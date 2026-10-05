// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Layer-major prefill (see glm_engine.h): for a prompt of T > max_chunk rows (up to tiles × max_chunk), each layer runs over every block
//   of max_chunk rows before the next layer starts. Attention and hyper-connections run block by block (KDA state and DSA caches continue
//   from block to block exactly as in the chunked path); the MoE runs once over all T rows, so a routed expert that is not resident crosses
//   PCIe once per layer for the whole call instead of once per block. Measured motivation (16K block, 14.7K tokens): 6.5 s of
//   8.0 s were expert streaming at the PCIe 4.0 limit (11,583 records ≈ 156 GB at 27.9 GB/s).
#include <algorithm>
#include <cstring>

#include "hive/clock.h"
#include "hive/glm/glm_engine.h"
#include "hive/glm/glm_kernels.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"

namespace hive::glm {
using namespace hive::k;

void GlmEngine::prefill_layer_major(GlmSeq& s, const int32_t* ids, int T, float* logits_last) {
  const int H = c_.hidden, HC = c_.hc * H, C = max_chunk_, E = c_.n_routed, K = c_.n_act, I = c_.moe_inter;
  HIVE_CHECK(T <= C * tiles_ && s.pos + T <= s.cap, "GLM layer-major prefill size");
  const int64_t pos0 = s.pos;
  GlmSeq* sp[1] = {&s};
  static const bool dbg = getenv("HIVE_GLM_DEBUG") && atoi(getenv("HIVE_GLM_DEBUG")) > 0;
  auto D = [&](const char* what, int l) { if (dbg) { CUDA_CHECK(cudaStreamSynchronize(stream_)); fprintf(stderr, "[glm-dbg] %s layer %d\n", what, l); } };
  const int nblk = (T + C - 1) / C;
  auto blk = [&](int b, int& r0, int& n) { r0 = b * C; n = std::min(C, T - r0); };
  // whole-prompt buffers: borrowed from the expert-cache tail (elastic, default) or the startup reservation
  bf16* hall; bf16* xnall; float* outall; float* postall; float* comball; float* rlog_all; int32_t* rids_all; float* rw_all;
  if (elastic_) {
    const size_t a = 256;
    auto up = [&](size_t x) { return (x + a - 1) / a * a; };
    const size_t sz[8] = {up((size_t)T * HC * 2), up((size_t)T * H * 2), up((size_t)T * H * 4), up((size_t)T * c_.hc * 4),
                          up((size_t)T * c_.hc * c_.hc * 4), up((size_t)T * E * 4), up((size_t)T * K * 4), up((size_t)T * K * 4)};
    size_t total = 0; for (size_t x : sz) total += x;
    const size_t rec = ex_.layout().total;
    const int n = (int)((total + rec - 1) / rec);
    uint8_t* p = ex_.lend_tail_slots(n);
    hall = (bf16*)p; p += sz[0]; xnall = (bf16*)p; p += sz[1]; outall = (float*)p; p += sz[2]; postall = (float*)p; p += sz[3];
    comball = (float*)p; p += sz[4]; rlog_all = (float*)p; p += sz[5]; rids_all = (int32_t*)p; p += sz[6]; rw_all = (float*)p;
  } else {
    hall = hall_.as<bf16>(); xnall = xnall_.as<bf16>(); outall = outall_.as<float>(); postall = postall_.as<float>(); comball = comball_.as<float>();
    rlog_all = rlog_all_.as<float>(); rids_all = rids_all_.as<int32_t>(); rw_all = rw_all_.as<float>();
  }
  // on an exception (e.g. from work run inside a layer yield) give the lent slots back and leave hcur_ cleared — the caller marks the sequence broken
  struct Lent {
    GlmEngine& e; bool on;
    ~Lent() { if (!on) return; cudaStreamSynchronize(e.stream_); if (e.elastic_) e.ex_.return_tail_slots(); e.hcur_ = nullptr; }
  } lent{*this, true};
  // embeddings → hc streams of every row
  {
    static thread_local std::vector<bf16> emb;
    for (int b = 0; b < nblk; ++b) {
      int r0, n; blk(b, r0, n);
      emb.resize((size_t)n * H);
      for (int r = 0; r < n; ++r) std::memcpy(emb.data() + (size_t)r * H, m_.embed_host() + (size_t)ids[r0 + r] * H, (size_t)H * 2);
      CUDA_CHECK(cudaMemcpyAsync(x_.p, emb.data(), (size_t)n * H * 2, cudaMemcpyHostToDevice, stream_));
      apply_embed_override(x_.as<bf16>(), emb_base_ + r0, n);
      for (int c = 0; c < c_.hc; ++c)
        CUDA_CHECK(cudaMemcpy2DAsync(hall + (size_t)r0 * HC + (size_t)c * H, (size_t)HC * 2, x_.p, (size_t)H * 2, (size_t)H * 2, n, cudaMemcpyDeviceToDevice, stream_));
    }
  }
  D("embeddings", -1);
  for (int l = 0; l < c_.n_layers; ++l) {
    const LayerW& L = m_.layer(l);
    for (int b = 0; b < nblk; ++b) {
      int r0, n; blk(b, r0, n);
      hcur_ = hall + (size_t)r0 * HC;
      s.pos = pos0 + r0;
      hc_pre_norm(L.hc_attn, L.in_norm, n);
      if (L.kda) kda_layer(l, sp, 1, n, true); else dsa_layer(l, sp, 1, n, true);
      hc_post_apply(n);
      hc_pre_norm(L.hc_ffn, L.post_norm, n);
      if (!L.moe) { mlp_layer(l, n, true); hc_post_apply(n); continue; }
      dcopy(xnall + (size_t)r0 * H, xn_.p, (size_t)n * H * 2, stream_);
      dcopy(postall + (size_t)r0 * c_.hc, post_.p, (size_t)n * c_.hc * 4, stream_);
      dcopy(comball + (size_t)r0 * c_.hc * c_.hc, comb_.p, (size_t)n * c_.hc * c_.hc * 4, stream_);
    }
    D("attention blocks", l);
    if (!L.moe) {
      if (layer_hook) { CUDA_CHECK(cudaStreamSynchronize(stream_)); layer_hook(l, hall, T); }
      if (layer_boundary && l + 1 < c_.n_layers) layer_boundary(l);  // layer yield (see glm_engine.h — hcur_ still points into hall)
      continue;
    }
    // MoE over all T rows: router, then every expert once
    const MoeW& w = L.moe_w;
    for (int b = 0; b < nblk; ++b) {
      int r0, n; blk(b, r0, n);
      blas_.gemm_bf16_f32out(xnall + (size_t)r0 * H, w.gate.p(), rlog_all + (size_t)r0 * E, n, E, H);
    }
    router_topk2(rlog_all, E, w.bias.as<float>(), nullptr, T, E, K, c_.routed_scale, rids_all, rw_all, nullptr, nullptr,
                 stream_);
    h_ids_all_.resize((size_t)T * K); h_w_all_.resize((size_t)T * K);
    CUDA_CHECK(cudaMemcpyAsync(h_ids_all_.data(), rids_all, (size_t)T * K * 4, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_w_all_.data(), rw_all, (size_t)T * K * 4, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemsetAsync(outall, 0, (size_t)T * H * 4, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    D("router", l);
    const double t0 = hive::mono_ms();
    ex_.prefill_layer(c_.moe_index[l], T, h_ids_all_.data(), h_w_all_.data(), xnall, outall, stream_);
    st_.ms_moe += hive::mono_ms() - t0;
    D("experts", l);
    // per block: shared expert, routed + shared, ffn hc_post
    for (int b = 0; b < nblk; ++b) {
      int r0, n; blk(b, r0, n);
      hcur_ = hall + (size_t)r0 * HC;
      const bf16* xb = xnall + (size_t)r0 * H;
      bf16* sgu = sh_gu_.as<bf16>();
      lin(w.sh_gate, xb, H, n, sgu, 2 * I);
      lin(w.sh_up, xb, H, n, sgu + I, 2 * I);
      swiglu_rows(sgu, n, I, c_.swiglu_limit, sh_y_.as<bf16>(), stream_);
      lin(w.sh_down, sh_y_.as<bf16>(), I, n, mlp_out_.as<bf16>(), H);
      f32_to_bf16(outall + (size_t)r0 * H, (size_t)n * H, attn_.as<bf16>(), stream_);
      add_bf16(attn_.as<bf16>(), mlp_out_.as<bf16>(), (size_t)n * H, attn_.as<bf16>(), stream_);
      dcopy(post_.p, postall + (size_t)r0 * c_.hc, (size_t)n * c_.hc * 4, stream_);
      dcopy(comb_.p, comball + (size_t)r0 * c_.hc * c_.hc, (size_t)n * c_.hc * c_.hc * 4, stream_);
      hc_post_apply(n);
    }
    if (layer_hook) { CUDA_CHECK(cudaStreamSynchronize(stream_)); layer_hook(l, hall, T); }  // all T rows of the hc streams (validation)
    if (layer_boundary && l + 1 < c_.n_layers) layer_boundary(l);  // layer yield
  }
  // head of the last row; MTP entries for every row (final hidden per block)
  s.pos = pos0;
  const bool mtp = mtp_on();
  if (mtp && s.h_last_valid && pos0 > 0 && s.mtp_written < pos0) mtp_kv(s, pos0 - 1, 1, s.h_last.as<bf16>(), ids);
  for (int b = 0; b < nblk; ++b) {
    int r0, n; blk(b, r0, n);
    const bool last = b + 1 == nblk;
    if (!mtp && !last) continue;
    const int rr0 = mtp ? 0 : n - 1, rn = mtp ? n : 1;
    hc_stream_mean(hall + (size_t)(r0 + rr0) * HC, rn, c_.hc, H, fx_.p ? fx_.as<bf16>() : final_x_.as<bf16>(), stream_);
    bf16* fh = fh_.p ? fh_.as<bf16>() : final_xn_.as<bf16>();
    rmsnorm(fx_.p ? fx_.as<bf16>() : final_x_.as<bf16>(), m_.final_norm().as<bf16>(), c_.rms_eps, rn, H, fh, stream_);
    if (mtp) {
      const int ne = last ? n - 1 : n;  // entry j pairs h_j with x_{j+1}: the last row's entry waits for the next token
      if (ne > 0) mtp_kv(s, pos0 + r0, ne, fh, ids + r0 + 1);
      if (last) { dcopy(s.h_last.p, fh + (size_t)(n - 1) * H, (size_t)H * 2, stream_); s.h_last_valid = true; }
    }
    if (last) {
      blas_.gemm_bf16_f32out(fh + (size_t)(rn - 1) * H, m_.lm_head().p(), logits_dev_.as<float>(), 1, c_.vocab, H);
      if (logits_last) CUDA_CHECK(cudaMemcpyAsync(logits_last, logits_dev_.p, (size_t)c_.vocab * 4, cudaMemcpyDeviceToHost, stream_));
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  if (elastic_) ex_.return_tail_slots();
  hcur_ = nullptr;
  lent.on = false;
  s.pos = pos0 + T;
  s.tokens.insert(s.tokens.end(), ids, ids + T);
}

}  // namespace hive::glm
