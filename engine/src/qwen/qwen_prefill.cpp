// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Layer-major prefill (see GlmEngine's, hive/glm/glm_prefill.cpp — the same reason): for a prompt of T > max_chunk rows (up to tiles ×
//   max_chunk) each layer runs over every block of max_chunk rows before the next layer; the MoE runs once over all T rows, so a routed
//   expert that is not resident crosses PCIe once per layer for the call instead of once per block. Only used while some expert is not
//   resident (QwenEngine::prefill) — with every expert in VRAM there is nothing to stream and the chunked path is used.
//   The whole-prompt buffers are borrowed from the expert cache's tail slots for the call (GlmExperts::lend_tail_slots).
#include <algorithm>
#include <cstring>

#include "hive/clock.h"
#include "hive/glm/glm_kernels.h"
#include "hive/glm/glm_moe.h"
#include "hive/qwen/qwen_engine.h"
#include "hive/qwen/qwen_kernels.h"

namespace hive::qwen {

void QwenEngine::prefill_layer_major(QwenSeq& s, const int32_t* ids, int T, float* logits_last) {
  const QwenConfig& c = c_;
  const int H = c.hidden, HC = c.hcw(), C = max_chunk_, E = c.n_experts, K = c.n_act, ND = c.hc_rank + c.hc, LD = E + 1;
  HIVE_CHECK(T <= C * tiles_ && s.pos + T <= s.cap, "qwen: layer-major prefill size");
  const int64_t pos0 = s.pos;
  const int nblk = (T + C - 1) / C;
  auto blk = [&](int b, int& r0, int& n) { r0 = b * C; n = std::min(C, T - r0); };
  const size_t a = 256;
  auto up = [&](size_t x) { return (x + a - 1) / a * a; };
  const size_t sz[7] = {up((size_t)T * HC * 2), up((size_t)T * H * 2), up((size_t)T * H * 4), up((size_t)T * ND * 2), up((size_t)T * LD * 4),
                        up((size_t)T * K * 4), up((size_t)T * K * 4)};
  size_t total = 0;
  for (size_t x : sz) total += x;
  const size_t rec = ex_.layout().total;
  uint8_t* p = ex_.lend_tail_slots((int)((total + rec - 1) / rec));
  bf16* hall = (bf16*)p; p += sz[0];
  bf16* xnall = (bf16*)p; p += sz[1];
  float* outall = (float*)p; p += sz[2];
  bf16* dlall = (bf16*)p; p += sz[3];
  float* rlog = (float*)p; p += sz[4];
  int32_t* rids = (int32_t*)p; p += sz[5];
  float* rw = (float*)p;
  struct Lent {  // also on an exception: the slots come back and hcur_ is cleared (the caller marks the sequence broken)
    QwenEngine& e; bool on;
    ~Lent() { if (!on) return; cudaStreamSynchronize(e.stream_); e.ex_.return_tail_slots(); e.hcur_ = nullptr; }
  } lent{*this, true};
  std::vector<std::vector<RowInfo>> brows(nblk);
  for (int b = 0; b < nblk; ++b) {
    int r0, n; blk(b, r0, n);
    brows[b].resize(n);
    for (int i = 0; i < n; ++i) brows[b][i] = {&s, pos0 + r0 + i};
  }
  // rope coordinates and embeddings of every block (the image-span bookkeeping in set_rope walks the rows in order)
  const int64_t base0 = emb_base_;
  for (int b = 0; b < nblk; ++b) {
    int r0, n; blk(b, r0, n);
    set_rope(brows[b], base0 + r0, true);
    embed(brows[b], ids + r0, base0 + r0, hall + (size_t)r0 * HC);
    CUDA_CHECK(cudaStreamSynchronize(stream_));  // emb_pin_ / rope_pin_ are reused by the next block
  }
  // the PLE reads each block's own n-gram history from s.tokens: keep it in step with the blocks already processed
  const size_t tok0 = s.tokens.size();
  for (int l = 0; l < c.n_layers; ++l) {
    const LayerW& L = m_.layer(l);
    for (int b = 0; b < nblk; ++b) {
      int r0, n; blk(b, r0, n);
      hcur_ = hall + (size_t)r0 * HC;
      if (L.ple) { s.tokens.resize(tok0); s.tokens.insert(s.tokens.end(), ids, ids + r0); }
      layer_front(l, brows[b], ids + r0, true, hcur_);
      CUDA_CHECK(cudaMemcpyAsync(xnall + (size_t)r0 * H, x_.p, (size_t)n * H * 2, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(dlall + (size_t)r0 * ND, dl_.p, (size_t)n * ND * 2, cudaMemcpyDeviceToDevice, stream_));
    }
    if (L.ple) s.tokens.resize(tok0);
    // MoE over all T rows: router, then every expert once
    const MoeW& w = L.moe;
    for (int b = 0; b < nblk; ++b) {
      int r0, n; blk(b, r0, n);
      blas_.gemm_bf16_f32out(xnall + (size_t)r0 * H, w.router.p(), rlog + (size_t)r0 * LD, n, LD, H);
    }
    router_softmax_topk(rlog, LD, T, E, K, c.norm_topk, rids, rw, nullptr, nullptr, stream_);
    h_ids_.resize((size_t)T * K); h_w_.resize((size_t)T * K);
    CUDA_CHECK(cudaMemcpyAsync(h_ids_.data(), rids, (size_t)T * K * 4, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_w_.data(), rw, (size_t)T * K * 4, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemsetAsync(outall, 0, (size_t)T * H * 4, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const double t0 = hive::mono_ms();
    ex_.prefill_layer(l, T, h_ids_.data(), h_w_.data(), xnall, outall, stream_);
    st_.ms_moe += hive::mono_ms() - t0;
    for (int b = 0; b < nblk; ++b) {
      int r0, n; blk(b, r0, n);
      hcur_ = hall + (size_t)r0 * HC;
      moe_shared(w, xnall + (size_t)r0 * H, n, shared_out_.as<bf16>());
      moe_combine(outall + (size_t)r0 * H, shared_out_.as<bf16>(), rlog + (size_t)r0 * LD, LD, E, n, H, y_.as<bf16>(), stream_);
      hc_combine_(L.hc_mlp, hcur_, y_.as<bf16>(), n, dlall + (size_t)r0 * ND);
    }
    if (layer_hook) { CUDA_CHECK(cudaStreamSynchronize(stream_)); layer_hook(l, hall, T); }
    if (layer_boundary && l + 1 < c.n_layers) layer_boundary(l);
  }
  // head of the last row; MTP entries of every row (the final hidden streams, block by block)
  const bool mtp = mtp_on();
  if (mtp) {
    if (s.h_last_valid && pos0 > 0 && s.mtp_written < pos0) mtp_kv(s, pos0 - 1, 1, s.h_last.as<bf16>(), ids, base0);
    for (int b = 0; b < nblk; ++b) {
      int r0, n; blk(b, r0, n);
      const bool last = b + 1 == nblk;
      const int ne = last ? n - 1 : n;  // entry j pairs h_j with x_{j+1}: the last row's entry waits for the next token
      if (ne > 0) mtp_kv(s, pos0 + r0, ne, hall + (size_t)r0 * HC, ids + r0 + 1, base0 + r0 + 1);
    }
    CUDA_CHECK(cudaMemcpyAsync(s.h_last.p, hall + (size_t)(T - 1) * HC, (size_t)HC * 2, cudaMemcpyDeviceToDevice, stream_));
    s.h_last_valid = true;
  }
  head(hall + (size_t)(T - 1) * HC, 1, logits_last);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  ex_.return_tail_slots();
  hcur_ = nullptr;
  lent.on = false;
  s.pos = pos0 + T;
  s.tokens.insert(s.tokens.end(), ids, ids + T);
}

}  // namespace hive::qwen
