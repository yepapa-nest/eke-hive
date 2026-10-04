// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash speculative decoding with the NextN (MTP) layer — see glm_engine.h for the contract.
//   The NextN layer (layer 45 of the checkpoint) is one plain-residual decoder layer (no hyper-connections): DSA attention with its own
//   cache, a MoE with its own 288 routed experts (Z.ai FP8, resident in VRAM) and a shared expert. Its input for entry j is
//   eh_proj([enorm(embedding of x_{j+1}) | hnorm(h_j)]) where h_j is the main model's final normed hidden at position j; the embedding is
//   zeroed at position 0. Output: shared_head.norm(mlp_out + residual) → the main lm_head (the checkpoint has no MTP head); that normed
//   output is also the hidden fed to the next draft step.
//   (Definition read from the public GLM-5 Next inference code of vllm-ascend, models/glm5next/mtp.py — math only, no code taken.)
#include <algorithm>
#include <cstring>

#include "hive/glm/glm_engine.h"
#include "hive/glm/glm_kernels.h"
#include "hive/glm/glm_moe_fp8.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"

namespace hive::glm {
using namespace hive::k;

// Exact MTP KV entries pos0 .. pos0+n-1: entry j = f(h_rows[j - pos0], next_ids[j - pos0]). Only the attention-cache inputs are computed
//   (kv_a → norm → latent c, indexer key and pool gate); queries, attention and the MoE are not needed for cache entries.
void GlmEngine::mtp_kv(GlmSeq& s, int64_t pos0, int n, const bf16* h_rows, const int32_t* next_ids) {
  if (n <= 0) return;
  const MtpW& w = *m_.mtp();
  const int H = c_.hidden;
  HIVE_CHECK(n <= max_chunk_ && pos0 + n <= s.cap, "GLM MTP entries");
  for (int i = 0; i < n; ++i) {
    if (pos0 + i == 0) std::memset(mtp_emb_pin_ + (size_t)i * H, 0, (size_t)H * 2);
    else std::memcpy(mtp_emb_pin_ + (size_t)i * H, m_.embed_host() + (size_t)next_ids[i] * H, (size_t)H * 2);
  }
  bf16* en = mtp_en_.as<bf16>(); bf16* hn = mtp_hn_.as<bf16>(); bf16* cat = mtp_cat_.as<bf16>();
  bf16* x0 = mtp_x0_.as<bf16>(); bf16* xn = mtp_xn_.as<bf16>();
  rmsnorm(mtp_emb_pin_, w.enorm.as<bf16>(), c_.rms_eps, n, H, en, stream_);   // pinned rows read directly (UVA)
  rmsnorm(h_rows, w.hnorm.as<bf16>(), c_.rms_eps, n, H, hn, stream_);
  cat_rows(en, hn, n, H, cat, stream_);
  lin(w.eh_proj, cat, 2 * H, n, x0, H);
  rmsnorm(x0, w.in_norm.as<bf16>(), c_.rms_eps, n, H, xn, stream_);
  const DsaW& a = w.dsa_w;
  lin(a.kv_a, xn, H, n, ckv_.as<bf16>(), c_.kv_lora);
  rmsnorm(ckv_.as<bf16>(), a.kv_a_norm.as<bf16>(), c_.rms_eps, n, c_.kv_lora, cn_.as<bf16>(), stream_);
  lin(a.idx_wk, xn, H, n, kI_raw_.as<bf16>(), c_.index_dim);
  layernorm(kI_raw_.as<bf16>(), a.idx_k_norm_w.as<bf16>(), a.idx_k_norm_b.as<bf16>(), 1e-6f, n, c_.index_dim, kI_.as<bf16>(), stream_);
  lin(a.idx_gate, xn, H, n, gs_.as<bf16>(), c_.index_dim);
  dsa_append(DsaLayerW{a.kv_b.p(), a.idx_ape.as<bf16>()}, s.mtp_cache, (int)pos0, n, cn_.as<bf16>(), kI_.as<bf16>(), gs_.as<bf16>(), stream_);
  s.mtp_written = std::max(s.mtp_written, pos0 + n);
}

// One full draft step at entry `pos` (rows = 1), no host synchronization: step 0 embeds `tok` (host), later steps embed the previous step's
//   draft straight from the device (pinned embedding table, UVA). Writes the draft id to mtp_ids_d_[step] and the softmax max / sum-exp of
//   its logits to mtp_conf_d_[2·step, 2·step+1]; the routed experts run from device-side ids (glm_moe_fp8_decode).
void GlmEngine::mtp_step(GlmSeq& s, int step, int32_t tok, const bf16* h_in, int64_t pos, bf16* h_out) {
  const MtpW& w = *m_.mtp();
  const int H = c_.hidden, E = c_.n_routed, K = c_.n_act, I = c_.moe_inter;
  HIVE_CHECK(pos + 1 <= s.cap, "GLM MTP draft beyond the sequence capacity");
  bf16* en = mtp_en_.as<bf16>(); bf16* hn = mtp_hn_.as<bf16>(); bf16* cat = mtp_cat_.as<bf16>();
  bf16* x0 = mtp_x0_.as<bf16>(); bf16* xn = mtp_xn_.as<bf16>(); bf16* h1 = mtp_h1_.as<bf16>(); bf16* tmp = mtp_tmp_.as<bf16>();
  const bf16* emb;
  if (step == 0) {
    if (pos == 0) std::memset(mtp_emb_pin_, 0, (size_t)H * 2);
    else std::memcpy(mtp_emb_pin_, m_.embed_host() + (size_t)tok * H, (size_t)H * 2);
    emb = mtp_emb_pin_;
  } else {
    embed_gather_dev(m_.embed_host(), mtp_ids_d_.as<int32_t>() + step - 1, H, tmp, stream_);
    emb = tmp;
  }
  rmsnorm(emb, w.enorm.as<bf16>(), c_.rms_eps, 1, H, en, stream_);
  rmsnorm(h_in, w.hnorm.as<bf16>(), c_.rms_eps, 1, H, hn, stream_);
  cat_rows(en, hn, 1, H, cat, stream_);
  { GemvSeg e{w.eh_proj.p(), cat, x0, H, 2 * H, 2 * H, H}; glm_gemv_bf16_multi(&e, 1, 1, stream_); }
  rmsnorm(x0, w.in_norm.as<bf16>(), c_.rms_eps, 1, H, xn, stream_);
  // attention (DSA, own cache)
  const DsaW& a = w.dsa_w;
  lin(a.q_a, xn, H, 1, qa_.as<bf16>(), c_.q_lora);
  rmsnorm(qa_.as<bf16>(), a.q_a_norm.as<bf16>(), c_.rms_eps, 1, c_.q_lora, qres_.as<bf16>(), stream_);
  lin(a.q_b, qres_.as<bf16>(), c_.q_lora, 1, q_.as<bf16>(), a.q_b.out);
  lin(a.kv_a, xn, H, 1, ckv_.as<bf16>(), c_.kv_lora);
  rmsnorm(ckv_.as<bf16>(), a.kv_a_norm.as<bf16>(), c_.rms_eps, 1, c_.kv_lora, cn_.as<bf16>(), stream_);
  {
    GemvSeg sg[3] = {{a.idx_wq_b.p(), qres_.as<bf16>(), qI_.as<bf16>(), a.idx_wq_b.out, c_.q_lora, c_.q_lora, a.idx_wq_b.out},
                     {a.idx_wk.p(), xn, kI_raw_.as<bf16>(), c_.index_dim, H, H, c_.index_dim},
                     {a.idx_gate.p(), xn, gs_.as<bf16>(), c_.index_dim, H, H, c_.index_dim}};
    glm_gemv_bf16_multi(sg, 3, 1, stream_);
  }
  layernorm(kI_raw_.as<bf16>(), a.idx_k_norm_w.as<bf16>(), a.idx_k_norm_b.as<bf16>(), 1e-6f, 1, c_.index_dim, kI_.as<bf16>(), stream_);
  blas_.gemm_bf16_f32out(xn, a.idx_weights.p(), wI_.as<float>(), 1, c_.index_heads, H);
  const DsaLayerW dw{a.kv_b.p(), a.idx_ape.as<bf16>()};
  dsa_append(dw, s.mtp_cache, (int)pos, 1, cn_.as<bf16>(), kI_.as<bf16>(), gs_.as<bf16>(), stream_);
  dsa_select(s.mtp_cache, (int)pos, 1, qI_.as<bf16>(), wI_.as<float>(), idx_.as<int32_t>(), ws_.p, ws_bytes_, stream_);
  dsa_attention(dw, s.mtp_cache, (int)pos, 1, q_.as<bf16>(), idx_.as<int32_t>(), dsa_out_.as<bf16>(), ws_.p, ws_bytes_, stream_);
  lin(a.o, dsa_out_.as<bf16>(), c_.n_heads * c_.v_head, 1, attn_.as<bf16>(), H);
  // residual + post-attention norm, MoE (device-side routing)
  add_bf16(x0, attn_.as<bf16>(), H, h1, stream_);
  rmsnorm(h1, w.post_norm.as<bf16>(), c_.rms_eps, 1, H, xn, stream_);
  const MoeW& mw = w.moe_w;
  glm_router_decode(xn, H, 1, mw.gate.p(), mw.bias.as<float>(), E, H, K, c_.routed_scale, rlog_.as<float>(), router_ids_.as<int32_t>(),
                    router_w_.as<float>(), nullptr, nullptr, stream_);
  float* acc = mlp_out_f32_.as<float>();
  CUDA_CHECK(cudaMemsetAsync(acc, 0, (size_t)H * 4, stream_));
  glm_moe_fp8_decode(mtp_moe_tab_, E, router_ids_.as<int32_t>(), router_w_.as<float>(), xn, 1, K, H, I, c_.swiglu_limit, acc, mtp_moe_ws_.p, stream_);
  if (mw.sh_gate.fp8 && mw.sh_up.fp8 && mw.sh_down.fp8)
    glm_shared_fp8_decode(mw.sh_gate.f8(), mw.sh_up.f8(), mw.sh_down.f8(), xn, H, 1, c_.swiglu_limit, sh_y_.as<bf16>(), mlp_out_.as<bf16>(), H, stream_);
  else {
    bf16* sgu = sh_gu_.as<bf16>();
    lin(mw.sh_gate, xn, H, 1, sgu, 2 * I); lin(mw.sh_up, xn, H, 1, sgu + I, 2 * I);
    swiglu_rows(sgu, 1, I, c_.swiglu_limit, sh_y_.as<bf16>(), stream_);
    lin(mw.sh_down, sh_y_.as<bf16>(), I, 1, mlp_out_.as<bf16>(), H);
  }
  f32_to_bf16(acc, H, tmp, stream_);
  add_bf16(tmp, mlp_out_.as<bf16>(), H, tmp, stream_);   // routed + shared
  add_bf16(tmp, h1, H, tmp, stream_);                     // + residual
  rmsnorm(tmp, w.head_norm.as<bf16>(), c_.rms_eps, 1, H, h_out, stream_);
  // head (main lm_head) → greedy draft and its softmax statistics, left on the device
  blas_.gemm_bf16_f32out(h_out, m_.lm_head().p(), logits_dev_.as<float>(), 1, c_.vocab, H);
  argmax_rows(logits_dev_.as<float>(), 1, c_.vocab, mtp_ids_d_.as<int32_t>() + step, stream_);
  row_max_sumexp(logits_dev_.as<float>(), 1, c_.vocab, one_f_.as<float>(), mtp_conf_d_.as<float>() + 2 * step, mtp_conf_d_.as<float>() + 2 * step + 1, stream_);
}

int GlmEngine::mtp_draft(GlmSeq& s, int32_t tok, int k, int32_t* drafts, float* conf) {
  if (!mtp_on() || !s.h_last_valid || k <= 0) return 0;
  k = std::min(k, mtp_k_);
  const int64_t p = s.pos - 1;  // entry index of (h_last, tok)
  if (p < 0 || p + k + 1 > s.cap) return 0;
  kv_reserve(s, std::min<int64_t>(s.cap, s.pos + k + kKvSlack));
  const bf16* hin = s.h_last.as<bf16>();
  for (int i = 0; i < k; ++i) {
    bf16* hout = mtp_hout_.as<bf16>() + (size_t)(i & 1) * c_.hidden;
    mtp_step(s, i, tok, hin, p + i, hout);
    hin = hout;
  }
  int32_t ids[8]; float ms[16];
  CUDA_CHECK(cudaMemcpyAsync(ids, mtp_ids_d_.p, (size_t)k * 4, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(ms, mtp_conf_d_.p, (size_t)k * 8, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  for (int i = 0; i < k; ++i) { drafts[i] = ids[i]; conf[i] = ms[2 * i + 1] > 0.f ? 1.f / ms[2 * i + 1] : 0.f; }  // p(argmax) = 1 / Σ exp(l − max)
  s.mtp_written = std::max(s.mtp_written, p + 1);  // the first entry used the real hidden: exact
  return k;
}

void GlmEngine::verify(GlmSeq& s, const int32_t* ids, int M, float* logits_host) {
  GlmSeq* sp[1] = {&s};
  verify_batch(sp, &M, 1, ids, logits_host);
}

void GlmEngine::verify_batch(GlmSeq* const* seqs, const int* Ms, int S, const int32_t* ids, float* logits_host) {
  HIVE_CHECK(S >= 1 && S <= snap_parts_, "GLM verify: more parts than snapshots (enable_batch_verify)");
  int R = 0;
  vparts_.clear();
  GlmSeq* sp[8];
  for (int p = 0; p < S; ++p) {
    const int M = Ms[p];
    HIVE_CHECK(M >= 1 && R + M <= 8 && M - 1 <= std::max(1, mtp_k_), "GLM verify rows");
    HIVE_CHECK(seqs[p]->pos + M <= seqs[p]->cap, "GLM: sequence capacity exceeded");
    kv_reserve(*seqs[p], std::min<int64_t>(seqs[p]->cap, seqs[p]->pos + M + kKvSlack));
    for (int q = 0; q < p; ++q) HIVE_CHECK(seqs[q] != seqs[p], "GLM verify: a sequence appears in two parts");
    vparts_.push_back({seqs[p], seqs[p]->pos, M, R});
    for (int r = 0; r < M; ++r) { sp[R + r] = seqs[p]; vrow_off_[R + r] = r; }
    R += M;
  }
  verify_ids_.assign(ids, ids + R);
  verify_ = true;
  try {
    forward(sp, R, ids, 1, false, logits_host, true);
  } catch (...) { verify_ = false; vparts_.clear(); throw; }
  verify_ = false;
  ex_.after_step(R);
}

void GlmEngine::rollback(GlmSeq& s, int n_keep) {
  HIVE_CHECK(vparts_.size() == 1 && vparts_[0].s == &s, "GLM rollback without a matching verify");
  rollback_batch(&n_keep, 1);
}

void GlmEngine::rollback_batch(const int* n_keep, int S) {
  HIVE_CHECK(S == (int)vparts_.size() && S >= 1, "GLM rollback without a matching verify");
  const int H = c_.hidden, kmax = std::max(1, mtp_k_);
  for (int p = 0; p < S; ++p) {
    const VPart& vp = vparts_[p];
    GlmSeq& s = *vp.s;
    const int M = vp.M, nk = n_keep[p];
    HIVE_CHECK(nk >= 1 && nk <= M && s.pos == vp.pos0 + M, "GLM rollback without a matching verify");
    if (nk < M)
      for (int ki = 0; ki < n_kda_; ++ki) {
        dcopy(s.state[ki].p, snap_state_[((size_t)p * kmax + nk - 1) * n_kda_ + ki].p, s.state[ki].n, stream_);
        dcopy(s.conv[ki].p, snap_conv_[((size_t)p * kmax + nk - 1) * n_kda_ + ki].p, s.conv[ki].n, stream_);
      }
    s.pos = vp.pos0 + nk;
    s.tokens.resize(s.tokens.size() - (size_t)(M - nk));
    if (mtp_on()) {
      if (nk >= 2) mtp_kv(s, vp.pos0, nk - 1, fh_.as<bf16>() + (size_t)vp.r0 * H, verify_ids_.data() + vp.r0 + 1);
      dcopy(s.h_last.p, fh_.as<bf16>() + (size_t)(vp.r0 + nk - 1) * H, (size_t)H * 2, stream_);
      s.h_last_valid = true;
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  vparts_.clear();
}

void GlmEngine::mtp_after_forward(GlmSeq* const* seqs, int M, const int32_t* ids, int T, bool prefill, const int64_t* pos_before) {
  const int H = c_.hidden;
  if (verify_) {  // per part: the entry of the position before the part (h_last ↔ its first token); the rows are entered by rollback
    for (const VPart& vp : vparts_) {
      GlmSeq& s = *vp.s;
      if (s.h_last_valid && vp.pos0 > 0 && s.mtp_written < vp.pos0) mtp_kv(s, vp.pos0 - 1, 1, s.h_last.as<bf16>(), ids + vp.r0);
    }
  } else if (prefill) {
    GlmSeq& s = *seqs[0];
    const int64_t p0 = pos_before[0];
    if (s.h_last_valid && p0 > 0 && s.mtp_written < p0) mtp_kv(s, p0 - 1, 1, s.h_last.as<bf16>(), ids);
    {
      if (T > 1) mtp_kv(s, p0, T - 1, fh_.as<bf16>(), ids + 1);
      dcopy(s.h_last.p, fh_.as<bf16>() + (size_t)(T - 1) * H, (size_t)H * 2, stream_);
      s.h_last_valid = true;
    }
  } else {
    for (int m = 0; m < M; ++m) {
      GlmSeq& s = *seqs[m];
      const int64_t p0 = pos_before[m];
      if (s.h_last_valid && p0 > 0 && s.mtp_written < p0) mtp_kv(s, p0 - 1, 1, s.h_last.as<bf16>(), ids + m);
      dcopy(s.h_last.p, fh_.as<bf16>() + (size_t)m * H, (size_t)H * 2, stream_);
      s.h_last_valid = true;
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace hive::glm
