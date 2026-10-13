// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next speculative decoding with its MTP layer — the GlmEngine contract (mtp_draft → verify → rollback, hive/glm/glm_mtp.cpp).
//   Entry j of the MTP layer combines the main model's final hidden stream h_j (before the final mixer) and the token x_{j+1}:
//     e = fc_embedding(norm_e(embed(x_{j+1}))) · s_c = fc_hidden(norm_h(h_j) copy c) · input stream copy c = s_c + e
//   (norm_e over 2560, norm_h over the whole 4 · 2560, both (1 + w)); then one full-attention decoder layer (hyper-connections, QSA attention
//   with its own cache, a MoE whose 512 routed experts are FP8 in VRAM, the gated shared expert) at rope position j; its own final mixer
//   and the main lm_head give the draft. The layer's output stream is the hidden of the next draft step.
#include <algorithm>
#include <cstring>

#include "hive/glm/fp8b.h"
#include "hive/glm/glm_kernels.h"
#include "hive/glm/glm_moe.h"
#include "hive/model_kernels.h"
#include "hive/qwen/qwen_engine.h"
#include "hive/qwen/fn_moe.h"
#include "hive/qwen/fn_gemv.h"
#include "hive/qwen/qwen_kernels.h"

namespace hive::qwen {

// Verify snapshots: capacity for `parts` parts (pointer tables), buffers for the parts in use only — part 0 always; parts 1.. are allocated
//   when a batched verify needs them and freed again after kSnapIdle verify steps of one part (16 parts held 2.3 GiB that a 64K context's KV
//   then took from the expert cache, 2026-10-10: 24039 of 24576 resident, host routing)
void QwenEngine::alloc_snapshots(int parts) {
  const int kmax = std::max(1, mtp_k_);
  HIVE_CHECK(kmax <= 8 && parts >= 1 && parts <= kDecRows / 2, "qwen verify snapshots: at most 8 rows per part and kDecRows / 2 parts");
  // HIVE_QWEN_SNAP_REPLAY (default on): parts ≥ 1 keep ONE state snapshot (the state before the verify) and replay the accepted rows on a
  //   rejection; part 0 keeps one per row (c1: rollback by buffer swap). 16 parts: 48 → 18 state copies per layer (5.4 → 2.0 GiB)
  snap_replay_ = !(getenv("HIVE_QWEN_SNAP_REPLAY") && strcmp(getenv("HIVE_QWEN_SNAP_REPLAY"), "0") == 0);
  const int live = std::min(snap_live_, parts);
  release_parts(0);
  snap_state_.clear(); snap_conv_.clear(); snap_conv0_.clear(); snap_ple_.clear();
  snap_state_.resize((size_t)parts * kmax * n_lin_); snap_conv_.resize((size_t)parts * kmax * n_lin_);
  if (snap_replay_) snap_conv0_.resize((size_t)parts * n_lin_);
  if (m_.ple_layer() >= 0) snap_ple_.resize((size_t)parts * kmax);
  if (snap_replay_ && parts > 1) {
    if (!proj_save_.p) proj_save_.alloc((size_t)n_lin_ * kDecRows * kGdnProjWidth * 2);
    if (rb_parts_ < parts) {   // one pointer section per part: every replayed part runs in the same launches (gdn_decode_rows_multi)
      if (rb_pin_) CUDA_CHECK(cudaFreeHost(rb_pin_));
      CUDA_CHECK(cudaMallocHost(&rb_pin_, (size_t)parts * n_lin_ * 2 * sizeof(float*)));
      rb_dev_.alloc((size_t)parts * n_lin_ * 2 * sizeof(float*));
      rb_parts_ = parts;
    }
  }
  if (snap_ptr_pin_) CUDA_CHECK(cudaFreeHost(snap_ptr_pin_));
  ptr_snap_n_ = (size_t)parts * n_lin_ * 16 + (size_t)parts * 8;
  CUDA_CHECK(cudaMallocHost(&snap_ptr_pin_, ptr_snap_n_ * sizeof(float*)));
  std::memset(snap_ptr_pin_, 0, ptr_snap_n_ * sizeof(float*));
  snap_parts_ = parts;
  snap_live_ = 0;
  ensure_parts(std::max(1, live));
  fprintf(stderr, "[qwen] MTP: up to %d drafts per step · verify snapshots: capacity %d part(s), %.0f MiB per batched part (allocated on use)\n", mtp_k_,
          parts, (double)(n_lin_ * ((snap_replay_ ? 1 : kmax) * kGdnStateFloats + (kmax + 1) * kGdnConvStateFloats) * 4.0) / 1048576.0);
}

void QwenEngine::ensure_parts(int n) {
  if (n <= snap_live_) return;
  const int kmax = std::max(1, mtp_k_);
  size_t need = 0;
  for (int p = snap_live_; p < n; ++p)
    need += (size_t)n_lin_ * (((!snap_replay_ || p == 0) ? kmax : 1) * kGdnStateFloats + (kmax + 1) * kGdnConvStateFloats) * 4;
  make_room(need);
  for (int p = snap_live_; p < n; ++p) {
    for (int li = 0; li < n_lin_; ++li)
      for (int r = 0; r < kmax; ++r) {
        const size_t i = ((size_t)p * kmax + r) * n_lin_ + li;
        if (!snap_replay_ || p == 0 || r == 0) snap_state_[i].alloc(kGdnStateFloats * 4);
        snap_conv_[i].alloc(kGdnConvStateFloats * 4);
        snap_ptr_pin_[((size_t)p * n_lin_ + li) * 16 + r] = snap_conv_[i].as<float>();
        snap_ptr_pin_[((size_t)p * n_lin_ + li) * 16 + 8 + r] = snap_state_[i].as<float>();
      }
    if (replay_part(p))
      for (int li = 0; li < n_lin_; ++li) {
        snap_conv0_[(size_t)p * n_lin_ + li].alloc(kGdnConvStateFloats * 4);
        snap_ptr_pin_[((size_t)p * n_lin_ + li) * 16 + 7] = snap_conv0_[(size_t)p * n_lin_ + li].as<float>();   // conv window before row 0
      }
    if (!snap_ple_.empty())
      for (int r = 0; r < kmax; ++r) {
        snap_ple_[(size_t)p * kmax + r].alloc((size_t)kPleConvState * c_.hcw() * 4);
        snap_ptr_pin_[(size_t)snap_parts_ * n_lin_ * 16 + (size_t)p * 8 + r] = snap_ple_[(size_t)p * kmax + r].as<float>();
      }
  }
  snap_live_ = n;
}

void QwenEngine::release_parts(int keep) {
  if (snap_live_ <= keep) return;
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  const int kmax = std::max(1, mtp_k_);
  for (int p = keep; p < snap_live_; ++p) {
    for (int li = 0; li < n_lin_; ++li) {
      for (int r = 0; r < kmax; ++r) {
        const size_t i = ((size_t)p * kmax + r) * n_lin_ + li;
        if (i < snap_state_.size()) { snap_state_[i].free(); snap_conv_[i].free(); }
      }
      if ((size_t)p * n_lin_ + li < snap_conv0_.size()) snap_conv0_[(size_t)p * n_lin_ + li].free();
      if (snap_ptr_pin_) std::memset(snap_ptr_pin_ + ((size_t)p * n_lin_ + li) * 16, 0, 16 * sizeof(float*));
    }
    for (int r = 0; r < kmax && (size_t)p * kmax + r < snap_ple_.size(); ++r) snap_ple_[(size_t)p * kmax + r].free();
  }
  snap_live_ = keep;
  give_back_room();
}

void QwenEngine::enable_batch_verify(int parts) { if (mtp_on() && parts > snap_parts_) alloc_snapshots(parts); }

void QwenEngine::mtp_input(const bf16* emb, const bf16* h_rows, int n, bf16* hc_out) {
  const MtpW& w = *m_.mtp();
  const int H = c_.hidden, HC = c_.hcw();
  gemma_rmsnorm(emb, H, w.pre_emb_norm.as<bf16>(), n, H, H, c_.rms_eps, mtp_en_.as<bf16>(), H, stream_);
  lin(w.fc_emb, mtp_en_.as<bf16>(), H, n, mtp_fc_.as<bf16>(), H);
  gemma_rmsnorm(h_rows, HC, w.pre_hidden_norm.as<bf16>(), n, HC, HC, c_.rms_eps, mtp_hn_.as<bf16>(), HC, stream_);
  lin(w.fc_hidden, mtp_hn_.as<bf16>(), H, n * c_.hc, hc_out, H);   // each copy separately: [n · hc, H] rows
  hc_add_rows(hc_out, mtp_fc_.as<bf16>(), n, c_.hc, H, stream_);
}

// Exact MTP cache entries pos0 .. pos0+n-1 (entry j = f(h_rows[j − pos0], next_ids[j − pos0]) at rope position j): only the attention-cache
//   inputs (keys, values, indexer keys, completed groups) — queries, attention and the MoE are not needed for cache entries.
void QwenEngine::mtp_kv(QwenSeq& s, int64_t pos0, int n, const bf16* h_rows, const int32_t* next_ids, int64_t ov_row) {
  if (n <= 0) return;
  const MtpW& w = *m_.mtp();
  const int H = c_.hidden;
  HIVE_CHECK(n <= max_chunk_ && pos0 + n <= s.cap, "qwen MTP entries");
  for (int i = 0; i < n; ++i) std::memcpy(mtp_emb_pin_ + (size_t)i * H, m_.embed_host() + (size_t)next_ids[i] * H, (size_t)H * 2);
  bf16* hc = mtp_h_.as<bf16>();
  const bf16* emb = mtp_emb_pin_;           // pinned rows read directly (UVA)
  bool over = false;
  if (ov_row >= 0)
    for (const EmbedOverride& o : emb_over_) over = over || (o.row < ov_row + n && o.row + o.n > ov_row);
  if (over) {  // image / video pad rows: the vision encoder's rows, as in the main model's embedding
    CUDA_CHECK(cudaMemcpyAsync(mtp_emb_.p, mtp_emb_pin_, (size_t)n * H * 2, cudaMemcpyHostToDevice, stream_));
    apply_embed_override(mtp_emb_.as<bf16>(), ov_row, n);
    emb = mtp_emb_.as<bf16>();
  }
  mtp_input(emb, h_rows, n, hc);
  hc_mix(w.layer.hc_attn, hc, n, x_.as<bf16>());
  const AttnW& a = w.layer.att;
  lin(a.in, x_.as<bf16>(), H, n, proj_.as<bf16>(), kQsaProjWidth);
  QsaRow* qr = qrows_alloc(n);
  for (int i = 0; i < n; ++i) qr[i] = QsaRow{s.mtp.view(s.rope.as<int32_t>()), (int)(pos0 + i)};
  const QsaWeights qw{a.q_norm.as<bf16>(), a.k_norm.as<bf16>(), a.iq_norm.as<bf16>(), a.ik_norm.as<bf16>(), c_.rms_eps};
  const QsaRope rope{rope_inv_freq_.as<float>(), rope_axis_.as<int32_t>()};
  QsaRow* qd = qrows_dev_.as<QsaRow>() + (qr - qrows_pin_);
  CUDA_CHECK(cudaMemcpyAsync(qd, qr, (size_t)n * sizeof(QsaRow), cudaMemcpyHostToDevice, stream_));
  qsa_prep(qw, rope, qd, n, proj_.as<bf16>(), kQsaProjWidth, q_.as<bf16>(), iq_.as<bf16>(), stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));  // mtp_emb_pin_ is rewritten by the next call
  s.mtp_written = std::max(s.mtp_written, pos0 + n);
}

// one pass over the accepted rows of every part (rollback_batch called mtp_kv per part, each ending in a stream sync: post 1.2 ms per
//   step at 4 sequences, 2.0 at 16 — HIVE_PROFILE 2026-10-10); HIVE_QWEN_MTPKV_SERIAL=1 keeps the per-part calls
void QwenEngine::mtp_kv_multi(const MtpKvPart* parts, int P) {
  int n = 0;
  for (int p = 0; p < P; ++p) n += parts[p].n;
  if (n <= 0) return;
  static const bool serial = getenv("HIVE_QWEN_MTPKV_SERIAL") && atoi(getenv("HIVE_QWEN_MTPKV_SERIAL")) > 0;
  if (serial || P == 1 || n > max_chunk_) {
    for (int p = 0; p < P; ++p) mtp_kv(*parts[p].s, parts[p].pos0, parts[p].n, parts[p].h_rows, parts[p].next_ids);
    return;
  }
  const MtpW& w = *m_.mtp();
  const int H = c_.hidden, HC = c_.hcw();
  if (mtp_hin_.n < (size_t)n * HC * 2) mtp_hin_.alloc((size_t)std::max(n, (int)kDecRows) * HC * 2);
  bf16* hin = mtp_hin_.as<bf16>();
  QwenSeq* last = nullptr;
  for (int p = 0, r = 0; p < P; ++p) {
    const MtpKvPart& q = parts[p];
    if (q.n <= 0) continue;
    HIVE_CHECK(q.pos0 + q.n <= q.s->cap && q.s != last, "qwen MTP entries");
    last = q.s;
    for (int i = 0; i < q.n; ++i) std::memcpy(mtp_emb_pin_ + (size_t)(r + i) * H, m_.embed_host() + (size_t)q.next_ids[i] * H, (size_t)H * 2);
    CUDA_CHECK(cudaMemcpyAsync(hin + (size_t)r * HC, q.h_rows, (size_t)q.n * HC * 2, cudaMemcpyDeviceToDevice, stream_));
    r += q.n;
  }
  // passes of ≤ 64 / hc rows (fc_hidden runs n · hc rows: above 64 the GEMV would give way to the prefill GEMM, other arithmetic than the
  //   per-part calls); each pass reads its own rows of the pinned / gathered buffers, so only the end waits
  const AttnW& a = w.layer.att;
  const QsaWeights qw{a.q_norm.as<bf16>(), a.k_norm.as<bf16>(), a.iq_norm.as<bf16>(), a.ik_norm.as<bf16>(), c_.rms_eps};
  const QsaRope rope{rope_inv_freq_.as<float>(), rope_axis_.as<int32_t>()};
  const int cap = std::max(1, 64 / std::max(1, c_.hc));
  for (int p0 = 0, r0 = 0; p0 < P;) {
    int p1 = p0, m = 0;
    while (p1 < P && (m + parts[p1].n <= cap || p1 == p0)) m += parts[p1++].n;
    if (m > 0) {
      bf16* hc = mtp_h_.as<bf16>();
      mtp_input(mtp_emb_pin_ + (size_t)r0 * H, hin + (size_t)r0 * HC, m, hc);
      hc_mix(w.layer.hc_attn, hc, m, x_.as<bf16>());
      lin(a.in, x_.as<bf16>(), H, m, proj_.as<bf16>(), kQsaProjWidth);
      QsaRow* qr = qrows_alloc(m);
      for (int p = p0, r = 0; p < p1; ++p)
        for (int i = 0; i < parts[p].n; ++i, ++r) qr[r] = QsaRow{parts[p].s->mtp.view(parts[p].s->rope.as<int32_t>()), (int)(parts[p].pos0 + i)};
      QsaRow* qd = qrows_dev_.as<QsaRow>() + (qr - qrows_pin_);
      CUDA_CHECK(cudaMemcpyAsync(qd, qr, (size_t)m * sizeof(QsaRow), cudaMemcpyHostToDevice, stream_));
      qsa_prep(qw, rope, qd, m, proj_.as<bf16>(), kQsaProjWidth, q_.as<bf16>(), iq_.as<bf16>(), stream_);
    }
    p0 = p1; r0 += m;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));  // mtp_emb_pin_ is rewritten by the next call
  for (int p = 0; p < P; ++p) if (parts[p].n > 0) parts[p].s->mtp_written = std::max(parts[p].s->mtp_written, parts[p].pos0 + parts[p].n);
}

// MoE of the MTP layer for one row: the router on the device, the 10 chosen FP8 experts as GEMVs (the ids come to the host once).
void QwenEngine::mtp_moe(const MoeW& w, const bf16* x, int rows, bf16* out) {
  HIVE_CHECK(rows == 1 || (mtp_tab_ && rows <= kDraftRows), "qwen MTP MoE: one row (several only on the device path)");
  const MtpW& mt = *m_.mtp();
  const int H = c_.hidden, E = c_.n_experts, K = c_.n_act, I = c_.moe_inter, ld = E + 1;
  float* lg = router_.as<float>();
  float* acc = mlp_out_f32_.as<float>();
  if (mtp_tab_) {   // device path (fused router + shared gate|up, FN FP8 expert kernels with the combine): no host round trip
    const int SI = c_.shared_inter;
    FnEpi e; e.mode = 3; e.yf = lg; e.ldyf = ld; e.nr = ld; e.I = SI; e.ys = sh_y_.as<bf16>();
    fn_gemv_epi(mtp_rs_.as<bf16>(), ld + 2 * SI, H, x, H, rows, e, gemv_ws_.as<float>(), stream_);
    router_softmax_topk(lg, ld, rows, E, K, c_.norm_topk, rids_.as<int32_t>(), rw_.as<float>(), nullptr, nullptr, stream_);
    lin(w.sh_down, sh_y_.as<bf16>(), SI, rows, shared_out_.as<bf16>(), H);
    fn_moe_fp8_routed(mtp_tab_, E, rids_.as<int32_t>(), rw_.as<float>(), rows, K, x, H, I, shared_out_.as<bf16>(), lg, ld, E, out, moe_ws_.p, stream_);
    (void)mt; (void)acc;
    return;
  }
  router_logits(w.router, x, 1, lg, ld);
  router_softmax_topk(lg, ld, 1, E, K, c_.norm_topk, rids_.as<int32_t>(), rw_.as<float>(), rids_h_, rw_h_, stream_);
  moe_shared(w, x, 1, shared_out_.as<bf16>());
  CUDA_CHECK(cudaMemsetAsync(acc, 0, (size_t)H * 4, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  bf16* gu = sh_gu_.as<bf16>();
  bf16* yv = sh_y_.as<bf16>();
  bf16* eo = attn_out_.as<bf16>();
  for (int k = 0; k < K; ++k) {
    const int e = rids_h_[k];
    if (e < 0 || e >= E) continue;
    const auto& m3 = mt.experts[e];
    glm::fp8b_gemv(m3[0], x, H, 1, gu, 2 * I, stream_);
    glm::fp8b_gemv(m3[1], x, H, 1, gu + I, 2 * I, stream_);
    glm::swiglu_rows(gu, 1, I, 0.f, yv, stream_);
    glm::fp8b_gemv(m3[2], yv, I, 1, eo, H, stream_);
    glm::accum_scaled(acc, eo, rw_h_[k], H, stream_);
  }
  moe_combine(acc, shared_out_.as<bf16>(), lg, ld, E, 1, H, out, stream_);
}

// One draft step at entry `pos` (rows = 1). Step 0 embeds `tok` (host); later steps embed the previous draft from the device (mapped table).
void QwenEngine::mtp_step(QwenSeq& s, int step, int32_t tok, const bf16* h_in, int64_t pos, bf16* h_out) {
  const MtpW& w = *m_.mtp();
  const LayerW& L = w.layer;
  const int H = c_.hidden, HC = c_.hcw();
  HIVE_CHECK(pos + 1 <= s.cap, "qwen MTP draft beyond the sequence capacity");
  const bf16* emb;
  // graph phases (mtp_draft): 1 = host part only, 2 = device part only
  if (step == 0) {
    if (gphase_ != 2) std::memcpy(mtp_emb_pin_, m_.embed_host() + (size_t)tok * H, (size_t)H * 2);
    emb = mtp_emb_pin_;
  } else {
    if (gphase_ != 1) embed_rows(m_.embed_host(), mtp_ids_d_.as<int32_t>() + step - 1, 1, H, mtp_emb_.as<bf16>(), stream_);
    emb = mtp_emb_.as<bf16>();
  }
  const std::vector<RowInfo> rows{{&s, pos}};
  mtp_slot_ = step;
  // rope coordinates of the draft position (a scratch entry past the processed tokens gets the text coordinate it will have)
  int32_t* o = rope_pin_ + (size_t)step * 3;
  if (gphase_ != 2) {
    o[0] = o[1] = o[2] = (int32_t)(pos + s.rope_delta);
    rope_dst_pin_[step] = s.rope.as<int32_t>() + (size_t)pos * 3;
  }
  if (gphase_ == 1) { attn_layer(-1, rows, nullptr, nullptr, true); return; }
  bf16* hc = mtp_h_.as<bf16>();
  mtp_input(emb, h_in, 1, hc);
  write_rope(o, rope_dst_pin_ + step, 1, stream_);
  hc_mix(L.hc_attn, hc, 1, x_.as<bf16>());
  attn_layer(-1, rows, x_.as<bf16>(), y_.as<bf16>(), true);
  hc_combine_(L.hc_attn, hc, y_.as<bf16>(), 1, dl_.as<bf16>());
  hc_mix(L.hc_mlp, hc, 1, x_.as<bf16>());
  mtp_moe(L.moe, x_.as<bf16>(), 1, y_.as<bf16>());
  hc_combine_(L.hc_mlp, hc, y_.as<bf16>(), 1, dl_.as<bf16>());
  CUDA_CHECK(cudaMemcpyAsync(h_out, hc, (size_t)HC * 2, cudaMemcpyDeviceToDevice, stream_));
  hc_mix(w.mixer, hc, 1, xfin_.as<bf16>());
  float* lg = logits_dev_.as<float>();
  if (draft_q_.p) {   // FP8 draft head (all rows, or the HIVE_QWEN_DRAFT_VOCAB rows)
    const int V = draft_n_ > 0 ? draft_n_ : c_.vocab;
    const glm::Fp8BMat W{draft_q_.as<uint8_t>(), draft_s_.as<float>(), V, H};
    glm::fp8b_gemv(W, xfin_.as<bf16>(), H, 1, draft_lg_.as<bf16>(), V, stream_);
    fn_draft_pick(draft_lg_.as<bf16>(), V, draft_n_ > 0 ? draft_rows_.as<int32_t>() : nullptr, mtp_ids_d_.as<int32_t>() + step,
                  mtp_conf_d_.as<float>() + 2 * step, draft_part_.as<float>(), stream_);
    return;
  } else if (draft_n_ > 0) {
    glm::gemv_rows_bf16(m_.lm_head().p(), draft_rows_.as<int32_t>(), draft_n_, xfin_.as<bf16>(), H, lg, stream_);
    k::argmax_rows(lg, 1, draft_n_, draft_idx_.as<int32_t>() + step, stream_);
    glm::map_id_dev(draft_idx_.as<int32_t>() + step, draft_rows_.as<int32_t>(), mtp_ids_d_.as<int32_t>() + step, stream_);
    k::row_max_sumexp(lg, 1, draft_n_, one_f_.as<float>(), mtp_conf_d_.as<float>() + 2 * step, mtp_conf_d_.as<float>() + 2 * step + 1, stream_);
    return;
  } else {
    blas_.gemm_bf16_f32out(xfin_.as<bf16>(), m_.lm_head().p(), lg, 1, c_.vocab, H);
  }
  k::argmax_rows(lg, 1, c_.vocab, mtp_ids_d_.as<int32_t>() + step, stream_);
  k::row_max_sumexp(lg, 1, c_.vocab, one_f_.as<float>(), mtp_conf_d_.as<float>() + 2 * step, mtp_conf_d_.as<float>() + 2 * step + 1, stream_);
}

int QwenEngine::mtp_draft(QwenSeq& s, int32_t tok, int k, int32_t* drafts, float* conf) {
  if (!mtp_on() || !s.h_last_valid || k <= 0) return 0;
  k = std::min(k, mtp_k_);
  const int64_t p = s.pos - 1;  // entry of (h_last, tok)
  if (p < 0 || p + k + 1 > s.cap) return 0;
  kv_reserve(s, std::min<int64_t>(s.cap, s.pos + k + kKvSlack));
  const int HC = c_.hcw();
  auto chain = [&] {
    const bf16* hin = s.h_last.as<bf16>();
    for (int i = 0; i < k; ++i) {
      bf16* hout = mtp_hout_.as<bf16>() + (size_t)(i & 1) * HC;
      mtp_step(s, i, tok, hin, p + i, hout);
      hin = hout;
    }
  };
  // CUDA graph of the draft chain (needs the device MoE path): host part, then capture once per shape / replay
  static const bool graphs = [] { const char* v = getenv("HIVE_QWEN_GRAPH"); return !(v && strcmp(v, "0") == 0); }();
  bool done = false;
  if (graphs && mtp_tab_ && !layer_hook && !layer_inject) {
    const bool scoring = (p + k) / kQsaRatio > kQsaTopBlocks;
    const int ctx = scoring ? (int)((p + k + 4095) / 4096 * 4096) : 0;
    const std::string key = "mtp" + std::to_string(k) + ":" + std::to_string(scoring) + ":" + std::to_string(ctx) + ":" + std::to_string((uintptr_t)s.h_last.p);
    graph_ctx_ = ctx;
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    gphase_ = 1;
    chain();
    gphase_ = 0;
    auto it = graphs_.find(key);
    if (it == graphs_.end() && graph_seen_[key]++ >= 1) {
      gphase_ = 2;
      cudaGraph_t g = nullptr;
      CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
      try { chain(); } catch (...) {
        cudaGraph_t bad = nullptr; cudaStreamEndCapture(stream_, &bad); if (bad) cudaGraphDestroy(bad); cudaGetLastError(); gphase_ = 0; throw;
      }
      gphase_ = 0;
      CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
      cudaGraphExec_t ex = nullptr;
      CUDA_CHECK(cudaGraphInstantiate(&ex, g, 0));
      CUDA_CHECK(cudaGraphDestroy(g));
      it = graph_put(key, ex);
    }
    if (it != graphs_.end()) { graph_used(key); CUDA_CHECK(cudaGraphLaunch(it->second, stream_)); done = true; }
  }
  if (!done) chain();
  int32_t ids[8]; float ms[16];
  CUDA_CHECK(cudaMemcpyAsync(ids, mtp_ids_d_.p, (size_t)k * 4, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(ms, mtp_conf_d_.p, (size_t)k * 8, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  for (int i = 0; i < k; ++i) { drafts[i] = ids[i]; conf[i] = ms[2 * i + 1] > 0.f ? 1.f / ms[2 * i + 1] : 0.f; }  // p(argmax) = 1 / Σ exp(l − max)
  s.mtp_written = std::max(s.mtp_written, p + 1);  // the first entry used the real hidden: exact
  return k;
}

// One draft step for S sequences (row r = sequence r at entry pos[r]); step 0 embeds toks (host), later steps the previous step's drafts
//   (device). h_in / h_out: [S, HC]. FP8 draft head (draft_q_) only. The same per-row arithmetic as mtp_step with rows = 1.
void QwenEngine::mtp_step_rows(QwenSeq* const* seqs, int S, int step, const int32_t* toks, const int64_t* pos, const bf16* h_in, bf16* h_out) {
  const MtpW& w = *m_.mtp();
  const LayerW& L = w.layer;
  const int H = c_.hidden, HC = c_.hcw();
  const bf16* emb;
  if (step == 0) {
    if (gphase_ != 2) for (int r = 0; r < S; ++r) std::memcpy(mtp_emb_pin_ + (size_t)r * H, m_.embed_host() + (size_t)toks[r] * H, (size_t)H * 2);
    emb = mtp_emb_pin_;
  } else {
    if (gphase_ != 1) embed_rows(m_.embed_host(), mtpb_ids_d_.as<int32_t>() + (size_t)(step - 1) * kDraftRows, S, H, mtp_emb_.as<bf16>(), stream_);
    emb = mtp_emb_.as<bf16>();
  }
  std::vector<RowInfo> rows(S);
  for (int r = 0; r < S; ++r) rows[r] = RowInfo{seqs[r], pos[r]};
  mtp_slot_ = step;
  int32_t* o = rope_pin_ + (size_t)step * S * 3;
  if (gphase_ != 2)
    for (int r = 0; r < S; ++r) {
      HIVE_CHECK(pos[r] + 1 <= seqs[r]->cap, "qwen MTP draft beyond the sequence capacity");
      o[r * 3] = o[r * 3 + 1] = o[r * 3 + 2] = (int32_t)(pos[r] + seqs[r]->rope_delta);
      rope_dst_pin_[step * S + r] = seqs[r]->rope.as<int32_t>() + (size_t)pos[r] * 3;
    }
  if (gphase_ == 1) { attn_layer(-1, rows, nullptr, nullptr, true); return; }
  bf16* hc = mtp_h_.as<bf16>();
  mtp_input(emb, h_in, S, hc);
  write_rope(o, rope_dst_pin_ + step * S, S, stream_);
  hc_mix(L.hc_attn, hc, S, x_.as<bf16>());
  attn_layer(-1, rows, x_.as<bf16>(), y_.as<bf16>(), true);
  hc_combine_(L.hc_attn, hc, y_.as<bf16>(), S, dl_.as<bf16>());
  hc_mix(L.hc_mlp, hc, S, x_.as<bf16>());
  mtp_moe(L.moe, x_.as<bf16>(), S, y_.as<bf16>());
  hc_combine_(L.hc_mlp, hc, y_.as<bf16>(), S, dl_.as<bf16>());
  CUDA_CHECK(cudaMemcpyAsync(h_out, hc, (size_t)S * HC * 2, cudaMemcpyDeviceToDevice, stream_));
  hc_mix(w.mixer, hc, S, xfin_.as<bf16>());
  const int V = draft_n_ > 0 ? draft_n_ : c_.vocab;
  const glm::Fp8BMat W{draft_q_.as<uint8_t>(), draft_s_.as<float>(), V, H};
  for (int r0 = 0; r0 < S; r0 += 8)
    glm::fp8b_gemv(W, xfin_.as<bf16>() + (size_t)r0 * H, H, std::min(8, S - r0), draft_lg_.as<bf16>() + (size_t)r0 * V, V, stream_);
  const int np = fn_draft_pick_parts(V);
  for (int r = 0; r < S; ++r)
    fn_draft_pick(draft_lg_.as<bf16>() + (size_t)r * V, V, draft_n_ > 0 ? draft_rows_.as<int32_t>() : nullptr,
                  mtpb_ids_d_.as<int32_t>() + (size_t)step * kDraftRows + r, mtpb_conf_d_.as<float>() + ((size_t)step * kDraftRows + r) * 2,
                  draft_part_.as<float>() + (size_t)r * np * 3, stream_);
}

void QwenEngine::mtp_draft_batch(QwenSeq* const* seqs, const int32_t* toks, int S, int k, int32_t* drafts, float* conf, int* n_out) {
  for (int b = 0; b < S; ++b) n_out[b] = 0;
  if (!mtp_on() || k <= 0 || S <= 0) return;
  k = std::min(k, mtp_k_);
  // sequences the chain can serve together: hidden state present, room for k entries; the rest (and a non-FP8 head) take mtp_draft
  std::vector<int> use;
  for (int b = 0; b < S; ++b) {
    QwenSeq& s = *seqs[b];
    const int64_t p = s.pos - 1;
    if (!s.h_last_valid || p < 0 || p + k + 1 > s.cap) continue;
    use.push_back(b);
  }
  if (!draft_q_.p || !mtp_tab_ || use.size() < 2) {
    for (int b : use) n_out[b] = mtp_draft(*seqs[b], toks[b], k, drafts + (size_t)b * k, conf + (size_t)b * k);
    return;
  }
  for (size_t u0 = 0; u0 < use.size(); u0 += kDraftRows) {
    const int n = (int)std::min<size_t>(kDraftRows, use.size() - u0);
    std::vector<QwenSeq*> sq(n);
    std::vector<int32_t> tk(n);
    std::vector<int64_t> p0(n);
    for (int r = 0; r < n; ++r) {
      QwenSeq& s = *seqs[use[u0 + r]];
      sq[r] = &s; tk[r] = toks[use[u0 + r]]; p0[r] = s.pos - 1;
      kv_reserve(s, std::min<int64_t>(s.cap, s.pos + k + kKvSlack));
    }
    const int HC = c_.hcw();
    bf16* hb0 = mtpb_h_.as<bf16>();   // [0]: the sequences' h_last rows, [1] / [2]: step outputs
    for (int r = 0; r < n; ++r) CUDA_CHECK(cudaMemcpyAsync(hb0 + (size_t)r * HC, sq[r]->h_last.p, (size_t)HC * 2, cudaMemcpyDeviceToDevice, stream_));
    for (int i = 0; i < k; ++i) {
      std::vector<int64_t> pp(n);
      for (int r = 0; r < n; ++r) pp[r] = p0[r] + i;
      const bf16* hin = i == 0 ? hb0 : hb0 + (size_t)(1 + ((i - 1) & 1)) * kDraftRows * HC;
      bf16* hout = hb0 + (size_t)(1 + (i & 1)) * kDraftRows * HC;
      int max_pos = 0;
      for (int r = 0; r < n; ++r) max_pos = std::max<int>(max_pos, (int)pp[r]);
      const bool scoring = (max_pos + 1) / kQsaRatio > kQsaTopBlocks;
      graph_ctx_ = scoring ? (max_pos + 1 + 4095) / 4096 * 4096 : 0;
      gphase_ = 0;
      mtp_step_rows(sq.data(), n, i, tk.data(), pp.data(), hin, hout);
    }
    std::vector<int32_t> ids((size_t)k * kDraftRows);
    std::vector<float> ms((size_t)2 * k * kDraftRows);
    CUDA_CHECK(cudaMemcpyAsync(ids.data(), mtpb_ids_d_.p, ids.size() * 4, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(ms.data(), mtpb_conf_d_.p, ms.size() * 4, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (int r = 0; r < n; ++r) {
      const int b = use[u0 + r];
      for (int i = 0; i < k; ++i) {
        drafts[(size_t)b * k + i] = ids[(size_t)i * kDraftRows + r];
        const float sum = ms[((size_t)i * kDraftRows + r) * 2 + 1];
        conf[(size_t)b * k + i] = sum > 0.f ? 1.f / sum : 0.f;
      }
      n_out[b] = k;
      sq[r]->mtp_written = std::max(sq[r]->mtp_written, p0[r] + 1);
    }
  }
}

void QwenEngine::verify_batch(QwenSeq* const* seqs, const int* Ms, int S, const int32_t* ids, float* logits_host) {
  HIVE_CHECK(S >= 1 && S <= snap_parts_, "qwen verify: more parts than snapshots (enable_batch_verify)");
  ensure_parts(S);
  if (S > 1) snap_idle_ = 0;
  else if (snap_live_ > 1 && ++snap_idle_ >= kSnapIdle) release_parts(1);
  int R = 0;
  vparts_.clear();
  std::vector<RowInfo> rows;
  for (int p = 0; p < S; ++p) {
    const int M = Ms[p];
    HIVE_CHECK(M >= 1 && R + M <= kDecRows && M - 1 <= std::max(1, mtp_k_), "qwen verify rows");
    HIVE_CHECK(seqs[p]->pos + M <= seqs[p]->cap, "qwen: sequence capacity exceeded");
    for (int q = 0; q < p; ++q) HIVE_CHECK(seqs[q] != seqs[p], "qwen verify: a sequence appears in two parts");
    kv_reserve(*seqs[p], std::min<int64_t>(seqs[p]->cap, seqs[p]->pos + M + kKvSlack));
    vparts_.push_back({seqs[p], seqs[p]->pos, M, R});
    for (int r = 0; r < M; ++r) rows.push_back({seqs[p], seqs[p]->pos + r});
    R += M;
  }
  verify_ids_.assign(ids, ids + R);
  verify_ = true;
  try {
    forward(rows, ids, false, logits_host, true);
  } catch (...) { verify_ = false; vparts_.clear(); throw; }
  verify_ = false;
  ex_.after_step(R);
}

void QwenEngine::rollback_batch(const int* n_keep, int S) {
  HIVE_CHECK(S == (int)vparts_.size() && S >= 1, "qwen rollback without a matching verify");
  const int HC = c_.hcw(), kmax = std::max(1, mtp_k_);
  struct Replay { int p, nk, r0; };
  std::vector<Replay> replays;
  std::vector<MtpKvPart> kv_parts;   // the accepted rows' MTP entries, all parts in one mtp_kv_multi below
  for (int p = 0; p < S; ++p) {
    const VPart& vp = vparts_[p];
    QwenSeq& s = *vp.s;
    const int M = vp.M, nk = n_keep[p];
    HIVE_CHECK(nk >= 1 && nk <= M && s.pos == vp.pos0 + M, "qwen rollback without a matching verify");
    if (nk < M && replay_part(p)) {
      // replay: the state before the verify (snapshot slot 0) becomes the sequence's state, then rows 0 .. nk−1 are run again from the saved
      //   projections (same kernel, same arithmetic — bit-identical to the verify's state after row nk−1); the conv window after row nk−1
      //   is a snapshot as on the per-row path
      const size_t s0 = (size_t)p * kmax;
      const size_t si = (size_t)p * kmax + nk - 1;
      float** rb = rb_pin_ + (size_t)p * n_lin_ * 2;
      for (int li = 0; li < n_lin_; ++li) {
        std::swap(s.state[li], snap_state_[s0 * n_lin_ + li]);
        snap_ptr_pin_[((size_t)p * n_lin_ + li) * 16 + 8] = snap_state_[s0 * n_lin_ + li].as<float>();
        rb[li * 2] = snap_conv0_[(size_t)p * n_lin_ + li].as<float>();
        rb[li * 2 + 1] = s.state[li].as<float>();
      }
      replays.push_back({p, nk, vp.r0});   // run below, all replayed parts together (the replay writes only the pointers captured above)
      for (int li = 0; li < n_lin_; ++li) {
        std::swap(s.conv[li], snap_conv_[si * n_lin_ + li]);
        snap_ptr_pin_[((size_t)p * n_lin_ + li) * 16 + (nk - 1)] = snap_conv_[si * n_lin_ + li].as<float>();
      }
      if (s.ple_state.p) glm::dcopy(s.ple_state.p, snap_ple_[si].p, s.ple_state.n, stream_);
    } else if (nk < M) {
      const size_t si = (size_t)p * kmax + nk - 1;
      // the snapshot after row nk−1 becomes the sequence's state by exchanging the buffers (36 × 3 MiB of copies per rejected step
      //   before); the snapshot slot takes the old buffer and the pinned pointer table follows (the graphs read it at run time)
      for (int li = 0; li < n_lin_; ++li) {
        std::swap(s.state[li], snap_state_[si * n_lin_ + li]);
        std::swap(s.conv[li], snap_conv_[si * n_lin_ + li]);
        snap_ptr_pin_[((size_t)p * n_lin_ + li) * 16 + (nk - 1)] = snap_conv_[si * n_lin_ + li].as<float>();
        snap_ptr_pin_[((size_t)p * n_lin_ + li) * 16 + 8 + (nk - 1)] = snap_state_[si * n_lin_ + li].as<float>();
      }
      if (s.ple_state.p) glm::dcopy(s.ple_state.p, snap_ple_[si].p, s.ple_state.n, stream_);
    }
    s.pos = vp.pos0 + nk;
    s.tokens.resize(s.tokens.size() - (size_t)(M - nk));
    if (mtp_on()) {
      if (nk >= 2) kv_parts.push_back({&s, vp.pos0, nk - 1, fh_.as<bf16>() + (size_t)vp.r0 * HC, verify_ids_.data() + vp.r0 + 1});
      CUDA_CHECK(cudaMemcpyAsync(s.h_last.p, fh_.as<bf16>() + (size_t)(vp.r0 + nk - 1) * HC, (size_t)HC * 2, cudaMemcpyDeviceToDevice, stream_));
      s.h_last_valid = true;
    }
  }
  mtp_kv_multi(kv_parts.data(), (int)kv_parts.size());
  if (!replays.empty()) {
    // rows 0 .. nk−1 of every replayed part again from the saved projections, one launch pair per linear layer for all of them (was one
    //   per part per layer plus a stream sync per part)
    CUDA_CHECK(cudaMemcpyAsync(rb_dev_.p, rb_pin_, (size_t)S * n_lin_ * 2 * sizeof(float*), cudaMemcpyHostToDevice, stream_));
    float** rd = rb_dev_.as<float*>();
    std::vector<GdnRowsPart> gp_parts(replays.size());
    for (int l = 0; l < c_.n_layers; ++l) {
      const int li = l < (int)lin_index_.size() ? lin_index_[l] : -1;
      if (li < 0) continue;
      const LinW& w = m_.layer(l).lin;
      const GdnParams gp{w.conv_w.as<float>(), w.A_log.as<float>(), w.dt_bias.as<float>(), w.norm.as<bf16>(), c_.rms_eps};
      for (size_t q = 0; q < replays.size(); ++q) {
        const Replay& rp = replays[q];
        float** rb = rd + (size_t)rp.p * n_lin_ * 2;
        gp_parts[q] = GdnRowsPart{proj_save_.as<bf16>() + ((size_t)li * kDecRows + rp.r0) * kGdnProjWidth, rp.nk, rb + li * 2, rb + li * 2 + 1,
                                  nullptr, nullptr, gdn_out_.as<bf16>() + (size_t)rp.r0 * kGdnVWidth, 0, nullptr};
      }
      gdn_decode_rows_multi(gp, kGdnProjWidth, (int)gp_parts.size(), gp_parts.data(), stream_);
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  vparts_.clear();
}

void QwenEngine::mtp_after_forward(const std::vector<RowInfo>& rows, const int32_t* ids, bool prefill) {
  const int HC = c_.hcw(), R = (int)rows.size();
  if (verify_) {  // per part: the entry of the position before the part (h_last ↔ its first token); the rows are entered by rollback
    for (const VPart& vp : vparts_) {
      QwenSeq& s = *vp.s;
      if (s.h_last_valid && vp.pos0 > 0 && s.mtp_written < vp.pos0) mtp_kv(s, vp.pos0 - 1, 1, s.h_last.as<bf16>(), ids + vp.r0);
    }
  } else if (prefill) {
    QwenSeq& s = *rows[0].s;
    const int64_t p0 = rows[0].pos;
    if (s.h_last_valid && p0 > 0 && s.mtp_written < p0) mtp_kv(s, p0 - 1, 1, s.h_last.as<bf16>(), ids, emb_base_);
    if (R > 1) mtp_kv(s, p0, R - 1, fh_.as<bf16>(), ids + 1, emb_base_ + 1);
    CUDA_CHECK(cudaMemcpyAsync(s.h_last.p, fh_.as<bf16>() + (size_t)(R - 1) * HC, (size_t)HC * 2, cudaMemcpyDeviceToDevice, stream_));
    s.h_last_valid = true;
  } else {
    for (int m = 0; m < R; ++m) {
      QwenSeq& s = *rows[m].s;
      const int64_t p0 = rows[m].pos;
      if (s.h_last_valid && p0 > 0 && s.mtp_written < p0) mtp_kv(s, p0 - 1, 1, s.h_last.as<bf16>(), ids + m);
      CUDA_CHECK(cudaMemcpyAsync(s.h_last.p, fh_.as<bf16>() + (size_t)m * HC, (size_t)HC * 2, cudaMemcpyDeviceToDevice, stream_));
      s.h_last_valid = true;
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace hive::qwen
