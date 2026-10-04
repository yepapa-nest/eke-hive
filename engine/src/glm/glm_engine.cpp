// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/glm/glm_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "hive/glm/glm_kernels.h"
#include "hive/glm/glm_moe_fp8.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"

namespace hive::glm {
using namespace hive::k;

namespace {
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// C[M, N] at column offset (row stride ldc) = A[M, K] · W[N, K]ᵀ
void gemm_into(Blas& b, const bf16* A, int lda, const bf16* W, bf16* C, int ldc, int M, int N, int K) {
  b.gemm_bf16_batched_ld(A, lda, 0, W, K, 0, C, ldc, 0, M, N, K, 1);
}
}  // namespace

void GlmEngine::prof_mark(double* acc) {
  if (!prof_) return;
  if (prof_ == 2) {
    if (ev_marks_.size() >= ev_pool_.size()) { cudaEvent_t e; CUDA_CHECK(cudaEventCreate(&e)); ev_pool_.push_back(e); }
    cudaEvent_t e = ev_pool_[ev_marks_.size()];
    CUDA_CHECK(cudaEventRecord(e, stream_));
    ev_marks_.push_back({e, acc});
    return;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  const double t = now_ms();
  if (acc) *acc += t - prof_t_;
  prof_t_ = t;
}

// HIVE_GLM_PROF=2: the time between consecutive events goes to the category named by the later mark
void GlmEngine::prof_flush() {
  if (prof_ != 2 || ev_marks_.empty()) return;
  CUDA_CHECK(cudaEventSynchronize(ev_marks_.back().first));
  for (size_t i = 1; i < ev_marks_.size(); ++i) {
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev_marks_[i - 1].first, ev_marks_[i].first));
    if (ev_marks_[i].second) *ev_marks_[i].second += ms;
  }
  ev_marks_.clear();
}

GlmEngine::GlmEngine(GlmModel& m, GlmExperts& x, int max_chunk) : m_(m), ex_(x), c_(m.cfg()), blas_(nullptr), max_chunk_(max_chunk) {
  prof_ = getenv("HIVE_GLM_PROF") ? atoi(getenv("HIVE_GLM_PROF")) : 0;
  pred_eval_ = getenv("HIVE_GLM_PREDICT_EVAL") ? atoi(getenv("HIVE_GLM_PREDICT_EVAL")) : 0;
  pf_n_ = getenv("HIVE_GLM_PREFETCH") ? std::max(0, atoi(getenv("HIVE_GLM_PREFETCH"))) : 0;
  probe_every_ = getenv("HIVE_GLM_LAUNCH_PROBE") ? std::max(0, atoi(getenv("HIVE_GLM_LAUNCH_PROBE"))) : 0;
  if (const char* cp = getenv("HIVE_CACHE_PRIOR")) cache_prior_ = std::max(0.f, (float)atof(cp));
  if (ex_.defer_enabled()) post_def_.alloc((size_t)8 * c_.hc * 4);
  if (const char* cj = getenv("HIVE_CACHE_PRIOR_TOPJ")) cache_prior_j_ = std::min(4, std::max(0, atoi(cj)));
  if (cache_prior_ > 0.f) {
    CUDA_CHECK(cudaMallocHost(&cache_mask_h_, (size_t)c_.n_moe * c_.n_routed));
    std::memset(cache_mask_h_, 0, (size_t)c_.n_moe * c_.n_routed);
    prior_range_.alloc((size_t)c_.n_moe * sizeof(float));
    CUDA_CHECK(cudaMemset(prior_range_.p, 0, (size_t)c_.n_moe * sizeof(float)));
    fprintf(stderr, "[glm] cache prior on: λ %.3f · top-%d always kept (decode routing biased toward VRAM-resident experts)\n", cache_prior_, cache_prior_j_);
  }
  if (probe_every_ > 0) {
    for (auto& e : probe_ev_) CUDA_CHECK(cudaEventCreate(&e));
  }
  fused_ = !(getenv("HIVE_GLM_DECODE_FUSED") && strcmp(getenv("HIVE_GLM_DECODE_FUSED"), "0") == 0);  // default on; "0" = off
  hc_sync_.alloc(kGlmDecodeSyncInts * 4); CUDA_CHECK(cudaMemset(hc_sync_.p, 0, kGlmDecodeSyncInts * 4));
  rlog_.alloc(glm_router_logits_floats(8, c_.n_routed, c_.hidden) * 4);
  CUDA_CHECK(cudaHostAlloc(&rids_h_, 8 * c_.n_act * 4, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc(&rw_h_, 8 * c_.n_act * 4, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc(&pf_ids_map_, 8 * c_.n_act * 4, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc(&pf_w_map_, 8 * c_.n_act * 4, cudaHostAllocMapped));
  next_moe_.assign(c_.n_layers, -1);
  for (int l = 0; l < c_.n_layers; ++l)
    for (int t = l + 1; t < c_.n_layers; ++t) if (c_.is_moe[t]) { next_moe_[l] = t; break; }
  if (pf_n_ > 0 && !fused_) {  // stacked router gates [gate(l) ; gate(next MoE layer)] per MoE layer — only the unfused path uses them
    pair_gate_.resize(c_.n_moe);
    const size_t gb = (size_t)c_.n_routed * c_.hidden * 2;
    for (int l = 0; l < c_.n_layers; ++l) {
      if (!c_.is_moe[l] || next_moe_[l] < 0) continue;
      DevBuf& g = pair_gate_[c_.moe_index[l]];
      g.alloc(2 * gb);
      CUDA_CHECK(cudaMemcpy(g.p, m_.layer(l).moe_w.gate.p(), gb, cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy((uint8_t*)g.p + gb, m_.layer(next_moe_[l]).moe_w.gate.p(), gb, cudaMemcpyDeviceToDevice));
    }
    fprintf(stderr, "[glm] prediction prefetch (unfused path): stacked router gates %.0f MiB\n", 2.0 * gb * c_.n_moe / 1048576.0);
  }
  if (pf_n_ > 0) {
    pf_ids_d_.alloc((size_t)max_chunk * c_.n_act * 4); pf_w_d_.alloc((size_t)max_chunk * c_.n_act * 4);
    fprintf(stderr, "[glm] prediction prefetch: up to %d experts per MoE layer\n", pf_n_);
  }
  if (pred_eval_) {
    pred_logits_.alloc((size_t)8 * c_.n_routed * 4); pred_ids_d_.alloc((size_t)8 * c_.n_act * 4); pred_w_d_.alloc((size_t)8 * c_.n_act * 4);
    for (auto& v : pred_) v.assign(c_.n_moe, {});
    for (auto& v : pred_nonres_) v.assign(c_.n_moe, {});
    p_mix_.alloc(8 * 24 * 4); p_rsq_.alloc(8 * 4); p_pre_.alloc(8 * 4 * 4); p_post_.alloc(8 * 4 * 4); p_comb_.alloc(8 * 16 * 4);
    p_x_.alloc((size_t)8 * c_.hidden * 2); p_xn_.alloc((size_t)8 * c_.hidden * 2);
  }
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  blas_.set_stream(stream_);
  kv_base_ = max_chunk;
  if (const char* v = getenv("HIVE_GLM_KV_BASE")) { const long long x = atoll(v); if (x >= 0 && *v) kv_base_ = x; }
  if (const char* v = getenv("HIVE_GLM_KV_STEP")) { const long long x = atoll(v); if (x > 0) kv_step_ = x; }
  kv_headroom_ = cache_reserve_bytes();
  kda_index_.assign(c_.n_layers, -1); dsa_index_.assign(c_.n_layers, -1);
  for (int l = 0; l < c_.n_layers; ++l) { if (c_.is_kda[l]) kda_index_[l] = n_kda_++; else dsa_index_[l] = n_dsa_++; }
  CUDA_CHECK(cudaMallocHost(&kda_ptr_h_, (size_t)2 * n_kda_ * 8 * sizeof(float*) + 64));
  CUDA_CHECK(cudaMallocHost(&dsa_rows_h_, (size_t)n_dsa_ * 8 * sizeof(DsaDecodeRow)));
  CUDA_CHECK(cudaMallocHost(&emb_pin_, (size_t)8 * c_.hidden * 2));
  CUDA_CHECK(cudaMallocHost(&xin_pin_, (size_t)64 * c_.hidden * 2));
  { int32_t io[8] = {0, 1, 2, 3, 4, 5, 6, 7}; iota_.alloc(32); CUDA_CHECK(cudaMemcpy(iota_.p, io, 32, cudaMemcpyHostToDevice)); }
  const size_t R = max_chunk, H = c_.hidden, HC = (size_t)c_.hc * H;
  auto al = [](DevBuf& b, size_t bytes) { b.alloc(std::max<size_t>(bytes, 256)); };
  al(h_, R * HC * 2); al(x_, R * H * 2); al(xn_, R * H * 2); al(attn_, R * H * 2);
  // fp32 copy of h for the cuBLAS hc reference path only (HIVE_GLM_HC_CUBLAS) — 1 GiB at a 16K chunk
  if (getenv("HIVE_GLM_HC_CUBLAS") && atoi(getenv("HIVE_GLM_HC_CUBLAS")) > 0) al(mix_f32_, R * HC * 4);
  al(mixes_, R * 24 * 4); al(rsq_, R * 4); al(pre_, R * 4 * 4); al(post_, R * 4 * 4); al(comb_, R * 16 * 4);
  const size_t W = (size_t)c_.kda_heads * c_.kda_dim;  // 8192
  al(qkv_, R * 3 * W * 2); al(gpre_, R * W * 2); al(gate_, R * W * 2); al(beta_, R * c_.kda_heads * 2); al(fa_, R * 128 * 2); al(ga_, R * 128 * 2);
  al(kda_out_, R * W * 2);
  al(qa_, R * c_.q_lora * 2); al(qres_, R * c_.q_lora * 2); al(q_, R * c_.n_heads * c_.qk_head * 2); al(ckv_, R * c_.kv_lora * 2); al(cn_, R * c_.kv_lora * 2);
  al(qI_, R * c_.index_heads * c_.index_dim * 2); al(kI_raw_, R * c_.index_dim * 2); al(kI_, R * c_.index_dim * 2); al(gs_, R * c_.index_dim * 2);
  al(wI_, R * c_.index_heads * 4); al(idx_, R * (c_.index_topk + 3) * 4); al(dsa_out_, R * c_.n_heads * c_.v_head * 2);
  al(mlp_out_f32_, R * H * 4); al(mlp_out_, R * H * 2); al(sh_gu_, R * 2 * c_.moe_inter * 2); al(sh_y_, R * c_.moe_inter * 2);
  al(dense_gu_, R * 2 * c_.dense_inter * 2); al(dense_y_, R * c_.dense_inter * 2);
  al(router_logits_, R * 2 * c_.n_routed * 4); al(router_ids_, R * c_.n_act * 4); al(router_w_, R * c_.n_act * 4);
  al(deq_, std::max<size_t>({(size_t)c_.dense_inter * H, (size_t)c_.n_heads * c_.qk_head * c_.q_lora, (size_t)H * c_.n_heads * c_.v_head}) * 2);
  ws_bytes_ = std::max({kda_workspace_bytes(max_chunk), dsa_select_ws_bytes(max_chunk, 262144), dsa_attention_ws_bytes(max_chunk),
                        dsa_select_decode_ws_bytes(8, 262144), dsa_attention_decode_ws_bytes(8)});
  al(ws_, ws_bytes_);
  al(logits_dev_, (size_t)64 * c_.vocab * 4); al(final_x_, 64 * H * 2); al(final_xn_, 64 * H * 2);
  // layer-major prefill buffers for up to tiles × max_chunk rows (HIVE_GLM_PREFILL_TILES, default 4; 1 = off)
  tiles_ = std::max(1, getenv("HIVE_GLM_PREFILL_TILES") ? atoi(getenv("HIVE_GLM_PREFILL_TILES")) : 4);
  elastic_ = !(getenv("HIVE_GLM_PREFILL_ELASTIC") && strcmp(getenv("HIVE_GLM_PREFILL_ELASTIC"), "0") == 0);
  if (tiles_ > 1 && elastic_) {
    fprintf(stderr, "[glm] layer-major prefill: up to %zu rows per call (%d blocks of %d) · buffers borrowed from the expert cache tail while it runs\n",
            (size_t)tiles_ * R, tiles_, max_chunk);
  } else if (tiles_ > 1) {
    const size_t RA = (size_t)tiles_ * R;
    al(hall_, RA * HC * 2); al(xnall_, RA * H * 2); al(outall_, RA * H * 4);
    al(postall_, RA * c_.hc * 4); al(comball_, RA * c_.hc * c_.hc * 4);
    al(rlog_all_, RA * c_.n_routed * 4); al(rids_all_, RA * c_.n_act * 4); al(rw_all_, RA * c_.n_act * 4);
    fprintf(stderr, "[glm] layer-major prefill: up to %zu rows per call (%d blocks of %d) · %.2f GiB\n", RA, tiles_, max_chunk,
            (double)(hall_.n + xnall_.n + outall_.n + postall_.n + comball_.n + rlog_all_.n + rids_all_.n + rw_all_.n) / 1073741824.0);
  }
  // MTP (HIVE_GLM_MTP_K drafts per step, default 3; 0 = off) — needs the NextN layer (GlmModel load_mtp + FP8 experts)
  mtp_k_ = m_.mtp() ? (getenv("HIVE_GLM_MTP_K") ? std::max(0, std::min(7, atoi(getenv("HIVE_GLM_MTP_K")))) : 3) : 0;
  if (m_.mtp()) {
    al(fx_, R * H * 2); al(fh_, R * H * 2);
    CUDA_CHECK(cudaMallocHost(&mtp_emb_pin_, R * H * 2));
    al(mtp_en_, R * H * 2); al(mtp_hn_, R * H * 2); al(mtp_cat_, R * 2 * H * 2); al(mtp_x0_, R * H * 2); al(mtp_xn_, R * H * 2);
    al(mtp_h1_, 8 * H * 2); al(mtp_tmp_, 8 * H * 2); al(mtp_hout_, 2 * H * 2);
    const int kmax = std::max(1, mtp_k_);
    alloc_snapshots(1);
    al(mtp_ids_d_, 16 * 4); al(mtp_conf_d_, 32 * 4);
    if (!one_f_.p) { one_f_.alloc(16); const float one = 1.f; CUDA_CHECK(cudaMemcpy(one_f_.p, &one, 4, cudaMemcpyHostToDevice)); }
    mtp_moe_tab_ = glm_moe_fp8_table_create(m_.mtp()->experts.data()->data(), (int)m_.mtp()->experts.size(), c_.hidden, c_.moe_inter);
    al(mtp_moe_ws_, glm_moe_fp8_ws_bytes(1, c_.n_act, c_.moe_inter));
    // the embedding table read by the draft chain on the device (embed_gather_dev): pin + map it (it stays in host RAM)
    CUDA_CHECK(cudaHostRegister(const_cast<bf16*>(m_.embed_host()), (size_t)c_.vocab * c_.hidden * 2, cudaHostRegisterMapped));
  }
  fprintf(stderr, "[glm] engine: chunk %d · work buffers ready (attention workspace %.0f MiB)\n", max_chunk, ws_bytes_ / 1048576.0);
}

std::unique_ptr<GlmSeq> GlmEngine::new_seq(int64_t cap) {
  auto s = std::make_unique<GlmSeq>();
  init_seq(*s, cap);
  return s;
}

void GlmEngine::init_seq(GlmSeq& seq, int64_t cap) {
  GlmSeq* s = &seq;
  s->cap = cap; s->device_bytes = 0; s->kv_tokens = 0;
  const size_t W = (size_t)c_.kda_heads * c_.kda_dim;
  s->conv.resize(n_kda_); s->state.resize(n_kda_);
  for (int i = 0; i < n_kda_; ++i) {
    s->conv[i].alloc(3 * 3 * W * 4); s->state[i].alloc((size_t)c_.kda_heads * c_.kda_dim * c_.kda_dim * 4);
    s->device_bytes += s->conv[i].n + s->state[i].n;
  }
  // DSA / MTP caches: addresses for cap positions, memory for the first kv_base_ positions (kv_reserve grows it)
  s->c.resize(n_dsa_); s->kI.resize(n_dsa_); s->gs.resize(n_dsa_); s->pooled.resize(n_dsa_); s->dsa.resize(n_dsa_);
  for (int i = 0; i < n_dsa_; ++i) {
    s->c[i].reserve((size_t)cap * c_.kv_lora * 2); s->kI[i].reserve((size_t)cap * c_.index_dim * 2); s->gs[i].reserve((size_t)cap * c_.index_dim * 2);
    s->pooled[i].reserve((size_t)(cap / c_.index_kpool + 1) * c_.index_dim * 2);
    s->dsa[i] = DsaSeqCache{s->c[i].as<bf16>(), s->kI[i].as<bf16>(), s->gs[i].as<bf16>(), s->pooled[i].as<bf16>(), (int)cap};
  }
  if (m_.mtp()) {
    s->mtp_c.reserve((size_t)cap * c_.kv_lora * 2); s->mtp_kI.reserve((size_t)cap * c_.index_dim * 2); s->mtp_gs.reserve((size_t)cap * c_.index_dim * 2);
    s->mtp_pooled.reserve((size_t)(cap / c_.index_kpool + 1) * c_.index_dim * 2);
    s->mtp_cache = DsaSeqCache{s->mtp_c.as<bf16>(), s->mtp_kI.as<bf16>(), s->mtp_gs.as<bf16>(), s->mtp_pooled.as<bf16>(), (int)cap};
    s->h_last.alloc((size_t)c_.hidden * 2);
    s->device_bytes += s->h_last.n;
  }
  kv_reserve(*s, std::min<int64_t>(kv_base_, cap));
  reset_seq(*s);
}

size_t GlmEngine::kv_bytes(const GlmSeq& s, int64_t t) const {
  auto cost = [&](const GrowBuf& b, size_t bytes) { return b.grow_cost(bytes); };
  size_t n = 0;
  const size_t T = (size_t)t;
  for (size_t i = 0; i < s.c.size(); ++i)
    n += cost(s.c[i], T * c_.kv_lora * 2) + cost(s.kI[i], T * c_.index_dim * 2) + cost(s.gs[i], T * c_.index_dim * 2) +
         cost(s.pooled[i], (T / c_.index_kpool + 1) * c_.index_dim * 2);
  if (s.h_last.p)
    n += cost(s.mtp_c, T * c_.kv_lora * 2) + cost(s.mtp_kI, T * c_.index_dim * 2) + cost(s.mtp_gs, T * c_.index_dim * 2) +
         cost(s.mtp_pooled, (T / c_.index_kpool + 1) * c_.index_dim * 2);
  return n;
}

void GlmEngine::kv_reserve(GlmSeq& s, int64_t tokens) {
  if (tokens <= s.kv_tokens) return;
  HIVE_CHECK(tokens <= s.cap, "GLM: sequence capacity exceeded");
  const int64_t t = std::min<int64_t>(s.cap, (tokens + kv_step_ - 1) / kv_step_ * kv_step_);
  const size_t need = kv_bytes(s, t);
  if (need > 0) {
    size_t fr = 0, tot = 0;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    if (fr < need + kv_headroom_) {  // keep the configured headroom: take the difference from the expert cache tail
      CUDA_CHECK(cudaStreamSynchronize(stream_));  // kernels of the last step may still read tail slots
      ex_.shrink_cache(need + kv_headroom_ - fr);
    }
  }
  const size_t T = (size_t)t;
  bool ok = true;
  auto grow = [&](GrowBuf& b, size_t bytes) { ok = ok && b.ensure(bytes); };
  for (size_t i = 0; i < s.c.size(); ++i) {
    grow(s.c[i], T * c_.kv_lora * 2); grow(s.kI[i], T * c_.index_dim * 2); grow(s.gs[i], T * c_.index_dim * 2);
    grow(s.pooled[i], (T / c_.index_kpool + 1) * c_.index_dim * 2);
  }
  if (s.h_last.p) {
    grow(s.mtp_c, T * c_.kv_lora * 2); grow(s.mtp_kI, T * c_.index_dim * 2); grow(s.mtp_gs, T * c_.index_dim * 2);
    grow(s.mtp_pooled, (T / c_.index_kpool + 1) * c_.index_dim * 2);
  }
  if (!ok) {
    kv_trim(s, s.kv_tokens);  // give back what this call mapped before failing (kv_tokens is still the old size)
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    throw std::runtime_error("GLM: no VRAM for the KV cache of " + std::to_string((long long)tokens) + " positions (free " + std::to_string((long long)(fr >> 20)) +
                             " MiB, expert cache " + std::to_string(ex_.n_slots()) + " slots)");
  }
  s.kv_tokens = t;
}

void GlmEngine::apply_embed_override(bf16* x_dev, int64_t base, int rows) {
  const size_t H = c_.hidden;
  for (const EmbedOverride& o : emb_over_) {
    const int64_t a = std::max<int64_t>(o.row, base), b = std::min<int64_t>(o.row + o.n, base + rows);
    if (a >= b) continue;
    CUDA_CHECK(cudaMemcpyAsync(x_dev + (size_t)(a - base) * H, o.dev + (size_t)(a - o.row) * H, (size_t)(b - a) * H * 2,
                               cudaMemcpyDeviceToDevice, stream_));
  }
}

void GlmEngine::make_room(size_t bytes) {
  size_t fr = 0, tot = 0;
  CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
  if (fr >= bytes + kv_headroom_) return;
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  ex_.shrink_cache(bytes + kv_headroom_ - fr);
}

void GlmEngine::kv_trim(GlmSeq& s, int64_t tokens) {
  const size_t T = (size_t)std::max<int64_t>(tokens, 0);
  size_t freed = 0;
  for (size_t i = 0; i < s.c.size(); ++i)
    freed += s.c[i].trim(T * c_.kv_lora * 2) + s.kI[i].trim(T * c_.index_dim * 2) + s.gs[i].trim(T * c_.index_dim * 2) +
             s.pooled[i].trim((T / c_.index_kpool + 1) * c_.index_dim * 2);
  if (s.h_last.p)
    freed += s.mtp_c.trim(T * c_.kv_lora * 2) + s.mtp_kI.trim(T * c_.index_dim * 2) + s.mtp_gs.trim(T * c_.index_dim * 2) +
             s.mtp_pooled.trim((T / c_.index_kpool + 1) * c_.index_dim * 2);
  s.kv_tokens = tokens;
  if (freed) ex_.regrow_cache(kv_headroom_);
}

void GlmEngine::reset_seq(GlmSeq& s) {
  for (auto& b : s.conv) CUDA_CHECK(cudaMemsetAsync(b.p, 0, b.n, stream_));
  for (auto& b : s.state) CUDA_CHECK(cudaMemsetAsync(b.p, 0, b.n, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  kv_trim(s, std::min<int64_t>(kv_base_, s.cap));  // a long conversation's KV goes back to the expert cache
  s.pos = 0; s.tokens.clear(); s.broken = false;
  s.h_last_valid = false; s.mtp_written = 0;
}

void GlmEngine::hc_pre_norm(const HcW& w, const DevBuf& norm_w, int rows) {
  const int H = c_.hidden, HC = c_.hc * H;
  static const bool cublas_path = getenv("HIVE_GLM_HC_CUBLAS") && atoi(getenv("HIVE_GLM_HC_CUBLAS")) > 0;  // reference path for A/B
  if (cublas_path) {
    hc_flatten_f32(hb(), rows, HC, mix_f32_.as<float>(), stream_);
    blas_.gemm_f32(mix_f32_.as<float>(), w.fn.as<float>(), mixes_.as<float>(), rows, 24, HC);
    hc_row_rsqrt(hb(), rows, HC, c_.rms_eps, rsq_.as<float>(), stream_);
  } else if (rows <= 8) {
    k::hc_mix(hb(), w.fn.as<float>(), rows, HC, 24, c_.rms_eps, mixes_.as<float>(), rsq_.as<float>(), stream_);
  } else {
    k::hc_mix_rows(hb(), w.fn.as<float>(), rows, HC, 24, c_.rms_eps, mixes_.as<float>(), rsq_.as<float>(), stream_);
  }
  hc_split_sinkhorn(mixes_.as<float>(), rsq_.as<float>(), w.scale.as<float>(), w.base.as<float>(), rows, c_.hc, c_.sinkhorn_iters, c_.hc_eps,
                    pre_.as<float>(), post_.as<float>(), comb_.as<float>(), stream_);
  if (!cublas_path && rows <= 8) {
    k::hc_pre_norm(hb(), pre_.as<float>(), rows, c_.hc, H, norm_w.as<bf16>(), c_.rms_eps, x_.as<bf16>(), xn_.as<bf16>(), stream_);
  } else {
    hc_pre(hb(), pre_.as<float>(), rows, c_.hc, H, x_.as<bf16>(), stream_);
    rmsnorm(x_.as<bf16>(), norm_w.as<bf16>(), c_.rms_eps, rows, H, xn_.as<bf16>(), stream_);
  }
}

void GlmEngine::lin(const Mat& W, const bf16* A, int lda, int rows, bf16* C, int ldc) {
  if (!W.fp8) { gemm_into(blas_, A, lda, W.p(), C, ldc, rows, W.out, W.in); return; }
  if (rows <= 8) { fp8b_gemv(W.f8(), A, lda, rows, C, ldc, stream_); return; }
  fp8b_dequant(W.f8(), deq_.as<bf16>(), stream_);
  gemm_into(blas_, A, lda, deq_.as<bf16>(), C, ldc, rows, W.out, W.in);
}

// KDA verify snapshots for `parts` parts × kmax rows (conv window + recurrent state after each row but the last) and the pointer table
//   kda_decode_rows reads through UVA ([part][kda layer][conv 8 | state 8]). The buffers never move (with HIVE_SLEEP_VMM they come back
//   at the same VA after sleep level 3). Growing keeps the existing part-0 buffers.
void GlmEngine::alloc_snapshots(int parts) {
  const int kmax = std::max(1, mtp_k_);
  HIVE_CHECK(kmax <= 8 && parts >= 1 && parts <= 8, "GLM verify snapshots: at most 8 rows and 8 parts");
  const size_t old = snap_state_.size(), want = (size_t)parts * kmax * n_kda_;
  if (want > old) {
    snap_state_.resize(want); snap_conv_.resize(want);
    for (size_t i = old; i < want; ++i) {
      snap_state_[i].alloc((size_t)c_.kda_heads * c_.kda_dim * c_.kda_dim * 4);
      snap_conv_[i].alloc((size_t)3 * 3 * c_.kda_heads * c_.kda_dim * 4);
    }
  }
  if (snap_ptr_h_) CUDA_CHECK(cudaFreeHost(snap_ptr_h_));
  CUDA_CHECK(cudaMallocHost(&snap_ptr_h_, (size_t)parts * n_kda_ * 16 * sizeof(float*)));
  for (int p = 0; p < parts; ++p)
    for (int ki = 0; ki < n_kda_; ++ki)
      for (int r = 0; r < kmax; ++r) {
        snap_ptr_h_[((size_t)p * n_kda_ + ki) * 16 + r] = snap_conv_[((size_t)p * kmax + r) * n_kda_ + ki].as<float>();
        snap_ptr_h_[((size_t)p * n_kda_ + ki) * 16 + 8 + r] = snap_state_[((size_t)p * kmax + r) * n_kda_ + ki].as<float>();
      }
  snap_parts_ = parts;
  fprintf(stderr, "[glm] MTP: up to %d drafts per step · verify snapshots for %d part(s) %.0f MiB\n", mtp_k_, parts,
          (double)want * (snap_state_[0].n + snap_conv_[0].n) / 1048576.0);
}

void GlmEngine::enable_batch_verify(int parts) { if (mtp_on() && parts > snap_parts_) alloc_snapshots(parts); }

void GlmEngine::hc_post_apply(int rows) {
  hc_post(attn_.as<bf16>(), post_.as<float>(), comb_.as<float>(), rows, c_.hc, c_.hidden, hb(), stream_);
}

void GlmEngine::kda_layer(int l, GlmSeq* const* seqs, int M, int T, bool prefill) {
  const KdaW& w = m_.layer(l).kda_w;
  const int H = c_.hidden, W = c_.kda_heads * c_.kda_dim, rows = prefill ? T : M;
  bf16* xn = xn_.as<bf16>();
  bf16* qkv = qkv_.as<bf16>();
  const bool fast = dec_fast(prefill, rows) && w.f_a.out % 32 == 0 && w.g_a.out % 32 == 0;
  if (fast) {  // q|k|v|f_a|g_a|b in one launch, then f_b|g_b
    GemvSeg a[6] = {{w.q.p(), xn, qkv, W, H, H, 3 * W}, {w.k.p(), xn, qkv + W, W, H, H, 3 * W}, {w.v.p(), xn, qkv + 2 * W, W, H, H, 3 * W},
                    {w.f_a.p(), xn, fa_.as<bf16>(), w.f_a.out, H, H, w.f_a.out}, {w.g_a.p(), xn, ga_.as<bf16>(), w.g_a.out, H, H, w.g_a.out},
                    {w.b.p(), xn, beta_.as<bf16>(), c_.kda_heads, H, H, c_.kda_heads}};
    glm_gemv_bf16_multi(a, 6, rows, stream_);
    GemvSeg b[2] = {{w.f_b.p(), fa_.as<bf16>(), gpre_.as<bf16>(), W, w.f_b.in, w.f_b.in, W}, {w.g_b.p(), ga_.as<bf16>(), gate_.as<bf16>(), W, w.g_b.in, w.g_b.in, W}};
    glm_gemv_bf16_multi(b, 2, rows, stream_);
  } else {
  gemm_into(blas_, xn, H, w.q.p(), qkv, 3 * W, rows, W, H);
  gemm_into(blas_, xn, H, w.k.p(), qkv + W, 3 * W, rows, W, H);
  gemm_into(blas_, xn, H, w.v.p(), qkv + 2 * W, 3 * W, rows, W, H);
  blas_.gemm_bf16(xn, w.f_a.p(), fa_.as<bf16>(), rows, w.f_a.out, H);
  blas_.gemm_bf16(fa_.as<bf16>(), w.f_b.p(), gpre_.as<bf16>(), rows, W, w.f_b.in);
  blas_.gemm_bf16(xn, w.g_a.p(), ga_.as<bf16>(), rows, w.g_a.out, H);
  blas_.gemm_bf16(ga_.as<bf16>(), w.g_b.p(), gate_.as<bf16>(), rows, W, w.g_b.in);
  blas_.gemm_bf16(xn, w.b.p(), beta_.as<bf16>(), rows, c_.kda_heads, H);
  }
  prof_mark(&st_.ms_kda_proj);
  KdaParams p{w.conv_w.as<float>(), w.A_log.as<float>(), w.dt_bias.as<float>(), w.o_norm.as<bf16>()};
  const int ki = kda_index_[l];
  if (prefill) {
    kda_forward_seq(p, qkv, gpre_.as<bf16>(), beta_.as<bf16>(), gate_.as<bf16>(), T, seqs[0]->conv[ki].as<float>(), seqs[0]->state[ki].as<float>(),
                    kda_out_.as<bf16>(), ws_.p, ws_bytes_, stream_);
  } else if (verify_) {
    // verify: per part, rows are consecutive tokens of one sequence — the recurrence runs row by row; the state after each row (but the
    //   last) is kept so rollback can return to any accepted prefix
    float** dptr = kda_ptr_h_ + (size_t)ki * 16;  // row r holds its part's sequence
    const int kmax = std::max(1, mtp_k_);
    // HIVE_GLM_KDA_ROWS (default on; "0" = one launch per row + dcopy snapshots): a part's rows in one launch (kda_decode_rows, bit-identical)
    static const bool rows1 = !(getenv("HIVE_GLM_KDA_ROWS") && strcmp(getenv("HIVE_GLM_KDA_ROWS"), "0") == 0);
    for (size_t pi = 0; pi < vparts_.size(); ++pi) {
      const VPart& vp = vparts_[pi];
      HIVE_CHECK(vp.M - 1 <= kmax && (int)pi < snap_parts_, "GLM verify: more rows or parts than snapshots (HIVE_GLM_MTP_K / enable_batch_verify)");
      const int r0 = vp.r0;
      if (rows1) {
        float** sp = snap_ptr_h_ + ((size_t)pi * n_kda_ + ki) * 16;
        kda_decode_rows(p, qkv + (size_t)r0 * 3 * W, gpre_.as<bf16>() + (size_t)r0 * W, beta_.as<bf16>() + (size_t)r0 * c_.kda_heads,
                        gate_.as<bf16>() + (size_t)r0 * W, vp.M, dptr + r0, dptr + 8 + r0, sp, sp + 8, kda_out_.as<bf16>() + (size_t)r0 * W, stream_);
        continue;
      }
      GlmSeq& sq = *vp.s;
      for (int r = 0; r < vp.M; ++r) {
        const int row = r0 + r;
        kda_decode_batch(p, qkv + (size_t)row * 3 * W, gpre_.as<bf16>() + (size_t)row * W, beta_.as<bf16>() + (size_t)row * c_.kda_heads,
                         gate_.as<bf16>() + (size_t)row * W, 1, dptr + r0, dptr + 8 + r0, kda_out_.as<bf16>() + (size_t)row * W, stream_);
        if (r + 1 < vp.M) {
          dcopy(snap_state_[((size_t)pi * kmax + r) * n_kda_ + ki].p, sq.state[ki].p, sq.state[ki].n, stream_);
          dcopy(snap_conv_[((size_t)pi * kmax + r) * n_kda_ + ki].p, sq.conv[ki].p, sq.conv[ki].n, stream_);
        }
      }
    }
  } else {
    // per-step pointer table (uploaded once per decode step in forward): [ki][conv 8 | state 8]
    float** dptr = kda_ptr_h_ + (size_t)ki * 16;  // pinned host table (UVA)
    kda_decode_batch(p, qkv, gpre_.as<bf16>(), beta_.as<bf16>(), gate_.as<bf16>(), M, dptr, dptr + 8, kda_out_.as<bf16>(), stream_);
  }
  prof_mark(&st_.ms_kda_core);
  if (fast) { GemvSeg o{w.o.p(), kda_out_.as<bf16>(), attn_.as<bf16>(), H, W, W, H}; glm_gemv_bf16_multi(&o, 1, rows, stream_); }
  else blas_.gemm_bf16(kda_out_.as<bf16>(), w.o.p(), attn_.as<bf16>(), rows, H, W);
}

void GlmEngine::dsa_layer(int l, GlmSeq* const* seqs, int M, int T, bool prefill) {
  const DsaW& w = m_.layer(l).dsa_w;
  const int H = c_.hidden, rows = prefill ? T : M;
  bf16* xn = xn_.as<bf16>();
  lin(w.q_a, xn, H, rows, qa_.as<bf16>(), c_.q_lora);
  rmsnorm(qa_.as<bf16>(), w.q_a_norm.as<bf16>(), c_.rms_eps, rows, c_.q_lora, qres_.as<bf16>(), stream_);
  lin(w.q_b, qres_.as<bf16>(), c_.q_lora, rows, q_.as<bf16>(), w.q_b.out);
  lin(w.kv_a, xn, H, rows, ckv_.as<bf16>(), c_.kv_lora);
  rmsnorm(ckv_.as<bf16>(), w.kv_a_norm.as<bf16>(), c_.rms_eps, rows, c_.kv_lora, cn_.as<bf16>(), stream_);
  const bool fast = dec_fast(prefill, rows);
  if (fast) {  // indexer projections wq_b (from q_resid) | wk | kpool gate (from xn) in one launch
    GemvSeg sg[3] = {{w.idx_wq_b.p(), qres_.as<bf16>(), qI_.as<bf16>(), w.idx_wq_b.out, c_.q_lora, c_.q_lora, w.idx_wq_b.out},
                     {w.idx_wk.p(), xn, kI_raw_.as<bf16>(), c_.index_dim, H, H, c_.index_dim},
                     {w.idx_gate.p(), xn, gs_.as<bf16>(), c_.index_dim, H, H, c_.index_dim}};
    glm_gemv_bf16_multi(sg, 3, rows, stream_);
  } else {
    blas_.gemm_bf16(qres_.as<bf16>(), w.idx_wq_b.p(), qI_.as<bf16>(), rows, w.idx_wq_b.out, c_.q_lora);
    blas_.gemm_bf16(xn, w.idx_wk.p(), kI_raw_.as<bf16>(), rows, c_.index_dim, H);
    blas_.gemm_bf16(xn, w.idx_gate.p(), gs_.as<bf16>(), rows, c_.index_dim, H);
  }
  layernorm(kI_raw_.as<bf16>(), w.idx_k_norm_w.as<bf16>(), w.idx_k_norm_b.as<bf16>(), 1e-6f, rows, c_.index_dim, kI_.as<bf16>(), stream_);
  // head weights as weights_proj(x) — the DSA selection kernels apply the 32^-0.5 head scale themselves (dsa.cu score_kernel)
  blas_.gemm_bf16_f32out(xn, w.idx_weights.p(), wI_.as<float>(), rows, c_.index_heads, H);
  DsaLayerW dw{w.kv_b.p(), w.idx_ape.as<bf16>()};
  const int di = dsa_index_[l];
  const int qI_w = c_.index_heads * c_.index_dim, q_w = c_.n_heads * c_.qk_head, out_w = c_.n_heads * c_.v_head, sel_w = c_.index_topk + 3;
  if (prefill) {
    GlmSeq& s = *seqs[0];
    dsa_append(dw, s.dsa[di], (int)s.pos, T, cn_.as<bf16>(), kI_.as<bf16>(), gs_.as<bf16>(), stream_);
    dsa_select(s.dsa[di], (int)s.pos, T, qI_.as<bf16>(), wI_.as<float>(), idx_.as<int32_t>(), ws_.p, ws_bytes_, stream_);
    dsa_attention(dw, s.dsa[di], (int)s.pos, T, q_.as<bf16>(), idx_.as<int32_t>(), dsa_out_.as<bf16>(), ws_.p, ws_bytes_, stream_);
  } else if (fast) {  // batched decode: row table for this DSA layer was uploaded once for the step (forward)
    const DsaDecodeRow* dr = dsa_rows_h_ + (size_t)di * 8;  // pinned host table (UVA)
    dsa_append_decode(dw, dr, M, cn_.as<bf16>(), kI_.as<bf16>(), gs_.as<bf16>(), stream_);
    dsa_select_decode(dr, M, dsa_max_ctx_, qI_.as<bf16>(), wI_.as<float>(), idx_.as<int32_t>(), ws_.p, ws_bytes_, stream_);
    dsa_attention_decode(dw, dr, M, q_.as<bf16>(), idx_.as<int32_t>(), dsa_out_.as<bf16>(), ws_.p, ws_bytes_, stream_);
  } else {
    for (int m = 0; m < M; ++m) {
      GlmSeq& s = *seqs[m];
      dsa_append(dw, s.dsa[di], (int)s.pos, 1, cn_.as<bf16>() + (size_t)m * c_.kv_lora, kI_.as<bf16>() + (size_t)m * c_.index_dim,
                 gs_.as<bf16>() + (size_t)m * c_.index_dim, stream_);
      dsa_select(s.dsa[di], (int)s.pos, 1, qI_.as<bf16>() + (size_t)m * qI_w, wI_.as<float>() + (size_t)m * c_.index_heads,
                 idx_.as<int32_t>() + (size_t)m * sel_w, ws_.p, ws_bytes_, stream_);
      dsa_attention(dw, s.dsa[di], (int)s.pos, 1, q_.as<bf16>() + (size_t)m * q_w, idx_.as<int32_t>() + (size_t)m * sel_w,
                    dsa_out_.as<bf16>() + (size_t)m * out_w, ws_.p, ws_bytes_, stream_);
    }
  }
  lin(w.o, dsa_out_.as<bf16>(), out_w, rows, attn_.as<bf16>(), H);
}

void GlmEngine::predict_eval(int l, int rows, const bf16* xn) {
  const int li = c_.moe_index[l], K = c_.n_act, E = c_.n_routed;
  // 1) score the predictions made earlier for this layer
  for (int d = 1; d <= 2; ++d) {
    std::vector<int32_t>& p = pred_[d][li];
    if ((int)p.size() != rows * K) { p.clear(); continue; }
    for (int m = 0; m < rows; ++m)
      for (int k = 0; k < K; ++k) {
        const int id = h_ids_[m * K + k];
        const bool hit = std::find(p.begin() + m * K, p.begin() + (m + 1) * K, id) != p.begin() + (m + 1) * K;
        const bool miss = ex_.slot_of(li * E + id) < 0;
        ++ps_.pairs[d]; ps_.pred[d] += hit; ps_.miss[d] += miss; ps_.miss_pred[d] += miss && hit;
      }
    // prefetch precision: distinct predicted ids that were non-resident at prediction time, and whether any row used them
    const std::vector<uint8_t>& nr = pred_nonres_[d][li];
    std::vector<int32_t> seen;
    for (size_t i = 0; i < p.size(); ++i) {
      if (i >= nr.size() || !nr[i] || std::find(seen.begin(), seen.end(), p[i]) != seen.end()) continue;
      seen.push_back(p[i]);
      ++ps_.pf_issued[d];
      ps_.pf_used[d] += std::find(h_ids_.begin(), h_ids_.begin() + rows * K, p[i]) != h_ids_.begin() + rows * K;
    }
    p.clear();
    pred_nonres_[d][li].clear();
  }
  // 2) predict the next two MoE layers from this layer's FFN input
  int found = 0;
  for (int t = l + 1; t < c_.n_layers && found < 2; ++t) {
    if (!c_.is_moe[t]) continue;
    ++found;
    const MoeW& w = m_.layer(t).moe_w;
    const bf16* in = xn;
    if (pred_eval_ == 2) {
      // router input proxy: the target layer's own hc mixing (ffn side) + post_norm applied to the current streams h
      const LayerW& T = m_.layer(t);
      const int HC = c_.hc * c_.hidden;
      k::hc_mix(hb(), T.hc_ffn.fn.as<float>(), rows, HC, 24, c_.rms_eps, p_mix_.as<float>(), p_rsq_.as<float>(), stream_);
      hc_split_sinkhorn(p_mix_.as<float>(), p_rsq_.as<float>(), T.hc_ffn.scale.as<float>(), T.hc_ffn.base.as<float>(), rows, c_.hc, c_.sinkhorn_iters,
                        c_.hc_eps, p_pre_.as<float>(), p_post_.as<float>(), p_comb_.as<float>(), stream_);
      k::hc_pre_norm(hb(), p_pre_.as<float>(), rows, c_.hc, c_.hidden, T.post_norm.as<bf16>(), c_.rms_eps, p_x_.as<bf16>(), p_xn_.as<bf16>(), stream_);
      in = p_xn_.as<bf16>();
    }
    blas_.gemm_bf16_f32out(in, w.gate.p(), pred_logits_.as<float>(), rows, E, c_.hidden);
    router_topk(pred_logits_.as<float>(), w.bias.as<float>(), rows, E, K, c_.routed_scale, pred_ids_d_.as<int32_t>(), pred_w_d_.as<float>(), stream_);
    std::vector<int32_t>& p = pred_[found][c_.moe_index[t]];
    p.resize((size_t)rows * K);
    CUDA_CHECK(cudaMemcpyAsync(p.data(), pred_ids_d_.p, (size_t)rows * K * 4, cudaMemcpyDeviceToHost, stream_));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  for (int d = 1; d <= 2; ++d)
    for (int t = 0; t < c_.n_moe; ++t) {
      const std::vector<int32_t>& p = pred_[d][t];
      if (p.empty() || !pred_nonres_[d][t].empty()) continue;
      pred_nonres_[d][t].resize(p.size());
      for (size_t i = 0; i < p.size(); ++i) pred_nonres_[d][t][i] = ex_.slot_of(t * E + p[i]) < 0;
    }
}

void GlmEngine::mlp_layer(int l, int rows, bool prefill) {
  const LayerW& L = m_.layer(l);
  const int H = c_.hidden;
  bf16* xn = xn_.as<bf16>();
  if (!L.moe) {
    const int I = c_.dense_inter;
    bf16* gu = dense_gu_.as<bf16>();
    if (dec_fast(prefill, rows)) {
      glm_dense_nvfp4_decode(m_.dense_mat(l, 0), m_.dense_mat(l, 1), m_.dense_mat(l, 2), xn, H, rows, c_.swiglu_limit, dense_y_.as<bf16>(), attn_.as<bf16>(), H, stream_);
    } else if (rows <= 4) {  // (nvfp4_gemv stages rows·12288 bf16 of the down input in ≤ 96 KB shared memory)
      // NVFP4 gemv: fp32 outputs → bf16 (reuse mlp_out_f32_ as scratch)
      float* tmp = mlp_out_f32_.as<float>();
      nvfp4_gemv(m_.dense_mat(l, 0), xn, H, rows, tmp, 2 * I, stream_);
      nvfp4_gemv(m_.dense_mat(l, 1), xn, H, rows, tmp + I, 2 * I, stream_);
      f32_to_bf16(tmp, (size_t)rows * 2 * I, gu, stream_);
      swiglu_rows(gu, rows, I, c_.swiglu_limit, dense_y_.as<bf16>(), stream_);
      nvfp4_gemv(m_.dense_mat(l, 2), dense_y_.as<bf16>(), I, rows, tmp, H, stream_);
      f32_to_bf16(tmp, (size_t)rows * H, attn_.as<bf16>(), stream_);
    } else {
      bf16* dq = deq_.as<bf16>();
      nvfp4_dequant(m_.dense_mat(l, 0), dq, stream_);
      gemm_into(blas_, xn, H, dq, gu, 2 * I, rows, I, H);
      nvfp4_dequant(m_.dense_mat(l, 1), dq, stream_);
      gemm_into(blas_, xn, H, dq, gu + I, 2 * I, rows, I, H);
      swiglu_rows(gu, rows, I, c_.swiglu_limit, dense_y_.as<bf16>(), stream_);
      nvfp4_dequant(m_.dense_mat(l, 2), dq, stream_);
      blas_.gemm_bf16(dense_y_.as<bf16>(), dq, attn_.as<bf16>(), rows, H, I);
    }
    return;
  }
  const MoeW& w = L.moe_w;
  const int I = c_.moe_inter, E = c_.n_routed, K = c_.n_act;
  // Router; with prediction prefetch (HIVE_GLM_PREFETCH=N, decode) the next MoE layer's router runs in the same GEMV (gates stacked) and the
  //   same top-k launch, on this layer's FFN input — its choice is a prediction used only to copy experts ahead (outputs unchanged).
  const int li_cur = c_.moe_index[l];
  // Short chunks (rows ≤ HIVE_GLM_SHORT_PREFILL, default 64) take the decode path — resident experts on the GPU, missed ones on the
  //   CPU — instead of streaming every used expert over PCIe (a 26-token prompt streamed 3,594 records = 2.5 s on a cold cache, hived log).
  static const int short_rows = getenv("HIVE_GLM_SHORT_PREFILL") ? atoi(getenv("HIVE_GLM_SHORT_PREFILL")) : 64;
  const bool prefill_streams = prefill && rows > std::min(short_rows, 64);
  const int nxt = next_moe_[l];
  const bool pf = pf_n_ > 0 && !prefill && rows <= 8 && nxt >= 0 && (dec_fast(prefill, rows) || !pair_gate_.empty());
  const bool pf_fast = pf && dec_fast(prefill, rows);
  // HIVE_CACHE_PRIOR: this layer's residency → its mask row (written before the launch that reads it; the previous reader of the row,
  //   last step's router of this layer, finished before this step's host work on the layer)
  GlmCachePrior cprior{};
  const GlmCachePrior* cpp = nullptr;
  if (cache_prior_ > 0.f && !prefill) {
    uint8_t* mrow = cache_mask_h_ + (size_t)li_cur * E;
    for (int e = 0; e < E; ++e) mrow[e] = ex_.slot_of(li_cur * E + e) >= 0;
    cprior = GlmCachePrior{mrow, cache_prior_, prior_range_.as<float>() + li_cur, cache_prior_j_};
    cpp = &cprior;
  }
  if (pf_fast) {  // two fused router launches (this layer → mapped rids/rw, next layer → mapped prediction buffers)
    glm_router_decode(xn, H, rows, w.gate.p(), w.bias.as<float>(), E, H, K, c_.routed_scale, rlog_.as<float>(), router_ids_.as<int32_t>(),
                      router_w_.as<float>(), rids_h_, rw_h_, stream_, cpp);
    const MoeW& nw = m_.layer(nxt).moe_w;
    glm_router_decode(xn, H, rows, nw.gate.p(), nw.bias.as<float>(), E, H, K, c_.routed_scale, rlog_.as<float>(), pf_ids_d_.as<int32_t>(),
                      pf_w_d_.as<float>(), pf_ids_map_, pf_w_map_, stream_);
  } else if (pf) {
    blas_.gemm_bf16_f32out(xn, pair_gate_[li_cur].as<bf16>(), router_logits_.as<float>(), rows, 2 * E, H);
    router_topk2(router_logits_.as<float>(), 2 * E, w.bias.as<float>(), m_.layer(nxt).moe_w.bias.as<float>(), rows, E, K, c_.routed_scale,
                 router_ids_.as<int32_t>(), router_w_.as<float>(), pf_ids_d_.as<int32_t>(), pf_w_d_.as<float>(), stream_);
    pf_ids_h_.resize((size_t)rows * K); pf_w_h_.resize((size_t)rows * K);
    CUDA_CHECK(cudaMemcpyAsync(pf_ids_h_.data(), pf_ids_d_.p, (size_t)rows * K * 4, cudaMemcpyDeviceToHost, stream_));  // (slow path only)
    CUDA_CHECK(cudaMemcpyAsync(pf_w_h_.data(), pf_w_d_.p, (size_t)rows * K * 4, cudaMemcpyDeviceToHost, stream_));
  } else if (dec_fast(prefill, rows)) {
    glm_router_decode(xn, H, rows, w.gate.p(), w.bias.as<float>(), E, H, K, c_.routed_scale, rlog_.as<float>(), router_ids_.as<int32_t>(),
                      router_w_.as<float>(), rids_h_, rw_h_, stream_, cpp);
  } else {
    blas_.gemm_bf16_f32out(xn, w.gate.p(), router_logits_.as<float>(), rows, E, H);
    router_topk2(router_logits_.as<float>(), E, w.bias.as<float>(), nullptr, rows, E, K, c_.routed_scale, router_ids_.as<int32_t>(), router_w_.as<float>(),
                 nullptr, nullptr, stream_);
  }
  const bool mapped_route = dec_fast(prefill, rows) && (!pf || pf_fast);
  h_ids_.resize((size_t)rows * K); h_w_.resize((size_t)rows * K);
  if (!mapped_route) {
    CUDA_CHECK(cudaMemcpyAsync(h_ids_.data(), router_ids_.p, (size_t)rows * K * 4, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_w_.data(), router_w_.p, (size_t)rows * K * 4, cudaMemcpyDeviceToHost, stream_));
  }
  prof_mark(&st_.ms_router);
  // shared expert (BF16)
  bf16* sgu = sh_gu_.as<bf16>();
  if (dec_fast(prefill, rows) && w.sh_gate.fp8 && w.sh_up.fp8 && w.sh_down.fp8) {
    glm_shared_fp8_decode(w.sh_gate.f8(), w.sh_up.f8(), w.sh_down.f8(), xn, H, rows, c_.swiglu_limit, sh_y_.as<bf16>(), mlp_out_.as<bf16>(), H, stream_);
  } else {
  lin(w.sh_gate, xn, H, rows, sgu, 2 * I);
  lin(w.sh_up, xn, H, rows, sgu + I, 2 * I);
  swiglu_rows(sgu, rows, I, c_.swiglu_limit, sh_y_.as<bf16>(), stream_);
  lin(w.sh_down, sh_y_.as<bf16>(), I, rows, mlp_out_.as<bf16>(), H);
  }
  CUDA_CHECK(cudaMemsetAsync(mlp_out_f32_.p, 0, (size_t)rows * H * 4, stream_));
  if (!prefill_streams) dcopy(xin_pin_, xn, (size_t)rows * H * 2, stream_);  // CPU expert input: pinned host rows (UVA), ready at the sync below
  CUDA_CHECK(cudaStreamSynchronize(stream_));  // routing on the host
  prof_mark(&st_.ms_shared);
  if (mapped_route) { std::memcpy(h_ids_.data(), rids_h_, (size_t)rows * K * 4); std::memcpy(h_w_.data(), rw_h_, (size_t)rows * K * 4); }
  if (pf) {
    if (pf_fast) { pf_ids_h_.assign(pf_ids_map_, pf_ids_map_ + rows * K); pf_w_h_.assign(pf_w_map_, pf_w_map_ + rows * K); }
    // steps with several rows (MTP verify, batches) keep the CPU busier for longer, leaving PCIe idle: their own budget (HIVE_GLM_PREFETCH_ROWS2)
    static const int pf_multi = getenv("HIVE_GLM_PREFETCH_ROWS2") ? atoi(getenv("HIVE_GLM_PREFETCH_ROWS2")) : 0;
    ex_.prefetch(c_.moe_index[nxt], pf_ids_h_.data(), pf_w_h_.data(), rows, rows >= 2 && pf_multi > 0 ? pf_multi : pf_n_, stream_);
  }
  if (pred_eval_ && !prefill) predict_eval(l, rows, xn);
  prof_mark(&st_.ms_predict);
  const int li = c_.moe_index[l];
  const double t0 = now_ms();
  if (prefill_streams) ex_.prefill_layer(li, rows, h_ids_.data(), h_w_.data(), xn, mlp_out_f32_.as<float>(), stream_);
  else {
    // expert deferral (HIVE_GLM_DEFER — GlmExperts::decode_layer): the previous MoE layer's deferred CPU experts are added to the streams
    //   now (they reach the next layers' inputs one MoE layer late — this layer's attention and FFN input did not see them); the last MoE
    //   layer is never deferred, so nothing is pending after the forward
    if (ex_.deferred_pending()) {
      ex_.deferred_wait();
      hc_inject(hb(), post_def_.as<float>(), ex_.deferred_y(), ex_.deferred_rows(), c_.hc, H, stream_);
    }
    const bool allow_defer = !prefill && ex_.defer_enabled() && next_moe_[l] >= 0;
    ex_.decode_layer(li, rows, h_ids_.data(), h_w_.data(), xn, mlp_out_f32_.as<float>(), stream_, xin_pin_, allow_defer);
    if (ex_.deferred_pending()) dcopy(post_def_.p, post_.p, (size_t)rows * c_.hc * 4, stream_);  // this FFN's post coefficients (post_ is reused)
  }
  f32_to_bf16(mlp_out_f32_.as<float>(), (size_t)rows * H, attn_.as<bf16>(), stream_);
  add_bf16(attn_.as<bf16>(), mlp_out_.as<bf16>(), (size_t)rows * H, attn_.as<bf16>(), stream_);
  st_.ms_moe += now_ms() - t0;
  prof_mark(&st_.ms_experts);
}

void GlmEngine::forward(GlmSeq* const* seqs, int M, const int32_t* ids, int T, bool prefill, float* logits_host, bool all_logits) {
  const int H = c_.hidden, rows = prefill ? T : M;
  int64_t pos_before[8] = {};
  if (prefill) pos_before[0] = seqs[0]->pos; else for (int m = 0; m < M && m < 8; ++m) pos_before[m] = seqs[m]->pos;
  const double t0 = now_ms();
  prof_t_ = t0;
  if (prof_ == 2) { ev_marks_.clear(); prof_mark(nullptr); }
  // embeddings: gather on the host (table in RAM), upload, replicate into the hc streams
  if (!prefill && rows <= 8) {
    // decode: rows gathered into a pinned buffer and expanded into the hc streams by a kernel that reads it directly (UVA) — no H2D
    //   copy on the critical path (it would queue behind promotion / prefetch copies on the copy engine)
    for (int r = 0; r < rows; ++r) std::memcpy(emb_pin_ + (size_t)r * H, m_.embed_host() + (size_t)ids[r] * H, (size_t)H * 2);
    k::embed_expand(emb_pin_, iota_.as<int32_t>(), rows, H, c_.hc, hb(), stream_);
  } else {
    static thread_local std::vector<bf16> emb;
    emb.resize((size_t)rows * H);
    for (int r = 0; r < rows; ++r) std::memcpy(emb.data() + (size_t)r * H, m_.embed_host() + (size_t)ids[r] * H, (size_t)H * 2);
    CUDA_CHECK(cudaMemcpyAsync(x_.p, emb.data(), (size_t)rows * H * 2, cudaMemcpyHostToDevice, stream_));
    if (prefill) apply_embed_override(x_.as<bf16>(), emb_base_, rows);
    for (int c = 0; c < c_.hc; ++c)
      CUDA_CHECK(cudaMemcpy2DAsync(hb() + (size_t)c * H, (size_t)c_.hc * H * 2, x_.p, (size_t)H * 2, (size_t)H * 2, rows, cudaMemcpyDeviceToDevice, stream_));
  }
  if (!prefill) {  // KDA state pointers of every KDA layer for this step: one upload (pinned host table, read by the async copy)
    HIVE_CHECK(M <= 8, "GLM decode batch > 8");
    for (int k2 = 0; k2 < n_kda_; ++k2)
      for (int m = 0; m < M; ++m) {
        kda_ptr_h_[(size_t)k2 * 16 + m] = seqs[m]->conv[k2].as<float>();
        kda_ptr_h_[(size_t)k2 * 16 + 8 + m] = seqs[m]->state[k2].as<float>();
      }
    // (the KDA / DSA tables are read by the kernels straight from pinned host memory — see emb_pin_ above for why no H2D copy)
    dsa_max_ctx_ = 0;
    for (int d2 = 0; d2 < n_dsa_; ++d2)
      for (int m = 0; m < M; ++m) {
        const DsaSeqCache& sc = seqs[m]->dsa[d2];
        dsa_rows_h_[(size_t)d2 * 8 + m] = DsaDecodeRow{sc.c, sc.kI, sc.gs, sc.pooled, (int)seqs[m]->pos + (verify_ ? vrow_off_[m] : 0)};
      }
    for (int m = 0; m < M; ++m) dsa_max_ctx_ = std::max<int>(dsa_max_ctx_, (int)seqs[m]->pos + 1 + (verify_ ? vrow_off_[m] : 0));

  }
  prof_mark(&st_.ms_embed);
  const bool fast = dec_fast(prefill, rows);
  // HIVE_GLM_LAUNCH_PROBE (see glm_engine.h): probe mode for this step — 0 normal launch, 1 held stream, -1 off
  const int probe = !prefill && probe_every_ > 0 && st_.steps % probe_every_ == 0 ? 0 : -1;
  double probe_host = 0;
  for (int l = 0; l < c_.n_layers; ++l) {
    const LayerW& L = m_.layer(l);
    double ph0 = 0;
    if (probe >= 0) {
      CUDA_CHECK(cudaEventRecord(probe_ev_[2 * l], stream_));
      ph0 = now_ms();
    }
    // fast decode: hc_post of the previous sub-layer is fused into the next hc pre-stage (post/comb carry over in post_/comb_)
    if (fast) glm_hc_pre_decode(hb(), rows, H, L.hc_attn.fn.as<float>(), L.hc_attn.scale.as<float>(), L.hc_attn.base.as<float>(), L.in_norm.as<bf16>(),
                                c_.rms_eps, c_.hc_eps, c_.sinkhorn_iters, mixes_.as<float>(), rsq_.as<float>(), pre_.as<float>(), post_.as<float>(),
                                comb_.as<float>(), x_.as<bf16>(), xn_.as<bf16>(), hc_sync_.as<int>(), stream_, l == 0 ? nullptr : attn_.as<bf16>());
    else hc_pre_norm(L.hc_attn, L.in_norm, rows);
    prof_mark(&st_.ms_hc);
    if (L.kda) { kda_layer(l, seqs, M, T, prefill); prof_mark(&st_.ms_kda); }
    else { dsa_layer(l, seqs, M, T, prefill); prof_mark(&st_.ms_dsa); }
    if (fast) glm_hc_pre_decode(hb(), rows, H, L.hc_ffn.fn.as<float>(), L.hc_ffn.scale.as<float>(), L.hc_ffn.base.as<float>(), L.post_norm.as<bf16>(),
                                c_.rms_eps, c_.hc_eps, c_.sinkhorn_iters, mixes_.as<float>(), rsq_.as<float>(), pre_.as<float>(), post_.as<float>(),
                                comb_.as<float>(), x_.as<bf16>(), xn_.as<bf16>(), hc_sync_.as<int>(), stream_, attn_.as<bf16>());
    else { hc_post_apply(rows); hc_pre_norm(L.hc_ffn, L.post_norm, rows); }
    prof_mark(&st_.ms_hc);
    if (probe >= 0) {
      CUDA_CHECK(cudaEventRecord(probe_ev_[2 * l + 1], stream_));
      probe_host += now_ms() - ph0;
    }
    mlp_layer(l, rows, prefill);
    if (!L.moe) prof_mark(&st_.ms_dense);
    if (fast) { if (l + 1 == c_.n_layers) glm_hc_post_decode(attn_.as<bf16>(), post_.as<float>(), comb_.as<float>(), rows, H, hb(), stream_); }
    else hc_post_apply(rows);
    prof_mark(&st_.ms_hc);
    if (layer_hook) { CUDA_CHECK(cudaStreamSynchronize(stream_)); layer_hook(l, hb(), rows); }
  }
  // head: mean over streams → norm → lm_head (only the rows whose logits are needed)
  const int r0 = all_logits ? 0 : rows - 1, nr = all_logits ? rows : 1;
  HIVE_CHECK(nr <= 64, "logits rows");
  const bf16* head_in;
  if (mtp_on()) {  // MTP needs the final normed hidden of every row (its entries pair h_j with the next token)
    hc_stream_mean(hb(), rows, c_.hc, H, fx_.as<bf16>(), stream_);
    rmsnorm(fx_.as<bf16>(), m_.final_norm().as<bf16>(), c_.rms_eps, rows, H, fh_.as<bf16>(), stream_);
    head_in = fh_.as<bf16>() + (size_t)r0 * H;
  } else {
    hc_stream_mean(hb() + (size_t)r0 * c_.hc * H, nr, c_.hc, H, final_x_.as<bf16>(), stream_);
    rmsnorm(final_x_.as<bf16>(), m_.final_norm().as<bf16>(), c_.rms_eps, nr, H, final_xn_.as<bf16>(), stream_);
    head_in = final_xn_.as<bf16>();
  }
  blas_.gemm_bf16_f32out(head_in, m_.lm_head().p(), logits_dev_.as<float>(), nr, c_.vocab, H);
  if (logits_host) CUDA_CHECK(cudaMemcpyAsync(logits_host, logits_dev_.p, (size_t)nr * c_.vocab * 4, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  prof_mark(&st_.ms_head);
  prof_flush();
  if (probe >= 0) {  // HIVE_GLM_LAUNCH_PROBE totals (the stream was synchronized above)
    double gpu = 0;
    for (int l = 0; l < c_.n_layers; ++l) { float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, probe_ev_[2 * l], probe_ev_[2 * l + 1])); gpu += ms; }
    probe_gpu_[probe] += gpu; probe_host_[probe] += probe_host; ++probe_n_[probe];
    if (++probe_steps_ % 50 == 0)
      fprintf(stderr, "[glm-probe] layer fronts per step (last %d rows): GPU span %.2f ms · host enqueue %.2f ms · %ld steps\n", rows,
              probe_gpu_[0] / probe_n_[0], probe_host_[0] / probe_n_[0], probe_n_[0]);
  }
  if (!prefill) ++st_.steps;
  if (mtp_on()) mtp_after_forward(seqs, M, ids, T, prefill, pos_before);
  if (prefill) { seqs[0]->pos += T; seqs[0]->tokens.insert(seqs[0]->tokens.end(), ids, ids + T); }
  else for (int m = 0; m < M; ++m) { seqs[m]->pos += 1; seqs[m]->tokens.push_back(ids[m]); }
  st_.ms_total += now_ms() - t0;
}

void GlmEngine::prefill(GlmSeq& s, const int32_t* ids, int T, float* logits_last) {
  GlmSeq* sp[1] = {&s};
  HIVE_CHECK(s.pos + T <= s.cap, "GLM: sequence capacity exceeded");
  kv_reserve(s, std::min<int64_t>(s.cap, s.pos + T + kKvSlack));  // before any expert-cache slot is lent to the layer-major buffers
  if (tiles_ > 1 && T > max_chunk_) {  // layer-major over up to tiles × max_chunk rows per call
    const int span = tiles_ * max_chunk_;
    for (int off = 0; off < T; off += span) {
      const int n = std::min(span, T - off);
      emb_base_ = off;
      if (n > max_chunk_) prefill_layer_major(s, ids + off, n, off + n == T ? logits_last : nullptr);
      else forward(sp, 1, ids + off, n, true, off + n == T ? logits_last : nullptr, false);
    }
    ex_.warm_quota(2048);
    return;
  }
  for (int off = 0; off < T; off += max_chunk_) {
    const int n = std::min(max_chunk_, T - off);
    HIVE_CHECK(s.pos + n <= s.cap, "GLM: sequence capacity exceeded");
    emb_base_ = off;
    forward(sp, 1, ids + off, n, true, off + n == T ? logits_last : nullptr, false);
  }
  ex_.warm_quota(2048);
}

void GlmEngine::decode(GlmSeq* const* seqs, const int32_t* ids, int M, float* logits) {
  for (int m = 0; m < M; ++m) HIVE_CHECK(seqs[m]->pos + 1 <= seqs[m]->cap, "GLM: sequence capacity exceeded");
  for (int m = 0; m < M; ++m) kv_reserve(*seqs[m], std::min<int64_t>(seqs[m]->cap, seqs[m]->pos + 1 + kKvSlack));
  forward(seqs, M, ids, 1, false, logits, true);
  ex_.after_step();
}

}  // namespace hive::glm
