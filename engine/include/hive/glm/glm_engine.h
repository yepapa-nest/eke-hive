// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash forward pass (text): prefill of one sequence in chunks and batched decode (one token per sequence).
#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "hive/cublas_ops.h"
#include "hive/grow_buf.h"
#include "hive/glm/dsa.h"
#include "hive/glm/glm_decode.h"
#include "hive/glm/glm_experts.h"
#include "hive/glm/glm_model.h"
#include "hive/glm/kda.h"

namespace hive::glm {

// Per-sequence state: KDA conv + recurrent state per KDA layer, DSA latent/indexer caches per DSA layer.
struct GlmSeq {
  int64_t pos = 0, cap = 0;
  std::vector<int32_t> tokens;
  std::vector<DevBuf> conv, state;         // per KDA layer: fp32 [3·24576], fp32 [64·128·128]
  std::vector<GrowBuf> c, kI, gs, pooled;  // per DSA layer — address range for cap positions, memory mapped for kv_tokens (GlmEngine::kv_reserve)
  std::vector<DsaSeqCache> dsa;            // views of the buffers above
  std::vector<int> kda_slot, dsa_slot;     // layer → index (-1 if other type)
  DevBuf conv_ptrs, state_ptrs;            // device arrays of pointers (batched decode)
  bool broken = false;
  size_t device_bytes = 0;
  // MTP (NextN layer): its own DSA cache; entry j = f(main hidden h_j, token x_{j+1}). Entries are written by the main path as soon as
  //   (h_j, x_{j+1}) are known, so entries < mtp_written are exact; draft steps write scratch entries above that, overwritten later.
  GrowBuf mtp_c, mtp_kI, mtp_gs, mtp_pooled;
  int64_t kv_tokens = 0;         // positions backed by memory in every DSA / MTP cache (≤ cap)
  DsaSeqCache mtp_cache{};
  DevBuf h_last;                 // bf16 [hidden]: final normed hidden of the last processed position
  bool h_last_valid = false;
  int64_t mtp_written = 0;
};

// HIVE_GLM_PREDICT_EVAL=1 (decode only, measurement — outputs unchanged): at each MoE layer l the routers of the next MoE layers (distance
//   d = 1, 2) are applied to layer l's FFN input; when layer l+d routes, its real choice is compared with the prediction.
//   pairs = (row, expert) choices · pred = of them predicted · miss = not resident in VRAM at routing time · miss_pred = misses predicted.
//   prefetch precision: of the predicted experts that were not resident when predicted (= what a prefetch would copy), how many were used.
struct GlmPredictStats { uint64_t pairs[3] = {}, pred[3] = {}, miss[3] = {}, miss_pred[3] = {}, pf_issued[3] = {}, pf_used[3] = {}; };

struct GlmForwardStats {
  double ms_attn = 0, ms_moe = 0, ms_total = 0;
  // HIVE_GLM_PROF=1: synchronized per-block timings (adds syncs; for breakdowns only)
  double ms_hc = 0, ms_kda = 0, ms_dsa = 0, ms_dense = 0, ms_router = 0, ms_shared = 0, ms_experts = 0, ms_head = 0, ms_embed = 0, ms_kda_proj = 0, ms_kda_core = 0, ms_predict = 0;
  int steps = 0;
};

class GlmEngine {
 public:
  GlmEngine(GlmModel& m, GlmExperts& x, int max_chunk);
  std::unique_ptr<GlmSeq> new_seq(int64_t cap);
  void init_seq(GlmSeq& s, int64_t cap);   // allocate the per-sequence state into an existing object (derived Seq types)
  // Device logits of the last forward: rows = 1 (prefill, last token) or M (decode), fp32 [rows, vocab]. Valid until the next forward.
  const float* logits_dev() const { return logits_dev_.as<float>(); }
  void reset_seq(GlmSeq& s);
  // KV memory follows use (HIVE_MAX_CTX only reserves addresses): kv_reserve backs the DSA / MTP caches of s up to `tokens` positions,
  //   in steps of HIVE_GLM_KV_STEP positions (default 4096). The memory comes from the free VRAM above HIVE_CACHE_RESERVE_MB; when that is
  //   not enough, tail chunks of the expert cache are given back first (GlmExperts::shrink_cache). Throws if even that is not enough.
  //   reset_seq trims a sequence back to HIVE_GLM_KV_BASE positions (default max_chunk — what every sequence keeps from startup) and returns
  //   the freed memory to the expert cache (regrow_cache).
  void kv_reserve(GlmSeq& s, int64_t tokens);
  // Vision input: rows of the next prefill whose embedding comes from the vision encoder instead of the token table. `row` is the index in
  //   the ids passed to prefill(), `dev` holds n rows of bf16 [hidden] on the device. The caller clears it after the prefill.
  struct EmbedOverride { int64_t row; int n; const bf16* dev; };
  void set_embed_override(std::vector<EmbedOverride> v) { emb_over_ = std::move(v); emb_base_ = 0; }
  // Device memory for `bytes` while HIVE_CACHE_RESERVE_MB stays free: expert-cache tail slots are given back first when needed (as for KV
  //   growth); give_back_room() lets the cache take them again once the memory is freed.
  void make_room(size_t bytes);
  void give_back_room() { ex_.regrow_cache(kv_headroom_); }
  // Positions a call may write past pos + its own rows: verify writes up to 8 rows (decode kernel M ≤ 8) and the MTP draft chain writes
  //   scratch entries up to k ≤ 8 positions ahead — reserving this much more keeps every write inside mapped memory.
  static constexpr int64_t kKvSlack = 16;
  // Prefill T tokens of one sequence (in chunks of ≤ max_chunk); logits of the last token → logits (host fp32 [vocab]) if non-null.
  void prefill(GlmSeq& s, const int32_t* ids, int T, float* logits_last);
  // Decode one token for each of M sequences; logits host fp32 [M, vocab].
  void decode(GlmSeq* const* seqs, const int32_t* ids, int M, float* logits);
  // Speculative decoding with the NextN (MTP) layer — the hived contract (mtp_draft → verify → rollback):
  //   mtp_draft: from token tok (not yet processed by the main model) and the sequence's last hidden, k greedy drafts (conf = MTP softmax
  //   probability of each draft). verify: the main model over [tok, drafts…] (M ≤ 8 rows of one sequence), logits of every row → host.
  //   rollback: keep the first n_keep rows (KDA states restored from the per-row snapshots, positions trimmed, MTP entries written).
  bool mtp_on() const { return mtp_k_ > 0 && m_.mtp() != nullptr; }
  int mtp_k() const { return mtp_k_; }
  int mtp_draft(GlmSeq& s, int32_t tok, int k, int32_t* drafts, float* conf);
  void verify(GlmSeq& s, const int32_t* ids, int M, float* logits_host);
  void rollback(GlmSeq& s, int n_keep);
  // Batched verify (hived HIVE_MTP_BATCH): S sequences, part s = [token, drafts…] of M[s] rows (ids concatenated), one forward of ΣM rows ≤ 8;
  //   logits of every row → logits_host. rollback_batch keeps n_keep[s] rows of part s. A single verify is the S = 1 case of the same path.
  //   enable_batch_verify(parts) allocates KDA snapshots for that many parts (call once, before serving).
  void verify_batch(GlmSeq* const* seqs, const int* M, int S, const int32_t* ids, float* logits_host);
  void rollback_batch(const int* n_keep, int S);
  void enable_batch_verify(int parts);
  int verify_parts_cap() const { return snap_parts_; }
  int prefill_tiles() const { return tiles_; }
  GlmForwardStats& stats() { return st_; }
  GlmPredictStats& predict_stats() { return ps_; }
  cudaStream_t stream() const { return stream_; }
  // Validation hook: called after every layer with the hc streams h [rows, hc, hidden] (device, bf16, stream synchronized).
  std::function<void(int layer, const bf16* h_dev, int rows)> layer_hook;

 private:
  void forward(GlmSeq* const* seqs, int M, const int32_t* ids, int T, bool prefill, float* logits_host, bool all_logits);
  void kda_layer(int l, GlmSeq* const* seqs, int M, int T, bool prefill);
  void dsa_layer(int l, GlmSeq* const* seqs, int M, int T, bool prefill);
  void mlp_layer(int l, int rows, bool prefill);
  void hc_pre_norm(const HcW& w, const DevBuf& norm_w, int rows);
  void hc_post_apply(int rows);
  // current hc-stream rows: the work buffer h_, or a block of the whole-prompt buffer hall_ during a layer-major prefill
  bf16* hcur_ = nullptr;
  bf16* hb() const { return hcur_ ? hcur_ : h_.as<bf16>(); }
  // Layer-major prefill (HIVE_GLM_PREFILL_TILES blocks of max_chunk rows per call, default 4): every layer runs over all blocks before the
  //   next layer, and its MoE runs once over all rows — each routed expert crosses PCIe once per layer per call instead of once per block.
  void prefill_layer_major(GlmSeq& s, const int32_t* ids, int T, float* logits_last);
  int tiles_ = 1;
  bool elastic_ = true;   // HIVE_GLM_PREFILL_ELASTIC: layer-major buffers live in lent expert-cache slots only while a long prompt runs
  DevBuf hall_, xnall_, outall_, postall_, comball_, rlog_all_, rids_all_, rw_all_;
  std::vector<int32_t> h_ids_all_; std::vector<float> h_w_all_;
  // C[rows, N] (row stride ldc) = A[rows, K] (row stride lda) · Wᵀ — BF16 cuBLAS, or FP8 (gemv for rows ≤ 8, dequantize + cuBLAS above)
  void lin(const Mat& W, const bf16* A, int lda, int rows, bf16* C, int ldc);
  // MTP helpers (glm_mtp.cpp)
  void mtp_kv(GlmSeq& s, int64_t pos0, int n, const bf16* h_rows, const int32_t* next_ids);   // exact entries pos0..pos0+n-1
  void mtp_step(GlmSeq& s, int step, int32_t tok, const bf16* h_in, int64_t pos, bf16* h_out);
  DevBuf mtp_ids_d_, mtp_conf_d_, mtp_moe_ws_;
  Fp8BMat* mtp_moe_tab_ = nullptr;
  void mtp_after_forward(GlmSeq* const* seqs, int M, const int32_t* ids, int T, bool prefill, const int64_t* pos_before);
  int mtp_k_ = 0;
  bool verify_ = false;                    // forward() is running a verify chunk (rows = consecutive tokens of one sequence)
  struct VPart { GlmSeq* s; int64_t pos0; int M, r0; };
  std::vector<VPart> vparts_;              // parts of the running / last verify (rows r0 .. r0+M−1 belong to part)
  int vrow_off_[8] = {};                   // verify row → index within its part (DSA position offset)
  std::vector<int32_t> verify_ids_;        // ids of the last verify (all parts, concatenated)
  int snap_parts_ = 1;                     // parts with KDA snapshots (snapshot (p, r, ki) = snap_*_[(p·kmax + r)·n_kda + ki])
  void alloc_snapshots(int parts);
  std::vector<DevBuf> snap_state_, snap_conv_;  // [row r < kmax][kda layer]: KDA state after verify row r
  DevBuf fx_, fh_;                         // [rows, hidden] stream mean / final normed hidden of every row of the last forward
  bf16* mtp_emb_pin_ = nullptr;            // pinned [max_chunk, hidden] embedding rows for MTP entries (read by rmsnorm via UVA)
  DevBuf mtp_en_, mtp_hn_, mtp_cat_, mtp_x0_, mtp_xn_, mtp_h1_, mtp_tmp_, mtp_hout_, one_f_;
  GlmModel& m_;
  GlmExperts& ex_;
  const GlmConfig& c_;
  cudaStream_t stream_ = nullptr;
  Blas blas_;
  int max_chunk_;
  int64_t kv_base_ = 0, kv_step_ = 4096;   // HIVE_GLM_KV_BASE / HIVE_GLM_KV_STEP (positions)
  std::vector<EmbedOverride> emb_over_;
  int64_t emb_base_ = 0;                   // index (in the prefill call's ids) of the first row of the forward / block being embedded
  void apply_embed_override(bf16* x_dev, int64_t base, int rows);
  size_t kv_headroom_ = 0;                 // HIVE_CACHE_RESERVE_MB (hive/cache_reserve.h)
  size_t kv_bytes(const GlmSeq& s, int64_t tokens) const;
  void kv_trim(GlmSeq& s, int64_t tokens);
  int n_kda_ = 0, n_dsa_ = 0;
  std::vector<int> kda_index_, dsa_index_;
  // work buffers (rows ≤ max_chunk)
  DevBuf h_, x_, xn_, attn_, mix_f32_, mixes_, rsq_, pre_, post_, comb_;
  DevBuf qkv_, gpre_, gate_, beta_, fa_, ga_, kda_out_;
  DevBuf qa_, qres_, q_, ckv_, cn_, qI_, kI_raw_, kI_, gs_, wI_, idx_, dsa_out_;
  DevBuf mlp_out_f32_, mlp_out_, sh_gu_, sh_y_, dense_gu_, dense_y_, router_logits_, router_ids_, router_w_, deq_;
  DevBuf ws_, logits_dev_, final_x_, final_xn_;
  size_t ws_bytes_ = 0;
  std::vector<int32_t> h_ids_; std::vector<float> h_w_;
  GlmForwardStats st_;
  GlmPredictStats ps_;
  int pred_eval_ = 0;
  // decode fast path (rows ≤ 8, HIVE_GLM_DECODE_FUSED, default on): glm_decode.h kernels. Per-step tables are pinned host memory read by the
  //   kernels directly (UVA) — no H2D copy on the critical path (it queues behind promotion / prefetch copies on the copy engine).
  bool fused_ = true;
  DevBuf hc_sync_, rlog_;
  // HIVE_CACHE_PRIOR=λ (decode routing biased toward VRAM-resident experts — GlmCachePrior in glm_decode.h); 0/unset = off
  float cache_prior_ = 0.f; int cache_prior_j_ = 2;
  uint8_t* cache_mask_h_ = nullptr;                // pinned [n_moe][E]: residency at routing time, read by the router kernel (UVA)
  DevBuf prior_range_;
  DevBuf post_def_;                                // expert deferral: post coefficients of the layer whose CPU experts are pending                             // [n_moe] running average of the selection-score range
  float** snap_ptr_h_ = nullptr;                   // [n_kda · 16]: verify snapshot pointers per KDA layer, rows 0..7 (conv 8 | state 8)
  float** kda_ptr_h_ = nullptr;                    // [n_kda · 16]: per KDA layer 8 conv-state then 8 recurrent-state pointers
  bf16* xin_pin_ = nullptr;                        // pinned [64, hidden]: MoE input rows for the CPU experts (written by dcopy before the routing sync)
  bf16* emb_pin_ = nullptr; DevBuf iota_;          // decode embedding rows (read by embed_expand) and row ids 0..7
  DsaDecodeRow* dsa_rows_h_ = nullptr; int dsa_max_ctx_ = 0;  // [n_dsa · 8] decode row table
  int32_t* rids_h_ = nullptr; float* rw_h_ = nullptr;   // mapped host router outputs
  int32_t* pf_ids_map_ = nullptr; float* pf_w_map_ = nullptr;  // mapped host next-layer predictions
  bool dec_fast(bool prefill, int rows) const { return fused_ && !prefill && rows <= 8 && !layer_hook; }
  DevBuf pf_ids_d_, pf_w_d_;
  int pf_n_ = 0;
  std::vector<int> next_moe_;
  std::vector<DevBuf> pair_gate_;
  std::vector<int32_t> pf_ids_h_; std::vector<float> pf_w_h_;
  DevBuf pred_logits_, pred_ids_d_, pred_w_d_, p_mix_, p_rsq_, p_pre_, p_post_, p_comb_, p_x_, p_xn_;
  std::vector<std::vector<uint8_t>> pred_nonres_[3];  // [distance][moe layer] → per predicted id: was it non-resident at prediction time
  std::vector<std::vector<int32_t>> pred_[3];  // [distance][moe layer] → ids [rows·K] (empty = none)
  void predict_eval(int l, int rows, const bf16* xn);
  int prof_ = 0;              // HIVE_GLM_PROF: 1 = host-synchronized marks, 2 = CUDA events (no syncs; GPU time between marks)
  double prof_t_ = 0;
  void prof_mark(double* acc);
  std::vector<cudaEvent_t> ev_pool_;
  // HIVE_GLM_LAUNCH_PROBE=N (measurement): every N-th decode step, each layer's front (hc pre, attention, hc pre of the FFN) is timed on
  //   the GPU (events) and on the host (enqueue time). A GPU span close to the host enqueue time means the GPU waited for launches (what a
  //   CUDA graph removes). Totals go to stderr every 50 probed steps. (A variant that held the stream until the whole front was enqueued
  //   hung and was removed.)
  int probe_every_ = 0; long probe_steps_ = 0, probe_n_[2] = {0, 0}; double probe_gpu_[2] = {0, 0}, probe_host_[2] = {0, 0};
  cudaEvent_t probe_ev_[2 * 64] = {};
  std::vector<std::pair<cudaEvent_t, double*>> ev_marks_;
  void prof_flush();
};

}  // namespace hive::glm
