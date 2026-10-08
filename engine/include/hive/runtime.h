// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Runtime — sequence state (caches) + forward pass (shared prefill-chunk/decode path). Batch 1.
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "hive/cublas_ops.h"
#include "hive/engram_hash.h"
#include "hive/expert_store.h"
#include "hive/model.h"
#include "hive/topk.h"
#include "hive/vision.h"
#include "hive/host_image.h"
#include "hive/layer_yield.h"  // HIVE_LAYER_YIELD (yield at layer boundaries — host scaffolding)

#include <cstdio>

namespace hive {

struct SnapshotFence {
  cudaEvent_t event = nullptr;
  SnapshotFence() { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
  ~SnapshotFence() { if (event) { cudaEventSynchronize(event); cudaEventDestroy(event); } }
  void wait() const { CUDA_CHECK(cudaEventSynchronize(event)); }
};

struct Seq {
  // Row tag for the expert trace (HIVE_EXPERT_TRACE) — the session pool reuses Seq objects, so a request boundary is (uid, pos returning to 0).
  static uint32_t next_uid() { static uint32_t n = 0; return ++n; }
  uint32_t uid = next_uid();
  uint32_t generation = uid;
  int64_t pos = 0;   // tokens processed
  bool broken = false;  // an exception during forward left the state half-advanced (compressor, engram history) — must not be reused; reset_seq clears it
  int64_t cap = 0;   // max tokens
  std::vector<DevBuf> ring;                       // [L] bf16 [win, D]
  std::map<int, DevBuf> comp_cache, idx_k_cache;  // per kv-source layer
  std::map<int, DevBuf> comp_state_kv, comp_state_score;
  std::vector<int64_t> engram_history;
  std::vector<int32_t> tokens;
  // DSpark (MTP): per-stage window KV ring for the drafter (main target-layer hidden -> main_proj -> per-stage wkv; written by mtp_sync for every processed position) + target-layer hidden of the last processed position
  std::vector<DevBuf> mtp_ring;   // [n_mtp] bf16 [win, D]
  DevBuf mtp_hidden;              // bf16 [n_targets·dim]
  bool mtp_hidden_valid = false;
  int64_t mtp_pos = -1;           // position of mtp_hidden (= last position written to the ring)
  mutable std::shared_ptr<SnapshotFence> snapshot_fence; // destroyed before source device buffers
  ~Seq() { if(snapshot_fence) snapshot_fence->wait(); } // fence may also be owned by an image
};

// Host image of a sequence state — for conversation KV reuse (end-of-prompt checkpoints) and RAM storage outside the session pool.
//   Compressed KV and index-key caches are append-only, so only the **completed group rows [0, pos/ratio)** are stored (later rows are rewritten
//   before they are read after a restore — same logic as rollback). Window ring, compressor partial state, MTP ring/hidden, engram history and tokens are copied whole. A 9.7K-token conversation is a few tens of MB.
struct SeqImage {
  int64_t pos = 0;
  uint32_t generation = 0;
  std::vector<int32_t> tokens;
  std::vector<int64_t> engram_history;
  std::vector<HostImageBuffer> ring, mtp_ring;
  std::map<int, HostImageBuffer> comp, idx, st_kv, st_score;
  HostImageBuffer mtp_hidden;
  bool mtp_hidden_valid = false;
  int64_t mtp_pos = -1;
  std::shared_ptr<SnapshotFence> fence; // waits before buffers are freed
  ~SeqImage() { if(fence) fence->wait(); } // shared_ptr destruction alone does not wait if Seq still owns it
  size_t allocated_bytes(std::set<const void*>& seen) const {
    if (!seen.insert(this).second) return 0;
    size_t b = sizeof(*this) + tokens.capacity() * 4 + engram_history.capacity() * 8 + mtp_hidden.allocated_bytes(seen);
    b += (ring.capacity()+mtp_ring.capacity())*sizeof(HostImageBuffer);
    for (auto& v : ring) b += v.allocated_bytes(seen);
    for (auto& v : mtp_ring) b += v.allocated_bytes(seen);
    for (auto* m : {&comp, &idx, &st_kv, &st_score}) for (auto& [l, v] : *m) b += v.allocated_bytes(seen) + sizeof(v) + 64;
    return b;
  }
  size_t bytes() const { std::set<const void*> seen; return allocated_bytes(seen); }
};

struct RuntimeOptions {
  int max_chunk = 64;        // max tokens processed at once
  int max_batch = 8;         // max sequences in a batched decode (<= max_chunk, <= 64)
  int64_t max_ctx = 4096;    // max sequence length
  std::string dump_dir;      // per-layer dumps (oracle comparison)
  std::string inject_dir;    // per-layer injection: before layer l, overwrite h and pre_mix with golden s{step}_h_L{l-1}.f32 (isolates per-layer error; eager mode only)
  bool cpu_for_misses = true;  // decode: run missed experts on the CPU (false = pull everything into VRAM)
  int prefill_threshold = 1024;  // M >= this is treated as prefill (experts streamed over PCIe). Below it, the decode policy applies (cache hits + CPU misses, grouped kernels: a few launches per layer) —
                               // streaming experts for a short chunk (0.7 ms per expert) turns a 10-token turn into seconds. CPU costs ~0.3 ms per expert + 0.05 ms per row
  int64_t index_budget = 32ll << 20;  // max elements in the indexer score buffer
  int decode_gpu_share = 1;   // decode: per layer, this many missed experts are handled by the GPU via staging DMA (<= 8 staging slots; the cache is not touched)
  int promote_per_token = 4;  // max experts promoted per token on the side stream (the copy engine is shared with main-stream H2D)
  int promote_misses = 0;     // >0: instead of score-based promotion, asynchronously promote up to this many experts missed in this step into LRU slots (simulated: hit rate +7.5 pt).
                              //   PCIe budget = N x 18.8 MB/token (16 -> ~11 ms @ 28 GB/s) — choose by A/B (0/8/16)
  float promote_miss_ratio = 0.f;  // >0: promotions per step = min(promote_misses, ceil(unique misses x this ratio)) — bandwidth split in the style of FreeToken (arXiv 2608.16157):
                                  //   PCIe 27.9 / CPU expert host bandwidth ~110 GB/s (measured) ~= 0.25. `--promote-misses auto[:cap]` enables it with 0.25 and cap 32
  float promote_min_score = 2.f;
  bool mtp_cache = false, phase_score = false, promote_score_victims = false;
  size_t promote_byte_budget = 0;  // 0 = no additional cap; experimental knobs are default off
  float score_decay = 0.97f;
  int warm_cap = 2048;        // max experts warm_cache() uploads at once after a prefill (>= prefill_threshold tokens) (2048 ~= 38 GB ~= 1.5 s)
  bool vision = true;         // load the vision encoder
  int decoder_tail = 128 * 21;  // in prefill, decoder layers (20+) compute only the last this-many rows (0 = off). 128 x (40-20+1) is the lower bound for exactness
  bool decoder_replay = false;  // Decoder SWA Bounded Replay recipe: tail rows do not see window keys before the replay span — use with decoder_tail=128 (approximate)
  int prefill_tile = 1;  // layer-major tiling: group a prefill chunk into this many sub-chunks (each <= max_chunk) so each encoder layer streams its experts once (1 = off)
  int vision_max_patches = 0; // 0 = model limit (max_image_tokens x r^2); can be lowered for tests
  bool vision_lazy = true;    // load the encoder on the first image (~1.7 GB VRAM) and unload it when idle — that VRAM goes to expert slots
  double vision_idle_ms = 120000;  // unload after this long without an image request
  bool mtp = true;            // DSpark drafting (enabled only when the model carries MTP stages)
  int sampler_cands = 1024;   // GPU sampler: after the head, per row copy the top-N logit candidates (value, index) plus max and sum-exp to the host — the host handles only candidates instead of a 129K partial sort (0 = off)
};

// One image: a span inside this chunk (start = position within the chunk). types follow the reference image_token_types (0 start, 1 image, 2 newline, 3 end).
struct ImageInput {
  int start = 0, n_vit_h = 0, n_vit_w = 0;
  std::vector<int8_t> types;
  const bf16* patches_dev = nullptr;  // [n_vit_h·n_vit_w, 3·p·p] bf16 (device)
};

// HIVE_BATCH_PREFILL: one sequence's share of forward_multi (one chunk = that sequence's [pos, pos+M) — text only).
//   Outputs: next (argmax, -1 without logits), tail_rows (rows in tail mode, else 0). upper_needed matches forward's argument of the same name.
struct PrefillPart {
  Seq* seq = nullptr;
  const int32_t* ids = nullptr;
  int M = 0;
  std::vector<float>* logits_out = nullptr;
  bool upper_needed = true;
  int32_t next = -1;
  int tail_rows = 0;
};
class HostStager;  // host-tile h staging (runtime.cpp)

struct ForwardStats {
  int n_routed = 0, n_hit = 0, n_cpu = 0, n_streamed = 0, tail_rows = 0;
  int n_dma_rows = 0, n_promoted = 0;
  int n_cpu_e = 0, n_dma_e = 0;  // experts (not rows) of missed experts computed on the CPU / moved by demand DMA (prefill profile; pre-copies not included)
  double ms_cpu_span = 0;  // overlaps GPU; ms_cpu_wait now means actual host blocking time
  std::vector<int> row_hit, row_cpu, row_dma;
  double ms_total = 0, ms_cpu_wait = 0, ms_host = 0;  // ms_host = host preparation in decode layers (GPU idle time)
  // HIVE_PREFILL_SPLIT=balance (all 0 when off): per streaming-prefill layer, the chosen DMA share and predicted vs actual completion (cumulative — read as per-chunk deltas).
  //   split_layers/split_cold/split_share_sum = layers decided, of which cold-start (0.7) layers, sum of shares (mean = sum/layers).
  //   CPU pair = CPU work completion (pool finish time - start) of layers that had a prediction (not cold start). GPU pair = the same layer's GPU window (events) — events are read at the next layer's decision,
  //   so the GPU pair of a chunk's last layer is added to the next forward (pairs always belong to the same layer).
  int split_layers = 0, split_cold = 0, split_cpu_n = 0, split_gpu_n = 0;
  double split_share_sum = 0, split_pred_cpu_ms = 0, split_act_cpu_ms = 0, split_pred_gpu_ms = 0, split_act_gpu_ms = 0;
};

class Runtime {
 public:
  Runtime(Model& model, ExpertStore& store, const EngramHash* eh, const RuntimeOptions& opt);
  ~Runtime();
  std::unique_ptr<Seq> new_seq() const;
  // Session reuse (session pool): keep the buffers, clear only the state (ring 0, compressor state reset, position, tokens, history, draft ring). Avoiding cudaMalloc at run time
  //   moves peak VRAM to startup (session KV was the only run-time allocation).
  void reset_seq(Seq& s) const;
  // Same Runtime only. Default sync; HIVE_CKPT_ASYNC uses pinned buffers/event fences.
  void save_image(const Seq& s, SeqImage& img, const SeqImage* base = nullptr) const;
  void load_image(Seq& s, const SeqImage& img) const;
  void quiesce_after_host_error();  // CUDA fatal failures still abort via CUDA_CHECK
  // Process ids[M] appended to seq. logits_out (host, [V]) gets the last token's logits (if there is a head). Returns argmax (-1 if none)
  int32_t forward(Seq& seq, const int32_t* ids, int M, std::vector<float>* logits_out, ForwardStats* stats) {
    return forward(seq, ids, M, nullptr, logits_out, stats);
  }
  // images: images in this chunk (each span must lie entirely inside the chunk). nullptr if none.
  int32_t forward(Seq& seq, const int32_t* ids, int M, const std::vector<ImageInput>* images, std::vector<float>* logits_out,
                  ForwardStats* stats, bool upper_needed = true);
  // A chunk whose next chunk runs in decoder tail mode may skip the tail layers (20+) with identical results — exposes the condition so the caller (hived) decides.
  //   (tail layers have no KV of their own and their ring is absorbed by the next chunk's "may be dirty" tail span; only the last chunk's logits and MTP hidden are used)
  bool tail_mode_for(int M) const;
  bool has_vision() const { return vision_can_; }  // can image requests be accepted (lazy load uploads on the first one)
  void ensure_vision();             // load the encoder (if not loaded yet)
  void release_vision_if_idle();    // unload if unused for more than vision_idle_ms — called by the daemon loop between steps
  // Batched decode: one token each for B sequences (<= max_batch). next[b] = argmax, logits (if requested) per row.
  void forward_batch(std::vector<Seq*>& seqs, const int32_t* ids, std::vector<int32_t>& next, std::vector<std::vector<float>>* logits_out,
                     ForwardStats* stats);
  int max_batch() const { return opt_.max_batch; }
  int prefill_tile() const { return opt_.prefill_tile; }  // effective value (the constructor sets it to 1 when conditions are not met)
  // Number of sub-chunks one forward can hold (= resident tiles prefill_tile + host tiles). Equals prefill_tile when host tiles are off.
  int prefill_slots() const { return opt_.prefill_tile + host_slots_; }
  int host_slots() const { return host_slots_; }
  bool batch_prefill() const { return batch_prefill_; }  // HIVE_BATCH_PREFILL (read by the constructor)
  int slots_for(int M) const;  // sub-chunks a chunk of M rows occupies (same as the tile plan: 1 if M <= max_chunk)
  // Text prefill chunks of several sequences in one forward — per layer: front part of all sub-chunks of all sequences -> experts once -> tail (same layer-major order as tiling).
  //   sum slots_for(M) <= prefill_slots(). Each sequence keeps its own KV, positions and attention (sequences never mix). On exception every part's seq is marked broken.
  //   Single-sequence chunks that use host tiles (forward hands over when M > max_chunk x prefill_tile) also take this path.
  void forward_multi(std::vector<PrefillPart>& parts, ForwardStats* stats);
  // Contract (hived <-> runtime): hived sets true for the duration of a multi-chunk prefill and false when done. Whether promotion is paused in decode steps
  //   between chunks is decided by the runtime via HIVE_PREFILL_PAUSE_PROMOTE (default off) — when off, this flag has no effect.
  void set_prefill_pending(bool on) { prefill_pending_ = on; }
  bool prefill_pending() const { return prefill_pending_; }
  // Short-prefill (TTFT) switches — read by the constructor with the env_on convention (runtime.cpp, all default off = baseline path).
  //   The setter exists for the A/B bench (engine/tests/bench_short_prefill.cpp), which toggles them within one model load — call only between forwards.
  struct ShortPrefillOpts {
    bool tail_short = false;      // HIVE_PREFILL_TAIL_SHORT: tail layers of a prefill forward (rows < prefill_threshold) use the short-prefill path (moe_decode_experts)
    bool prefetch_scale = false;  // HIVE_PREFETCH_SCALE: predicted prefetch rows = previous chunk rows x (current rows / previous rows)
    int adapt_rows = 0;           // HIVE_PREFILL_SHORT_ADAPT: streaming layers with rows <= N use their own adaptive DMA share instead of the fixed share (HIVE_DMA_FRAC_PREFILL) (0 = off)
    bool prof = false;            // HIVE_PREFILL_PROF: per prefill forward, a [prefill-prof] phase breakdown (host clock + GPU events)
    bool multi_tail_short = false;  // HIVE_PREFILL_MULTI_TAIL_SHORT: forward_multi upper-layer units (rows < threshold) use the short-prefill path
    int small_rows = 0;             // HIVE_PREFILL_SMALL: row lower bound (0 = off) — experts of chunks with bound <= M < threshold use the streaming path (tail mode stays off)
  };
  const ShortPrefillOpts& short_prefill_opts() const { return spo_; }
  void set_short_prefill_opts(const ShortPrefillOpts& o) { spo_ = o; }
  // Bench only (same reason): vary the short-prefill per-layer DMA cap (constructor value of HIVE_SHORT_DMA_CAP) — values below 1 are ignored (absorbed)
  int short_dma_cap() const { return short_dma_cap_; }
  void set_short_dma_cap(int n) { if (n >= 1) short_dma_cap_ = n; }
  std::vector<uint64_t> prefetch_stats() const { return {prefetch_issued_, prefetch_used_, prefetch_wasted_}; }
  double prefetch_copy_ms() const { return prefetch_copy_ms_; } // completed side-stream samples, overlaps compute
  size_t snapshot_pool_cached_bytes() const { return snapshot_pool_ ? snapshot_pool_->cached_bytes() : 0; }
  void trim_snapshot_pool() { if(snapshot_pool_) snapshot_pool_->trim(); }
  // Call after prefill: fill the VRAM cache with the top experts by usage score (synchronous). Returns the number uploaded
  int warm_cache();
  int warm_cap() const { return opt_.warm_cap; }          // HIVE_WARM_BUSY_CAP (hived): the cap is lowered for one call while other requests decode
  void set_warm_cap(int n) { opt_.warm_cap = n; }
  // HIVE_WARM_DEFER (hived): the warm after a prefill as paced promotions over the following decode steps instead of one synchronous upload (see runtime.cpp)
  void warm_defer(int per_step);
  int warm_left() const { return warm_left_; }
  int image_patch_bytes() const;  // bytes of one bf16 image patch (3 x patch x patch x 2) — hived checks a request's image geometry against its byte count
  int promote_after_step();
  // Warm start: seed scores from a saved resident-key list (score order) and upload until the slots are full (synchronous, all). Returns the number uploaded
  int warm_from_keys(const std::vector<int32_t>& keys);
  // Sleep/wake (hived {"op":"sleep"/"wake"} — see the "Z1" header comment in runtime.cpp). The default path never calls these.
  //   sleep_release: only between forwards (after hived has confirmed zero in-flight requests). Device sync -> commit promotions -> detach vision encoder and buffers inside the elastic range -> free store slots and staging.
  //     Returns device bytes freed (bookkeeping value — the caller measures with cudaMemGetInfo). 0 if already asleep.
  //   wake_restore: re-acquire slots and staging (ExpertStore::wake_alloc — reserve_bytes is the HIVE_CACHE_FIT operating margin, 0 = same count as before sleep) and rebind the buffers inside the
  //     elastic range (Work large layout, tiles, host-tile staging, multi-sequence mbufs) at their new addresses. Returns physical slot count, or -1 = not enough VRAM, still asleep (reason in *why).
  //   wake_warm: re-upload the resident keys saved at sleep (score order) (synchronous, scores unchanged — ExpertStore::restore_keys). Returns the number uploaded.
  size_t sleep_release();
  int wake_restore(size_t reserve_bytes, std::string* why);
  int wake_warm(const std::vector<int32_t>& keys, const std::function<void(int, int)>& progress = {});
  bool asleep() const { return asleep_; }
  // HIVE_LAYER_YIELD — at layer boundaries of a long prefill forward (forward, forward_multi; original rows >= prefill_threshold) run hived's yield body
  //   ("T11" header comment in runtime.cpp, hive/layer_yield.h). hived installs it once (empty run or period 0 = off). The body (run) calls back into this runtime:
  //   forward (rows <= R — R is run's first argument: only the first R rows of the outer prefill's Work are lent, so larger forwards, forward_multi and images must not be called),
  //   forward_batch, mtp_draft, forward_verify(_batch), rollback(_batch), save_image/load_image/reset_seq (on sequences other than the outer prefill's) are allowed;
  //   warm_cache is not (long synchronous upload — hived skips it inside a yield). Inside a yield, promotion is paused (same flag as set_prefill_pending) and elastic slots are not lent.
  void set_layer_yield(ly::Hooks h) { ly_.h = std::move(h); }
  bool layer_yield_on() const { return ly_.on(); }
  bool in_layer_yield() const { return ly_.depth > 0; }
  const ly::Stats& layer_yield_stats() const { return ly_.st; }
  double layer_yield_resume_ms() const { return ly_.last; }  // time of the last resume (or of the outer forward's start) (now_ms)
  // Share of the paused forward's work done at the current layer boundary (0..1) — hived's HIVE_LAYER_YIELD_MID estimates the forward's remaining rows
  //   from it. Encoder layers (all rows) carry 90 % of a decoder-tail forward, the tail layers (128 rows) the rest; without the tail, layers are equal.
  double layer_yield_progress() const { return ly_progress_; }
  // HIVE_LAYER_YIELD_INTRA ("T11b" comment in runtime.cpp): yield points inside a layer, decode only. layer_yield_kind() = kind of the point
  //   currently yielding (ly::kLayer at layer boundaries — hived admits only there). layer_yield_intra_on() = the switch was read on (the runtime
  //   still skips intra points when HIVE_CACHE_ELASTIC is off or incomplete, or CPU misses are off — see the startup line).
  int layer_yield_kind() const { return ly_.kind; }
  bool layer_yield_intra_on() const { return ly_.intra; }
  const RuntimeOptions& opt() const { return opt_; }
  // ---- DSpark speculative decoding (single sequence) ----
  //   Draft: target-layer hidden at seq's last processed position (seq.mtp_pos) + next token tok (position seq.pos) -> a block of B drafts d1..dB (positions pos+1..) and confidences.
  //   Verify: forward_verify processes [tok, d1..dk] (M = k+1 rows) and returns per-row logits (M x V, host). It snapshots ring, compressor state and history so that
  //   rollback(n_keep) keeps only the first n_keep rows, reverts state written by the rest and then updates the draft ring. Always call rollback after forward_verify
  //   (n_keep = M when everything is accepted).
  // ---- GPU sampler candidates (valid after forward_batch / forward_verify; row = that call's row) ----
  //   Write 1/temperature into row_inv_temp(m) before the call (sum-exp reflects temperature). Candidates are in ascending index order, padded with -1.
  int n_cands() const { return opt_.sampler_cands; }
  float* row_inv_temp() { return cand_it_h_; }
  const int32_t* cand_idx(int m) const { return cand_idx_h_ + (size_t)m * opt_.sampler_cands; }
  const float* cand_val(int m) const { return cand_val_h_ + (size_t)m * opt_.sampler_cands; }
  float row_max(int m) const { return cand_max_h_[m]; }
  float row_sumexp(int m) const { return cand_sum_h_[m]; }
  int32_t row_argmax(int m) const;
  bool mtp_enabled() const { return mtp_on_; }
  int mtp_block() const;
  void mtp_draft(Seq& seq, int32_t tok, std::vector<int32_t>& drafts, std::vector<float>& conf, ForwardStats* stats);
  void forward_verify(Seq& seq, const int32_t* ids, int M, std::vector<float>& logits_rows, ForwardStats* stats);
  void rollback(Seq& seq, int n_keep);
  // HIVE_MTP_VERIFY2 (env_on, constructor option block): run forward_verify on the decode path (decode_layer — layer graphs, fused attention kernel, moe_decode_experts,
  //   HOST_FAST/EARLY_ROUTE). Outside its conditions (fuse_ off, dump/inject, rows > mtp_batch_rows()) falls back to the forward path (absorbed). Numeric differences: verify_decode header comment in runtime.cpp.
  // HIVE_MTP_BATCH (env_on): verify several sequences in one forward — part p = one sequence's [ids, M_p >= 1] (M_p = 1 = a plain decode row without drafts).
  //   Rows are concatenated in part order (sum M_p <= mtp_batch_rows()). Logits = sum M_p rows (part order); GPU sampler candidate rows use the same numbering. Always follow with rollback_batch
  //   (n_keep[p] in [1, M_p], part order). Decode path only (this path regardless of HIVE_MTP_VERIFY2).
  struct VerifyPart { Seq* seq = nullptr; const int32_t* ids = nullptr; int M = 0; };
  void forward_verify_batch(std::vector<VerifyPart>& parts, std::vector<float>& logits_rows, ForwardStats* stats);
  void rollback_batch(const std::vector<int>& n_keep);
  bool mtp_batch_enabled() const { return mtp_on_ && mtp_batch_ && v2_ok_; }
  // {"op":"set","mtp_batch":0|1} (hived): switch batched speculation at run time for interleaved A/B in one process. Engine thread only.
  //   Has an effect only when the verify decode path was allocated at startup (HIVE_MTP_VERIFY2 or HIVE_MTP_BATCH — v2_ok_); otherwise mtp_batch_enabled() stays false.
  void set_mtp_batch(bool on) { mtp_batch_ = on; }
  int mtp_batch_rows() const { return v2_ok_ ? v2_rows_ : 0; }  // row cap for one verify (decode kernel M <= 8, row table max_batch)
  bool mtp_verify2() const { return verify2_ && v2_ok_; }
  void set_mtp_verify2_for_test(bool on) { verify2_ = on; }
  void set_mtp_verify_rowind_for_test(bool on) { verify_rowind_ = on; }  // CLI --verify2-test only (compare old/new path in the same process)
  long graph_captures() const { return graph_captures_; }     // CUDA graphs captured so far (the gate filters out step-cost samples that included a capture — HIVE_MTP_GATE3)
  // (HIVE_MTP_BATCH gate) number of times run_graph executed the **first use** of a graph key eagerly without a graph. First use (eager) -> second (capture) -> replay from the third,
  //   so the first step of a new row count or Tb bucket is slower than replay (hundreds of launches). Filtering captures alone lets that first sample into the cost table (observed: a
  //   batched c4 first 5-row verify of 47.5 ms vs T(4) 33 ms settled as "+14.5 ms per row" and every later draft decision was rejected, never re-measured). The batch gate filters samples by this count.
  long graph_eager_runs() const { return graph_eager_; }
  // Test only (CLI --verify2-test): while on, collect routing tables of decode-policy layers (moe_decode_experts — main and draft layers) in layer order (at the xtrace_layer site — entry = [layer, M, k, ids(M*k)]).
  //   Comparing expert sets per row and layer across two paths/runs tells whether a logit difference is a "routing flip (cascading amplification)". Off (default) does nothing.
  void set_route_capture(bool on) { route_cap_ = on; route_log_.clear(); }
  const std::vector<std::vector<int32_t>>& route_log() const { return route_log_; }

 private:
  struct Work;
  // HIVE_CACHE_ELASTIC: prefill-only work buffers <-> elastic expert-cache slots (lend, reclaim, switch Work between large/small layout — runtime.cpp header comment)
  struct ElasticScope;
  Model& model_;
  ExpertStore& store_;
  const EngramHash* eh_;
  RuntimeOptions opt_;
  cudaStream_t st_ = nullptr, side_ = nullptr, promo_ = nullptr;  // main, streaming/DMA share, promotion (so it uses a separate copy engine)
  cudaStream_t snapshot_st_ = nullptr;
  cudaStream_t hc_side_ = nullptr;                // decode sinkhorn branch (overlaps attention) — fork/join inside the graph
  cudaEvent_t hc_fork_[2] = {nullptr, nullptr}, hc_join_[2] = {nullptr, nullptr};
  std::unique_ptr<Blas> blas_;
  std::unique_ptr<Work> w_;
  // Expert usage trace (to analyse how experts move across steps) — only with HIVE_EXPERT_TRACE=<file>.
  //   Records (little endian): 'S' step header{u8 kind (0 batched decode, 1 chunk/single, 2 verify, 3 draft), u16 M, u16 n, f64 unix_ms, n x (u32 uid, u32 pos)} — n = M for batches, else 1 (one sequence)
  //     If M >= 65535, u16 M = 0xFFFF marker plus one extra trailing tag {0xFFFFFFFF, actual M} (counted in n — older analysers still skip the right length)
  //   'L' raw layer{u8 layer, u16 k, u16 M, M*k x u16 expert} (decode-policy path), 'H' layer histogram{u8 layer, u16 E, u32 M, E x u16 count} (streaming prefill)
  //   Analysis: tools/analyze_expert_trace.py. ~0.5 KB per decode token.
  FILE* xtrace_ = nullptr;
  double xtrace_flush_ms_ = 0;
  void xtrace_step(int kind, int M, Seq* const* seqs, int n);
  void xtrace_layer(int l, int M, int k, const int32_t* ids);
  void xtrace_hist(int l, int M, int k, int E, const int32_t* ids);
  TopK topk_;
  std::unique_ptr<Vision> vision_;
  bool vision_can_ = false;
  int vision_max_patches_ = 0;
  double vision_last_use_ = 0;
  // Shared across layers (reference SharedAttentionRuntime)
  const uint8_t* shared_comp_kv_ = nullptr;   // packed fp4 compressed KV cache (kv_pack.h)
  const uint8_t* shared_index_k_ = nullptr;   // packed fp4 index-key cache
  int shared_topk_cols_ = 0;   // number of columns of the shared topk
  bool have_candidates_ = false;
  bool in_prefill_ = false;  // this forward is a prefill chunk (original M >= threshold) — expert and promotion policy stay prefill even when tail mode shrinks M
  bool score_prefill_ = false;  // scoring phase, including short multi-token prefill, excluding verify/batched decode
  std::shared_ptr<HostImagePool> snapshot_pool_;
  int64_t win_min_pos_ = 0;  // prefill window indices never look before this position (> 0 only during tail replay)
  std::vector<int64_t> seq_hashes_;
  int step_ = 0;  // forward call number (dump file prefix s{step}_)
  // Profiling (HIVE_PROFILE=1): accumulated GPU time per phase (events)
  bool profile_ = false;
  int profile_every_ = 1;  // HIVE_PROFILE=N: only every N steps (service uses 64 — creating/printing 400 events per token is needless load)
  struct Prof { std::vector<cudaEvent_t> ev; std::vector<std::string> names; };
  Prof prof_;
  void pmark(const char* name);
  void preport(int M);
  cudaEvent_t promo_evt_ = nullptr;
  // Prefill streaming: per staging slot (copy done / slot freed) events; the side stream copies 8 ahead
  std::vector<cudaEvent_t> stage_copied_, stage_freed_;
  std::vector<char> stage_used_;
  uint32_t stage_next_ = 0;  // engram hashes of this forward (built by layer 1, reused by layer 14)

  void layer_forward(Seq& seq, int l, int M, int64_t start_pos, ForwardStats* stats, bool skip_compress = false);
  void attention(Seq& seq, const LayerWeights& L, int l, int M, int64_t start_pos, bool skip_compress);
  void compress_source(Seq& seq, const LayerWeights& L, int l, int M, int64_t start_pos);
  // Layer pieces shared by prefill and decode: hc mix before attention, hc mix before ffn, layer tail (acc -> bf16 -> hc_post -> pre_mix)
  void hc_attn_pre(const LayerWeights& L, int M);
  void hc_ffn_pre(const LayerWeights& L, int M);
  void layer_tail(int l, int M);
  // ---- Batched decode (row = sequence, one token each) ----
  // One layer = [host prep] -> graph A (previous layer tail + engram + attention + hc + router + shared expert + D2H) -> sync -> experts (eager).
  void decode_layer(std::vector<Seq*>& seqs, int l, int M, ForwardStats* stats);
  int attention_decode_host(std::vector<Seq*>& seqs, const LayerWeights& L, int l, int M);  // writes the row table, returns the indexer Tmax
  void attention_decode_dev(const LayerWeights& L, int l, int M, int Tmax);
  void engram_host(const LayerWeights& L, int M, const int32_t* ids_host, std::vector<Seq*>* seqs, Seq* seq);
  // HIVE_ENGRAM_SSD read-ahead (off or not on SSD = no-op): at forward start, compute layer 1/14 row addresses for this forward's tokens (row m's sequence = row_seqs[m] or seq)
  //   in advance (EngramHash::peek — does not change history) and issue the reads. Hashes are built by the read workers (the engine thread only copies).
  void engram_ssd_hint(const int32_t* ids, const int8_t* is_image, int M, Seq* const* row_seqs, Seq* seq);
  void engram_dev(const LayerWeights& L, int M);
  void moe_router_shared(const LayerWeights& L, int M, bool copy_acts);  // router -> shared expert -> issue D2H of routing results (+activations)
  void moe_experts(Seq& seq, const LayerWeights& L, int l, int M, ForwardStats* stats);         // prefill and small M: per-expert kernels
  // Expert inputs/outputs of one sub-chunk (routing tables in mapped pinned host memory, activations and accumulators on device)
  struct SubChunk { int M; const int32_t* route_ids_h; const float* rw_h; const uint8_t* xq_d; const uint8_t* xs_d; float* acc_d; };
  void moe_experts_multi(const LayerWeights& L, int l, const std::vector<SubChunk>& subs, ForwardStats* stats);
  void layer_front(Seq& seq, int l, int M, int64_t start_pos, bool skip_compress = false);  // layer front part (excluding experts and tail); skip_compress for multi-sequence tail layers
  void prologue_rows(const int32_t* rows, int n, int64_t start);  // embedding -> h, pre_mix identity, image marks 0 (chunk head of forward)
  int32_t head_last_row(int M, std::vector<float>* logits_out);   // last row -> logits/argmax (forward's default head)
  // Tile slots (inter-layer state of sub-chunks 2..T). Swapped with Work via std::swap (tile_select / tile_xchg).
  struct TileSlot {
    DevBuf h, pre_mix, idx, xq, xs, acc, pre_f, post_f, comb_f;
    int32_t *route_ids_h = nullptr, *route_ids_d = nullptr, *pos_h = nullptr, *pos_d = nullptr, *ids_h = nullptr, *visible_h = nullptr, *visible_d = nullptr,
            *gpos_h = nullptr, *gpos_d = nullptr;
    float *rw_h = nullptr, *rw_d = nullptr;
    int8_t *is_image_h = nullptr, *is_image_d = nullptr;
    int topk_cols = 0;
    std::vector<int64_t> hashes;
  };
  // Prefill prefetch state: pf_ = (expert, staging slot) in copy order, pf_l_ = target layer, last_rows_[l][e] = per-layer rows of the previous prefill (prediction)
  std::vector<std::pair<int, int>> pf_;
  int pf_l_ = -1;
  std::vector<std::vector<int>> last_rows_;
  void prefetch_layer_experts(int l, int rows_now = 0);  // rows_now = this chunk's rows for this layer (used only by HIVE_PREFETCH_SCALE; 0 = unknown -> no scaling)
  // State and instrumentation (see the ShortPrefillOpts comment). PrefillProf is defined in runtime.cpp (incomplete type — shared_ptr so the fake-runtime tests compile)
  ShortPrefillOpts spo_;
  bool small_fwd_ = false;  // this forward is a HIVE_PREFILL_SMALL chunk (set at forward start like in_prefill_, cleared at the end)
  bool prefill_short_moe(int M) const;  // use the short-prefill path for this layer's experts (tail_short, prefill forward, rows < threshold, CPU misses on)
  struct PrefillProf;
  std::shared_ptr<PrefillProf> pprof_;
  std::vector<int> step_miss_;  // expert keys (layer, expert) missed in this step (decode path) — input to promote_misses
  std::vector<int> draft_miss_;
  uint64_t prefetch_issued_ = 0, prefetch_used_ = 0, prefetch_wasted_ = 0;
  cudaEvent_t prefetch_begin_ = nullptr, prefetch_end_ = nullptr;
  bool prefetch_timing_pending_ = false;
  double prefetch_copy_ms_ = 0;
  std::vector<TileSlot> tiles_;
  // ---- Host tiles (HIVE_PREFILL_HOST_TILES) and multi-sequence prefill (HIVE_BATCH_PREFILL) ----
  //   The last host_slots_ entries of tiles_ are host tiles: h lives in pinned host memory (HostStager) and is uploaded to a staging buffer only while visited (tail + front).
  //   The remaining inter-layer state (pre_mix, idx, xq/xs, acc, pre_f/post_f/comb_f, mapped tables) stays in the slot's own device buffers (all needed in the expert phase).
  int host_slots_ = 0;
  bool batch_prefill_ = false;
  std::unique_ptr<HostStager> stager_;
  // Multi-sequence unit (= one sub-chunk): slot, sequence, inter-layer globals (they differ per sequence — in single-sequence tiling all sub-chunks shared the same values, so globals sufficed)
  struct UnitCtx {
    int part = 0, slot = 0, M = 0, off = 0; int64_t start = 0; bool last = false, up = false, tail = false;
    const uint8_t* comp_kv = nullptr; const uint8_t* index_k = nullptr; bool have_cand = false;
    int64_t win_min = 0; int mh_n = 0; int64_t mh_pos0 = 0; int mbuf = -1;
  };
  std::vector<UnitCtx> units_;
  int unit_cur_ = -1;
  struct MemberBufs { DevBuf cand, mh; };  // multi-sequence: candidate block and MTP capture for the second and later sequences running the upper layers (20+) (the first uses Work's)
  std::vector<MemberBufs> mbufs_;
  void unit_select(int u);  // idempotent: put unit u's slot and globals into Work (-1 = original state)
  void* work_addr_[4] = {};  // Work's original addresses (h, acc, route_ids_d, pos_d) — checked on return after tiling
  int tile_cur_ = 0;       // sub-chunk currently swapped into Work (0 = Work itself)
  void tile_select(int s);  // idempotent: put sub-chunk s into Work (0 = original state)
  void tile_xchg(int s);    // swap Work <-> tiles_[s-1] (internal)
  void moe_decode_experts(const LayerWeights& L, int l, int M, ForwardStats* stats);            // decode: grouped kernels + CPU
  // (runtime.cpp header comment "O1"): moe_decode_experts (4 args) = launch (7-arg overload: classify, start CPU, DMA, GPU launch) + finish (CPU wait, accumulate) — same order.
  //   HIVE_DECODE_HOST_FAST (env_on) = trim host work on the critical path, HIVE_DECODE_UBATCH (env_on) = two-half batch layer pipeline (hive/decode_overlap.h),
  //   a [decode-host] line on HIVE_PROFILE sample steps (per-layer host/GPU phases). State lives in runtime.cpp's DecodeOverlap (shared_ptr — the fake runtime compiles with an incomplete type).
  struct MoePend;
  struct DecodeOverlap;
  std::shared_ptr<DecodeOverlap> dov_;
  bool host_fast_ = false, ubatch_ = false;
  // HIVE_DECODE_PREGATE=K (hive/pregate.h header comment, 0 = off): on early-routed decode layers, a pre-gate (top K) before attention -> CPU pool prefetch.
  int pregate_k_ = 0;
  void pregate_dev(const LayerWeights& L, int M);  // inside the front graph (right after hc_attn_pre, separate stream) — predictive router + publish mapping
  void pregate_host(int l, int M);                 // right after the front launch (engine thread) — wait for publish -> request prefetch of non-resident predicted experts
  // F2 HIVE_DECODE_PREDICT_EVAL (measurement only, outputs unchanged): at each decode layer l, layer l+1's router runs on layer l's FFN input (side
  //   stream, scratch buffers); when layer l+1's real routing reaches the host it is compared with the prediction (recall of the top k / top 2k, of the
  //   experts that were not resident, and the precision of the first 1/2/4 non-resident candidates). Cumulative counts go to stderr every 2000 compared layers.
  struct PredEval;
  std::shared_ptr<PredEval> pe_;
  void predict_eval(const LayerWeights& L, int l, int M);
  // HIVE_CACHE_PRIOR=λ (decode routing biased toward VRAM-resident experts — k::CachePriorArgs in hive/decode_attn3.h); 0/unset = off.
  //   decode_layer sets cp_layer_ and fills the layer's mask row (residency is fixed within a step: promotions commit at the step head)
  float cache_prior_ = 0.f; int cache_prior_j_ = 2; int cp_layer_ = -1;
  // HIVE_DECODE_DEFER=1 (expert deferral, early-route decode only): CPU misses ranked 3rd or lower in their row run on the pool after the layer
  //   finishes and are added to the hidden streams at the next layer, before that layer builds its own CPU jobs — h += post_f(layer) · y. HIVE_DECODE_SKIP_MISS=f: CPU misses
  //   ranked 3rd or lower whose weight is below f of the row's routed weight are left out. Both off by default.
  bool dec_defer_ = false; float dec_skip_ = 0.f;
  std::vector<ExpertStore::Job> djobs_;
  float* def_out_h_ = nullptr; float* def_out_d_ = nullptr; int32_t* def_rows_h_ = nullptr; int32_t* def_rows_d_ = nullptr; float* def_scratch_h_ = nullptr;
  int def_rows_n_ = 0, def_M_ = 0; bool def_pending_ = false;
  DevBuf post_def_;
  long def_layers_ = 0, def_rows_total_ = 0, skip_rows_total_ = 0;
  void deferred_flush();   // wait for a pending deferred batch and add it to the hidden streams (stream order: now)
  uint8_t* cache_mask_h_ = nullptr;   // pinned [n_layers][n_routed]
  DevBuf prior_range_;                // [n_layers] running average of the selection-score range
  int dec_row0_ = 0;       // first row of a half decode (UBATCH) (stats row number, observe row) — default 0
  bool ub_active_ = false; // inside the UBATCH layer loop (do not shrink stats row arrays)
  void moe_decode_experts(const LayerWeights& L, int l, int M, ForwardStats* stats, MoePend& p, bool start_cpu, bool pump, bool cw_ok);  // = launch (steps 1-3)
  void moe_decode_start_cpu(MoePend& p);
  void moe_decode_wait(MoePend& p, ForwardStats* stats);   // wait for the CPU batch (+ DMA adaptation sample)
  void moe_decode_accum(MoePend& p);                        // launch accumulation of CPU results (st_)
  void moe_decode_finish(MoePend& p, ForwardStats* stats);  // = wait + accum (step 4 of moe_decode_experts)
  int decode_host_prep(std::vector<Seq*>& seqs, int l, int M, ForwardStats* stats);  // engram_host + attention_decode_host → Tmax
  void decode_layers_ubatch(std::vector<Seq*>& seqs, int M, ForwardStats* stats);
  void ub_view(int h, int M, bool on);  // map half h's row range into Work (row-based pointer offset); on=false restores
  void dh_report(int M);                // [decode-host] line (HIVE_PROFILE sample steps)
  double dh_cw_ms_ = 0; long dh_cw_step_ = 0;  // [decode-host] promo wait: commit_wait_ms and step_ at the last line
  // HIVE_DECODE_SPLIT=balance (hive/decode_split.h): cost-model state for the decode miss DMA/CPU split (runtime.cpp DecodeSplit — created only when enabled; incomplete type)
  struct DecodeSplit;
  std::shared_ptr<DecodeSplit> dsplit_;
  void ds_consume();      // if the pending sample (GPU events) finished, fold it into the model and totals (non-blocking — cudaEventQuery)
  void ds_report(int M);  // [decode-split] line (HIVE_PROFILE sample steps — called by dh_report)
  // Batched decode misses (hive/batch_miss.h): state for HIVE_DECODE_CPU_FIRST, HIVE_DECODE_STAGE_HIT, HIVE_DECODE_PREFETCH
  //   (runtime.cpp BatchMiss — created only when any is enabled; incomplete type)
  struct BatchMiss;
  std::shared_ptr<BatchMiss> bmiss_;
  void bm_prefetch(int l);  // prefetch layer l's predicted misses (only when the link is idle)
  void bm_report(int M);    // [decode-miss] line (HIVE_PROFILE sample steps — after forward_batch's dh_report)
  // CUDA graphs: captured per (layer, M, T bucket) from the second use and replayed. Off with HIVE_NO_GRAPH=1 or in dump mode.
  int attn_splits_ = 4;  // decode attention split-K pieces (HIVE_ATTN_SPLITS)
  bool fuse_ = true;     // fused decode kernels (HIVE_FUSE=0 disables — for bit comparison against the unfused chain)
  // Adaptive CPU/DMA split of missed experts (baseline: of a 35 ms c1 token, CPU misses took 12.4 ms — with a DMA share of 0 the copy engine sat idle).
  //   Per layer DMA share = round(frac x misses) (misses >= 3, <= 8 staging) — experts with the most rows go to DMA first. Compare the previous layer's CPU wait (ms_cpu) with DMA group completion (events)
  //   and adjust frac by +-0.05 (more to DMA when the CPU takes longer). Decode (M <= 8) and short prefill (M > 8) keep separate state. HIVE_DMA_FRAC=<fixed> disables adaptation.
  //   Three kinds (decode, short prefill, streaming prefill); each sample is bounded by its own kind's limits, and on layers where no sample
  //   can arise the value relaxes toward the default (runtime.cpp dma_frac_consume/dma_frac_relax).
  //   HIVE_PREFILL_SHORT_ADAPT: a fourth kind kDmaStreamShort = streaming prefill layers with rows <= N (short chunks) — split from long-chunk layers (unused when off).
  enum { kDmaDecode = 0, kDmaShort = 1, kDmaStream = 2, kDmaStreamShort = 3 };
  static constexpr float kDmaFracDefault[4] = {0.25f, 0.5f, 0.5f, 0.5f};
  float dma_frac_[4] = {kDmaFracDefault[0], kDmaFracDefault[1], kDmaFracDefault[2], kDmaFracDefault[3]};
  bool prefill_pending_ = false;  // M7: set_prefill_pending
  int warm_left_ = 0, warm_step_ = 0, warm_steps_ = 0, warm_done_ = 0;  // deferred warm (warm_defer): experts still to promote · cap per step · steps used · promoted so far
  // Opt-in switches (all default off — read by the constructor option block; off = baseline path)
  bool prefill_pause_promote_ = false;  // HIVE_PREFILL_PAUSE_PROMOTE: while prefill_pending_, promote_after_step pauses promotion only (decay and miss clearing continue)
  bool early_stream_ = false;           // HIVE_EARLY_STREAM: issue the first staging ring's H2D of a prefill layer before preparing the CPU share + parallel activation LUT unpack
  bool ly_small_ = false;               // HIVE_LAYER_YIELD_SMALL: layer-boundary yields also inside HIVE_PREFILL_SMALL forwards (below the prefill threshold)
  bool engram_par_ = false;             // HIVE_ENGRAM_PAR: engram row lookups (memcpy) on several threads
  bool engram_digest_ = false;          // HIVE_ENGRAM_DIGEST: one hash line of lookup results per layer (RAM vs SSD bit comparison)
  double essd_log_ms_ = 0;              // time of the periodic HIVE_ENGRAM_SSD summary line
  bool idx_msub_actual_ = false;        // HIVE_IDX_MSUB_ACTUAL: indexer sub-chunk rows = actual T (within the same scratch capacity)
  int host_par_threads_ = 1;            // parallel threads for EARLY_STREAM/ENGRAM_PAR = CPU expert pool thread count (the pool is idle at that moment)
  std::unique_ptr<VitCache> vit_cache_; // HIVE_VIT_CACHE_MB: LRU of image encoder outputs (host RAM, hit only on exact byte match)
  int short_dma_cap_ = 8;  // short-prefill per-layer DMA cap (HIVE_SHORT_DMA_CAP, unset = 8)
  void dma_frac_consume();          // fold one completed sample into the frac of its kind
  void dma_frac_relax(int kind);    // layer where no comparison sample could arise: one step toward the default
  float dma_frac_fixed_ = -1.f;
  // HIVE_PREFILL_SPLIT=balance: streaming-prefill split cost model (hive/prefill_split.h) — cost EMA per layer type + GPU-side sample events.
  //   Events are created on the first layer that uses them when enabled (not at all when off). No new sample is taken while one is pending (same rule as dma_evt_pending_).
  static constexpr int kSplitDmaSamples = 4;  // number of copies (the first few) used to time pure per-layer copy
  double split_job_ms_[2] = {0, 0}, split_rec_ms_[2] = {0, 0}, split_row_ms_[2] = {0, 0};
  bool split_has_job_[2] = {false, false}, split_has_rec_[2] = {false, false}, split_has_row_[2] = {false, false};
  cudaEvent_t split_g0_ = nullptr, split_g1_ = nullptr, split_c0_ = nullptr, split_c1_ = nullptr, split_d_[kSplitDmaSamples][2] = {};
  bool split_pending_ = false, split_pred_ok_ = false;
  int split_type_ = 0, split_nd_ = 0, split_comp_rows_ = 0;
  double split_pred_gpu_ = 0;
  void split_consume(ForwardStats* stats);  // if the pending GPU sample finished, fold it into costs/stats (non-blocking — cudaEventQuery)
  cudaEvent_t dma_e0_ = nullptr, dma_e1_ = nullptr;
  bool dma_evt_pending_ = false;
  int dma_evt_kind_ = 0;
  double dma_evt_cpu_ms_ = 0;
  // GPU sampler candidate buffers (pinned host + device)
  float* cand_it_h_ = nullptr; float* cand_it_d_ = nullptr;
  int32_t* cand_idx_h_ = nullptr; float* cand_val_h_ = nullptr; float* cand_max_h_ = nullptr; float* cand_sum_h_ = nullptr;
  DevBuf cand_idx_, cand_val_, cand_max_, cand_sum_;
  void sampler_cands_dev(int M);  // build candidates, max and sum-exp of logits_b rows [0,M) and copy them to the host (async, same stream)
  bool graphs_ = true;
  bool capturing_ = false;
  bool dump_final_ = false;  // window that allows only the final dump at the end of forward_batch in graph mode
  struct GraphEntry { cudaGraphExec_t exec = nullptr; int uses = 0; };
  std::map<std::tuple<int, int, int>, GraphEntry> graphs_map_;
  void run_graph(int l, int M, int Tb, const std::function<void()>& body);
  void decode_front(int l, int M, int Tb, bool copy_acts);  // decode_layer graph A body (shared with the step graph)
  // HIVE_DECODE_STEP_GRAPH (env_on, constructor option block): one forward_batch step (all layers + head) as a single CUDA graph — instead of a host sync per layer,
  //   a GPU plan (hs_plan) + CPU pool dispatcher handshake (hive/decode_handshake.h, decode_dispatch.h). State is runtime.cpp's StepGraph (created on the first step when enabled —
  //   shared_ptr so the fake-runtime tests compile with only an incomplete declaration). forward_batch_step: true if this step ran on that path (else the per-layer loop).
  bool step_graph_ = false;
  struct StepGraph;
  std::shared_ptr<StepGraph> sg_;
  bool asleep_ = false;  // sleeping (sleep_release .. wake_restore)
  bool forward_batch_step(std::vector<Seq*>& seqs, int M, const std::function<void()>& head_body, ForwardStats* stats);
  // HIVE_DECODE_COPY_PRIO (env_on, constructor stream block): instead of issuing promotion H2D in one batch at step end, queue it in pieces (ExpertStore::set_promo_pacing)
  //   and issue a budgeted amount per decode layer **after** that layer's demand DMA (staging copies) (promo_pump — runtime.cpp header comment). side_ = highest, promo_ = lowest stream priority.
  bool copy_prio_ = false;
  cudaEvent_t promo_gate_ = nullptr;   // marker after this layer's demand copy on side_ -> promo_ waits on it
  void promo_pump(int l, int nL);
  // Demand-copy instrumentation (HIVE_PROFILE sample steps with HIVE_TRACE_CACHE only — measures the same values on both A/B sides regardless of the switch): per layer [before side_ copy, after, just before st_ wait]
  std::vector<cudaEvent_t> cw_ev_;
  std::vector<uint8_t> cw_used_;       // per layer bit0 = copy span (before/after), bit1 = st_ wait marker
  bool cw_on_ = false;
  uint64_t cw_promo_last_ = 0;
  std::string cw_line_;                // this step's summary (appended to the preport [cache] line)
  void cw_begin(bool on);
  void cw_collect();
  std::vector<Seq*>* batch_ = nullptr;  // non-null only during forward_batch
  std::vector<int> shared_topk_rows_;    // per-row topk from the last index-source layer in batched decode
  void prepare_decoder_tail(Seq& seq, int l, int M, int64_t start_pos, int tail);
  void moe(Seq& seq, const LayerWeights& L, int l, int M, ForwardStats* stats);
  void engram(Seq& seq, const LayerWeights& L, int M, const int32_t* ids_host);
  void merge_images(int M, const std::vector<ImageInput>& images);
  void indexer(Seq& seq, const LayerWeights& L, int l, int M, int64_t start_pos, int G, int64_t g0);
  void dump(const std::string& name, const void* dev, size_t bytes);
  void dump_host(const std::string& name, const void* host, size_t bytes);
  void inject_layer_input(int l, int M);  // with inject_dir, replace layer l's input with the golden one
  // ---- DSpark ----
  bool mtp_on_ = false;
  int ntgt_ = 0;                 // number of target layers (main_proj input = ntgt*dim)
  int mh_n_ = 0; int64_t mh_pos0_ = 0;  // rows captured into w.mh by this forward and their first position (single-sequence path)
  bool verify_ = false;          // inside forward_verify (per-row logits, state snapshot, draft ring update deferred)
  std::vector<float>* verify_logits_out_ = nullptr;
  int64_t v_pos0_ = -1; int v_M_ = 0; size_t v_hist_ = 0;  // for rollback
  std::vector<int> v_src2_;      // kv source layers with ratio>1 (layers whose compressor state is reverted)
  int target_index(int l) const;  // ordinal of l if it is a target layer, else -1
  void mtp_capture(int l, int M, int64_t start_pos);      // on entering a target layer, hc-mean of h into w.mh (last min(M, row cap) rows)
  void mtp_sync_seq(Seq& seq, int64_t pos0, int n, int row0 = 0);  // w.mh rows [row0, row0+n) (positions pos0..) -> main_x -> per-stage ring + seq.mtp_hidden (row0 default 0)
  // Verify decode path (HIVE_MTP_VERIFY2, HIVE_MTP_BATCH — runtime.cpp verify_decode)
  bool verify2_ = false, mtp_batch_ = false;  // switches (constructor option block)
  bool verify_rowind_ = false;                // HIVE_MTP_VERIFY_ROWIND (row-independent expert sum for verify rows — runtime.cpp moe_decode_experts)
  bool v2_ok_ = false;                        // usable in this runtime (fuse_, no dump/inject, tables allocated — constructor)
  int v2_rows_ = 0;                           // row cap for one verify
  bool vdec_ = false;                         // decode_layer is running verify rows (seen by decode_host_prep, decode_front, run_graph)
  long graph_captures_ = 0;
  long graph_eager_ = 0;                      // first uses of a graph key (eager) — graph_eager_runs()
  bool route_cap_ = false;
  std::vector<std::vector<int32_t>> route_log_;
  struct VPart { Seq* seq = nullptr; int64_t pos0 = 0; int M = 0, row0 = 0; size_t hist = 0; };
  std::vector<VPart> vparts_;                 // pending verify (until rollback_batch)
  std::vector<Seq*> vseqs_;                   // row -> sequence
  static constexpr int kVParts = 8;           // max parts (snapshot table entries)
  static constexpr int kVerifyGraphKey = 1 << 20;  // verify front graph key = rows + this value (never collides with plain decode M or UBATCH half keys)
  void verify_decode(std::vector<VerifyPart>& parts, std::vector<float>& logits_rows, ForwardStats* stats);
  int attention_verify_host(const LayerWeights& L, int l, int M);            // same table as attention_decode_host (row position = w.pos_h[m])
  void attention_verify_dev(const LayerWeights& L, int l, int M, int Tmax);  // fuse_ variant of attention_decode_dev + the three verify sites (verify_decode.h)
  void mtp_sync_batch(std::vector<Seq*>& seqs, int M);     // batched decode: row m = this step's token of seqs[m]
  void mtp_attention(Seq& seq, int s, int B, int64_t pos0, int filled);
  // HIVE_LAYER_YIELD ("T11" header comment in runtime.cpp)
  ly::State ly_;
  double ly_progress_ = 0;                       // layer_yield_progress (set right before each layer_yield_point call)
  double ly_prog(int l, int L, int nl, bool tail) const {  // progress after layer l (L = tail layer, nl = layers)
    if (!tail || L <= 0 || L >= nl) return (l + 1.0) / std::max(1, nl);
    return l < L ? 0.9 * (l + 1.0) / L : 0.9 + 0.1 * (l + 1.0 - L) / std::max(1, nl - L);
  }
  std::vector<std::pair<Seq*, int>> ly_owner_[2];  // per yield level  // (sequence, rows) of the outer prefill forward — after a yield, xtrace header and row owners are rewritten
  struct LyPark;                                 // runtime.cpp (pinned storage — incomplete type: shared_ptr so the fake runtime compiles)
  std::shared_ptr<LyPark> lypark_[2];            // per yield level (ly::Hooks::max_depth ≤ 2)
  void ly_begin(std::vector<std::pair<Seq*, int>> owners);  // outer prefill forward start: clock and owners (forwards inside a yield do not touch it)
  // Layer boundary (kind ly::kLayer — called only by the outer prefill forward) or an intra-layer point (HIVE_LAYER_YIELD_INTRA: other kinds —
  //   layer and last select the look-ahead segment key; returns at once unless ly_live_ and the intra conditions hold)
  void layer_yield_point(int kind = ly::kLayer, int layer = -1, bool last = false);
  bool ly_live_ = false;        // the forward running now is yield-eligible (forward/forward_multi ly_ok) — intra points deeper in the call tree (moe, indexer) read it
  bool ly_intra_ = false;       // HIVE_LAYER_YIELD_INTRA read on (constructor option block); usable only with ly_intra_ok()
  bool ly_intra_ok() const;     // switch on · elastic small/big layouts complete · CPU misses on (runtime.cpp)
  cudaEvent_t ly_ev_[2] = {nullptr, nullptr};  // kIndex: host pacing behind the GPU (created on first use)
  int ly_ev_n_ = 0;
};

}  // namespace hive
