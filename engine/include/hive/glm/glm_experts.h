// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM routed experts: every expert record (NVFP4, glm_moe.h layout) lives in pinned host RAM; a VRAM slot cache holds the hottest
//   ones. Decode: resident experts run on the GPU (moe_decode), missed ones on the CPU pool (expert_forward_nvfp4 split across
//   threads) at the same time. Prefill: missed experts are copied to a staging ring over PCIe and run on the GPU.
// Cache policy (the one measured for the DeepSeek path): per-key usage score with decay after each decode step; after a step the
//   highest-scoring non-resident experts that were used are promoted asynchronously (side stream, ≤ promote per step) into the
//   lowest-scoring slots; a promotion becomes resident when its copy event has completed (checked at the next step). After a
//   prefill the cache is refilled from the top scores, paced over the following decode steps (warm quota, as HIVE_WARM_DEFER).
#pragma once
#include <algorithm>
#include <atomic>
#include <iterator>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hive/devmem.h"
#include "hive/grow_buf.h"
#include "hive/glm/glm_model.h"
#include "hive/glm/glm_moe.h"

namespace hive::glm {

struct GlmCacheStats {
  uint64_t routed = 0, hit = 0, cpu = 0, streamed = 0, promoted = 0, evicted = 0, prefetched = 0, skipped = 0, deferred = 0;
  // decode timing (host clock): whole decode_layer, CPU expert compute (cpu_experts), phase 1 / phase 2 of it, layers with CPU work
  double ms_layer = 0, ms_cpu = 0, ms_p1 = 0, ms_p2 = 0, ms_wait_x = 0; uint64_t cpu_layers = 0;
  uint64_t cpu_jobs = 0, cpu_bytes = 0;  // decode: distinct experts computed by the CPU and the record bytes they read
  int workers = 0;                      // CPU worker threads (constant)
  double ms_busy1 = 0, ms_busy2 = 0;     // summed task time of all workers in phase 1 / 2 (utilization = busy / (phase wall × workers))
};

class GlmExperts {
 public:
  // cache_bytes: VRAM for slots (0 = fit later with alloc_cache). cpu_threads: CPU expert workers.
  GlmExperts(GlmModel& m, size_t cache_bytes, int cpu_threads, int staging_slots);
  ~GlmExperts();
  void load_all(int threads);                         // read every routed expert of layers 3..44 into pinned RAM
  int alloc_cache(size_t bytes);                      // allocate the slot area (returns slots)
  // Sleep level 1 (hived {"op":"sleep"}): wait for every copy and CPU batch, forget residency (scores stay), free the slot area and the
  //   staging ring; returns the bytes freed. wake_alloc re-creates the staging ring and the slot area — `slots` (≥ 0) or, with -1, as
  //   many as the free VRAM minus `reserve` holds (HIVE_CACHE_FIT) — and returns the slot count (-1 if not even the staging ring fits).
  //   The caller refills the slots with warm_from_keys (resident_keys_by_score taken before the sleep).
  size_t sleep_release();
  int wake_alloc(int slots, size_t reserve);
  bool asleep() const { return asleep_; }
  int slots_before_sleep() const { return slots_before_sleep_; }
  int staging_slots() const { return n_staging_; }
  // Elastic tail: lend the last n slots' VRAM to the caller (their experts are evicted; in-flight copies finish first) and take it back
  //   later — the layer-major prefill buffers live there only while a long prompt is processed, so decode keeps the whole cache.
  uint8_t* lend_tail_slots(int n);
  void return_tail_slots();
  int lent_slots() const { return lent_; }
  // KV growth (GlmEngine::kv_reserve): with VMM the slot area is mapped in steps (slot_buf_), so the tail can be given back to the driver
  //   and taken again without moving the area.
  //   shrink_cache evicts tail slots and unmaps the steps they used until at least `bytes` are freed (returns the bytes freed; 0 without VMM, while slots are
  //   lent or asleep). regrow_cache maps tail chunks back, up to the startup count, while the free VRAM stays above `reserve` (returns the
  //   slots added; they start empty — promotions refill them). ⚠️ Both need every reader of the slots idle: the caller synchronizes its
  //   compute stream first (the copy stream is synchronized here).
  size_t shrink_cache(size_t bytes);
  int regrow_cache(size_t reserve);
  int max_slots() const { return max_slots_; }
  int n_slots() const { return n_slots_; }
  int n_keys() const { return n_moe_ * E_; }
  const ExpertRecLayout& layout() const { return L_; }
  const uint8_t* host_rec(int key) const { return host_ + (size_t)key * L_.total; }
  int slot_of(int key) const { return slot_of_[key]; }
  const uint8_t* dev_rec(int slot) const { return slots_ + (size_t)slot * L_.total; }
  size_t host_bytes() const { return host_bytes_; }

  // Decode step for one MoE layer (moe index li): rows M, routing (ids [M,K], weights [M,K] on host), x bf16 [M,H] on device,
  //   out fp32 [M,H] on device (accumulated into). Uses stream st for GPU work; CPU misses run on the pool concurrently and are added.
  //   x_host (optional): the same rows already in pinned host memory (written on `st` before the caller's routing sync) — skips the D2H + sync
  void decode_layer(int li, int M, const int32_t* ids, const float* w, const bf16* x_dev, float* out_dev, cudaStream_t st, const bf16* x_host = nullptr,
                    bool allow_defer = false);
  // Expert deferral (HIVE_GLM_DEFER=1, decode): with allow_defer, the CPU misses ranked 3rd or lower in their row are not waited for —
  //   decode_layer starts them on a helper thread (results in deferred_y(), fp32 [rows, H], already weighted) and returns; the caller adds
  //   them to the hidden streams later (deferred_wait first). Must be waited for before the next decode_layer / prefill on this object.
  bool defer_enabled() const { return defer_on_; }
  bool deferred_pending() const { return dpending_; }
  void deferred_wait();
  const float* deferred_y() const { return host_yd_[dbuf_]; }
  int deferred_rows() const { return drows_; }
  // Prefill for one MoE layer: T rows, routing on host; GPU only (resident slots or streamed records).
  void prefill_layer(int li, int T, const int32_t* ids, const float* w, const bf16* x_dev, float* out_dev, cudaStream_t st);
  // After a decode step: score decay + asynchronous promotions (+ warm quota). Commits finished promotions first.
  void after_step(int tokens = 1);   // promotion budget = promote_per_step_ × tokens processed in the step (a verify step processes several)
  void warm_quota(int n) { warm_left_ = n; }
  // Prediction prefetch (HIVE_GLM_PREFETCH=N): copy up to N predicted, non-resident experts of MoE layer li (ids/w host [rows·K]) into the
  //   lowest-score slots on the side stream, after the work already queued on `compute` (a victim slot may still be read by it). They become
  //   resident when their copy has finished (commit_ready at the start of that layer's decode_layer); unfinished ones are plain misses.
  int prefetch(int li, const int32_t* ids, const float* w, int rows, int max_n, cudaStream_t compute);
  GlmCacheStats& stats() { return stats_; }
  int n_resident() const { int n = 0; for (int k : key_of_slot_) n += k >= 0; return n; }
  std::vector<int32_t> resident_keys_by_score() const;   // resident keys, highest score first (warm-start list)
  int warm_from_keys(const std::vector<int32_t>& keys);  // fill free slots from the list in order (synchronous); returns how many
  int n_moe_layers() const { return n_moe_; }
  // per-row decode counters (rows of the current step): reset by the caller before a step
  void reset_row_stats() { std::fill(std::begin(row_hit_), std::end(row_hit_), 0); std::fill(std::begin(row_cpu_), std::end(row_cpu_), 0); }
  int row_hit(int m) const { return row_hit_[m]; }
  int row_cpu(int m) const { return row_cpu_[m]; }
  int n_experts() const { return E_; }

 private:
  void cpu_experts(int li, const std::vector<std::pair<int, std::vector<std::pair<int, float>>>>& jobs, const float* x_host, float* y_host);
  void commit_ready();
  void promote_key(int key, const std::vector<int>& victims, size_t& next);
  std::vector<int> vic_;  // after_step: victim candidates of this step, ascending (score, slot)
  GlmModel& m_;
  int H_, I_, E_, K_, n_moe_;
  float limit_;
  ExpertRecLayout L_;
  uint8_t* host_ = nullptr; size_t host_bytes_ = 0; bool numa_interleaved_ = false;
  uint8_t* slots_ = nullptr; int n_slots_ = 0; int lent_ = 0;
  // VMM slot area (alloc_cache): one address range for max_slots_, physical memory mapped in steps of slot_step_ bytes (a multiple of the
  //   allocation granularity, independent of the record size — slots that straddle a step boundary stay usable only while both steps are mapped)
  GrowBuf slot_buf_;
  bool slots_vmm_ = false; int max_slots_ = 0; size_t slot_step_ = 0;
  size_t slot_bytes_mapped(int n) const { return (((size_t)n * L_.total + slot_step_ - 1) / slot_step_) * slot_step_; }
  bool asleep_ = false; int slots_before_sleep_ = 0;
  std::vector<int> slot_of_, key_of_slot_;
  std::vector<float> score_;
  std::vector<uint8_t> pending_;           // per slot: promotion in flight
  struct Promo { int slot, key; cudaEvent_t ev; };
  std::deque<Promo> promos_;
  std::vector<cudaEvent_t> ev_pool_;
  cudaStream_t side_ = nullptr;
  cudaEvent_t pf_ev_ = nullptr;
  std::vector<int> victims_; size_t vpos_ = 0;  // slots in ascending score order (rebuilt once per step in after_step) — prefetch victims
  int next_victim(int target_layer);
  int promote_per_step_ = 8, warm_left_ = 0, warm_per_step_ = 32;
  std::vector<int> step_used_;              // keys used in this step (for promotion)
  // staging ring for prefill streaming
  uint8_t* staging_ = nullptr; int n_staging_ = 0;
  std::vector<cudaEvent_t> stage_ev_, copy_ev_;
  int32_t* pf_idx_h_ = nullptr; float* pf_w_h_ = nullptr; int32_t* pf_idx_d_ = nullptr; float* pf_w_d_ = nullptr; size_t pf_tab_cap_ = 0;
  // prefill CPU share: compact rows (pinned), their inputs (device gather → pinned bf16 → fp32), results (pinned fp32)
  int32_t* pf_crow_h_ = nullptr; bf16* pf_cx_h_ = nullptr; float* pf_cy_h_ = nullptr; bf16* pf_cx_d_ = nullptr; int32_t* pf_crow_d_ = nullptr; size_t pf_c_cap_ = 0;
  std::vector<float> pf_cxf_;  // prefill row table (pinned, read by the kernels)
  // device scratch
  void* moe_ws_ = nullptr; size_t moe_ws_bytes_ = 0;
  MoePair* dpairs_ = nullptr; int dpairs_cap_ = 0;
  bf16* deq_gu_ = nullptr; bf16* deq_d_ = nullptr; bf16* rows_x_ = nullptr; bf16* rows_gu_ = nullptr; bf16* rows_y_ = nullptr; bf16* rows_o_ = nullptr;
  int32_t* rows_idx_ = nullptr; float* rows_w_ = nullptr; int rows_cap_ = 0;
  // CPU pool — NUMA split (DeepSeek path layout): every expert is also kept as two half records, one per NUMA node (node k: rows
  //   [k·I/2, (k+1)·I/2) of gate and up, rows [k·H/2, (k+1)·H/2) of down, with their scales). Workers are pinned to their node and only
  //   read that node's halves, so both nodes' memory channels work without cross-node traffic. HIVE_GLM_NUMA_SPLIT=0 = the single
  //   interleaved copy (shared queue). Workers spin HIVE_GLM_SPIN_US (default 3000) after a batch before sleeping on the condvar.
  struct HalfLayout { size_t w1, s1, w3, s3, w2, s2, g, total; };
  HalfLayout HL_{};
  bool split_ = false;
  uint8_t* half_[2] = {nullptr, nullptr};
  size_t half_bytes_ = 0;
  const uint8_t* half_rec(int node, int key) const { return half_[node] + (size_t)key * HL_.total; }
  int cpu_threads_;
  std::vector<std::thread> workers_;
  std::mutex mu_; std::condition_variable cv_;
  std::function<void(int, int)> task_;          // (node, index)
  std::atomic<uint64_t> next_[2];  // [generation 32 | task count 16 | next index 16] — claimed with CAS: a worker of an older batch can never take an index
  std::atomic<int> pending_tasks_{0};
  std::atomic<uint64_t> gen_{0}; bool stop_ = false;
  int spin_us_ = 3000;
  void run_tasks(int n0, int n1, const std::function<void(int, int)>& f);  // n_k tasks for node k (shared queue when !split_: n0 only)
  void build_halves(int threads);
  float* host_x_ = nullptr; float* host_y_ = nullptr;  // pinned [64, H]
  // expert deferral: helper thread running cpu_experts for the deferred jobs of one layer
  bool defer_on_ = false, dpending_ = false; int drows_ = 0, dli_ = -1;
  float* host_yd_[2] = {nullptr, nullptr};             // pinned [64, H] ×2 — alternate, so a batch never overwrites the one the caller's add may still read
  int dbuf_ = 0;
  std::vector<std::pair<int, std::vector<std::pair<int, float>>>> djobs_;
  std::thread dthr_; std::mutex dmu_; std::condition_variable dcv_; bool dgo_ = false, ddone_ = true, dstop_ = false;
  std::vector<float> scr_g_, scr_u_, scr_y_, scr_o_;  // cpu_experts scratch (reused across calls)
  std::vector<size_t> job_off_;
  GlmCacheStats stats_;
  int row_hit_[64] = {}, row_cpu_[64] = {};
  void* blas_ = nullptr;
};

}  // namespace hive::glm
