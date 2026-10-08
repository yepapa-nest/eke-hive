// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Expert store — all experts live in pinned RAM; hot experts are cached in VRAM slots.
// Row split: each expert is stored **half on each NUMA node, by rows** (half of the inter rows of w1/w3 and half of the dim rows
//   of w2 — each node holds one half-record). With whole experts per node (even/odd experts), a single decode CPU miss was bound
//   by **one node's DRAM bandwidth (measured 61 GB/s)**: one job took 0.332 ms = 17.9 MB / 54 GB/s (bench-cpu, including work
//   stealing — stealing threads read the same node's DRAM, so it did not help).
//   Splitting in halves lets one job use both nodes' DRAM (~110 GB/s). The GPU record format is unchanged (contiguous per
//   matrix) — H2D copies the two half-records as 12 pieces.
// Missed experts are computed in place by the CPU pool (each node's workers take their node's half-record row ranges and
//   steal from the other node when they run out).
// Engram tables are also loaded into pinned RAM here (one table per layer, alternating nodes).
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <deque>
#include <vector>

#include "hive/common.h"
#include "hive/expert_cpu.h"
#include "hive/model.h"

namespace hive {

// hive/bulk_load.h (HIVE_LOAD_PAR) — only declared here: that header pulls in nlohmann/json, and a CPU test
// (test_runtime_host_cpu.cpp) includes this header after '#define private public', so no new standard headers may be added.
struct BulkLoadOpts;
struct BulkLoadStats;

// Expert record layout (byte offsets). Format E2M1_B32; reference sizes dim 5120, inter 2304.
struct ExpertLayout {
  size_t w1, s1, w3, s3, w2, s2, total;
  static ExpertLayout make(int dim, int inter) {
    ExpertLayout L{};
    size_t wsz = (size_t)inter * dim / 2, ssz = (size_t)inter * dim / 32;
    L.w1 = 0; L.s1 = L.w1 + wsz; L.w3 = L.s1 + ssz; L.s3 = L.w3 + wsz; L.w2 = L.s3 + ssz; L.s2 = L.w2 + wsz;
    L.total = align_up(L.s2 + ssz, 4096);
    return L;
  }
};
// Half-record of node n (0/1): [w1 rows n·I/2..(n+1)·I/2][s1 same rows][w3][s3][w2 rows n·dim/2..][s2]. Byte-identical to the
// corresponding half of each matrix in the GPU record (ExpertLayout).
struct HalfLayout {
  size_t w1, s1, w3, s3, w2, s2, total;
  size_t w13_rows, w2_rows;  // rows held by one half-record (I/2, dim/2)
  static HalfLayout make(int dim, int inter) {
    HalfLayout L{};
    HIVE_CHECK(inter % 64 == 0 && dim % 64 == 0, "half layout needs inter/2, dim/2 multiples of 32");
    L.w13_rows = (size_t)inter / 2; L.w2_rows = (size_t)dim / 2;
    const size_t w13 = L.w13_rows * dim / 2, s13 = L.w13_rows * dim / 32, w2 = L.w2_rows * inter / 2, s2 = L.w2_rows * inter / 32;
    L.w1 = 0; L.s1 = L.w1 + w13; L.w3 = L.s1 + s13; L.s3 = L.w3 + w13; L.w2 = L.s3 + s13; L.s2 = L.w2 + w2;
    L.total = align_up(L.s2 + s2, 4096);
    return L;
  }
};

struct PinnedArena {
  uint8_t* base = nullptr;
  size_t bytes = 0;
  int node = -1;
  void alloc(size_t n, int numa_node, bool huge = true);
  void pin(size_t unit = 0);  // cudaHostRegister (in chunks — if unit is given, chunks are multiples of unit: a record spanning two registered ranges makes DMA fail)
  size_t chunk_ = 0;
  PinnedArena() = default;
  // Non-copyable, move-only: the destructor frees base, so a shallow copy would kill the original (a real bug: resizing the
  // engram_ vector copied and freed table 0, and after loading the second engram table layer 1 output blew up to 1e35. It was
  // invisible in ≤13-layer checks that loaded only one table).
  PinnedArena& operator=(const PinnedArena&) = delete;
  PinnedArena(const PinnedArena&) = delete;
  PinnedArena(PinnedArena&& o) noexcept : base(o.base), bytes(o.bytes), node(o.node), chunk_(o.chunk_) { o.base = nullptr; o.bytes = 0; }
  PinnedArena& operator=(PinnedArena&& o) noexcept {
    if (this != &o) { this->~PinnedArena(); base = o.base; bytes = o.bytes; node = o.node; chunk_ = o.chunk_; o.base = nullptr; o.bytes = 0; }
    return *this;
  }
  ~PinnedArena();
};

// Cold load HIVE_LOAD_PREFAULT (unset, "" or "0" = off = default): reserve the host memory the expert arenas and engram tables
//   will use **before** Model is constructed, and let several threads do the first touch (page allocation, zero fill, THP
//   compaction) overlapped with the dense load. PinnedArena::alloc takes a region with the same (aligned size, node, huge).
//   Measured basis (engine/tests/bench_load_par.cpp, while a running hived held ~750 GB of RAM): reading with O_DIRECT straight
//   into a fresh node-bound MADV_HUGEPAGE arena gave 1.2–2.3 GB/s (4–16 threads) — the first touch alone, 16 threads, was
//   2.32 GB/s; without MADV_HUGEPAGE 17.8 GB/s · reading into a pre-touched arena 7.29 GB/s (= the disk limit). So the bottleneck
//   can be direct THP compaction rather than the read (defrag=madvise). Memory state at startup may differ (the page-cache
//   layers of the non-prefault startup, 4.2–5.2 GB/s, are the first-touch lower bound there).
//   Touching uses a value-preserving atomic (fetch_or 0), so a region already taken and being written by a loader thread is not
//   wiped (it is skipped without waiting).
//   If the prediction is wrong (take finds no matching region), all remaining regions are released and allocation proceeds as
//   usual — absorbed.
struct HostPrefault {
  // Predict (requested bytes, node) of the two ExpertStore arenas + engram tables (values, scales) from the checkpoint
  // (config.json · index · shard headers). Failure = empty list.
  static std::vector<std::pair<size_t, int>> predict(const std::string& dir, int max_layer, bool no_mtp);
  static void start(const std::vector<std::pair<size_t, int>>& regions, int threads);
  static uint8_t* take(size_t aligned_bytes, int node, bool huge);  // called by PinnedArena::alloc (nullptr if there is no pool)
  // Stop touching, free regions not taken, log one line (taken regions are freed by their PinnedArena). Returns (bytes taken, bytes freed) — for tests
  static std::pair<size_t, size_t> finish();
};

struct EngramTable {
  int layer = -1;
  int64_t rows = 0;
  int hd = 256;
  PinnedArena vals;    // [rows, hd] e4m3 (empty with HIVE_ENGRAM_SSD — rows are read from the file)
  PinnedArena scales;  // [rows, hd/32] e8m0 (empty in SSD mode 2)
  bool ssd = false;    // HIVE_ENGRAM_SSD: this table is read by EngramSsd (ExpertStore::engram_ssd())
};

// HIVE_ENGRAM_SSD (hive/engram_ssd.h) — only declared here (no new standard headers in this header — see the BulkLoadOpts comment above)
class EngramSsd;

class ExpertStore {
 public:
  // n_mtp_layers: number of DSpark draft layers — appended as layers n_layers.. (cfg.dspark_experts experts each, same record
  //   format). The slot cache and CPU pool are shared.
  // defer_cache (hived HIVE_CACHE_FIT): do not allocate the slot area here (n_slots 0) — alloc_cache allocates it from the
  //   remaining VRAM at the end of startup (after the session pool). false = allocate here (vram_cache_bytes / record size).
  ExpertStore(const Config& cfg, int n_layers, size_t vram_cache_bytes, int cpu_threads_per_node, int n_mtp_layers = 0, bool defer_cache = false);
  ~ExpertStore();
  // HIVE_CACHE_FIT: slot count = (free VRAM − reserve (operating headroom: vision encoder, image patches, graph growth) − overhead (allocation granularity slack)) / record size rec.
  static int fit_slots(size_t free_bytes, size_t reserve_bytes, size_t rec, size_t overhead = 64ull << 20) {
    if (rec == 0 || free_bytes <= reserve_bytes + overhead) return 0;
    const size_t n = (free_bytes - reserve_bytes - overhead) / rec;
    return (int)std::min<size_t>(n, (size_t)1 << 30);
  }
  // Allocate the deferred slot area once with n slots (one contiguous block — keeps the dev_rec(0) + slot·total assumption of
  //   the G1 step graph). If cudaMalloc fails, retry shrinking by 1/64 each time (absorbed).
  //   Returns the number of slots allocated. A non-deferred store or a second call does nothing and returns the current count.
  int alloc_cache(int n);
  bool cache_deferred() const { return defer_cache_ && !cache_allocated_; }
  // HIVE_CACHE_ELASTIC (runtime.cpp reads the switch and calls these — if they are never called, everything below is 0/no-op
  //   and behaviour is the default):
  //   Appends n_el = ⌈bytes / record⌉ elastic slots to the tail of the slot area (one contiguous block) = the space for the
  //   prefill-only work buffers (runtime Work's large panels, tiles, host-tile staging). While no prefill runs they are lent out
  //   as cache slots (lend) and reclaimed right before a prefill (reclaim — only a table update evicting their residents, no copy).
  //   Slot numbers and addresses stay dev_rec(s) = dev_rec(0) + s·total (keeps the G1 step-graph slots_base · D1 table · F2/F3
  //   kernel assumptions) — while reclaimed, n_slots() drops to the base slot count (n_base), so every choice (place · promote ·
  //   promote_keys victims and limits) only sees [0, n_base).
  //   elastic_reserve: for a deferred store (HIVE_CACHE_FIT) it only records the request and alloc_cache allocates one block.
  //   For an already-allocated store it reallocates the slot area as (n_slots + n_el) only when nothing is resident or being
  //   promoted (during startup). Returns the number of elastic slots. If that is impossible (0 — absorbed), the store allocates a
  //   separate range of the same size (at startup — so the VRAM peak stays at startup as before) and returns it from
  //   elastic_base() (never lent = same VRAM as without elastic slots). Same when alloc_cache of a deferred store gives up the
  //   elastic slots.
  int elastic_reserve(size_t bytes);
  int elastic_slots() const { return n_elastic_; }
  int base_slots() const { return n_elastic_ ? n_base_ : n_slots_; }
  bool elastic_lent() const { return n_elastic_ > 0 && n_slots_ > n_base_; }
  uint8_t* elastic_base() const {  // start of the elastic range (n_el·total bytes) · the separately allocated range if elastic was given up · nullptr if not requested or before a deferred store allocates
    return n_elastic_ ? slots_.as<uint8_t>() + (size_t)n_base_ * lay_.total : elastic_own_.as<uint8_t>();
  }
  // Reclaim: wait for promotions targeting elastic slots (after issuing all their pieces) up to their batch event and drop them
  //   (pending → 'D'), then evict residents ('E'). Returns the number of residents evicted.
  //   ⚠️Precondition: the caller has synchronized the streams that issued kernels reading elastic slots and place() copies
  //   (runtime calls this after synchronizing st_ and side_).
  int elastic_reclaim();
  // Lend: turn the elastic slots back into empty slots (n_slots() = all). ⚠️Precondition: the caller has synchronized the streams that used that range as a work buffer.
  void elastic_lend();
  struct ElasticStats { uint64_t reclaims = 0, lends = 0, evicted = 0, dropped = 0; double wait_ms = 0; };
  const ElasticStats& elastic_stats() const { return el_stats_; }
  // Z1 sleep/wake (hived {"op":"sleep"/"wake"} → called by runtime sleep_release/wake_restore — if never called, all of this is
  //   a no-op = default behaviour).
  //   sleep_release: commit all promotions (commit_all), clear the slot table and free the slot area (base + elastic = one block),
  //     the staging ring, and the separate range allocated when elastic was given up. n_slots() becomes 0 (place/promote are not
  //     called while asleep — hived does not run forward). Use counts, scores and D4 policy state (host) are kept.
  //     Returns the freed device bytes (accounted value). 0 if already asleep. ⚠️Precondition: no device work reads the slots or
  //     staging (runtime calls this after a device synchronize).
  //   wake_alloc: reallocate with the same size (physical slot count before sleeping) — the address may change (slot numbers and
  //     the contiguous dev_rec(0) + slot·total assumption still hold).
  //     With reserve_bytes > 0 (HIVE_CACHE_FIT) it does not exceed the count fitting VRAM free − reserve at that moment (same
  //     formula as alloc_cache at startup) · cudaMalloc failures are absorbed by shrinking 1/64 at a time (as alloc_cache).
  //     The elastic slots (prefill work-buffer space), the separate range and staging are mandatory, though — if even those cannot
  //     be allocated, nothing is allocated and -1 is returned (still asleep — the caller wakes again). The lent/reclaimed state is
  //     kept from sleep time. Returns the physical slot count (current count if already awake).
  //   restore_keys: load keys (residents in score order — resident_keys_by_score at sleep time) into empty slots, front to back, up
  //     to n_slots() (promote_keys batches + commit_all, synchronous). Scores and use counts are untouched (unlike seed_scores in
  //     warm_from_keys — the cache policy state from before sleeping stays as is). progress(loaded, target) is called per batch.
  size_t sleep_release();
  int wake_alloc(size_t reserve_bytes);
  int restore_keys(const std::vector<int32_t>& keys, cudaStream_t side, const std::function<void(int, int)>& progress = {});
  bool asleep() const { return asleep_; }
  int sleep_phys_slots() const { return sleep_phys_; }  // physical slot count before sleeping (for comparison while asleep / after waking)
  size_t device_bytes() const { return slots_.n + staging_.n + elastic_own_.n; }  // device bytes held by the store (slots + staging + separate elastic range)
  // Z1 host memory report (unchanged while asleep): expert arenas (two nodes · cudaHostRegister pinned) · engram tables (host RAM — read only by the CPU)
  size_t experts_host_bytes() const { return arena_[0].bytes + arena_[1].bytes; }
  size_t engram_host_bytes() const {
    size_t n = engram_ssd_cache_host_bytes();  // HIVE_ENGRAM_SSD row cache (off = 0)
    for (const auto& t : engram_) n += t.vals.bytes + t.scales.bytes;
    return n;
  }
  // HIVE_ENGRAM_SSD (unset, "" or "0" = off = nullptr — tables in RAM): when on, alloc/load_engram and load_bulk do not load the
  //   value tables (and, in mode 2, the scales) into RAM but register (file, offset) with EngramSsd. When this is set, runtime
  //   engram_host calls EngramSsd::gather instead of engram_gather (same output bytes).
  EngramSsd* engram_ssd() const { return essd_; }
  int engram_ssd_mode_now() const { return essd_mode_; }
  size_t engram_ssd_cache_host_bytes() const;
  // Call once after all tables are registered: allocates the row cache + startup log `[engram-ssd] mode … RAM saved …` (off or second call = no-op). Lookups are correct without it (reads without a cache — just slower).
  void engram_ssd_finish(Checkpoint& ck);
  const ExpertLayout& layout() const { return lay_; }
  int n_layers() const { return n_layers_; }
  int E() const { return E_; }                       // key stride (backbone expert count)
  int E_of(int l) const { return l < n_backbone_ ? E_ : cfg_.dspark_experts; }  // actual expert count of a layer

  // ---- loading ----
  void load_layer_experts(Checkpoint& ck, int l, int copy_threads = 16);
  bool layer_loaded(int l) const { return loaded_[l]; }
  void pin_all();  // cudaHostRegister the expert arenas (once, after loading)
  void load_engram(Checkpoint& ck, int layer, int hash_index);
  // Cold load (HIVE_LOAD_PAR — measurements in the hive/bulk_load.h header comment): read expert layers [0, n_layers_total) and
  //   the engram tables (layer, hash_index) in one parallel O_DIRECT pass into the same places as the default path (half-records ·
  //   engram arenas). pin = true overlaps registering the expert arenas (pin_all) with the reads.
  //   Returns true if the bulk load succeeded. On failure it reloads from scratch via the default path (load_layer_experts ·
  //   load_engram) inside this call (absorbed — the result is the same either way).
  bool load_bulk(Checkpoint& ck, int n_layers_total, const std::vector<std::pair<int, int>>& engram_tables, const BulkLoadOpts& opt, bool pin,
                 BulkLoadStats* stats = nullptr);
  void alloc_engram(Checkpoint& ck, int layer, int hash_index);  // the table-allocation part of load_engram (kept if already allocated)
  // HIVE_LOAD_CHECKSUM (verification log): 64-bit hashes of the whole host copy (two expert arenas · values and scales of each
  //   engram table) — fixed order (64 MiB block hashes folded with their index), so independent of the thread count. Start once
  //   with the default load and once with the new load and compare (bit-identity check). Order = node0, node1, (vals, scales) × tables.
  std::vector<uint64_t> host_checksums(int threads) const;
  const EngramTable* engram(int hash_index) const { return hash_index < (int)engram_.size() ? &engram_[hash_index] : nullptr; }

  // ---- host records (half-record × 2 nodes) ----
  const HalfLayout& half_layout() const { return hlay_; }
  const uint8_t* host_half(int l, int e, int node) const;
  // H2D one expert into a GPU record (ExpertLayout format) dst — 12 pieces from the two nodes' half-records (6 matrices × 2 halves)
  void copy_rec_async(uint8_t* dst_dev, int l, int e, cudaStream_t st) const;
  void copy_to_staging(int slot, int l, int e, cudaStream_t st);
  // HIVE_LAYER_YIELD_INTRA staging hold (runtime.cpp "T11b"): set while decode steps run inside an intra-layer yield and the paused prefill
  //   still owns pre-copied staging records (pf_). Decode writers check staging_held() and send their misses to the CPU; copy_to_staging
  //   itself aborts while held (data-corruption guard — comment in expert_store.cpp).
  void set_staging_hold(bool on) { staging_hold_.store(on, std::memory_order_relaxed); }
  bool staging_held() const { return staging_hold_.load(std::memory_order_relaxed); }
  // victim_ready: (HIVE_CACHE_REUSE_STAGE only) event marking that work reading the victim slot's old record is done — the dedicated D2D stream waits on it first (nullptr if none)
  void copy_for_promotion(uint8_t* dst_dev, int l, int e, cudaStream_t st, cudaEvent_t victim_ready = nullptr);
  bool reuse_staging() const { return reuse_staging_; }

  // ---- VRAM cache ----
  int n_slots() const { return n_slots_; }
  int slot_of(int l, int e) const { return slot_of_[(size_t)l * E_ + e]; }
  // HIVE_DECODE_STEP_GRAPH: the whole slot table (key l·E_ + e → slot, -1 = not resident) — the step head syncs a device copy
  //   (H2D only when it changed). Only the engine thread modifies it (place · promote · promote_keys · commit_*), always between
  //   steps, so the copy taken at the step head is the value for the whole step.
  const int32_t* slot_table() const { return slot_of_.data(); }
  size_t slot_table_size() const { return slot_of_.size(); }
  // Usage distribution: sort all (layer, expert) by use count descending; fraction of uses covered by the top n (cache slot count → upper bound on the theoretical hit rate)
  std::vector<std::pair<int, double>> coverage(const std::vector<int>& ns) const;
  uint64_t total_uses() const { uint64_t s = 0; for (auto v : use_count_) s += v; return s; }
  // Warm start: resident keys in descending score order · seed scores for a list (after restart, promote loads them in this order). Key = l·E + e.
  std::vector<int32_t> resident_keys_by_score() const;
  void seed_scores(const std::vector<int32_t>& keys, float top, float step);
  int n_resident() const { int n = 0; for (int s = 0; s < n_slots_; ++s) n += expert_of_slot_[s] >= 0; return n; }
  int n_pending() const { int n = 0; for (int s = 0; s < n_slots_; ++s) n += pending_[s] >= 0; return n; }
  struct CacheStats {
    int resident = 0, unique = 0, pending = 0, duplicates = 0, invalid_mappings = 0;
    uint64_t promotions = 0, commits = 0, evictions = 0, duplicate_skips = 0, h2d_records = 0, d2d_records = 0;
  };
  CacheStats cache_stats() const;  // Engine thread only; publish a copy to the API thread.
  // Mark a resident slot as used in this step (LRU update) — so it is not picked as a place() victim in the same step
  void touch(int slot) { slot_last_use_[slot] = ++use_total_; }
  // B6: slots used together in one batch (e.g. one prefill layer) get the same timestamp — calling touch() in e order would
  //   spread LRU times by expert number and bias promote_keys/place victim order toward numbering. Take one lru_stamp() per batch
  //   and mark with touch_at(slot, that value).
  uint64_t lru_stamp() { return ++use_total_; }
  void touch_at(int slot, uint64_t stamp) { if (slot_last_use_[slot] < stamp) slot_last_use_[slot] = stamp; }
  const uint8_t* dev_rec(int slot) const { return slots_.as<uint8_t>() + (size_t)slot * lay_.total; }
  // Load an expert into a slot (H2D). Returns its slot if already present. With no free slot, evicts a victim (least used).
  int place(int l, int e, cudaStream_t st);
  // Record an event that lets another stream wait for the copy into the slot after place
  void record_after_place(cudaStream_t st, cudaEvent_t evt) { CUDA_CHECK(cudaEventRecord(evt, st)); }
  // Statistics: routing observations (for the adaptive cache and distribution measurements). score is an exponentially decayed usage score.
  // D4 row: the row of this routing (row announced via set_row_owners → sequence, -1 = unknown). Only used with HIVE_CACHE_POLICY=seq (ignored when unset — default behaviour).
  void observe(int l, int e, float weight = 1.f, int row = -1) {
    size_t k = (size_t)l * E_ + e; ++use_count_[k]; score_[k] += weight; ++use_total_; trace_event('U',(int)k,slot_of_[k]);
    if (policy_) observe_policy(k, weight, row);
  }
  // D4: row → sequence uid for this forward (n = 1 means every row belongs to that sequence). prompt = prompt chunk (xtrace kind 1). No-op when the policy is off.
  void set_row_owners(const uint32_t* uids, int n, bool prompt) { if (policy_) set_row_owners_policy(uids, n, prompt); }
  void trace_event(char event,int key,int slot) const;
  const std::vector<uint64_t>& use_counts() const { return use_count_; }
  void decay_scores(float f) { for (auto& v : score_) v *= f; if (policy_) decay_policy(); }
  int cache_policy() const { return policy_; }  // D4: 0 = default (priority = score_) · 1 = seq (D4 policy block in expert_store.cpp)
  // Adaptive promotion: load up to max_n non-resident experts with the highest scores on the side stream (freeing the lowest-score slots).
  // Until the copy finishes the slot is 'pending' and not counted as resident. Returns the number issued.
  // Each promotion batch records its own event and is queued (re-recording a single event every token always looked like "latest
  //  copy in flight", so commit never happened, pending slots piled up and the cache starved — measured resident 102→43, hit 0).
  //  Nothing is issued while pending ≥ 2·max_n.
  // victim_ready: see copy_for_promotion (used only by the dedicated D2D stream of HIVE_CACHE_REUSE_STAGE — ignored when off)
  int promote(int max_n, float min_score, cudaStream_t side, cudaEvent_t victim_ready = nullptr);
  // Miss promotion: asynchronously copy up to max_n of keys (layer·expert keys, in priority order) that are neither resident nor
  //   being promoted into the **least recently used slots**.
  //   Offline replay (tools/cache_policy_sim.py rec16, one-hour natural-language trace): decode hit rate 74.5 → 82.0%. Returns the number of copies started.
  int promote_keys(const std::vector<int>& keys, int max_n, cudaStream_t side, int backlog_n = 0, bool score_victims = false,
                   cudaEvent_t victim_ready = nullptr);
  float score_of(int key) const { return score_[(size_t)key]; }
  // D4 priority used for promotion, victim and resident-list ordering: default = score_ · policy = prio_
  float prio_of(int key) const { return policy_ ? prio_[(size_t)key] : score_[(size_t)key]; }
  // Commit waiting promotions as resident once done_evt has completed
  void commit_pending();  // commit batches whose copy has finished (from the queue front, in order)
  void commit_all();      // wait for every batch's copy and commit (synchronous)
  double commit_wait_ms() const { return commit_wait_ms_; }  // total host time commit_pending waited for paced promotion copies
  // E4 HIVE_DECODE_COPY_PRIO (promotion copy pacing — see the promo_pump header comment in runtime.cpp): when on, promote/
  //   promote_keys decide **what goes into which slot** exactly as before (pending marks, victim release, statistics and the
  //   REUSE_STAGE D2D all at decision time), but only queue the H2D as pieces (the 12 pieces of copy_rec_async — same bytes, same
  //   order). The caller drains them gradually via issue_promotions(byte budget); after a batch's last piece is issued its batch
  //   event is recorded (= "issued"). commit_pending only looks at issued batches, and paced batches are committed by waiting on
  //   that event rather than querying it (fixes the residency point to call order — deterministic).
  //   The head of promote/promote_keys and commit_all first issue all remaining pieces (flush_promotions) — no decision or
  //   synchronous wait happens with pieces still pending.
  void set_promo_pacing(bool on) { pace_ = on; }
  bool promo_pacing() const { return pace_; }
  size_t promo_backlog_bytes() const { return pace_bytes_; }                  // bytes of pieces not yet issued
  bool promo_backlog() const { return pace_unissued_ > 0; }                    // there is a batch not yet fully issued (including batches with 0 pieces)
  size_t issue_promotions(size_t budget);  // issue pieces in queue order until the issued bytes reach budget (at least one piece if there is a backlog). Returns bytes issued
  void flush_promotions() { if (pace_unissued_ > 0) issue_promotions(SIZE_MAX); }
  uint64_t promo_h2d_bytes() const { return promo_h2d_bytes_; }  // cumulative bytes actually issued as promotion H2D (pacing off = at decision time · on = when pieces are issued)
  // Staging slots (for prefill streaming): a ring of n, separate from the cache
  int staging_slots() const { return n_staging_; }
  uint8_t* staging_rec(int i) const { return staging_.as<uint8_t>() + (size_t)i * lay_.total; }

  // ---- CPU pool ----
  // Job = one (token, expert). The pool splits it into row ranges (phase 1: inter rows of w1/w3, phase 2: dim rows of w2) and all
  // threads of a node share the computation (so all cores are used even when decode misses only a few experts).
  static constexpr int kMaxRows = 8;
  struct Job {  // one expert × R activations (rows of a batch calling the same expert form one job)
    int layer, e, R;
    const float* a_f[kMaxRows];   // [dim] activation values (fp8 values as fp32, scales not applied)
    const float* a_s[kMaxRows];   // [dim/32] block scales, fp32
    float route_w[kMaxRows];
    float* out[kMaxRows];         // [dim] fp32 (host)
    float* scratch;               // R × job_scratch_floats(inter)
  };
  static size_t job_scratch_floats(int inter) { return (size_t)3 * inter + inter / 32 + 64; }
  // Synchronous: blocks until all jobs are done
  void run_jobs(std::vector<Job>& jobs);
  // Asynchronous: start wakes the pool (the caller launches GPU work meanwhile) and wait waits for completion. One batch at a time (the jobs vector must stay alive until wait).
  // P1: owned = request an owned batch (decode layer — only effective when the store is in owned mode and the batch is not split; other calls such as prefill use the default claiming)
  void start_jobs(std::vector<Job>& jobs, bool owned = false);
  void wait_jobs();
  double jobs_done_ms() const;  // CPU completion time of the previous batch (now_ms clock) — valid after wait_jobs
  int cpu_threads() const { return 2 * threads_per_node_; }  // worker count of the two node pools (--cpu-threads × 2)
  // P1 HIVE_DECODE_PREGATE (see the hive/pregate.h header comment): owned mode (fixed worker per item · one worker = one physical core) + prefetch.
  //   prefetch_experts = each worker pre-reads only the rows it will compute from the half-records of layer l's experts
  //   (asynchronous — returns immediately). The next start_jobs or next request cancels it.
  //   No-op outside owned mode. Called only by the engine thread.
  bool owned_mode() const { return owned_; }
  void prefetch_experts(int l, const int* e, int n);
  struct PfStats { long req = 0, experts = 0, kib = 0, aborted = 0, finished = 0; };
  PfStats pf_stats() const;

 private:
  FILE* event_trace_ = nullptr;
  const Config& cfg_;
  ExpertLayout lay_;
  HalfLayout hlay_;
  int n_layers_, E_, n_backbone_;
  std::vector<size_t> rec_off_;  // [n_layers+1] index of a layer's first half-record within the node arena (E_of per layer — every expert is split across both nodes)
  PinnedArena arena_[2];
  std::vector<std::atomic<int>> p1_left_;  // remaining phase-1 chunks per job (both nodes) — phase 2 starts only when this is 0
  // B2 HIVE_CPU_SPLIT13 (off = empty and never read): when phase 1 is split into w1 items and w3 items, count of finished halves per
  //   (job, node, 32-row block) — the item that finishes second does that block's swiglu and fp8 quantization (expert_store.cpp
  //   run_item). split13_/p2_prefetch_ = switches read by the constructor.
  std::vector<std::atomic<int>> p13_done_;
  bool split13_ = false, p2_prefetch_ = false;
  bool copy2d_ = false;  // B3 HIVE_STAGE_COPY2D (copy_rec_async — 8 calls including 2D copies · same bytes)
  bool fine_ = false;  // B3 HIVE_CPU_FINE (start_jobs parts_of — multi-row jobs: phase-1 blocks in 4 pieces · phase-2 P2/2 rows)
  bool jobs_active_ = false;
  std::vector<char> loaded_;
  std::deque<EngramTable> engram_;  // stable element addresses (no relocation)
  int essd_mode_ = 0;               // HIVE_ENGRAM_SSD (read in the constructor · 0 = off)
  EngramSsd* essd_ = nullptr;       // owned (deleted first in the destructor: workers only read table files, not arenas, but the order is fixed)
  // cache
  DevBuf slots_, staging_;
  int n_slots_ = 0, n_staging_ = 8;
  bool defer_cache_ = false, cache_allocated_ = false;  // R5 HIVE_CACHE_FIT (constructor defer_cache · alloc_cache)
  // Q1 HIVE_CACHE_ELASTIC: elastic slots = physical slots [n_base_, n_base_ + n_elastic_) · while reclaimed n_slots_ = n_base_
  //   (while lent = all) · elastic_req_ = request recorded for a deferred store (used by alloc_cache). When off all three are 0 and n_slots_ is unchanged.
  int n_base_ = 0, n_elastic_ = 0, elastic_req_ = 0;
  size_t elastic_bytes_ = 0;  // requested bytes (size of the separate range when elastic is given up)
  DevBuf elastic_own_;        // range used when elastic is given up (runtime work buffer — never lent)
  ElasticStats el_stats_;
  // Z1 sleep: asleep_ · physical slot count before sleeping · whether it was lent · size of the separate elastic range (reallocated on wake)
  bool asleep_ = false, sleep_lent_ = false;
  int sleep_phys_ = 0;
  size_t sleep_own_bytes_ = 0;
  void elastic_size_tables(int n_phys);  // resize the slot-count-dependent tables to n_phys (new entries = empty slots)
  std::vector<int32_t> slot_of_;       // [L*E] → slot or -1
  std::vector<int32_t> expert_of_slot_; // [slots] → l*E+e or -1
  std::vector<uint64_t> use_count_;
  std::vector<float> score_;
  // D4 cache policy (HIVE_CACHE_POLICY) — with policy_ 0 (unset, "" or "0") everything below is empty and never read.
  //   seq: prio_[k] = sum over active sequences of the "expected uses in the next step" = Σ_o (c_ok + α·p_k) / (n_o + α) — the
  //   Bayesian posterior mean mixing sequence o's use count c in this request (decay od) and token count n with the global
  //   per-token usage rate p_k (decay gd) as a prior (strength α tokens). When a request ends (not seen for the idle steps) its share is removed.
  int policy_ = 0;
  std::vector<float> prio_;           // [L*E]
  struct Owner {
    uint32_t uid = 0, last = 0;       // last = policy step in which it was last seen
    bool live = false, prompt = false;
    float n = 0.f, scale = 1.f;       // n = token count of this request (decayed) · true c = raw · scale (decay as one scalar)
    std::vector<float> raw;           // [L*E] (allocated on first use)
    std::vector<int32_t> touched;     // keys with raw > 0 (for sparse summation and reset)
  };
  std::vector<Owner> owners_;         // 16 entries (concurrent sequences ≤ max_batch 8 + ones being prefilled)
  std::vector<int> row_owner_;        // set_row_owners: row → owners_ entry
  std::vector<float> g_raw_, g_layer_raw_;  // global use counts (decode rows) — true value = raw · g_scale_ · p_k = raw_k / layer sum × n_act, so the scale cancels
  float g_scale_ = 1.f;
  uint32_t pstep_ = 0;                // policy step (+1 per decay_scores)
  float p_alpha_ = 8.f, p_pre_ = 8.f, p_min_ = 0.025f, p_gdecay_ = 0.999f, p_odecay_ = 0.99f;
  int p_idle_ = 8;
  int parse_cache_policy(const char* v);
  void observe_policy(size_t k, float weight, int row);
  void set_row_owners_policy(const uint32_t* uids, int n, bool prompt);
  void decay_policy();
  int n_act_of(int l) const { return l < n_backbone_ ? cfg_.n_act : cfg_.dspark_act; }
  std::vector<int32_t> pending_;    // [slots] → key of the expert being promoted, or -1 (slot busy)
  std::vector<int32_t> pending_slot_;  // key -> pending slot; shared by every insertion path
  uint64_t promotions_ = 0, commits_ = 0, evictions_ = 0, duplicate_skips_ = 0, d2d_records_ = 0;
  mutable uint64_t h2d_records_ = 0;
  std::atomic<bool> staging_hold_{false};  // set_staging_hold (atomic: the G1 dispatcher thread calls copy_to_staging)
  bool reuse_staging_ = false;
  std::vector<int> staging_key_;
  std::vector<cudaEvent_t> staging_ready_, staging_reused_;
  std::vector<char> staging_reuse_pending_;
  // Dedicated D2D stream for REUSE_STAGE: staging → promotion slot D2D and staging_reused_ run outside the promotion stream (the
  //   H2D queue) — so staging overwrites are not pushed behind all promotion H2D. Batch completion (commit) covers the D2D by
  //   joining it into the promotion stream via reuse_done_.
  cudaStream_t reuse_st_ = nullptr;
  cudaEvent_t reuse_done_ = nullptr;
  bool reuse_join_pending_ = false;
  void join_reuse(cudaStream_t side);  // if this batch had a D2D, make side wait for it (right before recording the batch event)
  double commit_wait_ms_ = 0;  // commit_wait_ms()
  // E4: paced = paced batch (H2D pieces issued later) · issued = batch event recorded (always true for non-paced batches) · pieces_left = pieces not yet issued · st = promotion stream at decision time
  struct PromoBatch { cudaEvent_t evt; std::vector<int> slots; bool paced = false, issued = true; int pieces_left = 0; cudaStream_t st = nullptr; };
  struct PromoPiece { uint8_t* dst; const uint8_t* src; size_t n; };
  bool pace_ = false;
  std::deque<PromoPiece> pace_pieces_;  // pieces of all paced batches (batch order = order of the unissued tail of promo_q_)
  size_t pace_bytes_ = 0;
  int pace_unissued_ = 0;               // number of unissued batches at the tail of promo_q_ (unissued batches always gather at the tail — FIFO)
  int pace_batch_pieces_ = 0;           // pieces queued for the batch being built (only inside promote/promote_keys)
  uint64_t promo_h2d_bytes_ = 0;
  template <class F> void for_each_rec_piece(uint8_t* dst, int l, int e, F&& f) const;  // piece list shared by copy_rec_async and defer_rec (used only inside expert_store.cpp)
  void defer_rec(uint8_t* dst, int l, int e);  // the same 12 pieces as copy_rec_async into pace_pieces_ (h2d_records_ is counted at decision time — statistics unchanged)
  std::deque<PromoBatch> promo_q_;
  std::vector<cudaEvent_t> evt_pool_;
  std::vector<uint64_t> slot_last_use_;
  uint64_t use_total_ = 0;
  int clock_hand_ = 0;
  // pool
  struct Pool;
  Pool* pool_[2] = {nullptr, nullptr};
  std::atomic<int> pool_busy_{0};  // D2: shared by both pools (stealing) — number of workers currently touching next/items
  int threads_per_node_;
  bool owned_ = false;   // P1 (constructor: HIVE_DECODE_PREGATE > 0)
  int pf_spin_us_ = 0;   // P1 upper bound (µs) a worker that finished prefetching spins waiting for the job batch — sleeping would add wake-up latency to its owned items
  struct PfReq;          // P1 prefetch request (seqlock · expert_store.cpp)
  PfReq* pf_ = nullptr;
};

}  // namespace hive
