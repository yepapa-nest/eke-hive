// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_ENGRAM_SSD — offload mode that keeps the engram tables (layers 1 and 14, values FP8 [rows, 256] + scales E8M0 [rows, 8]) out
//   of RAM and reads only the needed rows from the checkpoint shards with O_DIRECT (default off = tables resident in RAM). No CUDA,
//   header-only (so the source list of the fake-CUDA CPU test build does not change).
//
// Measured basis (engine/tests/bench_engram_ssd.cpp — random rows of layers.1.engram.embed.weight in the real shard model-00047,
//   O_DIRECT pread, while a serving hived was running): the minimal 512-aligned span covering a row (256 B row → 512 B or 1024 B;
//   50 % straddle a 512 boundary)
//     1 thread = 16.4K IOPS, p50 58 µs / 8 = 126K, 60 µs / 32 = 458K, 66 µs / 64 = 783K, 73 µs / 128 = 1.12M IOPS, p50 100 µs p99 225 µs.
//   4 KiB block reads give the same IOPS (1.12M @128), 16 KiB gives 342K @128, the scale tensor (8 B rows) is the same (1.12M @128).
//   Device logical block 512 B.
//   → one row = one read (≈60–100 µs); IOPS is bought with queue depth. Hence (1) a thread-pool pread (no io_uring/liburing — headers
//     unavailable in both container and host), (2) a per-row RAM cache (CLOCK), (3) read-ahead as soon as the tokens are known (row
//     addresses depend only on token ids — EngramHash::peek).
// Measured locality (tools/engram_locality.py, hash matches engram_hash_selftest.json): 48 rows per token (2 layers × 24 columns).
//   Code corpus, 100K tokens: 55.5 % unique within a 16K window, LRU 1M rows (264 MiB) hit 53.9 %, 64K rows 37.7 %. Korean documents,
//   100K: 75.9 % unique in a 16K window, LRU 1M 33.0 %, 64K 15.7 %.
//
// Result bytes equal RAM mode (the same file bytes are copied to the same places — same output layout and same range-check message
// as engram_gather).
// Read failures are absorbed: per-row retries (EINTR, short read, EIO, … → up to `retries` times with 200 µs / 2 ms / 20 ms backoff);
//   an O_DIRECT alignment EINVAL switches to 4 KiB alignment and retries (absorbed); a file system that cannot open with O_DIRECT is
//   opened buffered (one warning line). If a row still cannot be read, gather throws std::runtime_error — hived ends only that
//   request (in a batch, that step's requests) with an error and the engine keeps running (hived.cpp admit_error, forward_batch
//   BatchGuard). Failed rows are not left in the cache (the next request reads them again). This rejection means "never compute on
//   bytes we do not have" (data integrity) — the one place where failing closed is the intended behavior.
#pragma once
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hive/clock.h"

namespace hive {

// ---- Switches (unset / "" / "0" = off = tables resident in RAM) ----
// HIVE_ENGRAM_SSD: 1 (or any other "on" value) = value tables on SSD, scales (3.07 GB per table) in RAM / "2" or "all" = scales on SSD too (two reads per row)
inline int engram_ssd_mode() {
  const char* v = getenv("HIVE_ENGRAM_SSD");
  if (!(v && *v && strcmp(v, "0") != 0)) return 0;
  return (strcmp(v, "2") == 0 || strcmp(v, "all") == 0) ? 2 : 1;
}
// HIVE_ENGRAM_SSD_CACHE_MB: row cache MiB (unset / "" / non-numeric = 2048, "0" = no cache — every row is read on demand, cap 1 TiB)
//   Rationale for 2048 (estimate): unique rows of one 100K-token chunk ≈ 100K × 48 × 0.56–0.76 ≈ 2.7–3.6M rows × 256 B ≈ 0.7–0.9 GiB (protects layer 1 + layer 14 read-ahead at once)
inline size_t engram_ssd_cache_bytes() {
  const char* v = getenv("HIVE_ENGRAM_SSD_CACHE_MB");
  if (!(v && *v)) return 2048ull << 20;
  char* end = nullptr;
  const double x = strtod(v, &end);
  if (!(end != v && end && *end == 0) || !(x >= 0) || x != x) return 2048ull << 20;
  return (size_t)(std::min(x, 1048576.0) * 1048576.0);
}
// HIVE_ENGRAM_SSD_THREADS: read threads = queue depth (unset / "" / "0" / non-numeric = 64, range 1..256). 64 = 783K IOPS measured, 128 = 1.12M
inline int engram_ssd_threads() {
  const char* v = getenv("HIVE_ENGRAM_SSD_THREADS");
  const int n = v && *v ? atoi(v) : 0;
  return n <= 0 ? 64 : std::min(n, 256);
}
// HIVE_ENGRAM_SSD_NO_PREFETCH=1: disable read-ahead (for comparison — rows are read in place at the layer 1/14 lookups) / 0
inline bool engram_ssd_no_prefetch() { const char* v = getenv("HIVE_ENGRAM_SSD_NO_PREFETCH"); return v && *v && strcmp(v, "0") != 0; }
// HIVE_ENGRAM_DIGEST=1 (verification log, both RAM and SSD modes): one `[engram-digest]` line per layer with a 64-bit hash of the engram lookup result (value and scale bytes) / 0
inline bool engram_digest_on() { const char* v = getenv("HIVE_ENGRAM_DIGEST"); return v && *v && strcmp(v, "0") != 0; }

inline uint64_t engram_digest(const uint8_t* p, size_t n, uint64_t x = 0xcbf29ce484222325ull) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, p + i, 8); x = (x ^ w) * 0x100000001b3ull; x ^= x >> 29; }
  for (; i < n; ++i) x = (x ^ p[i]) * 0x100000001b3ull;
  return x;
}

struct EngramSsdOpts {
  int mode = 1;                       // 1 = values on SSD, 2 = values + scales
  size_t cache_bytes = 2048ull << 20;  // row cache (in mode 2, split between values and scales by row byte ratio)
  int threads = 64;
  bool prefetch = true;
  int retries = 3;                    // extra attempts per row (not counting the first)
  static EngramSsdOpts from_env() {
    EngramSsdOpts o;
    o.mode = std::max(1, engram_ssd_mode());
    o.cache_bytes = engram_ssd_cache_bytes();
    o.threads = engram_ssd_threads();
    o.prefetch = !engram_ssd_no_prefetch();
    return o;
  }
};

// ---- Row cache: sharded (mutex + open-addressing hash + CLOCK) ----
//   States: EMPTY → LOADING (read issued) → READY | FAILED. Entries with pins > 0 (a lookup is waiting), LOADING, or protected
//   (read ahead and not yet taken by a lookup — until a later read-ahead epoch) are never evicted (protection = kProtectEpochs
//   epochs). If only protected entries remain and there is no free entry: read-ahead drops the row (the lookup reads it directly
//   later) and a lookup reads straight into the output, bypassing the cache.
//   → progress and results are the same at any cache size, including 0 (size only affects performance).
class EngramRowCache {
 public:
  enum : uint8_t { EMPTY = 0, LOADING = 1, READY = 2, FAILED = 3 };
  static constexpr uint32_t NONE = 0xffffffffu;
  struct Entry { uint64_t key = 0; uint32_t pins = 0; uint32_t prot = 0; uint8_t state = EMPTY; uint8_t ref = 0; };
  struct Shard {
    std::mutex mu;
    std::condition_variable cv;
    uint32_t base = 0, n = 0, hand = 0, mask = 0;
    std::vector<uint32_t> idx;  // open addressing (linear probing) → entry index, NONE = empty
  };
  void init(size_t entries, int row_bytes) {
    rb_ = row_bytes;
    entries = std::min<size_t>(entries, (size_t)NONE - 1);
    const int ns = entries == 0 ? 0 : (int)std::min<size_t>(64, entries);
    sh_ = std::vector<std::unique_ptr<Shard>>();
    for (int s = 0; s < ns; ++s) sh_.push_back(std::make_unique<Shard>());
    e_.assign(entries, Entry{});
    data_.reset(entries ? new uint8_t[entries * (size_t)rb_] : nullptr);
    size_t at = 0;
    for (int s = 0; s < ns; ++s) {
      Shard& S = *sh_[s];
      S.base = (uint32_t)at;
      S.n = (uint32_t)(entries / ns + ((size_t)s < entries % ns ? 1 : 0));
      at += S.n;
      uint32_t cap = 2;
      while (cap < 2 * S.n) cap <<= 1;
      S.mask = cap - 1;
      S.idx.assign(cap, NONE);
    }
  }
  size_t entries() const { return e_.size(); }
  size_t bytes() const { return e_.size() * ((size_t)rb_ + sizeof(Entry)) + idx_bytes(); }
  int row_bytes() const { return rb_; }
  bool empty() const { return sh_.empty(); }
  static uint64_t mix(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ull; k ^= k >> 33; return k; }
  Shard& shard_of(uint64_t key) { return *sh_[mix(key) % sh_.size()]; }
  Entry& e(uint32_t i) { return e_[i]; }
  uint8_t* data(uint32_t i) { return data_.get() + (size_t)i * rb_; }
  // ↓ all called with S.mu held
  uint32_t find(Shard& S, uint64_t key) const {
    for (uint32_t p = (uint32_t)(mix(key ^ 0x9E3779B97F4A7C15ull)) & S.mask;; p = (p + 1) & S.mask) {
      const uint32_t i = S.idx[p];
      if (i == NONE) return NONE;
      if (e_[i].key == key) return i;
    }
  }
  void insert(Shard& S, uint64_t key, uint32_t i) {
    e_[i].key = key;
    uint32_t p = (uint32_t)(mix(key ^ 0x9E3779B97F4A7C15ull)) & S.mask;
    while (S.idx[p] != NONE) p = (p + 1) & S.mask;
    S.idx[p] = i;
  }
  void erase(Shard& S, uint32_t i) {  // linear-probing backward-shift deletion (no tombstones)
    uint32_t p = (uint32_t)(mix(e_[i].key ^ 0x9E3779B97F4A7C15ull)) & S.mask;
    while (S.idx[p] != i) p = (p + 1) & S.mask;
    uint32_t hole = p;
    for (uint32_t q = (hole + 1) & S.mask; S.idx[q] != NONE; q = (q + 1) & S.mask) {
      const uint32_t home = (uint32_t)(mix(e_[S.idx[q]].key ^ 0x9E3779B97F4A7C15ull)) & S.mask;
      // if home lies outside the cyclic interval (hole, q], moving it into hole keeps it reachable by probing
      const bool in = hole <= q ? (home > hole && home <= q) : (home > hole || home <= q);
      if (!in) { S.idx[hole] = S.idx[q]; hole = q; }
    }
    S.idx[hole] = NONE;
    e_[i].state = EMPTY; e_[i].pins = 0; e_[i].prot = 0; e_[i].ref = 0;
  }
  // Protection window = 16 read-ahead epochs (one per forward — so the layer 14 read-ahead survives even when layer yielding (T11)
  //   inside a long prefill forward interleaves a few decode steps; the protection of an aborted forward expires after 16 forwards).
  //   The size only affects performance (results unaffected).
  static constexpr uint32_t kProtectEpochs = 16;
  static bool protected_now(const Entry& E, uint32_t epoch) { return E.prot != 0 && epoch - (E.prot - 1) < kProtectEpochs; }
  // A free or evictable entry (CLOCK — an entry with its reference bit set gets one more lap). NONE if there is none.
  uint32_t alloc(Shard& S, uint32_t epoch) {
    for (uint32_t step = 0; step < 2 * S.n + 1; ++step) {
      const uint32_t i = S.base + S.hand;
      S.hand = S.hand + 1 == S.n ? 0 : S.hand + 1;
      Entry& E = e_[i];
      if (E.state == EMPTY) return i;
      if (E.state == LOADING || E.pins || protected_now(E, epoch)) continue;
      if (E.ref) { E.ref = 0; continue; }
      erase(S, i);
      return i;
    }
    return NONE;
  }

 private:
  size_t idx_bytes() const { size_t n = 0; for (const auto& s : sh_) n += s->idx.size() * 4; return n; }
  int rb_ = 0;
  std::vector<std::unique_ptr<Shard>> sh_;
  std::vector<Entry> e_;
  std::unique_ptr<uint8_t[]> data_;
};

class EngramSsd {
 public:
  struct Stats {
    // Lookups (gather): rows, already in cache, being read by read-ahead, read into the cache by the lookup, read straight into the output without cache
    uint64_t gathers = 0, rows = 0, hit = 0, inflight = 0, sync = 0, bypass = 0;
    uint64_t pf_calls = 0, pf_rows = 0, pf_issued = 0, pf_present = 0, pf_dropped = 0;  // read-ahead
    uint64_t reads = 0, read_bytes = 0, retries = 0, failed = 0, align_switch = 0;
    double wait_ms = 0, wait_max_ms = 0;  // time lookups waited for reads to finish (sum, max of a single lookup)
  };
  struct GatherInfo { uint64_t rows = 0, hit = 0, inflight = 0, sync = 0, bypass = 0; double wait_ms = 0; };

  explicit EngramSsd(const EngramSsdOpts& o) : opt_(o) {
    opt_.threads = std::max(1, opt_.threads);
    for (int k = 0; k < 2; ++k) cache_[k].init(0, k == 0 ? 256 : 8);
    for (int t = 0; t < opt_.threads; ++t) th_.emplace_back([this] { worker(); });
  }
  ~EngramSsd() {
    { std::lock_guard<std::mutex> lk(qmu_); stop_ = true; }
    qcv_.notify_all();
    for (auto& t : th_) t.join();
    for (auto& T : tables_) for (int k = 0; k < 2; ++k) if (T.fd[k] >= 0) ::close(T.fd[k]);
  }
  EngramSsd(const EngramSsd&) = delete;
  EngramSsd& operator=(const EngramSsd&) = delete;

  const EngramSsdOpts& opts() const { return opt_; }
  // Register a table (in hash_index order). vals_off/scales_off = tensor start byte within the file. If scales_ram != nullptr the
  //   scales come from that RAM copy (mode 1). Returns whether the file actually opened with O_DIRECT (false = absorbed as buffered).
  //   If the file cannot be opened: std::runtime_error (startup failure — same as the open in the RAM-mode load_engram).
  bool add_table(int hidx, const std::string& vals_path, uint64_t vals_off, const std::string& scales_path, uint64_t scales_off, int64_t rows, int hd,
                 const uint8_t* scales_ram) {
    if (hd != 256) throw std::runtime_error("engram ssd: head_dim 256 expected");
    if (opt_.mode != 2 && !scales_ram) throw std::runtime_error("engram ssd: mode 1 needs the RAM scale table");
    std::lock_guard<std::mutex> lk(cfg_mu_);
    if ((int)tables_.size() <= hidx) tables_.resize(hidx + 1);
    Table& T = tables_[hidx];
    for (int k = 0; k < 2; ++k) if (T.fd[k] >= 0) { ::close(T.fd[k]); T.fd[k] = -1; }
    T.off[0] = vals_off; T.off[1] = scales_off; T.rb[0] = hd; T.rb[1] = hd / 32; T.rows = rows; T.scales_ram = scales_ram;
    for (int k = 0; k < (opt_.mode == 2 ? 2 : 1); ++k) {
      const std::string& path = k == 0 ? vals_path : scales_path;
      T.path[k] = path;
      T.fd[k] = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
      T.direct[k] = T.fd[k] >= 0;
      if (T.fd[k] < 0) {  // tmpfs etc. cannot do O_DIRECT → buffered (page cache will grow — warn)
        T.fd[k] = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (T.fd[k] < 0) throw std::runtime_error("engram ssd: cannot open " + path + ": " + strerror(errno));
        fprintf(stderr, "[engram-ssd] ⚠️O_DIRECT open failed for %s — buffered reads (page cache will grow)\n", path.c_str());
      }
    }
    return T.direct[0];
  }
  // Allocate the cache (once, after table registration; later calls are ignored). In mode 2, values:scales = 256:8 byte ratio.
  bool alloc_cache() {  // returns whether this call allocated
    std::lock_guard<std::mutex> lk(cfg_mu_);
    if (cache_ready_) return false;
    // Per-row bound: data + entry + index (shard index slots = power of two >= 2·n and < 4·n → < 16 B per row, per cache). Mode 2
    //   must include the 16 B index of the scale cache as well, otherwise the real size exceeds the budget (CPU test: 1,062,688 B for a 1 MiB budget).
    const size_t per_row = 256 + (opt_.mode == 2 ? 8 + 16 : 0) + 2 * sizeof(EngramRowCache::Entry) + 16;
    const size_t rows = opt_.cache_bytes / per_row;
    cache_[0].init(rows, 256);
    cache_[1].init(opt_.mode == 2 ? rows : 0, 8);
    cache_ready_ = true;
    return true;
  }
  size_t cache_host_bytes() const { return cache_[0].bytes() + cache_[1].bytes(); }
  size_t cache_rows() const { return cache_[0].entries(); }
  bool direct(int hidx) const { return hidx < (int)tables_.size() && tables_[hidx].direct[0]; }
  int n_tables() const { return (int)tables_.size(); }

  // ---- Read-ahead (asynchronous, no effect on results — reading a wrong row only occupies a cache entry) ----
  // make(hashes): builds the hashes [M, nL, cols] on a worker thread (so the engine thread is not held — hashing 100K rows takes tens of ms). Reads the table rows from hidx_from onward.
  void prefetch(std::function<void(std::vector<int64_t>&)> make, int M, int nL, int cols, int hidx_from) {
    if (!opt_.prefetch || cache_[0].empty() || M <= 0) return;
    const uint32_t ep = epoch_.fetch_add(1) + 1;
    { std::lock_guard<std::mutex> lk(st_mu_); st_.pf_calls++; }
    push_plan([this, make = std::move(make), M, nL, cols, hidx_from, ep] {
      auto h = std::make_shared<std::vector<int64_t>>();
      make(*h);
      if (h->size() != (size_t)M * nL * cols) return;
      plan_all(h, M, nL, cols, hidx_from, ep);
    });
  }
  void prefetch_hashes(const int64_t* hashes, int M, int nL, int cols, int hidx_from) {
    if (!opt_.prefetch || cache_[0].empty() || M <= 0 || hidx_from >= nL) return;
    const uint32_t ep = epoch_.load();  // same epoch (layer 14 from layer 1 — protected under the same epoch as the read-ahead at the start of the forward)
    auto h = std::make_shared<std::vector<int64_t>>(hashes, hashes + (size_t)M * nL * cols);
    { std::lock_guard<std::mutex> lk(st_mu_); st_.pf_calls++; }
    push_plan([this, h, M, nL, cols, hidx_from, ep] { plan_all(h, M, nL, cols, hidx_from, ep); });
  }

  // ---- Lookup: same output as engram_gather (runtime.cpp) — v_out[(m·cols + col)·hd], s_out[(m·cols + col)·hd/32] ----
  //   All range checks first (same message "engram row range"), then reads. With threads > 1 and M >= 64 the rows are split so
  //   bookkeeping and waiting run in parallel (each row has its own destination, so the bytes are identical).
  GatherInfo gather(const int64_t* hashes, int M, int nL, int hidx, int cols, int hd, int64_t n_rows, uint8_t* v_out, uint8_t* s_out, int threads) {
    for (int m = 0; m < M; ++m)
      for (int col = 0; col < cols; ++col) {
        const int64_t row = hashes[((size_t)m * nL + hidx) * cols + col];
        if (!(row >= 0 && row < n_rows)) throw std::runtime_error("engram row range");
      }
    const auto t0 = hive::SteadyClock::now();
    const int nt = (threads > 1 && M >= 2 * 64) ? std::min(threads, M / 64) : 1;
    std::vector<GatherInfo> gi(nt);
    std::vector<std::string> err(nt);
    auto run = [&](int t) {
      const int per = (M + nt - 1) / nt, m0 = t * per, m1 = std::min(M, m0 + per);
      try { gather_rows(hashes, m0, m1, nL, hidx, cols, hd, v_out, s_out, gi[t]); } catch (const std::exception& e) { err[t] = e.what(); }
    };
    if (nt == 1) run(0);
    else {
      std::vector<std::thread> th;
      for (int t = 1; t < nt; ++t) th.emplace_back(run, t);
      run(0);
      for (auto& x : th) x.join();
    }
    GatherInfo g;
    for (auto& x : gi) { g.rows += x.rows; g.hit += x.hit; g.inflight += x.inflight; g.sync += x.sync; g.bypass += x.bypass; g.wait_ms = std::max(g.wait_ms, x.wait_ms); }
    (void)t0;
    {
      std::lock_guard<std::mutex> lk(st_mu_);
      st_.gathers++; st_.rows += g.rows; st_.hit += g.hit; st_.inflight += g.inflight; st_.sync += g.sync; st_.bypass += g.bypass;
      st_.wait_ms += g.wait_ms; st_.wait_max_ms = std::max(st_.wait_max_ms, g.wait_ms);
    }
    for (auto& e : err) if (!e.empty()) throw std::runtime_error(e);
    return g;
  }

  Stats stats(bool reset = false) {
    std::lock_guard<std::mutex> lk(st_mu_);
    Stats s = st_;
    s.reads = reads_.load(); s.read_bytes = read_bytes_.load(); s.retries = retries_.load(); s.failed = failed_.load(); s.align_switch = align_switch_.load();
    s.pf_rows = pf_rows_.load(); s.pf_issued = pf_issued_.load(); s.pf_present = pf_present_.load(); s.pf_dropped = pf_dropped_.load();
    if (reset) {
      st_ = Stats{};
      reads_ = 0; read_bytes_ = 0; retries_ = 0; failed_ = 0; align_switch_ = 0; pf_rows_ = 0; pf_issued_ = 0; pf_present_ = 0; pf_dropped_ = 0;
    }
    return s;
  }
  // Until the queue is empty (tests, before shutdown). Covers both read-ahead planning and reads.
  void quiesce() {
    std::unique_lock<std::mutex> lk(qmu_);
    idle_cv_.wait(lk, [&] { return busy_ == 0 && urgent_.empty() && pf_.empty() && plans_.empty(); });
  }
  // Test-only failure injection: (kind 0 values / 1 scales, row, attempt number) → 0 = normal, an errno value = fail that attempt with that error.
  std::function<int(int, int64_t, int)> fault;

 private:
  struct Table { std::string path[2]; int fd[2] = {-1, -1}; bool direct[2] = {false, false}; uint64_t off[2] = {0, 0}; int rb[2] = {256, 8}; int64_t rows = 0; const uint8_t* scales_ram = nullptr; };
  struct Waiter {
    std::mutex mu; std::condition_variable cv; int pending = 0; int failed = 0; std::string err;
    void done(bool ok, const std::string& e) { std::lock_guard<std::mutex> lk(mu); if (!ok) { failed++; if (err.empty()) err = e; } if (--pending == 0) cv.notify_all(); }
  };
  struct Job { uint8_t kind = 0; int table = 0; int64_t row = 0; uint32_t ei = EngramRowCache::NONE; uint8_t* dst = nullptr; Waiter* w = nullptr; };

  static uint64_t key_of(int table, int64_t row) { return ((uint64_t)table << 40) | (uint64_t)row; }

  void push_plan(std::function<void()> f) {
    { std::lock_guard<std::mutex> lk(qmu_); plans_.push_back(std::move(f)); }
    qcv_.notify_one();
  }
  void push_jobs(std::vector<Job>& jobs, bool urgent) {
    if (jobs.empty()) return;
    {
      std::lock_guard<std::mutex> lk(qmu_);
      auto& q = urgent ? urgent_ : pf_;
      for (auto& j : jobs) q.push_back(j);
    }
    if (jobs.size() == 1) qcv_.notify_one(); else qcv_.notify_all();
    jobs.clear();
  }
  void worker() {
    for (;;) {
      Job j; std::function<void()> plan; bool is_plan = false;
      {
        std::unique_lock<std::mutex> lk(qmu_);
        qcv_.wait(lk, [&] { return stop_ || !urgent_.empty() || !pf_.empty() || !plans_.empty(); });
        if (stop_ && urgent_.empty() && pf_.empty() && plans_.empty()) return;
        if (!urgent_.empty()) { j = urgent_.front(); urgent_.pop_front(); }
        else if (!pf_.empty()) { j = pf_.front(); pf_.pop_front(); }
        else { plan = std::move(plans_.front()); plans_.pop_front(); is_plan = true; }
        busy_++;
      }
      if (is_plan) { try { plan(); } catch (...) {} }  // read-ahead planning failures are ignored (the lookup reads directly)
      else run_job(j);
      {
        std::lock_guard<std::mutex> lk(qmu_);
        busy_--;
        if (busy_ == 0 && urgent_.empty() && pf_.empty() && plans_.empty()) idle_cv_.notify_all();
      }
    }
  }
  // Read one row from the file into dst (retries, alignment absorbed). Returns success; err = last error.
  bool read_row(int kind, int table, int64_t row, uint8_t* dst, std::string& err) {
    const Table& T = tables_[table];
    const uint64_t off = T.off[kind] + (uint64_t)row * T.rb[kind];
    const size_t len = (size_t)T.rb[kind];
    thread_local std::unique_ptr<uint8_t, void (*)(void*)> buf(nullptr, free);
    if (!buf) { void* p = nullptr; if (posix_memalign(&p, 4096, 1 << 16)) { err = "posix_memalign"; return false; } buf.reset((uint8_t*)p); }
    static const int backoff_us[] = {200, 2000, 20000};
    for (int attempt = 0; attempt <= opt_.retries; ++attempt) {
      const uint64_t al = T.direct[kind] ? align_.load() : 1;
      const uint64_t a = off / al * al, b = (off + len + al - 1) / al * al;
      const int inj = fault ? fault(kind, row, attempt) : 0;
      ssize_t r;
      int e = 0;
      if (inj) { r = -1; e = inj; }
      else { do { r = pread(T.fd[kind], buf.get(), (size_t)(b - a), (off_t)a); } while (r < 0 && errno == EINTR); e = r < 0 ? errno : 0; }
      if (r >= 0 && (uint64_t)r >= off + len - a) {
        memcpy(dst, buf.get() + (off - a), len);
        reads_++; read_bytes_ += (uint64_t)(b - a);
        return true;
      }
      if (r < 0 && e == EINVAL && T.direct[kind] && al < 4096) {  // device logical block larger than 512 → switch to 4 KiB alignment (absorbed, not counted as an attempt)
        uint64_t want = al;
        if (align_.compare_exchange_strong(want, 4096)) { align_switch_++; fprintf(stderr, "[engram-ssd] O_DIRECT 512 B alignment rejected (EINVAL) — using 4096\n"); }
        --attempt;
        continue;
      }
      err = r < 0 ? std::string(strerror(e)) : ("short read " + std::to_string(r) + " of " + std::to_string(b - a) + " at " + std::to_string(a));
      if (attempt < opt_.retries) { retries_++; std::this_thread::sleep_for(std::chrono::microseconds(backoff_us[std::min(attempt, 2)])); }
    }
    failed_++;
    return false;
  }
  void run_job(const Job& j) {
    std::string err;
    if (j.ei == EngramRowCache::NONE) {  // straight into the output, no cache
      const bool ok = read_row(j.kind, j.table, j.row, j.dst, err);
      j.w->done(ok, err.empty() ? "" : "engram ssd read failed (table " + std::to_string(j.table) + " row " + std::to_string(j.row) + "): " + err);
      return;
    }
    EngramRowCache& C = cache_[j.kind];
    const bool ok = read_row(j.kind, j.table, j.row, C.data(j.ei), err);  // a LOADING entry is written only by this job (not an eviction candidate)
    EngramRowCache::Shard& S = C.shard_of(key_of(j.table, j.row));
    {
      std::lock_guard<std::mutex> lk(S.mu);
      EngramRowCache::Entry& E = C.e(j.ei);
      if (ok) E.state = EngramRowCache::READY;
      else if (E.pins == 0) C.erase(S, j.ei);  // no waiting lookup → free it now (the next request reads it again)
      else { E.state = EngramRowCache::FAILED; std::lock_guard<std::mutex> le(err_mu_); last_err_ = err; }
    }
    S.cv.notify_all();
  }
  void plan_all(const std::shared_ptr<std::vector<int64_t>>& h, int M, int nL, int cols, int hidx_from, uint32_t ep) {
    // Queue per-chunk plans in table (layer) order — layer 1 row reads go out before layer 14 rows (FIFO queue)
    const int chunk = 2048;
    for (int hidx = hidx_from; hidx < nL && hidx < (int)tables_.size(); ++hidx)
      for (int m0 = 0; m0 < M; m0 += chunk) {
        const int m1 = std::min(M, m0 + chunk);
        if (hidx == hidx_from && m0 == 0) plan_rows(*h, m0, m1, nL, hidx, cols, ep);  // the first chunk is planned right here on this thread
        else push_plan([this, h, m0, m1, nL, hidx, cols, ep] { plan_rows(*h, m0, m1, nL, hidx, cols, ep); });
      }
  }
  void plan_rows(const std::vector<int64_t>& h, int m0, int m1, int nL, int hidx, int cols, uint32_t ep) {
    const Table& T = tables_[hidx];
    std::vector<Job> jobs;
    uint64_t rows = 0, issued = 0, present = 0, dropped = 0;
    for (int m = m0; m < m1; ++m)
      for (int col = 0; col < cols; ++col) {
        const int64_t row = h[((size_t)m * nL + hidx) * cols + col];
        if (!(row >= 0 && row < T.rows)) continue;  // read-ahead is a hint — out-of-range rows are rejected by the lookup
        for (int kind = 0; kind < (opt_.mode == 2 ? 2 : 1); ++kind) {
          rows++;
          EngramRowCache& C = cache_[kind];
          const uint64_t key = key_of(hidx, row);
          EngramRowCache::Shard& S = C.shard_of(key);
          std::lock_guard<std::mutex> lk(S.mu);
          uint32_t i = C.find(S, key);
          if (i != EngramRowCache::NONE) {
            EngramRowCache::Entry& E = C.e(i);
            if (E.state != EngramRowCache::FAILED) { E.prot = ep + 1; E.ref = 1; }
            present++;
            continue;
          }
          i = C.alloc(S, ep);
          if (i == EngramRowCache::NONE) { dropped++; continue; }
          C.insert(S, key, i);
          EngramRowCache::Entry& E = C.e(i);
          E.state = EngramRowCache::LOADING; E.pins = 0; E.prot = ep + 1; E.ref = 1;
          jobs.push_back(Job{(uint8_t)kind, hidx, row, i, nullptr, nullptr});
          issued++;
        }
      }
    pf_rows_ += rows; pf_issued_ += issued; pf_present_ += present; pf_dropped_ += dropped;
    push_jobs(jobs, false);
  }
  void gather_rows(const int64_t* hashes, int m0, int m1, int nL, int hidx, int cols, int hd, uint8_t* v_out, uint8_t* s_out, GatherInfo& g) {
    const Table& T = tables_.at(hidx);
    const int sb = hd / 32;
    const uint32_t ep = epoch_.load();
    struct Wait { uint8_t kind; uint32_t ei; uint64_t key; uint8_t* dst; };
    std::vector<Wait> waits;
    std::vector<Job> jobs;
    Waiter bw;
    for (int m = m0; m < m1; ++m)
      for (int col = 0; col < cols; ++col) {
        const int64_t row = hashes[((size_t)m * nL + hidx) * cols + col];
        uint8_t* dv = v_out + ((size_t)m * cols + col) * hd;
        uint8_t* ds = s_out + ((size_t)m * cols + col) * sb;
        if (opt_.mode != 2) memcpy(ds, T.scales_ram + (size_t)row * sb, sb);  // mode 1: scales from the RAM copy (same bytes as RAM mode)
        for (int kind = 0; kind < (opt_.mode == 2 ? 2 : 1); ++kind) {
          uint8_t* dst = kind == 0 ? dv : ds;
          g.rows++;
          EngramRowCache& C = cache_[kind];
          if (C.empty()) { jobs.push_back(Job{(uint8_t)kind, hidx, row, EngramRowCache::NONE, dst, &bw}); g.bypass++; continue; }
          const uint64_t key = key_of(hidx, row);
          EngramRowCache::Shard& S = C.shard_of(key);
          std::lock_guard<std::mutex> lk(S.mu);
          uint32_t i = C.find(S, key);
          if (i != EngramRowCache::NONE) {
            EngramRowCache::Entry& E = C.e(i);
            if (E.state == EngramRowCache::READY) { memcpy(dst, C.data(i), C.row_bytes()); E.ref = 1; E.prot = 0; g.hit++; continue; }
            if (E.state == EngramRowCache::LOADING) { E.pins++; waits.push_back({(uint8_t)kind, i, key, dst}); g.inflight++; continue; }
            // FAILED (an earlier failure, kept because a lookup was waiting): read again
            E.state = EngramRowCache::LOADING; E.pins++;
            waits.push_back({(uint8_t)kind, i, key, dst});
            jobs.push_back(Job{(uint8_t)kind, hidx, row, i, nullptr, nullptr});
            g.sync++;
            continue;
          }
          i = C.alloc(S, ep);
          if (i == EngramRowCache::NONE) { jobs.push_back(Job{(uint8_t)kind, hidx, row, EngramRowCache::NONE, dst, &bw}); g.bypass++; continue; }
          C.insert(S, key, i);
          EngramRowCache::Entry& E = C.e(i);
          E.state = EngramRowCache::LOADING; E.pins = 1; E.prot = 0; E.ref = 1;
          waits.push_back({(uint8_t)kind, i, key, dst});
          jobs.push_back(Job{(uint8_t)kind, hidx, row, i, nullptr, nullptr});
          g.sync++;
        }
      }
    { std::lock_guard<std::mutex> lk(bw.mu); for (const Job& j : jobs) if (j.ei == EngramRowCache::NONE) bw.pending++; }
    push_jobs(jobs, true);
    const auto t0 = hive::SteadyClock::now();
    std::string err;
    for (const Wait& w : waits) {
      EngramRowCache& C = cache_[w.kind];
      EngramRowCache::Shard& S = C.shard_of(w.key);
      std::unique_lock<std::mutex> lk(S.mu);
      EngramRowCache::Entry& E = C.e(w.ei);
      S.cv.wait(lk, [&] { return E.state != EngramRowCache::LOADING; });
      if (E.state == EngramRowCache::READY) { memcpy(w.dst, C.data(w.ei), C.row_bytes()); E.prot = 0; }
      else if (err.empty()) { std::lock_guard<std::mutex> le(err_mu_); err = "engram ssd read failed (table " + std::to_string(hidx) + " row " + std::to_string(w.key & ((1ull << 40) - 1)) + "): " + last_err_; }
      E.pins--;
      if (E.state == EngramRowCache::FAILED && E.pins == 0) C.erase(S, w.ei);
    }
    {
      std::unique_lock<std::mutex> lk(bw.mu);
      bw.cv.wait(lk, [&] { return bw.pending == 0; });
      if (bw.failed && err.empty()) err = bw.err;
    }
    g.wait_ms = hive::ms_since(t0);
    if (!err.empty()) throw std::runtime_error(err);
  }

  EngramSsdOpts opt_;
  std::mutex cfg_mu_;
  std::vector<Table> tables_;
  bool cache_ready_ = false;
  EngramRowCache cache_[2];
  std::atomic<uint32_t> epoch_{1};
  std::atomic<uint64_t> align_{512};
  std::mutex qmu_;
  std::condition_variable qcv_, idle_cv_;
  std::deque<Job> urgent_, pf_;
  std::deque<std::function<void()>> plans_;
  int busy_ = 0;
  bool stop_ = false;
  std::vector<std::thread> th_;
  std::mutex st_mu_;
  Stats st_;
  std::mutex err_mu_;
  std::string last_err_;  // last read failure message (for the error text — not an exact row mapping)
  std::atomic<uint64_t> reads_{0}, read_bytes_{0}, retries_{0}, failed_{0}, align_switch_{0}, pf_rows_{0}, pf_issued_{0}, pf_present_{0}, pf_dropped_{0};
};

}  // namespace hive
