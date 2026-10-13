// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// PLE hash and table — see hive/qwen/ple.h.
//   RAM table: every shard copied once into pinned host memory (47.7 GiB); a lookup is 16 memcpy of 160 B per token.
//   SSD table (HIVE_QWEN_PLE=ssd): the shards stay in the checkpoint file; rows are read with O_DIRECT pread by a thread pool into a row
//     cache (CLOCK, hive/engram_ssd.h EngramRowCache), with read-ahead as soon as a request's tokens are known (rows depend only on token
//     ids). The design and the reasons are those of HIVE_ENGRAM_SSD (engram_ssd.h header: one row = one read, IOPS bought with queue
//     depth). A row that cannot be read after the retries fails the request (never computed on bytes we do not have).
//   Switches: HIVE_QWEN_PLE = ram (default) | ssd · HIVE_QWEN_PLE_SSD_CACHE_MB (default 2048) · HIVE_QWEN_PLE_SSD_THREADS (default 64).
#include "hive/qwen/ple.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <cuda_runtime.h>

#include "hive/clock.h"
#include "hive/common.h"
#include "hive/engram_ssd.h"

namespace hive::qwen {

namespace {

struct Shards {
  std::vector<const TensorInfo*> t;   // valid only while the checkpoint is open (hived releases it after startup — the RAM copy is made before)
  std::vector<uint64_t> off;          // file offset of each shard's data (kept: the SSD reader needs it after the release)
  std::vector<int64_t> start;   // first global row of each shard
  int64_t rows = 0;
  size_t shard_of(int64_t r) const { return (size_t)(std::upper_bound(start.begin(), start.end(), r) - start.begin()) - 1; }
};

// ---- RAM ----------------------------------------------------------------------------------------------------------------------------------
class PleRam final : public PleTable {
 public:
  explicit PleRam(const Shards& s) : rows_(s.rows) {
    bytes_ = (size_t)rows_ * kPleHeadDim;
    CUDA_CHECK(cudaHostAlloc(&data_, bytes_, cudaHostAllocDefault));
    const double t0 = hive::mono_ms();
    std::atomic<size_t> next{0};
    const int nt = std::max(8, std::min(32, (int)std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int k = 0; k < nt; ++k)
      th.emplace_back([&] {
        for (size_t i; (i = next.fetch_add(1)) < s.t.size();)
          std::memcpy(data_ + (size_t)s.start[i] * kPleHeadDim, s.t[i]->data, s.t[i]->nbytes);
      });
    for (auto& x : th) x.join();
    fprintf(stderr, "[qwen] PLE table in RAM: %lld rows · %.1f GiB pinned in %.1f s\n", (long long)rows_, bytes_ / 1073741824.0, (hive::mono_ms() - t0) / 1000);
  }
  ~PleRam() override { if (data_) cudaFreeHost(data_); }
  void gather(const int64_t* rows, int n, uint8_t* dst) override {
    auto run = [&](int a, int b) {
      for (int i = a; i < b; ++i)
        for (int j = 0; j < kPleHeads; ++j) {
          const int64_t r = rows[(size_t)i * kPleHeads + j];
          if (!(r >= 0 && r < rows_)) throw std::runtime_error("PLE row range");
          std::memcpy(dst + ((size_t)i * kPleHeads + j) * kPleHeadDim, data_ + (size_t)r * kPleHeadDim, kPleHeadDim);
        }
    };
    if (n < 4096) { run(0, n); return; }
    const int nt = std::min(16, n / 2048);
    std::vector<std::thread> th;
    std::vector<std::string> err(nt);
    for (int k = 0; k < nt; ++k)
      th.emplace_back([&, k] { try { run((int)((int64_t)n * k / nt), (int)((int64_t)n * (k + 1) / nt)); } catch (const std::exception& e) { err[k] = e.what(); } });
    for (auto& x : th) x.join();
    for (auto& e : err) if (!e.empty()) throw std::runtime_error(e);
  }
  size_t host_bytes() const override { return bytes_; }
  const char* kind() const override { return "ram"; }

 private:
  int64_t rows_;
  size_t bytes_ = 0;
  uint8_t* data_ = nullptr;
};

// ---- SSD ----------------------------------------------------------------------------------------------------------------------------------
class PleSsd final : public PleTable {
 public:
  PleSsd(const Shards& s, const std::string& path, size_t cache_bytes, int threads) : sh_(s) {
    sh_.t.clear();  // TensorInfo pointers die with Checkpoint::release_all — only offsets and row starts are used
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    direct_ = fd_ >= 0;
    if (fd_ < 0) {
      fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
      if (fd_ < 0) throw std::runtime_error("PLE ssd: cannot open " + path + ": " + strerror(errno));
      fprintf(stderr, "[qwen] ⚠️PLE ssd: O_DIRECT open failed for %s — buffered reads (the page cache will grow)\n", path.c_str());
    }
    const size_t per_row = kPleHeadDim + sizeof(EngramRowCache::Entry) + 16;
    cache_.init(cache_bytes / per_row, kPleHeadDim);
    for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { worker(); });
    fprintf(stderr, "[qwen] PLE table on SSD: %lld rows in %s · %s · row cache %.0f MiB · %d readers\n", (long long)s.rows, path.c_str(),
            direct_ ? "O_DIRECT" : "buffered", cache_.bytes() / 1048576.0, threads);
  }
  ~PleSsd() override {
    { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
    if (fd_ >= 0) ::close(fd_);
  }
  void prefetch(const int64_t* rows, int n) override {
    if (cache_.empty()) return;
    const uint32_t ep = ++epoch_;
    std::vector<Job> jobs;
    for (size_t i = 0; i < (size_t)n * kPleHeads; ++i) {
      const int64_t r = rows[i];
      if (!(r >= 0 && r < sh_.rows)) continue;
      EngramRowCache::Shard& S = cache_.shard_of((uint64_t)r);
      std::lock_guard<std::mutex> lk(S.mu);
      if (cache_.find(S, (uint64_t)r) != EngramRowCache::NONE) continue;
      const uint32_t e = cache_.alloc(S, ep);
      if (e == EngramRowCache::NONE) break;  // only protected / in-flight entries left: the lookup reads directly
      cache_.insert(S, (uint64_t)r, e);
      EngramRowCache::Entry& E = cache_.e(e);
      E.state = EngramRowCache::LOADING; E.pins = 0; E.prot = ep + 1; E.ref = 1;
      jobs.push_back(Job{r, e, nullptr, nullptr});
    }
    push(jobs, false);
  }
  void gather(const int64_t* rows, int n, uint8_t* dst) override {
    const size_t total = (size_t)n * kPleHeads;
    for (size_t i = 0; i < total; ++i) if (!(rows[i] >= 0 && rows[i] < sh_.rows)) throw std::runtime_error("PLE row range");
    const uint32_t ep = epoch_.load();
    struct Wait { uint32_t e; uint64_t key; uint8_t* dst; };
    std::vector<Wait> waits;
    std::vector<Job> jobs;
    Waiter bw;
    for (size_t i = 0; i < total; ++i) {
      const int64_t r = rows[i];
      uint8_t* d = dst + i * kPleHeadDim;
      if (cache_.empty()) { jobs.push_back(Job{r, EngramRowCache::NONE, d, &bw}); continue; }
      EngramRowCache::Shard& S = cache_.shard_of((uint64_t)r);
      std::lock_guard<std::mutex> lk(S.mu);
      uint32_t e = cache_.find(S, (uint64_t)r);
      if (e != EngramRowCache::NONE) {
        EngramRowCache::Entry& E = cache_.e(e);
        if (E.state == EngramRowCache::READY) { std::memcpy(d, cache_.data(e), kPleHeadDim); E.ref = 1; E.prot = 0; ++hit_; continue; }
        E.pins++;
        if (E.state == EngramRowCache::FAILED) { E.state = EngramRowCache::LOADING; jobs.push_back(Job{r, e, nullptr, nullptr}); }
        waits.push_back({e, (uint64_t)r, d});
        continue;
      }
      e = cache_.alloc(S, ep);
      if (e == EngramRowCache::NONE) { jobs.push_back(Job{r, EngramRowCache::NONE, d, &bw}); continue; }
      cache_.insert(S, (uint64_t)r, e);
      EngramRowCache::Entry& E = cache_.e(e);
      E.state = EngramRowCache::LOADING; E.pins = 1; E.prot = 0; E.ref = 1;
      waits.push_back({e, (uint64_t)r, d});
      jobs.push_back(Job{r, e, nullptr, nullptr});
    }
    { std::lock_guard<std::mutex> lk(bw.mu); for (const Job& j : jobs) if (j.e == EngramRowCache::NONE) bw.pending++; }
    push(jobs, true);
    std::string err;
    for (const Wait& w : waits) {
      EngramRowCache::Shard& S = cache_.shard_of(w.key);
      std::unique_lock<std::mutex> lk(S.mu);
      EngramRowCache::Entry& E = cache_.e(w.e);
      S.cv.wait(lk, [&] { return E.state != EngramRowCache::LOADING; });
      if (E.state == EngramRowCache::READY) std::memcpy(w.dst, cache_.data(w.e), kPleHeadDim);
      else if (err.empty()) err = "PLE ssd read failed (row " + std::to_string(w.key) + ")";
      E.pins--;
      if (E.state == EngramRowCache::FAILED && E.pins == 0) cache_.erase(S, w.e);
    }
    {
      std::unique_lock<std::mutex> lk(bw.mu);
      bw.cv.wait(lk, [&] { return bw.pending == 0; });
      if (bw.failed && err.empty()) err = bw.err;
    }
    if (!err.empty()) throw std::runtime_error(err);
  }
  size_t host_bytes() const override { return cache_.bytes(); }
  const char* kind() const override { return "ssd"; }

 private:
  struct Waiter { std::mutex mu; std::condition_variable cv; int pending = 0, failed = 0; std::string err; };
  struct Job { int64_t row; uint32_t e; uint8_t* dst; Waiter* w; };
  void push(std::vector<Job>& jobs, bool urgent) {
    if (jobs.empty()) return;
    { std::lock_guard<std::mutex> lk(mu_); for (auto& j : jobs) { if (urgent) q_.push_front(j); else q_.push_back(j); } }
    cv_.notify_all();
  }
  void worker() {
    std::vector<uint8_t> buf_store(8192 + 4096);
    uint8_t* buf = reinterpret_cast<uint8_t*>(((uintptr_t)buf_store.data() + 4095) & ~(uintptr_t)4095);
    for (;;) {
      Job j;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
        if (stop_ && q_.empty()) return;
        j = q_.front(); q_.pop_front();
      }
      std::string err;
      uint8_t tmp[kPleHeadDim];
      const bool ok = read_row(j.row, j.e == EngramRowCache::NONE ? j.dst : tmp, buf, err);
      if (j.e == EngramRowCache::NONE) {
        std::lock_guard<std::mutex> lk(j.w->mu);
        if (!ok) { j.w->failed++; if (j.w->err.empty()) j.w->err = err; }
        if (--j.w->pending == 0) j.w->cv.notify_all();
        continue;
      }
      EngramRowCache::Shard& S = cache_.shard_of((uint64_t)j.row);
      std::lock_guard<std::mutex> lk(S.mu);
      EngramRowCache::Entry& E = cache_.e(j.e);
      if (ok) { std::memcpy(cache_.data(j.e), tmp, kPleHeadDim); E.state = EngramRowCache::READY; }
      else { E.state = EngramRowCache::FAILED; if (E.pins == 0) cache_.erase(S, j.e); }
      S.cv.notify_all();
    }
  }
  bool read_row(int64_t r, uint8_t* dst, uint8_t* buf, std::string& err) {
    const size_t si = sh_.shard_of(r);
    const uint64_t off = sh_.off[si] + (uint64_t)(r - sh_.start[si]) * kPleHeadDim;
    for (int attempt = 0; attempt <= 3; ++attempt) {
      const uint64_t al = direct_ ? align_.load() : 1;
      const uint64_t a = off / al * al, len = (off + kPleHeadDim - a + al - 1) / al * al;
      const ssize_t got = ::pread(fd_, buf, len, (off_t)a);
      if (got >= 0 && (uint64_t)got >= off + kPleHeadDim - a) { std::memcpy(dst, buf + (off - a), kPleHeadDim); return true; }
      if (got < 0 && errno == EINVAL && direct_ && align_ < 4096) { align_ = 4096; --attempt; continue; }  // 4 KiB logical blocks
      err = got < 0 ? std::string(strerror(errno)) : "short read";
      std::this_thread::sleep_for(std::chrono::microseconds(attempt == 0 ? 200 : attempt == 1 ? 2000 : 20000));
    }
    return false;
  }
  Shards sh_;
  int fd_ = -1;
  bool direct_ = false;
  std::atomic<uint64_t> align_{512};
  EngramRowCache cache_;
  std::atomic<uint32_t> epoch_{1};
  std::mutex mu_; std::condition_variable cv_;
  std::deque<Job> q_;
  bool stop_ = false;
  std::vector<std::thread> workers_;
  uint64_t hit_ = 0;
};

int64_t i64_scalar(const TensorInfo& t, size_t i) {
  HIVE_CHECK(t.dtype == "I64" && (size_t)t.numel() > i, "PLE: I64 buffer expected: " + t.name);
  int64_t v; std::memcpy(&v, t.data + i * 8, 8); return v;
}

}  // namespace

std::unique_ptr<PleTable> ple_open(Checkpoint& ck, const std::string& prefix, PleHash& hash, float& weight_scale) {
  const std::string e = prefix + "ple_embedding.";
  const TensorInfo& lm = ck.get(e + "layer_multipliers");
  const TensorInfo& vs = ck.get(e + "ngram_heads_vocab_sizes");
  const TensorInfo& vo = ck.get(e + "ngram_heads_offsets");
  HIVE_CHECK(lm.numel() == hash.ngram && vs.numel() == kPleHeads && vo.numel() == kPleHeads, "PLE: hash buffer shapes");
  hash.mult.resize(hash.ngram); hash.size.resize(kPleHeads); hash.offset.resize(kPleHeads);
  for (int i = 0; i < hash.ngram; ++i) hash.mult[i] = i64_scalar(lm, i);
  int64_t top = 0;
  for (int j = 0; j < kPleHeads; ++j) {
    hash.size[j] = i64_scalar(vs, j); hash.offset[j] = i64_scalar(vo, j);
    HIVE_CHECK(hash.size[j] > 0 && hash.offset[j] >= 0, "PLE: bad head size / offset");
    top = std::max(top, hash.offset[j] + hash.size[j]);
  }
  // token ids are < vocab, so x·m fits int64 when m ≤ INT64_MAX / vocab (the bound the multipliers are drawn under)
  HIVE_CHECK(hash.vocab > 0, "PLE: vocabulary size not set");
  for (int64_t m : hash.mult) HIVE_CHECK(m > 0 && m <= INT64_MAX / hash.vocab, "PLE: multiplier outside the no-overflow bound");
  {
    const TensorInfo& ws = ck.get(e + "ngram_embedding.weight_scale");
    HIVE_CHECK(ws.dtype == "BF16" && ws.numel() == 1, "PLE: weight_scale must be one BF16 value");
    weight_scale = bf2f(*reinterpret_cast<const bf16*>(ws.data));
  }
  Shards s;
  for (int i = 0;; ++i) {
    const std::string n = e + "ngram_embedding.shard_" + std::to_string(i) + ".weight";
    if (!ck.has(n)) break;
    const TensorInfo& t = ck.get(n);
    HIVE_CHECK(t.dtype == "F8_E4M3" && t.shape.size() == 2 && t.shape[1] == kPleHeadDim, "PLE: shard must be FP8 [rows, 160]: " + n);
    s.t.push_back(&t); s.off.push_back(t.offset); s.start.push_back(s.rows); s.rows += t.shape[0];
  }
  HIVE_CHECK(!s.t.empty(), "PLE: no ngram_embedding shards under " + e);
  for (size_t i = 0; i + 1 < s.t.size(); ++i) HIVE_CHECK(s.t[i]->shape[0] == s.t[0]->shape[0], "PLE: shards of unequal size (only the last may be shorter)");
  HIVE_CHECK(top <= s.rows, "PLE: hash range exceeds the table rows");
  hash.rows = s.rows;
  const char* mode = getenv("HIVE_QWEN_PLE");
  const bool ssd = mode && (std::string(mode) == "ssd" || std::string(mode) == "1");
  if (!ssd) return std::make_unique<PleRam>(s);
  // every shard must sit in one file for the SSD reader (the NVIDIA checkpoint keeps them in model-fp8-mtp-ple.safetensors)
  const std::string path = ck.shard_for(e + "ngram_embedding.shard_0.weight").path();
  for (size_t i = 1; i < s.t.size(); ++i)
    HIVE_CHECK(ck.shard_for(e + "ngram_embedding.shard_" + std::to_string(i) + ".weight").path() == path, "PLE ssd: shards in more than one file");
  const char* cm = getenv("HIVE_QWEN_PLE_SSD_CACHE_MB");
  const char* th = getenv("HIVE_QWEN_PLE_SSD_THREADS");
  const size_t cache = (size_t)((cm && *cm ? std::max(0.0, atof(cm)) : 2048.0) * 1048576.0);
  const int threads = th && *th && atoi(th) > 0 ? std::min(atoi(th), 256) : 64;
  return std::make_unique<PleSsd>(s, path, cache, threads);
}

}  // namespace hive::qwen
