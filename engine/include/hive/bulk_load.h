// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Faster cold load: parallel O_DIRECT bulk loading (HIVE_LOAD_PAR). No CUDA, header-only (so the fake-CUDA CPU test
// builds do not need to change their source lists).
//
// Measured basis:
//  · A 478.3 s startup (`ready in 478.3s`, per-stage lines in hived.log) broke down as: experts, 43 layers, 224.8 s total
//    (1.4–10.2 s per layer; disk-bound layers ≈1.0 GB/s — page-fault reads through mmap+memcpy, read_ahead_kb 128) ·
//    pinned registration 10.5 s · two engram tables 48.0 s + 125.3 s (buffered pread, 16 threads — 0.5–6.2 GB/s, varying
//    between startups) · warm start 3.1 s · the rest (dense model, runtime, session pool; no timestamps in the log) ≈ 66.6 s.
//  · All weights live on a single NVMe (ext4, PCIe4 x4). O_DIRECT dd: 1 stream bs64M 6.2 GB/s · 4 streams 16 GiB in 2.42 s
//    (7.1 GB/s) · 6 streams 18 GiB in 2.63 s (7.3 GB/s). Bytes to read = experts 296.0 GB + engram tables 202.8 GB + rest 11.5 GB.
//    → disk lower bound ≈ 510 GB / 7.2 GB/s ≈ 71 s (estimate).
//  · The buffered path fills the page cache with the 508 GB checkpoint and then drops it with release_all (log line
//    `checkpoint page cache released: 508 GB`). Combined with ~476 GB of pinned memory this triggers reclaim (kswapd), and
//    one startup saw engram table reads drop to 0.5 GB/s (`engram layer 1 … 209.6s (0.5 GB/s)`).
//
// Design: sort the destination segment list (file, offset, length, destination pointer) by file and offset, merge it into
//  contiguous extents, cut those into 4 KiB-aligned chunks; threads O_DIRECT-pread each chunk into an aligned bounce buffer and
//  memcpy only the overlapping segments to their destinations. The resulting bytes are identical to the non-parallel paths
//  (mmap memcpy · buffered pread) — the same file bytes land at the same destination offsets (tools/test_load_par_cpu.py
//  compares bit-for-bit against a real ExpertStore).
//  Files that cannot be opened with O_DIRECT (tmpfs etc.) fall back to buffered pread for that file only. Read errors and
//  truncated files return false — the caller reloads from scratch via the non-parallel path (not a new guard: a failure is
//  absorbed by the fallback path rather than turned into a rejection).
#pragma once
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "hive/clock.h"
#include "nlohmann/json.hpp"

namespace hive {

struct LoadSeg {
  int file = 0;        // index into paths
  uint64_t off = 0;    // absolute offset within the file
  uint64_t len = 0;
  uint8_t* dst = nullptr;
};

struct BulkLoadOpts {
  int threads = 8;                 // measured: the disk saturates at 4–6 streams (7.1–7.3 GB/s); 8 so the copy work overlaps
  size_t chunk = 32ull << 20;      // size of one pread (rounded to a multiple of 4 KiB)
  bool direct = true;              // false = buffered pread (HIVE_LOAD_BUFFERED)
  size_t merge_gap = 1ull << 20;   // segments separated by a gap no larger than this are read as one extent
};

struct BulkLoadStats {
  size_t dst_bytes = 0, read_bytes = 0;
  double sec = 0;
  int files = 0, direct_files = 0, jobs = 0, threads = 0;
  std::string err;
};

// ---- Switches (convention: unset, "" or "0" = off = non-parallel path) ----
// HIVE_LOAD_PAR: off = non-parallel path (per-layer mmap memcpy · engram buffered pread · after pinning). "1" (or any non-numeric
// "on" value) = 8 threads · N ≥ 2 = N threads (max 64).
inline int load_par_threads() {
  const char* v = getenv("HIVE_LOAD_PAR");
  if (!v || !*v || strcmp(v, "0") == 0) return 0;
  char* end = nullptr;
  const long n = strtol(v, &end, 10);
  if (end == v || *end != 0 || n <= 1) return 8;
  return (int)std::min<long>(n, 64);
}
// HIVE_LOAD_CHUNK_MB: size of one read in MiB (unset, non-numeric or out of range = 32 · valid range 1..1024).
inline size_t load_chunk_bytes() {
  const char* v = getenv("HIVE_LOAD_CHUNK_MB");
  long mb = 32;
  if (v && *v) { char* end = nullptr; const long n = strtol(v, &end, 10); if (end != v && *end == 0 && n >= 1 && n <= 1024) mb = n; }
  return (size_t)mb << 20;
}
// HIVE_LOAD_BUFFERED: on = buffered pread instead of O_DIRECT (still parallel — for comparison and as an emergency fallback). off = O_DIRECT.
inline bool load_buffered() { const char* v = getenv("HIVE_LOAD_BUFFERED"); return v && *v && strcmp(v, "0") != 0; }
// HIVE_LOAD_PREFAULT: off = default. "1" (or any non-numeric "on" value) = 16 threads · N ≥ 2 = N (max 64). See the HostPrefault header comment in expert_store.h.
inline int load_prefault_threads() {
  const char* v = getenv("HIVE_LOAD_PREFAULT");
  if (!v || !*v || strcmp(v, "0") == 0) return 0;
  char* end = nullptr;
  const long n = strtol(v, &end, 10);
  if (end == v || *end != 0 || n <= 1) return 16;
  return (int)std::min<long>(n, 64);
}
// HIVE_LOAD_CHECKSUM: on = log one hash line of the host copy right after loading (verification log, hived).
inline bool load_checksum_on() { const char* v = getenv("HIVE_LOAD_CHECKSUM"); return v && *v && strcmp(v, "0") != 0; }
inline BulkLoadOpts load_opts_from_env(int threads) {
  BulkLoadOpts o;
  o.threads = std::max(1, threads);
  o.chunk = load_chunk_bytes();
  o.direct = !load_buffered();
  return o;
}

namespace bulk_detail {
inline double now_s() { return hive::Seconds(hive::SteadyClock::now().time_since_epoch()).count(); }
constexpr uint64_t kAlign = 4096;
inline uint64_t down(uint64_t x) { return x / kAlign * kAlign; }
inline uint64_t up(uint64_t x) { return (x + kAlign - 1) / kAlign * kAlign; }
}  // namespace bulk_detail

// Copy [off, off+len) of paths[seg.file] to seg.dst. Returns true on success. On failure stats->err holds the reason (the
// destination may be partially filled — the caller reloads).
inline bool bulk_load(const std::vector<std::string>& paths, std::vector<LoadSeg> segs, const BulkLoadOpts& opt, BulkLoadStats* stats) {
  using namespace bulk_detail;
  BulkLoadStats st;
  const double t0 = now_s();
  st.files = (int)paths.size();
  std::vector<int> fds(paths.size(), -1);
  auto close_all = [&] { for (int fd : fds) if (fd >= 0) ::close(fd); };
  for (size_t i = 0; i < paths.size(); ++i) {
    int fd = -1;
    if (opt.direct) fd = ::open(paths[i].c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd >= 0) ++st.direct_files;
    else fd = ::open(paths[i].c_str(), O_RDONLY | O_CLOEXEC);  // filesystem without O_DIRECT → buffered for this file only (fallback)
    if (fd < 0) { st.err = "open " + paths[i] + ": " + strerror(errno); close_all(); if (stats) *stats = st; return false; }
    fds[i] = fd;
  }
  segs.erase(std::remove_if(segs.begin(), segs.end(), [](const LoadSeg& s) { return s.len == 0; }), segs.end());
  std::sort(segs.begin(), segs.end(), [](const LoadSeg& a, const LoadSeg& b) { return a.file != b.file ? a.file < b.file : a.off < b.off; });
  for (const auto& s : segs) st.dst_bytes += s.len;
  // contiguous extents → aligned chunk jobs
  struct Job { int file; uint64_t a, b; size_t s0, s1; };
  std::vector<Job> jobs;
  const uint64_t chunk = std::max<uint64_t>(kAlign, opt.chunk / kAlign * kAlign);
  for (size_t i = 0; i < segs.size();) {
    size_t j = i + 1;
    uint64_t end = segs[i].off + segs[i].len;
    while (j < segs.size() && segs[j].file == segs[i].file && segs[j].off <= end + opt.merge_gap) { end = std::max(end, segs[j].off + segs[j].len); ++j; }
    const uint64_t A = down(segs[i].off), B = up(end);
    for (uint64_t x = A; x < B; x += chunk) jobs.push_back({segs[i].file, x, std::min(x + chunk, B), i, j});
    i = j;
  }
  st.jobs = (int)jobs.size();
  const int nt = std::max(1, std::min<int>(opt.threads, (int)std::max<size_t>(1, jobs.size())));
  st.threads = nt;
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  std::atomic<uint64_t> read_bytes{0};
  std::vector<std::string> errs(nt);
  auto worker = [&](int t) {
    uint8_t* buf = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&buf), kAlign, chunk) != 0 || !buf) { errs[t] = "bounce buffer alloc"; failed = true; return; }
    for (;;) {
      if (failed.load(std::memory_order_relaxed)) break;
      const size_t k = next.fetch_add(1);
      if (k >= jobs.size()) break;
      const Job& J = jobs[k];
      const uint64_t want = J.b - J.a;
      uint64_t got = 0;
      while (got < want) {
        const ssize_t r = pread(fds[J.file], buf + got, want - got, (off_t)(J.a + got));
        if (r < 0) {
          if (errno == EINTR) continue;
          errs[t] = "pread " + paths[J.file] + " @" + std::to_string(J.a + got) + ": " + strerror(errno);
          failed = true;
          break;
        }
        if (r == 0) break;  // end of file (tail extended by alignment)
        got += (uint64_t)r;
        if ((uint64_t)r % kAlign) break;  // short O_DIRECT read = end of file
      }
      if (failed.load(std::memory_order_relaxed)) break;
      read_bytes += got;
      const uint64_t have = J.a + got;
      for (size_t s = J.s0; s < J.s1; ++s) {
        const LoadSeg& g = segs[s];
        const uint64_t ge = g.off + g.len;
        if (g.off >= J.b) break;  // segments are sorted by off
        if (ge <= J.a) continue;
        const uint64_t lo = std::max(g.off, J.a), need_hi = std::min(ge, J.b), hi = std::min(ge, have);
        if (hi < need_hi) {  // file is shorter than the segment (truncated file)
          errs[t] = "short read " + paths[J.file] + " @" + std::to_string(have) + " < " + std::to_string(need_hi);
          failed = true;
          break;
        }
        memcpy(g.dst + (lo - g.off), buf + (lo - J.a), hi - lo);
      }
    }
    free(buf);
  };
  std::vector<std::thread> th;
  for (int t = 0; t < nt; ++t) th.emplace_back(worker, t);
  for (auto& x : th) x.join();
  close_all();
  st.read_bytes = read_bytes.load();
  st.sec = now_s() - t0;
  for (auto& e : errs) if (!e.empty()) { st.err = e; break; }
  if (stats) *stats = st;
  return !failed.load();
}

// ---- Dense prefetch (hived starts this before constructing Model when HIVE_LOAD_PAR is on) ----
// Model reads dense, embedding, head, MTP dense and engram q/k tensors from mmap one tensor at a time with cudaMemcpy/memcpy
// (page-fault reads). A few threads first pull those tensors (everything except experts and engram tables, measured 11.5 GB)
// into the page cache with large buffered preads — the data path itself is unchanged (mmap), so the bytes are identical.
// All failures are ignored (it is only a hint). The page cache is still dropped by release_all (Checkpoint opens those shards).
inline bool dense_tensor_name(const std::string& k, int max_layer) {
  if (k.find(".ffn.experts.") != std::string::npos || k.find(".engram.embed.") != std::string::npos) return false;
  if (k.rfind("embed.", 0) == 0 || k.rfind("head.", 0) == 0 || k.rfind("norm.", 0) == 0 || k.rfind("mtp.", 0) == 0) return true;
  if (k.rfind("layers.", 0) == 0) {
    if (max_layer < 0) return true;
    return atoi(k.c_str() + 7) <= max_layer;
  }
  return false;  // vision.* · aligner.* are loaded lazily (first image) — not prefetched
}
struct DensePrefetch {
  std::thread th;
  std::atomic<uint64_t> bytes{0};
  double sec = 0;
  void start(const std::string& dir, int max_layer, int threads) {
    th = std::thread([this, dir, max_layer, threads] {
      const double t0 = bulk_detail::now_s();
      try {
        std::ifstream f(dir + "/model.safetensors.index.json");
        if (!f) return;
        nlohmann::json idx = nlohmann::json::parse(f);
        std::map<std::string, int> shard_of;
        for (auto& [k, v] : idx["weight_map"].items()) if (dense_tensor_name(k, max_layer)) shard_of[v.get<std::string>()] = 1;
        struct R { int fd; uint64_t a, b; };
        constexpr uint64_t kPiece = 8ull << 20;
        std::vector<R> rs;
        std::vector<int> fds;
        for (auto& [shard, _] : shard_of) {
          const int fd = ::open((dir + "/" + shard).c_str(), O_RDONLY | O_CLOEXEC);
          if (fd < 0) continue;
          fds.push_back(fd);
          uint64_t hl = 0;
          if (pread(fd, &hl, 8, 0) != 8 || hl > (64ull << 20)) continue;
          std::string h(hl, '\0');
          if (pread(fd, h.data(), hl, 8) != (ssize_t)hl) continue;
          nlohmann::json hdr = nlohmann::json::parse(h);
          std::vector<std::pair<uint64_t, uint64_t>> v;
          for (auto& [k, t] : hdr.items()) {
            if (k == "__metadata__" || !dense_tensor_name(k, max_layer)) continue;
            v.push_back({8 + hl + t["data_offsets"][0].get<uint64_t>(), 8 + hl + t["data_offsets"][1].get<uint64_t>()});
          }
          std::sort(v.begin(), v.end());
          for (size_t i = 0; i < v.size();) {
            uint64_t a = v[i].first, b = v[i].second;
            size_t j = i + 1;
            while (j < v.size() && v[j].first <= b + (1u << 20)) { b = std::max(b, v[j].second); ++j; }
            for (uint64_t x = a; x < b; x += kPiece) rs.push_back({fd, x, std::min<uint64_t>(b, x + kPiece)});
            i = j;
          }
        }
        std::atomic<size_t> next{0};
        std::vector<std::thread> ws;
        for (int t = 0; t < std::max(1, threads); ++t)
          ws.emplace_back([&] {
            std::vector<uint8_t> buf(8ull << 20);
            for (size_t k; (k = next.fetch_add(1)) < rs.size();) {
              const ssize_t r = pread(rs[k].fd, buf.data(), rs[k].b - rs[k].a, (off_t)rs[k].a);
              if (r > 0) bytes += (uint64_t)r;
            }
          });
        for (auto& w : ws) w.join();
        for (int fd : fds) ::close(fd);
      } catch (...) {
      }
      sec = bulk_detail::now_s() - t0;
    });
  }
  void join() { if (th.joinable()) th.join(); }
  ~DensePrefetch() { join(); }
};

}  // namespace hive
