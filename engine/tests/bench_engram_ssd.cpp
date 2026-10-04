// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Measurement behind the HIVE_ENGRAM_SSD design — IOPS and latency of random row reads from an engram table file (O_DIRECT pread, threads = queue depth).
//   Read-only; O_DIRECT (does not grow the page cache — a running hived holds the RAM); total read = reads x read size (a few GB by default).
//   Usage: bench_engram_ssd FILE TENSOR_OFFSET ROWS ROW_BYTES READS THREADS[,THREADS...] [MODE ...]
//     MODE: row = smallest 512-aligned range covering the row (256 B row -> 512 or 1024 B); 4k = 4 KiB-aligned block covering the row (8 KiB if it straddles a boundary); 16k = 16 KiB block
//   Output per mode and thread count: IOPS, MB/s, latency p50/p90/p99/max (µs), boundary-straddle ratio
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "hive/clock.h"

static double now_us() {
  using Micros = std::chrono::duration<double, std::micro>;
  return Micros(hive::SteadyClock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr, "usage: %s FILE TENSOR_OFFSET ROWS ROW_BYTES READS THREADS[,..] [row|4k|16k ...]\n", argv[0]);
    return 2;
  }
  const char* path = argv[1];
  const uint64_t base = strtoull(argv[2], nullptr, 10), rows = strtoull(argv[3], nullptr, 10), rb = strtoull(argv[4], nullptr, 10);
  const int reads = atoi(argv[5]);
  std::vector<int> tl;
  for (char* s = argv[6]; *s;) { tl.push_back(atoi(s)); char* c = strchr(s, ','); if (!c) break; s = c + 1; }
  std::vector<std::string> modes;
  for (int i = 7; i < argc; ++i) modes.push_back(argv[i]);
  if (modes.empty()) modes = {"row", "4k"};
  int fd = open(path, O_RDONLY | O_DIRECT);
  if (fd < 0) { perror("open O_DIRECT"); return 1; }
  std::mt19937_64 rng(12345);
  for (const auto& mode : modes) {
    const uint64_t align = mode == "row" ? 512 : mode == "4k" ? 4096 : 16384;
    for (int T : tl) {
      std::vector<uint64_t> off(reads), len(reads);
      int straddle = 0;
      for (int i = 0; i < reads; ++i) {
        const uint64_t r = rng() % rows, s = base + r * rb, e = s + rb;
        const uint64_t a = s / align * align, b = (e + align - 1) / align * align;
        off[i] = a; len[i] = b - a;
        straddle += (b - a) > align;
      }
      std::vector<double> lat(reads);
      std::atomic<int> next{0};
      std::atomic<long> bytes{0};
      std::atomic<int> errs{0};
      const double t0 = now_us();
      std::vector<std::thread> th;
      for (int t = 0; t < T; ++t)
        th.emplace_back([&] {
          void* buf = nullptr;
          if (posix_memalign(&buf, 4096, 1 << 16)) return;
          for (int i; (i = next.fetch_add(1)) < reads;) {
            const double a = now_us();
            ssize_t r = pread(fd, buf, len[i], (off_t)off[i]);
            lat[i] = now_us() - a;
            if (r != (ssize_t)len[i]) errs++;
            else bytes += r;
          }
          free(buf);
        });
      for (auto& x : th) x.join();
      const double sec = (now_us() - t0) / 1e6;
      std::sort(lat.begin(), lat.end());
      auto pct = [&](double p) { return lat[std::min<size_t>(lat.size() - 1, (size_t)(p * lat.size()))]; };
      printf("mode %-4s threads %3d reads %d: %.0f IOPS · %.0f MB/s · lat p50 %.0f p90 %.0f p99 %.0f max %.0f us · straddle %.1f%% · err %d\n", mode.c_str(), T,
             reads, reads / sec, bytes / sec / 1e6, pct(0.5), pct(0.9), pct(0.99), lat.back(), 100.0 * straddle / reads, errs.load());
      fflush(stdout);
    }
  }
  close(fd);
  return 0;
}
