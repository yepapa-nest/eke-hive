// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Measurement tool for the HIVE_LOAD_PAR cold load (CPU only — no GPU/CUDA, independent of a running server). Reads a few expert layers (+ the head of the engram
//   tables) of a real checkpoint with hive/bulk_load.h into fresh NUMA arenas (half records per layer on both nodes — the same first touch as the real load) and
//   measures GB/s. Verification = reading the same bytes separately with single-threaded O_DIRECT and comparing (no buffered reads — the page cache is not filled).
// Build (CPU-only): g++ -std=c++20 -O2 -pthread -Iengine/include -Iengine/third_party engine/tests/bench_load_par.cpp engine/src/safetensors.cpp -lnuma -o /tmp/bench_load_par
// Run: bench_load_par <ckpt dir> <first layer> <layer count> <engram GB (0 = skip)> <threads> <chunk MiB> [verify 0/1]
#include <numa.h>
#include <sys/mman.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hive/bulk_load.h"
#include "hive/clock.h"
#include "hive/safetensors.h"

using namespace hive;

static double now_s() { return hive::Seconds(hive::SteadyClock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  if (argc < 7) { fprintf(stderr, "usage: %s ckpt first_layer n_layers engram_gb threads chunk_mb [verify]\n", argv[0]); return 2; }
  const std::string dir = argv[1];
  const int l0 = atoi(argv[2]), nl = atoi(argv[3]);
  const double eg_gb = atof(argv[4]);
  BulkLoadOpts o;
  o.threads = atoi(argv[5]);
  o.chunk = (size_t)atoi(argv[6]) << 20;
  const bool verify = argc > 7 && atoi(argv[7]);
  Checkpoint ck(dir);
  std::vector<std::string> paths;
  std::map<std::string, int> idx;
  auto file_of = [&](const std::string& t) {
    const std::string& p = ck.shard_for(t).path();
    auto it = idx.find(p);
    if (it != idx.end()) return it->second;
    paths.push_back(p);
    return idx[p] = (int)paths.size() - 1;
  };
  // Real half-record layout (dim 4096 and inter 2048 are read from the checkpoint tensor sizes): each tensor in two halves -> node 0/1 arenas
  struct Piece { int file; uint64_t off, len; int node; size_t at; };
  std::vector<Piece> pieces;
  size_t node_bytes[2] = {0, 0};
  for (int l = l0; l < l0 + nl; ++l)
    for (int e = 0;; ++e) {
      const std::string p = "layers." + std::to_string(l) + ".ffn.experts." + std::to_string(e) + ".";
      if (!ck.has(p + "w1.weight")) break;
      for (const char* m : {"w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale"}) {
        const TensorInfo& t = ck.get(p + m);
        const int f = file_of(p + m);
        for (int n = 0; n < 2; ++n) { pieces.push_back({f, t.offset + n * (t.nbytes / 2), t.nbytes / 2, n, node_bytes[n]}); node_bytes[n] += t.nbytes / 2; }
      }
    }
  size_t eg_bytes = 0;
  if (eg_gb > 0) {
    const TensorInfo& w = ck.get("layers.1.engram.embed.weight");
    eg_bytes = std::min<size_t>(w.nbytes, (size_t)(eg_gb * 1e9));
    pieces.push_back({file_of("layers.1.engram.embed.weight"), w.offset, eg_bytes, 2, 0});
  }
  const bool numa = numa_available() >= 0 && numa_max_node() >= 1;
  uint8_t* base[3];
  const size_t sz[3] = {node_bytes[0], node_bytes[1], eg_bytes};
  for (int n = 0; n < 3; ++n) {
    base[n] = nullptr;
    if (!sz[n]) continue;
    base[n] = (uint8_t*)(numa ? numa_alloc_onnode(sz[n], n & 1) : mmap(nullptr, sz[n], PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (!getenv("BENCH_NOHUGE")) madvise(base[n], sz[n], MADV_HUGEPAGE);
  }
  if (const char* v = getenv("BENCH_PREFAULT")) {  // measure first page touch (allocation, zero fill, THP compaction) separately from reading
    const int nt = atoi(v);
    const double f0 = now_s();
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
      th.emplace_back([&, t] {
        for (int n = 0; n < 3; ++n)
          for (size_t off = (size_t)t * 4096; off < sz[n]; off += (size_t)nt * 4096) base[n][off] = 0;
      });
    for (auto& x : th) x.join();
    printf("prefault %.2f GB with %d threads in %.2f s = %.2f GB/s\n", (sz[0] + sz[1] + sz[2]) / 1e9, nt, now_s() - f0, (sz[0] + sz[1] + sz[2]) / 1e9 / (now_s() - f0));
  }
  std::vector<LoadSeg> segs;
  for (auto& p : pieces) segs.push_back({p.file, p.off, p.len, base[p.node] + p.at});
  BulkLoadStats st;
  const double t0 = now_s();
  const bool ok = bulk_load(paths, segs, o, &st);
  const double sec = now_s() - t0;
  printf("bulk_load %s · layers %d..%d + engram %.1f GB · dst %.2f GB read %.2f GB in %.2f s = %.2f GB/s · O_DIRECT %d/%d files · %d threads × %zu MiB · %d jobs%s%s\n",
         ok ? "ok" : "FAILED", l0, l0 + nl - 1, eg_bytes / 1e9, st.dst_bytes / 1e9, st.read_bytes / 1e9, sec, st.read_bytes / sec / 1e9, st.direct_files, st.files,
         st.threads, o.chunk >> 20, st.jobs, st.err.empty() ? "" : " · ", st.err.c_str());
  int bad = 0;
  if (ok && verify) {  // read separately with single-threaded O_DIRECT and compare
    const size_t B = 64ull << 20;
    uint8_t* buf = nullptr;
    if (posix_memalign((void**)&buf, 4096, B + 8192)) return 3;
    std::vector<int> fds;
    for (auto& p : paths) fds.push_back(open(p.c_str(), O_RDONLY | O_DIRECT));
    const double v0 = now_s();
    for (auto& p : pieces) {
      for (uint64_t done = 0; done < p.len;) {
        const uint64_t off = p.off + done, a = off / 4096 * 4096, n = std::min<uint64_t>(p.len - done, B);
        const uint64_t want = (off + n - a + 4095) / 4096 * 4096;
        uint64_t got = 0;
        while (got < want) { ssize_t r = pread(fds[p.file], buf + got, want - got, (off_t)(a + got)); if (r <= 0) break; got += r; if (r % 4096) break; }
        bad += got < off + n - a || memcmp(buf + (off - a), base[p.node] + p.at + done, n) != 0;
        done += n;
      }
    }
    printf("verify (independent single-thread O_DIRECT): %zu pieces · %d mismatch(es) · %.1f s\n", pieces.size(), bad, now_s() - v0);
    for (int fd : fds) close(fd);
    free(buf);
  }
  for (int n = 0; n < 3; ++n) if (base[n]) { if (numa) numa_free(base[n], sz[n]); else munmap(base[n], sz[n]); }
  return ok && bad == 0 ? 0 : 1;
}
