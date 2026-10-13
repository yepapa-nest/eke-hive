// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// DeepSeek-V4.1-Flash CPU expert kernels in the ExpertStore::Pool shape — pool microbench of gemv_e2m1_rows_multi variants (ref / v2 / v3 tiled / v4 register): 2 nodes × T pinned threads, node-local half
//   records (w13 1152 rows × K 5120, w2 2560 rows × K 2304, e8m0/32 scales), 32-row items like ExpertStore::Pool; interleaved, median of 3.
//   (Phase 2 here does not wait for phase 1 per job as the pool does — a kernel comparison, not a pool model.) Node CPU lists: 3975WX layout.
//   g++ -std=c++20 -O3 -mavx2 -mfma -mf16c -pthread -Itools/cpu_fake/include -Iengine/include engine/tests/bench_e2m1_pool_cpu.cpp \
//       engine/src/cpu/expert_cpu.cpp -o /tmp/bep && /tmp/bep 300 16      (args: experts-per-node threads-per-node)
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <thread>
#include <vector>
#include "hive/expert_cpu.h"
using clk = std::chrono::steady_clock;
static void pin_node(int node) {
  cpu_set_t cs; CPU_ZERO(&cs);
  for (int c = 0; c < 16; ++c) { CPU_SET(node * 16 + c, &cs); CPU_SET(32 + node * 16 + c, &cs); }
  pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
}
struct Rec { uint8_t *w1, *s1, *w3, *s3, *w2, *s2; };
using Fn = void (*)(const uint8_t*, const uint8_t*, int, int, const float* const*, const float* const*, int, int, int, float* const*);
int main(int argc, char** argv) {
  const int H = 5120, I = 2304, NE = argc > 1 ? atoi(argv[1]) : 300, T = argc > 2 ? atoi(argv[2]) : 16;
  const int Ih = I / 2, Hh = H / 2;
  const size_t b13 = (size_t)Ih * H / 2, s13 = (size_t)Ih * H / 32, b2 = (size_t)Hh * I / 2, s2 = (size_t)Hh * I / 32;
  const size_t rec = 2 * (b13 + s13) + b2 + s2;
  std::vector<uint8_t*> mem(2);
  std::vector<std::vector<Rec>> recs(2, std::vector<Rec>(NE));
  for (int k = 0; k < 2; ++k)
    std::thread([&, k] {
      pin_node(k);
      uint8_t* m = (uint8_t*)aligned_alloc(4096, (rec * NE + 4095) / 4096 * 4096);
      std::mt19937 rng(k + 1);
      for (size_t i = 0; i + 4 <= rec * NE; i += 4) { uint32_t v = rng(); memcpy(m + i, &v, 4); }
      for (int e = 0; e < NE; ++e) {
        uint8_t* r = m + rec * e; Rec& R = recs[k][e];
        R.w1 = r; R.s1 = r + b13; R.w3 = R.s1 + s13; R.s3 = R.w3 + b13; R.w2 = R.s3 + s13; R.s2 = R.w2 + b2;
        for (size_t i = 0; i < s13; ++i) { R.s1[i] = 117 + R.s1[i] % 6; R.s3[i] = 117 + R.s3[i] % 6; }
        for (size_t i = 0; i < s2; ++i) R.s2[i] = 117 + R.s2[i] % 6;
      }
      mem[k] = m;
    }).join();
  std::vector<float> X(64 * (size_t)H), SX(64 * (size_t)H / 32), Y(64 * (size_t)I), SY(64 * (size_t)I / 32);
  std::mt19937 rng(9); std::normal_distribution<float> nd;
  for (auto& v : X) v = nd(rng); for (auto& v : Y) v = nd(rng);
  for (auto& v : SX) v = 0.01f; for (auto& v : SY) v = 0.02f;
  std::vector<float> G(64 * (size_t)I), U(64 * (size_t)I), O(64 * (size_t)H);
  std::atomic<long> gen{0}; std::atomic<int> next[2]; std::atomic<int> pend{0}; int ntask[2] = {0, 0};
  std::function<void(int, int)> task; std::atomic<bool> stop{false};
  std::vector<std::thread> W;
  for (int k = 0; k < 2; ++k)
    for (int t = 0; t < T; ++t)
      W.emplace_back([&, k] {
        pin_node(k); long seen = 0;
        while (!stop) {
          long g; while ((g = gen.load(std::memory_order_acquire)) == seen && !stop) _mm_pause();
          if (stop) return; seen = g;
          for (;;) { int i = next[k].fetch_add(1); if (i >= ntask[k]) break; task(k, i); pend.fetch_sub(1); }
        }
      });
  auto run = [&](int n, const std::function<void(int, int)>& f) {
    task = f; ntask[0] = ntask[1] = n; next[0] = 0; next[1] = 0; pend = 2 * n; gen.fetch_add(1, std::memory_order_release);
    while (pend.load(std::memory_order_acquire) > 0) _mm_pause();
  };
  Fn fns[4] = {hive::cpu::gemv_e2m1_rows_multi_ref, hive::cpu::gemv_e2m1_rows_multi_v2, hive::cpu::gemv_e2m1_rows_multi_v3, hive::cpu::gemv_e2m1_rows_multi_v4};
  const char* names[4] = {"ref", "v2", "v3tile", "v4reg"};
  int ecur = 0;
  for (int R : {2, 3, 4, 5, 8})
    for (int J : {1, 2, 4, 8}) {
      if (J * R > 64) continue;
      std::vector<double> res[4];
      for (int round = 0; round < 3; ++round)
        for (int kk = 0; kk < 4; ++kk) {
          Fn gemv = fns[kk];
          double tp = 0; long bytes = 0;
          const auto Tstart = clk::now();
          while (std::chrono::duration<double>(clk::now() - Tstart).count() < 0.25) {
            std::vector<int> ex(J); for (auto& e : ex) { e = ecur; ecur = (ecur + 7) % NE; }
            auto phase = [&](int rows, bool p1) {
              const int CH = 32, chunks = (rows + CH - 1) / CH;
              run(J * chunks, [&](int k, int t) {
                const int j = t / chunks, c = t % chunks, n0 = c * CH, n1 = std::min(rows, n0 + CH);
                const Rec& rc = recs[k][ex[j]];
                const float* a[8]; const float* sa[8]; float* y1[8]; float* y2[8];
                for (int r = 0; r < R; ++r) {
                  const int row = j * R + r;
                  if (p1) { a[r] = X.data() + (size_t)row * H; sa[r] = SX.data() + (size_t)row * H / 32; y1[r] = G.data() + (size_t)row * I + k * Ih; y2[r] = U.data() + (size_t)row * I + k * Ih; }
                  else { a[r] = Y.data() + (size_t)row * I; sa[r] = SY.data() + (size_t)row * I / 32; y1[r] = O.data() + (size_t)row * H + k * Hh; }
                }
                if (p1) { gemv(rc.w1, rc.s1, Ih, H, a, sa, R, n0, n1, y1); gemv(rc.w3, rc.s3, Ih, H, a, sa, R, n0, n1, y2); }
                else gemv(rc.w2, rc.s2, Hh, I, a, sa, R, n0, n1, y1);
              });
            };
            const auto t0 = clk::now(); phase(Ih, true); phase(Hh, false);
            tp += std::chrono::duration<double>(clk::now() - t0).count();
            bytes += (long)J * 2 * rec;
          }
          res[kk].push_back(bytes / tp / 1e9);
        }
      printf("DS H%d I%d T%d R=%d J=%d", H, I, T, R, J);
      for (int kk = 0; kk < 4; ++kk) { std::sort(res[kk].begin(), res[kk].end()); printf("  %s %.1f [%.1f-%.1f]", names[kk], res[kk][1], res[kk][0], res[kk][2]); }
      printf("\n"); fflush(stdout);
    }
  stop = true; gen.fetch_add(1);
  for (auto& t : W) t.join();
}
