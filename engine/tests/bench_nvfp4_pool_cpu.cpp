// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// NVFP4 CPU expert kernels in the GlmExperts::cpu_experts shape (GLM-5.3-Flash and Qwen3.8-Flash-Next VRAM-cache misses): 2 NUMA nodes ×
//   T worker threads pinned per node, node-local half records (first touch by a thread of that node — like build_halves), per "MoE layer"
//   J experts × R activation rows: phase 1 gate+up [I/2 rows, K = H] then phase 2 down [H/2 rows, K = I], chunks of 128 rows halved down
//   to 32 until every worker of a node has one (HIVE_GLM_CPU_ADAPT). Kernels _ref and _v2 interleaved in one process (3 rounds × 0.25 s,
//   median + range) — separate runs on a shared machine differed by up to ±25 %, so only numbers from one run are compared.
//   Prints the read ceiling of the same threads over the same memory first. GB/s = weight bytes (incl. scales) per second.
//   Node CPU lists assume the 3975WX layout (node k = cores 16k..16k+15 and their SMT siblings 32+16k..).
//   g++ -std=c++20 -O3 -mavx2 -mfma -mf16c -pthread -Itools/cpu_fake/include -Iengine/include engine/tests/bench_nvfp4_pool_cpu.cpp \
//       engine/src/cpu/expert_cpu_nvfp4.cpp -o /tmp/bnp && /tmp/bnp 4096 2048 400 16   (GLM)   ·   /tmp/bnp 2560 640 2000 16   (FN)
//   args: hidden inter experts-per-node threads-per-node
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
int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s hidden inter experts [threads/node]\n", argv[0]); return 2; }
  const int H = atoi(argv[1]), I = atoi(argv[2]), NE = atoi(argv[3]);  // hidden, inter, experts in pool (per node half records)
  const int T = argc > 4 ? atoi(argv[4]) : 16;                          // threads per node
  using Fn = void (*)(const uint8_t*, const uint8_t*, float, int, int, const float* const*, int, int, int, float* const*);
  Fn gemv = nullptr;
  const int Ih = I / 2, Hh = H / 2;
  const size_t b13 = (size_t)Ih * H / 2, s13 = (size_t)Ih * H / 16, b2 = (size_t)Hh * I / 2, s2 = (size_t)Hh * I / 16;
  const size_t rec = 2 * (b13 + s13) + b2 + s2;
  std::vector<uint8_t*> mem(2);
  std::vector<std::vector<Rec>> recs(2, std::vector<Rec>(NE));
  for (int k = 0; k < 2; ++k) {
    std::thread([&, k] {
      pin_node(k);
      uint8_t* m = (uint8_t*)aligned_alloc(4096, (rec * NE + 4095) / 4096 * 4096);
      std::mt19937 rng(k + 1);
      for (size_t i = 0; i < rec * NE; i += 4) { uint32_t v = rng(); memcpy(m + i, &v, std::min<size_t>(4, rec * NE - i)); }
      for (int e = 0; e < NE; ++e) {
        uint8_t* r = m + rec * e;
        Rec& R = recs[k][e];
        R.w1 = r; R.s1 = r + b13; R.w3 = R.s1 + s13; R.s3 = R.w3 + b13; R.w2 = R.s3 + s13; R.s2 = R.w2 + b2;
        for (size_t i = 0; i < s13; ++i) { R.s1[i] = 0x28 + (R.s1[i] & 15); R.s3[i] = 0x28 + (R.s3[i] & 15); }
        for (size_t i = 0; i < s2; ++i) R.s2[i] = 0x28 + (R.s2[i] & 15);
      }
      mem[k] = m;
    }).join();
  }
  // activations
  std::vector<float> X(64 * (size_t)H), Y(64 * (size_t)I);
  std::mt19937 rng(9); std::normal_distribution<float> nd;
  for (auto& v : X) v = nd(rng);
  for (auto& v : Y) v = nd(rng);
  std::vector<float> G(64 * (size_t)I), U(64 * (size_t)I), O(64 * (size_t)H);
  // pool
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
  // read ceiling: each node's threads stream its own memory
  {
    const size_t chunk = 1 << 20, nch = rec * NE / chunk;
    std::atomic<double> sink{0};
    for (int rep = 0; rep < 2; ++rep) {
      const auto t0 = clk::now();
      run((int)nch, [&](int k, int i) {
        const __m256* p = (const __m256*)(mem[k] + (size_t)i * chunk);
        __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
        for (size_t j = 0; j < chunk / 32; j += 4) { a = _mm256_add_ps(a, p[j]); b = _mm256_add_ps(b, p[j + 1]); c = _mm256_add_ps(c, p[j + 2]); d = _mm256_add_ps(d, p[j + 3]); }
        float o[8]; _mm256_storeu_ps(o, _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d))); if (o[0] == 12345.f) sink = sink + 1;
      });
      const double s = std::chrono::duration<double>(clk::now() - t0).count();
      if (rep) printf("read ceiling (both nodes, %d thr/node): %.1f GB/s\n", T, 2.0 * nch * chunk / s / 1e9);
    }
  }
  // expert layers: J jobs of R rows each, experts drawn round robin (cold). Kernels interleaved: 3 rounds × (ref, v2), 0.25 s each, median.
  int ecur = 0;
  constexpr int NK = 2;
  Fn fns[NK] = {hive::cpu::gemv_nvfp4_rows_multi_ref, hive::cpu::gemv_nvfp4_rows_multi_v2};
  const char* names[NK] = {"ref", "v2"};
  for (int R : {1, 2, 3, 4, 5, 6, 8})
    for (int J : {1, 2, 4, 8}) {
      std::vector<double> res[NK];
      for (int round = 0; round < 3; ++round)
        for (int kk = 0; kk < NK; ++kk) {
          gemv = fns[kk];
          double tp = 0; long bytes = 0;
          const auto Tstart = clk::now();
          while (std::chrono::duration<double>(clk::now() - Tstart).count() < 0.25) {
            std::vector<int> ex(J); for (auto& e : ex) { e = ecur; ecur = (ecur + 7) % NE; }
            auto phase = [&](int rows, bool p1) {
              int CH = 128; while (CH > 32 && J * ((rows + CH - 1) / CH) < T) CH /= 2;
              const int chunks = (rows + CH - 1) / CH;
              run(J * chunks, [&](int k, int t) {
                const int j = t / chunks, c = t % chunks, n0 = c * CH, n1 = std::min(rows, n0 + CH);
                const Rec& rc = recs[k][ex[j]];
                const float* a[8]; float* y1[8]; float* y2[8];
                for (int r = 0; r < R; ++r) {
                  const int row = j * R + r;
                  if (p1) { a[r] = X.data() + (size_t)row * H; y1[r] = G.data() + (size_t)row * I + k * Ih; y2[r] = U.data() + (size_t)row * I + k * Ih; }
                  else { a[r] = Y.data() + (size_t)row * I; y1[r] = O.data() + (size_t)row * H + k * Hh; }
                }
                if (p1) { gemv(rc.w1, rc.s1, 0.01f, Ih, H, a, R, n0, n1, y1); gemv(rc.w3, rc.s3, 0.01f, Ih, H, a, R, n0, n1, y2); }
                else gemv(rc.w2, rc.s2, 0.01f, Hh, I, a, R, n0, n1, y1);
              });
            };
            const auto t0 = clk::now(); phase(Ih, true); phase(Hh, false);
            tp += std::chrono::duration<double>(clk::now() - t0).count();
            bytes += (long)J * 2 * rec;
          }
          res[kk].push_back(bytes / tp / 1e9);
        }
      printf("H%d I%d T%d R=%d J=%d GB/s median of 3:", H, I, T, R, J);
      for (int kk = 0; kk < NK; ++kk) { std::sort(res[kk].begin(), res[kk].end()); printf("  %s %.1f [%.1f-%.1f]", names[kk], res[kk][1], res[kk][0], res[kk][2]); }
      printf("\n"); fflush(stdout);
    }
  stop = true; gen.fetch_add(1);
  for (auto& t : W) t.join();
}
