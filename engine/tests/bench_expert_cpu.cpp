// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_CPU_GEMV2 — CPU expert GEMV baseline (_ref) vs new version (_v2): output bit comparison on real shapes + ms per expert (single thread · T threads).
//   Runs on the CPU without a GPU (tools/test_cpu.sh runs the short `--quick` variant for the bit comparison only — timings are those of the machine it runs on, not the Zen2 3975WX target).
//   Build (one line): g++ -std=c++20 -O3 -mavx2 -mfma -mf16c -pthread -Itools/cpu_fake/include -Iengine/include engine/tests/bench_expert_cpu.cpp \
//                engine/src/cpu/expert_cpu.cpp -o /tmp/bench_expert_cpu      (same -O3 -mavx2 -mfma -mf16c as HIVE_CPU_FLAGS in the CMake build)
//   Run: bench_expert_cpu [--quick] [--threads T] [--experts E] [--reps N]
//   Shapes (DeepSeek-V4.1-Flash): dim 5120 · inter 2304 · expert = w1/w3 [2304, 5120/2] + w2 [5120, 2304/2] nibbles + e8m0 scales ≈ 18.8 MB.
//   The pool (expert_store.cpp) splits an expert into half records per node (w13 1152 rows · w2 2560 rows) processed as 32-row items — the threaded measurement uses the same 32-row items.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "hive/common.h"
#include "hive/expert_cpu.h"

using namespace hive;
using clk = std::chrono::steady_clock;

namespace {
constexpr int kDim = 5120, kInter = 2304;
struct Expert {
  std::vector<uint8_t> w1, s1, w3, s3, w2, s2;
  size_t bytes() const { return w1.size() + s1.size() + w3.size() + s3.size() + w2.size() + s2.size(); }
};
uint8_t rnd_scale(std::mt19937& rng, bool allow_zero) {  // near real checkpoint scales (2^-10 ~ 2^-5) + occasional e8m0 0 (subnormal 2^-127 — e8m0_fast's special path)
  std::uniform_int_distribution<int> d(0, 99);
  if (allow_zero && d(rng) == 0) return 0;
  return (uint8_t)(117 + d(rng) % 6);
}
void fill_expert(Expert& e, std::mt19937& rng, bool allow_zero) {
  std::uniform_int_distribution<int> byte(0, 255);
  e.w1.resize((size_t)kInter * kDim / 2); e.w3.resize(e.w1.size()); e.w2.resize((size_t)kDim * kInter / 2);
  e.s1.resize((size_t)kInter * kDim / 32); e.s3.resize(e.s1.size()); e.s2.resize((size_t)kDim * kInter / 32);
  for (auto* v : {&e.w1, &e.w3, &e.w2}) for (auto& x : *v) x = (uint8_t)byte(rng);
  for (auto* v : {&e.s1, &e.s3, &e.s2}) for (auto& x : *v) x = rnd_scale(rng, allow_zero);
}
std::vector<float> rnd_act(std::mt19937& rng, int K) {  // e4m3 values (including sign · 0 · -0)
  std::uniform_int_distribution<int> byte(0, 255);
  std::vector<float> a((size_t)K);
  for (auto& v : a) { uint8_t q; do { q = (uint8_t)byte(rng); } while ((q & 0x7F) == 0x7F); v = e4m3_to_f32(q); }
  return a;
}
std::vector<float> rnd_sa(std::mt19937& rng, int K) {
  std::vector<float> s((size_t)K / 32);
  for (auto& v : s) v = e8m0_to_f32(rnd_scale(rng, true));
  return s;
}
double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
std::string cpu_model() {
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  while (std::getline(f, line)) if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
  return "?";
}

// bit comparison: do _ref and _v2 produce memcmp-equal output for the same input?
int compare_rows(const uint8_t* W, const uint8_t* sw, int N, int K, const float* a, const float* sa, int n0, int n1) {
  std::vector<float> y0((size_t)N, 7.f), y1((size_t)N, 7.f);
  cpu::gemv_e2m1_rows_ref(W, sw, N, K, a, sa, n0, n1, y0.data());
  cpu::gemv_e2m1_rows_v2(W, sw, N, K, a, sa, n0, n1, y1.data());
  int bad = 0;
  for (int n = 0; n < N; ++n) if (std::memcmp(&y0[n], &y1[n], 4) != 0) ++bad;
  return bad;
}
int compare_multi(const uint8_t* W, const uint8_t* sw, int N, int K, const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& sa, int R,
                  int n0, int n1) {
  std::vector<std::vector<float>> y0(R, std::vector<float>((size_t)N, 7.f)), y1 = y0;
  std::vector<const float*> ap(R), sp(R);
  std::vector<float*> p0(R), p1(R);
  for (int r = 0; r < R; ++r) { ap[r] = a[r].data(); sp[r] = sa[r].data(); p0[r] = y0[r].data(); p1[r] = y1[r].data(); }
  cpu::gemv_e2m1_rows_multi_ref(W, sw, N, K, ap.data(), sp.data(), R, n0, n1, p0.data());
  cpu::gemv_e2m1_rows_multi_v2(W, sw, N, K, ap.data(), sp.data(), R, n0, n1, p1.data());
  int bad = 0;
  for (int r = 0; r < R; ++r) for (int n = 0; n < N; ++n) if (std::memcmp(&y0[r][n], &y1[r][n], 4) != 0) ++bad;
  // HIVE_CPU_MULTIROW2: _v3 (row groups × K tiles) is also bit-compared with _ref — rows not written (out of range) must stay 7
  std::vector<std::vector<float>> y2(R, std::vector<float>((size_t)N, 7.f));
  std::vector<float*> p2(R);
  for (int r = 0; r < R; ++r) p2[r] = y2[r].data();
  cpu::gemv_e2m1_rows_multi_v3(W, sw, N, K, ap.data(), sp.data(), R, n0, n1, p2.data());
  for (int r = 0; r < R; ++r) for (int n = 0; n < N; ++n) if (std::memcmp(&y0[r][n], &y2[r][n], 4) != 0) ++bad;
  return bad;
}

// one expert (R=1) with the same calls as the pool: w1·w3 [0, inter) · w2 [0, dim) (single thread = all rows · threaded = 32-row items)
using RowsFn = void (*)(const uint8_t*, const uint8_t*, int, int, const float*, const float*, int, int, float*);
void expert_rows(RowsFn fn, const Expert& e, const float* a, const float* sa, const float* yq, const float* sy, float* g, float* u, float* o, int r0, int r1,
                 int phase) {
  if (phase == 1) { fn(e.w1.data(), e.s1.data(), kInter, kDim, a, sa, r0, r1, g); fn(e.w3.data(), e.s3.data(), kInter, kDim, a, sa, r0, r1, u); }
  else fn(e.w2.data(), e.s2.data(), kDim, kInter, yq, sy, r0, r1, o);
}
}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  int n_exp = 24, reps = 3;
  int threads = std::max<unsigned>(std::thread::hardware_concurrency() / 2, 1u);
  for (int i = 1; i < argc; i++) {
    const std::string a(argv[i]);
    if (a == "--quick") quick = true;
    else if (a == "--threads" && i + 1 < argc) threads = std::max(1, atoi(argv[++i]));
    else if (a == "--experts" && i + 1 < argc) n_exp = std::max(2, atoi(argv[++i]));
    else if (a == "--reps" && i + 1 < argc) reps = std::max(1, atoi(argv[++i]));
  }
  std::mt19937 rng(20260930);
  int fails = 0;
  // ---- 1) bit comparison ----------------------------------------------------------------------------------------------
  {
    // (a) exhaustive over 16 nibble codes × 16 byte positions: one weight row cycles every byte value (0..255) through every position, activations = a different power of two per position (exact products → the comparison sees every single value)
    const int K = 32 * 16;  // 16 blocks · 256 rows = every byte value × every position (byte at position j of row v = (v + 17 j) & 255)
    std::vector<uint8_t> W((size_t)256 * K / 2), S((size_t)256 * K / 32, 127);
    for (int v = 0; v < 256; ++v) for (int j = 0; j < K / 2; ++j) W[(size_t)v * (K / 2) + j] = (uint8_t)((v + 17 * j) & 255);
    std::vector<float> a((size_t)K), sa((size_t)K / 32, 1.f);
    for (int k = 0; k < K; ++k) a[(size_t)k] = std::ldexp((k & 1) ? -1.f : 1.f, (k % 37) - 18);
    const int bad = compare_rows(W.data(), S.data(), 256, K, a.data(), sa.data(), 0, 256);
    printf("bench_expert_cpu: exhaustive nibbles (256 bytes x 16 positions) mismatches %d %s\n", bad, bad ? "FAIL" : "ok");
    fails += bad != 0;
    // (b) exhaustive e8m0 scales: all block scales of row v = v (including 0 = subnormal 2^-127 special path) · activation scale 2^20 so sa·sw = 2^(v-107) — v = 0 is normal too and visible in the result.
    //     255 (NaN) is assumed absent from checkpoints (e8m0_fast comment) — only checks that both versions produce the same bits (0x7F800000 = inf).
    std::vector<uint8_t> S2((size_t)256 * K / 32);
    for (int v = 0; v < 256; ++v) for (int b = 0; b < K / 32; ++b) S2[(size_t)v * (K / 32) + b] = (uint8_t)v;
    std::vector<float> sa2((size_t)K / 32);
    for (int b = 0; b < K / 32; ++b) sa2[(size_t)b] = std::ldexp(1.f, 20);
    const int bad2 = compare_rows(W.data(), S2.data(), 256, K, a.data(), sa2.data(), 0, 256);
    printf("bench_expert_cpu: exhaustive e8m0 scales (0..255) mismatches %d %s\n", bad2, bad2 ? "FAIL" : "ok");
    fails += bad2 != 0;
  }
  Expert e;
  fill_expert(e, rng, /*allow_zero=*/true);
  {
    int bad = 0, cases = 0;
    const auto a = rnd_act(rng, kDim), sa = rnd_sa(rng, kDim), yq = rnd_act(rng, kInter), sy = rnd_sa(rng, kInter);
    // all rows · the pool's half-record row ranges (1152 · 2560 rows) · odd boundaries · 32-row items
    const int ranges[][2] = {{0, kInter}, {0, 1152}, {1152, 2304}, {3, 1000}, {1, 2}, {5, 6}, {64, 96}};
    for (auto& r : ranges) { bad += compare_rows(e.w1.data(), e.s1.data(), kInter, kDim, a.data(), sa.data(), r[0], r[1]); ++cases; }
    const int ranges2[][2] = {{0, kDim}, {0, 2560}, {2560, 5120}, {7, 4999}, {4095, 4096}};
    for (auto& r : ranges2) { bad += compare_rows(e.w2.data(), e.s2.data(), kDim, kInter, yq.data(), sy.data(), r[0], r[1]); ++cases; }
    // nb % 8 != 0 tail path (absent from real shapes): K = 5120 + 3·32
    const int Kt = kDim + 96;
    std::vector<uint8_t> Wt((size_t)64 * Kt / 2), St((size_t)64 * Kt / 32);
    for (auto& x : Wt) x = (uint8_t)rng();
    for (auto& x : St) x = rnd_scale(rng, true);
    const auto at = rnd_act(rng, Kt), sat = rnd_sa(rng, Kt);
    bad += compare_rows(Wt.data(), St.data(), 64, Kt, at.data(), sat.data(), 0, 63); ++cases;
    // multi-row R = 1..8
    for (int R = 1; R <= 8; ++R) {
      std::vector<std::vector<float>> am, sm;
      for (int r = 0; r < R; ++r) { am.push_back(rnd_act(rng, kDim)); sm.push_back(rnd_sa(rng, kDim)); }
      bad += compare_multi(e.w1.data(), e.s1.data(), kInter, kDim, am, sm, R, 0, kInter); ++cases;
      bad += compare_multi(e.w1.data(), e.s1.data(), kInter, kDim, am, sm, R, 33, 97); ++cases;
      std::vector<std::vector<float>> amt, smt;
      for (int r = 0; r < R; ++r) { amt.push_back(rnd_act(rng, Kt)); smt.push_back(rnd_sa(rng, Kt)); }
      bad += compare_multi(Wt.data(), St.data(), 64, Kt, amt, smt, R, 0, 64); ++cases;
    }
    // _v3 tile boundaries: inside/outside row-group (16) boundaries · K tile (16/32/64 blocks) boundaries · shape whose nb%8 tail falls in the last tile (K = 5120 + 3·32 → nb 163)
    for (int R = 2; R <= 8; ++R) {
      std::vector<std::vector<float>> am, sm, amt, smt;
      for (int r = 0; r < R; ++r) { am.push_back(rnd_act(rng, kInter)); sm.push_back(rnd_sa(rng, kInter)); amt.push_back(rnd_act(rng, Kt)); smt.push_back(rnd_sa(rng, Kt)); }
      bad += compare_multi(e.w2.data(), e.s2.data(), kDim, kInter, am, sm, R, 15, 49); ++cases;     // w2 shape (nb 72 — with 16-block tiles: 16·16·16·16·8)
      bad += compare_multi(e.w2.data(), e.s2.data(), kDim, kInter, am, sm, R, 2560, 2592); ++cases; // the pool's 32-row item
      bad += compare_multi(e.w2.data(), e.s2.data(), kDim, kInter, am, sm, R, 100, 116); ++cases;   // FINE 16-row item
      bad += compare_multi(Wt.data(), St.data(), 64, Kt, amt, smt, R, 1, 64); ++cases;
    }
    // full expert forward (entry function = process switch): expert_forward uses gemv_e2m1_rows, so _ref/_v2 depends on the switch — here both versions are assembled directly and compared
    printf("bench_expert_cpu: real-shape bit comparison, %d cases (full/half record, odd boundary, nb%%8 tail, R=1..8), mismatches %d %s\n", cases, bad, bad ? "FAIL" : "ok");
    fails += bad != 0;
  }
  {  // HIVE_CPU_UNPACK2: vector activation unpack == scalar formula (e4m3_to_f32 · e8m0_to_f32) — exhaustive over 256 codes (8 positions × every value) + random rows (including dim · dim/32 tails)
    int bad = 0;
    for (int dim : {256 * 8, kDim, kDim + 40, 8, 13}) {
      std::vector<uint8_t> q((size_t)dim), sc((size_t)std::max(1, dim / 32));
      for (int d = 0; d < dim; ++d) q[(size_t)d] = dim == 256 * 8 ? (uint8_t)(d / 8 + 37 * (d % 8)) : (uint8_t)rng();
      for (size_t b = 0; b < sc.size(); ++b) sc[b] = dim == 256 * 8 ? (uint8_t)(b * 4 + 3) : (uint8_t)rng();
      std::vector<float> f0((size_t)dim, 7.f), f1 = f0, s0(sc.size(), 7.f), s1 = s0;
      cpu::unpack_act_row_ref(q.data(), sc.data(), dim, f0.data(), s0.data());
      cpu::unpack_act_row(q.data(), sc.data(), dim, f1.data(), s1.data());
      bad += std::memcmp(f0.data(), f1.data(), f0.size() * 4) != 0;
      bad += std::memcmp(s0.data(), s1.data(), s0.size() * 4) != 0;
    }
    std::vector<uint8_t> all(256);
    for (int v = 0; v < 256; ++v) all[(size_t)v] = (uint8_t)v;
    std::vector<float> f(256), sv(8), s8(256);
    cpu::unpack_act_row(all.data(), all.data(), 256, f.data(), sv.data());
    for (int v = 0; v < 256; ++v) { const float r = e4m3_to_f32((uint8_t)v); bad += std::memcmp(&r, &f[(size_t)v], 4) != 0; }
    for (int b0 = 0; b0 < 256; b0 += 8) {  // exhaustive e8m0: 8 at a time through the vector path
      std::vector<uint8_t> qq(256, 0);
      std::vector<float> ff(256);
      cpu::unpack_act_row(qq.data(), all.data() + b0, 256, ff.data(), s8.data() + b0);
    }
    for (int v = 0; v < 256; ++v) { const float r = e8m0_to_f32((uint8_t)v); bad += std::memcmp(&r, &s8[(size_t)v], 4) != 0; }
    printf("bench_expert_cpu: [B3] vector activation unpack == baseline (exhaustive e4m3 256, e8m0 256 + random rows, tail) mismatches %d %s\n", bad, bad ? "FAIL" : "ok");
    fails += bad != 0;
  }
  printf("bench_expert_cpu: entry switch HIVE_CPU_GEMV2 -> %s · [B3] HIVE_CPU_MULTIROW2 min rows %d (0 = off)\n", cpu::gemv2_enabled() ? "_v2" : "_ref (baseline)",
         cpu::multirow2_min_rows());
  if (quick) {
    printf("bench_expert_cpu --quick: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
  }
  // ---- 2) timing: cycle through several experts so reads come from DRAM (working set > L3) -------------------------------------------------------
  std::vector<Expert> ex((size_t)n_exp);
  ex[0] = e;
  for (int i = 1; i < n_exp; ++i) fill_expert(ex[(size_t)i], rng, false);
  const double mb = e.bytes() / 1e6;
  printf("bench_expert_cpu: CPU = %s · expert %.1f MB x %d (working set %.0f MB) · reps %d\n", cpu_model().c_str(), mb, n_exp, mb * n_exp, reps);
  const auto a = rnd_act(rng, kDim), sa = rnd_sa(rng, kDim), yq = rnd_act(rng, kInter), sy = rnd_sa(rng, kInter);
  // (a) single thread: one whole expert (w1·w3·w2)
  for (int v = 0; v < 2; ++v) {
    const RowsFn fn = v ? cpu::gemv_e2m1_rows_v2 : cpu::gemv_e2m1_rows_ref;
    std::vector<float> g(kInter), u(kInter), o(kDim);
    double best = 1e30;
    for (int rep = 0; rep < reps; ++rep) {
      const auto t0 = clk::now();
      for (int i = 0; i < n_exp; ++i) { expert_rows(fn, ex[(size_t)i], a.data(), sa.data(), yq.data(), sy.data(), g.data(), u.data(), o.data(), 0, kInter, 1);
                                         expert_rows(fn, ex[(size_t)i], a.data(), sa.data(), yq.data(), sy.data(), g.data(), u.data(), o.data(), 0, kDim, 2); }
      best = std::min(best, ms_since(t0) / n_exp);
    }
    printf("  single thread %s: %.3f ms/expert (%.1f GB/s)\n", v ? "_v2 " : "_ref", best, mb / best);
  }
  // (b) T threads: per expert, 32-row items (72 for w13 → 160 for w2 — same size as the pool's P1/P2 = 32) claimed via an atomic counter (including the wait between phases, like the pool)
  for (int T : {threads / 2, threads}) {
    if (T < 1) continue;
    for (int v = 0; v < 2; ++v) {
      const RowsFn fn = v ? cpu::gemv_e2m1_rows_v2 : cpu::gemv_e2m1_rows_ref;
      const int n1 = kInter / 32, n2 = kDim / 32;
      std::vector<float> g(kInter), u(kInter), o(kDim);
      double best = 1e30;
      for (int rep = 0; rep < reps; ++rep) {
        std::atomic<int> next{0}, p1_done{0};
        std::atomic<int> cur{0};
        const auto t0 = clk::now();
        for (int i = 0; i < n_exp; ++i) {  // one expert = one "job" (same unit as one CPU miss of a decode M=1 layer)
          next.store(0); p1_done.store(0);
          std::vector<std::thread> th;
          for (int t = 0; t < T; ++t) th.emplace_back([&, i] {
            for (;;) {
              const int it = next.fetch_add(1);
              if (it >= n1 + n2) break;
              if (it < n1) { expert_rows(fn, ex[(size_t)i], a.data(), sa.data(), yq.data(), sy.data(), g.data(), u.data(), o.data(), it * 32, it * 32 + 32, 1); p1_done.fetch_add(1); }
              else { while (p1_done.load() < n1) std::this_thread::yield();
                     const int r = (it - n1) * 32; expert_rows(fn, ex[(size_t)i], a.data(), sa.data(), yq.data(), sy.data(), g.data(), u.data(), o.data(), r, r + 32, 2); }
            }
          });
          for (auto& x : th) x.join();
        }
        (void)cur;
        best = std::min(best, ms_since(t0) / n_exp);
      }
      printf("  %2d threads %s: %.3f ms/expert (%.1f GB/s, includes thread create/join — unlike a pool wake-up)\n", T, v ? "_v2 " : "_ref", best, mb / best);
    }
  }
  // (c) T threads · persistent threads (no creation cost): each thread streams its whole share of experts — throughput in the bandwidth-saturated regime (ms/expert = total time / number of experts)
  for (int T : {threads / 2, threads}) {
    if (T < 1) continue;
    for (int v = 0; v < 2; ++v) {
      const RowsFn fn = v ? cpu::gemv_e2m1_rows_v2 : cpu::gemv_e2m1_rows_ref;
      double best = 1e30;
      for (int rep = 0; rep < reps; ++rep) {
        std::atomic<int> next{0};
        const int items_per = kInter / 32 + kDim / 32, total = items_per * n_exp;
        const auto t0 = clk::now();
        std::vector<std::thread> th;
        for (int t = 0; t < T; ++t) th.emplace_back([&] {
          std::vector<float> g(kInter), u(kInter), o(kDim);
          for (;;) {
            const int it = next.fetch_add(1);
            if (it >= total) break;
            const int i = it / items_per, k = it % items_per;
            if (k < kInter / 32) expert_rows(fn, ex[(size_t)i], a.data(), sa.data(), yq.data(), sy.data(), g.data(), u.data(), o.data(), k * 32, k * 32 + 32, 1);
            else expert_rows(fn, ex[(size_t)i], a.data(), sa.data(), yq.data(), sy.data(), g.data(), u.data(), o.data(), (k - kInter / 32) * 32, (k - kInter / 32) * 32 + 32, 2);
          }
        });
        for (auto& x : th) x.join();
        best = std::min(best, ms_since(t0) / n_exp);
      }
      printf("  %2d threads throughput %s: %.3f ms/expert (%.1f GB/s)\n", T, v ? "_v2 " : "_ref", best, mb / best);
    }
  }
  // (d) HIVE_CPU_MULTIROW2: multi-row _v2 vs _v3 — T threads claim w1 32-row items (72) per expert via an atomic counter (persistent threads · no barrier). ms per matrix (w1 2304×5120)
  {
    std::vector<std::vector<float>> am, sm;
    for (int r = 0; r < 8; ++r) { am.push_back(rnd_act(rng, kDim)); sm.push_back(rnd_sa(rng, kDim)); }
    const float* ap[8]; const float* sp[8];
    for (int r = 0; r < 8; ++r) { ap[r] = am[(size_t)r].data(); sp[r] = sm[(size_t)r].data(); }
    for (int R : {2, 3, 4, 5, 6, 8}) {
      double best[2] = {1e30, 1e30};
      for (int rep = 0; rep < reps; ++rep)
        for (int v = 0; v < 2; ++v) {  // alternate (_v2 → _v3 within the same repetition)
          std::atomic<int> next{0};
          const int per = kInter / 32, total = per * n_exp;
          const auto t0 = clk::now();
          std::vector<std::thread> th;
          for (int t = 0; t < threads; ++t) th.emplace_back([&] {
            std::vector<std::vector<float>> y(8, std::vector<float>(kInter));
            float* yp[8];
            for (int r = 0; r < 8; ++r) yp[r] = y[(size_t)r].data();
            for (;;) {
              const int it = next.fetch_add(1);
              if (it >= total) break;
              const Expert& x = ex[(size_t)(it / per)];
              const int r0 = (it % per) * 32;
              if (v == 0) cpu::gemv_e2m1_rows_multi_v2(x.w1.data(), x.s1.data(), kInter, kDim, ap, sp, R, r0, r0 + 32, yp);
              else cpu::gemv_e2m1_rows_multi_v3(x.w1.data(), x.s1.data(), kInter, kDim, ap, sp, R, r0, r0 + 32, yp);
            }
          });
          for (auto& x : th) x.join();
          best[v] = std::min(best[v], ms_since(t0) / n_exp);
        }
      printf("  [B3] %2d threads multi-row R=%d: _v2 %.3f · _v3 %.3f ms/matrix (w1) (%+.1f%%)\n", threads, R, best[0], best[1], 100.0 * (best[1] / best[0] - 1.0));
    }
  }
  printf("bench_expert_cpu: %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
