// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU matmul timings, current kernels against the ab_legacy snapshot in the same process (rounds interleaved), CPU only.
// usage: bench_cpu_matmul_ab <mode> [seconds per case = 0.6]
//   gemv    engine-shaped decode matrices (FN / GLM / DS / 27B), M = 1..8: µs per matrix and GB/s of weights, new vs legacy. Every set of
//           matrices is cycled over > 1.5 GiB so the L3 does not serve it.
//   shapes  the same with every kernel shape R×G (tune::gemv_force), new only — picks the default shape table
//   gemm    GFLOP/s per weight format and M (prefill), new vs legacy
//   gemm1 <fmt> <M> <N> <K>   one GEMM case, new only, for perf stat
//   stream  read bandwidth per NUMA node (node-local workers), both nodes, and cross-node
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <immintrin.h>

#include "hive/cpu/cpu_backend.h"

namespace hive::cpu {
namespace ab_legacy {
void matmul(const MatView& W, const float* X, int M, int ldx, float* Y, int ldy);
void matmul_multi(const MatJob* jobs_in, int n_jobs, const float* X, int M, int ldx);
}  // namespace ab_legacy
namespace tune { extern int gemv_force_r, gemv_force_g, gemm_mb, gemm_order, gemm_iloop; }
}  // namespace hive::cpu

using namespace hive::cpu;

namespace {
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
double secs = 0.6;

std::vector<uint8_t> rnd(size_t n, uint8_t mask = 0xff) {
  std::vector<uint8_t> v(n);
  std::mt19937 r(7);
  for (auto& b : v) b = (uint8_t)(r() & mask);
  return v;
}
const char* tname(WType t) {
  switch (t) { case WType::BF16: return "BF16"; case WType::F32: return "F32"; case WType::FP8: return "FP8"; case WType::NVFP4: return "NVFP4";
               case WType::MXFP4: return "MXFP4"; case WType::I8: return "I8"; }
  return "?";
}
WType parse_t(const std::string& s) {
  if (s == "BF16") return WType::BF16; if (s == "FP8") return WType::FP8; if (s == "NVFP4") return WType::NVFP4;
  if (s == "MXFP4") return WType::MXFP4; if (s == "I8") return WType::I8; return WType::F32;
}

Mat make_one(WType t, int N, int K) {
  switch (t) {
    case WType::BF16: { auto w = rnd((size_t)N * K * 2, 0x3f); return make_mat(t, N, K, w.data()); }
    case WType::F32: { std::vector<float> w((size_t)N * K, 0.01f); return make_mat(t, N, K, w.data()); }
    case WType::FP8: {
      auto w = rnd((size_t)N * K); for (auto& b : w) if ((b & 0x7f) == 0x7f) b = 0x30;
      std::vector<float> s((size_t)((N + 127) / 128) * ((K + 127) / 128), 0.01f);
      return make_mat(t, N, K, w.data(), s.data(), SType::F32);
    }
    case WType::NVFP4: { auto w = rnd((size_t)N * K / 2); auto s = rnd((size_t)N * K / 16, 0x3f); for (auto& b : s) b |= 0x20;
                         return make_mat(t, N, K, w.data(), s.data(), SType::E4M3, 0.01f); }
    case WType::MXFP4: { auto w = rnd((size_t)N * K / 2); std::vector<uint8_t> s((size_t)N * K / 32, 127);
                         return make_mat(t, N, K, w.data(), s.data(), SType::E8M0); }
    case WType::I8: { std::vector<uint16_t> w((size_t)N * K, 0x3c00); return make_mat_i8(N, K, w.data(), 64); }
  }
  return Mat{};
}

struct GvCase { const char* model; const char* what; WType t; int N, K; };
const GvCase kGv[] = {
    {"FN", "gdn in_proj", WType::BF16, 12288, 2560},  {"FN", "gdn out_proj", WType::BF16, 2560, 6144},
    {"FN", "expert gate/up", WType::NVFP4, 640, 2560}, {"FN", "expert down", WType::NVFP4, 2560, 640},
    {"GLM", "q_a (fp8)", WType::FP8, 1536, 4096},      {"GLM", "q_b (fp8)", WType::FP8, 12288, 1536},
    {"GLM", "expert gate/up", WType::NVFP4, 2048, 4096}, {"GLM", "expert down", WType::NVFP4, 4096, 2048},
    {"DS", "wq_b (fp8)", WType::FP8, 32768, 1280},     {"DS", "expert gate/up", WType::MXFP4, 2304, 5120},
    {"DS", "expert down", WType::MXFP4, 5120, 2304},
    {"27B", "mlp gate/up", WType::NVFP4, 17408, 5120}, {"27B", "mlp down", WType::NVFP4, 5120, 17408},
    {"27B", "attn q (bf16)", WType::BF16, 12288, 5120},
    {"any", "I8 dense", WType::I8, 12288, 2560},
};

struct Set { std::vector<Mat> m; size_t bytes = 0; };
Set make_set(WType t, int N, int K, size_t total) {
  Set s;
  Mat one = make_one(t, N, K);
  const int cnt = (int)std::max<size_t>(2, total / std::max<size_t>(1, one.bytes));
  s.m.push_back(one); s.bytes += one.bytes;
  for (int i = 1; i < cnt; ++i) { s.m.push_back(make_one(t, N, K)); s.bytes += s.m.back().bytes; }
  return s;
}

// µs per matrix (whole set passes), interleaved rounds: returns {new, legacy}
std::pair<double, double> time_gemv(Set& s, int M, const std::vector<float>& X, std::vector<float>& Y, int K, int N, bool legacy_too) {
  double tn = 0, tl = 0; long cn = 0, cl = 0;
  const double t_end = now() + secs * (legacy_too ? 2 : 1);
  for (auto& m : s.m) matmul(m, X.data(), M, K, Y.data(), N);
  if (legacy_too) for (auto& m : s.m) ab_legacy::matmul(MatView{&m, 0, m.N}, X.data(), M, K, Y.data(), N);
  while (now() < t_end) {
    double t0 = now();
    for (auto& m : s.m) matmul(m, X.data(), M, K, Y.data(), N);
    tn += now() - t0; cn += (long)s.m.size();
    if (legacy_too) {
      t0 = now();
      for (auto& m : s.m) ab_legacy::matmul(MatView{&m, 0, m.N}, X.data(), M, K, Y.data(), N);
      tl += now() - t0; cl += (long)s.m.size();
    }
  }
  return {tn / cn * 1e6, legacy_too ? tl / cl * 1e6 : 0.0};
}

void bench_gemv(bool shapes) {
  const size_t G = 3ull << 29;  // 1.5 GiB
  printf("%-4s %-15s %-6s %5s×%-5s %2s %10s %10s %8s %8s %7s %s\n", "mdl", "matrix", "fmt", "N", "K", "M", "new µs", "old µs", "new GB/s", "old GB/s",
         "speedup", "new µs / M=1 µs");
  const char* only = getenv("GV_CASES");  // comma list of kGv indexes
  for (int ci = 0; ci < (int)(sizeof kGv / sizeof kGv[0]); ++ci) {
    const GvCase& c = kGv[ci];
    if (only && !strstr(("," + std::string(only) + ",").c_str(), ("," + std::to_string(ci) + ",").c_str())) continue;
    Set s = make_set(c.t, c.N, c.K, c.t == WType::NVFP4 && c.N * c.K < 4000000 ? G / 2 : G);
    std::vector<float> X((size_t)8 * c.K), Y((size_t)8 * c.N);
    std::mt19937 r(1);
    for (auto& v : X) v = (float)(r() % 1000) / 1000.f - 0.5f;
    const double mb = (double)s.m[0].bytes;
    double t1 = 0;
    for (int M = 1; M <= 8; ++M) {
      if (!shapes) {
        auto [tn, tl] = time_gemv(s, M, X, Y, c.K, c.N, true);
        if (M == 1) t1 = tn;
        printf("%-4s %-15s %-6s %5d×%-5d %2d %10.1f %10.1f %8.1f %8.1f %6.2fx %5.2f\n", c.model, c.what, tname(c.t), c.N, c.K, M, tn, tl, mb / tn / 1e3,
               mb / tl / 1e3, tl / tn, tn / t1);
        fflush(stdout);
      } else {
        if (M == 5 || M == 7) continue;
        printf("%-4s %-15s %-6s %5d×%-5d M=%d:", c.model, c.what, tname(c.t), c.N, c.K, M);
        std::vector<std::pair<int, int>> shp{{1, M}, {2, M}};
        if (M <= 4) shp.push_back({4, M});
        if (M > 4) { shp.push_back({2, 4}); shp.push_back({4, 4}); shp.push_back({1, 4}); }
        if (M == 1) shp = {{1, 1}, {2, 1}, {4, 1}};
        for (auto [R, g] : shp) {
          tune::gemv_force_r = R; tune::gemv_force_g = g;
          auto [tn, tl] = time_gemv(s, M, X, Y, c.K, c.N, false);
          printf(" %dx%d=%.1f", R, g, tn);
          fflush(stdout);
        }
        tune::gemv_force_r = tune::gemv_force_g = 0;
        auto [tn, tl] = time_gemv(s, M, X, Y, c.K, c.N, true);
        printf(" | default %.1f legacy %.1f\n", tn, tl);
      }
    }
    for (auto& m : s.m) free_mat(m);
  }
}

double gemm_time(const Mat& m, int M, const std::vector<float>& X, std::vector<float>& Y, bool legacy) {
  if (legacy) ab_legacy::matmul(MatView{&m, 0, m.N}, X.data(), M, m.K, Y.data(), m.N); else matmul(m, X.data(), M, m.K, Y.data(), m.N);
  int it = 0; const double t0 = now();
  while (now() - t0 < secs) {
    if (legacy) ab_legacy::matmul(MatView{&m, 0, m.N}, X.data(), M, m.K, Y.data(), m.N); else matmul(m, X.data(), M, m.K, Y.data(), m.N);
    ++it;
  }
  return (now() - t0) / it;
}

void bench_gemm() {
  struct C { WType t; int N, K; const char* what; };
  const C cs[] = {{WType::BF16, 12288, 2560, "FN gdn in"}, {WType::FP8, 12288, 2560, ""}, {WType::NVFP4, 12288, 2560, ""},
                  {WType::MXFP4, 12288, 2560, ""}, {WType::I8, 12288, 2560, ""}, {WType::F32, 12288, 2560, ""},
                  {WType::NVFP4, 640, 2560, "FN expert"}, {WType::NVFP4, 17408, 5120, "27B mlp"}, {WType::MXFP4, 2304, 5120, "DS expert"},
                  {WType::FP8, 32768, 1280, "DS wq_b"}};
  printf("%-6s %5s×%-5s %-10s %5s %9s %9s %9s %9s %7s\n", "fmt", "N", "K", "", "M", "new ms", "old ms", "new GF/s", "old GF/s", "speedup");
  for (const C& c : cs) {
    Mat m = make_one(c.t, c.N, c.K);
    for (int M : {128, 512, 2048}) {
      std::vector<float> X((size_t)M * c.K, 0.25f), Y((size_t)M * c.N);
      const double tn = gemm_time(m, M, X, Y, false), tl = gemm_time(m, M, X, Y, true);
      const double f = 2.0 * M * c.N * c.K;
      printf("%-6s %5d×%-5d %-10s %5d %9.2f %9.2f %9.0f %9.0f %6.2fx\n", tname(c.t), c.N, c.K, c.what, M, tn * 1e3, tl * 1e3, f / tn / 1e9, f / tl / 1e9,
             tl / tn);
      fflush(stdout);
    }
    free_mat(m);
  }
}

// MoE decode layer: top-k of E experts, gate + up of every chosen expert in one dispatch, then the k down projections (own X) in a second;
//   all M rows use the same experts (the per-expert M of a batch that agrees on its experts — the worst case for the per-expert work).
//   The expert set is cycled over > 1.5 GiB so the L3 does not serve it.
void bench_moe_decode() {
  struct C { const char* name; WType t; int E, top, I, H; };
  const C cs[] = {{"FN", WType::NVFP4, 512, 10, 640, 2560}, {"GLM", WType::NVFP4, 288, 8, 2048, 4096}, {"DS", WType::MXFP4, 384, 6, 2304, 5120}};
  for (const C& c : cs) {
    Mat one = make_one(c.t, c.I, c.H);
    const int E = (int)std::min<size_t>(c.E, std::max<size_t>(3 * c.top, (3ull << 29) / (3 * one.bytes)));
    free_mat(one);
    std::vector<Mat> g(E), u(E), d(E);
    for (int e = 0; e < E; ++e) { g[e] = make_one(c.t, c.I, c.H); u[e] = make_one(c.t, c.I, c.H); d[e] = make_one(c.t, c.H, c.I); }
    std::vector<float> X((size_t)8 * c.H, 0.25f), GU((size_t)c.top * 2 * 8 * c.I, 0.1f), Dn((size_t)c.top * 8 * c.H);
    double t1 = 0;
    for (int M = 1; M <= 8; ++M) {
      double tt[2] = {0, 0}; long n[2] = {0, 0};
      std::mt19937 r(3);
      const double t_end = now() + secs * 2;
      int step = 0;
      while (now() < t_end || step < 4) {
        int ex[16];
        for (int i = 0; i < c.top; ++i) ex[i] = (int)(r() % E);
        std::vector<MatJob> j1(2 * c.top), j2(c.top);
        for (int i = 0; i < c.top; ++i) {
          j1[2 * i] = {MatView{&g[ex[i]], 0, c.I}, GU.data() + (size_t)(2 * i) * M * c.I, c.I};
          j1[2 * i + 1] = {MatView{&u[ex[i]], 0, c.I}, GU.data() + (size_t)(2 * i + 1) * M * c.I, c.I};
          j2[i] = {MatView{&d[ex[i]], 0, c.H}, Dn.data() + (size_t)i * M * c.H, c.H, GU.data() + (size_t)(2 * i) * M * c.I};
        }
        const int lg = step & 1;
        const double t0 = now();
        if (lg) { ab_legacy::matmul_multi(j1.data(), 2 * c.top, X.data(), M, c.H); ab_legacy::matmul_multi(j2.data(), c.top, nullptr, M, c.I); }
        else { matmul_multi(j1.data(), 2 * c.top, X.data(), M, c.H); matmul_multi(j2.data(), c.top, nullptr, M, c.I); }
        tt[lg] += now() - t0; ++n[lg]; ++step;
      }
      const double a = tt[0] / n[0] * 1e6, b = tt[1] / n[1] * 1e6;
      if (M == 1) t1 = a;
      printf("MoE decode %-3s %-5s top-%d M=%d: new %7.1f µs · old %7.1f µs · %.2fx · new/M=1 %.2f\n", c.name, tname(c.t), c.top, M, a, b, b / a, a / t1);
      fflush(stdout);
    }
    for (int e = 0; e < E; ++e) { free_mat(g[e]); free_mat(u[e]); free_mat(d[e]); }
  }
}

// MoE prefill layer: E experts × (gate + up in one dispatch, then down with per-job X), every expert with Me rows (2048 tokens · top-k / E)
void bench_moe_prefill() {
  struct C { const char* name; WType t; int E, I, H, Me; };
  const C cs[] = {{"FN", WType::NVFP4, 512, 640, 2560, 40}, {"GLM", WType::NVFP4, 288, 2048, 4096, 57}, {"DS", WType::MXFP4, 384, 2304, 5120, 32}};
  for (const C& c : cs) {
    const int E = std::min(c.E, 128);  // 128 experts per dispatch is plenty to fill the pool
    std::vector<Mat> g(E), u(E), d(E);
    for (int e = 0; e < E; ++e) { g[e] = make_one(c.t, c.I, c.H); u[e] = make_one(c.t, c.I, c.H); d[e] = make_one(c.t, c.H, c.I); }
    std::vector<float> X((size_t)E * c.Me * c.H, 0.25f), GU((size_t)E * 2 * c.Me * c.I, 0.1f), Dn((size_t)E * c.Me * c.H);
    std::vector<MatJob> j1(2 * E), j2(E);
    for (int e = 0; e < E; ++e) {
      j1[2 * e] = {MatView{&g[e], 0, c.I}, GU.data() + (size_t)(2 * e) * c.Me * c.I, c.I, X.data() + (size_t)e * c.Me * c.H, c.Me, c.H};
      j1[2 * e + 1] = {MatView{&u[e], 0, c.I}, GU.data() + (size_t)(2 * e + 1) * c.Me * c.I, c.I, X.data() + (size_t)e * c.Me * c.H, c.Me, c.H};
      j2[e] = {MatView{&d[e], 0, c.H}, Dn.data() + (size_t)e * c.Me * c.H, c.H, GU.data() + (size_t)(2 * e) * c.Me * c.I, c.Me, c.I};
    }
    double tt[2];
    for (int legacy = 0; legacy < 2; ++legacy) {
      auto run = [&] {
        if (legacy) { ab_legacy::matmul_multi(j1.data(), 2 * E, nullptr, 0, 0); ab_legacy::matmul_multi(j2.data(), E, nullptr, 0, 0); }
        else { matmul_multi(j1.data(), 2 * E, nullptr, 0, 0); matmul_multi(j2.data(), E, nullptr, 0, 0); }
      };
      run();
      int it = 0; const double t0 = now();
      while (now() - t0 < secs * 2) { run(); ++it; }
      tt[legacy] = (now() - t0) / it;
    }
    const double f = 2.0 * E * c.Me * 3.0 * c.I * c.H;
    printf("MoE prefill %-3s %-5s %d experts × M=%d (%d×%d): new %.2f ms %.0f GF/s · old %.2f ms %.0f GF/s · %.2fx\n", c.name, tname(c.t), E, c.Me, c.I, c.H,
           tt[0] * 1e3, f / tt[0] / 1e9, tt[1] * 1e3, f / tt[1] / 1e9, tt[1] / tt[0]);
    fflush(stdout);
    for (int e = 0; e < E; ++e) { free_mat(g[e]); free_mat(u[e]); free_mat(d[e]); }
  }
}

// GEMM loop variants (tune::gemm_*), new kernels only
void bench_gemm_sweep() {
  struct C { WType t; int M, N, K; const char* what; };
  const C cs[] = {{WType::BF16, 2048, 12288, 2560, "FN gdn in"}, {WType::NVFP4, 512, 17408, 5120, "27B mlp"}, {WType::MXFP4, 2048, 2304, 5120, "DS expert"},
                  {WType::FP8, 2048, 32768, 1280, "DS wq_b"}};
  printf("pool %d threads + %d SMT helpers\n", Pool::get().threads(), Pool::get().smt_threads());
  for (const C& c : cs) {
    Mat m = make_one(c.t, c.N, c.K);
    std::vector<float> X((size_t)c.M * c.K, 0.25f), Y((size_t)c.M * c.N);
    printf("%-6s %-9s M=%d N=%d K=%d GF/s:", tname(c.t), c.what, c.M, c.N, c.K);
    for (int order : {0, 1}) for (int il : {0, 1}) for (int mb : {48, 96, 192}) {
      tune::gemm_order = order; tune::gemm_iloop = il; tune::gemm_mb = mb;
      const double tn = gemm_time(m, c.M, X, Y, false);
      printf(" o%d/i%d/mb%d=%.0f", order, il, mb, 2.0 * c.M * c.N * c.K / tn / 1e9);
      fflush(stdout);
    }
    tune::gemm_order = 0; tune::gemm_iloop = 0; tune::gemm_mb = 192;
    printf("\n");
    free_mat(m);
  }
}

// interleaved rounds of three loop variants (default o0/i0, o0/i1, o1/i1 at mb 192): median GFLOP/s
void bench_gemm_loop_ab() {
  struct C { WType t; int M, N, K; const char* what; };
  const C cs[] = {{WType::BF16, 2048, 12288, 2560, "FN gdn in"}, {WType::NVFP4, 512, 17408, 5120, "27B mlp"}, {WType::MXFP4, 2048, 2304, 5120, "DS expert"},
                  {WType::FP8, 2048, 32768, 1280, "DS wq_b"}, {WType::NVFP4, 2048, 640, 2560, "FN expert"}};
  const int V[3][2] = {{0, 0}, {0, 1}, {1, 1}};
  printf("pool %d threads + %d SMT helpers\n", Pool::get().threads(), Pool::get().smt_threads());
  for (const C& c : cs) {
    Mat m = make_one(c.t, c.N, c.K);
    std::vector<float> X((size_t)c.M * c.K, 0.25f), Y((size_t)c.M * c.N);
    std::vector<double> g[3];
    for (int round = 0; round < 7; ++round)
      for (int v = 0; v < 3; ++v) {
        tune::gemm_order = V[v][0]; tune::gemm_iloop = V[v][1]; tune::gemm_mb = 192;
        g[v].push_back(2.0 * c.M * c.N * c.K / gemm_time(m, c.M, X, Y, false) / 1e9);
      }
    tune::gemm_order = 0; tune::gemm_iloop = 0;
    printf("%-6s %-9s M=%d N=%d K=%d median GF/s:", tname(c.t), c.what, c.M, c.N, c.K);
    for (int v = 0; v < 3; ++v) { std::sort(g[v].begin(), g[v].end()); printf(" o%d/i%d=%.0f [%.0f-%.0f]", V[v][0], V[v][1], g[v][3], g[v][0], g[v][6]); }
    printf("\n");
    fflush(stdout);
    free_mat(m);
  }
}

void bench_gemm1(WType t, int M, int N, int K) {
  Mat m = make_one(t, N, K);
  std::vector<float> X((size_t)M * K, 0.25f), Y((size_t)M * N);
  const double tn = gemm_time(m, M, X, Y, getenv("LEGACY") != nullptr);
  printf("%s M=%d N=%d K=%d%s: %.2f ms · %.0f GFLOP/s\n", tname(t), M, N, K, getenv("LEGACY") ? " (legacy)" : "", tn * 1e3, 2.0 * M * N * K / tn / 1e9);
  free_mat(m);
}

// read bandwidth: each task sums one 1 MiB slice with 4 independent 256-bit chains
void bench_stream() {
  Pool& P = Pool::get();
  const size_t per = 4ull << 30, slice = 1u << 20;  // 4 GiB per node
  void* buf[2] = {numa_alloc_node(per, 0), P.nodes() > 1 ? numa_alloc_node(per, 1) : nullptr};
  for (int k = 0; k < P.nodes(); ++k) {
    const int ns = (int)(per / slice);
    std::vector<int> n(2, 0); n[k] = ns;
    P.run_nodes(n, [&](int kk, int i) { memset((char*)buf[kk] + (size_t)i * slice, 1, slice); });
  }
  static float sink[64 * 16];
  auto rd = [&](const void* p, int tid) {
    const __m256* v = (const __m256*)p;
    __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
    for (size_t i = 0; i < slice / 32; i += 4) {
      a = _mm256_add_ps(a, v[i]); b = _mm256_add_ps(b, v[i + 1]); c = _mm256_add_ps(c, v[i + 2]); d = _mm256_add_ps(d, v[i + 3]);
    }
    _mm256_storeu_ps(sink + (tid % 64) * 16, _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d)));
  };
  const int ns = (int)(per / slice);
  auto run = [&](const char* name, std::vector<int> n, std::function<const void*(int, int)> src) {
    P.run_nodes(n, [&](int k, int i) { rd(src(k, i), i); });  // warm
    double best = 0;
    for (int rep = 0; rep < 5; ++rep) {
      const double t0 = now();
      P.run_nodes(n, [&](int k, int i) { rd(src(k, i), i); });
      const double dt = now() - t0;
      best = std::max(best, (double)(n[0] + n[1]) * slice / dt / 1e9);
    }
    printf("STREAM read %-40s %7.1f GB/s (best of 5)\n", name, best);
  };
  printf("pool %d threads (%d + %d per node)\n", P.threads(), P.threads_on(0), P.nodes() > 1 ? P.threads_on(1) : 0);
  run("node 0 workers → node 0 memory", {ns, 0}, [&](int, int i) { return (const char*)buf[0] + (size_t)i * slice; });
  if (P.nodes() > 1) {
    run("node 1 workers → node 1 memory", {0, ns}, [&](int, int i) { return (const char*)buf[1] + (size_t)i * slice; });
    run("both nodes, node-local", {ns, ns}, [&](int k, int i) { return (const char*)buf[k] + (size_t)i * slice; });
    run("node 0 workers → node 1 memory (remote)", {ns, 0}, [&](int, int i) { return (const char*)buf[1] + (size_t)i * slice; });
    run("both nodes, all remote", {ns, ns}, [&](int k, int i) { return (const char*)buf[1 - k] + (size_t)i * slice; });
  }
  numa_free_node(buf[0], per);
  if (buf[1]) numa_free_node(buf[1], per);
}
}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "gemv";
  Pool& P = Pool::get();
  printf("pool: %d threads on %d node(s)\n", P.threads(), P.nodes());
  if (mode == "gemm1") { secs = 3; bench_gemm1(parse_t(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5])); return 0; }
  if (argc > 2) secs = atof(argv[2]);
  if (mode == "gemv") bench_gemv(false);
  else if (mode == "shapes") bench_gemv(true);
  else if (mode == "gemm") { bench_gemm(); bench_moe_prefill(); }
  else if (mode == "moe") bench_moe_prefill();
  else if (mode == "moedec") bench_moe_decode();
  else if (mode == "gemmsweep") bench_gemm_sweep();
  else if (mode == "gemmab") bench_gemm_loop_ab();
  else if (mode == "stream") bench_stream();
  return 0;
}
