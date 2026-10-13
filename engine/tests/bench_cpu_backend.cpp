// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU backend timings: decode GEMV bandwidth per weight format (the matrices of one pass are cycled over > 1 GiB so the 128 MiB of L3 does
//   not serve them), expert-sized GEMV calls (dispatch included), GEMM GFLOP/s, pool dispatch latency. CPU only.
// usage: bench_cpu_backend [seconds per case = 1.5]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <algorithm>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/cpu/cpu_backend.h"

using namespace hive::cpu;

namespace {
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
double secs = 1.5;

std::vector<uint8_t> rnd(size_t n, uint8_t mask = 0xff, uint8_t avoid = 0) {
  std::vector<uint8_t> v(n);
  std::mt19937 r(7);
  for (auto& b : v) { b = (uint8_t)(r() & mask); if (avoid && (b & 0x7f) == avoid) b = 0x30; }
  return v;
}

struct Set { std::vector<Mat> m; size_t bytes = 0; };
Set make_set(WType t, int N, int K, size_t total) {
  Set s;
  size_t rb = t == WType::BF16 ? (size_t)K * 2 : t == WType::FP8 ? (size_t)K : (size_t)K / 2;
  const size_t one = rb * N;
  const int cnt = (int)std::max<size_t>(2, total / one);
  std::vector<uint8_t> w = rnd(one, 0xff, t == WType::FP8 ? 0x7f : 0);
  std::vector<uint8_t> sc;
  SType st = SType::F32;
  if (t == WType::FP8) { std::vector<float> f((size_t)((N + 127) / 128) * ((K + 127) / 128), 0.01f); sc.resize(f.size() * 4); memcpy(sc.data(), f.data(), sc.size()); }
  if (t == WType::NVFP4) { sc = rnd((size_t)N * K / 16, 0x3f); for (auto& b : sc) b |= 0x20; st = SType::E4M3; }
  if (t == WType::MXFP4) { sc.assign((size_t)N * K / 32, 127); st = SType::E8M0; }
  for (int i = 0; i < cnt; ++i) { s.m.push_back(make_mat(t, N, K, w.data(), sc.empty() ? nullptr : sc.data(), st, 0.01f)); s.bytes += s.m.back().bytes; }
  return s;
}

void gemv_bw(const char* name, WType t, int N, int K, size_t total) {
  Set s = make_set(t, N, K, total);
  std::vector<float> X((size_t)8 * K, 0.5f), Y((size_t)8 * N);
  for (int M : {1, 4, 8}) {
    for (auto& m : s.m) matmul(m, X.data(), M, K, Y.data(), N);  // warm
    size_t bytes = 0; int calls = 0;
    const double t0 = now();
    while (now() - t0 < secs) for (auto& m : s.m) { matmul(m, X.data(), M, K, Y.data(), N); bytes += m.bytes; ++calls; }
    const double dt = now() - t0;
    printf("GEMV %-6s %5d×%-5d M=%d: %6.1f GB/s · %7.1f µs/call (set %.2f GiB, %zu matrices)\n", name, N, K, M, bytes / dt / 1e9, dt / calls * 1e6,
           s.bytes / 1073741824.0, s.m.size());
  }
  for (auto& m : s.m) free_mat(m);
}

// one FN MoE layer of decode: top-10 of 512 experts (~2.4 GiB, so L3 does not serve them), gate + up (20 × 640×2560 NVFP4) in one dispatch,
// then the 10 down projections (2560×640, each with its own input) in a second one.
void moe_layer(int M) {
  const int E = 512, top = 10, I = 640, H = 2560;
  std::vector<uint8_t> wg = rnd((size_t)I * H / 2), sg = rnd((size_t)I * H / 16, 0x3f);
  for (auto& b : sg) b |= 0x20;
  std::vector<Mat> g(E), u(E), d(E);
  for (int e = 0; e < E; ++e) {
    g[e] = make_mat(WType::NVFP4, I, H, wg.data(), sg.data(), SType::E4M3, 0.01f);
    u[e] = make_mat(WType::NVFP4, I, H, wg.data(), sg.data(), SType::E4M3, 0.01f);
    d[e] = make_mat(WType::NVFP4, H, I, wg.data(), sg.data(), SType::E4M3, 0.01f);
  }
  std::vector<float> X((size_t)M * H, 0.5f), GU((size_t)top * 2 * M * I), Dn((size_t)top * M * H);
  std::mt19937 r(3);
  int layers = 0; size_t bytes = 0;
  const double t0 = now();
  while (now() - t0 < secs) {
    int ex[top];
    for (int i = 0; i < top; ++i) ex[i] = (int)(r() % E);
    MatJob j1[2 * top], j2[top];
    for (int i = 0; i < top; ++i) {
      j1[2 * i] = {MatView{&g[ex[i]], 0, I}, GU.data() + (size_t)(2 * i) * M * I, I};
      j1[2 * i + 1] = {MatView{&u[ex[i]], 0, I}, GU.data() + (size_t)(2 * i + 1) * M * I, I};
      j2[i] = {MatView{&d[ex[i]], 0, H}, Dn.data() + (size_t)i * M * H, H, GU.data() + (size_t)(2 * i) * M * I};
      bytes += g[ex[i]].bytes + u[ex[i]].bytes + d[ex[i]].bytes;
    }
    matmul_multi(j1, 2 * top, X.data(), M, H);
    matmul_multi(j2, top, nullptr, M, I);
    ++layers;
  }
  const double dt = (now() - t0) / layers;
  printf("MoE layer NVFP4 top-%d M=%d: %6.1f µs/layer · %5.1f GB/s · ×48 layers = %.1f ms\n", top, M, dt * 1e6, bytes / (now() - t0) / 1e9, dt * 48e3);
  for (int e = 0; e < E; ++e) { free_mat(g[e]); free_mat(u[e]); free_mat(d[e]); }
}

void gemm_flops(int M, int N, int K) {
  std::vector<uint8_t> w = rnd((size_t)N * K * 2, 0x3f);
  Mat m = make_mat(WType::BF16, N, K, w.data());
  std::vector<float> X((size_t)M * K, 0.25f), Y((size_t)M * N);
  matmul(m, X.data(), M, K, Y.data(), N);
  int it = 0;
  const double t0 = now();
  while (now() - t0 < secs) { matmul(m, X.data(), M, K, Y.data(), N); ++it; }
  const double dt = (now() - t0) / it;
  printf("GEMM BF16 M=%d N=%d K=%d: %.1f ms · %.0f GFLOP/s\n", M, N, K, dt * 1e3, 2.0 * M * N * K / dt / 1e9);
  free_mat(m);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) secs = atof(argv[1]);
  Pool& P = Pool::get();
  printf("pool: %d threads on %d node(s)\n", P.threads(), P.nodes());
  {
    int it = 0; const double t0 = now();
    const std::function<void(int)> f = [](int) {};
    while (now() - t0 < 0.5) { P.run(P.threads(), f); ++it; }
    printf("dispatch (run, %d tasks): %.2f µs\n", P.threads(), (now() - t0) / it * 1e6);
  }
  const size_t G = 3ull << 30;
  gemv_bw("BF16", WType::BF16, 12288, 2560, G);
  gemv_bw("BF16", WType::BF16, 2560, 6144, G);
  gemv_bw("FP8", WType::FP8, 12288, 2560, G);
  gemv_bw("NVFP4", WType::NVFP4, 640, 2560, G / 2);
  gemv_bw("NVFP4", WType::NVFP4, 2560, 640, G / 2);
  gemv_bw("MXFP4", WType::MXFP4, 2048, 7168, G);
  for (int M : {1, 4, 8}) moe_layer(M);
  gemm_flops(512, 10240, 2560);
  gemm_flops(4096, 10240, 2560);
  return 0;
}
