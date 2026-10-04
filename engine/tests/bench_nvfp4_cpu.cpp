// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// NVFP4 CPU GEMV: fp32-activation kernel (gemv_nvfp4_rows_multi) vs W4A8 (gemv_nvfp4_q8_rows_multi) — accuracy against an fp64
//   reference and single-thread / threaded throughput on a GLM expert shape (gate [2048, 4096]). CPU only.
//   g++ -std=c++20 -O3 -mavx2 -mfma -mf16c -pthread -Itools/cpu_fake/include -Iengine/include engine/tests/bench_nvfp4_cpu.cpp \
//       engine/src/cpu/expert_cpu_nvfp4.cpp engine/src/cpu/expert_cpu.cpp -o /tmp/bn
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "hive/expert_cpu.h"

using namespace hive;
using clk = std::chrono::steady_clock;

static double e4m3d(uint8_t v) {  // e4m3 (fn) → value, for the fp64 reference
  const int s = v >> 7, e = (v >> 3) & 0xF, m = v & 7;
  const double x = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
  return s ? -x : x;
}

int main(int argc, char** argv) {
  const int N = 2048, K = 4096, NM = 48;  // 48 matrices = 192 MB (out of cache)
  std::mt19937 rng(1);
  std::vector<std::vector<uint8_t>> W(NM, std::vector<uint8_t>((size_t)N * K / 2)), S(NM, std::vector<uint8_t>((size_t)N * K / 16));
  for (int m = 0; m < NM; ++m) {
    for (auto& v : W[m]) v = rng() & 0xFF;
    for (auto& v : S[m]) v = 0x28 + rng() % 0x18;
  }
  const float g = 0.0031f;
  std::normal_distribution<float> nd(0.f, 1.f);
  int fails = 0;
  for (int R : {1, 2, 4, 8}) {
    std::vector<std::vector<float>> a(R, std::vector<float>(K));
    for (auto& r : a) for (auto& v : r) v = nd(rng) * (1.f + 3.f * (rng() % 50 == 0));  // a few outliers
    std::vector<std::vector<int8_t>> q(R, std::vector<int8_t>(K)); std::vector<std::vector<float>> qs(R, std::vector<float>(K / 32));
    for (int r = 0; r < R; ++r) cpu::quantize_q8_nvfp4(a[r].data(), K, q[r].data(), qs[r].data());
    std::vector<const float*> ap(R); std::vector<const int8_t*> qp(R); std::vector<const float*> sp(R);
    std::vector<std::vector<float>> y1(R, std::vector<float>(N)), y2(R, std::vector<float>(N));
    std::vector<float*> y1p(R), y2p(R);
    for (int r = 0; r < R; ++r) { ap[r] = a[r].data(); qp[r] = q[r].data(); sp[r] = qs[r].data(); y1p[r] = y1[r].data(); y2p[r] = y2[r].data(); }
    cpu::gemv_nvfp4_rows_multi(W[0].data(), S[0].data(), g, N, K, ap.data(), R, 0, N, y1p.data());
    cpu::gemv_nvfp4_q8_rows_multi(W[0].data(), S[0].data(), g, N, K, qp.data(), sp.data(), R, 0, N, y2p.data());
    // fp64 reference for row 0 of the batch
    double e1 = 0, e2 = 0, den = 0;
    static const float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    for (int r = 0; r < R; ++r)
      for (int n = 0; n < N; ++n) {
        double ref = 0;
        for (int k = 0; k < K; ++k) {
          const uint8_t b = W[0][(size_t)n * K / 2 + k / 2];
          ref += (double)a[r][k] * kE2M1[(k & 1) ? (b >> 4) : (b & 15)] * e4m3d(S[0][(size_t)n * K / 16 + k / 16]);
        }
        ref *= g;
        e1 += (y1[r][n] - ref) * (y1[r][n] - ref); e2 += (y2[r][n] - ref) * (y2[r][n] - ref); den += ref * ref;
      }
    const double r1 = std::sqrt(e1 / den), r2 = std::sqrt(e2 / den);
    const bool ok = r1 < 1e-5 && r2 < 1.5e-2;
    fails += !ok;
    // timing: single thread, all NM matrices
    auto bench = [&](bool q8, int threads) {
      const auto t0 = clk::now();
      const int reps = 2;
      for (int rep = 0; rep < reps; ++rep) {
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t)
          th.emplace_back([&, t] {
            std::vector<std::vector<float>> yy(R, std::vector<float>(N)); std::vector<float*> yp(R);
            for (int r = 0; r < R; ++r) yp[r] = yy[r].data();
            for (int m = t; m < NM; m += threads) {
              if (q8) cpu::gemv_nvfp4_q8_rows_multi(W[m].data(), S[m].data(), g, N, K, qp.data(), sp.data(), R, 0, N, yp.data());
              else cpu::gemv_nvfp4_rows_multi(W[m].data(), S[m].data(), g, N, K, ap.data(), R, 0, N, yp.data());
            }
          });
        for (auto& x : th) x.join();
      }
      const double s = std::chrono::duration<double>(clk::now() - t0).count() / reps;
      return (double)NM * ((size_t)N * K / 2 + (size_t)N * K / 16) / s / 1e9;
    };
    printf("R=%d  rel err vs fp64: fp32-act %.2e · W4A8 %.2e  %s | GB/s 1 thread: fp32-act %.1f · W4A8 %.1f | 12 threads: fp32-act %.1f · W4A8 %.1f\n", R, r1, r2,
           ok ? "ok" : "FAIL", bench(false, 1), bench(true, 1), bench(false, 12), bench(true, 12));
  }
  printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
