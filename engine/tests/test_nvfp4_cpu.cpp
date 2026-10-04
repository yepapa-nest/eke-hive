// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// NVFP4 CPU kernels: AVX2 gemv == scalar reference (relative error at fp32 reassociation level) for single and multi-row, odd row
//   ranges, random scales (incl. zero / negative codes) and a real-sized expert (4096 × 2048). Also a negative control: swapping the
//   two 16-column scales of a 32-column block must change the result.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "hive/expert_cpu.h"

using namespace hive::cpu;
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static double rel(const std::vector<float>& a, const std::vector<float>& b, int n0, int n1) {
  double num = 0, den = 0;
  for (int i = n0; i < n1; ++i) { num += (a[i] - b[i]) * (double)(a[i] - b[i]); den += (double)b[i] * b[i]; }
  return std::sqrt(num / (den + 1e-30));
}

int main() {
  std::mt19937 rng(7);
  for (auto [N, K] : {std::pair{37, 64}, {2048, 4096}, {4096, 2048}}) {
    std::vector<uint8_t> W((size_t)N * K / 2), S((size_t)N * K / 16);
    for (auto& b : W) b = rng() & 0xFF;
    for (auto& b : S) b = (uint8_t)(0x20 + rng() % 0x40);  // positive e4m3 scales 2^-4 .. ~2^3
    S[0] = 0;                                                // a zero scale
    std::vector<float> a(K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : a) v = nd(rng);
    std::vector<float> y(N, 0.f), yr(N, 0.f);
    const float g = 0.0137f;
    gemv_nvfp4_rows(W.data(), S.data(), g, N, K, a.data(), 3, N - 1, y.data());
    gemv_nvfp4_rows_scalar(W.data(), S.data(), g, N, K, a.data(), 3, N - 1, yr.data());
    const double e = rel(y, yr, 3, N - 1);
    EXPECT(e < 1e-5, "N %d K %d rel err %.3g", N, K, e);
    EXPECT(y[0] == 0.f && y[N - 1] == 0.f, "row range respected");
    // multi-row == single row (same accumulation order per row)
    const int R = 5;
    std::vector<std::vector<float>> as(R, std::vector<float>(K)), ys(R, std::vector<float>(N, 0.f));
    std::vector<const float*> ap; std::vector<float*> yp;
    for (int r = 0; r < R; ++r) { for (auto& v : as[r]) v = nd(rng); ap.push_back(as[r].data()); yp.push_back(ys[r].data()); }
    gemv_nvfp4_rows_multi(W.data(), S.data(), g, N, K, ap.data(), R, 0, N, yp.data());
    for (int r = 0; r < R; ++r) {
      std::vector<float> one(N);
      gemv_nvfp4_rows(W.data(), S.data(), g, N, K, as[r].data(), 0, N, one.data());
      bool same = true; for (int n = 0; n < N; ++n) same &= one[n] == ys[r][n];
      EXPECT(same, "multi row %d == single (bitwise)", r);
    }
    // negative control: the scale of the second 16 columns matters
    std::vector<uint8_t> S2 = S; for (int n = 0; n < N; ++n) std::swap(S2[(size_t)n * (K / 16) + 0], S2[(size_t)n * (K / 16) + 1]);
    std::vector<float> ys2(N);
    gemv_nvfp4_rows(W.data(), S2.data(), g, N, K, a.data(), 0, N, ys2.data());
    std::vector<float> y0(N); gemv_nvfp4_rows(W.data(), S.data(), g, N, K, a.data(), 0, N, y0.data());
    EXPECT(rel(ys2, y0, 0, N) > 1e-3, "swapped 16-column scales must change the result");
  }
  if (fails) { fprintf(stderr, "nvfp4 CPU: %d failures\n", fails); return 1; }
  puts("nvfp4 CPU: AVX2 gemv == scalar reference · multi-row == single row (bitwise) · per-16 scales matter");
  return 0;
}
