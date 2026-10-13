// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// NVFP4 CPU kernels: AVX2 gemv == scalar reference (relative error at fp32 reassociation level) for single and multi-row, odd row
//   ranges, random scales (incl. zero / negative codes) and a real-sized expert (4096 × 2048). Also a negative control: swapping the
//   two 16-column scales of a 32-column block must change the result.
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>
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
  // _v2 == _ref bit for bit: GLM (4096/2048, NUMA halves 1024×4096 / 2048×2048) and FN (2560/640, halves 320×2560 / 1280×640) shapes,
  //   R = 1..8, odd row ranges (tails of the 4-row and 16-row groups), scale codes over the whole e4m3 range incl. zero / negative / subnormal,
  //   activations with outliers, zeros and negative zeros. Negative control: changing one block of activations changes the output bits.
  {
    long cmp = 0, diff = 0;
    for (auto [N, K] : {std::pair{1024, 4096}, {2048, 2048}, {320, 2560}, {1280, 640}, {37, 96}}) {
      std::vector<uint8_t> W((size_t)N * K / 2), S((size_t)N * K / 16);
      for (auto& b : W) b = rng() & 0xFF;
      for (auto& b : S) b = (uint8_t)(rng() & 0xFF);
      for (auto& b : S) if ((b & 0x7F) == 0x7F) b = 0x38;  // 0x7F / 0xFF = NaN in e4m3fn: not produced by the checkpoints
      std::normal_distribution<float> nd(0.f, 1.f);
      std::vector<std::vector<float>> as(8, std::vector<float>(K));
      for (auto& r : as) for (auto& v : r) { v = nd(rng) * (rng() % 64 == 0 ? 50.f : 1.f); if (rng() % 97 == 0) v = 0.f; if (rng() % 101 == 0) v = -0.f; }
      for (int R = 1; R <= 8; ++R)
        for (auto [n0, n1] : {std::pair{0, N}, {1, N - 2}, {5, std::min(N, 23)}}) {
          std::vector<std::vector<float>> y1(R, std::vector<float>(N, 7.f)), y2(R, std::vector<float>(N, 7.f));
          std::vector<const float*> ap; std::vector<float*> p1, p2;
          for (int r = 0; r < R; ++r) { ap.push_back(as[r].data()); p1.push_back(y1[r].data()); p2.push_back(y2[r].data()); }
          const float g = 0.0137f;
          gemv_nvfp4_rows_multi_ref(W.data(), S.data(), g, N, K, ap.data(), R, n0, n1, p1.data());
          gemv_nvfp4_rows_multi_v2(W.data(), S.data(), g, N, K, ap.data(), R, n0, n1, p2.data());
          for (int r = 0; r < R; ++r)
            for (int n = 0; n < N; ++n) { ++cmp; diff += memcmp(&y1[r][n], &y2[r][n], 4) != 0; }
          if (R == 1) {  // single-row entry too
            std::vector<float> s1(N, 7.f), s2(N, 7.f);
            gemv_nvfp4_rows_ref(W.data(), S.data(), g, N, K, as[0].data(), n0, n1, s1.data());
            gemv_nvfp4_rows_v2(W.data(), S.data(), g, N, K, as[0].data(), n0, n1, s2.data());
            for (int n = 0; n < N; ++n) { ++cmp; diff += memcmp(&s1[n], &s2[n], 4) != 0; }
          }
        }
      // negative control (the comparison can see a change)
      std::vector<float> b0(N), b1(N); std::vector<float> a2 = as[0]; for (int k = 0; k < 32; ++k) a2[K / 2 + k] += 0.25f;  // one block: some row with a nonzero weight there must change
      gemv_nvfp4_rows_v2(W.data(), S.data(), 1.f, N, K, as[0].data(), 0, N, b0.data());
      gemv_nvfp4_rows_v2(W.data(), S.data(), 1.f, N, K, a2.data(), 0, N, b1.data());
      EXPECT(memcmp(b0.data(), b1.data(), 4 * N) != 0, "negative control: changed activation must change the output (N %d K %d)", N, K);
    }
    EXPECT(diff == 0, "_v2 != _ref in %ld of %ld outputs", diff, cmp);
    printf("nvfp4 CPU: _v2 == _ref bitwise over %ld outputs (R 1..8, GLM/FN half shapes, odd ranges)\n", cmp);
  }
  if (fails) { fprintf(stderr, "nvfp4 CPU: %d failures\n", fails); return 1; }
  puts("nvfp4 CPU: AVX2 gemv == scalar reference · multi-row == single row (bitwise) · per-16 scales matter");
  return 0;
}
