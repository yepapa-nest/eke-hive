// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU expert kernel test — compared against a double scalar reference (same cast points). Random e2m1 weights, fp8 activations.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "hive/clock.h"
#include "hive/common.h"
#include "hive/expert_cpu.h"

using namespace hive;
using Floats = std::vector<float>;

static double ref_row(const std::vector<uint8_t>& W, const std::vector<uint8_t>& sw, int n, int K, const Floats& a, const Floats& sa) {
  double acc = 0;
  for (int b = 0; b < K / 32; ++b) {
    double d = 0;
    for (int i = 0; i < 32; ++i) {
      int k = b * 32 + i;
      uint8_t byte = W[(size_t)n * (K / 2) + k / 2];
      d += (double)a[k] * e2m1_to_f32((k & 1) ? (byte >> 4) : (byte & 0xF));
    }
    acc += d * sa[b] * e8m0_to_f32(sw[(size_t)n * (K / 32) + b]);
  }
  return acc;
}

int main() {
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> byte(0, 255);
  const int K = 5120, I = 2304;
  auto rnd_e4m3 = [&]() { uint8_t v; do { v = (uint8_t)byte(rng); } while ((v & 0x7F) == 0x7F || (v & 0x78) > 0x50); return v; };
  std::vector<uint8_t> w1((size_t)I * K / 2), w3((size_t)I * K / 2), w2((size_t)K * I / 2);
  std::vector<uint8_t> s1((size_t)I * K / 32), s3((size_t)I * K / 32), s2((size_t)K * I / 32);
  for (auto& v : w1) v = (uint8_t)byte(rng);
  for (auto& v : w3) v = (uint8_t)byte(rng);
  for (auto& v : w2) v = (uint8_t)byte(rng);
  for (auto& v : s1) v = (uint8_t)(127 - 10 + byte(rng) % 6);
  for (auto& v : s3) v = (uint8_t)(127 - 10 + byte(rng) % 6);
  for (auto& v : s2) v = (uint8_t)(127 - 10 + byte(rng) % 6);
  std::vector<uint8_t> aq(K), as(K / 32);
  for (auto& v : aq) v = rnd_e4m3();
  for (auto& v : as) v = (uint8_t)(127 - 6 + byte(rng) % 4);

  // 1) raw gemv vs reference
  std::vector<float> af(K), saf(K / 32);
  for (int k = 0; k < K; ++k) af[k] = e4m3_to_f32(aq[k]);
  for (int b = 0; b < K / 32; ++b) saf[b] = e8m0_to_f32(as[b]);
  std::vector<float> y(I);
  cpu::gemv_e2m1_rows(w1.data(), s1.data(), I, K, af.data(), saf.data(), 0, I, y.data());
  double maxrel = 0;
  for (int n = 0; n < I; ++n) {
    double r = ref_row(w1, s1, n, K, af, saf);
    maxrel = std::max(maxrel, std::fabs(y[n] - r) / (std::fabs(r) + 1e-2));
  }
  printf("gemv_e2m1_rows: maxrel %.3e %s\n", maxrel, maxrel < 1e-4 ? "ok" : "FAIL");

  // 2) full expert forward vs reference (same casts)
  cpu::ExpertDesc e{cpu::ExpertFormat::E2M1_B32, K, I, w1.data(), s1.data(), w2.data(), s2.data(), w3.data(), s3.data()};
  std::vector<float> out(K), tmp(4 * I + 2 * K + 64);
  const float route_w = 0.37f, limit = 10.f;
  auto t0 = std::chrono::steady_clock::now();
  cpu::expert_forward(e, aq.data(), as.data(), route_w, limit, out.data(), tmp.data());
  auto t1 = std::chrono::steady_clock::now();
  // reference
  std::vector<float> yq(I), sy(I / 32);
  for (int i = 0; i < I; ++i) {
    float g = bf2f(f2bf((float)ref_row(w1, s1, i, K, af, saf)));
    float u = bf2f(f2bf((float)ref_row(w3, s3, i, K, af, saf)));
    u = std::min(std::max(u, -limit), limit);
    g = std::min(g, limit);
    float v = g / (1.f + expf(-g)) * u * route_w;
    yq[i] = bf2f(f2bf(v));
  }
  for (int b = 0; b < I / 32; ++b) {
    float amax = 0;
    for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(yq[b * 32 + i]));
    amax = std::max(amax, 1e-4f);
    float sc = e8m0_to_f32(f32_ceil_pow2_e8m0(amax * (1.0f / 448.0f)));
    sy[b] = sc;
    for (int i = 0; i < 32; ++i) yq[b * 32 + i] = e4m3_to_f32(f32_to_e4m3_host(std::min(std::max(yq[b * 32 + i] / sc, -448.f), 448.f)));
  }
  double maxrel2 = 0; int bf_mismatch = 0;
  for (int k = 0; k < K; ++k) {
    double r = ref_row(w2, s2, k, I, yq, sy);
    float rb = bf2f(f2bf((float)r));
    maxrel2 = std::max(maxrel2, std::fabs(out[k] - rb) / (std::fabs(rb) + 1e-2));  // both are bf16, so they should match
    if (out[k] != rb) ++bf_mismatch;
  }
  double ms = hive::Millis(t1 - t0).count();
  printf("expert_forward: maxrel %.3e (bf16 mismatches %d/%d — rounding boundaries from accumulation order) %s · %.2f ms/expert single thread (%.1f GB/s)\n",
         maxrel2, bf_mismatch, K, maxrel2 < 1e-3 ? "ok" : "FAIL", ms, (w1.size() + w2.size() + w3.size() + s1.size() + s2.size() + s3.size()) / ms / 1e6);

  // 3) R-row gemv (rows of a batch decode that call the same expert) must be bit-identical to per-row single gemv (same accumulation order)
  int multi_mismatch = 0; double ms_multi = 0, ms_single = 0;
  for (int R = 2; R <= 8; R += 3) {
    std::vector<std::vector<float>> afs(R, std::vector<float>(K)), safs(R, std::vector<float>(K / 32)), ym(R, std::vector<float>(I)), ys(R, std::vector<float>(I));
    for (int r = 0; r < R; ++r) {
      for (int k = 0; k < K; ++k) afs[r][k] = e4m3_to_f32(rnd_e4m3());
      for (int b = 0; b < K / 32; ++b) safs[r][b] = e8m0_to_f32((uint8_t)(127 - 6 + byte(rng) % 4));
    }
    std::vector<const float*> ap(R), sp(R); std::vector<float*> yp(R);
    for (int r = 0; r < R; ++r) { ap[r] = afs[r].data(); sp[r] = safs[r].data(); yp[r] = ym[r].data(); }
    auto ta = std::chrono::steady_clock::now();
    cpu::gemv_e2m1_rows_multi(w1.data(), s1.data(), I, K, ap.data(), sp.data(), R, 0, I, yp.data());
    auto tb = std::chrono::steady_clock::now();
    for (int r = 0; r < R; ++r) cpu::gemv_e2m1_rows(w1.data(), s1.data(), I, K, ap[r], sp[r], 0, I, ys[r].data());
    auto tc = std::chrono::steady_clock::now();
    for (int r = 0; r < R; ++r)
      for (int n = 0; n < I; ++n) if (ym[r][n] != ys[r][n]) ++multi_mismatch;
    ms_multi += std::chrono::duration<double, std::milli>(tb - ta).count() / R;
    ms_single += std::chrono::duration<double, std::milli>(tc - tb).count() / R;
  }
  printf("gemv_e2m1_rows_multi (R=2,5,8): mismatches %d %s · %.3f ms per row (single %.3f ms)\n", multi_mismatch, multi_mismatch == 0 ? "ok" : "FAIL",
         ms_multi / 3, ms_single / 3);
  return (maxrel < 1e-4 && maxrel2 < 1e-3 && multi_mismatch == 0) ? 0 : 1;
}
