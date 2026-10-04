// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Check: fp32 indexer scores (HIVE_IDX_F32) — ① fp32 version vs host double (formula without per-head rounding), relative error < 1e-4 ② top-512 selection overlap with the bf16 version
//   (reference formula; informational — quality is judged by a real-model A/B) ③ the bf16 version matches host double (reference formula: bf16 rounding per head) (regression guard).
//   Run: scripts/hive-run.sh "./build-dev/test_idx_f32"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "hive/common.h"
#include "hive/kernels.h"
#include "hive/kv_pack.h"
#include "hive/model_kernels.h"
#include "test_util.h"

using namespace hive;
static uint16_t f2bits(float f) { bf16 b = f2bf(f); uint16_t u; memcpy(&u, &b, 2); return u; }
static float bits2f(uint16_t b) { return hive::test::bf16_bits_to_f32(b); }
static float bfr(double v) { return bits2f(f2bits((float)v)); }

int main() {
  const int M = 4, T = 5000, Hi = 32, Di = kvp::IDX_D, R = kvp::IDX_ROW, K = 512;
  std::mt19937 rng(29);
  std::normal_distribution<float> nd;
  std::vector<uint16_t> Q((size_t)M * Hi * Di), Kb((size_t)T * Di), W((size_t)M * Hi);
  for (auto& v : Q) v = f2bits(nd(rng) * 2.f);
  for (auto& v : Kb) v = f2bits(nd(rng));
  for (auto& v : W) v = f2bits(nd(rng) * 0.02f);
  std::vector<int32_t> vis(M, T);
  bf16 *dQ, *dK, *dW, *sb; uint8_t* dKp; int32_t *dV, *sel_b, *sel_f; float *sbf, *sf;
  CUDA_CHECK(cudaMalloc(&dQ, Q.size() * 2)); CUDA_CHECK(cudaMalloc(&dK, Kb.size() * 2)); CUDA_CHECK(cudaMalloc(&dW, W.size() * 2));
  CUDA_CHECK(cudaMalloc(&dKp, (size_t)T * R)); CUDA_CHECK(cudaMalloc(&dV, M * 4)); CUDA_CHECK(cudaMalloc(&sb, (size_t)M * T * 2));
  CUDA_CHECK(cudaMalloc(&sbf, (size_t)M * T * 4)); CUDA_CHECK(cudaMalloc(&sf, (size_t)M * T * 4));
  CUDA_CHECK(cudaMalloc(&sel_b, (size_t)M * K * 4)); CUDA_CHECK(cudaMalloc(&sel_f, (size_t)M * K * 4));
  CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dK, Kb.data(), Kb.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dV, vis.data(), M * 4, cudaMemcpyHostToDevice));
  k::fp4_pack(dK, T, Di, 32, false, dKp, R, 0);
  k::fp4_quant_roundtrip(dQ, M, Hi * Di, 32, false, 0);
  k::indexer_scores(dQ, dKp, dW, M, Hi, Di, T, dV, sb, 0);
  k::indexer_scores_f32(dQ, dKp, dW, M, Hi, Di, T, dV, sf, 0);
  k::bf16_rows_to_f32(sb, M * T, sbf, 0);
  k::topk_select_rows(sbf, M, T, K, T, sel_b, K, 0);
  k::topk_select_rows(sf, M, T, K, T, sel_f, K, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> hq(Q.size()), hb((size_t)M * T); std::vector<float> hf((size_t)M * T); std::vector<uint8_t> kp((size_t)T * R);
  std::vector<int32_t> selb((size_t)M * K), self((size_t)M * K);
  CUDA_CHECK(cudaMemcpy(hq.data(), dQ, hq.size() * 2, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(kp.data(), dKp, kp.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb.data(), sb, hb.size() * 2, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hf.data(), sf, hf.size() * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(selb.data(), sel_b, selb.size() * 4, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(self.data(), sel_f, self.size() * 4, cudaMemcpyDeviceToHost));
  auto kval = [&](int t, int d) { const uint8_t* row = &kp[(size_t)t * R]; const int nib = (row[kvp::IDX_HDR + d / 2] >> (4 * (d & 1))) & 0xF;
                                  return (double)e2m1_to_f32(nib) * e8m0_to_f32(row[d >> 5]); };
  double e_f = 0, e_b = 0, sc = 1e-9; size_t ties_b = 0;
  for (int m = 0; m < M; ++m)
    for (int t = 0; t < T; t += 7) {
      double tf = 0, tb = 0;
      for (int h = 0; h < Hi; ++h) {
        double d = 0;
        for (int dd = 0; dd < Di; ++dd) d += (double)bits2f(hq[((size_t)m * Hi + h) * Di + dd]) * kval(t, dd);
        const double w = bits2f(W[(size_t)m * Hi + h]);
        tf += std::max(d, 0.0) * w;
        double b = bfr(d); b = b > 0 ? b : 0; tb += bfr(b * w);
      }
      sc = std::max(sc, std::fabs(tf));
      e_f = std::max(e_f, std::fabs(hf[(size_t)m * T + t] - tf));
      e_b = std::max(e_b, (double)std::fabs(bits2f(hb[(size_t)m * T + t]) - bfr(tb)));
    }
  for (int m = 0; m < M; ++m) {  // tie count of the bf16 scores (number of scores equal to the 512th value — more ties make the selection more arbitrary)
    std::vector<float> v(T); for (int t = 0; t < T; ++t) v[t] = bits2f(hb[(size_t)m * T + t]);
    std::nth_element(v.begin(), v.begin() + K - 1, v.end(), std::greater<float>());
    const float kth = v[K - 1]; for (int t = 0; t < T; ++t) ties_b += bits2f(hb[(size_t)m * T + t]) == kth;
  }
  size_t overlap = 0;
  for (int m = 0; m < M; ++m) {
    std::set<int> a(selb.begin() + (size_t)m * K, selb.begin() + (size_t)(m + 1) * K);
    for (int i = 0; i < K; ++i) overlap += a.count(self[(size_t)m * K + i]);
  }
  const bool ok = e_f / sc < 1e-4 && e_b / sc < 2e-2;
  printf("fp32 vs host |Δ|/max %.2e · bf16 vs host (bf16 formula) %.2e · bf16 ties at the 512th value %zu (rows %d) · top-%d overlap %.1f%%  %s\n", e_f / sc, e_b / sc, ties_b, M, K,
         100.0 * overlap / (M * K), ok ? "PASS" : "FAIL");
  return !ok;
}
