// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// D-2 check: indexer_scores_tc (e2m1×e2m1 tensor cores) vs indexer_scores (CUDA cores, round-tripped bf16 queries) vs a host double reference.
//   Queries and keys are built with fp4_pack/fp4_quant_roundtrip as in the real path. M 37 (16·2+5 — cuts a warp tile) · T 203 (cuts a 32 tile) · visible differs per row.
//   Pass = same -inf positions + relative error of finite values (relative to the row max |reference|) < 2e-2 for both GPU paths (allows 1-ulp flips at per-head bf16 rounding boundaries).
//   Run: scripts/hive-run.sh "./build/test_idx_tc"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/kernels.h"
#include "hive/kv_pack.h"
#include "hive/model_kernels.h"
#include "test_util.h"

using namespace hive;
static float bits2f(uint16_t b) { return hive::test::bf16_bits_to_f32(b); }
static uint16_t f2bits(float f) { bf16 b = f2bf(f); uint16_t u; memcpy(&u, &b, 2); return u; }
static float bfr(double v) { return bits2f(f2bits((float)v)); }
static double pval(const uint8_t* row, int d) {
  const int nib = (row[kvp::IDX_HDR + (d >> 1)] >> (4 * (d & 1))) & 0xF;
  return (double)e2m1_to_f32(nib) * e8m0_to_f32(row[d >> 5]);
}

int main() {
  bool all_ok = true;
  for (const int M : {1, 9, 37, 130}) {
    const int Hi = 64, Di = kvp::IDX_D, T = 203, R = kvp::IDX_ROW;
    std::mt19937 rng(11 + M);
    std::normal_distribution<float> nd;
    std::vector<uint16_t> Q((size_t)M * Hi * Di), K((size_t)T * Di), W((size_t)M * Hi);
    for (auto& v : Q) v = f2bits(nd(rng) * 2.f);
    for (auto& v : K) v = f2bits(nd(rng));
    for (auto& v : W) v = f2bits(nd(rng) * 0.02f);
    std::vector<int32_t> vis(M);
    for (int m = 0; m < M; ++m) vis[m] = (m == 0) ? T : (int)(rng() % (T + 1));
    bf16 *dQ, *dK, *dW, *s_ref, *s_tc; uint8_t *dQp, *dKp; int32_t* dV;
    CUDA_CHECK(cudaMalloc(&dQ, Q.size() * 2)); CUDA_CHECK(cudaMalloc(&dK, K.size() * 2)); CUDA_CHECK(cudaMalloc(&dW, W.size() * 2));
    CUDA_CHECK(cudaMalloc(&dQp, (size_t)M * Hi * R)); CUDA_CHECK(cudaMalloc(&dKp, (size_t)T * R)); CUDA_CHECK(cudaMalloc(&dV, M * 4));
    CUDA_CHECK(cudaMalloc(&s_ref, (size_t)M * T * 2)); CUDA_CHECK(cudaMalloc(&s_tc, (size_t)M * T * 2));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dV, vis.data(), M * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dQp, 0, (size_t)M * Hi * R)); CUDA_CHECK(cudaMemset(dKp, 0, (size_t)T * R));
    k::fp4_pack(dK, T, Di, 32, false, dKp, R, 0);
    k::fp4_pack(dQ, M * Hi, Di, 32, false, dQp, R, 0);          // same order as runtime: pack before the round trip
    k::fp4_quant_roundtrip(dQ, M, Hi * Di, 32, false, 0);
    k::indexer_scores(dQ, dKp, dW, M, Hi, Di, T, dV, s_ref, 0);
    k::indexer_scores_tc(dQp, dKp, dW, M, Hi, T, dV, s_tc, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> Qp((size_t)M * Hi * R), Kp((size_t)T * R);
    std::vector<uint16_t> Qr(Q.size()), h_ref((size_t)M * T), h_tc((size_t)M * T);
    CUDA_CHECK(cudaMemcpy(Qp.data(), dQp, Qp.size(), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(Kp.data(), dKp, Kp.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Qr.data(), dQ, Qr.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), s_ref, h_ref.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tc.data(), s_tc, h_tc.size() * 2, cudaMemcpyDeviceToHost));
    // decoded values of the packed queries == round-tripped bf16 queries (must be bit-identical so both paths get the same input)
    size_t qmis = 0;
    for (int rr = 0; rr < M * Hi; ++rr)
      for (int d = 0; d < Di; ++d)
        if ((float)pval(&Qp[(size_t)rr * R], d) != bits2f(Qr[(size_t)rr * Di + d])) ++qmis;
    double e_ref = 0, e_tc = 0; size_t inf_mis = 0;
    for (int m = 0; m < M; ++m) {
      std::vector<double> host(T, 0.0); double scale = 1e-6;
      for (int t = 0; t < vis[m]; ++t) {
        double tot = 0;
        for (int h = 0; h < Hi; ++h) {
          double d = 0;
          for (int dd = 0; dd < Di; ++dd) d += pval(&Qp[((size_t)m * Hi + h) * R], dd) * pval(&Kp[(size_t)t * R], dd);
          double sb = bfr(d); sb = sb > 0 ? sb : 0; sb = bfr(sb * bits2f(W[(size_t)m * Hi + h]));
          tot += sb;
        }
        host[t] = tot; scale = std::max(scale, std::fabs(tot));
      }
      for (int t = 0; t < T; t++) {
        const float a = bits2f(h_ref[(size_t)m * T + t]), b = bits2f(h_tc[(size_t)m * T + t]);
        const bool ia = std::isinf(a) && a < 0, ib = std::isinf(b) && b < 0, ih = t >= vis[m];
        if (ia != ih || ib != ih) { ++inf_mis; continue; }
        if (ih) continue;
        e_ref = std::max(e_ref, std::fabs(a - host[t]) / scale);
        e_tc = std::max(e_tc, std::fabs(b - host[t]) / scale);
      }
    }
    const bool ok = qmis == 0 && inf_mis == 0 && e_ref < 2e-2 && e_tc < 2e-2;
    printf("M=%-4d q_mismatch=%zu inf_mismatch=%zu err(cuda-core)=%.3e err(tc)=%.3e  %s\n", M, qmis, inf_mis, e_ref, e_tc, ok ? "PASS" : "FAIL");
    all_ok = all_ok && ok;
    cudaFree(dQ); cudaFree(dK); cudaFree(dW); cudaFree(dQp); cudaFree(dKp); cudaFree(dV); cudaFree(s_ref); cudaFree(s_tc);
  }
  printf(all_ok ? "ALL PASS\n" : "SOME FAIL\n");
  return all_ok ? 0 : 1;
}
