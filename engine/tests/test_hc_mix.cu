// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Check: hc_mix_rows (fused) vs the unfused path (hc_flatten_f32 + cuBLAS fp32 GEMM + hc_row_rsqrt) — both fp32, only the accumulation order differs.
// Second version: hc_mix_rows16 (16-row blocks sharing the W window in smem, default) vs hc_mix_rows4 (previous) vs cuBLAS vs host double (first 3 rows + last row)
//   + determinism (same input twice → bit-identical). M mixes values that are not multiples of 16 (1·7·17·300) to exercise the tail block.
//   Run: scripts/hive-run.sh "./build-dev/test_hc_mix"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/cublas_ops.h"
#include "hive/model_kernels.h"
#include "test_util.h"

using namespace hive;
static uint16_t f2bits(float f) { bf16 b = f2bf(f); uint16_t u; memcpy(&u, &b, 2); return u; }
static float bits2f(uint16_t b) { return hive::test::bf16_bits_to_f32(b); }

using Floats = std::vector<float>;
static double rel_err(const Floats& a, const Floats& b) {
  double scale = 1e-6, e = 0;
  for (const float v : a) scale = std::max(scale, (double)std::fabs(v));
  for (size_t i = 0; i < a.size(); i++) e = std::max(e, std::fabs((double)a[i] - b[i]) / scale);
  return e;
}
static double rsq_err(const Floats& a, const Floats& b) {
  double e = 0;
  for (size_t m = 0; m < a.size(); ++m) e = std::max(e, std::fabs((double)a[m] - b[m]) / std::fabs(a[m]));
  return e;
}

int main() {
  const int hc = 4, dim = 5120, hcdim = hc * dim, mix = (2 + hc) * hc;
  const float eps = 1e-20f;
  bool all_ok = true;
  Blas blas(0);
  for (const int M : {1, 7, 17, 300, 2048}) {
    std::mt19937 rng(3 + M);
    std::normal_distribution<float> nd;
    std::vector<uint16_t> h((size_t)M * hcdim);
    std::vector<float> W((size_t)mix * hcdim);
    for (auto& v : h) v = f2bits(nd(rng));
    for (auto& v : W) v = nd(rng) * 0.01f;
    bf16* dh; float *dW, *hflat, *mix_ref, *rsq_ref, *mix4, *rsq4, *mix16, *rsq16, *mix16b, *rsq16b;
    CUDA_CHECK(cudaMalloc(&dh, h.size() * 2)); CUDA_CHECK(cudaMalloc(&dW, W.size() * 4)); CUDA_CHECK(cudaMalloc(&hflat, h.size() * 4));
    CUDA_CHECK(cudaMalloc(&mix_ref, (size_t)M * mix * 4)); CUDA_CHECK(cudaMalloc(&mix4, (size_t)M * mix * 4));
    CUDA_CHECK(cudaMalloc(&mix16, (size_t)M * mix * 4)); CUDA_CHECK(cudaMalloc(&mix16b, (size_t)M * mix * 4));
    CUDA_CHECK(cudaMalloc(&rsq_ref, M * 4)); CUDA_CHECK(cudaMalloc(&rsq4, M * 4)); CUDA_CHECK(cudaMalloc(&rsq16, M * 4)); CUDA_CHECK(cudaMalloc(&rsq16b, M * 4));
    CUDA_CHECK(cudaMemcpy(dh, h.data(), h.size() * 2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size() * 4, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float ms_ref = 0, ms4 = 0, ms16 = 0;
    for (int rep = 0; rep < 2; ++rep) {
      cudaEventRecord(e0);
      k::hc_flatten_f32(dh, M, hcdim, hflat, 0);
      blas.gemm_f32(hflat, dW, mix_ref, M, mix, hcdim);
      k::hc_row_rsqrt(dh, M, hcdim, eps, rsq_ref, 0);
      cudaEventRecord(e1); CUDA_CHECK(cudaDeviceSynchronize()); cudaEventElapsedTime(&ms_ref, e0, e1);
      cudaEventRecord(e0);
      k::hc_mix_rows4(dh, dW, M, hcdim, mix, eps, mix4, rsq4, 0);
      cudaEventRecord(e1); CUDA_CHECK(cudaDeviceSynchronize()); cudaEventElapsedTime(&ms4, e0, e1);
      cudaEventRecord(e0);
      k::hc_mix_rows16(dh, dW, M, hcdim, mix, eps, rep == 0 ? mix16b : mix16, rep == 0 ? rsq16b : rsq16, 0);  // twice → determinism check
      cudaEventRecord(e1); CUDA_CHECK(cudaDeviceSynchronize()); cudaEventElapsedTime(&ms16, e0, e1);
    }
    std::vector<float> a((size_t)M * mix), b4((size_t)M * mix), b16((size_t)M * mix), b16b((size_t)M * mix), ra(M), r4(M), r16(M), r16b(M);
    CUDA_CHECK(cudaMemcpy(a.data(), mix_ref, a.size() * 4, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(b4.data(), mix4, b4.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b16.data(), mix16, b16.size() * 4, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(b16b.data(), mix16b, b16b.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ra.data(), rsq_ref, M * 4, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(r4.data(), rsq4, M * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r16.data(), rsq16, M * 4, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(r16b.data(), rsq16b, M * 4, cudaMemcpyDeviceToHost));
    const double e4 = rel_err(a, b4), e16 = rel_err(a, b16), e416 = rel_err(b4, b16), er4 = rsq_err(ra, r4), er16 = rsq_err(ra, r16);
    const bool det = memcmp(b16.data(), b16b.data(), b16.size() * 4) == 0 && memcmp(r16.data(), r16b.data(), (size_t)M * 4) == 0;
    // host double reference: rows 0..2 and the last row (tail block)
    double eh = 0, erh = 0, sh = 1e-6;
    for (const int m : {0, 1, 2, M - 1}) {
      if (m < 0 || m >= M) continue;
      double ss = 0;
      for (int kk = 0; kk < hcdim; ++kk) { const double x = bits2f(h[(size_t)m * hcdim + kk]); ss += x * x; }
      const double rs = 1.0 / std::sqrt(ss / hcdim + eps);
      erh = std::max(erh, std::fabs(rs - r16[m]) / rs);
      for (int j = 0; j < mix; ++j) {
        double d = 0;
        for (int kk = 0; kk < hcdim; ++kk) d += (double)bits2f(h[(size_t)m * hcdim + kk]) * W[(size_t)j * hcdim + kk];
        sh = std::max(sh, std::fabs(d));
        eh = std::max(eh, std::fabs(d - b16[(size_t)m * mix + j]));
      }
    }
    eh /= sh;
    const bool ok = e4 < 1e-4 && e16 < 1e-4 && e416 < 1e-4 && er4 < 1e-5 && er16 < 1e-5 && eh < 1e-4 && erh < 1e-5 && det;
    printf("M=%-5d vs cuBLAS: rows4 %.2e rows16 %.2e | rows4↔16 %.2e | rsq %.2e/%.2e | host-double mix %.2e rsq %.2e | det %s | ref %.3f ms  rows4 %.3f ms  rows16 %.3f ms  %s\n",
           M, e4, e16, e416, er4, er16, eh, erh, det ? "ok" : "BAD", ms_ref, ms4, ms16, ok ? "PASS" : "FAIL");
    all_ok = all_ok && ok;
    cudaFree(dh); cudaFree(dW); cudaFree(hflat); cudaFree(mix_ref); cudaFree(mix4); cudaFree(mix16); cudaFree(mix16b);
    cudaFree(rsq_ref); cudaFree(rsq4); cudaFree(rsq16); cudaFree(rsq16b);
  }
  printf(all_ok ? "ALL PASS\n" : "SOME FAIL\n");
  return all_ok ? 0 : 1;
}
