// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// fp8b_gemv / fp8b_dequant (e4m3 + 128×128 fp32 block scales) vs a CPU double reference; shapes of the GLM DSA/shared-expert matrices.
// Negative control: -DFP8B_NEGCTRL swaps the scale-block index (n/128 ↔ k/128) in the reference — the comparison must then fail.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <cuda_fp8.h>

#include "hive/common.h"
#include "hive/glm/fp8b.h"

using namespace hive;
static int fails = 0;

static float e4m3(uint8_t v) { __nv_fp8_e4m3 t; t.__x = v; return float(t); }

int main() {
  std::mt19937 rng(7);
  struct Shape { int N, K; } shapes[] = {{1536, 4096}, {16384, 1536}, {512, 4096}, {4096, 16384}, {2048, 4096}, {4096, 2048}, {200, 384}};
  for (auto sh : shapes) {
    const int N = sh.N, K = sh.K, nb = (N + 127) / 128, kb = (K + 127) / 128;
    std::vector<uint8_t> w((size_t)N * K);
    for (auto& v : w) { do v = rng() & 0xFF; while ((v & 0x7F) == 0x7F); }  // skip NaN codes
    std::vector<float> s((size_t)nb * kb);
    std::uniform_real_distribution<float> us(1e-4f, 3e-3f);
    for (auto& v : s) v = us(rng);
    for (int M : {1, 3, 8}) {
      std::vector<__nv_bfloat16> x((size_t)M * K);
      std::normal_distribution<float> nd(0.f, 1.f);
      for (auto& v : x) v = __float2bfloat16(nd(rng));
      uint8_t* dw; float* ds; __nv_bfloat16 *dx, *dy;
      CUDA_CHECK(cudaMalloc(&dw, w.size())); CUDA_CHECK(cudaMalloc(&ds, s.size() * 4));
      CUDA_CHECK(cudaMalloc(&dx, x.size() * 2)); CUDA_CHECK(cudaMalloc(&dy, (size_t)M * N * 2));
      CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(ds, s.data(), s.size() * 4, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
      glm::Fp8BMat W{dw, ds, N, K};
      glm::fp8b_gemv(W, dx, K, M, dy, N, 0);
      std::vector<__nv_bfloat16> y((size_t)M * N);
      CUDA_CHECK(cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost));
      double num = 0, den = 0;
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          double r = 0;
          for (int k = 0; k < K; ++k) {
#ifdef FP8B_NEGCTRL
            const double sc = s[(size_t)(k / 128 % nb) * kb + (n / 128 % kb)];
#else
            const double sc = s[(size_t)(n / 128) * kb + k / 128];
#endif
            r += (double)e4m3(w[(size_t)n * K + k]) * sc * __bfloat162float(x[(size_t)m * K + k]);
          }
          const double a = __bfloat162float(y[(size_t)m * N + n]);
          num += (a - r) * (a - r); den += r * r;
        }
      const double rel = std::sqrt(num / den);
      const bool ok = rel < 5e-3;
      if (!ok) ++fails;
      printf("gemv    N=%5d K=%5d M=%d  rel %.2e  %s\n", N, K, M, rel, ok ? "ok" : "FAIL");
      if (M == 1) {
        __nv_bfloat16* dq; CUDA_CHECK(cudaMalloc(&dq, (size_t)N * K * 2));
        glm::fp8b_dequant(W, dq, 0);
        std::vector<__nv_bfloat16> q((size_t)N * K);
        CUDA_CHECK(cudaMemcpy(q.data(), dq, q.size() * 2, cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (int n = 0; n < N; ++n)
          for (int k = 0; k < K; ++k) {
            const float ref = __bfloat162float(__float2bfloat16(e4m3(w[(size_t)n * K + k]) * s[(size_t)(n / 128) * kb + k / 128]));
            bad += __bfloat162float(q[(size_t)n * K + k]) != ref;
          }
        if (bad) ++fails;
        printf("dequant N=%5d K=%5d      mismatches %zu  %s\n", N, K, bad, bad ? "FAIL" : "ok");
        // timing (L2-cold-ish: matrix ≥ 6 MB, repeated)
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int i = 0; i < 5; ++i) glm::fp8b_gemv(W, dx, K, 1, dy, N, 0);
        cudaEventRecord(e0);
        for (int i = 0; i < 50; ++i) glm::fp8b_gemv(W, dx, K, 1, dy, N, 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 50;
        printf("        time M=1 %.4f ms  %.0f GB/s\n", ms, (double)N * K / ms / 1e6);
        cudaFree(dq);
      }
      cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dy);
    }
  }
  printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
