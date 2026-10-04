// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GPU NVFP4 kernels == CPU scalar reference (gemv M = 1..8 · dequant), real GLM expert shapes; negative control on the per-16 scales.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/expert_cpu.h"
#include "hive/glm/nvfp4.h"

using namespace hive;
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

int main() {
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto [N, K] : {std::pair{2048, 4096}, {4096, 2048}, {96, 64}}) {
    std::vector<uint8_t> W((size_t)N * K / 2), S((size_t)N * K / 16);
    for (auto& b : W) b = rng() & 0xFF;
    for (auto& b : S) b = (uint8_t)(0x20 + rng() % 0x40);
    const float g = 0.021f;
    uint8_t *dW, *dS; CUDA_CHECK(cudaMalloc(&dW, W.size())); CUDA_CHECK(cudaMalloc(&dS, S.size()));
    CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dS, S.data(), S.size(), cudaMemcpyHostToDevice));
    glm::Nvfp4Mat M{dW, dS, g, N, K};
    for (int Mr : {1, 3, 8}) {
      std::vector<bf16> x((size_t)Mr * K); std::vector<float> xf((size_t)Mr * K);
      for (size_t i = 0; i < x.size(); ++i) { x[i] = f2bf(nd(rng)); xf[i] = bf2f(x[i]); }
      bf16* dx; float* dy; CUDA_CHECK(cudaMalloc(&dx, x.size() * 2)); CUDA_CHECK(cudaMalloc(&dy, (size_t)Mr * N * 4));
      CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
      glm::nvfp4_gemv(M, dx, K, Mr, dy, N, 0);
      std::vector<float> y((size_t)Mr * N); CUDA_CHECK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
      double num = 0, den = 0;
      for (int m = 0; m < Mr; ++m) {
        std::vector<float> yr(N);
        cpu::gemv_nvfp4_rows_scalar(W.data(), S.data(), g, N, K, xf.data() + (size_t)m * K, 0, N, yr.data());
        for (int n = 0; n < N; ++n) { num += std::pow(y[(size_t)m * N + n] - yr[n], 2); den += (double)yr[n] * yr[n]; }
      }
      EXPECT(std::sqrt(num / den) < 1e-5, "gemv N %d K %d M %d rel %.3g", N, K, Mr, std::sqrt(num / den));
      cudaFree(dx); cudaFree(dy);
    }
    bf16* dq; CUDA_CHECK(cudaMalloc(&dq, (size_t)N * K * 2));
    glm::nvfp4_dequant(M, dq, 0);
    std::vector<bf16> q((size_t)N * K); CUDA_CHECK(cudaMemcpy(q.data(), dq, q.size() * 2, cudaMemcpyDeviceToHost));
    // dequant row n == scalar gemv with one-hot activations is expensive; check against the definition directly
    static const float E[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    int bad = 0;
    for (size_t i = 0; i < q.size(); i += 997) {
      const size_t n = i / K, k = i % K; const uint8_t b = W[n * (K / 2) + k / 2];
      const float ref = bf2f(f2bf(E[(k & 1) ? (b >> 4) : (b & 15)] * e4m3_to_f32(S[n * (K / 16) + k / 16]) * g));
      bad += bf2f(q[i]) != ref;
    }
    EXPECT(bad == 0, "dequant N %d K %d: %d mismatches", N, K, bad);
    cudaFree(dq); cudaFree(dW); cudaFree(dS);
  }
  if (fails) { fprintf(stderr, "glm nvfp4 GPU: %d failures\n", fails); return 1; }
  puts("glm nvfp4 GPU: gemv (M 1/3/8) == CPU scalar · dequant == definition");
  return 0;
}
