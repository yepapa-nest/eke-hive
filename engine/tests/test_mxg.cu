// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Validates gemm_bs_mx_tiled (block-scaled tiled tensor-core GEMM, fp8 · fp4 B) vs gemm_bs_tc (bf16 WMMA) vs a host double reference.
//   Shapes cut across tile boundaries (M 37·300·2048 · N 136(=128+8) · K 192(=64·3)). Pass = both GPU paths within relative 1e-2 (bf16) / 1e-5 (f32) of the reference.
//   Speed sample: real shapes (shared expert w1 2048×2304×5120 fp8 · expert w1 2048×2304×5120 fp4 · wq_b 2048×32768×1280 fp8) compared with tc.
//   Run: scripts/hive-run.sh "./build-dev/test_mxg"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/kernels.h"
#include "test_util.h"

using namespace hive;
static float bits2f(uint16_t b) { return hive::test::bf16_bits_to_f32(b); }

static bool check(int M, int N, int K, bool fp4, unsigned seed) {
  const int nb = K / 32;
  std::mt19937 rng(seed);
  auto rnd = [&](int lo, int hi) { return (uint8_t)(lo + rng() % (hi - lo + 1)); };
  std::vector<uint8_t> W((size_t)N * (fp4 ? K / 2 : K)), SW((size_t)N * nb), A((size_t)M * K), SA((size_t)M * nb);
  for (auto& v : W) { do { v = rnd(0, 255); } while (!fp4 && ((v & 0x7F) == 0x7F || (v & 0x78) > 0x48)); }
  for (auto& v : SW) v = rnd(125, 129);
  for (auto& v : A) { do { v = rnd(0, 255); } while ((v & 0x7F) == 0x7F || (v & 0x78) > 0x48); }
  for (auto& v : SA) v = rnd(125, 129);
  std::vector<double> ref((size_t)M * N, 0.0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      for (int b = 0; b < nb; ++b) {
        double d = 0;
        for (int i = 0; i < 32; ++i) {
          const int k = b * 32 + i;
          const double wv = fp4 ? e2m1_to_f32((k & 1) ? (W[(size_t)n * K / 2 + k / 2] >> 4) : (W[(size_t)n * K / 2 + k / 2] & 0xF)) : e4m3_to_f32(W[(size_t)n * K + k]);
          d += (double)e4m3_to_f32(A[(size_t)m * K + k]) * wv;
        }
        ref[(size_t)m * N + n] += d * e8m0_to_f32(SA[(size_t)m * nb + b]) * e8m0_to_f32(SW[(size_t)n * nb + b]);
      }
  uint8_t *dW, *dSW, *dA, *dSA; bf16 *o_tc, *o_mx; float* f_mx;
  CUDA_CHECK(cudaMalloc(&dW, W.size())); CUDA_CHECK(cudaMalloc(&dSW, SW.size())); CUDA_CHECK(cudaMalloc(&dA, A.size())); CUDA_CHECK(cudaMalloc(&dSA, SA.size()));
  CUDA_CHECK(cudaMalloc(&o_tc, (size_t)M * N * 2)); CUDA_CHECK(cudaMalloc(&o_mx, (size_t)M * N * 2)); CUDA_CHECK(cudaMalloc(&f_mx, (size_t)M * N * 4));
  CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dSW, SW.data(), SW.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dSA, SA.data(), SA.size(), cudaMemcpyHostToDevice));
  k::gemm_bs_tc(dA, dSA, dW, dSW, fp4, M, N, K, o_tc, nullptr, 0);
  k::gemm_bs_mx_tiled(dA, dSA, dW, dSW, fp4, M, N, K, o_mx, f_mx, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> h_tc((size_t)M * N), h_mx((size_t)M * N);
  std::vector<float> h_f((size_t)M * N);
  CUDA_CHECK(cudaMemcpy(h_tc.data(), o_tc, h_tc.size() * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_mx.data(), o_mx, h_mx.size() * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_f.data(), f_mx, h_f.size() * 4, cudaMemcpyDeviceToHost));
  double e_tc = 0, e_mx = 0, e_f32 = 0, scale = 1e-6;
  for (const double r : ref) scale = std::max(scale, std::fabs(r));
  for (size_t i = 0; i < ref.size(); i++) {
    e_tc = std::max(e_tc, std::fabs(bits2f(h_tc[i]) - ref[i]) / scale);
    e_mx = std::max(e_mx, std::fabs(bits2f(h_mx[i]) - ref[i]) / scale);
    e_f32 = std::max(e_f32, std::fabs(h_f[i] - ref[i]) / scale);
  }
  const bool ok = e_mx < 1e-2 && e_f32 < 1e-5;
  printf("M=%-5d N=%-4d K=%-4d %s  err tc %.2e · tiled bf16 %.2e · tiled f32 %.2e  %s\n", M, N, K, fp4 ? "fp4" : "fp8", e_tc, e_mx, e_f32, ok ? "PASS" : "FAIL");
  cudaFree(dW); cudaFree(dSW); cudaFree(dA); cudaFree(dSA); cudaFree(o_tc); cudaFree(o_mx); cudaFree(f_mx);
  return ok;
}

static void speed(int M, int N, int K, bool fp4) {
  const int nb = K / 32;
  uint8_t *dW, *dSW, *dA, *dSA; bf16* o;
  CUDA_CHECK(cudaMalloc(&dW, (size_t)N * (fp4 ? K / 2 : K))); CUDA_CHECK(cudaMalloc(&dSW, (size_t)N * nb)); CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K));
  CUDA_CHECK(cudaMalloc(&dSA, (size_t)M * nb)); CUDA_CHECK(cudaMalloc(&o, (size_t)M * N * 2));
  CUDA_CHECK(cudaMemset(dW, 0x22, (size_t)N * (fp4 ? K / 2 : K))); CUDA_CHECK(cudaMemset(dSW, 127, (size_t)N * nb));
  CUDA_CHECK(cudaMemset(dA, 0x30, (size_t)M * K)); CUDA_CHECK(cudaMemset(dSA, 127, (size_t)M * nb));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  float ms_tc = 0, ms_mx = 0;
  for (int rep = 0; rep < 3; ++rep) {
    cudaEventRecord(e0); k::gemm_bs_tc(dA, dSA, dW, dSW, fp4, M, N, K, o, nullptr, 0); cudaEventRecord(e1); CUDA_CHECK(cudaDeviceSynchronize()); cudaEventElapsedTime(&ms_tc, e0, e1);
    cudaEventRecord(e0); k::gemm_bs_mx_tiled(dA, dSA, dW, dSW, fp4, M, N, K, o, nullptr, 0); cudaEventRecord(e1); CUDA_CHECK(cudaDeviceSynchronize()); cudaEventElapsedTime(&ms_mx, e0, e1);
  }
  const double flop = 2.0 * M * N * K;
  printf("speed M=%d N=%d K=%d %s: tc %.3f ms (%.0f TFLOPS) · tiled %.3f ms (%.0f TFLOPS) · ×%.1f\n", M, N, K, fp4 ? "fp4" : "fp8", ms_tc, flop / ms_tc / 1e9, ms_mx,
         flop / ms_mx / 1e9, ms_tc / ms_mx);
  cudaFree(dW); cudaFree(dSW); cudaFree(dA); cudaFree(dSA); cudaFree(o);
}

int main() {
  bool ok = true;
  ok &= check(37, 136, 192, false, 1);
  ok &= check(37, 136, 192, true, 2);
  ok &= check(300, 256, 320, false, 3);
  ok &= check(300, 256, 320, true, 4);
  ok &= check(2048, 136, 128, true, 5);
  ok &= check(9, 8, 64, false, 6);
  speed(2048, 2304, 5120, false);
  speed(2048, 2304, 5120, true);
  speed(2048, 32768, 1280, false);
  speed(16384, 2304, 5120, true);
  printf(ok ? "ALL PASS\n" : "SOME FAIL\n");
  return ok ? 0 : 1;
}
