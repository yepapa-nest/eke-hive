// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// mx_grouped_w2 vs CUDA-core gemv_grouped_w2 vs a host double reference — a small shape tells which one is wrong.
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

int main() {
  const int K = 64, N = 16, R = 1;  // K: 2 blocks, N: 2 tiles, 1 row
  std::mt19937 rng(3);
  std::vector<uint8_t> W((size_t)N * K / 2), SW((size_t)N * K / 32), A((size_t)R * K), SA((size_t)R * K / 32);
  auto rnd = [&](int lo, int hi) { return (uint8_t)(lo + rng() % (hi - lo + 1)); };
  for (auto& v : W) v = rnd(0, 255);
  for (auto& v : SW) v = rnd(125, 129);
  for (auto& v : A) { do { v = rnd(0, 255); } while ((v & 0x7F) == 0x7F || (v & 0x78) > 0x48); }
  for (auto& v : SA) v = rnd(125, 129);
  // host reference
  std::vector<double> ref(N, 0.0);
  for (int n = 0; n < N; ++n)
    for (int b = 0; b < K / 32; ++b) {
      double d = 0;
      for (int i = 0; i < 32; ++i) {
        const int k = b * 32 + i;
        const uint8_t byte = W[(size_t)n * K / 2 + k / 2];
        const float w = e2m1_to_f32((k & 1) ? (byte >> 4) : (byte & 0xF));
        d += (double)e4m3_to_f32(A[k]) * w;
      }
      ref[n] += d * e8m0_to_f32(SA[b]) * e8m0_to_f32(SW[(size_t)n * (K / 32) + b]);
    }
  uint8_t *dW, *dSW, *dA, *dSA; bf16 *o1, *o2;
  CUDA_CHECK(cudaMalloc(&dW, W.size())); CUDA_CHECK(cudaMalloc(&dSW, SW.size())); CUDA_CHECK(cudaMalloc(&dA, A.size())); CUDA_CHECK(cudaMalloc(&dSA, SA.size()));
  CUDA_CHECK(cudaMalloc(&o1, N * 2)); CUDA_CHECK(cudaMalloc(&o2, N * 2));
  CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dSW, SW.data(), SW.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dSA, SA.data(), SA.size(), cudaMemcpyHostToDevice));
  k::GroupDesc gd{dW, dSW, dW, dSW, dW, dSW, 0, R};  // w2 = W (dim = N, I = K)
  k::GroupDesc* dgd; CUDA_CHECK(cudaMalloc(&dgd, sizeof(gd))); CUDA_CHECK(cudaMemcpy(dgd, &gd, sizeof(gd), cudaMemcpyHostToDevice));
  k::gemv_grouped_w2(dgd, 1, dA, dSA, N, K, o1, 0);
  k::mx_grouped_w2(dgd, 1, dA, dSA, N, K, o2, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> h1(N), h2(N);
  CUDA_CHECK(cudaMemcpy(h1.data(), o1, N * 2, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(h2.data(), o2, N * 2, cudaMemcpyDeviceToHost));
  printf("%4s %14s %14s %14s\n", "n", "ref", "cuda-core", "mx");
  for (int n = 0; n < N; ++n) printf("%4d %14.6g %14.6g %14.6g\n", n, ref[n], bits2f(h1[n]), bits2f(h2[n]));
  return 0;
}
