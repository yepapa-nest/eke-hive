// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Check: gemm_bs_mx (prefill block-scaled tensor cores) vs gemm_bs_tc (bf16 WMMA) vs a host double reference.
//   Shapes deliberately cut tile boundaries (M 37 = 16·2+5 · N 72 = 64+8 · K 96 = 3 blocks). Pass = both GPU paths within bf16 rounding of the reference (relative 1e-2)
//   and within the same range of each other. Run: scripts/hive-run.sh "./build/test_mx3"
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
  bool all_ok = true;
  for (const int M : {9, 16, 37, 300}) {
    const int N = 72, K = 96, nb = K / 32;
    std::mt19937 rng(7 + M);
    std::vector<uint8_t> W((size_t)N * K / 2), SW((size_t)N * nb), A((size_t)M * K), SA((size_t)M * nb);
    auto rnd = [&](int lo, int hi) { return (uint8_t)(lo + rng() % (hi - lo + 1)); };
    for (auto& v : W) v = rnd(0, 255);
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
            const uint8_t byte = W[(size_t)n * K / 2 + k / 2];
            d += (double)e4m3_to_f32(A[(size_t)m * K + k]) * e2m1_to_f32((k & 1) ? (byte >> 4) : (byte & 0xF));
          }
          ref[(size_t)m * N + n] += d * e8m0_to_f32(SA[(size_t)m * nb + b]) * e8m0_to_f32(SW[(size_t)n * nb + b]);
        }
    uint8_t *dW, *dSW, *dA, *dSA; bf16 *o_tc, *o_mx; float *f_mx;
    CUDA_CHECK(cudaMalloc(&dW, W.size())); CUDA_CHECK(cudaMalloc(&dSW, SW.size())); CUDA_CHECK(cudaMalloc(&dA, A.size())); CUDA_CHECK(cudaMalloc(&dSA, SA.size()));
    CUDA_CHECK(cudaMalloc(&o_tc, (size_t)M * N * 2)); CUDA_CHECK(cudaMalloc(&o_mx, (size_t)M * N * 2)); CUDA_CHECK(cudaMalloc(&f_mx, (size_t)M * N * 4));
    CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dSW, SW.data(), SW.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dSA, SA.data(), SA.size(), cudaMemcpyHostToDevice));
    k::gemm_bs_tc(dA, dSA, dW, dSW, true, M, N, K, o_tc, nullptr, 0);
    k::gemm_bs_mx(dA, dSA, dW, dSW, M, N, K, o_mx, f_mx, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> h_tc((size_t)M * N), h_mx((size_t)M * N);
    std::vector<float> h_f((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(h_tc.data(), o_tc, h_tc.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mx.data(), o_mx, h_mx.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_f.data(), f_mx, h_f.size() * 4, cudaMemcpyDeviceToHost));
    double e_tc = 0, e_mx = 0, e_f32 = 0, e_pair = 0, scale = 1e-6;
    for (const double r : ref) scale = std::max(scale, std::fabs(r));
    for (size_t i = 0; i < ref.size(); i++) {
      e_tc = std::max(e_tc, std::fabs(bits2f(h_tc[i]) - ref[i]) / scale);
      e_mx = std::max(e_mx, std::fabs(bits2f(h_mx[i]) - ref[i]) / scale);
      e_f32 = std::max(e_f32, std::fabs(h_f[i] - ref[i]) / scale);
      e_pair = std::max(e_pair, std::fabs((double)bits2f(h_mx[i]) - bits2f(h_tc[i])) / scale);
    }
    const bool ok = e_tc < 1e-2 && e_mx < 1e-2 && e_f32 < 1e-5 && e_pair < 1e-2;
    all_ok &= ok;
    printf("M=%4d N=%d K=%d · max rel err(÷max|ref|): tc %.2e · mx %.2e · mx-f32 %.2e · mx↔tc %.2e → %s\n", M, N, K, e_tc, e_mx, e_f32, e_pair, ok ? "OK" : "FAIL");
    cudaFree(dW); cudaFree(dSW); cudaFree(dA); cudaFree(dSA); cudaFree(o_tc); cudaFree(o_mx); cudaFree(f_mx);
  }
  printf("%s\n", all_ok ? "test_mx3 PASS" : "test_mx3 FAIL");
  return all_ok ? 0 : 1;
}
