// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// P3 validation: wo_a prefill — dequant_fp8_rows_to_bf16 + cuBLAS batched bf16 GEMM vs gemm_bf16a_fp8b_tc (bf16 activations × fp8 weights) vs host double.
//   Both paths use identical weight values (e4m3×2^k is exact in bf16); only the accumulation order differs. Run: scripts/hive-run.sh "./build-dev/test_woa"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

#include "hive/common.h"
#include "hive/cublas_ops.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"
#include "test_util.h"

using namespace hive;
static float bits2f(uint16_t b) { return hive::test::bf16_bits_to_f32(b); }
static uint16_t f2bits(float f) { bf16 b = f2bf(f); uint16_t u; memcpy(&u, &b, 2); return u; }

int main() {
  const int H = 64, D = 512, HD = H * D, G = 8, R = 1024, sub = HD / G;  // wo_a: [R, sub] per group g
  bool all_ok = true;
  Blas blas(0);
  for (const int M : {9, 70, 2048}) {
    std::mt19937 rng(5 + M);
    std::normal_distribution<float> nd;
    std::vector<uint8_t> Wq((size_t)G * R * sub), Sq((size_t)G * R * (sub / 32));
    for (auto& v : Wq) { do { v = (uint8_t)(rng() & 0xFF); } while ((v & 0x7F) == 0x7F); }
    for (auto& v : Sq) v = (uint8_t)(120 + rng() % 9);
    std::vector<uint16_t> o((size_t)M * HD);
    for (auto& v : o) v = f2bits(nd(rng) * 0.1f);
    uint8_t *dW, *dS; bf16 *do_, *woa, *og_ref, *og_new;
    CUDA_CHECK(cudaMalloc(&dW, Wq.size())); CUDA_CHECK(cudaMalloc(&dS, Sq.size())); CUDA_CHECK(cudaMalloc(&do_, o.size() * 2));
    CUDA_CHECK(cudaMalloc(&woa, Wq.size() * 2)); CUDA_CHECK(cudaMalloc(&og_ref, (size_t)M * G * R * 2)); CUDA_CHECK(cudaMalloc(&og_new, (size_t)M * G * R * 2));
    CUDA_CHECK(cudaMemcpy(dW, Wq.data(), Wq.size(), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dS, Sq.data(), Sq.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(do_, o.data(), o.size() * 2, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float ms_ref = 0, ms_new = 0;
    for (int rep = 0; rep < 2; ++rep) {
      cudaEventRecord(e0);
      for (int g = 0; g < G; ++g)
        k::gemm_bf16a_fp8b_tc(do_ + (size_t)g * sub, HD, dW + (size_t)g * R * sub, dS + (size_t)g * R * (sub / 32), M, R, sub, og_ref + (size_t)g * R, G * R, 0);
      cudaEventRecord(e1); CUDA_CHECK(cudaDeviceSynchronize()); cudaEventElapsedTime(&ms_ref, e0, e1);
      cudaEventRecord(e0);
      k::dequant_fp8_rows_to_bf16(dW, dS, G * R, sub, woa, 0);
      blas.gemm_bf16_batched_ld(do_, HD, sub, woa, sub, (long)R * sub, og_new, G * R, R, M, R, sub, G);
      cudaEventRecord(e1); CUDA_CHECK(cudaDeviceSynchronize()); cudaEventElapsedTime(&ms_new, e0, e1);
    }
    // Host reference (double): 3 sample rows
    std::vector<uint16_t> a((size_t)M * G * R), b((size_t)M * G * R);
    CUDA_CHECK(cudaMemcpy(a.data(), og_ref, a.size() * 2, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(b.data(), og_new, b.size() * 2, cudaMemcpyDeviceToHost));
    double e_pair = 0, e_ref = 0, e_new = 0, scale = 1e-6;
    for (const uint16_t v : a) scale = std::max(scale, (double)std::fabs(bits2f(v)));
    for (size_t i = 0; i < a.size(); i++) e_pair = std::max(e_pair, std::fabs((double)bits2f(a[i]) - bits2f(b[i])) / scale);
    for (const int m : {0, M / 2, M - 1})
      for (int g = 0; g < G; ++g)
        for (int rr = 0; rr < R; rr += 97) {
          double acc = 0;
          for (int k = 0; k < sub; ++k) {
            const size_t wi = ((size_t)g * R + rr) * sub + k;
            acc += (double)bits2f(o[(size_t)m * HD + g * sub + k]) * e4m3_to_f32(Wq[wi]) * e8m0_to_f32(Sq[((size_t)g * R + rr) * (sub / 32) + k / 32]);
          }
          const size_t oi = (size_t)m * G * R + g * R + rr;
          e_ref = std::max(e_ref, std::fabs(bits2f(a[oi]) - acc) / scale);
          e_new = std::max(e_new, std::fabs(bits2f(b[oi]) - acc) / scale);
        }
    const bool ok = e_pair < 2e-2 && e_new < 1e-2;
    printf("M=%-5d tc↔cublas %.2e  tc↔ref %.2e  cublas↔ref %.2e  tc %.3f ms  dequant+cublas %.3f ms  %s\n", M, e_pair, e_ref, e_new, ms_ref, ms_new, ok ? "PASS" : "FAIL");
    all_ok = all_ok && ok;
    cudaFree(dW); cudaFree(dS); cudaFree(do_); cudaFree(woa); cudaFree(og_ref); cudaFree(og_new);
  }
  printf(all_ok ? "ALL PASS\n" : "SOME FAIL\n");
  return all_ok ? 0 : 1;
}
