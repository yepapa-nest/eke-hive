// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_HEAD_ROWS check: the multi-row head kernel (head_logits, M ≤ 8) against the per-row kernel (head_logits_per_row) — logits compared
//   bit for bit for M = 1..8 on the real shape (vocabulary 129,280 × 5,120, bf16 head, fp32 input), and µs per call with L2 flushed.
//   Run: ./build/test_head_rows [reps=20]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "hive/common.h"
#include "hive/model_kernels.h"

using namespace hive;

namespace {
__global__ void fill_bf16(bf16* p, size_t n, uint32_t seed) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t h = (uint32_t)(i * 2654435761u) ^ seed; h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12;
  p[i] = __float2bfloat16(((int)(h & 0xFFFF) - 32768) / 65536.f * 0.08f);
}
__global__ void fill_f32(float* p, size_t n, uint32_t seed) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t h = (uint32_t)(i * 2246822519u) ^ seed; h ^= h >> 13; h *= 0x297a2d39u; h ^= h >> 16;
  p[i] = ((int)(h & 0xFFFF) - 32768) / 16384.f;
}
}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::max(3, atoi(argv[1])) : 20;
  const int V = 129280, D = 5120;
  bf16* head; float *x, *a, *b; uint8_t* flush;
  const size_t flush_n = 256u << 20;
  CUDA_CHECK(cudaMalloc(&head, (size_t)V * D * 2)); CUDA_CHECK(cudaMalloc(&x, (size_t)8 * D * 4));
  CUDA_CHECK(cudaMalloc(&a, (size_t)8 * V * 4)); CUDA_CHECK(cudaMalloc(&b, (size_t)8 * V * 4)); CUDA_CHECK(cudaMalloc(&flush, flush_n));
  fill_bf16<<<(unsigned)(((size_t)V * D + 255) / 256), 256>>>(head, (size_t)V * D, 7);
  fill_f32<<<(8 * D + 255) / 256, 256>>>(x, (size_t)8 * D, 11);
  cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
  cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  bool ok = true;
  printf("%-3s %-12s %-12s %-8s %s\n", "M", "per-row µs", "rows µs", "ratio", "verdict");
  for (int M = 1; M <= 8; ++M) {
    double t[2] = {0, 0};
    for (int v = 0; v < 2; ++v)
      for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaMemsetAsync(flush, r & 0xFF, flush_n, st));
        CUDA_CHECK(cudaEventRecord(e0, st));
        if (v == 0) k::head_logits_per_row(x, head, M, V, D, a, st); else k::head_logits(x, head, M, V, D, b, st);
        CUDA_CHECK(cudaEventRecord(e1, st));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (r) t[v] += ms;
      }
    std::vector<float> ha((size_t)M * V), hb((size_t)M * V);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, ha.size() * 4, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hb.data(), b, hb.size() * 4, cudaMemcpyDeviceToHost));
    size_t mism = 0; for (size_t i = 0; i < ha.size(); ++i) mism += std::memcmp(&ha[i], &hb[i], 4) != 0;
    ok = ok && mism == 0;
    const double u0 = t[0] / (reps - 1) * 1000, u1 = t[1] / (reps - 1) * 1000;
    printf("%-3d %-12.1f %-12.1f ×%-7.2f %s\n", M, u0, u1, u0 / u1, mism ? "FAIL" : "PASS (bit-exact)");
  }
  printf("RESULT: %s\n", ok ? "PASS (all bit-exact)" : "FAIL");
  return ok ? 0 : 1;
}
