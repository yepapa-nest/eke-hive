// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// NVFP4 GPU kernels for GLM-5.3-Flash (ModelOpt checkpoints): e2m1 nibbles [N, K/2] (low nibble = even column), e4m3 scale per 16 columns
//   [N, K/16], one fp32 global scale per matrix. Activations stay bf16 (W4A16).
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace hive::glm {

// One weight matrix view (device pointers).
struct Nvfp4Mat {
  const uint8_t* w = nullptr;   // [N, K/2]
  const uint8_t* s = nullptr;   // [N, K/16] e4m3
  float g = 1.f;                // global scale
  int N = 0, K = 0;
};

// y[m, n] = Σ_k x[m, k] · W[n, k]   for m < M (M ≤ 8). x bf16 [M, K] (row stride ldx), y fp32 [M, N] (row stride ldy).
void nvfp4_gemv(const Nvfp4Mat& W, const __nv_bfloat16* x, int ldx, int M, float* y, int ldy, cudaStream_t st);
// Dequantize W to bf16 [N, K] (row-major) — prefill path (followed by a bf16 GEMM).
void nvfp4_dequant(const Nvfp4Mat& W, __nv_bfloat16* out, cudaStream_t st);

}  // namespace hive::glm
