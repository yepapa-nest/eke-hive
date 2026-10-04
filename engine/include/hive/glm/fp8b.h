// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// FP8 e4m3 weights with 128×128 block scales (fp32 "weight_scale_inv", the Z.ai GLM-5.3-Flash / DeepSeek-V3 format):
//   W[n, k] = e4m3(w[n, k]) · s[n / 128, k / 128]. Used for the tensors Z.ai ships in FP8 (DSA MLA projections, shared experts).
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace hive::glm {

struct Fp8BMat {
  const uint8_t* w = nullptr;  // [N, K] e4m3
  const float* s = nullptr;    // [ceil(N/128), ceil(K/128)] fp32
  int N = 0, K = 0;
};

// y[m, n] = bf16( Σ_k x[m, k] · W[n, k] ) for M ≤ 8 rows (decode). fp32 accumulation per 128-column block, then × block scale.
void fp8b_gemv(const Fp8BMat& W, const __nv_bfloat16* x, int ldx, int M, __nv_bfloat16* y, int ldy, cudaStream_t st);
// out[N, K] = bf16(e4m3(w) · s) — the bf16 weight transformers uses (prefill: dequantize, then a bf16 GEMM).
void fp8b_dequant(const Fp8BMat& W, __nv_bfloat16* out, cudaStream_t st);

}  // namespace hive::glm
