// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Decode GEMV (M ≤ 8 rows) for FN's small dense matrices: K split inside the block and across blocks, parts added in a fixed order
//   (src/qwen/fn_gemv.cu). y = bf16(x · Wᵀ) [M, N] at row stride ldy — or fp32 into yf when yf != nullptr. ws ≥ fn_gemv_ws_bytes.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace hive::qwen {

bool fn_gemv_ok(int N, int K, int M);
int fn_gemv_parts(int N, int K);
size_t fn_gemv_ws_bytes(int N, int K, int M);
void fn_gemv(const __nv_bfloat16* W, int N, int K, const __nv_bfloat16* x, int ldx, int M, __nv_bfloat16* y, int ldy, float* yf, float* ws,
             cudaStream_t st);

// Split-K GEMV with a fused epilogue (see fn_gemv.cu): mode 2 = hyper-connection mix down projection (y bf16 [M, N] at ldy and
//   s = bf16(silu(bf16(y / hc))) [M, rank] for the first rank columns); mode 3 = router logits + shared expert gate|up (N = nr + 2I rows:
//   fp32 logits [M, nr] at ldyf, ys = bf16(silu(gate) · up) [M, I]).
struct FnEpi {
  int mode = 0;
  __nv_bfloat16* y = nullptr; int ldy = 0;
  float* yf = nullptr; int ldyf = 0;
  __nv_bfloat16* s = nullptr; int rank = 0; float inv_hc = 1.f;
  int nr = 0, I = 0; __nv_bfloat16* ys = nullptr;
};
void fn_gemv_epi(const __nv_bfloat16* W, int N, int K, const __nv_bfloat16* x, int ldx, int M, const FnEpi& e, float* ws, cudaStream_t st);

// 8-bit dense weights: fmt 1 = FP8 e4m3, 2 = INT8; W[n, k] = q[n, k] · s[n, k / gs] (s fp32 [N, ceil(K/gs)]); y = bf16(x · Wᵀ)
bool fn_gemv8_ok(int N, int K, int gs, int M);
void fn_quant8(const __nv_bfloat16* w, int N, int K, int fmt, int gs, uint8_t* q, float* s, cudaStream_t st);
// yf (optional): fp32 output instead of y (the verify head's logits, HIVE_QWEN_DENSE8_HEAD)
void fn_gemv8(const uint8_t* W, const float* S, int fmt, int gs, int N, int K, const __nv_bfloat16* x, int ldx, int M, __nv_bfloat16* y, int ldy,
              cudaStream_t st, float* yf = nullptr);

// bf16 [N, K] = q · s (the inverse of fn_quant8 up to its rounding) — prefill operand when the bf16 original was dropped
void fn_dequant8(const uint8_t* q, const float* s, int fmt, int gs, int N, int K, __nv_bfloat16* w, cudaStream_t st);

// split-K 8-bit GEMV + fn_gemv_epi epilogue (modes 2 / 3); ws ≥ fn_gemv8_split_parts(N, K, gs) · M · N floats
int fn_gemv8_split_parts(int N, int K, int gs);
void fn_gemv8_epi(const uint8_t* W, const float* S, int fmt, int gs, int N, int K, const __nv_bfloat16* x, int ldx, int M, const FnEpi& e, float* ws,
                  cudaStream_t st);

// batch-invariant bf16 GEMV (1..64 rows; the GLM gemv_tc schedule): y bf16, or yf fp32 when non-null
void fn_gemv_tc(const __nv_bfloat16* W, int N, int K, const __nv_bfloat16* x, int ldx, int M, __nv_bfloat16* y, int ldy, float* yf, cudaStream_t st);

}  // namespace hive::qwen
