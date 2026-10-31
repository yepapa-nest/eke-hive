// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next prefill dense GEMMs in MXFP8 (HIVE_QWEN_DENSE_MX=1, lossy — see docs/qwen.md): both operands e4m3 with a power-of-two
//   (ue8m0) scale per 32 values along K, the format NVIDIA's FP4-Local release of this model keeps for these projections; the GEMM itself is
//   cuBLASLt's block-scaled FP8 (CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 — measured on this card 2026-10-31: supported, 886 TFLOP/s at
//   16384 × 16480 × 2560 against ~357 TFLOP/s for the bf16 GEMM it replaces; OUTER_VEC_32F is not supported here).
//   The weight comes from the 8-bit copy (HIVE_QWEN_DENSE8=int8) expanded per GEMM, as the bf16 path expands it today (no extra VRAM).
//   Scales: block exponent e = ceil(log2(amax / 448)) — the largest value maps to ≤ 448, nothing is clipped; values round to nearest e4m3.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "hive/devbuf.h"

#include <cstddef>
#include <cstdint>

namespace hive::qwen {

// bytes of the tiled ue8m0 scale table (cuBLASLt block-scale layout: 128-row × 4-column tiles) for rows × K/32 scales
size_t fn_mx_scale_bytes(int rows, int K);
// x bf16 [rows, K] (row stride ld) → q e4m3 [rows, K], s tiled ue8m0
void fn_mx_quant_bf16(const __nv_bfloat16* x, int ld, int rows, int K, uint8_t* q, uint8_t* s, cudaStream_t st);
// the 8-bit copy (fn_quant8 layout: INT8 [N, K] with fp32 scales per row and group of gs) → q e4m3 [N, K], s tiled ue8m0
void fn_mx_quant_w8(const uint8_t* q8, const float* s8, int gs, int N, int K, uint8_t* q, uint8_t* s, cudaStream_t st);

// the checkpoint's own MXFP8 scales (ue8m0 bytes [N, K/32], row-major — NVIDIA FP4-Local dense projections) → the tiled layout
void fn_mx_tile_scales(const uint8_t* s, int N, int K, uint8_t* tiled, cudaStream_t st);

class FnMx {
 public:
  explicit FnMx(cudaStream_t st);
  ~FnMx();
  bool ok() const { return h_ != nullptr; }
  // C bf16 [M, N] (row stride ldc) = A [M, K] · W [N, K]ᵀ, both MXFP8 (q, tiled scales). false when cuBLASLt has no algorithm.
  bool gemm(const uint8_t* aq, const uint8_t* as, const uint8_t* wq, const uint8_t* ws, int M, int N, int K, __nv_bfloat16* C, int ldc);
 private:
  cudaStream_t st_;
  void* h_ = nullptr;
  DevBuf ws_;                       // cuBLASLt workspace — DevBuf so the level-3 sleep releases it with the rest (2026-10-31 review)
  size_t ws_bytes_ = 32u << 20;
};

}  // namespace hive::qwen
