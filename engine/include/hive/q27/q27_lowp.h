// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-27B low-precision PREFILL GEMMs (option HIVE_Q27_PREFILL_QUANT=1 — a lossy path, see docs/q27.md): the activations are quantized
//   the way the checkpoint was calibrated for (compressed-tensors quantization_config):
//   · FP8 layers (group_0: attention / linear-attention projections, MLP 56-63): dynamic per-token FP8 e4m3, scale = amax / 448 per row;
//     GEMM e4m3 × e4m3 (fp32 accumulate, cuBLASLt), then y = bf16(acc · s_token[m] · s_channel[n]).
//   · NVFP4 MLP (group_1, layers 0-55): e2m1 activations with an e4m3 scale per 16 values and the checkpoint's static input_global_scale g_a
//     (block scale = e4m3(amax16 / 6 · g_a), value = e2m1(x · g_a / block scale)); block-scaled FP4 GEMM (cuBLASLt VEC16_UE4M3), alpha =
//     1 / (g_a · g_w), y = bf16(acc · alpha).
// Decode and verify keep the bf16-activation GEMV path (hive/q27/q27_kernels.h).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace hive::q27::lowp {

using bf16 = __nv_bfloat16;

// bytes of a block-scale table in cuBLASLt's tiled layout (128-row × 4-column tiles of 512 bytes) for rows × (K / 16) scales
size_t swz_bytes(int rows, int K);
// row-major e4m3 scales [rows, K/16] → the tiled layout (dst zero-filled for the padding)
void swizzle_scales(const uint8_t* src, int rows, int K, uint8_t* dst, cudaStream_t st);
// per-token FP8: q e4m3 [M, K], s fp32 [M]
void quant_fp8_rows(const bf16* x, int ldx, int M, int K, uint8_t* q, float* s, cudaStream_t st);
// NVFP4 activations: q e2m1 pairs [M, K/2], scales in the tiled layout (swz_bytes(M, K)), input global scale g_a
void quant_nvfp4_rows(const bf16* x, int ldx, int M, int K, float g_a, uint8_t* q, uint8_t* s_swz, cudaStream_t st);
// y[m, n] = bf16(acc[m, n] · alpha · (rs ? rs[m] : 1) · (cs ? cs[n] : 1))
void scale_out(const float* acc, int M, int N, float alpha, const float* rs, const float* cs, bf16* y, int ldy, cudaStream_t st);

// W8A8 decode GEMM (option HIVE_Q27_DECODE_QUANT=1), 1 ≤ M ≤ 16: xq e4m3 [M, K] (row stride ldq, row scales xs), w e4m3 [N, K] with
//   per-row scales ws → y[m, n] = bf16(acc · xs[m] · ws[n]) (yb) or fp32 (yf). mma.sync m16n8k32 e4m3, permuted-k fragments loaded straight
//   from global memory (16 bytes per lane), 4 K-slices per output tile added in slice order: the same arithmetic for every M.
void fp8_gemm_rows(const uint8_t* xq, int ldq, const float* xs, int M, const uint8_t* w, const float* ws, int N, int K, bf16* yb, float* yf,
                   int ldy, cudaStream_t st);

// W4A4 decode GEMM (option HIVE_Q27_DECODE_QUANT=1, NVFP4 matrices), 1 ≤ M ≤ 16: activations quantized to e2m1 with e4m3 scales per 16
//   (row-major [M, K/16]) under the static input global scale; mma.sync kind::mxf4nvf4 block_scale scale_vec::4X (ue4m3) m16n8k64;
//   y = bf16(acc · alpha), alpha = 1 / (g_a · g_w).
void quant_nvfp4_rm(const bf16* x, int ldx, int M, int K, float ga, uint8_t* q, uint8_t* s, cudaStream_t st);
void nvfp4_gemm_rows(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* w, const uint8_t* ws, int N, int K, float alpha, bf16* yb,
                     float* yf, int ldy, cudaStream_t st);

// NVFP4 activation quantization with a dynamic global scale: sc (device, 4 floats) ← {amax, ga = 448·6/amax, alpha = g / ga, beta = 0}
void quant_nvfp4_rows_dyn(const bf16* x, int ldx, int M, int K, float g, float* sc, uint8_t* q, uint8_t* s_swz, cudaStream_t st);

class Lt {
 public:
  explicit Lt(cudaStream_t st);
  ~Lt();
  // acc fp32 [M, N] (row-major) = Xq [M, K] · Wq [N, K]ᵀ, both e4m3 (unit scales). false if cuBLASLt has no algorithm.
  bool fp8(const uint8_t* xq, const uint8_t* wq, int M, int N, int K, float* acc);
  // acc fp32 [M, N] = Σ_k (xq·xs)(wq·ws) · alpha — e2m1 pairs with tiled e4m3 block scales (VEC16_UE4M3). false if unsupported.
  bool nvfp4(const uint8_t* xq, const uint8_t* xs_swz, const uint8_t* wq, const uint8_t* ws_swz, int M, int N, int K, float alpha, float* acc);
  // the same products written as bf16 rows (row stride ldy) by the GEMM itself — FP8 with per-row scales of both operands (outer-vector
  //   scaling; false when cuBLASLt has none — then never tried again), NVFP4 with alpha
  bool fp8_bf16(const uint8_t* xq, const float* xs, const uint8_t* wq, const float* wsc, int M, int N, int K, bf16* y, int ldy);
  // alpha and beta from the device (alpha_beta[0], alpha_beta[1]) — the dynamic-scale W4A4 prefill
  bool nvfp4_bf16_dev(const uint8_t* xq, const uint8_t* xs_swz, const uint8_t* wq, const uint8_t* ws_swz, int M, int N, int K, const float* alpha_beta,
                      bf16* y, int ldy);
  bool nvfp4_bf16(const uint8_t* xq, const uint8_t* xs_swz, const uint8_t* wq, const uint8_t* ws_swz, int M, int N, int K, float alpha, bf16* y, int ldy);
 private:
  bool fp8_outer_off_ = false;
  void* h_ = nullptr;
  void* ws_ = nullptr;
  size_t ws_bytes_ = 64ull << 20;
  cudaStream_t st_;
};

}  // namespace hive::q27::lowp
