// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// MXFP8 prefill GEMMs — see hive/qwen/fn_mx.h.
#include "hive/qwen/fn_mx.h"

#include <cublasLt.h>
#include <cuda_fp8.h>

#include <map>
#include <tuple>

#include "hive/common.h"

namespace hive::qwen {

namespace {
using bf16 = __nv_bfloat16;
inline int pad_up(int x, int m) { return (x + m - 1) / m * m; }
// tiled block-scale layout (the same tiles as the NVFP4 scales of q27_lowp.cu): 128 rows × 4 scale columns per 512-byte tile, row tiles outer
__device__ __forceinline__ size_t swz(int r, int c, int cols4) {
  return ((size_t)(r / 128) * (cols4 / 4) + (c / 4)) * 512 + (size_t)(r % 32) * 16 + (size_t)((r % 128) / 32) * 4 + (c % 4);
}
// one 32-value block: amax → e = ceil(log2(amax / 448)) (amax · 2^−e ≤ 448), values · 2^−e rounded to nearest e4m3
__device__ __forceinline__ void mx_block(const float (&v)[32], uint8_t* q, uint8_t& sbyte) {
  float a = 0.f;
#pragma unroll
  for (int j = 0; j < 32; ++j) a = fmaxf(a, fabsf(v[j]));
  int e = -127;
  if (a > 0.f) {
    e = (int)ceilf(log2f(a / 448.f));
    if (ldexpf(a, -e) > 448.f) ++e;   // (log2f rounding at an exact power of two)
    e = max(-127, min(127, e));
  }
  sbyte = (uint8_t)(e + 127);
  const float f = a > 0.f ? ldexpf(1.f, -e) : 0.f;
  unsigned w[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v[4 * j] * f, v[4 * j + 1] * f), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v[4 * j + 2] * f, v[4 * j + 3] * f), __NV_SATFINITE, __NV_E4M3);
    w[j] = (unsigned)lo | ((unsigned)hi << 16);
  }
  uint4* d = reinterpret_cast<uint4*>(q);
  d[0] = make_uint4(w[0], w[1], w[2], w[3]);
  d[1] = make_uint4(w[4], w[5], w[6], w[7]);
}
// thread per 32-value block
__global__ void mx_quant_bf16_kernel(const bf16* __restrict__ x, int ld, int rows, int K, uint8_t* __restrict__ q, uint8_t* __restrict__ s, int cols4) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int nb = K / 32;
  if (i >= (size_t)rows * nb) return;
  const int r = (int)(i / nb), b = (int)(i % nb);
  const uint4* p = reinterpret_cast<const uint4*>(x + (size_t)r * ld + (size_t)b * 32);
  float v[32];
#pragma unroll
  for (int h = 0; h < 4; ++h) {
    const uint4 u = p[h];
    const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[j]));
      v[h * 8 + j * 2] = f.x; v[h * 8 + j * 2 + 1] = f.y;
    }
  }
  uint8_t sb;
  mx_block(v, q + (size_t)r * K + (size_t)b * 32, sb);
  s[swz(r, b, cols4)] = sb;
}
__global__ void mx_quant_w8_kernel(const uint8_t* __restrict__ q8, const float* __restrict__ s8, int gs, int N, int K, uint8_t* __restrict__ q,
                                   uint8_t* __restrict__ s, int cols4) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int nb = K / 32, ng = (K + gs - 1) / gs;
  if (i >= (size_t)N * nb) return;
  const int r = (int)(i / nb), b = (int)(i % nb);
  const float sc = s8[(size_t)r * ng + (b * 32) / gs];
  const uint4* p = reinterpret_cast<const uint4*>(q8 + (size_t)r * K + (size_t)b * 32);
  float v[32];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint4 u = p[h];
    const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int j = 0; j < 16; ++j) v[h * 16 + j] = (float)(int8_t)((w[j / 4] >> (8 * (j % 4))) & 0xFF) * sc;
  }
  uint8_t sb;
  mx_block(v, q + (size_t)r * K + (size_t)b * 32, sb);
  s[swz(r, b, cols4)] = sb;
}
// ue8m0 bytes [N, K/32] (the checkpoint's row-major scales) → the tiled layout
__global__ void mx_tile_kernel(const uint8_t* __restrict__ s, int N, int nb, uint8_t* __restrict__ t, int cols4) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)N * nb) return;
  const int r = (int)(i / nb), b = (int)(i % nb);
  t[swz(r, b, cols4)] = s[i];
}
inline unsigned nblk(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }
#define LT_OK(expr) ((expr) == CUBLAS_STATUS_SUCCESS)
}  // namespace

size_t fn_mx_scale_bytes(int rows, int K) { return (size_t)pad_up(rows, 128) * pad_up(K / 32, 4); }

void fn_mx_quant_bf16(const bf16* x, int ld, int rows, int K, uint8_t* q, uint8_t* s, cudaStream_t st) {
  if (rows <= 0) return;
  HIVE_CHECK(K % 32 == 0 && ld % 8 == 0 && ((uintptr_t)x & 15) == 0, "fn_mx_quant_bf16 shape");
  CUDA_CHECK(cudaMemsetAsync(s, 0, fn_mx_scale_bytes(rows, K), st));
  mx_quant_bf16_kernel<<<nblk((size_t)rows * (K / 32)), 256, 0, st>>>(x, ld, rows, K, q, s, pad_up(K / 32, 4));
  CUDA_CHECK(cudaGetLastError());
}
void fn_mx_quant_w8(const uint8_t* q8, const float* s8, int gs, int N, int K, uint8_t* q, uint8_t* s, cudaStream_t st) {
  HIVE_CHECK(K % 32 == 0 && gs % 32 == 0, "fn_mx_quant_w8 shape");
  CUDA_CHECK(cudaMemsetAsync(s, 0, fn_mx_scale_bytes(N, K), st));
  mx_quant_w8_kernel<<<nblk((size_t)N * (K / 32)), 256, 0, st>>>(q8, s8, gs, N, K, q, s, pad_up(K / 32, 4));
  CUDA_CHECK(cudaGetLastError());
}

void fn_mx_tile_scales(const uint8_t* s, int N, int K, uint8_t* tiled, cudaStream_t st) {
  HIVE_CHECK(K % 32 == 0, "fn_mx_tile_scales shape");
  CUDA_CHECK(cudaMemsetAsync(tiled, 0, fn_mx_scale_bytes(N, K), st));
  mx_tile_kernel<<<nblk((size_t)N * (K / 32)), 256, 0, st>>>(s, N, K / 32, tiled, pad_up(K / 32, 4));
  CUDA_CHECK(cudaGetLastError());
}

FnMx::FnMx(cudaStream_t st) : st_(st) {
  cublasLtHandle_t h;
  if (cublasLtCreate(&h) == CUBLAS_STATUS_SUCCESS) h_ = h;
  ws_.alloc(ws_bytes_);
}
FnMx::~FnMx() {
  if (h_) cublasLtDestroy((cublasLtHandle_t)h_);
}

// D (col-major N × M = row-major [M, N], ld ldc) = op(A = W, stored K × N col-major, transposed) · (B = X, K × M col-major)
bool FnMx::gemm(const uint8_t* aq, const uint8_t* as, const uint8_t* wq, const uint8_t* wsc, int M, int N, int K, bf16* C, int ldc) {
  if (!h_) return false;
  cublasLtHandle_t h = (cublasLtHandle_t)h_;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  bool ok = LT_OK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
  ok = ok && LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA));
  ok = ok && LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB));
  const int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  ok = ok && LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof mode));
  ok = ok && LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof mode));
  ok = ok && LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wsc, sizeof wsc));
  ok = ok && LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &as, sizeof as));
  ok = ok && LT_OK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, K, N, K));
  ok = ok && LT_OK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, K, M, K));
  ok = ok && LT_OK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, N, M, ldc));
  ok = ok && LT_OK(cublasLtMatmulPreferenceCreate(&pref));
  ok = ok && LT_OK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes_, sizeof ws_bytes_));
  static std::map<std::tuple<int, int, int, int>, cublasLtMatmulAlgo_t> algo_cache;   // the heuristic's pick per shape
  const auto key = std::make_tuple(M, N, K, ldc);
  cublasLtMatmulHeuristicResult_t res{};
  int n = 0;
  if (auto it = algo_cache.find(key); ok && it != algo_cache.end()) { res.algo = it->second; n = 1; }
  else {
    ok = ok && LT_OK(cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, ld, ld, pref, 1, &res, &n)) && n > 0;
    if (ok) algo_cache.emplace(key, res.algo);
  }
  const float alpha = 1.f, beta = 0.f;
  ok = ok && LT_OK(cublasLtMatmul(h, op, &alpha, wq, la, aq, lb, &beta, C, ld, C, ld, &res.algo, ws_.p, ws_bytes_, st_));
  if (pref) cublasLtMatmulPreferenceDestroy(pref);
  if (ld) cublasLtMatrixLayoutDestroy(ld);
  if (lb) cublasLtMatrixLayoutDestroy(lb);
  if (la) cublasLtMatrixLayoutDestroy(la);
  if (op) cublasLtMatmulDescDestroy(op);
  return ok;
}

}  // namespace hive::qwen
