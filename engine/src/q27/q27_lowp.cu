// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-27B low-precision prefill GEMMs — see hive/q27/q27_lowp.h. Quantization kernels are this project's own; the GEMMs are cuBLASLt.
#include "hive/q27/q27_lowp.h"

#include <map>
#include <tuple>

#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdio>

#include "hive/common.h"

namespace hive::q27::lowp {

namespace {
// cuBLASLt's 1-D block-scale layout: (row, scale column) → byte offset; tiles of 128 rows × 4 columns, 512 bytes each, row tiles outer
__host__ __device__ inline size_t swz_off(int r, int c, int cols_pad4) {
  const size_t tile = (size_t)(r / 128) * (cols_pad4 / 4) + (c / 4);
  return tile * 512 + (size_t)(r % 32) * 16 + (size_t)((r % 128) / 32) * 4 + (c % 4);
}
inline int pad_up(int x, int m) { return (x + m - 1) / m * m; }

__global__ void swizzle_kernel(const uint8_t* src, int rows, int cols, int cols4, uint8_t* dst) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)rows * cols) return;
  const int r = (int)(i / cols), c = (int)(i % cols);
  dst[swz_off(r, c, cols4)] = src[i];
}

__global__ void quant_fp8_kernel(const bf16* x, int ldx, int K, uint8_t* q, float* s) {
  const int m = blockIdx.x;
  const bf16* xr = x + (size_t)m * ldx;
  __shared__ float red[32];
  float a = 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) a = fmaxf(a, fabsf(__bfloat162float(xr[i])));
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = a;
  __syncthreads();
  if (threadIdx.x == 0) { float t = 0.f; for (int i = 0; i < (int)(blockDim.x / 32); ++i) t = fmaxf(t, red[i]); red[0] = t; }
  __syncthreads();
  const float sc = red[0] > 0.f ? red[0] / 448.f : 1.f, inv = 1.f / sc;
  if (threadIdx.x == 0) s[m] = sc;
  for (int i = threadIdx.x * 2; i < K; i += blockDim.x * 2) {
    const float2 v = make_float2(__bfloat162float(xr[i]) * inv, __bfloat162float(xr[i + 1]) * inv);
    *reinterpret_cast<__nv_fp8x2_storage_t*>(q + (size_t)m * K + i) = __nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
  }
}

// one thread per 16-value block
__global__ void quant_nvfp4_kernel(const bf16* x, int ldx, int M, int K, float ga, uint8_t* q, uint8_t* s, int cols4) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int nb = K / 16;
  if (i >= (size_t)M * nb) return;
  const int m = (int)(i / nb), b = (int)(i % nb);
  const bf16* xr = x + (size_t)m * ldx + (size_t)b * 16;
  float v[16], a = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) { v[j] = __bfloat162float(xr[j]); a = fmaxf(a, fabsf(v[j])); }
  const __nv_fp8_storage_t se = __nv_cvt_float_to_fp8(a / 6.f * ga, __NV_SATFINITE, __NV_E4M3);
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(se, __NV_E4M3);
  const float sv = __half2float(*reinterpret_cast<const __half*>(&hr));
  const float f = sv > 0.f ? ga / sv : 0.f;
  uint8_t* qo = q + (size_t)m * (K / 2) + (size_t)b * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) qo[j] = (uint8_t)__nv_cvt_float2_to_fp4x2(make_float2(v[2 * j] * f, v[2 * j + 1] * f), __NV_E2M1, cudaRoundNearest);
  s[swz_off(m, b, cols4)] = (uint8_t)se;
}

// dynamic activation global scale (W4A4 prefill of requantized FP8 matrices): amax of the chunk → ga = 448 · 6 / amax, alpha = g / ga
__global__ void amax_kernel(const bf16* x, int ldx, int M, int K, float* amax) {
  float m = 0.f;
  const size_t n = (size_t)M * K;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    m = fmaxf(m, fabsf(__bfloat162float(x[(i / K) * ldx + i % K])));
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) atomicMax(reinterpret_cast<int*>(amax), __float_as_int(m));
}
__global__ void dyn_scale_kernel(float* sc, float g) {
  const float a = sc[0];
  const float ga = a > 0.f ? 448.f * 6.f / a : 1.f;
  sc[1] = ga; sc[2] = g / ga; sc[3] = 0.f;
}
__global__ void quant_nvfp4_dyn_kernel(const bf16* x, int ldx, int M, int K, const float* sc, uint8_t* q, uint8_t* s, int cols4) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int nb = K / 16;
  if (i >= (size_t)M * nb) return;
  const float ga = sc[1];
  const int m = (int)(i / nb), b = (int)(i % nb);
  const bf16* xr = x + (size_t)m * ldx + (size_t)b * 16;
  float v[16], a = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) { v[j] = __bfloat162float(xr[j]); a = fmaxf(a, fabsf(v[j])); }
  const __nv_fp8_storage_t se = __nv_cvt_float_to_fp8(a / 6.f * ga, __NV_SATFINITE, __NV_E4M3);
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(se, __NV_E4M3);
  const float sv = __half2float(*reinterpret_cast<const __half*>(&hr));
  const float f = sv > 0.f ? ga / sv : 0.f;
  uint8_t* qo = q + (size_t)m * (K / 2) + (size_t)b * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) qo[j] = (uint8_t)__nv_cvt_float2_to_fp4x2(make_float2(v[2 * j] * f, v[2 * j + 1] * f), __NV_E2M1, cudaRoundNearest);
  s[swz_off(m, b, cols4)] = (uint8_t)se;
}

__global__ void scale_out_kernel(const float* acc, int M, int N, float alpha, const float* rs, const float* cs, bf16* y, int ldy) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * N) return;
  const int m = (int)(i / N), n = (int)(i % N);
  float v = acc[i] * alpha;
  if (rs) v *= rs[m];
  if (cs) v *= cs[n];
  y[(size_t)m * ldy + n] = __float2bfloat16_rn(v);
}
inline unsigned blocks(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }

// ---- W8A8 decode GEMM ----------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma_e4m3(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// block = 2 tiles of 8 weight rows × 4 K-slices; a quad lane q holds 16 consecutive bytes of its row: step j ∈ {0,1} uses bytes [8j, 8j+4)
//   as the logical k (4q … 4q+3) and [8j+4, 8j+8) as (16+4q …), the activation fragment takes the same physical bytes.
__global__ void __launch_bounds__(256) fp8_gemm_kernel(const uint8_t* __restrict__ xq, int ldq, const float* xs, int M, const uint8_t* __restrict__ w,
                                                       const float* ws, int N, int K, bf16* yb, float* yf, int ldy) {
  constexpr int S = 4;
  __shared__ float part[2][S][16][8];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, tl = warp / S, sl = warp % S;
  const int n0 = (blockIdx.x * 2 + tl) * 8, q = lane % 4, r = lane / 4;
  const int span = 64, nsp = K / span, s0 = sl * nsp / S, s1 = (sl + 1) * nsp / S;
  const uint8_t* wr = w + (size_t)min(n0 + r, N - 1) * K;
  const bool a0v = r < M, a1v = r + 8 < M;
  const uint8_t* xa = xq + (size_t)min(r, M - 1) * ldq;
  const uint8_t* xb = xq + (size_t)min(r + 8, M - 1) * ldq;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int sb = s0; sb < s1; sb += 4) {
    uint4 wv[4], av[4], bv[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int e0 = min(sb + u, s1 - 1) * span + q * 16;
      wv[u] = __ldg(reinterpret_cast<const uint4*>(wr + e0));
      av[u] = a0v ? *reinterpret_cast<const uint4*>(xa + e0) : make_uint4(0, 0, 0, 0);
      bv[u] = a1v ? *reinterpret_cast<const uint4*>(xb + e0) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (sb + u >= s1) break;
      const uint32_t* W4 = reinterpret_cast<const uint32_t*>(&wv[u]);
      const uint32_t* A4 = reinterpret_cast<const uint32_t*>(&av[u]);
      const uint32_t* B4 = reinterpret_cast<const uint32_t*>(&bv[u]);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint32_t a[4] = {A4[2 * j], B4[2 * j], A4[2 * j + 1], B4[2 * j + 1]};
        mma_e4m3(acc, a, W4[2 * j], W4[2 * j + 1]);
      }
    }
  }
  part[tl][sl][r][2 * q] = acc[0]; part[tl][sl][r][2 * q + 1] = acc[1];
  part[tl][sl][r + 8][2 * q] = acc[2]; part[tl][sl][r + 8][2 * q + 1] = acc[3];
  __syncthreads();
  if (sl == 0)
    for (int i = lane; i < 16 * 8; i += 32) {
      const int m = i / 8, c = i % 8, n = n0 + c;
      if (m >= M || n >= N) continue;
      float v = part[tl][0][m][c];
      for (int z = 1; z < S; ++z) v += part[tl][z][m][c];
      v *= xs[m] * ws[n];
      if (yb) yb[(size_t)m * ldy + n] = __float2bfloat16_rn(v); else yf[(size_t)m * ldy + n] = v;
    }
}

// ---- W4A4 decode GEMM ----------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma_nvf4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  const uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}
// per row the k64 steps of a K slice; operands in the standard fragment layout (packed nibbles: a0 = row r, k 8q … 8q+7; a2 = k 32+8q …;
//   b0 / b1 likewise for weight row n0 + r); scales: lane t supplies activation row (t>>2) + 8·(t&1) and weight row n0 + (t>>2), 4 bytes =
//   the 4 groups of 16 of the step (lower k in the lower byte)
__global__ void __launch_bounds__(256) nvfp4_gemm_kernel(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs, int M,
                                                         const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws, int N, int K, float alpha,
                                                         bf16* yb, float* yf, int ldy) {
  constexpr int S = 4;
  __shared__ float part[2][S][16][8];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, tl = warp / S, sl = warp % S;
  const int n0 = (blockIdx.x * 2 + tl) * 8, q = lane % 4, r = lane / 4;
  const int nst = K / 64, s0 = sl * nst / S, s1 = (sl + 1) * nst / S;
  const int kb = K / 2, kg = K / 16;   // bytes per row (values / 2) · scale groups per row
  const uint8_t* wr = w + (size_t)min(n0 + r, N - 1) * kb;
  const uint8_t* wsr = ws + (size_t)min(n0 + r, N - 1) * kg;
  const int ar = r + 8 * (lane & 1);   // activation row whose scales this lane supplies
  const uint8_t* xsr = xs + (size_t)min(ar, M - 1) * kg;
  const bool a0v = r < M, a1v = r + 8 < M, sav = ar < M;
  const uint8_t* xa = xq + (size_t)min(r, M - 1) * kb;
  const uint8_t* xb = xq + (size_t)min(r + 8, M - 1) * kb;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  constexpr int U = 8;   // steps whose loads are issued together (memory-level parallelism: 4-byte fragment loads)
  for (int sb = s0; sb < s1; sb += U) {
    uint32_t B0[U], B1[U], SB[U], A0[U], A1[U], A2[U], A3[U], SA[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int st = min(sb + u, s1 - 1), bo = st * 32;
      B0[u] = __ldg(reinterpret_cast<const uint32_t*>(wr + bo + 4 * q));
      B1[u] = __ldg(reinterpret_cast<const uint32_t*>(wr + bo + 16 + 4 * q));
      SB[u] = __ldg(reinterpret_cast<const uint32_t*>(wsr + st * 4));
      A0[u] = a0v ? *reinterpret_cast<const uint32_t*>(xa + bo + 4 * q) : 0u;
      A2[u] = a0v ? *reinterpret_cast<const uint32_t*>(xa + bo + 16 + 4 * q) : 0u;
      A1[u] = a1v ? *reinterpret_cast<const uint32_t*>(xb + bo + 4 * q) : 0u;
      A3[u] = a1v ? *reinterpret_cast<const uint32_t*>(xb + bo + 16 + 4 * q) : 0u;
      SA[u] = sav ? *reinterpret_cast<const uint32_t*>(xsr + st * 4) : 0u;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (sb + u >= s1) break;
      const uint32_t a[4] = {A0[u], A1[u], A2[u], A3[u]};
      mma_nvf4(acc, a, B0[u], B1[u], SA[u], SB[u]);
    }
  }
  part[tl][sl][r][2 * q] = acc[0]; part[tl][sl][r][2 * q + 1] = acc[1];
  part[tl][sl][r + 8][2 * q] = acc[2]; part[tl][sl][r + 8][2 * q + 1] = acc[3];
  __syncthreads();
  if (sl == 0)
    for (int i = lane; i < 16 * 8; i += 32) {
      const int m = i / 8, c = i % 8, n = n0 + c;
      if (m >= M || n >= N) continue;
      float v = part[tl][0][m][c];
      for (int z = 1; z < S; ++z) v += part[tl][z][m][c];
      v *= alpha;
      if (yb) yb[(size_t)m * ldy + n] = __float2bfloat16_rn(v); else yf[(size_t)m * ldy + n] = v;
    }
}

// row-major NVFP4 activation quantization (one thread per 16-value block)
__global__ void quant_nvfp4_rm_kernel(const bf16* x, int ldx, int M, int K, float ga, uint8_t* q, uint8_t* s) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int nb = K / 16;
  if (i >= (size_t)M * nb) return;
  const int m = (int)(i / nb), b = (int)(i % nb);
  const bf16* xr = x + (size_t)m * ldx + (size_t)b * 16;
  float v[16], a = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) { v[j] = __bfloat162float(xr[j]); a = fmaxf(a, fabsf(v[j])); }
  const __nv_fp8_storage_t se = __nv_cvt_float_to_fp8(a / 6.f * ga, __NV_SATFINITE, __NV_E4M3);
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(se, __NV_E4M3);
  const float sv = __half2float(*reinterpret_cast<const __half*>(&hr));
  const float f = sv > 0.f ? ga / sv : 0.f;
  uint8_t* qo = q + (size_t)m * (K / 2) + (size_t)b * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) qo[j] = (uint8_t)__nv_cvt_float2_to_fp4x2(make_float2(v[2 * j] * f, v[2 * j + 1] * f), __NV_E2M1, cudaRoundNearest);
  s[(size_t)m * nb + b] = (uint8_t)se;
}

#define LT_OK(expr) ((expr) == CUBLAS_STATUS_SUCCESS)
}  // namespace

size_t swz_bytes(int rows, int K) { return (size_t)pad_up(rows, 128) * pad_up(K / 16, 4); }

void swizzle_scales(const uint8_t* src, int rows, int K, uint8_t* dst, cudaStream_t st) {
  const int cols = K / 16;
  CUDA_CHECK(cudaMemsetAsync(dst, 0, swz_bytes(rows, K), st));
  swizzle_kernel<<<blocks((size_t)rows * cols), 256, 0, st>>>(src, rows, cols, pad_up(cols, 4), dst);
  CUDA_CHECK(cudaGetLastError());
}
void quant_fp8_rows(const bf16* x, int ldx, int M, int K, uint8_t* q, float* s, cudaStream_t st) {
  if (M <= 0) return;
  quant_fp8_kernel<<<M, 256, 0, st>>>(x, ldx, K, q, s);
  CUDA_CHECK(cudaGetLastError());
}
void quant_nvfp4_rows(const bf16* x, int ldx, int M, int K, float ga, uint8_t* q, uint8_t* s_swz, cudaStream_t st) {
  if (M <= 0) return;
  CUDA_CHECK(cudaMemsetAsync(s_swz, 0, swz_bytes(M, K), st));
  quant_nvfp4_kernel<<<blocks((size_t)M * (K / 16)), 256, 0, st>>>(x, ldx, M, K, ga, q, s_swz, pad_up(K / 16, 4));
  CUDA_CHECK(cudaGetLastError());
}
void quant_nvfp4_rows_dyn(const bf16* x, int ldx, int M, int K, float g, float* sc, uint8_t* q, uint8_t* s_swz, cudaStream_t st) {
  if (M <= 0) return;
  CUDA_CHECK(cudaMemsetAsync(sc, 0, 16, st));
  amax_kernel<<<256, 256, 0, st>>>(x, ldx, M, K, sc);
  dyn_scale_kernel<<<1, 1, 0, st>>>(sc, g);
  CUDA_CHECK(cudaMemsetAsync(s_swz, 0, swz_bytes(M, K), st));
  quant_nvfp4_dyn_kernel<<<blocks((size_t)M * (K / 16)), 256, 0, st>>>(x, ldx, M, K, sc, q, s_swz, pad_up(K / 16, 4));
  CUDA_CHECK(cudaGetLastError());
}
void scale_out(const float* acc, int M, int N, float alpha, const float* rs, const float* cs, bf16* y, int ldy, cudaStream_t st) {
  if (M <= 0) return;
  scale_out_kernel<<<blocks((size_t)M * N), 256, 0, st>>>(acc, M, N, alpha, rs, cs, y, ldy);
  CUDA_CHECK(cudaGetLastError());
}

void fp8_gemm_rows(const uint8_t* xq, int ldq, const float* xs, int M, const uint8_t* w, const float* ws, int N, int K, bf16* yb, float* yf,
                   int ldy, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= 16 && K % 256 == 0 && ldq % 16 == 0, "q27 fp8 decode gemm shape");
  fp8_gemm_kernel<<<(N + 15) / 16, 256, 0, st>>>(xq, ldq, xs, M, w, ws, N, K, yb, yf, ldy);
  CUDA_CHECK(cudaGetLastError());
}

void quant_nvfp4_rm(const bf16* x, int ldx, int M, int K, float ga, uint8_t* q, uint8_t* s, cudaStream_t st) {
  if (M <= 0) return;
  quant_nvfp4_rm_kernel<<<blocks((size_t)M * (K / 16)), 256, 0, st>>>(x, ldx, M, K, ga, q, s);
  CUDA_CHECK(cudaGetLastError());
}
void nvfp4_gemm_rows(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* w, const uint8_t* ws, int N, int K, float alpha, bf16* yb,
                     float* yf, int ldy, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= 16 && K % 256 == 0, "q27 nvfp4 decode gemm shape");
  nvfp4_gemm_kernel<<<(N + 15) / 16, 256, 0, st>>>(xq, xs, M, w, ws, N, K, alpha, yb, yf, ldy);
  CUDA_CHECK(cudaGetLastError());
}

Lt::Lt(cudaStream_t st) : st_(st) {
  cublasLtHandle_t h;
  if (cublasLtCreate(&h) == CUBLAS_STATUS_SUCCESS) h_ = h;
  CUDA_CHECK(cudaMalloc(&ws_, ws_bytes_));
}
Lt::~Lt() {
  if (h_) cublasLtDestroy((cublasLtHandle_t)h_);
  if (ws_) cudaFree(ws_);
}

namespace {
// D (col-major N × M, = row-major [M, N], row stride ldd; fp32 or bf16) = op(A = W, stored K × N col-major, transposed) · (B = X, stored
//   K × M col-major). Scales: VEC16_UE4M3 block scales (NVFP4), or OUTER_VEC_32F (FP8: per weight row, per activation row).
bool lt_run(cublasLtHandle_t h, void* ws, size_t wsb, cudaStream_t st, cudaDataType_t ab, const void* w, const void* x, int M, int N, int K,
            float alpha, void* d, cudaDataType_t dt, int ldd, const void* ws_scale, const void* xs_scale, bool outer,
            const float* alpha_dev = nullptr) {
  if (!h) return false;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  bool ok = LT_OK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
  ok = ok && LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA));
  ok = ok && LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB));
  if (alpha_dev) {   // alpha (and beta = alpha_dev[1]) read from the device
    const cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_DEVICE;
    ok = ok && LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof pm));
  }
  if (ws_scale) {
    const int32_t mode = outer ? CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F : CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    ok = ok && LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof mode));
    ok = ok && LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof mode));
    ok = ok && LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ws_scale, sizeof ws_scale));
    ok = ok && LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &xs_scale, sizeof xs_scale));
  }
  ok = ok && LT_OK(cublasLtMatrixLayoutCreate(&la, ab, K, N, K));
  ok = ok && LT_OK(cublasLtMatrixLayoutCreate(&lb, ab, K, M, K));
  ok = ok && LT_OK(cublasLtMatrixLayoutCreate(&ld, dt, N, M, ldd));
  ok = ok && LT_OK(cublasLtMatmulPreferenceCreate(&pref));
  ok = ok && LT_OK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb));
  // the heuristic's pick per shape is kept (the query cost ~tens of µs per GEMM call: ~450 calls per prefill chunk)
  static std::map<std::tuple<int, int, int, int, int, int, int, int>, cublasLtMatmulAlgo_t> algo_cache;
  const auto key = std::make_tuple((int)ab, (int)dt, M, N, K, ldd, (ws_scale ? (outer ? 2 : 1) : 0) + (alpha_dev ? 4 : 0), (int)wsb);
  cublasLtMatmulHeuristicResult_t res{};
  int n = 0;
  if (auto it = algo_cache.find(key); ok && it != algo_cache.end()) { res.algo = it->second; n = 1; }
  else {
    ok = ok && LT_OK(cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, ld, ld, pref, 1, &res, &n)) && n > 0;
    if (ok) algo_cache.emplace(key, res.algo);
  }
  const float beta = 0.f;
  if (alpha_dev) ok = ok && LT_OK(cublasLtMatmul(h, op, alpha_dev, w, la, x, lb, alpha_dev + 1, d, ld, d, ld, &res.algo, ws, wsb, st));
  else ok = ok && LT_OK(cublasLtMatmul(h, op, &alpha, w, la, x, lb, &beta, d, ld, d, ld, &res.algo, ws, wsb, st));
  if (pref) cublasLtMatmulPreferenceDestroy(pref);
  if (ld) cublasLtMatrixLayoutDestroy(ld);
  if (lb) cublasLtMatrixLayoutDestroy(lb);
  if (la) cublasLtMatrixLayoutDestroy(la);
  if (op) cublasLtMatmulDescDestroy(op);
  return ok;
}
}  // namespace

bool Lt::fp8(const uint8_t* xq, const uint8_t* wq, int M, int N, int K, float* acc) {
  return lt_run((cublasLtHandle_t)h_, ws_, ws_bytes_, st_, CUDA_R_8F_E4M3, wq, xq, M, N, K, 1.f, acc, CUDA_R_32F, N, nullptr, nullptr, false);
}
bool Lt::fp8_bf16(const uint8_t* xq, const float* xs, const uint8_t* wq, const float* wsc, int M, int N, int K, bf16* y, int ldy) {
  if (fp8_outer_off_) return false;
  const bool ok = lt_run((cublasLtHandle_t)h_, ws_, ws_bytes_, st_, CUDA_R_8F_E4M3, wq, xq, M, N, K, 1.f, y, CUDA_R_16BF, ldy, wsc, xs, true);
  if (!ok) fp8_outer_off_ = true;   // no outer-vector scaling here: the fp32 + scale_out path from now on
  return ok;
}
bool Lt::nvfp4(const uint8_t* xq, const uint8_t* xs_swz, const uint8_t* wq, const uint8_t* ws_swz, int M, int N, int K, float alpha, float* acc) {
  return lt_run((cublasLtHandle_t)h_, ws_, ws_bytes_, st_, CUDA_R_4F_E2M1, wq, xq, M, N, K, alpha, acc, CUDA_R_32F, N, ws_swz, xs_swz, false);
}
bool Lt::nvfp4_bf16_dev(const uint8_t* xq, const uint8_t* xs_swz, const uint8_t* wq, const uint8_t* ws_swz, int M, int N, int K, const float* alpha_beta,
                        bf16* y, int ldy) {
  return lt_run((cublasLtHandle_t)h_, ws_, ws_bytes_, st_, CUDA_R_4F_E2M1, wq, xq, M, N, K, 0.f, y, CUDA_R_16BF, ldy, ws_swz, xs_swz, false, alpha_beta);
}
bool Lt::nvfp4_bf16(const uint8_t* xq, const uint8_t* xs_swz, const uint8_t* wq, const uint8_t* ws_swz, int M, int N, int K, float alpha, bf16* y, int ldy) {
  return lt_run((cublasLtHandle_t)h_, ws_, ws_bytes_, st_, CUDA_R_4F_E2M1, wq, xq, M, N, K, alpha, y, CUDA_R_16BF, ldy, ws_swz, xs_swz, false);
}

}  // namespace hive::q27::lowp
