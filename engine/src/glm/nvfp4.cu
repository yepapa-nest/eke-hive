// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/glm/nvfp4.h"

#include "hive/common.h"

namespace hive::glm {

namespace {
__device__ __constant__ float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__device__ __forceinline__ float e4m3f(uint8_t v) {
  __nv_fp8_e4m3 t; t.__x = v; return float(t);
}

// One warp per output row n, all M activation rows at once. Each lane walks 16-byte chunks (32 columns, two scales) of the row.
template <int MM>
__global__ void gemv_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, float g, int N, int K,
                            const __nv_bfloat16* __restrict__ x, int ldx, int M, float* __restrict__ y, int ldy) {
  extern __shared__ __nv_bfloat16 xs[];  // [M, K] bf16 (RTX Blackwell: ≤ 99 KB dynamic smem per block)
  for (int i = threadIdx.x; i < M * K; i += blockDim.x) xs[i] = x[(size_t)(i / K) * ldx + (i % K)];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * (blockDim.x >> 5) + warp;
  if (n >= N) return;
  const uint4* wr = reinterpret_cast<const uint4*>(W + (size_t)n * (K / 2));
  const uint8_t* sr = S + (size_t)n * (K / 16);
  float acc[MM];
#pragma unroll
  for (int m = 0; m < MM; ++m) acc[m] = 0.f;
  for (int c = lane; c < K / 32; c += 32) {
    const uint4 q = __ldg(wr + c);
    const float s0 = e4m3f(sr[2 * c]), s1 = e4m3f(sr[2 * c + 1]);
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
    float w[32];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const uint32_t byte = (words[j] >> (8 * b)) & 0xFF;
        w[j * 8 + b * 2] = kE2M1[byte & 15];
        w[j * 8 + b * 2 + 1] = kE2M1[byte >> 4];
      }
#pragma unroll
    for (int m = 0; m < MM; ++m) {
      if (m >= M) break;
      const __nv_bfloat16* xr = xs + (size_t)m * K + c * 32;
      float lo = 0.f, hi = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) lo = fmaf(w[i], __bfloat162float(xr[i]), lo);
#pragma unroll
      for (int i = 16; i < 32; ++i) hi = fmaf(w[i], __bfloat162float(xr[i]), hi);
      acc[m] = fmaf(lo, s0, fmaf(hi, s1, acc[m]));
    }
  }
#pragma unroll
  for (int m = 0; m < MM; ++m) {
    if (m >= M) break;
    float v = acc[m];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (lane == 0) y[(size_t)m * ldy + n] = v * g;
  }
}

__global__ void dequant_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, float g, int N, int K, __nv_bfloat16* __restrict__ out) {
  const size_t pairs = (size_t)N * K / 2;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < pairs; i += (size_t)gridDim.x * blockDim.x) {
    const size_t n = i / (K / 2), kp = i % (K / 2);
    const float s = e4m3f(S[n * (K / 16) + (kp * 2) / 16]) * g;
    const uint8_t b = W[i];
    __nv_bfloat162 v; v.x = __float2bfloat16(kE2M1[b & 15] * s); v.y = __float2bfloat16(kE2M1[b >> 4] * s);
    reinterpret_cast<__nv_bfloat162*>(out)[i] = v;
  }
}
}  // namespace

void nvfp4_gemv(const Nvfp4Mat& W, const __nv_bfloat16* x, int ldx, int M, float* y, int ldy, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= 8 && W.K % 32 == 0, "nvfp4_gemv shape");
  const int warps = 8;
  const size_t smem = (size_t)M * W.K * sizeof(__nv_bfloat16);
  HIVE_CHECK(smem <= 96 * 1024, "nvfp4_gemv smem");
  static bool attr = false;
  if (!attr) { CUDA_CHECK(cudaFuncSetAttribute(gemv_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); attr = true; }
  gemv_kernel<8><<<(W.N + warps - 1) / warps, warps * 32, smem, st>>>(W.w, W.s, W.g, W.N, W.K, x, ldx, M, y, ldy);
  CUDA_CHECK(cudaGetLastError());
}

void nvfp4_dequant(const Nvfp4Mat& W, __nv_bfloat16* out, cudaStream_t st) {
  const size_t pairs = (size_t)W.N * W.K / 2;
  const int blocks = (int)std::min<size_t>((pairs + 255) / 256, 65535 * 4);
  dequant_kernel<<<blocks, 256, 0, st>>>(W.w, W.s, W.g, W.N, W.K, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::glm
