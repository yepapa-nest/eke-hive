// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/glm/fp8b.h"

#include <cuda_fp8.h>

#include <algorithm>

#include "hive/common.h"

namespace hive::glm {

namespace {
__device__ __forceinline__ float e4m3f(uint32_t v) {
  __nv_fp8_e4m3 t; t.__x = (uint8_t)v; return float(t);
}

// One warp per output row; each lane takes 16-byte chunks (16 columns). Chunks c..c+7 share one 128-column scale block.
// Activations are read through the read-only cache (x is small and shared by every warp).
template <int MM>
__global__ void __launch_bounds__(256) fp8b_gemv_kernel(const uint8_t* __restrict__ W, const float* __restrict__ S, int N, int K,
                                                         const __nv_bfloat16* __restrict__ x, int ldx, int M, __nv_bfloat16* __restrict__ y, int ldy) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * (blockDim.x >> 5) + warp;
  if (n >= N) return;
  const int kb_n = (K + 127) / 128;
  const uint4* wr = reinterpret_cast<const uint4*>(W + (size_t)n * K);
  const float* sr = S + (size_t)(n / 128) * kb_n;
  float acc[MM];
#pragma unroll
  for (int m = 0; m < MM; ++m) acc[m] = 0.f;
  for (int c = lane; c < K / 16; c += 32) {
    const uint4 q = __ldg(wr + c);
    const float s = __ldg(sr + (c >> 3));
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
    float w[16];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int b = 0; b < 4; ++b) w[j * 4 + b] = e4m3f((words[j] >> (8 * b)) & 0xFF);
#pragma unroll
    for (int m = 0; m < MM; ++m) {
      if (m >= M) break;
      const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)m * ldx + c * 16);
      const uint4 a0 = __ldg(xr), a1 = __ldg(xr + 1);
      const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
      const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&a1);
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(p0[i]);
        d = fmaf(w[2 * i], f.x, d); d = fmaf(w[2 * i + 1], f.y, d);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(p1[i]);
        d = fmaf(w[8 + 2 * i], f.x, d); d = fmaf(w[8 + 2 * i + 1], f.y, d);
      }
      acc[m] = fmaf(d, s, acc[m]);
    }
  }
#pragma unroll
  for (int m = 0; m < MM; ++m) {
    if (m >= M) break;
    float v = acc[m];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (lane == 0) y[(size_t)m * ldy + n] = __float2bfloat16(v);
  }
}

__global__ void fp8b_dequant_kernel(const uint8_t* __restrict__ W, const float* __restrict__ S, int N, int K, __nv_bfloat16* __restrict__ out) {
  const int kb_n = (K + 127) / 128;
  const size_t quads = (size_t)N * K / 4;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < quads; i += (size_t)gridDim.x * blockDim.x) {
    const size_t e = i * 4, n = e / K, k = e % K;
    const float s = S[(n / 128) * kb_n + k / 128];
    const uint32_t q = reinterpret_cast<const uint32_t*>(W)[i];
    __nv_bfloat162 a, b;
    a.x = __float2bfloat16(e4m3f(q & 0xFF) * s); a.y = __float2bfloat16(e4m3f((q >> 8) & 0xFF) * s);
    b.x = __float2bfloat16(e4m3f((q >> 16) & 0xFF) * s); b.y = __float2bfloat16(e4m3f(q >> 24) * s);
    reinterpret_cast<__nv_bfloat162*>(out)[2 * i] = a;
    reinterpret_cast<__nv_bfloat162*>(out)[2 * i + 1] = b;
  }
}
}  // namespace

void fp8b_gemv(const Fp8BMat& W, const __nv_bfloat16* x, int ldx, int M, __nv_bfloat16* y, int ldy, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= 8 && W.K % 16 == 0 && ldx % 8 == 0, "fp8b_gemv shape");
  const int warps = 8;
  fp8b_gemv_kernel<8><<<(W.N + warps - 1) / warps, warps * 32, 0, st>>>(W.w, W.s, W.N, W.K, x, ldx, M, y, ldy);
  CUDA_CHECK(cudaGetLastError());
}

void fp8b_dequant(const Fp8BMat& W, __nv_bfloat16* out, cudaStream_t st) {
  HIVE_CHECK(W.K % 4 == 0, "fp8b_dequant shape");
  const size_t quads = (size_t)W.N * W.K / 4;
  const int blocks = (int)std::min<size_t>((quads + 255) / 256, 65535 * 4);
  fp8b_dequant_kernel<<<blocks, 256, 0, st>>>(W.w, W.s, W.N, W.K, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::glm
