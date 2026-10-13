// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// PLE block kernels — see hive/qwen/ple.h. Roundings follow the reference's bf16 tensors: the dequantized row (× weight_scale), the normed
//   key / query and their product, the gate, the gated value, the conv output and its SiLU, the block output and the updated hidden.
//   The three conv paths (sequence, batch, verify rows) use the same tap expression, so a token gives the same bits in all of them.
#include "hive/qwen/ple.h"

#include <cuda_fp8.h>

#include <algorithm>

#include "hive/common.h"

namespace hive::qwen {
namespace {

__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }

__global__ void ple_dequant_kernel(const uint8_t* __restrict__ rows, int M, float scale, bf16* __restrict__ e) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * kPleRow) return;
  __nv_fp8_e4m3 v; v.__x = rows[i];
  e[i] = __float2bfloat16(rbf(float(v)) * scale);
}

__device__ __forceinline__ float block_sum(float v, float* red) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
  __syncthreads();
  if (l == 0) red[w] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < nw; ++i) t += red[i];
  return t;
}

// block = (row m, copy c)
__global__ void ple_gate_kernel(const bf16* __restrict__ key, const bf16* __restrict__ h, const bf16* __restrict__ nk, const bf16* __restrict__ nq,
                                const bf16* __restrict__ value, int hc, int dim, float eps, bf16* __restrict__ gv) {
  __shared__ float red[32];
  const int m = blockIdx.x, c = blockIdx.y;
  const size_t o = (size_t)m * hc * dim + (size_t)c * dim;
  const bf16* kr = key + o;
  const bf16* hr = h + o;
  float sk = 0.f, sq = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float a = __bfloat162float(kr[i]), b = __bfloat162float(hr[i]);
    sk = fmaf(a, a, sk); sq = fmaf(b, b, sq);
  }
  const float rk = rsqrtf(block_sum(sk, red) / (float)dim + eps);
  const float rq = rsqrtf(block_sum(sq, red) / (float)dim + eps);
  float dot = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float kn = rbf(__bfloat162float(kr[i]) * rk * (1.f + __bfloat162float(nk[(size_t)c * dim + i])));
    const float qn = rbf(__bfloat162float(hr[i]) * rq * (1.f + __bfloat162float(nq[(size_t)c * dim + i])));
    dot += rbf(kn * qn);
  }
  float g = rbf(rbf(block_sum(dot, red)) * rsqrtf((float)dim));
  const float a = rbf(sqrtf(fmaxf(fabsf(g), 1e-6f)));
  g = rbf(a * (g > 0.f ? 1.f : g < 0.f ? -1.f : 0.f));
  g = rbf(1.f / (1.f + expf(-g)));
  for (int i = threadIdx.x; i < dim; i += blockDim.x) gv[o + i] = __float2bfloat16(g * __bfloat162float(value[(size_t)m * dim + i]));
}

// conv taps of channel ch: x(t−9), x(t−6), x(t−3), x(t) — one expression for every path
__device__ __forceinline__ float conv4(const float* __restrict__ w, float x0, float x1, float x2, float x3) {
  float y = w[0] * x0;
  y = fmaf(w[1], x1, y);
  y = fmaf(w[2], x2, y);
  y = fmaf(w[3], x3, y);
  return y;
}
__device__ __forceinline__ float ple_out(float gv, float conv, float h) {
  const float cv = rbf(conv);
  const float s = rbf(cv / (1.f + expf(-cv)));
  return rbf(h + rbf(gv + s));
}

// sequence: thread per (row t, channel); x(j) for j < 0 comes from the state (slot 9 + j)
__global__ void ple_conv_seq_kernel(const bf16* __restrict__ gv, const bf16* __restrict__ gvn, const float* __restrict__ w, int T, int C,
                                    const float* __restrict__ state, bf16* __restrict__ h) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)T * C) return;
  const int t = (int)(i / C), ch = (int)(i % C);
  auto x = [&](int j) { return j >= 0 ? __bfloat162float(gvn[(size_t)j * C + ch]) : state[(size_t)(kPleConvState + j) * C + ch]; };
  const float y = conv4(w + (size_t)ch * 4, x(t - 9), x(t - 6), x(t - 3), x(t));
  h[i] = __float2bfloat16(ple_out(__bfloat162float(gv[i]), y, __bfloat162float(h[i])));
}
__global__ void ple_conv_state_kernel(const bf16* __restrict__ gvn, int T, int C, float* __restrict__ state) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= C) return;
  float nv[kPleConvState];
#pragma unroll
  for (int s = 0; s < kPleConvState; ++s) {
    const int j = T - kPleConvState + s;  // input index relative to the call
    nv[s] = j >= 0 ? __bfloat162float(gvn[(size_t)j * C + ch]) : state[(size_t)(kPleConvState + j) * C + ch];
  }
#pragma unroll
  for (int s = 0; s < kPleConvState; ++s) state[(size_t)s * C + ch] = nv[s];
}

__global__ void ple_conv_batch_kernel(const bf16* __restrict__ gv, const bf16* __restrict__ gvn, const float* __restrict__ w, int M, int C,
                                      float* const* __restrict__ states, bf16* __restrict__ h) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * C) return;
  const int m = (int)(i / C), ch = (int)(i % C);
  float* s = states[m];
  float st[kPleConvState];
#pragma unroll
  for (int k = 0; k < kPleConvState; ++k) st[k] = s[(size_t)k * C + ch];
  const float xt = __bfloat162float(gvn[i]);
  const float y = conv4(w + (size_t)ch * 4, st[0], st[3], st[6], xt);
  h[i] = __float2bfloat16(ple_out(__bfloat162float(gv[i]), y, __bfloat162float(h[i])));
#pragma unroll
  for (int k = 0; k + 1 < kPleConvState; ++k) s[(size_t)k * C + ch] = st[k + 1];
  s[(size_t)(kPleConvState - 1) * C + ch] = xt;
}

__global__ void ple_conv_rows_kernel(const bf16* __restrict__ gv, const bf16* __restrict__ gvn, const float* __restrict__ w, int R, int C,
                                     float* const* __restrict__ state_pp, float* const* __restrict__ snap, bf16* __restrict__ h) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= C) return;
  float* state = *state_pp;   // through a pointer table: the verify graphs are replayed for other sequences
  float st[kPleConvState];
#pragma unroll
  for (int k = 0; k < kPleConvState; ++k) st[k] = state[(size_t)k * C + ch];
  for (int r = 0; r < R; ++r) {
    const size_t i = (size_t)r * C + ch;
    const float xt = __bfloat162float(gvn[i]);
    const float y = conv4(w + (size_t)ch * 4, st[0], st[3], st[6], xt);
    h[i] = __float2bfloat16(ple_out(__bfloat162float(gv[i]), y, __bfloat162float(h[i])));
#pragma unroll
    for (int k = 0; k + 1 < kPleConvState; ++k) st[k] = st[k + 1];
    st[kPleConvState - 1] = xt;
    if (snap && r + 1 < R)
#pragma unroll
      for (int k = 0; k < kPleConvState; ++k) snap[r][(size_t)k * C + ch] = st[k];
  }
#pragma unroll
  for (int k = 0; k < kPleConvState; ++k) state[(size_t)k * C + ch] = st[k];
}

unsigned blocks_for(size_t n, int t = 256) { return (unsigned)std::max<size_t>(1, (n + t - 1) / t); }

}  // namespace

void ple_dequant(const uint8_t* rows, int M, float scale, bf16* e, cudaStream_t st) {
  if (M <= 0) return;
  ple_dequant_kernel<<<blocks_for((size_t)M * kPleRow), 256, 0, st>>>(rows, M, scale, e);
  CUDA_CHECK(cudaGetLastError());
}

void ple_gate(const bf16* key, const bf16* h, const bf16* norm_key, const bf16* norm_query, const bf16* value, int M, int hc, int dim, float eps, bf16* gv,
              cudaStream_t st) {
  if (M <= 0) return;
  ple_gate_kernel<<<dim3(M, hc), 256, 0, st>>>(key, h, norm_key, norm_query, value, hc, dim, eps, gv);
  CUDA_CHECK(cudaGetLastError());
}

void ple_conv_seq(const bf16* gv, const bf16* gvn, const float* conv_w, int T, int C, float* state, bf16* h, cudaStream_t st) {
  if (T <= 0) return;
  ple_conv_seq_kernel<<<blocks_for((size_t)T * C), 256, 0, st>>>(gv, gvn, conv_w, T, C, state, h);
  ple_conv_state_kernel<<<blocks_for((size_t)C), 256, 0, st>>>(gvn, T, C, state);
  CUDA_CHECK(cudaGetLastError());
}

void ple_conv_batch(const bf16* gv, const bf16* gvn, const float* conv_w, int M, int C, float* const* states, bf16* h, cudaStream_t st) {
  if (M <= 0) return;
  ple_conv_batch_kernel<<<blocks_for((size_t)M * C), 256, 0, st>>>(gv, gvn, conv_w, M, C, states, h);
  CUDA_CHECK(cudaGetLastError());
}

void ple_conv_rows(const bf16* gv, const bf16* gvn, const float* conv_w, int R, int C, float* const* state, float* const* snap, bf16* h, cudaStream_t st) {
  if (R <= 0) return;
  ple_conv_rows_kernel<<<blocks_for((size_t)C), 256, 0, st>>>(gv, gvn, conv_w, R, C, state, snap, h);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::qwen
