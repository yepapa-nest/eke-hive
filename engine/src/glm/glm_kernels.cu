// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/glm/glm_kernels.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>

#include "hive/common.h"

namespace hive::glm {
namespace {
__global__ void layernorm_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ b, float eps,
                                 int K, __nv_bfloat16* __restrict__ out) {
  const __nv_bfloat16* xr = x + (size_t)blockIdx.x * K;
  __shared__ float red[2][32];
  float s = 0.f, s2 = 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) { const float v = __bfloat162float(xr[i]); s += v; s2 += v * v; }
  for (int o = 16; o > 0; o >>= 1) { s += __shfl_xor_sync(0xffffffff, s, o); s2 += __shfl_xor_sync(0xffffffff, s2, o); }
  if ((threadIdx.x & 31) == 0) { red[0][threadIdx.x >> 5] = s; red[1][threadIdx.x >> 5] = s2; }
  __syncthreads();
  if (threadIdx.x < 32) {
    const int nw = blockDim.x >> 5;
    s = threadIdx.x < nw ? red[0][threadIdx.x] : 0.f; s2 = threadIdx.x < nw ? red[1][threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) { s += __shfl_xor_sync(0xffffffff, s, o); s2 += __shfl_xor_sync(0xffffffff, s2, o); }
    if (threadIdx.x == 0) { red[0][0] = s; red[1][0] = s2; }
  }
  __syncthreads();
  const float mean = red[0][0] / K, var = fmaxf(red[1][0] / K - mean * mean, 0.f), inv = rsqrtf(var + eps);
  for (int i = threadIdx.x; i < K; i += blockDim.x)
    out[(size_t)blockIdx.x * K + i] = __float2bfloat16((__bfloat162float(xr[i]) - mean) * inv * __bfloat162float(w[i]) + __bfloat162float(b[i]));
}

// one block per row; E ≤ 1024, k ≤ 16
__global__ void router_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int E, int k, float scale, int32_t* __restrict__ ids,
                              float* __restrict__ w) {
  extern __shared__ float sm[];
  float* s = sm; float* c = sm + E;
  const float* lr = logits + (size_t)blockIdx.x * E;
  for (int e = threadIdx.x; e < E; e += blockDim.x) { const float v = 1.f / (1.f + expf(-lr[e])); s[e] = v; c[e] = v + bias[e]; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.f;
    int32_t sel[16]; float sv[16];
    for (int j = 0; j < k; ++j) {
      int best = -1; float bv = -FLT_MAX;
      for (int e = 0; e < E; ++e) if (c[e] > bv) { bv = c[e]; best = e; }
      sel[j] = best; sv[j] = s[best]; c[best] = -FLT_MAX; sum += s[best];
    }
    const float inv = 1.f / (sum + 1e-20f);
    for (int j = 0; j < k; ++j) { ids[(size_t)blockIdx.x * k + j] = sel[j]; w[(size_t)blockIdx.x * k + j] = sv[j] * inv * scale; }
  }
}

__global__ void stream_mean_kernel(const __nv_bfloat16* __restrict__ h, int hc, int dim, __nv_bfloat16* __restrict__ x) {
  const int m = blockIdx.y;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += gridDim.x * blockDim.x) {
    float s = 0.f;
    for (int c = 0; c < hc; ++c) s += __bfloat162float(h[((size_t)m * hc + c) * dim + i]);
    x[(size_t)m * dim + i] = __float2bfloat16(s / hc);
  }
}
__global__ void f2b_kernel(const float* __restrict__ a, size_t n, __nv_bfloat16* __restrict__ o) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) o[i] = __float2bfloat16(a[i]);
}
__global__ void scale_kernel(float* __restrict__ a, size_t n, float s) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) a[i] *= s;
}
__global__ void add_bf16_kernel(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, size_t n, __nv_bfloat16* __restrict__ o) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    o[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}
inline int blocks_for(size_t n) { return (int)std::min<size_t>((n + 255) / 256, 8192); }
}  // namespace

void layernorm(const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* b, float eps, int M, int K, __nv_bfloat16* out, cudaStream_t st) {
  if (M <= 0) return;
  layernorm_kernel<<<M, 128, 0, st>>>(x, w, b, eps, K, out);
  CUDA_CHECK(cudaGetLastError());
}
// Parallel top-k router (same selection as router_kernel: largest biased score, ties → smaller expert id; weights from the unbiased
//   sigmoid scores, summed in selection order). blockIdx.y = half h of a row whose logits hold two layers back to back ([cur E | next E],
//   row stride ld); half 1 (the next layer's prediction) uses bias1 / ids1 / w1.
__global__ void router2_kernel(const float* __restrict__ logits, int ld, const float* __restrict__ bias0, const float* __restrict__ bias1, int E, int k,
                               float scale, int32_t* __restrict__ ids0, float* __restrict__ w0, int32_t* __restrict__ ids1, float* __restrict__ w1) {
  extern __shared__ float sm[];
  float* s = sm; float* c = sm + E;
  __shared__ float wv[32]; __shared__ int wi[32]; __shared__ int sel[16];
  const int h = blockIdx.y;
  const float* bias = h ? bias1 : bias0;
  const float* lr = logits + (size_t)blockIdx.x * ld + (size_t)h * E;
  for (int e = threadIdx.x; e < E; e += blockDim.x) { const float v = 1.f / (1.f + expf(-lr[e])); s[e] = v; c[e] = v + bias[e]; }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int j = 0; j < k; ++j) {
    float bv = -FLT_MAX; int bi = 0x7fffffff;
    for (int e = threadIdx.x; e < E; e += blockDim.x) { const float v = c[e]; if (v > bv || (v == bv && e < bi)) { bv = v; bi = e; } }
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffff, bv, o); const int oi = __shfl_xor_sync(0xffffffff, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { wv[warp] = bv; wi[warp] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
      float v = wv[0]; int i = wi[0];
      for (int q = 1; q < nw; ++q) if (wv[q] > v || (wv[q] == v && wi[q] < i)) { v = wv[q]; i = wi[q]; }
      sel[j] = i; c[i] = -FLT_MAX;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    int32_t* ids = h ? ids1 : ids0; float* w = h ? w1 : w0;
    float sum = 0.f;
    for (int j = 0; j < k; ++j) sum += s[sel[j]];
    const float inv = 1.f / (sum + 1e-20f);
    for (int j = 0; j < k; ++j) { ids[(size_t)blockIdx.x * k + j] = sel[j]; w[(size_t)blockIdx.x * k + j] = s[sel[j]] * inv * scale; }
  }
}

void router_topk2(const float* logits, int ld, const float* bias0, const float* bias1, int M, int E, int k, float scale, int32_t* ids0, float* w0,
                  int32_t* ids1, float* w1, cudaStream_t st) {
  HIVE_CHECK(E <= 1024 && k <= 16, "router_topk2 shape");
  if (M <= 0) return;
  router2_kernel<<<dim3(M, bias1 ? 2 : 1), 256, 2 * E * sizeof(float), st>>>(logits, ld, bias0, bias1, E, k, scale, ids0, w0, ids1, w1);
  CUDA_CHECK(cudaGetLastError());
}

void router_topk(const float* logits, const float* bias, int M, int E, int k, float scale, int32_t* ids, float* w, cudaStream_t st) {
  HIVE_CHECK(E <= 1024 && k <= 16, "router_topk shape");
  if (M <= 0) return;
  router_kernel<<<M, 256, 2 * E * sizeof(float), st>>>(logits, bias, E, k, scale, ids, w);
  CUDA_CHECK(cudaGetLastError());
}
void hc_stream_mean(const __nv_bfloat16* h, int M, int hc, int dim, __nv_bfloat16* x, cudaStream_t st) {
  if (M <= 0) return;
  stream_mean_kernel<<<dim3((dim + 255) / 256, M), 256, 0, st>>>(h, hc, dim, x);
  CUDA_CHECK(cudaGetLastError());
}
void f32_to_bf16(const float* a, size_t n, __nv_bfloat16* out, cudaStream_t st) { if (n) { f2b_kernel<<<blocks_for(n), 256, 0, st>>>(a, n, out); CUDA_CHECK(cudaGetLastError()); } }
void scale_f32(float* x, size_t n, float s, cudaStream_t st) { if (n) { scale_kernel<<<blocks_for(n), 256, 0, st>>>(x, n, s); CUDA_CHECK(cudaGetLastError()); } }
void add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, size_t n, __nv_bfloat16* out, cudaStream_t st) {
  if (n) { add_bf16_kernel<<<blocks_for(n), 256, 0, st>>>(a, b, n, out); CUDA_CHECK(cudaGetLastError()); }
}
// ---- small helpers for MTP / speculative verify ---------------------------------------------------------------------------------------------
namespace {
__global__ void dcopy_kernel(uint4* __restrict__ d, const uint4* __restrict__ s, size_t n16) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) d[i] = s[i];
}
__global__ void cat_rows_kernel(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, int n, int H, __nv_bfloat16* __restrict__ out) {
  const size_t total = (size_t)n * 2 * H;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const size_t r = i / (2 * H), c = i % (2 * H);
    out[i] = c < (size_t)H ? a[r * H + c] : b[r * H + c - H];
  }
}
__global__ void accum_scaled_kernel(float* __restrict__ acc, const __nv_bfloat16* __restrict__ x, float w, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) acc[i] += w * __bfloat162float(x[i]);
}
}  // namespace

void dcopy(void* dst, const void* src, size_t bytes, cudaStream_t st) {
  HIVE_CHECK(bytes % 16 == 0 && ((uintptr_t)dst % 16) == 0 && ((uintptr_t)src % 16) == 0, "dcopy alignment");
  const size_t n16 = bytes / 16;
  if (!n16) return;
  const int blocks = (int)std::min<size_t>((n16 + 255) / 256, 1024);
  dcopy_kernel<<<blocks, 256, 0, st>>>(reinterpret_cast<uint4*>(dst), reinterpret_cast<const uint4*>(src), n16);
  CUDA_CHECK(cudaGetLastError());
}
void cat_rows(const __nv_bfloat16* a, const __nv_bfloat16* b, int n, int H, __nv_bfloat16* out, cudaStream_t st) {
  const size_t total = (size_t)n * 2 * H;
  cat_rows_kernel<<<(int)std::min<size_t>((total + 255) / 256, 4096), 256, 0, st>>>(a, b, n, H, out);
  CUDA_CHECK(cudaGetLastError());
}
void accum_scaled(float* acc, const __nv_bfloat16* x, float w, size_t n, cudaStream_t st) {
  accum_scaled_kernel<<<(int)std::min<size_t>((n + 255) / 256, 1024), 256, 0, st>>>(acc, x, w, n);
  CUDA_CHECK(cudaGetLastError());
}

namespace {
__global__ void scatter_add_f32_kernel(const float* __restrict__ src, const int32_t* __restrict__ idx, int n, int H, float* __restrict__ out) {
  const size_t total = (size_t)n * H;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const size_t r = i / H, c = i % H;
    out[(size_t)idx[r] * H + c] += src[i];
  }
}
}  // namespace
void scatter_add_f32_rows(const float* src, const int32_t* idx, int n, int H, float* out, cudaStream_t st) {
  if (n <= 0) return;
  const size_t total = (size_t)n * H;
  scatter_add_f32_kernel<<<(int)std::min<size_t>((total + 255) / 256, 8192), 256, 0, st>>>(src, idx, n, H, out);
  CUDA_CHECK(cudaGetLastError());
}

namespace {
__global__ void embed_gather_dev_kernel(const __nv_bfloat16* __restrict__ table, const int32_t* __restrict__ id, int H, __nv_bfloat16* __restrict__ out) {
  const size_t row = (size_t)*id;
  const uint4* src = reinterpret_cast<const uint4*>(table + row * H);
  uint4* dst = reinterpret_cast<uint4*>(out);
  for (int i = threadIdx.x; i < H / 8; i += blockDim.x) dst[i] = src[i];
}
}  // namespace
void embed_gather_dev(const __nv_bfloat16* table, const int32_t* id_dev, int H, __nv_bfloat16* out, cudaStream_t st) {
  HIVE_CHECK(H % 8 == 0, "embed_gather_dev");
  embed_gather_dev_kernel<<<1, 256, 0, st>>>(table, id_dev, H, out);
  CUDA_CHECK(cudaGetLastError());
}

namespace {
constexpr int kRowsPerBlock = 8;  // 8 warps
__global__ void gemv_rows_kernel(const __nv_bfloat16* __restrict__ W, const int32_t* __restrict__ rows, int n,
                                 const __nv_bfloat16* __restrict__ x, int H, float* __restrict__ out) {
  extern __shared__ uint4 xs[];  // x as 16-byte chunks (H / 8)
  const int nv = H / 8;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) xs[i] = reinterpret_cast<const uint4*>(x)[i];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int r = blockIdx.x * kRowsPerBlock + warp;
  if (r >= n) return;
  const uint4* w = reinterpret_cast<const uint4*>(W + (size_t)rows[r] * H);
  float acc = 0.f;
  for (int i = lane; i < nv; i += 32) {
    const uint4 a = w[i], b = xs[i];
    const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 fa = __bfloat1622float2(pa[j]), fb = __bfloat1622float2(pb[j]);
      acc = fmaf(fa.x, fb.x, acc);
      acc = fmaf(fa.y, fb.y, acc);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) out[r] = acc;
}
__global__ void map_id_kernel(const int32_t* __restrict__ idx, const int32_t* __restrict__ table, int32_t* __restrict__ out) { *out = table[*idx]; }
}  // namespace

void gemv_rows_bf16(const __nv_bfloat16* W, const int32_t* rows, int n, const __nv_bfloat16* x, int H, float* out, cudaStream_t st) {
  HIVE_CHECK(H % 8 == 0 && n > 0, "gemv_rows_bf16");
  gemv_rows_kernel<<<(n + kRowsPerBlock - 1) / kRowsPerBlock, 32 * kRowsPerBlock, (size_t)H * 2, st>>>(W, rows, n, x, H, out);
  CUDA_CHECK(cudaGetLastError());
}

void map_id_dev(const int32_t* idx, const int32_t* table, int32_t* out, cudaStream_t st) {
  map_id_kernel<<<1, 1, 0, st>>>(idx, table, out);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void hc_inject_kernel(bf16* __restrict__ h, const float* __restrict__ post, const float* __restrict__ y, int M, int hc, int dim) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * dim) return;
  const int m = (int)(i / dim), d = (int)(i % dim);
  const float yv = y[i];
  for (int c = 0; c < hc; ++c) {
    bf16* p = h + ((size_t)m * hc + c) * dim + d;
    *p = __float2bfloat16(__bfloat162float(*p) + post[m * hc + c] * yv);
  }
}
void hc_inject(bf16* h, const float* post, const float* y, int M, int hc, int dim, cudaStream_t st) {
  const size_t n = (size_t)M * dim;
  hc_inject_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(h, post, y, M, hc, dim);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::glm
