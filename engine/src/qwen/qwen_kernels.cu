// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Small Qwen3.8-Flash-Next kernels — see hive/qwen/qwen_kernels.h for the contracts.
#include "hive/qwen/qwen_kernels.h"

#include <algorithm>

#include "hive/common.h"
#include "hive/qwen/fn_pdl.h"
#include "hive/glm/glm_moe.h"

namespace hive::qwen {
namespace {

__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }

__device__ __forceinline__ float block_sum(float v, float* red) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
  __syncthreads();  // red may still be read by the previous call
  if (l == 0) red[w] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < nw; ++i) t += red[i];  // fixed order: every thread gets the same value
  return t;
}

// block = (row m, group g): out = x · rsqrt(mean(x²) + eps) · (1 + w)
__global__ void gemma_rmsnorm_kernel(const bf16* __restrict__ x, int ldx, const bf16* __restrict__ w, int group, float eps, bf16* __restrict__ out,
                                     int ldo) {
  pdl_wait();
  __shared__ float red[32];
  const int m = blockIdx.x, g = blockIdx.y;
  const bf16* xr = x + (size_t)m * ldx + (size_t)g * group;
  float ss = 0.f;
  for (int i = threadIdx.x; i < group; i += blockDim.x) { const float v = __bfloat162float(xr[i]); ss = fmaf(v, v, ss); }
  const float r = rsqrtf(block_sum(ss, red) / (float)group + eps);
  bf16* o = out + (size_t)m * ldo + (size_t)g * group;
  const bf16* wg = w + (size_t)g * group;
  for (int i = threadIdx.x; i < group; i += blockDim.x)
    o[i] = __float2bfloat16(__bfloat162float(xr[i]) * r * (1.f + __bfloat162float(wg[i])));
}

__global__ void hc_mix_act_kernel(const bf16* __restrict__ dl, int ld, int M, int rank, float inv_hc, bf16* __restrict__ s) {
  pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * rank) return;
  const int m = i / rank, j = i % rank;
  const float v = rbf(__bfloat162float(dl[(size_t)m * ld + j]) * inv_hc);
  s[i] = __float2bfloat16(v * sigm(v));
}

__global__ void hc_mix_out_kernel(const bf16* __restrict__ u, const bf16* __restrict__ hn, int M, int hc, int dim, bf16* __restrict__ x) {
  pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * dim) return;
  const int m = i / dim, d = i % dim;
  float acc = 0.f;
  for (int c = 0; c < hc; ++c) {
    const size_t o = (size_t)m * hc * dim + (size_t)c * dim + d;
    acc += rbf(rbf(sigm(__bfloat162float(u[o]))) * __bfloat162float(hn[o]));
  }
  x[i] = __float2bfloat16(acc / (float)hc);
}

__global__ void hc_combine_kernel(bf16* __restrict__ h, const bf16* __restrict__ y, int ldy, const bf16* __restrict__ dl, int ld, int rank, int M, int hc,
                                  int dim, float inv_hc) {
  pdl_wait();
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * hc * dim) return;
  const int m = (int)(i / ((size_t)hc * dim)), c = (int)((i / dim) % hc), d = (int)(i % dim);
  const float g = rbf(__bfloat162float(dl[(size_t)m * ld + rank + c]) * inv_hc);
  const float s2 = 2.f * rbf(sigm(g));
  const float inj = rbf(__bfloat162float(y[(size_t)m * ldy + d]) * s2);
  h[i] = __float2bfloat16(__bfloat162float(h[i]) + inj);
}

// 8 elements per thread (dim % 8 == 0): the same expressions as hc_mix_out_kernel / hc_combine_kernel, 16-byte loads and stores — the
//   prefill calls (16K rows) ran at ~630 GB/s with one element per thread (nsys 2026-10-10: hc_combine 1.2 ms per call). Prefill
//   rows only (M ≥ 64): on decode rows the 8× fewer threads made hc_mix_out slower (0.18 → 0.36 ms per MTP step)
__global__ void hc_mix_out8_kernel(const bf16* __restrict__ u, const bf16* __restrict__ hn, int M, int hc, int dim, bf16* __restrict__ x) {
  pdl_wait();
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= (size_t)M * dim) return;
  const int m = (int)(i / dim), d = (int)(i % dim);
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int c = 0; c < hc; ++c) {
    const size_t o = (size_t)m * hc * dim + (size_t)c * dim + d;
    const uint4 uv = *reinterpret_cast<const uint4*>(u + o), hv = *reinterpret_cast<const uint4*>(hn + o);
    const bf16* ue = reinterpret_cast<const bf16*>(&uv);
    const bf16* he = reinterpret_cast<const bf16*>(&hv);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] += rbf(rbf(sigm(__bfloat162float(ue[j]))) * __bfloat162float(he[j]));
  }
  uint4 ov;
  bf16* oe = reinterpret_cast<bf16*>(&ov);
#pragma unroll
  for (int j = 0; j < 8; ++j) oe[j] = __float2bfloat16(acc[j] / (float)hc);
  *reinterpret_cast<uint4*>(x + i) = ov;
}

__global__ void hc_combine8_kernel(bf16* __restrict__ h, const bf16* __restrict__ y, int ldy, const bf16* __restrict__ dl, int ld, int rank, int M, int hc,
                                   int dim, float inv_hc) {
  pdl_wait();
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= (size_t)M * hc * dim) return;
  const int m = (int)(i / ((size_t)hc * dim)), c = (int)((i / dim) % hc), d = (int)(i % dim);
  const float g = rbf(__bfloat162float(dl[(size_t)m * ld + rank + c]) * inv_hc);
  const float s2 = 2.f * rbf(sigm(g));
  uint4 hv = *reinterpret_cast<const uint4*>(h + i);
  const uint4 yv = *reinterpret_cast<const uint4*>(y + (size_t)m * ldy + d);
  bf16* he = reinterpret_cast<bf16*>(&hv);
  const bf16* ye = reinterpret_cast<const bf16*>(&yv);
#pragma unroll
  for (int j = 0; j < 8; ++j) he[j] = __float2bfloat16(__bfloat162float(he[j]) + rbf(__bfloat162float(ye[j]) * s2));
  *reinterpret_cast<uint4*>(h + i) = hv;
}

// hc_combine fused with the gemma_rmsnorm that follows it (the next mix's input norm, group = dim per copy): block = (row, copy), the
//   rmsnorm kernel's thread count and loop order, so h and hn are bit-identical to hc_combine_kernel → gemma_rmsnorm_kernel
__global__ void hc_combine_norm_kernel(bf16* __restrict__ h, const bf16* __restrict__ y, int ldy, const bf16* __restrict__ dl, int ld, int rank, int hc,
                                       int dim, float inv_hc, const bf16* __restrict__ w, float eps, bf16* __restrict__ hn) {
  pdl_wait();
  __shared__ float red[32];
  const int m = blockIdx.x, c = blockIdx.y;
  const float g = rbf(__bfloat162float(dl[(size_t)m * ld + rank + c]) * inv_hc);
  const float s2 = 2.f * rbf(sigm(g));
  bf16* hr = h + ((size_t)m * hc + c) * dim;
  const bf16* yr = y + (size_t)m * ldy;
  float ss = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const bf16 nv = __float2bfloat16(__bfloat162float(hr[i]) + rbf(__bfloat162float(yr[i]) * s2));
    hr[i] = nv;
    const float v = __bfloat162float(nv);
    ss = fmaf(v, v, ss);
  }
  const float r = rsqrtf(block_sum(ss, red) / (float)dim + eps);
  bf16* o = hn + ((size_t)m * hc + c) * dim;
  const bf16* wg = w + (size_t)c * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) o[i] = __float2bfloat16(__bfloat162float(hr[i]) * r * (1.f + __bfloat162float(wg[i])));
}

__global__ void hc_expand_kernel(const bf16* __restrict__ x, int M, int hc, int dim, bf16* __restrict__ h) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * hc * dim) return;
  const int m = (int)(i / ((size_t)hc * dim)), d = (int)(i % dim);
  h[i] = x[(size_t)m * dim + d];
}

// one warp per row: softmax over E bf16-rounded logits, k argmax rounds (strict >: the smaller id wins ties), optional renormalisation
// kMaxPer = logits per lane (E ≤ 32·kMaxPer). Every index into p[] is a compile-time constant (unrolled), so it stays in registers; lane r
//   keeps the r-th pick and every lane sums the picked weights in pick order (the same fp32 order as a serial sum)
template <int kMaxPer>
__global__ void router_kernel(const float* __restrict__ logits, int ld, int M, int E, int k, bool renorm, int32_t* __restrict__ ids, float* __restrict__ w,
                              int32_t* ids_host, float* w_host) {
  pdl_wait();
  const int m = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (m >= M) return;
  float p[kMaxPer];
  const float* lr = logits + (size_t)m * ld;
  float mx = -INFINITY;
#pragma unroll
  for (int j = 0; j < kMaxPer; ++j) {
    const int e = lane + 32 * j;
    p[j] = e < E ? rbf(lr[e]) : -INFINITY;
    mx = fmaxf(mx, p[j]);
  }
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < kMaxPer; ++j) { p[j] = lane + 32 * j < E ? expf(p[j] - mx) : 0.f; sum += p[j]; }
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  const float inv = 1.f / sum;
  float my_w = 0.f, tot = 0.f;
  int my_id = 0;
  for (int r = 0; r < k; ++r) {
    float best = -1.f; int bid = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < kMaxPer; ++j) {
      const int e = lane + 32 * j;
      if (e < E && p[j] >= 0.f && (p[j] > best || (p[j] == best && e < bid))) { best = p[j]; bid = e; }
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bid, o);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
    }
    const float sw = best * inv;
    tot += sw;
    if (lane == r) { my_id = bid; my_w = sw; }
#pragma unroll
    for (int j = 0; j < kMaxPer; ++j) if (lane + 32 * j == bid) p[j] = -1.f;  // taken
  }
  if (lane < k) {
    const float sc = renorm ? 1.f / tot : 1.f;
    const float wv = my_w * sc;
    ids[(size_t)m * k + lane] = my_id; w[(size_t)m * k + lane] = wv;
    if (ids_host) { ids_host[(size_t)m * k + lane] = my_id; w_host[(size_t)m * k + lane] = wv; }
  }
}

__global__ void moe_combine_kernel(const float* __restrict__ routed, const bf16* __restrict__ shared, const float* __restrict__ logits, int ld, int col, int M,
                                   int dim, bf16* __restrict__ out) {
  pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * dim) return;
  const int m = i / dim;
  const float g = rbf(sigm(rbf(logits[(size_t)m * ld + col])));
  out[i] = __float2bfloat16(rbf(routed[i]) + rbf(g * __bfloat162float(shared[i])));
}

__global__ void build_pairs_kernel(const int32_t* __restrict__ ids, const float* __restrict__ w, int M, int K, const uint8_t* const* __restrict__ tab,
                                   int li, int E, MoePairDev* __restrict__ pairs) {
  pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * K) return;
  const int m = i / K;
  const int e = ids[i];
  const uint8_t* rec = (e >= 0 && e < E) ? tab[(size_t)li * E + e] : nullptr;
  float wv = w[i];
  if (!rec) {  // not resident (the caller should not have taken this path) — contribute nothing, keep a valid record pointer
    wv = 0.f;
    for (int k = 0; k < K && !rec; ++k) { const int e2 = ids[m * K + k]; if (e2 >= 0 && e2 < E) rec = tab[(size_t)li * E + e2]; }
    if (!rec) rec = tab[(size_t)li * E];
  }
  pairs[i] = MoePairDev{m, rec, wv};
}

__global__ void embed_rows_kernel(const bf16* __restrict__ table, const int32_t* __restrict__ ids, int M, int dim, bf16* __restrict__ out) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * dim) return;
  const int m = (int)(i / dim), d = (int)(i % dim);
  out[i] = table[(size_t)ids[m] * dim + d];
}

__global__ void copy_rows_kernel(const bf16* __restrict__ src, int lds, int n, int dim, bf16* __restrict__ dst, int ldd) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n * dim) return;
  const int r = (int)(i / dim), d = (int)(i % dim);
  dst[(size_t)r * ldd + d] = src[(size_t)r * lds + d];
}

__global__ void hc_add_rows_kernel(bf16* __restrict__ h, const bf16* __restrict__ x, int M, int hc, int dim) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * hc * dim) return;
  const int m = (int)(i / ((size_t)hc * dim)), d = (int)(i % dim);
  h[i] = __float2bfloat16(__bfloat162float(h[i]) + __bfloat162float(x[(size_t)m * dim + d]));
}

__global__ void write_rope_kernel(const int32_t* __restrict__ src, int32_t* const* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * 3) return;
  dst[i / 3][i % 3] = src[i];
}

unsigned blocks_for(size_t n, int t = 256) { return (unsigned)std::max<size_t>(1, (n + t - 1) / t); }

}  // namespace

static_assert(sizeof(MoePairDev) == sizeof(glm::MoePair) && alignof(MoePairDev) == alignof(glm::MoePair), "MoePairDev mirrors glm::MoePair");

void gemma_rmsnorm(const bf16* x, int ldx, const bf16* w, int M, int K, int group, float eps, bf16* out, int ldo, cudaStream_t st) {
  if (M <= 0) return;
  HIVE_CHECK(group > 0 && K % group == 0, "gemma_rmsnorm: group must divide K");
  launch_pdl(gemma_rmsnorm_kernel, dim3(M, K / group), dim3(group >= 1024 ? 512 : 256), 0, st, x, ldx, w, group, eps, out, ldo);
  CUDA_CHECK(cudaGetLastError());
}

void hc_mix_act(const bf16* dl, int ld, int M, int rank, int hc, bf16* s, cudaStream_t st) {
  if (M <= 0) return;
  launch_pdl(hc_mix_act_kernel, dim3(blocks_for((size_t)M * rank)), dim3(256), 0, st, dl, ld, M, rank, 1.f / hc, s);
  CUDA_CHECK(cudaGetLastError());
}

void hc_mix_out(const bf16* u, const bf16* hn, int M, int hc, int dim, bf16* x, cudaStream_t st) {
  if (M <= 0) return;
  if (M >= 64 && dim % 8 == 0 && ((uintptr_t)u & 15) == 0 && ((uintptr_t)hn & 15) == 0 && ((uintptr_t)x & 15) == 0)
    launch_pdl(hc_mix_out8_kernel, dim3(blocks_for((size_t)M * dim / 8)), dim3(256), 0, st, u, hn, M, hc, dim, x);
  else launch_pdl(hc_mix_out_kernel, dim3(blocks_for((size_t)M * dim)), dim3(256), 0, st, u, hn, M, hc, dim, x);
  CUDA_CHECK(cudaGetLastError());
}

void hc_combine(bf16* h, const bf16* y, int ldy, const bf16* dl, int ld, int rank, int M, int hc, int dim, cudaStream_t st) {
  if (M <= 0) return;
  if (M >= 64 && dim % 8 == 0 && ldy % 8 == 0 && ((uintptr_t)h & 15) == 0 && ((uintptr_t)y & 15) == 0)
    launch_pdl(hc_combine8_kernel, dim3(blocks_for((size_t)M * hc * dim / 8)), dim3(256), 0, st, h, y, ldy, dl, ld, rank, M, hc, dim, 1.f / hc);
  else launch_pdl(hc_combine_kernel, dim3(blocks_for((size_t)M * hc * dim)), dim3(256), 0, st, h, y, ldy, dl, ld, rank, M, hc, dim, 1.f / hc);
  CUDA_CHECK(cudaGetLastError());
}

void hc_combine_norm(bf16* h, const bf16* y, int ldy, const bf16* dl, int ld, int rank, int M, int hc, int dim, const bf16* w, float eps, bf16* hn,
                     cudaStream_t st) {
  if (M <= 0) return;
  launch_pdl(hc_combine_norm_kernel, dim3(M, hc), dim3(dim >= 1024 ? 512 : 256), 0, st, h, y, ldy, dl, ld, rank, hc, dim, 1.f / hc, w, eps, hn);
  CUDA_CHECK(cudaGetLastError());
}

void hc_expand(const bf16* x, int M, int hc, int dim, bf16* h, cudaStream_t st) {
  if (M <= 0) return;
  hc_expand_kernel<<<blocks_for((size_t)M * hc * dim), 256, 0, st>>>(x, M, hc, dim, h);
  CUDA_CHECK(cudaGetLastError());
}

void router_softmax_topk(const float* logits, int ld, int M, int E, int k, bool renorm, int32_t* ids, float* w, int32_t* ids_host, float* w_host,
                         cudaStream_t st) {
  if (M <= 0) return;
  HIVE_CHECK(E >= 1 && E <= 1024 && k >= 1 && k <= 16 && k <= E, "router_softmax_topk shape");
  if (E <= 256) launch_pdl(router_kernel<8>, dim3((M + 7) / 8), dim3(256), 0, st, logits, ld, M, E, k, renorm, ids, w, ids_host, w_host);
  else if (E <= 512) launch_pdl(router_kernel<16>, dim3((M + 7) / 8), dim3(256), 0, st, logits, ld, M, E, k, renorm, ids, w, ids_host, w_host);
  else launch_pdl(router_kernel<32>, dim3((M + 7) / 8), dim3(256), 0, st, logits, ld, M, E, k, renorm, ids, w, ids_host, w_host);
  CUDA_CHECK(cudaGetLastError());
}

void moe_combine(const float* routed, const bf16* shared, const float* logits, int ld, int col, int M, int dim, bf16* out, cudaStream_t st) {
  if (M <= 0) return;
  launch_pdl(moe_combine_kernel, dim3(blocks_for((size_t)M * dim)), dim3(256), 0, st, routed, shared, logits, ld, col, M, dim, out);
  CUDA_CHECK(cudaGetLastError());
}

void build_moe_pairs(const int32_t* ids, const float* w, int M, int K, const uint8_t* const* rec_tab, int li, int E, MoePairDev* pairs, cudaStream_t st) {
  if (M <= 0) return;
  launch_pdl(build_pairs_kernel, dim3(blocks_for((size_t)M * K, 128)), dim3(128), 0, st, ids, w, M, K, rec_tab, li, E, pairs);
  CUDA_CHECK(cudaGetLastError());
}

void hc_add_rows(bf16* h, const bf16* x, int M, int hc, int dim, cudaStream_t st) {
  if (M <= 0) return;
  hc_add_rows_kernel<<<blocks_for((size_t)M * hc * dim), 256, 0, st>>>(h, x, M, hc, dim);
  CUDA_CHECK(cudaGetLastError());
}

void write_rope(const int32_t* src, int32_t* const* dst, int n, cudaStream_t st) {
  if (n <= 0) return;
  write_rope_kernel<<<blocks_for((size_t)n * 3), 256, 0, st>>>(src, dst, n);
  CUDA_CHECK(cudaGetLastError());
}

void embed_rows(const bf16* table, const int32_t* ids, int M, int dim, bf16* out, cudaStream_t st) {
  if (M <= 0) return;
  embed_rows_kernel<<<blocks_for((size_t)M * dim), 256, 0, st>>>(table, ids, M, dim, out);
  CUDA_CHECK(cudaGetLastError());
}

void copy_rows(const bf16* src, int lds, int n, int dim, bf16* dst, int ldd, cudaStream_t st) {
  if (n <= 0) return;
  copy_rows_kernel<<<blocks_for((size_t)n * dim), 256, 0, st>>>(src, lds, n, dim, dst, ldd);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::qwen
