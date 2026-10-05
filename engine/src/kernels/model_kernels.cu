// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Model body kernels — numeric semantics follow the reference model.py location noted above each function.
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <cstring>

#include "hive/model_kernels.h"
#include "hive/warp_reduce.cuh"
#include "hc_decode_fused.cuh"  // HIVE_HC_DECODE_FUSED: 16-lane sinkhorn + last-block tail

namespace hive::k {

namespace {

__device__ inline float warp_sum(float v) { return hive::cu::warp_allreduce(v); }
__device__ inline float warp_max(float v) { return hive::cu::warp_allreduce<hive::cu::Max>(v); }
// Block-wide sum (no blockDim 256 assumption: any size <= 1024)
template <int THREADS>
__device__ inline float block_sum(float v, float* red) {
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
  if (threadIdx.x < 32) {
    t = threadIdx.x < THREADS / 32 ? red[threadIdx.x] : 0.f;
    t = warp_sum(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  t = red[0];
  __syncthreads();
  return t;
}

__global__ void expand_block_scale_kernel(const uint8_t* __restrict__ src, int N, int nbk, uint8_t* __restrict__ dst) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)N * nbk) return;
  int n = i / nbk, b = i % nbk;
  dst[i] = src[(size_t)(n / 32) * nbk + b];
}

// fp8 weights with per-row expanded scales (load_fp8's expand_block_scale: [N, K/32]) → bf16. e4m3 (4 significant bits) × 2^k fits exactly in bf16, so
//   the values equal the bf16 dequantization of the reference convert.py. Used to expand prefill wo_a per layer into scratch for a cuBLAS bf16 GEMM (8 elements at a time).
__global__ void dequant_fp8_rows_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ s, int N, int K, bf16* __restrict__ out) {
  const size_t i8 = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i8 >= (size_t)N * K) return;
  const int n = i8 / K, k = i8 % K;
  const float sc = e8m0_to_f32(s[(size_t)n * (K / 32) + k / 32]);
  const uint2 raw = *reinterpret_cast<const uint2*>(w + i8);
  const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
  uint4 o;
  o.x = (uint32_t)__bfloat16_as_ushort(f2bf(e4m3_to_f32(b[0]) * sc)) | ((uint32_t)__bfloat16_as_ushort(f2bf(e4m3_to_f32(b[1]) * sc)) << 16);
  o.y = (uint32_t)__bfloat16_as_ushort(f2bf(e4m3_to_f32(b[2]) * sc)) | ((uint32_t)__bfloat16_as_ushort(f2bf(e4m3_to_f32(b[3]) * sc)) << 16);
  o.z = (uint32_t)__bfloat16_as_ushort(f2bf(e4m3_to_f32(b[4]) * sc)) | ((uint32_t)__bfloat16_as_ushort(f2bf(e4m3_to_f32(b[5]) * sc)) << 16);
  o.w = (uint32_t)__bfloat16_as_ushort(f2bf(e4m3_to_f32(b[6]) * sc)) | ((uint32_t)__bfloat16_as_ushort(f2bf(e4m3_to_f32(b[7]) * sc)) << 16);
  *reinterpret_cast<uint4*>(out + i8) = o;
}

// Prefill hc mix coefficients — fuses the three steps bf16→fp32 copy (hc_flatten) + cuBLAS fp32 GEMM (N=24 skinny; 0.85 ms = 2.4 TFLOPS at M=2048) + row rsqrt
//   into one kernel. Block = HCRT rows; 256 threads split K (hcdim=20480) 8 elements at a time and stream W (fp32, 2 MB, L2-resident) once.
//   mixes[m][j] = Σ_k f32(h[m][k])·W[j][k] (fp32 FMA), rsq[m] = rsqrt(mean(h²)+eps) — same formula as hc_mix_kernel (decode); only the accumulation order differs.
constexpr int HCMIX = 24, HCRT = 4;
__global__ void __launch_bounds__(256) hc_mix_rows_kernel(const bf16* __restrict__ h, const float* __restrict__ W, int M, int hcdim, float eps,
                                                          float* __restrict__ mixes, float* __restrict__ rsq) {
  __shared__ float red[8][HCRT * (HCMIX + 1)];
  const int m0 = blockIdx.x * HCRT;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  float acc[HCRT][HCMIX + 1];
#pragma unroll
  for (int r = 0; r < HCRT; ++r)
#pragma unroll
    for (int j = 0; j <= HCMIX; ++j) acc[r][j] = 0.f;
  for (int k = tid * 8; k < hcdim; k += 256 * 8) {
    float hv[HCRT][8];
#pragma unroll
    for (int r = 0; r < HCRT; ++r) {
      const int m = m0 + r;
      uint4 raw = make_uint4(0, 0, 0, 0);
      if (m < M) raw = *reinterpret_cast<const uint4*>(h + (size_t)m * hcdim + k);
      const uint32_t u[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) { hv[r][2 * i] = __uint_as_float(u[i] << 16); hv[r][2 * i + 1] = __uint_as_float(u[i] & 0xFFFF0000u); }
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[r][HCMIX] = fmaf(hv[r][i], hv[r][i], acc[r][HCMIX]);
    }
#pragma unroll
    for (int j = 0; j < HCMIX; ++j) {  // must be fully unrolled so acc[r][j] stays in registers (partial unrolling spilled 400 B to local memory — measured with cuobjdump)
      const float4 w0 = *reinterpret_cast<const float4*>(W + (size_t)j * hcdim + k);
      const float4 w1 = *reinterpret_cast<const float4*>(W + (size_t)j * hcdim + k + 4);
      const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
      for (int r = 0; r < HCRT; ++r)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[r][j] = fmaf(hv[r][i], wv[i], acc[r][j]);
    }
  }
#pragma unroll
  for (int r = 0; r < HCRT; ++r)
#pragma unroll
    for (int j = 0; j <= HCMIX; ++j) {
      float v = acc[r][j];
      v = hive::cu::warp_allreduce(v);
      if (lane == 0) red[warp][r * (HCMIX + 1) + j] = v;
    }
  __syncthreads();
  for (int i = tid; i < HCRT * (HCMIX + 1); i += 256) {
    float t = 0.f;
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) t += red[wv][i];
    const int r = i / (HCMIX + 1), j = i % (HCMIX + 1);
    const int m = m0 + r;
    if (m >= M) continue;
    if (j < HCMIX) mixes[(size_t)m * HCMIX + j] = t;
    else rsq[m] = rsqrtf(t / (float)hcdim + eps);
  }
}

// hc_mix_rows, 16-row version: the kernel above streams all of W (24 × 20480 fp32 = 1.97 MB) from L2 per block (4 rows), ≈1 GB of L2 traffic for W alone at M=2048.
//   Block = 16 rows (8 warps = 4-row groups rg 0..3 × window halves kh 0..1). K is split into 512-element windows; the W window [24][512] (48 KB) is loaded into
//   smem with a 2-stage cp.async pipeline and shared by the four row groups → 4× less W traffic (M/16 blocks · 1.97 MB). The per-thread micro-kernel (4 rows × 25
//   output accumulators, 8-element steps) is the same as above.
//   · ~180 registers means 1 block/SM anyway → using 98,304 B of smem (<= 99 KB opt-in) costs no occupancy. M=2048 → 128 blocks (one wave).
//   · smem layout: within a window half (256 elements) a lane's 8 elements are split into the first 4 (A, 128 slots) and last 4 (B, 128 slots) at lane*4 — LDS.128
//     is then 16 B-contiguous across lanes, so there are no bank conflicts (a contiguous 32 B layout would be 2-way).
//   · Deterministic sum: fixed k order within a thread → warp xor shuffle (fixed) → kh 0 + kh 1 (fixed). No atomics. Differs from the 4-row version only in fp32 accumulation order (test_hc_mix).
constexpr int HCR16 = 16, HCKT = 512;
constexpr size_t HCW_STAGE = (size_t)HCMIX * HCKT;               // float 12,288 = 48KB
constexpr size_t HCMIX16_SMEM = 2 * HCW_STAGE * sizeof(float);   // 98,304
static_assert(HCMIX16_SMEM <= 99 * 1024, "sm_120 opt-in dynamic smem per block is 99KB");
static_assert((size_t)2 * HCR16 * (HCMIX + 1) <= 2 * HCW_STAGE, "reduction scratch reuses the W stages");
__device__ __forceinline__ void hc_cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(smem)), "l"(gmem));
}
__global__ void __launch_bounds__(256, 1) hc_mix_rows16_kernel(const bf16* __restrict__ h, const float* __restrict__ W, int M, int hcdim, float eps,
                                                               float* __restrict__ mixes, float* __restrict__ rsq) {
  extern __shared__ __align__(16) float hsw[];  // [2][24][512] W windows → afterwards [2 kh][16 rows][25] sum buffer
  const int m0 = blockIdx.x * HCR16;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rg = warp & 3, kh = warp >> 2;
  const int nsteps = hcdim / HCKT;
  // W of window s into buffer b: 16 B chunk c (4 floats, window elements 4c) → [j][kh′·256 + part·128 + lane′·4]
  auto stage = [&](int s, int b) {
    if (s < nsteps) {
      float* dst = hsw + (size_t)b * HCW_STAGE;
      for (int i = tid; i < HCMIX * (HCKT / 4); i += 256) {
        const int j = i / (HCKT / 4), c = i % (HCKT / 4);
        const int e0 = 4 * c, khp = e0 >> 8, e = e0 & 255, lp = e >> 3, part = (e >> 2) & 1;
        hc_cp_async16(dst + j * HCKT + khp * 256 + part * 128 + lp * 4, W + (size_t)j * hcdim + (size_t)s * HCKT + e0);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);  // commit even an empty group — keeps the wait count uniform
  };
  float acc[4][HCMIX + 1];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int j = 0; j <= HCMIX; ++j) acc[r][j] = 0.f;
  stage(0, 0);
  for (int s = 0; s < nsteps; ++s) {
    const int b = s & 1;
    stage(s + 1, b ^ 1);  // b^1 = buffer of window s-1 — nobody reads it after the sync at the end of the previous iteration
    const int k = s * HCKT + kh * 256 + lane * 8;
    float hv[4][8];
#pragma unroll
    for (int r = 0; r < 4; r++) {  // issue the h loads first to overlap with the W wait
      const int m = m0 + rg * 4 + r;
      uint4 raw = make_uint4(0, 0, 0, 0);
      if (m < M) raw = *reinterpret_cast<const uint4*>(h + (size_t)m * hcdim + k);
      const uint32_t u[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) { hv[r][2 * i] = __uint_as_float(u[i] << 16); hv[r][2 * i + 1] = __uint_as_float(u[i] & 0xFFFF0000u); }
    }
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[r][HCMIX] = fmaf(hv[r][i], hv[r][i], acc[r][HCMIX]);
    const float* wb = hsw + (size_t)b * HCW_STAGE + kh * 256 + lane * 4;
#pragma unroll
    for (int j = 0; j < HCMIX; ++j) {  // fully unrolled (same reason as the kernel above — partial unrolling spills acc to local memory)
      const float4 w0 = *reinterpret_cast<const float4*>(wb + j * HCKT);
      const float4 w1 = *reinterpret_cast<const float4*>(wb + j * HCKT + 128);
      const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
      for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[r][j] = fmaf(hv[r][i], wv[i], acc[r][j]);
    }
    __syncthreads();  // before stage(s+2) of the next iteration overwrites buffer b
  }
  asm volatile("cp.async.wait_group 0;\n" ::);
  // Sum: warp xor shuffle → red[kh][row][j] → kh 0 + kh 1. The window buffers are reusable after the last sync above.
  float* red = hsw;
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int j = 0; j <= HCMIX; ++j) {
      float v = acc[r][j];
      v = hive::cu::warp_allreduce(v);
      if (lane == 0) red[(kh * HCR16 + rg * 4 + r) * (HCMIX + 1) + j] = v;
    }
  __syncthreads();
  for (int i = tid; i < HCR16 * (HCMIX + 1); i += 256) {
    const float t = red[i] + red[HCR16 * (HCMIX + 1) + i];
    const int r = i / (HCMIX + 1), j = i % (HCMIX + 1);
    const int m = m0 + r;
    if (m >= M) continue;
    if (j < HCMIX) mixes[(size_t)m * HCMIX + j] = t;
    else rsq[m] = rsqrtf(t / (float)hcdim + eps);
  }
}

// Dump (golden comparison) only: set positions outside the candidates to -inf — so index_score has the same shape as the keep-mask output of the full-T path (cand is ascending → binary search)
__global__ void mask_by_cand_kernel(bf16* __restrict__ score, int M, int T, int bs, const int32_t* __restrict__ cand, int cand_stride, int kb) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * T) return;
  const int m = i / T, b = (int)(i % T) / bs;
  const int32_t* c = cand + (size_t)m * cand_stride;
  int lo = 0, hi = kb;
  while (lo < hi) { const int mid = (lo + hi) >> 1; const int32_t v = c[mid]; if (v < 0 || v > b) hi = mid; else if (v < b) lo = mid + 1; else return; }
  score[i] = f2bf(-INFINITY);
}

__global__ void dequant_fp8_block_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ s, int N, int K,
                                         bf16* __restrict__ out) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)N * K) return;
  int n = i / K, k = i % K;
  float sc = e8m0_to_f32(s[(size_t)(n / 32) * (K / 32) + k / 32]);
  out[i] = f2bf(e4m3_to_f32(w[i]) * sc);
}

__global__ void embed_expand_kernel(const bf16* __restrict__ embed, const int32_t* __restrict__ ids, int M, int dim, int hc,
                                    bf16* __restrict__ h) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * hc * dim) return;
  int m = i / ((size_t)hc * dim), d = i % dim;
  h[i] = embed[(size_t)ids[m] * dim + d];
}

__global__ void identity_pre_mix_kernel(float* pre, int M, int hc) {
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= M * hc) return;
  pre[i] = (i % hc == 0) ? 1.f : 0.f;
}

__global__ void hc_row_rsqrt_kernel(const bf16* __restrict__ h, int hcdim, float eps, float* __restrict__ rsq) {
  __shared__ float red[32];
  const bf16* p = h + (size_t)blockIdx.x * hcdim;
  float ss = 0.f;
  for (int i = threadIdx.x; i < hcdim; i += 256) {
    float v = bf2f(p[i]);
    ss += v * v;
  }
  ss = block_sum<256>(ss, red);
  if (threadIdx.x == 0) rsq[blockIdx.x] = rsqrtf(ss / (float)hcdim + eps);
}

__global__ void bf16_to_f32_kernel(const bf16* __restrict__ src, size_t n, float* __restrict__ dst) {
  const size_t i = hive::cu::global_tid();
  if (i < n) dst[i] = bf2f(src[i]);
}
__global__ void f32_to_bf16_kernel(const float* __restrict__ src, size_t n, bf16* __restrict__ dst) {
  const size_t i = hive::cu::global_tid();
  if (i < n) dst[i] = f2bf(src[i]);
}

// Reference hc_split_sinkhorn_kernel. One thread = one token (hc=4 → 16 matrices; sequential is enough).
// HC is a template constant: with a runtime hc loop, c[16] is dynamically indexed → spilled to local memory, and 20 iterations took ~20 µs (measured).
template <int HC>
__device__ __forceinline__ void sinkhorn_one(const float* __restrict__ mx, float r, const float* __restrict__ scale,
                                             const float* __restrict__ base, int iters, float eps, float* __restrict__ pre,
                                             float* __restrict__ post, float* __restrict__ comb) {
  constexpr int hc = HC;
  float c[hc * hc];
#pragma unroll
  for (int j = 0; j < hc; ++j) pre[j] = 1.f / (1.f + expf(-(mx[j] * r * scale[0] + base[j]))) + eps;
#pragma unroll
  for (int j = 0; j < hc; ++j) post[j] = 2.f / (1.f + expf(-(mx[j + hc] * r * scale[1] + base[j + hc])));
#pragma unroll
  for (int j = 0; j < hc; ++j)
#pragma unroll
    for (int kk = 0; kk < hc; ++kk) c[j * hc + kk] = mx[j * hc + kk + 2 * hc] * r * scale[2] + base[j * hc + kk + 2 * hc];
  // softmax(-1) + eps
#pragma unroll
  for (int j = 0; j < hc; ++j) {
    float mxv = -FLT_MAX;
#pragma unroll
    for (int kk = 0; kk < hc; ++kk) mxv = fmaxf(mxv, c[j * hc + kk]);
    float s = 0.f;
#pragma unroll
    for (int kk = 0; kk < hc; ++kk) { c[j * hc + kk] = expf(c[j * hc + kk] - mxv); s += c[j * hc + kk]; }
#pragma unroll
    for (int kk = 0; kk < hc; ++kk) c[j * hc + kk] = c[j * hc + kk] / s + eps;
  }
  // col normalize, then (iters-1) × (row, col)
#pragma unroll
  for (int kk = 0; kk < hc; ++kk) {
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < hc; ++j) s += c[j * hc + kk];
#pragma unroll
    for (int j = 0; j < hc; ++j) c[j * hc + kk] = c[j * hc + kk] / (s + eps);
  }
  for (int it = 0; it < iters - 1; ++it) {
#pragma unroll
    for (int j = 0; j < hc; ++j) {
      float s = 0.f;
#pragma unroll
      for (int kk = 0; kk < hc; ++kk) s += c[j * hc + kk];
#pragma unroll
      for (int kk = 0; kk < hc; ++kk) c[j * hc + kk] = c[j * hc + kk] / (s + eps);
    }
#pragma unroll
    for (int kk = 0; kk < hc; ++kk) {
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < hc; ++j) s += c[j * hc + kk];
#pragma unroll
      for (int j = 0; j < hc; ++j) c[j * hc + kk] = c[j * hc + kk] / (s + eps);
    }
  }
#pragma unroll
  for (int j = 0; j < hc * hc; ++j) comb[j] = c[j];
}

template <int HC>
__global__ void sinkhorn_kernel(const float* __restrict__ mixes, const float* __restrict__ rsq, const float* __restrict__ scale,
                                const float* __restrict__ base, int M, int iters, float eps, float* __restrict__ pre,
                                float* __restrict__ post, float* __restrict__ comb) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M) return;
  constexpr int mix_hc = (2 + HC) * HC;
  sinkhorn_one<HC>(mixes + (size_t)m * mix_hc, rsq[m], scale, base, iters, eps, pre + m * HC, post + m * HC, comb + (size_t)m * HC * HC);
}

// ---- Decode hc fusion (M <= 8): (1) block (m, j) = dot product for mix coefficient j (or the row rsqrt if j == mix_hc)
// The bodies are device functions — the separate kernels and the combined kernel (hc_mix_pre_norm) run **the same code**, so they are bit-identical (test_hc_fuse)
__device__ __forceinline__ void hc_mix_body(const bf16* __restrict__ h, const float* __restrict__ W, int hcdim, int mix_hc, float eps,
                                            float* __restrict__ mixes, float* __restrict__ rsq, int m, int j, float* red) {
  const bf16* p = h + (size_t)m * hcdim;
  float acc = 0.f;
  if (j < mix_hc) {
    const float* wr = W + (size_t)j * hcdim;
    for (int i = threadIdx.x; i < hcdim; i += 256) acc = fmaf(bf2f(p[i]), wr[i], acc);
  } else {
    for (int i = threadIdx.x; i < hcdim; i += 256) { float v = bf2f(p[i]); acc += v * v; }  // keep this exact formula (changing it may change the bits)
  }
  acc = hive::cu::warp_allreduce(acc);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int i = 0; i < 8; ++i) t += red[i];
    if (j < mix_hc) mixes[(size_t)m * mix_hc + j] = t;
    else rsq[m] = rsqrtf(t / (float)hcdim + eps);
  }
}
__global__ void hc_mix_kernel(const bf16* __restrict__ h, const float* __restrict__ W, int hcdim, int mix_hc, float eps,
                              float* __restrict__ mixes, float* __restrict__ rsq) {
  __shared__ float red[8];
  hc_mix_body(h, W, hcdim, mix_hc, eps, mixes, rsq, blockIdx.x, blockIdx.y, red);
}
// (2′) block m: pre + rmsnorm only (sinkhorn runs separately in sinkhorn_kernel on the side stream) — bit-identical to hc_pre + rmsnorm
template <int HC>
__device__ __forceinline__ void hc_pre_norm_body(const bf16* __restrict__ h, const float* __restrict__ pre_in, int dim, const bf16* __restrict__ norm_w,
                                                 float norm_eps, bf16* __restrict__ x, bf16* __restrict__ xn, int m, float* red, float* pin) {
  constexpr int hc = HC;
  if (threadIdx.x < hc) pin[threadIdx.x] = pre_in[m * hc + threadIdx.x];
  __syncthreads();
  const bf16* hm = h + (size_t)m * hc * dim;
  float ss = 0.f;
  for (int d = threadIdx.x; d < dim; d += 256) {
    float s = 0.f;
    for (int c = 0; c < hc; ++c) s += pin[c] * bf2f(hm[(size_t)c * dim + d]);
    const bf16 xv = f2bf(s);
    x[(size_t)m * dim + d] = xv;
    const float v = bf2f(xv);
    ss += v * v;
  }
  ss = hive::cu::warp_allreduce(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
    t = hive::cu::warp_allreduce(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  const float rs = rsqrtf(red[0] / (float)dim + norm_eps);
  for (int d = threadIdx.x; d < dim; d += 256) {
    const float v = bf2f(x[(size_t)m * dim + d]);
    xn[(size_t)m * dim + d] = f2bf(bf2f(norm_w[d]) * (v * rs));
  }
}
template <int HC>
__global__ void hc_pre_norm_kernel(const bf16* __restrict__ h, const float* __restrict__ pre_in, int dim, const bf16* __restrict__ norm_w,
                                   float norm_eps, bf16* __restrict__ x, bf16* __restrict__ xn) {
  __shared__ float red[8];
  __shared__ float pin[HC];
  hc_pre_norm_body<HC>(h, pre_in, dim, norm_w, norm_eps, x, xn, blockIdx.x, red, pin);
}
// The two decode hc kernels in one launch: block (m, j <= mix_hc) = hc_mix, (m, mix_hc+1) = hc_pre_norm. Both only read h and their outputs do not overlap.
template <int HC>
__global__ void hc_mix_pre_norm_kernel(const bf16* __restrict__ h, const float* __restrict__ W, int hcdim, int mix_hc, float eps, float* __restrict__ mixes,
                                       float* __restrict__ rsq, const float* __restrict__ pre_in, int dim, const bf16* __restrict__ norm_w, float norm_eps,
                                       bf16* __restrict__ x, bf16* __restrict__ xn) {
  __shared__ float red[8];
  __shared__ float pin[HC];
  if ((int)blockIdx.y <= mix_hc) hc_mix_body(h, W, hcdim, mix_hc, eps, mixes, rsq, blockIdx.x, blockIdx.y, red);
  else hc_pre_norm_body<HC>(h, pre_in, dim, norm_w, norm_eps, x, xn, blockIdx.x, red, pin);
}
// HIVE_HC_DECODE_FUSED: the kernel above + sinkhorn (the last mix block handles the M tokens, 16 lanes each — hc_decode_fused.cuh). The two bodies are the same device functions as above, so
//   mixes/rsq/x/xn are bit-identical to hc_mix_pre_norm and pre/post/comb to sinkhorn_kernel (side stream) (test_hc_decode). cnt = zero-initialized int (the kernel resets it to 0).
template <int HC>
__global__ void hc_mix_pre_norm_sk_kernel(const bf16* __restrict__ h, const float* __restrict__ W, int hcdim, int mix_hc, float eps, float* mixes, float* rsq,
                                          const float* __restrict__ pre_in, int dim, const bf16* __restrict__ norm_w, float norm_eps, bf16* __restrict__ x,
                                          bf16* __restrict__ xn, const float* __restrict__ scale, const float* __restrict__ base, int M, int iters, float hc_eps,
                                          float* __restrict__ pre_out, float* __restrict__ post, float* __restrict__ comb, int* cnt) {
  __shared__ float red[8];
  __shared__ float pin[HC];
  __shared__ int last;
  if ((int)blockIdx.y <= mix_hc) {
    hc_mix_body(h, W, hcdim, mix_hc, eps, mixes, rsq, blockIdx.x, blockIdx.y, red);
    hcdf::sk_tail<HC>(mixes, rsq, scale, base, M, iters, hc_eps, pre_out, post, comb, cnt, M * (mix_hc + 1), &last);
  } else {
    hc_pre_norm_body<HC>(h, pre_in, dim, norm_w, norm_eps, x, xn, blockIdx.x, red, pin);
  }
}

// (2) block m: sinkhorn (thread 0) → pre_out/post/comb, x = Σ_c pre_in[c]·h[c] → bf16 → rmsnorm → xn  (bit-identical to hc_pre + rmsnorm)
template <int HC>
__global__ void hc_pre_fused_kernel(const bf16* __restrict__ h, const float* __restrict__ mixes, const float* __restrict__ rsq,
                                    const float* __restrict__ scale, const float* __restrict__ base, const float* __restrict__ pre_in,
                                    int dim, int iters, float hc_eps, const bf16* __restrict__ norm_w, float norm_eps,
                                    float* __restrict__ pre_out, float* __restrict__ post, float* __restrict__ comb,
                                    bf16* __restrict__ x, bf16* __restrict__ xn) {
  constexpr int hc = HC, mix_hc = (2 + HC) * HC;
  __shared__ float red[8];
  __shared__ float pin[HC];
  const int m = blockIdx.x;
  if (threadIdx.x == 0)
    sinkhorn_one<HC>(mixes + (size_t)m * mix_hc, rsq[m], scale, base, iters, hc_eps, pre_out + m * hc, post + m * hc, comb + (size_t)m * hc * hc);
  if (threadIdx.x < hc) pin[threadIdx.x] = pre_in[m * hc + threadIdx.x];
  __syncthreads();
  const bf16* hm = h + (size_t)m * hc * dim;
  float ss = 0.f;
  for (int d = threadIdx.x; d < dim; d += 256) {
    float s = 0.f;
    for (int c = 0; c < hc; ++c) s += pin[c] * bf2f(hm[(size_t)c * dim + d]);
    const bf16 xv = f2bf(s);
    x[(size_t)m * dim + d] = xv;
    const float v = bf2f(xv);
    ss += v * v;
  }
  ss = hive::cu::warp_allreduce(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
    t = hive::cu::warp_allreduce(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  const float rs = rsqrtf(red[0] / (float)dim + norm_eps);
  for (int d = threadIdx.x; d < dim; d += 256) {
    const float v = bf2f(x[(size_t)m * dim + d]);
    xn[(size_t)m * dim + d] = f2bf(bf2f(norm_w[d]) * (v * rs));
  }
}

// Writes the routing result (+activations) directly into a mapped pinned host buffer — replaces 4 cudaMemcpyAsync D2H (each ~10 µs of serialized copy-engine latency) with one kernel
__global__ void route_to_host_kernel(const int32_t* __restrict__ ids, const float* __restrict__ rw, const uint8_t* __restrict__ xq,
                                     const uint8_t* __restrict__ xs, int n_route, int n_act, int32_t* __restrict__ ids_h,
                                     float* __restrict__ rw_h, uint8_t* __restrict__ xq_h, uint8_t* __restrict__ xs_h) {
  const size_t i = hive::cu::global_tid();
  if (i < (size_t)n_route) { ids_h[i] = ids[i]; rw_h[i] = rw[i]; }
  if (i < (size_t)n_act) { xq_h[i] = xq[i]; if (i % 32 == 0) xs_h[i / 32] = xs[i / 32]; }
}

__global__ void hc_pre_kernel(const bf16* __restrict__ h, const float* __restrict__ pre, int M, int hc, int dim,
                              bf16* __restrict__ x) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * dim) return;
  int m = i / dim, d = i % dim;
  float s = 0.f;
  for (int c = 0; c < hc; ++c) s += pre[m * hc + c] * bf2f(h[((size_t)m * hc + c) * dim + d]);
  x[i] = f2bf(s);
}

__global__ void hc_post_kernel(const bf16* __restrict__ x, const float* __restrict__ post, const float* __restrict__ comb, int M,
                               int hc, int dim, bf16* __restrict__ h) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * dim) return;
  int m = i / dim, d = i % dim;
  float res[4];
  for (int j = 0; j < hc; ++j) res[j] = bf2f(h[((size_t)m * hc + j) * dim + d]);
  float xv = bf2f(x[i]);
  for (int c = 0; c < hc; ++c) {
    float s = post[m * hc + c] * xv;
    for (int j = 0; j < hc; ++j) s += comb[((size_t)m * hc + j) * hc + c] * res[j];
    h[((size_t)m * hc + c) * dim + d] = f2bf(s);
  }
}

// x [M,H,D], rotate the last rd dimensions: pairs (2i, 2i+1) as complex numbers (the reference view_as_complex uses adjacent pairs)
__global__ void rope_last_kernel(bf16* __restrict__ x, int M, int H, int D, int rd, const float2* __restrict__ freqs,
                                 const int32_t* __restrict__ pos, bool inverse) {
  const size_t i = hive::cu::global_tid();
  const int half = rd / 2;
  if (i >= (size_t)M * H * half) return;
  int m = i / ((size_t)H * half);
  int rem = i % ((size_t)H * half);
  int hh = rem / half, p = rem % half;
  float2 f = freqs[(size_t)pos[m] * half + p];
  if (inverse) f.y = -f.y;
  bf16* base = x + ((size_t)m * H + hh) * D + (D - rd) + 2 * p;
  float a = bf2f(base[0]), b = bf2f(base[1]);
  base[0] = f2bf(a * f.x - b * f.y);
  base[1] = f2bf(a * f.y + b * f.x);
}

// ---- sparse attention: block = (1 query, head group ATT_HG=8), 256 threads → per thread (1 head, a 128-dim range… → 8 heads × 512 = 4096 outputs / 256 = 16)
//   Reference kernel semantics: score = bf16 q · bf16 kv (fp32 accumulation) × scale; P is cast to bf16 and multiplied with V (fp32 accumulation); denominator fp32 (+sink).
constexpr int ATT_D = 512;
constexpr int ATT_BLK = 32;   // key block (32 KB smem)
constexpr int ATT_HG = 8;     // heads per block

__device__ inline bf16 kv_elem(const KvSources& kv, int idx, int d) {
  if (idx < kv.win) return kv.ring[(size_t)idx * ATT_D + d];
  idx -= kv.win;
  if (idx < kv.chunk_len) return kv.chunk[(size_t)idx * ATT_D + d];
  idx -= kv.chunk_len;
  return f2bf(kvp::comp_val(kv.comp + (size_t)idx * kvp::COMP_ROW, d));
}

__global__ void __launch_bounds__(256) sparse_attn_kernel(const bf16* __restrict__ q, int H, KvSources kv,
                                                          const int32_t* __restrict__ idx, int idx_stride, int topk,
                                                          const float* __restrict__ sink, float scale, bf16* __restrict__ o) {
  extern __shared__ float smem[];
  bf16* qs = reinterpret_cast<bf16*>(smem);                            // HG*D
  bf16* kvs = qs + (size_t)ATT_HG * ATT_D;                             // BLK*D
  float* P = reinterpret_cast<float*>(kvs + (size_t)ATT_BLK * ATT_D);  // HG*BLK
  float* mrow = P + (size_t)ATT_HG * ATT_BLK;                          // HG
  float* lrow = mrow + ATT_HG;                                         // HG
  float* alpha = lrow + ATT_HG;                                        // HG
  const int m = blockIdx.x / (H / ATT_HG);
  const int hg = blockIdx.x % (H / ATT_HG);
  const int h0 = hg * ATT_HG;
  const int tid = threadIdx.x;
  for (int i = tid; i < ATT_HG * ATT_D; i += 256) qs[i] = q[((size_t)m * H + h0) * ATT_D + i];
  if (tid < ATT_HG) { mrow[tid] = -1e30f; lrow[tid] = 0.f; }
  // Output accumulation: thread = (head hh = tid/32, dimension range (tid%32)*16 .. +16)
  const int hh = tid / 32;
  const int dseg = (tid % 32) * 16;
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.f;
  __syncthreads();
  const int32_t* irow = idx + (size_t)m * idx_stride;
  for (int t0 = 0; t0 < topk; t0 += ATT_BLK) {
    for (int i = tid; i < ATT_BLK * ATT_D; i += 256) {
      int j = i / ATT_D, d = i % ATT_D;
      int ix = (t0 + j < topk) ? irow[t0 + j] : -1;
      kvs[i] = (ix >= 0) ? kv_elem(kv, ix, d) : f2bf(0.f);
    }
    __syncthreads();
    // Scores S[h][j]: thread = (h = tid/32, j = tid%32) — each thread computes a 512-long dot product
    {
      const int j = tid % 32;
      int ix = (t0 + j < topk) ? irow[t0 + j] : -1;
      float d = -INFINITY;
      if (ix >= 0) {
        const bf16* qh = qs + (size_t)hh * ATT_D;
        const bf16* kr = kvs + (size_t)j * ATT_D;
        d = 0.f;
        for (int dd = 0; dd < ATT_D; dd += 2) {
          __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(qh + dd);
          __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(kr + dd);
          d = fmaf(__bfloat162float(a.x), __bfloat162float(b.x), d);
          d = fmaf(__bfloat162float(a.y), __bfloat162float(b.y), d);
        }
        d *= scale;
      }
      P[hh * ATT_BLK + j] = d;
    }
    __syncthreads();
    if (tid < ATT_HG) {
      float mprev = mrow[tid];
      float mnew = mprev;
      for (int j = 0; j < ATT_BLK; ++j) mnew = fmaxf(mnew, P[tid * ATT_BLK + j]);
      float a = expf(mprev - mnew);
      float ssum = 0.f;
      for (int j = 0; j < ATT_BLK; ++j) {
        float p = expf(P[tid * ATT_BLK + j] - mnew);
        P[tid * ATT_BLK + j] = bf2f(f2bf(p));  // P is cast to bf16 before multiplying with V
        ssum += p;                              // the denominator uses the fp32 p before the cast
      }
      lrow[tid] = lrow[tid] * a + ssum;
      mrow[tid] = mnew;
      alpha[tid] = a;
    }
    __syncthreads();
    {
      float a = alpha[hh];
#pragma unroll
      for (int i = 0; i < 16; ++i) acc[i] *= a;
      for (int j = 0; j < ATT_BLK; ++j) {
        float p = P[hh * ATT_BLK + j];
        if (p == 0.f) continue;
        const bf16* kr = kvs + (size_t)j * ATT_D + dseg;
#pragma unroll
        for (int i = 0; i < 16; i += 2) {
          __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(kr + i);
          acc[i] = fmaf(p, __bfloat162float(b.x), acc[i]);
          acc[i + 1] = fmaf(p, __bfloat162float(b.y), acc[i + 1]);
        }
      }
    }
    __syncthreads();
  }
  float denom = lrow[hh] + expf(sink[h0 + hh] - mrow[hh]);
  bf16* orow_out = o + ((size_t)m * H + h0 + hh) * ATT_D + dseg;
#pragma unroll
  for (int i = 0; i < 16; ++i) orow_out[i] = f2bf(acc[i] / denom);
}

__global__ void window_idxs_kernel(int M, int win, int64_t start_pos, int64_t min_src, int32_t* __restrict__ out, int out_stride) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * win) return;
  int m = i / win, j = i % win;
  int64_t p = start_pos + m;                // global position of the query
  int64_t src = p - (win - 1) + j;          // attended position (oldest first)
  int32_t v;
  if (src < min_src) v = -1;                // P6 bounded replay: keys before the replay range are not attended (min_src = replay start, normally 0)
  else if (src < start_pos) v = (int32_t)(src % win);          // ring slot (earlier chunk / tokens before decode)
  else v = (int32_t)(win + (src - start_pos));                 // chunk row
  out[(size_t)m * out_stride + j] = v;
}

__global__ void window_idxs_rows_kernel(int M, int win, const int32_t* __restrict__ pos, int32_t* __restrict__ out, int out_stride) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * win) return;
  int m = i / win, j = i % win;
  int64_t p = pos[m];
  int64_t src = p - (win - 1) + j;
  int32_t v;
  if (src < 0) v = -1;
  else if (src < p) v = (int32_t)(src % win);
  else v = (int32_t)(win + m);
  out[(size_t)m * out_stride + j] = v;
}
__global__ void ring_write_rows_kernel(const bf16* __restrict__ kv, int M, int D, int win, const int32_t* __restrict__ pos,
                                       bf16* const* __restrict__ ring_ptrs) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * D) return;
  int m = i / D, d = i % D;
  ring_ptrs[m][(size_t)(pos[m] % win) * D + d] = kv[i];
}
__global__ void compressor_step_rows_kernel(const float* __restrict__ kv, const float* __restrict__ score, int M, int D, int ratio,
                                            const int32_t* __restrict__ pos, float* const* __restrict__ state_kv,
                                            float* const* __restrict__ state_score, float* __restrict__ out, uint8_t* __restrict__ valid) {
  const int m = blockIdx.x;
  const int p = pos[m];
  const int slot = p % ratio;
  float* skv = state_kv[m];
  float* ssc = state_score[m];
  const bool should = ((p + 1) % ratio) == 0;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    skv[slot * D + d] = kv[(size_t)m * D + d];
    ssc[slot * D + d] = score[(size_t)m * D + d];
  }
  __syncthreads();
  if (should) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      float mx = -FLT_MAX;
      for (int r = 0; r < ratio; ++r) mx = fmaxf(mx, ssc[r * D + d]);
      float sum = 0.f, acc = 0.f, e[8];
      for (int r = 0; r < ratio; ++r) { e[r] = expf(ssc[r * D + d] - mx); sum += e[r]; }
      for (int r = 0; r < ratio; ++r) acc += skv[r * D + d] * (e[r] / sum);
      out[(size_t)m * D + d] = acc;
    }
  }
  if (threadIdx.x == 0) valid[m] = should ? 1 : 0;
}
__global__ void write_cache_rows_kernel(const bf16* __restrict__ src, int M, int D, bf16* const* __restrict__ dst_ptrs,
                                        const int32_t* __restrict__ dst_index, const uint8_t* __restrict__ valid) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * D) return;
  int m = i / D, d = i % D;
  if (!valid[m]) return;
  dst_ptrs[m][(size_t)dst_index[m] * D + d] = src[i];
}

__global__ void ring_write_kernel(const bf16* __restrict__ kv, int M, int D, int win, int64_t start_pos, bf16* __restrict__ ring) {
  // only the last min(M, win) tokens are written (earlier ones would be overwritten anyway)
  int first = M > win ? M - win : 0;
  size_t n = (size_t)(M - first) * D;
  const size_t i = hive::cu::global_tid();
  if (i >= n) return;
  int s = first + (int)(i / D), d = (int)(i % D);
  ring[(size_t)((start_pos + s) % win) * D + d] = kv[(size_t)s * D + d];
}
__global__ void scale_bf16_kernel(bf16* __restrict__ x, size_t n, float c) {
  const size_t i = hive::cu::global_tid();
  if (i < n) x[i] = f2bf(bf2f(x[i]) * c);
}
__global__ void iota_rows_kernel(int32_t* __restrict__ out, size_t n, int T) {
  const size_t i = hive::cu::global_tid();
  if (i < n) out[i] = (int32_t)(i % T);
}

// Compressor pooling: group = global position // ratio. The partial group in the state (pending, positions start_pos - (start_pos % ratio) .. start_pos-1) is
// combined with this chunk's tokens to form complete groups. Block = one group, thread = dimension.
__global__ void compressor_pool_kernel(const float* __restrict__ kv, const float* __restrict__ score, int M, int D, int ratio,
                                       int64_t start_pos, float* __restrict__ state_kv, float* __restrict__ state_score,
                                       float* __restrict__ out) {
  const int g = blockIdx.x;  // complete group index (0.. relative to this chunk)
  const int64_t g0 = start_pos / ratio;  // global index of the first (possibly partial) group
  const int64_t gpos = (g0 + g) * ratio;  // first position of the group
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float v[8], s[8];  // ratio ≤ 8
    for (int r = 0; r < ratio; ++r) {
      int64_t p = gpos + r;
      if (p < start_pos) { v[r] = state_kv[r * D + d]; s[r] = state_score[r * D + d]; }
      else { int64_t mrow = p - start_pos; v[r] = kv[mrow * D + d]; s[r] = score[mrow * D + d]; }
    }
    float mx = -FLT_MAX;
    for (int r = 0; r < ratio; ++r) mx = fmaxf(mx, s[r]);
    float sum = 0.f, acc = 0.f;
    float e[8];
    for (int r = 0; r < ratio; ++r) { e[r] = expf(s[r] - mx); sum += e[r]; }
    for (int r = 0; r < ratio; ++r) acc += v[r] * (e[r] / sum);
    out[(size_t)g * D + d] = acc;
  }
}
// Store the tail (incomplete group) in the state: positions p (after this chunk's last complete group) up to end
__global__ void compressor_tail_kernel(const float* __restrict__ kv, const float* __restrict__ score, int M, int D, int ratio,
                                       int64_t start_pos, float* __restrict__ state_kv, float* __restrict__ state_score) {
  const int64_t end = start_pos + M;
  const int64_t tail0 = end - (end % ratio);  // first position of the last incomplete group
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    for (int r = 0; r < ratio; ++r) {
      int64_t p = tail0 + r;
      if (p >= end) { state_score[r * D + d] = -INFINITY; state_kv[r * D + d] = 0.f; continue; }
      if (p < start_pos) continue;  // already in the state (the chunk did not complete the group)
      int64_t mrow = p - start_pos;
      state_kv[r * D + d] = kv[mrow * D + d];
      state_score[r * D + d] = score[mrow * D + d];
    }
  }
}

// Indexer scores: block = (query m, key tile of 128), thread = one key. q [M,Hi,Di] bf16 in smem, k tile in smem.
constexpr int IDX_T = 128;
// Score of one key — shared so the three kernels (full T, row table, candidate pool) use the same arithmetic (P1: candidate-pool results must be bit-identical to full-T results)
__device__ __forceinline__ float idx_score_one(const float* __restrict__ qsm, const float* __restrict__ wsm, const uint8_t* __restrict__ kr, int Hi, int Di) {
  // A per-key kf[128] array would spill to local memory (512 B of stack) and be re-read 128 times per head. Instead heads are grouped by 32 with the dimension loop outside and the head loop inside —
  //   the 32 accumulators stay in registers. Per head fmaf over ascending d and the head sum over ascending h, so the result is bit-identical.
  float total = 0.f;
  for (int h0 = 0; h0 < Hi; h0 += 32) {
    float acc[32];
#pragma unroll
    for (int h = 0; h < 32; ++h) acc[h] = 0.f;
    for (int d = 0; d < Di; ++d) {
      const float kv = kvp::idx_val(kr, d);
#pragma unroll
      for (int h = 0; h < 32; ++h) if (h0 + h < Hi) acc[h] = fmaf(qsm[(h0 + h) * Di + d], kv, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < 32; ++h) {
      if (h0 + h >= Hi) break;
      float sb = bf2f(f2bf(acc[h]));      // einsum output is bf16
      sb = fmaxf(sb, 0.f);                // relu
      sb = bf2f(f2bf(sb * wsm[h0 + h]));  // bf16 product
      total += sb;                        // head sum (fp32 accumulation → bf16 at the end)
    }
  }
  return total;
}
__global__ void indexer_scores_kernel(const bf16* __restrict__ q, const uint8_t* __restrict__ k, const bf16* __restrict__ w, int Hi,
                                      int Di, int T, const int32_t* __restrict__ visible, bf16* __restrict__ score) {
  extern __shared__ float qsm[];  // Hi*Di fp32 + Hi (weights)
  const int m = blockIdx.y;
  const int t = blockIdx.x * IDX_T + threadIdx.x;
  float* wsm = qsm + Hi * Di;
  for (int i = threadIdx.x; i < Hi * Di; i += IDX_T) qsm[i] = bf2f(q[(size_t)m * Hi * Di + i]);
  for (int i = threadIdx.x; i < Hi; i += IDX_T) wsm[i] = bf2f(w[(size_t)m * Hi + i]);
  __syncthreads();
  if (t >= T) return;
  const int vis = visible[m];
  if (t < vis) score[(size_t)m * T + t] = f2bf(idx_score_one(qsm, wsm, k + (size_t)t * kvp::IDX_ROW, Hi, Di));
  else score[(size_t)m * T + t] = f2bf(-INFINITY);
}

__global__ void indexer_scores_rows_kernel(const bf16* __restrict__ q, const uint8_t* const* __restrict__ k_ptrs,
                                           const int32_t* __restrict__ T_rows, int Tmax, const bf16* __restrict__ w, int Hi, int Di,
                                           const int32_t* __restrict__ visible, bf16* __restrict__ score) {
  extern __shared__ float qsm[];
  const int m = blockIdx.y;
  const int t = blockIdx.x * IDX_T + threadIdx.x;
  float* wsm = qsm + Hi * Di;
  for (int i = threadIdx.x; i < Hi * Di; i += IDX_T) qsm[i] = bf2f(q[(size_t)m * Hi * Di + i]);
  for (int i = threadIdx.x; i < Hi; i += IDX_T) wsm[i] = bf2f(w[(size_t)m * Hi + i]);
  __syncthreads();
  if (t >= Tmax) return;
  const int vis = min(visible[m], T_rows[m]);
  if (t < vis) score[(size_t)m * Tmax + t] = f2bf(idx_score_one(qsm, wsm, k_ptrs[m] + (size_t)t * kvp::IDX_ROW, Hi, Di));
  else score[(size_t)m * Tmax + t] = f2bf(-INFINITY);
}

// P1 candidate-pool scores (hierarchical indexer): the Reindex layers (24, 28, 32, 36) score only positions inside the block list cand[m][0..kb) chosen by layer 20
//   (block start positions of size bs, ascending, -1 when short) — cost is bounded by kb·bs (16,384) instead of the context length. Selection equals the full-T path
//   (full-T scores, then -inf via the keep mask): positions outside the candidates were -inf there too, and candidate index ci follows position order, so top-k tie
//   handling (smaller index first) is the same. Output score[m, ci], ci = bi·bs + j.
//   Either k_ptrs (row table, batched decode) or k (single cache). With T_rows, positions are also clipped to the row's key count.
__global__ void indexer_scores_cand_kernel(const bf16* __restrict__ q, const uint8_t* __restrict__ k, const uint8_t* const* __restrict__ k_ptrs,
                                           const int32_t* __restrict__ T_rows, const bf16* __restrict__ w, int Hi, int Di,
                                           const int32_t* __restrict__ cand, int cand_stride, int bs, int ncand,
                                           const int32_t* __restrict__ visible, bf16* __restrict__ score) {
  extern __shared__ float qsm[];
  const int m = blockIdx.y;
  const int ci = blockIdx.x * IDX_T + threadIdx.x;
  float* wsm = qsm + Hi * Di;
  for (int i = threadIdx.x; i < Hi * Di; i += IDX_T) qsm[i] = bf2f(q[(size_t)m * Hi * Di + i]);
  for (int i = threadIdx.x; i < Hi; i += IDX_T) wsm[i] = bf2f(w[(size_t)m * Hi + i]);
  __syncthreads();
  if (ci >= ncand) return;
  int vis = visible[m];
  if (T_rows) vis = min(vis, T_rows[m]);
  const int32_t b = cand[(size_t)m * cand_stride + ci / bs];
  const int t = b >= 0 ? b * bs + ci % bs : -1;
  if (t >= 0 && t < vis) {
    const uint8_t* kb = k_ptrs ? k_ptrs[m] : k;
    score[(size_t)m * ncand + ci] = f2bf(idx_score_one(qsm, wsm, kb + (size_t)t * kvp::IDX_ROW, Hi, Di));
  } else {
    score[(size_t)m * ncand + ci] = f2bf(-INFINITY);
  }
}
// fp32 score version (HIVE_IDX_F32) — the default kernel above is left untouched (machine code verified unchanged); this is a copy with only the per-head bf16 rounding removed
// Score of one key — shared so the three kernels (full T, row table, candidate pool) use the same arithmetic (P1: candidate-pool results must be bit-identical to full-T results)
__device__ __forceinline__ float idx_score_one_f32(const float* __restrict__ qsm, const float* __restrict__ wsm, const uint8_t* __restrict__ kr, int Hi, int Di) {
  // A per-key kf[128] array would spill to local memory (512 B of stack) and be re-read 128 times per head. Instead heads are grouped by 32 with the dimension loop outside and the head loop inside —
  //   the 32 accumulators stay in registers. Per head fmaf over ascending d and the head sum over ascending h, so the result is bit-identical.
  float total = 0.f;
  for (int h0 = 0; h0 < Hi; h0 += 32) {
    float acc[32];
#pragma unroll
    for (int h = 0; h < 32; ++h) acc[h] = 0.f;
    for (int d = 0; d < Di; ++d) {
      const float kv = kvp::idx_val(kr, d);
#pragma unroll
      for (int h = 0; h < 32; ++h) if (h0 + h < Hi) acc[h] = fmaf(qsm[(h0 + h) * Di + d], kv, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < 32; ++h) {
      if (h0 + h >= Hi) break;
      total += fmaxf(acc[h], 0.f) * wsm[h0 + h];  // fp32 as is (no rounding)
    }
  }
  return total;
}
__global__ void indexer_scores_f32_kernel(const bf16* __restrict__ q, const uint8_t* __restrict__ k, const bf16* __restrict__ w, int Hi,
                                      int Di, int T, const int32_t* __restrict__ visible, float* __restrict__ score) {
  extern __shared__ float qsm[];  // Hi*Di fp32 + Hi (weights)
  const int m = blockIdx.y;
  const int t = blockIdx.x * IDX_T + threadIdx.x;
  float* wsm = qsm + Hi * Di;
  for (int i = threadIdx.x; i < Hi * Di; i += IDX_T) qsm[i] = bf2f(q[(size_t)m * Hi * Di + i]);
  for (int i = threadIdx.x; i < Hi; i += IDX_T) wsm[i] = bf2f(w[(size_t)m * Hi + i]);
  __syncthreads();
  if (t >= T) return;
  const int vis = visible[m];
  if (t < vis) score[(size_t)m * T + t] = float(idx_score_one_f32(qsm, wsm, k + (size_t)t * kvp::IDX_ROW, Hi, Di));
  else score[(size_t)m * T + t] = -INFINITY;
}

__global__ void indexer_scores_rows_f32_kernel(const bf16* __restrict__ q, const uint8_t* const* __restrict__ k_ptrs,
                                           const int32_t* __restrict__ T_rows, int Tmax, const bf16* __restrict__ w, int Hi, int Di,
                                           const int32_t* __restrict__ visible, float* __restrict__ score) {
  extern __shared__ float qsm[];
  const int m = blockIdx.y;
  const int t = blockIdx.x * IDX_T + threadIdx.x;
  float* wsm = qsm + Hi * Di;
  for (int i = threadIdx.x; i < Hi * Di; i += IDX_T) qsm[i] = bf2f(q[(size_t)m * Hi * Di + i]);
  for (int i = threadIdx.x; i < Hi; i += IDX_T) wsm[i] = bf2f(w[(size_t)m * Hi + i]);
  __syncthreads();
  if (t >= Tmax) return;
  const int vis = min(visible[m], T_rows[m]);
  if (t < vis) score[(size_t)m * Tmax + t] = float(idx_score_one_f32(qsm, wsm, k_ptrs[m] + (size_t)t * kvp::IDX_ROW, Hi, Di));
  else score[(size_t)m * Tmax + t] = -INFINITY;
}

// P1 candidate-pool scores (hierarchical indexer): the Reindex layers (24, 28, 32, 36) score only positions inside the block list cand[m][0..kb) chosen by layer 20
//   (block start positions of size bs, ascending, -1 when short) — cost is bounded by kb·bs (16,384) instead of the context length. Selection equals the full-T path
//   (full-T scores, then -inf via the keep mask): positions outside the candidates were -inf there too, and candidate index ci follows position order, so top-k tie
//   handling (smaller index first) is the same. Output score[m, ci], ci = bi·bs + j.
//   Either k_ptrs (row table, batched decode) or k (single cache). With T_rows, positions are also clipped to the row's key count.
__global__ void indexer_scores_cand_f32_kernel(const bf16* __restrict__ q, const uint8_t* __restrict__ k, const uint8_t* const* __restrict__ k_ptrs,
                                           const int32_t* __restrict__ T_rows, const bf16* __restrict__ w, int Hi, int Di,
                                           const int32_t* __restrict__ cand, int cand_stride, int bs, int ncand,
                                           const int32_t* __restrict__ visible, float* __restrict__ score) {
  extern __shared__ float qsm[];
  const int m = blockIdx.y;
  const int ci = blockIdx.x * IDX_T + threadIdx.x;
  float* wsm = qsm + Hi * Di;
  for (int i = threadIdx.x; i < Hi * Di; i += IDX_T) qsm[i] = bf2f(q[(size_t)m * Hi * Di + i]);
  for (int i = threadIdx.x; i < Hi; i += IDX_T) wsm[i] = bf2f(w[(size_t)m * Hi + i]);
  __syncthreads();
  if (ci >= ncand) return;
  int vis = visible[m];
  if (T_rows) vis = min(vis, T_rows[m]);
  const int32_t b = cand[(size_t)m * cand_stride + ci / bs];
  const int t = b >= 0 ? b * bs + ci % bs : -1;
  if (t >= 0 && t < vis) {
    const uint8_t* kb = k_ptrs ? k_ptrs[m] : k;
    score[(size_t)m * ncand + ci] = float(idx_score_one_f32(qsm, wsm, kb + (size_t)t * kvp::IDX_ROW, Hi, Di));
  } else {
    score[(size_t)m * ncand + ci] = -INFINITY;
  }
}
// Candidate index → position (in place). -1 stays -1.
__global__ void cand_to_pos_kernel(int32_t* __restrict__ sel, int M, int k, const int32_t* __restrict__ cand, int cand_stride, int bs) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * k) return;
  const int m = i / k;
  const int32_t ci = sel[i];
  if (ci < 0) return;
  const int32_t b = cand[(size_t)m * cand_stride + ci / bs];
  sel[i] = b >= 0 ? b * bs + ci % bs : -1;
}

__global__ void block_max_kernel(const bf16* __restrict__ score, int M, int T, int bs, const int32_t* __restrict__ visible,
                                 float* __restrict__ bmax) {
  const int nblocks = (T + bs - 1) / bs;
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * nblocks) return;
  int m = i / nblocks, b = i % nblocks;
  float mx = -INFINITY;
  for (int j = b * bs; j < min(T, (b + 1) * bs); ++j) mx = fmaxf(mx, bf2f(score[(size_t)m * T + j]));
  // the last block (the one the query reaches) is pinned to +inf
  int last = (visible[m] - 1) / bs;
  if (b == last) mx = INFINITY;
  bmax[i] = mx;
}

__global__ void block_max_f32_kernel(const float* __restrict__ score, int M, int T, int bs, const int32_t* __restrict__ visible,
                                 float* __restrict__ bmax) {
  const int nblocks = (T + bs - 1) / bs;
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * nblocks) return;
  int m = i / nblocks, b = i % nblocks;
  float mx = -INFINITY;
  for (int j = b * bs; j < min(T, (b + 1) * bs); ++j) mx = fmaxf(mx, score[(size_t)m * T + j]);
  // the last block (the one the query reaches) is pinned to +inf
  int last = (visible[m] - 1) / bs;
  if (b == last) mx = INFINITY;
  bmax[i] = mx;
}

__global__ void scatter_keep_kernel(const int32_t* __restrict__ pos, const float* __restrict__ bmax, int M, int k, int nblocks,
                                    uint8_t* __restrict__ keep, int keep_stride) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * k) return;
  int m = i / k;
  int32_t p = pos[i];
  if (p < 0 || p >= nblocks) return;
  if (bmax[(size_t)m * nblocks + p] > -INFINITY) keep[(size_t)m * keep_stride + p] = 1;
}

__global__ void apply_block_mask_kernel(bf16* __restrict__ score, int M, int T, int bs, const uint8_t* __restrict__ keep,
                                        int nblocks) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * T) return;
  int m = i / T, t = i % T;
  if (!keep[(size_t)m * nblocks + t / bs]) score[i] = f2bf(-INFINITY);
}

__global__ void offset_idxs_kernel(const int32_t* __restrict__ pos, int M, int topk, const int32_t* __restrict__ visible,
                                   int32_t offset, int32_t* __restrict__ out, int out_stride) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * topk) return;
  int m = i / topk, j = i % topk;
  int32_t p = pos[i];
  out[(size_t)m * out_stride + j] = (p >= 0 && p < visible[m]) ? p + offset : -1;
}

// Router: block = token, 256 threads, E <= 512. top-k on sqrt(softplus(score)) + bias (ties → lower id — not guaranteed to match torch topk's
// tie handling); weights are the scores without bias, normalized by their sum × scale.
__global__ void router_topk_kernel(const float* __restrict__ scores, const float* __restrict__ bias, const float* __restrict__ bias_vl,
                                   const int8_t* __restrict__ is_image, int E, int k, float route_scale, int32_t* __restrict__ ids,
                                   float* __restrict__ w) {
  __shared__ float s[512];
  __shared__ float sel[512];
  __shared__ int best_i;
  __shared__ float best_v;
  const int m = blockIdx.x;
  const float* b = (bias_vl && is_image && is_image[m]) ? bias_vl : bias;
  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    float x = scores[(size_t)m * E + e];
    float sp = x > 20.f ? x : log1pf(expf(x));  // softplus (torch: threshold 20)
    s[e] = sqrtf(sp);
    sel[e] = s[e] + b[e];
  }
  __syncthreads();
  float wsum = 0.f;
  for (int j = 0; j < k; ++j) {
    // argmax over sel (thread 0, sequentially — cheap enough with E=384)
    if (threadIdx.x == 0) {
      int bi = -1; float bv = -FLT_MAX;
      for (int e = 0; e < E; ++e) if (sel[e] > bv) { bv = sel[e]; bi = e; }
      best_i = bi; best_v = bv;
      sel[bi] = -FLT_MAX;
      ids[m * k + j] = bi;
      w[m * k + j] = s[bi];
      wsum += s[bi];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float inv = route_scale / (wsum + 1e-20f);
    for (int j = 0; j < k; ++j) w[m * k + j] = w[m * k + j] / (wsum + 1e-20f) * route_scale;
    (void)inv;
  }
}

__global__ void swiglu_route_kernel(const bf16* __restrict__ gate, const bf16* __restrict__ up, const float* __restrict__ rw, int M,
                                    int I, float limit, bf16* __restrict__ y) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * I) return;
  int m = i / I;
  float g = bf2f(gate[i]), u = bf2f(up[i]);
  if (limit > 0.f) { u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); }
  float v = g / (1.f + expf(-g)) * u;
  if (rw) v *= rw[m];
  y[i] = f2bf(v);
}

__global__ void accum_bf16_rows_kernel(const bf16* __restrict__ src, const int32_t* __restrict__ row_map, int M, int dim,
                                       float* __restrict__ acc) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * dim) return;
  int m = i / dim, d = i % dim;
  int r = row_map ? row_map[m] : m;
  atomicAdd(acc + (size_t)r * dim + d, bf2f(src[i]));
}
__global__ void accum_f32_rows_kernel(const float* __restrict__ src, const int32_t* __restrict__ row_map, int M, int dim,
                                      float* __restrict__ acc) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * dim) return;
  int m = i / dim, d = i % dim;
  int r = row_map ? row_map[m] : m;
  atomicAdd(acc + (size_t)r * dim + d, src[i]);
}

// Deterministic accumulation: acc[m,d] += Σ_{r: row_map[r]==m} src[r,d] (fixed r order — independent of atomic-add ordering)
__global__ void accum_bf16_rows_seq_kernel(const bf16* __restrict__ src, const int32_t* __restrict__ row_map, int R, int M, int dim,
                                           float* __restrict__ acc) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * dim) return;
  int m = i / dim, d = i % dim;
  float s = acc[i];
  for (int r = 0; r < R; ++r)
    if (row_map[r] == m) s += bf2f(src[(size_t)r * dim + d]);
  acc[i] = s;
}

__global__ void gather_rows_bf16_kernel(const bf16* __restrict__ src, const int32_t* __restrict__ rows, int n, int dim,
                                        bf16* __restrict__ dst) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)n * dim) return;
  int r = i / dim, d = i % dim;
  dst[i] = src[(size_t)rows[r] * dim + d];
}
__global__ void gather_rows_u8_kernel(const uint8_t* __restrict__ src, const int32_t* __restrict__ rows, int n, int rb,
                                      uint8_t* __restrict__ dst) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)n * rb) return;
  int r = i / rb, d = i % rb;
  dst[i] = src[(size_t)rows[r] * rb + d];
}

__global__ void engram_dequant_kernel(const uint8_t* __restrict__ vals, const uint8_t* __restrict__ scales, int M, int cols, int hd,
                                      bf16* __restrict__ out) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * cols * hd) return;
  size_t row = i / hd;
  int d = i % hd;
  out[i] = f2bf(e4m3_to_f32(vals[i]) * e8m0_to_f32(scales[row * (hd / 32) + d / 32]));
}

// engram gate: block = (token m, copy c), 256 threads. dot = Σ_d h·w·key · rstd(h)·rstd(key)·dim^-0.5
__global__ void engram_gate_kernel(const bf16* __restrict__ kv, const float* __restrict__ qk, const int8_t* __restrict__ is_image,
                                   int hc, int dim, float eps, bf16* __restrict__ h) {
  __shared__ float red[32];
  const int m = blockIdx.x / hc, c = blockIdx.x % hc;
  if (is_image && is_image[m]) return;
  const bf16* key = kv + (size_t)m * dim * (hc + 1) + (size_t)c * dim;
  const bf16* value = kv + (size_t)m * dim * (hc + 1) + (size_t)hc * dim;
  bf16* hrow = h + ((size_t)m * hc + c) * dim;
  float hh = 0.f, kk = 0.f, dot = 0.f;
  for (int d = threadIdx.x; d < dim; d += 256) {
    float hv = bf2f(hrow[d]), kv_ = bf2f(key[d]);
    hh += hv * hv;
    kk += kv_ * kv_;
    dot += hv * qk[c * dim + d] * kv_;
  }
  hh = block_sum<256>(hh, red);
  kk = block_sum<256>(kk, red);
  dot = block_sum<256>(dot, red);
  float rstd = rsqrtf(hh / dim + eps) * rsqrtf(kk / dim + eps);
  float dd = dot * rstd * rsqrtf((float)dim);
  float g = sqrtf(fmaxf(fabsf(dd), 1e-6f));
  g = copysignf(g, dd);
  g = 1.f / (1.f + expf(-g));
  for (int d = threadIdx.x; d < dim; d += 256) hrow[d] = f2bf(bf2f(hrow[d]) + g * bf2f(value[d]));
}

// Head: one warp = one vocabulary entry
__global__ void head_logits_kernel(const float* __restrict__ x, const bf16* __restrict__ head, int M, int V, int dim,
                                   float* __restrict__ logits) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= V) return;
  const bf16* hr = head + (size_t)warp * dim;
  for (int m = 0; m < M; ++m) {
    const float* xr = x + (size_t)m * dim;
    float acc = 0.f;
    for (int d = lane * 2; d < dim; d += 64) {
      __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(hr + d);
      acc = fmaf(xr[d], __bfloat162float(b.x), acc);
      acc = fmaf(xr[d + 1], __bfloat162float(b.y), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) logits[(size_t)m * V + warp] = acc;
  }
}

// Head for M ≥ 2 rows (HIVE_HEAD_ROWS, default on): the kernel above reads a vocabulary row once per input row and every warp reads the
//   whole input again for each row — measured (CUPTI, real chat, c8): 1.68 ms per call at M = 8 against 1.03 ms at the mix of M of a single
//   stream and ~0.74 ms for one pass over the 1.32 GB head. Here a warp takes HR vocabulary rows and keeps HR × MR accumulators, so a vocabulary
//   row is read once for all input rows, and the block stages the input rows in shared memory one 1,024-column slice at a time (the 8 warps of
//   a block share it). Per (vocabulary row, input row) every lane accumulates the same products in the same order as head_logits_kernel —
//   d = 2·lane + 64·k ascending, x[d]·h[d] then x[d+1]·h[d+1], one fmaf each — and the same warp_sum follows, so the logits are bit-identical.
constexpr int kHeadHR = 4, kHeadChunk = 1024;
template <int MR>
__global__ void __launch_bounds__(256) head_logits_rows_kernel(const float* __restrict__ x, const bf16* __restrict__ head, int M, int V, int dim,
                                                               float* __restrict__ logits) {
  __shared__ float xs[MR * kHeadChunk];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int v0 = (blockIdx.x * 8 + warp) * kHeadHR;
  float acc[kHeadHR][MR];
#pragma unroll
  for (int r = 0; r < kHeadHR; ++r)
#pragma unroll
    for (int m = 0; m < MR; ++m) acc[r][m] = 0.f;
  const bf16* hr[kHeadHR];
#pragma unroll
  for (int r = 0; r < kHeadHR; ++r) hr[r] = head + (size_t)min(v0 + r, V - 1) * dim;
  for (int c0 = 0; c0 < dim; c0 += kHeadChunk) {
    const int cn = min(kHeadChunk, dim - c0);
    __syncthreads();
    for (int i = threadIdx.x; i < MR * kHeadChunk; i += 256) {
      const int m = i / kHeadChunk, d = i % kHeadChunk;
      xs[i] = m < M && d < cn ? x[(size_t)m * dim + c0 + d] : 0.f;
    }
    __syncthreads();
    if (v0 >= V) continue;
    for (int d = lane * 2; d < cn; d += 64) {
      float h0[kHeadHR], h1[kHeadHR];
#pragma unroll
      for (int r = 0; r < kHeadHR; ++r) {
        const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(hr[r] + c0 + d);
        h0[r] = __bfloat162float(b.x); h1[r] = __bfloat162float(b.y);
      }
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        const float xa = xs[m * kHeadChunk + d], xb = xs[m * kHeadChunk + d + 1];
#pragma unroll
        for (int r = 0; r < kHeadHR; ++r) { acc[r][m] = fmaf(xa, h0[r], acc[r][m]); acc[r][m] = fmaf(xb, h1[r], acc[r][m]); }
      }
    }
  }
  if (v0 >= V) return;
#pragma unroll
  for (int r = 0; r < kHeadHR; ++r)
#pragma unroll
    for (int m = 0; m < MR; ++m) {
      const float a = warp_sum(acc[r][m]);
      if (lane == 0 && m < M && v0 + r < V) logits[(size_t)m * V + v0 + r] = a;
    }
}

__global__ void argmax_rows_kernel(const float* __restrict__ logits, int V, int32_t* __restrict__ out) {
  __shared__ float bv[1024];
  __shared__ int bi[1024];
  const int m = blockIdx.x;
  float best = -FLT_MAX; int besti = 0;
  for (int v = threadIdx.x; v < V; v += blockDim.x) {
    float x = logits[(size_t)m * V + v];
    if (x > best) { best = x; besti = v; }
  }
  bv[threadIdx.x] = best; bi[threadIdx.x] = besti;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      float ov = bv[threadIdx.x + s]; int oi = bi[threadIdx.x + s];
      if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) { bv[threadIdx.x] = ov; bi[threadIdx.x] = oi; }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[m] = bi[0];
}

inline int grid1d(size_t n, int block = 256) { return (int)((n + block - 1) / block); }

}  // namespace

void expand_block_scale(const uint8_t* src, int N, int K, uint8_t* dst, cudaStream_t st) {
  int nbk = K / 32;
  expand_block_scale_kernel<<<grid1d((size_t)N * nbk), 256, 0, st>>>(src, N, nbk, dst);
}
void dequant_fp8_block_to_bf16(const uint8_t* w, const uint8_t* s, int N, int K, bf16* out, cudaStream_t st) {
  dequant_fp8_block_kernel<<<grid1d((size_t)N * K), 256, 0, st>>>(w, s, N, K, out);
}
void dequant_fp8_rows_to_bf16(const uint8_t* w, const uint8_t* s, int N, int K, bf16* out, cudaStream_t st) {
  dequant_fp8_rows_kernel<<<grid1d((size_t)N * K / 8), 256, 0, st>>>(w, s, N, K, out);
}
// 4-row-block version — for HIVE_HC_MIX_ROWS4=1 A/B
void hc_mix_rows4(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, cudaStream_t st) {
  HIVE_CHECK(mix_hc == HCMIX && hcdim % 2048 == 0, "hc_mix_rows shape");
  if (M <= 0) return;
  hc_mix_rows_kernel<<<(M + HCRT - 1) / HCRT, 256, 0, st>>>(h, W, M, hcdim, eps, mixes, rsq);
}
void hc_mix_rows16(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, cudaStream_t st) {
  HIVE_CHECK(mix_hc == HCMIX && hcdim % HCKT == 0, "hc_mix_rows16 shape");
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(hc_mix_rows16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)HCMIX16_SMEM));
    configured = true;
  }
  if (M <= 0) return;
  hc_mix_rows16_kernel<<<(M + HCR16 - 1) / HCR16, 256, HCMIX16_SMEM, st>>>(h, W, M, hcdim, eps, mixes, rsq);
}
// Default = 16-row blocks (1/4 of the W traffic); HIVE_HC_MIX_ROWS4=1 selects the 4-row version
void hc_mix_rows(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, cudaStream_t st) {
  static const bool rows4 = getenv("HIVE_HC_MIX_ROWS4") && atoi(getenv("HIVE_HC_MIX_ROWS4")) != 0;
  // rows16 has block = 16 rows and 1 block per SM, so with ⌈M/16⌉ < SM count it cannot fill the GPU (M=2048 → 128 blocks / 188 SMs). Then use rows4 (4-row blocks, reads W more but fills the SMs).
  static int n_sm = 0;
  if (!n_sm) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev); if (n_sm <= 0) n_sm = 1; }
  if (rows4 || (M + 15) / 16 < n_sm) hc_mix_rows4(h, W, M, hcdim, mix_hc, eps, mixes, rsq, st);
  else hc_mix_rows16(h, W, M, hcdim, mix_hc, eps, mixes, rsq, st);
}
void mask_by_cand(bf16* score, int M, int T, int bs, const int32_t* cand, int cand_stride, int kb, cudaStream_t st) {
  mask_by_cand_kernel<<<grid1d((size_t)M * T), 256, 0, st>>>(score, M, T, bs, cand, cand_stride, kb);
}
void embed_expand(const bf16* embed, const int32_t* ids, int M, int dim, int hc, bf16* h, cudaStream_t st) {
  embed_expand_kernel<<<grid1d((size_t)M * hc * dim), 256, 0, st>>>(embed, ids, M, dim, hc, h);
}
void identity_pre_mix(float* pre, int M, int hc, cudaStream_t st) {
  identity_pre_mix_kernel<<<grid1d((size_t)M * hc), 256, 0, st>>>(pre, M, hc);
}
void hc_row_rsqrt(const bf16* h, int M, int hcdim, float eps, float* rsq, cudaStream_t st) {
  hc_row_rsqrt_kernel<<<M, 256, 0, st>>>(h, hcdim, eps, rsq);
}
void hc_flatten_f32(const bf16* h, int M, int hcdim, float* out, cudaStream_t st) {
  bf16_to_f32_kernel<<<grid1d((size_t)M * hcdim), 256, 0, st>>>(h, (size_t)M * hcdim, out);
}
void bf16_to_f32(const bf16* src, int n, float* dst, cudaStream_t st) {
  bf16_to_f32_kernel<<<grid1d((size_t)n), 256, 0, st>>>(src, (size_t)n, dst);
}
void f32_to_bf16(const float* src, int n, bf16* dst, cudaStream_t st) {
  f32_to_bf16_kernel<<<grid1d((size_t)n), 256, 0, st>>>(src, (size_t)n, dst);
}
void hc_split_sinkhorn(const float* mixes, const float* rsq, const float* scale, const float* base, int M, int hc, int iters,
                       float eps, float* pre, float* post, float* comb, cudaStream_t st) {
  HIVE_CHECK(hc == 4, "hc==4");
  sinkhorn_kernel<4><<<grid1d(M, 128), 128, 0, st>>>(mixes, rsq, scale, base, M, iters, eps, pre, post, comb);
}
void hc_mix(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, cudaStream_t st) {
  hc_mix_kernel<<<dim3(M, mix_hc + 1), 256, 0, st>>>(h, W, hcdim, mix_hc, eps, mixes, rsq);
}
void hc_mix_pre_norm(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, const float* pre_in, int hc, int dim,
                     const bf16* norm_w, float norm_eps, bf16* x, bf16* xn, cudaStream_t st) {
  HIVE_CHECK(hc == 4 && hcdim == hc * dim, "hc_mix_pre_norm shape");
  hc_mix_pre_norm_kernel<4><<<dim3(M, mix_hc + 2), 256, 0, st>>>(h, W, hcdim, mix_hc, eps, mixes, rsq, pre_in, dim, norm_w, norm_eps, x, xn);
}
void hc_mix_pre_norm_sk(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, const float* pre_in, int hc, int dim,
                        const bf16* norm_w, float norm_eps, bf16* x, bf16* xn, const float* scale, const float* base, int iters, float hc_eps, float* pre_out,
                        float* post, float* comb, int* cnt, cudaStream_t st) {
  HIVE_CHECK(hc == 4 && hcdim == hc * dim && mix_hc == (2 + hc) * hc && M >= 1 && M <= 16, "hc_mix_pre_norm_sk shape");
  hc_mix_pre_norm_sk_kernel<4><<<dim3(M, mix_hc + 2), 256, 0, st>>>(h, W, hcdim, mix_hc, eps, mixes, rsq, pre_in, dim, norm_w, norm_eps, x, xn, scale, base, M,
                                                                     iters, hc_eps, pre_out, post, comb, cnt);
}
void hc_mix_pre_norm_sk_preload() {
  cudaFuncAttributes fa;
  CUDA_CHECK(cudaFuncGetAttributes(&fa, hc_mix_pre_norm_sk_kernel<4>));
}
void hc_pre_norm(const bf16* h, const float* pre_in, int M, int hc, int dim, const bf16* norm_w, float norm_eps, bf16* x, bf16* xn, cudaStream_t st) {
  HIVE_CHECK(hc == 4, "hc==4");
  hc_pre_norm_kernel<4><<<M, 256, 0, st>>>(h, pre_in, dim, norm_w, norm_eps, x, xn);
}
void hc_pre_fused(const bf16* h, const float* mixes, const float* rsq, const float* scale, const float* base, const float* pre_in, int M, int hc,
                  int dim, int iters, float hc_eps, const bf16* norm_w, float norm_eps, float* pre_out, float* post, float* comb, bf16* x, bf16* xn,
                  cudaStream_t st) {
  HIVE_CHECK(hc == 4, "hc==4");
  hc_pre_fused_kernel<4><<<M, 256, 0, st>>>(h, mixes, rsq, scale, base, pre_in, dim, iters, hc_eps, norm_w, norm_eps, pre_out, post, comb, x, xn);
}
void route_to_host(const int32_t* ids, const float* rw, const uint8_t* xq, const uint8_t* xs, int n_route, int n_act, int32_t* ids_h,
                   float* rw_h, uint8_t* xq_h, uint8_t* xs_h, cudaStream_t st) {
  route_to_host_kernel<<<grid1d((size_t)std::max(n_route, n_act)), 256, 0, st>>>(ids, rw, xq, xs, n_route, n_act, ids_h, rw_h, xq_h, xs_h);
}
void hc_pre(const bf16* h, const float* pre, int M, int hc, int dim, bf16* x, cudaStream_t st) {
  hc_pre_kernel<<<grid1d((size_t)M * dim), 256, 0, st>>>(h, pre, M, hc, dim, x);
}
void hc_post(const bf16* x, const float* post, const float* comb, int M, int hc, int dim, bf16* h, cudaStream_t st) {
  HIVE_CHECK(hc <= 4, "hc<=4");
  hc_post_kernel<<<grid1d((size_t)M * dim), 256, 0, st>>>(x, post, comb, M, hc, dim, h);
}
void rope_last(bf16* x, int M, int H, int D, int rd, const float2* freqs, const int32_t* pos, bool inverse, cudaStream_t st) {
  rope_last_kernel<<<grid1d((size_t)M * H * (rd / 2)), 256, 0, st>>>(x, M, H, D, rd, freqs, pos, inverse);
}
void sparse_attn(const bf16* q, int M, int H, int D, const KvSources& kv, const int32_t* idx, int idx_stride, int topk,
                 const float* sink, float scale, bf16* o, cudaStream_t st) {
  HIVE_CHECK(D == ATT_D && H % ATT_HG == 0, "sparse_attn shape");
  size_t smem = (size_t)ATT_HG * ATT_D * 2 + (size_t)ATT_BLK * ATT_D * 2 + (size_t)ATT_HG * ATT_BLK * 4 + 3 * ATT_HG * 4;
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(sparse_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    configured = true;
  }
  sparse_attn_kernel<<<M * (H / ATT_HG), 256, smem, st>>>(q, H, kv, idx, idx_stride, topk, sink, scale, o);
}
void window_idxs(int M, int win, int64_t start_pos, int32_t* out, int out_stride, cudaStream_t st, int64_t min_src) {
  window_idxs_kernel<<<grid1d((size_t)M * win), 256, 0, st>>>(M, win, start_pos, min_src, out, out_stride);
}
void window_idxs_rows(int M, int win, const int32_t* pos, int32_t* out, int out_stride, cudaStream_t st) {
  window_idxs_rows_kernel<<<grid1d((size_t)M * win), 256, 0, st>>>(M, win, pos, out, out_stride);
}
void ring_write_rows(const bf16* kv, int M, int D, int win, const int32_t* pos, bf16* const* ring_ptrs, cudaStream_t st) {
  ring_write_rows_kernel<<<grid1d((size_t)M * D), 256, 0, st>>>(kv, M, D, win, pos, ring_ptrs);
}
void compressor_step_rows(const float* kv, const float* score, int M, int D, int ratio, const int32_t* pos, float* const* state_kv,
                          float* const* state_score, float* out, uint8_t* valid, cudaStream_t st) {
  compressor_step_rows_kernel<<<M, 256, 0, st>>>(kv, score, M, D, ratio, pos, state_kv, state_score, out, valid);
}
void write_cache_rows(const bf16* src, int M, int D, bf16* const* dst_ptrs, const int32_t* dst_index, const uint8_t* valid, cudaStream_t st) {
  write_cache_rows_kernel<<<grid1d((size_t)M * D), 256, 0, st>>>(src, M, D, dst_ptrs, dst_index, valid);
}
void indexer_scores_rows(const bf16* q, const uint8_t* const* k_ptrs, const int32_t* T_rows, int Tmax, const bf16* w, int M, int Hi, int Di,
                         const int32_t* visible, bf16* score, cudaStream_t st) {
  HIVE_CHECK(Di <= 128, "indexer Di");
  dim3 grid((Tmax + IDX_T - 1) / IDX_T, M);
  size_t smem = (size_t)Hi * Di * 4 + Hi * 4;
  indexer_scores_rows_kernel<<<grid, IDX_T, smem, st>>>(q, k_ptrs, T_rows, Tmax, w, Hi, Di, visible, score);
}
void indexer_scores_rows_f32(const bf16* q, const uint8_t* const* k_ptrs, const int32_t* T_rows, int Tmax, const bf16* w, int M, int Hi, int Di,
                             const int32_t* visible, float* score, cudaStream_t st) {
  HIVE_CHECK(Di <= 128, "indexer Di");
  dim3 grid((Tmax + IDX_T - 1) / IDX_T, M);
  size_t smem = (size_t)Hi * Di * 4 + Hi * 4;
  indexer_scores_rows_f32_kernel<<<grid, IDX_T, smem, st>>>(q, k_ptrs, T_rows, Tmax, w, Hi, Di, visible, score);
}
// P1
void indexer_scores_cand(const bf16* q, const uint8_t* k, const uint8_t* const* k_ptrs, const int32_t* T_rows, const bf16* w, int M, int Hi, int Di,
                         const int32_t* cand, int cand_stride, int kb, int bs, const int32_t* visible, bf16* score, cudaStream_t st) {
  HIVE_CHECK(Di <= 128 && kb >= 1 && bs >= 1, "indexer_scores_cand shape");
  const int ncand = kb * bs;
  const size_t smem = (size_t)(Hi * Di + Hi) * 4;  // 16.5 KB (Hi 32, Di 128) — within the static limit
  dim3 grid((ncand + IDX_T - 1) / IDX_T, M);
  indexer_scores_cand_kernel<<<grid, IDX_T, smem, st>>>(q, k, k_ptrs, T_rows, w, Hi, Di, cand, cand_stride, bs, ncand, visible, score);
}
void indexer_scores_cand_f32(const bf16* q, const uint8_t* k, const uint8_t* const* k_ptrs, const int32_t* T_rows, const bf16* w, int M, int Hi, int Di,
                             const int32_t* cand, int cand_stride, int kb, int bs, const int32_t* visible, float* score, cudaStream_t st) {
  HIVE_CHECK(Di <= 128 && kb >= 1 && bs >= 1, "indexer_scores_cand shape");
  const int ncand = kb * bs;
  const size_t smem = (size_t)(Hi * Di + Hi) * 4;
  dim3 grid((ncand + IDX_T - 1) / IDX_T, M);
  indexer_scores_cand_f32_kernel<<<grid, IDX_T, smem, st>>>(q, k, k_ptrs, T_rows, w, Hi, Di, cand, cand_stride, bs, ncand, visible, score);
}
void cand_to_pos(int32_t* sel, int M, int k, const int32_t* cand, int cand_stride, int bs, cudaStream_t st) {
  cand_to_pos_kernel<<<grid1d((size_t)M * k), 256, 0, st>>>(sel, M, k, cand, cand_stride, bs);
}
void ring_write(const bf16* kv, int M, int D, int win, int64_t start_pos, bf16* ring, cudaStream_t st) {
  int first = M > win ? M - win : 0;
  size_t n = (size_t)(M - first) * D;
  ring_write_kernel<<<grid1d(n), 256, 0, st>>>(kv, M, D, win, start_pos, ring);
}
void scale_bf16(bf16* x, int n, float c, cudaStream_t st) { scale_bf16_kernel<<<grid1d((size_t)n), 256, 0, st>>>(x, (size_t)n, c); }
void iota_rows(int32_t* out, int M, int T, cudaStream_t st) {
  iota_rows_kernel<<<grid1d((size_t)M * T), 256, 0, st>>>(out, (size_t)M * T, T);
}
void bf16_rows_to_f32(const bf16* src, int n, float* dst, cudaStream_t st) { bf16_to_f32(src, n, dst, st); }
void compressor_pool(const float* kv, const float* score, int M, int D, int ratio, int64_t start_pos, float* state_kv,
                     float* state_score, float* out, cudaStream_t st) {
  const int64_t end = start_pos + M;
  const int64_t g0 = start_pos / ratio;
  const int64_t G = end / ratio - g0;  // number of complete groups
  if (G > 0) compressor_pool_kernel<<<(int)G, 256, 0, st>>>(kv, score, M, D, ratio, start_pos, state_kv, state_score, out);
  compressor_tail_kernel<<<1, 256, 0, st>>>(kv, score, M, D, ratio, start_pos, state_kv, state_score);
}
void indexer_scores(const bf16* q, const uint8_t* k, const bf16* w, int M, int Hi, int Di, int T, const int32_t* visible,
                    bf16* score, cudaStream_t st) {
  HIVE_CHECK(Di <= 128, "indexer Di");
  dim3 grid((T + IDX_T - 1) / IDX_T, M);
  size_t smem = (size_t)Hi * Di * 4 + Hi * 4;
  indexer_scores_kernel<<<grid, IDX_T, smem, st>>>(q, k, w, Hi, Di, T, visible, score);
}
void indexer_scores_f32(const bf16* q, const uint8_t* k, const bf16* w, int M, int Hi, int Di, int T, const int32_t* visible, float* score, cudaStream_t st) {
  HIVE_CHECK(Di <= 128, "indexer Di");
  dim3 grid((T + IDX_T - 1) / IDX_T, M);
  size_t smem = (size_t)Hi * Di * 4 + Hi * 4;
  indexer_scores_f32_kernel<<<grid, IDX_T, smem, st>>>(q, k, w, Hi, Di, T, visible, score);
}
void block_max(const bf16* score, int M, int T, int bs, const int32_t* visible, float* bmax, cudaStream_t st) {
  int nblocks = (T + bs - 1) / bs;
  block_max_kernel<<<grid1d((size_t)M * nblocks), 256, 0, st>>>(score, M, T, bs, visible, bmax);
}
void block_max_f32(const float* score, int M, int T, int bs, const int32_t* visible, float* bmax, cudaStream_t st) {
  int nblocks = (T + bs - 1) / bs;
  block_max_f32_kernel<<<grid1d((size_t)M * nblocks), 256, 0, st>>>(score, M, T, bs, visible, bmax);
}
void scatter_keep(const int32_t* pos, const float* bmax, int M, int k, int nblocks, uint8_t* keep, int keep_stride, cudaStream_t st) {
  scatter_keep_kernel<<<grid1d((size_t)M * k), 256, 0, st>>>(pos, bmax, M, k, nblocks, keep, keep_stride);
}
void apply_block_mask(bf16* score, int M, int T, int bs, const uint8_t* keep_block, int nblocks, cudaStream_t st) {
  apply_block_mask_kernel<<<grid1d((size_t)M * T), 256, 0, st>>>(score, M, T, bs, keep_block, nblocks);
}
void offset_idxs(const int32_t* pos, int M, int topk, const int32_t* visible, int32_t offset, int32_t* out, int out_stride,
                 cudaStream_t st) {
  offset_idxs_kernel<<<grid1d((size_t)M * topk), 256, 0, st>>>(pos, M, topk, visible, offset, out, out_stride);
}
void router_topk(const float* scores, const float* bias, const float* bias_vl, const int8_t* is_image, int M, int E, int k,
                 float route_scale, int32_t* ids, float* w, cudaStream_t st) {
  HIVE_CHECK(E <= 512, "E<=512");
  router_topk_kernel<<<M, 256, 0, st>>>(scores, bias, bias_vl, is_image, E, k, route_scale, ids, w);
}
void swiglu_route(const bf16* gate, const bf16* up, const float* route_w, int M, int I, float limit, bf16* y, cudaStream_t st) {
  swiglu_route_kernel<<<grid1d((size_t)M * I), 256, 0, st>>>(gate, up, route_w, M, I, limit, y);
}
void accum_bf16_rows(const bf16* src, const int32_t* row_map, int M, int dim, float* acc, cudaStream_t st) {
  accum_bf16_rows_kernel<<<grid1d((size_t)M * dim), 256, 0, st>>>(src, row_map, M, dim, acc);
}
void accum_f32_rows(const float* src, const int32_t* row_map, int M, int dim, float* acc, cudaStream_t st) {
  accum_f32_rows_kernel<<<grid1d((size_t)M * dim), 256, 0, st>>>(src, row_map, M, dim, acc);
}
void accum_bf16_rows_seq(const bf16* src, const int32_t* row_map, int R, int M, int dim, float* acc, cudaStream_t st) {
  if (R <= 0) return;
  accum_bf16_rows_seq_kernel<<<grid1d((size_t)M * dim), 256, 0, st>>>(src, row_map, R, M, dim, acc);
}
void gather_rows_bf16(const bf16* src, const int32_t* rows, int n, int dim, bf16* dst, cudaStream_t st) {
  gather_rows_bf16_kernel<<<grid1d((size_t)n * dim), 256, 0, st>>>(src, rows, n, dim, dst);
}
void gather_rows_u8(const uint8_t* src, const int32_t* rows, int n, int row_bytes, uint8_t* dst, cudaStream_t st) {
  gather_rows_u8_kernel<<<grid1d((size_t)n * row_bytes), 256, 0, st>>>(src, rows, n, row_bytes, dst);
}
void engram_dequant(const uint8_t* vals, const uint8_t* scales, int M, int cols, int hd, bf16* out, cudaStream_t st) {
  engram_dequant_kernel<<<grid1d((size_t)M * cols * hd), 256, 0, st>>>(vals, scales, M, cols, hd, out);
}
void engram_gate(const bf16* kv, const float* qk_weight, const int8_t* is_image, int M, int hc, int dim, float eps, bf16* h,
                 cudaStream_t st) {
  engram_gate_kernel<<<M * hc, 256, 0, st>>>(kv, qk_weight, is_image, hc, dim, eps, h);
}
namespace {
__global__ void merge_image_kernel(bf16* __restrict__ h, int hc, int dim, int start, int span, const int8_t* __restrict__ types,
                                   const int32_t* __restrict__ row_of, const bf16* __restrict__ rows, const bf16* __restrict__ vs,
                                   const bf16* __restrict__ vn, const bf16* __restrict__ ve) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)span * dim) return;
  int p = i / dim, d = i % dim;
  int t = types[p];
  bf16 v;
  if (t == 0) v = vs[d];
  else if (t == 2) v = vn[d];
  else if (t == 3) v = ve[d];
  else v = rows[(size_t)row_of[p] * dim + d];
  for (int c = 0; c < hc; ++c) h[((size_t)(start + p) * hc + c) * dim + d] = v;
}
}  // namespace
void merge_image(bf16* h, int hc, int dim, int start, int span, const int8_t* types, const int32_t* row_of, const bf16* rows,
                 const bf16* v_start, const bf16* v_newline, const bf16* v_end, cudaStream_t st) {
  merge_image_kernel<<<grid1d((size_t)span * dim), 256, 0, st>>>(h, hc, dim, start, span, types, row_of, rows, v_start, v_newline, v_end);
}
void head_logits(const float* x, const bf16* head, int M, int V, int dim, float* logits, cudaStream_t st) {
  static const bool rows = !(getenv("HIVE_HEAD_ROWS") && strcmp(getenv("HIVE_HEAD_ROWS"), "0") == 0);  // default on; "0" = the per-row kernel
  if (rows && M >= 1 && M <= 8 && dim % 2 == 0) {
    const int blocks = (V + 8 * kHeadHR - 1) / (8 * kHeadHR);
    if (M == 1) head_logits_rows_kernel<1><<<blocks, 256, 0, st>>>(x, head, M, V, dim, logits);
    else if (M == 2) head_logits_rows_kernel<2><<<blocks, 256, 0, st>>>(x, head, M, V, dim, logits);
    else if (M <= 4) head_logits_rows_kernel<4><<<blocks, 256, 0, st>>>(x, head, M, V, dim, logits);
    else head_logits_rows_kernel<8><<<blocks, 256, 0, st>>>(x, head, M, V, dim, logits);
    return;
  }
  head_logits_kernel<<<grid1d((size_t)V * 32), 256, 0, st>>>(x, head, M, V, dim, logits);
}
void head_logits_per_row(const float* x, const bf16* head, int M, int V, int dim, float* logits, cudaStream_t st) {  // the previous kernel (tests)
  head_logits_kernel<<<grid1d((size_t)V * 32), 256, 0, st>>>(x, head, M, V, dim, logits);
}
void argmax_rows(const float* logits, int M, int V, int32_t* out, cudaStream_t st) {
  argmax_rows_kernel<<<M, 1024, 0, st>>>(logits, V, out);
}

// ---- DSpark (MTP) helper kernels ----------------------------------------------------------------------------------------
namespace {
// hc mean of the target-layer attention input (reference h.mean(dim=2) — bf16 tensor mean: fp32 accumulation → bf16): dst[i, col_off + d] = mean_c h[row0+i, c, d]
__global__ void hc_mean_rows_kernel(const bf16* __restrict__ h, int hc, int dim, int row0, int n, bf16* __restrict__ dst, int dst_stride, int col_off) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)n * dim) return;
  int r = (int)(i / dim), d = (int)(i % dim);
  const bf16* p = h + (size_t)(row0 + r) * hc * dim + d;
  float acc = 0.f;
  for (int c = 0; c < hc; ++c) acc += bf2f(p[(size_t)c * dim]);
  dst[(size_t)r * dst_stride + col_off + d] = f2bf(acc / (float)hc);
}
// markov embedding row gather: dst[rank] = embed[id]
__global__ void markov_gather_kernel(const bf16* __restrict__ embed, const int32_t* __restrict__ id, int rank, bf16* __restrict__ dst) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rank) dst[r] = embed[(size_t)(*id) * rank + r];
}
// logits[v] += Σ_r head[v,r]·e[r]  (reference ParallelHead: F.linear(embed.float(), head fp32) added to the fp32 logits) — warp = 1 vocabulary entry
__global__ void markov_add_bias_kernel(float* __restrict__ logits, const bf16* __restrict__ head, const bf16* __restrict__ e, int V, int rank) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= V) return;
  const bf16* hr = head + (size_t)warp * rank;
  float acc = 0.f;
  for (int r = lane; r < rank; r += 32) acc = fmaf(bf2f(hr[r]), bf2f(e[r]), acc);
  acc = warp_sum(acc);
  if (lane == 0) logits[warp] += acc;
}
// Confidence: out[b] = Σ_d w[d]·x[b,d] + Σ_r w[dim+r]·e[b,r]  (reference proj(cat[hidden, markov_embed].float()) — fp32)
__global__ void conf_dot_kernel(const bf16* __restrict__ x, int dim, const bf16* __restrict__ e, int rank, const float* __restrict__ w,
                                float* __restrict__ out) {
  __shared__ float red[32];
  const int b = blockIdx.x;
  float acc = 0.f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) acc = fmaf(w[d], bf2f(x[(size_t)b * dim + d]), acc);
  for (int r = threadIdx.x; r < rank; r += blockDim.x) acc = fmaf(w[dim + r], bf2f(e[(size_t)b * rank + r]), acc);
  acc = block_sum<256>(acc, red);
  if (threadIdx.x == 0) out[b] = acc;
}
// Ring snapshot: dst[l, i, :] = rings[l][(start_pos+i) % win, :]   / restore: rings[l][(start_pos+i) % win] = src[l, i]  (i ∈ [i0, n))
__global__ void ring_gather_layers_kernel(bf16* const* __restrict__ rings, int L, int win, int64_t start_pos, int n, int D, bf16* __restrict__ dst) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)L * n * D) return;
  int d = (int)(i % D), r = (int)((i / D) % n), l = (int)(i / ((size_t)n * D));
  dst[i] = rings[l][(size_t)((start_pos + r) % win) * D + d];
}
__global__ void ring_scatter_layers_kernel(const bf16* __restrict__ src, bf16* const* __restrict__ rings, int L, int win, int64_t start_pos, int i0,
                                           int n, int D) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)L * n * D) return;
  int d = (int)(i % D), r = (int)((i / D) % n), l = (int)(i / ((size_t)n * D));
  if (r < i0) return;
  rings[l][(size_t)((start_pos + r) % win) * D + d] = src[i];
}
// Draft block attention indices (reference get_dspark_topk_idxs): per row, ring slots [0, filled), block rows [win, win+B), rest -1. Columns = win + B.
__global__ void dspark_idxs_kernel(int B, int filled, int win, int32_t* __restrict__ out, int stride) {
  const size_t i = hive::cu::global_tid();
  const int ncols = win + B;
  if (i >= (size_t)B * ncols) return;
  int m = (int)(i / ncols), j = (int)(i % ncols);
  int32_t v;
  if (j < filled) v = j;
  else if (j < filled + B) v = win + (j - filled);
  else v = -1;
  out[(size_t)m * stride + j] = v;
}
// Scatter through per-row destination pointers: dst_ptrs[m][0..n) = src[m, 0..n)
__global__ void scatter_rows_ptrs_kernel(const bf16* __restrict__ src, int M, int n, bf16* const* __restrict__ dst_ptrs) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * n) return;
  int m = (int)(i / n), j = (int)(i % n);
  dst_ptrs[m][j] = src[i];
}
}  // namespace
// ---- GPU sampler helpers: row max, Σexp((l−max)·inv_temp) (fp32), gather of candidate values ----------------------------------------------
namespace {
__global__ void row_max_sumexp_kernel(const float* __restrict__ logits, int V, const float* __restrict__ inv_temp, float* __restrict__ rowmax,
                                      float* __restrict__ rowsum) {
  __shared__ float red[32];
  __shared__ float smx;
  const int m = blockIdx.x;
  const float* row = logits + (size_t)m * V;
  float mx = -FLT_MAX;
  for (int v = threadIdx.x; v < V; v += blockDim.x) mx = fmaxf(mx, row[v]);
  mx = warp_max(mx);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x == 0) { float t = red[0]; for (int i = 1; i < (int)(blockDim.x >> 5); ++i) t = fmaxf(t, red[i]); smx = t; }
  __syncthreads();
  mx = smx;
  const float it = inv_temp[m];
  float s = 0.f;
  for (int v = threadIdx.x; v < V; v += blockDim.x) s += expf((row[v] - mx) * it);
  s = block_sum<1024>(s, red);
  if (threadIdx.x == 0) { rowmax[m] = mx; rowsum[m] = s; }
}
__global__ void gather_vals_kernel(const float* __restrict__ logits, const int32_t* __restrict__ idx, int M, int NC, int V, float* __restrict__ vals) {
  const size_t i = hive::cu::global_tid();
  if (i >= (size_t)M * NC) return;
  const int m = (int)(i / NC), j = idx[i];
  vals[i] = j >= 0 ? logits[(size_t)m * V + j] : -INFINITY;
}
}  // namespace
void row_max_sumexp(const float* logits, int M, int V, const float* inv_temp, float* rowmax, float* rowsum, cudaStream_t st) {
  row_max_sumexp_kernel<<<M, 1024, 0, st>>>(logits, V, inv_temp, rowmax, rowsum);
}
void gather_vals(const float* logits, const int32_t* idx, int M, int NC, int V, float* vals, cudaStream_t st) {
  gather_vals_kernel<<<grid1d((size_t)M * NC), 256, 0, st>>>(logits, idx, M, NC, V, vals);
}

void hc_mean_rows(const bf16* h, int hc, int dim, int row0, int n, bf16* dst, int dst_stride, int col_off, cudaStream_t st) {
  if (n <= 0) return;
  hc_mean_rows_kernel<<<grid1d((size_t)n * dim), 256, 0, st>>>(h, hc, dim, row0, n, dst, dst_stride, col_off);
}
void markov_gather(const bf16* embed, const int32_t* id, int rank, bf16* dst, cudaStream_t st) {
  markov_gather_kernel<<<grid1d((size_t)rank), 256, 0, st>>>(embed, id, rank, dst);
}
void markov_add_bias(float* logits, const bf16* head, const bf16* e, int V, int rank, cudaStream_t st) {
  markov_add_bias_kernel<<<grid1d((size_t)V * 32), 256, 0, st>>>(logits, head, e, V, rank);
}
void conf_dot(const bf16* x, int dim, const bf16* e, int rank, const float* w, int B, float* out, cudaStream_t st) {
  conf_dot_kernel<<<B, 256, 0, st>>>(x, dim, e, rank, w, out);
}
void ring_gather_layers(bf16* const* rings, int L, int win, int64_t start_pos, int n, int D, bf16* dst, cudaStream_t st) {
  if (L <= 0 || n <= 0) return;
  ring_gather_layers_kernel<<<grid1d((size_t)L * n * D), 256, 0, st>>>(rings, L, win, start_pos, n, D, dst);
}
void ring_scatter_layers(const bf16* src, bf16* const* rings, int L, int win, int64_t start_pos, int i0, int n, int D, cudaStream_t st) {
  if (L <= 0 || n <= i0) return;
  ring_scatter_layers_kernel<<<grid1d((size_t)L * n * D), 256, 0, st>>>(src, rings, L, win, start_pos, i0, n, D);
}
void dspark_idxs(int B, int filled, int win, int32_t* out, int stride, cudaStream_t st) {
  dspark_idxs_kernel<<<grid1d((size_t)B * (win + B)), 256, 0, st>>>(B, filled, win, out, stride);
}
void scatter_rows_ptrs(const bf16* src, int M, int n, bf16* const* dst_ptrs, cudaStream_t st) {
  scatter_rows_ptrs_kernel<<<grid1d((size_t)M * n), 256, 0, st>>>(src, M, n, dst_ptrs);
}

}  // namespace hive::k
