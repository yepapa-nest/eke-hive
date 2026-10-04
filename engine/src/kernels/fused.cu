// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Fused decode kernels — halve the chain of ~40 launches per layer. Profiling showed that of a 24.6 ms token, the 19 ms graph
// section is dominated not by bandwidth (dense 175 MB per layer = 0.11 ms) but by kernel launch and dependency latency (~30 per
// layer × a few µs), so small elementwise kernels are attached before and after the GEMVs.
//
// Numeric contract: aims for **bit identity** with the unfused chain by using the same formulas and the same reduction order.
//   · Input quantization (act_quant_fp8): 32-block amax → ceil-pow2 e8m0 → clamp(±448) → e4m3 (RNE). Built once per block in
//     smem, then the dot product follows the same order as gemv_bs_kernel (lanes rotate over 32-blocks, 32 fmaf, multiply by
//     block scale, xor reduction).
//   · RMSNorm (last block): same thread assignment and warp reduction order as rmsnorm_kernel<256>.
//   · RoPE (reference apply_rotary_emb): rotate the bf16-rounded values in fp32 → bf16.
//   · fp8 round trip, swiglu, accumulation: exactly the formulas of the respective kernels.
// Block = 32 output columns (OUT) × M ≤ 8 rows · 256 threads (8 warps, 4 columns per warp). With 32 columns per block, the output
// 32-block quantization and RoPE pairs close within the block.
#include <cuda_fp8.h>

#include <cfloat>
#include <stdexcept>
#include <string>

#include "hive/kernels.h"
#include "hive/model_kernels.h"
#include "hive/warp_reduce.cuh"

namespace hive::k {

namespace {

// Block = 8 columns (one column per warp) — same parallelism as gemv_bs_kernel (N/8 blocks). Measured: with 32 columns per block
//   and 4 per warp the block count is 1/4, and eager execution was slower than the chain (q+kv+shared 79→92 µs). The output
//   quantization (y) that needs 32 columns was moved to the consumer (w2), which quantizes its input in smem.
constexpr int FT = 256;
constexpr int FW = FT / 32;
constexpr int OUT = 8;
constexpr int CPW = OUT / FW;
constexpr int MMAX = 8;
constexpr float kFp8MaxInv = 1.0f / 448.0f;

__device__ __forceinline__ float2 cvt_e4m3x2_f(uint32_t pair16) {
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t)pair16));
  return __half22float2(*reinterpret_cast<__half2*>(&h2));
}
__device__ __forceinline__ float e8m0_fast_f(uint8_t b) { return b ? __uint_as_float((uint32_t)b << 23) : __uint_as_float(0x00400000u); }
__device__ __forceinline__ void decode8(const uint4 r0, const uint4 r1, float (&w)[32]) {
  const uint32_t x[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float2 a = cvt_e4m3x2_f(x[i] & 0xFFFFu), b = cvt_e4m3x2_f(x[i] >> 16);
    w[4 * i] = a.x; w[4 * i + 1] = a.y; w[4 * i + 2] = b.x; w[4 * i + 3] = b.y;
  }
}

// A[M,K] bf16 → smem e4m3 + scales (values and codes) — same formula as act_quant_fp8_kernel
__device__ __forceinline__ void quant_rows_smem(const bf16* __restrict__ A, int lda, int M, int K, uint8_t* aq, float* asv, uint8_t* asc) {
  const int nb = K / 32;
  for (int idx = threadIdx.x; idx < M * nb; idx += FT) {
    const int m = idx / nb, b = idx % nb;
    const bf16* p = A + (size_t)m * lda + b * 32;
    float v[32];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) { v[i] = bf2f(p[i]); amax = fmaxf(amax, fabsf(v[i])); }
    amax = fmaxf(amax, 1e-4f);
    const uint8_t code = f32_ceil_pow2_e8m0(amax * kFp8MaxInv);
    const float sc = e8m0_to_f32(code);
    uint8_t* q = aq + (size_t)m * K + b * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) q[i] = f32_to_e4m3(fminf(fmaxf(v[i] / sc, -448.f), 448.f));
    asv[m * nb + b] = sc;
    asc[m * nb + b] = code;
  }
}

// Warp: one column n × M rows · A in smem (e4m3 + scale values) or global (e4m3 + scale codes). Same order as gemv_bs_kernel.
template <bool A_SMEM>
__device__ __forceinline__ void warp_dot(const uint8_t* __restrict__ aq, const float* __restrict__ asv, const uint8_t* __restrict__ asc, int lda_q,
                                         const uint8_t* __restrict__ B, const uint8_t* __restrict__ sb, int M, int K, int n, int lane, float (&acc)[MMAX]) {
  const int nb = K / 32;
#pragma unroll
  for (int m = 0; m < MMAX; ++m) acc[m] = 0.f;
  for (int b = lane; b < nb; b += 32) {
    float w[32];
    const uint8_t* pb = B + (size_t)n * K + b * 32;
    decode8(*reinterpret_cast<const uint4*>(pb), *reinterpret_cast<const uint4*>(pb + 16), w);
    const float sbv = e8m0_fast_f(sb[(size_t)n * nb + b]);
#pragma unroll
    for (int m = 0; m < MMAX; ++m) {
      if (m < M) {
        const uint8_t* pa = aq + (size_t)m * lda_q + b * 32;
        float av[32];
        decode8(*reinterpret_cast<const uint4*>(pa), *reinterpret_cast<const uint4*>(pa + 16), av);
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) d = fmaf(av[i], w[i], d);
        const float sav = A_SMEM ? asv[m * nb + b] : e8m0_fast_f(asc[(size_t)m * nb + b]);
        acc[m] = fmaf(d, sav * sbv, acc[m]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MMAX; ++m) {
    float v = acc[m];
    v = hive::cu::warp_allreduce(v);
    acc[m] = v;
  }
}

// Same reduction as rmsnorm_kernel<256>: row x[N] (bf16) → out[N] (bf16). red[8] in smem.
__device__ __forceinline__ void rmsnorm_row_256(const bf16* __restrict__ x, const bf16* __restrict__ w, float eps, int N, bf16* __restrict__ out, float* red) {
  float ss = 0.f;
  for (int i = threadIdx.x; i < N; i += 256) { float v = bf2f(x[i]); ss += v * v; }
  ss = hive::cu::warp_allreduce(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
    t = hive::cu::warp_allreduce(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  const float rs = rsqrtf(red[0] / (float)N + eps);
  for (int i = threadIdx.x; i < N; i += 256) out[i] = f2bf(bf2f(w[i]) * (bf2f(x[i]) * rs));
  __syncthreads();
}

// Last-block test (threadfence reduction). Returns whether this block is the last one. The last block resets the counter to 0 (safe for graph replay).
__device__ __forceinline__ bool last_block(int* counter, int* flag_smem) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) *flag_smem = (atomicAdd(counter, 1) == (int)gridDim.x - 1);
  __syncthreads();
  const bool last = *flag_smem != 0;
  if (last) { __threadfence(); if (threadIdx.x == 0) *counter = 0; }
  return last;
}

// ---------------------------------------------------------------------------------------------------------------------
// ① q first half: qr = wq_a(quant(xn)) → [last block] qrn = rmsnorm(bf16(qr), q_norm)
__global__ void __launch_bounds__(FT) gemv_qa_kernel(const bf16* __restrict__ xn, int K, const uint8_t* __restrict__ B, const uint8_t* __restrict__ sb, int M,
                                                     int N, const bf16* __restrict__ norm_w, float eps, bf16* __restrict__ qr, bf16* __restrict__ qrn,
                                                     int* __restrict__ counter) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* aq = smem;
  float* asv = reinterpret_cast<float*>(smem + (size_t)M * K);
  uint8_t* asc = reinterpret_cast<uint8_t*>(asv + M * (K / 32));
  __shared__ float outv[MMAX][OUT];
  __shared__ float red[8];
  __shared__ int flag;
  quant_rows_smem(xn, K, M, K, aq, asv, asc);
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, n0 = blockIdx.x * OUT;
  for (int j = 0; j < CPW; ++j) {
    const int c = warp * CPW + j;
    float acc[MMAX];
    warp_dot<true>(aq, asv, asc, K, B, sb, M, K, n0 + c, lane, acc);
    if (lane == 0) for (int m = 0; m < M; ++m) outv[m][c] = acc[m];
  }
  __syncthreads();
  if (threadIdx.x < M * OUT) { const int m = threadIdx.x / OUT, c = threadIdx.x % OUT; qr[(size_t)m * N + n0 + c] = f2bf(outv[m][c]); }
  if (!last_block(counter, &flag)) return;
  for (int m = 0; m < M; ++m) rmsnorm_row_256(qr + (size_t)m * N, norm_w, eps, N, qrn + (size_t)m * N, red);
}

// ② q second half: q = wq_b(quant(qrn)) → RoPE (tail rd of each head). Side output: qrn quantized by block 0 (qrq·qrs — used by the indexer)
__global__ void __launch_bounds__(FT) gemv_qb_rope_kernel(const bf16* __restrict__ qrn, int K, const uint8_t* __restrict__ B, const uint8_t* __restrict__ sb,
                                                          int M, int N, int H, int D, int rd, const float2* __restrict__ freqs, const int32_t* __restrict__ pos,
                                                          bf16* __restrict__ q, uint8_t* __restrict__ side_q, uint8_t* __restrict__ side_s) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* aq = smem;
  float* asv = reinterpret_cast<float*>(smem + (size_t)M * K);
  uint8_t* asc = reinterpret_cast<uint8_t*>(asv + M * (K / 32));
  __shared__ float outv[MMAX][OUT];
  quant_rows_smem(qrn, K, M, K, aq, asv, asc);
  __syncthreads();
  if (blockIdx.x == 0 && side_q) {
    const int nb = K / 32;
    for (int i = threadIdx.x; i < M * K; i += FT) side_q[i] = aq[i];
    for (int i = threadIdx.x; i < M * nb; i += FT) side_s[i] = asc[i];
  }
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, n0 = blockIdx.x * OUT;
  for (int j = 0; j < CPW; ++j) {
    const int c = warp * CPW + j;
    float acc[MMAX];
    warp_dot<true>(aq, asv, asc, K, B, sb, M, K, n0 + c, lane, acc);
    if (lane == 0) for (int m = 0; m < M; ++m) outv[m][c] = acc[m];
  }
  __syncthreads();
  if (threadIdx.x >= M * OUT) return;
  const int m = threadIdx.x / OUT, c = threadIdx.x % OUT, n = n0 + c;
  const int dpos = n % D;
  (void)H;
  if (dpos < D - rd) { q[(size_t)m * N + n] = f2bf(outv[m][c]); return; }
  if (c & 1) return;  // pair (2p, 2p+1): the even column writes both
  const int p = (dpos - (D - rd)) >> 1;
  const float2 f = freqs[(size_t)pos[m] * (rd / 2) + p];
  const float a = bf2f(f2bf(outv[m][c])), b = bf2f(f2bf(outv[m][c + 1]));
  q[(size_t)m * N + n] = f2bf(a * f.x - b * f.y);
  q[(size_t)m * N + n + 1] = f2bf(a * f.y + b * f.x);
}

// ③ kv: kv = wkv(quant(xn)) → [last block] rmsnorm(kv_norm) → RoPE (tail rd, 1 head) → fp8 round trip → kv row + ring[(pos % win)] (when ring_ptrs is given)
__global__ void __launch_bounds__(FT) gemv_kv_kernel(const bf16* __restrict__ xn, int K, const uint8_t* __restrict__ B, const uint8_t* __restrict__ sb, int M,
                                                     int D, const bf16* __restrict__ norm_w, float eps, int rd, const float2* __restrict__ freqs,
                                                     const int32_t* __restrict__ pos, bf16* __restrict__ kv, bf16* const* __restrict__ ring_ptrs, int win,
                                                     int32_t* __restrict__ idx_out, int idx_stride, int* __restrict__ counter) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* aq = smem;
  float* asv = reinterpret_cast<float*>(smem + (size_t)M * K);
  uint8_t* asc = reinterpret_cast<uint8_t*>(asv + M * (K / 32));
  __shared__ float outv[MMAX][OUT];
  __shared__ __align__(16) bf16 row[512];
  __shared__ float red[8];
  __shared__ int flag;
  quant_rows_smem(xn, K, M, K, aq, asv, asc);
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, n0 = blockIdx.x * OUT;
  for (int j = 0; j < CPW; ++j) {
    const int c = warp * CPW + j;
    float acc[MMAX];
    warp_dot<true>(aq, asv, asc, K, B, sb, M, K, n0 + c, lane, acc);
    if (lane == 0) for (int m = 0; m < M; ++m) outv[m][c] = acc[m];
  }
  __syncthreads();
  if (threadIdx.x < M * OUT) { const int m = threadIdx.x / OUT, c = threadIdx.x % OUT; kv[(size_t)m * D + n0 + c] = f2bf(outv[m][c]); }
  if (!last_block(counter, &flag)) return;
  if (idx_out) {  // window_idxs_rows: position p of row m; earlier tokens are ring slots (src % win), its own kv is chunk row (win + m), otherwise -1
    for (int i = threadIdx.x; i < M * win; i += FT) {
      const int m = i / win, j = i % win;
      const int64_t p = pos[m], src = p - (win - 1) + j;
      idx_out[(size_t)m * idx_stride + j] = src < 0 ? -1 : (src < p ? (int32_t)(src % win) : (int32_t)(win + m));
    }
  }
  const int half = rd / 2;
  for (int m = 0; m < M; ++m) {
    bf16* kvm = kv + (size_t)m * D;
    rmsnorm_row_256(kvm, norm_w, eps, D, row, red);  // row = normalized (bf16)
    if (threadIdx.x < half) {  // RoPE: pairs (2p, 2p+1) are adjacent
      const float2 f = freqs[(size_t)pos[m] * half + threadIdx.x];
      bf16* base = row + (D - rd) + 2 * threadIdx.x;
      const float a = bf2f(base[0]), b = bf2f(base[1]);
      base[0] = f2bf(a * f.x - b * f.y);
      base[1] = f2bf(a * f.y + b * f.x);
    }
    __syncthreads();
    if (threadIdx.x < D / 32) {  // fp8 round trip (act_quant_fp8_roundtrip)
      bf16* p = row + threadIdx.x * 32;
      float v[32];
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 32; ++i) { v[i] = bf2f(p[i]); amax = fmaxf(amax, fabsf(v[i])); }
      amax = fmaxf(amax, 1e-4f);
      const float sc = e8m0_to_f32(f32_ceil_pow2_e8m0(amax * kFp8MaxInv));
#pragma unroll
      for (int i = 0; i < 32; i++) {
        const float t = fminf(fmaxf(v[i] / sc, -448.f), 448.f);
        p[i] = f2bf(e4m3_to_f32(f32_to_e4m3(t)) * sc);
      }
    }
    __syncthreads();
    bf16* ring = ring_ptrs ? ring_ptrs[m] + (size_t)(pos[m] % win) * D : nullptr;
    for (int i = threadIdx.x; i < D; i += FT) { kvm[i] = row[i]; if (ring) ring[i] = row[i]; }
    __syncthreads();
  }
}

// ④ generic: C = W(quant(A)) bf16 (wo_b: og → attn_out). No side outputs.
__global__ void __launch_bounds__(FT) gemv_quantin_kernel(const bf16* __restrict__ A, int K, const uint8_t* __restrict__ B, const uint8_t* __restrict__ sb, int M,
                                                          int N, bf16* __restrict__ C) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* aq = smem;
  float* asv = reinterpret_cast<float*>(smem + (size_t)M * K);
  uint8_t* asc = reinterpret_cast<uint8_t*>(asv + M * (K / 32));
  __shared__ float outv[MMAX][OUT];
  quant_rows_smem(A, K, M, K, aq, asv, asc);
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, n0 = blockIdx.x * OUT;
  for (int j = 0; j < CPW; ++j) {
    const int c = warp * CPW + j;
    float acc[MMAX];
    warp_dot<true>(aq, asv, asc, K, B, sb, M, K, n0 + c, lane, acc);
    if (lane == 0) for (int m = 0; m < M; ++m) outv[m][c] = acc[m];
  }
  __syncthreads();
  if (threadIdx.x < M * OUT) { const int m = threadIdx.x / OUT, c = threadIdx.x % OUT; C[(size_t)m * N + n0 + c] = f2bf(outv[m][c]); }
}

// ⑤ shared expert, first half: gate = w1(quant(xn)), up = w3(quant(xn)) → y = bf16(swiglu(bf16 gate, bf16 up)) → yq/ys (32-block quantization). Block 0 also emits xq/xs (quantized xn).
__global__ void __launch_bounds__(FT) gemv_w13_swiglu_kernel(const bf16* __restrict__ xn, int K, const uint8_t* __restrict__ W1, const uint8_t* __restrict__ S1,
                                                             const uint8_t* __restrict__ W3, const uint8_t* __restrict__ S3, int M, int I, float limit,
                                                             bf16* __restrict__ y, uint8_t* __restrict__ xq, uint8_t* __restrict__ xs) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* aq = smem;
  float* asv = reinterpret_cast<float*>(smem + (size_t)M * K);
  uint8_t* asc = reinterpret_cast<uint8_t*>(asv + M * (K / 32));
  __shared__ float outg[MMAX][OUT];
  __shared__ float outu[MMAX][OUT];
  quant_rows_smem(xn, K, M, K, aq, asv, asc);
  __syncthreads();
  if (blockIdx.x == 0 && xq) {
    const int nb = K / 32;
    for (int i = threadIdx.x; i < M * K; i += FT) xq[i] = aq[i];
    for (int i = threadIdx.x; i < M * nb; i += FT) xs[i] = asc[i];
  }
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, n0 = blockIdx.x * OUT;
  for (int j = 0; j < CPW; ++j) {
    const int c = warp * CPW + j;
    float acc[MMAX];
    warp_dot<true>(aq, asv, asc, K, W1, S1, M, K, n0 + c, lane, acc);
    if (lane == 0) for (int m = 0; m < M; ++m) outg[m][c] = acc[m];
    warp_dot<true>(aq, asv, asc, K, W3, S3, M, K, n0 + c, lane, acc);
    if (lane == 0) for (int m = 0; m < M; ++m) outu[m][c] = acc[m];
  }
  __syncthreads();
  if (threadIdx.x < M * OUT) {  // swiglu_route(rw = nullptr)
    const int m = threadIdx.x / OUT, c = threadIdx.x % OUT;
    float g = bf2f(f2bf(outg[m][c])), u = bf2f(f2bf(outu[m][c]));
    if (limit > 0.f) { u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); }
    y[(size_t)m * I + n0 + c] = f2bf(g / (1.f + expf(-g)) * u);
  }
}

// ⑥ shared expert, second half: acc[m,:] = bf16 value of w2(quant y) (written as fp32 — replaces memset + accum_bf16_rows). y is quantized in smem per block.
__global__ void __launch_bounds__(FT) gemv_w2_acc_kernel(const bf16* __restrict__ y, int I, const uint8_t* __restrict__ W2, const uint8_t* __restrict__ S2, int M,
                                                         int dim, float* __restrict__ acc) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* aq = smem;
  float* asv = reinterpret_cast<float*>(smem + (size_t)M * I);
  uint8_t* asc = reinterpret_cast<uint8_t*>(asv + M * (I / 32));
  __shared__ float outv[MMAX][OUT];
  quant_rows_smem(y, I, M, I, aq, asv, asc);
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, n0 = blockIdx.x * OUT;
  for (int j = 0; j < CPW; ++j) {
    const int c = warp * CPW + j;
    float a[MMAX];
    warp_dot<true>(aq, asv, asc, I, W2, S2, M, I, n0 + c, lane, a);
    if (lane == 0) for (int m = 0; m < M; ++m) outv[m][c] = a[m];
  }
  __syncthreads();
  if (threadIdx.x < M * OUT) { const int m = threadIdx.x / OUT, c = threadIdx.x % OUT; acc[(size_t)m * dim + n0 + c] = bf2f(f2bf(outv[m][c])); }
}

// ⑦ router: scores[m,e] = Σ_k f32(xn[m,k])·W[e,k] (warp = expert) → [last block] same selection as router_topk (one warp per row, serial argmax in lane 0)
__global__ void __launch_bounds__(FT) router_fused_kernel(const bf16* __restrict__ xn, int K, const float* __restrict__ W, int M, int E, const float* __restrict__ bias,
                                                          const float* __restrict__ bias_vl, const int8_t* __restrict__ is_image, int k, float route_scale,
                                                          float* __restrict__ scores, int32_t* __restrict__ ids, float* __restrict__ w, int* __restrict__ counter) {
  __shared__ float s[FW][512];
  __shared__ float sel[FW][512];
  __shared__ int flag;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int e = blockIdx.x * FW + warp;
  if (e < E) {
    const float* We = W + (size_t)e * K;
    for (int m = 0; m < M; ++m) {
      const bf16* xm = xn + (size_t)m * K;
      float d = 0.f;
      for (int kk = lane; kk < K; kk += 32) d = fmaf(bf2f(xm[kk]), We[kk], d);
      d = hive::cu::warp_allreduce(d);
      if (lane == 0) scores[(size_t)m * E + e] = d;
    }
  }
  if (!last_block(counter, &flag)) return;
  for (int m = warp; m < M; m += FW) {
    const float* b = (bias_vl && is_image && is_image[m]) ? bias_vl : bias;
    for (int ee = lane; ee < E; ee += 32) {
      const float x = scores[(size_t)m * E + ee];
      const float sp = x > 20.f ? x : log1pf(expf(x));
      s[warp][ee] = sqrtf(sp);
      sel[warp][ee] = s[warp][ee] + b[ee];
    }
    __syncwarp();
    if (lane == 0) {
      float wsum = 0.f;
      for (int j = 0; j < k; ++j) {
        int bi = -1; float bv = -FLT_MAX;
        for (int ee = 0; ee < E; ++ee) if (sel[warp][ee] > bv) { bv = sel[warp][ee]; bi = ee; }
        sel[warp][bi] = -FLT_MAX;
        ids[m * k + j] = bi;
        w[m * k + j] = s[warp][bi];
        wsum += s[warp][bi];
      }
      for (int j = 0; j < k; ++j) w[m * k + j] = w[m * k + j] / (wsum + 1e-20f) * route_scale;
    }
    __syncwarp();
  }
}

// ⑧ layer tail: h = hc_post(bf16(acc), post, comb) · pre_mix = pre_f  (f32_to_bf16 + hc_post + memcpy)
__global__ void hc_post_tail_kernel(const float* __restrict__ acc, const float* __restrict__ post, const float* __restrict__ comb, int M, int hc, int dim,
                                    bf16* __restrict__ h, const float* __restrict__ pre_f, float* __restrict__ pre_mix) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)M * hc) pre_mix[i] = pre_f[i];
  if (i >= (size_t)M * dim) return;
  const int m = (int)(i / dim), d = (int)(i % dim);
  float res[4];
  for (int j = 0; j < hc; ++j) res[j] = bf2f(h[((size_t)m * hc + j) * dim + d]);
  const float xv = bf2f(f2bf(acc[i]));
  for (int c = 0; c < hc; ++c) {
    float sum = post[m * hc + c] * xv;
    for (int j = 0; j < hc; ++j) sum += comb[((size_t)m * hc + j) * hc + c] * res[j];
    h[((size_t)m * hc + c) * dim + d] = f2bf(sum);
  }
}

inline int grid1d(size_t n, int block = 256) { return (int)((n + block - 1) / block); }
inline size_t smem_bytes(int M, int K) { return (size_t)M * K + (size_t)M * (K / 32) * 4 + (size_t)M * (K / 32) + 16; }
// Dynamic smem limit attribute: if static smem (output table, row buffers ≤ 4 KB) plus dynamic smem exceeds 48 KB, the launch
//   fails silently (measured with M=8 kv/w13 — the outputs were empty). The attribute is set from 44 KB dynamic upward
//   (static + dynamic ≤ 99 KB opt-in).
template <class Kern>
void ensure_smem(Kern kern, size_t bytes) {
  static size_t configured = 0;
  if (bytes + 4096 > 48 * 1024 && bytes > configured) {
    CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(bytes + 4096)));
    configured = bytes;
  }
}
inline void launch_check(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) return;
  throw std::runtime_error(std::string("fused launch failed: ").append(what).append(": ").append(cudaGetErrorString(e)));
}

}  // namespace

size_t fused_smem_max_k() { return 8192; }  // input K limit (M ≤ 8) — anything larger (e.g. main_proj K=15360) runs unfused

void fused_q_proj(const bf16* xn, int K, const uint8_t* wqa, const uint8_t* sqa, int Nqa, const bf16* q_norm, float eps, const uint8_t* wqb,
                  const uint8_t* sqb, int H, int D, int rd, const float2* freqs, const int32_t* pos, int M, bf16* qr, bf16* qrn, bf16* q, uint8_t* qrq,
                  uint8_t* qrs, int* counter, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= MMAX && K % 32 == 0 && Nqa % OUT == 0 && (H * D) % OUT == 0 && D % OUT == 0 && rd % 2 == 0 && (D - rd) % OUT == 0,
             "fused_q_proj shape");
  const size_t s1 = smem_bytes(M, K), s2 = smem_bytes(M, Nqa);
  ensure_smem(gemv_qa_kernel, s1);
  ensure_smem(gemv_qb_rope_kernel, s2);
  gemv_qa_kernel<<<Nqa / OUT, FT, s1, st>>>(xn, K, wqa, sqa, M, Nqa, q_norm, eps, qr, qrn, counter);
  launch_check("q_a");
  gemv_qb_rope_kernel<<<(H * D) / OUT, FT, s2, st>>>(qrn, Nqa, wqb, sqb, M, H * D, H, D, rd, freqs, pos, q, qrq, qrs);
  launch_check("q_b");
}

void fused_kv_proj(const bf16* xn, int K, const uint8_t* wkv, const uint8_t* skv, int D, const bf16* kv_norm, float eps, int rd, const float2* freqs,
                   const int32_t* pos, int M, bf16* kv, bf16* const* ring_ptrs, int win, int32_t* idx_out, int idx_stride, int* counter, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= MMAX && K % 32 == 0 && D % OUT == 0 && D <= 512 && rd / 2 <= FT && D % 32 == 0, "fused_kv_proj shape");
  const size_t s1 = smem_bytes(M, K);
  ensure_smem(gemv_kv_kernel, s1);
  gemv_kv_kernel<<<D / OUT, FT, s1, st>>>(xn, K, wkv, skv, M, D, kv_norm, eps, rd, freqs, pos, kv, ring_ptrs, win, idx_out, idx_stride, counter);
  launch_check("kv");
}

void fused_gemv_quantin(const bf16* A, int K, const uint8_t* B, const uint8_t* sb, int M, int N, bf16* C, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= MMAX && K % 32 == 0 && N % OUT == 0, "fused_gemv_quantin shape");
  const size_t s1 = smem_bytes(M, K);
  ensure_smem(gemv_quantin_kernel, s1);
  gemv_quantin_kernel<<<N / OUT, FT, s1, st>>>(A, K, B, sb, M, N, C);
  launch_check("quantin");
}

void fused_shared_experts(const bf16* xn, int K, const uint8_t* w1, const uint8_t* s1, const uint8_t* w3, const uint8_t* s3, const uint8_t* w2,
                          const uint8_t* s2, int M, int I, float limit, bf16* y, uint8_t* xq, uint8_t* xs, float* acc, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= MMAX && K % 32 == 0 && I % 32 == 0 && I % OUT == 0 && K % OUT == 0, "fused_shared_experts shape");
  const size_t sm = smem_bytes(M, K), sm2 = smem_bytes(M, I);
  ensure_smem(gemv_w13_swiglu_kernel, sm);
  ensure_smem(gemv_w2_acc_kernel, sm2);
  gemv_w13_swiglu_kernel<<<I / OUT, FT, sm, st>>>(xn, K, w1, s1, w3, s3, M, I, limit, y, xq, xs);
  launch_check("w13");
  gemv_w2_acc_kernel<<<K / OUT, FT, sm2, st>>>(y, I, w2, s2, M, K, acc);
  launch_check("w2");
}

void fused_router(const bf16* xn, int K, const float* gate_w, int M, int E, const float* bias, const float* bias_vl, const int8_t* is_image, int k,
                  float route_scale, float* scores, int32_t* ids, float* w, int* counter, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= MMAX && E <= 512 && k <= 16, "fused_router shape");
  router_fused_kernel<<<(E + FW - 1) / FW, FT, 0, st>>>(xn, K, gate_w, M, E, bias, bias_vl, is_image, k, route_scale, scores, ids, w, counter);
  launch_check("router");
}

void hc_post_tail(const float* acc, const float* post, const float* comb, int M, int hc, int dim, bf16* h, const float* pre_f, float* pre_mix, cudaStream_t st) {
  hc_post_tail_kernel<<<grid1d((size_t)M * dim), 256, 0, st>>>(acc, post, comb, M, hc, dim, h, pre_f, pre_mix);
}

}  // namespace hive::k
