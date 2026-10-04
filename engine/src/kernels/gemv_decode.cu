// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// D2b HIVE_DECODE_GEMV2 — bandwidth-oriented versions of the decode (M ≤ 8) dense GEMVs, router and hc mixing. Contract and switches are in the include/hive/gemv_decode.h header comment.
//
// Background (analysis with the service configuration, decode comp layer M=1): ≈178 MB of weights per layer (wq_b 43 · wo_b 43 ·
//   wo_a 34.5 · shared expert 35.4 · router 7.9 · wq_a 6.6 · hc 3.9 · wkv 2.7) → ≈99 µs at 1.8 TB/s, but measured 311 µs. The dense
//   GEMVs reached ≈1 TB/s, the router argmax was a 2,304-iteration serial loop in lane 0, and hc mixing used 26 blocks.
//
// Why they were slow (structure confirmed by reading the code):
//   ① The GEMVs in fused.cu quantize the input per block (quant_rows_smem) → __syncthreads → and only then read the weights. Weight
//      loads queue up behind the quantization latency.
//   ② The `for (b = lane; b < nb; b += 32)` loop in warp_dot is not unrolled, so each lane has only 32 B (one block) in flight at a
//      time — wo_b (K 8192 → 8 blocks per lane) needs 8 round trips. wo_a (gemm.cu gemv_bf16_fp8_grouped_kernel) has the same loop
//      (4 blocks per lane).
//   ③ Router: warp = expert, lanes load kk = lane + 32j sequentially as fp32 scalars · 48 blocks (E 384 / 8 warps) → only 48 of
//      188 SMs are used.
// This version (numerics unchanged):
//   ① Each warp loads all of its lanes' weight share for its columns (NBL blocks × 32 B + scales) into registers **before
//      quantization** (ld.global.nc.L1::no_allocate.v4). With PDL the loads are issued before griddepcontrol.wait → they overlap
//      the tail of the previous kernel.
//   ② The dot product uses the same formula as the default warp_dot<true>: a lane's blocks b = lane, lane+32, … in ascending order
//      · per block d = fmaf(a_i, w_i, d) (i 0..31) → acc[m] = fmaf(d, sa·sb, acc[m]) → xor 16,8,4,2,1. Only the column → warp
//      assignment may differ (CPW columns per warp); each column is still computed entirely by one warp.
//   ③ Router: block = 2 warps (2 experts) → 192 blocks. Each lane's gate_w share is preloaded 32 at a time (the next batch is
//      issued first) and fmaf'd in the same kk order for each row.
//      top-k: per row one warp takes, per round, the lane-local max (ascending scan · `>` — equal values keep the earlier index)
//      → xor shuffle (larger value; on ties the smaller index). Updating only on `>` and starting from -FLT_MAX means NaN and
//      values ≤ -FLT_MAX are never picked — the same choice as the default serial lane-0 scan (first maximum).
//      The weight sum (wsum) and normalization are done by lane 0 in j order, as before.
//   ④ hc mixing: each thread's share of hcdim/256 elements is loaded in batches of 16, then fmaf'd in the same order (same
//      splitting and reduction partitioning — block (m, j) · 256 threads · xor · serial sum of red[0..7]).
// The only case that may not be bit-identical: a router row with fewer than k non-NaN candidates (only when activations are NaN)
//   — the default kernel then read s[warp][-1] (out of bounds) with bi = -1, while this version writes id -1 · weight 0.
#include <cuda_fp8.h>

#include <atomic>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include "hive/gemv_decode.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"
#include "hive/warp_reduce.cuh"
#include "hc_decode_fused.cuh"  // E3 HIVE_HC_DECODE_FUSED

namespace hive::k {

namespace {

// Switch rule = same as env_on in runtime.cpp (unset, "" or "0" = off). The runtime one lives in an anonymous namespace, so it is duplicated here (same reason as in expert_store.cpp).
inline bool gd_env_on(const char* name) { const char* v = getenv(name); return v && *v && strcmp(v, "0") != 0; }
std::atomic<int> g_force_pdl{-1};

constexpr int FT = 256;        // block = 8 warps
constexpr int FW = FT / 32;
constexpr int MMAX = 8;
constexpr float kFp8MaxInv = 1.0f / 448.0f;

// ---- PDL(griddepcontrol, sm_90+) --------------------------------------------------------------------------------------------
// wait: if this grid was launched with a programmatic dependency, wait until the previous grid completes and its memory is visible. Otherwise returns immediately (per the docs: a no-op without a preceding grid).
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
// launch_dependents: this CTA allows early launch of the next grid (the next grid starts once every CTA has called this or exited, and waits for our completion in its own wait).
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

__device__ __forceinline__ uint4 ld_stream16(const void* p) {  // weight streaming: read-only · no L1 allocation (no reuse)
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ float ld_stream_f32(const float* p) {
  float r;
  asm volatile("ld.global.nc.L1::no_allocate.f32 %0, [%1];" : "=f"(r) : "l"(p));
  return r;
}

// ---- Same device formulas as fused.cu (copied — bit identity is the contract, so the formulas must not change) --------------------------------------------------
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
// Same formula as fused.cu quant_rows_smem
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
// Same reduction as fused.cu rmsnorm_row_256
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
// Same as fused.cu last_block (the last block resets the counter to 0 — safe for graph replay)
__device__ __forceinline__ bool last_block(int* counter, int* flag_smem) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) *flag_smem = (atomicAdd(counter, 1) == (int)gridDim.x - 1);
  __syncthreads();
  const bool last = *flag_smem != 0;
  if (last) { __threadfence(); if (threadIdx.x == 0) *counter = 0; }
  return last;
}

// ---- Weight register preload + dot product (same order as warp_dot<true>) -----------------------------------------------------------------
template <int NBL>
struct WReg {
  uint4 lo[NBL], hi[NBL];
  uint32_t s[NBL];
};
template <int NBL>
__device__ __forceinline__ void wload(const uint8_t* __restrict__ B, const uint8_t* __restrict__ sb, int K, int nb, int n, int lane, WReg<NBL>& r) {
#pragma unroll
  for (int t = 0; t < NBL; ++t) {
    const int b = lane + 32 * t;
    if (b < nb) {
      const uint8_t* pb = B + (size_t)n * K + b * 32;
      r.lo[t] = ld_stream16(pb);
      r.hi[t] = ld_stream16(pb + 16);
      r.s[t] = sb[(size_t)n * nb + b];
    }
  }
}
template <int NBL>
__device__ __forceinline__ void wdot(const uint8_t* __restrict__ aq, const float* __restrict__ asv, int K, int nb, const WReg<NBL>& r, int M, int lane,
                                     float (&acc)[MMAX]) {
#pragma unroll
  for (int m = 0; m < MMAX; ++m) acc[m] = 0.f;
#pragma unroll
  for (int t = 0; t < NBL; ++t) {
    const int b = lane + 32 * t;
    if (b < nb) {
      float w[32];
      decode8(r.lo[t], r.hi[t], w);
      const float sbv = e8m0_fast_f((uint8_t)r.s[t]);
#pragma unroll
      for (int m = 0; m < MMAX; ++m) {
        if (m < M) {
          const uint8_t* pa = aq + (size_t)m * K + b * 32;
          float av[32];
          decode8(*reinterpret_cast<const uint4*>(pa), *reinterpret_cast<const uint4*>(pa + 16), av);
          float d = 0.f;
#pragma unroll
          for (int i = 0; i < 32; ++i) d = fmaf(av[i], w[i], d);
          const float sav = asv[m * nb + b];
          acc[m] = fmaf(d, sav * sbv, acc[m]);
        }
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

// ---- One template for GEMVs with quantized input (only the epilogue differs) ------------------------------------------------------------------------------------
enum Epi { EPI_QA = 0, EPI_QB = 1, EPI_QUANTIN = 2, EPI_W13 = 3, EPI_W2 = 4 };
struct G2Args {
  const bf16* A; int K;                      // input [M, K] bf16 (row stride = K)
  const uint8_t* B; const uint8_t* sb;       // weights [N, K] e4m3 + [N, K/32] e8m0
  const uint8_t* B2; const uint8_t* sb2;     // w3 of w13
  int M, N;
  bf16* out;                                 // QA: qr · QB: q · QUANTIN: C · W13: y
  float* out_f;                              // W2: acc
  const bf16* norm_w; float eps; bf16* out2; int* counter;               // QA: q_norm · qrn · counter
  int D, rd; const float2* freqs; const int32_t* pos; uint8_t* side_q; uint8_t* side_s;  // QB (RoPE · quantized qrn) · W13 (xq/xs)
  float limit;                               // W13
};

template <int EPI, int NBL, int CPW>
__global__ void __launch_bounds__(FT) g2_kernel(const G2Args a) {
  constexpr int OUTB = FW * CPW;  // output columns per block
  constexpr int NMAT = EPI == EPI_W13 ? 2 : 1;
  extern __shared__ __align__(16) uint8_t smem[];
  const int M = a.M, K = a.K, nb = K / 32;
  uint8_t* aq = smem;
  float* asv = reinterpret_cast<float*>(smem + (size_t)M * K);
  uint8_t* asc = reinterpret_cast<uint8_t*>(asv + M * nb);
  __shared__ float outv[NMAT][MMAX][OUTB];
  __shared__ float red[8];
  __shared__ int flag;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, n0 = blockIdx.x * OUTB;
  // ① weight preload (independent of the previous kernel — with PDL it overlaps the previous kernel's tail)
  WReg<NBL> wr[NMAT][CPW];
#pragma unroll
  for (int j = 0; j < CPW; ++j) {
    const int n = n0 + warp * CPW + j;
    wload<NBL>(a.B, a.sb, K, nb, n, lane, wr[0][j]);
    if constexpr (NMAT == 2) wload<NBL>(a.B2, a.sb2, K, nb, n, lane, wr[NMAT - 1][j]);
  }
  pdl_wait();     // from here on read the previous kernel's output (activations) and write outputs
  pdl_trigger();
  // ② input quantization (same formula as fused.cu) + side outputs (block 0)
  quant_rows_smem(a.A, K, M, K, aq, asv, asc);
  __syncthreads();
  if constexpr (EPI == EPI_QB || EPI == EPI_W13) {
    if (blockIdx.x == 0 && a.side_q) {
      for (int i = threadIdx.x; i < M * K; i += FT) a.side_q[i] = aq[i];
      for (int i = threadIdx.x; i < M * nb; i += FT) a.side_s[i] = asc[i];
    }
  }
  // ③ dot product per column (default warp_dot<true> order)
#pragma unroll
  for (int j = 0; j < CPW; ++j) {
    const int c = warp * CPW + j;
#pragma unroll
    for (int mt = 0; mt < NMAT; ++mt) {
      float acc[MMAX];
      wdot<NBL>(aq, asv, K, nb, wr[mt][j], M, lane, acc);
      if (lane == 0) for (int m = 0; m < M; ++m) outv[mt][m][c] = acc[m];
    }
  }
  __syncthreads();
  // ④ epilogue (exactly the formulas of the corresponding fused.cu kernels)
  const int N = a.N;
  if constexpr (EPI == EPI_QA) {
    if (threadIdx.x < M * OUTB) { const int m = threadIdx.x / OUTB, c = threadIdx.x % OUTB; a.out[(size_t)m * N + n0 + c] = f2bf(outv[0][m][c]); }
    if (!last_block(a.counter, &flag)) return;
    for (int m = 0; m < M; ++m) rmsnorm_row_256(a.out + (size_t)m * N, a.norm_w, a.eps, N, a.out2 + (size_t)m * N, red);
  } else if constexpr (EPI == EPI_QB) {
    if (threadIdx.x >= M * OUTB) return;
    const int m = threadIdx.x / OUTB, c = threadIdx.x % OUTB, n = n0 + c;
    const int D = a.D, rd = a.rd;
    const int dpos = n % D;
    if (dpos < D - rd) { a.out[(size_t)m * N + n] = f2bf(outv[0][m][c]); return; }
    if (c & 1) return;  // pair (2p, 2p+1): the even column writes both (block boundaries are multiples of OUTB, which is even, so pairs close within a block)
    const int p = (dpos - (D - rd)) >> 1;
    const float2 f = a.freqs[(size_t)a.pos[m] * (rd / 2) + p];
    const float x0 = bf2f(f2bf(outv[0][m][c])), x1 = bf2f(f2bf(outv[0][m][c + 1]));
    a.out[(size_t)m * N + n] = f2bf(x0 * f.x - x1 * f.y);
    a.out[(size_t)m * N + n + 1] = f2bf(x0 * f.y + x1 * f.x);
  } else if constexpr (EPI == EPI_QUANTIN) {
    if (threadIdx.x < M * OUTB) { const int m = threadIdx.x / OUTB, c = threadIdx.x % OUTB; a.out[(size_t)m * N + n0 + c] = f2bf(outv[0][m][c]); }
  } else if constexpr (EPI == EPI_W13) {
    if (threadIdx.x < M * OUTB) {  // swiglu_route(rw = nullptr)
      const int m = threadIdx.x / OUTB, c = threadIdx.x % OUTB;
      float g = bf2f(f2bf(outv[0][m][c])), u = bf2f(f2bf(outv[NMAT - 1][m][c]));
      const float limit = a.limit;
      if (limit > 0.f) { u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); }
      a.out[(size_t)m * N + n0 + c] = f2bf(g / (1.f + expf(-g)) * u);
    }
  } else {  // EPI_W2
    if (threadIdx.x < M * OUTB) { const int m = threadIdx.x / OUTB, c = threadIdx.x % OUTB; a.out_f[(size_t)m * N + n0 + c] = bf2f(f2bf(outv[0][m][c])); }
  }
}

// ---- wo_a: bf16 activations × fp8 weights (grouped) — same order as gemm.cu gemv_bf16_fp8_grouped_kernel<8> -----------------------------------------
template <int NBL>
__global__ void __launch_bounds__(FT) woa2_kernel(const bf16* __restrict__ A, int lda, const uint8_t* __restrict__ B, const uint8_t* __restrict__ sb, int M,
                                                  int G, int R, int K, bf16* __restrict__ C) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const bool valid = warp < G * R;
  const int n = valid ? warp : 0, g = n / R;
  const int nb = K / 32;
  WReg<NBL> r;
  if (valid) wload<NBL>(B, sb, K, nb, n, lane, r);
  pdl_wait();  // every CTA passes the wait (the early return below comes after it)
  pdl_trigger();
  if (!valid) return;
  float acc[MMAX];
#pragma unroll
  for (int m = 0; m < MMAX; ++m) acc[m] = 0.f;
#pragma unroll
  for (int t = 0; t < NBL; ++t) {
    const int b = lane + 32 * t;
    if (b < nb) {
      float w[32];
      decode8(r.lo[t], r.hi[t], w);
      const float sbv = e8m0_fast_f((uint8_t)r.s[t]);
#pragma unroll
      for (int m = 0; m < MMAX; ++m) {
        if (m < M) {
          const uint4* pa = reinterpret_cast<const uint4*>(A + (size_t)m * lda + (size_t)g * K + b * 32);
          float d = 0.f;
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            const uint4 raw = pa[v];
            const uint32_t x[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              d = fmaf(__uint_as_float(x[i] << 16), w[v * 8 + 2 * i], d);
              d = fmaf(__uint_as_float(x[i] & 0xFFFF0000u), w[v * 8 + 2 * i + 1], d);
            }
          }
          acc[m] = fmaf(d, sbv, acc[m]);
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MMAX; ++m) {
    float v = acc[m];
    v = hive::cu::warp_allreduce(v);
    if (lane == 0 && m < M) C[(size_t)m * (G * R) + n] = f2bf(v);
  }
}

// ---- Router ---------------------------------------------------------------------------------------------------------------------
constexpr int RW = 2;        // block = 2 warps (2 experts) — E 384 → 192 blocks (all SMs)
constexpr int RU = 32;       // per-lane gate_w batch (32 scalars = 4 KB in flight per warp)
__global__ void __launch_bounds__(RW * 32) router2_kernel(const bf16* __restrict__ xn, int K, const float* __restrict__ W, int M, int E,
                                                          const float* __restrict__ bias, const float* __restrict__ bias_vl, const int8_t* __restrict__ is_image,
                                                          int k, float route_scale, float* __restrict__ scores, int32_t* __restrict__ ids, float* __restrict__ w,
                                                          int* __restrict__ counter) {
  __shared__ float s[RW][512];
  __shared__ float sel[RW][512];
  __shared__ int flag;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int e = blockIdx.x * RW + warp;
  const bool valid = e < E;
  const float* We = W + (size_t)(valid ? e : 0) * K;
  float cur[RU];
#pragma unroll
  for (int u = 0; u < RU; ++u) { const int kk = lane + 32 * u; cur[u] = (valid && kk < K) ? ld_stream_f32(We + kk) : 0.f; }
  pdl_wait();
  pdl_trigger();
  if (valid) {
    float d[MMAX];
#pragma unroll
    for (int m = 0; m < MMAX; ++m) d[m] = 0.f;
    for (int k0 = 0; k0 < K; k0 += 32 * RU) {
      float nxt[RU];
      const int k1 = k0 + 32 * RU;
#pragma unroll
      for (int u = 0; u < RU; ++u) { const int kk = k1 + lane + 32 * u; nxt[u] = kk < K ? ld_stream_f32(We + kk) : 0.f; }
#pragma unroll
      for (int m = 0; m < MMAX; ++m) {
        if (m < M) {
          const bf16* xm = xn + (size_t)m * K;
          float dm = d[m];
#pragma unroll
          for (int u = 0; u < RU; ++u) {  // default: d = fmaf(x, w, d) in the order kk = lane, lane+32, … — ascending kk within a batch as well
            const int kk = k0 + lane + 32 * u;
            if (kk < K) dm = fmaf(bf2f(xm[kk]), cur[u], dm);
          }
          d[m] = dm;
        }
      }
#pragma unroll
      for (int u = 0; u < RU; ++u) cur[u] = nxt[u];
    }
#pragma unroll
    for (int m = 0; m < MMAX; ++m) {
      if (m < M) {
        float v = d[m];
        v = hive::cu::warp_allreduce(v);
        if (lane == 0) scores[(size_t)m * E + e] = v;
      }
    }
  }
  if (!last_block(counter, &flag)) return;
  for (int m = warp; m < M; m += RW) {
    const float* b = (bias_vl && is_image && is_image[m]) ? bias_vl : bias;
    for (int ee = lane; ee < E; ee += 32) {
      const float x = scores[(size_t)m * E + ee];
      const float sp = x > 20.f ? x : log1pf(expf(x));
      s[warp][ee] = sqrtf(sp);
      sel[warp][ee] = s[warp][ee] + b[ee];
    }
    __syncwarp();
    float wsum = 0.f;
    for (int j = 0; j < k; ++j) {
      float bv = -FLT_MAX; int bi = -1;
      for (int ee = lane; ee < E; ee += 32) { const float v = sel[warp][ee]; if (v > bv) { bv = v; bi = ee; } }
#pragma unroll
      for (int o = 16; o >= 1; o /= 2) {
        const float ov = __shfl_xor_sync(0xffffffff, bv, o);
        const int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
      }
      if (lane == 0) {
        const float sv = bi >= 0 ? s[warp][bi] : 0.f;
        if (bi >= 0) sel[warp][bi] = -FLT_MAX;
        ids[m * k + j] = bi;
        w[m * k + j] = sv;
        wsum += sv;
      }
      __syncwarp();
    }
    if (lane == 0) for (int j = 0; j < k; ++j) w[m * k + j] = w[m * k + j] / (wsum + 1e-20f) * route_scale;
    __syncwarp();
  }
}

// ---- hc mixing + pre+norm (same splitting and reduction as model_kernels.cu hc_mix_pre_norm_kernel<4>) -----------------------------------------------
constexpr int HU = 16;  // per-thread load batch
__device__ __forceinline__ void hc_mix_body2(const bf16* __restrict__ h, const float* __restrict__ W, int hcdim, int mix_hc, float eps,
                                             float* __restrict__ mixes, float* __restrict__ rsq, int m, int j, float* red) {
  const bf16* p = h + (size_t)m * hcdim;
  const float* wr = W + (size_t)(j < mix_hc ? j : 0) * hcdim;
  float wpre[HU];  // first batch of weights (independent of the previous kernel) — loaded before the wait
#pragma unroll
  for (int u = 0; u < HU; ++u) { const int i = threadIdx.x + 256 * u; wpre[u] = (j < mix_hc && i < hcdim) ? ld_stream_f32(wr + i) : 0.f; }
  pdl_wait();
  pdl_trigger();
  float acc = 0.f;
  if (j < mix_hc) {
    int i0 = 0;
    for (; i0 < hcdim; i0 += 256 * HU) {  // default: acc = fmaf(h, w, acc) for i = tid, tid+256, … — same order within a batch
      float hv[HU], wv[HU];
#pragma unroll
      for (int u = 0; u < HU; ++u) {
        const int i = i0 + threadIdx.x + 256 * u;
        hv[u] = i < hcdim ? bf2f(p[i]) : 0.f;
        wv[u] = i0 == 0 ? wpre[u] : (i < hcdim ? ld_stream_f32(wr + i) : 0.f);
      }
#pragma unroll
      for (int u = 0; u < HU; ++u) { const int i = i0 + threadIdx.x + 256 * u; if (i < hcdim) acc = fmaf(hv[u], wv[u], acc); }
    }
  } else {
    for (int i0 = 0; i0 < hcdim; i0 += 256 * HU) {
      float hv[HU];
#pragma unroll
      for (int u = 0; u < HU; ++u) { const int i = i0 + threadIdx.x + 256 * u; hv[u] = i < hcdim ? bf2f(p[i]) : 0.f; }
#pragma unroll
      for (int u = 0; u < HU; ++u) { const int i = i0 + threadIdx.x + 256 * u; if (i < hcdim) { float v = hv[u]; acc += v * v; } }  // same formula as the default
    }
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
constexpr int HPU = 10;  // pre+norm: per-thread batch of d (dim 5120 → 20 = 2 batches)
template <int HC>
__device__ __forceinline__ void hc_pre_norm_body2(const bf16* __restrict__ h, const float* __restrict__ pre_in, int dim, const bf16* __restrict__ norm_w,
                                                  float norm_eps, bf16* __restrict__ x, bf16* __restrict__ xn, int m, float* red, float* pin) {
  constexpr int hc = HC;
  pdl_wait();
  pdl_trigger();
  if (threadIdx.x < hc) pin[threadIdx.x] = pre_in[m * hc + threadIdx.x];
  __syncthreads();
  const bf16* hm = h + (size_t)m * hc * dim;
  float ss = 0.f;
  for (int d0 = threadIdx.x; d0 < dim; d0 += 256 * HPU) {
    float hv[HPU][HC];
#pragma unroll
    for (int u = 0; u < HPU; ++u) {
      const int d = d0 + 256 * u;
#pragma unroll
      for (int c = 0; c < hc; ++c) hv[u][c] = d < dim ? bf2f(hm[(size_t)c * dim + d]) : 0.f;
    }
#pragma unroll
    for (int u = 0; u < HPU; ++u) {
      const int d = d0 + 256 * u;
      if (d < dim) {  // default: s += pin[c]·h (c order) → bf16 → ss += v·v (d order)
        float s = 0.f;
        for (int c = 0; c < hc; ++c) s += pin[c] * hv[u][c];
        const bf16 xv = f2bf(s);
        x[(size_t)m * dim + d] = xv;
        const float v = bf2f(xv);
        ss += v * v;
      }
    }
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
__global__ void __launch_bounds__(256) hc_mix_pre_norm2_kernel(const bf16* __restrict__ h, const float* __restrict__ W, int hcdim, int mix_hc, float eps,
                                                               float* __restrict__ mixes, float* __restrict__ rsq, const float* __restrict__ pre_in, int dim,
                                                               const bf16* __restrict__ norm_w, float norm_eps, bf16* __restrict__ x, bf16* __restrict__ xn) {
  __shared__ float red[8];
  __shared__ float pin[HC];
  if ((int)blockIdx.y <= mix_hc) hc_mix_body2(h, W, hcdim, mix_hc, eps, mixes, rsq, blockIdx.x, blockIdx.y, red);
  else hc_pre_norm_body2<HC>(h, pre_in, dim, norm_w, norm_eps, x, xn, blockIdx.x, red, pin);
}
// E3 HIVE_HC_DECODE_FUSED: the kernel above + the sinkhorn tail (hc_decode_fused.cuh — last mixing block). The tail comes after
//   the body's griddepcontrol.wait, so even when launched with PDL it does not overlap the previous grid. mixes·rsq·x·xn are
//   bit-identical to hc_mix_pre_norm2 and pre·post·comb to sinkhorn_kernel (test_hc_decode).
template <int HC>
__global__ void __launch_bounds__(256) hc_mix_pre_norm2_sk_kernel(const bf16* __restrict__ h, const float* __restrict__ W, int hcdim, int mix_hc, float eps,
                                                                  float* mixes, float* rsq, const float* __restrict__ pre_in, int dim,
                                                                  const bf16* __restrict__ norm_w, float norm_eps, bf16* __restrict__ x, bf16* __restrict__ xn,
                                                                  const float* __restrict__ scale, const float* __restrict__ base, int M, int iters, float hc_eps,
                                                                  float* __restrict__ pre_out, float* __restrict__ post, float* __restrict__ comb, int* cnt) {
  __shared__ float red[8];
  __shared__ float pin[HC];
  __shared__ int last;
  if ((int)blockIdx.y <= mix_hc) {
    hc_mix_body2(h, W, hcdim, mix_hc, eps, mixes, rsq, blockIdx.x, blockIdx.y, red);
    hcdf::sk_tail<HC>(mixes, rsq, scale, base, M, iters, hc_eps, pre_out, post, comb, cnt, M * (mix_hc + 1), &last);
  } else {
    hc_pre_norm_body2<HC>(h, pre_in, dim, norm_w, norm_eps, x, xn, blockIdx.x, red, pin);
  }
}

// ---- Launch (PDL optional) ------------------------------------------------------------------------------------------------------------
std::atomic<int> g_pdl_broken{0};  // once a PDL launch has been rejected, all later launches are plain (absorbed — no rejection)
template <class... KArgs, class... Args>
void launch(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, const char* what, const Args&... args) {
  const bool pdl = decode_pdl_on() && !g_pdl_broken.load(std::memory_order_relaxed);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = pdl ? at : nullptr;
  cfg.numAttrs = pdl ? 1 : 0;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern, args...);
  if (e != cudaSuccess && pdl) {  // if the driver or capture does not accept the PDL attribute, fall back to a plain launch (reported once)
    (void)cudaGetLastError();
    if (!g_pdl_broken.exchange(1)) fprintf(stderr, "[gemv2] PDL launch refused (%s: %s) — plain launches from now on\n", what, cudaGetErrorString(e));
    cfg.attrs = nullptr; cfg.numAttrs = 0;
    e = cudaLaunchKernelEx(&cfg, kern, args...);
  }
  if (e != cudaSuccess) throw std::runtime_error(std::string("gemv2 launch failed: ") + what + ": " + cudaGetErrorString(e));
}

inline size_t smem_bytes(int M, int K) { return (size_t)M * K + (size_t)M * (K / 32) * 4 + (size_t)M * (K / 32) + 16; }
// Dynamic smem limit attribute (per instance — fused.cu ensure_smem uses a static per kernel type, so instances with the same signature would share it)
template <int EPI, int NBL, int CPW>
void g2_ensure_smem(size_t bytes) {
  static std::atomic<size_t> configured{0};
  static std::mutex mu;
  if (bytes + 4096 <= 48 * 1024 || bytes <= configured.load()) return;
  const std::lock_guard<std::mutex> lk(mu);
  if (bytes <= configured.load()) return;
  CUDA_CHECK(cudaFuncSetAttribute(g2_kernel<EPI, NBL, CPW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(bytes + 4096)));
  configured.store(bytes);
}
template <int EPI, int NBL>
void g2_launch_nbl(const G2Args& a, cudaStream_t st, const char* what) {
  constexpr int CPW = NBL <= 2 ? 2 : 1;  // with a small per-lane share use 2 columns per warp (to balance the preload bytes)
  const size_t sm = smem_bytes(a.M, a.K);
  g2_ensure_smem<EPI, NBL, CPW>(sm);
  launch(g2_kernel<EPI, NBL, CPW>, dim3(a.N / (FW * CPW)), dim3(FT), sm, st, what, a);
}
// Whether this version accepts the shape: per-lane block count NBL = ceil(K/32/32) ∈ {2,3,4,5,8} · N a multiple of the block column count (8·CPW) · smem ≤ 99 KB. Otherwise the caller uses the default function.
inline int g2_nbl(int K) { return (K / 32 + 31) / 32; }
inline int g2_outb(int K) { return FW * (g2_nbl(K) <= 2 ? 2 : 1); }
inline bool g2_ok(int M, int K, int N) {
  const int nbl = g2_nbl(K);
  return M >= 1 && M <= MMAX && K % 32 == 0 && (nbl == 2 || nbl == 3 || nbl == 4 || nbl == 5 || nbl == 8) && N % g2_outb(K) == 0 &&
         smem_bytes(M, K) + 4096 <= 99 * 1024;
}
template <int EPI>
void g2_launch(const G2Args& a, cudaStream_t st, const char* what) {  // the caller has checked g2_ok
  switch (g2_nbl(a.K)) {
    case 2: g2_launch_nbl<EPI, 2>(a, st, what); break;
    case 3: g2_launch_nbl<EPI, 3>(a, st, what); break;
    case 4: g2_launch_nbl<EPI, 4>(a, st, what); break;
    case 5: g2_launch_nbl<EPI, 5>(a, st, what); break;
    default: g2_launch_nbl<EPI, 8>(a, st, what); break;
  }
}
G2Args g2_args(const bf16* A, int K, const uint8_t* B, const uint8_t* sb, int M, int N) {
  G2Args a = {};
  a.A = A; a.K = K; a.B = B; a.sb = sb; a.M = M; a.N = N;
  return a;
}

}  // namespace

bool decode_gemv2_on() { static const bool on = gd_env_on("HIVE_DECODE_GEMV2"); return on; }
bool decode_pdl_on() {
  const int f = g_force_pdl.load(std::memory_order_relaxed);
  if (f >= 0) return f != 0;
  static const bool on = gd_env_on("HIVE_DECODE_PDL");
  return on;
}
void gemv2_force_pdl(int mode) { g_force_pdl.store(mode); g_pdl_broken.store(0); }

void gemv2_q_proj(const bf16* xn, int K, const uint8_t* wqa, const uint8_t* sqa, int Nqa, const bf16* q_norm, float eps, const uint8_t* wqb,
                  const uint8_t* sqb, int H, int D, int rd, const float2* freqs, const int32_t* pos, int M, bf16* qr, bf16* qrn, bf16* q, uint8_t* qrq,
                  uint8_t* qrs, int* counter, cudaStream_t st) {
  // shape condition = that of fused_q_proj + this version's templates (both kernels must fit for this version — D·(D−rd) must be a multiple of the block column count so RoPE pairs close within a block)
  const int outb_b = g2_outb(Nqa);
  const bool ok = g2_ok(M, K, Nqa) && g2_ok(M, Nqa, H * D) && D % outb_b == 0 && rd % 2 == 0 && (D - rd) % outb_b == 0;
  if (!ok) { fused_q_proj(xn, K, wqa, sqa, Nqa, q_norm, eps, wqb, sqb, H, D, rd, freqs, pos, M, qr, qrn, q, qrq, qrs, counter, st); return; }
  G2Args a = g2_args(xn, K, wqa, sqa, M, Nqa);
  a.out = qr; a.norm_w = q_norm; a.eps = eps; a.out2 = qrn; a.counter = counter;
  g2_launch<EPI_QA>(a, st, "q_a");
  G2Args b = g2_args(qrn, Nqa, wqb, sqb, M, H * D);
  b.out = q; b.D = D; b.rd = rd; b.freqs = freqs; b.pos = pos; b.side_q = qrq; b.side_s = qrs;
  g2_launch<EPI_QB>(b, st, "q_b");
}

void gemv2_quantin(const bf16* A, int K, const uint8_t* B, const uint8_t* sb, int M, int N, bf16* C, cudaStream_t st) {
  G2Args a = g2_args(A, K, B, sb, M, N);
  a.out = C;
  if (!g2_ok(M, K, N)) { fused_gemv_quantin(A, K, B, sb, M, N, C, st); return; }
  g2_launch<EPI_QUANTIN>(a, st, "quantin");
}

void gemv2_shared_experts(const bf16* xn, int K, const uint8_t* w1, const uint8_t* s1, const uint8_t* w3, const uint8_t* s3, const uint8_t* w2,
                          const uint8_t* s2, int M, int I, float limit, bf16* y, uint8_t* xq, uint8_t* xs, float* acc, cudaStream_t st) {
  const bool ok = g2_ok(M, K, I) && g2_ok(M, I, K);
  if (!ok) { fused_shared_experts(xn, K, w1, s1, w3, s3, w2, s2, M, I, limit, y, xq, xs, acc, st); return; }
  G2Args a = g2_args(xn, K, w1, s1, M, I);
  a.B2 = w3; a.sb2 = s3; a.out = y; a.limit = limit; a.side_q = xq; a.side_s = xs;
  g2_launch<EPI_W13>(a, st, "w13");
  G2Args b = g2_args(y, I, w2, s2, M, K);
  b.out_f = acc;
  g2_launch<EPI_W2>(b, st, "w2");
}

void gemv2_router(const bf16* xn, int K, const float* gate_w, int M, int E, const float* bias, const float* bias_vl, const int8_t* is_image, int k,
                  float route_scale, float* scores, int32_t* ids, float* w, int* counter, cudaStream_t st) {
  if (!(M >= 1 && M <= MMAX && E <= 512 && k <= 16)) {
    fused_router(xn, K, gate_w, M, E, bias, bias_vl, is_image, k, route_scale, scores, ids, w, counter, st);
    return;
  }
  launch(router2_kernel, dim3((E + RW - 1) / RW), dim3(RW * 32), 0, st, "router", xn, K, gate_w, M, E, bias, bias_vl, is_image, k, route_scale, scores, ids,
         w, counter);
}

void gemv2_bf16_fp8_grouped(const bf16* A, int lda, const uint8_t* B, const uint8_t* sb, int M, int G, int R, int sub, bf16* C, cudaStream_t st) {
  const int nbl = (sub / 32 + 31) / 32;
  if (!(M >= 1 && M <= MMAX && sub % 32 == 0) || (nbl != 2 && nbl != 4 && nbl != 8)) { gemv_bf16_fp8_grouped(A, lda, B, sb, M, G, R, sub, C, st); return; }
  const int warps = G * R, blocks = (warps * 32 + FT - 1) / FT;
  switch (nbl) {
    case 2: launch(woa2_kernel<2>, dim3(blocks), dim3(FT), 0, st, "wo_a", A, lda, B, sb, M, G, R, sub, C); break;
    case 4: launch(woa2_kernel<4>, dim3(blocks), dim3(FT), 0, st, "wo_a", A, lda, B, sb, M, G, R, sub, C); break;
    default: launch(woa2_kernel<8>, dim3(blocks), dim3(FT), 0, st, "wo_a", A, lda, B, sb, M, G, R, sub, C); break;
  }
}

void hc_mix_pre_norm2(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, const float* pre_in, int hc, int dim,
                      const bf16* norm_w, float norm_eps, bf16* x, bf16* xn, cudaStream_t st) {
  if (!(hc == 4 && hcdim == hc * dim)) { hc_mix_pre_norm(h, W, M, hcdim, mix_hc, eps, mixes, rsq, pre_in, hc, dim, norm_w, norm_eps, x, xn, st); return; }
  launch(hc_mix_pre_norm2_kernel<4>, dim3(M, mix_hc + 2), dim3(256), 0, st, "hc_mix", h, W, hcdim, mix_hc, eps, mixes, rsq, pre_in, dim, norm_w, norm_eps, x,
         xn);
}

void hc_mix_pre_norm2_sk(const bf16* h, const float* W, int M, int hcdim, int mix_hc, float eps, float* mixes, float* rsq, const float* pre_in, int hc, int dim,
                         const bf16* norm_w, float norm_eps, bf16* x, bf16* xn, const float* scale, const float* base, int iters, float hc_eps, float* pre_out,
                         float* post, float* comb, int* cnt, cudaStream_t st) {
  if (!(hc == 4 && hcdim == hc * dim)) {  // same fallback as hc_mix_pre_norm2 (to that version — same shape condition)
    hc_mix_pre_norm_sk(h, W, M, hcdim, mix_hc, eps, mixes, rsq, pre_in, hc, dim, norm_w, norm_eps, x, xn, scale, base, iters, hc_eps, pre_out, post, comb, cnt, st);
    return;
  }
  HIVE_CHECK(mix_hc == (2 + hc) * hc && M >= 1 && M <= 16, "hc_mix_pre_norm2_sk shape");
  launch(hc_mix_pre_norm2_sk_kernel<4>, dim3(M, mix_hc + 2), dim3(256), 0, st, "hc_mix_sk", h, W, hcdim, mix_hc, eps, mixes, rsq, pre_in, dim, norm_w, norm_eps,
         x, xn, scale, base, M, iters, hc_eps, pre_out, post, comb, cnt);
}
void hc_mix_pre_norm2_sk_preload() {
  cudaFuncAttributes fa;
  CUDA_CHECK(cudaFuncGetAttributes(&fa, hc_mix_pre_norm2_sk_kernel<4>));
}

}  // namespace hive::k
