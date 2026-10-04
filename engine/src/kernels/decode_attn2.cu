// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_ATTN2: bandwidth-oriented decode (M <= 8) versions of the router, wq_b (q_b), wo_b, the shared expert
//   (w1||w3, w2) and wq_a. Contract and switch: see the header comment of include/hive/decode_attn2.h.
//
// Background (nsys, release build, decode M=1, last 128 steps; per layer = step value / 40): router_fused 29us
//   (weights 7.9MB -> 0.27 TB/s), dec_qb 42.5us (wq_b 43MB -> 1.0 TB/s), gemv_quantin(wo_b) 35us (43MB -> 1.2 TB/s),
//   gemv_w13_swiglu 21us (23.6MB -> 1.1 TB/s), gemv_w2_acc 12.8us (11.8MB -> 0.92 TB/s). HBM ~1.8 TB/s, 188 SMs.
//   (gemv_bf16_fp8_grouped at 0.94ms/step is wo_a (34.5MB, 23.5us -> 1.47 TB/s), not the shared expert; it is already
//   fast and is left alone.)
//
// Why the previous kernels are slow (established from the code structure):
//   1. fused.cu / attn_decode_fused.cu GEMV: every block (8 columns) quantizes the whole input **first**
//      (quant_rows_smem: only K/32 threads work, each serially over 32 elements: 32 scalar bf16 loads + 32 IEEE
//      divisions + 32 e4m3 conversions). No weight load is in flight meanwhile. dec_qb repeats this prologue in 4,096
//      blocks (K=1280 -> 40 threads per block).
//   2. warp_dot's `for (b = lane; b < nb; b += 32)` is not unrolled, so a warp has only 1KB (32B per lane) in flight and
//      waits for it before the next block: w13 does w1 5x + w3 5x = 10 serial round trips, wo_b 8.
//   3. Router: only 48 blocks (warp = expert) occupy SMs, lanes load fp32 scalars serially, and the last block's top-k
//      is a sequential scan of 384x6 = 2,304 steps by lane 0 alone.
// Why HIVE_DECODE_GEMV2 (gemv_decode.cu), a bit-identical rewrite of the same kernels, was **slower** (test_decode_gemv:
//   shared expert 0.56-0.66x, router 0.50-0.96x, layer chain 0.94-0.97x; cause confirmed by reading the code):
//   1. All weights (the lane's NBL blocks x 32B + scales, two matrices for w13) are preloaded into **registers** ->
//      97-146 registers per thread (w2 spills 8B) -> 1 block per SM (256 threads x 146 = 37K registers; two blocks would
//      exceed 65,536). The w13 grid of 288 blocks exceeds the 188 SMs, so it runs in **two waves** (188 + 100), w2's
//      640 blocks in four; each wave repeats the full "preload -> quantize -> dot" latency (the original kernels use
//      few registers, so the whole grid is resident in one wave).
//   2. Router RW=2 (2 warps per block) -> 64 threads per SM, and for each row m the same warp re-reads xn as scalars,
//      so the dot products are serial over M at M=4/8 (M times).
//   3. Input quantization is still K/32 serial threads per block; even if the preload finished earlier, the weights are
//      used only after quantization.
// This version (design rules: do not add registers but increase bytes in flight per SM; one wave; hide quantization
// behind the weight loads):
//   - One GEMV template (a2_gemv_kernel<EPI, S>): block = 8 warps, __launch_bounds__(256, 2) (registers <= 128 -> 2
//     blocks per SM). Grid = min(column groups / 8, occupancy x SMs), i.e. one wave. Each warp owns a contiguous column
//     range (split evenly over all warps of the grid) and receives its columns' weights through a **per-warp cp.async
//     ring** (S stages x (1KB weights + 32B scales)). Stage = (column, matrix, t): lane l receives the 32B of block
//     32t + l and the scales (4B groups, lanes < 8). The first S-1 stages are issued **before input quantization**.
//     In flight = (S-1)KB per warp -> 16 warps per SM x 3KB = 48KB (needed ~ 1.8 TB/s x ~1.5us / 188 ~ 14KB).
//   - Input quantization (a2_quant_rows): one 32-element block is handled by 4 lanes, 8 elements each (16B loads);
//     amax via 2 xor shuffles (fmaxf: order-independent, same NaN-dropping rule); the scale, code and e4m3 formulas
//     are those of quant_rows_smem (including the v / sc division). All 256 threads of the block work.
//   - Dot product: lane l takes blocks b = 32t + l (t ascending); per block d = fmaf(a_i, w_i, d) (i 0..31) ->
//     acc = fmaf(d, sa·sb, acc) -> xor 16,8,4,2,1, the same chain as warp_dot. Only the column -> warp assignment
//     differs (it does not affect values). sa = e8m0_fast_f(code) equals e8m0_to_f32(code) for code <= 254, and
//     f32_ceil_pow2_e8m0 clamps activation codes to 254.
//   - Epilogue: a warp writes as soon as its column is done (for QB one warp owns the RoPE pair (2p, 2p+1), so no
//     block synchronization). The QA tail (qrn = rmsnorm(qr)) is done by the last block with rmsnorm_row_256, as before.
//   - Router (a2_router_kernel): block = one expert (384 blocks, 256 threads, smem 20-24KB -> 2 blocks per SM, all
//     resident). The whole block fetches the gate_w row (20KB) with cp.async at once; row m is computed by warp m with
//     the original formula (fmaf over kk = lane + 32j ascending), xn prefetched into registers 16 at a time, one group
//     ahead. Top-k: one warp per row; each round takes a lane-local maximum (ascending, `>`) and reduces with xor
//     shuffles (larger value, on ties the smaller index), which selects the same first maximum as the original
//     sequential lane-0 scan. softplus, sqrt, bias, weight sum (in j order) and normalization are unchanged.
// Expected (arithmetic estimate, not a measurement; the GPU test test_decode_attn2 measures it): per layer at M=1
//   router 29 -> ~8us, q_b 42.5 -> ~30us, wo_b 35 -> ~29us, w13 21 -> ~17us, w2 12.8 -> ~9.6us => ~-46us per layer,
//   ~-1.8ms per step (40 layers) on the front segment, the critical path before the experts start.
#include <cuda_fp8.h>

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hive/decode_attn2.h"
#include "hive/kernels.h"
#include "hive/model_kernels.h"
#include "hive/warp_reduce.cuh"

namespace hive::k {

namespace {

// Switch test = same rule as runtime.cpp env_on (unset, "" or "0" = off). The runtime one lives in an anonymous namespace, so it is duplicated here (same reason as gemv_decode.cu).
inline bool a2_env_on(const char* name) { const char* v = getenv(name); return v && *v && strcmp(v, "0") != 0; }

constexpr int A2T = 256;           // block = 8 warps
constexpr int A2W = A2T / 32;
constexpr int A2M = 8;             // max rows (decode batch)
constexpr int A2_SLOT = 1024;      // weight bytes per ring stage (32 lanes x 32B block)
constexpr int A2_SSLOT = 32;       // scale bytes per ring stage (32 blocks)
constexpr size_t kA2SmemMax = 98 * 1024;  // dynamic smem cap (stays within 99KB after static smem and 1KB headroom)
constexpr float kFp8MaxInv = 1.0f / 448.0f;
enum A2Epi { A2_QA = 0, A2_QB = 1, A2_QUANTIN = 2, A2_W13 = 3, A2_W2 = 4 };

// ---- device helpers (same formulas as fused.cu; bit identity is the contract, so the formulas must not change) -------------------------------------------------------
__device__ __forceinline__ float2 a2_cvt_e4m3x2(uint32_t pair16) {
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t)pair16));
  return __half22float2(*reinterpret_cast<__half2*>(&h2));
}
__device__ __forceinline__ float a2_e8m0_fast(uint8_t b) { return b ? __uint_as_float((uint32_t)b << 23) : __uint_as_float(0x00400000u); }
__device__ __forceinline__ void a2_decode8(const uint4 r0, const uint4 r1, float (&w)[32]) {
  const uint32_t x[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 lo = a2_cvt_e4m3x2(x[i] & 0xFFFFu), hi = a2_cvt_e4m3x2(x[i] >> 16);
    w[4 * i] = lo.x; w[4 * i + 1] = lo.y; w[4 * i + 2] = hi.x; w[4 * i + 3] = hi.y;
  }
}
// cp.async in the most conservative form, as in moe_decode.cu (no hints, no prefetch: the cache_hint variant raised an illegal instruction on sm_120a, measured)
__device__ __forceinline__ uint32_t a2_su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void a2_cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a2_su32(s)), "l"(g) : "memory"); }
__device__ __forceinline__ void a2_cp4(void* s, const void* g) { asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(a2_su32(s)), "l"(g) : "memory"); }
__device__ __forceinline__ void a2_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void a2_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// A[M, K] bf16 (row stride K) -> aq[M·K] e4m3 + ac[M·K/32] e8m0 codes. **Same values** as quant_rows_smem (fused.cu):
//   element formula v / sc, clamp +-448, e4m3 (RNE) unchanged; amax = max(0, |v_i|...) uses fmaxf, so grouping order does
//   not matter (fmaxf is exact and drops NaN arguments, same result as a sequential fmaxf) -> fmaxf(amax, 1e-4).
//   Only the work split differs: one block = 4 lanes (8 elements, 16B loads), amax via xor 1 and 2 within the warp. Each
//   thread first issues the loads for A2QR blocks per pass (M=8, K=8192 = 2,048 blocks -> 8 passes, instead of 8 serial
//   blocks per thread). The loop trip count is the same for the whole block, so the whole warp reaches the shuffles.
constexpr int A2QR = 4;
__device__ __forceinline__ void a2_quant_rows(const bf16* __restrict__ A, int M, int K, uint8_t* aq, uint8_t* ac) {
  const int nb = K / 32, total = M * nb, q = threadIdx.x & 3;
  for (int i0 = 0; i0 < total; i0 += A2QR * (A2T / 4)) {
    uint4 raw[A2QR];
#pragma unroll
    for (int r = 0; r < A2QR; ++r) {
      const int idx = i0 + r * (A2T / 4) + (int)(threadIdx.x >> 2);
      if (idx < total) {
        const int m = idx / nb, b = idx - m * nb;
        raw[r] = *reinterpret_cast<const uint4*>(A + (size_t)m * K + (size_t)b * 32 + q * 8);
      } else {
        raw[r] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
#pragma unroll
    for (int r = 0; r < A2QR; ++r) {
      const int idx = i0 + r * (A2T / 4) + (int)(threadIdx.x >> 2);
      const bool ok = idx < total;
      const uint32_t x[4] = {raw[r].x, raw[r].y, raw[r].z, raw[r].w};
      float v[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) { v[2 * i] = __uint_as_float(x[i] << 16); v[2 * i + 1] = __uint_as_float(x[i] & 0xFFFF0000u); }  // = bf2f(A[·])
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
      if (ok) {
        const int m = idx / nb, b = idx - m * nb;
        amax = fmaxf(amax, 1e-4f);
        const uint8_t code = f32_ceil_pow2_e8m0(amax * kFp8MaxInv);
        const float sc = e8m0_to_f32(code);
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) lo |= (uint32_t)f32_to_e4m3(fminf(fmaxf(v[i] / sc, -448.f), 448.f)) << (8 * i);
#pragma unroll
        for (int i = 0; i < 4; ++i) hi |= (uint32_t)f32_to_e4m3(fminf(fmaxf(v[4 + i] / sc, -448.f), 448.f)) << (8 * i);
        *reinterpret_cast<uint2*>(aq + (size_t)m * K + (size_t)b * 32 + q * 8) = make_uint2(lo, hi);
        if (q == 0) ac[idx] = code;  // idx = m·nb + b
      }
    }
  }
}
// same reduction as fused.cu rmsnorm_row_256
__device__ __forceinline__ void a2_rmsnorm_row_256(const bf16* __restrict__ x, const bf16* __restrict__ w, float eps, int N, bf16* __restrict__ out, float* red) {
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
// same as fused.cu last_block (the last block resets the counter to 0, safe for graph replay)
__device__ __forceinline__ bool a2_last_block(int* counter, int* flag_smem) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) *flag_smem = (atomicAdd(counter, 1) == (int)gridDim.x - 1);
  __syncthreads();
  const bool last = *flag_smem != 0;
  if (last) { __threadfence(); if (threadIdx.x == 0) *counter = 0; }
  return last;
}

// ---- GEMV template --------------------------------------------------------------------------------------------------------------
struct A2Args {
  const bf16* A;                              // input [M, K] bf16 (row stride K)
  const uint8_t *B, *sb, *B2, *sb2;           // weights [N, K] e4m3 + [N, K/32] e8m0 (W13: B = w1, B2 = w3)
  int M, K, N, items;                         // items = number of warp work items (QB: N/2 column pairs, otherwise N)
  bf16* out;                                  // QA: qr · QB: q · QUANTIN: C · W13: y
  float* out_f;                               // W2: acc
  bf16* out2; const bf16* norm_w; float eps; int* counter;  // QA tail: qrn = rmsnorm(qr, q_norm)
  int D, rd; const float2* freqs; const int32_t* pos; int pos_stride;  // QB RoPE (position = pos[m · pos_stride])
  uint8_t *side_q, *side_s;                   // QB: quantized qrn, W13: quantized xn (written by block 0)
  float limit;                                // W13 swiglu
};

template <int EPI, int S>
__global__ void __launch_bounds__(A2T, 2) a2_gemv_kernel(const A2Args a) {
  constexpr int NMAT = EPI == A2_W13 ? 2 : 1;
  constexpr int CW = EPI == A2_QB ? 2 : 1;  // columns per warp work item (QB = RoPE pair)
  extern __shared__ __align__(16) uint8_t a2_sm[];
  __shared__ float red[8];
  __shared__ int flag;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int M = a.M, K = a.K, N = a.N, nb = K / 32, nbl = (nb + 31) / 32;
  uint8_t* const wr = a2_sm + warp * (S * A2_SLOT);                                   // this warp's weight ring
  uint8_t* const sr = a2_sm + A2W * S * A2_SLOT + warp * (S * A2_SSLOT);             // this warp's scale ring
  uint8_t* const aq = a2_sm + A2W * S * (A2_SLOT + A2_SSLOT);                        // quantized input [M, K]
  uint8_t* const ac = aq + (size_t)M * K;                                            // input scale codes [M, nb]
  // this warp's work items [it0, it1), split evenly over all warps of the grid (difference <= 1)
  const long long nw = (long long)gridDim.x * A2W, gw = (long long)blockIdx.x * A2W + warp;
  const int it0 = (int)(gw * a.items / nw), it1 = (int)((gw + 1) * a.items / nw);
  const int c0 = it0 * CW, ncol = (it1 - it0) * CW;
  const int nst = ncol * NMAT * nbl;  // number of stages this warp receives
  // issue cursor (stage order = column -> matrix -> t, same as the consumption order below)
  int is = 0, ic = c0, im = 0, itt = 0;
  auto issue = [&]() {
    if (is < nst) {
      const uint8_t* Bm = a.B;
      const uint8_t* Sm = a.sb;
      if constexpr (NMAT == 2) { if (im) { Bm = a.B2; Sm = a.sb2; } }
      const int slot = is % S, b = itt * 32 + lane;
      if (b < nb) {
        const uint8_t* g = Bm + (size_t)ic * K + (size_t)b * 32;
        uint8_t* d = wr + slot * A2_SLOT + lane * 32;
        a2_cp16(d, g);
        a2_cp16(d + 16, g + 16);
      }
      const int nv = min(32, nb - itt * 32);  // blocks in this stage (a multiple of 4; the launcher checks nb % 4 == 0)
      if (lane * 4 < nv) a2_cp4(sr + slot * A2_SSLOT + lane * 4, Sm + (size_t)ic * nb + itt * 32 + lane * 4);
      if (++itt == nbl) { itt = 0; if (++im == NMAT) { im = 0; ++ic; } }
    }
    a2_commit();  // commit empty groups too (keeps wait_group counts aligned with stage numbers)
    ++is;
  };
  // (1) first S-1 weight stages (independent of the previous kernel's output) -> (2) input quantization (weights are in flight meanwhile)
#pragma unroll
  for (int s0 = 0; s0 < S - 1; ++s0) issue();
  a2_quant_rows(a.A, M, K, aq, ac);
  __syncthreads();
  if constexpr (EPI == A2_QB || EPI == A2_W13) {  // side output (block 0, as in the original kernels; same bytes)
    if (blockIdx.x == 0 && a.side_q) {
      for (int i = threadIdx.x; i < M * K; i += A2T) a.side_q[i] = aq[i];
      for (int i = threadIdx.x; i < M * nb; i += A2T) a.side_s[i] = ac[i];
    }
  }
  // (3) per-column dot product (same chain as warp_dot) + epilogue
  float keep[A2M];  // QB: first column of the pair
#pragma unroll
  for (int m = 0; m < A2M; ++m) keep[m] = 0.f;
  int s = 0;
  for (int ci = 0; ci < ncol; ++ci) {
    const int n = c0 + ci;
    float acc[NMAT][A2M];
#pragma unroll
    for (int mt = 0; mt < NMAT; ++mt) {
#pragma unroll
      for (int m = 0; m < A2M; ++m) acc[mt][m] = 0.f;
      for (int t = 0; t < nbl; ++t, ++s) {
        a2_wait<S - 2>();  // this lane's share of stage s has arrived
        __syncwarp();      // makes other lanes' shares visible, and ensures all lanes finished reading the previous stage's slot before the issue below overwrites it
        issue();           // stage s + S - 1 -> slot (s - 1) % S
        const int slot = s % S, b = t * 32 + lane;
        if (b < nb) {
          const uint4* pw = reinterpret_cast<const uint4*>(wr + slot * A2_SLOT + lane * 32);
          float w[32];
          a2_decode8(pw[0], pw[1], w);
          const float sbv = a2_e8m0_fast(sr[slot * A2_SSLOT + lane]);
#pragma unroll
          for (int m = 0; m < A2M; ++m) {
            if (m < M) {
              const uint4* pa = reinterpret_cast<const uint4*>(aq + (size_t)m * K + (size_t)b * 32);
              float av[32];
              a2_decode8(pa[0], pa[1], av);
              float d = 0.f;
#pragma unroll
              for (int i = 0; i < 32; ++i) d = fmaf(av[i], w[i], d);
              const float sav = a2_e8m0_fast(ac[m * nb + b]);
              acc[mt][m] = fmaf(d, sav * sbv, acc[mt][m]);
            }
          }
        }
      }
#pragma unroll
      for (int m = 0; m < A2M; ++m) {
        float v = acc[mt][m];
        v = hive::cu::warp_allreduce(v);
        acc[mt][m] = v;
      }
    }
    // epilogue (formulas of the respective original kernels, unchanged); every lane holds the same values
    if constexpr (EPI == A2_QA || EPI == A2_QUANTIN) {
      if (lane == 0) {
#pragma unroll
        for (int m = 0; m < A2M; ++m) if (m < M) a.out[(size_t)m * N + n] = f2bf(acc[0][m]);
      }
    } else if constexpr (EPI == A2_W2) {
      if (lane == 0) {
#pragma unroll
        for (int m = 0; m < A2M; ++m) if (m < M) a.out_f[(size_t)m * N + n] = bf2f(f2bf(acc[0][m]));
      }
    } else if constexpr (EPI == A2_W13) {  // swiglu_route(rw = nullptr); row m is handled by lane m
#pragma unroll
      for (int m = 0; m < A2M; ++m) {
        if (m < M && lane == m) {
          float g = bf2f(f2bf(acc[0][m])), u = bf2f(f2bf(acc[NMAT - 1][m]));
          const float limit = a.limit;
          if (limit > 0.f) { u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); }
          a.out[(size_t)m * N + n] = f2bf(g / (1.f + expf(-g)) * u);
        }
      }
    } else {  // A2_QB: pair (n0, n0+1); kept at the even column, written at the odd column (q-block formula of gemv_qb_rope_kernel / dec_qb_kernel)
      if ((ci & 1) == 0) {
#pragma unroll
        for (int m = 0; m < A2M; ++m) keep[m] = acc[0][m];
      } else {
        const int n0 = n - 1, dpos = n0 % a.D;
#pragma unroll
        for (int m = 0; m < A2M; ++m) {
          if (m < M && lane == m) {
            bf16* qm = a.out + (size_t)m * N;
            if (dpos < a.D - a.rd) {
              qm[n0] = f2bf(keep[m]);
              qm[n0 + 1] = f2bf(acc[0][m]);
            } else {
              const int p = (dpos - (a.D - a.rd)) >> 1;
              const float2 f = a.freqs[(size_t)a.pos[(size_t)m * a.pos_stride] * (a.rd / 2) + p];
              const float x0 = bf2f(f2bf(keep[m])), x1 = bf2f(f2bf(acc[0][m]));
              qm[n0] = f2bf(x0 * f.x - x1 * f.y);
              qm[n0 + 1] = f2bf(x0 * f.y + x1 * f.x);
            }
          }
        }
      }
    }
  }
  a2_wait<0>();
  if constexpr (EPI == A2_QA) {  // tail: qrn = rmsnorm(qr), done by the last block as in gemv_qa_kernel
    if (!a2_last_block(a.counter, &flag)) return;
    for (int m = 0; m < M; ++m) a2_rmsnorm_row_256(a.out + (size_t)m * N, a.norm_w, a.eps, N, a.out2 + (size_t)m * N, red);
  }
  (void)red;
  (void)flag;
}

// ---- router ---------------------------------------------------------------------------------------------------------------------
constexpr int A2RT = 256;  // block = one expert, row m = warp m
constexpr int A2RU = 16;   // xn prefetch group (16 per lane = one group ahead)
__global__ void __launch_bounds__(A2RT) a2_router_kernel(const bf16* __restrict__ xn, int K, const float* __restrict__ W, int M, int E,
                                                         const float* __restrict__ bias, const float* __restrict__ bias_vl,
                                                         const int8_t* __restrict__ is_image, int k, float route_scale, float* __restrict__ scores,
                                                         int32_t* __restrict__ ids, float* __restrict__ w, int* __restrict__ counter) {
  extern __shared__ __align__(16) float a2_rs[];  // [K] gate_w row; reused as s[M][E] and sel[M][E] in the tail
  __shared__ int flag;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, e = blockIdx.x;
  const float* We = W + (size_t)e * K;
  for (int c = threadIdx.x; c < K / 4; c += A2RT) a2_cp16(a2_rs + 4 * c, We + 4 * c);
  a2_commit();
  const bool act = warp < M;
  const bf16* xm = xn + (size_t)(act ? warp : 0) * K;
  float xa[A2RU];
#pragma unroll
  for (int u = 0; u < A2RU; ++u) { const int kk = lane + 32 * u; xa[u] = (act && kk < K) ? bf2f(xm[kk]) : 0.f; }
  a2_wait<0>();
  __syncthreads();
  if (act) {  // same as the original: d = fmaf(bf2f(xm[kk]), We[kk], d), kk = lane, lane+32, ... ascending -> xor reduction
    float d = 0.f;
    for (int j0 = 0; j0 < K; j0 += 32 * A2RU) {
      float xb[A2RU];
      const int j1 = j0 + 32 * A2RU;
#pragma unroll
      for (int u = 0; u < A2RU; ++u) { const int kk = j1 + lane + 32 * u; xb[u] = kk < K ? bf2f(xm[kk]) : 0.f; }
#pragma unroll
      for (int u = 0; u < A2RU; ++u) { const int kk = j0 + lane + 32 * u; if (kk < K) d = fmaf(xa[u], a2_rs[kk], d); }
#pragma unroll
      for (int u = 0; u < A2RU; ++u) xa[u] = xb[u];
    }
    d = hive::cu::warp_allreduce(d);
    if (lane == 0) scores[(size_t)warp * E + e] = d;
  }
  if (!a2_last_block(counter, &flag)) return;
  float* s = a2_rs;
  float* sel = a2_rs + (size_t)M * E;
  for (int m = warp; m < M; m += A2RT / 32) {
    float* sm = s + (size_t)m * E;
    float* sl = sel + (size_t)m * E;
    const float* b = (bias_vl && is_image && is_image[m]) ? bias_vl : bias;
    for (int ee = lane; ee < E; ee += 32) {  // original formula, unchanged
      const float x = __ldcg(scores + (size_t)m * E + ee);
      const float sp = x > 20.f ? x : log1pf(expf(x));
      sm[ee] = sqrtf(sp);
      sl[ee] = sm[ee] + b[ee];
    }
    __syncwarp();
    float wsum = 0.f;
    for (int j = 0; j < k; ++j) {
      float bv = -FLT_MAX;
      int bi = -1;
      for (int ee = lane; ee < E; ee += 32) { const float v = sl[ee]; if (v > bv) { bv = v; bi = ee; } }
#pragma unroll
      for (int o = 16; o >= 1; o /= 2) {
        const float ov = __shfl_xor_sync(0xffffffff, bv, o);
        const int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
      }
      if (lane == 0) {
        const float sv = bi >= 0 ? sm[bi] : 0.f;
        if (bi >= 0) sl[bi] = -FLT_MAX;
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

// ---- launch -----------------------------------------------------------------------------------------------------------------------
inline void a2_check(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) return;
  throw std::runtime_error(std::string("attn2 launch failed: ").append(what).append(": ").append(cudaGetErrorString(e)));
}
inline int a2_sms() {
  static const int n = [] {
    int dev = 0, v = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev));
    return v;
  }();
  return n;
}
inline bool a2_al(const void* p, size_t a) { return ((uintptr_t)p % a) == 0; }
inline size_t a2_smem(int S, int M, int K) { return (size_t)A2W * S * (A2_SLOT + A2_SSLOT) + (size_t)M * K + (size_t)M * (K / 32); }
// resident blocks per SM (dynamic smem limit attribute -> occupancy query), computed once per smem size
template <int EPI, int S>
int a2_blocks_per_sm(size_t smem) {
  static std::mutex mu;
  static std::vector<std::pair<size_t, int>> cache;
  static bool configured = false;
  std::lock_guard<std::mutex> lk(mu);
  for (const auto& p : cache) if (p.first == smem) return p.second;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(a2_gemv_kernel<EPI, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kA2SmemMax));
    configured = true;
  }
  int n = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, a2_gemv_kernel<EPI, S>, A2T, smem));
  cache.emplace_back(smem, n);
  return n;
}
// shape/alignment: M 1..8, K % 128 == 0 (block count nb is a multiple of 4, for 4B scale groups), input 16B, weights 16B, scales 4B, smem <= cap
inline bool a2_ok(const bf16* A, const uint8_t* B, const uint8_t* sb, int M, int K, int N) {
  return M >= 1 && M <= A2M && K >= 128 && K % 128 == 0 && N >= 1 && a2_al(A, 16) && a2_al(B, 16) && a2_al(sb, 4) && a2_smem(3, M, K) <= kA2SmemMax;
}
// Stage count S in {4, 3}: pick the one with more stages in flight per SM (resident blocks x 8 warps x (S-1)); on a tie
//   the one with more blocks (S=3). E.g. M=1, K=5120 -> S=4 (2 blocks x 3); M=4, K=5120 -> S=3 (2 blocks x 2 > 1 block
//   x 3); M=8, K=8192 -> S=3 (S=4 exceeds the smem cap).
template <int EPI>
void a2_launch(const A2Args& a, cudaStream_t st, const char* what) {  // the caller has checked a2_ok
  const size_t sm4 = a2_smem(4, a.M, a.K), sm3 = a2_smem(3, a.M, a.K);
  const int b4 = sm4 <= kA2SmemMax ? a2_blocks_per_sm<EPI, 4>(sm4) : 0, b3 = a2_blocks_per_sm<EPI, 3>(sm3);
  const int S = b4 * 3 > b3 * 2 ? 4 : 3;
  const size_t smem = S == 4 ? sm4 : sm3;
  const int bps = std::max(1, S == 4 ? b4 : b3);
  const int grid = std::max(1, std::min((a.items + A2W - 1) / A2W, bps * a2_sms()));
  if (S == 4) a2_gemv_kernel<EPI, 4><<<grid, A2T, smem, st>>>(a);
  else a2_gemv_kernel<EPI, 3><<<grid, A2T, smem, st>>>(a);
  a2_check(what);
}
A2Args a2_args(const bf16* A, int K, const uint8_t* B, const uint8_t* sb, int M, int N) {
  A2Args a = {};
  a.A = A; a.K = K; a.B = B; a.sb = sb; a.M = M; a.N = N; a.items = N; a.pos_stride = 1;
  return a;
}
inline bool a2_qb_ok(const bf16* qrn, int K, const uint8_t* B, const uint8_t* sb, int M, int N, int D, int rd) {
  return a2_ok(qrn, B, sb, M, K, N) && D >= 2 && D % 2 == 0 && rd >= 0 && rd % 2 == 0 && rd <= D && N % D == 0;
}

}  // namespace

bool decode_attn2_on() { static const bool on = a2_env_on("HIVE_DECODE_ATTN2"); return on; }

bool attn2_qb(const bf16* qrn, int K, const uint8_t* B, const uint8_t* sb, int M, int N, int D, int rd, const float2* freqs, const int32_t* pos,
              int pos_stride, bf16* q, uint8_t* side_q, uint8_t* side_s, cudaStream_t st) {
  if (!a2_qb_ok(qrn, K, B, sb, M, N, D, rd)) return false;
  A2Args a = a2_args(qrn, K, B, sb, M, N);
  a.items = N / 2; a.out = q; a.D = D; a.rd = rd; a.freqs = freqs; a.pos = pos; a.pos_stride = pos_stride; a.side_q = side_q; a.side_s = side_s;
  a2_launch<A2_QB>(a, st, "q_b");
  return true;
}

void attn2_q_proj(const bf16* xn, int K, const uint8_t* wqa, const uint8_t* sqa, int Nqa, const bf16* q_norm, float eps, const uint8_t* wqb,
                  const uint8_t* sqb, int H, int D, int rd, const float2* freqs, const int32_t* pos, int M, bf16* qr, bf16* qrn, bf16* q, uint8_t* qrq,
                  uint8_t* qrs, int* counter, cudaStream_t st) {
  if (!(a2_ok(xn, wqa, sqa, M, K, Nqa) && a2_qb_ok(qrn, Nqa, wqb, sqb, M, H * D, D, rd))) {  // fallback: outside this kernel's envelope, use the original function
    fused_q_proj(xn, K, wqa, sqa, Nqa, q_norm, eps, wqb, sqb, H, D, rd, freqs, pos, M, qr, qrn, q, qrq, qrs, counter, st);
    return;
  }
  A2Args a = a2_args(xn, K, wqa, sqa, M, Nqa);
  a.out = qr; a.out2 = qrn; a.norm_w = q_norm; a.eps = eps; a.counter = counter;
  a2_launch<A2_QA>(a, st, "q_a");
  attn2_qb(qrn, Nqa, wqb, sqb, M, H * D, D, rd, freqs, pos, 1, q, qrq, qrs, st);
}

void attn2_gemv_quantin(const bf16* A, int K, const uint8_t* B, const uint8_t* sb, int M, int N, bf16* C, cudaStream_t st) {
  if (!a2_ok(A, B, sb, M, K, N)) { fused_gemv_quantin(A, K, B, sb, M, N, C, st); return; }
  A2Args a = a2_args(A, K, B, sb, M, N);
  a.out = C;
  a2_launch<A2_QUANTIN>(a, st, "quantin");
}

void attn2_shared_experts(const bf16* xn, int K, const uint8_t* w1, const uint8_t* s1, const uint8_t* w3, const uint8_t* s3, const uint8_t* w2,
                          const uint8_t* s2, int M, int I, float limit, bf16* y, uint8_t* xq, uint8_t* xs, float* acc, cudaStream_t st) {
  const bool ok = a2_ok(xn, w1, s1, M, K, I) && a2_al(w3, 16) && a2_al(s3, 4) && a2_ok(y, w2, s2, M, I, K);
  if (!ok) { fused_shared_experts(xn, K, w1, s1, w3, s3, w2, s2, M, I, limit, y, xq, xs, acc, st); return; }
  A2Args a = a2_args(xn, K, w1, s1, M, I);
  a.B2 = w3; a.sb2 = s3; a.out = y; a.limit = limit; a.side_q = xq; a.side_s = xs;
  a2_launch<A2_W13>(a, st, "w13");
  A2Args b = a2_args(y, I, w2, s2, M, K);
  b.out_f = acc;
  a2_launch<A2_W2>(b, st, "w2");
}

void attn2_router(const bf16* xn, int K, const float* gate_w, int M, int E, const float* bias, const float* bias_vl, const int8_t* is_image, int k,
                  float route_scale, float* scores, int32_t* ids, float* w, int* counter, cudaStream_t st) {
  const size_t smem = (size_t)std::max<size_t>((size_t)K, (size_t)2 * M * E) * 4;
  const bool ok = M >= 1 && M <= A2M && E >= 1 && E <= 512 && k >= 1 && k <= 16 && k <= E && K >= 4 && K % 4 == 0 && a2_al(gate_w, 16) &&
                  smem <= 48 * 1024;
  if (!ok) { fused_router(xn, K, gate_w, M, E, bias, bias_vl, is_image, k, route_scale, scores, ids, w, counter, st); return; }
  a2_router_kernel<<<E, A2RT, smem, st>>>(xn, K, gate_w, M, E, bias, bias_vl, is_image, k, route_scale, scores, ids, w, counter);
  a2_check("router");
}

}  // namespace hive::k
