// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-27B GPU kernels — see hive/q27/q27_kernels.h. This project's own code (the Gated DeltaNet recurrence follows the math contract of
//   hive/q27/q27_model.h; the work split of the recurrence — one thread per state column, the column in registers — is the one of the CPU
//   engine's AVX2 loop transposed to a block).
#include "hive/q27/q27_kernels.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <vector>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "hive/common.h"

namespace hive::q27::k {

namespace {
constexpr int kWarp = 32;
struct DevBufLite {   // grow-only device scratch (never freed: process lifetime)
  void* p = nullptr;
  size_t n = 0;
  void ensure(size_t b) { if (b <= n) return; if (p) CUDA_CHECK(cudaFree(p)); CUDA_CHECK(cudaMalloc(&p, b)); n = b; }
};

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.f + __expf(-x)); }

// ---- GEMV: one warp per output row, M accumulators -----------------------------------------------------------------------------------
// Weights are decoded with the hardware conversions (cvt.rn.f16x2.e4m3x2 / .e2m1x2 → exact halves → fp32); activations are loaded 8 bf16
//   at a time. Per lane the k order is fixed (chunk c = lane, lane + 32, …), then the warp tree — independent of the launch shape.
__device__ __forceinline__ void bf8(const bf16* p, float (&f)[8]) {
  const uint4 v = *reinterpret_cast<const uint4*>(p);
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) { f[2 * i] = __uint_as_float(w[i] << 16); f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u); }
}
__device__ __forceinline__ float2 fp8x2(uint16_t b) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)b, __NV_E4M3);
  return __half22float2(*reinterpret_cast<const __half2*>(&h));
}
__device__ __forceinline__ float2 fp4x2(uint8_t b) {
  const __half2_raw h = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b, __NV_E2M1);
  return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

// R weight rows (pointers to each row's bytes / scales) against M activation rows: per lane chunk c = lane, lane + 32, …; the activation
//   values of a chunk are loaded once and applied to all R rows (decode GEMV is limited by these loads, not by the weight stream, at M ≥ 2).
//   res[r][m] = warp-summed dot (scales of NVFP4 groups applied; the per-row / global scale is the caller's).
template <int M, WFmt F, int R>
__device__ __forceinline__ void rows_part(const uint8_t* const (&w)[R], const uint8_t* const (&sc)[R], const bf16* x, int ldx, int kb, int ke,
                                          int lane, float (&res)[R][M]) {
  if constexpr (F == WFmt::FP8ROW) {
    for (int c = kb / 16 + lane; c < ke / 16; c += kWarp) {
      uint4 wv[R];
#pragma unroll
      for (int r = 0; r < R; ++r) wv[r] = __ldg(reinterpret_cast<const uint4*>(w[r]) + c);
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        float xf[M][8];
#pragma unroll
        for (int m = 0; m < M; ++m) bf8(x + (size_t)m * ldx + (c * 16 - kb) + hf * 8, xf[m]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const uint16_t* b = reinterpret_cast<const uint16_t*>(&wv[r]) + hf * 4;
          float wf[8];
#pragma unroll
          for (int i = 0; i < 4; ++i) { const float2 t = fp8x2(b[i]); wf[2 * i] = t.x; wf[2 * i + 1] = t.y; }
#pragma unroll
          for (int m = 0; m < M; ++m)
#pragma unroll
            for (int i = 0; i < 8; ++i) res[r][m] = fmaf(wf[i], xf[m][i], res[r][m]);
        }
      }
    }
  } else if constexpr (F == WFmt::NVFP4) {
    for (int c = kb / 32 + lane; c < ke / 32; c += kWarp) {
      uint4 wv[R];
      float2 sv[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        wv[r] = __ldg(reinterpret_cast<const uint4*>(w[r]) + c);
        sv[r] = fp8x2(__ldg(reinterpret_cast<const uint16_t*>(sc[r]) + c));
      }
#pragma unroll
      for (int g = 0; g < 2; ++g) {   // scale group g: 16 values
        float part[R][M];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int m = 0; m < M; ++m) part[r][m] = 0.f;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          float xf[M][8];
#pragma unroll
          for (int m = 0; m < M; ++m) bf8(x + (size_t)m * ldx + (c * 32 - kb) + g * 16 + q * 8, xf[m]);
#pragma unroll
          for (int r = 0; r < R; ++r) {
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&wv[r]) + g * 8 + q * 4;
            float wf[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) { const float2 t = fp4x2(b[i]); wf[2 * i] = t.x; wf[2 * i + 1] = t.y; }
#pragma unroll
            for (int m = 0; m < M; ++m)
#pragma unroll
              for (int i = 0; i < 8; ++i) part[r][m] = fmaf(wf[i], xf[m][i], part[r][m]);
          }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int m = 0; m < M; ++m) res[r][m] = fmaf(part[r][m], g ? sv[r].y : sv[r].x, res[r][m]);
      }
    }
  } else {
    for (int c = kb / 8 + lane; c < ke / 8; c += kWarp) {
      float xf[M][8];
#pragma unroll
      for (int m = 0; m < M; ++m) bf8(x + (size_t)m * ldx + (c * 8 - kb), xf[m]);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float wf[8];
        bf8(reinterpret_cast<const bf16*>(w[r]) + (size_t)c * 8, wf);
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int i = 0; i < 8; ++i) res[r][m] = fmaf(wf[i], xf[m][i], res[r][m]);
      }
    }
  }
}

template <int M, WFmt F, int R>
__device__ __forceinline__ void rows_dot(const uint8_t* const (&w)[R], const uint8_t* const (&sc)[R], const bf16* __restrict__ x, int ldx, int K,
                                         int lane, float (&res)[R][M]) {
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) res[r][m] = 0.f;
  rows_part<M, F, R>(w, sc, x, ldx, 0, K, lane, res);
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) res[r][m] = warp_sum(res[r][m]);
}
// The same per-lane chunk order with the activation rows staged in shared memory per block (KX columns at a time — a multiple of every
//   format's warp step, so each lane visits exactly the chunks it would visit in rows_dot, in the same order: bit-identical results).
constexpr int kKX = 2048;
template <int M, WFmt F, int R>
__device__ __forceinline__ void rows_dot_smem(const uint8_t* const (&w)[R], const uint8_t* const (&sc)[R], const bf16* __restrict__ x, int ldx, int K,
                                              int lane, bool active, bf16* xs, float (&res)[R][M]) {
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) res[r][m] = 0.f;
  for (int kb = 0; kb < K; kb += kKX) {
    const int kx = min(kKX, K - kb);
    __syncthreads();
    for (int i = threadIdx.x; i < M * (kx / 8); i += blockDim.x) {
      const int m = i / (kx / 8), c8 = (i % (kx / 8)) * 8;
      *reinterpret_cast<uint4*>(xs + m * kKX + c8) = *reinterpret_cast<const uint4*>(x + (size_t)m * ldx + kb + c8);
    }
    __syncthreads();
    if (active) rows_part<M, F, R>(w, sc, xs, kKX, kb, kb + kx, lane, res);
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) res[r][m] = warp_sum(res[r][m]);
}

__device__ __forceinline__ size_t row_bytes(const GMat& W) { return W.fmt == WFmt::BF16 ? (size_t)W.K * 2 : W.fmt == WFmt::FP8ROW ? (size_t)W.K : (size_t)W.K / 2; }
__device__ __forceinline__ float row_scale(const GMat& W, int n) {
  return W.fmt == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[n] : W.fmt == WFmt::NVFP4 ? W.g : 1.f;
}

// rows [n0, n0 + R) of W per warp (the last warp clamps to row N − 1 and does not write the duplicates)
template <int M, WFmt F, int R>
__global__ void __launch_bounds__(256) gemv_kernel(GMat W, const bf16* __restrict__ x, int ldx, bf16* yb, float* yf, int ldy) {
  const int n0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * R, lane = threadIdx.x % kWarp;
  constexpr bool kStage = M >= 2;
  __shared__ __align__(16) bf16 xs[kStage ? M * kKX : 8];
  if (!kStage && n0 >= W.N) return;   // with staging every warp takes part in the block's copies
  const bool active = n0 < W.N;
  const uint8_t* w[R];
  const uint8_t* sc[R];
  const size_t rb = row_bytes(W);
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int n = min(max(n0, 0) + r, W.N - 1);
    w[r] = W.w + (size_t)n * rb;
    sc[r] = F == WFmt::NVFP4 ? reinterpret_cast<const uint8_t*>(W.s) + (size_t)n * (W.K / 16) : nullptr;
  }
  float res[R][M];
  if constexpr (kStage) rows_dot_smem<M, F, R>(w, sc, x, ldx, W.K, lane, active, xs, res);
  else rows_dot<M, F, R>(w, sc, x, ldx, W.K, lane, res);
  if (!active) return;
  if (lane == 0)
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = n0 + r;
      if (n >= W.N) break;
      const float s = row_scale(W, n);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        if (yb) yb[(size_t)m * ldy + n] = __float2bfloat16_rn(res[r][m] * s);
        else yf[(size_t)m * ldy + n] = res[r][m] * s;
      }
    }
}

// split-K GEMV for short-N / long-K matrices (N ≤ 8192: down, o_proj, out_proj, MTP fc): a block = 2 rows × 4 K parts (8 warps); part p
//   covers the warp iterations [p·I/4, (p+1)·I/4) of the row (I = warp iterations over K), the 4 warp sums are added in part order.
//   The split depends on the matrix only, so every M gives the same per-row arithmetic (decode == verify bit for bit).
template <int M, WFmt F>
__global__ void __launch_bounds__(256) gemv_splitk_kernel(GMat W, const bf16* __restrict__ x, int ldx, bf16* yb, float* yf, int ldy) {
  constexpr int S = 4, CW = F == WFmt::NVFP4 ? 32 : F == WFmt::FP8ROW ? 16 : 8;
  __shared__ float part[2][S][M];
  const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
  const int rl = warp / S, p = warp % S, n = blockIdx.x * 2 + rl;
  const int nr = min(n, W.N - 1);
  const int iters = (W.K / CW + kWarp - 1) / kWarp;
  const int it0 = p * iters / S, it1 = (p + 1) * iters / S;
  const int kb = min(W.K, it0 * kWarp * CW), ke = min(W.K, it1 * kWarp * CW);
  const uint8_t* w[1] = {W.w + (size_t)nr * (F == WFmt::BF16 ? (size_t)W.K * 2 : F == WFmt::FP8ROW ? (size_t)W.K : (size_t)W.K / 2)};
  const uint8_t* sc[1] = {F == WFmt::NVFP4 ? reinterpret_cast<const uint8_t*>(W.s) + (size_t)nr * (W.K / 16) : nullptr};
  float res[1][M];
#pragma unroll
  for (int m = 0; m < M; ++m) res[0][m] = 0.f;
  rows_part<M, F, 1>(w, sc, x + kb, ldx, kb, ke, lane, res);
#pragma unroll
  for (int m = 0; m < M; ++m) res[0][m] = warp_sum(res[0][m]);
  if (lane == 0)
#pragma unroll
    for (int m = 0; m < M; ++m) part[rl][p][m] = res[0][m];
  __syncthreads();
  if (p == 0 && lane == 0 && n < W.N) {
    const float sc2 = row_scale(W, n);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float t = part[rl][0][m];
#pragma unroll
      for (int q = 1; q < S; ++q) t += part[rl][q][m];
      if (yb) yb[(size_t)m * ldy + n] = __float2bfloat16_rn(t * sc2); else yf[(size_t)m * ldy + n] = t * sc2;
    }
  }
}

// fused MLP front: rows [n0, n0 + R/2) of G and of U per warp
template <int M, WFmt F, int R>
__global__ void __launch_bounds__(256) gemv_gu_kernel(GMat G, GMat U, const bf16* __restrict__ x, int ldx, bf16* act, int lda) {
  constexpr int RH = R / 2;
  const int n0 = (blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp) * RH, lane = threadIdx.x % kWarp;
  constexpr bool kStage = M >= 2;
  __shared__ __align__(16) bf16 xs[kStage ? M * kKX : 8];
  if (!kStage && n0 >= G.N) return;
  const bool active = n0 < G.N;
  const uint8_t* w[R];
  const uint8_t* sc[R];
  const size_t rb = row_bytes(G);
#pragma unroll
  for (int r = 0; r < RH; ++r) {
    const int n = min(n0 + r, G.N - 1);
    w[r] = G.w + (size_t)n * rb; w[RH + r] = U.w + (size_t)n * rb;
    sc[r] = F == WFmt::NVFP4 ? reinterpret_cast<const uint8_t*>(G.s) + (size_t)n * (G.K / 16) : nullptr;
    sc[RH + r] = F == WFmt::NVFP4 ? reinterpret_cast<const uint8_t*>(U.s) + (size_t)n * (U.K / 16) : nullptr;
  }
  float res[R][M];
  if constexpr (kStage) rows_dot_smem<M, F, R>(w, sc, x, ldx, G.K, lane, active, xs, res);
  else rows_dot<M, F, R>(w, sc, x, ldx, G.K, lane, res);
  if (!active) return;
  if (lane == 0)
#pragma unroll
    for (int r = 0; r < RH; ++r) {
      const int n = n0 + r;
      if (n >= G.N) break;
      const float sg = row_scale(G, n), su = row_scale(U, n);
#pragma unroll
      for (int m = 0; m < M; ++m) act[(size_t)m * lda + n] = __float2bfloat16_rn(bfr(siluf_(bfr(res[r][m] * sg))) * bfr(res[RH + r][m] * su));
    }
}

// ---- tensor-core decode GEMM (2 ≤ M ≤ 16): mma.sync m16n8k16, weights dequantized to exact bf16 in shared memory ------------------------
// Warp = 8 output columns (weight rows n0 … n0 + 7) × all M activation rows (padded to 16). K in chunks: the warp copies its 8 rows' bytes
//   of the chunk to shared memory (16-byte loads), then each k-step of 16 builds the B fragment from shared memory and the A fragment from x.
//   GU: two weight matrices (gate, up) share the A fragments; the epilogue writes bf16(bf16(silu(bf16 g)) · bf16 u).
namespace tc {
constexpr int KC_BYTES = 256;   // bytes per row per chunk (FP8: 256 values · NVFP4: 512 · BF16: 128) — the GU stage stays under 48 KB static
__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&p);
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// values per byte and k values per chunk
template <WFmt F> constexpr int vals_per_byte() { return F == WFmt::NVFP4 ? 2 : F == WFmt::FP8ROW ? 1 : 0; }
template <WFmt F> constexpr int chunk_k() { return F == WFmt::NVFP4 ? 2 * KC_BYTES : F == WFmt::FP8ROW ? KC_BYTES : KC_BYTES / 2; }
// B fragment word (k pair kk, kk + 1) of the staged row r: bf16x2
template <WFmt F>
__device__ __forceinline__ uint32_t bword(const uint8_t* row, int kk, float gscale) {
  if constexpr (F == WFmt::FP8ROW) {
    const float2 t = fp8x2(*reinterpret_cast<const uint16_t*>(row + kk));
    return pack_bf2(t.x, t.y);
  } else if constexpr (F == WFmt::NVFP4) {
    const float2 t = fp4x2(row[kk / 2]);
    return pack_bf2(t.x * gscale, t.y * gscale);   // e2m1 · e4m3: exact in bf16
  } else {
    return *reinterpret_cast<const uint32_t*>(row + kk * 2);
  }
}
}  // namespace tc

template <WFmt F, bool GU>
__global__ void __launch_bounds__(256) tc_gemm_kernel(GMat W, GMat U, const bf16* __restrict__ x, int ldx, int M, bf16* yb, float* yf, int ldy) {
  using namespace tc;
  constexpr int NM = GU ? 2 : 1;
  __shared__ __align__(16) uint8_t stage[8][NM][8][KC_BYTES];   // [warp][matrix][row][bytes]
  __shared__ __align__(16) uint8_t sstage[8][NM][8][F == WFmt::NVFP4 ? KC_BYTES / 8 : 16];   // NVFP4 group scales of the chunk
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n0 = (blockIdx.x * 8 + warp) * 8;
  if (n0 >= W.N) return;
  const int K = W.K, CK = chunk_k<F>();
  const size_t rb = F == WFmt::BF16 ? (size_t)K * 2 : F == WFmt::FP8ROW ? (size_t)K : (size_t)K / 2;
  float acc[NM][4];
#pragma unroll
  for (int q = 0; q < NM; ++q) acc[q][0] = acc[q][1] = acc[q][2] = acc[q][3] = 0.f;
  const int ar = lane / 4, ac = (lane % 4) * 2;
  const bf16* x0 = x + (size_t)min(ar, M - 1) * ldx;
  const bf16* x1 = x + (size_t)min(ar + 8, M - 1) * ldx;
  const bool v0 = ar < M, v1 = ar + 8 < M;
  for (int k0 = 0; k0 < K; k0 += CK) {
    const int kc = min(CK, K - k0);
    const int cbytes = F == WFmt::BF16 ? kc * 2 : F == WFmt::FP8ROW ? kc : kc / 2;
    __syncwarp();
#pragma unroll
    for (int q = 0; q < NM; ++q) {
      const GMat& Wq = q ? U : W;
      for (int i = lane; i < 8 * (cbytes / 16); i += 32) {
        const int r = i / (cbytes / 16), c16 = i % (cbytes / 16);
        const int n = min(n0 + r, Wq.N - 1);
        const size_t boff = F == WFmt::BF16 ? (size_t)k0 * 2 : F == WFmt::FP8ROW ? (size_t)k0 : (size_t)k0 / 2;
        *reinterpret_cast<uint4*>(&stage[warp][q][r][c16 * 16]) = __ldg(reinterpret_cast<const uint4*>(Wq.w + (size_t)n * rb + boff) + c16);
      }
      if constexpr (F == WFmt::NVFP4)
        for (int i = lane; i < 8 * (kc / 16); i += 32) {
          const int r = i / (kc / 16), gi = i % (kc / 16);
          const int n = min(n0 + r, Wq.N - 1);
          sstage[warp][q][r][gi] = reinterpret_cast<const uint8_t*>(Wq.s)[(size_t)n * (K / 16) + k0 / 16 + gi];
        }
    }
    __syncwarp();
    for (int kk = 0; kk < kc; kk += 16) {
      uint32_t a[4];
      a[0] = v0 ? *reinterpret_cast<const uint32_t*>(x0 + k0 + kk + ac) : 0u;
      a[1] = v1 ? *reinterpret_cast<const uint32_t*>(x1 + k0 + kk + ac) : 0u;
      a[2] = v0 ? *reinterpret_cast<const uint32_t*>(x0 + k0 + kk + ac + 8) : 0u;
      a[3] = v1 ? *reinterpret_cast<const uint32_t*>(x1 + k0 + kk + ac + 8) : 0u;
#pragma unroll
      for (int q = 0; q < NM; ++q) {
        const uint8_t* row = stage[warp][q][ar];   // B column n = ar
        const float gs = F == WFmt::NVFP4 ? fp8x2((uint16_t)sstage[warp][q][ar][kk / 16]).x : 1.f;
        tc::mma(acc[q], a, tc::bword<F>(row, kk + ac, gs), tc::bword<F>(row, kk + ac + 8, gs));
      }
    }
  }
  // C fragment: rows ar (acc[0..1]) and ar + 8 (acc[2..3]), columns n0 + ac, n0 + ac + 1
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const int n = n0 + ac + e;
    if (n >= W.N) continue;
    if constexpr (GU) {
      const float sg = W.fmt == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[n] : W.fmt == WFmt::NVFP4 ? W.g : 1.f;
      const float su = U.fmt == WFmt::FP8ROW ? reinterpret_cast<const float*>(U.s)[n] : U.fmt == WFmt::NVFP4 ? U.g : 1.f;
      if (v0) yb[(size_t)ar * ldy + n] = __float2bfloat16_rn(bfr(siluf_(bfr(acc[0][e] * sg))) * bfr(acc[1][e] * su));
      if (v1) yb[(size_t)(ar + 8) * ldy + n] = __float2bfloat16_rn(bfr(siluf_(bfr(acc[0][2 + e] * sg))) * bfr(acc[1][2 + e] * su));
    } else {
      const float sc = W.fmt == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[n] : W.fmt == WFmt::NVFP4 ? W.g : 1.f;
      if (v0) { if (yb) yb[(size_t)ar * ldy + n] = __float2bfloat16_rn(acc[0][e] * sc); else yf[(size_t)ar * ldy + n] = acc[0][e] * sc; }
      if (v1) { if (yb) yb[(size_t)(ar + 8) * ldy + n] = __float2bfloat16_rn(acc[0][2 + e] * sc); else yf[(size_t)(ar + 8) * ldy + n] = acc[0][2 + e] * sc; }
    }
  }
}

// ---- tensor-core GEMM v2 (1 ≤ M ≤ 16): permuted-k fragments straight from global memory --------------------------------------------------
// A quad (4 lanes, same B column n = weight row n0 + lane/4) loads 4 × 16 consecutive bytes of the row; lane q's 16 bytes hold its B
//   fragment values for the next mma steps: step j uses the lane's elements [4j, 4j + 4) as the logical k pairs (2q, 2q+1) and (2q+8, 2q+9).
//   The activation (A) fragment takes x at the same physical positions, so Σ_k x·w is unchanged (k is only summed in another order).
//   FP8: 16 elements per lane → 4 steps per load · NVFP4: 32 elements (2 scale groups, applied to the B values: e2m1 · e4m3 exact in bf16)
//   → 8 steps · BF16: 8 elements → 2 steps. Block = 2 tiles of 8 rows × 4 K-slices (8 warps); slice sums added in slice order. The
//   arithmetic of a row does not depend on M (absent activation rows are zeros), so decode and verify agree bit for bit.
namespace tc2 {
template <WFmt F> constexpr int elems() { return F == WFmt::NVFP4 ? 32 : F == WFmt::FP8ROW ? 16 : 8; }   // per 16-byte load
}
template <WFmt F, bool GU>
__global__ void __launch_bounds__(256) tc2_kernel(GMat W, GMat U, const bf16* __restrict__ x, int ldx, int M, bf16* yb, float* yf, int ldy) {
  constexpr int E = tc2::elems<F>(), STEPS = E / 4, S = 4, NM = GU ? 2 : 1;
  __shared__ float part[2][S][NM][16][8];   // [tile][slice][matrix][activation row][column]
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, tl = warp / S, sl = warp % S;
  const int n0 = (blockIdx.x * 2 + tl) * 8;
  const int q = lane % 4, r = lane / 4;
  const int K = W.K, span = 4 * E;
  const int nsp = K / span;
  const int s0 = sl * nsp / S, s1 = (sl + 1) * nsp / S;
  const int nr = min(n0 + r, W.N - 1);
  const size_t rb = F == WFmt::BF16 ? (size_t)K * 2 : F == WFmt::FP8ROW ? (size_t)K : (size_t)K / 2;
  const uint8_t* wrow[NM];
  const uint8_t* srow[NM];
#pragma unroll
  for (int t = 0; t < NM; ++t) {
    const GMat& Wt = t ? U : W;
    wrow[t] = Wt.w + (size_t)nr * rb;
    srow[t] = F == WFmt::NVFP4 ? reinterpret_cast<const uint8_t*>(Wt.s) + (size_t)nr * (K / 16) : nullptr;
  }
  const bool a0v = r < M, a1v = r + 8 < M;
  const bf16* xa = x + (size_t)min(r, M - 1) * ldx;
  const bf16* xb = x + (size_t)min(r + 8, M - 1) * ldx;
  float acc[NM][4];
#pragma unroll
  for (int t = 0; t < NM; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
  constexpr int UN = GU ? 2 : 4;   // spans whose weight loads are issued together (memory-level parallelism)
  for (int sb = s0; sb < s1; sb += UN) {
    uint4 wvs[UN][NM];
    uint16_t gss[UN][NM];
#pragma unroll
    for (int u = 0; u < UN; ++u) {
      const int sp = min(sb + u, s1 - 1);
      const int e0 = sp * span + q * E;
#pragma unroll
      for (int t = 0; t < NM; ++t) {
        wvs[u][t] = __ldg(reinterpret_cast<const uint4*>(wrow[t] + (F == WFmt::BF16 ? (size_t)e0 * 2 : F == WFmt::FP8ROW ? (size_t)e0 : (size_t)e0 / 2)));
        if constexpr (F == WFmt::NVFP4) gss[u][t] = __ldg(reinterpret_cast<const uint16_t*>(srow[t] + e0 / 16));
      }
    }
#pragma unroll
    for (int u = 0; u < UN; ++u) {
    if (sb + u >= s1) break;
    const int e0 = (sb + u) * span + q * E;
    uint4 wv[NM];
    float gsc[NM][2];
#pragma unroll
    for (int t = 0; t < NM; ++t) {
      wv[t] = wvs[u][t];
      if constexpr (F == WFmt::NVFP4) { const float2 g2 = fp8x2(gss[u][t]); gsc[t][0] = g2.x; gsc[t][1] = g2.y; }
    }
#pragma unroll
    for (int j = 0; j < STEPS; ++j) {
      const int e = e0 + 4 * j;
      uint32_t a[4];
      const uint2 xa2 = a0v ? *reinterpret_cast<const uint2*>(xa + e) : make_uint2(0, 0);
      const uint2 xb2 = a1v ? *reinterpret_cast<const uint2*>(xb + e) : make_uint2(0, 0);
      a[0] = xa2.x; a[1] = xb2.x; a[2] = xa2.y; a[3] = xb2.y;
#pragma unroll
      for (int t = 0; t < NM; ++t) {
        uint32_t b0, b1;
        if constexpr (F == WFmt::FP8ROW) {
          const uint16_t* h = reinterpret_cast<const uint16_t*>(&wv[t]);
          const float2 u0 = fp8x2(h[2 * j]), u1 = fp8x2(h[2 * j + 1]);
          b0 = tc::pack_bf2(u0.x, u0.y); b1 = tc::pack_bf2(u1.x, u1.y);
        } else if constexpr (F == WFmt::NVFP4) {
          const uint8_t* by = reinterpret_cast<const uint8_t*>(&wv[t]);
          const float g = gsc[t][j / 4];
          const float2 u0 = fp4x2(by[2 * j]), u1 = fp4x2(by[2 * j + 1]);
          b0 = tc::pack_bf2(u0.x * g, u0.y * g); b1 = tc::pack_bf2(u1.x * g, u1.y * g);
        } else {
          const uint32_t* wd = reinterpret_cast<const uint32_t*>(&wv[t]);
          b0 = wd[2 * j]; b1 = wd[2 * j + 1];
        }
        tc::mma(acc[t], a, b0, b1);
      }
    }
    }
  }
#pragma unroll
  for (int t = 0; t < NM; ++t) {
    part[tl][sl][t][r][2 * q] = acc[t][0]; part[tl][sl][t][r][2 * q + 1] = acc[t][1];
    part[tl][sl][t][r + 8][2 * q] = acc[t][2]; part[tl][sl][t][r + 8][2 * q + 1] = acc[t][3];
  }
  __syncthreads();
  if (sl == 0) {
    for (int i = lane; i < 16 * 8; i += 32) {
      const int m = i / 8, c = i % 8, n = n0 + c;
      if (m >= M || n >= W.N) continue;
      float v[NM];
#pragma unroll
      for (int t = 0; t < NM; ++t) { v[t] = part[tl][0][t][m][c]; for (int z = 1; z < S; ++z) v[t] += part[tl][z][t][m][c]; }
      if constexpr (GU) {
        const float sg = row_scale(W, n), su = row_scale(U, n);
        yb[(size_t)m * ldy + n] = __float2bfloat16_rn(bfr(siluf_(bfr(v[0] * sg))) * bfr(v[1] * su));
      } else {
        const float sc = row_scale(W, n);
        if (yb) yb[(size_t)m * ldy + n] = __float2bfloat16_rn(v[0] * sc); else yf[(size_t)m * ldy + n] = v[0] * sc;
      }
    }
  }
}

// one block per row, the row's values in registers between the two passes (K ≤ 16 · blockDim)
__global__ void __launch_bounds__(512) add_norm_kernel(bf16* h, const bf16* y, const bf16* w, int K, float eps, bf16* out) {
  const int m = blockIdx.x;
  bf16* hr = h + (size_t)m * K;
  const bf16* yr = y + (size_t)m * K;
  __shared__ float red[32];
  float v[16], wv[16];
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {   // the norm weights are fetched with the rows (one memory round trip before the reduction)
    const int i = threadIdx.x + j * blockDim.x;
    wv[j] = i < K ? __bfloat162float(w[i]) : 0.f;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int i = threadIdx.x + j * blockDim.x;
    v[j] = 0.f;
    if (i < K) { v[j] = bfr(__bfloat162float(hr[i]) + __bfloat162float(yr[i])); hr[i] = __float2bfloat16_rn(v[j]); s += v[j] * v[j]; }
  }
  s = warp_sum(s);
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    t = warp_sum(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  const float r = 1.f / sqrtf(red[0] / (float)K + eps);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int i = threadIdx.x + j * blockDim.x;
    if (i < K) out[(size_t)m * K + i] = __float2bfloat16_rn(v[j] * r * (1.f + wv[j]));
  }
}

__global__ void dequant_kernel(GMat W, bf16* out) {
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 16, total = (size_t)W.N * W.K;
  if (i >= total) return;
  float v[16];
  if (W.fmt == WFmt::FP8ROW) {
    const uint4 q = *reinterpret_cast<const uint4*>(W.w + i);
    const uint16_t* b = reinterpret_cast<const uint16_t*>(&q);
#pragma unroll
    for (int j = 0; j < 8; ++j) { const float2 t = fp8x2(b[j]); v[2 * j] = t.x; v[2 * j + 1] = t.y; }
  } else if (W.fmt == WFmt::NVFP4) {
    const uint2 q = *reinterpret_cast<const uint2*>(W.w + i / 2);
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&q);
    const size_t n = i / W.K, kk = i % W.K;
    const float2 sc = fp8x2((uint16_t)reinterpret_cast<const uint8_t*>(W.s)[n * (W.K / 16) + kk / 16]);   // low byte = this group's scale
#pragma unroll
    for (int j = 0; j < 8; ++j) { const float2 t = fp4x2(b[j]); v[2 * j] = t.x * sc.x; v[2 * j + 1] = t.y * sc.x; }
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) v[j] = __bfloat162float(reinterpret_cast<const bf16*>(W.w)[i + j]);
  }
  uint4 o[2];
  uint32_t* ow = reinterpret_cast<uint32_t*>(o);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const __nv_bfloat162 p2 = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]);
    ow[j] = *reinterpret_cast<const uint32_t*>(&p2);
  }
  reinterpret_cast<uint4*>(out + i)[0] = o[0];
  reinterpret_cast<uint4*>(out + i)[1] = o[1];
}

__global__ void epilogue_kernel(GMat W, const float* acc, int lda, int M, bf16* yb, float* yf, int ldy) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * W.N) return;
  const int m = (int)(i / W.N), n = (int)(i % W.N);
  const float scale = W.fmt == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[n] : W.fmt == WFmt::NVFP4 ? W.g : 1.f;
  const float v = acc[(size_t)m * lda + n] * scale;
  if (yb) yb[(size_t)m * ldy + n] = __float2bfloat16_rn(v); else yf[(size_t)m * ldy + n] = v;
}

// ---- norms / elementwise -----------------------------------------------------------------------------------------------------------------
__global__ void rmsnorm1p_kernel(const bf16* x, int ldx, const bf16* w, int K, int group, float eps, bf16* out, int ldo) {
  const int m = blockIdx.y, g = blockIdx.x;
  const bf16* xr = x + (size_t)m * ldx + (size_t)g * group;
  __shared__ float red[32];
  float s = 0.f;
  for (int i = threadIdx.x; i < group; i += blockDim.x) { const float v = __bfloat162float(xr[i]); s += v * v; }
  s = warp_sum(s);
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    t = warp_sum(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  const float r = 1.f / sqrtf(red[0] / (float)group + eps);
  for (int i = threadIdx.x; i < group; i += blockDim.x)
    out[(size_t)m * ldo + (size_t)g * group + i] = __float2bfloat16_rn(__bfloat162float(xr[i]) * r * (1.f + __bfloat162float(w[(size_t)g * group + i])));
}

__global__ void add_round_kernel(bf16* h, const bf16* y, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) h[i] = __float2bfloat16_rn(__bfloat162float(h[i]) + __bfloat162float(y[i]));
}

__global__ void silu_mul_kernel(bf16* g, int ldg, const bf16* u, int ldu, int M, int I) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * I) return;
  const size_t m = i / I, j = i % I;
  const float a = bfr(siluf_(__bfloat162float(g[m * ldg + j])));
  g[m * ldg + j] = __float2bfloat16_rn(a * __bfloat162float(u[m * ldu + j]));
}

__global__ void attn_gate_kernel(const float* att, const bf16* proj, int ld, int M, int H, int D, bf16* out) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * H * D) return;
  const size_t m = i / ((size_t)H * D), hd = i % ((size_t)H * D), h = hd / D, d = hd % D;
  const float gt = __bfloat162float(proj[m * ld + h * 2 * D + D + d]);
  out[i] = __float2bfloat16_rn(bfr(att[i]) * sigmoidf_(gt));
}

// ---- attention -------------------------------------------------------------------------------------------------------------------------
// one block per row, one warp per head (norm, rope): q heads then (kv_only or not) k heads, v copy
__global__ void attn_prep_kernel(const AttnRow* rows, const bf16* proj, int ld, const bf16* q_norm, const bf16* k_norm, float eps, const int32_t* coords,
                                 const double* inv_freq, const int32_t* axis, int rot, int H, int Hkv, int D, bf16* q_out, bool kv_only,
                                 const float2* tab, int tab_n) {
  const int m = blockIdx.x, warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp, nw = blockDim.x / kWarp;
  const AttnRow R = rows[m];
  const bf16* pr = proj + (size_t)m * ld;
  const int half = rot / 2;
  __shared__ float cs[64], sn[64];
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    const int p = coords[m * 3 + axis[i]];
    if (tab && p >= 0 && p < tab_n) { const float2 t = tab[(size_t)p * half + i]; cs[i] = t.x; sn[i] = t.y; continue; }   // same values, precomputed
    double s, c;
    sincos((double)p * inv_freq[i], &s, &c);
    cs[i] = (float)c; sn[i] = (float)s;
  }
  __syncthreads();
  const int QW = 2 * H * D;
  for (int t = warp; t < H + Hkv; t += nw) {
    const bool isq = t < H;
    if (isq && kv_only) continue;
    const bf16* src = isq ? pr + (size_t)t * 2 * D : pr + QW + (size_t)(t - H) * D;
    const bf16* w = isq ? q_norm : k_norm;
    float v[8];   // D = 256: 8 per lane
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { v[i] = __bfloat162float(src[lane + 32 * i]); ss += v[i] * v[i]; }
    ss = warp_sum(ss);
    const float r = 1.f / sqrtf(ss / (float)D + eps);
    __shared__ float buf[8][256];
    float* b = buf[warp % 8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { const int d = lane + 32 * i; b[d] = bfr(v[i] * r * (1.f + __bfloat162float(w[d]))); }
    __syncwarp();
    float o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int d = lane + 32 * i;
      if (d < half) o[i] = bfr(b[d] * cs[d] - b[d + half] * sn[d]);
      else if (d < rot) o[i] = bfr(b[d] * cs[d - half] + b[d - half] * sn[d - half]);
      else o[i] = b[d];
    }
    __syncwarp();
    if (isq) {
#pragma unroll
      for (int i = 0; i < 8; ++i) q_out[((size_t)m * H + t) * D + lane + 32 * i] = __float2bfloat16_rn(o[i]);
    } else {
      const int g = t - H;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int d = lane + 32 * i;
        R.k[((size_t)R.pos * Hkv + g) * D + d] = __float2bfloat16_rn(o[i]);
        R.v[((size_t)R.pos * Hkv + g) * D + d] = pr[QW + (size_t)Hkv * D + (size_t)g * D + d];
      }
    }
  }
}

// one block (256 threads = D) per (row, head): tiles of 32 keys, scores by 8 warps (4 keys each), online softmax, thread d accumulates out[d]
__global__ void attention_kernel(const AttnRow* rows, const bf16* q, int H, int Hkv, int D, float scale, float* out) {
  const int m = blockIdx.x / H, h = blockIdx.x % H, g = h / (H / Hkv);
  const AttnRow R = rows[m];
  const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp, d = threadIdx.x;
  __shared__ float qs[256], sc[32];
  qs[d] = __bfloat162float(q[((size_t)m * H + h) * D + d]);
  __syncthreads();
  float mx = -INFINITY, sum = 0.f, acc = 0.f;
  for (int j0 = 0; j0 < R.n_ctx; j0 += 32) {
    const int nt = min(32, R.n_ctx - j0);
    for (int jj = warp; jj < nt; jj += 8) {
      const bf16* kr = R.k + ((size_t)(j0 + jj) * Hkv + g) * D;
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) s += qs[lane + 32 * i] * __bfloat162float(kr[lane + 32 * i]);
      s = warp_sum(s);
      if (lane == 0) sc[jj] = s * scale;
    }
    __syncthreads();
    float tm = -INFINITY;
    for (int jj = 0; jj < nt; ++jj) tm = fmaxf(tm, sc[jj]);
    const float nm = fmaxf(mx, tm);
    const float corr = mx == -INFINITY ? 0.f : expf(mx - nm);
    acc *= corr; sum *= corr;
    for (int jj = 0; jj < nt; ++jj) {
      const float p = expf(sc[jj] - nm);
      sum += p;
      acc = fmaf(p, __bfloat162float(R.v[((size_t)(j0 + jj) * Hkv + g) * D + d]), acc);
    }
    mx = nm;
    __syncthreads();
  }
  out[((size_t)m * H + h) * D + d] = sum > 0.f ? acc / sum : 0.f;
}

// ---- prefill attention (mma.sync m16n8k16, bf16 in, fp32 accumulate) ---------------------------------------------------------------------
namespace fa {
constexpr int BQ = 64, BK = 32, D = 256, LDS = D + 8;   // padded row stride (bf16) against bank conflicts
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t ld32(const bf16* p) { return *reinterpret_cast<const uint32_t*>(p); }
}  // namespace fa

// grid (ceil(M / 64), H), 256 threads. smem: Q [64][264] · K [32][264] · V [32][264] bf16 · P [64][40] bf16 · row state
__global__ void __launch_bounds__(256) attention_prefill_kernel(const bf16* __restrict__ q, int M, int pos0, const bf16* __restrict__ K,
                                                                 const bf16* __restrict__ V, int H, int Hkv, float scale, float* out) {
  using namespace fa;
  extern __shared__ __align__(16) unsigned char smem[];
  bf16* Qs = reinterpret_cast<bf16*>(smem);
  bf16* Ks = Qs + BQ * LDS;
  bf16* Vs = Ks + BK * LDS;
  bf16* Ps = Vs + BK * LDS;                 // [64][BK + 8]
  float* rmax = reinterpret_cast<float*>(Ps + BQ * (BK + 8));
  float* rsum = rmax + BQ;
  float* rcor = rsum + BQ;
  const int h = blockIdx.y, g = h / (H / Hkv), q0 = blockIdx.x * BQ;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, tid = threadIdx.x;
  const int nq = min(BQ, M - q0);
  // Q tile
  for (int i = tid; i < BQ * (D / 8); i += 256) {
    const int r = i / (D / 8), c8 = (i % (D / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < nq) v = *reinterpret_cast<const uint4*>(q + ((size_t)(q0 + r) * H + h) * D + c8);
    *reinterpret_cast<uint4*>(Qs + r * LDS + c8) = v;
  }
  if (tid < BQ) { rmax[tid] = -INFINITY; rsum[tid] = 0.f; rcor[tid] = 1.f; }
  // P·V accumulators: warp w → rows 16·(w % 4) … +15, d half (w / 4) · 128: 16 n-tiles of 8 → 64 floats
  float acc[16][4];
#pragma unroll
  for (int t = 0; t < 16; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
  const int pr0 = 16 * (warp % 4), dh = (warp / 4) * 128;
  const int last_pos = pos0 + q0 + nq - 1;               // the tile's last query position: keys [0, last_pos]
  for (int k0 = 0; k0 <= last_pos; k0 += BK) {
    __syncthreads();
    for (int i = tid; i < BK * (D / 8); i += 256) {
      const int r = i / (D / 8), c8 = (i % (D / 8)) * 8, kp = k0 + r;
      uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
      if (kp <= last_pos) {
        kv = *reinterpret_cast<const uint4*>(K + ((size_t)kp * Hkv + g) * D + c8);
        vv = *reinterpret_cast<const uint4*>(V + ((size_t)kp * Hkv + g) * D + c8);
      }
      *reinterpret_cast<uint4*>(Ks + r * LDS + c8) = kv;
      *reinterpret_cast<uint4*>(Vs + r * LDS + c8) = vv;
    }
    __syncthreads();
    if (warp < 4) {   // S = Q · Kᵀ for rows 16·warp … +15, keys k0 … k0 + 31 (4 n-tiles of 8)
      float s[4][4];
#pragma unroll
      for (int t = 0; t < 4; ++t) s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.f;
      const int ar = 16 * warp + lane / 4, ac = (lane % 4) * 2;
#pragma unroll 4
      for (int kk = 0; kk < D; kk += 16) {
        uint32_t a[4] = {ld32(Qs + ar * LDS + kk + ac), ld32(Qs + (ar + 8) * LDS + kk + ac), ld32(Qs + ar * LDS + kk + ac + 8),
                         ld32(Qs + (ar + 8) * LDS + kk + ac + 8)};
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const int bn = t * 8 + lane / 4;   // key row in the tile
          uint32_t b[2] = {ld32(Ks + bn * LDS + kk + ac), ld32(Ks + bn * LDS + kk + ac + 8)};
          mma16816(s[t], a, b);
        }
      }
      // online softmax per row: this lane holds rows ar and ar + 8, columns t·8 + ac + {0, 1}
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int r = ar + 8 * half, qp = pos0 + q0 + r;
        float mx = -INFINITY;
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int kp = k0 + t * 8 + ac + e;
            float v = s[t][2 * half + e] * scale;
            if (kp > qp || r >= nq) v = -INFINITY;
            s[t][2 * half + e] = v;
            mx = fmaxf(mx, v);
          }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float om = rmax[r];
        const float nm = fmaxf(om, mx);
        float ps = 0.f;
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float p = nm == -INFINITY ? 0.f : expf(s[t][2 * half + e] - nm);
            ps += p;
            Ps[r * (BK + 8) + t * 8 + ac + e] = __float2bfloat16_rn(p);
          }
        ps += __shfl_xor_sync(0xffffffffu, ps, 1);
        ps += __shfl_xor_sync(0xffffffffu, ps, 2);
        __syncwarp();
        if (lane % 4 == 0) {
          const float cor = om == -INFINITY ? 0.f : expf(om - nm);
          rcor[r] = cor; rsum[r] = rsum[r] * cor + ps; rmax[r] = nm;
        }
      }
    }
    __syncthreads();
    // O = O · cor + P · V  (rows pr0 … +15, d in [dh, dh + 128))
    {
      const int ar = pr0 + lane / 4, ac = (lane % 4) * 2;
      const float c0 = rcor[ar], c1 = rcor[ar + 8];
#pragma unroll
      for (int t = 0; t < 16; ++t) { acc[t][0] *= c0; acc[t][1] *= c0; acc[t][2] *= c1; acc[t][3] *= c1; }
#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        uint32_t a[4] = {ld32(Ps + ar * (BK + 8) + kk + ac), ld32(Ps + (ar + 8) * (BK + 8) + kk + ac), ld32(Ps + ar * (BK + 8) + kk + ac + 8),
                         ld32(Ps + (ar + 8) * (BK + 8) + kk + ac + 8)};
#pragma unroll
        for (int t = 0; t < 16; ++t) {
          const int dn = dh + t * 8 + lane / 4;   // output column; B[k][n] = V[k0 + kk + k][dn]
          const bf16* v0 = Vs + (kk + ac) * LDS + dn;
          const bf16* v1 = Vs + (kk + ac + 8) * LDS + dn;
          __nv_bfloat162 b0, b1;
          b0.x = v0[0]; b0.y = v0[LDS];
          b1.x = v1[0]; b1.y = v1[LDS];
          uint32_t b[2] = {*reinterpret_cast<uint32_t*>(&b0), *reinterpret_cast<uint32_t*>(&b1)};
          mma16816(acc[t], a, b);
        }
      }
    }
  }
  __syncthreads();
  {
    const int ar = pr0 + lane / 4, ac = (lane % 4) * 2;
    const float l0 = rsum[ar], l1 = rsum[ar + 8];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      const int d = dh + t * 8 + ac;
      if (ar < nq) { float* o = out + ((size_t)(q0 + ar) * H + h) * D + d; o[0] = acc[t][0] / l0; o[1] = acc[t][1] / l0; }
      if (ar + 8 < nq) { float* o = out + ((size_t)(q0 + ar + 8) * H + h) * D + d; o[0] = acc[t][2] / l1; o[1] = acc[t][3] / l1; }
    }
  }
}

// ---- Gated DeltaNet ----------------------------------------------------------------------------------------------------------------------
// conv + SiLU: thread per channel of one segment (rows sequential inside)
__global__ void gdn_conv_kernel(const GdnSeg* segs, const bf16* proj, int ld, const float* cw, int C, bf16* y) {
  const GdnSeg S = segs[blockIdx.y];
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= C) return;
  float h0 = S.conv[ch], h1 = S.conv[C + ch], h2 = S.conv[2 * C + ch];
  const float w0 = cw[ch * 4], w1 = cw[ch * 4 + 1], w2 = cw[ch * 4 + 2], w3 = cw[ch * 4 + 3];
  for (int t = 0; t < S.n; ++t) {
    const float x = __bfloat162float(proj[(size_t)(S.row0 + t) * ld + ch]);
    float v = w0 * h0;
    v += w1 * h1; v += w2 * h2; v += w3 * x;
    y[(size_t)(S.row0 + t) * C + ch] = __float2bfloat16_rn(siluf_(v));
    h0 = h1; h1 = h2; h2 = x;
    if (S.snap_conv && t + 1 < S.n) { float* sc = S.snap_conv[t]; sc[ch] = h0; sc[C + ch] = h1; sc[2 * C + ch] = h2; }
  }
  S.conv[ch] = h0; S.conv[C + ch] = h1; S.conv[2 * C + ch] = h2;
}

// the same conv with the rows split over blockIdx.z (prefill): x[t − j] from the rows or, before the segment, from the conv state; the
//   state itself is advanced afterwards by gdn_conv_state_kernel (every reader of the old state has finished by then)
__global__ void gdn_conv_par_kernel(const GdnSeg* segs, const bf16* proj, int ld, const float* cw, int C, bf16* y) {
  const GdnSeg S = segs[blockIdx.y];
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= C) return;
  const int per = (S.n + gridDim.z - 1) / gridDim.z, t0 = blockIdx.z * per, t1 = min(S.n, t0 + per);
  if (t0 >= t1) return;
  const float w0 = cw[ch * 4], w1 = cw[ch * 4 + 1], w2 = cw[ch * 4 + 2], w3 = cw[ch * 4 + 3];
  auto xin = [&](int t) { return t >= 0 ? __bfloat162float(proj[(size_t)(S.row0 + t) * ld + ch]) : S.conv[(3 + t) * C + ch]; };
  float h0 = xin(t0 - 3), h1 = xin(t0 - 2), h2 = xin(t0 - 1);
  for (int t = t0; t < t1; ++t) {
    const float x = xin(t);
    float v = w0 * h0;
    v += w1 * h1; v += w2 * h2; v += w3 * x;
    y[(size_t)(S.row0 + t) * C + ch] = __float2bfloat16_rn(siluf_(v));
    h0 = h1; h1 = h2; h2 = x;
    if (S.snap_conv && t + 1 < S.n) { float* sc = S.snap_conv[t]; sc[ch] = h0; sc[C + ch] = h1; sc[2 * C + ch] = h2; }
  }
}
__global__ void gdn_conv_state_kernel(const GdnSeg* segs, const bf16* proj, int ld, int C) {
  const GdnSeg S = segs[blockIdx.y];
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= C) return;
  float h[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) { const int t = S.n - 3 + j; h[j] = t >= 0 ? __bfloat162float(proj[(size_t)(S.row0 + t) * ld + ch]) : S.conv[(3 + t) * C + ch]; }
  S.conv[ch] = h[0]; S.conv[C + ch] = h[1]; S.conv[2 * C + ch] = h[2];
}

// q, k l2 norm (q · 128^−0.5) per (row, key head): qk [rows, 2·KW]; decay / beta per (row, value head): gb [rows, 2·VH]
__global__ void gdn_prep_kernel(const bf16* y, int C, int KH, int VH, const bf16* proj, int ld, int bo, const float* A_log, const float* dt_bias, float* qk,
                                float* gb) {
  const int r = blockIdx.x, warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp, nw = blockDim.x / kWarp;
  const int KW = KH * 128;
  for (int t = warp; t < 2 * KH; t += nw) {
    const bf16* src = y + (size_t)r * C + (size_t)t * 128;   // q heads then k heads
    float v[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[i] = __bfloat162float(src[lane + 32 * i]); ss += v[i] * v[i]; }
    ss = warp_sum(ss);
    const float nrm = sqrtf(ss + 1e-6f), qs = t < KH ? 1.f / sqrtf(128.f) : 1.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) qk[(size_t)r * 2 * KW + (size_t)t * 128 + lane + 32 * i] = v[i] / nrm * qs;
  }
  for (int vh = threadIdx.x; vh < VH; vh += blockDim.x) {
    const float b = __bfloat162float(proj[(size_t)r * ld + bo + vh]), a = __bfloat162float(proj[(size_t)r * ld + bo + VH + vh]);
    float sp = a + dt_bias[vh];
    sp = sp > 20.f ? sp : log1pf(expf(sp));
    gb[(size_t)r * 2 * VH + vh] = expf(-expf(A_log[vh]) * sp);
    gb[(size_t)r * 2 * VH + VH + vh] = bfr(1.f / (1.f + expf(-b)));
  }
}

// recurrence: block = (segment, value head, 8 state columns), one warp; lane = 4 · c + kq holds rows [32 kq, 32 kq + 32) of state column
//   j = 8 · blockIdx.z + c in registers. A row's k-reductions: 4 interleaved partial sums over the lane's 32 rows (i ≡ 0..3 mod 4, then
//   (p0 + p1) + (p2 + p3)), then the 4 quarters by butterfly (xor 1, xor 2) — the same order for every launch shape (decode, verify,
//   prefill and the rollback replay agree). The next row's q / k / gate / beta / v are fetched while the current row is applied.
//   Verify (snap_state set, HIVE_Q27_VSNAP=1): the state before the segment's first row is copied to snap_state[0] (rollback replays the kept
//   rows from it). Verify default (keep_state): the state is only read; the rollback replays the kept rows from it in place.
__device__ __forceinline__ float gdn_row(float (&col)[32], const float* kr, const float* qr, float e, float bt, float v) {
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
  for (int i = 0; i < 32; i += 4) {
    col[i] *= e; col[i + 1] *= e; col[i + 2] *= e; col[i + 3] *= e;
    a0 = fmaf(col[i], kr[i], a0); a1 = fmaf(col[i + 1], kr[i + 1], a1); a2 = fmaf(col[i + 2], kr[i + 2], a2); a3 = fmaf(col[i + 3], kr[i + 3], a3);
  }
  float kv = (a0 + a1) + (a2 + a3);
  kv += __shfl_xor_sync(0xffffffffu, kv, 1);
  kv += __shfl_xor_sync(0xffffffffu, kv, 2);
  const float dl = (v - kv) * bt;
  a0 = a1 = a2 = a3 = 0.f;
#pragma unroll
  for (int i = 0; i < 32; i += 4) {
    col[i] = fmaf(kr[i], dl, col[i]); col[i + 1] = fmaf(kr[i + 1], dl, col[i + 1]);
    col[i + 2] = fmaf(kr[i + 2], dl, col[i + 2]); col[i + 3] = fmaf(kr[i + 3], dl, col[i + 3]);
    a0 = fmaf(col[i], qr[i], a0); a1 = fmaf(col[i + 1], qr[i + 1], a1); a2 = fmaf(col[i + 2], qr[i + 2], a2); a3 = fmaf(col[i + 3], qr[i + 3], a3);
  }
  float oo = (a0 + a1) + (a2 + a3);
  oo += __shfl_xor_sync(0xffffffffu, oo, 1);
  oo += __shfl_xor_sync(0xffffffffu, oo, 2);
  return oo;
}
// rows [row0, row0 + n) of one (state, value head, column block): q / k fp32 rows (stride 2·KW, q first), gate / beta fp32 (stride 2·VH),
//   v bf16 (stride C, at 2·KW); outputs o (nullptr: none)
__device__ __forceinline__ void gdn_rows(float (&col)[32], int row0, int n, const bf16* y, int C, int KH, int VH, const float* qk, const float* gb,
                                         float* o, int vh, int j) {
  const int lane = threadIdx.x, kq = lane & 3, rep = VH / KH, kh = vh / rep, KW = KH * 128, r0 = kq * 32;
  __shared__ float ks[2][128], qs[2][128];
  float pq[4], pk[4], pe = 0.f, pb = 0.f, pv = 0.f;
  auto fetch = [&](int t) {
    const size_t row = (size_t)row0 + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      pq[i] = qk[row * 2 * KW + (size_t)kh * 128 + lane + 32 * i];
      pk[i] = qk[row * 2 * KW + KW + (size_t)kh * 128 + lane + 32 * i];
    }
    pe = gb[row * 2 * VH + vh]; pb = gb[row * 2 * VH + VH + vh];
    pv = __bfloat162float(y[row * C + 2 * KW + (size_t)vh * 128 + j]);
  };
  if (n > 0) fetch(0);
  for (int t = 0; t < n; ++t) {
    const int bsel = t & 1;
#pragma unroll
    for (int i = 0; i < 4; ++i) { qs[bsel][lane + 32 * i] = pq[i]; ks[bsel][lane + 32 * i] = pk[i]; }
    const float e = pe, bt = pb, v = pv;
    __syncwarp();
    if (t + 1 < n) fetch(t + 1);
    const float oo = gdn_row(col, ks[bsel] + r0, qs[bsel] + r0, e, bt, v);
    if (o && kq == 0) o[((size_t)row0 + t) * (size_t)VH * 128 + (size_t)vh * 128 + j] = bfr(oo);
  }
}
__global__ void __launch_bounds__(32) gdn_rec_kernel(const GdnSeg* segs, const bf16* y, int C, int KH, int VH, const float* qk, const float* gb, float* o) {
  const GdnSeg S = segs[blockIdx.z];   // grid (column block, value head, segment): neighbouring blocks share one head's 64 KiB state
  const int vh = blockIdx.y, lane = threadIdx.x, j = blockIdx.x * 8 + (lane >> 2), r0 = (lane & 3) * 32;
  float* st = S.state + (size_t)vh * 128 * 128;
  float col[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) col[i] = st[(size_t)(r0 + i) * 128 + j];
  if (S.snap_state) {
    float* sp = S.snap_state[0] + (size_t)vh * 128 * 128;
#pragma unroll
    for (int i = 0; i < 32; ++i) sp[(size_t)(r0 + i) * 128 + j] = col[i];
  }
  gdn_rows(col, S.row0, S.n, y, C, KH, VH, qk, gb, o, vh, j);
  if (S.keep_state) return;   // verify without snapshot: the state stays as it was before the verify (the rollback replays from it)
#pragma unroll
  for (int i = 0; i < 32; ++i) st[(size_t)(r0 + i) * 128 + j] = col[i];
}
// rollback replay: the state from `init` through `n` kept rows of the verify (inputs saved per layer) into `state`
__global__ void __launch_bounds__(32) gdn_replay_kernel(const GdnReplay* items, int C, int KH, int VH) {
  // grid (column block, value head, item): neighbouring blocks read / write one head's contiguous 64 KiB (measured 2026-10-10, nsys S4 k7:
  //   with the item as blockIdx.x the 192-item replay ran at ~0.35 TB/s, 3.3 ms per round)
  const GdnReplay R = items[blockIdx.z];
  const int vh = blockIdx.y, lane = threadIdx.x, j = blockIdx.x * 8 + (lane >> 2), r0 = (lane & 3) * 32;
  const float* src = R.init + (size_t)vh * 128 * 128;
  float col[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) col[i] = src[(size_t)(r0 + i) * 128 + j];
  gdn_rows(col, R.row0, R.n, R.y, C, KH, VH, R.qk, R.gb, nullptr, vh, j);
  float* dst = R.state + (size_t)vh * 128 * 128;
#pragma unroll
  for (int i = 0; i < 32; ++i) dst[(size_t)(r0 + i) * 128 + j] = col[i];
}

// gated norm with a SiLU gate per (row, value head): 128 threads
__global__ void gdn_norm_kernel(const float* o, const bf16* proj, int ld, int zo, const bf16* w, float eps, int VH, bf16* out) {
  const int r = blockIdx.x, vh = blockIdx.y, j = threadIdx.x;
  const float v = o[(size_t)r * VH * 128 + (size_t)vh * 128 + j];
  __shared__ float red[4];
  float s = warp_sum(v * v);
  if (j % 32 == 0) red[j / 32] = s;
  __syncthreads();
  const float tot = red[0] + red[1] + red[2] + red[3];
  const float rs = 1.f / sqrtf(tot / 128.f + eps);
  const float z = __bfloat162float(proj[(size_t)r * ld + zo + (size_t)vh * 128 + j]);
  const float a = bfr(v * rs), b = bfr(a * __bfloat162float(w[j]));
  out[(size_t)r * VH * 128 + (size_t)vh * 128 + j] = __float2bfloat16_rn(b * siluf_(z));
}

// two stages: 64 blocks each reduce a slice to (max, index, Σ exp(l − slice max)); one block combines them (ties → smaller index)
constexpr int kAmBlocks = 64;
__device__ __forceinline__ void am_pair(float& bv, int& bi, float ov, int oi) { if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
__global__ void __launch_bounds__(256) argmax_part_kernel(const float* lg, int V, float* part) {
  __shared__ float sv[8]; __shared__ int si[8]; __shared__ float ss[8];
  const int per = (V + gridDim.x - 1) / gridDim.x, a = blockIdx.x * per, e = min(V, a + per);
  float bv = -INFINITY; int bi = 0x7fffffff;
  for (int i = a + threadIdx.x; i < e; i += blockDim.x) am_pair(bv, bi, lg[i], i);
  for (int o = 16; o > 0; o >>= 1) am_pair(bv, bi, __shfl_xor_sync(0xffffffffu, bv, o), __shfl_xor_sync(0xffffffffu, bi, o));
  const int w = threadIdx.x / 32, ln = threadIdx.x % 32;
  if (ln == 0) { sv[w] = bv; si[w] = bi; }
  __syncthreads();
  if (threadIdx.x == 0) { for (int i = 1; i < 8; ++i) am_pair(bv, bi, sv[i], si[i]); sv[0] = bv; si[0] = bi; }
  __syncthreads();
  const float mx = sv[0];
  float s = 0.f;
  for (int i = a + threadIdx.x; i < e; i += blockDim.x) s += expf(lg[i] - mx);
  s = warp_sum(s);
  if (ln == 0) ss[w] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int i = 0; i < 8; ++i) t += ss[i];
    part[blockIdx.x * 3] = mx; part[blockIdx.x * 3 + 1] = __int_as_float(si[0]); part[blockIdx.x * 3 + 2] = t;
  }
}
__global__ void argmax_final_kernel(const float* part, int n, float* out) {
  if (threadIdx.x != 0) return;
  float bv = -INFINITY; int bi = 0x7fffffff;
  for (int i = 0; i < n; ++i) am_pair(bv, bi, part[i * 3], __float_as_int(part[i * 3 + 1]));
  float t = 0.f;
  for (int i = 0; i < n; ++i) if (part[i * 3] > -INFINITY) t += part[i * 3 + 2] * expf(part[i * 3] - bv);
  out[0] = __int_as_float(bi); out[1] = 1.f / t;
}

__global__ void embed_by_id_kernel(const bf16* table, const int32_t* id_ptr, int dim, bf16* out) {
  const int id = *id_ptr;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dim / 8; i += blockDim.x * gridDim.x)
    reinterpret_cast<uint4*>(out)[i] = reinterpret_cast<const uint4*>(table + (size_t)id * dim)[i];
}

__global__ void embed_rows_kernel(const bf16* table, const int32_t* ids, int dim, bf16* out) {
  const int r = blockIdx.y;
  const uint4* src = reinterpret_cast<const uint4*>(table + (size_t)ids[r] * dim);
  uint4* dst = reinterpret_cast<uint4*>(out + (size_t)r * dim);
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dim / 8; i += blockDim.x * gridDim.x) dst[i] = src[i];
}

__global__ void map_ids_kernel(int32_t* id, const int32_t* map) { *id = map[*id]; }

__global__ void b2f_kernel(const bf16* x, size_t n, float* y) { const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = __bfloat162float(x[i]); }
__global__ void f2b_kernel(const float* x, size_t n, bf16* y) { const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = __float2bfloat16_rn(x[i]); }

inline unsigned blocks(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }
}  // namespace

// rows per warp: activation loads are shared by R rows (register budget: R·M accumulators + 8·M activation values)
template <int M> constexpr int rows_per_warp() { return M <= 4 ? 4 : 2; }
template <int M, int R>
void gemv_mr(const GMat& W, const bf16* x, int ldx, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  const int rpb = 8 * R;
  dim3 grid((W.N + rpb - 1) / rpb), block(256);
  if (W.fmt == WFmt::BF16) gemv_kernel<M, WFmt::BF16, R><<<grid, block, 0, st>>>(W, x, ldx, yb, yf, ldy);
  else if (W.fmt == WFmt::FP8ROW) gemv_kernel<M, WFmt::FP8ROW, R><<<grid, block, 0, st>>>(W, x, ldx, yb, yf, ldy);
  else gemv_kernel<M, WFmt::NVFP4, R><<<grid, block, 0, st>>>(W, x, ldx, yb, yf, ldy);
}
// Rows per warp: the most (≤ the register-safe maximum for M) that still gives ≥ 2 blocks per SM — x reuse without leaving SMs idle
//   (measured 2026-10-10: 4 rows per warp gave the 5120-row down projection 160 blocks on 188 SMs, 1.26 TB/s). HIVE_Q27_GEMV_R1 forces M = 1.
int sm_count() { static int n = 0; if (!n) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); if (n <= 0) n = 1; } return n; }
int pick_rows(int N, int rmax) {
  static const int forced = getenv("HIVE_Q27_GEMV_R1") ? atoi(getenv("HIVE_Q27_GEMV_R1")) : 0;
  (void)forced;
  int r = rmax;
  while (r > 1 && (N + 8 * r - 1) / (8 * r) < 2 * sm_count()) r /= 2;
  return r;
}
int gemv_r1() { static const int r = getenv("HIVE_Q27_GEMV_R1") ? atoi(getenv("HIVE_Q27_GEMV_R1")) : 0; return r; }
int splitk_maxn() { static const int v = getenv("HIVE_Q27_SPLITK_N") ? atoi(getenv("HIVE_Q27_SPLITK_N")) : 0; return v; }   // off: measured 2026-10-10 slower (M=4 20.1 → 22.3 ms)
template <int M>
void gemv_m(const GMat& W, const bf16* x, int ldx, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  if (W.N <= splitk_maxn()) {   // HIVE_Q27_SPLITK_N=0 turns it off
    dim3 grid((W.N + 1) / 2), block(256);
    if (W.fmt == WFmt::BF16) gemv_splitk_kernel<M, WFmt::BF16><<<grid, block, 0, st>>>(W, x, ldx, yb, yf, ldy);
    else if (W.fmt == WFmt::FP8ROW) gemv_splitk_kernel<M, WFmt::FP8ROW><<<grid, block, 0, st>>>(W, x, ldx, yb, yf, ldy);
    else gemv_splitk_kernel<M, WFmt::NVFP4><<<grid, block, 0, st>>>(W, x, ldx, yb, yf, ldy);
    return;
  }
  // adaptive rows only for M = 1 (measured 2026-10-10: at M = 4 fewer rows per warp lost the activation reuse, 19.6 → 21.9 ms/step)
  int r = M == 1 ? pick_rows(W.N, rows_per_warp<M>()) : rows_per_warp<M>();
  if (M == 1 && gemv_r1() > 0) r = gemv_r1();
  if (r >= 4 && rows_per_warp<M>() >= 4) { gemv_mr<M, (rows_per_warp<M>() >= 4 ? 4 : 1)>(W, x, ldx, yb, yf, ldy, st); return; }
  if (r >= 2) { gemv_mr<M, 2>(W, x, ldx, yb, yf, ldy, st); return; }
  gemv_mr<M, 1>(W, x, ldx, yb, yf, ldy, st);
}
// rows from which the tensor-core GEMM replaces the GEMV (HIVE_Q27_TC_MIN; it must cover every row count to keep decode == verify bit for bit)
int tc_min() { static const int v = getenv("HIVE_Q27_TC_MIN") ? atoi(getenv("HIVE_Q27_TC_MIN")) : 17; return v; }   // off by default: measured 2026-10-10 2.5x slower than the GEMV path (40.1 vs 15.7 ms/step at M=1)
template <bool GU>
void tc_launch(const GMat& W, const GMat& U, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  const int span = 4 * (W.fmt == WFmt::NVFP4 ? 32 : W.fmt == WFmt::FP8ROW ? 16 : 8);
  HIVE_CHECK(W.K % (4 * span) == 0 && ldx % 8 == 0, "q27 tc2: K must be a multiple of 4 spans");
  dim3 grid((W.N + 15) / 16), block(256);
  if (W.fmt == WFmt::BF16) tc2_kernel<WFmt::BF16, GU><<<grid, block, 0, st>>>(W, U, x, ldx, M, yb, yf, ldy);
  else if (W.fmt == WFmt::FP8ROW) tc2_kernel<WFmt::FP8ROW, GU><<<grid, block, 0, st>>>(W, U, x, ldx, M, yb, yf, ldy);
  else tc2_kernel<WFmt::NVFP4, GU><<<grid, block, 0, st>>>(W, U, x, ldx, M, yb, yf, ldy);
  CUDA_CHECK(cudaGetLastError());
}
// ---- swapped tensor-core GEMV (HIVE_Q27_SW, 1 ≤ M ≤ 16): weights are the mma's A operand (16 rows per tile), the activation rows its B
//   operand (8 tokens per n-tile) — mma.m16n8k16 bf16, fp32 accumulation. A lane (g = lane / 4, t = lane % 4) loads 16 consecutive bytes
//   of weight rows g and g + 8 of each tile per span (span = 4 lanes × E elements); in step j it uses its elements [4j, 4j + 4) as the
//   logical k pairs (2t, 2t+1) and (2t+8, 2t+9), and takes the activations at the same physical positions (so Σ_k x·w is unchanged, only the
//   order differs). Weight decode: cvt e4m3x2 / e2m1x2 → f16x2 (exact; NVFP4 · group scale in f16, exact), f16 → bf16 by bit moves (exact:
//   at most 7 mantissa bits, never subnormal). The activations of a span are loaded once and reused by both tiles. Block = 8 warps splitting
//   the spans of K; their sums are added in warp order. A token's arithmetic does not depend on M (absent tokens are zero columns), so
//   decode, verify and batch rows agree bit for bit.
namespace sw {
template <WFmt F> constexpr int elems() { return F == WFmt::NVFP4 ? 32 : F == WFmt::FP8ROW ? 16 : 8; }
constexpr int kSplit = 8;
__device__ __forceinline__ uint32_t h2_bf2(uint32_t h) {
  const uint32_t s = h & 0x80008000u, a = h & 0x7fff7fffu;
  const uint32_t v = ((a >> 3) & 0x0fff0fffu) + 0x38003800u;
  return (v & __vcmpne2(a, 0u)) | s;
}
__device__ __forceinline__ uint32_t fp8_bf2(uint16_t b) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)b, __NV_E4M3);
  return h2_bf2(*reinterpret_cast<const uint32_t*>(&h));
}
__device__ __forceinline__ uint32_t fp4_bf2(uint8_t b, __half2 sc) {
  const __half2_raw h = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b, __NV_E2M1);
  const __half2 p = __hmul2(*reinterpret_cast<const __half2*>(&h), sc);
  return h2_bf2(*reinterpret_cast<const uint32_t*>(&p));
}
__device__ __forceinline__ __half2 sc_h2(uint8_t s) {   // e4m3 group scale → f16 (exact), broadcast
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(uint16_t)s, __NV_E4M3);
  const __half lo = *reinterpret_cast<const __half*>(&h.x);
  return __half2half2(lo);
}
}  // namespace sw

template <WFmt F, bool GU, int NT, int TLX, int UN, int SPL = 8, int MINB = 1, bool JIT = false>
__global__ void __launch_bounds__(32 * SPL, MINB) sw_kernel(GMat W, GMat U, const bf16* __restrict__ x, int ldx, int M, bf16* yb, float* yf, int ldy) {
  // TLX: 16-row weight tiles per warp (GU: always 2 = gate and up of the same 16 rows); UN: spans per batch; the next batch's weights and
  //   activations are loaded while the current one is multiplied
  constexpr int E = sw::elems<F>(), STEPS = E / 4, S = SPL, TL = GU ? 2 : TLX, XW = E / 2;
  constexpr int RPB = GU ? 16 : 16 * TL;
  constexpr int NTP = NT > 4 ? 4 : NT;   // n-tiles per reduction pass (8 n-tiles of gate+up would need 64 KiB of static shared memory)
  __shared__ float part[S][TL][NTP][32][4];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int n0 = blockIdx.x * RPB;
  const int K = W.K, span = 4 * E, nsp = K / span;
  const int s0 = warp * nsp / S, s1 = (warp + 1) * nsp / S;
  const size_t rb = F == WFmt::BF16 ? (size_t)K * 2 : F == WFmt::FP8ROW ? (size_t)K : (size_t)K / 2;
  const uint8_t* wr[TL][2];
  const uint8_t* sr[TL][2];
#pragma unroll
  for (int tt = 0; tt < TL; ++tt) {
    const GMat& Wt = (GU && tt == 1) ? U : W;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int n = min(n0 + (GU ? 0 : tt * 16) + g + 8 * h, W.N - 1);
      wr[tt][h] = Wt.w + (size_t)n * rb;
      sr[tt][h] = F == WFmt::NVFP4 ? reinterpret_cast<const uint8_t*>(Wt.s) + (size_t)n * (K / 16) : nullptr;
    }
  }
  bool tv[NT];
  const bf16* xr[NT];
#pragma unroll
  for (int ni = 0; ni < NT; ++ni) { tv[ni] = ni * 8 + g < M; xr[ni] = x + (size_t)min(ni * 8 + g, M - 1) * ldx; }
  float acc[TL][NT][4];
#pragma unroll
  for (int tt = 0; tt < TL; ++tt)
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) acc[tt][ni][0] = acc[tt][ni][1] = acc[tt][ni][2] = acc[tt][ni][3] = 0.f;
  struct Batch { uint4 w[UN][TL][2]; uint16_t gs[UN][TL][2]; uint4 xv[JIT ? 1 : UN][JIT ? 1 : NT][JIT ? 1 : XW / 4]; };
  auto load = [&](Batch& B, int sb) {
#pragma unroll
    for (int u = 0; u < UN; ++u) {
      const int e0 = min(sb + u, s1 - 1) * span + t * E;
      const size_t bo = F == WFmt::BF16 ? (size_t)e0 * 2 : F == WFmt::FP8ROW ? (size_t)e0 : (size_t)e0 / 2;
#pragma unroll
      for (int tt = 0; tt < TL; ++tt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          B.w[u][tt][h] = __ldg(reinterpret_cast<const uint4*>(wr[tt][h] + bo));
          if constexpr (F == WFmt::NVFP4) B.gs[u][tt][h] = __ldg(reinterpret_cast<const uint16_t*>(sr[tt][h] + e0 / 16));
        }
      if constexpr (!JIT) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni)
#pragma unroll
          for (int c = 0; c < XW / 4; ++c) B.xv[u][ni][c] = tv[ni] ? __ldg(reinterpret_cast<const uint4*>(xr[ni] + e0) + c) : make_uint4(0, 0, 0, 0);
      }
    }
  };
  auto compute = [&](const Batch& B, int sb) {
#pragma unroll
    for (int u = 0; u < UN; ++u) {
      if (sb + u >= s1) break;
      __half2 sc[TL][2][2];
      if constexpr (F == WFmt::NVFP4) {
#pragma unroll
        for (int tt = 0; tt < TL; ++tt)
#pragma unroll
          for (int h = 0; h < 2; ++h) { sc[tt][h][0] = sw::sc_h2((uint8_t)(B.gs[u][tt][h] & 0xff)); sc[tt][h][1] = sw::sc_h2((uint8_t)(B.gs[u][tt][h] >> 8)); }
      }
      // JIT: the span's activations one 16-byte piece (two k-steps) at a time, the next piece loaded while the current one is multiplied
      uint4 xa[JIT ? 2 : 1][JIT ? NT : 1];
      const int e0j = (sb + u) * span + t * E;
      if constexpr (JIT) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) xa[0][ni] = tv[ni] ? __ldg(reinterpret_cast<const uint4*>(xr[ni] + e0j)) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int j = 0; j < STEPS; ++j) {
        uint32_t b0[NT], b1[NT];
        if constexpr (JIT) {
          if (j % 2 == 0 && j / 2 + 1 < XW / 4) {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) xa[(j / 2 + 1) & 1][ni] = tv[ni] ? __ldg(reinterpret_cast<const uint4*>(xr[ni] + e0j) + j / 2 + 1) : make_uint4(0, 0, 0, 0);
          }
#pragma unroll
          for (int ni = 0; ni < NT; ++ni) {
            const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xa[(j / 2) & 1][ni]);
            b0[ni] = xw[2 * (j % 2)]; b1[ni] = xw[2 * (j % 2) + 1];
          }
        } else {
#pragma unroll
          for (int ni = 0; ni < NT; ++ni) {
            const uint32_t* xw = reinterpret_cast<const uint32_t*>(&B.xv[u][ni][0]);
            b0[ni] = xw[2 * j]; b1[ni] = xw[2 * j + 1];
          }
        }
#pragma unroll
        for (int tt = 0; tt < TL; ++tt) {
          uint32_t a[4];
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            if constexpr (F == WFmt::FP8ROW) {
              const uint16_t* hw = reinterpret_cast<const uint16_t*>(&B.w[u][tt][h]);
              a[h] = sw::fp8_bf2(hw[2 * j]); a[2 + h] = sw::fp8_bf2(hw[2 * j + 1]);
            } else if constexpr (F == WFmt::NVFP4) {
              const uint8_t* by = reinterpret_cast<const uint8_t*>(&B.w[u][tt][h]);
              a[h] = sw::fp4_bf2(by[2 * j], sc[tt][h][j / 4]); a[2 + h] = sw::fp4_bf2(by[2 * j + 1], sc[tt][h][j / 4]);
            } else {
              const uint32_t* wd = reinterpret_cast<const uint32_t*>(&B.w[u][tt][h]);
              a[h] = wd[2 * j]; a[2 + h] = wd[2 * j + 1];
            }
          }
#pragma unroll
          for (int ni = 0; ni < NT; ++ni) tc::mma(acc[tt][ni], a, b0[ni], b1[ni]);
        }
      }
    }
  };
  if (s0 < s1) {
    Batch cur, nxt;
    load(cur, s0);
    for (int sb = s0; sb < s1; sb += UN) {
      if (sb + UN < s1) load(nxt, sb + UN);
      compute(cur, sb);
      cur = nxt;
    }
  }
#pragma unroll
  for (int pass = 0; pass < NT / NTP; ++pass) {
    if (pass) __syncthreads();
#pragma unroll
    for (int tt = 0; tt < TL; ++tt)
#pragma unroll
      for (int nj = 0; nj < NTP; ++nj)
#pragma unroll
        for (int c = 0; c < 4; ++c) part[warp][tt][nj][lane][c] = acc[tt][pass * NTP + nj][c];
    __syncthreads();
    // outputs: D (row g / g + 8, token 2t / 2t + 1) per lane and fragment element c
    constexpr int NOUT = (GU ? 1 : TL) * NTP * 32 * 4;
    for (int i = threadIdx.x; i < NOUT; i += blockDim.x) {
      const int c = i & 3, ln = (i >> 2) & 31, nj = (i >> 7) % NTP, tt = GU ? 0 : (i >> 7) / NTP;
      const int row = n0 + (GU ? 0 : tt * 16) + ln / 4 + (c >= 2 ? 8 : 0), tok = (pass * NTP + nj) * 8 + 2 * (ln % 4) + (c & 1);
      if (row >= W.N || tok >= M) continue;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < S; ++w) v += part[w][tt][nj][ln][c];
      if constexpr (GU) {
        float u = 0.f;
#pragma unroll
        for (int w = 0; w < S; ++w) u += part[w][1][nj][ln][c];
        const float sg = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : F == WFmt::NVFP4 ? W.g : 1.f;
        const float su = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(U.s)[row] : F == WFmt::NVFP4 ? U.g : 1.f;
        yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(bfr(siluf_(bfr(v * sg))) * bfr(u * su));
      } else {
        const float sc = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : F == WFmt::NVFP4 ? W.g : 1.f;
        if (yb) yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(v * sc); else yf[(size_t)tok * ldy + row] = v * sc;
      }
    }
  }
}
// ---- staged swapped GEMV (HIVE_Q27_SW=15, 9..32 rows, NVFP4 / FP8 rows): sw_kernel's work split, span order, warp order and mma
//   sequence, but each warp stages its activations half a span at a time in shared memory (cp.async, double buffered, zero-filled for
//   absent tokens) instead of holding up to 4 n-tiles × a whole span in registers (ncu 2026-10-11: 239-251 registers → 1 block of 8 warps
//   per SM, DRAM 45-61 % at 17..32 rows). B fragments come from shared memory as 16-byte pieces (two k-steps); the layout puts the 8 lanes
//   of every quarter-warp phase on distinct 4-bank groups. The partial sums of the reduction reuse the staging memory. The per-token
//   arithmetic is sw_kernel's, so every row is bit-identical to the M = 1 call.
namespace sws {
template <WFmt F> constexpr int hw() { return sw::elems<F>() / 4; }                       // words per (token, lane-t) half chunk: NVFP4 8 · FP8 4
template <WFmt F> constexpr int sg() { return 4 * hw<F>() + (hw<F>() == 8 ? 4 : 0); }     // words per token row of a half buffer
template <WFmt F, int NT> constexpr int buf_words() { return NT * 8 * sg<F>(); }           // one half buffer of one warp
template <WFmt F, bool GU, int NT, int TL> constexpr size_t smem_bytes() {
  const size_t st = (size_t)8 * 2 * buf_words<F, NT>() * 4, pr = (size_t)8 * (GU ? 2 : TL) * NT * 32 * 4 * 4;
  return st > pr ? st : pr;
}
__device__ __forceinline__ void cp16(uint32_t d, const void* src, int bytes) {
  // .ca (through L1): with .cg the L2 traffic was 2.5x the register kernel's, whose __ldg activations hit L1 across blocks (ncu 2026-10-11)
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
}  // namespace sws

template <WFmt F, bool GU, int NT, int TLX, int MINB>
__global__ void __launch_bounds__(256, MINB) sws_kernel(GMat W, GMat U, const bf16* __restrict__ x, int ldx, int M, bf16* yb, float* yf, int ldy) {
  constexpr int E = sw::elems<F>(), S = 8, TL = GU ? 2 : TLX, HW = sws::hw<F>(), SG = sws::sg<F>(), P = HW / 4, BW = sws::buf_words<F, NT>();
  constexpr int RPB = GU ? 16 : 16 * TL, HSTEPS = E / 8;   // k-steps per half span
  extern __shared__ __align__(16) uint32_t sws_smem[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int n0 = blockIdx.x * RPB;
  const int K = W.K, span = 4 * E, nsp = K / span;
  const int s0 = warp * nsp / S, s1 = (warp + 1) * nsp / S;
  const size_t rb = F == WFmt::FP8ROW ? (size_t)K : (size_t)K / 2;
  uint32_t* wb = sws_smem + (size_t)warp * 2 * BW;   // this warp's two half buffers
  const uint32_t wb_s = (uint32_t)__cvta_generic_to_shared(wb);
  const uint8_t* wr[TL][2];
  const uint8_t* sr[TL][2];
#pragma unroll
  for (int tt = 0; tt < TL; ++tt) {
    const GMat& Wt = (GU && tt == 1) ? U : W;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int n = min(n0 + (GU ? 0 : tt * 16) + g + 8 * h, W.N - 1);
      wr[tt][h] = Wt.w + (size_t)n * rb;
      sr[tt][h] = F == WFmt::NVFP4 ? reinterpret_cast<const uint8_t*>(Wt.s) + (size_t)n * (K / 16) : nullptr;
    }
  }
  // stage half hf of span sb into buffer b: pieces (ni, token g', lane-t t', piece q) — NT·32·P per warp, NT·P per lane
  auto stage = [&](int sb, int hf, int b) {
#pragma unroll
    for (int i = 0; i < NT * P; ++i) {
      const int idx = lane + 32 * i, q = idx % P, tq = (idx / P) % 4, gq = (idx / P / 4) % 8, ni = idx / P / 32;
      const int tok = ni * 8 + gq;
      const bf16* src = x + (size_t)min(tok, M - 1) * ldx + (size_t)sb * span + tq * E + hf * (E / 2) + q * 8;
      sws::cp16(wb_s + 4u * (uint32_t)(b * BW + ni * 8 * SG + gq * SG + tq * HW + q * 4), src, tok < M ? 16 : 0);
    }
    sws::commit();
  };
  float acc[TL][NT][4];
#pragma unroll
  for (int tt = 0; tt < TL; ++tt)
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) acc[tt][ni][0] = acc[tt][ni][1] = acc[tt][ni][2] = acc[tt][ni][3] = 0.f;
  struct WB { uint4 w[TL][2]; uint16_t gs[TL][2]; };
  auto wload = [&](WB& B, int sb) {
    const int e0 = sb * span + t * E;
    const size_t bo = F == WFmt::FP8ROW ? (size_t)e0 : (size_t)e0 / 2;
#pragma unroll
    for (int tt = 0; tt < TL; ++tt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        B.w[tt][h] = __ldg(reinterpret_cast<const uint4*>(wr[tt][h] + bo));
        if constexpr (F == WFmt::NVFP4) B.gs[tt][h] = __ldg(reinterpret_cast<const uint16_t*>(sr[tt][h] + e0 / 16));
      }
  };
  if (s0 < s1) {
    WB cur, nxt;
    wload(cur, s0);
    stage(s0, 0, 0);
    int b = 0;
    for (int sb = s0; sb < s1; ++sb) {
      if (sb + 1 < s1) wload(nxt, sb + 1);
      __half2 sc[TL][2][2];
      if constexpr (F == WFmt::NVFP4) {
#pragma unroll
        for (int tt = 0; tt < TL; ++tt)
#pragma unroll
          for (int h = 0; h < 2; ++h) { sc[tt][h][0] = sw::sc_h2((uint8_t)(cur.gs[tt][h] & 0xff)); sc[tt][h][1] = sw::sc_h2((uint8_t)(cur.gs[tt][h] >> 8)); }
      }
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const bool more = hf == 0 || sb + 1 < s1;
        if (more) stage(hf == 0 ? sb : sb + 1, hf ^ 1, b ^ 1);
        if (more) sws::wait<1>(); else sws::wait<0>();
        __syncwarp();
        const uint32_t* bb = wb + b * BW + g * SG + t * HW;
#pragma unroll
        for (int jj = 0; jj < HSTEPS; jj += 2) {   // one 16-byte piece = two k-steps
          uint4 xq[NT];
#pragma unroll
          for (int ni = 0; ni < NT; ++ni) xq[ni] = *reinterpret_cast<const uint4*>(bb + ni * 8 * SG + (jj / 2) * 4);
#pragma unroll
          for (int js = 0; js < 2; ++js) {
            const int j = hf * HSTEPS + jj + js;
            uint32_t b0[NT], b1[NT];
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
              const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xq[ni]);
              b0[ni] = xw[2 * js]; b1[ni] = xw[2 * js + 1];
            }
#pragma unroll
            for (int tt = 0; tt < TL; ++tt) {
              uint32_t a[4];
#pragma unroll
              for (int h = 0; h < 2; ++h) {
                if constexpr (F == WFmt::FP8ROW) {
                  const uint16_t* hw16 = reinterpret_cast<const uint16_t*>(&cur.w[tt][h]);
                  a[h] = sw::fp8_bf2(hw16[2 * j]); a[2 + h] = sw::fp8_bf2(hw16[2 * j + 1]);
                } else {
                  const uint8_t* by = reinterpret_cast<const uint8_t*>(&cur.w[tt][h]);
                  a[h] = sw::fp4_bf2(by[2 * j], sc[tt][h][j / 4]); a[2 + h] = sw::fp4_bf2(by[2 * j + 1], sc[tt][h][j / 4]);
                }
              }
#pragma unroll
              for (int ni = 0; ni < NT; ++ni) tc::mma(acc[tt][ni], a, b0[ni], b1[ni]);
            }
          }
        }
        __syncwarp();
        b ^= 1;
      }
      cur = nxt;
    }
  }
  sws::wait<0>();
  __syncthreads();   // the staging memory becomes the partial-sum area
  float* part = reinterpret_cast<float*>(sws_smem);   // [S][TL][NT][32][4]
#pragma unroll
  for (int tt = 0; tt < TL; ++tt)
#pragma unroll
    for (int ni = 0; ni < NT; ++ni)
#pragma unroll
      for (int c = 0; c < 4; ++c) part[(((size_t)warp * TL + tt) * NT + ni) * 128 + lane * 4 + c] = acc[tt][ni][c];
  __syncthreads();
  constexpr int NOUT = (GU ? 1 : TL) * NT * 32 * 4;
  for (int i = threadIdx.x; i < NOUT; i += blockDim.x) {
    const int c = i & 3, ln = (i >> 2) & 31, ni = (i >> 7) % NT, tt = GU ? 0 : (i >> 7) / NT;
    const int row = n0 + (GU ? 0 : tt * 16) + ln / 4 + (c >= 2 ? 8 : 0), tok = ni * 8 + 2 * (ln % 4) + (c & 1);
    if (row >= W.N || tok >= M) continue;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < S; ++w) v += part[(((size_t)w * TL + tt) * NT + ni) * 128 + ln * 4 + c];
    if constexpr (GU) {
      float u = 0.f;
#pragma unroll
      for (int w = 0; w < S; ++w) u += part[(((size_t)w * TL + 1) * NT + ni) * 128 + ln * 4 + c];
      const float sg_ = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : W.g;
      const float su = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(U.s)[row] : U.g;
      yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(bfr(siluf_(bfr(v * sg_))) * bfr(u * su));
    } else {
      const float scl = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : W.g;
      if (yb) yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(v * scl); else yf[(size_t)tok * ldy + row] = v * scl;
    }
  }
}
template <WFmt F, bool GU, int NT, int TL, int MINB>
void sws_go(const GMat& W, const GMat& U, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  constexpr size_t sm = sws::smem_bytes<F, GU, NT, TL>();
  static bool attr = false;
  if (!attr) {   // the largest shared-memory carveout: otherwise the driver's default left 1 block per SM (ncu 2026-10-11, "Block Limit Shared Mem 1")
    CUDA_CHECK(cudaFuncSetAttribute(sws_kernel<F, GU, NT, TL, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    static const int carve = getenv("HIVE_Q27_SWS_CARVE") ? atoi(getenv("HIVE_Q27_SWS_CARVE")) : -1;   // percent of L1 as shared memory (-1: driver default)
    if (carve >= 0) CUDA_CHECK(cudaFuncSetAttribute(sws_kernel<F, GU, NT, TL, MINB>, cudaFuncAttributePreferredSharedMemoryCarveout, carve));
    attr = true;
  }
  const int rpb = GU ? 16 : 16 * TL;
  sws_kernel<F, GU, NT, TL, MINB><<<(W.N + rpb - 1) / rpb, 256, sm, st>>>(W, U, x, ldx, M, yb, yf, ldy);
}
// HIVE_Q27_SW=v (v ≥ 1): the swapped tensor-core GEMV for every M ≤ 16 (decode, verify, batch) when K is a whole number of spans.
//   Variants (tile rows per warp · spans per batch): 1 = 1 tile · 4 (2 for NVFP4), 2 = 2 tiles · 2 (1), 3 = 1 tile · 2 (1)
int sw_ver() { static const int v = getenv("HIVE_Q27_SW") ? atoi(getenv("HIVE_Q27_SW")) : 0; return v; }
bool sw_on() { return sw_ver() > 0; }
template <WFmt F, bool GU, int NT, int TL, int UN, int SPL = 8, int MINB = 1, bool JIT = false>
void sw_go(const GMat& W, const GMat& U, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  const int rpb = GU ? 16 : 16 * TL;
  sw_kernel<F, GU, NT, TL, UN, SPL, MINB, JIT><<<(W.N + rpb - 1) / rpb, 32 * SPL, 0, st>>>(W, U, x, ldx, M, yb, yf, ldy);
}
template <WFmt F, bool GU, int NT>
void sw_ver_go(const GMat& W, const GMat& U, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  constexpr bool F4 = F == WFmt::NVFP4;
  switch (sw_ver()) {
    case 15: case 16: case 17:   // staged activations (sws_kernel) for 9..32 rows of NVFP4 / FP8 rows; 15 = 2 blocks per SM, 16 = 3, 17 = 2 with 1 weight tile
      if constexpr (NT >= 2 && F != WFmt::BF16) {
        if (sw_ver() == 16) sws_go<F, GU, NT, 2, 3>(W, U, x, ldx, M, yb, yf, ldy, st);
        else if (sw_ver() == 17) sws_go<F, GU, NT, 1, 2>(W, U, x, ldx, M, yb, yf, ldy, st);
        else sws_go<F, GU, NT, 2, 2>(W, U, x, ldx, M, yb, yf, ldy, st);
      } else if constexpr (NT >= 2) sw_go<F, GU, NT, 2, F4 ? 1 : 2>(W, U, x, ldx, M, yb, yf, ldy, st);
      else sw_go<F, GU, NT, 1, F4 ? 2 : 4>(W, U, x, ldx, M, yb, yf, ldy, st);
      break;
    case 2: sw_go<F, GU, NT, 2, F4 ? 1 : 2>(W, U, x, ldx, M, yb, yf, ldy, st); break;
    case 3: sw_go<F, GU, NT, 1, F4 ? 1 : 2>(W, U, x, ldx, M, yb, yf, ldy, st); break;
    case 5:   // 16 warps per block (K split 16) for matrices of ≤ 6144 rows, else variant 1
      if (W.N <= 6144) sw_go<F, GU, NT, NT == 4 ? 2 : 1, F4 ? (NT == 4 ? 1 : 2) : (NT == 4 ? 2 : 4), 16>(W, U, x, ldx, M, yb, yf, ldy, st);
      else if constexpr (NT == 4) sw_go<F, GU, NT, 2, F4 ? 1 : 2>(W, U, x, ldx, M, yb, yf, ldy, st);
      else sw_go<F, GU, NT, 1, F4 ? 2 : 4>(W, U, x, ldx, M, yb, yf, ldy, st);
      break;
    case 4: sw_go<F, GU, NT, NT >= 2 ? 2 : 1, F4 ? (NT >= 2 ? 1 : 2) : (NT >= 2 ? 2 : 4)>(W, U, x, ldx, M, yb, yf, ldy, st); break;   // 2 tiles from 9 rows
    // 6..8 (2026-10-10, verify widths 9..32 ran at ~1.25 TB/s vs ~1.45 for ≤ 8 rows): more weight bytes in flight per warp for NT ≥ 2;
    //   M ≤ 8 keeps variant 1. Same span order per warp (SPL 8) → the same bits as every other variant.
    // 9 / 10 / 11 (ncu 2026-10-11: 239-251 registers per thread → 1 block of 8 warps per SM, 16.7 % occupancy, DRAM 45-61 %): variant 4's
    //   tiles with ≤ 128 (2 blocks per SM) / ≤ 80 (3) registers forced; 11 = 2 blocks and 1 tile for matrices of ≤ 8192 rows at 17..32 rows
    //   (the 5120-row matrices filled 0.85 waves)
    case 9: sw_go<F, GU, NT, NT >= 2 ? 2 : 1, F4 ? (NT >= 2 ? 1 : 2) : (NT >= 2 ? 2 : 4), 8, 2>(W, U, x, ldx, M, yb, yf, ldy, st); break;
    case 10: sw_go<F, GU, NT, NT >= 2 ? 2 : 1, F4 ? (NT >= 2 ? 1 : 2) : (NT >= 2 ? 2 : 4), 8, 3>(W, U, x, ldx, M, yb, yf, ldy, st); break;
    case 11:
      if (NT == 4 && W.N <= 8192) sw_go<F, GU, NT, 1, F4 ? 1 : 2, 8, 2>(W, U, x, ldx, M, yb, yf, ldy, st);
      else sw_go<F, GU, NT, NT >= 2 ? 2 : 1, F4 ? (NT >= 2 ? 1 : 2) : (NT >= 2 ? 2 : 4), 8, 2>(W, U, x, ldx, M, yb, yf, ldy, st);
      break;
    // 12 / 13 / 14: activations loaded just in time (two k-steps per 16-byte piece, one piece ahead) instead of double-buffered per span —
    //   frees ~64-128 registers at 17..32 rows; 12 = variant 4's tiles, 13 = + ≤ 128 registers (2 blocks per SM), 14 = 13 with spans in pairs
    case 12: sw_go<F, GU, NT, NT >= 2 ? 2 : 1, F4 ? (NT >= 2 ? 1 : 2) : (NT >= 2 ? 2 : 4), 8, 1, true>(W, U, x, ldx, M, yb, yf, ldy, st); break;
    case 13: sw_go<F, GU, NT, NT >= 2 ? 2 : 1, F4 ? (NT >= 2 ? 1 : 2) : (NT >= 2 ? 2 : 4), 8, 2, true>(W, U, x, ldx, M, yb, yf, ldy, st); break;
    case 14: sw_go<F, GU, NT, NT >= 2 ? 2 : 1, 2, 8, 2, true>(W, U, x, ldx, M, yb, yf, ldy, st); break;
    case 6:
      if constexpr (NT >= 2) sw_go<F, GU, NT, 2, 2>(W, U, x, ldx, M, yb, yf, ldy, st);
      else sw_go<F, GU, NT, 1, F4 ? 2 : 4>(W, U, x, ldx, M, yb, yf, ldy, st);
      break;
    case 7:
      if constexpr (NT >= 2) sw_go<F, GU, NT, 1, 4>(W, U, x, ldx, M, yb, yf, ldy, st);
      else sw_go<F, GU, NT, 1, F4 ? 2 : 4>(W, U, x, ldx, M, yb, yf, ldy, st);
      break;
    case 8:
      if constexpr (NT >= 2) sw_go<F, GU, NT, 1, 2>(W, U, x, ldx, M, yb, yf, ldy, st);
      else sw_go<F, GU, NT, 1, F4 ? 2 : 4>(W, U, x, ldx, M, yb, yf, ldy, st);
      break;
    default:   // 9..32 rows: the activations serve 2 tiles (9..16 too since 2026-10-11 — DRAM bench M12 / M16 689 / 770 → 617 / 624 us)
      if constexpr (NT >= 2) sw_go<F, GU, NT, 2, F4 ? 1 : 2>(W, U, x, ldx, M, yb, yf, ldy, st);
      else sw_go<F, GU, NT, 1, F4 ? 2 : 4>(W, U, x, ldx, M, yb, yf, ldy, st);
      break;
  }
}
// ---- split-N tensor-core GEMV (HIVE_Q27_SW=20): every warp owns 16 output rows (gate+up: 16 of each) of a 128-row block over the block's
//   K range, and a K-chunk of all M rows' activations is staged once per block in shared memory (cp.async, double buffered) for all 8 warps.
//   The swapped kernel above splits K over a block's 8 warps, so each warp re-reads its activations and every 32-row block re-reads all of them:
//   ncu 2026-10-11 (gate+up 17408×5120, 32 rows) ~348 MB of L2 activation reads against 100 MB of weights, 2 warps per scheduler (239-251
//   registers), DRAM 45 %. Small-N matrices split K over blocks (split count from the shape only); the partials are added in split order by
//   sn_reduce_kernel. Per token the arithmetic does not depend on M (the same spans per warp, the same split order), so decode, verify and
//   batch rows agree bit for bit. Weight decode and fragment layout as sw_kernel (lane t holds E consecutive k of rows g / g + 8).
namespace sn {
constexpr int kWarps = 4, kRowsPB = 16 * kWarps, kUN = 2;   // 4 warps: more resident blocks, whole-K loops for the large matrices
float* g_part = nullptr;
size_t g_part_bytes = 0;
inline int splits(int N, int K, int span) {
  const int bn = (N + kRowsPB - 1) / kRowsPB, nsp = K / span;
  static const int bps = getenv("HIVE_Q27_SN_BPS") ? std::max(1, atoi(getenv("HIVE_Q27_SN_BPS"))) : 4;   // target resident blocks per SM
  int sk = (bps * 188 + bn / 2) / bn;
  return std::max(1, std::min({sk, 16, nsp / (2 * kUN)}));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
}  // namespace sn
template <WFmt F, bool GU, int NT>
__global__ void __launch_bounds__(32 * sn::kWarps) sn_kernel(GMat W, GMat U, const bf16* __restrict__ x, int ldx, int M, int SK, float* part, bf16* yb, float* yf,
                                                 int ldy) {
  constexpr int E = sw::elems<F>(), STEPS = E / 4, TL = GU ? 2 : 1, UN = sn::kUN, SPAN = 4 * E, CHK = UN * SPAN, RS = CHK + 8, ROWS = NT * 8;
  extern __shared__ uint4 sn_smem[];
  bf16* xs = reinterpret_cast<bf16*>(sn_smem);   // [2][ROWS][RS]
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int n0 = blockIdx.x * sn::kRowsPB + warp * 16, ks = blockIdx.y;
  const int K = W.K, nsp = K / SPAN, s0 = ks * nsp / SK, s1 = (ks + 1) * nsp / SK;
  const size_t rb = F == WFmt::BF16 ? (size_t)K * 2 : F == WFmt::FP8ROW ? (size_t)K : (size_t)K / 2;
  const uint8_t* wr[TL][2];
  const uint8_t* sr[TL][2];
#pragma unroll
  for (int tt = 0; tt < TL; ++tt) {
    const GMat& Wt = (GU && tt == 1) ? U : W;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int n = min(n0 + g + 8 * h, W.N - 1);
      wr[tt][h] = Wt.w + (size_t)n * rb;
      sr[tt][h] = F == WFmt::NVFP4 ? reinterpret_cast<const uint8_t*>(Wt.s) + (size_t)n * (K / 16) : nullptr;
    }
  }
  float acc[TL][NT][4];
#pragma unroll
  for (int tt = 0; tt < TL; ++tt)
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) acc[tt][ni][0] = acc[tt][ni][1] = acc[tt][ni][2] = acc[tt][ni][3] = 0.f;
  struct WB { uint4 w[UN][TL][2]; uint16_t gs[UN][TL][2]; };
  auto loadw = [&](WB& B, int sb) {
#pragma unroll
    for (int u = 0; u < UN; ++u) {
      const int e0 = min(sb + u, s1 - 1) * SPAN + t * E;
      const size_t bo = F == WFmt::BF16 ? (size_t)e0 * 2 : F == WFmt::FP8ROW ? (size_t)e0 : (size_t)e0 / 2;
#pragma unroll
      for (int tt = 0; tt < TL; ++tt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          B.w[u][tt][h] = __ldg(reinterpret_cast<const uint4*>(wr[tt][h] + bo));
          if constexpr (F == WFmt::NVFP4) B.gs[u][tt][h] = __ldg(reinterpret_cast<const uint16_t*>(sr[tt][h] + e0 / 16));
        }
    }
  };
  auto stage = [&](int sb, int b) {   // activations of spans [sb, sb + UN) ∩ [s0, s1) for all rows (rows ≥ M: zeros)
    const int k0 = sb * SPAN, nk = min(UN, s1 - sb) * SPAN;
    bf16* dst = xs + (size_t)b * ROWS * RS;
    for (int i = threadIdx.x; i < ROWS * (nk / 8); i += blockDim.x) {
      const int r = i / (nk / 8), c8 = i % (nk / 8);
      if (r < M) sn::cp16(dst + r * RS + c8 * 8, x + (size_t)r * ldx + k0 + c8 * 8);
      else *reinterpret_cast<uint4*>(dst + r * RS + c8 * 8) = make_uint4(0, 0, 0, 0);
    }
    sn::commit();
  };
  auto compute = [&](const WB& B, int sb, int b) {
    const bf16* xb = xs + (size_t)b * ROWS * RS;
#pragma unroll
    for (int u = 0; u < UN; ++u) {
      if (sb + u >= s1) break;
      __half2 sc[TL][2][2];
      if constexpr (F == WFmt::NVFP4) {
#pragma unroll
        for (int tt = 0; tt < TL; ++tt)
#pragma unroll
          for (int h = 0; h < 2; ++h) { sc[tt][h][0] = sw::sc_h2((uint8_t)(B.gs[u][tt][h] & 0xff)); sc[tt][h][1] = sw::sc_h2((uint8_t)(B.gs[u][tt][h] >> 8)); }
      }
#pragma unroll
      for (int j = 0; j < STEPS; ++j) {
        uint32_t b0[NT], b1[NT];
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
          const uint2 v = *reinterpret_cast<const uint2*>(xb + (ni * 8 + g) * RS + u * SPAN + t * E + 4 * j);
          b0[ni] = v.x; b1[ni] = v.y;
        }
#pragma unroll
        for (int tt = 0; tt < TL; ++tt) {
          uint32_t a[4];
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            if constexpr (F == WFmt::FP8ROW) {
              const uint16_t* hw = reinterpret_cast<const uint16_t*>(&B.w[u][tt][h]);
              a[h] = sw::fp8_bf2(hw[2 * j]); a[2 + h] = sw::fp8_bf2(hw[2 * j + 1]);
            } else if constexpr (F == WFmt::NVFP4) {
              const uint8_t* by = reinterpret_cast<const uint8_t*>(&B.w[u][tt][h]);
              a[h] = sw::fp4_bf2(by[2 * j], sc[tt][h][j / 4]); a[2 + h] = sw::fp4_bf2(by[2 * j + 1], sc[tt][h][j / 4]);
            } else {
              const uint32_t* wd = reinterpret_cast<const uint32_t*>(&B.w[u][tt][h]);
              a[h] = wd[2 * j]; a[2 + h] = wd[2 * j + 1];
            }
          }
#pragma unroll
          for (int ni = 0; ni < NT; ++ni) tc::mma(acc[tt][ni], a, b0[ni], b1[ni]);
        }
      }
    }
  };
  if (s0 < s1) {
    WB cur, nxt;
    stage(s0, 0);
    loadw(cur, s0);
    int b = 0;
    for (int sb = s0; sb < s1; sb += UN, b ^= 1) {
      if (sb + UN < s1) { stage(sb + UN, b ^ 1); loadw(nxt, sb + UN); sn::wait<1>(); } else sn::wait<0>();
      __syncthreads();
      compute(cur, sb, b);
      __syncthreads();
      cur = nxt;
    }
  }
  // outputs: D (row g / g + 8, token 2t / 2t + 1) per lane and fragment element c
#pragma unroll
  for (int ni = 0; ni < NT; ++ni)
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int row = n0 + g + (c >= 2 ? 8 : 0), tok = ni * 8 + 2 * t + (c & 1);
      if (row >= W.N || tok >= M) continue;
      if (SK > 1) {   // partial of split ks: part[(ks · M + tok) · TL·N + tile·N + row]
#pragma unroll
        for (int tt = 0; tt < TL; ++tt) part[((size_t)ks * M + tok) * TL * W.N + (size_t)tt * W.N + row] = acc[tt][ni][c];
        continue;
      }
      const float v = acc[0][ni][c];
      if constexpr (GU) {
        const float u = acc[TL - 1][ni][c];
        const float sg = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : F == WFmt::NVFP4 ? W.g : 1.f;
        const float su = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(U.s)[row] : F == WFmt::NVFP4 ? U.g : 1.f;
        yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(bfr(siluf_(bfr(v * sg))) * bfr(u * su));
      } else {
        const float sc = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : F == WFmt::NVFP4 ? W.g : 1.f;
        if (yb) yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(v * sc); else yf[(size_t)tok * ldy + row] = v * sc;
      }
    }
}
template <WFmt F, bool GU>
__global__ void sn_reduce_kernel(GMat W, GMat U, int M, int SK, const float* part, bf16* yb, float* yf, int ldy) {
  const int TL = GU ? 2 : 1;
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * W.N) return;
  const int tok = (int)(i / W.N), row = (int)(i % W.N);
  float v = 0.f, u = 0.f;
  for (int k = 0; k < SK; ++k) {   // split order
    v += part[((size_t)k * M + tok) * TL * W.N + row];
    if (GU) u += part[((size_t)k * M + tok) * TL * W.N + W.N + row];
  }
  if constexpr (GU) {
    const float sg = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : F == WFmt::NVFP4 ? W.g : 1.f;
    const float su = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(U.s)[row] : F == WFmt::NVFP4 ? U.g : 1.f;
    yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(bfr(siluf_(bfr(v * sg))) * bfr(u * su));
  } else {
    const float sc = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : F == WFmt::NVFP4 ? W.g : 1.f;
    if (yb) yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(v * sc); else yf[(size_t)tok * ldy + row] = v * sc;
  }
}
template <WFmt F, bool GU, int NT>
void sn_go(const GMat& W, const GMat& U, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, int SK, cudaStream_t st) {
  constexpr int E = sw::elems<F>(), RS = sn::kUN * 4 * E + 8;
  const size_t smem = (size_t)2 * NT * 8 * RS * 2;
  static const bool attr = cudaFuncSetAttribute(sn_kernel<F, GU, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess;
  (void)attr;
  dim3 grid((W.N + sn::kRowsPB - 1) / sn::kRowsPB, SK);
  sn_kernel<F, GU, NT><<<grid, 32 * sn::kWarps, smem, st>>>(W, U, x, ldx, M, SK, sn::g_part, yb, yf, ldy);
  if (SK > 1) sn_reduce_kernel<F, GU><<<(unsigned)(((size_t)M * W.N + 255) / 256), 256, 0, st>>>(W, U, M, SK, sn::g_part, yb, yf, ldy);
}
template <bool GU>
bool sn_launch(const GMat& W, const GMat& U, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  if (sw_ver() != 20 || !sn::g_part) return false;
  const int E = W.fmt == WFmt::NVFP4 ? 32 : W.fmt == WFmt::FP8ROW ? 16 : 8, span = 4 * E;
  if (W.K % span != 0 || ldx % 8 != 0 || M < 1 || M > 32) return false;
  const int SK = sn::splits(W.N, W.K, span);
  if (SK > 1 && (size_t)SK * M * (GU ? 2 : 1) * W.N * 4 > sn::g_part_bytes) return false;
#define Q27_SN(F, NT) sn_go<F, GU, NT>(W, U, x, ldx, M, yb, yf, ldy, SK, st)
  if (M <= 8) {
    if (W.fmt == WFmt::BF16) Q27_SN(WFmt::BF16, 1); else if (W.fmt == WFmt::FP8ROW) Q27_SN(WFmt::FP8ROW, 1); else Q27_SN(WFmt::NVFP4, 1);
  } else if (M <= 16) {
    if (W.fmt == WFmt::BF16) Q27_SN(WFmt::BF16, 2); else if (W.fmt == WFmt::FP8ROW) Q27_SN(WFmt::FP8ROW, 2); else Q27_SN(WFmt::NVFP4, 2);
  } else {
    if (W.fmt == WFmt::BF16) Q27_SN(WFmt::BF16, 4); else if (W.fmt == WFmt::FP8ROW) Q27_SN(WFmt::FP8ROW, 4); else Q27_SN(WFmt::NVFP4, 4);
  }
#undef Q27_SN
  CUDA_CHECK(cudaGetLastError());
  return true;
}
// ---- split-N GEMV v2 (HIVE_Q27_SW=21): blocks of 4 warps × 16 rows over a K range of ≤ 768 values — that range's activations of all M rows are
//   staged once (one cp.async group, one barrier) and every warp issues all of its weight loads for the range before multiplying (6 NVFP4
//   spans = 96 B per lane in flight); the split count comes from the shape only (K / 768), partials are added in split order by
//   sn_reduce_kernel. Same arithmetic for every M (decode / verify / batch rows bit for bit). v1 (SW=20) synchronised every 2 spans and its
//   warps streamed the whole K: ~1000 GB/s at M = 1 against 1477 for the swapped kernel (DRAM bench 2026-10-11).
namespace sn2 {
constexpr int kWarps = 4, kRowsPB = 16 * kWarps, kKR = 768;   // rows per block · K values per block
}
template <WFmt F, bool GU, int NT>
__global__ void __launch_bounds__(32 * sn2::kWarps) sn2_kernel(GMat W, GMat U, const bf16* __restrict__ x, int ldx, int M, int SK, float* part, bf16* yb,
                                                              float* yf, int ldy) {
  constexpr int E = sw::elems<F>(), STEPS = E / 4, TL = GU ? 2 : 1, SPAN = 4 * E, NS = sn2::kKR / SPAN, RS = sn2::kKR + 8, ROWS = NT * 8;
  extern __shared__ uint4 sn2_smem[];
  bf16* xs = reinterpret_cast<bf16*>(sn2_smem);   // [ROWS][RS]
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int n0 = blockIdx.x * sn2::kRowsPB + warp * 16, ks = blockIdx.y;
  const int K = W.K, nsp = K / SPAN, s0 = ks * NS, s1 = min(nsp, s0 + NS);
  const size_t rb = F == WFmt::BF16 ? (size_t)K * 2 : F == WFmt::FP8ROW ? (size_t)K : (size_t)K / 2;
  {   // stage the range's activations (rows ≥ M: zeros)
    const int k0 = s0 * SPAN, nk = (s1 - s0) * SPAN;
    for (int i = threadIdx.x; i < ROWS * (nk / 8); i += blockDim.x) {
      const int r = i / (nk / 8), c8 = i % (nk / 8);
      if (r < M) sn::cp16(xs + r * RS + c8 * 8, x + (size_t)r * ldx + k0 + c8 * 8);
      else *reinterpret_cast<uint4*>(xs + r * RS + c8 * 8) = make_uint4(0, 0, 0, 0);
    }
    sn::commit();
  }
  uint4 wv[NS][TL][2];
  uint16_t gs[NS][TL][2];
#pragma unroll
  for (int tt = 0; tt < TL; ++tt) {
    const GMat& Wt = (GU && tt == 1) ? U : W;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int n = min(n0 + g + 8 * h, W.N - 1);
      const uint8_t* wr = Wt.w + (size_t)n * rb;
      const uint8_t* sr = F == WFmt::NVFP4 ? reinterpret_cast<const uint8_t*>(Wt.s) + (size_t)n * (K / 16) : nullptr;
#pragma unroll
      for (int u = 0; u < NS; ++u) {
        const int e0 = min(s0 + u, s1 - 1) * SPAN + t * E;
        const size_t bo = F == WFmt::BF16 ? (size_t)e0 * 2 : F == WFmt::FP8ROW ? (size_t)e0 : (size_t)e0 / 2;
        wv[u][tt][h] = __ldg(reinterpret_cast<const uint4*>(wr + bo));
        if constexpr (F == WFmt::NVFP4) gs[u][tt][h] = __ldg(reinterpret_cast<const uint16_t*>(sr + e0 / 16));
      }
    }
  }
  float acc[TL][NT][4];
#pragma unroll
  for (int tt = 0; tt < TL; ++tt)
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) acc[tt][ni][0] = acc[tt][ni][1] = acc[tt][ni][2] = acc[tt][ni][3] = 0.f;
  sn::wait<0>();
  __syncthreads();
#pragma unroll
  for (int u = 0; u < NS; ++u) {
    if (s0 + u >= s1) break;
    __half2 sc[TL][2][2];
    if constexpr (F == WFmt::NVFP4) {
#pragma unroll
      for (int tt = 0; tt < TL; ++tt)
#pragma unroll
        for (int h = 0; h < 2; ++h) { sc[tt][h][0] = sw::sc_h2((uint8_t)(gs[u][tt][h] & 0xff)); sc[tt][h][1] = sw::sc_h2((uint8_t)(gs[u][tt][h] >> 8)); }
    }
#pragma unroll
    for (int j = 0; j < STEPS; ++j) {
      uint32_t b0[NT], b1[NT];
#pragma unroll
      for (int ni = 0; ni < NT; ++ni) {
        const uint2 v = *reinterpret_cast<const uint2*>(xs + (ni * 8 + g) * RS + u * SPAN + t * E + 4 * j);
        b0[ni] = v.x; b1[ni] = v.y;
      }
#pragma unroll
      for (int tt = 0; tt < TL; ++tt) {
        uint32_t a[4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          if constexpr (F == WFmt::FP8ROW) {
            const uint16_t* hw = reinterpret_cast<const uint16_t*>(&wv[u][tt][h]);
            a[h] = sw::fp8_bf2(hw[2 * j]); a[2 + h] = sw::fp8_bf2(hw[2 * j + 1]);
          } else if constexpr (F == WFmt::NVFP4) {
            const uint8_t* by = reinterpret_cast<const uint8_t*>(&wv[u][tt][h]);
            a[h] = sw::fp4_bf2(by[2 * j], sc[tt][h][j / 4]); a[2 + h] = sw::fp4_bf2(by[2 * j + 1], sc[tt][h][j / 4]);
          } else {
            const uint32_t* wd = reinterpret_cast<const uint32_t*>(&wv[u][tt][h]);
            a[h] = wd[2 * j]; a[2 + h] = wd[2 * j + 1];
          }
        }
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) tc::mma(acc[tt][ni], a, b0[ni], b1[ni]);
      }
    }
  }
#pragma unroll
  for (int ni = 0; ni < NT; ++ni)
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int row = n0 + g + (c >= 2 ? 8 : 0), tok = ni * 8 + 2 * t + (c & 1);
      if (row >= W.N || tok >= M) continue;
      if (SK > 1) {
#pragma unroll
        for (int tt = 0; tt < TL; ++tt) part[((size_t)ks * M + tok) * TL * W.N + (size_t)tt * W.N + row] = acc[tt][ni][c];
        continue;
      }
      const float v = acc[0][ni][c];
      if constexpr (GU) {
        const float u = acc[TL - 1][ni][c];
        const float sg = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : F == WFmt::NVFP4 ? W.g : 1.f;
        const float su = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(U.s)[row] : F == WFmt::NVFP4 ? U.g : 1.f;
        yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(bfr(siluf_(bfr(v * sg))) * bfr(u * su));
      } else {
        const float sc = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : F == WFmt::NVFP4 ? W.g : 1.f;
        if (yb) yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(v * sc); else yf[(size_t)tok * ldy + row] = v * sc;
      }
    }
}
template <WFmt F, bool GU, int NT>
void sn2_go(const GMat& W, const GMat& U, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, int SK, cudaStream_t st) {
  const size_t smem = (size_t)NT * 8 * (sn2::kKR + 8) * 2;
  static const bool attr = cudaFuncSetAttribute(sn2_kernel<F, GU, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess;
  (void)attr;
  dim3 grid((W.N + sn2::kRowsPB - 1) / sn2::kRowsPB, SK);
  sn2_kernel<F, GU, NT><<<grid, 32 * sn2::kWarps, smem, st>>>(W, U, x, ldx, M, SK, sn::g_part, yb, yf, ldy);
  if (SK > 1) sn_reduce_kernel<F, GU><<<(unsigned)(((size_t)M * W.N + 255) / 256), 256, 0, st>>>(W, U, M, SK, sn::g_part, yb, yf, ldy);
}
template <bool GU>
bool sn2_launch(const GMat& W, const GMat& U, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  if (sw_ver() != 21 || !sn::g_part) return false;
  const int E = W.fmt == WFmt::NVFP4 ? 32 : W.fmt == WFmt::FP8ROW ? 16 : 8, span = 4 * E;
  if (W.K % span != 0 || sn2::kKR % span != 0 || ldx % 8 != 0 || M < 1 || M > 32) return false;
  const int SK = (W.K + sn2::kKR - 1) / sn2::kKR;
  if (SK > 1 && (size_t)SK * M * (GU ? 2 : 1) * W.N * 4 > sn::g_part_bytes) return false;
#define Q27_SN2(F, NT) sn2_go<F, GU, NT>(W, U, x, ldx, M, yb, yf, ldy, SK, st)
  if (M <= 8) {
    if (W.fmt == WFmt::BF16) Q27_SN2(WFmt::BF16, 1); else if (W.fmt == WFmt::FP8ROW) Q27_SN2(WFmt::FP8ROW, 1); else Q27_SN2(WFmt::NVFP4, 1);
  } else if (M <= 16) {
    if (W.fmt == WFmt::BF16) Q27_SN2(WFmt::BF16, 2); else if (W.fmt == WFmt::FP8ROW) Q27_SN2(WFmt::FP8ROW, 2); else Q27_SN2(WFmt::NVFP4, 2);
  } else {
    if (W.fmt == WFmt::BF16) Q27_SN2(WFmt::BF16, 4); else if (W.fmt == WFmt::FP8ROW) Q27_SN2(WFmt::FP8ROW, 4); else Q27_SN2(WFmt::NVFP4, 4);
  }
#undef Q27_SN2
  CUDA_CHECK(cudaGetLastError());
  return true;
}
// ---- W4A4 decode / verify GEMV (GMat.act == 4) ------------------------------------------------------------------------------------------
// NInfer's production 27B artifact allows NVFP4 activations for the MLP (AllowA4) and its resolver takes the W4A4 route for every T on the
//   gate+up matrix (read 2026-10-11 from its artifact metadata and dispatch thresholds — facts only). Our version: the M ≤ 32 activation rows
//   are quantized per row (global = amax / (6 · 448), e4m3 scale per 16 values, e2m1 round-to-nearest), then the swapped layout of
//   sw_kernel (weights = mma A, 16 rows; tokens = B, 8 per n-tile) with mma.m16n8k64 kind::mxf4nvf4 block_scale scale_vec::4X: lane (g, t)
//   holds A rows g / g + 8 at k 8t … 8t+7 and 32+8t … (4 bytes each), B column g at the same k, the row / column's 4 e4m3 scales of the
//   k64 step come from lanes t = 0 (A row g, B column g) and t = 1 (A row g + 8). 8 warps split K (fixed order), so every M gives the same
//   bits for a token. Lossy (activation quantization): behind HIVE_Q27_A4_* and the quality gate.
namespace a4 {
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
uint8_t* g_xq = nullptr;   // [32][K/2] e2m1 pairs
uint8_t* g_xs = nullptr;   // [32][K/16] e4m3
float* g_xg = nullptr;     // [32] global decode factor per row
constexpr int kMaxK = 17408;
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%12, %13};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(0), "n"(0));
}
}  // namespace a4
// one block per activation row: amax → global, then e4m3 scale + e2m1 codes per 16 values
__global__ void __launch_bounds__(256) a4_quant_kernel(const bf16* x, int ldx, int K, int M, uint8_t* xq, uint8_t* xs, float* xg, int search) {
  const int m = blockIdx.x, tid = threadIdx.x;   // rows m ≥ M (up to the n-tile) are written as zeros
  const bf16* xr = x + (size_t)min(m, M - 1) * ldx;
  float am = 0.f;
  if (m < M) for (int i = tid; i < K; i += 256) am = fmaxf(am, fabsf(__bfloat162float(xr[i])));
  am = a4::warp_max(am);
  __shared__ float red[8];
  if ((tid & 31) == 0) red[tid >> 5] = am;
  __syncthreads();
  if (tid < 32) { float v = tid < 8 ? red[tid] : 0.f; v = a4::warp_max(v); if (tid == 0) red[0] = v; }
  __syncthreads();
  const float gx = red[0] > 0.f ? red[0] / (6.f * 448.f) : 1.f;
  if (tid == 0 && m < M) xg[m] = gx;
  for (int b = tid; b < K / 16; b += 256) {
    float f[16], bm = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) { f[j] = m < M ? __bfloat162float(xr[b * 16 + j]) : 0.f; bm = fmaxf(bm, fabsf(f[j])); }
    // HIVE_Q27_A4_SEARCH=1: the block scale is the best of {1, .95, .9, .85, 1.05} × amax / 6 by squared error (as tools/q27_reencode.py
    //   does for weights); default: amax / 6
    const int nf = search ? 5 : 1;
    const float facs[5] = {1.f, 0.95f, 0.9f, 0.85f, 1.05f};
    __nv_fp8_storage_t sc = 0;
    float sd = 0.f, best = INFINITY;
    for (int fi = 0; fi < nf; ++fi) {
      const __nv_fp8_storage_t c = __nv_cvt_float_to_fp8(bm * facs[fi] / 6.f / gx, __NV_SATFINITE, __NV_E4M3);
      const __half_raw chr = __nv_cvt_fp8_to_halfraw(c, __NV_E4M3);
      const float cd = __half2float(*reinterpret_cast<const __half*>(&chr)) * gx;
      float err = 0.f;
      if (nf > 1 && cd > 0.f) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const __half_raw qh = __nv_cvt_fp4_to_halfraw(__nv_cvt_float_to_fp4(f[j] / cd, __NV_E2M1, cudaRoundNearest), __NV_E2M1);
          const float d = f[j] - __half2float(*reinterpret_cast<const __half*>(&qh)) * cd;
          err += d * d;
        }
      }
      if (fi == 0 || err < best) { best = err; sc = c; sd = cd; }
    }
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const uint32_t v = sd > 0.f ? ((uint32_t)__nv_cvt_float_to_fp4(f[j] / sd, __NV_E2M1, cudaRoundNearest) & 0xfu) : 0u;
      if (j < 8) lo |= v << (4 * j); else hi |= v << (4 * (j - 8));
    }
    // packed for the B fragment: k64 step ks = b / 4, quarter q = b % 4 (k 16q … 16q+15): lane t = (q & 1) · 2 + (j ≥ 8) holds k 8t … of
    //   the step's first half (q < 2) or second half (q ≥ 2) as its b0 / b1 word
    {
      const int ks = b / 4, q = b % 4, ni = m / 8, gg = m % 8, half = q >> 1, tb = (q & 1) * 2;
      uint32_t* base = reinterpret_cast<uint32_t*>(xq) + (((size_t)ks * 4 + ni) * 32) * 2;
      base[((gg * 4 + tb) * 2) + half] = lo;
      base[((gg * 4 + tb + 1) * 2) + half] = hi;
      xs[(((size_t)ks * 4 + ni) * 8 + gg) * 4 + q] = (uint8_t)sc;
    }
  }
}
// repack: one warp per (16-row tile, k64 step); lane (g, t) writes its A fragment (rows g / g + 8, k 8t … and 32+8t …) as 16 bytes and
//   lanes t < 2 the 4 scales of row g + 8t
__global__ void a4_repack_kernel(GMat W, uint8_t* dw, uint8_t* ds) {
  const int tile = blockIdx.x, ks = blockIdx.y, lane = threadIdx.x, g = lane / 4, t = lane % 4, K = W.K, nks = K / 64;
  uint32_t r[4];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = tile * 16 + g + 8 * h;
    const uint8_t* src = W.w + (size_t)row * (K / 2) + ks * 32;
    r[h] = row < W.N ? *reinterpret_cast<const uint32_t*>(src + 4 * t) : 0u;
    r[2 + h] = row < W.N ? *reinterpret_cast<const uint32_t*>(src + 16 + 4 * t) : 0u;
  }
  reinterpret_cast<uint4*>(dw)[((size_t)tile * nks + ks) * 32 + lane] = make_uint4(r[0], r[1], r[2], r[3]);
  if (t < 2) {
    const int row = tile * 16 + g + 8 * t;
    const uint32_t sc = row < W.N ? *reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(W.s) + (size_t)row * (K / 16) + ks * 4) : 0u;
    reinterpret_cast<uint32_t*>(ds)[((size_t)tile * nks + ks) * 16 + g * 2 + t] = sc;
  }
}
template <bool GU, int NT>
__global__ void __launch_bounds__(256) a4_kernel(GMat W, GMat U, int M, const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                                                 const float* __restrict__ xg, bf16* yb, float* yf, int ldy) {
  constexpr int TL = GU ? 2 : 1, S = 8;
  __shared__ float part[S][TL][NT][32][4];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int n0 = blockIdx.x * 16, K = W.K, nst = K / 64, s0 = warp * nst / S, s1 = (warp + 1) * nst / S;
  const uint4* wp[TL];
  const uint32_t* sp[TL];
#pragma unroll
  for (int tt = 0; tt < TL; ++tt) {
    const GMat& Wt = (GU && tt == 1) ? U : W;
    wp[tt] = reinterpret_cast<const uint4*>(Wt.a4w) + (size_t)blockIdx.x * nst * 32 + lane;
    sp[tt] = reinterpret_cast<const uint32_t*>(Wt.a4s) + (size_t)blockIdx.x * nst * 16 + g * 2 + (t & 1);
  }
  float acc[TL][NT][4];
#pragma unroll
  for (int tt = 0; tt < TL; ++tt)
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) acc[tt][ni][0] = acc[tt][ni][1] = acc[tt][ni][2] = acc[tt][ni][3] = 0.f;
  struct St { uint32_t a[TL][4], sa[TL], b[NT][2], sb[NT]; };
  auto load = [&](St& L, int stp) {
    const int ks = min(stp, s1 - 1);
#pragma unroll
    for (int tt = 0; tt < TL; ++tt) {
      const uint4 v = __ldg(wp[tt] + (size_t)ks * 32);
      L.a[tt][0] = v.x; L.a[tt][1] = v.y; L.a[tt][2] = v.z; L.a[tt][3] = v.w;
      L.sa[tt] = __ldg(sp[tt] + (size_t)ks * 16);
    }
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) {
      const uint2 v = __ldg(reinterpret_cast<const uint2*>(xq) + ((size_t)ks * 4 + ni) * 32 + lane);   // rows ≥ M were written as zeros
      L.b[ni][0] = v.x; L.b[ni][1] = v.y;
      L.sb[ni] = __ldg(reinterpret_cast<const uint32_t*>(xs) + ((size_t)ks * 4 + ni) * 8 + g);
    }
  };
  if (s0 < s1) {
    St cur, nxt;
    load(cur, s0);
    for (int st = s0; st < s1; ++st) {
      if (st + 1 < s1) load(nxt, st + 1);
#pragma unroll
      for (int tt = 0; tt < TL; ++tt)
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) a4::mma(acc[tt][ni], cur.a[tt], cur.b[ni][0], cur.b[ni][1], cur.sa[tt], cur.sb[ni]);
      cur = nxt;
    }
  }
#pragma unroll
  for (int tt = 0; tt < TL; ++tt)
#pragma unroll
    for (int ni = 0; ni < NT; ++ni)
#pragma unroll
      for (int c = 0; c < 4; ++c) part[warp][tt][ni][lane][c] = acc[tt][ni][c];
  __syncthreads();
  for (int i = threadIdx.x; i < NT * 32 * 4; i += blockDim.x) {
    const int c = i & 3, ln = (i >> 2) & 31, ni = i >> 7;
    const int row = n0 + ln / 4 + (c >= 2 ? 8 : 0), tok = ni * 8 + 2 * (ln % 4) + (c & 1);
    if (row >= W.N || tok >= M) continue;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < S; ++w) v += part[w][0][ni][ln][c];
    const float gx = xg[tok];
    if constexpr (GU) {
      float u = 0.f;
#pragma unroll
      for (int w = 0; w < S; ++w) u += part[w][TL - 1][ni][ln][c];
      yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(bfr(siluf_(bfr(v * (W.g * gx)))) * bfr(u * (U.g * gx)));
    } else {
      const float y = v * (W.g * gx);
      if (yb) yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(y); else yf[(size_t)tok * ldy + row] = y;
    }
  }
}
template <bool GU>
static bool a4_go(const GMat& W, const GMat& U, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  if (!a4::g_xq || W.act != 4 || W.fmt != WFmt::NVFP4 || !W.a4w || (GU && (U.fmt != WFmt::NVFP4 || !U.a4w || U.N != W.N || U.K != W.K)) || W.K % 64 ||
      W.K > a4::kMaxK || M < 1 || M > 32 || ldx % 8)
    return false;
  static const int search = getenv("HIVE_Q27_A4_SEARCH") && atoi(getenv("HIVE_Q27_A4_SEARCH")) == 1;
  a4_quant_kernel<<<(M + 7) / 8 * 8, 256, 0, st>>>(x, ldx, W.K, M, a4::g_xq, a4::g_xs, a4::g_xg, search);
  const dim3 grid((W.N + 15) / 16);
  if (M <= 8) a4_kernel<GU, 1><<<grid, 256, 0, st>>>(W, U, M, a4::g_xq, a4::g_xs, a4::g_xg, yb, yf, ldy);
  else if (M <= 16) a4_kernel<GU, 2><<<grid, 256, 0, st>>>(W, U, M, a4::g_xq, a4::g_xs, a4::g_xg, yb, yf, ldy);
  else a4_kernel<GU, 4><<<grid, 256, 0, st>>>(W, U, M, a4::g_xq, a4::g_xs, a4::g_xg, yb, yf, ldy);
  CUDA_CHECK(cudaGetLastError());
  return true;
}
size_t a4_repack(const GMat& W, uint8_t* dst, cudaStream_t st) {
  const size_t tiles = (W.N + 15) / 16, cb = tiles * 16 * (size_t)W.K / 2, sb = tiles * 16 * (size_t)W.K / 16;
  if (!dst) return cb + sb;
  HIVE_CHECK(W.fmt == WFmt::NVFP4 && W.K % 64 == 0, "q27 a4_repack: NVFP4, K % 64");
  a4_repack_kernel<<<dim3((unsigned)tiles, W.K / 64), 32, 0, st>>>(W, dst, dst + cb);
  CUDA_CHECK(cudaGetLastError());
  return cb + sb;
}
bool gemv_a4(const GMat& W, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  return a4_go<false>(W, W, x, ldx, M, yb, yf, ldy, st);
}
bool gemv_gu_a4(const GMat& G, const GMat& U, const bf16* x, int ldx, int M, bf16* act, int lda, cudaStream_t st) {
  return a4_go<true>(G, U, x, ldx, M, act, nullptr, lda, st);
}
// ---- GEMM-tile GEMV (HIVE_Q27_SW=30, every M ≤ 32: decode, verify, drafter): a block of 8 warps owns 128 output rows (gate+up: 64 + 64),
//   each warp one 16-row weight tile over the block's K range (no K split inside the block). The activations of a 256-wide K chunk
//   ([M ≤ 32][256] bf16) are staged once per block with cp.async (3 stages) and shared by all 8 warps; every warp streams its own weights
//   straight into registers one chunk ahead and dequantises them once per k16 step for all n-tiles. The swapped kernel above re-reads all
//   M × K activations per 16-32-row block (327 KB vs 87 KB of weights per block at M = 32, ncu 2026-10-11; 239-254 registers, 1 block per
//   SM): here a block's activation : weight bytes are 0.89 : 1 at M = 32. Small-N matrices split K over blocks — the split count depends on
//   the shape only — and gt_reduce_kernel adds the fp32 partials in split order. Per (row, token) the sum order is chunk, span, k-step,
//   split — the same for every M (each mma column is one token), so a row is bit-identical whatever M it is computed in.
//   Weight fragments as sw_kernel (lane t holds E consecutive k of rows g / g + 8 per span, k-step j uses its elements 4j..4j+3).
namespace gt {
constexpr int KC = 256, KS = 512, STG = 3, RB = 128;   // KC: default chunk · KS: split granularity (fixed — the sum order must not depend on the chunk)
template <WFmt F> constexpr int elems() { return sw::elems<F>(); }
float* g_part = nullptr;
int* g_cnt = nullptr;   // per-tile arrival counters of the last-block fixup (zeroed; the last block resets its own)
size_t g_part_bytes = 0;
// 16-byte piece p of a staged token row (KC / 8 = 32 pieces) → its slot: lanes (g, t) of one quarter-warp phase read 8 distinct 4-bank groups
template <WFmt F> __device__ __forceinline__ int slot(int row, int p) {
  return F == WFmt::NVFP4 ? (p ^ ((((p >> 3) & 1) << 1) | (row & 1))) : (p ^ (row & 1));
}
__device__ __forceinline__ void cp16(uint32_t d, const void* src, int bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
int bps() { static const int v = getenv("HIVE_Q27_GT_BPS") ? std::max(1, atoi(getenv("HIVE_Q27_GT_BPS"))) : 2; return v; }   // target blocks per SM
int fixup() { static const int v = getenv("HIVE_Q27_GT_FIX") ? atoi(getenv("HIVE_Q27_GT_FIX")) : 1; return v; }   // 1 = last block adds the splits
int kc_small() { static const int v = getenv("HIVE_Q27_GT_KC1") ? atoi(getenv("HIVE_Q27_GT_KC1")) : 512; return v == 256 ? 256 : 512; }   // chunk at ≤ 16 rows
int splits(int N, bool gu, int K) {   // shape-only: enough blocks for ~bps per SM, whole KS units per split
  const int tiles = (N + (gu ? RB / 2 : RB) - 1) / (gu ? RB / 2 : RB), nks = K / KS;
  int ns = std::max(1, std::min(nks, (bps() * sm_count() + tiles - 1) / tiles));
  const int per = (nks + ns - 1) / ns;
  return (nks + per - 1) / per;
}
}  // namespace gt

template <WFmt F, bool GU, int NT, int KCT>
__global__ void __launch_bounds__(256, 2) gt_kernel(GMat W, GMat U, const bf16* __restrict__ x, int ldx, int M, int nsplit, bf16* yb, float* yf, int ldy,
                                                    float* part, int* cnt, int fix) {
  constexpr int E = gt::elems<F>(), SPAN = 4 * E, NSP = KCT / SPAN, STEPS = E / 4, ROWS = NT * 8, TPR = KCT / 8;   // pieces per staged row
  constexpr int RBK = GU ? gt::RB / 2 : gt::RB;
  extern __shared__ __align__(16) uint4 gt_smem[];   // [STG][ROWS][TPR] activation pieces; reused for the partial sums at the end
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int K = W.K, nks = K / gt::KS, per = (nks + nsplit - 1) / nsplit, sp = blockIdx.y;
  const int c0 = sp * per * (gt::KS / KCT), c1 = min(K / KCT, (sp + 1) * per * (gt::KS / KCT));   // chunks of this split
  const int n0 = blockIdx.x * RBK, tile = GU ? (warp & 3) : warp;
  const GMat& Wt = (GU && warp >= 4) ? U : W;
  const size_t rb = F == WFmt::FP8ROW ? (size_t)K : (size_t)K / 2;
  const uint8_t* wr[2];
  const uint8_t* sr[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int n = min(n0 + tile * 16 + g + 8 * h, W.N - 1);
    wr[h] = Wt.w + (size_t)n * rb;
    sr[h] = F == WFmt::NVFP4 ? reinterpret_cast<const uint8_t*>(Wt.s) + (size_t)n * (K / 16) : nullptr;
  }
  const uint32_t sm0 = (uint32_t)__cvta_generic_to_shared(gt_smem);
  auto stage = [&](int c, int b) {
    if (c < c1) {
      for (int i = threadIdx.x; i < ROWS * TPR; i += 256) {
        const int row = i / TPR, pc = i % TPR;
        const bf16* src = x + (size_t)min(row, M - 1) * ldx + (size_t)c * KCT + pc * 8;
        gt::cp16(sm0 + 16u * (uint32_t)((b * ROWS + row) * TPR + gt::slot<F>(row, pc)), src, row < M ? 16 : 0);
      }
    }
    gt::commit();
  };
  struct WB { uint4 w[NSP][2]; uint16_t gs[NSP][2]; };
  auto wload = [&](WB& B, int c) {
#pragma unroll
    for (int s = 0; s < NSP; ++s) {
      const int e0 = c * KCT + s * SPAN + t * E;
      const size_t bo = F == WFmt::FP8ROW ? (size_t)e0 : (size_t)e0 / 2;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        B.w[s][h] = __ldg(reinterpret_cast<const uint4*>(wr[h] + bo));
        if constexpr (F == WFmt::NVFP4) B.gs[s][h] = __ldg(reinterpret_cast<const uint16_t*>(sr[h] + e0 / 16));
      }
    }
  };
  float acc[NT][4];
#pragma unroll
  for (int ni = 0; ni < NT; ++ni) acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;
  if (c0 < c1) {
#pragma unroll
    for (int i = 0; i < gt::STG - 1; ++i) stage(c0 + i, i);
    WB cur, nxt;
    wload(cur, c0);
    for (int c = c0; c < c1; ++c) {
      const int b = (c - c0) % gt::STG;
      if (c + 1 < c1) wload(nxt, c + 1);
      gt::wait<gt::STG - 2>();
      __syncthreads();
      stage(c + gt::STG - 1, (c - c0 + gt::STG - 1) % gt::STG);
      const uint4* xb = gt_smem + (size_t)b * ROWS * TPR;
#pragma unroll
      for (int s = 0; s < NSP; ++s) {
        __half2 sc[2][2];
        if constexpr (F == WFmt::NVFP4) {
#pragma unroll
          for (int h = 0; h < 2; ++h) { sc[h][0] = sw::sc_h2((uint8_t)(cur.gs[s][h] & 0xff)); sc[h][1] = sw::sc_h2((uint8_t)(cur.gs[s][h] >> 8)); }
        }
#pragma unroll
        for (int jj = 0; jj < STEPS; jj += 2) {   // one 16-byte activation piece = two k-steps
          uint4 xq[NT];
          const int pc = (s * SPAN + t * E) / 8 + jj / 2;
#pragma unroll
          for (int ni = 0; ni < NT; ++ni) { const int row = ni * 8 + g; xq[ni] = xb[row * TPR + gt::slot<F>(row, pc)]; }
#pragma unroll
          for (int js = 0; js < 2; ++js) {
            const int j = jj + js;
            uint32_t a[4];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              if constexpr (F == WFmt::FP8ROW) {
                const uint16_t* hw16 = reinterpret_cast<const uint16_t*>(&cur.w[s][h]);
                a[h] = sw::fp8_bf2(hw16[2 * j]); a[2 + h] = sw::fp8_bf2(hw16[2 * j + 1]);
              } else {
                const uint8_t* by = reinterpret_cast<const uint8_t*>(&cur.w[s][h]);
                a[h] = sw::fp4_bf2(by[2 * j], sc[h][j / 4]); a[2 + h] = sw::fp4_bf2(by[2 * j + 1], sc[h][j / 4]);
              }
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
              const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xq[ni]);
              tc::mma(acc[ni], a, xw[2 * js], xw[2 * js + 1]);
            }
          }
        }
      }
      cur = nxt;
    }
  }
  gt::wait<0>();
  __syncthreads();
  float* pt = reinterpret_cast<float*>(gt_smem);   // [8 warps][NT][32 lanes][4]
#pragma unroll
  for (int ni = 0; ni < NT; ++ni)
#pragma unroll
    for (int e = 0; e < 4; ++e) pt[((warp * NT + ni) * 32 + lane) * 4 + e] = acc[ni][e];
  __syncthreads();
  // outputs: D element (tile row ln / 4 + 8 · (e ≥ 2), token ni · 8 + 2 · (ln % 4) + (e & 1)); GU: gate of warp w with up of warp w + 4
  constexpr int NOUT = (GU ? 4 : 8) * NT * 32 * 4;
  const size_t ldp = GU ? 2 * (size_t)W.N : (size_t)W.N;
  auto finish = [&](int row, int tok, float v, float u) {
    if constexpr (GU) {
      const float sg_ = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : W.g;
      const float su = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(U.s)[row] : U.g;
      yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(bfr(siluf_(bfr(v * sg_))) * bfr(u * su));
    } else {
      (void)u;
      const float scl = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : W.g;
      if (yb) yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(v * scl); else yf[(size_t)tok * ldy + row] = v * scl;
    }
  };
  for (int i = threadIdx.x; i < NOUT; i += 256) {
    const int e = i & 3, ln = (i >> 2) & 31, ni = (i >> 7) % NT, w = (i >> 7) / NT;
    const int row = n0 + w * 16 + ln / 4 + (e >= 2 ? 8 : 0), tok = ni * 8 + 2 * (ln % 4) + (e & 1);
    if (row >= W.N || tok >= M) continue;
    const float v = pt[((w * NT + ni) * 32 + ln) * 4 + e];
    const float u = GU ? pt[(((w + 4) * NT + ni) * 32 + ln) * 4 + e] : 0.f;
    if (nsplit > 1) {   // raw partials [split][M][N] (GU: [split][M][2N], up at N + row)
      float* pp = part + ((size_t)sp * M + tok) * ldp;
      pp[row] = v;
      if constexpr (GU) pp[W.N + row] = u;
      continue;
    }
    finish(row, tok, v, u);
  }
  if (nsplit > 1 && fix) {   // last-block fixup: the block that arrives last for this tile adds the splits in order (no reduce launch)
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) { last = atomicAdd(cnt + blockIdx.x, 1) == nsplit - 1; if (last) cnt[blockIdx.x] = 0; }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int nrow = min(RBK, W.N - n0);
    for (int i = threadIdx.x; i < nrow * M; i += 256) {
      const int tok = i / nrow, row = n0 + i % nrow;
      float v = 0.f, u = 0.f;
      for (int q = 0; q < nsplit; ++q) {
        const float* pp = part + ((size_t)q * M + tok) * ldp;
        v += __ldcg(pp + row);
        if constexpr (GU) u += __ldcg(pp + W.N + row);
      }
      finish(row, tok, v, u);
    }
  }
}
// split partials → outputs, splits added in order (then gt_kernel's epilogue)
template <WFmt F, bool GU>
__global__ void gt_reduce_kernel(GMat W, GMat U, const float* part, int M, int nsplit, bf16* yb, float* yf, int ldy) {
  const size_t ld = GU ? 2 * (size_t)W.N : (size_t)W.N;
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * W.N) return;
  const int tok = (int)(i / W.N), row = (int)(i % W.N);
  float v = 0.f, u = 0.f;
  for (int s = 0; s < nsplit; ++s) {
    const float* pp = part + ((size_t)s * M + tok) * ld;
    v += pp[row];
    if constexpr (GU) u += pp[W.N + row];
  }
  if constexpr (GU) {
    const float sg_ = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : W.g;
    const float su = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(U.s)[row] : U.g;
    yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(bfr(siluf_(bfr(v * sg_))) * bfr(u * su));
  } else {
    const float scl = F == WFmt::FP8ROW ? reinterpret_cast<const float*>(W.s)[row] : W.g;
    if (yb) yb[(size_t)tok * ldy + row] = __float2bfloat16_rn(v * scl); else yf[(size_t)tok * ldy + row] = v * scl;
  }
}
template <WFmt F, bool GU, int NT, int KCT>
void gt_go(const GMat& W, const GMat& U, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  const int ns = gt::splits(W.N, GU, W.K), fix = gt::fixup();
  constexpr size_t st_b = (size_t)gt::STG * NT * 8 * KCT * 2, pt_b = (size_t)8 * NT * 32 * 4 * 4, sm = st_b > pt_b ? st_b : pt_b;
  static bool attr = false;
  if (!attr) { CUDA_CHECK(cudaFuncSetAttribute(gt_kernel<F, GU, NT, KCT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm)); attr = true; }
  const int rbk = GU ? gt::RB / 2 : gt::RB;
  gt_kernel<F, GU, NT, KCT><<<dim3((W.N + rbk - 1) / rbk, ns), 256, sm, st>>>(W, U, x, ldx, M, ns, yb, yf, ldy, gt::g_part, gt::g_cnt, fix);
  if (ns > 1 && !fix) gt_reduce_kernel<F, GU><<<(unsigned)(((size_t)M * W.N + 255) / 256), 256, 0, st>>>(W, U, gt::g_part, M, ns, yb, yf, ldy);
}
template <bool GU>
bool gt_launch(const GMat& W, const GMat& U, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  if (sw_ver() != 30 || (W.fmt != WFmt::NVFP4 && W.fmt != WFmt::FP8ROW) || W.K % gt::KS != 0 || ldx % 8 != 0 || M < 1 || M > 32 || !gt::g_part) return false;
  const int ns = gt::splits(W.N, GU, W.K);
  if (ns > 1 && (size_t)ns * M * (GU ? 2 : 1) * W.N * 4 > gt::g_part_bytes) return false;
  const bool k5 = gt::kc_small() == 512;
#define Q27_GT(F) do { if (M <= 8) { if (k5) gt_go<F, GU, 1, 512>(W, U, x, ldx, M, yb, yf, ldy, st); else gt_go<F, GU, 1, 256>(W, U, x, ldx, M, yb, yf, ldy, st); } \
    else if (M <= 16) { if (k5) gt_go<F, GU, 2, 512>(W, U, x, ldx, M, yb, yf, ldy, st); else gt_go<F, GU, 2, 256>(W, U, x, ldx, M, yb, yf, ldy, st); } \
    else gt_go<F, GU, 4, 256>(W, U, x, ldx, M, yb, yf, ldy, st); } while (0)
  if (W.fmt == WFmt::NVFP4) Q27_GT(WFmt::NVFP4); else Q27_GT(WFmt::FP8ROW);
#undef Q27_GT
  CUDA_CHECK(cudaGetLastError());
  return true;
}
void gemv_prepare() {
  if (!a4::g_xq && ((getenv("HIVE_Q27_A4_MLP") && atoi(getenv("HIVE_Q27_A4_MLP")) >= 1) || (getenv("HIVE_Q27_A4_PROJ") && atoi(getenv("HIVE_Q27_A4_PROJ")) == 1))) {
    if (cudaMalloc(&a4::g_xq, (size_t)32 * a4::kMaxK / 2) != cudaSuccess || cudaMalloc(&a4::g_xs, (size_t)32 * a4::kMaxK / 16) != cudaSuccess ||
        cudaMalloc(&a4::g_xg, 32 * 4) != cudaSuccess) { cudaGetLastError(); a4::g_xq = nullptr; }
  }   // split-N partial scratch (outside any graph capture): 16 splits × 32 rows × 2 · 17408 rows fp32 ≈ 71 MB
  static const bool draft_gt = !(getenv("HIVE_Q27_DRAFT_GT") && strcmp(getenv("HIVE_Q27_DRAFT_GT"), "0") == 0);
  if ((sw_ver() == 30 || draft_gt) && !gt::g_part) {   // GEMM-tile split partials: ≤ 32 rows × splits × N (largest: down 5120 × 9, gate+up 2 × 34816 × 2) — 48 MB
    gt::g_part_bytes = (size_t)48 << 20;
    if (cudaMalloc(&gt::g_part, gt::g_part_bytes) != cudaSuccess) { cudaGetLastError(); gt::g_part = nullptr; gt::g_part_bytes = 0; }
    if (gt::g_part && (cudaMalloc(&gt::g_cnt, 65536 * 4) != cudaSuccess || cudaMemset(gt::g_cnt, 0, 65536 * 4) != cudaSuccess)) { cudaGetLastError(); cudaFree(gt::g_part); gt::g_part = nullptr; gt::g_part_bytes = 0; }
  }
  if ((sw_ver() != 20 && sw_ver() != 21) || sn::g_part) return;
  sn::g_part_bytes = (size_t)24 * 32 * 2 * 17408 * 4;   // ≤ 24 splits (17408 / 768 = 23) × 32 rows × gate+up
  if (cudaMalloc(&sn::g_part, sn::g_part_bytes) != cudaSuccess) { cudaGetLastError(); sn::g_part = nullptr; sn::g_part_bytes = 0; }
}
template <bool GU>
bool sw_launch(const GMat& W, const GMat& U, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  const int span = 4 * (W.fmt == WFmt::NVFP4 ? 32 : W.fmt == WFmt::FP8ROW ? 16 : 8);
  if (!sw_on() || W.K % span != 0 || ldx % 8 != 0 || M < 1 || M > 64) return false;
  if (gt_launch<GU>(W, U, x, ldx, M, yb, yf, ldy, st)) return true;
  if (M > 32) {   // 33..64 verify rows (HIVE_Q27_VROWS=64): 8 n-tiles in one pass over the weights — the per-token arithmetic is the
    //   M ≤ 32 path's (same spans per warp, same warp order, one independent mma column per token), so rows stay bit-identical across M.
    //   HIVE_Q27_SW64: 1 (default) = 1 tile, activations just in time, 2 spans per batch · 2 = the same with 1 span per batch
    //   (2 tiles would need 64 KiB of static shared partials; double-buffered activations of 8 n-tiles spill)
    static const int v64 = getenv("HIVE_Q27_SW64") ? atoi(getenv("HIVE_Q27_SW64")) : 1;
#define Q27_SW64(F) do { if (v64 == 2) sw_go<F, GU, 8, 1, 1, 8, 1, true>(W, U, x, ldx, M, yb, yf, ldy, st); \
    else sw_go<F, GU, 8, 1, 2, 8, 1, true>(W, U, x, ldx, M, yb, yf, ldy, st); } while (0)
    if (W.fmt == WFmt::BF16) Q27_SW64(WFmt::BF16); else if (W.fmt == WFmt::FP8ROW) Q27_SW64(WFmt::FP8ROW); else Q27_SW64(WFmt::NVFP4);
#undef Q27_SW64
    CUDA_CHECK(cudaGetLastError());
    return true;
  }
  if (sn_launch<GU>(W, U, x, ldx, M, yb, yf, ldy, st)) return true;
  if (sn2_launch<GU>(W, U, x, ldx, M, yb, yf, ldy, st)) return true;
#define Q27_SW(F, NT) sw_ver_go<F, GU, NT>(W, U, x, ldx, M, yb, yf, ldy, st)
  if (M <= 8) {
    if (W.fmt == WFmt::BF16) Q27_SW(WFmt::BF16, 1); else if (W.fmt == WFmt::FP8ROW) Q27_SW(WFmt::FP8ROW, 1); else Q27_SW(WFmt::NVFP4, 1);
  } else if (M <= 16) {
    if (W.fmt == WFmt::BF16) Q27_SW(WFmt::BF16, 2); else if (W.fmt == WFmt::FP8ROW) Q27_SW(WFmt::FP8ROW, 2); else Q27_SW(WFmt::NVFP4, 2);
  } else {
    if (W.fmt == WFmt::BF16) Q27_SW(WFmt::BF16, 4); else if (W.fmt == WFmt::FP8ROW) Q27_SW(WFmt::FP8ROW, 4); else Q27_SW(WFmt::NVFP4, 4);
  }
#undef Q27_SW
  CUDA_CHECK(cudaGetLastError());
  return true;
}
// Drafter rows (HIVE_Q27_DRAFT_GT, default on): the DFlash2 block of 8 rows per stream gives 33..64 rows at c5..c8, which ran as two 32-row passes
//   of the swapped GEMV (the drafter weights read twice: 2.08 ms of drafter GEMV per c8 round vs 1.57 at c4, nsys 2026-10-11). One pass of the
//   GEMM-tile kernel with 8 n-tiles instead. Drafts are not outputs (the verify decides every token), so the drafter needs no batch invariance.
bool gemv_draft(const GMat& W, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  static const bool on = !(getenv("HIVE_Q27_DRAFT_GT") && strcmp(getenv("HIVE_Q27_DRAFT_GT"), "0") == 0);
  if (!on || M <= 32 || M > 64 || (W.fmt != WFmt::NVFP4 && W.fmt != WFmt::FP8ROW) || W.K % gt::KS != 0 || ldx % 8 != 0 || !gt::g_part) return false;
  const int ns = gt::splits(W.N, false, W.K);
  if (ns > 1 && (size_t)ns * M * W.N * 4 > gt::g_part_bytes) return false;
  if (W.fmt == WFmt::NVFP4) gt_go<WFmt::NVFP4, false, 8, 128>(W, W, x, ldx, M, yb, yf, ldy, st);
  else gt_go<WFmt::FP8ROW, false, 8, 128>(W, W, x, ldx, M, yb, yf, ldy, st);
  CUDA_CHECK(cudaGetLastError());
  return true;
}
void gemv(const GMat& W, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= 64 && W.K % 32 == 0 && ldx % 8 == 0, "q27 gemv shape");
  if (W.act == 4 && gemv_a4(W, x, ldx, M, yb, yf, ldy, st)) return;
  if (sw_launch<false>(W, W, x, ldx, M, yb, yf, ldy, st)) return;
  if (M > 32) {   // without the swapped GEMV: 32-row passes
    gemv(W, x, ldx, 32, yb, yf, ldy, st);
    gemv(W, x + (size_t)32 * ldx, ldx, M - 32, yb ? yb + (size_t)32 * ldy : nullptr, yf ? yf + (size_t)32 * ldy : nullptr, ldy, st);
    return;
  }
  if (M > 16) {   // without the swapped GEMV: 16-row passes
    gemv(W, x, ldx, 16, yb, yf, ldy, st);
    gemv(W, x + (size_t)16 * ldx, ldx, M - 16, yb ? yb + (size_t)16 * ldy : nullptr, yf ? yf + (size_t)16 * ldy : nullptr, ldy, st);
    return;
  }
  if (M >= tc_min() || M > 8) { tc_launch<false>(W, W, x, ldx, M, yb, yf, ldy, st); return; }
  switch (M) {
    case 1: gemv_m<1>(W, x, ldx, yb, yf, ldy, st); break;
    case 2: gemv_m<2>(W, x, ldx, yb, yf, ldy, st); break;
    case 3: gemv_m<3>(W, x, ldx, yb, yf, ldy, st); break;
    case 4: gemv_m<4>(W, x, ldx, yb, yf, ldy, st); break;
    case 5: gemv_m<5>(W, x, ldx, yb, yf, ldy, st); break;
    case 6: gemv_m<6>(W, x, ldx, yb, yf, ldy, st); break;
    case 7: gemv_m<7>(W, x, ldx, yb, yf, ldy, st); break;
    default: gemv_m<8>(W, x, ldx, yb, yf, ldy, st); break;
  }
  CUDA_CHECK(cudaGetLastError());
}
template <int M, int R>
void gemv_gu_mr(const GMat& G, const GMat& U, const bf16* x, int ldx, bf16* act, int lda, cudaStream_t st) {
  const int rpb = 8 * (R / 2);
  dim3 grid((G.N + rpb - 1) / rpb), block(256);
  if (G.fmt == WFmt::BF16) gemv_gu_kernel<M, WFmt::BF16, R><<<grid, block, 0, st>>>(G, U, x, ldx, act, lda);
  else if (G.fmt == WFmt::FP8ROW) gemv_gu_kernel<M, WFmt::FP8ROW, R><<<grid, block, 0, st>>>(G, U, x, ldx, act, lda);
  else gemv_gu_kernel<M, WFmt::NVFP4, R><<<grid, block, 0, st>>>(G, U, x, ldx, act, lda);
}
template <int M>
void gemv_gu_m(const GMat& G, const GMat& U, const bf16* x, int ldx, bf16* act, int lda, cudaStream_t st) {
  const int rh = M == 1 ? pick_rows(G.N, 2) : (rows_per_warp<M>() >= 4 ? 2 : 1);   // rows of each matrix per warp
  if (rh >= 2 && rows_per_warp<M>() >= 4) gemv_gu_mr<M, (rows_per_warp<M>() >= 4 ? 4 : 2)>(G, U, x, ldx, act, lda, st);
  else gemv_gu_mr<M, 2>(G, U, x, ldx, act, lda, st);
}
void gemv_gu(const GMat& G, const GMat& U, const bf16* x, int ldx, int M, bf16* act, int lda, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= 64 && G.fmt == U.fmt && G.N == U.N && G.K == U.K && G.K % 32 == 0, "q27 gemv_gu shape");
  if (G.act == 4 && gemv_gu_a4(G, U, x, ldx, M, act, lda, st)) return;
  if (sw_launch<true>(G, U, x, ldx, M, act, nullptr, lda, st)) return;
  if (M > 32) {
    gemv_gu(G, U, x, ldx, 32, act, lda, st);
    gemv_gu(G, U, x + (size_t)32 * ldx, ldx, M - 32, act + (size_t)32 * lda, lda, st);
    return;
  }
  if (M > 16) {
    gemv_gu(G, U, x, ldx, 16, act, lda, st);
    gemv_gu(G, U, x + (size_t)16 * ldx, ldx, M - 16, act + (size_t)16 * lda, lda, st);
    return;
  }
  if (M >= tc_min() || M > 8) { tc_launch<true>(G, U, x, ldx, M, act, nullptr, lda, st); return; }
  switch (M) {
    case 1: gemv_gu_m<1>(G, U, x, ldx, act, lda, st); break;
    case 2: gemv_gu_m<2>(G, U, x, ldx, act, lda, st); break;
    case 3: gemv_gu_m<3>(G, U, x, ldx, act, lda, st); break;
    case 4: gemv_gu_m<4>(G, U, x, ldx, act, lda, st); break;
    case 5: gemv_gu_m<5>(G, U, x, ldx, act, lda, st); break;
    case 6: gemv_gu_m<6>(G, U, x, ldx, act, lda, st); break;
    case 7: gemv_gu_m<7>(G, U, x, ldx, act, lda, st); break;
    default: gemv_gu_m<8>(G, U, x, ldx, act, lda, st); break;
  }
  CUDA_CHECK(cudaGetLastError());
}
namespace {
__global__ void w4_amax_kernel(GMat W, float* amax) {
  const size_t nb = (size_t)W.N * W.K / 16;
  float m = 0.f;
  for (size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x; b < nb; b += (size_t)gridDim.x * blockDim.x) {
    const int n = (int)(b / (W.K / 16));
    const float sr = reinterpret_cast<const float*>(W.s)[n];
    const uint4 q = *reinterpret_cast<const uint4*>(W.w + b * 16);
    const uint16_t* h = reinterpret_cast<const uint16_t*>(&q);
#pragma unroll
    for (int j = 0; j < 8; ++j) { const float2 t = fp8x2(h[j]); m = fmaxf(m, fmaxf(fabsf(t.x), fabsf(t.y)) * fabsf(sr)); }
  }
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) atomicMax(reinterpret_cast<int*>(amax), __float_as_int(m));
}
__global__ void w4_quant_kernel(GMat W, const float* amax, uint8_t* w4, uint8_t* s4) {
  const size_t nb = (size_t)W.N * W.K / 16;
  const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nb) return;
  const int n = (int)(b / (W.K / 16));
  const float sr = reinterpret_cast<const float*>(W.s)[n];
  const float gs = *amax > 0.f ? 448.f * 6.f / *amax : 1.f;
  const uint4 q = *reinterpret_cast<const uint4*>(W.w + b * 16);
  const uint16_t* h = reinterpret_cast<const uint16_t*>(&q);
  float f[16], m = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) { const float2 t = fp8x2(h[j]); f[2 * j] = t.x * sr; f[2 * j + 1] = t.y * sr; m = fmaxf(m, fmaxf(fabsf(f[2 * j]), fabsf(f[2 * j + 1]))); }
  const __nv_fp8_storage_t sc = __nv_cvt_float_to_fp8(m * gs / 6.f, __NV_SATFINITE, __NV_E4M3);
  const __half_raw shr = __nv_cvt_fp8_to_halfraw(sc, __NV_E4M3);
  const float sd = __half2float(*reinterpret_cast<const __half*>(&shr));
  s4[b] = (uint8_t)sc;
  uint32_t lo = 0, hi = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint32_t v = sd > 0.f ? ((uint32_t)__nv_cvt_float_to_fp4(f[j] * gs / sd, __NV_E2M1, cudaRoundNearest) & 0xfu) : 0u;
    if (j < 8) lo |= v << (4 * j); else hi |= v << (4 * (j - 8));
  }
  reinterpret_cast<uint2*>(w4)[b] = make_uint2(lo, hi);
}
}  // namespace
float requant_nvfp4(const GMat& W, uint8_t* w4, uint8_t* s4, float* amax, cudaStream_t st) {
  HIVE_CHECK(W.fmt == WFmt::FP8ROW && W.K % 16 == 0, "q27 requant_nvfp4: FP8 matrix");
  CUDA_CHECK(cudaMemsetAsync(amax, 0, 4, st));
  w4_amax_kernel<<<1024, 256, 0, st>>>(W, amax);
  const size_t nb = (size_t)W.N * W.K / 16;
  w4_quant_kernel<<<(unsigned)((nb + 255) / 256), 256, 0, st>>>(W, amax, w4, s4);
  CUDA_CHECK(cudaGetLastError());
  float a = 0.f;
  CUDA_CHECK(cudaMemcpyAsync(&a, amax, 4, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  return a > 0.f ? a / (448.f * 6.f) : 1.f;
}
void add_norm(bf16* h, const bf16* y, const bf16* w, int M, int K, float eps, bf16* out, cudaStream_t st) {
  if (M <= 0) return;
  HIVE_CHECK(K <= 16 * 512, "q27 add_norm: K");
  add_norm_kernel<<<M, 512, 0, st>>>(h, y, w, K, eps, out);
  CUDA_CHECK(cudaGetLastError());
}
void dequant_pre(const GMat& W, bf16* out, cudaStream_t st) {
  HIVE_CHECK(W.K % 16 == 0, "q27 dequant: K % 16");
  dequant_kernel<<<blocks((size_t)W.N * W.K / 16), 256, 0, st>>>(W, out);
  CUDA_CHECK(cudaGetLastError());
}
void epilogue(const GMat& W, const float* acc, int lda, int M, bf16* yb, float* yf, int ldy, cudaStream_t st) {
  epilogue_kernel<<<blocks((size_t)M * W.N), 256, 0, st>>>(W, acc, lda, M, yb, yf, ldy);
  CUDA_CHECK(cudaGetLastError());
}
void rmsnorm1p(const bf16* x, int ldx, const bf16* w, int M, int K, int group, float eps, bf16* out, int ldo, cudaStream_t st) {
  if (M <= 0) return;
  rmsnorm1p_kernel<<<dim3(K / group, M), 256, 0, st>>>(x, ldx, w, K, group, eps, out, ldo);
  CUDA_CHECK(cudaGetLastError());
}
void add_round(bf16* h, const bf16* y, size_t n, cudaStream_t st) { if (n) add_round_kernel<<<blocks(n), 256, 0, st>>>(h, y, n); }
void silu_mul(bf16* g, int ldg, const bf16* u, int ldu, int M, int I, cudaStream_t st) {
  if (M > 0) silu_mul_kernel<<<blocks((size_t)M * I), 256, 0, st>>>(g, ldg, u, ldu, M, I);
}
void attn_gate(const float* att, const bf16* proj, int ld, int M, int H, int D, bf16* out, cudaStream_t st) {
  if (M > 0) attn_gate_kernel<<<blocks((size_t)M * H * D), 256, 0, st>>>(att, proj, ld, M, H, D, out);
}
const float2* g_rope_tab = nullptr;
int g_rope_tab_n = 0;
void set_rope_table(const float2* tab, int n) { g_rope_tab = tab; g_rope_tab_n = n; }
void attn_prep(const AttnRow* rows, int M, const bf16* proj, int ld, const bf16* q_norm, const bf16* k_norm, float eps, const int32_t* coords,
               const double* inv_freq, const int32_t* axis, int rot, int H, int Hkv, int D, bf16* q_out, bool kv_only, cudaStream_t st) {
  HIVE_CHECK(D == 256 && rot <= 128, "q27 attn_prep: head 256");
  if (M <= 0) return;
  attn_prep_kernel<<<M, 256, 0, st>>>(rows, proj, ld, q_norm, k_norm, eps, coords, inv_freq, axis, rot, H, Hkv, D, q_out, kv_only, g_rope_tab, g_rope_tab_n);
  CUDA_CHECK(cudaGetLastError());
}
// ---- decode / verify attention v2 (HIVE_Q27_ATTN2, default on): split keys (flash decoding). grid (rows, kv heads, kDecSplits): a block
//   takes a contiguous key range of one row and one kv head and scores it for all H / Hkv query heads at once (each K / V row read once);
//   4 warps interleave the keys (lane = 8 dims), online softmax per head; partials (max, sum, 256 values per head) are merged by
//   attn_combine_kernel. Split count fixed (graph replays keep the grid), the range adapts to the row's context.
constexpr int kDecSplits = 16, kDecRep = 6;
__global__ void __launch_bounds__(128) attn_dec_kernel(const AttnRow* rows, const bf16* q, int H, int Hkv, float scale, float* part) {
  constexpr int D = 256, R = kDecRep;
  const int m = blockIdx.x, g = blockIdx.y, sp = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const AttnRow Rw = rows[m];
  const int per = (Rw.n_ctx + kDecSplits - 1) / kDecSplits, lo = sp * per, hi = min(Rw.n_ctx, lo + per);
  float qv[R][8];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const uint4 u = *reinterpret_cast<const uint4*>(q + ((size_t)m * H + g * R + r) * D + lane * 8);
    const bf16* e = reinterpret_cast<const bf16*>(&u);
#pragma unroll
    for (int i = 0; i < 8; ++i) qv[r][i] = __bfloat162float(e[i]) * scale;
  }
  float mx[R], ls[R], acc[R][8];
#pragma unroll
  for (int r = 0; r < R; ++r) { mx[r] = -INFINITY; ls[r] = 0.f; for (int i = 0; i < 8; ++i) acc[r][i] = 0.f; }
  for (int j = lo + warp; j < hi; j += 4) {
    const size_t off = ((size_t)j * Hkv + g) * D + lane * 8;
    const uint4 ku = *reinterpret_cast<const uint4*>(Rw.k + off), vu = *reinterpret_cast<const uint4*>(Rw.v + off);
    const bf16* ke = reinterpret_cast<const bf16*>(&ku);
    const bf16* ve = reinterpret_cast<const bf16*>(&vu);
    float kf[8], vf[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { kf[i] = __bfloat162float(ke[i]); vf[i] = __bfloat162float(ve[i]); }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) s = fmaf(qv[r][i], kf[i], s);
      s = warp_sum(s);
      const float nm = fmaxf(mx[r], s);
      const float c = mx[r] == -INFINITY ? 0.f : __expf(mx[r] - nm), p = __expf(s - nm);
      ls[r] = ls[r] * c + p;
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[r][i] = fmaf(p, vf[i], acc[r][i] * c);
      mx[r] = nm;
    }
  }
  __shared__ float wm[4][R], wl[4][R], wa[4][R][D];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    if (lane == 0) { wm[warp][r] = mx[r]; wl[warp][r] = ls[r]; }
#pragma unroll
    for (int i = 0; i < 8; ++i) wa[warp][r][lane * 8 + i] = acc[r][i];
  }
  __syncthreads();
  float* pb = part + (((size_t)m * Hkv + g) * kDecSplits + sp) * R * (D + 2);
  for (int i = threadIdx.x; i < R * D; i += 128) {
    const int r = i / D, d = i % D;
    float M = -INFINITY;
    for (int w = 0; w < 4; ++w) M = fmaxf(M, wm[w][r]);
    float L = 0.f, A = 0.f;
    if (M != -INFINITY)
      for (int w = 0; w < 4; ++w) { if (wm[w][r] == -INFINITY) continue; const float f = __expf(wm[w][r] - M); L += wl[w][r] * f; A += wa[w][r][d] * f; }
    pb[r * (D + 2) + 2 + d] = A;
    if (d == 0) { pb[r * (D + 2)] = M; pb[r * (D + 2) + 1] = L; }
  }
}
// Grouped variant (HIVE_Q27_ATTN3=1, off by default — see attention()): the rows of one sequence in a verify (≤ 8 consecutive rows with the same KV cache) share one
//   block, and each K / V chunk of the split's key range is read from DRAM once into shared memory for all of them. attn_dec_kernel gives
//   every row its own blocks, so a verify of R rows read the whole cache R times — measured 2026-10-11 (600 W, prefill64k: TPOT 11.6 ms vs
//   4.6 at 1K; at 64K one row's cache is ~4.2 GB per step over the 16 attention layers). Arithmetic per row is that of attn_dec_kernel
//   unchanged — the same split boundaries (from the row's own context), 4 warps per row interleaving its keys in the same order, the same
//   merge — so the partials are bit-identical and decode / verify rows still agree bit for bit.
constexpr int kGrpRows = 4, kGrpChunk = 32;   // 4 rows × 4 warps = 512 threads (≤ 128 registers per thread)
__device__ __forceinline__ void dec_cp16(void* dst, const void* src) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void dec_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void dec_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__global__ void __launch_bounds__(128 * kGrpRows) attn_dec_grp_kernel(const AttnRow* rows, int M, const bf16* q, int H, int Hkv, float scale, float* part) {
  constexpr int D = 256, R = kDecRep;
  const int m = blockIdx.x, g = blockIdx.y, sp = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // group = consecutive rows on the same cache, cut every kGrpRows rows; only the group's first row's blocks work
  int st = m;
  while (st > 0 && rows[st - 1].k == rows[m].k && m - st < 64) --st;
  if ((m - st) % kGrpRows) return;
  int T = 1;
  while (T < kGrpRows && m + T < M && rows[m + T].k == rows[m].k) ++T;
  const int t = warp >> 2, sub = warp & 3;
  const bool active = t < T;
  const AttnRow Rw = rows[m + (active ? t : 0)];
  const int per = (Rw.n_ctx + kDecSplits - 1) / kDecSplits, lo = sp * per, hi = min(Rw.n_ctx, lo + per);
  // union of the group's ranges for this split
  int ulo = INT_MAX, uhi = 0;
  for (int i = 0; i < T; ++i) {
    const int n = rows[m + i].n_ctx, pi = (n + kDecSplits - 1) / kDecSplits, l = sp * pi, h = min(n, l + pi);
    if (l < h) { ulo = min(ulo, l); uhi = max(uhi, h); }
  }
  extern __shared__ __align__(16) unsigned char gsm[];
  bf16* sbuf = reinterpret_cast<bf16*>(gsm);   // [2 buffers][K | V][kGrpChunk][D] — chunk c + 1 is loaded (cp.async) while c is scored
  uint4 qp[R];   // q kept as bf16 (registers); bf16 → fp32 · scale on use gives the same value as attn_dec_kernel's precomputed qv
  float mx[R], ls[R], acc[R][8];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    qp[r] = *reinterpret_cast<const uint4*>(q + ((size_t)(m + (active ? t : 0)) * H + g * R + r) * D + lane * 8);
    mx[r] = -INFINITY; ls[r] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[r][i] = 0.f;
  }
  const bf16* kb = rows[m].k;
  const bf16* vb = rows[m].v;
  auto issue = [&](int c, int b) {
    const int nk = min(kGrpChunk, uhi - c);
    bf16* sk = sbuf + (size_t)b * 2 * kGrpChunk * D;
    bf16* sv = sk + kGrpChunk * D;
    for (int i = threadIdx.x; i < nk * (D / 8); i += blockDim.x) {   // 16-byte pieces of the chunk's K and V rows (kv head g)
      const int j = i / (D / 8), d8 = i % (D / 8);
      const size_t off = ((size_t)(c + j) * Hkv + g) * D + d8 * 8;
      dec_cp16(sk + j * D + d8 * 8, kb + off);
      dec_cp16(sv + j * D + d8 * 8, vb + off);
    }
    dec_commit();
  };
  if (ulo < uhi) issue(ulo, 0);
  for (int c = ulo, b = 0; c < uhi; c += kGrpChunk, b ^= 1) {
    const int nk = min(kGrpChunk, uhi - c);
    if (c + kGrpChunk < uhi) { issue(c + kGrpChunk, b ^ 1); dec_wait<1>(); } else dec_wait<0>();
    __syncthreads();
    const bf16* sk = sbuf + (size_t)b * 2 * kGrpChunk * D;
    const bf16* sv = sk + kGrpChunk * D;
    if (active && lo < hi) {
      // this warp's keys: j ≡ lo + sub (mod 4), lo ≤ j < hi, inside [c, c + nk) — the same sequence as attn_dec_kernel's warp `sub`
      const int first = max(c, lo);
      int j = first + ((lo + sub - first) % 4 + 4) % 4;
      const int end = min(c + nk, hi);
      for (; j < end; j += 4) {
        const uint4 ku = reinterpret_cast<const uint4*>(sk)[(j - c) * (D / 8) + lane], vu = reinterpret_cast<const uint4*>(sv)[(j - c) * (D / 8) + lane];
        const bf16* ke = reinterpret_cast<const bf16*>(&ku);
        const bf16* ve = reinterpret_cast<const bf16*>(&vu);
        float kf[8], vf[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { kf[i] = __bfloat162float(ke[i]); vf[i] = __bfloat162float(ve[i]); }
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const bf16* qe = reinterpret_cast<const bf16*>(&qp[r]);
          float s = 0.f;
#pragma unroll
          for (int i = 0; i < 8; ++i) s = fmaf(__bfloat162float(qe[i]) * scale, kf[i], s);
          s = warp_sum(s);
          const float nm = fmaxf(mx[r], s);
          const float cc = mx[r] == -INFINITY ? 0.f : __expf(mx[r] - nm), p = __expf(s - nm);
          ls[r] = ls[r] * cc + p;
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[r][i] = fmaf(p, vf[i], acc[r][i] * cc);
          mx[r] = nm;
        }
      }
    }
    __syncthreads();
  }
  // merge the 4 warps of each row (as attn_dec_kernel), one row at a time through the chunk buffer
  float* wm = reinterpret_cast<float*>(gsm);   // [4][R]
  float* wl = wm + 4 * R;                       // [4][R]
  float* wa = wl + 4 * R;                       // [4][R][D]
  for (int rr = 0; rr < T; ++rr) {
    if (t == rr) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (lane == 0) { wm[sub * R + r] = mx[r]; wl[sub * R + r] = ls[r]; }
#pragma unroll
        for (int i = 0; i < 8; ++i) wa[(sub * R + r) * D + lane * 8 + i] = acc[r][i];
      }
    }
    __syncthreads();
    float* pb = part + (((size_t)(m + rr) * Hkv + g) * kDecSplits + sp) * R * (D + 2);
    for (int i = threadIdx.x; i < R * D; i += blockDim.x) {
      const int r = i / D, d = i % D;
      float Mx = -INFINITY;
      for (int w = 0; w < 4; ++w) Mx = fmaxf(Mx, wm[w * R + r]);
      float L = 0.f, A = 0.f;
      if (Mx != -INFINITY)
        for (int w = 0; w < 4; ++w) { if (wm[w * R + r] == -INFINITY) continue; const float f = __expf(wm[w * R + r] - Mx); L += wl[w * R + r] * f; A += wa[(w * R + r) * D + d] * f; }
      pb[r * (D + 2) + 2 + d] = A;
      if (d == 0) { pb[r * (D + 2)] = Mx; pb[r * (D + 2) + 1] = L; }
    }
    __syncthreads();
  }
}
__global__ void attn_combine_kernel(const float* part, int H, int Hkv, float* out) {
  constexpr int D = 256, R = kDecRep;
  const int m = blockIdx.x / H, h = blockIdx.x % H, g = h / R, r = h % R, d = threadIdx.x;
  const float* pb = part + ((size_t)m * Hkv + g) * kDecSplits * R * (D + 2) + r * (D + 2);
  float M = -INFINITY;
  for (int s = 0; s < kDecSplits; ++s) M = fmaxf(M, pb[(size_t)s * R * (D + 2)]);
  float L = 0.f, A = 0.f;
  if (M != -INFINITY)
    for (int s = 0; s < kDecSplits; ++s) {
      const float* q = pb + (size_t)s * R * (D + 2);
      if (q[0] == -INFINITY) continue;
      const float f = __expf(q[0] - M);
      L += q[1] * f; A += q[2 + d] * f;
    }
  out[((size_t)m * H + h) * D + d] = L > 0.f ? A / L : 0.f;
}
float* g_attn_part = nullptr;
void attention_tc(const AttnRow* rows, int M, const bf16* q, int H, int Hkv, float scale, float* part, float* out, cudaStream_t st);
int g_attn_part_rows = 0;
void set_attention_scratch(float* p, int rows) { g_attn_part = p; g_attn_part_rows = rows; }
size_t attention_scratch_bytes(int rows, int Hkv) { return (size_t)rows * Hkv * kDecSplits * kDecRep * (256 + 2) * 4; }

void attention(const AttnRow* rows, int M, const bf16* q, int H, int Hkv, int D, float scale, float* out, cudaStream_t st) {
  static const bool v2 = !(getenv("HIVE_Q27_ATTN2") && strcmp(getenv("HIVE_Q27_ATTN2"), "0") == 0);
  // tensor-core decode / verify attention: default since 2026-10-11 (quality gate gs81: every category identical on / off; hived prefill32k
  //   post-prompt TPOT 8.76 → 5.98 ms); HIVE_Q27_ATTN4=0 keeps the scalar kernel
  static const bool v4 = !(getenv("HIVE_Q27_ATTN4") && strcmp(getenv("HIVE_Q27_ATTN4"), "0") == 0);
  if (v4 && g_attn_part && M <= g_attn_part_rows && D == 256 && H == Hkv * kDecRep) { attention_tc(rows, M, q, H, Hkv, scale, g_attn_part, out, st); return; }
  if (v2 && g_attn_part && M <= g_attn_part_rows && D == 256 && H == Hkv * kDecRep) {
    // HIVE_Q27_ATTN3=1 (off by default): measured 2026-10-11 slower (p32k spec 152.9 → 124.6 tok/s; hived prefill32k TPOT 8.90 → 9.21 ms) —
    //   the per-row blocks of one split already run together and share K / V in L2; the 32K cost is the scalar score / softmax path
    static const bool grp = getenv("HIVE_Q27_ATTN3") && strcmp(getenv("HIVE_Q27_ATTN3"), "1") == 0;
    constexpr size_t smem = std::max((size_t)2 * 2 * kGrpChunk * 256 * 2, (size_t)(8 * kDecRep + 4 * kDecRep * 256) * 4);
    static const bool attr = [] { return cudaFuncSetAttribute(attn_dec_grp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess; }();
    if (grp && attr && M > 1) attn_dec_grp_kernel<<<dim3(M, Hkv, kDecSplits), 128 * kGrpRows, smem, st>>>(rows, M, q, H, Hkv, scale, g_attn_part);
    else attn_dec_kernel<<<dim3(M, Hkv, kDecSplits), 128, 0, st>>>(rows, q, H, Hkv, scale, g_attn_part);
    attn_combine_kernel<<<M * H, 256, 0, st>>>(g_attn_part, H, Hkv, out);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  HIVE_CHECK(D == 256, "q27 attention: head 256");
  if (M <= 0) return;
  attention_kernel<<<M * H, 256, 0, st>>>(rows, q, H, Hkv, D, scale, out);
  CUDA_CHECK(cudaGetLastError());
}
// ---- prefill attention v2 (HIVE_Q27_FA2, default on): FlashAttention-2-style work split — 4 warps × 16 query rows own their rows end to
//   end (scores, online softmax and P·V in registers, no block-wide row state); K / V tiles of 32 keys double-buffered with cp.async;
//   XOR-swizzled shared rows (16-byte chunk c of row r at c ^ (r & 7)) for conflict-free ldmatrix; heavy (late) query tiles first.
//   Scores fp32 (scale · log2 e, exp2), probabilities rounded to bf16 for P·V (fp32 accumulate), row sums of the fp32 probabilities.
namespace fa2 {
constexpr int BQ = 64, BK = 32, D = 256, CH = D / 8;   // CH: 16-byte chunks per row
__device__ __forceinline__ int swz(int r, int c) { return r * CH + (c ^ (r & 7)); }   // chunk index in a swizzled tile
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  const int n = ok ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldm4(uint32_t (&r)[4], const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldm4t(uint32_t (&r)[4], const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ uint32_t pk(float a, float b) { const __nv_bfloat162 p = __floats2bfloat162_rn(a, b); return *reinterpret_cast<const uint32_t*>(&p); }
}  // namespace fa2

__global__ void __launch_bounds__(128) attention_prefill2_kernel(const bf16* __restrict__ q, int M, int pos0, const bf16* __restrict__ K,
                                                                  const bf16* __restrict__ V, int H, int Hkv, float scale, float* out) {
  using namespace fa2;
  extern __shared__ __align__(128) unsigned char smem[];
  uint4* Qs = reinterpret_cast<uint4*>(smem);            // [BQ][CH]
  uint4* Ks = Qs + BQ * CH;                               // [2][BK][CH]
  uint4* Vs = Ks + 2 * BK * CH;                           // [2][BK][CH]
  const int tile = gridDim.x - 1 - blockIdx.x, h = blockIdx.y, g = h / (H / Hkv), q0 = tile * BQ;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
  const int nq = min(BQ, M - q0);
  for (int i = tid; i < BQ * CH; i += 128) {
    const int r = i / CH, c = i % CH;
    cp16(Qs + swz(r, c), q + ((size_t)(q0 + min(r, nq - 1)) * H + h) * D + c * 8, r < nq);
  }
  const int last_pos = pos0 + q0 + nq - 1, nkt = last_pos / BK + 1;
  auto load_kv = [&](int kt, int b) {
    for (int i = tid; i < BK * CH; i += 128) {
      const int r = i / CH, c = i % CH, kp = kt * BK + r;
      const bool ok = kp <= last_pos;
      const size_t off = ((size_t)min(kp, last_pos) * Hkv + g) * D + c * 8;
      cp16(Ks + (b * BK) * CH + swz(r, c), K + off, ok);
      cp16(Vs + (b * BK) * CH + swz(r, c), V + off, ok);
    }
  };
  load_kv(0, 0);
  cp_commit();
  const int gq = lane >> 2, tq = lane & 3;
  const int wr0 = warp * 16, qp0 = pos0 + q0 + wr0 + gq, qp1 = qp0 + 8;   // this lane's two query positions
  const int wmax = pos0 + q0 + min(wr0 + 15, nq - 1);                    // the warp's last valid query position
  const float sl2 = scale * 1.4426950408889634f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  float o[32][4];
#pragma unroll
  for (int t = 0; t < 32; ++t) o[t][0] = o[t][1] = o[t][2] = o[t][3] = 0.f;
  for (int kt = 0; kt < nkt; ++kt) {
    const int b = kt & 1;
    if (kt + 1 < nkt) { load_kv(kt + 1, b ^ 1); cp_commit(); cp_wait<1>(); } else cp_wait<0>();
    __syncthreads();
    const int k0 = kt * BK;
    if (k0 <= wmax && wr0 < nq) {
      const uint4* Kb = Ks + (b * BK) * CH;
      const uint4* Vb = Vs + (b * BK) * CH;
      float s[BK / 8][4];
#pragma unroll
      for (int t = 0; t < BK / 8; ++t) s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < D / 16; ++ks) {
        uint32_t a[4];
        { const int r = wr0 + (lane & 7) + 8 * ((lane >> 3) & 1), c = 2 * ks + (lane >> 4); ldm4(a, Qs + swz(r, c)); }
#pragma unroll
        for (int np = 0; np < BK / 16; ++np) {
          uint32_t bb[4];
          const int mi = lane >> 3, r = np * 16 + (lane & 7) + 8 * (mi >> 1), c = 2 * ks + (mi & 1);
          ldm4(bb, Kb + swz(r, c));
          const uint32_t b0[2] = {bb[0], bb[1]}, b1[2] = {bb[2], bb[3]};
          fa::mma16816(s[2 * np], a, b0);
          fa::mma16816(s[2 * np + 1], a, b1);
        }
      }
      const bool diag = k0 + BK - 1 > pos0 + q0 + wr0;   // some key of the tile lies after some row of the warp
      float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
      for (int t = 0; t < BK / 8; ++t)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int kp = k0 + t * 8 + 2 * tq + e;
          float v0 = s[t][e] * sl2, v1 = s[t][2 + e] * sl2;
          if (diag) { if (kp > qp0) v0 = -INFINITY; if (kp > qp1) v1 = -INFINITY; }
          s[t][e] = v0; s[t][2 + e] = v1;
          mx0 = fmaxf(mx0, v0); mx1 = fmaxf(mx1, v1);
        }
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
      const float c0 = n0 == -INFINITY ? 1.f : exp2f(m0 - n0), c1 = n1 == -INFINITY ? 1.f : exp2f(m1 - n1);
      m0 = n0; m1 = n1;
      float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
      for (int t = 0; t < BK / 8; ++t)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float p0 = n0 == -INFINITY ? 0.f : exp2f(s[t][e] - n0), p1 = n1 == -INFINITY ? 0.f : exp2f(s[t][2 + e] - n1);
          s[t][e] = p0; s[t][2 + e] = p1; ps0 += p0; ps1 += p1;
        }
      l0 = l0 * c0 + ps0; l1 = l1 * c1 + ps1;   // per-lane partial sums (the quad is added at the end)
#pragma unroll
      for (int t = 0; t < 32; ++t) { o[t][0] *= c0; o[t][1] *= c0; o[t][2] *= c1; o[t][3] *= c1; }
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk) {
        const uint32_t a[4] = {pk(s[2 * kk][0], s[2 * kk][1]), pk(s[2 * kk][2], s[2 * kk][3]), pk(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                               pk(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
        for (int dp = 0; dp < 16; ++dp) {   // 16 pairs of 8-column d tiles
          uint32_t bb[4];
          const int mi = lane >> 3, r = kk * 16 + (lane & 7) + 8 * (mi & 1), c = 2 * dp + (mi >> 1);
          ldm4t(bb, Vb + swz(r, c));
          const uint32_t b0[2] = {bb[0], bb[1]}, b1[2] = {bb[2], bb[3]};
          fa::mma16816(o[2 * dp], a, b0);
          fa::mma16816(o[2 * dp + 1], a, b1);
        }
      }
    }
    __syncthreads();
  }
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const int r0 = wr0 + gq, r1 = r0 + 8;
  const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
  for (int t = 0; t < 32; ++t) {
    const int d = t * 8 + 2 * tq;
    if (r0 < nq) *reinterpret_cast<float2*>(out + ((size_t)(q0 + r0) * H + h) * D + d) = make_float2(o[t][0] * i0, o[t][1] * i0);
    if (r1 < nq) *reinterpret_cast<float2*>(out + ((size_t)(q0 + r1) * H + h) * D + d) = make_float2(o[t][2] * i1, o[t][3] * i1);
  }
}

// ---- decode / verify attention on tensor cores (default; HIVE_Q27_ATTN4=0 = the scalar kernel): grid (rows, kv heads, kDecSplits) as attn_dec_kernel, but per 32-key tile
//   the 6 query heads of the row (an m16 tile, rows 6..15 zero) are scored by mma (each warp one 8-key n-tile over the 256 dims), the online
//   softmax runs once per row in a fixed order, and P · V is mma again (each warp 64 of the 256 dims). attn_dec_kernel's scalar dot product +
//   warp shuffles per key was the long-context cost: measured 2026-10-11 (hived, 32K prompt) verify R8 13.3 → 18.0 ms while the cache read is
//   ~1.3 ms. Split ranges are whole 32-key tiles from the row's own context; consecutive rows of one sequence share a block in pairs (row A's
//   heads in tile rows 0..5, row B's in 8..13 — the cache is read once for both). A row's result depends only on its own query, its own tiles
//   and keys (everything else is masked, and a fully masked tile is an exact no-op), so decode and verify rows stay bit-identical.
//   Partials in attn_dec_kernel's layout (max in natural-log units) for attn_combine_kernel.
namespace tcd {
constexpr int BK = 32, D = 256, CH = D / 8, PS = BK + 8;
__device__ __forceinline__ void ldm2(uint32_t (&r)[2], const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
}  // namespace tcd
__global__ void __launch_bounds__(128) attn_tcdec_kernel(const AttnRow* rows, int M, const bf16* q, int H, int Hkv, float scale, float* part) {
  using namespace fa2;
  constexpr int R = kDecRep;
  extern __shared__ __align__(128) unsigned char tsm[];
  uint4* Qs = reinterpret_cast<uint4*>(tsm);                         // [16][CH] swizzled: rows 0..5 = row A's heads, 8..13 = row B's
  uint4* Ks = Qs + 16 * CH;                                           // [2][BK][CH]
  uint4* Vs = Ks + 2 * tcd::BK * CH;                                  // [2][BK][CH]
  float* Ss = reinterpret_cast<float*>(Vs + 2 * tcd::BK * CH);        // [16][BK + 1] scores (log2 units)
  bf16* Ps = reinterpret_cast<bf16*>(Ss + 16 * (tcd::BK + 1));        // [16][PS] probabilities
  float* st = reinterpret_cast<float*>(Ps + 16 * tcd::PS);            // m [16] · l [16] · c [16]
  const int m = blockIdx.x, g = blockIdx.y, sp = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
  // rows of one sequence go in pairs (the cache read once for both); a pair's second row has no blocks of its own
  int s0 = m;
  while (s0 > 0 && rows[s0 - 1].k == rows[m].k && m - s0 < 64) --s0;
  if ((m - s0) & 1) return;
  const int T2 = (m + 1 < M && rows[m + 1].k == rows[m].k) ? 2 : 1;
  int nr[2], lo[2], hi[2];   // per row: context, own split range in whole tiles
  int ulo = INT_MAX, uhi = 0;
  for (int i = 0; i < 2; ++i) {
    nr[i] = i < T2 ? rows[m + i].n_ctx : 0;
    const int ntl = (nr[i] + tcd::BK - 1) / tcd::BK, per = (ntl + kDecSplits - 1) / kDecSplits;
    lo[i] = sp * per; hi[i] = i < T2 ? min(ntl, lo[i] + per) : 0;
    if (lo[i] < hi[i]) { ulo = min(ulo, lo[i]); uhi = max(uhi, hi[i]); }
  }
  const AttnRow Rw = rows[m];
  for (int i = tid; i < 16 * tcd::CH; i += 128) {
    const int r = i / tcd::CH, c = i % tcd::CH, row = r >> 3, h = r & 7;
    const bool ok = h < R && row < T2;
    cp16(Qs + swz(r, c), q + ((size_t)(m + (ok ? row : 0)) * H + g * R + (ok ? h : 0)) * tcd::D + c * 8, ok);
  }
  cp_commit();   // group 1: Q
  auto load_kv = [&](int kt, int b) {
    for (int i = tid; i < tcd::BK * tcd::CH; i += 128) {
      const int r = i / tcd::CH, c = i % tcd::CH, kp = kt * tcd::BK + r, nmax = nr[T2 - 1];
      const bool ok = kp < nmax;
      const size_t off = ((size_t)min(kp, nmax - 1) * Hkv + g) * tcd::D + c * 8;
      cp16(Ks + (b * tcd::BK) * tcd::CH + swz(r, c), Rw.k + off, ok);
      cp16(Vs + (b * tcd::BK) * tcd::CH + swz(r, c), Rw.v + off, ok);
    }
  };
  if (ulo < uhi) load_kv(ulo, 0);
  cp_commit();
  if (tid < 16) { st[tid] = -INFINITY; st[16 + tid] = 0.f; st[32 + tid] = 1.f; }
  for (int i = tid; i < 16 * tcd::PS; i += 128) Ps[i] = __float2bfloat16_rn(0.f);   // padding rows stay 0
  const int gq = lane >> 2, tq = lane & 3;
  cp_wait<1>();
  __syncthreads();
  uint32_t qa[tcd::D / 16][4];   // the 16 Q fragments, loaded once
#pragma unroll
  for (int ks = 0; ks < tcd::D / 16; ++ks) { const int r = (lane & 7) + 8 * ((lane >> 3) & 1), c = 2 * ks + (lane >> 4); ldm4(qa[ks], Qs + swz(r, c)); }
  const float sl2 = scale * 1.4426950408889634f;
  float o[8][4];
#pragma unroll
  for (int t = 0; t < 8; ++t) o[t][0] = o[t][1] = o[t][2] = o[t][3] = 0.f;
  for (int kt = ulo, b = 0; kt < uhi; ++kt, b ^= 1) {
    if (kt + 1 < uhi) { load_kv(kt + 1, b ^ 1); cp_commit(); cp_wait<1>(); } else cp_wait<0>();
    __syncthreads();
    const uint4* Kb = Ks + (b * tcd::BK) * tcd::CH;
    const uint4* Vb = Vs + (b * tcd::BK) * tcd::CH;
    const int k0 = kt * tcd::BK;
    {   // scores of keys 8 · warp … + 7; a key outside the row's own tiles or context is masked (a fully masked tile changes nothing)
      float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ks = 0; ks < tcd::D / 16; ++ks) {
        uint32_t bb[2];
        { const int r = warp * 8 + (lane & 7), c = 2 * ks + ((lane >> 3) & 1); tcd::ldm2(bb, Kb + swz(r, c)); }
        fa::mma16816(sc, qa[ks], bb);
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int vr = e >= 2 ? 1 : 0, row = gq + 8 * vr, key = warp * 8 + 2 * tq + (e & 1);
        const bool ok = vr < T2 && kt >= lo[vr] && kt < hi[vr] && k0 + key < nr[vr];
        Ss[row * (tcd::BK + 1) + key] = ok ? sc[e] * sl2 : -INFINITY;
      }
    }
    __syncthreads();
    if (tid < 96) {   // online softmax: tile rows 0..5 and 8..13 (8 threads per row, 4 keys each, fixed shuffle trees)
      const int rr = tid / 8, row = (rr / R) * 8 + rr % R, j = tid % 8;
      const float* sr = Ss + row * (tcd::BK + 1);
      float sv[4], mx = -INFINITY;
#pragma unroll
      for (int i = 0; i < 4; ++i) { sv[i] = sr[4 * j + i]; mx = fmaxf(mx, sv[i]); }
#pragma unroll
      for (int o2 = 4; o2 > 0; o2 >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o2));
      const float mo = st[row], nm = fmaxf(mo, mx);
      float ps = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float p = nm == -INFINITY ? 0.f : exp2f(sv[i] - nm);
        Ps[row * tcd::PS + 4 * j + i] = __float2bfloat16_rn(p);
        ps += p;
      }
#pragma unroll
      for (int o2 = 4; o2 > 0; o2 >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o2);
      __syncwarp();
      if (j == 0) { const float c = nm == -INFINITY ? 1.f : exp2f(mo - nm); st[row] = nm; st[16 + row] = st[16 + row] * c + ps; st[32 + row] = c; }
    }
    __syncthreads();
    {   // O (rows g / g + 8, this warp's 64 dims) · c + P · V
      const float c0 = st[32 + gq], c1 = st[32 + gq + 8];
#pragma unroll
      for (int t = 0; t < 8; ++t) { o[t][0] *= c0; o[t][1] *= c0; o[t][2] *= c1; o[t][3] *= c1; }
#pragma unroll
      for (int kk = 0; kk < tcd::BK / 16; ++kk) {
        uint32_t a[4];
        { const int r = (lane & 7) + 8 * ((lane >> 3) & 1), c = kk * 2 + (lane >> 4); ldm4(a, Ps + r * tcd::PS + c * 8); }
#pragma unroll
        for (int dp = 0; dp < 4; ++dp) {
          uint32_t bb[4];
          const int mi = lane >> 3, r = kk * 16 + (lane & 7) + 8 * (mi & 1), c = 2 * (warp * 4 + dp) + (mi >> 1);
          ldm4t(bb, Vb + swz(r, c));
          const uint32_t b0[2] = {bb[0], bb[1]}, b1[2] = {bb[2], bb[3]};
          fa::mma16816(o[2 * dp], a, b0);
          fa::mma16816(o[2 * dp + 1], a, b1);
        }
      }
    }
    __syncthreads();
  }
  // partials per row of the pair (attn_dec_kernel's layout; a row with an empty own range gets max −inf)
  for (int vr = 0; vr < T2; ++vr) {
    float* pb = part + (((size_t)(m + vr) * Hkv + g) * kDecSplits + sp) * R * (tcd::D + 2);
    if (gq < R) {
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const int d = warp * 64 + t * 8 + 2 * tq;
        pb[gq * (tcd::D + 2) + 2 + d] = o[t][2 * vr];
        pb[gq * (tcd::D + 2) + 2 + d + 1] = o[t][2 * vr + 1];
      }
    }
    if (tid < R) {
      const float mm = st[8 * vr + tid];
      pb[tid * (tcd::D + 2)] = mm == -INFINITY ? -INFINITY : mm * 0.6931471805599453f;
      pb[tid * (tcd::D + 2) + 1] = st[16 + 8 * vr + tid];
    }
  }
}
void attention_tc(const AttnRow* rows, int M, const bf16* q, int H, int Hkv, float scale, float* part, float* out, cudaStream_t st) {
  const size_t smem = (size_t)(16 + 4 * tcd::BK) * tcd::CH * 16 + (size_t)16 * (tcd::BK + 1) * 4 + (size_t)16 * tcd::PS * 2 + 48 * 4;
  static const bool attr = cudaFuncSetAttribute(attn_tcdec_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess;
  (void)attr;
  attn_tcdec_kernel<<<dim3(M, Hkv, kDecSplits), 128, smem, st>>>(rows, M, q, H, Hkv, scale, part);
  attn_combine_kernel<<<M * H, 256, 0, st>>>(part, H, Hkv, out);
  CUDA_CHECK(cudaGetLastError());
}

void attention_prefill(const bf16* q, int M, int pos0, const bf16* K, const bf16* V, int H, int Hkv, int D, float scale, float* out, cudaStream_t st) {
  static const bool v2 = !(getenv("HIVE_Q27_FA2") && strcmp(getenv("HIVE_Q27_FA2"), "0") == 0);
  if (v2 && D == fa2::D && M > 0) {
    const size_t smem2 = (size_t)(fa2::BQ + 4 * fa2::BK) * fa2::CH * 16;
    static bool attr2 = false;
    if (!attr2) { CUDA_CHECK(cudaFuncSetAttribute(attention_prefill2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2)); attr2 = true; }
    attention_prefill2_kernel<<<dim3((M + fa2::BQ - 1) / fa2::BQ, H), 128, smem2, st>>>(q, M, pos0, K, V, H, Hkv, scale, out);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  HIVE_CHECK(D == fa::D, "q27 attention_prefill: head 256");
  if (M <= 0) return;
  const size_t smem = (size_t)(fa::BQ + 2 * fa::BK) * fa::LDS * 2 + (size_t)fa::BQ * (fa::BK + 8) * 2 + 3 * fa::BQ * 4;
  static bool attr = false;
  if (!attr) { CUDA_CHECK(cudaFuncSetAttribute(attention_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
  attention_prefill_kernel<<<dim3((M + fa::BQ - 1) / fa::BQ, H), 256, smem, st>>>(q, M, pos0, K, V, H, Hkv, scale, out);
  CUDA_CHECK(cudaGetLastError());
}
// ---- chunked Gated DeltaNet recurrence for prefill (one segment, ≥ 128 rows; HIVE_Q27_GDN_CHUNK=0 keeps the row loop) ------------------
// Per value head, chunks of 64 rows with S0 the state before the chunk, Lg_t = Σ_{i ≤ t} log e_i (within the chunk), u_i the row update
//   (dl in the row loop): u = T · W with W_t = β_t (v_t − exp(Lg_t) · S0ᵀ k_t) and T = (I + A)⁻¹, A_ti = β_t exp(Lg_t − Lg_i) (k_t · k_i) (i < t);
//   o_t = exp(Lg_t) · S0ᵀ q_t + Σ_{i ≤ t} exp(Lg_t − Lg_i) (q_t · k_i) u_i;  S ← exp(Lg_C) · S0 + Σ_i exp(Lg_C − Lg_i) k_i u_iᵀ.
//   (This project's derivation from the row recurrence above; fp32 throughout, another summation order than the row loop.)
namespace gc {
constexpr int CK = 64;
constexpr int kQReg = (CK * 132 > 2 * CK * (CK + 1)) ? CK * 132 : 2 * CK * (CK + 1);   // the scan's q-tile / T·M region (floats)
}
// kernel 1 (chunks in parallel): T [C][C], M = masked exp-weighted Q·Kᵀ [C][C], Lg [C] per (chunk, value head)
__global__ void __launch_bounds__(256) gdn_chunk_prep_kernel(const GdnSeg* segs, int KH, int VH, const float* qk, const float* gb, float* Tw,
                                                              float* Mw, float* Lw) {
  using namespace gc;
  const GdnSeg S = segs[0];
  const int ch = blockIdx.x, vh = blockIdx.y, rep = VH / KH, kh = vh / rep, KW = KH * 128, tid = threadIdx.x;
  const int t0 = ch * CK, nr = min(CK, S.n - t0);
  extern __shared__ float smp[];   // ks [CK][129] · qs [CK][129] · A [CK][CK + 1] · lg [CK] · bt [CK]
  float (*ks)[129] = reinterpret_cast<float (*)[129]>(smp);
  float (*qs)[129] = reinterpret_cast<float (*)[129]>(smp + CK * 129);
  float (*A)[CK + 1] = reinterpret_cast<float (*)[CK + 1]>(smp + 2 * CK * 129);
  float* lg = smp + 2 * CK * 129 + CK * (CK + 1);
  float* bt = lg + CK;
  for (int i = tid; i < CK * 128; i += 256) {
    const int r = i / 128, d = i % 128;
    const size_t row = (size_t)S.row0 + t0 + r;
    ks[r][d] = r < nr ? qk[row * 2 * KW + KW + (size_t)kh * 128 + d] : 0.f;
    qs[r][d] = r < nr ? qk[row * 2 * KW + (size_t)kh * 128 + d] : 0.f;
  }
  if (tid < CK) {
    const size_t row = (size_t)S.row0 + t0 + tid;
    // log of the decay gate, floored: a gate that underflowed to 0 (large softplus) gave log 0 = −inf and −inf − (−inf) = NaN in every
    //   exp(Lg_t − Lg_j) after it (measured 2026-10-10: hived tool prompts / a 32K ruler prompt decoded garbage; the row loop multiplies by 0
    //   and was fine). e < 1e-30 contributes exp(−69) — nothing — so the floor does not change the result.
    lg[tid] = tid < nr ? logf(fmaxf(gb[row * 2 * VH + vh], 1e-30f)) : 0.f;
    bt[tid] = tid < nr ? gb[row * 2 * VH + VH + vh] : 0.f;
  }
  __syncthreads();
  if (tid == 0) for (int r = 1; r < CK; ++r) lg[r] += lg[r - 1];   // inclusive prefix sum (64 adds)
  __syncthreads();
  float* T = Tw + ((size_t)ch * VH + vh) * CK * CK;
  float* Mq = Mw + ((size_t)ch * VH + vh) * CK * CK;
  {   // 4 × 4 (t, j) block per thread: k_t · k_j and q_t · k_j
    const int tb = (tid / 16) * 4, jb = (tid % 16) * 4;
    float kk[4][4] = {}, qq[4][4] = {};
    for (int d = 0; d < 128; ++d) {
      float kt[4], qt[4], kj[4];
#pragma unroll
      for (int x = 0; x < 4; ++x) { kt[x] = ks[tb + x][d]; qt[x] = qs[tb + x][d]; kj[x] = ks[jb + x][d]; }
#pragma unroll
      for (int x = 0; x < 4; ++x)
#pragma unroll
        for (int z = 0; z < 4; ++z) { kk[x][z] = fmaf(kt[x], kj[z], kk[x][z]); qq[x][z] = fmaf(qt[x], kj[z], qq[x][z]); }
    }
#pragma unroll
    for (int x = 0; x < 4; ++x)
#pragma unroll
      for (int z = 0; z < 4; ++z) {
        const int t = tb + x, j = jb + z;
        const float dec = j <= t ? expf(lg[t] - lg[j]) : 0.f;
        A[t][j] = j < t ? bt[t] * dec * kk[x][z] : 0.f;
        Mq[t * CK + j] = j <= t ? dec * qq[x][z] : 0.f;
      }
  }
  if (tid < CK) Lw[((size_t)ch * VH + vh) * CK + tid] = lg[tid];
  __syncthreads();
  // T = (I + A)⁻¹ by forward substitution, one column per thread (columns in shared memory, reusing the k tile):
  //   T[t][j] = δ_tj − Σ_{j ≤ m < t} A[t][m] T[m][j]
  float (*Ts)[CK + 1] = reinterpret_cast<float (*)[CK + 1]>(&ks[0][0]);
  __syncthreads();
  {   // column j = tid / 4 by 4 lanes (m ≡ part mod 4), quad sum by butterfly; columns are independent (no block barrier inside)
    const int j = tid >> 2, part = tid & 3;
    for (int t = 0; t < CK; ++t) {
      float sum = 0.f;
      for (int m = j + part; m < t; m += 4) sum = fmaf(A[t][m], Ts[m][j], sum);
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      if (part == 0) Ts[t][j] = (t == j ? 1.f : 0.f) - (t > j ? sum : 0.f);
      __syncwarp();
    }
  }
  __syncthreads();
  for (int i = tid; i < CK * CK; i += 256) T[i] = Ts[i / CK][i % CK];
}

// kernel 2 (chunks in order): per (value head, 16 state columns), 256 threads; S0 [128][16] in shared memory
__global__ void __launch_bounds__(256) gdn_chunk_scan_kernel(const GdnSeg* segs, const bf16* y, int C, int KH, int VH, const float* qk,
                                                              const float* gb, const float* Tw, const float* Mw, const float* Lw, float* o) {
  using namespace gc;
  extern __shared__ float sm[];
  float* S0 = sm;                       // [128][16]
  // k / q rows padded to 132 floats (16-byte aligned rows: float4 reads of a row, conflict-free across the warp's 8 rows) — 2026-10-11
  //   ncu: the kernel was shared-memory bound (stalls short scoreboard / MIO throttle; L1 73 %); the arithmetic order is unchanged
  constexpr int KS = 132;
  float* ks = S0 + 128 * 16;            // [CK][KS]
  float* qs = ks + CK * KS;             // [CK][KS]
  float* W = qs + gc::kQReg;            // [CK][16]  (the q tile region also holds T and M [CK][CK + 1] after Q · S0)
  float* U = W + CK * 16;               // [CK][16]
  float* lg = U + CK * 16;              // [CK]
  float* fac = lg + CK;                 // [CK] exp(Lg_last − Lg_i)
  const GdnSeg S = segs[0];
  const int vh = blockIdx.x, c0 = blockIdx.y * 16, rep = VH / KH, kh = vh / rep, KW = KH * 128, tid = threadIdx.x;
  float* st = S.state + (size_t)vh * 128 * 128;
  for (int i = tid; i < 128 * 16; i += 256) S0[i] = st[(size_t)(i / 16) * 128 + c0 + i % 16];
  const int nch = (S.n + CK - 1) / CK;
  const int tr = tid / 4, tc = (tid % 4) * 4;   // row-phase layout: row tr, 4 columns tc..tc+3
  // the next chunk's k / q tiles and T / M are loaded into registers while the current chunk is computed (2026-10-11: one block per SM —
  //   the un-prefetched tile loads of every chunk were exposed; the arithmetic and its order are unchanged)
  float4 kv4[8], qv4[8], t4[4], m4[4];
  auto fetch_kq = [&](int c) {
    const int t0c = c * CK, nrc = min(CK, S.n - t0c);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int i = tid + 256 * u, r = i / 32, d4 = i % 32;
      const size_t row = (size_t)S.row0 + t0c + r;
      const bool ok = c < nch && r < nrc;
      kv4[u] = ok ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
      qv4[u] = ok ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
  };
  auto fetch_tm = [&](int c) {
    if (c >= nch) return;
    const float* Tg = Tw + ((size_t)c * VH + vh) * CK * CK;
    const float* Mg = Mw + ((size_t)c * VH + vh) * CK * CK;
#pragma unroll
    for (int u = 0; u < 4; ++u) { const int i = tid + 256 * u; t4[u] = reinterpret_cast<const float4*>(Tg)[i]; m4[u] = reinterpret_cast<const float4*>(Mg)[i]; }
  };
  float bv_b, bv_v[4];   // β and v of row tr (columns tc..tc+3), one chunk ahead
  auto fetch_bv = [&](int c) {
    const int t0c = c * CK, nrc = min(CK, S.n - t0c);
    const size_t row = (size_t)S.row0 + t0c + tr;
    const bool ok = c < nch && tr < nrc;
    bv_b = ok ? gb[row * 2 * VH + VH + vh] : 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) bv_v[e] = ok ? __bfloat162float(y[row * C + 2 * KW + (size_t)vh * 128 + c0 + tc + e]) : 0.f;
  };
  fetch_kq(0);
  fetch_tm(0);
  fetch_bv(0);
  for (int ch = 0; ch < nch; ++ch) {
    const int t0 = ch * CK, nr = min(CK, S.n - t0);
    __syncthreads();
    {   // k / q tiles (fetched one chunk ahead) into shared memory, then the next chunk's are requested
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int i = tid + 256 * u, r = i / 32, d = (i % 32) * 4;
        *reinterpret_cast<float4*>(ks + r * KS + d) = kv4[u];
        *reinterpret_cast<float4*>(qs + r * KS + d) = qv4[u];
      }
      fetch_kq(ch + 1);
    }
    if (tid < CK) lg[tid] = Lw[((size_t)ch * VH + vh) * CK + tid];
    __syncthreads();
    if (tid < CK) fac[tid] = expf(lg[CK - 1] - lg[tid]);
    // P = K · S0 and Q · S0 (row tr, columns tc..tc+3); W_t = β_t (v_t − exp(Lg_t) P_t)
    float p[4] = {0.f, 0.f, 0.f, 0.f}, qsv[4] = {0.f, 0.f, 0.f, 0.f};
    for (int d = 0; d < 128; d += 4) {
      const float4 k4 = *reinterpret_cast<const float4*>(ks + tr * KS + d), q4 = *reinterpret_cast<const float4*>(qs + tr * KS + d);
      const float kv[4] = {k4.x, k4.y, k4.z, k4.w}, qv[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
      for (int dd = 0; dd < 4; ++dd) {
        const float4 s4 = *reinterpret_cast<const float4*>(S0 + (d + dd) * 16 + tc);
        const float sv[4] = {s4.x, s4.y, s4.z, s4.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { p[e] = fmaf(kv[dd], sv[e], p[e]); qsv[e] = fmaf(qv[dd], sv[e], qsv[e]); }
      }
    }
    const size_t row = (size_t)S.row0 + t0 + tr;
    const float gt = expf(lg[tr]);
    const float bta = bv_b;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float v = bv_v[e];
      W[tr * 16 + tc + e] = bta * (v - gt * p[e]);
    }
    fetch_bv(ch + 1);
    __syncthreads();   // every Q · S0 is done: the q tile region takes T and M
    float* Ts = qs;
    float* Ms = qs + CK * (CK + 1);
    {   // T / M (fetched one chunk ahead), then the next chunk's are requested
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int i = (tid + 256 * u) * 4, r = i / CK, c = i % CK;
        float* td = Ts + r * (CK + 1) + c;
        float* md = Ms + r * (CK + 1) + c;
        td[0] = t4[u].x; td[1] = t4[u].y; td[2] = t4[u].z; td[3] = t4[u].w;
        md[0] = m4[u].x; md[1] = m4[u].y; md[2] = m4[u].z; md[3] = m4[u].w;
      }
      fetch_tm(ch + 1);
    }
    __syncthreads();
    // U = T · W (T lower, unit diagonal)
    float u[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i = 0; i <= tr; ++i) {
      const float tv = Ts[tr * (CK + 1) + i];
#pragma unroll
      for (int e = 0; e < 4; ++e) u[e] = fmaf(tv, W[i * 16 + tc + e], u[e]);
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) U[tr * 16 + tc + e] = u[e];
    __syncthreads();
    // o_t = exp(Lg_t) · (Q S0)_t + Σ_{i ≤ t} M_ti u_i
    float oo[4] = {gt * qsv[0], gt * qsv[1], gt * qsv[2], gt * qsv[3]};
    for (int i = 0; i <= tr; ++i) {
      const float mv = Ms[tr * (CK + 1) + i];
#pragma unroll
      for (int e = 0; e < 4; ++e) oo[e] = fmaf(mv, U[i * 16 + tc + e], oo[e]);
    }
    if (tr < nr)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[row * (size_t)VH * 128 + (size_t)vh * 128 + c0 + tc + e] = bfr(oo[e]);
    // S ← exp(Lg_last) · S0 + Σ_i exp(Lg_last − Lg_i) k_i u_iᵀ  (state row d = tid / 2, 8 columns)
    __syncthreads();
    {
      const float gl = lg[CK - 1];
      const int d = tid / 2, cb = (tid % 2) * 8;
      float acc[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = expf(gl) * S0[d * 16 + cb + e];
      for (int i = 0; i < CK; ++i) {
        const float f = fac[i] * ks[i * KS + d];
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = fmaf(f, U[i * 16 + cb + e], acc[e]);
      }
      __syncthreads();
#pragma unroll
      for (int e = 0; e < 8; ++e) S0[d * 16 + cb + e] = acc[e];
    }
  }
  __syncthreads();
  for (int i = tid; i < 128 * 16; i += 256) st[(size_t)(i / 16) * 128 + c0 + i % 16] = S0[i];
}

// ---- chunked GDN in WY form (HIVE_Q27_GDN_WY=1): the scan above keeps K·S0, Q·S0, T·W, M·U and the state update in its sequential chain.
//   With W_t = β_t (v_t − e^{Lg_t} S0ᵀ k_t) and U = T W:  U = Uv − Wk S0,  Uv = T (β ⊙ v),  Wk = T (β e^{Lg} ⊙ K)  — Uv / Wk do not depend on
//   the state, so they are formed per chunk in parallel (prep); the sequential kernel keeps only U = Uv − Wk S0 and S ← e^{Lg_C} S0 + (fac ⊙ K)ᵀ U
//   (half of the serial arithmetic, three barriers per chunk instead of six) and stores each chunk's entry state and U; the outputs
//   o_t = e^{Lg_t} q_t S_c + Σ_{i ≤ t} M_ti U_i are formed for all chunks in parallel afterwards. fp32 throughout (another summation order than
//   the scan: compared against the row loop, not bit for bit). This project's derivation from the recurrence above.
namespace gwy {
constexpr int CK = gc::CK, KS = 132;
}
// kernel 1 (chunks in parallel): as gdn_chunk_prep_kernel (M, Lg) plus Wk [CK][128] and Uv [CK][128] per (chunk, value head)
__global__ void __launch_bounds__(256) gdn_wy_prep_kernel(const GdnSeg* segs, const bf16* y, int C, int KH, int VH, const float* qk, const float* gb,
                                                           float* Wkw, float* Uvw, float* Mw, float* Lw) {
  using namespace gwy;
  const GdnSeg S = segs[0];
  const int ch = blockIdx.x, vh = blockIdx.y, rep = VH / KH, kh = vh / rep, KW = KH * 128, tid = threadIdx.x;
  const int t0 = ch * CK, nr = min(CK, S.n - t0);
  extern __shared__ float smw[];   // ks [CK][129] · qs [CK][129] · A [CK][CK + 1] · Ts [CK][CK + 1] · lg [CK] · bt [CK]
  float (*ks)[129] = reinterpret_cast<float (*)[129]>(smw);
  float (*qs)[129] = reinterpret_cast<float (*)[129]>(smw + CK * 129);
  float (*A)[CK + 1] = reinterpret_cast<float (*)[CK + 1]>(smw + 2 * CK * 129);
  float (*Ts)[CK + 1] = reinterpret_cast<float (*)[CK + 1]>(smw + 2 * CK * 129 + CK * (CK + 1));
  float* lg = smw + 2 * CK * 129 + 2 * CK * (CK + 1);
  float* bt = lg + CK;
  for (int i = tid; i < CK * 128; i += 256) {
    const int r = i / 128, d = i % 128;
    const size_t row = (size_t)S.row0 + t0 + r;
    ks[r][d] = r < nr ? qk[row * 2 * KW + KW + (size_t)kh * 128 + d] : 0.f;
    qs[r][d] = r < nr ? qk[row * 2 * KW + (size_t)kh * 128 + d] : 0.f;
  }
  if (tid < CK) {
    const size_t row = (size_t)S.row0 + t0 + tid;
    lg[tid] = tid < nr ? logf(fmaxf(gb[row * 2 * VH + vh], 1e-30f)) : 0.f;   // floored as in gdn_chunk_prep_kernel
    bt[tid] = tid < nr ? gb[row * 2 * VH + VH + vh] : 0.f;
  }
  __syncthreads();
  if (tid == 0) for (int r = 1; r < CK; ++r) lg[r] += lg[r - 1];
  __syncthreads();
  float* Mq = Mw + ((size_t)ch * VH + vh) * CK * CK;
  {
    const int tb = (tid / 16) * 4, jb = (tid % 16) * 4;
    float kk[4][4] = {}, qq[4][4] = {};
    for (int d = 0; d < 128; ++d) {
      float kt[4], qt[4], kj[4];
#pragma unroll
      for (int x = 0; x < 4; ++x) { kt[x] = ks[tb + x][d]; qt[x] = qs[tb + x][d]; kj[x] = ks[jb + x][d]; }
#pragma unroll
      for (int x = 0; x < 4; ++x)
#pragma unroll
        for (int z = 0; z < 4; ++z) { kk[x][z] = fmaf(kt[x], kj[z], kk[x][z]); qq[x][z] = fmaf(qt[x], kj[z], qq[x][z]); }
    }
#pragma unroll
    for (int x = 0; x < 4; ++x)
#pragma unroll
      for (int z = 0; z < 4; ++z) {
        const int t = tb + x, j = jb + z;
        const float dec = j <= t ? expf(lg[t] - lg[j]) : 0.f;
        A[t][j] = j < t ? bt[t] * dec * kk[x][z] : 0.f;
        Mq[t * CK + j] = j <= t ? dec * qq[x][z] : 0.f;
      }
  }
  if (tid < CK) Lw[((size_t)ch * VH + vh) * CK + tid] = lg[tid];
  __syncthreads();
  {   // T = (I + A)⁻¹ (forward substitution, one column per 4 lanes — as gdn_chunk_prep_kernel)
    const int j = tid >> 2, part = tid & 3;
    for (int t = 0; t < CK; ++t) {
      float sum = 0.f;
      for (int m = j + part; m < t; m += 4) sum = fmaf(A[t][m], Ts[m][j], sum);
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      if (part == 0) Ts[t][j] = (t == j ? 1.f : 0.f) - (t > j ? sum : 0.f);
      __syncwarp();
    }
  }
  // the q tile is no longer needed: it takes β e^{Lg} ⊙ k, then β ⊙ v
  for (int i = tid; i < CK * 128; i += 256) { const int r = i / 128, d = i % 128; qs[r][d] = bt[r] * expf(lg[r]) * ks[r][d]; }
  __syncthreads();
  float* Wk = Wkw + ((size_t)ch * VH + vh) * CK * 128;
  float* Uv = Uvw + ((size_t)ch * VH + vh) * CK * 128;
  // thread: row t = tid / 4, columns (tid % 4) + 4 m (m < 32): Wk[t][d] = Σ_{i ≤ t} T[t][i] X[i][d]
  auto tri_mul = [&](float* dst) {
    const int t = tid / 4, cq = tid % 4;
    float acc[32];
#pragma unroll
    for (int m = 0; m < 32; ++m) acc[m] = 0.f;
    for (int i = 0; i <= t; ++i) {
      const float tv = Ts[t][i];
#pragma unroll
      for (int m = 0; m < 32; ++m) acc[m] = fmaf(tv, qs[i][cq + 4 * m], acc[m]);
    }
#pragma unroll
    for (int m = 0; m < 32; ++m) dst[t * 128 + cq + 4 * m] = acc[m];
  };
  tri_mul(Wk);
  __syncthreads();
  for (int i = tid; i < CK * 128; i += 256) {
    const int r = i / 128, c = i % 128;
    const size_t row = (size_t)S.row0 + t0 + r;
    qs[r][c] = r < nr ? bt[r] * __bfloat162float(y[row * C + 2 * KW + (size_t)vh * 128 + c]) : 0.f;
  }
  __syncthreads();
  tri_mul(Uv);
}
// kernel 2 (chunks in order): per (value head, 16 state columns): U = Uv − Wk S0, S ← e^{Lg_C} S0 + (fac ⊙ K)ᵀ U; stores S_c (entry) and U
__global__ void __launch_bounds__(256) gdn_wy_state_kernel(const GdnSeg* segs, int KH, int VH, const float* qk, const float* Wkw, const float* Uvw,
                                                            const float* Lw, float* Scw, float* Uw) {
  using namespace gwy;
  extern __shared__ float sm[];
  float* S0 = sm;                  // [128][16]
  float* wks = S0 + 128 * 16;      // [CK][KS]
  float* ks = wks + CK * KS;       // [CK][KS]
  float* U = ks + CK * KS;         // [CK][16]
  float* lg = U + CK * 16;         // [CK]
  float* fac = lg + CK;            // [CK]
  const GdnSeg S = segs[0];
  const int vh = blockIdx.x, c0 = blockIdx.y * 16, rep = VH / KH, kh = vh / rep, KW = KH * 128, tid = threadIdx.x;
  float* st = S.state + (size_t)vh * 128 * 128;
  for (int i = tid; i < 128 * 16; i += 256) S0[i] = st[(size_t)(i / 16) * 128 + c0 + i % 16];
  const int nch = (S.n + CK - 1) / CK;
  const int tr = tid / 4, tc = (tid % 4) * 4;
  float4 w4[8], k4[8], uv4;
  auto fetch = [&](int c) {
    if (c >= nch) return;
    const int t0c = c * CK, nrc = min(CK, S.n - t0c);
    const float* Wg = Wkw + ((size_t)c * VH + vh) * CK * 128;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int i = tid + 256 * u, r = i / 32, d4 = i % 32;
      w4[u] = reinterpret_cast<const float4*>(Wg)[i];
      const size_t row = (size_t)S.row0 + t0c + r;
      k4[u] = r < nrc ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    uv4 = *reinterpret_cast<const float4*>(Uvw + ((size_t)c * VH + vh) * CK * 128 + tr * 128 + c0 + tc);
  };
  fetch(0);
  for (int ch = 0; ch < nch; ++ch) {
    __syncthreads();
    float* Sc = Scw + ((size_t)ch * VH + vh) * 128 * 128;   // entry state of this chunk (for the output kernel)
    for (int i = tid; i < 128 * 16; i += 256) Sc[(size_t)(i / 16) * 128 + c0 + i % 16] = S0[i];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int i = tid + 256 * u, r = i / 32, d = (i % 32) * 4;
      *reinterpret_cast<float4*>(wks + r * KS + d) = w4[u];
      *reinterpret_cast<float4*>(ks + r * KS + d) = k4[u];
    }
    const float uvv[4] = {uv4.x, uv4.y, uv4.z, uv4.w};
    if (tid < CK) lg[tid] = Lw[((size_t)ch * VH + vh) * CK + tid];
    fetch(ch + 1);
    __syncthreads();
    if (tid < CK) fac[tid] = expf(lg[CK - 1] - lg[tid]);
    float p[4] = {0.f, 0.f, 0.f, 0.f};
    for (int d = 0; d < 128; d += 4) {
      const float4 a4 = *reinterpret_cast<const float4*>(wks + tr * KS + d);
      const float av[4] = {a4.x, a4.y, a4.z, a4.w};
#pragma unroll
      for (int dd = 0; dd < 4; ++dd) {
        const float4 s4 = *reinterpret_cast<const float4*>(S0 + (d + dd) * 16 + tc);
        const float sv[4] = {s4.x, s4.y, s4.z, s4.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) p[e] = fmaf(av[dd], sv[e], p[e]);
      }
    }
    float* Ug = Uw + ((size_t)ch * VH + vh) * CK * 128;
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float uu = uvv[e] - p[e]; U[tr * 16 + tc + e] = uu; Ug[tr * 128 + c0 + tc + e] = uu; }
    __syncthreads();
    {
      const float gl = lg[CK - 1];
      const int d = tid / 2, cb = (tid % 2) * 8;
      float acc[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = expf(gl) * S0[d * 16 + cb + e];
      for (int i = 0; i < CK; ++i) {
        const float f = fac[i] * ks[i * KS + d];
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = fmaf(f, U[i * 16 + cb + e], acc[e]);
      }
      __syncthreads();
#pragma unroll
      for (int e = 0; e < 8; ++e) S0[d * 16 + cb + e] = acc[e];
    }
  }
  __syncthreads();
  for (int i = tid; i < 128 * 16; i += 256) st[(size_t)(i / 16) * 128 + c0 + i % 16] = S0[i];
}
// kernel 3 (chunks in parallel): o_t = e^{Lg_t} q_t S_c + Σ_{i ≤ t} M_ti U_i for 32 of the 128 value columns (grid z)
__global__ void __launch_bounds__(256) gdn_wy_out_kernel(const GdnSeg* segs, int KH, int VH, const float* qk, const float* Scw, const float* Uw,
                                                          const float* Mw, const float* Lw, float* o) {
  using namespace gwy;
  extern __shared__ float smo[];
  float* qs = smo;                  // [CK][KS]
  float* Ss = qs + CK * KS;         // [128][32]
  float* Ms = Ss + 128 * 32;        // [CK][CK + 1]
  float* Us = Ms + CK * (CK + 1);   // [CK][32]
  float* lg = Us + CK * 32;         // [CK]
  const GdnSeg S = segs[0];
  const int ch = blockIdx.x, vh = blockIdx.y, cz = blockIdx.z * 32, rep = VH / KH, kh = vh / rep, KW = KH * 128, tid = threadIdx.x;
  const int t0 = ch * CK, nr = min(CK, S.n - t0);
  const float* Sc = Scw + ((size_t)ch * VH + vh) * 128 * 128;
  const float* Ug = Uw + ((size_t)ch * VH + vh) * CK * 128;
  const float* Mg = Mw + ((size_t)ch * VH + vh) * CK * CK;
  for (int i = tid; i < CK * 32; i += 256) {
    const int r = i / 32, d4 = i % 32;
    const size_t row = (size_t)S.row0 + t0 + r;
    *reinterpret_cast<float4*>(qs + r * KS + d4 * 4) = r < nr ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  for (int i = tid; i < 128 * 32; i += 256) Ss[i] = Sc[(size_t)(i / 32) * 128 + cz + i % 32];
  for (int i = tid; i < CK * CK; i += 256) Ms[(i / CK) * (CK + 1) + i % CK] = Mg[i];
  for (int i = tid; i < CK * 32; i += 256) Us[i] = Ug[(size_t)(i / 32) * 128 + cz + i % 32];
  if (tid < CK) lg[tid] = Lw[((size_t)ch * VH + vh) * CK + tid];
  __syncthreads();
  const int t = tid / 4, cb = (tid % 4) * 8;
  float qsv[8] = {}, mu[8] = {};
  for (int d = 0; d < 128; ++d) {
    const float qv = qs[t * KS + d];
#pragma unroll
    for (int e = 0; e < 8; ++e) qsv[e] = fmaf(qv, Ss[d * 32 + cb + e], qsv[e]);
  }
  for (int i = 0; i <= t; ++i) {
    const float mv = Ms[t * (CK + 1) + i];
#pragma unroll
    for (int e = 0; e < 8; ++e) mu[e] = fmaf(mv, Us[i * 32 + cb + e], mu[e]);
  }
  if (t < nr) {
    const float gt = expf(lg[t]);
    const size_t row = (size_t)S.row0 + t0 + t;
#pragma unroll
    for (int e = 0; e < 8; ++e) o[row * (size_t)VH * 128 + (size_t)vh * 128 + cz + cb + e] = bfr(gt * qsv[e] + mu[e]);
  }
}
// ---- chunked GDN scan on tensor cores (HIVE_Q27_GDN_TC=1): the per-chunk products of gdn_chunk_scan_kernel (P = K·S0, Q·S0, U = T·W,
//   M·U, S ← e^{Lg_C} S0 + (f∘K)ᵀ U) as mma m16n8k8 in 3 × TF32 (x = hi + lo, hi·hi + hi·lo + lo·hi; ~fp32 accuracy, another summation
//   order than the scalar kernel → oracle + quality gate). The scalar scan ran at ~20 % of fp32 peak (2026-10-11: 32K prefill 0.74 s of
//   ~4.1 s; 1K 21 ms of ~105): dependent 128-deep dot products per thread with shared-memory operands.
namespace gtc {
constexpr int SS = 24;   // row stride of the 16-column tiles (S0, W, U): conflict-free B fragments
__device__ __forceinline__ uint32_t tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x)); return r; }
__device__ __forceinline__ void mma1(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// c += a · b with a, b split into tf32 hi + lo (small products first)
__device__ __forceinline__ void mma3(float (&c)[4], const float (&a)[4], const float (&b)[2]) {
  uint32_t ah[4], al[4], bh[2], bl[2];
#pragma unroll
  for (int i = 0; i < 4; ++i) { ah[i] = tf32(a[i]); al[i] = tf32(a[i] - __uint_as_float(ah[i])); }
#pragma unroll
  for (int i = 0; i < 2; ++i) { bh[i] = tf32(b[i]); bl[i] = tf32(b[i] - __uint_as_float(bh[i])); }
  mma1(c, al, bh);
  mma1(c, ah, bl);
  mma1(c, ah, bh);
}
}  // namespace gtc
__global__ void __launch_bounds__(256) gdn_chunk_scan_tc_kernel(const GdnSeg* segs, const bf16* y, int C, int KH, int VH, const float* qk,
                                                                 const float* gb, const float* Tw, const float* Mw, const float* Lw, float* o) {
  using namespace gc;
  using gtc::SS;
  extern __shared__ float sm[];
  float* S0 = sm;                       // [128][SS]
  float* ks = S0 + 128 * SS;            // [CK][129]
  float* qs = ks + CK * 129;            // [CK][129] — then T and M [CK][CK + 1]
  float* W = qs + CK * (2 * (CK + 1));  // [CK][SS]
  float* U = W + CK * SS;               // [CK][SS]
  float* lg = U + CK * SS;              // [CK]
  float* fac = lg + CK;                 // [CK]
  const GdnSeg S = segs[0];
  const int vh = blockIdx.x, c0 = blockIdx.y * 16, rep = VH / KH, kh = vh / rep, KW = KH * 128, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int mi = warp >> 1, ni = warp & 1;   // the warp's 16 × 8 tile of the 64 × 16 chunk products
  float* st = S.state + (size_t)vh * 128 * 128;
  for (int i = tid; i < 128 * 16; i += 256) S0[(i / 16) * SS + i % 16] = st[(size_t)(i / 16) * 128 + c0 + i % 16];
  const int nch = (S.n + CK - 1) / CK;
  float4 kv4[8], qv4[8], t4[4], m4[4];
  auto fetch_kq = [&](int c) {
    const int t0c = c * CK, nrc = min(CK, S.n - t0c);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int i = tid + 256 * u, r = i / 32, d4 = i % 32;
      const size_t row = (size_t)S.row0 + t0c + r;
      const bool ok = c < nch && r < nrc;
      kv4[u] = ok ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
      qv4[u] = ok ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
  };
  auto fetch_tm = [&](int c) {
    if (c >= nch) return;
    const float* Tg = Tw + ((size_t)c * VH + vh) * CK * CK;
    const float* Mg = Mw + ((size_t)c * VH + vh) * CK * CK;
#pragma unroll
    for (int u = 0; u < 4; ++u) { const int i = tid + 256 * u; t4[u] = reinterpret_cast<const float4*>(Tg)[i]; m4[u] = reinterpret_cast<const float4*>(Mg)[i]; }
  };
  float bv_b[2], bv_v[4];   // β of the tile rows (g, g + 8) and v of the 4 tile elements, one chunk ahead
  auto fetch_bv = [&](int c) {
    const int t0c = c * CK, nrc = min(CK, S.n - t0c);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = 16 * mi + g + 8 * h;
      const size_t row = (size_t)S.row0 + t0c + r;
      const bool ok = c < nch && r < nrc;
      bv_b[h] = ok ? gb[row * 2 * VH + VH + vh] : 0.f;
#pragma unroll
      for (int e = 0; e < 2; ++e) bv_v[2 * h + e] = ok ? __bfloat162float(y[row * C + 2 * KW + (size_t)vh * 128 + c0 + 8 * ni + 2 * t + e]) : 0.f;
    }
  };
  fetch_kq(0);
  fetch_tm(0);
  fetch_bv(0);
  for (int ch = 0; ch < nch; ++ch) {
    const int t0 = ch * CK, nr = min(CK, S.n - t0);
    __syncthreads();
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int i = tid + 256 * u, r = i / 32, d = (i % 32) * 4;
      float* kd = ks + r * 129 + d;
      float* qd = qs + r * 129 + d;
      kd[0] = kv4[u].x; kd[1] = kv4[u].y; kd[2] = kv4[u].z; kd[3] = kv4[u].w;
      qd[0] = qv4[u].x; qd[1] = qv4[u].y; qd[2] = qv4[u].z; qd[3] = qv4[u].w;
    }
    fetch_kq(ch + 1);
    if (tid < CK) lg[tid] = Lw[((size_t)ch * VH + vh) * CK + tid];
    __syncthreads();
    if (tid < CK) fac[tid] = expf(lg[CK - 1] - lg[tid]);
    // P = K · S0, QS = Q · S0 for the warp's tile (rows 16 mi + g / + 8, columns 8 ni + 2t / + 1)
    float p[4] = {0.f, 0.f, 0.f, 0.f}, qsv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
    for (int kk = 0; kk < 16; ++kk) {
      const int r0 = 16 * mi + g, k0 = 8 * kk + t;
      const float ak[4] = {ks[r0 * 129 + k0], ks[(r0 + 8) * 129 + k0], ks[r0 * 129 + k0 + 4], ks[(r0 + 8) * 129 + k0 + 4]};
      const float aq[4] = {qs[r0 * 129 + k0], qs[(r0 + 8) * 129 + k0], qs[r0 * 129 + k0 + 4], qs[(r0 + 8) * 129 + k0 + 4]};
      const float b[2] = {S0[k0 * SS + 8 * ni + g], S0[(k0 + 4) * SS + 8 * ni + g]};
      gtc::mma3(p, ak, b);
      gtc::mma3(qsv, aq, b);
    }
    // W_t = β_t (v_t − exp(Lg_t) P_t)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int r = 16 * mi + g + (e >= 2 ? 8 : 0), col = 8 * ni + 2 * t + (e & 1);
      W[r * SS + col] = bv_b[e >> 1] * (bv_v[e] - expf(lg[r]) * p[e]);
    }
    fetch_bv(ch + 1);
    __syncthreads();   // every Q · S0 is done: the q tile region takes T and M
    float* Ts = qs;
    float* Ms = qs + CK * (CK + 1);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int i = (tid + 256 * u) * 4, r = i / CK, c = i % CK;
      float* td = Ts + r * (CK + 1) + c;
      float* md = Ms + r * (CK + 1) + c;
      td[0] = t4[u].x; td[1] = t4[u].y; td[2] = t4[u].z; td[3] = t4[u].w;
      md[0] = m4[u].x; md[1] = m4[u].y; md[2] = m4[u].z; md[3] = m4[u].w;
    }
    fetch_tm(ch + 1);
    __syncthreads();
    // U = T · W (T lower triangular: k-steps up to the tile's last row)
    {
      float uu[4] = {0.f, 0.f, 0.f, 0.f};
      for (int kk = 0; kk <= 2 * mi + 1; ++kk) {
        const int r0 = 16 * mi + g, k0 = 8 * kk + t;
        const float a[4] = {Ts[r0 * (CK + 1) + k0], Ts[(r0 + 8) * (CK + 1) + k0], Ts[r0 * (CK + 1) + k0 + 4], Ts[(r0 + 8) * (CK + 1) + k0 + 4]};
        const float b[2] = {W[k0 * SS + 8 * ni + g], W[(k0 + 4) * SS + 8 * ni + g]};
        gtc::mma3(uu, a, b);
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) U[(16 * mi + g + (e >= 2 ? 8 : 0)) * SS + 8 * ni + 2 * t + (e & 1)] = uu[e];
    }
    __syncthreads();
    // o_t = exp(Lg_t) · (Q S0)_t + Σ_{i ≤ t} M_ti u_i
    {
      float oo[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) oo[e] = expf(lg[16 * mi + g + (e >= 2 ? 8 : 0)]) * qsv[e];
      for (int kk = 0; kk <= 2 * mi + 1; ++kk) {
        const int r0 = 16 * mi + g, k0 = 8 * kk + t;
        const float a[4] = {Ms[r0 * (CK + 1) + k0], Ms[(r0 + 8) * (CK + 1) + k0], Ms[r0 * (CK + 1) + k0 + 4], Ms[(r0 + 8) * (CK + 1) + k0 + 4]};
        const float b[2] = {U[k0 * SS + 8 * ni + g], U[(k0 + 4) * SS + 8 * ni + g]};
        gtc::mma3(oo, a, b);
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int r = 16 * mi + g + (e >= 2 ? 8 : 0), col = 8 * ni + 2 * t + (e & 1);
        if (r < nr) o[((size_t)S.row0 + t0 + r) * (size_t)VH * 128 + (size_t)vh * 128 + c0 + col] = bfr(oo[e]);
      }
    }
    // S ← exp(Lg_last) · S0 + Σ_i (fac_i k_i) u_iᵀ: warp w owns state rows 16 w … + 15, both 8-column halves
    {
      const float eg = expf(lg[CK - 1]);
#pragma unroll
      for (int nh = 0; nh < 2; ++nh) {
        float acc[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[e] = eg * S0[(16 * warp + g + (e >= 2 ? 8 : 0)) * SS + 8 * nh + 2 * t + (e & 1)];
#pragma unroll 2
        for (int kk = 0; kk < 8; ++kk) {
          const int k0 = 8 * kk + t, d0 = 16 * warp + g;
          const float a[4] = {fac[k0] * ks[k0 * 129 + d0], fac[k0] * ks[k0 * 129 + d0 + 8], fac[k0 + 4] * ks[(k0 + 4) * 129 + d0],
                              fac[k0 + 4] * ks[(k0 + 4) * 129 + d0 + 8]};
          const float b[2] = {U[k0 * SS + 8 * nh + g], U[(k0 + 4) * SS + 8 * nh + g]};
          gtc::mma3(acc, a, b);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) S0[(16 * warp + g + (e >= 2 ? 8 : 0)) * SS + 8 * nh + 2 * t + (e & 1)] = acc[e];
      }
    }
  }
  __syncthreads();
  for (int i = tid; i < 128 * 16; i += 256) st[(size_t)(i / 16) * 128 + c0 + i % 16] = S0[(i / 16) * SS + i % 16];
}

// ---- WY form on tensor cores (HIVE_Q27_GDN_WY=2): the state passing and the outputs of gdn_wy_* as 3 × TF32 mma (hi / lo split: fp32-level
//   products, gtc::mma3). Same algebra and data as the SIMT kernels above.
namespace gwt {
constexpr int KS = 136, SS = 24, QS = 132, CS = 40, MS = 68;
// precision of the WY products: 3 × TF32 (fp32-level, default for HIVE_Q27_GDN_WY=2) or 1 × TF32 (HIVE_Q27_GDN_WY=3)
__device__ __forceinline__ void mm(bool one, float (&c)[4], const float (&a)[4], const float (&b)[2]) {
  if (one) {
    const uint32_t ua[4] = {gtc::tf32(a[0]), gtc::tf32(a[1]), gtc::tf32(a[2]), gtc::tf32(a[3])}, ub[2] = {gtc::tf32(b[0]), gtc::tf32(b[1])};
    gtc::mma1(c, ua, ub);
  } else gtc::mma3(c, a, b);
}
}
__global__ void __launch_bounds__(256) gdn_wy_state_tc_kernel(const GdnSeg* segs, int KH, int VH, const float* qk, const float* Wkw, const float* Uvw,
                                                               const float* Lw, float* Scw, float* Uw, bool one) {
  using namespace gwy;
  using gwt::SS;
  constexpr int KS2 = gwt::KS;
  extern __shared__ float sm[];
  float* S0 = sm;                  // [128][SS]
  float* wks = S0 + 128 * SS;      // [CK][KS2]
  float* ks = wks + CK * KS2;      // [CK][KS2]  fac-scaled k
  float* U = ks + CK * KS2;        // [CK][SS]
  float* lg = U + CK * SS;         // [CK]
  const GdnSeg S = segs[0];
  const int vh = blockIdx.x, c0 = blockIdx.y * 16, rep = VH / KH, kh = vh / rep, KW = KH * 128, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  float* st = S.state + (size_t)vh * 128 * 128;
  for (int i = tid; i < 128 * 16; i += 256) S0[(i / 16) * SS + i % 16] = st[(size_t)(i / 16) * 128 + c0 + i % 16];
  const int nch = (S.n + CK - 1) / CK;
  float4 w4[8], k4[8];
  const int um = (warp >> 1) * 16, un = (warp & 1) * 8, sd = warp * 16;
  float uvn[4];   // this lane's Uv fragment of the next chunk
  auto fetch = [&](int c) {
    if (c >= nch) return;
    const int t0c = c * CK, nrc = min(CK, S.n - t0c);
    const float* Wg = Wkw + ((size_t)c * VH + vh) * CK * 128;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int i = tid + 256 * u, r = i / 32, d4 = i % 32;
      w4[u] = reinterpret_cast<const float4*>(Wg)[i];
      const size_t row = (size_t)S.row0 + t0c + r;
      k4[u] = r < nrc ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    const float* Uvg = Uvw + ((size_t)c * VH + vh) * CK * 128;
    const int r0 = um + g, cc = un + 2 * t;
    uvn[0] = Uvg[r0 * 128 + c0 + cc]; uvn[1] = Uvg[r0 * 128 + c0 + cc + 1]; uvn[2] = Uvg[(r0 + 8) * 128 + c0 + cc]; uvn[3] = Uvg[(r0 + 8) * 128 + c0 + cc + 1];
  };
  fetch(0);
  // U tile of this warp: rows um .. + 15, columns un .. + 7 · S tiles: rows sd .. + 15 (d), both column halves
  for (int ch = 0; ch < nch; ++ch) {
    __syncthreads();
    float* Sc = Scw + ((size_t)ch * VH + vh) * 128 * 128;
    for (int i = tid; i < 128 * 16; i += 256) Sc[(size_t)(i / 16) * 128 + c0 + i % 16] = S0[(i / 16) * SS + i % 16];
    if (tid < CK) lg[tid] = Lw[((size_t)ch * VH + vh) * CK + tid];
    __syncthreads();
    const float gl = lg[CK - 1];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int i = tid + 256 * u, r = i / 32, d = (i % 32) * 4;
      *reinterpret_cast<float4*>(wks + r * KS2 + d) = w4[u];
      const float f = expf(gl - lg[r]);
      *reinterpret_cast<float4*>(ks + r * KS2 + d) = make_float4(f * k4[u].x, f * k4[u].y, f * k4[u].z, f * k4[u].w);
    }
    // U = Uv − Wk · S0 (this warp's 16 × 8 tile; Uv straight from global)
    const float uv[4] = {uvn[0], uvn[1], uvn[2], uvn[3]};
    float uc[4] = {0.f, 0.f, 0.f, 0.f};
    fetch(ch + 1);
    __syncthreads();
#pragma unroll 4
    for (int k0 = 0; k0 < 128; k0 += 8) {
      const float a[4] = {wks[(um + g) * KS2 + k0 + t], wks[(um + g + 8) * KS2 + k0 + t], wks[(um + g) * KS2 + k0 + t + 4], wks[(um + g + 8) * KS2 + k0 + t + 4]};
      const float b[2] = {S0[(k0 + t) * SS + un + g], S0[(k0 + t + 4) * SS + un + g]};
      gwt::mm(one, uc, a, b);
    }
    {
      float* Ug = Uw + ((size_t)ch * VH + vh) * CK * 128;
      const int r0 = um + g, cc = un + 2 * t;
      const float v0 = uv[0] - uc[0], v1 = uv[1] - uc[1], v2 = uv[2] - uc[2], v3 = uv[3] - uc[3];
      U[r0 * SS + cc] = v0; U[r0 * SS + cc + 1] = v1; U[(r0 + 8) * SS + cc] = v2; U[(r0 + 8) * SS + cc + 1] = v3;
      Ug[r0 * 128 + c0 + cc] = v0; Ug[r0 * 128 + c0 + cc + 1] = v1; Ug[(r0 + 8) * 128 + c0 + cc] = v2; Ug[(r0 + 8) * 128 + c0 + cc + 1] = v3;
    }
    __syncthreads();
    // S ← e^{Lg_C} S0 + (fac ⊙ K)ᵀ U : rows d = sd .. sd + 15, columns 0..7 and 8..15
    const float eg = expf(gl);
    float sc2[2][4];
#pragma unroll
    for (int nh = 0; nh < 2; ++nh) {
      const int cc = nh * 8 + 2 * t;
      sc2[nh][0] = eg * S0[(sd + g) * SS + cc]; sc2[nh][1] = eg * S0[(sd + g) * SS + cc + 1];
      sc2[nh][2] = eg * S0[(sd + g + 8) * SS + cc]; sc2[nh][3] = eg * S0[(sd + g + 8) * SS + cc + 1];
    }
#pragma unroll 2
    for (int k0 = 0; k0 < CK; k0 += 8) {
      const float a[4] = {ks[(k0 + t) * KS2 + sd + g], ks[(k0 + t) * KS2 + sd + g + 8], ks[(k0 + t + 4) * KS2 + sd + g], ks[(k0 + t + 4) * KS2 + sd + g + 8]};
#pragma unroll
      for (int nh = 0; nh < 2; ++nh) {
        const float b[2] = {U[(k0 + t) * SS + nh * 8 + g], U[(k0 + t + 4) * SS + nh * 8 + g]};
        gwt::mm(one, sc2[nh], a, b);
      }
    }
    __syncthreads();   // every warp has read S0
#pragma unroll
    for (int nh = 0; nh < 2; ++nh) {
      const int cc = nh * 8 + 2 * t;
      S0[(sd + g) * SS + cc] = sc2[nh][0]; S0[(sd + g) * SS + cc + 1] = sc2[nh][1];
      S0[(sd + g + 8) * SS + cc] = sc2[nh][2]; S0[(sd + g + 8) * SS + cc + 1] = sc2[nh][3];
    }
  }
  __syncthreads();
  for (int i = tid; i < 128 * 16; i += 256) st[(size_t)(i / 16) * 128 + c0 + i % 16] = S0[(i / 16) * SS + i % 16];
}
// prep on tensor cores: K·Kᵀ and Q·Kᵀ, then T·(β e^{Lg} ⊙ K) and T·(β ⊙ v) as 3 × TF32 mma; T = (I + A)⁻¹ by the same forward substitution
__global__ void __launch_bounds__(256) gdn_wy_prep_tc_kernel(const GdnSeg* segs, const bf16* y, int C, int KH, int VH, const float* qk, const float* gb,
                                                              float* Wkw, float* Uvw, float* Mw, float* Lw, bool one) {
  using namespace gwy;
  constexpr int RS = 132, TS = 68;
  const GdnSeg S = segs[0];
  const int ch = blockIdx.x, vh = blockIdx.y, rep = VH / KH, kh = vh / rep, KW = KH * 128, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int t0 = ch * CK, nr = min(CK, S.n - t0);
  extern __shared__ float smw[];   // ks [CK][RS] (then A [CK][CK + 1]) · xs [CK][RS] (q, then β e^{Lg} k, then β v) · Ts [CK][TS] · lg · bt
  float* ks = smw;
  float* xs = ks + CK * RS;
  float* Ts = xs + CK * RS;
  float* lg = Ts + CK * TS;
  float* bt = lg + CK;
  for (int i = tid; i < CK * 32; i += 256) {
    const int r = i / 32, d4 = i % 32;
    const size_t row = (size_t)S.row0 + t0 + r;
    const bool ok = r < nr;
    *reinterpret_cast<float4*>(ks + r * RS + d4 * 4) = ok ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    *reinterpret_cast<float4*>(xs + r * RS + d4 * 4) = ok ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  if (tid < CK) {
    const size_t row = (size_t)S.row0 + t0 + tid;
    lg[tid] = tid < nr ? logf(fmaxf(gb[row * 2 * VH + vh], 1e-30f)) : 0.f;
    bt[tid] = tid < nr ? gb[row * 2 * VH + VH + vh] : 0.f;
  }
  __syncthreads();
  if (tid == 0) for (int r = 1; r < CK; ++r) lg[r] += lg[r - 1];
  // warp: row tile (warp >> 1) · 16, column tiles 32 (warp & 1) .. + 31 (four n8) of K Kᵀ and of Q Kᵀ
  const int mr = (warp >> 1) * 16, nc = (warp & 1) * 32;
  float kk[4][4] = {}, qq[4][4] = {};
#pragma unroll 2
  for (int k0 = 0; k0 < 128; k0 += 8) {
    const float ak[4] = {ks[(mr + g) * RS + k0 + t], ks[(mr + g + 8) * RS + k0 + t], ks[(mr + g) * RS + k0 + t + 4], ks[(mr + g + 8) * RS + k0 + t + 4]};
    const float aq[4] = {xs[(mr + g) * RS + k0 + t], xs[(mr + g + 8) * RS + k0 + t], xs[(mr + g) * RS + k0 + t + 4], xs[(mr + g + 8) * RS + k0 + t + 4]};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float b[2] = {ks[(nc + 8 * j + g) * RS + k0 + t], ks[(nc + 8 * j + g) * RS + k0 + t + 4]};
      gwt::mm(one, kk[j], ak, b);
      gwt::mm(one, qq[j], aq, b);
    }
  }
  __syncthreads();   // lg prefix sum visible; every warp is done with ks / xs
  float* Mq = Mw + ((size_t)ch * VH + vh) * CK * CK;
  float (*A)[CK + 1] = reinterpret_cast<float (*)[CK + 1]>(ks);
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int tt = mr + g + (e >= 2 ? 8 : 0), jj = nc + 8 * j + 2 * t + (e & 1);
      const float dec = jj <= tt ? expf(lg[tt] - lg[jj]) : 0.f;
      A[tt][jj] = jj < tt ? bt[tt] * dec * kk[j][e] : 0.f;
      Mq[tt * CK + jj] = jj <= tt ? dec * qq[j][e] : 0.f;
    }
  if (tid < CK) Lw[((size_t)ch * VH + vh) * CK + tid] = lg[tid];
  __syncthreads();
  {   // T = (I + A)⁻¹ (forward substitution, as gdn_chunk_prep_kernel)
    const int j = tid >> 2, part = tid & 3;
    for (int tt = 0; tt < CK; ++tt) {
      float sum = 0.f;
      for (int m = j + part; m < tt; m += 4) sum = fmaf(A[tt][m], Ts[m * TS + j], sum);
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      if (part == 0) Ts[tt * TS + j] = (tt == j ? 1.f : 0.f) - (tt > j ? sum : 0.f);
      __syncwarp();
    }
  }
  __syncthreads();
  // X = β e^{Lg} ⊙ k (the k tile is gone: A took its place — reload from global / L2)
  for (int i = tid; i < CK * 32; i += 256) {
    const int r = i / 32, d4 = i % 32;
    const size_t row = (size_t)S.row0 + t0 + r;
    const float f = bt[r] * expf(lg[r]);
    const float4 kv = r < nr ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    *reinterpret_cast<float4*>(xs + r * RS + d4 * 4) = make_float4(f * kv.x, f * kv.y, f * kv.z, f * kv.w);
  }
  __syncthreads();
  // dst [CK][128] = T · X: warp rows (warp >> 1) · 16, columns 64 (warp & 1) .. + 63 (eight n8)
  auto tmul = [&](float* dst) {
    const int cb = (warp & 1) * 64;
    float acc[8][4] = {};
#pragma unroll 2
    for (int k0 = 0; k0 < CK; k0 += 8) {
      if (k0 > mr + 15) break;   // T is lower triangular
      const float a[4] = {Ts[(mr + g) * TS + k0 + t], Ts[(mr + g + 8) * TS + k0 + t], Ts[(mr + g) * TS + k0 + t + 4], Ts[(mr + g + 8) * TS + k0 + t + 4]};
#pragma unroll
      for (int j = 0; j < 8; ++j) { const float b[2] = {xs[(k0 + t) * RS + cb + 8 * j + g], xs[(k0 + t + 4) * RS + cb + 8 * j + g]}; gwt::mm(one, acc[j], a, b); }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int r = mr + g, c = cb + 8 * j + 2 * t;
      dst[r * 128 + c] = acc[j][0]; dst[r * 128 + c + 1] = acc[j][1]; dst[(r + 8) * 128 + c] = acc[j][2]; dst[(r + 8) * 128 + c + 1] = acc[j][3];
    }
  };
  tmul(Wkw + ((size_t)ch * VH + vh) * CK * 128);
  __syncthreads();
  for (int i = tid; i < CK * 128; i += 256) {
    const int r = i / 128, c = i % 128;
    const size_t row = (size_t)S.row0 + t0 + r;
    xs[r * RS + c] = r < nr ? bt[r] * __bfloat162float(y[row * C + 2 * KW + (size_t)vh * 128 + c]) : 0.f;
  }
  __syncthreads();
  tmul(Uvw + ((size_t)ch * VH + vh) * CK * 128);
}
// outputs on tensor cores: per (chunk, value head, 32 columns) O = diag(e^Lg) Q S_c + M U — each warp one 16-row × 8-column tile pair
__global__ void __launch_bounds__(256) gdn_wy_out_tc_kernel(const GdnSeg* segs, int KH, int VH, const float* qk, const float* Scw, const float* Uw,
                                                             const float* Mw, const float* Lw, float* o, bool one) {
  using namespace gwy;
  using gwt::QS; using gwt::CS; using gwt::MS;
  extern __shared__ float smo[];
  float* qs = smo;                  // [CK][QS]
  float* Ss = qs + CK * QS;         // [128][CS]
  float* Ms = Ss + 128 * CS;        // [CK][MS]
  float* Us = Ms + CK * MS;         // [CK][CS]
  float* lg = Us + CK * CS;         // [CK]
  const GdnSeg S = segs[0];
  const int ch = blockIdx.x, vh = blockIdx.y, cz = blockIdx.z * 32, rep = VH / KH, kh = vh / rep, KW = KH * 128, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int t0 = ch * CK, nr = min(CK, S.n - t0);
  const float* Sc = Scw + ((size_t)ch * VH + vh) * 128 * 128;
  const float* Ug = Uw + ((size_t)ch * VH + vh) * CK * 128;
  const float* Mg = Mw + ((size_t)ch * VH + vh) * CK * CK;
  for (int i = tid; i < CK * 32; i += 256) {
    const int r = i / 32, d4 = i % 32;
    const size_t row = (size_t)S.row0 + t0 + r;
    *reinterpret_cast<float4*>(qs + r * QS + d4 * 4) = r < nr ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  for (int i = tid; i < 128 * 8; i += 256) { const int r = i / 8, c4 = i % 8; *reinterpret_cast<float4*>(Ss + r * CS + c4 * 4) = *reinterpret_cast<const float4*>(Sc + (size_t)r * 128 + cz + c4 * 4); }
  for (int i = tid; i < CK * 16; i += 256) { const int r = i / 16, c4 = i % 16; *reinterpret_cast<float4*>(Ms + r * MS + c4 * 4) = *reinterpret_cast<const float4*>(Mg + (size_t)r * CK + c4 * 4); }
  for (int i = tid; i < CK * 8; i += 256) { const int r = i / 8, c4 = i % 8; *reinterpret_cast<float4*>(Us + r * CS + c4 * 4) = *reinterpret_cast<const float4*>(Ug + (size_t)r * 128 + cz + c4 * 4); }
  if (tid < CK) lg[tid] = Lw[((size_t)ch * VH + vh) * CK + tid];
  __syncthreads();
  const int mr = (warp >> 1) * 16, nb = (warp & 1) * 16;   // rows mr..mr+15, columns nb..nb+15 (two n8 tiles)
  float qa[2][4] = {}, ma[2][4] = {};
#pragma unroll 4
  for (int k0 = 0; k0 < 128; k0 += 8) {
    const float a[4] = {qs[(mr + g) * QS + k0 + t], qs[(mr + g + 8) * QS + k0 + t], qs[(mr + g) * QS + k0 + t + 4], qs[(mr + g + 8) * QS + k0 + t + 4]};
#pragma unroll
    for (int j = 0; j < 2; ++j) { const float b[2] = {Ss[(k0 + t) * CS + nb + 8 * j + g], Ss[(k0 + t + 4) * CS + nb + 8 * j + g]}; gwt::mm(one, qa[j], a, b); }
  }
#pragma unroll 2
  for (int k0 = 0; k0 < CK; k0 += 8) {
    if (k0 > mr + 15) break;   // M is lower triangular
    const float a[4] = {Ms[(mr + g) * MS + k0 + t], Ms[(mr + g + 8) * MS + k0 + t], Ms[(mr + g) * MS + k0 + t + 4], Ms[(mr + g + 8) * MS + k0 + t + 4]};
#pragma unroll
    for (int j = 0; j < 2; ++j) { const float b[2] = {Us[(k0 + t) * CS + nb + 8 * j + g], Us[(k0 + t + 4) * CS + nb + 8 * j + g]}; gwt::mm(one, ma[j], a, b); }
  }
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int r = mr + g + (e >= 2 ? 8 : 0), c = nb + 8 * j + 2 * t + (e & 1);
      if (r < nr) o[((size_t)S.row0 + t0 + r) * (size_t)VH * 128 + (size_t)vh * 128 + cz + c] = bfr(expf(lg[r]) * qa[j][e] + ma[j][e]);
    }
}
// ---- WY form v2 (HIVE_Q27_GDN_WY=4): bf16 operands / storage, fp32 accumulation, the serial state in fp32 registers ----------------------
//   stored per chunk: Wk, Uv [CK][128] (prep), S_cᵀ [128 c][128 d] (entry state, transposed), Uᵀ [128 c][CK] — all bf16. mma m16n8k16 bf16:
//   A row-major (k contiguous), B as [n][k] (k contiguous) so every fragment is one 32-bit shared-memory read; row strides padded by 8 bf16.
namespace gv2 {
constexpr int CK = 64;
__device__ __forceinline__ uint32_t ld32(const bf16* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  const uint32_t a[4] = {a0, a1, a2, a3};
  tc::mma(c, a, b0, b1);
}
// A fragment of a 16 × 16 tile at (r0, k0) of a row-major bf16 matrix with row stride ld (elements)
__device__ __forceinline__ void afrag(const bf16* A, int ld, int r0, int k0, int g, int t, uint32_t (&a)[4]) {
  a[0] = ld32(A + (r0 + g) * ld + k0 + 2 * t); a[1] = ld32(A + (r0 + g + 8) * ld + k0 + 2 * t);
  a[2] = ld32(A + (r0 + g) * ld + k0 + 2 * t + 8); a[3] = ld32(A + (r0 + g + 8) * ld + k0 + 2 * t + 8);
}
// B fragment of a 16 × 8 tile at (k0, n0) of a matrix stored as [n][k] with row stride ld
__device__ __forceinline__ void bfrag(const bf16* Bt, int ld, int k0, int n0, int g, int t, uint32_t& b0, uint32_t& b1) {
  b0 = ld32(Bt + (n0 + g) * ld + k0 + 2 * t); b1 = ld32(Bt + (n0 + g) * ld + k0 + 2 * t + 8);
}
}  // namespace gv2
namespace { DevBufLite g_gdn_ws, g_gdn_wsv; }   // chunked GDN prefill workspaces (gdn_reserve / gdn)
// state passing: grid (value head, 4 groups of 32 columns), 8 warps. Per chunk: U = Uv − Wk S (warp: rows 16 (w / 2), columns 16 (w % 2) .. + 15),
//   S ← e^{Lg_C} S + (fac ⊙ K)ᵀ U (warp: rows d = 16 w .. + 15, all 32 columns — the accumulators ARE the state, fp32)
__global__ void __launch_bounds__(256) gdn_v2_state_kernel(const GdnSeg* segs, int KH, int VH, const float* qk, const bf16* Wkw, const bf16* Uvw,
                                                            const float* Lw, bf16* Sctw, bf16* Utw) {
  using namespace gv2;
  constexpr int LW = 128 + 8, LS = 128 + 8, LK = CK + 8, LU = CK + 8;
  extern __shared__ __align__(16) unsigned char smst[];
  bf16* Wks = reinterpret_cast<bf16*>(smst);   // Wk [i][d]
  bf16* Kts = Wks + CK * LW;                    // (fac ⊙ K)ᵀ [d][i]
  bf16* Sts = Kts + 128 * LK;                   // Sᵀ [c][d] (bf16 copy for U = Wk S)
  bf16* Uts = Sts + 32 * LS;                    // Uᵀ [c][i]
  float* lg = reinterpret_cast<float*>(Uts + 32 * LU);
  const GdnSeg S = segs[0];
  const int vh = blockIdx.x, c0 = blockIdx.y * 32, rep = VH / KH, kh = vh / rep, KW = KH * 128, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  float* st = S.state + (size_t)vh * 128 * 128;
  const int sd = warp * 16;
  float sacc[4][4];   // S rows sd + g / + 8, columns 8 j + 2 t / + 1 (j = n-tile)
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int cc = c0 + 8 * j + 2 * t;
    sacc[j][0] = st[(size_t)(sd + g) * 128 + cc]; sacc[j][1] = st[(size_t)(sd + g) * 128 + cc + 1];
    sacc[j][2] = st[(size_t)(sd + g + 8) * 128 + cc]; sacc[j][3] = st[(size_t)(sd + g + 8) * 128 + cc + 1];
  }
  auto put_st = [&]() {   // bf16 Sᵀ from the accumulators
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int cc = 8 * j + 2 * t;
      const int rr[4] = {cc * LS + sd + g, (cc + 1) * LS + sd + g, cc * LS + sd + g + 8, (cc + 1) * LS + sd + g + 8};
#pragma unroll
      for (int e = 0; e < 4; ++e) Sts[rr[e]] = __float2bfloat16_rn(sacc[j][e]);
    }
  };
  put_st();
  const int nch = (S.n + CK - 1) / CK;
  uint4 wk4[4];   // the next chunk's Wk (64 × 128 bf16 = 1024 uint4, 4 per thread)
  float4 k4[8];   // the next chunk's k rows (fp32)
  float uvn[2][4];
  const int um = (warp >> 1) * 16, unb = (warp & 1) * 16;
  auto fetch = [&](int c) {
    if (c >= nch) return;
    const int t0c = c * CK, nrc = min(CK, S.n - t0c);
    const uint4* Wg = reinterpret_cast<const uint4*>(Wkw + ((size_t)c * VH + vh) * CK * 128);
#pragma unroll
    for (int u = 0; u < 4; ++u) wk4[u] = Wg[tid + 256 * u];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int i = tid + 256 * u, r = i / 32, d4 = i % 32;
      const size_t row = (size_t)S.row0 + t0c + r;
      k4[u] = r < nrc ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    const bf16* Uvg = Uvw + ((size_t)c * VH + vh) * CK * 128;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int cc = c0 + unb + 8 * j + 2 * t;
      uvn[j][0] = __bfloat162float(Uvg[(um + g) * 128 + cc]); uvn[j][1] = __bfloat162float(Uvg[(um + g) * 128 + cc + 1]);
      uvn[j][2] = __bfloat162float(Uvg[(um + g + 8) * 128 + cc]); uvn[j][3] = __bfloat162float(Uvg[(um + g + 8) * 128 + cc + 1]);
    }
  };
  fetch(0);
  for (int ch = 0; ch < nch; ++ch) {
    // entry state of this chunk → global (Sᵀ rows c0 .. c0 + 31), from the bf16 copy
    __syncthreads();
    {
      bf16* Sg = Sctw + ((size_t)ch * VH + vh) * 128 * 128;
      for (int i = tid; i < 32 * 16; i += 256) { const int cr = i / 16, q = i % 16; *reinterpret_cast<uint4*>(Sg + (size_t)(c0 + cr) * 128 + q * 8) = *reinterpret_cast<const uint4*>(Sts + cr * LS + q * 8); }
    }
    if (tid < CK) lg[tid] = Lw[((size_t)ch * VH + vh) * CK + tid];
#pragma unroll
    for (int u = 0; u < 4; ++u) { const int i = tid + 256 * u, r = i / 16, q = i % 16; *reinterpret_cast<uint4*>(Wks + r * LW + q * 8) = wk4[u]; }
    __syncthreads();
    const float gl = lg[CK - 1];
#pragma unroll
    for (int u = 0; u < 8; ++u) {   // (fac ⊙ k)ᵀ
      const int i = tid + 256 * u, r = i / 32, d = (i % 32) * 4;
      const float f = expf(gl - lg[r]);
      Kts[(d + 0) * LK + r] = __float2bfloat16_rn(f * k4[u].x); Kts[(d + 1) * LK + r] = __float2bfloat16_rn(f * k4[u].y);
      Kts[(d + 2) * LK + r] = __float2bfloat16_rn(f * k4[u].z); Kts[(d + 3) * LK + r] = __float2bfloat16_rn(f * k4[u].w);
    }
    float uv[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) uv[j][e] = uvn[j][e];
    fetch(ch + 1);
    // U = Uv − Wk S
    float uc[2][4] = {};
#pragma unroll
    for (int k0 = 0; k0 < 128; k0 += 16) {
      uint32_t a[4]; afrag(Wks, LW, um, k0, g, t, a);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        uint32_t b0, b1; bfrag(Sts, LS, k0, unb + 8 * j, g, t, b0, b1); mma(uc[j], a[0], a[1], a[2], a[3], b0, b1);
      }
    }
    __syncthreads();   // Kts complete; every warp done with Sts (its update follows)
    {
      bf16* Ug = Utw + ((size_t)ch * VH + vh) * 128 * CK;
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int r = um + g + (e >= 2 ? 8 : 0), cl = unb + 8 * j + 2 * t + (e & 1);
          const float uf = uv[j][e] - uc[j][e];
          const bf16 v = __float2bfloat16_rn(uf);
          Uts[cl * LU + r] = v;
          Ug[(size_t)(c0 + cl) * CK + r] = v;
        }
    }
    __syncthreads();
    // S ← e^{Lg_C} S + (fac ⊙ K)ᵀ U
    const float eg = expf(gl);
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) sacc[j][e] *= eg;
#pragma unroll
    for (int k0 = 0; k0 < CK; k0 += 16) {
      uint32_t a[4]; afrag(Kts, LK, sd, k0, g, t, a);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t b0, b1; bfrag(Uts, LU, k0, 8 * j, g, t, b0, b1); mma(sacc[j], a[0], a[1], a[2], a[3], b0, b1);
      }
    }
    put_st();
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int cc = c0 + 8 * j + 2 * t;
    st[(size_t)(sd + g) * 128 + cc] = sacc[j][0]; st[(size_t)(sd + g) * 128 + cc + 1] = sacc[j][1];
    st[(size_t)(sd + g + 8) * 128 + cc] = sacc[j][2]; st[(size_t)(sd + g + 8) * 128 + cc + 1] = sacc[j][3];
  }
}
// outputs: one block per (chunk, value head), 8 warps; warp: rows 16 (w / 2) .. + 15, columns 64 (w % 2) .. + 63
__global__ void __launch_bounds__(256) gdn_v2_out_kernel(const GdnSeg* segs, int KH, int VH, const float* qk, const bf16* Sctw, const bf16* Utw,
                                                          const bf16* Mw, const float* Lw, float* o) {
  using namespace gv2;
  constexpr int LQ = 128 + 8, LS = 128 + 8, LM = CK + 8, LU = CK + 8;
  extern __shared__ __align__(16) unsigned char smv2[];
  bf16* Qs = reinterpret_cast<bf16*>(smv2);   // [CK][LQ]
  bf16* Ss = Qs + CK * LQ;                     // S_cᵀ [128][LS]
  bf16* Ms = Ss + 128 * LS;                    // [CK][LM]
  bf16* Us = Ms + CK * LM;                     // Uᵀ [128][LU]
  float* lg = reinterpret_cast<float*>(Us + 128 * LU);
  const GdnSeg S = segs[0];
  const int ch = blockIdx.x, vh = blockIdx.y, rep = VH / KH, kh = vh / rep, KW = KH * 128, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int t0 = ch * CK, nr = min(CK, S.n - t0);
  for (int i = tid; i < CK * 32; i += 256) {
    const int r = i / 32, d4 = i % 32;
    const size_t row = (size_t)S.row0 + t0 + r;
    const float4 q4 = r < nr ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    *reinterpret_cast<uint32_t*>(Qs + r * LQ + d4 * 4) = tc::pack_bf2(q4.x, q4.y);
    *reinterpret_cast<uint32_t*>(Qs + r * LQ + d4 * 4 + 2) = tc::pack_bf2(q4.z, q4.w);
  }
  const bf16* Sg = Sctw + ((size_t)ch * VH + vh) * 128 * 128;
  for (int i = tid; i < 128 * 16; i += 256) { const int r = i / 16, q = i % 16; *reinterpret_cast<uint4*>(Ss + r * LS + q * 8) = *reinterpret_cast<const uint4*>(Sg + (size_t)r * 128 + q * 8); }
  const bf16* Mg = Mw + ((size_t)ch * VH + vh) * CK * CK;
  for (int i = tid; i < CK * 8; i += 256) { const int r = i / 8, q = i % 8; *reinterpret_cast<uint4*>(Ms + r * LM + q * 8) = *reinterpret_cast<const uint4*>(Mg + (size_t)r * CK + q * 8); }
  const bf16* Ug = Utw + ((size_t)ch * VH + vh) * 128 * CK;
  for (int i = tid; i < 128 * 8; i += 256) { const int r = i / 8, q = i % 8; *reinterpret_cast<uint4*>(Us + r * LU + q * 8) = *reinterpret_cast<const uint4*>(Ug + (size_t)r * CK + q * 8); }
  if (tid < CK) lg[tid] = Lw[((size_t)ch * VH + vh) * CK + tid];
  __syncthreads();
  const int mr = (warp >> 1) * 16, nb = (warp & 1) * 64;
  float qa[8][4] = {}, ma[8][4] = {};
#pragma unroll 2
  for (int k0 = 0; k0 < 128; k0 += 16) {
    uint32_t a[4]; afrag(Qs, LQ, mr, k0, g, t, a);
#pragma unroll
    for (int j = 0; j < 8; ++j) { uint32_t b0, b1; bfrag(Ss, LS, k0, nb + 8 * j, g, t, b0, b1); mma(qa[j], a[0], a[1], a[2], a[3], b0, b1); }
  }
  for (int k0 = 0; k0 <= mr; k0 += 16) {   // M lower triangular: k blocks up to the row block
    uint32_t a[4]; afrag(Ms, LM, mr, k0, g, t, a);
#pragma unroll
    for (int j = 0; j < 8; ++j) { uint32_t b0, b1; bfrag(Us, LU, k0, nb + 8 * j, g, t, b0, b1); mma(ma[j], a[0], a[1], a[2], a[3], b0, b1); }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int r = mr + g + (e >= 2 ? 8 : 0), c = nb + 8 * j + 2 * t + (e & 1);
      if (r < nr) o[((size_t)S.row0 + t0 + r) * (size_t)VH * 128 + (size_t)vh * 128 + c] = bfr(expf(lg[r]) * qa[j][e] + ma[j][e]);
    }
}
// prep v2: as gdn_wy_prep_tc_kernel (K·Kᵀ, Q·Kᵀ, T solve and T·X in 3 × TF32 / fp32), Wk / Uv / M stored as bf16
__global__ void __launch_bounds__(256) gdn_v2_prep_kernel(const GdnSeg* segs, const bf16* y, int C, int KH, int VH, const float* qk, const float* gb,
                                                              bf16* Wkw, bf16* Uvw, bf16* Mw, float* Lw, bool one) {
  using namespace gwy;
  constexpr int RS = 132, TS = 68;
  const GdnSeg S = segs[0];
  const int ch = blockIdx.x, vh = blockIdx.y, rep = VH / KH, kh = vh / rep, KW = KH * 128, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int t0 = ch * CK, nr = min(CK, S.n - t0);
  extern __shared__ float smw[];   // ks [CK][RS] (then A [CK][CK + 1]) · xs [CK][RS] (q, then β e^{Lg} k, then β v) · Ts [CK][TS] · lg · bt
  float* ks = smw;
  float* xs = ks + CK * RS;
  float* Ts = xs + CK * RS;
  float* lg = Ts + CK * TS;
  float* bt = lg + CK;
  for (int i = tid; i < CK * 32; i += 256) {
    const int r = i / 32, d4 = i % 32;
    const size_t row = (size_t)S.row0 + t0 + r;
    const bool ok = r < nr;
    *reinterpret_cast<float4*>(ks + r * RS + d4 * 4) = ok ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    *reinterpret_cast<float4*>(xs + r * RS + d4 * 4) = ok ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  if (tid < CK) {
    const size_t row = (size_t)S.row0 + t0 + tid;
    lg[tid] = tid < nr ? logf(fmaxf(gb[row * 2 * VH + vh], 1e-30f)) : 0.f;
    bt[tid] = tid < nr ? gb[row * 2 * VH + VH + vh] : 0.f;
  }
  __syncthreads();
  if (tid == 0) for (int r = 1; r < CK; ++r) lg[r] += lg[r - 1];
  // warp: row tile (warp >> 1) · 16, column tiles 32 (warp & 1) .. + 31 (four n8) of K Kᵀ and of Q Kᵀ
  const int mr = (warp >> 1) * 16, nc = (warp & 1) * 32;
  float kk[4][4] = {}, qq[4][4] = {};
#pragma unroll 2
  for (int k0 = 0; k0 < 128; k0 += 8) {
    const float ak[4] = {ks[(mr + g) * RS + k0 + t], ks[(mr + g + 8) * RS + k0 + t], ks[(mr + g) * RS + k0 + t + 4], ks[(mr + g + 8) * RS + k0 + t + 4]};
    const float aq[4] = {xs[(mr + g) * RS + k0 + t], xs[(mr + g + 8) * RS + k0 + t], xs[(mr + g) * RS + k0 + t + 4], xs[(mr + g + 8) * RS + k0 + t + 4]};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float b[2] = {ks[(nc + 8 * j + g) * RS + k0 + t], ks[(nc + 8 * j + g) * RS + k0 + t + 4]};
      gtc::mma3(kk[j], ak, b);
      gtc::mma3(qq[j], aq, b);
    }
  }
  __syncthreads();   // lg prefix sum visible; every warp is done with ks / xs
  bf16* Mq = Mw + ((size_t)ch * VH + vh) * CK * CK;
  float (*A)[CK + 1] = reinterpret_cast<float (*)[CK + 1]>(ks);
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int tt = mr + g + (e >= 2 ? 8 : 0), jj = nc + 8 * j + 2 * t + (e & 1);
      const float dec = jj <= tt ? expf(lg[tt] - lg[jj]) : 0.f;
      A[tt][jj] = jj < tt ? bt[tt] * dec * kk[j][e] : 0.f;
      Mq[tt * CK + jj] = __float2bfloat16_rn(jj <= tt ? dec * qq[j][e] : 0.f);
    }
  if (tid < CK) Lw[((size_t)ch * VH + vh) * CK + tid] = lg[tid];
  __syncthreads();
  {   // T = (I + A)⁻¹ (forward substitution, as gdn_chunk_prep_kernel)
    const int j = tid >> 2, part = tid & 3;
    for (int tt = 0; tt < CK; ++tt) {
      float sum = 0.f;
      for (int m = j + part; m < tt; m += 4) sum = fmaf(A[tt][m], Ts[m * TS + j], sum);
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      if (part == 0) Ts[tt * TS + j] = (tt == j ? 1.f : 0.f) - (tt > j ? sum : 0.f);
      __syncwarp();
    }
  }
  __syncthreads();
  // X = β e^{Lg} ⊙ k (the k tile is gone: A took its place — reload from global / L2)
  for (int i = tid; i < CK * 32; i += 256) {
    const int r = i / 32, d4 = i % 32;
    const size_t row = (size_t)S.row0 + t0 + r;
    const float f = bt[r] * expf(lg[r]);
    const float4 kv = r < nr ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    *reinterpret_cast<float4*>(xs + r * RS + d4 * 4) = make_float4(f * kv.x, f * kv.y, f * kv.z, f * kv.w);
  }
  __syncthreads();
  // dst [CK][128] = T · X: warp rows (warp >> 1) · 16, columns 64 (warp & 1) .. + 63 (eight n8)
  // T and X as bf16 tiles for the big T·X products (T itself solved in fp32 above): Tb [CK][CK + 8] over the A region, Xb over the k region
  bf16* Tb = reinterpret_cast<bf16*>(ks);
  bf16* Xb = Tb + CK * (CK + 8);
  constexpr int LT = CK + 8, LX = CK + 8;   // Xb stored transposed [d][i] (B operand as [n][k])
  for (int i = tid; i < CK * CK; i += 256) { const int r = i / CK, c = i % CK; Tb[r * LT + c] = __float2bfloat16_rn(Ts[r * TS + c]); }
  auto tmul = [&](bf16* dst) {
    for (int i = tid; i < CK * 128; i += 256) { const int r = i / 128, c = i % 128; Xb[c * LX + r] = __float2bfloat16_rn(xs[r * RS + c]); }
    __syncthreads();
    const int cb = (warp & 1) * 64;
    float acc[8][4] = {};
    for (int k0 = 0; k0 <= mr; k0 += 16) {   // T lower triangular
      uint32_t a[4]; gv2::afrag(Tb, LT, mr, k0, g, t, a);
#pragma unroll
      for (int j = 0; j < 8; ++j) { uint32_t b0, b1; gv2::bfrag(Xb, LX, k0, cb + 8 * j, g, t, b0, b1); gv2::mma(acc[j], a[0], a[1], a[2], a[3], b0, b1); }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int r = mr + g, c = cb + 8 * j + 2 * t;
      *reinterpret_cast<uint32_t*>(dst + r * 128 + c) = tc::pack_bf2(acc[j][0], acc[j][1]); *reinterpret_cast<uint32_t*>(dst + (r + 8) * 128 + c) = tc::pack_bf2(acc[j][2], acc[j][3]);
    }
  };
  tmul(Wkw + ((size_t)ch * VH + vh) * CK * 128);
  __syncthreads();
  for (int i = tid; i < CK * 128; i += 256) {
    const int r = i / 128, c = i % 128;
    const size_t row = (size_t)S.row0 + t0 + r;
    xs[r * RS + c] = r < nr ? bt[r] * __bfloat162float(y[row * C + 2 * KW + (size_t)vh * 128 + c]) : 0.f;
  }
  __syncthreads();
  tmul(Uvw + ((size_t)ch * VH + vh) * CK * 128);
}
// prep v3: one block per (chunk, KEY head) — K·Kᵀ and Q·Kᵀ do not depend on the value head (only β and the gate do), so they are formed once
//   (3 × TF32) for the key head's value heads; then per value head: A / M from the same fragments, T by forward substitution (fp32), T·X (bf16).
__global__ void __launch_bounds__(256) gdn_v3_prep_kernel(const GdnSeg* segs, const bf16* y, int C, int KH, int VH, const float* qk, const float* gb,
                                                           bf16* Wkw, bf16* Uvw, bf16* Mw, float* Lw) {
  using namespace gwy;
  constexpr int RS = 132, TS = 68, LT = CK + 8, LX = CK + 8;
  const GdnSeg S = segs[0];
  const int ch = blockIdx.x, kh = blockIdx.y, rep = VH / KH, KW = KH * 128, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int t0 = ch * CK, nr = min(CK, S.n - t0);
  extern __shared__ float smw3[];   // ks [CK][RS] (kept) · xs [CK][RS] (q, then A / Tb / Xb) · Ts [CK][TS] · lg · bt
  float* ks = smw3;
  float* xs = ks + CK * RS;
  float* Ts = xs + CK * RS;
  float* lg = Ts + CK * TS;
  float* bt = lg + CK;
  for (int i = tid; i < CK * 32; i += 256) {
    const int r = i / 32, d4 = i % 32;
    const size_t row = (size_t)S.row0 + t0 + r;
    const bool ok = r < nr;
    *reinterpret_cast<float4*>(ks + r * RS + d4 * 4) = ok ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    *reinterpret_cast<float4*>(xs + r * RS + d4 * 4) = ok ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  __syncthreads();
  const int mr = (warp >> 1) * 16, nc = (warp & 1) * 32;
  float kk[4][4] = {}, qq[4][4] = {};
#pragma unroll 2
  for (int k0 = 0; k0 < 128; k0 += 8) {
    const float ak[4] = {ks[(mr + g) * RS + k0 + t], ks[(mr + g + 8) * RS + k0 + t], ks[(mr + g) * RS + k0 + t + 4], ks[(mr + g + 8) * RS + k0 + t + 4]};
    const float aq[4] = {xs[(mr + g) * RS + k0 + t], xs[(mr + g + 8) * RS + k0 + t], xs[(mr + g) * RS + k0 + t + 4], xs[(mr + g + 8) * RS + k0 + t + 4]};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float b[2] = {ks[(nc + 8 * j + g) * RS + k0 + t], ks[(nc + 8 * j + g) * RS + k0 + t + 4]};
      gtc::mma3(kk[j], ak, b);
      gtc::mma3(qq[j], aq, b);
    }
  }
  float (*A)[CK + 1] = reinterpret_cast<float (*)[CK + 1]>(xs);
  bf16* Tb = reinterpret_cast<bf16*>(xs);
  bf16* Xb = Tb + CK * LT;
  for (int v = 0; v < rep; ++v) {
    const int vh = kh * rep + v;
    __syncthreads();   // previous head done with xs / Ts / lg (and, first time, every warp done with q)
    if (tid < CK) {
      const size_t row = (size_t)S.row0 + t0 + tid;
      lg[tid] = tid < nr ? logf(fmaxf(gb[row * 2 * VH + vh], 1e-30f)) : 0.f;
      bt[tid] = tid < nr ? gb[row * 2 * VH + VH + vh] : 0.f;
    }
    __syncthreads();
    if (tid == 0) for (int r = 1; r < CK; ++r) lg[r] += lg[r - 1];
    __syncthreads();
    bf16* Mq = Mw + ((size_t)ch * VH + vh) * CK * CK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int tt = mr + g + (e >= 2 ? 8 : 0), jj = nc + 8 * j + 2 * t + (e & 1);
        const float dec = jj <= tt ? expf(lg[tt] - lg[jj]) : 0.f;
        A[tt][jj] = jj < tt ? bt[tt] * dec * kk[j][e] : 0.f;
        Mq[tt * CK + jj] = __float2bfloat16_rn(jj <= tt ? dec * qq[j][e] : 0.f);
      }
    if (tid < CK) Lw[((size_t)ch * VH + vh) * CK + tid] = lg[tid];
    __syncthreads();
    {   // T = (I + A)⁻¹, fp32 forward substitution
      const int j = tid >> 2, part = tid & 3;
      for (int tt = 0; tt < CK; ++tt) {
        float sum = 0.f;
        for (int m = j + part; m < tt; m += 4) sum = fmaf(A[tt][m], Ts[m * TS + j], sum);
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        if (part == 0) Ts[tt * TS + j] = (tt == j ? 1.f : 0.f) - (tt > j ? sum : 0.f);
        __syncwarp();
      }
    }
    __syncthreads();   // A is dead: its region takes Tb and Xb
    for (int i = tid; i < CK * CK; i += 256) { const int r = i / CK, c = i % CK; Tb[r * LT + c] = __float2bfloat16_rn(Ts[r * TS + c]); }
    auto tmul = [&](bf16* dst) {
      __syncthreads();
      const int cb = (warp & 1) * 64;
      float acc[8][4] = {};
      for (int k0 = 0; k0 <= mr; k0 += 16) {
        uint32_t a[4]; gv2::afrag(Tb, LT, mr, k0, g, t, a);
#pragma unroll
        for (int j = 0; j < 8; ++j) { uint32_t b0, b1; gv2::bfrag(Xb, LX, k0, cb + 8 * j, g, t, b0, b1); gv2::mma(acc[j], a[0], a[1], a[2], a[3], b0, b1); }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int r = mr + g, c = cb + 8 * j + 2 * t;
        *reinterpret_cast<uint32_t*>(dst + r * 128 + c) = tc::pack_bf2(acc[j][0], acc[j][1]); *reinterpret_cast<uint32_t*>(dst + (r + 8) * 128 + c) = tc::pack_bf2(acc[j][2], acc[j][3]);
      }
    };
    for (int i = tid; i < CK * 128; i += 256) {   // Xbᵀ = β e^{Lg} ⊙ k (from the kept fp32 k tile)
      const int r = i % CK, d = i / CK;
      Xb[d * LX + r] = __float2bfloat16_rn(bt[r] * expf(lg[r]) * ks[r * RS + d]);
    }
    tmul(Wkw + ((size_t)ch * VH + vh) * CK * 128);
    __syncthreads();
    for (int i = tid; i < CK * 128; i += 256) {   // Xbᵀ = β ⊙ v
      const int r = i % CK, d = i / CK;
      const size_t row = (size_t)S.row0 + t0 + r;
      Xb[d * LX + r] = __float2bfloat16_rn(r < nr ? bt[r] * __bfloat162float(y[row * C + 2 * KW + (size_t)vh * 128 + d]) : 0.f);
    }
    tmul(Uvw + ((size_t)ch * VH + vh) * CK * 128);
  }
}
// prep v4: as v3, with K·Kᵀ / Q·Kᵀ over four 32-wide d slices (≈ 37 KB of shared memory → 2 blocks per SM) and T = (I + A)⁻¹ blocked: the
//   four 16 × 16 diagonal blocks by forward substitution in parallel, then T_ij = −T_ii Σ_{j ≤ k < i} A_ik T_kj by distance i − j (fp32).
__global__ void __launch_bounds__(256, 2) gdn_v4_prep_kernel(const GdnSeg* segs, const bf16* y, int C, int KH, int VH, const float* qk, const float* gb,
                                                              bf16* Wkw, bf16* Uvw, bf16* Mw, float* Lw) {
  using namespace gwy;
  constexpr int QS4 = 36, TS = 72, LT = CK + 8, LX = CK + 8, AS = CK + 1;
  const GdnSeg S = segs[0];
  const int ch = blockIdx.x, kh = blockIdx.y, rep = VH / KH, KW = KH * 128, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int t0 = ch * CK, nr = min(CK, S.n - t0);
  extern __shared__ float smw4[];   // R1 [CK][TS] floats: k / q slices → A → Tb (bf16) · R2 [CK][TS]: Ts → Xb (bf16) · lg · bt
  float* R1 = smw4;
  float* R2 = R1 + CK * TS;
  float* lg = R2 + CK * TS;
  float* bt = lg + CK;
  float* ks = R1;               // [CK][QS4]
  float* xs = R1 + CK * QS4;    // [CK][QS4]
  const int mr = (warp >> 1) * 16, nc = (warp & 1) * 32;
  float kk[4][4] = {}, qq[4][4] = {};
  for (int dq = 0; dq < 128; dq += 32) {
    __syncthreads();
    for (int i = tid; i < CK * 8; i += 256) {
      const int r = i / 8, d4 = i % 8;
      const size_t row = (size_t)S.row0 + t0 + r;
      const bool ok = r < nr;
      *reinterpret_cast<float4*>(ks + r * QS4 + d4 * 4) = ok ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + KW + (size_t)kh * 128 + dq + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
      *reinterpret_cast<float4*>(xs + r * QS4 + d4 * 4) = ok ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + (size_t)kh * 128 + dq + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();
#pragma unroll
    for (int k0 = 0; k0 < 32; k0 += 8) {
      const float ak[4] = {ks[(mr + g) * QS4 + k0 + t], ks[(mr + g + 8) * QS4 + k0 + t], ks[(mr + g) * QS4 + k0 + t + 4], ks[(mr + g + 8) * QS4 + k0 + t + 4]};
      const float aq[4] = {xs[(mr + g) * QS4 + k0 + t], xs[(mr + g + 8) * QS4 + k0 + t], xs[(mr + g) * QS4 + k0 + t + 4], xs[(mr + g + 8) * QS4 + k0 + t + 4]};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float b[2] = {ks[(nc + 8 * j + g) * QS4 + k0 + t], ks[(nc + 8 * j + g) * QS4 + k0 + t + 4]};
        gtc::mma3(kk[j], ak, b);
        gtc::mma3(qq[j], aq, b);
      }
    }
  }
  float* A = R1;      // [CK][AS]
  float* Ts = R2;     // [CK][TS]
  bf16* Tb = reinterpret_cast<bf16*>(R1);   // [CK][LT]
  bf16* Xb = reinterpret_cast<bf16*>(R2);   // Xbᵀ [128][LX]
  for (int v = 0; v < rep; ++v) {
    const int vh = kh * rep + v;
    __syncthreads();
    if (tid < CK) {
      const size_t row = (size_t)S.row0 + t0 + tid;
      lg[tid] = tid < nr ? logf(fmaxf(gb[row * 2 * VH + vh], 1e-30f)) : 0.f;
      bt[tid] = tid < nr ? gb[row * 2 * VH + VH + vh] : 0.f;
    }
    __syncthreads();
    if (tid == 0) for (int r = 1; r < CK; ++r) lg[r] += lg[r - 1];
    __syncthreads();
    bf16* Mq = Mw + ((size_t)ch * VH + vh) * CK * CK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int tt = mr + g + (e >= 2 ? 8 : 0), jj = nc + 8 * j + 2 * t + (e & 1);
        const float dec = jj <= tt ? expf(lg[tt] - lg[jj]) : 0.f;
        A[tt * AS + jj] = jj < tt ? bt[tt] * dec * kk[j][e] : 0.f;
        Mq[tt * CK + jj] = __float2bfloat16_rn(jj <= tt ? dec * qq[j][e] : 0.f);
      }
    if (tid < CK) Lw[((size_t)ch * VH + vh) * CK + tid] = lg[tid];
    for (int i = tid; i < CK * CK; i += 256) Ts[(i / CK) * TS + i % CK] = 0.f;
    __syncthreads();
    {   // diagonal blocks: block b = tid / 64, column j = 16 b + (tid % 64) / 4, 4 lanes per column
      const int b = tid >> 6, j = 16 * b + ((tid & 63) >> 2), part = tid & 3;
      for (int tt = 16 * b; tt < 16 * b + 16; ++tt) {
        float sum = 0.f;
        for (int m = j + part; m < tt; m += 4) sum = fmaf(A[tt * AS + m], Ts[m * TS + j], sum);
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        if (part == 0) Ts[tt * TS + j] = (tt == j ? 1.f : 0.f) - (tt > j ? sum : 0.f);
        __syncwarp();
      }
    }
    // off-diagonal blocks by distance: R = Σ_{k = j..i−1} A_ik T_kj, then T_ij = −T_ii R (one element per thread per block)
    __shared__ float Rt[3][16 * 17];
    for (int dist = 1; dist < 4; ++dist) {
      __syncthreads();
      const int nb = 4 - dist;   // blocks (i, j = i − dist)
      for (int e = tid; e < nb * 256; e += 256) {
        const int bi = e / 256 + dist, bj = bi - dist, r = (e % 256) / 16, c = e % 16;
        float sum = 0.f;
        for (int kb = bj; kb < bi; ++kb)
#pragma unroll
          for (int q = 0; q < 16; ++q) sum = fmaf(A[(16 * bi + r) * AS + 16 * kb + q], Ts[(16 * kb + q) * TS + 16 * bj + c], sum);
        Rt[e / 256][r * 17 + c] = sum;
      }
      __syncthreads();
      for (int e = tid; e < nb * 256; e += 256) {
        const int bi = e / 256 + dist, bj = bi - dist, r = (e % 256) / 16, c = e % 16;
        float sum = 0.f;
#pragma unroll
        for (int q = 0; q < 16; ++q) sum = fmaf(Ts[(16 * bi + r) * TS + 16 * bi + q], Rt[e / 256][q * 17 + c], sum);
        Ts[(16 * bi + r) * TS + 16 * bj + c] = -sum;
      }
    }
    __syncthreads();   // A dead → Tb
    for (int i = tid; i < CK * CK; i += 256) { const int r = i / CK, c = i % CK; Tb[r * LT + c] = __float2bfloat16_rn(Ts[r * TS + c]); }
    __syncthreads();   // Ts dead → Xb
    auto tmul = [&](bf16* dst) {
      __syncthreads();
      const int cb = (warp & 1) * 64;
      float acc[8][4] = {};
      for (int k0 = 0; k0 <= mr; k0 += 16) {
        uint32_t a[4]; gv2::afrag(Tb, LT, mr, k0, g, t, a);
#pragma unroll
        for (int j = 0; j < 8; ++j) { uint32_t b0, b1; gv2::bfrag(Xb, LX, k0, cb + 8 * j, g, t, b0, b1); gv2::mma(acc[j], a[0], a[1], a[2], a[3], b0, b1); }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int r = mr + g, c = cb + 8 * j + 2 * t;
        *reinterpret_cast<uint32_t*>(dst + r * 128 + c) = tc::pack_bf2(acc[j][0], acc[j][1]); *reinterpret_cast<uint32_t*>(dst + (r + 8) * 128 + c) = tc::pack_bf2(acc[j][2], acc[j][3]);
      }
    };
    for (int i = tid; i < CK * 128; i += 256) {   // Xbᵀ = β e^{Lg} ⊙ k (k from global / L2)
      const int r = i % CK, d = i / CK;
      const size_t row = (size_t)S.row0 + t0 + r;
      const float kv = r < nr ? qk[row * 2 * KW + KW + (size_t)kh * 128 + d] : 0.f;
      Xb[d * LX + r] = __float2bfloat16_rn(bt[r] * expf(lg[r]) * kv);
    }
    tmul(Wkw + ((size_t)ch * VH + vh) * CK * 128);
    __syncthreads();
    for (int i = tid; i < CK * 128; i += 256) {
      const int r = i % CK, d = i / CK;
      const size_t row = (size_t)S.row0 + t0 + r;
      Xb[d * LX + r] = __float2bfloat16_rn(r < nr ? bt[r] * __bfloat162float(y[row * C + 2 * KW + (size_t)vh * 128 + d]) : 0.f);
    }
    tmul(Uvw + ((size_t)ch * VH + vh) * CK * 128);
  }
}
// outputs v3: as gdn_v2_out_kernel, the 128 columns in four passes of 32 (≈ 40 KB → 2 blocks per SM)
__global__ void __launch_bounds__(256, 2) gdn_v3_out_kernel(const GdnSeg* segs, int KH, int VH, const float* qk, const bf16* Sctw, const bf16* Utw,
                                                             const bf16* Mw, const float* Lw, float* o) {
  using namespace gv2;
  constexpr int LQ = 128 + 8, LS = 128 + 8, LM = CK + 8, LU = CK + 8;
  extern __shared__ __align__(16) unsigned char smv3[];
  bf16* Qs = reinterpret_cast<bf16*>(smv3);   // [CK][LQ]
  bf16* Ms = Qs + CK * LQ;                     // [CK][LM]
  bf16* Ss = Ms + CK * LM;                     // S_cᵀ slice [32][LS]
  bf16* Us = Ss + 32 * LS;                     // Uᵀ slice [32][LU]
  float* lg = reinterpret_cast<float*>(Us + 32 * LU);
  const GdnSeg S = segs[0];
  const int ch = blockIdx.x, vh = blockIdx.y, rep = VH / KH, kh = vh / rep, KW = KH * 128, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int t0 = ch * CK, nr = min(CK, S.n - t0);
  for (int i = tid; i < CK * 32; i += 256) {
    const int r = i / 32, d4 = i % 32;
    const size_t row = (size_t)S.row0 + t0 + r;
    const float4 q4 = r < nr ? *reinterpret_cast<const float4*>(qk + row * 2 * KW + (size_t)kh * 128 + d4 * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    *reinterpret_cast<uint32_t*>(Qs + r * LQ + d4 * 4) = tc::pack_bf2(q4.x, q4.y);
    *reinterpret_cast<uint32_t*>(Qs + r * LQ + d4 * 4 + 2) = tc::pack_bf2(q4.z, q4.w);
  }
  const bf16* Mg = Mw + ((size_t)ch * VH + vh) * CK * CK;
  for (int i = tid; i < CK * 8; i += 256) { const int r = i / 8, q = i % 8; *reinterpret_cast<uint4*>(Ms + r * LM + q * 8) = *reinterpret_cast<const uint4*>(Mg + (size_t)r * CK + q * 8); }
  if (tid < CK) lg[tid] = Lw[((size_t)ch * VH + vh) * CK + tid];
  const bf16* Sg = Sctw + ((size_t)ch * VH + vh) * 128 * 128;
  const bf16* Ug = Utw + ((size_t)ch * VH + vh) * 128 * CK;
  const int mr = (warp >> 1) * 16, nb = (warp & 1) * 16;
  for (int cz = 0; cz < 128; cz += 32) {
    __syncthreads();
    for (int i = tid; i < 32 * 16; i += 256) { const int r = i / 16, q = i % 16; *reinterpret_cast<uint4*>(Ss + r * LS + q * 8) = *reinterpret_cast<const uint4*>(Sg + (size_t)(cz + r) * 128 + q * 8); }
    for (int i = tid; i < 32 * 8; i += 256) { const int r = i / 8, q = i % 8; *reinterpret_cast<uint4*>(Us + r * LU + q * 8) = *reinterpret_cast<const uint4*>(Ug + (size_t)(cz + r) * CK + q * 8); }
    __syncthreads();
    float qa[2][4] = {}, ma[2][4] = {};
#pragma unroll
    for (int k0 = 0; k0 < 128; k0 += 16) {
      uint32_t a[4]; afrag(Qs, LQ, mr, k0, g, t, a);
#pragma unroll
      for (int j = 0; j < 2; ++j) { uint32_t b0, b1; bfrag(Ss, LS, k0, nb + 8 * j, g, t, b0, b1); mma(qa[j], a[0], a[1], a[2], a[3], b0, b1); }
    }
    for (int k0 = 0; k0 <= mr; k0 += 16) {
      uint32_t a[4]; afrag(Ms, LM, mr, k0, g, t, a);
#pragma unroll
      for (int j = 0; j < 2; ++j) { uint32_t b0, b1; bfrag(Us, LU, k0, nb + 8 * j, g, t, b0, b1); mma(ma[j], a[0], a[1], a[2], a[3], b0, b1); }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int r = mr + g + (e >= 2 ? 8 : 0), c = cz + nb + 8 * j + 2 * t + (e & 1);
        if (r < nr) o[((size_t)S.row0 + t0 + r) * (size_t)VH * 128 + (size_t)vh * 128 + c] = bfr(expf(lg[r]) * qa[j][e] + ma[j][e]);
      }
  }
}
// the chunked prefill's workspaces at their size for `rows` rows, allocated up front (they grew on the first long chunk: measured 2026-10-12,
//   32 GB KV tier, ~190 MiB on the first 2048-row chunk of a 32K prompt with the VRAM already handed to KV and the staging copy → the daemon
//   aborted on this cudaMalloc). Same sizes as gdn() computes for the default (WY v2) path.
void gdn_reserve(int rows, int KH, int VH) {
  (void)KH;
  const int nch = (rows + gc::CK - 1) / gc::CK;
  g_gdn_ws.ensure((size_t)nch * VH * (2 * gc::CK * gc::CK + gc::CK) * 4);
  const bool v2_on = !getenv("HIVE_Q27_GDN_WY") || strcmp(getenv("HIVE_Q27_GDN_WY"), "4") == 0;
  if (v2_on) {
    const size_t per = (size_t)nch * VH;
    const size_t nWk = per * gv2::CK * 128, nSc = per * 128 * 128, nU = per * 128 * gv2::CK, nM = per * gv2::CK * gv2::CK;
    g_gdn_wsv.ensure((2 * nWk + nSc + nU + nM) * 2 + per * gv2::CK * 4 + 256);
  }
}

void gdn(const GdnW& w, const GdnSeg* segs, int n_segs, int rows, const bf16* proj, int ld, int KH, int VH, bf16* y, float* qk, float* gb, float* o,
         bf16* out, cudaStream_t st) {
  if (rows <= 0) return;
  const int KW = KH * 128, VW = VH * 128, C = 2 * KW + VW;
  if (rows >= 64 * n_segs) {   // prefill: rows in parallel, then the state
    gdn_conv_par_kernel<<<dim3(blocks(C, 128), n_segs, 64), 128, 0, st>>>(segs, proj, ld, w.conv_w, C, y);
    gdn_conv_state_kernel<<<dim3(blocks(C, 128), n_segs), 128, 0, st>>>(segs, proj, ld, C);
  } else {
    gdn_conv_kernel<<<dim3(blocks(C, 128), n_segs), 128, 0, st>>>(segs, proj, ld, w.conv_w, C, y);
  }
  gdn_prep_kernel<<<rows, 256, 0, st>>>(y, C, KH, VH, proj, ld, C + VW, w.A_log, w.dt_bias, qk, gb);
  // HIVE_Q27_GDN_CHUNK: 0 = row loop only · N ≥ 2 = chunked from N rows · unset / 1 = from 2 · CK rows
  static const int chunk_min = [] {
    const char* v = getenv("HIVE_Q27_GDN_CHUNK");
    if (!v || !*v || strcmp(v, "1") == 0) return 2 * gc::CK;
    const int n = atoi(v);
    return n <= 0 ? INT_MAX : std::max(n, 2 * gc::CK);
  }();
  static const bool chunked = chunk_min != INT_MAX;
  if (chunked && n_segs == 1 && rows >= chunk_min) {   // prefill of one sequence (verify snapshots never take this path: ≤ 16 rows)
    DevBufLite& ws = g_gdn_ws;
    const int nch = (rows + gc::CK - 1) / gc::CK;
    const size_t need = (size_t)nch * VH * (2 * gc::CK * gc::CK + gc::CK) * 4;
    ws.ensure(need);
    float* Tw = reinterpret_cast<float*>(ws.p);
    float* Mw = Tw + (size_t)nch * VH * gc::CK * gc::CK;
    float* Lw = Mw + (size_t)nch * VH * gc::CK * gc::CK;
    const size_t smp = (size_t)(2 * gc::CK * 129 + gc::CK * (gc::CK + 1) + 2 * gc::CK) * 4;
    static bool attrp = false;
    if (!attrp) { CUDA_CHECK(cudaFuncSetAttribute(gdn_chunk_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smp)); attrp = true; }
    static const bool wy_prep_skip = !getenv("HIVE_Q27_GDN_WY") || atoi(getenv("HIVE_Q27_GDN_WY")) >= 1;
    if (!wy_prep_skip) gdn_chunk_prep_kernel<<<dim3(nch, VH), 256, smp, st>>>(segs, KH, VH, qk, gb, Tw, Mw, Lw);
    const size_t smem = (size_t)(128 * 16 + gc::CK * 132 + gc::kQReg + 2 * gc::CK * 16 + 2 * gc::CK) * 4;
    static bool attr = false;
    if (!attr) { CUDA_CHECK(cudaFuncSetAttribute(gdn_chunk_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
    static const bool gtc_on = getenv("HIVE_Q27_GDN_TC") && strcmp(getenv("HIVE_Q27_GDN_TC"), "1") == 0;
    static const bool wy_on = getenv("HIVE_Q27_GDN_WY") && (strcmp(getenv("HIVE_Q27_GDN_WY"), "1") == 0 || strcmp(getenv("HIVE_Q27_GDN_WY"), "2") == 0 || strcmp(getenv("HIVE_Q27_GDN_WY"), "3") == 0);
    // default since 2026-10-11 (quality gate passed: quality 147/157 = baseline, oracle tokens identical); HIVE_Q27_GDN_WY=0 = the fp32 chunk scan
    static const bool v2_on = !getenv("HIVE_Q27_GDN_WY") || strcmp(getenv("HIVE_Q27_GDN_WY"), "4") == 0;
    if (v2_on) {   // WY v2: bf16 storage / operands, fp32 accumulation and serial state
      DevBufLite& wsv = g_gdn_wsv;
      const size_t per = (size_t)nch * VH;
      const size_t nWk = per * gv2::CK * 128, nSc = per * 128 * 128, nU = per * 128 * gv2::CK, nM = per * gv2::CK * gv2::CK;
      wsv.ensure((2 * nWk + nSc + nU + nM) * 2 + per * gv2::CK * 4 + 256);
      bf16* Wk2 = reinterpret_cast<bf16*>(wsv.p);
      bf16* Uv2 = Wk2 + nWk;
      bf16* Sc2 = Uv2 + nWk;
      bf16* Ut2 = Sc2 + nSc;
      bf16* M2 = Ut2 + nU;
      float* L2 = reinterpret_cast<float*>(M2 + nM);
      const size_t sp1t = (size_t)(2 * gc::CK * 132 + gc::CK * 68 + 2 * gc::CK) * 4;
      const size_t sps = (size_t)(gv2::CK * 136 + 128 * 72 + 32 * 136 + 32 * 72) * 2 + gv2::CK * 4;
      const size_t spo = (size_t)(gv2::CK * 136 + 128 * 136 + gv2::CK * 72 + 128 * 72) * 2 + gv2::CK * 4;
      static bool c1 = false, c2 = false, c3 = false;
      if (!c1) { CUDA_CHECK(cudaFuncSetAttribute(gdn_v2_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sp1t)); c1 = true; }
      if (!c2) { CUDA_CHECK(cudaFuncSetAttribute(gdn_v2_state_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sps)); c2 = true; }
      if (!c3) { CUDA_CHECK(cudaFuncSetAttribute(gdn_v2_out_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)spo)); c3 = true; }
      static bool c0 = false;
      if (!c0) { CUDA_CHECK(cudaFuncSetAttribute(gdn_v3_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sp1t)); c0 = true; }
      static const int v4 = getenv("HIVE_Q27_GDN_V4") ? atoi(getenv("HIVE_Q27_GDN_V4")) : 1;   // 0 = iteration-3 prep / output kernels
      if (v4) {
        const size_t sp4 = (size_t)(2 * gc::CK * 72 + 2 * gc::CK) * 4;
        static bool c4 = false;
        if (!c4) { CUDA_CHECK(cudaFuncSetAttribute(gdn_v4_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sp4)); c4 = true; }
        gdn_v4_prep_kernel<<<dim3(nch, KH), 256, sp4, st>>>(segs, y, C, KH, VH, qk, gb, Wk2, Uv2, M2, L2);
      } else
      gdn_v3_prep_kernel<<<dim3(nch, KH), 256, sp1t, st>>>(segs, y, C, KH, VH, qk, gb, Wk2, Uv2, M2, L2);
      gdn_v2_state_kernel<<<dim3(VH, 4), 256, sps, st>>>(segs, KH, VH, qk, Wk2, Uv2, L2, Sc2, Ut2);
      if (v4) {
        const size_t spo3 = (size_t)(gv2::CK * 136 + gv2::CK * 72 + 32 * 136 + 32 * 72) * 2 + gv2::CK * 4;
        static bool c5 = false;
        if (!c5) { CUDA_CHECK(cudaFuncSetAttribute(gdn_v3_out_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)spo3)); c5 = true; }
        gdn_v3_out_kernel<<<dim3(nch, VH), 256, spo3, st>>>(segs, KH, VH, qk, Sc2, Ut2, M2, L2, o);
      } else
      gdn_v2_out_kernel<<<dim3(nch, VH), 256, spo, st>>>(segs, KH, VH, qk, Sc2, Ut2, M2, L2, o);
    } else if (wy_on) {   // WY form: prep (Wk, Uv, M, Lg) · state passing (S_c, U) · outputs
      static DevBufLite wsw;
      const size_t per = (size_t)nch * VH;
      wsw.ensure(per * ((size_t)3 * gc::CK * 128 + (size_t)128 * 128) * 4);
      float* Wkw = reinterpret_cast<float*>(wsw.p);
      float* Uvw = Wkw + per * gc::CK * 128;
      float* Uw = Uvw + per * gc::CK * 128;
      float* Scw = Uw + per * gc::CK * 128;
      const size_t sp1 = (size_t)(2 * gc::CK * 129 + 2 * gc::CK * (gc::CK + 1) + 2 * gc::CK) * 4;
      const size_t sp2 = (size_t)(128 * 16 + 2 * gc::CK * gwy::KS + gc::CK * 16 + 2 * gc::CK) * 4;
      const size_t sp3 = (size_t)(gc::CK * gwy::KS + 128 * 32 + gc::CK * (gc::CK + 1) + gc::CK * 32 + gc::CK) * 4;
      static bool a1 = false, a2 = false, a3 = false;
      if (!a1) { CUDA_CHECK(cudaFuncSetAttribute(gdn_wy_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sp1)); a1 = true; }
      if (!a2) { CUDA_CHECK(cudaFuncSetAttribute(gdn_wy_state_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sp2)); a2 = true; }
      if (!a3) { CUDA_CHECK(cudaFuncSetAttribute(gdn_wy_out_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sp3)); a3 = true; }
      static const bool wy_tc1 = getenv("HIVE_Q27_GDN_WY") && (strcmp(getenv("HIVE_Q27_GDN_WY"), "2") == 0 || strcmp(getenv("HIVE_Q27_GDN_WY"), "3") == 0);
      static const bool wy_one = getenv("HIVE_Q27_GDN_WY") && strcmp(getenv("HIVE_Q27_GDN_WY"), "3") == 0;
      if (wy_tc1) {
        const size_t sp1t = (size_t)(2 * gc::CK * 132 + gc::CK * 68 + 2 * gc::CK) * 4;
        static bool b1 = false;
        if (!b1) { CUDA_CHECK(cudaFuncSetAttribute(gdn_wy_prep_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sp1t)); b1 = true; }
        gdn_wy_prep_tc_kernel<<<dim3(nch, VH), 256, sp1t, st>>>(segs, y, C, KH, VH, qk, gb, Wkw, Uvw, Mw, Lw, wy_one);
      } else
      gdn_wy_prep_kernel<<<dim3(nch, VH), 256, sp1, st>>>(segs, y, C, KH, VH, qk, gb, Wkw, Uvw, Mw, Lw);
      static const bool wy_tc = wy_tc1;
      if (wy_tc) {
        const size_t st2 = (size_t)(128 * gwt::SS + 2 * gc::CK * gwt::KS + gc::CK * gwt::SS + gc::CK) * 4;
        const size_t st3 = (size_t)(gc::CK * gwt::QS + 128 * gwt::CS + gc::CK * gwt::MS + gc::CK * gwt::CS + gc::CK) * 4;
        static bool b2 = false, b3 = false;
        if (!b2) { CUDA_CHECK(cudaFuncSetAttribute(gdn_wy_state_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)st2)); b2 = true; }
        if (!b3) { CUDA_CHECK(cudaFuncSetAttribute(gdn_wy_out_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)st3)); b3 = true; }
        gdn_wy_state_tc_kernel<<<dim3(VH, 8), 256, st2, st>>>(segs, KH, VH, qk, Wkw, Uvw, Lw, Scw, Uw, wy_one);
        gdn_wy_out_tc_kernel<<<dim3(nch, VH, 4), 256, st3, st>>>(segs, KH, VH, qk, Scw, Uw, Mw, Lw, o, wy_one);
      } else {
      gdn_wy_state_kernel<<<dim3(VH, 8), 256, sp2, st>>>(segs, KH, VH, qk, Wkw, Uvw, Lw, Scw, Uw);
      gdn_wy_out_kernel<<<dim3(nch, VH, 4), 256, sp3, st>>>(segs, KH, VH, qk, Scw, Uw, Mw, Lw, o);
      }
    } else if (gtc_on) {
      const size_t smt = (size_t)(128 * gtc::SS + gc::CK * 129 + 2 * gc::CK * (gc::CK + 1) + 2 * gc::CK * gtc::SS + 2 * gc::CK) * 4;
      static bool attrt = false;
      if (!attrt) { CUDA_CHECK(cudaFuncSetAttribute(gdn_chunk_scan_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smt)); attrt = true; }
      gdn_chunk_scan_tc_kernel<<<dim3(VH, 8), 256, smt, st>>>(segs, y, C, KH, VH, qk, gb, Tw, Mw, Lw, o);
    } else
    gdn_chunk_scan_kernel<<<dim3(VH, 8), 256, smem, st>>>(segs, y, C, KH, VH, qk, gb, Tw, Mw, Lw, o);
  } else {
    gdn_rec_kernel<<<dim3(16, VH, n_segs), 32, 0, st>>>(segs, y, C, KH, VH, qk, gb, o);
  }
  static const bool dbg = getenv("HIVE_Q27_GDN_DEBUG") && strcmp(getenv("HIVE_Q27_GDN_DEBUG"), "1") == 0;
  if (dbg && n_segs == 1 && rows >= 64) {   // diagnostics (prefill calls only — decode runs inside graph captures): magnitudes after this layer's recurrence (state, outputs, gates, T)
    static long call = 0;
    CUDA_CHECK(cudaStreamSynchronize(st));
    GdnSeg hs; CUDA_CHECK(cudaMemcpy(&hs, segs, sizeof hs, cudaMemcpyDeviceToHost));
    std::vector<float> sv((size_t)VH * 128 * 128), ov((size_t)rows * VW), gv((size_t)rows * 2 * VH);
    CUDA_CHECK(cudaMemcpy(sv.data(), hs.state, sv.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ov.data(), o, ov.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gv.data(), gb, gv.size() * 4, cudaMemcpyDeviceToHost));
    auto mx = [](const std::vector<float>& v, size_t a, size_t b, int& bad) { float m = 0; for (size_t i = a; i < b; ++i) { if (!std::isfinite(v[i])) ++bad; else m = std::max(m, std::fabs(v[i])); } return m; };
    int b1 = 0, b2 = 0, b3 = 0;
    const float ms = mx(sv, 0, sv.size(), b1), mo = mx(ov, 0, ov.size(), b2);
    float gmin = 1e30f, gmax = 0, bmax = 0;
    for (int r = 0; r < rows; ++r) for (int h = 0; h < VH; ++h) { const float g = gv[(size_t)r * 2 * VH + h], b = gv[(size_t)r * 2 * VH + VH + h]; gmin = std::min(gmin, g); gmax = std::max(gmax, g); bmax = std::max(bmax, b); if (!std::isfinite(g) || !std::isfinite(b)) ++b3; }
    if (const char* dd = getenv("HIVE_Q27_GDN_DUMP")) {   // states after each call and the last 64 rows' outputs: <dir>/s<call>.bin
      char fn[512]; snprintf(fn, sizeof fn, "%s/s%04ld.bin", dd, call);
      if (FILE* f = fopen(fn, "wb")) { fwrite(sv.data(), 4, sv.size(), f); fwrite(ov.data() + (size_t)std::max(0, rows - 64) * VW, 4, (size_t)std::min(64, rows) * VW, f); fclose(f); }
    }
    fprintf(stderr, "[gdn-dbg] call %ld rows %d n %d: |S| max %.3g (bad %d) · |o| max %.3g (bad %d) · gate [%.3g, %.3g] β max %.3g (bad %d)\n", call++, rows, hs.n, ms, b1, mo, b2, gmin, gmax, bmax, b3);
  }
  gdn_norm_kernel<<<dim3(rows, VH), 128, 0, st>>>(o, proj, ld, C, w.norm, w.eps, VH, out);
  CUDA_CHECK(cudaGetLastError());
}
namespace {
__global__ void multi_copy_kernel(const CopyItem* items) {
  const CopyItem it = items[blockIdx.x];
  for (int i = threadIdx.x; i < it.n; i += blockDim.x) it.dst[i] = it.src[i];
}
}  // namespace
__global__ void copy_chunks_kernel(const ByteChunk* chunks) {
  const ByteChunk c = chunks[blockIdx.x];
  if ((((uintptr_t)c.src | (uintptr_t)c.dst | c.n) & 15) == 0) {
    const uint4* s = reinterpret_cast<const uint4*>(c.src);
    uint4* d = reinterpret_cast<uint4*>(c.dst);
    for (unsigned i = threadIdx.x; i < c.n / 16; i += blockDim.x) d[i] = s[i];
  } else {
    for (unsigned i = threadIdx.x; i < c.n; i += blockDim.x) c.dst[i] = c.src[i];
  }
}
void copy_chunks(const ByteChunk* chunks, int n, cudaStream_t st) {
  if (n <= 0) return;
  copy_chunks_kernel<<<n, 256, 0, st>>>(chunks);
  CUDA_CHECK(cudaGetLastError());
}
void multi_copy(const CopyItem* items, int n, cudaStream_t st) {
  if (n <= 0) return;
  multi_copy_kernel<<<n, 256, 0, st>>>(items);
  CUDA_CHECK(cudaGetLastError());
}
void gdn_replay(const GdnReplay* items, int n, int KH, int VH, int C, cudaStream_t st) {
  if (n <= 0) return;
  gdn_replay_kernel<<<dim3(16, VH, n), 32, 0, st>>>(items, C, KH, VH);
  CUDA_CHECK(cudaGetLastError());
}
void argmax_prob(const float* logits, int V, float* out, cudaStream_t st) {
  // out must hold 2 + 3·kAmBlocks floats (the partials follow the result)
  argmax_part_kernel<<<kAmBlocks, 256, 0, st>>>(logits, V, out + 2);
  argmax_final_kernel<<<1, 32, 0, st>>>(out + 2, kAmBlocks, out);
  CUDA_CHECK(cudaGetLastError());
}
void embed_by_id(const bf16* table, const int32_t* id_ptr, int dim, bf16* out, cudaStream_t st) {
  embed_by_id_kernel<<<4, 160, 0, st>>>(table, id_ptr, dim, out);
  CUDA_CHECK(cudaGetLastError());
}
void embed_rows(const bf16* table, const int32_t* ids, int M, int dim, bf16* out, cudaStream_t st) {
  if (M <= 0) return;
  embed_rows_kernel<<<dim3(2, M), 320, 0, st>>>(table, ids, dim, out);
  CUDA_CHECK(cudaGetLastError());
}
void map_ids(int32_t* id, const int32_t* map, cudaStream_t st) { map_ids_kernel<<<1, 1, 0, st>>>(id, map); CUDA_CHECK(cudaGetLastError()); }
void bf16_to_f32(const bf16* x, size_t n, float* y, cudaStream_t st) { if (n) b2f_kernel<<<blocks(n), 256, 0, st>>>(x, n, y); }
void f32_to_bf16(const float* x, size_t n, bf16* y, cudaStream_t st) { if (n) f2b_kernel<<<blocks(n), 256, 0, st>>>(x, n, y); }

}  // namespace hive::q27::k
