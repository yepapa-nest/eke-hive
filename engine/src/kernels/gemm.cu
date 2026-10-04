// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Block-scaled GEMM v1 (CUDA cores, correctness first).
//   C[m,n] = Σ_b ( Σ_{k∈block b} A[m,k]·B[n,k] ) · sa[m,b] · sb[n,b]   (same formula as the per-block scale accumulation of the reference fp8_gemm/fp4_gemm)
//   · M <= 8: warp-per-n GEMV (decode) · otherwise: 64×64×32 tiled GEMM (prefill v1)
// The block-scaled tensor-core version (mma.sync.m16n8k32.kind::mxf8f6f4.block_scale, e4m3×{e4m3,e2m1}, ue8m0) lives in gemm_mx.cu.
#include <cstdlib>

#include "hive/kernels.h"
#include "hive/warp_reduce.cuh"

namespace hive::k {

namespace {

__constant__ float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// ---- Hardware conversion (sm_89+ e4m3x2, sm_120a e2m1x2): the values fit exactly in fp16, so results are bit-identical to the scalar ldexpf path. -------------
// The scalar e4m3_to_f32 (ldexpf + branches) has a large instruction count and made the GEMV instruction-bound (measured: 73–390 GB/s, M=8 at 4× the time of M=1).
__device__ __forceinline__ float2 cvt_e4m3x2(uint32_t pair16) {  // low byte → .x, high byte → .y
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t)pair16));
  return __half22float2(*reinterpret_cast<__half2*>(&h2));
}
__device__ __forceinline__ float2 cvt_e2m1x2(uint32_t byte8) {  // low nibble → .x, high nibble → .y
  uint32_t h2;
  asm("{ .reg .b8 lo, hi;\n mov.b16 {lo, hi}, %1;\n cvt.rn.f16x2.e2m1x2 %0, lo; }" : "=r"(h2) : "h"((uint16_t)byte8));
  return __half22float2(*reinterpret_cast<__half2*>(&h2));
}
__device__ __forceinline__ float e8m0_fast_dev(uint8_t b) {  // 2^(b-127); b=0 is the fp32 subnormal 2^-127
  return b ? __uint_as_float((uint32_t)b << 23) : __uint_as_float(0x00400000u);
}
// Decode a 32-element block: fp8 32 B (uint4×2) or fp4 16 B (uint4) → w[32]
__device__ __forceinline__ void decode_fp8_block(const uint4 r0, const uint4 r1, float (&w)[32]) {
  const uint32_t x[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float2 a = cvt_e4m3x2(x[i] & 0xFFFFu), b = cvt_e4m3x2(x[i] >> 16);
    w[4 * i] = a.x; w[4 * i + 1] = a.y; w[4 * i + 2] = b.x; w[4 * i + 3] = b.y;
  }
}
__device__ __forceinline__ void decode_fp4_block(const uint4 r, float (&w)[32]) {
  const uint32_t x[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 v = cvt_e2m1x2((x[i] >> (8 * j)) & 0xFFu);
      w[8 * i + 2 * j] = v.x; w[8 * i + 2 * j + 1] = v.y;
    }
  }
}

// ---- GEMV: one warp per output column n. Lanes take the 32-element blocks in round-robin (one block = 32 elements = 32 products). -----------
template <bool B_FP4, int MMAX>
__global__ void gemv_bs_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa, const uint8_t* __restrict__ B,
                               const uint8_t* __restrict__ sb, int M, int N, int K, bf16* __restrict__ C,
                               float* __restrict__ C32) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= N) return;
  const int n = warp;
  const int nb = K / 32;
  float acc[MMAX];
#pragma unroll
  for (int m = 0; m < MMAX; ++m) acc[m] = 0.f;
  for (int b = lane; b < nb; b += 32) {
    // 32 values of the B block
    float w[32];
    if (B_FP4) {
      decode_fp4_block(*reinterpret_cast<const uint4*>(B + (size_t)n * (K / 2) + b * 16), w);
    } else {
      const uint8_t* pb = B + (size_t)n * K + b * 32;
      decode_fp8_block(*reinterpret_cast<const uint4*>(pb), *reinterpret_cast<const uint4*>(pb + 16), w);
    }
    const float sbv = e8m0_fast_dev(sb[(size_t)n * nb + b]);
#pragma unroll
    for (int m = 0; m < MMAX; ++m) {
      if (m < M) {
        const uint8_t* pa = A + (size_t)m * K + b * 32;
        float av[32];
        decode_fp8_block(*reinterpret_cast<const uint4*>(pa), *reinterpret_cast<const uint4*>(pa + 16), av);
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) d = fmaf(av[i], w[i], d);
        acc[m] = fmaf(d, e8m0_fast_dev(sa[(size_t)m * nb + b]) * sbv, acc[m]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MMAX; ++m) {
    float v = acc[m];
    v = hive::cu::warp_allreduce(v);
    if (lane == 0 && m < M) {
      if (C) C[(size_t)m * N + n] = f2bf(v);
      if (C32) C32[(size_t)m * N + n] = v;
    }
  }
}

// ---- Grouped GEMV (decode experts): one warp per (group g, column n). Same loop and reduction order as gemv_bs_kernel<true,8> → same fp32.
__device__ __forceinline__ void gemv_fp4_warp(const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa, const uint8_t* __restrict__ B,
                                              const uint8_t* __restrict__ sb, int M, int K, int n, int lane, float (&acc)[8]) {
  const int nb = K / 32;
#pragma unroll
  for (int m = 0; m < 8; ++m) acc[m] = 0.f;
  for (int b = lane; b < nb; b += 32) {
    float w[32];
    decode_fp4_block(*reinterpret_cast<const uint4*>(B + (size_t)n * (K / 2) + b * 16), w);
    const float sbv = e8m0_fast_dev(sb[(size_t)n * nb + b]);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      if (m < M) {
        const uint8_t* pa = A + (size_t)m * K + b * 32;
        float av[32];
        decode_fp8_block(*reinterpret_cast<const uint4*>(pa), *reinterpret_cast<const uint4*>(pa + 16), av);
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) d = fmaf(av[i], w[i], d);
        acc[m] = fmaf(d, e8m0_fast_dev(sa[(size_t)m * nb + b]) * sbv, acc[m]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < 8; ++m) {
    float v = acc[m];
    v = hive::cu::warp_allreduce(v);
    acc[m] = v;
  }
}

// ---- Decode wo_a: one warp per output column (g, r). A is bf16 (exact), B is fp8 + block scales. fp32 accumulation (sequential within a block → times block scale → sum). --------
template <int MMAX>
__global__ void gemv_bf16_fp8_grouped_kernel(const bf16* __restrict__ A, int lda, const uint8_t* __restrict__ B, const uint8_t* __restrict__ sb,
                                             int M, int G, int R, int K, bf16* __restrict__ C) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= G * R) return;
  const int g = warp / R, n = warp;  // n = g·R + r (row of B)
  const int nb = K / 32;
  float acc[MMAX];
#pragma unroll
  for (int m = 0; m < MMAX; ++m) acc[m] = 0.f;
  for (int b = lane; b < nb; b += 32) {
    float w[32];
    const uint8_t* pb = B + (size_t)n * K + b * 32;
    decode_fp8_block(*reinterpret_cast<const uint4*>(pb), *reinterpret_cast<const uint4*>(pb + 16), w);
    const float sbv = e8m0_fast_dev(sb[(size_t)n * nb + b]);
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
#pragma unroll
  for (int m = 0; m < MMAX; ++m) {
    float v = acc[m];
    v = hive::cu::warp_allreduce(v);
    if (lane == 0 && m < M) C[(size_t)m * (G * R) + n] = f2bf(v);
  }
}

constexpr int GRP_WARPS = 8;  // block = 8 columns

__global__ void __launch_bounds__(GRP_WARPS * 32) gemv_grouped_w13_kernel(const GroupDesc* __restrict__ g, const uint8_t* __restrict__ A,
                                                                          const uint8_t* __restrict__ sa, int I, int K,
                                                                          const float* __restrict__ rw, float limit, bf16* __restrict__ y) {
  const GroupDesc d = g[blockIdx.y];
  const int n = blockIdx.x * GRP_WARPS + (threadIdx.x / 32);
  const int lane = threadIdx.x % 32;
  if (n >= I) return;
  const uint8_t* Ar = A + (size_t)d.row0 * K;
  const uint8_t* sar = sa + (size_t)d.row0 * (K / 32);
  float gate[8], up[8];
  gemv_fp4_warp(Ar, sar, d.w1, d.s1, d.n, K, n, lane, gate);
  gemv_fp4_warp(Ar, sar, d.w3, d.s3, d.n, K, n, lane, up);
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      if (m < d.n) {
        float gv = bf2f(f2bf(gate[m])), uv = bf2f(f2bf(up[m]));  // gemm output is bf16 → same formula as swiglu_route
        if (limit > 0.f) { uv = fminf(fmaxf(uv, -limit), limit); gv = fminf(gv, limit); }
        float v = gv / (1.f + expf(-gv)) * uv;
        v *= rw[d.row0 + m];
        y[(size_t)(d.row0 + m) * I + n] = f2bf(v);
      }
    }
  }
}

__global__ void __launch_bounds__(GRP_WARPS * 32) gemv_grouped_w2_kernel(const GroupDesc* __restrict__ g, const uint8_t* __restrict__ Yq,
                                                                         const uint8_t* __restrict__ ys, int dim, int I,
                                                                         bf16* __restrict__ eout) {
  const GroupDesc d = g[blockIdx.y];
  const int n = blockIdx.x * GRP_WARPS + (threadIdx.x / 32);
  const int lane = threadIdx.x % 32;
  if (n >= dim) return;
  float acc[8];
  gemv_fp4_warp(Yq + (size_t)d.row0 * I, ys + (size_t)d.row0 * (I / 32), d.w2, d.s2, d.n, I, n, lane, acc);
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < 8; ++m)
      if (m < d.n) eout[(size_t)(d.row0 + m) * dim + n] = f2bf(acc[m]);
  }
}

// ---- Tiled GEMM: 64×64 output tile, K in 32-element blocks. 256 threads = 16×16, 4×4 outputs per thread. -----------------------------
constexpr int BM = 64, BN = 64, BK = 32;

template <bool B_FP4>
__global__ void __launch_bounds__(256) gemm_bs_tile_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa,
                                                           const uint8_t* __restrict__ B, const uint8_t* __restrict__ sb,
                                                           int M, int N, int K, bf16* __restrict__ C, float* __restrict__ C32) {
  __shared__ float As[BM][BK + 1];
  __shared__ float Bs[BN][BK + 1];
  __shared__ float Sa[BM];
  __shared__ float Sb[BN];
  const int tm = blockIdx.y * BM, tn = blockIdx.x * BN;
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int nb = K / 32;
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  for (int b = 0; b < nb; ++b) {
    // A tile: 64 rows × 32 = 2048 elements, 256 threads → 8 each
    for (int e = threadIdx.x; e < BM * BK; e += 256) {
      int r = e / BK, c = e % BK;
      int gm = tm + r;
      As[r][c] = gm < M ? e4m3_to_f32(A[(size_t)gm * K + b * 32 + c]) : 0.f;
    }
    for (int e = threadIdx.x; e < BN * BK; e += 256) {
      int r = e / BK, c = e % BK;
      int gn = tn + r;
      float v = 0.f;
      if (gn < N) {
        if (B_FP4) {
          uint8_t byte = B[(size_t)gn * (K / 2) + b * 16 + (c >> 1)];
          v = kE2M1[(c & 1) ? (byte >> 4) : (byte & 0xF)];
        } else {
          v = e4m3_to_f32(B[(size_t)gn * K + b * 32 + c]);
        }
      }
      Bs[r][c] = v;
    }
    if (threadIdx.x < BM) {
      int gm = tm + threadIdx.x;
      Sa[threadIdx.x] = gm < M ? e8m0_to_f32(sa[(size_t)gm * nb + b]) : 0.f;
    } else if (threadIdx.x < BM + BN) {
      int gn = tn + threadIdx.x - BM;
      Sb[threadIdx.x - BM] = gn < N ? e8m0_to_f32(sb[(size_t)gn * nb + b]) : 0.f;
    }
    __syncthreads();
    float part[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) part[i][j] = 0.f;
#pragma unroll 8
    for (int k = 0; k < BK; ++k) {
      float a[4], w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[ty * 4 + i][k];
#pragma unroll
      for (int j = 0; j < 4; ++j) w[j] = Bs[tx * 4 + j][k];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) part[i][j] = fmaf(a[i], w[j], part[i][j]);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(part[i][j], Sa[ty * 4 + i] * Sb[tx * 4 + j], acc[i][j]);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int gm = tm + ty * 4 + i;
    if (gm >= M) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int gn = tn + tx * 4 + j;
      if (gn >= N) continue;
      if (C) C[(size_t)gm * N + gn] = f2bf(acc[i][j]);
      if (C32) C32[(size_t)gm * N + gn] = acc[i][j];
    }
  }
}

}  // namespace

static int g_force_path = 0;
void gemm_bs_force_path(int p) { g_force_path = p; }

void gemv_bf16_fp8_grouped(const bf16* A, int lda, const uint8_t* B, const uint8_t* sb, int M, int G, int R, int sub, bf16* C, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= 8 && sub % 32 == 0, "gemv_bf16_fp8_grouped shape");
  const int threads = 256, warps = G * R;
  gemv_bf16_fp8_grouped_kernel<8><<<(warps * 32 + threads - 1) / threads, threads, 0, st>>>(A, lda, B, sb, M, G, R, sub, C);
}

void gemv_grouped_w13(const GroupDesc* g, int ngroups, const uint8_t* A, const uint8_t* sa, int I, int K, const float* rw, float limit,
                      bf16* y, cudaStream_t st) {
  if (ngroups <= 0) return;
  dim3 grid((I + GRP_WARPS - 1) / GRP_WARPS, ngroups);
  gemv_grouped_w13_kernel<<<grid, GRP_WARPS * 32, 0, st>>>(g, A, sa, I, K, rw, limit, y);
}
void gemv_grouped_w2(const GroupDesc* g, int ngroups, const uint8_t* Yq, const uint8_t* ys, int dim, int I, bf16* eout, cudaStream_t st) {
  if (ngroups <= 0) return;
  dim3 grid((dim + GRP_WARPS - 1) / GRP_WARPS, ngroups);
  gemv_grouped_w2_kernel<<<grid, GRP_WARPS * 32, 0, st>>>(g, Yq, ys, dim, I, eout);
}

// Path selection in one place (shared by gemm_bs and the grouped prefill path).
int gemm_bs_route(bool b_fp4, int M, int K) {
  // D-1: fp4 weights with M > 8 go to block-scaled tensor cores (only with HIVE_MX_PREFILL=1 — test_mx3 checks it against gemm_bs_tc)
  static const bool mx_prefill = getenv("HIVE_MX_PREFILL") && atoi(getenv("HIVE_MX_PREFILL")) != 0;
  static const bool mx_gemm = getenv("HIVE_MX_GEMM") && atoi(getenv("HIVE_MX_GEMM")) != 0;  // tiled version (fp8 and fp4, K % 64)
  if (M > 8 && mx_gemm && K % 64 == 0 && g_force_path == 0) return kGemmMxTiled;
  if (M > 8 && b_fp4 && mx_prefill && g_force_path == 0) return kGemmMxRows;
  if (M > 8 && g_force_path != 1) return kGemmTc;
  if (M <= 8) return kGemmGemv;
  return kGemmTile;
}

void gemm_bs(const uint8_t* A, const uint8_t* sa, const uint8_t* B, const uint8_t* sb, bool b_fp4, int M, int N, int K,
             bf16* C, float* C32, cudaStream_t st) {
  HIVE_CHECK(K % 32 == 0, "K % 32");
  switch (gemm_bs_route(b_fp4, M, K)) {
    case kGemmMxTiled: gemm_bs_mx_tiled(A, sa, B, sb, b_fp4, M, N, K, C, C32, st); return;
    case kGemmMxRows: gemm_bs_mx(A, sa, B, sb, M, N, K, C, C32, st); return;
    case kGemmTc: gemm_bs_tc(A, sa, B, sb, b_fp4, M, N, K, C, C32, st); return;
    case kGemmGemv: {
      const int threads = 256;
      const int blocks = (N * 32 + threads - 1) / threads;
      if (b_fp4) gemv_bs_kernel<true, 8><<<blocks, threads, 0, st>>>(A, sa, B, sb, M, N, K, C, C32);
      else gemv_bs_kernel<false, 8><<<blocks, threads, 0, st>>>(A, sa, B, sb, M, N, K, C, C32);
      return;
    }
    default: break;
  }
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  if (b_fp4) gemm_bs_tile_kernel<true><<<grid, 256, 0, st>>>(A, sa, B, sb, M, N, K, C, C32);
  else gemm_bs_tile_kernel<false><<<grid, 256, 0, st>>>(A, sa, B, sb, M, N, K, C, C32);
}

void gemm_fp4_grouped(int route, const GroupDesc* tiles, int ntiles, int which, const uint8_t* const* a_base, const uint8_t* const* s_base,
                      const int32_t* rref, const uint8_t* A, const uint8_t* sa, int N, int K, bf16* C0, bf16* C1, cudaStream_t st) {
  if (ntiles <= 0) return;
  HIVE_CHECK(route == kGemmMxTiled || route == kGemmTc, "gemm_fp4_grouped route");
  if (route == kGemmMxTiled) gemm_bs_mx_tiled_grouped(tiles, ntiles, which, a_base, s_base, rref, A, sa, N, K, C0, C1, st);
  else gemm_bs_tc_grouped(tiles, ntiles, which, a_base, s_base, rref, A, sa, N, K, C0, C1, st);
}

}  // namespace hive::k
