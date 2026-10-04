// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Block-scaled GEMM v2 — tensor cores (WMMA bf16). On load, e4m3/e2m1 values are multiplied by their scales, unpacked to bf16 in smem, then bf16 MMA (fp32 accumulation).
//   a(e4m3)*2^k and w(e2m1)*2^k are exact in bf16, so sum (a*sa)(w*sb) = sa*sb*sum a*w — differs from the reference fp4_gemm/fp8_gemm only in accumulation order.
//   For prefill with M >= 16. Tile 128x128x32, 8 warps (each 32x64 = 2x4 WMMA 16x16), smem double buffering.
#include <mma.h>

#include <cmath>
#include <cstring>

#include "hive/kernels.h"

namespace hive::k {

namespace {
using namespace nvcuda;

constexpr int TBM = 128, TBN = 128, TBK = 32;
constexpr int TC_THREADS = 256;
constexpr int SMEM_LD = TBK + 8;  // padding against bank conflicts (40 bf16 = 80 B, keeps 16 B alignment)

// e2m1 -> bf16 bits (0, .5, 1, 1.5, 2, 3, 4, 6, and negatives)
__constant__ uint16_t kE2M1bf[16] = {0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
                                     0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0};
// e4m3 -> bf16 bit table (256 entries): copied into smem at block start
__constant__ uint16_t kE4M3bf[256];
static bool g_lut_ready = false;

// Multiply bf16 bits by 2^k (0 stays 0, exponent saturates). k = e8m0 code - 127
__device__ inline uint16_t bf16_scale2(uint16_t bits, int k) {
  if ((bits & 0x7FFF) == 0) return bits;
  int e = (int)((bits >> 7) & 0xFF) + k;
  if (e <= 0) return (uint16_t)(bits & 0x8000);     // underflow -> 0 (does not occur within the scale range used in practice)
  if (e >= 255) e = 254;
  return (uint16_t)((bits & 0x807F) | ((uint16_t)e << 7));
}

// A tile (128x32) load: 256 threads x 16 elements. Row r = t/2, column c0 = (t%2)*16. fp8 16 B -> bf16 bit table -> add the scale to the exponent
__device__ inline void load_a_tile(const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa, int M, int K, int nb, int m0, int kb,
                                   bf16* __restrict__ As, const uint16_t* __restrict__ lut) {
  const int t = threadIdx.x;
  const int r = t >> 1, c0 = (t & 1) * 16;
  const int gm = m0 + r;
  uint16_t* dst = reinterpret_cast<uint16_t*>(As + r * SMEM_LD + c0);
  if (gm < M) {
    const uint8_t* src = A + (size_t)gm * K + kb * 32 + c0;
    uint4 raw = *reinterpret_cast<const uint4*>(src);
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
    const int k = (int)sa[(size_t)gm * nb + kb] - 127;
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = bf16_scale2(lut[b[i]], k);
  } else {
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = 0;
  }
}
// B tile (128x32): row n, e4m3 32 B or e2m1 16 B
template <bool B_FP4>
__device__ inline void load_b_tile(const uint8_t* __restrict__ B, const uint8_t* __restrict__ sb, int N, int K, int nb, int n0, int kb,
                                   bf16* __restrict__ Bs, const uint16_t* __restrict__ lut) {
  const int t = threadIdx.x;
  const int r = t >> 1, c0 = (t & 1) * 16;
  const int gn = n0 + r;
  uint16_t* dst = reinterpret_cast<uint16_t*>(Bs + r * SMEM_LD + c0);
  if (gn < N) {
    const int k = (int)sb[(size_t)gn * nb + kb] - 127;
    if (B_FP4) {
      const uint8_t* src = B + (size_t)gn * (K / 2) + kb * 16 + c0 / 2;
      uint2 raw = *reinterpret_cast<const uint2*>(src);
      const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        dst[2 * i] = bf16_scale2(kE2M1bf[b[i] & 0xF], k);
        dst[2 * i + 1] = bf16_scale2(kE2M1bf[b[i] >> 4], k);
      }
    } else {
      const uint8_t* src = B + (size_t)gn * K + kb * 32 + c0;
      uint4 raw = *reinterpret_cast<const uint4*>(src);
      const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
      for (int i = 0; i < 16; ++i) dst[i] = bf16_scale2(lut[b[i]], k);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = 0;
  }
}

template <bool B_FP4>
__global__ void __launch_bounds__(TC_THREADS) gemm_bs_tc_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa,
                                                                const uint8_t* __restrict__ B, const uint8_t* __restrict__ sb, int M, int N,
                                                                int K, bf16* __restrict__ C, float* __restrict__ C32) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  bf16* As[2] = {reinterpret_cast<bf16*>(smem_raw), reinterpret_cast<bf16*>(smem_raw) + TBM * SMEM_LD};
  bf16* Bs[2] = {As[1] + TBM * SMEM_LD, As[1] + TBM * SMEM_LD + TBN * SMEM_LD};
  float* Cw = reinterpret_cast<float*>(Bs[1] + TBN * SMEM_LD);  // per-warp epilogue staging [8][16][68]
  __shared__ uint16_t lut[256];
  for (int i = threadIdx.x; i < 256; i += TC_THREADS) lut[i] = kE4M3bf[i];
  const int m0 = blockIdx.y * TBM, n0 = blockIdx.x * TBN;
  const int warp = threadIdx.x >> 5;
  const int wm = warp >> 1, wn = warp & 1;  // 4x2 warp grid: warp tile 32x64
  const int nb = K / 32;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  __syncthreads();  // lut
  load_a_tile(A, sa, M, K, nb, m0, 0, As[0], lut);
  load_b_tile<B_FP4>(B, sb, N, K, nb, n0, 0, Bs[0], lut);
  __syncthreads();
  for (int kb = 0; kb < nb; ++kb) {
    const int cur = kb & 1;
    if (kb + 1 < nb) {
      load_a_tile(A, sa, M, K, nb, m0, kb + 1, As[cur ^ 1], lut);
      load_b_tile<B_FP4>(B, sb, N, K, nb, n0, kb + 1, Bs[cur ^ 1], lut);
    }
#pragma unroll
    for (int kk = 0; kk < TBK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> fa[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> fb[4];
#pragma unroll
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(fa[i], As[cur] + (wm * 32 + i * 16) * SMEM_LD + kk, SMEM_LD);
#pragma unroll
      for (int j = 0; j < 4; ++j) wmma::load_matrix_sync(fb[j], Bs[cur] + (wn * 64 + j * 16) * SMEM_LD + kk, SMEM_LD);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
    __syncthreads();
  }
  // Epilogue: each warp writes 16 rows x 64 columns twice through its own staging (fp32 [16][68]) with bounds checks
  const int ldw = 64 + 4;
  float* stage = Cw + warp * 16 * ldw;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::store_matrix_sync(stage + j * 16, acc[i][j], ldw, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < 16 * 64; e += 32) {
      const int r = e / 64, cidx = e % 64;
      const int gm = m0 + wm * 32 + i * 16 + r, gn = n0 + wn * 64 + cidx;
      if (gm >= M || gn >= N) continue;
      const float v = stage[r * ldw + cidx];
      if (C) C[(size_t)gm * N + gn] = f2bf(v);
      if (C32) C32[(size_t)gm * N + gn] = v;
    }
    __syncwarp();
  }
}

constexpr size_t TC_SMEM = (size_t)2 * TBM * SMEM_LD * 2 + (size_t)2 * TBN * SMEM_LD * 2 + (size_t)8 * 16 * 68 * 4;

// Grouped variant (HIVE_GROUPED_PREFILL, when HIVE_MX_GEMM is off): **same body** as gemm_bs_tc_kernel<true>; the only differences are that row tiles come from a table (tiles[blockIdx.y]),
//   A row addresses go through a row map (rref) and blockIdx.z selects w1/w3 — row-independent WMMA with the same k order, so bit-identical to calling gemm_bs_tc per expert.
__device__ inline void load_a_tile_ptr(const uint8_t* __restrict__ a_row, const uint8_t* __restrict__ s_row, bool ok, int kb, bf16* __restrict__ As,
                                       const uint16_t* __restrict__ lut) {
  const int t = threadIdx.x;
  const int r = t >> 1, c0 = (t & 1) * 16;
  uint16_t* dst = reinterpret_cast<uint16_t*>(As + r * SMEM_LD + c0);
  if (ok) {
    const uint8_t* src = a_row + kb * 32 + c0;
    uint4 raw = *reinterpret_cast<const uint4*>(src);
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
    const int k = (int)s_row[kb] - 127;
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = bf16_scale2(lut[b[i]], k);
  } else {
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = 0;
  }
}

__global__ void __launch_bounds__(TC_THREADS) gemm_bs_tc_grouped_kernel(const GroupDesc* __restrict__ tiles, int which, const uint8_t* const* __restrict__ a_base,
                                                                        const uint8_t* const* __restrict__ s_base, const int32_t* __restrict__ rref,
                                                                        const uint8_t* __restrict__ A0, const uint8_t* __restrict__ sa0, int N, int K,
                                                                        bf16* __restrict__ C0, bf16* __restrict__ C1) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  bf16* As[2] = {reinterpret_cast<bf16*>(smem_raw), reinterpret_cast<bf16*>(smem_raw) + TBM * SMEM_LD};
  bf16* Bs[2] = {As[1] + TBM * SMEM_LD, As[1] + TBM * SMEM_LD + TBN * SMEM_LD};
  float* Cw = reinterpret_cast<float*>(Bs[1] + TBN * SMEM_LD);
  __shared__ uint16_t lut[256];
  for (int i = threadIdx.x; i < 256; i += TC_THREADS) lut[i] = kE4M3bf[i];
  const GroupDesc d = tiles[blockIdx.y];
  const int z = blockIdx.z;
  const uint8_t* __restrict__ B = which == 0 ? (z ? d.w3 : d.w1) : d.w2;
  const uint8_t* __restrict__ sb = which == 0 ? (z ? d.s3 : d.s1) : d.s2;
  bf16* __restrict__ C = z ? C1 : C0;
  const int M = d.n;
  const int n0 = blockIdx.x * TBN;
  const int warp = threadIdx.x >> 5;
  const int wm = warp >> 1, wn = warp & 1;
  const int nb = K / 32;
  // Resolve this thread's A row (r = t/2 in load_a_tile) address once
  const int a_lr = threadIdx.x >> 1;
  const bool a_ok = a_lr < M;
  const uint8_t* a_ptr = nullptr;
  const uint8_t* s_ptr = nullptr;
  if (a_ok) {
    const int gr = d.row0 + a_lr;
    if (rref) { const int v = rref[gr]; a_ptr = a_base[v >> 24] + (size_t)(v & 0xFFFFFF) * K; s_ptr = s_base[v >> 24] + (size_t)(v & 0xFFFFFF) * nb; }
    else { a_ptr = A0 + (size_t)gr * K; s_ptr = sa0 + (size_t)gr * nb; }
  }
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  __syncthreads();  // lut
  load_a_tile_ptr(a_ptr, s_ptr, a_ok, 0, As[0], lut);
  load_b_tile<true>(B, sb, N, K, nb, n0, 0, Bs[0], lut);
  __syncthreads();
  for (int kb = 0; kb < nb; ++kb) {
    const int cur = kb & 1;
    if (kb + 1 < nb) {
      load_a_tile_ptr(a_ptr, s_ptr, a_ok, kb + 1, As[cur ^ 1], lut);
      load_b_tile<true>(B, sb, N, K, nb, n0, kb + 1, Bs[cur ^ 1], lut);
    }
#pragma unroll
    for (int kk = 0; kk < TBK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> fa[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> fb[4];
#pragma unroll
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(fa[i], As[cur] + (wm * 32 + i * 16) * SMEM_LD + kk, SMEM_LD);
#pragma unroll
      for (int j = 0; j < 4; ++j) wmma::load_matrix_sync(fb[j], Bs[cur] + (wn * 64 + j * 16) * SMEM_LD + kk, SMEM_LD);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
    __syncthreads();
  }
  const int ldw = 64 + 4;
  float* stage = Cw + warp * 16 * ldw;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::store_matrix_sync(stage + j * 16, acc[i][j], ldw, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < 16 * 64; e += 32) {
      const int r = e / 64, cidx = e % 64;
      const int lm = wm * 32 + i * 16 + r, gn = n0 + wn * 64 + cidx;
      if (lm >= M || gn >= N) continue;
      C[(size_t)(d.row0 + lm) * N + gn] = f2bf(stage[r * ldw + cidx]);
    }
    __syncwarp();
  }
}

// A tile (plain bf16, row stride lda) — wo_a prefill: activations are bf16 and only weights are fp8 (dequantisation is exact, so values match the cuBLAS bf16 path)
__device__ inline void load_a_tile_bf16(const bf16* __restrict__ A, int lda, int M, int m0, int kb, bf16* __restrict__ As) {
  const int t = threadIdx.x;
  const int r = t >> 1, c0 = (t & 1) * 16;
  const int gm = m0 + r;
  uint4* dst = reinterpret_cast<uint4*>(As + r * SMEM_LD + c0);
  if (gm < M) {
    const uint4* src = reinterpret_cast<const uint4*>(A + (size_t)gm * lda + kb * 32 + c0);
    dst[0] = src[0]; dst[1] = src[1];
  } else {
    dst[0] = make_uint4(0, 0, 0, 0); dst[1] = dst[0];
  }
}

__global__ void __launch_bounds__(TC_THREADS) gemm_bf16a_fp8b_tc_kernel(const bf16* __restrict__ A, int lda, const uint8_t* __restrict__ B,
                                                                        const uint8_t* __restrict__ sb, int M, int N, int K, bf16* __restrict__ C,
                                                                        int ldc) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  bf16* As[2] = {reinterpret_cast<bf16*>(smem_raw), reinterpret_cast<bf16*>(smem_raw) + TBM * SMEM_LD};
  bf16* Bs[2] = {As[1] + TBM * SMEM_LD, As[1] + TBM * SMEM_LD + TBN * SMEM_LD};
  float* Cw = reinterpret_cast<float*>(Bs[1] + TBN * SMEM_LD);
  __shared__ uint16_t lut[256];
  for (int i = threadIdx.x; i < 256; i += TC_THREADS) lut[i] = kE4M3bf[i];
  const int m0 = blockIdx.y * TBM, n0 = blockIdx.x * TBN;
  const int warp = threadIdx.x >> 5;
  const int wm = warp >> 1, wn = warp & 1;
  const int nb = K / 32;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.f);
  __syncthreads();
  load_a_tile_bf16(A, lda, M, m0, 0, As[0]);
  load_b_tile<false>(B, sb, N, K, nb, n0, 0, Bs[0], lut);
  __syncthreads();
  for (int kb = 0; kb < nb; ++kb) {
    const int cur = kb & 1;
    if (kb + 1 < nb) {
      load_a_tile_bf16(A, lda, M, m0, kb + 1, As[cur ^ 1]);
      load_b_tile<false>(B, sb, N, K, nb, n0, kb + 1, Bs[cur ^ 1], lut);
    }
#pragma unroll
    for (int kk = 0; kk < TBK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> fa[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> fb[4];
#pragma unroll
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(fa[i], As[cur] + (wm * 32 + i * 16) * SMEM_LD + kk, SMEM_LD);
#pragma unroll
      for (int j = 0; j < 4; ++j) wmma::load_matrix_sync(fb[j], Bs[cur] + (wn * 64 + j * 16) * SMEM_LD + kk, SMEM_LD);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
    __syncthreads();
  }
  const int ldw = 64 + 4;
  float* stage = Cw + warp * 16 * ldw;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::store_matrix_sync(stage + j * 16, acc[i][j], ldw, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < 16 * 64; e += 32) {
      const int r = e / 64, cidx = e % 64;
      const int gm = m0 + wm * 32 + i * 16 + r, gn = n0 + wn * 64 + cidx;
      if (gm >= M || gn >= N) continue;
      C[(size_t)gm * ldc + gn] = f2bf(stage[r * ldw + cidx]);
    }
    __syncwarp();
  }
}

}  // namespace

void gemm_bf16a_fp8b_tc(const bf16* A, int lda, const uint8_t* B, const uint8_t* sb, int M, int N, int K, bf16* C, int ldc, cudaStream_t st) {
  HIVE_CHECK(K % 32 == 0 && lda % 8 == 0, "gemm_bf16a_fp8b_tc shape");
  static bool configured = false;
  if (!configured) {
    {  // e4m3 -> bf16 bit table (same table as gemm_bs_tc — built here too so it is filled whichever is called first)
      uint16_t tab[256];
      for (int i = 0; i < 256; ++i) { float v = e4m3_to_f32((uint8_t)i); bf16 b = f2bf(std::isnan(v) ? 0.f : v); memcpy(&tab[i], &b, 2); }
      CUDA_CHECK(cudaMemcpyToSymbol(kE4M3bf, tab, sizeof(tab)));
    }
    CUDA_CHECK(cudaFuncSetAttribute(gemm_bf16a_fp8b_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM));
    configured = true;
  }
  dim3 grid((N + TBN - 1) / TBN, (M + TBM - 1) / TBM);
  gemm_bf16a_fp8b_tc_kernel<<<grid, TC_THREADS, TC_SMEM, st>>>(A, lda, B, sb, M, N, K, C, ldc);
}

void gemm_bs_tc(const uint8_t* A, const uint8_t* sa, const uint8_t* B, const uint8_t* sb, bool b_fp4, int M, int N, int K, bf16* C,
                float* C32, cudaStream_t st) {
  HIVE_CHECK(K % 32 == 0, "K % 32");
  static bool configured = false;
  if (!configured) {
    // e4m3 -> bf16 bit table (built on the host, stored in constant memory)
    uint16_t tab[256];
    for (int i = 0; i < 256; ++i) {
      float v = e4m3_to_f32((uint8_t)i);
      bf16 b = f2bf(std::isnan(v) ? 0.f : v);
      memcpy(&tab[i], &b, 2);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(kE4M3bf, tab, sizeof(tab)));
    CUDA_CHECK(cudaFuncSetAttribute(gemm_bs_tc_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM));
    CUDA_CHECK(cudaFuncSetAttribute(gemm_bs_tc_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM));
    configured = true;
  }
  dim3 grid((N + TBN - 1) / TBN, (M + TBM - 1) / TBM);
  if (b_fp4) gemm_bs_tc_kernel<true><<<grid, TC_THREADS, TC_SMEM, st>>>(A, sa, B, sb, M, N, K, C, C32);
  else gemm_bs_tc_kernel<false><<<grid, TC_THREADS, TC_SMEM, st>>>(A, sa, B, sb, M, N, K, C, C32);
}

void gemm_bs_tc_grouped(const GroupDesc* tiles, int ntiles, int which, const uint8_t* const* a_base, const uint8_t* const* s_base, const int32_t* rref,
                        const uint8_t* A, const uint8_t* sa, int N, int K, bf16* C0, bf16* C1, cudaStream_t st) {
  if (ntiles <= 0) return;
  HIVE_CHECK(K % 32 == 0 && ntiles <= 65535 && (which == 0 ? C1 != nullptr : true), "gemm_bs_tc_grouped shape");
  static bool configured = false;
  if (!configured) {
    {  // e4m3 -> bf16 bit table (same table as gemm_bs_tc — built here too so it is filled whichever is called first)
      uint16_t tab[256];
      for (int i = 0; i < 256; ++i) { float v = e4m3_to_f32((uint8_t)i); bf16 b = f2bf(std::isnan(v) ? 0.f : v); memcpy(&tab[i], &b, 2); }
      CUDA_CHECK(cudaMemcpyToSymbol(kE4M3bf, tab, sizeof(tab)));
    }
    CUDA_CHECK(cudaFuncSetAttribute(gemm_bs_tc_grouped_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM));
    configured = true;
  }
  dim3 grid((N + TBN - 1) / TBN, ntiles, which == 0 ? 2 : 1);
  gemm_bs_tc_grouped_kernel<<<grid, TC_THREADS, TC_SMEM, st>>>(tiles, which, a_base, s_base, rref, A, sa, N, K, C0, C1);
}

}  // namespace hive::k
