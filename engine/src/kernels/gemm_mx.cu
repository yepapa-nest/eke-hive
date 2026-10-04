// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Block-scaled tensor-core grouped GEMM (sm_120a: mma.sync kind::mxf8f6f4.block_scale) — for decode resident experts and short prefills.
//   D[16×8] += Σ_k (A[m,k]·2^sfa[m]) · (B[k,n]·2^sfb[n]),  A = e4m3 (activations, fp8 bytes as is), B = e2m1 (experts, nibble → byte container bits[2:6])
//   · scales = ue8m0, scale_vec::1X = one mma covers one K=32 block (same as our block-scale unit) → algebraically identical to the reference formula; only the fp32 accumulation order differs.
// Fragment layout (CuTe SM120_16x8x32_TN_VS traits, verified):
//   A: a0 = A[t/4][4q..4q+3], a1 = A[t/4+8][…], a2 = A[t/4][16+4q..], a3 = A[t/4+8][16+4q..]  (q = t%4, 4 bytes = 1 register, lower k in the lower byte)
//   B: b0 = B[k=4q..4q+3][n=t/4], b1 = B[16+4q..][t/4]   C/D: d0,d1 = D[t/4][2q,2q+1], d2,d3 = D[t/4+8][2q,2q+1]
//   sfa: thread t supplies the scale byte of row (t>>2)+8·(t&1) · sfb: the scale byte of column t>>2 (bid/tid = 0)
// k may be permuted freely: within one mma A and B only need to use the same actual k → thread q takes actual k [8q, 8q+8) within the block (contiguous loads of 4 B of B and 8 B of A).
#include "hive/kernels.h"
#include "hive/kv_pack.h"

namespace hive::k {

namespace {

__device__ __forceinline__ void mma_mx_e4m3_e2m1(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sfa, uint32_t sfb) {
  const uint16_t zero = 0;
  const uint32_t ar0 = a[0], ar1 = a[1], ar2 = a[2], ar3 = a[3], br0 = b[0], br1 = b[1];
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(ar0), "r"(ar1), "r"(ar2), "r"(ar3), "r"(br0), "r"(br1), "r"(sfa), "h"(zero), "h"(zero), "r"(sfb), "h"(zero), "h"(zero));
}

// e4m3 × e4m3 (dense fp8 weights: wq_a/wq_b/wkv/wo_b/shared experts) — for the tiled GEMM
__device__ __forceinline__ void mma_mx_e4m3_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sfa, uint32_t sfb) {
  const uint16_t zero = 0;
  const uint32_t ar0 = a[0], ar1 = a[1], ar2 = a[2], ar3 = a[3], br0 = b[0], br1 = b[1];
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(ar0), "r"(ar1), "r"(ar2), "r"(ar3), "r"(br0), "r"(br1), "r"(sfa), "h"(zero), "h"(zero), "r"(sfb), "h"(zero), "h"(zero));
}

// e2m1 × e2m1 (both with ue8m0 block scales) — for indexer scores (query and key both fp4, block 32)
__device__ __forceinline__ void mma_mx_e2m1_e2m1(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sfa, uint32_t sfb) {
  const uint16_t zero = 0;
  const uint32_t ar0 = a[0], ar1 = a[1], ar2 = a[2], ar3 = a[3], br0 = b[0], br1 = b[1];
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(ar0), "r"(ar1), "r"(ar2), "r"(ar3), "r"(br0), "r"(br1), "r"(sfa), "h"(zero), "h"(zero), "r"(sfb), "h"(zero), "h"(zero));
}

// 4 nibbles (packed 2 bytes P = p0 | p1<<8, lower k in the lower nibble) → 4 byte containers (bits 2..5)
__device__ __forceinline__ uint32_t e2m1x4_to_bytes(uint32_t P) {
  return ((P & 0x000Fu) << 2) | ((P & 0x00F0u) << 6) | ((P & 0x0F00u) << 10) | ((P & 0xF000u) << 14);
}

constexpr int MX_WARPS = 4;            // block = 4 warps
constexpr int TILES_PER_WARP = 2;      // n tiles (8 columns each) per warp
constexpr int KU = 4;                  // k-step unroll (overlaps loads)

// One warp: rows <= 8 (the upper 8 rows of A are 0) × T n-tiles × NM matrices (2 for w13: w1 and w3, 1 for w2). Returns the accumulator fragments.
// rows: group row → row index into A (nullptr = consecutive rows of A from row0 — a gathered buffer). Fused decode reads xq/xs directly through the row map without a gather.
template <int NM, int T>
__device__ __forceinline__ void mx_warp_tiles(const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa, const int32_t* __restrict__ rows, int row0,
                                              int nrows, int K, const uint8_t* const (&W)[NM], const uint8_t* const (&SW)[NM], int n0, int N,
                                              float (&acc)[NM][T][4]) {
  const int lane = threadIdx.x & 31, q = lane & 3, r = lane >> 2;
  const int nb = K / 32;
  const int am = r;                      // a0/a2 row
  const bool have_a = am < nrows;
#pragma unroll
  for (int j = 0; j < NM; ++j)
#pragma unroll
    for (int i = 0; i < T; ++i) { acc[j][i][0] = acc[j][i][1] = acc[j][i][2] = acc[j][i][3] = 0.f; }
  const int sfa_row = r + 8 * (lane & 1);  // row supplying the scale (any value if >= nrows)
  const int sfa_src = sfa_row < nrows ? sfa_row : 0;
  const uint8_t* sa_row = sa + (size_t)(rows ? rows[row0 + sfa_src] : row0 + sfa_src) * nb;
  const uint8_t* a_row = A + (size_t)(rows ? rows[row0 + (have_a ? am : 0)] : row0 + am) * K;
  for (int kb0 = 0; kb0 < nb; kb0 += KU) {
    uint32_t a[KU][4];
    uint32_t sfa[KU];
    uint32_t b[KU][NM][T][2];
    uint32_t sfb[KU][NM][T];
#pragma unroll
    for (int u = 0; u < KU; ++u) {
      const int kb = kb0 + u;
      const bool ok = kb < nb;
      if (ok && have_a) {
        const uint2 av = *reinterpret_cast<const uint2*>(a_row + (size_t)kb * 32 + 8 * q);
        a[u][0] = av.x; a[u][2] = av.y;
      } else { a[u][0] = 0u; a[u][2] = 0u; }
      a[u][1] = 0u; a[u][3] = 0u;  // rows 8..15 absent
      sfa[u] = ok ? (uint32_t)sa_row[kb] : 127u;
#pragma unroll
      for (int j = 0; j < NM; ++j)
#pragma unroll
        for (int i = 0; i < T; ++i) {
          const int n = n0 + i * 8 + r;
          if (ok && n < N) {
            const uint32_t P = *reinterpret_cast<const uint32_t*>(W[j] + (size_t)n * (K / 2) + (size_t)kb * 16 + 4 * q);
            b[u][j][i][0] = e2m1x4_to_bytes(P & 0xFFFFu);
            b[u][j][i][1] = e2m1x4_to_bytes(P >> 16);
            sfb[u][j][i] = (uint32_t)SW[j][(size_t)n * nb + kb];
          } else { b[u][j][i][0] = b[u][j][i][1] = 0u; sfb[u][j][i] = 127u; }
        }
    }
#pragma unroll
    for (int u = 0; u < KU; ++u)
#pragma unroll
      for (int j = 0; j < NM; ++j)
#pragma unroll
        for (int i = 0; i < T; ++i) mma_mx_e4m3_e2m1(acc[j][i], a[u], b[u][j][i], sfa[u], sfb[u][j][i]);
  }
}

// w1/w3 + swiglu → y[row, n] bf16. Block = (bundle of n tiles, group). Warp w handles tiles [tile0 + w·T, +T).
// With yq/ys, each block (64 columns = two 32-element blocks) quantizes its own y values per 32-element block (same formula as act_quant_fp8) and writes them directly — no separate act_quant launch.
__global__ void __launch_bounds__(MX_WARPS * 32) mx_grouped_w13_kernel(const GroupDesc* __restrict__ g, const uint8_t* __restrict__ A,
                                                                       const uint8_t* __restrict__ sa, const int32_t* __restrict__ rows, int I, int K,
                                                                       const float* __restrict__ rw, float limit, bf16* __restrict__ y,
                                                                       uint8_t* __restrict__ yq, uint8_t* __restrict__ ys) {
  constexpr int BCOLS = MX_WARPS * TILES_PER_WARP * 8;  // 64
  __shared__ bf16 ysm[8][BCOLS];
  const GroupDesc d = g[blockIdx.y];
  if (d.n <= 0) return;  // step graph: empty group slot of the fixed grid (d is uniform across the block — eager calls only pass n >= 1)
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, q = lane & 3, r = lane >> 2;
  const int tile0 = (blockIdx.x * MX_WARPS + warp) * TILES_PER_WARP;
  const int n0 = tile0 * 8;
  const int nb0 = blockIdx.x * BCOLS;  // first column of the block
  if (nb0 >= I) return;
  const uint8_t* const W[2] = {d.w1, d.w3};
  const uint8_t* const SW[2] = {d.s1, d.s3};
  float acc[2][TILES_PER_WARP][4];
  if (n0 < I) mx_warp_tiles<2, TILES_PER_WARP>(A, sa, rows, d.row0, d.n, K, W, SW, n0, I, acc);
  // Epilogue: row r (= t/4), columns n0 + i·8 + 2q, +1 (d0, d1). gemm output bf16 → swiglu → ·rw → bf16 (same formula as swiglu_route)
  if (n0 < I && r < d.n) {
    const float rwv = rw[d.row0 + r];
#pragma unroll
    for (int i = 0; i < TILES_PER_WARP; ++i) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int n = n0 + i * 8 + 2 * q + c;
        if (n >= I) continue;
        float gv = bf2f(f2bf(acc[0][i][c])), uv = bf2f(f2bf(acc[1][i][c]));
        if (limit > 0.f) { uv = fminf(fmaxf(uv, -limit), limit); gv = fminf(gv, limit); }
        float v = gv / (1.f + expf(-gv)) * uv * rwv;
        const bf16 vb = f2bf(v);
        y[(size_t)(d.row0 + r) * I + n] = vb;
        ysm[r][n - nb0] = vb;
      }
    }
  }
  if (!yq) return;
  __syncthreads();
  if (threadIdx.x < d.n * (BCOLS / 32)) {  // one (row, 32-element block) per thread
    const int rr = threadIdx.x / (BCOLS / 32), bb = threadIdx.x % (BCOLS / 32);
    const int n_start = nb0 + bb * 32;
    if (n_start < I) {
      float v[32];
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 32; ++i) { v[i] = bf2f(ysm[rr][bb * 32 + i]); amax = fmaxf(amax, fabsf(v[i])); }
      amax = fmaxf(amax, 1e-4f);
      const uint8_t code = f32_ceil_pow2_e8m0(amax * (1.0f / 448.0f));
      const float sc = e8m0_to_f32(code);
      uint8_t* qp = yq + (size_t)(d.row0 + rr) * I + n_start;
#pragma unroll
      for (int i = 0; i < 32; ++i) qp[i] = f32_to_e4m3(fminf(fmaxf(v[i] / sc, -448.f), 448.f));
      ys[(size_t)(d.row0 + rr) * (I / 32) + n_start / 32] = code;
    }
  }
}

// w2: eout[row, n] = bf16(Σ_k yq·w2)
__global__ void __launch_bounds__(MX_WARPS * 32) mx_grouped_w2_kernel(const GroupDesc* __restrict__ g, const uint8_t* __restrict__ Yq,
                                                                      const uint8_t* __restrict__ ys, int dim, int I, bf16* __restrict__ eout) {
  const GroupDesc d = g[blockIdx.y];
  if (d.n <= 0) return;  // step graph: empty group slot (as above)
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, q = lane & 3, r = lane >> 2;
  const int tile0 = (blockIdx.x * MX_WARPS + warp) * TILES_PER_WARP;
  const int n0 = tile0 * 8;
  if (n0 >= dim) return;
  const uint8_t* const W[1] = {d.w2};
  const uint8_t* const SW[1] = {d.s2};
  float acc[1][TILES_PER_WARP][4];
  mx_warp_tiles<1, TILES_PER_WARP>(Yq, ys, nullptr, d.row0, d.n, I, W, SW, n0, dim, acc);
  if (r < d.n) {
#pragma unroll
    for (int i = 0; i < TILES_PER_WARP; ++i)
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int n = n0 + i * 8 + 2 * q + c;
        if (n < dim) eout[(size_t)(d.row0 + r) * dim + n] = f2bf(acc[0][i][c]);
      }
  }
}

// D-1: block-scaled GEMM with arbitrary row count for prefill — C[M,N] = A[M,K](e4m3, sa ue8m0/32) · B[N,K](e2m1 nibbles, sb ue8m0/32)ᵀ.
//   The default prefill expert path (gemm_bs → gemm_bs_tc) expands weights and activations to bf16 and multiplies with WMMA bf16. Products of
//   e2m1/e4m3 values and 2^k scales are exact in fp32, so the two paths differ only in fp32 accumulation order (same reasoning as the
//   mx_warp_tiles header comment). Warp = all 16 rows (upper 8 rows a0/a2, lower 8 rows a1/a3) × T tiles.
//   Block = 4 warps on different 16-row groups with the same bundle of n tiles (B reused via L1/L2). Enable = HIVE_MX_PREFILL=1 (default off pending validation — test_mx3).
template <int T>
__global__ void __launch_bounds__(MX_WARPS * 32) mx_gemm_rows_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa, int M, int K,
                                                                     const uint8_t* __restrict__ W, const uint8_t* __restrict__ SW, int N,
                                                                     bf16* __restrict__ C, float* __restrict__ C32) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, q = lane & 3, r = lane >> 2;
  const int m0 = (blockIdx.y * MX_WARPS + warp) * 16;
  const int n0 = blockIdx.x * T * 8;
  if (m0 >= M || n0 >= N) return;
  const int nb = K / 32;
  const int ra = m0 + r, rb = m0 + r + 8;           // a0/a2 row, a1/a3 row
  const bool ha = ra < M, hb = rb < M;
  const int sfa_row = m0 + r + 8 * (lane & 1);      // row supplying the scale (any row if out of range — its results are not written)
  const uint8_t* sa_row = sa + (size_t)(sfa_row < M ? sfa_row : m0) * nb;
  const uint8_t* a_ra = A + (size_t)(ha ? ra : m0) * K;
  const uint8_t* a_rb = A + (size_t)(hb ? rb : m0) * K;
  float acc[T][4];
#pragma unroll
  for (int i = 0; i < T; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  for (int kb0 = 0; kb0 < nb; kb0 += KU) {
    uint32_t a[KU][4], sfa[KU], b[KU][T][2], sfb[KU][T];
#pragma unroll
    for (int u = 0; u < KU; ++u) {
      const int kb = kb0 + u;
      const bool ok = kb < nb;
      if (ok && ha) { const uint2 v = *reinterpret_cast<const uint2*>(a_ra + (size_t)kb * 32 + 8 * q); a[u][0] = v.x; a[u][2] = v.y; }
      else { a[u][0] = 0u; a[u][2] = 0u; }
      if (ok && hb) { const uint2 v = *reinterpret_cast<const uint2*>(a_rb + (size_t)kb * 32 + 8 * q); a[u][1] = v.x; a[u][3] = v.y; }
      else { a[u][1] = 0u; a[u][3] = 0u; }
      sfa[u] = ok ? (uint32_t)sa_row[kb] : 127u;
#pragma unroll
      for (int i = 0; i < T; ++i) {
        const int n = n0 + i * 8 + r;
        if (ok && n < N) {
          const uint32_t P = *reinterpret_cast<const uint32_t*>(W + (size_t)n * (K / 2) + (size_t)kb * 16 + 4 * q);
          b[u][i][0] = e2m1x4_to_bytes(P & 0xFFFFu);
          b[u][i][1] = e2m1x4_to_bytes(P >> 16);
          sfb[u][i] = (uint32_t)SW[(size_t)n * nb + kb];
        } else { b[u][i][0] = b[u][i][1] = 0u; sfb[u][i] = 127u; }
      }
    }
#pragma unroll
    for (int u = 0; u < KU; ++u)
#pragma unroll
      for (int i = 0; i < T; ++i) mma_mx_e4m3_e2m1(acc[i], a[u], b[u][i], sfa[u], sfb[u][i]);
  }
  // d0,d1 = (row r, columns 2q, 2q+1) · d2,d3 = (row r+8, same columns)
#pragma unroll
  for (int i = 0; i < T; ++i)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int n = n0 + i * 8 + 2 * q + c;
      if (n >= N) continue;
      if (ha) { const float v = acc[i][c]; if (C) C[(size_t)ra * N + n] = f2bf(v); if (C32) C32[(size_t)ra * N + n] = v; }
      if (hb) { const float v = acc[i][2 + c]; if (C) C[(size_t)rb * N + n] = f2bf(v); if (C32) C32[(size_t)rb * N + n] = v; }
    }
}

// D-2: tensor-core indexer scores. score[m, t] = bf16( Σ_h bf16( relu( bf16(q[m,h]·k[t]) ) · w[m,h] ) ), -inf if t >= visible[m].
//   Query and key are both kvp::IDX_ROW (80 B) packed rows (scale row[b] ue8m0, nibbles row[16 + d/2]) — query row = (m, h). One mma = one
//   32-element block (scale_vec::1X), and e2m1·ue8m0 products are exact in fp32 → differs from the CUDA-core version (indexer_scores_kernel:
//   round-tripped bf16 query × dequantized key fp32 FMA) only in fp32 accumulation order.
//   Warp = 16 query rows × 8·NT keys. 4 block mmas per head → the epilogue (per-head bf16 rounding, relu, weight) follows the same order as the CUDA-core version.
template <int NT>
__global__ void __launch_bounds__(MX_WARPS * 32) idx_scores_tc_kernel(const uint8_t* __restrict__ qp, const uint8_t* __restrict__ kp,
                                                                      const bf16* __restrict__ w, int M, int Hi, int T,
                                                                      const int32_t* __restrict__ visible, bf16* __restrict__ score) {
  constexpr int ROW = kvp::IDX_ROW, HDR = kvp::IDX_HDR, NBLK = kvp::IDX_D / 32;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, q = lane & 3, r = lane >> 2;
  const int m0 = (blockIdx.y * MX_WARPS + warp) * 16;
  const int t0 = blockIdx.x * NT * 8;
  if (m0 >= M || t0 >= T) return;
  const int ra = m0 + r, rb = m0 + r + 8;
  const bool ha = ra < M, hb = rb < M;
  const int sfr = m0 + r + 8 * (lane & 1);
  const int sfr_c = sfr < M ? sfr : m0;
  float total[NT][4];
#pragma unroll
  for (int i = 0; i < NT; ++i) total[i][0] = total[i][1] = total[i][2] = total[i][3] = 0.f;
  for (int h = 0; h < Hi; ++h) {
    const uint8_t* qa = qp + ((size_t)(ha ? ra : m0) * Hi + h) * ROW;
    const uint8_t* qb = qp + ((size_t)(hb ? rb : m0) * Hi + h) * ROW;
    const uint8_t* qs = qp + ((size_t)sfr_c * Hi + h) * ROW;
    float acc[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
    for (int b = 0; b < NBLK; ++b) {
      uint32_t a[4];
      if (ha) { const uint32_t P = *reinterpret_cast<const uint32_t*>(qa + HDR + b * 16 + 4 * q); a[0] = e2m1x4_to_bytes(P & 0xFFFFu); a[2] = e2m1x4_to_bytes(P >> 16); }
      else { a[0] = a[2] = 0u; }
      if (hb) { const uint32_t P = *reinterpret_cast<const uint32_t*>(qb + HDR + b * 16 + 4 * q); a[1] = e2m1x4_to_bytes(P & 0xFFFFu); a[3] = e2m1x4_to_bytes(P >> 16); }
      else { a[1] = a[3] = 0u; }
      const uint32_t sfa = (uint32_t)qs[b];
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const int t = t0 + i * 8 + r;
        uint32_t bb[2];
        uint32_t sfb = 127u;
        if (t < T) {
          const uint8_t* kr = kp + (size_t)t * ROW;
          const uint32_t P = *reinterpret_cast<const uint32_t*>(kr + HDR + b * 16 + 4 * q);
          bb[0] = e2m1x4_to_bytes(P & 0xFFFFu); bb[1] = e2m1x4_to_bytes(P >> 16);
          sfb = (uint32_t)kr[b];
        } else { bb[0] = bb[1] = 0u; }
        mma_mx_e2m1_e2m1(acc[i], a, bb, sfa, sfb);
      }
    }
    const float wa = ha ? bf2f(w[(size_t)ra * Hi + h]) : 0.f, wb = hb ? bf2f(w[(size_t)rb * Hi + h]) : 0.f;
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        float sb = bf2f(f2bf(acc[i][c]));      // einsum output is bf16
        sb = fmaxf(sb, 0.f);                   // relu
        sb = bf2f(f2bf(sb * (c < 2 ? wa : wb)));
        total[i][c] += sb;
      }
  }
  const int va = ha ? visible[ra] : 0, vb = hb ? visible[rb] : 0;
#pragma unroll
  for (int i = 0; i < NT; ++i)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int t = t0 + i * 8 + 2 * q + c;
      if (t >= T) continue;
      if (ha) score[(size_t)ra * T + t] = t < va ? f2bf(total[i][c]) : f2bf(-INFINITY);
      if (hb) score[(size_t)rb * T + t] = t < vb ? f2bf(total[i][2 + c]) : f2bf(-INFINITY);
    }
}

// Tensor-core indexer scores over a candidate pool (the e2m1×e2m1 version of indexer_scores_cand). The full-T version above uses warp rows = 16 queries, so
//   queries share the keys (B columns); with a candidate pool each query has its own key list, so that grouping is impossible → rows = 16 **heads** (of one
//   query), columns = 8 of that query's candidates. The head tile (16) is the outer loop.
//   · Warp = 32 candidates (4 n tiles), block = 4 warps = 128 candidates (same grid as IDX_T of the CUDA-core version). B (key fragments, scales) is loaded
//     into registers once and reused for every head tile.
//   · Candidate index ci → block cand[m][ci / bs] (ascending, -1 = none) → position t = blk·bs + ci % bs. -inf if t >= min(visible, T_rows) (key not read).
//   · Epilogue order matches the CUDA-core version (idx_score_one): per head bf16 rounding → relu → × w (bf16 rounding) → fp32 sum over heads 0→Hi-1 ascending.
//     Heads are spread across lanes, so they are gathered in per-warp smem [32 candidates][Hi+1], then lane = one candidate sums in ascending order (fixed order).
//   · One (head, key) value = accumulation of 4 mma blocks (b 0→3) — same instructions and order as the full-T version above. Differs from the CUDA-core
//     version only in fp32 accumulation order (test_idx_cand_tc).
constexpr int IC_WARPS = 4, IC_NT = 4;
__global__ void __launch_bounds__(IC_WARPS * 32) idx_scores_cand_tc_kernel(const uint8_t* __restrict__ qp, const uint8_t* __restrict__ k,
                                                                           const uint8_t* const* __restrict__ k_ptrs, const int32_t* __restrict__ T_rows,
                                                                           const bf16* __restrict__ w, int Hi, const int32_t* __restrict__ cand,
                                                                           int cand_stride, int bs, int ncand, const int32_t* __restrict__ visible,
                                                                           bf16* __restrict__ score) {
  constexpr int ROW = kvp::IDX_ROW, HDR = kvp::IDX_HDR, NBLK = kvp::IDX_D / 32;
  extern __shared__ float icred[];  // [warp][32 candidates][Hi+1]
  const int m = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, q = lane & 3, r = lane >> 2;
  const int c0 = (blockIdx.x * IC_WARPS + warp) * IC_NT * 8;
  if (c0 >= ncand) return;  // warp-level exit — only __syncwarp is used below
  int vis = visible[m];
  if (T_rows) vis = min(vis, T_rows[m]);
  const uint8_t* kbase = k_ptrs ? k_ptrs[m] : k;
  const int32_t* crow = cand + (size_t)m * cand_stride;
  auto cand_pos = [&](int ci) -> int {  // t if the position is visible, else -1
    if (ci >= ncand) return -1;
    const int32_t blk = crow[ci / bs];
    const int t = blk >= 0 ? blk * bs + ci % bs : -1;
    return (t >= 0 && t < vis) ? t : -1;
  };
  // B: column r of n tile i = candidate c0 + 8i + r — lane q takes actual k [8q, 8q+8) within the block
  uint32_t bb[IC_NT][NBLK][2], sfb[IC_NT][NBLK];
#pragma unroll
  for (int i = 0; i < IC_NT; ++i) {
    const int t = cand_pos(c0 + i * 8 + r);
    const uint8_t* kr = kbase + (size_t)(t >= 0 ? t : 0) * ROW;
#pragma unroll
    for (int b = 0; b < NBLK; ++b) {
      if (t >= 0) {
        const uint32_t P = *reinterpret_cast<const uint32_t*>(kr + HDR + b * 16 + 4 * q);
        bb[i][b][0] = e2m1x4_to_bytes(P & 0xFFFFu); bb[i][b][1] = e2m1x4_to_bytes(P >> 16);
        sfb[i][b] = (uint32_t)kr[b];
      } else { bb[i][b][0] = bb[i][b][1] = 0u; sfb[i][b] = 127u; }
    }
  }
  float* red = icred + (size_t)warp * 32 * (Hi + 1);
  const uint8_t* qm = qp + (size_t)m * Hi * ROW;
  for (int h0 = 0; h0 < Hi; h0 += 16) {
    // A: rows = heads h0 + r (a0,a2) and h0 + r + 8 (a1,a3); sfa = scale of head h0 + r + 8·(lane&1)
    const uint8_t* qa = qm + (size_t)(h0 + r) * ROW;
    const uint8_t* qb = qa + (size_t)8 * ROW;
    const uint8_t* qs = qm + (size_t)(h0 + r + 8 * (lane & 1)) * ROW;
    float acc[IC_NT][4];
#pragma unroll
    for (int i = 0; i < IC_NT; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
    for (int b = 0; b < NBLK; ++b) {
      uint32_t a[4];
      const uint32_t PA = *reinterpret_cast<const uint32_t*>(qa + HDR + b * 16 + 4 * q);
      const uint32_t PB = *reinterpret_cast<const uint32_t*>(qb + HDR + b * 16 + 4 * q);
      a[0] = e2m1x4_to_bytes(PA & 0xFFFFu); a[2] = e2m1x4_to_bytes(PA >> 16);
      a[1] = e2m1x4_to_bytes(PB & 0xFFFFu); a[3] = e2m1x4_to_bytes(PB >> 16);
      const uint32_t sfa = (uint32_t)qs[b];
#pragma unroll
      for (int i = 0; i < IC_NT; ++i) mma_mx_e2m1_e2m1(acc[i], a, bb[i][b], sfa, sfb[i][b]);
    }
    // d0,d1 = (head h0+r, candidate 8i+2q+c) · d2,d3 = (head h0+r+8, …)
    const float wa = bf2f(w[(size_t)m * Hi + h0 + r]), wb = bf2f(w[(size_t)m * Hi + h0 + r + 8]);
#pragma unroll
    for (int i = 0; i < IC_NT; ++i)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        float sb = bf2f(f2bf(acc[i][c]));  // einsum output is bf16
        sb = fmaxf(sb, 0.f);               // relu
        sb = bf2f(f2bf(sb * (c < 2 ? wa : wb)));
        red[(i * 8 + 2 * q + (c & 1)) * (Hi + 1) + h0 + r + (c < 2 ? 0 : 8)] = sb;
      }
  }
  __syncwarp();
  const int ci = c0 + lane;
  if (ci >= ncand) return;
  float total = 0.f;
  for (int h = 0; h < Hi; ++h) total += red[lane * (Hi + 1) + h];  // heads in ascending order (same order as the CUDA-core version)
  score[(size_t)m * ncand + ci] = cand_pos(ci) >= 0 ? f2bf(total) : f2bf(-INFINITY);
}

// Block-scaled tiled GEMM: C[M,N] = A(e4m3 [M,K], sa [M,K/32]) × B(e4m3 [N,K] or e2m1 nibbles [N,K/2], sb [N,K/32])ᵀ, fp32 accumulation.
//   gemm_bs_tc expands every element's e4m3/e2m1·2^k to bf16 in smem and runs WMMA bf16 (measured 50–60 TFLOPS at M=2048 — expansion cost + bf16 tensor rate).
//   Here the raw bytes go into smem unchanged via cp.async and are multiplied with mma kind::mxf8f6f4 (native block scaling, no expansion) — the only value
//   difference is fp32 accumulation order (e4m3·e2m1·2^k products are exact). Fragment layout and k permutation match mx_gemm_rows_kernel (D-1, test_mx3):
//   lane q takes actual k [8q, 8q+8).
//   Tile 128×128×64 (two 32-element k blocks), 8 warps = (64 rows × 2) × (32 columns × 4), pipeline fp4 3 stages (38.4 KB) / fp8 2 stages (33.8 KB) —
//   sm_120 has 100 KB smem per SM (1 KB reserved per block), so 3-stage fp8 (50.7 KB) would allow only one block per SM → 2 stages for two blocks.
//   With an smem row stride of 32 B (fp8) / 16 B (fp4), the 8 B / 4 B fragment loads read a contiguous 128 B within a half-warp / warp, so there are no bank conflicts (no padding needed).
//   Assumes K % 64 == 0 and N % 8 == 0 (current model: K 1280/2304/5120/8192, N 512–32768). Enable = HIVE_MX_GEMM=1 (default off pending validation — test_mxg checks it against gemm_bs_tc).
constexpr int TG_BM = 128, TG_BN = 128, TG_BK = 64, TG_THREADS = 256;
template <bool B_FP4, int TG_STAGES>
struct TgSmem {
  static constexpr int A_BYTES = 2 * TG_BM * 32;                     // [kblk][row][32B]
  static constexpr int B_ROW = B_FP4 ? 16 : 32;
  static constexpr int B_BYTES = 2 * TG_BN * B_ROW;                  // [kblk][n][row]
  static constexpr int S_BYTES = 2 * TG_BM;                          // [kblk][row] ue8m0
  static constexpr int STAGE = A_BYTES + B_BYTES + 2 * S_BYTES;
  static constexpr int TOTAL = TG_STAGES * STAGE;
};
__device__ __forceinline__ uint32_t tg_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void tg_cp_async16(void* smem, const void* gmem, bool valid) {
  const int n = valid ? 16 : 0;  // src-size 0 → fill 16 B with zeros (out-of-range row)
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(tg_smem_u32(smem)), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void tg_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void tg_cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <bool B_FP4, int TG_STAGES>
__global__ void __launch_bounds__(TG_THREADS, 2) mx_gemm_tiled_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa,
                                                                     const uint8_t* __restrict__ B, const uint8_t* __restrict__ sb, int M, int N, int K,
                                                                     bf16* __restrict__ C, float* __restrict__ C32) {
  using SM = TgSmem<B_FP4, TG_STAGES>;
  static_assert(SM::TOTAL + 1024 <= 100 * 1024 / 2, "two blocks per SM on sm_120");
  extern __shared__ __align__(128) uint8_t tg_smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, q = lane & 3, r = lane >> 2;
  const int wm = warp & 1, wn = warp >> 1;  // 64 rows × 2 · 32 columns × 4
  const int m0 = blockIdx.y * TG_BM, n0 = blockIdx.x * TG_BN;
  const int nb = K / 32, KT = K / TG_BK;
  auto As = [&](int st, int kb) { return tg_smem + st * SM::STAGE + kb * (TG_BM * 32); };
  auto Bs = [&](int st, int kb) { return tg_smem + st * SM::STAGE + SM::A_BYTES + kb * (TG_BN * SM::B_ROW); };
  auto SAs = [&](int st, int kb) { return tg_smem + st * SM::STAGE + SM::A_BYTES + SM::B_BYTES + kb * TG_BM; };
  auto SBs = [&](int st, int kb) { return tg_smem + st * SM::STAGE + SM::A_BYTES + SM::B_BYTES + SM::S_BYTES + kb * TG_BN; };
  // Stage load: A 512 chunks (2 per thread), B fp8 512 (2) / fp4 256 (1); scales are byte-granular, so plain load → st.shared
  auto load_stage = [&](int st, int kt) {
    const int k0 = kt * TG_BK;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int c = tid + i * TG_THREADS;           // 0..511: kb = c>>8, row = (c>>1)&127, half = c&1
      const int kb = c >> 8, row = (c >> 1) & 127, half = c & 1;
      const int gm = m0 + row;
      const bool ok = gm < M;
      tg_cp_async16(As(st, kb) + row * 32 + half * 16, A + (size_t)(ok ? gm : 0) * K + k0 + kb * 32 + half * 16, ok);
    }
    if (B_FP4) {
      const int c = tid;                             // 0..255: kb = c>>7, n = c&127
      const int kb = c >> 7, n = c & 127;
      const int gn = n0 + n;
      const bool ok = gn < N;
      tg_cp_async16(Bs(st, kb) + n * 16, B + (size_t)(ok ? gn : 0) * (K / 2) + (k0 + kb * 32) / 2, ok);
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int c = tid + i * TG_THREADS;
        const int kb = c >> 8, n = (c >> 1) & 127, half = c & 1;
        const int gn = n0 + n;
        const bool ok = gn < N;
        tg_cp_async16(Bs(st, kb) + n * 32 + half * 16, B + (size_t)(ok ? gn : 0) * K + k0 + kb * 32 + half * 16, ok);
      }
    }
  };
  // Scales (bytes, non-contiguous) do not fit cp.async's 4 B minimum, so they go through registers. Doing ld.global → st.shared directly inside
  //   load_stage would make every warp wait for the global round trip before the MMA; instead the next stage's loads are issued before the compute (ld is asynchronous) and written to smem after it.
  const int s_kb = tid >> 7, s_row = tid & 127;
  auto load_scales = [&](int kt, uint8_t& va, uint8_t& vb) {
    const int k0 = kt * TG_BK, gm = m0 + s_row, gn = n0 + s_row;
    va = gm < M ? sa[(size_t)gm * nb + k0 / 32 + s_kb] : (uint8_t)127;
    vb = gn < N ? sb[(size_t)gn * nb + k0 / 32 + s_kb] : (uint8_t)127;
  };
  auto store_scales = [&](int st, uint8_t va, uint8_t vb) { SAs(st, s_kb)[s_row] = va; SBs(st, s_kb)[s_row] = vb; };
  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  // Prologue: the first STAGES-1 stages
#pragma unroll
  for (int st = 0; st < TG_STAGES - 1; ++st) {
    if (st < KT) { load_stage(st, st); uint8_t va, vb; load_scales(st, va, vb); store_scales(st, va, vb); }
    tg_cp_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    tg_cp_wait<TG_STAGES - 2>();
    __syncthreads();  // stage kt (data and scales) is visible to everyone, and compute on stage (kt-1)%S is done (the next load overwrites it)
    const int nk = kt + TG_STAGES - 1;
    uint8_t nsa = 127, nsb = 127;
    if (nk < KT) { load_stage(nk % TG_STAGES, nk); load_scales(nk, nsa, nsb); }
    tg_cp_commit();
    const int st = kt % TG_STAGES;
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      const uint8_t* as = As(st, kb);
      const uint8_t* bs = Bs(st, kb);
      const uint8_t* sas = SAs(st, kb);
      const uint8_t* sbs = SBs(st, kb);
      uint32_t bfr[4][2], sfb[4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int n = wn * 32 + nt * 8 + r;
        if (B_FP4) {
          const uint32_t P = *reinterpret_cast<const uint32_t*>(bs + n * 16 + 4 * q);
          bfr[nt][0] = e2m1x4_to_bytes(P & 0xFFFFu); bfr[nt][1] = e2m1x4_to_bytes(P >> 16);
        } else {
          const uint2 v = *reinterpret_cast<const uint2*>(bs + n * 32 + 8 * q);
          bfr[nt][0] = v.x; bfr[nt][1] = v.y;
        }
        sfb[nt] = sbs[n];
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        const int ra = wm * 64 + mt * 16 + r;
        const uint2 va = *reinterpret_cast<const uint2*>(as + ra * 32 + 8 * q);
        const uint2 vb = *reinterpret_cast<const uint2*>(as + (ra + 8) * 32 + 8 * q);
        const uint32_t a[4] = {va.x, vb.x, va.y, vb.y};
        const uint32_t sfa = sas[wm * 64 + mt * 16 + r + 8 * (lane & 1)];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          if (B_FP4) mma_mx_e4m3_e2m1(acc[mt][nt], a, bfr[nt], sfa, sfb[nt]);
          else mma_mx_e4m3_e4m3(acc[mt][nt], a, bfr[nt], sfa, sfb[nt]);
        }
      }
    }
    if (nk < KT) store_scales(nk % TG_STAGES, nsa, nsb);  // slot nk%S is not read in this iteration — the sync at the top of the next iteration publishes it
  }
  tg_cp_wait<0>();
  // Epilogue: d0,d1 = (row r, columns 2q, 2q+1) · d2,d3 = (row r+8)
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int ra = m0 + wm * 64 + mt * 16 + r, rb = ra + 8;
      const int n = n0 + wn * 32 + nt * 8 + 2 * q;
      if (n >= N) continue;
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        if (n + c >= N) continue;
        if (ra < M) { const float v = acc[mt][nt][c]; if (C) C[(size_t)ra * N + n + c] = f2bf(v); if (C32) C32[(size_t)ra * N + n + c] = v; }
        if (rb < M) { const float v = acc[mt][nt][2 + c]; if (C) C[(size_t)rb * N + n + c] = f2bf(v); if (C32) C32[(size_t)rb * N + n + c] = v; }
      }
    }
}

// Grouped version (HIVE_GROUPED_PREFILL): **the same body** as mx_gemm_tiled_kernel<true, S> above (loads, pipeline, mma order, epilogue); the differences are
//   (1) the block's row tile comes from a table (tiles[blockIdx.y]: expert weights + gathered rows [row0, row0+n), n <= 128), (2) A row addresses go through a row map
//   (rref → row of the sub-chunk xq), (3) blockIdx.z selects w1/w3. Independent per-row mmas with the same k order → bit-identical to calling gemm_bs_mx_tiled
//   separately per expert (test_grouped_prefill). The original kernel is unchanged (default path untouched). fp4 weights only (experts).
template <int TG_STAGES>
__global__ void __launch_bounds__(TG_THREADS, 2) mx_gemm_tiled_grouped_kernel(const GroupDesc* __restrict__ tiles, int which,
                                                                             const uint8_t* const* __restrict__ a_base,
                                                                             const uint8_t* const* __restrict__ s_base, const int32_t* __restrict__ rref,
                                                                             const uint8_t* __restrict__ A0, const uint8_t* __restrict__ sa0, int N, int K,
                                                                             bf16* __restrict__ C0, bf16* __restrict__ C1) {
  using SM = TgSmem<true, TG_STAGES>;
  extern __shared__ __align__(128) uint8_t tg_smem[];
  const GroupDesc d = tiles[blockIdx.y];
  const int z = blockIdx.z;
  const uint8_t* __restrict__ B = which == 0 ? (z ? d.w3 : d.w1) : d.w2;
  const uint8_t* __restrict__ sb = which == 0 ? (z ? d.s3 : d.s1) : d.s2;
  bf16* __restrict__ C = z ? C1 : C0;
  const int M = d.n;  // valid rows in the tile (the original's gm < M check = local row < n here)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, q = lane & 3, r = lane >> 2;
  const int wm = warp & 1, wn = warp >> 1;
  const int n0 = blockIdx.x * TG_BN;
  const int nb = K / 32, KT = K / TG_BK;
  // gathered row r → A row and scale row addresses (original: A + gm·K, sa + gm·nb)
  auto a_row = [&](int lr) -> const uint8_t* {
    const int gr = d.row0 + lr;
    if (!rref) return A0 + (size_t)gr * K;
    const int v = rref[gr];
    return a_base[v >> 24] + (size_t)(v & 0xFFFFFF) * K;
  };
  auto s_row = [&](int lr) -> const uint8_t* {
    const int gr = d.row0 + lr;
    if (!rref) return sa0 + (size_t)gr * nb;
    const int v = rref[gr];
    return s_base[v >> 24] + (size_t)(v & 0xFFFFFF) * nb;
  };
  auto As = [&](int st, int kb) { return tg_smem + st * SM::STAGE + kb * (TG_BM * 32); };
  auto Bs = [&](int st, int kb) { return tg_smem + st * SM::STAGE + SM::A_BYTES + kb * (TG_BN * SM::B_ROW); };
  auto SAs = [&](int st, int kb) { return tg_smem + st * SM::STAGE + SM::A_BYTES + SM::B_BYTES + kb * TG_BM; };
  auto SBs = [&](int st, int kb) { return tg_smem + st * SM::STAGE + SM::A_BYTES + SM::B_BYTES + SM::S_BYTES + kb * TG_BN; };
  // A thread's A load row is (tid >> 1) & 127 for both chunks (i = 0, 1) — resolve the address once. Out-of-range rows use row 0's address (the original also uses gm 0) + src-size 0.
  const int a_lr = (tid >> 1) & 127;
  const bool a_ok = a_lr < M;
  const uint8_t* __restrict__ a_ptr = a_row(a_ok ? a_lr : 0);
  auto load_stage = [&](int st, int kt) {
    const int k0 = kt * TG_BK;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int c = tid + i * TG_THREADS;
      const int kb = c >> 8, row = (c >> 1) & 127, half = c & 1;
      tg_cp_async16(As(st, kb) + row * 32 + half * 16, a_ptr + k0 + kb * 32 + half * 16, a_ok);
    }
    {
      const int c = tid;
      const int kb = c >> 7, n = c & 127;
      const int gn = n0 + n;
      const bool ok = gn < N;
      tg_cp_async16(Bs(st, kb) + n * 16, B + (size_t)(ok ? gn : 0) * (K / 2) + (k0 + kb * 32) / 2, ok);
    }
  };
  const int s_kb = tid >> 7, s_lr = tid & 127;
  const bool s_ok = s_lr < M;
  const uint8_t* __restrict__ s_ptr = s_ok ? s_row(s_lr) : nullptr;
  auto load_scales = [&](int kt, uint8_t& va, uint8_t& vb) {
    const int k0 = kt * TG_BK, gn = n0 + s_lr;
    va = s_ok ? s_ptr[k0 / 32 + s_kb] : (uint8_t)127;
    vb = gn < N ? sb[(size_t)gn * nb + k0 / 32 + s_kb] : (uint8_t)127;
  };
  auto store_scales = [&](int st, uint8_t va, uint8_t vb) { SAs(st, s_kb)[s_lr] = va; SBs(st, s_kb)[s_lr] = vb; };
  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
#pragma unroll
  for (int st = 0; st < TG_STAGES - 1; ++st) {
    if (st < KT) { load_stage(st, st); uint8_t va, vb; load_scales(st, va, vb); store_scales(st, va, vb); }
    tg_cp_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    tg_cp_wait<TG_STAGES - 2>();
    __syncthreads();
    const int nk = kt + TG_STAGES - 1;
    uint8_t nsa = 127, nsb = 127;
    if (nk < KT) { load_stage(nk % TG_STAGES, nk); load_scales(nk, nsa, nsb); }
    tg_cp_commit();
    const int st = kt % TG_STAGES;
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      const uint8_t* as = As(st, kb);
      const uint8_t* bs = Bs(st, kb);
      const uint8_t* sas = SAs(st, kb);
      const uint8_t* sbs = SBs(st, kb);
      uint32_t bfr[4][2], sfb[4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int n = wn * 32 + nt * 8 + r;
        const uint32_t P = *reinterpret_cast<const uint32_t*>(bs + n * 16 + 4 * q);
        bfr[nt][0] = e2m1x4_to_bytes(P & 0xFFFFu); bfr[nt][1] = e2m1x4_to_bytes(P >> 16);
        sfb[nt] = sbs[n];
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        const int ra = wm * 64 + mt * 16 + r;
        const uint2 va = *reinterpret_cast<const uint2*>(as + ra * 32 + 8 * q);
        const uint2 vb = *reinterpret_cast<const uint2*>(as + (ra + 8) * 32 + 8 * q);
        const uint32_t a[4] = {va.x, vb.x, va.y, vb.y};
        const uint32_t sfa = sas[wm * 64 + mt * 16 + r + 8 * (lane & 1)];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma_mx_e4m3_e2m1(acc[mt][nt], a, bfr[nt], sfa, sfb[nt]);
      }
    }
    if (nk < KT) store_scales(nk % TG_STAGES, nsa, nsb);
  }
  tg_cp_wait<0>();
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int ra = wm * 64 + mt * 16 + r, rb = ra + 8;  // local row within the tile
      const int n = n0 + wn * 32 + nt * 8 + 2 * q;
      if (n >= N) continue;
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        if (n + c >= N) continue;
        if (ra < M) C[(size_t)(d.row0 + ra) * N + n + c] = f2bf(acc[mt][nt][c]);
        if (rb < M) C[(size_t)(d.row0 + rb) * N + n + c] = f2bf(acc[mt][nt][2 + c]);
      }
    }
}

}  // namespace

void mx_grouped_w13(const GroupDesc* g, int ngroups, const uint8_t* A, const uint8_t* sa, int I, int K, const float* rw, float limit, bf16* y,
                    cudaStream_t st, const int32_t* rows, uint8_t* yq, uint8_t* ys) {
  if (ngroups <= 0) return;
  HIVE_CHECK(K % 32 == 0 && I % 8 == 0 && (!yq || I % 32 == 0), "mx_grouped_w13 shape");
  const int tiles = I / 8, per_block = MX_WARPS * TILES_PER_WARP;
  dim3 grid((tiles + per_block - 1) / per_block, ngroups);
  mx_grouped_w13_kernel<<<grid, MX_WARPS * 32, 0, st>>>(g, A, sa, rows, I, K, rw, limit, y, yq, ys);
}
void mx_grouped_w2(const GroupDesc* g, int ngroups, const uint8_t* Yq, const uint8_t* ys, int dim, int I, bf16* eout, cudaStream_t st) {
  if (ngroups <= 0) return;
  HIVE_CHECK(I % 32 == 0 && dim % 8 == 0, "mx_grouped_w2 shape");
  const int tiles = dim / 8, per_block = MX_WARPS * TILES_PER_WARP;
  dim3 grid((tiles + per_block - 1) / per_block, ngroups);
  mx_grouped_w2_kernel<<<grid, MX_WARPS * 32, 0, st>>>(g, Yq, ys, dim, I, eout);
}

void gemm_bs_mx(const uint8_t* A, const uint8_t* sa, const uint8_t* B, const uint8_t* sb, int M, int N, int K, bf16* C, float* C32, cudaStream_t st) {
  HIVE_CHECK(K % 32 == 0 && M > 0 && N > 0, "gemm_bs_mx shape");
  constexpr int T = 8;  // 64 columns per warp
  dim3 grid((N + T * 8 - 1) / (T * 8), (M + MX_WARPS * 16 - 1) / (MX_WARPS * 16));
  mx_gemm_rows_kernel<T><<<grid, MX_WARPS * 32, 0, st>>>(A, sa, M, K, B, sb, N, C, C32);
}

void indexer_scores_tc(const uint8_t* q_packed, const uint8_t* k_packed, const bf16* w, int M, int Hi, int T, const int32_t* visible, bf16* score,
                       cudaStream_t st) {
  if (M <= 0 || T <= 0) return;
  constexpr int NT = 4;  // 32 keys per warp
  dim3 grid((T + NT * 8 - 1) / (NT * 8), (M + MX_WARPS * 16 - 1) / (MX_WARPS * 16));
  idx_scores_tc_kernel<NT><<<grid, MX_WARPS * 32, 0, st>>>(q_packed, k_packed, w, M, Hi, T, visible, score);
}

void indexer_scores_cand_tc(const uint8_t* q_packed, const uint8_t* k_packed, const uint8_t* const* k_ptrs, const int32_t* T_rows, const bf16* w, int M,
                            int Hi, const int32_t* cand, int cand_stride, int kb, int bs, const int32_t* visible, bf16* score, cudaStream_t st) {
  HIVE_CHECK(Hi % 16 == 0 && Hi > 0 && Hi <= 64 && kb >= 1 && bs >= 1, "indexer_scores_cand_tc shape (Hi % 16, ≤ 64)");
  if (M <= 0) return;
  const int ncand = kb * bs;
  const size_t smem = (size_t)IC_WARPS * 32 * (Hi + 1) * sizeof(float);  // Hi 32: 16.9 KB, 64: 33.3 KB (within the 48 KB static limit)
  dim3 grid((ncand + IC_WARPS * IC_NT * 8 - 1) / (IC_WARPS * IC_NT * 8), M);
  idx_scores_cand_tc_kernel<<<grid, IC_WARPS * 32, smem, st>>>(q_packed, k_packed, k_ptrs, T_rows, w, Hi, cand, cand_stride, bs, ncand, visible, score);
}

void gemm_bs_mx_tiled(const uint8_t* A, const uint8_t* sa, const uint8_t* B, const uint8_t* sb, bool b_fp4, int M, int N, int K, bf16* C, float* C32,
                      cudaStream_t st) {
  HIVE_CHECK(K % TG_BK == 0 && N % 8 == 0 && N > 0 && M > 0, "gemm_bs_mx_tiled shape (K % 64, N % 8)");
  constexpr int S4 = 3, S8 = 2;  // fp4 3 stages (38.4 KB), fp8 2 stages (33.8 KB)
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(mx_gemm_tiled_kernel<true, S4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TgSmem<true, S4>::TOTAL));
    CUDA_CHECK(cudaFuncSetAttribute(mx_gemm_tiled_kernel<false, S8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TgSmem<false, S8>::TOTAL));
    configured = true;
  }
  dim3 grid((N + TG_BN - 1) / TG_BN, (M + TG_BM - 1) / TG_BM);
  if (b_fp4) mx_gemm_tiled_kernel<true, S4><<<grid, TG_THREADS, TgSmem<true, S4>::TOTAL, st>>>(A, sa, B, sb, M, N, K, C, C32);
  else mx_gemm_tiled_kernel<false, S8><<<grid, TG_THREADS, TgSmem<false, S8>::TOTAL, st>>>(A, sa, B, sb, M, N, K, C, C32);
}

void gemm_bs_mx_tiled_grouped(const GroupDesc* tiles, int ntiles, int which, const uint8_t* const* a_base, const uint8_t* const* s_base, const int32_t* rref,
                              const uint8_t* A, const uint8_t* sa, int N, int K, bf16* C0, bf16* C1, cudaStream_t st) {
  if (ntiles <= 0) return;
  HIVE_CHECK(K % TG_BK == 0 && N % 8 == 0 && N > 0 && ntiles <= 65535 && (which == 0 ? C1 != nullptr : true), "gemm_bs_mx_tiled_grouped shape");
  constexpr int S4 = 3;  // same fp4 stage count as gemm_bs_mx_tiled
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(mx_gemm_tiled_grouped_kernel<S4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TgSmem<true, S4>::TOTAL));
    configured = true;
  }
  dim3 grid((N + TG_BN - 1) / TG_BN, ntiles, which == 0 ? 2 : 1);
  mx_gemm_tiled_grouped_kernel<S4><<<grid, TG_THREADS, TgSmem<true, S4>::TOTAL, st>>>(tiles, which, a_base, s_base, rref, A, sa, N, K, C0, C1);
}

}  // namespace hive::k
