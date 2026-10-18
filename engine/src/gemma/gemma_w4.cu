// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// int4 group-32 weight kernels for the Gemma family (hive/gemma/gemma_kernels.h).
//
// Decode / verify GEMV ("swapped" tensor-core form): the weight is the mma A operand (16 output rows × k16), the activations are B
//   (k16 × 8 tokens; a second B tile covers tokens 8..15). Per 32-input group the two k16 halves accumulate the integer products
//   (nibble - 8, exact in bf16) in a fresh fp32 tile, which is then scaled by the group's bf16 scale into the fp32 accumulator — the
//   scale is applied in fp32, never rounded into the weight. A warp walks its 16 rows over a K range (the matrix's fixed split), so a
//   row's sum never depends on how many tokens share the call (batch-invariant decode / verify).
//   mma.m16n8k16 bf16 fragments (row-major A, column-major B): with g = lane / 4, c = lane % 4,
//     a0 = A[g][2c, 2c+1], a1 = A[g+8][2c, 2c+1], a2 = A[g][2c+8, 2c+9], a3 = A[g+8][2c+8, 2c+9]
//     b0 = B[2c, 2c+1][g], b1 = B[2c+8, 2c+9][g];  d0, d1 = D[g][2c, 2c+1], d2, d3 = D[g+8][2c, 2c+1]
//   Integer pairs are made with the 128-bias trick: bf16 bits 0x4300 | n = 128 + n exactly (n < 16), minus 136 → n - 8, exact.
//   The weight is read in fragment order (w4_repack): per 2 groups a lane loads one 16-byte word quad that holds exactly its fragments,
//   so a warp reads its tile's 512 bytes once, contiguously (the checkpoint order had 4 lanes re-reading each 16 bytes).
//
// Prefill GEMM (more than 16 rows): the same math as the GEMV, row for row and bit for bit — the same K split over the block's warps
//   (W.split, warp_range), the same per-group integer tile and fp32 scaling, the same warp-order sum — so a row's result is still
//   independent of how many rows share the call, and equals the decode / verify GEMV's. What changes is reuse: the GEMV re-reads every
//   token's activations from L1 / L2 for each 16-row tile (at 32 rows that read, not the weight stream, bounds it: q + k + v + o + down
//   of a layer took 92 µs against ~15 µs of weight bytes); here a warp owns GM_TN tiles (64 rows) × GM_BM tokens and stages the tokens'
//   activations of its K range in shared memory (cp.async), feeding the mma B operands with ldmatrix.
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "hive/common.h"
#include "hive/gemma/gemma_kernels.h"
#include "gemma_pdl.cuh"

namespace hive::gemma::k {

namespace {

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// nibble pair p of a fragment word (first value at bits 4p, second at 16 + 4p) → bf16x2 {first - 8, second - 8}
__device__ __forceinline__ uint32_t pair(uint32_t w, int p) {
  const uint32_t biased = ((w >> (4 * p)) & 0x000F000Fu) | 0x43004300u;   // {128 + first, 128 + second}
  __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&biased);
  v = __hsub2(v, __floats2bfloat162_rn(136.f, 136.f));
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t ld_x2(const bf16* p) { return *reinterpret_cast<const uint32_t*>(p); }

__device__ __forceinline__ float gelu_tanh(float x) {
  return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Block = one 16-row tile; its KW warps split the K range (group pairs) evenly and their partial tiles are summed in shared memory in
//   warp order — a fixed order that depends on K only, so a row's sum is the same for any M (batch-invariant decode / verify). A warp
//   issues the loads of up to PC group pairs at once (weights, then scales) before the mma work, so each SM keeps several KB in flight.
constexpr int PC = 8;
constexpr int KW_MAX = 8;

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// One warp: rows [16 tile, 16 tile + 16) of W over group pairs [p0, p1); NT = token tiles of 8 (1 or 2). acc[t][i] per D fragment.
template <int NT>
__device__ __forceinline__ void w4_range(const W4Mat& W, int tile, int p0, int p1, const bf16* __restrict__ x, int ldx, int M, float (&acc)[NT][4]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  const int G = W.K / 32;
  const uint4* qt = reinterpret_cast<const uint4*>(W.q) + (size_t)tile * (G / 2) * 32 + lane;
  const uint32_t* st_ = W.s + (size_t)tile * G * 8 + g;
  const bf16* xb[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int tok = t * 8 + g;
    xb[t] = x + (size_t)(tok < M ? tok : 0) * ldx;   // tokens >= M read row 0 and are never stored
  }
  for (int pc = p0; pc < p1; pc += PC) {
    uint4 w[PC];
    uint32_t sp[PC][2];
#pragma unroll
    for (int i = 0; i < PC; ++i)
      if (pc + i < p1) w[i] = ld_stream(qt + (size_t)(pc + i) * 32);   // {group 2p: half 0, half 1, group 2p+1: half 0, half 1}
#pragma unroll
    for (int i = 0; i < PC; ++i)
      if (pc + i < p1) { sp[i][0] = __ldg(st_ + (size_t)(2 * (pc + i)) * 8); sp[i][1] = __ldg(st_ + (size_t)(2 * (pc + i) + 1) * 8); }
#pragma unroll
    for (int i = 0; i < PC; ++i) {
      if (pc + i >= p1) break;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(&sp[i][j]);
        const float sc0 = __low2float(s2), sc1 = __high2float(s2);
        float tmp[NT][4];
#pragma unroll
        for (int t = 0; t < NT; ++t) tmp[t][0] = tmp[t][1] = tmp[t][2] = tmp[t][3] = 0.f;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t wd = j ? (h ? w[i].w : w[i].z) : (h ? w[i].y : w[i].x);
          const uint32_t a0 = pair(wd, 0), a1 = pair(wd, 1), a2 = pair(wd, 2), a3 = pair(wd, 3);
          const int k = (2 * (pc + i) + j) * 32 + h * 16 + 2 * c;
#pragma unroll
          for (int t = 0; t < NT; ++t) mma_bf16(tmp[t], a0, a1, a2, a3, ld_x2(xb[t] + k), ld_x2(xb[t] + k + 8));
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          acc[t][0] += tmp[t][0] * sc0; acc[t][1] += tmp[t][1] * sc0;
          acc[t][2] += tmp[t][2] * sc1; acc[t][3] += tmp[t][3] * sc1;
        }
      }
    }
  }
}

// sum the block's warp partials in warp order; on return lane values of warp 0 hold the tile's totals (other warps: garbage)
template <int NT, int NA>
__device__ __forceinline__ void block_reduce(float (&acc)[NA][NT][4], float* red) {
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, KW = blockDim.x >> 5;
  if (KW == 1) return;
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) red[(((size_t)wid * NA + a) * NT * 4 + t * 4 + i) * 32 + lane] = acc[a][t][i];
  __syncthreads();
  if (wid != 0) return;
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float v = red[(((size_t)0 * NA + a) * NT * 4 + t * 4 + i) * 32 + lane];
        for (int w = 1; w < KW; ++w) v += red[(((size_t)w * NA + a) * NT * 4 + t * 4 + i) * 32 + lane];
        acc[a][t][i] = v;
      }
}

// store D fragments: rows (n) n0 + g / n0 + g + 8, tokens 2c, 2c+1 (+8 per tile)
template <int NT>
__device__ __forceinline__ void store_tile(const float (&acc)[NT][4], int n0, int M, bf16* yb, float* yf, int ldy) {
  const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int tok = t * 8 + 2 * c + (i & 1), n = n0 + g + (i >> 1) * 8;
      if (tok >= M) continue;
      if (yf) yf[(size_t)tok * ldy + n] = acc[t][i];
      else yb[(size_t)tok * ldy + n] = __float2bfloat16(acc[t][i]);
    }
}

__device__ __forceinline__ void warp_range(int K, int& p0, int& p1) {
  const int P = K / 64, KW = blockDim.x >> 5, per = (P + KW - 1) / KW, w = threadIdx.x >> 5;
  p0 = min(P, w * per);
  p1 = min(P, p0 + per);
}

// the warp's weight / scale bytes of tile over pairs [p0, p1) into L2 (independent of the chain: issued before pdl_wait)
__device__ __forceinline__ void prefetch_range(const W4Mat& W, int tile, int p0, int p1) {
  if ((threadIdx.x & 31) != 0 || p1 <= p0) return;
  const int G = W.K / 32;
  prefetch_l2(reinterpret_cast<const uint4*>(W.q) + ((size_t)tile * (G / 2) + p0) * 32, (uint32_t)(p1 - p0) * 512u);
  prefetch_l2(W.s + ((size_t)tile * G + 2 * p0) * 8, (uint32_t)(p1 - p0) * 64u);
}

// up to 3 matrices with the same K and input, tiles concatenated along blockIdx.x
template <int NT>
__global__ void __launch_bounds__(KW_MAX * 32) w4_gemv_kernel(W4Multi J, const bf16* __restrict__ x, int ldx, int M) {
  extern __shared__ float red[];
  int tile = blockIdx.x, m = 0;
  while (m + 1 < J.n && tile >= J.W[m].N / 16) { tile -= J.W[m].N / 16; ++m; }
  const W4Mat& W = J.W[m];
  int p0, p1;
  warp_range(W.K, p0, p1);
  prefetch_range(W, tile, p0, p1);
  pdl_wait();
  pdl_trigger();   // after the wait: the next GEMV may prefetch while this one streams, but not two ahead
  float acc[1][NT][4] = {};
  w4_range<NT>(W, tile, p0, p1, x, ldx, M, acc[0]);
  block_reduce<NT, 1>(acc, red);
  if ((threadIdx.x >> 5) == 0) store_tile<NT>(acc[0], tile * 16, M, J.yb[m], J.yf[m], J.ldy[m]);
}

template <int NT>
__global__ void __launch_bounds__(KW_MAX * 32) w4_gemv_gu_kernel(W4Mat Gm, W4Mat Um, const bf16* __restrict__ x, int ldx, int M, bf16* act, int lda) {
  extern __shared__ float red[];
  const int tile = blockIdx.x;
  int p0, p1;
  warp_range(Gm.K, p0, p1);
  prefetch_range(Gm, tile, p0, p1);
  prefetch_range(Um, tile, p0, p1);
  pdl_wait();
  pdl_trigger();
  float acc[2][NT][4] = {};
  w4_range<NT>(Gm, tile, p0, p1, x, ldx, M, acc[0]);
  w4_range<NT>(Um, tile, p0, p1, x, ldx, M, acc[1]);
  block_reduce<NT, 2>(acc, red);
  if ((threadIdx.x >> 5) != 0) return;
  const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int tok = t * 8 + 2 * c + (i & 1), n = tile * 16 + g + (i >> 1) * 8;
      if (tok < M) act[(size_t)tok * lda + n] = __float2bfloat16(gelu_tanh(acc[0][t][i]) * acc[1][t][i]);
    }
}

// Variants (template): TN 16-row tiles per warp × BM tokens per block, ST activation stages per warp (one group pair = 64 k each), PF =
//   the next pair's weight words loaded into registers before the current pair's mma work, MB = blocks per SM the launch bounds aim at.
//   Every variant computes the same sums in the same order (bit for bit the GEMV's); HIVE_GEMMA_GEMM_V picks one (gm_variant).
constexpr int GM_LDS = 72;   // staged row stride in bf16 (144 B: the 8 rows of an ldmatrix land in distinct banks)

__device__ __forceinline__ void cp_async16(void* s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"((uint32_t)__cvta_generic_to_shared(p)));
}

template <int TN, int BM, int ST>
constexpr size_t gm_smem_bytes(int kw) {
  return (size_t)kw * ST * BM * GM_LDS * 2 > (size_t)kw * TN * (BM / 8) * 4 * 32 * sizeof(float)
             ? (size_t)kw * ST * BM * GM_LDS * 2 : (size_t)kw * TN * (BM / 8) * 4 * 32 * sizeof(float);
}

// GU = false: J's matrices, tiles concatenated along blockIdx.y in groups of TN; outputs J.yb / J.yf.
// GU = true: J.W[0] = gate, J.W[1] = up; a warp's tiles are TN / 2 gate tiles and the same rows of up; act = bf16(gelu(g) · u).
template <bool GU, int TN, int BM, int ST, bool PF, int MB>
__global__ void __launch_bounds__(KW_MAX * 32, MB) w4_gemm_kernel(W4Multi J, const bf16* __restrict__ x, int ldx, int M, bf16* act, int lda) {
  extern __shared__ __align__(16) uint8_t gsm[];
  constexpr int TPM = GU ? TN / 2 : TN;   // tiles of one matrix per block
  constexpr int NTT = BM / 8;
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, c = lane & 3;
  int nb = blockIdx.y, m = 0;
  if (!GU) while (m + 1 < J.n && nb >= J.W[m].N / (16 * TPM)) { nb -= J.W[m].N / (16 * TPM); ++m; }
  const W4Mat& W0 = J.W[m];
  const int G = W0.K / 32, t0 = blockIdx.x * BM;
  int p0, p1;
  warp_range(W0.K, p0, p1);
  const uint4* qt[TN];
  const uint32_t* stp[TN];
#pragma unroll
  for (int t = 0; t < TN; ++t) {
    const W4Mat& W = GU ? J.W[t < TPM ? 0 : 1] : W0;
    const int tile = nb * TPM + (GU ? t % TPM : t);
    qt[t] = reinterpret_cast<const uint4*>(W.q) + (size_t)tile * (G / 2) * 32 + lane;
    stp[t] = W.s + (size_t)tile * G * 8 + g;
    if (blockIdx.x == 0) prefetch_range(W, tile, p0, p1);   // one token block pulls the weights into L2 for all
  }
  pdl_wait();
  pdl_trigger();
  bf16* xs = reinterpret_cast<bf16*>(gsm) + (size_t)wid * ST * BM * GM_LDS;
  auto stage = [&](int p, int sidx) {   // activations of tokens t0 .. t0 + BM, k [64 p, 64 p + 64) → stage sidx
#pragma unroll
    for (int it = 0; it < BM * 8 / 32; ++it) {
      const int idx = it * 32 + lane, row = idx >> 3, ch = idx & 7, tok = t0 + row;
      cp_async16(xs + ((size_t)sidx * BM + row) * GM_LDS + ch * 8, x + (size_t)(tok < M ? tok : 0) * ldx + (size_t)p * 64 + ch * 8);
    }
  };
  float acc[TN][NTT][4] = {};
  const int np = p1 - p0;
#pragma unroll
  for (int s0 = 0; s0 < ST - 1; ++s0) { if (s0 < np) stage(p0 + s0, s0); cp_async_commit(); }
  uint4 wn[TN];
  uint32_t spn[TN][2];
#define GM_WLOAD(P, W_, SP_)                                                                                                \
  _Pragma("unroll") for (int t = 0; t < TN; ++t) {                                                                         \
    W_[t] = ld_stream(qt[t] + (size_t)(P) * 32);                                                                          \
    SP_[t][0] = __ldg(stp[t] + (size_t)(2 * (P)) * 8);                                                                     \
    SP_[t][1] = __ldg(stp[t] + (size_t)(2 * (P) + 1) * 8);                                                                 \
  }
  if (PF && np > 0) { GM_WLOAD(p0, wn, spn) }
  for (int i = 0; i < np; ++i) {
    const int p = p0 + i;
    if (i + ST - 1 < np) stage(p + ST - 1, (i + ST - 1) % ST);
    cp_async_commit();
    uint4 w[TN];
    uint32_t sp[TN][2];
    if (PF) {
#pragma unroll
      for (int t = 0; t < TN; ++t) { w[t] = wn[t]; sp[t][0] = spn[t][0]; sp[t][1] = spn[t][1]; }
      if (i + 1 < np) { GM_WLOAD(p + 1, wn, spn) }
    } else {
      GM_WLOAD(p, w, sp)
    }
#undef GM_WLOAD
    cp_async_wait<ST - 1>();
    __syncwarp();
    const bf16* xst = xs + (size_t)(i % ST) * BM * GM_LDS;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float tmp[TN][NTT][4];
#pragma unroll
      for (int t = 0; t < TN; ++t)
#pragma unroll
        for (int u = 0; u < NTT; ++u) tmp[t][u][0] = tmp[t][u][1] = tmp[t][u][2] = tmp[t][u][3] = 0.f;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int ks = (2 * j + h) * 16;
        uint32_t b[NTT][2];
        if constexpr (NTT >= 2) {
#pragma unroll
          for (int tp = 0; tp < NTT / 2; ++tp) {   // matrices: (tile 2tp, k lo), (2tp, k hi), (2tp + 1, k lo), (2tp + 1, k hi)
            uint32_t r[4];
            ldmatrix_x4(r, xst + (size_t)(tp * 16 + ((lane >> 4) << 3) + (lane & 7)) * GM_LDS + ks + ((lane >> 3) & 1) * 8);
            b[2 * tp][0] = r[0]; b[2 * tp][1] = r[1]; b[2 * tp + 1][0] = r[2]; b[2 * tp + 1][1] = r[3];
          }
        } else {
          uint32_t r[2];
          ldmatrix_x2(r, xst + (size_t)(lane & 7) * GM_LDS + ks + ((lane >> 3) & 1) * 8);
          b[0][0] = r[0]; b[0][1] = r[1];
        }
#pragma unroll
        for (int t = 0; t < TN; ++t) {
          const uint32_t wd = j ? (h ? w[t].w : w[t].z) : (h ? w[t].y : w[t].x);
          const uint32_t a0 = pair(wd, 0), a1 = pair(wd, 1), a2 = pair(wd, 2), a3 = pair(wd, 3);
#pragma unroll
          for (int u = 0; u < NTT; ++u) mma_bf16(tmp[t][u], a0, a1, a2, a3, b[u][0], b[u][1]);
        }
      }
#pragma unroll
      for (int t = 0; t < TN; ++t) {
        const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(&sp[t][j]);
        const float sc0 = __low2float(s2), sc1 = __high2float(s2);
#pragma unroll
        for (int u = 0; u < NTT; ++u) {
          acc[t][u][0] += tmp[t][u][0] * sc0; acc[t][u][1] += tmp[t][u][1] * sc0;
          acc[t][u][2] += tmp[t][u][2] * sc1; acc[t][u][3] += tmp[t][u][3] * sc1;
        }
      }
    }
    __syncwarp();   // the stage read above is the one the next iteration refills
  }
  cp_async_wait<0>();
  // warp-order sum (the GEMV's block_reduce), the staging memory reused
  const int KW = blockDim.x >> 5;
  if (KW > 1) {
    __syncthreads();
    float* red = reinterpret_cast<float*>(gsm);
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int u = 0; u < NTT; ++u)
#pragma unroll
        for (int e = 0; e < 4; ++e) red[(((size_t)wid * TN + t) * NTT * 4 + u * 4 + e) * 32 + lane] = acc[t][u][e];
    __syncthreads();
    if (wid != 0) return;
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int u = 0; u < NTT; ++u)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float v = red[(((size_t)0 * TN + t) * NTT * 4 + u * 4 + e) * 32 + lane];
          for (int ww = 1; ww < KW; ++ww) v += red[(((size_t)ww * TN + t) * NTT * 4 + u * 4 + e) * 32 + lane];
          acc[t][u][e] = v;
        }
  }
#pragma unroll
  for (int t = 0; t < TPM; ++t)
#pragma unroll
    for (int u = 0; u < NTT; ++u)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int tok = t0 + u * 8 + 2 * c + (e & 1), n = (nb * TPM + t) * 16 + g + (e >> 1) * 8;
        if (tok >= M) continue;
        if (GU) act[(size_t)tok * lda + n] = __float2bfloat16(gelu_tanh(acc[t][u][e]) * acc[t + TPM][u][e]);
        else if (J.yf[m]) J.yf[m][(size_t)tok * J.ldy[m] + n] = acc[t][u][e];
        else J.yb[m][(size_t)tok * J.ldy[m] + n] = __float2bfloat16(acc[t][u][e]);
      }
}

// one fragment word per thread: (tile, group, lane, half) → its 8 values at their [n, k]
__global__ void w4_dequant_kernel(W4Mat W, bf16* out) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int G = W.K / 32;
  if (i >= (size_t)W.N * (W.K / 8)) return;
  const int q4 = (int)(i & 3), lane = (int)((i >> 2) & 31);
  const size_t pr = i >> 7;   // (tile · G/2 + pair)
  const int tile = (int)(pr / (G / 2)), gr = (int)(pr % (G / 2)) * 2 + (q4 >> 1), h = q4 & 1;
  const int g = lane >> 2, c = lane & 3;
  const uint32_t w = W.q[i];
  const uint32_t sp = W.s[((size_t)tile * G + gr) * 8 + g];
  const float sc[2] = {__bfloat162float(*reinterpret_cast<const bf16*>(&sp)), __bfloat162float(*(reinterpret_cast<const bf16*>(&sp) + 1))};
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const int row = tile * 16 + g + (p & 1) * 8, k = gr * 32 + h * 16 + 2 * c + (p >> 1) * 8;
    const float f0 = (float)((int)((w >> (4 * p)) & 0xF) - 8), f1 = (float)((int)((w >> (16 + 4 * p)) & 0xF) - 8);
    out[(size_t)row * W.K + k] = __float2bfloat16(f0 * sc[p & 1]);
    out[(size_t)row * W.K + k + 1] = __float2bfloat16(f1 * sc[p & 1]);
  }
}

__device__ __forceinline__ uint32_t ck_nib(const uint32_t* q_ck, int K, int row, int k) {
  return (q_ck[(size_t)row * (K / 8) + k / 8] >> (4 * (k % 8))) & 0xF;
}

__global__ void w4_repack_kernel(const uint32_t* __restrict__ q_ck, const bf16* __restrict__ s_ck, int N, int K, uint32_t* q, uint32_t* s) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int G = K / 32;
  if (i < (size_t)N * (K / 8)) {
    const int q4 = (int)(i & 3), lane = (int)((i >> 2) & 31);
    const size_t pr = i >> 7;
    const int tile = (int)(pr / (G / 2)), gr = (int)(pr % (G / 2)) * 2 + (q4 >> 1), h = q4 & 1;
    const int g = lane >> 2, c = lane & 3;
    uint32_t w = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int row = tile * 16 + g + (p & 1) * 8, k = gr * 32 + h * 16 + 2 * c + (p >> 1) * 8;
      w |= ck_nib(q_ck, K, row, k) << (4 * p);
      w |= ck_nib(q_ck, K, row, k + 1) << (16 + 4 * p);
    }
    q[i] = w;
  }
  if (i < (size_t)(N / 16) * G * 8) {
    const int g = (int)(i % 8);
    const size_t tg = i / 8;
    const int tile = (int)(tg / G), gr = (int)(tg % G);
    const bf16 a = s_ck[(size_t)(tile * 16 + g) * G + gr], b = s_ck[(size_t)(tile * 16 + g + 8) * G + gr];
    __nv_bfloat162 v;
    v.x = a; v.y = b;
    s[i] = *reinterpret_cast<const uint32_t*>(&v);
  }
}

}  // namespace

int w4_split_for(int N, int K) {
  // warps per 16-row tile (the K split inside a block): ~PC group pairs per warp, at most KW_MAX — a function of K only
  (void)N;
  const int P = K / 64;
  return std::max(1, std::min(KW_MAX, (P + PC - 1) / PC));
}

namespace {
template <int NT>
void launch_multi(const W4Multi& J, const bf16* x, int ldx, int M, cudaStream_t st) {
  int tiles = 0;
  for (int i = 0; i < J.n; ++i) tiles += J.W[i].N / 16;
  const int KW = J.W[0].split;
  const size_t sm = KW > 1 ? (size_t)KW * NT * 4 * 32 * sizeof(float) : 0;
  launch_pdl(w4_gemv_kernel<NT>, dim3(tiles), dim3(KW * 32), sm, st, J, x, ldx, M);
}
}  // namespace

void w4_gemv_multi(const W4Multi& J, const bf16* x, int ldx, int M, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && M <= kGemvRows && J.n >= 1 && J.n <= 3, "w4_gemv_multi: rows / matrices");
  for (int i = 0; i < J.n; ++i)
    HIVE_CHECK(J.W[i].N % 16 == 0 && J.W[i].K == J.W[0].K && J.W[i].split == J.W[0].split && J.W[0].K % 64 == 0 &&
                   ((J.yb[i] == nullptr) != (J.yf[i] == nullptr)), "w4_gemv_multi: shapes / outputs");
  switch ((M + 7) / 8) {   // token tiles of 8: up to 32 rows read each weight once (a prefill tail of a conversation turn is ~70 rows)
    case 1: launch_multi<1>(J, x, ldx, M, st); break;
    case 2: launch_multi<2>(J, x, ldx, M, st); break;
    case 3: launch_multi<3>(J, x, ldx, M, st); break;
    default: launch_multi<4>(J, x, ldx, M, st); break;
  }
}

namespace {
// the variants (TN, BM, ST, PF, MB) timed by tests/test_gemma_w4.cu (cold L2, 2026-10-18, sum of q+k+v, o, gate+up, down of a layer):
//   up to ~80 rows variant 2 (16 tokens per block, two blocks per SM: 70 rows 305 µs against 422 for variant 0 and 788 for the GEMV
//   passes), above it variant 1 (32 tokens per block, the next pair's weights loaded ahead: 128 rows 477 µs, variant 2 489, 192 rows 659
//   against 716 for dequant + cuBLAS). HIVE_GEMMA_GEMM_V forces one (0 … 5). Every variant gives the same bits.
constexpr int kGemmVariants = 6;
int gm_variant(int M) {
  static const int forced = [] { const char* e = getenv("HIVE_GEMMA_GEMM_V"); const int x = e ? atoi(e) : -1; return x >= 0 && x < kGemmVariants ? x : -1; }();
  return forced >= 0 ? forced : M <= 80 ? 2 : 1;
}
template <bool GU, int TN, int BM, int ST, bool PF, int MB>
void launch_v(const W4Multi& J, int rows_per_y, const bf16* x, int ldx, int M, bf16* act, int lda, cudaStream_t st) {
  static const bool attr = [] {
    CUDA_CHECK(cudaFuncSetAttribute(w4_gemm_kernel<GU, TN, BM, ST, PF, MB>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)gm_smem_bytes<TN, BM, ST>(KW_MAX)));
    return true;
  }();
  (void)attr;
  const int KW = J.W[0].split;
  int ty = 0;
  for (int i = 0; i < (GU ? 1 : J.n); ++i) ty += J.W[i].N / (16 * (GU ? TN / 2 : TN));
  (void)rows_per_y;
  launch_pdl(w4_gemm_kernel<GU, TN, BM, ST, PF, MB>, dim3((M + BM - 1) / BM, ty), dim3(KW * 32), gm_smem_bytes<TN, BM, ST>(KW), st, J, x, ldx, M,
             act, lda);
}
template <bool GU>
void launch_gemm(const W4Multi& J, const bf16* x, int ldx, int M, bf16* act, int lda, cudaStream_t st) {
  switch (gm_variant(M)) {
    case 0: launch_v<GU, 4, 32, 2, false, 1>(J, 64, x, ldx, M, act, lda, st); break;
    case 1: launch_v<GU, 4, 32, 2, true, 1>(J, 64, x, ldx, M, act, lda, st); break;
    case 2: launch_v<GU, 4, 16, 2, true, 2>(J, 64, x, ldx, M, act, lda, st); break;
    case 3: launch_v<GU, 2, 32, 2, true, 2>(J, 32, x, ldx, M, act, lda, st); break;
    case 4: launch_v<GU, 2, 16, 2, true, 2>(J, 32, x, ldx, M, act, lda, st); break;
    default: launch_v<GU, 8, 16, 2, true, 1>(J, 128, x, ldx, M, act, lda, st); break;
  }
}
// rows (N) one block covers, for the shape check: the largest over the variants a row count can pick (64 unless one is forced)
int gm_rows_per_block() {
  const int v = gm_variant(1) == gm_variant(1 << 20) ? gm_variant(1) : 0;
  return v == 3 || v == 4 ? 32 : v == 5 ? 128 : 64;
}
}  // namespace

bool w4_gemm_fits(const W4Mat& W) { return W.N % gm_rows_per_block() == 0 && W.K % 64 == 0 && W.split >= 1 && W.split <= KW_MAX; }

void w4_gemm_multi(const W4Multi& J, const bf16* x, int ldx, int M, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && J.n >= 1 && J.n <= 3, "w4_gemm_multi: rows / matrices");
  for (int i = 0; i < J.n; ++i)
    HIVE_CHECK(w4_gemm_fits(J.W[i]) && J.W[i].K == J.W[0].K && J.W[i].split == J.W[0].split && ((J.yb[i] == nullptr) != (J.yf[i] == nullptr)),
               "w4_gemm_multi: shapes / outputs");
  launch_gemm<false>(J, x, ldx, M, nullptr, 0, st);
}

void w4_gemm_gu(const W4Mat& Gm, const W4Mat& Um, const bf16* x, int ldx, int M, bf16* act, int lda, cudaStream_t st) {
  HIVE_CHECK(M >= 1 && Gm.N == Um.N && Gm.K == Um.K && Gm.split == Um.split && Gm.N % (gm_rows_per_block() / 2) == 0 && Gm.K % 64 == 0 &&
                 Gm.split <= KW_MAX, "w4_gemm_gu shapes");
  W4Multi J;
  J.n = 2; J.W[0] = Gm; J.W[1] = Um;
  launch_gemm<true>(J, x, ldx, M, act, lda, st);
}

void w4_gemv(const W4Mat& W, const bf16* x, int ldx, int M, bf16* yb, float* yf, int ldy, float* ws, cudaStream_t st) {
  (void)ws;
  W4Multi J;
  J.W[0] = W; J.yb[0] = yb; J.yf[0] = yf; J.ldy[0] = ldy; J.n = 1;
  w4_gemv_multi(J, x, ldx, M, st);
}

void w4_gemv_gu(const W4Mat& G, const W4Mat& U, const bf16* x, int ldx, int M, bf16* act, int lda, float* ws, cudaStream_t st) {
  (void)ws;
  HIVE_CHECK(M >= 1 && M <= kGemvRows && G.N == U.N && G.K == U.K && G.split == U.split && G.N % 16 == 0 && G.K % 64 == 0, "w4_gemv_gu shapes");
  const int KW = G.split;
  const size_t sm1 = KW > 1 ? (size_t)KW * 2 * 4 * 32 * sizeof(float) : 0;
  const int nt = (M + 7) / 8;
  const dim3 g(G.N / 16), b(KW * 32);
  if (nt == 1) launch_pdl(w4_gemv_gu_kernel<1>, g, b, sm1, st, G, U, x, ldx, M, act, lda);
  else if (nt == 2) launch_pdl(w4_gemv_gu_kernel<2>, g, b, sm1 * 2, st, G, U, x, ldx, M, act, lda);
  else if (nt == 3) launch_pdl(w4_gemv_gu_kernel<3>, g, b, sm1 * 3, st, G, U, x, ldx, M, act, lda);
  else launch_pdl(w4_gemv_gu_kernel<4>, g, b, sm1 * 4, st, G, U, x, ldx, M, act, lda);
}

void w4_repack(const uint32_t* q_ck, const bf16* s_ck, int N, int K, uint32_t* q, uint32_t* s, cudaStream_t st) {
  HIVE_CHECK(N % 16 == 0 && K % 64 == 0, "w4_repack shape");
  const size_t n = (size_t)N * (K / 8);
  w4_repack_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(q_ck, s_ck, N, K, q, s);
}

void w4_dequant(const W4Mat& W, bf16* out, cudaStream_t st) {
  const size_t words = (size_t)W.N * (W.K / 8);
  w4_dequant_kernel<<<(unsigned)((words + 255) / 256), 256, 0, st>>>(W, out);
}

}  // namespace hive::gemma::k
