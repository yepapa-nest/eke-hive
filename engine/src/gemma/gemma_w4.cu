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
#include <algorithm>
#include <cmath>

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
