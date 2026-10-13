// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Full-attention + QSA indexer kernels of Qwen3.8-Flash-Next — see hive/qwen/qsa.h for the math contract.
//
// Kernels (all row-table driven: every query row carries its own cache pointers and position, so prefill chunks, verify rows and
//   decode rows of several sequences use the same code):
//   qsa_prep_kernel      block per row, one warp per head unit (24 q heads, 2 k heads, 4 indexer heads): (1 + w) RMS norm, bf16 round,
//                        NeoX rope on dims 0..63, bf16 round → q / iq outputs and k into the cache; v and the raw indexer key are copied.
//   qsa_compress_kernel  warp per row whose token completes a 4-token group: fp32 mean of the 4 raw keys (bf16 round), (1 + w) norm (bf16
//                        round), rope at the group's first token's coordinates (bf16 round) → gk[G]. A separate launch, so the members
//                        written by qsa_prep_kernel in the same call are visible.
//   qsa_score_kernel     (8-row tile, 1024-group chunk) blocks, warp per group: score = Σ_h relu(iq_h · gk_G) / sqrt(128) in fp32
//                        (warp-reduced in a fixed order — deterministic). Rows with ≤ 512 visible groups are skipped (dense).
//   top-k                hive::k::topk_select_rows_t (radix select, ascending ids, −1 padded, ties → smaller id).
//   qsa_expand_kernel    group ids → token positions + the open group's tail; dense rows get every token ≤ pos.
//   qsa_attn_kernel      block per (row, kv head): the 12 query heads of that kv head, one warp each; selected positions are gathered in
//                        tiles of 32 into shared memory (K rows padded to an odd word stride → conflict-free per-token dot products), fp32
//                        online softmax, P·V with each lane owning 8 output dims; out = bf16(bf16(attn) · sigmoid(gate)).
//                        CUDA-core version: a tensor-core variant (mma over a [16 heads × 32 tokens] tile, K/V staged with cp.async) is the
//                        obvious next step for long prefills.
// Rounding points follow the reference's bf16 module boundaries: norm output bf16, rope output bf16, pooled mean bf16; scores, softmax and
//   P·V accumulate in fp32.
// RoPE angles: (float)coordinate · inv_freq in fp32, then sincosf (the accurate libm version, not __sincosf — the fast intrinsic loses
//   accuracy for the large arguments of 262K-token positions); inv_freq = 1 / θ^(2i/64) computed in fp32 on the host (powf), as a
//   framework would build its cos/sin cache.
#include "hive/qwen/qsa.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "hive/common.h"
#include "hive/qwen/fn_pdl.h"
#include "hive/model_kernels.h"

namespace hive::qwen {

namespace {

constexpr int D = kQsaDim, DI = kQsaIdxDim, R2 = kQsaRot / 2;
constexpr int kHeadsPerKv = kQsaHeads / kQsaKvHeads;   // 12
constexpr int kKvW = kQsaKvHeads * kQsaDim;            // 512
constexpr float kIdxScale = 11.313708498984761f;       // sqrt(128) (divided, as the reference does)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float rb(float x) { return __bfloat162float(__float2bfloat16(x)); }

// (1 + w) RMS norm of a head held lane-strided (element d = lane + 32·j, J values per lane), rounded to bf16.
template <int J>
__device__ __forceinline__ void norm_head(float (&x)[J], const bf16* __restrict__ w, float eps, int lane) {
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < J; ++j) ss = fmaf(x[j], x[j], ss);
  ss = warp_sum(ss);
  const float r = 1.0f / sqrtf(ss / (float)(32 * J) + eps);
#pragma unroll
  for (int j = 0; j < J; ++j) x[j] = rb(x[j] * r * (1.0f + __bfloat162float(w[lane + 32 * j])));
}
// NeoX rope on dims 0..63 (pairs (i, i + 32), i = lane): x[0] = dim lane, x[1] = dim lane + 32. Rounded to bf16.
template <int J>
__device__ __forceinline__ void rope_head(float (&x)[J], const QsaRope& rp, const int32_t* __restrict__ coord3, int lane) {
  const float ang = (float)coord3[rp.axis[lane]] * rp.inv_freq[lane];
  float s, c;
  sincosf(ang, &s, &c);
  const float a = x[0], b = x[1];
  x[0] = rb(a * c - b * s);
  x[1] = rb(b * c + a * s);
}

// ---- prep --------------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) qsa_prep_kernel(QsaWeights w, QsaRope rp, const QsaRow* __restrict__ rows, const bf16* __restrict__ proj, int ld,
                                                       bf16* __restrict__ q, bf16* __restrict__ iq) {
  pdl_wait();
  const int r = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const QsaRow row = rows[r];
  const bf16* pr = proj + (size_t)r * ld;
  const int32_t* coord = row.c.rope + (size_t)row.pos * 3;
  // v and the raw indexer key: plain copies
  for (int i = tid; i < kKvW; i += blockDim.x) row.c.v[(size_t)row.pos * kKvW + i] = pr[kQsaVOff + i];
  for (int i = tid; i < DI; i += blockDim.x) row.c.ik[(size_t)row.pos * DI + i] = pr[kQsaIkOff + i];
  // head units: 0..23 query heads, 24..25 key heads, 26..29 indexer query heads
  for (int u = warp; u < kQsaHeads + kQsaKvHeads + kQsaIdxHeads; u += 8) {
    if (u < kQsaHeads + kQsaKvHeads) {
      const bool isq = u < kQsaHeads;
      const bf16* src = isq ? pr + (size_t)u * 2 * D : pr + kQsaKOff + (size_t)(u - kQsaHeads) * D;
      float x[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) x[j] = __bfloat162float(src[lane + 32 * j]);
      norm_head<8>(x, isq ? w.q_norm : w.k_norm, w.eps, lane);
      rope_head<8>(x, rp, coord, lane);
      bf16* dst = isq ? q + ((size_t)r * kQsaHeads + u) * D : row.c.k + ((size_t)row.pos * kQsaKvHeads + (u - kQsaHeads)) * D;
#pragma unroll
      for (int j = 0; j < 8; ++j) dst[lane + 32 * j] = __float2bfloat16(x[j]);
    } else {
      const int hi = u - kQsaHeads - kQsaKvHeads;
      const bf16* src = pr + kQsaIqOff + (size_t)hi * DI;
      float x[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) x[j] = __bfloat162float(src[lane + 32 * j]);
      norm_head<4>(x, w.iq_norm, w.eps, lane);
      rope_head<4>(x, rp, coord, lane);
      bf16* dst = iq + ((size_t)r * kQsaIdxHeads + hi) * DI;
#pragma unroll
      for (int j = 0; j < 4; ++j) dst[lane + 32 * j] = __float2bfloat16(x[j]);
    }
  }
}

__global__ void __launch_bounds__(256) qsa_compress_kernel(QsaWeights w, QsaRope rp, const QsaRow* __restrict__ rows, int n) {
  pdl_wait();
  const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= n) return;
  const QsaRow row = rows[r];
  if ((row.pos & 3) != 3) return;
  const int G = row.pos >> 2;
  const bf16* ik = row.c.ik + (size_t)(4 * G) * DI;
  float x[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int d = lane + 32 * j;
    float s = __bfloat162float(ik[d]);
    s += __bfloat162float(ik[DI + d]);
    s += __bfloat162float(ik[2 * DI + d]);
    s += __bfloat162float(ik[3 * DI + d]);
    x[j] = rb(s * 0.25f);
  }
  norm_head<4>(x, w.ik_norm, w.eps, lane);
  rope_head<4>(x, rp, row.c.rope + (size_t)(4 * G) * 3, lane);
  bf16* dst = row.c.gk + (size_t)G * DI;
#pragma unroll
  for (int j = 0; j < 4; ++j) dst[lane + 32 * j] = __float2bfloat16(x[j]);
}

// ---- select ------------------------------------------------------------------------------------------------------------------------
constexpr int kScoreRows = 8, kScoreChunk = 1024;

// T_rows[i] = visible groups of row r0 + i when it needs scoring (> 512), else 0
__global__ void qsa_visible_kernel(const QsaRow* __restrict__ rows, int r0, int nb, int32_t* __restrict__ T_rows) {
  pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  const int V = (rows[r0 + i].pos + 1) / kQsaRatio;
  T_rows[i] = V > kQsaTopBlocks ? V : 0;
}

__global__ void __launch_bounds__(256) qsa_score_kernel(const QsaRow* __restrict__ rows, int r0, int nb, const bf16* __restrict__ iq, float* __restrict__ score,
                                                        int Gs, int chunk) {
  pdl_wait();
  __shared__ float sq[kScoreRows][kQsaIdxHeads][DI];
  __shared__ int sV[kScoreRows];
  __shared__ const bf16* sgk[kScoreRows];
  const int tile0 = blockIdx.x * kScoreRows, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nt = min(kScoreRows, nb - tile0);
  if (tid < kScoreRows) {
    if (tid < nt) {
      const QsaRow row = rows[r0 + tile0 + tid];
      const int V = (row.pos + 1) / kQsaRatio;
      sV[tid] = V > kQsaTopBlocks ? V : 0;
      sgk[tid] = row.c.gk;
    } else { sV[tid] = 0; sgk[tid] = nullptr; }
  }
  for (int i = tid; i < nt * kQsaIdxHeads * DI; i += blockDim.x)
    (&sq[0][0][0])[i] = __bfloat162float(iq[(size_t)(r0 + tile0) * kQsaIdxHeads * DI + i]);
  __syncthreads();
  int vmax = 0;
  for (int t = 0; t < nt; ++t) vmax = max(vmax, sV[t]);
  const int g0 = blockIdx.y * chunk;
  if (g0 >= vmax) return;
  const int g1 = min(g0 + chunk, vmax);
  for (int G = g0 + warp; G < g1; G += 8) {
    const bf16* prev = nullptr;
    float k[4] = {0.f, 0.f, 0.f, 0.f};
    for (int t = 0; t < nt; ++t) {
      if (G >= sV[t]) continue;
      if (sgk[t] != prev) {  // rows of one sequence share the group keys: load once per tile run
        prev = sgk[t];
        const bf16* g = prev + (size_t)G * DI;
#pragma unroll
        for (int j = 0; j < 4; ++j) k[j] = __bfloat162float(g[lane + 32 * j]);
      }
      float tot = 0.f;
#pragma unroll
      for (int h = 0; h < kQsaIdxHeads; ++h) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) s = fmaf(sq[t][h][lane + 32 * j], k[j], s);
        s = warp_sum(s);
        tot += fmaxf(s, 0.f);
      }
      if (lane == 0) score[(size_t)(tile0 + t) * Gs + G] = tot / kIdxScale;
    }
  }
}

// idx row (width kQsaSel): dense rows → 0..pos; scored rows → expanded top groups + open-group tail
__global__ void qsa_expand_kernel(const QsaRow* __restrict__ rows, int r0, int nb, const int32_t* __restrict__ top, int32_t* __restrict__ idx) {
  pdl_wait();
  const int i = blockIdx.x, tid = threadIdx.x;
  if (i >= nb) return;
  const int pos = rows[r0 + i].pos;
  const int V = (pos + 1) / kQsaRatio;
  int32_t* out = idx + (size_t)(r0 + i) * kQsaSel;
  if (V <= kQsaTopBlocks) {
    for (int j = tid; j < kQsaSel; j += blockDim.x) out[j] = j <= pos ? j : -1;
    return;
  }
  const int32_t* tp = top + (size_t)i * kQsaTopBlocks;
  for (int j = tid; j < kQsaTopBlocks * kQsaRatio; j += blockDim.x) {
    const int G = tp[j / kQsaRatio];
    out[j] = G >= 0 ? G * kQsaRatio + (j % kQsaRatio) : -1;
  }
  if (tid < kQsaRatio - 1) {
    const int t = V * kQsaRatio + tid;
    out[kQsaTopBlocks * kQsaRatio + tid] = t <= pos ? t : -1;
  }
}

struct SelLayout { size_t score, trows, top, total; int Gs; };
// Decode / verify scoring (≤ 8 rows of one sequence): thread = (group, indexer head) — 32 groups × 4 heads per block; the rows' queries in
//   shared memory, the group key streamed in 16-byte pieces once for every row; head sums by two shuffles. The warp-per-group kernel left
//   122-173 µs per call at 64K context (a warp walking its groups one after another; nsys 2026-10-10). The dot products add in a different
//   order than that kernel: only near-equal groups at the top-512 boundary can trade places.
constexpr int kSdG = 32;
__global__ void __launch_bounds__(128) qsa_score_dec_kernel(const QsaRow* __restrict__ rows, int r0, int nb, const bf16* __restrict__ iq,
                                                            float* __restrict__ score, int Gs) {
  pdl_wait();
  __shared__ float sq[kScoreRows][kQsaIdxHeads][DI];
  __shared__ int sV[kScoreRows];
  const int tid = threadIdx.x, nt = min(kScoreRows, nb);
  for (int i = tid; i < nt * kQsaIdxHeads * DI; i += blockDim.x) (&sq[0][0][0])[i] = __bfloat162float(iq[(size_t)r0 * kQsaIdxHeads * DI + i]);
  if (tid < kScoreRows) {
    int V = 0;
    if (tid < nt) { V = (rows[r0 + tid].pos + 1) / kQsaRatio; V = V > kQsaTopBlocks ? V : 0; }
    sV[tid] = V;
  }
  __syncthreads();
  const int h = tid & 3, G = blockIdx.x * kSdG + (tid >> 2);
  int vmax = 0;
  for (int t = 0; t < nt; ++t) vmax = max(vmax, sV[t]);
  if (G >= vmax) return;
  const bf16* gk = rows[r0].c.gk + (size_t)G * DI;
  float acc[kScoreRows];
#pragma unroll
  for (int t = 0; t < kScoreRows; ++t) acc[t] = 0.f;
#pragma unroll 4
  for (int d = 0; d < DI; d += 8) {
    const uint4 kv = *reinterpret_cast<const uint4*>(gk + d);
    const bf16* ke = reinterpret_cast<const bf16*>(&kv);
    float kf[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) kf[j] = __bfloat162float(ke[j]);
#pragma unroll
    for (int t = 0; t < kScoreRows; ++t) {
      if (t >= nt) break;
      float a = acc[t];
#pragma unroll
      for (int j = 0; j < 8; ++j) a = fmaf(sq[t][h][d + j], kf[j], a);
      acc[t] = a;
    }
  }
#pragma unroll
  for (int t = 0; t < kScoreRows; ++t) {
    if (t >= nt) break;
    float v = fmaxf(acc[t], 0.f);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    if (h == 0 && G < sV[t]) score[(size_t)t * Gs + G] = v / kIdxScale;
  }
}

// Tensor-core scoring (prefill: 16 rows per block, all of one sequence): A = the 16 rows' indexer queries as 64 rows (row·4 + head) × 128,
//   B = 64 group keys × 128 per tile (cp.async, double buffered), S = A·Bᵀ by mma.m16n8k16 (warp w: rows 4w..4w+3, all 4 heads); then
//   score = Σ_h relu(S) / sqrt(128) (heads added as (h0 + h1) + (h2 + h3)). The CUDA-core kernel adds the heads in order and its dot
//   products in a warp-reduction order: scores differ in the last bits, which can only reorder near-equal groups at the top-512 boundary.
constexpr int kScR = 16, kScG = 64, kScChunk = 2048, kScS = DI + 8;
__global__ void __launch_bounds__(128) qsa_score_tc_kernel(const QsaRow* __restrict__ rows, int r0, int nb, const bf16* __restrict__ iq,
                                                           float* __restrict__ score, int Gs, int chunk) {
  pdl_wait();
  __shared__ __align__(16) bf16 sa[kScR * 4 * kScS];
  __shared__ __align__(16) bf16 sb[2][kScG * kScS];
  __shared__ int sV[kScR];
  __shared__ const bf16* sgk;
  const int tile0 = blockIdx.x * kScR, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nt = min(kScR, nb - tile0);
  if (tid < kScR) {
    int V = 0;
    if (tid < nt) { const QsaRow row = rows[r0 + tile0 + tid]; V = (row.pos + 1) / kQsaRatio; V = V > kQsaTopBlocks ? V : 0; if (tid == 0) sgk = row.c.gk; }
    sV[tid] = V;
  }
  for (int i = tid; i < kScR * 4 * (DI / 8); i += 128) {
    const int r = i / (DI / 8), c = (i % (DI / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r / 4 < nt) v = *reinterpret_cast<const uint4*>(iq + ((size_t)(r0 + tile0) * kQsaIdxHeads + r) * DI + c);
    *reinterpret_cast<uint4*>(sa + r * kScS + c) = v;
  }
  __syncthreads();
  int vmax = 0;
  for (int t = 0; t < nt; ++t) vmax = max(vmax, sV[t]);
  const int g0 = blockIdx.y * chunk;
  if (g0 >= vmax) return;
  const int g1 = min(g0 + chunk, vmax);
  const bf16* gk = sgk;
  unsigned af[8][4];
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(sa + (warp * 16 + (lane & 15)) * kScS + ks * 16 + (lane >> 4) * 8);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(af[ks][0]), "=r"(af[ks][1]), "=r"(af[ks][2]), "=r"(af[ks][3]) : "r"(a));
  }
  auto issue = [&](int G0, int buf) {
    for (int i = tid; i < kScG * (DI / 8); i += 128) {
      const int gr = i / (DI / 8), c = (i % (DI / 8)) * 8, G = G0 + gr;
      const unsigned d = (unsigned)__cvta_generic_to_shared(sb[buf] + gr * kScS + c);
      const bf16* src = gk + (size_t)(G < g1 ? G : g0) * DI + c;
      const int n = G < g1 ? 16 : 0;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  const int ntile = (g1 - g0 + kScG - 1) / kScG;
  issue(g0, 0);
  const int g = lane >> 2, q2 = (lane & 3) * 2;
  for (int t = 0; t < ntile; ++t) {
    const int buf = t & 1, G0 = g0 + t * kScG;
    if (t + 1 < ntile) { issue(G0 + kScG, buf ^ 1); asm volatile("cp.async.wait_group 1;\n" ::); }
    else asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
#pragma unroll
    for (int np = 0; np < 4; ++np) {   // 16 groups per pass
      float c[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
      for (int ks = 0; ks < 8; ks += 2) {
        unsigned b[4], b2[4];
        const bf16* base = sb[buf] + (np * 16 + (lane & 7)) * kScS;
        const unsigned a0 = (unsigned)__cvta_generic_to_shared(base + (ks + (lane >> 4)) * 16 + ((lane >> 3) & 1) * 8);
        const unsigned a1 = (unsigned)__cvta_generic_to_shared(base + 8 * kScS + (ks + (lane >> 4)) * 16 + ((lane >> 3) & 1) * 8);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(a0));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(b2[0]), "=r"(b2[1]), "=r"(b2[2]), "=r"(b2[3]) : "r"(a1));
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                       : "+f"(c[0][0]), "+f"(c[0][1]), "+f"(c[0][2]), "+f"(c[0][3])
                       : "r"(af[ks + kk][0]), "r"(af[ks + kk][1]), "r"(af[ks + kk][2]), "r"(af[ks + kk][3]), "r"(b[kk * 2]), "r"(b[kk * 2 + 1]));
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                       : "+f"(c[1][0]), "+f"(c[1][1]), "+f"(c[1][2]), "+f"(c[1][3])
                       : "r"(af[ks + kk][0]), "r"(af[ks + kk][1]), "r"(af[ks + kk][2]), "r"(af[ks + kk][3]), "r"(b2[kk * 2]), "r"(b2[kk * 2 + 1]));
        }
      }
      // relu, then the 4 heads (lane bits 2..3 of g) summed by two shuffles
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float v = fmaxf(c[h][j], 0.f);
          v += __shfl_xor_sync(0xffffffffu, v, 4);
          v += __shfl_xor_sync(0xffffffffu, v, 8);
          c[h][j] = v;
        }
      if ((g & 3) == 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int hi = 0; hi < 2; ++hi) {
            const int rr = warp * 4 + g / 4 + hi * 2;   // M row g (+8) → query row 4w + g/4 (+2)
            if (rr >= nt) continue;
            const int V = sV[rr];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              const int G = G0 + np * 16 + h * 8 + q2 + j;
              if (G < V) score[(size_t)(tile0 + rr) * Gs + G] = c[h][hi * 2 + j] / kIdxScale;
            }
          }
      }
    }
    __syncthreads();
  }
}

SelLayout sel_layout(int n, int ctx) {
  SelLayout L{};
  L.Gs = std::max(1, ctx / kQsaRatio + 1);
  auto al = [](size_t x) { return (x + 255) / 256 * 256; };
  L.score = 0;
  L.trows = al((size_t)n * L.Gs * 4);
  L.top = L.trows + al((size_t)n * 4);
  L.total = L.top + al((size_t)n * kQsaTopBlocks * 4) + 256;
  return L;
}

// ---- attention ---------------------------------------------------------------------------------------------------------------------
constexpr int kTile = 32, kKS = D + 2;   // K row stride (bf16): 258 = 129 words → consecutive rows start on different banks
constexpr int kAttnThreads = kHeadsPerKv * 32;

__global__ void __launch_bounds__(kAttnThreads) qsa_attn_kernel(const QsaRow* __restrict__ rows, const bf16* __restrict__ q, const bf16* __restrict__ proj,
                                                                int ld, const int32_t* __restrict__ idx, bf16* __restrict__ out) {
  pdl_wait();
  __shared__ float sq[kHeadsPerKv][D];
  __shared__ __align__(16) bf16 sk[kTile * kKS];
  __shared__ __align__(16) bf16 sv[kTile * D];
  __shared__ int sidx[kTile];
  __shared__ int sany;
  const int r = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const QsaRow row = rows[r];
  const int h = g * kHeadsPerKv + warp;   // this warp's query head
  for (int i = tid; i < kHeadsPerKv * D; i += blockDim.x)
    (&sq[0][0])[i] = __bfloat162float(q[((size_t)r * kQsaHeads + g * kHeadsPerKv) * D + i]);
  const int32_t* ir = idx + (size_t)r * kQsaSel;
  float m = -INFINITY, l = 0.f, o[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = 0.f;
  for (int t0 = 0; t0 < kQsaSel; t0 += kTile) {
    __syncthreads();  // previous tile consumed (and sq loaded on the first pass)
    if (tid == 0) sany = 0;
    __syncthreads();
    if (tid < kTile) {
      const int p = t0 + tid < kQsaSel ? ir[t0 + tid] : -1;
      sidx[tid] = p;
      if (p >= 0) sany = 1;
    }
    __syncthreads();
    if (!sany) continue;  // an all-empty tile (the −1 tail of a dense row) is skipped
    // gather K / V rows of kv head g: 32 tokens × 256, 32-bit words
    for (int i = tid; i < kTile * (D / 2); i += blockDim.x) {
      const int t = i / (D / 2), w2 = i % (D / 2);
      const int p = sidx[t];
      uint32_t kw = 0, vw = 0;
      if (p >= 0) {
        kw = reinterpret_cast<const uint32_t*>(row.c.k + ((size_t)p * kQsaKvHeads + g) * D)[w2];
        vw = reinterpret_cast<const uint32_t*>(row.c.v + ((size_t)p * kQsaKvHeads + g) * D)[w2];
      }
      reinterpret_cast<uint32_t*>(sk + t * kKS)[w2] = kw;
      reinterpret_cast<uint32_t*>(sv + t * D)[w2] = vw;
    }
    __syncthreads();
    // scores: lane = token of the tile
    float s = -INFINITY;
    if (sidx[lane] >= 0) {
      const bf16* kr = sk + lane * kKS;
      float acc = 0.f;
      for (int d = 0; d < D; d += 2) {
        const __nv_bfloat162 kk = *reinterpret_cast<const __nv_bfloat162*>(kr + d);
        acc = fmaf(sq[warp][d], __bfloat162float(kk.x), acc);
        acc = fmaf(sq[warp][d + 1], __bfloat162float(kk.y), acc);
      }
      s = acc * (1.0f / 16.0f);
    }
    const float mt = warp_max(s);
    if (mt == -INFINITY) continue;
    const float mn = fmaxf(m, mt);
    const float corr = m == -INFINITY ? 0.f : __expf(m - mn);
    const float p = s == -INFINITY ? 0.f : __expf(s - mn);
    l = l * corr + warp_sum(p);
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] *= corr;
    for (int t = 0; t < kTile; ++t) {
      const float pt = __shfl_sync(0xffffffffu, p, t);
      if (pt == 0.f) continue;
      const bf16* vr = sv + t * D;
#pragma unroll
      for (int j = 0; j < 8; ++j) o[j] = fmaf(pt, __bfloat162float(vr[lane + 32 * j]), o[j]);
    }
    m = mn;
  }
  const bf16* gate = proj + (size_t)r * ld + (size_t)h * 2 * D + D;
  bf16* dst = out + ((size_t)r * kQsaHeads + h) * D;
  const float inv = l > 0.f ? 1.0f / l : 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int d = lane + 32 * j;
    const float a = rb(o[j] * inv);
    const float gt = __bfloat162float(gate[d]);
    dst[d] = __float2bfloat16(a * (1.0f / (1.0f + expf(-gt))));
  }
}


// ---- attention, tensor cores (prefill rows) -------------------------------------------------------------------------------------------
// block = (row, kv head), 4 warps; the 12 query heads of the kv head are the M = 16 rows of mma.m16n8k16 (rows 12..15 zero); selected
//   positions in tiles of 32, K / V rows gathered with cp.async (double buffered; a −1 position is zero-filled and masked).
//   S = Q·Kᵀ: warp w owns tokens 8w..8w+7 (16 k-steps over the 256 dims, Q fragments held in registers) → fp32 S / 16 in shared memory;
//   warp 0 runs the online softmax (fp32 max / Σexp, __expf as the CUDA-core kernel) and writes P in bf16; O += P·V: warp w owns dims
//   64w..64w+63 (V fragments by ldmatrix.trans). Differences from qsa_attn_kernel: P enters the P·V product rounded to bf16 and the
//   dot products accumulate in the mma order — validated against the oracle like every other path.
constexpr int kTT = 32, kQS = D + 8, kSS = kTT + 4, kPS = kTT + 8;
struct TcSmem {
  bf16 q[16 * kQS];
  bf16 k[2][kTT * kQS];
  bf16 v[2][kTT * kQS];
  float s[16 * kSS];
  bf16 p[16 * kPS];
  int idx[2][kTT];
  float m[16], l[16], corr[16];
};
__device__ __forceinline__ void tc_cp16(void* dst, const void* src, bool valid) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  const int n = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
}
__device__ __forceinline__ void tc_ldm4(unsigned (&r)[4], const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void tc_ldm4t(unsigned (&r)[4], const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void tc_mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(128) qsa_attn_tc_kernel(const QsaRow* __restrict__ rows, const bf16* __restrict__ q, const bf16* __restrict__ proj,
                                                          int ld, const int32_t* __restrict__ idx, bf16* __restrict__ out, int splits, float* __restrict__ part) {
  extern __shared__ __align__(16) unsigned char tc_raw[];
  TcSmem& S = *reinterpret_cast<TcSmem*>(tc_raw);
  const int r = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const QsaRow row = rows[r];
  const int32_t* ir = idx + (size_t)r * kQsaSel;
  const bf16* kb = row.c.k + (size_t)g * D;
  const bf16* vb = row.c.v + (size_t)g * D;
  // Q: 12 heads × 256 (rows 12..15 zero)
  for (int i = tid; i < 16 * (D / 8); i += 128) {
    const int hr = i / (D / 8), c = (i % (D / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (hr < kHeadsPerKv) v = *reinterpret_cast<const uint4*>(q + ((size_t)r * kQsaHeads + g * kHeadsPerKv + hr) * D + c);
    *reinterpret_cast<uint4*>(S.q + hr * kQS + c) = v;
  }
  if (tid < 16) { S.m[tid] = -INFINITY; S.l[tid] = 0.f; }
  const int ntiles_all = (kQsaSel + kTT - 1) / kTT;
  // split-KV (decode / verify rows: few blocks otherwise): block z takes tiles [t_lo, t_hi) and writes partial (m, l, o) for the combine
  const int t_lo = (int)((long)blockIdx.z * ntiles_all / splits), t_hi = (int)((long)(blockIdx.z + 1) * ntiles_all / splits);
  const int ntiles = t_hi - t_lo;
  // tile loader: 32 tokens × (K, V) × 32 chunks of 16 B = 2048 chunks, 16 per thread
  auto issue = [&](int t, int buf) {
    const int t0 = t * kTT;
    if (tid < kTT) S.idx[buf][tid] = t0 + tid < kQsaSel ? ir[t0 + tid] : -1;
#pragma unroll 4
    for (int i = tid; i < kTT * 64; i += 128) {
      const int tok = i >> 6, part = (i >> 5) & 1, c = (i & 31) * 8;
      const int pp = t0 + tok < kQsaSel ? __ldg(ir + t0 + tok) : -1;
      const bf16* src = (part ? vb : kb) + (size_t)(pp >= 0 ? pp : 0) * kKvW + c;
      tc_cp16((part ? S.v[buf] : S.k[buf]) + tok * kQS + c, src, pp >= 0);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  if (ntiles > 0) issue(t_lo, 0);
  __syncthreads();   // Q staged
  unsigned qa[16][4];
#pragma unroll
  for (int ks = 0; ks < 16; ++ks) tc_ldm4(qa[ks], S.q + (lane & 15) * kQS + ks * 16 + (lane >> 4) * 8);
  float o[8][4];
#pragma unroll
  for (int a = 0; a < 8; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) o[a][b] = 0.f;
  const int gq = lane >> 2, q2 = (lane & 3) * 2;
  for (int t = 0; t < ntiles; ++t) {
    const int buf = t & 1;
    if (t + 1 < ntiles) { issue(t_lo + t + 1, buf ^ 1); asm volatile("cp.async.wait_group 1;\n" ::); }
    else asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    const int any = __syncthreads_or(tid < kTT && S.idx[buf][tid] >= 0);
    if (!any) continue;
    // S = Q · Kᵀ for this warp's 8 tokens
    float sc[4] = {0.f, 0.f, 0.f, 0.f};
    const bf16* kt = S.k[buf] + (size_t)(warp * 8) * kQS;
#pragma unroll
    for (int ks = 0; ks < 16; ks += 2) {
      unsigned b[4];
      tc_ldm4(b, kt + (lane & 7) * kQS + (ks + (lane >> 4)) * 16 + ((lane >> 3) & 1) * 8);
      tc_mma(sc, qa[ks], b[0], b[1]);
      tc_mma(sc, qa[ks + 1], b[2], b[3]);
    }
    {
      const int c = warp * 8 + q2;
      S.s[gq * kSS + c] = sc[0] * (1.0f / 16.0f); S.s[gq * kSS + c + 1] = sc[1] * (1.0f / 16.0f);
      S.s[(gq + 8) * kSS + c] = sc[2] * (1.0f / 16.0f); S.s[(gq + 8) * kSS + c + 1] = sc[3] * (1.0f / 16.0f);
    }
    __syncthreads();
    if (warp == 0) {   // lane: row lane / 2, tokens 16·(lane & 1) … + 15
      const int hr = lane >> 1, h0 = (lane & 1) * 16;
      float mt = -INFINITY;
#pragma unroll
      for (int j = 0; j < 16; ++j) if (S.idx[buf][h0 + j] >= 0) mt = fmaxf(mt, S.s[hr * kSS + h0 + j]);
      mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
      const float mo = S.m[hr], mn = fmaxf(mo, mt);
      const float corr = mo == -INFINITY ? 0.f : __expf(mo - mn);
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < 16; j += 2) {
        const bool v0 = S.idx[buf][h0 + j] >= 0, v1 = S.idx[buf][h0 + j + 1] >= 0;
        const float p0 = v0 ? __expf(S.s[hr * kSS + h0 + j] - mn) : 0.f, p1 = v1 ? __expf(S.s[hr * kSS + h0 + j + 1] - mn) : 0.f;
        sum += p0 + p1;
        *reinterpret_cast<__nv_bfloat162*>(S.p + hr * kPS + h0 + j) = __floats2bfloat162_rn(p0, p1);
      }
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      __syncwarp();
      if ((lane & 1) == 0) { S.l[hr] = S.l[hr] * corr + sum; S.m[hr] = mn; S.corr[hr] = corr; }
    }
    __syncthreads();
    const float c0 = S.corr[gq], c1 = S.corr[gq + 8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) { o[nt][0] *= c0; o[nt][1] *= c0; o[nt][2] *= c1; o[nt][3] *= c1; }
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      unsigned pa[4];
      tc_ldm4(pa, S.p + (lane & 15) * kPS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        unsigned b[4];
        tc_ldm4t(b, S.v[buf] + (size_t)(ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * kQS + warp * 64 + np * 16 + (lane >> 4) * 8);
        tc_mma(o[np * 2], pa, b[0], b[1]);
        tc_mma(o[np * 2 + 1], pa, b[2], b[3]);
      }
    }
    __syncthreads();   // the buffer is refilled by the issue two tiles on, S / P by the next tile
  }
  if (splits > 1) {   // partial: per (row, kv head, split) m[16], l[16], o[16][256] (unnormalised)
    float* pp = part + (((size_t)r * kQsaKvHeads + g) * splits + blockIdx.z) * (32 + 16 * D);
    if (tid < 16) { pp[tid] = S.m[tid]; pp[16 + tid] = S.l[tid]; }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int d = warp * 64 + nt * 8 + q2;
        pp[32 + (size_t)(gq + hh * 8) * D + d] = o[nt][hh * 2];
        pp[32 + (size_t)(gq + hh * 8) * D + d + 1] = o[nt][hh * 2 + 1];
      }
    return;
  }
  // out = bf16(bf16(o / l) · sigmoid(gate)) for the 12 real heads
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const int hr = gq + hh * 8;
    if (hr >= kHeadsPerKv) continue;
    const float L = S.l[hr], inv = L > 0.f ? 1.0f / L : 0.f;
    const int h = g * kHeadsPerKv + hr;
    const bf16* gate = proj + (size_t)r * ld + (size_t)h * 2 * D + D;
    bf16* dst = out + ((size_t)r * kQsaHeads + h) * D;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int d = warp * 64 + nt * 8 + q2;
      const float a0 = rb(o[nt][hh * 2] * inv), a1 = rb(o[nt][hh * 2 + 1] * inv);
      const float g0 = __bfloat162float(gate[d]), g1 = __bfloat162float(gate[d + 1]);
      *reinterpret_cast<__nv_bfloat162*>(dst + d) = __floats2bfloat162_rn(a0 * (1.0f / (1.0f + expf(-g0))), a1 * (1.0f / (1.0f + expf(-g1))));
    }
  }
}
// combine of the split-KV partials, splits in order: block = (row, head), thread = 2 dims
__global__ void qsa_attn_combine_kernel(const float* __restrict__ part, int splits, const bf16* __restrict__ proj, int ld, bf16* __restrict__ out) {
  const int r = blockIdx.x, h = blockIdx.y, g = h / kHeadsPerKv, hr = h % kHeadsPerKv;
  const float* base = part + ((size_t)r * kQsaKvHeads + g) * splits * (32 + 16 * D);
  float M = -INFINITY;
  for (int s2 = 0; s2 < splits; ++s2) M = fmaxf(M, base[(size_t)s2 * (32 + 16 * D) + hr]);
  float L = 0.f, w[32];
  for (int s2 = 0; s2 < splits && s2 < 32; ++s2) {
    const float* p = base + (size_t)s2 * (32 + 16 * D);
    w[s2] = p[hr] == -INFINITY ? 0.f : __expf(p[hr] - M);
    L += p[16 + hr] * w[s2];
  }
  const float inv = L > 0.f ? 1.0f / L : 0.f;
  const bf16* gate = proj + (size_t)r * ld + (size_t)h * 2 * D + D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float o = 0.f;
    for (int s2 = 0; s2 < splits; ++s2) if (w[s2] != 0.f) o += base[(size_t)s2 * (32 + 16 * D) + 32 + (size_t)hr * D + d] * w[s2];
    const float a = rb(o * inv), gt = __bfloat162float(gate[d]);
    out[((size_t)r * kQsaHeads + h) * D + d] = __float2bfloat16(a * (1.0f / (1.0f + expf(-gt))));
  }
}

}  // namespace

void qsa_rope_tables(float theta, const int* section, bool interleaved, float* inv_freq_dev, int32_t* axis_dev, cudaStream_t st) {
  float inv[R2];
  int32_t ax[R2];
  for (int i = 0; i < R2; ++i) {
    inv[i] = 1.0f / powf(theta, (float)(2 * i) / (float)kQsaRot);
    if (interleaved) ax[i] = (i % 3 == 1 && i < 3 * section[1]) ? 1 : (i % 3 == 2 && i < 3 * section[2]) ? 2 : 0;
    else ax[i] = i < section[0] ? 0 : i < section[0] + section[1] ? 1 : 2;
  }
  CUDA_CHECK(cudaMemcpyAsync(inv_freq_dev, inv, sizeof inv, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(axis_dev, ax, sizeof ax, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaStreamSynchronize(st));  // the host arrays are on this stack frame
}

void qsa_prep(const QsaWeights& w, const QsaRope& rope, const QsaRow* rows, int n, const bf16* proj, int ld, bf16* q, bf16* iq, cudaStream_t st) {
  if (n <= 0) return;
  HIVE_CHECK(ld >= kQsaProjWidth && ld % 2 == 0, "qsa_prep: projection row stride");
  launch_pdl(qsa_prep_kernel, dim3(n), dim3(256), 0, st, w, rope, rows, proj, ld, q, iq);
  CUDA_CHECK(cudaGetLastError());
  launch_pdl(qsa_compress_kernel, dim3((n + 7) / 8), dim3(256), 0, st, w, rope, rows, n);
  CUDA_CHECK(cudaGetLastError());
}

size_t qsa_select_ws_bytes(int n, int ctx) { return sel_layout(std::max(n, 1), ctx).total; }

void qsa_select(const QsaRow* rows, int n, int max_pos, const bf16* iq, int32_t* idx, void* ws, size_t ws_bytes, cudaStream_t st, int ctx_cap, bool one_seq) {
  if (n <= 0) return;
  const int ctx = std::max(max_pos + 1, ctx_cap);
  int nb = n;
  while (nb > 1 && sel_layout(nb, ctx).total > ws_bytes) nb = std::max(1, std::min(nb - 1, (int)((double)nb * ws_bytes / sel_layout(nb, ctx).total)));
  const SelLayout L = sel_layout(nb, ctx);
  HIVE_CHECK(L.total <= ws_bytes, "qsa_select: workspace below qsa_select_ws_bytes(1, ctx)");
  uint8_t* w8 = static_cast<uint8_t*>(ws);
  float* score = reinterpret_cast<float*>(w8 + L.score);
  int32_t* trows = reinterpret_cast<int32_t*>(w8 + L.trows);
  int32_t* top = reinterpret_cast<int32_t*>(w8 + L.top);
  const bool scoring = (max_pos + 1) / kQsaRatio > kQsaTopBlocks;  // otherwise every row is dense
  for (int r0 = 0; r0 < n; r0 += nb) {
    const int m = std::min(nb, n - r0);
    if (scoring) {
      launch_pdl(qsa_visible_kernel, dim3((m + 255) / 256), dim3(256), 0, st, rows, r0, m, trows);
      // groups per block: 1024, or 32 for calls of ≤ 8 rows (decode / verify: one row tile — 16 blocks at 64K context with 1024; 128 still
      //   left 122 µs per call, a warp walking 16 groups back to back — nsys 2026-10-10)
      const int sch = m <= kScoreRows ? 128 : kScoreChunk;
      // HIVE_QWEN_QSA_TC also moves the scoring of prefill calls (> 64 rows of one sequence) to tensor cores
      static const bool tc = !(getenv("HIVE_QWEN_QSA_TC") && strcmp(getenv("HIVE_QWEN_QSA_TC"), "0") == 0);
      if (tc && one_seq && n > 64) {   // prefill chunks only — on verify rows (4 of the 16 rows used) the tensor-core scorer made the
        //   64K-context decode slower (hived TPOT 19.2 → 33.6 ms, 2026-10-10)
        const int ch = kScChunk;
        launch_pdl(qsa_score_tc_kernel, dim3((m + kScR - 1) / kScR, (L.Gs + ch - 1) / ch), dim3(128), 0, st, rows, r0, m, iq, score, L.Gs, ch);
      }
      else
        if (one_seq && m <= kScoreRows)
          launch_pdl(qsa_score_dec_kernel, dim3((L.Gs + kSdG - 1) / kSdG), dim3(128), 0, st, rows, r0, m, iq, score, L.Gs);
        else
          launch_pdl(qsa_score_kernel, dim3(dim3((m + kScoreRows - 1) / kScoreRows, (L.Gs + sch - 1) / sch)), dim3(256), 0, st, rows, r0, m, iq, score, L.Gs, sch);
      CUDA_CHECK(cudaGetLastError());
      hive::k::topk_select_rows_t(score, m, trows, L.Gs, kQsaTopBlocks, L.Gs, top, kQsaTopBlocks, st);
      CUDA_CHECK(cudaGetLastError());
    }
    launch_pdl(qsa_expand_kernel, dim3(m), dim3(256), 0, st, rows, r0, m, top, idx);
    CUDA_CHECK(cudaGetLastError());
  }
}

void qsa_attention(const QsaRow* rows, int n, const bf16* q, const bf16* proj, int ld, const int32_t* idx, bf16* out, cudaStream_t st) {
  if (n <= 0) return;
  // HIVE_QWEN_QSA_TC (default on): calls of more than 64 rows (prefill chunks) on the tensor-core kernel; decode / verify rows keep the
  //   CUDA-core kernel
  static const bool tc = !(getenv("HIVE_QWEN_QSA_TC") && strcmp(getenv("HIVE_QWEN_QSA_TC"), "0") == 0);
  if (tc && n > 64) {
    static bool attr = false;
    if (!attr) { CUDA_CHECK(cudaFuncSetAttribute(qsa_attn_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(TcSmem))); attr = true; }
    qsa_attn_tc_kernel<<<dim3(n, kQsaKvHeads, 1), 128, sizeof(TcSmem), st>>>(rows, q, proj, ld, idx, out, 1, nullptr);
  } else {
    launch_pdl(qsa_attn_kernel, dim3(dim3(n, kQsaKvHeads)), dim3(kAttnThreads), 0, st, rows, q, proj, ld, idx, out);
  }
  CUDA_CHECK(cudaGetLastError());
}

size_t qsa_attention_split_ws_bytes(int n, int splits) { return (size_t)n * kQsaKvHeads * splits * (32 + 16 * D) * 4; }

void qsa_attention_split(const QsaRow* rows, int n, const bf16* q, const bf16* proj, int ld, const int32_t* idx, bf16* out, int splits, void* ws,
                         cudaStream_t st) {
  if (n <= 0) return;
  HIVE_CHECK(splits >= 2 && splits <= 32, "qsa_attention_split: splits");
  static bool attr = false;
  if (!attr) { CUDA_CHECK(cudaFuncSetAttribute(qsa_attn_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(TcSmem))); attr = true; }
  qsa_attn_tc_kernel<<<dim3(n, kQsaKvHeads, splits), 128, sizeof(TcSmem), st>>>(rows, q, proj, ld, idx, out, splits, static_cast<float*>(ws));
  qsa_attn_combine_kernel<<<dim3(n, kQsaHeads), 128, 0, st>>>(static_cast<const float*>(ws), splits, proj, ld, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::qwen
