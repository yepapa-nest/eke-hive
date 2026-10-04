// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Sparse attention v2 — tensor cores (WMMA bf16). Same semantics as v1 (sparse_attn in model_kernels.cu):
//   S = bf16 q · bf16 kv (fp32 accumulation) x scale, index -1 is -inf; softmax statistics (max, sum) from fp32 p; P is cast to bf16 and multiplied with V (fp32 accumulation)
//   · denominator = sum + exp(sink - max). Three kernels: (1) scores (global scratch S[M,H,topk] fp32), (2) row statistics, (3) P·V accumulation (WMMA) -> o.
// Unlike v1's online softmax, P is formed after the max is final, so only the accumulation order differs (it also differs from the reference kernel).
#include <mma.h>

#include "hive/model_kernels.h"
#include "hive/warp_reduce.cuh"

namespace hive::k {

namespace {
using namespace nvcuda;
constexpr int D = 512;
constexpr int HG = 32;   // heads per block (Q smem 32x512 bf16 = 32KB)
constexpr int KB = 32;   // key block (kv smem 32x512 bf16 = 32KB)
constexpr int THREADS = 256;

// KV element (dimension d): ring and chunk are bf16 as is; compressed KV is unpacked from packed fp4 (kv_pack.h) (same values as the bf16 stored form)
__device__ inline bf16 kv_elem2(const KvSources& kv, int idx, int d) {
  if (idx < kv.win) return kv.ring[(size_t)idx * D + d];
  idx -= kv.win;
  if (idx < kv.chunk_len) return kv.chunk[(size_t)idx * D + d];
  idx -= kv.chunk_len;
  return f2bf(kvp::comp_val(kv.comp + (size_t)idx * kvp::COMP_ROW, d));
}

// (1) Scores: block = (query m, head group hg, key block t0/KB). S[m][h][t] = scale·q_h·kv_t (-inf if missing)
__global__ void __launch_bounds__(THREADS) attn_scores_kernel(const bf16* __restrict__ q, int H, KvSources kv, const int32_t* __restrict__ idx,
                                                              int idx_stride, int topk, float scale, float* __restrict__ S) {
  extern __shared__ __align__(128) uint8_t smem[];
  bf16* qs = reinterpret_cast<bf16*>(smem);                 // HG×D
  bf16* ks = qs + HG * D;                                   // KB×D
  float* ss = reinterpret_cast<float*>(ks + KB * D);        // HGxKB (fp32 result)
  const int nkb = (topk + KB - 1) / KB;
  const int t0 = (blockIdx.x % nkb) * KB;
  const int rest = blockIdx.x / nkb;
  const int hg = rest % (H / HG), m = rest / (H / HG);
  const int h0 = hg * HG;
  const int tid = threadIdx.x, warp = tid >> 5;
  for (int i = tid; i < HG * D; i += THREADS) qs[i] = q[((size_t)m * H + h0) * D + i];
  const int32_t* irow = idx + (size_t)m * idx_stride;
  for (int i = tid; i < KB * D; i += THREADS) {
    int j = i / D, d = i % D;
    int t = t0 + j;
    int ix = (t < topk) ? irow[t] : -1;
    ks[i] = (ix >= 0) ? kv_elem2(kv, ix, d) : f2bf(0.f);
  }
  __syncthreads();
  // S (HGxKB) = Q (HGxD) · K (KBxD)^T: 16x16 tiles, HG/16 x KB/16 = 2x2 = 4 tiles -> warps 0..3
  if (warp < (HG / 16) * (KB / 16)) {
    const int tm = warp / (KB / 16), tn = warp % (KB / 16);
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int kk = 0; kk < D; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> fa;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> fb;
      wmma::load_matrix_sync(fa, qs + tm * 16 * D + kk, D);
      wmma::load_matrix_sync(fb, ks + tn * 16 * D + kk, D);
      wmma::mma_sync(acc, fa, fb, acc);
    }
    wmma::store_matrix_sync(ss + tm * 16 * KB + tn * 16, acc, KB, wmma::mem_row_major);
  }
  __syncthreads();
  for (int i = tid; i < HG * KB; i += THREADS) {
    int hh = i / KB, j = i % KB;
    int t = t0 + j;
    if (t >= topk) continue;
    int ix = irow[t];
    S[((size_t)m * H + h0 + hh) * topk + t] = (ix >= 0) ? ss[i] * scale : -INFINITY;
  }
}

// (2) Row statistics: block = (m, h) -> mx[m,h], sum[m,h] (sum of fp32 p)
__global__ void attn_stats_kernel(const float* __restrict__ S, int H, int topk, float* __restrict__ mx, float* __restrict__ sum) {
  __shared__ float red[32];
  const size_t row = blockIdx.x;
  const float* s = S + row * topk;
  float m = -1e30f;
  for (int t = threadIdx.x; t < topk; t += blockDim.x) m = fmaxf(m, s[t]);
  m = hive::cu::warp_allreduce<hive::cu::Max>(m);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -1e30f;
    t = hive::cu::warp_allreduce<hive::cu::Max>(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  m = red[0];
  __syncthreads();
  float su = 0.f;
  for (int t = threadIdx.x; t < topk; t += blockDim.x) su += expf(s[t] - m);
  su = hive::cu::warp_allreduce(su);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = su;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    t = hive::cu::warp_allreduce(t);
    if (threadIdx.x == 0) { mx[row] = m; sum[row] = t; }
  }
}

// (3) P·V: block = (m, head group). For each key block, P (HGxKB) bf16 = exp(S-max) goes to smem, V (KBxD) is gathered, and O (HGxD) is accumulated with WMMA.
//   8 warps x (HG/16=2 x D/16=32 = 64 tiles) -> 8 tiles per warp (4 along columns x ...: warp w takes row tile w/4, column tiles (w%4)*8 .. +8)
__global__ void __launch_bounds__(THREADS) attn_pv_kernel(const float* __restrict__ S, const float* __restrict__ mx, const float* __restrict__ sum,
                                                          int H, KvSources kv, const int32_t* __restrict__ idx, int idx_stride, int topk,
                                                          const float* __restrict__ sink, bf16* __restrict__ o) {
  extern __shared__ __align__(128) uint8_t smem[];
  bf16* ps = reinterpret_cast<bf16*>(smem);           // HG×KB
  bf16* vs = ps + HG * KB;                            // KB×D
  float* os = reinterpret_cast<float*>(vs + KB * D);  // HGxD epilogue (64KB) — overlaps the vs/ps area (used afterwards)
  (void)os;
  const int hg = blockIdx.x % (H / HG), m = blockIdx.x / (H / HG);
  const int h0 = hg * HG;
  const int tid = threadIdx.x, warp = tid >> 5;
  const int32_t* irow = idx + (size_t)m * idx_stride;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) wmma::fill_fragment(acc[i], 0.f);
  const int tm = warp / 4, tn0 = (warp % 4) * 8;
  for (int t0 = 0; t0 < topk; t0 += KB) {
    for (int i = tid; i < HG * KB; i += THREADS) {
      int hh = i / KB, j = i % KB;
      int t = t0 + j;
      float p = 0.f;
      if (t < topk) {
        float s = S[((size_t)m * H + h0 + hh) * topk + t];
        p = expf(s - mx[(size_t)m * H + h0 + hh]);  // -inf → 0
      }
      ps[i] = f2bf(p);
    }
    for (int i = tid; i < KB * D; i += THREADS) {
      int j = i / D, d = i % D;
      int t = t0 + j;
      int ix = (t < topk) ? irow[t] : -1;
      vs[i] = (ix >= 0) ? kv_elem2(kv, ix, d) : f2bf(0.f);
    }
    __syncthreads();
    // O (HGxD) += P (HGxKB) · V (KBxD): k = KB = 32 -> 2 steps
#pragma unroll
    for (int kk = 0; kk < KB; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> fa;
      wmma::load_matrix_sync(fa, ps + tm * 16 * KB + kk, KB);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> fb;
        wmma::load_matrix_sync(fb, vs + kk * D + (tn0 + j) * 16, D);
        wmma::mma_sync(acc[j], fa, fb, acc[j]);
      }
    }
    __syncthreads();
  }
  // Epilogue: spill the accumulators to smem (reusing the ps/vs area: HGxD fp32 = 64KB), divide by the per-head denominator, write bf16
  float* out_s = reinterpret_cast<float*>(smem);
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j) wmma::store_matrix_sync(out_s + tm * 16 * D + (tn0 + j) * 16, acc[j], D, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < HG * D; i += THREADS) {
    int hh = i / D, d = i % D;
    size_t row = (size_t)m * H + h0 + hh;
    float denom = sum[row] + expf(sink[h0 + hh] - mx[row]);
    o[row * D + d] = f2bf(out_s[i] / denom);
  }
}

// ---- Row-table version: only kv_row differs, the rest is shared (via a template) ----
struct RowSrc {
  const KvRow* rows;
  const bf16* chunk;
  int chunk_len, win;
};
__device__ inline bf16 kv_elem_r(const RowSrc& s, int m, int idx, int d) {
  if (idx < s.win) return s.rows[m].ring[(size_t)idx * D + d];
  idx -= s.win;
  if (idx < s.chunk_len) return s.chunk[(size_t)idx * D + d];
  idx -= s.chunk_len;
  return f2bf(kvp::comp_val(s.rows[m].comp + (size_t)idx * kvp::COMP_ROW, d));
}
// For the decode kernel: ring/chunk row pointers (bf16) / compressed row pointers (packed)
__device__ inline bool is_comp_r(const RowSrc& s, int idx) { return idx >= s.win + s.chunk_len; }
__device__ inline const bf16* kv_row_bf16_r(const RowSrc& s, int m, int idx) {
  if (idx < s.win) return s.rows[m].ring + (size_t)idx * D;
  return s.chunk + (size_t)(idx - s.win) * D;
}
__device__ inline const uint8_t* comp_row_r(const RowSrc& s, int m, int idx) { return s.rows[m].comp + (size_t)(idx - s.win - s.chunk_len) * kvp::COMP_ROW; }
__device__ inline uint32_t pack2bf(float a, float b) {  // two fp32 values (representable in bf16) -> bf16x2 bits
  return (uint32_t)__bfloat16_as_ushort(f2bf(a)) | ((uint32_t)__bfloat16_as_ushort(f2bf(b)) << 16);
}
__device__ inline int topk_of(const RowSrc& s, int m) { return s.win + s.rows[m].topk; }

__global__ void __launch_bounds__(THREADS) attn_scores_rows_kernel(const bf16* __restrict__ q, int H, RowSrc src, const int32_t* __restrict__ idx,
                                                                   int idx_stride, int topk_max, float scale, float* __restrict__ S) {
  extern __shared__ __align__(128) uint8_t smem[];
  bf16* qs = reinterpret_cast<bf16*>(smem);
  bf16* ks = qs + HG * D;
  float* ss = reinterpret_cast<float*>(ks + KB * D);
  const int nkb = (topk_max + KB - 1) / KB;
  const int t0 = (blockIdx.x % nkb) * KB;
  const int rest = blockIdx.x / nkb;
  const int hg = rest % (H / HG), m = rest / (H / HG);
  const int topk = topk_of(src, m);
  const int h0 = hg * HG;
  const int tid = threadIdx.x, warp = tid >> 5;
  if (t0 >= topk) {  // outside this row's columns: fill with -inf so statistics and PV ignore it
    for (int i = tid; i < HG * KB; i += THREADS) {
      int hh = i / KB, t = t0 + i % KB;
      if (t < topk_max) S[((size_t)m * H + h0 + hh) * topk_max + t] = -INFINITY;
    }
    return;
  }
  for (int i = tid; i < HG * D; i += THREADS) qs[i] = q[((size_t)m * H + h0) * D + i];
  const int32_t* irow = idx + (size_t)m * idx_stride;
  for (int i = tid; i < KB * D; i += THREADS) {
    int j = i / D, d = i % D;
    int t = t0 + j;
    int ix = (t < topk) ? irow[t] : -1;
    ks[i] = (ix >= 0) ? kv_elem_r(src, m, ix, d) : f2bf(0.f);
  }
  __syncthreads();
  if (warp < (HG / 16) * (KB / 16)) {
    const int tm = warp / (KB / 16), tn = warp % (KB / 16);
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int kk = 0; kk < D; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> fa;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> fb;
      wmma::load_matrix_sync(fa, qs + tm * 16 * D + kk, D);
      wmma::load_matrix_sync(fb, ks + tn * 16 * D + kk, D);
      wmma::mma_sync(acc, fa, fb, acc);
    }
    wmma::store_matrix_sync(ss + tm * 16 * KB + tn * 16, acc, KB, wmma::mem_row_major);
  }
  __syncthreads();
  for (int i = tid; i < HG * KB; i += THREADS) {
    int hh = i / KB, j = i % KB;
    int t = t0 + j;
    if (t >= topk_max) continue;
    float v = -INFINITY;
    if (t < topk) { int ix = irow[t]; if (ix >= 0) v = ss[i] * scale; }
    S[((size_t)m * H + h0 + hh) * topk_max + t] = v;
  }
}

__global__ void __launch_bounds__(THREADS) attn_pv_rows_kernel(const float* __restrict__ S, const float* __restrict__ mx, const float* __restrict__ sum,
                                                               int H, RowSrc src, const int32_t* __restrict__ idx, int idx_stride, int topk_max,
                                                               const float* __restrict__ sink, bf16* __restrict__ o) {
  extern __shared__ __align__(128) uint8_t smem[];
  bf16* ps = reinterpret_cast<bf16*>(smem);
  bf16* vs = ps + HG * KB;
  const int hg = blockIdx.x % (H / HG), m = blockIdx.x / (H / HG);
  const int topk = topk_of(src, m);
  const int h0 = hg * HG;
  const int tid = threadIdx.x, warp = tid >> 5;
  const int32_t* irow = idx + (size_t)m * idx_stride;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) wmma::fill_fragment(acc[i], 0.f);
  const int tm = warp / 4, tn0 = (warp % 4) * 8;
  for (int t0 = 0; t0 < topk; t0 += KB) {
    for (int i = tid; i < HG * KB; i += THREADS) {
      int hh = i / KB, j = i % KB;
      int t = t0 + j;
      float p = 0.f;
      if (t < topk) {
        float s = S[((size_t)m * H + h0 + hh) * topk_max + t];
        p = expf(s - mx[(size_t)m * H + h0 + hh]);
      }
      ps[i] = f2bf(p);
    }
    for (int i = tid; i < KB * D; i += THREADS) {
      int j = i / D, d = i % D;
      int t = t0 + j;
      int ix = (t < topk) ? irow[t] : -1;
      vs[i] = (ix >= 0) ? kv_elem_r(src, m, ix, d) : f2bf(0.f);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < KB; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> fa;
      wmma::load_matrix_sync(fa, ps + tm * 16 * KB + kk, KB);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> fb;
        wmma::load_matrix_sync(fb, vs + kk * D + (tn0 + j) * 16, D);
        wmma::mma_sync(acc[j], fa, fb, acc[j]);
      }
    }
    __syncthreads();
  }
  float* out_s = reinterpret_cast<float*>(smem);
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j) wmma::store_matrix_sync(out_s + tm * 16 * D + (tn0 + j) * 16, acc[j], D, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < HG * D; i += THREADS) {
    int hh = i / D, d = i % D;
    size_t row = (size_t)m * H + h0 + hh;
    float denom = sum[row] + expf(sink[h0 + hh] - mx[row]);
    o[row * D + d] = f2bf(out_s[i] / denom);
  }
}
// Per-row statistics (topk differs per row): the valid columns of S = topk_of(m); the rest is filled with -inf, so scanning the whole topk_max is fine.

// ---- Fused decode version (M <= 8): block = (row m, head h), one kernel for scores -> statistics -> P·V. Three kernels + the global S round trip took ~70 µs at M=1.
//   Same semantics as above: fp32 scores, P cast to bf16 and multiplied with V (fp32 accumulation), denominator = fp32 sum of p + exp(sink - max). Only the dot-product order differs.
constexpr int DEC_THREADS = 256;
// split-K: block = (row m, head h, chunk s) — each chunk produces a partial (max, fp32 sum, acc[D]) and a merge kernel combines them (one block per head would use only 64 SMs).
// With S == 1, o is written directly. p is cast to bf16 relative to the chunk's local max (rounding differs slightly from the single-block version — verified against golden outputs).
// When storing o, apply inverse RoPE (reference apply_rotary_emb(inverse) — rotating the bf16-rounded values): if rope_freqs is given, the trailing rd dimensions are processed as pairs (2p, 2p+1)
__device__ __forceinline__ void store_o_pair(bf16* __restrict__ orow_base, int d0, float v0, float v1, int D, int rd, const float2* __restrict__ rope_freqs,
                                             const int32_t* __restrict__ rope_pos, int m) {
  if (rope_freqs && d0 >= D - rd) {
    const int half = rd / 2, p = (d0 - (D - rd)) >> 1;
    float2 f = rope_freqs[(size_t)rope_pos[m] * half + p];
    f.y = -f.y;
    const float a = bf2f(f2bf(v0)), b = bf2f(f2bf(v1));
    orow_base[d0] = f2bf(a * f.x - b * f.y);
    orow_base[d0 + 1] = f2bf(a * f.y + b * f.x);
  } else {
    orow_base[d0] = f2bf(v0);
    orow_base[d0 + 1] = f2bf(v1);
  }
}
__global__ void __launch_bounds__(DEC_THREADS) attn_decode_rows_kernel(const bf16* __restrict__ q, int H, RowSrc src, const int32_t* __restrict__ idx,
                                                                       int idx_stride, const float* __restrict__ sink, float scale, int S,
                                                                       bf16* __restrict__ o, float* __restrict__ pacc, float* __restrict__ pm,
                                                                       float* __restrict__ ps, const float2* __restrict__ rope_freqs,
                                                                       const int32_t* __restrict__ rope_pos, int rd) {
  extern __shared__ float sdyn[];                 // [chunk] scores -> p
  __shared__ float qs[D];
  __shared__ float red[DEC_THREADS / 32];
  __shared__ float stat[2];
  const int s = blockIdx.x % S;
  const int mh = blockIdx.x / S;
  const int h = mh % H, m = mh / H;
  // The row table (mapped pinned host memory) is read once per block — re-reading it per key would serialize as many PCIe round trips as there are keys
  const KvRow row = src.rows[m];
  const RowSrc srcl{&row, src.chunk, src.chunk_len, src.win};
  const int ncols = src.win + row.topk;
  const int chunk = (ncols + S - 1) / S;
  const int tb = s * chunk, te = min(ncols, tb + chunk), nloc = te > tb ? te - tb : 0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int32_t* irow = idx + (size_t)m * idx_stride + tb;
  const bf16* qh = q + ((size_t)m * H + h) * D;
  // 16 dims of a key (one lane's share) as bf16x16 bits: ring/chunk as is; compressed rows are exact when unpacked from fp4 and repacked as bf16 (kv_pack.h)
  auto load_key16 = [&](int ix, uint4& ra, uint4& rb) {
    if (is_comp_r(srcl, ix)) {
      float v[16];
      kvp::comp_block16(comp_row_r(srcl, 0, ix), lane, v);
      ra = make_uint4(pack2bf(v[0], v[1]), pack2bf(v[2], v[3]), pack2bf(v[4], v[5]), pack2bf(v[6], v[7]));
      rb = make_uint4(pack2bf(v[8], v[9]), pack2bf(v[10], v[11]), pack2bf(v[12], v[13]), pack2bf(v[14], v[15]));
    } else {
      const uint4* kp = reinterpret_cast<const uint4*>(kv_row_bf16_r(srcl, 0, ix)) + lane * 2;
      ra = kp[0]; rb = kp[1];
    }
  };
  auto load_pair = [&](int ix, int d0) -> uint32_t {
    if (is_comp_r(srcl, ix)) { const float2 p = kvp::comp_pair(comp_row_r(srcl, 0, ix), d0); return pack2bf(p.x, p.y); }
    return *reinterpret_cast<const uint32_t*>(kv_row_bf16_r(srcl, 0, ix) + d0);
  };
  for (int i = tid; i < D; i += DEC_THREADS) qs[i] = bf2f(qh[i]);
  __syncthreads();
  // (1) Scores: each warp handles 4 keys at a time (each lane 16 dims) — all 4 keys' loads are issued before computing
  const float q0 = qs[lane * 16 + 0], q1 = qs[lane * 16 + 1], q2 = qs[lane * 16 + 2], q3 = qs[lane * 16 + 3];
  const float q4 = qs[lane * 16 + 4], q5 = qs[lane * 16 + 5], q6 = qs[lane * 16 + 6], q7 = qs[lane * 16 + 7];
  const float q8 = qs[lane * 16 + 8], q9 = qs[lane * 16 + 9], q10 = qs[lane * 16 + 10], q11 = qs[lane * 16 + 11];
  const float q12 = qs[lane * 16 + 12], q13 = qs[lane * 16 + 13], q14 = qs[lane * 16 + 14], q15 = qs[lane * 16 + 15];
  auto dot16 = [&](const uint4 a, const uint4 b) -> float {
    const bf16* ka = reinterpret_cast<const bf16*>(&a);
    const bf16* kb = reinterpret_cast<const bf16*>(&b);
    float d = fmaf(q0, bf2f(ka[0]), 0.f);
    d = fmaf(q1, bf2f(ka[1]), d); d = fmaf(q2, bf2f(ka[2]), d); d = fmaf(q3, bf2f(ka[3]), d);
    d = fmaf(q4, bf2f(ka[4]), d); d = fmaf(q5, bf2f(ka[5]), d); d = fmaf(q6, bf2f(ka[6]), d); d = fmaf(q7, bf2f(ka[7]), d);
    d = fmaf(q8, bf2f(kb[0]), d); d = fmaf(q9, bf2f(kb[1]), d); d = fmaf(q10, bf2f(kb[2]), d); d = fmaf(q11, bf2f(kb[3]), d);
    d = fmaf(q12, bf2f(kb[4]), d); d = fmaf(q13, bf2f(kb[5]), d); d = fmaf(q14, bf2f(kb[6]), d); d = fmaf(q15, bf2f(kb[7]), d);
    return d;
  };
  constexpr int NW = DEC_THREADS / 32;
  for (int t0 = warp * 4; t0 < nloc; t0 += NW * 4) {
    int ixs[4];
    uint4 ra[4], rb[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int t = t0 + u;
      ixs[u] = (t < nloc) ? irow[t] : -1;
      if (ixs[u] >= 0) load_key16(ixs[u], ra[u], rb[u]);
      else { ra[u] = make_uint4(0, 0, 0, 0); rb[u] = ra[u]; }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      float d = dot16(ra[u], rb[u]);
      d = hive::cu::warp_allreduce(d);
      const int t = t0 + u;
      if (lane == 0 && t < nloc) sdyn[t] = (ixs[u] >= 0) ? d * scale : -INFINITY;
    }
  }
  __syncthreads();
  // (2) chunk max and fp32 sum
  float mx = -INFINITY;
  for (int t = tid; t < nloc; t += DEC_THREADS) mx = fmaxf(mx, sdyn[t]);
  mx = hive::cu::warp_allreduce<hive::cu::Max>(mx);
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  if (tid == 0) { float v = red[0]; for (int i = 1; i < DEC_THREADS / 32; ++i) v = fmaxf(v, red[i]); stat[0] = v; }
  __syncthreads();
  mx = stat[0];
  const bool empty = !(mx > -INFINITY);  // this chunk has no valid keys
  float su = 0.f;
  if (!empty) for (int t = tid; t < nloc; t += DEC_THREADS) su += expf(sdyn[t] - mx);
  su = hive::cu::warp_allreduce(su);
  __syncthreads();  // before red is reused
  if (lane == 0) red[warp] = su;
  __syncthreads();
  if (tid == 0) { float v = 0.f; for (int i = 0; i < DEC_THREADS / 32; ++i) v += red[i]; stat[1] = v; }
  // Cast p to bf16 (P·V uses bf16 P) — overwrites sdyn with p
  if (!empty) for (int t = tid; t < nloc; t += DEC_THREADS) sdyn[t] = bf2f(f2bf(expf(sdyn[t] - mx)));
  __syncthreads();
  // (3) P·V: thread = 2 output dims, loads of 8 keys at a time are overlapped
  float acc0 = 0.f, acc1 = 0.f;
  const int d0 = tid * 2;
  int t = 0;
  if (!empty) {
    for (; t + 8 <= nloc; t += 8) {
      uint32_t raw[8];
      float pv[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        pv[u] = sdyn[t + u];
        const int ix = irow[t + u];
        raw[u] = (ix >= 0 && pv[u] != 0.f) ? load_pair(ix, d0) : 0u;
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        acc0 = fmaf(pv[u], __uint_as_float(raw[u] << 16), acc0);
        acc1 = fmaf(pv[u], __uint_as_float(raw[u] & 0xFFFF0000u), acc1);
      }
    }
    for (; t < nloc; ++t) {
      const float p = sdyn[t];
      const int ix = irow[t];
      if (p == 0.f || ix < 0) continue;
      const uint32_t raw = load_pair(ix, d0);
      acc0 = fmaf(p, __uint_as_float(raw << 16), acc0);
      acc1 = fmaf(p, __uint_as_float(raw & 0xFFFF0000u), acc1);
    }
  }
  if (S == 1) {
    const float denom = stat[1] + expf(sink[h] - mx);
    store_o_pair(o + ((size_t)m * H + h) * D, d0, acc0 / denom, acc1 / denom, D, rd, rope_freqs, rope_pos, m);
  } else {
    const size_t pi = ((size_t)m * H + h) * S + s;
    float* pa = pacc + pi * D + d0;
    pa[0] = acc0; pa[1] = acc1;
    if (tid == 0) { pm[pi] = mx; ps[pi] = stat[1]; }
  }
}

// Merge: block = (m, h). M_g = max_s pm, scale_s = exp(pm_s - M_g), acc = sum scale_s·pacc_s, denominator = sum scale_s·ps_s + exp(sink - M_g)
__global__ void attn_decode_merge_kernel(const float* __restrict__ pacc, const float* __restrict__ pm, const float* __restrict__ ps, int H, int S,
                                         const float* __restrict__ sink, bf16* __restrict__ o, const float2* __restrict__ rope_freqs,
                                         const int32_t* __restrict__ rope_pos, int rd) {
  __shared__ float sc[16];
  __shared__ float stat[2];
  const int h = blockIdx.x % H, m = blockIdx.x / H;
  const size_t base = ((size_t)m * H + h) * S;
  if (threadIdx.x == 0) {
    float M_g = -INFINITY;
    for (int s = 0; s < S; ++s) M_g = fmaxf(M_g, pm[base + s]);
    float sum = 0.f;
    for (int s = 0; s < S; ++s) { const float f = pm[base + s] > -INFINITY ? expf(pm[base + s] - M_g) : 0.f; sc[s] = f; sum += f * ps[base + s]; }
    stat[0] = M_g; stat[1] = sum + expf(sink[h] - M_g);
  }
  __syncthreads();
  const int d0 = threadIdx.x * 2;
  float a0 = 0.f, a1 = 0.f;
  for (int s = 0; s < S; ++s) {
    const float* pa = pacc + (base + s) * D + d0;
    a0 = fmaf(sc[s], pa[0], a0); a1 = fmaf(sc[s], pa[1], a1);
  }
  store_o_pair(o + ((size_t)m * H + h) * D, d0, a0 / stat[1], a1 / stat[1], D, rd, rope_freqs, rope_pos, m);
}

constexpr size_t SMEM_SCORES = (size_t)HG * D * 2 + (size_t)KB * D * 2 + (size_t)HG * KB * 4;   // 32K + 32K + 4K
constexpr size_t SMEM_PV = (size_t)HG * KB * 2 + (size_t)KB * D * 2;                             // 2K + 32K (64KB epilogue separate)
constexpr size_t SMEM_PV_TOTAL = (size_t)HG * D * 4 > SMEM_PV ? (size_t)HG * D * 4 : SMEM_PV;  // 64KB
}  // namespace

void sparse_attn_tc(const bf16* q, int M, int H, int Dd, const KvSources& kv, const int32_t* idx, int idx_stride, int topk,
                    const float* sink, float scale, float* S, float* mx, float* sum, bf16* o, cudaStream_t st) {
  HIVE_CHECK(Dd == D && H % HG == 0, "sparse_attn_tc shape");
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(attn_scores_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_SCORES));
    CUDA_CHECK(cudaFuncSetAttribute(attn_pv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_PV_TOTAL));
    configured = true;
  }
  const int nkb = (topk + KB - 1) / KB;
  attn_scores_kernel<<<M * (H / HG) * nkb, THREADS, SMEM_SCORES, st>>>(q, H, kv, idx, idx_stride, topk, scale, S);
  attn_stats_kernel<<<M * H, 256, 0, st>>>(S, H, topk, mx, sum);
  attn_pv_kernel<<<M * (H / HG), THREADS, SMEM_PV_TOTAL, st>>>(S, mx, sum, H, kv, idx, idx_stride, topk, sink, o);
}

void sparse_attn_tc_rows(const bf16* q, int M, int H, int Dd, const KvRow* rows, const bf16* chunk, int chunk_len, int win,
                         const int32_t* idx, int idx_stride, int topk_max, const float* sink, float scale, float* S, float* mx, float* sum,
                         bf16* o, cudaStream_t st) {
  HIVE_CHECK(Dd == D && H % HG == 0, "sparse_attn_tc_rows shape");
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(attn_scores_rows_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_SCORES));
    CUDA_CHECK(cudaFuncSetAttribute(attn_pv_rows_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_PV_TOTAL));
    configured = true;
  }
  RowSrc src{rows, chunk, chunk_len, win};
  const int nkb = (topk_max + KB - 1) / KB;
  attn_scores_rows_kernel<<<M * (H / HG) * nkb, THREADS, SMEM_SCORES, st>>>(q, H, src, idx, idx_stride, topk_max, scale, S);
  attn_stats_kernel<<<M * H, 256, 0, st>>>(S, H, topk_max, mx, sum);
  attn_pv_rows_kernel<<<M * (H / HG), THREADS, SMEM_PV_TOTAL, st>>>(S, mx, sum, H, src, idx, idx_stride, topk_max, sink, o);
}

void sparse_attn_decode_rows(const bf16* q, int M, int H, int Dd, const KvRow* rows, const bf16* chunk, int chunk_len, int win,
                             const int32_t* idx, int idx_stride, int ncols_max, const float* sink, float scale, bf16* o, int splits, float* pacc,
                             float* pm, float* ps, cudaStream_t st, const float2* rope_freqs, const int32_t* rope_pos, int rd) {
  HIVE_CHECK(Dd == D && M <= 8 && splits >= 1 && splits <= 16, "sparse_attn_decode_rows shape");
  RowSrc src{rows, chunk, chunk_len, win};
  const int S = splits;
  const size_t smem = (size_t)((ncols_max + S - 1) / S) * 4 + 16;
  attn_decode_rows_kernel<<<M * H * S, DEC_THREADS, smem, st>>>(q, H, src, idx, idx_stride, sink, scale, S, o, pacc, pm, ps, rope_freqs, rope_pos, rd);
  if (S > 1) attn_decode_merge_kernel<<<M * H, DEC_THREADS, 0, st>>>(pacc, pm, ps, H, S, sink, o, rope_freqs, rope_pos, rd);
}

}  // namespace hive::k
