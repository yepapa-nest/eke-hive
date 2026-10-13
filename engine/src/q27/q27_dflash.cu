// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// DFlash2 drafter kernels — see hive/q27/q27_dflash.h (math after z-lab/dflash, MIT License, Copyright (c) 2026 Z Lab; own CUDA code).
#include "hive/q27/q27_dflash.h"

#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <cuda_fp8.h>

#include "hive/common.h"

namespace hive::q27::df {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// block-wide sum (blockDim.x multiple of 32, ≤ 1024)
__device__ float block_sum(float v, float* sh) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  float t = l < nw ? sh[l] : 0.f;
  t = warp_sum(t);
  return t;
}

__global__ void rmsnorm_kernel(const bf16* x, int ldx, const bf16* w, int K, float eps, bf16* out, int ldo) {
  __shared__ float sh[32];
  const bf16* r = x + (size_t)blockIdx.x * ldx;
  float ss = 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) { const float v = __bfloat162float(r[i]); ss += v * v; }
  ss = block_sum(ss, sh);
  const float inv = rsqrtf(ss / K + eps);
  bf16* o = out + (size_t)blockIdx.x * ldo;
  for (int i = threadIdx.x; i < K; i += blockDim.x) o[i] = __float2bfloat16(__bfloat162float(r[i]) * inv * __bfloat162float(w[i]));
}

__global__ void capture_kernel(const bf16* h, const bf16* y, int H, bf16* out, int ldo) {
  const size_t r = blockIdx.y;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < H; i += gridDim.x * blockDim.x) {
    const float v = __bfloat162float(h[r * H + i]) + (y ? __bfloat162float(y[r * H + i]) : 0.f);
    out[r * ldo + i] = __float2bfloat16(v);
  }
}

// one head of 128 values held one per thread (blockDim 128): norm (w) + neox rope at pos → returns the value of this thread's dim
__device__ float norm_rope(float x, const bf16* w, float eps, int pos, const float* inv_freq, float mscale, float* sh, float* shv) {
  const int d = threadIdx.x;
  const float ss = block_sum(x * x, sh);
  const float n = __bfloat162float(__float2bfloat16(x * rsqrtf(ss / kHeadDim + eps) * __bfloat162float(w[d])));
  shv[d] = n;
  __syncthreads();
  const int i = d & 63;
  float s, c;
  sincosf((float)pos * inv_freq[i], &s, &c);
  s *= mscale; c *= mscale;
  const float other = shv[d ^ 64];
  return d < 64 ? n * c - other * s : n * c + other * s;
}

__global__ void ctx_kv_kernel(const bf16* kv, const bf16* k_norm, const CtxRow* rows, float eps, const float* inv_freq, float mscale) {
  __shared__ float sh[32], shv[kHeadDim];
  const int r = blockIdx.x, l = blockIdx.y, h = blockIdx.z, d = threadIdx.x;
  const CtxRow row = rows[r];
  const bf16* src = kv + (size_t)r * kLayers * 2 * kKvW + (size_t)l * 2 * kKvW;
  const size_t slot = ((size_t)l * row.slots + (size_t)(row.pos % row.slots)) * kKvW + (size_t)h * kHeadDim + d;
  const float o = norm_rope(__bfloat162float(src[h * kHeadDim + d]), k_norm + l * kHeadDim, eps, row.pos, inv_freq, mscale, sh, shv);
  row.k[slot] = __float2bfloat16(o);
  row.v[slot] = src[kKvW + h * kHeadDim + d];
}

__global__ void blk_prep_kernel(const bf16* proj, const bf16* q_norm, const bf16* k_norm, const BlkSeq* seqs, int B, float eps, const float* inv_freq,
                                float mscale, bf16* q, bf16* k, bf16* v) {
  __shared__ float sh[32], shv[kHeadDim];
  const int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int pos0 = seqs[i / B].ctx - i / B * B;   // row i sits at ctx + i % B
  const int PW = (kHeads + 2 * kKvHeads) * kHeadDim;
  const bf16* src = proj + (size_t)i * PW;
  if (h < kHeads) {
    q[((size_t)i * kHeads + h) * kHeadDim + d] = __float2bfloat16(norm_rope(__bfloat162float(src[h * kHeadDim + d]), q_norm, eps, pos0 + i, inv_freq, mscale, sh, shv));
  } else if (h < kHeads + kKvHeads) {
    const int kh = h - kHeads;
    k[((size_t)i * kKvHeads + kh) * kHeadDim + d] =
        __float2bfloat16(norm_rope(__bfloat162float(src[kHeads * kHeadDim + kh * kHeadDim + d]), k_norm, eps, pos0 + i, inv_freq, mscale, sh, shv));
  } else {
    const int vh = h - kHeads - kKvHeads;
    v[((size_t)i * kKvHeads + vh) * kHeadDim + d] = src[(kHeads + kKvHeads) * kHeadDim + vh * kHeadDim + d];
  }
}

// grid (32 heads, L queries), 4 warps; each lane scores one key per tile of 32 (full 128-dim dot), then the tile's P·V with 4 dims per lane
__global__ void attention_kernel(const bf16* q, const bf16* bk_all, const bf16* bv_all, const BlkSeq* seqs, int layer, int B, int window, float scale,
                                 bf16* out) {
  const int L = B, sq = blockIdx.y / B, il = blockIdx.y % B;
  const int slots = seqs[sq].slots;
  const bf16* rk = seqs[sq].k + (size_t)layer * slots * kKvW;
  const bf16* rv = seqs[sq].v + (size_t)layer * slots * kKvW;
  const int ctx_len = seqs[sq].ctx;
  const bf16* bk = bk_all + (size_t)sq * B * kKvW;
  const bf16* bv = bv_all + (size_t)sq * B * kKvW;
  __shared__ float qs[kHeadDim];
  __shared__ float wm[4], wl[4], wacc[4][kHeadDim];
  const int h = blockIdx.x, i = blockIdx.y, kvh = h / (kHeads / kKvHeads);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  qs[threadIdx.x] = __bfloat162float(q[((size_t)i * kHeads + h) * kHeadDim + threadIdx.x]) * scale;
  __syncthreads();
  const int qpos = ctx_len + il;
  const int lo = window > 0 ? max(0, qpos - (window - 1)) : 0;
  const int nctx = ctx_len - lo, n = nctx + L;
  float m = -FLT_MAX, lsum = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int base = warp * 32; base < n; base += 128) {
    const int j = base + lane;
    const bf16* kp = nullptr;
    const bf16* vp = nullptr;
    if (j < n) {
      if (j < nctx) {
        const size_t s = (size_t)((lo + j) % slots) * kKvW + (size_t)kvh * kHeadDim;
        kp = rk + s; vp = rv + s;
      } else {
        const size_t s = (size_t)(j - nctx) * kKvW + (size_t)kvh * kHeadDim;
        kp = bk + s; vp = bv + s;
      }
    }
    float sc = -FLT_MAX;
    if (kp) {
      float dot = 0.f;
      const uint4* k4 = reinterpret_cast<const uint4*>(kp);
#pragma unroll 4
      for (int c = 0; c < kHeadDim / 8; ++c) {
        const uint4 u = k4[c];
        const bf16* e = reinterpret_cast<const bf16*>(&u);
#pragma unroll
        for (int t = 0; t < 8; ++t) dot += qs[c * 8 + t] * __bfloat162float(e[t]);
      }
      sc = dot;
    }
    const float tm = warp_max(sc);
    const float mn = fmaxf(m, tm);
    const float corr = __expf(m - mn);
    const float p = kp ? __expf(sc - mn) : 0.f;
    lsum = lsum * corr + warp_sum(p);
    for (int t = 0; t < 4; ++t) acc[t] *= corr;
    m = mn;
    const int cnt = min(32, n - base);
    for (int jj = 0; jj < cnt; ++jj) {
      const float pj = __shfl_sync(0xffffffffu, p, jj);
      const bf16* vj = reinterpret_cast<const bf16*>(__shfl_sync(0xffffffffu, reinterpret_cast<uintptr_t>(vp), jj));
      const uint2 u = reinterpret_cast<const uint2*>(vj)[lane];
      const bf16* e = reinterpret_cast<const bf16*>(&u);
#pragma unroll
      for (int t = 0; t < 4; ++t) acc[t] += pj * __bfloat162float(e[t]);
    }
  }
  if (lane == 0) { wm[warp] = m; wl[warp] = lsum; }
  for (int t = 0; t < 4; ++t) wacc[warp][lane * 4 + t] = acc[t];
  __syncthreads();
  const int d = threadIdx.x;
  float M = -FLT_MAX;
  for (int w = 0; w < 4; ++w) if (wl[w] > 0.f) M = fmaxf(M, wm[w]);
  float S = 0.f, A = 0.f;
  for (int w = 0; w < 4; ++w) {
    if (wl[w] <= 0.f) continue;
    const float f = __expf(wm[w] - M);
    S += wl[w] * f; A += wacc[w][d] * f;
  }
  out[(size_t)i * kHeads * kHeadDim + (size_t)h * kHeadDim + d] = __float2bfloat16(A / S);
}

// grouped variant (HIVE_Q27_DF_ATTN_GRP, default on): one block per (sequence, kv head, key split) scores all 4 q heads × B rows of that kv head
//   against one shared K / V tile in shared memory — attention_kernel above reads the whole window once per (head, row), 32 × the L2 traffic
//   (2026-10-11: 2048 blocks at c8, 0.13-0.46 ms of the draft phase; 0.64 ms per round at a 64K context). Partials (max, Σ, acc) per split go to
//   scratch and attn_grp_combine_kernel merges them. Same masking as attention_kernel (per-row window start, block keys always visible).
constexpr int kGrpSplits = 8, kGrpRows = 4 * 8;   // q heads per kv head × max block rows
float* g_grp_part = nullptr;   // partials of up to 64 sequences (attention_prepare, outside any graph capture)
__global__ void __launch_bounds__(256) attn_grp_kernel(const bf16* q, const bf16* bk_all, const bf16* bv_all, const BlkSeq* seqs, int layer, int B, int window,
                                                       float scale, float* part) {
  constexpr int KSD = 130, VSD = 136, HPK = kHeads / kKvHeads;   // V rows 16-byte aligned (272 B)
  __shared__ float qs[kGrpRows][kHeadDim];
  __shared__ __align__(16) bf16 Ks[32 * KSD];
  __shared__ __align__(16) bf16 Vs[32 * VSD];
  const int kvh = blockIdx.x, sq = blockIdx.y, sp = blockIdx.z, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int R = HPK * B;
  const int slots = seqs[sq].slots, ctx_len = seqs[sq].ctx;
  const bf16* rk = seqs[sq].k + (size_t)layer * slots * kKvW;
  const bf16* rv = seqs[sq].v + (size_t)layer * slots * kKvW;
  const bf16* bk = bk_all + (size_t)sq * B * kKvW;
  const bf16* bv = bv_all + (size_t)sq * B * kKvW;
  for (int i = tid; i < R * kHeadDim; i += 256) {   // row r = hh · B + il  →  head kvh · 4 + hh, block row il
    const int r = i / kHeadDim, d = i % kHeadDim, hh = r / B, il = r % B;
    qs[r][d] = __bfloat162float(q[((size_t)(sq * B + il) * kHeads + kvh * HPK + hh) * kHeadDim + d]) * scale;
  }
  const int lo0 = window > 0 ? max(0, ctx_len - (window - 1)) : 0;   // row il = 0 (the smallest window start)
  const int nctx = ctx_len - lo0, n = nctx + B;
  const int per = (n + kGrpSplits - 1) / kGrpSplits, k_lo = sp * per, k_hi = min(n, k_lo + per);
  float m[4], l[4], acc[4][4];
  int rr[4], lo_r[4];
#pragma unroll
  for (int x = 0; x < 4; ++x) {
    rr[x] = warp * 4 + x; m[x] = -FLT_MAX; l[x] = 0.f;
    const int il = rr[x] < R ? rr[x] % B : 0;
    lo_r[x] = window > 0 ? max(0, ctx_len + il - (window - 1)) : 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[x][e] = 0.f;
  }
  for (int base = k_lo; base < k_hi; base += 32) {
    __syncthreads();
    for (int i = tid; i < 32 * 16; i += 256) {   // 32 keys × 16 pieces of 8 bf16
      const int kk = i / 16, pc = i % 16, j = base + kk;
      uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
      if (j < k_hi) {
        const size_t off = j < nctx ? (size_t)((lo0 + j) % slots) * kKvW + (size_t)kvh * kHeadDim : (size_t)(j - nctx) * kKvW + (size_t)kvh * kHeadDim;
        const bf16* kp = j < nctx ? rk : bk;
        const bf16* vp = j < nctx ? rv : bv;
        kv = reinterpret_cast<const uint4*>(kp + off)[pc]; vv = reinterpret_cast<const uint4*>(vp + off)[pc];
      }
      const uint32_t* kw = reinterpret_cast<const uint32_t*>(&kv);
#pragma unroll
      for (int w = 0; w < 4; ++w) *reinterpret_cast<uint32_t*>(Ks + kk * KSD + pc * 8 + 2 * w) = kw[w];
      *reinterpret_cast<uint4*>(Vs + kk * VSD + pc * 8) = vv;
    }
    __syncthreads();
    const int j = base + lane;
    float sc[4];
    {
      float dot[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
      for (int d = 0; d < kHeadDim; d += 2) {
        const __nv_bfloat162 k2 = *reinterpret_cast<const __nv_bfloat162*>(Ks + lane * KSD + d);
        const float k0 = __low2float(k2), k1 = __high2float(k2);
#pragma unroll
        for (int x = 0; x < 4; ++x) dot[x] += qs[rr[x] < R ? rr[x] : 0][d] * k0 + qs[rr[x] < R ? rr[x] : 0][d + 1] * k1;
      }
#pragma unroll
      for (int x = 0; x < 4; ++x) {
        const bool vis = j < k_hi && rr[x] < R && (j >= nctx || lo0 + j >= lo_r[x]);
        sc[x] = vis ? dot[x] : -FLT_MAX;
      }
    }
    float p[4];
#pragma unroll
    for (int x = 0; x < 4; ++x) {
      const float tm = warp_max(sc[x]), mn = fmaxf(m[x], tm);
      const float corr = __expf(m[x] - mn);
      p[x] = sc[x] > -FLT_MAX ? __expf(sc[x] - mn) : 0.f;
      l[x] = l[x] * corr + warp_sum(p[x]);
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[x][e] *= corr;
      m[x] = mn;
    }
    const int cnt = min(32, k_hi - base);
    for (int jj = 0; jj < cnt; ++jj) {
      const uint2 u = *reinterpret_cast<const uint2*>(Vs + jj * VSD + lane * 4);
      const bf16* e4 = reinterpret_cast<const bf16*>(&u);
      const float v0 = __bfloat162float(e4[0]), v1 = __bfloat162float(e4[1]), v2 = __bfloat162float(e4[2]), v3 = __bfloat162float(e4[3]);
#pragma unroll
      for (int x = 0; x < 4; ++x) {
        const float pj = __shfl_sync(0xffffffffu, p[x], jj);
        acc[x][0] += pj * v0; acc[x][1] += pj * v1; acc[x][2] += pj * v2; acc[x][3] += pj * v3;
      }
    }
  }
#pragma unroll
  for (int x = 0; x < 4; ++x) {
    if (rr[x] >= R) continue;
    float* pb = part + ((((size_t)sq * kKvHeads + kvh) * kGrpRows + rr[x]) * kGrpSplits + sp) * (kHeadDim + 2);
    if (lane == 0) { pb[0] = m[x]; pb[1] = l[x]; }
#pragma unroll
    for (int e = 0; e < 4; ++e) pb[2 + lane * 4 + e] = acc[x][e];
  }
}
__global__ void attn_grp_combine_kernel(const float* part, int B, bf16* out) {
  const int kvh = blockIdx.x, sq = blockIdx.y, r = blockIdx.z, d = threadIdx.x, HPK = kHeads / kKvHeads;
  if (r >= HPK * B) return;
  const float* pb = part + (((size_t)sq * kKvHeads + kvh) * kGrpRows + r) * kGrpSplits * (kHeadDim + 2);
  float M = -FLT_MAX;
  for (int s2 = 0; s2 < kGrpSplits; ++s2) if (pb[s2 * (kHeadDim + 2) + 1] > 0.f) M = fmaxf(M, pb[s2 * (kHeadDim + 2)]);
  float S = 0.f, A = 0.f;
  for (int s2 = 0; s2 < kGrpSplits; ++s2) {
    const float* q2 = pb + s2 * (kHeadDim + 2);
    if (q2[1] <= 0.f) continue;
    const float f = __expf(q2[0] - M);
    S += q2[1] * f; A += q2[2 + d] * f;
  }
  const int hh = r / B, il = r % B;
  out[((size_t)(sq * B + il) * kHeads + kvh * HPK + hh) * kHeadDim + d] = __float2bfloat16(A / S);
}

__global__ void conv_kernel(const bf16* in, int ld_in, const float* dyn, int half, const float* base, int B, int H, bf16* out, bool residual) {
  const int t = blockIdx.y, tl = t % B;
  const int G = H / kGroup;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < H; c += gridDim.x * blockDim.x) {
    float v = 0.f;
    for (int o = 0; o < 2 && o <= tl; ++o) {
      const float k = base[(size_t)o * H + c] + dyn[(size_t)t * 4 * G + (size_t)half * 2 * G + (size_t)o * G + c / kGroup];
      v += k * __bfloat162float(in[(size_t)(t - o) * ld_in + c]);
    }
    bf16* op = out + (size_t)t * H + c;
    if (residual) *op = __float2bfloat16(__bfloat162float(*op) + __bfloat162float(__float2bfloat16(v)));
    else *op = __float2bfloat16(v);
  }
}

// candidate selector, part 1: every edge term of position p — e[s][p][a][c] = Σ_r P[a', r] · Hp[p, r] · S[c', r] for each predecessor a (the
//   anchor for p = 0, else the 16 candidates of p − 1) and candidate c. grid (S, 7), 16 warps (warp = candidate c), 8 rank values per lane.
__global__ void select_edges_kernel(const int32_t* cidx, const float* hp, const bf16* pred, const bf16* succ, const int32_t* map, const BlkSeq* seqs,
                                    float* edges) {
  const int P = kBlock - 1, sq = blockIdx.x, p = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int32_t* ci = cidx + (size_t)sq * P * kTopK;
  auto tok = [&](int pp, int c) { const int32_t i = ci[pp * kTopK + c]; return (i < 0 || i == 0x7fffffff) ? 0 : (map ? map[i] : i); };   // an empty top-16 slot (non-finite logits) → token 0
  const int32_t tc = tok(p, w);
  const float* hr = hp ? hp + ((size_t)sq * P + p) * kRank : nullptr;
  float u[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { const int r = lane + 32 * i; u[i] = (hp ? hr[r] : 1.f) * __bfloat162float(succ[(size_t)tc * kRank + r]); }
  const int na = p == 0 ? 1 : kTopK;
  float* eo = edges + (((size_t)sq * P + p) * kTopK) * kTopK;
  for (int a = 0; a < na; ++a) {
    const int32_t ta = p == 0 ? seqs[sq].anchor : tok(p - 1, a);
    float e = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) e += __bfloat162float(pred[(size_t)ta * kRank + lane + 32 * i]) * u[i];
    e = warp_sum(e);
    if (lane == 0) eo[a * kTopK + w] = e;
  }
}
// part 2: the greedy chain per sequence (one warp; lane c < 16 = candidate c): score = logit + edge from the previous pick
__global__ void select_chain_kernel(const int32_t* cidx, const float* cval, const int32_t* map, const float* edges, int32_t* ids, float* conf,
                                    const BlkSeq* seqs, const bf16* pred, const float* conf_h, const bf16* conf_w, float conf_b) {
  const int P = kBlock - 1, sq = blockIdx.x, lane = threadIdx.x;
  int prev = 0;   // index of the previous pick among its 16 (0 = the anchor row for p = 0)
  int32_t prev_tok = seqs[sq].anchor;
  for (int p = 0; p < P; ++p) {
    float sc = -FLT_MAX;
    int32_t tk = 0x7fffffff;
    if (lane < kTopK) {
      const size_t o = ((size_t)sq * P + p) * kTopK;
      sc = cval[o + lane] + edges[(o + prev) * kTopK + lane];
      const int32_t ci = cidx[o + lane];
      tk = (ci < 0 || ci == 0x7fffffff) ? 0 : (map ? map[ci] : ci);
    }
    float bs = sc; int32_t bt = tk; int bl = lane;
    for (int off = 16; off > 0; off >>= 1) {
      const float os = __shfl_xor_sync(0xffffffffu, bs, off);
      const int32_t ot = __shfl_xor_sync(0xffffffffu, bt, off);
      const int ol = __shfl_xor_sync(0xffffffffu, bl, off);
      if (os > bs || (os == bs && ot < bt)) { bs = os; bt = ot; bl = ol; }
    }
    const float z = warp_sum(lane < kTopK ? __expf(sc - bs) : 0.f);
    float cf = 1.f / z;
    if (conf_h) {   // DSpark confidence head: [hidden_p ; w1[token before p]]
      float e = 0.f;
      for (int r = lane; r < kRank; r += 32) e += __bfloat162float(conf_w[r]) * __bfloat162float(pred[(size_t)prev_tok * kRank + r]);
      e = warp_sum(e);
      cf = 1.f / (1.f + __expf(-(conf_h[sq * P + p] + e + conf_b)));
    }
    if (lane == 0) { ids[sq * P + p] = bt; conf[sq * P + p] = cf; }
    prev = bl;
    prev_tok = bt;
  }
}

// top-16 of each row (ties → smaller index): stage 1 = 32 slices per row → 16 each; stage 2 = 512 → 16 (values and indices)
constexpr int kTkSlices = 32;
__device__ __forceinline__ void tk_better(float& bv, int& bi, float ov, int oi) { if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
template <int PER>
__device__ void block_top16(float (&v)[PER], int (&ix)[PER], float* ov, int* oi) {   // 256 threads; out 16 (value, index)
  __shared__ float sv[8]; __shared__ int si[8];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int r = 0; r < kTopK; ++r) {
    float bv = -FLT_MAX; int bi = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < PER; ++j) tk_better(bv, bi, v[j], ix[j]);
    for (int off = 16; off > 0; off >>= 1) tk_better(bv, bi, __shfl_xor_sync(0xffffffffu, bv, off), __shfl_xor_sync(0xffffffffu, bi, off));
    if (lane == 0) { sv[w] = bv; si[w] = bi; }
    __syncthreads();
    bv = sv[0]; bi = si[0];
    for (int k = 1; k < 8; ++k) tk_better(bv, bi, sv[k], si[k]);
    if (threadIdx.x == 0) { ov[r] = bv; oi[r] = bi; }
#pragma unroll
    for (int j = 0; j < PER; ++j) if (ix[j] == bi) v[j] = -FLT_MAX, ix[j] = 0x7fffffff;
    __syncthreads();
  }
}
__global__ void __launch_bounds__(256) top16_stage1(const float* lg, int N, float* pv, int* pi) {
  constexpr int PER = 16;   // slice ≤ 4096
  const int row = blockIdx.y, sl = blockIdx.x, len = (N + kTkSlices - 1) / kTkSlices, a = sl * len, e = min(N, a + len);
  float v[PER]; int ix[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) { const int i = a + threadIdx.x + 256 * j; const bool ok = i < e; v[j] = ok ? lg[(size_t)row * N + i] : -FLT_MAX; ix[j] = ok ? i : 0x7fffffff; }
  block_top16<PER>(v, ix, pv + ((size_t)row * kTkSlices + sl) * kTopK, pi + ((size_t)row * kTkSlices + sl) * kTopK);
}
__global__ void __launch_bounds__(256) top16_stage2(const float* pv, const int* pi, int32_t* out_i, float* out_v) {
  const int row = blockIdx.x;
  float v[2]; int ix[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) { const size_t o = (size_t)row * kTkSlices * kTopK + threadIdx.x + 256 * j; v[j] = pv[o]; ix[j] = pi[o]; }
  __shared__ float ov[kTopK]; __shared__ int oi[kTopK];
  block_top16<2>(v, ix, ov, oi);
  if (threadIdx.x < kTopK) { out_i[row * kTopK + threadIdx.x] = oi[threadIdx.x]; out_v[row * kTopK + threadIdx.x] = ov[threadIdx.x]; }
}

__global__ void to_fp8_kernel(const bf16* w, int K, uint8_t* q, float* scale) {
  __shared__ float sh[32];
  const bf16* r = w + (size_t)blockIdx.x * K;
  float m = 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) m = fmaxf(m, fabsf(__bfloat162float(r[i])));
  m = warp_max(m);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? sh[threadIdx.x] : 0.f; v = warp_max(v); if (threadIdx.x == 0) sh[0] = v; }
  __syncthreads();
  const float sc = sh[0] > 0.f ? sh[0] / 448.f : 1.f;
  if (threadIdx.x == 0) scale[blockIdx.x] = sc;
  uint8_t* o = q + (size_t)blockIdx.x * K;
  for (int i = threadIdx.x; i < K; i += blockDim.x) o[i] = (uint8_t)__nv_cvt_float_to_fp8(__bfloat162float(r[i]) / sc, __NV_SATFINITE, __NV_E4M3);
}

}  // namespace

void rmsnorm(const bf16* x, int ldx, const bf16* w, int M, int K, float eps, bf16* out, int ldo, cudaStream_t st) {
  if (M > 0) rmsnorm_kernel<<<M, 256, 0, st>>>(x, ldx, w, K, eps, out, ldo);
  CUDA_CHECK(cudaGetLastError());
}
void capture(const bf16* h, const bf16* y, int M, int H, bf16* out, int ldo, cudaStream_t st) {
  if (M > 0) capture_kernel<<<dim3(5, M), 256, 0, st>>>(h, y, H, out, ldo);
  CUDA_CHECK(cudaGetLastError());
}
void ctx_kv(const bf16* kv, const bf16* k_norm, const CtxRow* rows, int M, float eps, const float* inv_freq, float mscale, cudaStream_t st) {
  if (M > 0) ctx_kv_kernel<<<dim3(M, kLayers, kKvHeads), kHeadDim, 0, st>>>(kv, k_norm, rows, eps, inv_freq, mscale);
  CUDA_CHECK(cudaGetLastError());
}
void blk_prep(const bf16* proj, const bf16* q_norm, const bf16* k_norm, const BlkSeq* seqs, int S, int B, float eps, const float* inv_freq,
              float mscale, bf16* q, bf16* k, bf16* v, cudaStream_t st) {
  blk_prep_kernel<<<dim3(B * S, kHeads + 2 * kKvHeads), kHeadDim, 0, st>>>(proj, q_norm, k_norm, seqs, B, eps, inv_freq, mscale, q, k, v);
  CUDA_CHECK(cudaGetLastError());
}
void attention_prepare() {
  if (g_grp_part) return;
  if (cudaMalloc(&g_grp_part, (size_t)64 * kKvHeads * kGrpRows * kGrpSplits * (kHeadDim + 2) * 4) != cudaSuccess) { cudaGetLastError(); g_grp_part = nullptr; }
}
void attention(const bf16* q, const bf16* bk, const bf16* bv, const BlkSeq* seqs, int layer, int S, int B, int window, float scale, bf16* out,
               cudaStream_t st) {
  static const bool grp = !(getenv("HIVE_Q27_DF_ATTN_GRP") && strcmp(getenv("HIVE_Q27_DF_ATTN_GRP"), "0") == 0);
  float* part = g_grp_part;
  if (grp && part && B <= 8 && S <= 64) {
    attn_grp_kernel<<<dim3(kKvHeads, S, kGrpSplits), 256, 0, st>>>(q, bk, bv, seqs, layer, B, window, scale, part);
    attn_grp_combine_kernel<<<dim3(kKvHeads, S, kGrpRows), kHeadDim, 0, st>>>(part, B, out);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  attention_kernel<<<dim3(kHeads, B * S), kHeadDim, 0, st>>>(q, bk, bv, seqs, layer, B, window, scale, out);
  CUDA_CHECK(cudaGetLastError());
}
void conv(const bf16* in, int ld_in, const float* dyn, int half, const float* base, int rows, int B, int H, bf16* out, bool residual, cudaStream_t st) {
  conv_kernel<<<dim3((H + 255) / 256, rows), 256, 0, st>>>(in, ld_in, dyn, half, base, B, H, out, residual);
  CUDA_CHECK(cudaGetLastError());
}
void select(const int32_t* cidx, const float* cval, const float* hp, const bf16* pred, const bf16* succ, const int32_t* map, const BlkSeq* seqs,
            int S, float* edges, int32_t* ids, float* conf, cudaStream_t st, const float* conf_h, const bf16* conf_w, float conf_b) {
  select_edges_kernel<<<dim3(S, kBlock - 1), 32 * kTopK, 0, st>>>(cidx, hp, pred, succ, map, seqs, edges);
  select_chain_kernel<<<S, 32, 0, st>>>(cidx, cval, map, edges, ids, conf, seqs, pred, conf_h, conf_w, conf_b);
  CUDA_CHECK(cudaGetLastError());
}
void top16(const float* logits, int rows, int N, float* work, int32_t* idx, float* val, cudaStream_t st) {
  HIVE_CHECK(N <= kTkSlices * 4096, "dflash top16: row too long");
  float* pv = work;
  int* pi = reinterpret_cast<int*>(work + (size_t)rows * kTkSlices * kTopK);
  top16_stage1<<<dim3(kTkSlices, rows), 256, 0, st>>>(logits, N, pv, pi);
  top16_stage2<<<rows, 256, 0, st>>>(pv, pi, idx, val);
  CUDA_CHECK(cudaGetLastError());
}

void to_fp8_rows(const bf16* w, int N, int K, uint8_t* q, float* scale, cudaStream_t st) {
  to_fp8_kernel<<<N, 256, 0, st>>>(w, K, q, scale);
  CUDA_CHECK(cudaGetLastError());
}

namespace {
__global__ void nf_bf16_kernel(const bf16* x, size_t n, int* out) {
  int c = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) c += !isfinite(__bfloat162float(x[i]));
  if (c) atomicAdd(out, c);
}
__global__ void nf_f32_kernel(const float* x, size_t n, int* out) {
  int c = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) c += !isfinite(x[i]);
  if (c) atomicAdd(out, c);
}
}  // namespace
void count_nonfinite_bf16(const bf16* x, size_t n, int* out, cudaStream_t st) { nf_bf16_kernel<<<256, 256, 0, st>>>(x, n, out); }
void count_nonfinite_f32(const float* x, size_t n, int* out, cudaStream_t st) { nf_f32_kernel<<<256, 256, 0, st>>>(x, n, out); }

}  // namespace hive::q27::df
