// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next routed-expert decode on the GPU (rows ≤ 64, every pair's record resident): out[row] += Σ w · expert(x[row]).
//   glm::moe_decode's fast path needs both K dims to be multiples of 1024 (GLM's H 4096 / I 2048); FN's I = 640 sent it to the reference
//   kernels (one warp per output row, e2m1 decoded through a divergent __constant__ lookup) — nsys 2026-10-09: 21 ms of a 39 ms
//   MTP verify step. This path:
//   (1) group: pairs that share a record are gathered (≤ kMem members per group, first appearance order), so a record is read once per
//       step for every row that routed to it;
//   (2) gu: block = (group, 16 rows of I); warp = 2 rows; a row's gate and up weights are read once for all members — e2m1 → f16 with the
//       hardware conversion, one 16-element scale block per lane step, fp32 accumulation;
//   (3) down: block = (group, 32 rows of H); 8 lanes per output row (K = 640 = 40 scale blocks, 5 per lane);
//   (4) reduce: out[row] += parts in pair order (the reference kernels' order — deterministic, no atomics).
//   Rounding points are the reference's: g, u → bf16 after their global scales, y = bf16(silu(g)·u), part = bf16(w2·y·g2) · w.
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>

#include "hive/common.h"
#include "hive/qwen/fn_pdl.h"
#include "hive/glm/glm_moe.h"
#include "hive/qwen/fn_moe.h"

namespace hive::qwen {

namespace {
using bf16 = __nv_bfloat16;
constexpr int kMem = 8;          // rows per group (a record routed by more rows forms another group)
constexpr int kGuRows = 16, kGuThreads = 256;   // 8 warps × 2 rows
constexpr int kDnRows = 32, kDnThreads = 256, kDnLanes = 8;   // 32 rows: 4 rows per warp, 8 lanes each

struct Group { const uint8_t* rec; int m; int p[kMem]; };

__device__ __forceinline__ float e4m3f(uint8_t v) { __nv_fp8_e4m3 t; t.__x = v; return float(t); }

// 16 e2m1 values (8 bytes) → floats
__device__ __forceinline__ void dec16(uint2 q, float (&w)[16]) {
  const uint32_t words[2] = {q.x, q.y};
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const __nv_fp4x2_storage_t byte = (__nv_fp4x2_storage_t)((words[j] >> (8 * b)) & 0xFF);
      const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2(byte, __NV_E2M1);
      const __half2 h = *reinterpret_cast<const __half2*>(&hr);
      const float2 f = __half22float2(h);
      w[j * 8 + b * 2] = f.x; w[j * 8 + b * 2 + 1] = f.y;   // low nibble = even column
    }
}

// one block, one thread per pair: groups of pairs by record in first-appearance order, at most kMem members each (a record routed by more
//   rows starts another group at its (kMem+1)-th pair) — the same groups, in the same order, as a serial scan
constexpr int kGroupMaxPairs = 1024, kGroupThreads = 256;   // 64 decode rows × 10 experts
__global__ void __launch_bounds__(kGroupThreads) group_kernel(const glm::MoePair* __restrict__ pairs, int n, Group* __restrict__ groups,
                                                              int* __restrict__ n_groups) {
  pdl_wait();
  __shared__ const uint8_t* rec[kGroupMaxPairs];
  __shared__ int lead[kGroupMaxPairs], rank[kGroupMaxPairs], start[kGroupMaxPairs];
  for (int p = threadIdx.x; p < n; p += blockDim.x) rec[p] = pairs[p].rec;
  __syncthreads();
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    int l = p, r = 0;
    for (int q = 0; q < p; ++q)
      if (rec[q] == rec[p]) { if (l == p) l = q; ++r; }
    lead[p] = l; rank[p] = r; start[p] = r % kMem == 0;
  }
  __syncthreads();
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    if (!start[p]) continue;
    int gid = 0;
    for (int q = 0; q < p; ++q) gid += start[q];
    Group g;
    g.rec = rec[p]; g.m = 0;
    for (int q = p; q < n && g.m < kMem; ++q)
      if (lead[q] == lead[p] && rank[q] >= rank[p]) g.p[g.m++] = q;
    groups[gid] = g;
  }
  if (threadIdx.x == 0) {
    int G = 0;
    for (int p = 0; p < n; ++p) G += start[p];
    *n_groups = G;
  }
}

__global__ void __launch_bounds__(kGuThreads) gu_kernel(glm::ExpertRecLayout L, const glm::MoePair* __restrict__ pairs, const Group* __restrict__ groups,
                                                        const int* __restrict__ n_groups, const bf16* __restrict__ x, int H, int I, bf16* __restrict__ y) {
  pdl_wait();
  if ((int)blockIdx.y >= *n_groups) return;
  extern __shared__ bf16 xs[];   // [m, H]
  const Group gr = groups[blockIdx.y];
  const int m = gr.m;
  for (int j = 0; j < m; ++j) {
    const bf16* src = x + (size_t)pairs[gr.p[j]].row * H;
    for (int i = threadIdx.x * 8; i < H; i += blockDim.x * 8) *reinterpret_cast<uint4*>(xs + (size_t)j * H + i) = *reinterpret_cast<const uint4*>(src + i);
  }
  __syncthreads();
  const uint8_t* rec = gr.rec;
  const float g1 = *reinterpret_cast<const float*>(rec + L.g), g3 = *reinterpret_cast<const float*>(rec + L.g + 4);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nb = H / 16;   // scale blocks per row
  for (int rr = 0; rr < kGuRows / 8; ++rr) {
    const int i = blockIdx.x * kGuRows + warp * (kGuRows / 8) + rr;
    if (i >= I) break;
    const uint2* wg = reinterpret_cast<const uint2*>(rec + L.w1 + (size_t)i * (H / 2));
    const uint2* wu = reinterpret_cast<const uint2*>(rec + L.w3 + (size_t)i * (H / 2));
    const uint8_t* sg = rec + L.s1 + (size_t)i * nb;
    const uint8_t* su = rec + L.s3 + (size_t)i * nb;
    float ag[kMem], au[kMem];
#pragma unroll
    for (int j = 0; j < kMem; ++j) { ag[j] = 0.f; au[j] = 0.f; }
    for (int c = lane; c < nb; c += 32) {
      float a[16], b[16];
      dec16(__ldg(wg + c), a);
      dec16(__ldg(wu + c), b);
      const float sa = e4m3f(__ldg(sg + c)), sb = e4m3f(__ldg(su + c));
#pragma unroll
      for (int j = 0; j < kMem; ++j) {
        if (j >= m) break;
        const bf16* xr = xs + (size_t)j * H + c * 16;
        float dg = 0.f, du = 0.f;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float xv = __bfloat162float(xr[k]);
          dg = fmaf(a[k], xv, dg);
          du = fmaf(b[k], xv, du);
        }
        ag[j] = fmaf(dg, sa, ag[j]);
        au[j] = fmaf(du, sb, au[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < kMem; ++j) {
      if (j >= m) break;
      float g = ag[j], u = au[j];
      for (int o = 16; o > 0; o >>= 1) { g += __shfl_xor_sync(0xffffffff, g, o); u += __shfl_xor_sync(0xffffffff, u, o); }
      if (lane == 0) {
        g = __bfloat162float(__float2bfloat16(g * g1));
        u = __bfloat162float(__float2bfloat16(u * g3));
        y[(size_t)gr.p[j] * I + i] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
      }
    }
  }
}

__global__ void __launch_bounds__(kDnThreads) down_kernel(glm::ExpertRecLayout L, const glm::MoePair* __restrict__ pairs, const Group* __restrict__ groups,
                                                          const int* __restrict__ n_groups, const bf16* __restrict__ y, int H, int I, float* __restrict__ part) {
  pdl_wait();
  if ((int)blockIdx.y >= *n_groups) return;
  extern __shared__ bf16 ys[];   // [m, I]
  const Group gr = groups[blockIdx.y];
  const int m = gr.m;
  for (int j = 0; j < m; ++j)
    for (int i = threadIdx.x * 8; i < I; i += blockDim.x * 8)
      *reinterpret_cast<uint4*>(ys + (size_t)j * I + i) = *reinterpret_cast<const uint4*>(y + (size_t)gr.p[j] * I + i);
  __syncthreads();
  const uint8_t* rec = gr.rec;
  const float g2 = *reinterpret_cast<const float*>(rec + L.g + 8);
  const int sub = threadIdx.x / kDnLanes, sl = threadIdx.x % kDnLanes;   // 32 rows per block, 8 lanes per row
  const int n = blockIdx.x * kDnRows + sub;
  if (n >= H) return;
  const int nb = I / 16;
  const uint2* wd = reinterpret_cast<const uint2*>(rec + L.w2 + (size_t)n * (I / 2));
  const uint8_t* sd = rec + L.s2 + (size_t)n * nb;
  float acc[kMem];
#pragma unroll
  for (int j = 0; j < kMem; ++j) acc[j] = 0.f;
  for (int c = sl; c < nb; c += kDnLanes) {
    float a[16];
    dec16(__ldg(wd + c), a);
    const float s = e4m3f(__ldg(sd + c));
#pragma unroll
    for (int j = 0; j < kMem; ++j) {
      if (j >= m) break;
      const bf16* yr = ys + (size_t)j * I + c * 16;
      float d = 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k) d = fmaf(a[k], __bfloat162float(yr[k]), d);
      acc[j] = fmaf(d, s, acc[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < kMem; ++j) {
    if (j >= m) break;
    float v = acc[j];
    for (int o = kDnLanes / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (sl == 0) {
      const glm::MoePair pr = pairs[gr.p[j]];
      part[(size_t)gr.p[j] * H + n] = __bfloat162float(__float2bfloat16(v * g2)) * pr.w;
    }
  }
}

__global__ void reduce_kernel(const glm::MoePair* __restrict__ pairs, int n_pairs, const float* __restrict__ part, int H, float* __restrict__ out) {
  pdl_wait();
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= H) return;
  for (int p = 0; p < n_pairs; ++p) out[(size_t)pairs[p].row * H + n] += part[(size_t)p * H + n];
}

// the pairs from the router's device output (build_pairs folded in): pair p = (row p / K, record of expert ids[p] in MoE layer li, w[p]);
//   a non-resident id (not expected on this path) contributes nothing through a valid record of the row, as build_pairs did
__global__ void __launch_bounds__(kGroupThreads) group_ids_kernel(const int32_t* __restrict__ ids, const float* __restrict__ wts, int M, int K,
                                                                  const uint8_t* const* __restrict__ tab, int li, int E, glm::MoePair* __restrict__ pairs,
                                                                  Group* __restrict__ groups, int* __restrict__ n_groups) {
  pdl_wait();
  __shared__ const uint8_t* rec[kGroupMaxPairs];
  __shared__ int lead[kGroupMaxPairs], rank[kGroupMaxPairs], start[kGroupMaxPairs];
  const int n = M * K;
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    const int m = p / K, e = ids[p];
    const uint8_t* r = (e >= 0 && e < E) ? tab[(size_t)li * E + e] : nullptr;
    float wv = wts[p];
    if (!r) {
      wv = 0.f;
      for (int k = 0; k < K && !r; ++k) { const int e2 = ids[m * K + k]; if (e2 >= 0 && e2 < E) r = tab[(size_t)li * E + e2]; }
      if (!r) r = tab[(size_t)li * E];
    }
    rec[p] = r;
    pairs[p] = glm::MoePair{m, r, wv};
  }
  __syncthreads();
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    int l = p, r = 0;
    for (int q = 0; q < p; ++q)
      if (rec[q] == rec[p]) { if (l == p) l = q; ++r; }
    lead[p] = l; rank[p] = r; start[p] = r % kMem == 0;
  }
  __syncthreads();
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    if (!start[p]) continue;
    int gid = 0;
    for (int q = 0; q < p; ++q) gid += start[q];
    Group g;
    g.rec = rec[p]; g.m = 0;
    for (int q = p; q < n && g.m < kMem; ++q)
      if (lead[q] == lead[p] && rank[q] >= rank[p]) g.p[g.m++] = q;
    groups[gid] = g;
  }
  if (threadIdx.x == 0) {
    int G = 0;
    for (int p = 0; p < n; ++p) G += start[p];
    *n_groups = G;
  }
}

// ---- v2 (default; fn_moe_decode_set_ref(true) keeps the kernels above for the bit-exactness test) -----------------------------------------
// nsys 2026-10-10, 6 sequences × 5 MTP rows with distinct prompts: gu 10.4 + down 4.4 + group_ids 0.97 ms per step (48 layers) — a third
//   of the batched verify. Same arithmetic, same order, so the outputs are bit-identical:
//   · group: one pass by expert id (shared counters + a block scan) instead of the O(n²) first-appearance scan; a group's members are any kMem
//     pairs of one record (each pair's result depends only on its own row, so the grouping never changes a value);
//   · gu / down: the activation rows are read from shared memory 16 bytes at a time (the scalar bf16 loads were 16 LDS per member per
//     scale block), and the next weight chunk is loaded before the current one is used.
__device__ __forceinline__ void x16(const bf16* __restrict__ p, float (&v)[16]) {   // 16 bf16 (32-byte aligned) → floats
  const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i]));
    v[2 * i] = f.x; v[2 * i + 1] = f.y;
  }
}
constexpr int kKeyMax = 2048;   // expert keys (E ≤ 1024) + one private key per pair without a resident record of its own
__global__ void __launch_bounds__(kGroupThreads) group_ids2_kernel(const int32_t* __restrict__ ids, const float* __restrict__ wts, int M, int K,
                                                                   const uint8_t* const* __restrict__ tab, int li, int E, glm::MoePair* __restrict__ pairs,
                                                                   Group* __restrict__ groups, int* __restrict__ n_groups,
                                                                   const uint8_t* const* __restrict__ host_tab, int pcie_div) {
  pdl_wait();
  __shared__ int cnt[kKeyMax], base[kKeyMax];
  __shared__ int mrank[1024];
  __shared__ int key[kGroupMaxPairs], rank[kGroupMaxPairs];
  __shared__ const uint8_t* rec[kGroupMaxPairs];
  __shared__ int wsum[kGroupThreads / 32];
  const int n = M * K, nk = E + n;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) cnt[i] = 0;
  // hybrid route with a PCIe share (pcie_div > 0): the non-resident experts this layer uses, ranked by id; rank % pcie_div == pcie_div − 1
  //   reads its record from pinned host memory (host_tab, over PCIe) instead of leaving it to the CPU — QwenEngine::hyb_cpu applies the
  //   same rule to the same routing, so every pair is computed exactly once
  const bool split = host_tab && pcie_div > 0 && E <= 1024;
  if (split) {
    for (int i = threadIdx.x; i < E; i += blockDim.x) mrank[i] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < n; p += blockDim.x) { const int e = ids[p]; if (e >= 0 && e < E && !tab[(size_t)li * E + e]) mrank[e] = 1; }
    __syncthreads();
    if (threadIdx.x < 32) {   // exclusive scan of the flags (warp 0: a contiguous chunk per lane)
      const int ln = threadIdx.x, per = (E + 31) / 32, a = ln * per, b = min(E, a + per);
      int loc = 0;
      for (int i = a; i < b; ++i) loc += mrank[i];
      int inc = loc;
      for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffff, inc, o); if (ln >= o) inc += v; }
      int off = inc - loc;
      for (int i = a; i < b; ++i) { const int f = mrank[i]; mrank[i] = f ? off : -1; off += f; }
    }
  }
  __syncthreads();
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    const int m = p / K, e = ids[p];
    const uint8_t* r = (e >= 0 && e < E) ? tab[(size_t)li * E + e] : nullptr;
    if (!r && split && e >= 0 && e < E && mrank[e] >= 0 && mrank[e] % pcie_div == pcie_div - 1) r = host_tab[(size_t)li * E + e];
    float wv = wts[p];
    // a pair without a record (hybrid route: the expert is not resident — the CPU computes it, its result arrives as `add` of the
    //   reduce) joins no group and is left out of the reduce; it never occurs while every routed expert is resident
    const int kk = r ? e : -1;
    rec[p] = r; key[p] = kk;
    pairs[p] = glm::MoePair{m, r, r ? wv : 0.f};
    rank[p] = kk >= 0 ? atomicAdd(&cnt[kk], 1) : 0;
  }
  __syncthreads();
  // groups per key = ceil(cnt / kMem); exclusive scan over the keys (chunk per thread, then a warp/block scan of the chunk sums)
  const int per = (nk + blockDim.x - 1) / blockDim.x, k0 = threadIdx.x * per;
  int loc = 0;
  for (int i = k0; i < min(k0 + per, nk); ++i) loc += (cnt[i] + kMem - 1) / kMem;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int inc = loc;
  for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffff, inc, o); if (lane >= o) inc += v; }
  if (lane == 31) wsum[warp] = inc;
  __syncthreads();
  int off = inc - loc;
  for (int w = 0; w < warp; ++w) off += wsum[w];
  for (int i = k0; i < min(k0 + per, nk); ++i) { base[i] = off; off += (cnt[i] + kMem - 1) / kMem; }
  if (threadIdx.x == blockDim.x - 1) *n_groups = off;
  __syncthreads();
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    if (key[p] < 0) continue;
    const int kk = key[p], r = rank[p], gi = base[kk] + r / kMem, sl = r % kMem;
    groups[gi].p[sl] = p;
    if (sl == 0) { groups[gi].rec = rec[p]; groups[gi].m = min(kMem, cnt[kk] - (r / kMem) * kMem); }
  }
}

template <int RPW>   // rows of I per warp (block = 8 warps · RPW rows)
__global__ void __launch_bounds__(kGuThreads) gu2_kernel(glm::ExpertRecLayout L, const glm::MoePair* __restrict__ pairs, const Group* __restrict__ groups,
                                                         const int* __restrict__ n_groups, const bf16* __restrict__ x, int H, int I, bf16* __restrict__ y) {
  pdl_wait();
  if ((int)blockIdx.y >= *n_groups) return;
  extern __shared__ __align__(16) unsigned char gsm[];
  bf16* xs = reinterpret_cast<bf16*>(gsm);   // [m, H]
  const Group gr = groups[blockIdx.y];
  const int m = gr.m;
  for (int j = 0; j < m; ++j) {
    const bf16* src = x + (size_t)pairs[gr.p[j]].row * H;
    for (int i = threadIdx.x * 8; i < H; i += blockDim.x * 8) *reinterpret_cast<uint4*>(xs + (size_t)j * H + i) = *reinterpret_cast<const uint4*>(src + i);
  }
  __syncthreads();
  const uint8_t* rec = gr.rec;
  const float g1 = *reinterpret_cast<const float*>(rec + L.g), g3 = *reinterpret_cast<const float*>(rec + L.g + 4);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nb = H / 16;
  for (int rr = 0; rr < RPW; ++rr) {
    const int i = blockIdx.x * (8 * RPW) + warp * RPW + rr;
    if (i >= I) break;
    const uint2* wg = reinterpret_cast<const uint2*>(rec + L.w1 + (size_t)i * (H / 2));
    const uint2* wu = reinterpret_cast<const uint2*>(rec + L.w3 + (size_t)i * (H / 2));
    const uint8_t* sg = rec + L.s1 + (size_t)i * nb;
    const uint8_t* su = rec + L.s3 + (size_t)i * nb;
    float ag[kMem], au[kMem];
#pragma unroll
    for (int j = 0; j < kMem; ++j) { ag[j] = 0.f; au[j] = 0.f; }
    uint2 qg = make_uint2(0, 0), qu = qg;
    uint8_t cg = 0, cu = 0;
    if (lane < nb) { qg = __ldg(wg + lane); qu = __ldg(wu + lane); cg = __ldg(sg + lane); cu = __ldg(su + lane); }
    for (int c = lane; c < nb; c += 32) {
      const uint2 tg = qg, tu = qu;
      const uint8_t tcg = cg, tcu = cu;
      if (c + 32 < nb) { qg = __ldg(wg + c + 32); qu = __ldg(wu + c + 32); cg = __ldg(sg + c + 32); cu = __ldg(su + c + 32); }
      float a[16], b[16];
      dec16(tg, a);
      dec16(tu, b);
      const float sa = e4m3f(tcg), sb = e4m3f(tcu);
#pragma unroll
      for (int j = 0; j < kMem; ++j) {
        if (j >= m) break;
        float xv[16];
        x16(xs + (size_t)j * H + c * 16, xv);
        float dg = 0.f, du = 0.f;
#pragma unroll
        for (int k = 0; k < 16; ++k) { dg = fmaf(a[k], xv[k], dg); du = fmaf(b[k], xv[k], du); }
        ag[j] = fmaf(dg, sa, ag[j]);
        au[j] = fmaf(du, sb, au[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < kMem; ++j) {
      if (j >= m) break;
      float g = ag[j], u = au[j];
      for (int o = 16; o > 0; o >>= 1) { g += __shfl_xor_sync(0xffffffff, g, o); u += __shfl_xor_sync(0xffffffff, u, o); }
      if (lane == 0) {
        g = __bfloat162float(__float2bfloat16(g * g1));
        u = __bfloat162float(__float2bfloat16(u * g3));
        y[(size_t)gr.p[j] * I + i] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
      }
    }
  }
}

template <int RPT>   // output rows per 8-lane group (block = 32 · RPT rows; the activation rows are staged once per block)
__global__ void __launch_bounds__(kDnThreads) down2_kernel(glm::ExpertRecLayout L, const glm::MoePair* __restrict__ pairs, const Group* __restrict__ groups,
                                                           const int* __restrict__ n_groups, const bf16* __restrict__ y, int H, int I, float* __restrict__ part) {
  pdl_wait();
  if ((int)blockIdx.y >= *n_groups) return;
  extern __shared__ __align__(16) unsigned char dsm[];
  bf16* ys = reinterpret_cast<bf16*>(dsm);   // [m, I]
  const Group gr = groups[blockIdx.y];
  const int m = gr.m;
  for (int j = 0; j < m; ++j)
    for (int i = threadIdx.x * 8; i < I; i += blockDim.x * 8)
      *reinterpret_cast<uint4*>(ys + (size_t)j * I + i) = *reinterpret_cast<const uint4*>(y + (size_t)gr.p[j] * I + i);
  __syncthreads();
  const uint8_t* rec = gr.rec;
  const float g2 = *reinterpret_cast<const float*>(rec + L.g + 8);
  const int sub = threadIdx.x / kDnLanes, sl = threadIdx.x % kDnLanes;
  const int nb = I / 16;
  for (int rt = 0; rt < RPT; ++rt) {
  const int n = (blockIdx.x * RPT + rt) * kDnRows + sub;
  if (n >= H) return;
  const uint2* wd = reinterpret_cast<const uint2*>(rec + L.w2 + (size_t)n * (I / 2));
  const uint8_t* sd = rec + L.s2 + (size_t)n * nb;
  float acc[kMem];
#pragma unroll
  for (int j = 0; j < kMem; ++j) acc[j] = 0.f;
  uint2 q = make_uint2(0, 0);
  uint8_t cs = 0;
  if (sl < nb) { q = __ldg(wd + sl); cs = __ldg(sd + sl); }
  for (int c = sl; c < nb; c += kDnLanes) {
    const uint2 t = q;
    const uint8_t tc = cs;
    if (c + kDnLanes < nb) { q = __ldg(wd + c + kDnLanes); cs = __ldg(sd + c + kDnLanes); }
    float a[16];
    dec16(t, a);
    const float s = e4m3f(tc);
#pragma unroll
    for (int j = 0; j < kMem; ++j) {
      if (j >= m) break;
      float yv[16];
      x16(ys + (size_t)j * I + c * 16, yv);
      float d = 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k) d = fmaf(a[k], yv[k], d);
      acc[j] = fmaf(d, s, acc[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < kMem; ++j) {
    if (j >= m) break;
    float v = acc[j];
    for (int o = kDnLanes / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (sl == 0) {
      const glm::MoePair pr = pairs[gr.p[j]];
      part[(size_t)gr.p[j] * H + n] = __bfloat162float(__float2bfloat16(v * g2)) * pr.w;
    }
  }
  }
}
bool g_moe_ref = false;

// out[m, n] = bf16( bf16(Σ_k part[m·K + k, n]) + bf16( bf16(sigmoid(bf16(gate logit))) · shared[m, n] ) )  — the reference reduce (pair
//   order: k ascending) followed by moe_combine, in one pass
//   pairs (optional): pairs without a record are skipped · add (optional, fp32 [M, H], may be pinned host memory): added after the parts
//   (the CPU share of the hybrid route, already weighted) — both absent / never taken while every expert is resident: the same sum
__global__ void reduce_combine_kernel(const float* __restrict__ part, int M, int K, int H, const bf16* __restrict__ shared, const float* __restrict__ lg,
                                      int ld, int col, bf16* __restrict__ out, const glm::MoePair* __restrict__ pairs = nullptr,
                                      const float* __restrict__ add = nullptr) {
  pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * H) return;
  const int m = i / H, n = i % H;
  float v = 0.f;
  for (int k = 0; k < K; ++k) {
    if (pairs && !pairs[m * K + k].rec) continue;
    v += part[(size_t)(m * K + k) * H + n];
  }
  if (add) v += add[i];
  const float rbf_l = __bfloat162float(__float2bfloat16(lg[(size_t)m * ld + col]));
  const float g = __bfloat162float(__float2bfloat16(1.f / (1.f + expf(-rbf_l))));
  out[i] = __float2bfloat16(__bfloat162float(__float2bfloat16(v)) + __bfloat162float(__float2bfloat16(g * __bfloat162float(shared[i]))));
}

// ---- FP8 experts (the MTP layer: e4m3 with 128×128 block scales, glm::Fp8BMat triples in a device table) -----------------------------------
// groups by expert id (the table entry), then the same structure as the NVFP4 kernels; 16 weights per lane step, block scale per 128 columns
__device__ __forceinline__ void dec16_e4m3(uint4 q, float (&w)[16]) {
  const uint32_t words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)((words[j] >> (16 * b)) & 0xFFFF), __NV_E4M3);
      const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
      w[j * 4 + b * 2] = f.x; w[j * 4 + b * 2 + 1] = f.y;
    }
}
__global__ void __launch_bounds__(kGroupThreads) group_e_kernel(const int32_t* __restrict__ ids, const float* __restrict__ wts, int M, int K, int E,
                                                                const glm::Fp8BMat* __restrict__ tab, glm::MoePair* __restrict__ pairs,
                                                                Group* __restrict__ groups, int* __restrict__ n_groups) {
  pdl_wait();
  __shared__ const uint8_t* rec[kGroupMaxPairs];
  __shared__ int lead[kGroupMaxPairs], rank[kGroupMaxPairs], start[kGroupMaxPairs];
  const int n = M * K;
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    const int e = ids[p];
    const bool ok = e >= 0 && e < E;
    rec[p] = reinterpret_cast<const uint8_t*>(tab + 3 * (ok ? e : 0));
    pairs[p] = glm::MoePair{p / K, rec[p], ok ? wts[p] : 0.f};
  }
  __syncthreads();
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    int l = p, r = 0;
    for (int q = 0; q < p; ++q)
      if (rec[q] == rec[p]) { if (l == p) l = q; ++r; }
    lead[p] = l; rank[p] = r; start[p] = r % kMem == 0;
  }
  __syncthreads();
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    if (!start[p]) continue;
    int gid = 0;
    for (int q = 0; q < p; ++q) gid += start[q];
    Group g;
    g.rec = rec[p]; g.m = 0;
    for (int q = p; q < n && g.m < kMem; ++q)
      if (lead[q] == lead[p] && rank[q] >= rank[p]) g.p[g.m++] = q;
    groups[gid] = g;
  }
  if (threadIdx.x == 0) { int G = 0; for (int p = 0; p < n; ++p) G += start[p]; *n_groups = G; }
}
__device__ __forceinline__ float fp8_row_dot(const glm::Fp8BMat& W, int row, const bf16* xs, int lane, int lanes) {
  const int K = W.K, nk = (K + 127) / 128;
  const uint4* wr = reinterpret_cast<const uint4*>(W.w + (size_t)row * K);
  const float* sr = W.s + (size_t)(row / 128) * nk;
  float acc = 0.f;
  for (int c = lane; c < K / 16; c += lanes) {
    float a[16];
    dec16_e4m3(__ldg(wr + c), a);
    float xv[16];
    x16(xs + c * 16, xv);   // 16-byte shared loads (were 16 scalar ones), same products in the same order
    float d = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) d = fmaf(a[k], xv[k], d);
    acc = fmaf(d, __ldg(sr + (c * 16) / 128), acc);
  }
  return acc;
}
__global__ void __launch_bounds__(kGuThreads) gu8_kernel(const glm::MoePair* __restrict__ pairs, const Group* __restrict__ groups, const int* __restrict__ n_groups,
                                                         const bf16* __restrict__ x, int H, int I, bf16* __restrict__ y) {
  pdl_wait();
  if ((int)blockIdx.y >= *n_groups) return;
  extern __shared__ __align__(16) bf16 xs[];
  const Group gr = groups[blockIdx.y];
  const glm::Fp8BMat* t = reinterpret_cast<const glm::Fp8BMat*>(gr.rec);
  const glm::Fp8BMat G = t[0], U = t[1];
  for (int j = 0; j < gr.m; ++j)
    for (int i = threadIdx.x * 8; i < H; i += blockDim.x * 8)
      *reinterpret_cast<uint4*>(xs + (size_t)j * H + i) = *reinterpret_cast<const uint4*>(x + (size_t)pairs[gr.p[j]].row * H + i);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int rr = 0; rr < kGuRows / 8; ++rr) {
    const int i = blockIdx.x * kGuRows + warp * (kGuRows / 8) + rr;
    if (i >= I) break;
    for (int j = 0; j < gr.m; ++j) {
      float g = fp8_row_dot(G, i, xs + (size_t)j * H, lane, 32), u = fp8_row_dot(U, i, xs + (size_t)j * H, lane, 32);
      for (int o = 16; o > 0; o >>= 1) { g += __shfl_xor_sync(0xffffffff, g, o); u += __shfl_xor_sync(0xffffffff, u, o); }
      if (lane == 0) {
        g = __bfloat162float(__float2bfloat16(g)); u = __bfloat162float(__float2bfloat16(u));
        y[(size_t)gr.p[j] * I + i] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
      }
    }
  }
}
__global__ void __launch_bounds__(kDnThreads) down8_kernel(const glm::MoePair* __restrict__ pairs, const Group* __restrict__ groups, const int* __restrict__ n_groups,
                                                           const bf16* __restrict__ y, int H, int I, float* __restrict__ part) {
  pdl_wait();
  if ((int)blockIdx.y >= *n_groups) return;
  extern __shared__ __align__(16) bf16 ys[];
  const Group gr = groups[blockIdx.y];
  const glm::Fp8BMat D = reinterpret_cast<const glm::Fp8BMat*>(gr.rec)[2];
  for (int j = 0; j < gr.m; ++j)
    for (int i = threadIdx.x * 8; i < I; i += blockDim.x * 8)
      *reinterpret_cast<uint4*>(ys + (size_t)j * I + i) = *reinterpret_cast<const uint4*>(y + (size_t)gr.p[j] * I + i);
  __syncthreads();
  const int sub = threadIdx.x / kDnLanes, sl = threadIdx.x % kDnLanes;
  const int n = blockIdx.x * kDnRows + sub;
  if (n >= H) return;
  for (int j = 0; j < gr.m; ++j) {
    float v = fp8_row_dot(D, n, ys + (size_t)j * I, sl, kDnLanes);
    for (int o = kDnLanes / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (sl == 0) part[(size_t)gr.p[j] * H + n] = __bfloat162float(__float2bfloat16(v)) * pairs[gr.p[j]].w;
  }
}

size_t align256(size_t b) { return (b + 255) / 256 * 256; }
}  // namespace

void fn_moe_decode_set_ref(bool ref) { g_moe_ref = ref; }
bool fn_moe_decode_supported(int H, int I) { return H % 256 == 0 && I % 128 == 0 && H % kDnRows == 0; }

size_t fn_moe_decode_ws_bytes(int n_pairs, int H, int I) {
  return align256((size_t)n_pairs * I * 2) + align256((size_t)n_pairs * H * 4) + align256((size_t)n_pairs * sizeof(Group)) + 256 +
         align256((size_t)n_pairs * sizeof(glm::MoePair));
}

void fn_moe_fp8_routed(const glm::Fp8BMat* table, int E, const int32_t* ids, const float* w, int M, int K, const bf16* x, int H, int I,
                       const bf16* shared, const float* lg, int ld, int col, bf16* out, void* ws, cudaStream_t st) {
  const int n = M * K;
  if (n <= 0) return;
  HIVE_CHECK(n <= kGroupMaxPairs && (size_t)kMem * H * 2 <= 48 * 1024 && H % 16 == 0 && I % 16 == 0, "fn_moe_fp8_routed: shape");
  uint8_t* p = reinterpret_cast<uint8_t*>(ws);
  bf16* y = reinterpret_cast<bf16*>(p); p += align256((size_t)n * I * 2);
  float* part = reinterpret_cast<float*>(p); p += align256((size_t)n * H * 4);
  Group* groups = reinterpret_cast<Group*>(p); p += align256((size_t)n * sizeof(Group));
  int* n_groups = reinterpret_cast<int*>(p); p += 256;
  glm::MoePair* pairs = reinterpret_cast<glm::MoePair*>(p);
  launch_pdl(group_e_kernel, dim3(1), dim3(kGroupThreads), 0, st, ids, w, M, K, E, table, pairs, groups, n_groups);
  CUDA_CHECK(cudaGetLastError());
  launch_pdl(gu8_kernel, dim3(dim3((I + kGuRows - 1) / kGuRows, n)), dim3(kGuThreads), (size_t)kMem * H * 2, st, pairs, groups, n_groups, x, H, I, y);
  CUDA_CHECK(cudaGetLastError());
  launch_pdl(down8_kernel, dim3(dim3(H / kDnRows, n)), dim3(kDnThreads), (size_t)kMem * I * 2, st, pairs, groups, n_groups, y, H, I, part);
  CUDA_CHECK(cudaGetLastError());
  launch_pdl(reduce_combine_kernel, dim3((M * H + 255) / 256), dim3(256), 0, st, part, M, K, H, shared, lg, ld, col, out, (const glm::MoePair*)nullptr, (const float*)nullptr);
  CUDA_CHECK(cudaGetLastError());
}

void fn_moe_decode_routed(const glm::ExpertRecLayout& L, const int32_t* ids, const float* w, int M, int K, const uint8_t* const* rec_tab, int li, int E,
                          const bf16* x, int H, int I, const bf16* shared, const float* lg, int ld, int col, bf16* out, void* ws, cudaStream_t st,
                          const float* add, cudaEvent_t join, const uint8_t* const* host_tab, int pcie_div) {
  const int n = M * K;
  if (n <= 0) return;
  HIVE_CHECK(n <= kGroupMaxPairs && (size_t)kMem * H * 2 <= 48 * 1024, "fn_moe_decode_routed: shape");
  uint8_t* p = reinterpret_cast<uint8_t*>(ws);
  bf16* y = reinterpret_cast<bf16*>(p); p += align256((size_t)n * I * 2);
  float* part = reinterpret_cast<float*>(p); p += align256((size_t)n * H * 4);
  Group* groups = reinterpret_cast<Group*>(p); p += align256((size_t)n * sizeof(Group));
  int* n_groups = reinterpret_cast<int*>(p); p += 256;
  glm::MoePair* pairs = reinterpret_cast<glm::MoePair*>(p);
  if (g_moe_ref) {
    launch_pdl(group_ids_kernel, dim3(1), dim3(kGroupThreads), 0, st, ids, w, M, K, rec_tab, li, E, pairs, groups, n_groups);
    CUDA_CHECK(cudaGetLastError());
    launch_pdl(gu_kernel, dim3(dim3((I + kGuRows - 1) / kGuRows, n)), dim3(kGuThreads), (size_t)kMem * H * 2, st, L, pairs, groups, n_groups, x, H, I, y);
    CUDA_CHECK(cudaGetLastError());
    launch_pdl(down_kernel, dim3(dim3(H / kDnRows, n)), dim3(kDnThreads), (size_t)kMem * I * 2, st, L, pairs, groups, n_groups, y, H, I, part);
  } else {
    HIVE_CHECK(E + n <= kKeyMax && H % 16 == 0 && I % 16 == 0, "fn_moe_decode_routed: v2 shape");
    launch_pdl(group_ids2_kernel, dim3(1), dim3(kGroupThreads), 0, st, ids, w, M, K, rec_tab, li, E, pairs, groups, n_groups, host_tab, pcie_div);
    CUDA_CHECK(cudaGetLastError());
    // block shapes (HIVE_QWEN_MOE_GU_RPW rows per warp 2|4, HIVE_QWEN_MOE_DN_RPT row groups per block 1|2|4) — per-row arithmetic unchanged
    static const int rpw = getenv("HIVE_QWEN_MOE_GU_RPW") && atoi(getenv("HIVE_QWEN_MOE_GU_RPW")) >= 4 ? 4 : 2;
    static const int rpt = getenv("HIVE_QWEN_MOE_DN_RPT") ? std::max(1, std::min(4, atoi(getenv("HIVE_QWEN_MOE_DN_RPT")))) : 1;
    const dim3 gg((I + 8 * rpw - 1) / (8 * rpw), n);
    if (rpw == 4) launch_pdl(gu2_kernel<4>, gg, dim3(kGuThreads), (size_t)kMem * H * 2, st, L, pairs, groups, n_groups, x, H, I, y);
    else launch_pdl(gu2_kernel<2>, gg, dim3(kGuThreads), (size_t)kMem * H * 2, st, L, pairs, groups, n_groups, x, H, I, y);
    CUDA_CHECK(cudaGetLastError());
    const int rt = rpt == 3 ? 2 : rpt;
    const dim3 dg((H / kDnRows + rt - 1) / rt, n);
    if (rt == 4) launch_pdl(down2_kernel<4>, dg, dim3(kDnThreads), (size_t)kMem * I * 2, st, L, pairs, groups, n_groups, y, H, I, part);
    else if (rt == 2) launch_pdl(down2_kernel<2>, dg, dim3(kDnThreads), (size_t)kMem * I * 2, st, L, pairs, groups, n_groups, y, H, I, part);
    else launch_pdl(down2_kernel<1>, dg, dim3(kDnThreads), (size_t)kMem * I * 2, st, L, pairs, groups, n_groups, y, H, I, part);
  }
  CUDA_CHECK(cudaGetLastError());
  if (join) CUDA_CHECK(cudaStreamWaitEvent(st, join, 0));   // the CPU share (hybrid route) is complete
  launch_pdl(reduce_combine_kernel, dim3((M * H + 255) / 256), dim3(256), 0, st, part, M, K, H, shared, lg, ld, col, out,
             (const glm::MoePair*)(g_moe_ref ? nullptr : pairs), add);
  CUDA_CHECK(cudaGetLastError());
}

void fn_moe_decode(const glm::ExpertRecLayout& L, const glm::MoePair* pairs, int n, const bf16* x, int H, int I, float* out, void* ws,
                   cudaStream_t st) {
  if (n <= 0) return;
  uint8_t* p = reinterpret_cast<uint8_t*>(ws);
  bf16* y = reinterpret_cast<bf16*>(p); p += align256((size_t)n * I * 2);
  float* part = reinterpret_cast<float*>(p); p += align256((size_t)n * H * 4);
  Group* groups = reinterpret_cast<Group*>(p); p += align256((size_t)n * sizeof(Group));
  int* n_groups = reinterpret_cast<int*>(p);
  HIVE_CHECK((size_t)kMem * H * 2 <= 48 * 1024, "fn_moe_decode: activation rows exceed the default shared memory");
  HIVE_CHECK(n <= kGroupMaxPairs, "fn_moe_decode: too many pairs");
  launch_pdl(group_kernel, dim3(1), dim3(kGroupThreads), 0, st, pairs, n, groups, n_groups);
  CUDA_CHECK(cudaGetLastError());
  launch_pdl(gu_kernel, dim3(dim3((I + kGuRows - 1) / kGuRows, n)), dim3(kGuThreads), (size_t)kMem * H * 2, st, L, pairs, groups, n_groups, x, H, I, y);
  CUDA_CHECK(cudaGetLastError());
  launch_pdl(down_kernel, dim3(dim3(H / kDnRows, n)), dim3(kDnThreads), (size_t)kMem * I * 2, st, L, pairs, groups, n_groups, y, H, I, part);
  CUDA_CHECK(cudaGetLastError());
  launch_pdl(reduce_kernel, dim3((H + 255) / 256), dim3(256), 0, st, pairs, n, part, H, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::qwen

// ---- FP8 block-scaled copy of a bf16 matrix (draft head) -----------------------------------------------------------------------------------
namespace hive::qwen {
namespace {
// block (128 rows × 128 cols): s = max|w| / 448, q = e4m3(w / s) (round to nearest, saturating)
__global__ void quant_fp8b_kernel(const __nv_bfloat16* __restrict__ w, int N, int K, uint8_t* __restrict__ q, float* __restrict__ s) {
  const int bn = blockIdx.y, bk = blockIdx.x, nk = (K + 127) / 128;
  const int n0 = bn * 128, k0 = bk * 128;
  __shared__ float red[256];
  float amax = 0.f;
  for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
    const int n = n0 + i / 128, k = k0 + i % 128;
    if (n < N && k < K) amax = fmaxf(amax, fabsf(__bfloat162float(w[(size_t)n * K + k])));
  }
  red[threadIdx.x] = amax;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) { if ((int)threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
  const float sc = red[0] > 0.f ? red[0] / 448.f : 1.f;
  if (threadIdx.x == 0) s[(size_t)bn * nk + bk] = sc;
  const float inv = 1.f / sc;
  for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
    const int n = n0 + i / 128, k = k0 + i % 128;
    if (n < N && k < K) q[(size_t)n * K + k] = __nv_fp8_e4m3(__bfloat162float(w[(size_t)n * K + k]) * inv).__x;
  }
}
__global__ void bf16_to_f32_kernel(const __nv_bfloat16* __restrict__ a, size_t n, float* __restrict__ b) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = __bfloat162float(a[i]);
}
__global__ void gather_rows_kernel(const __nv_bfloat16* __restrict__ src, const int32_t* __restrict__ rows, int K, __nv_bfloat16* __restrict__ dst) {
  const __nv_bfloat16* a = src + (size_t)rows[blockIdx.x] * K;
  __nv_bfloat16* b = dst + (size_t)blockIdx.x * K;
  for (int i = threadIdx.x; i < K; i += blockDim.x) b[i] = a[i];
}
// draft pick: per block of kPickPer logits the max, its lowest index and Σexp(l − block max); then one warp combines the blocks in order
constexpr int kPickPer = 2048;
__global__ void pick_part_kernel(const __nv_bfloat16* __restrict__ lg, int V, float* __restrict__ part) {
  pdl_wait();
  __shared__ float sm[8], ss[8];
  __shared__ int si[8];
  const int v0 = blockIdx.x * kPickPer, v1 = min(V, v0 + kPickPer), lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  float mx = -INFINITY; int mi = 0x7fffffff;
  for (int v = v0 + threadIdx.x; v < v1; v += blockDim.x) { const float x = __bfloat162float(lg[v]); if (x > mx) { mx = x; mi = v; } }
  for (int o = 16; o > 0; o >>= 1) {
    const float om = __shfl_xor_sync(0xffffffffu, mx, o); const int oi = __shfl_xor_sync(0xffffffffu, mi, o);
    if (om > mx || (om == mx && oi < mi)) { mx = om; mi = oi; }
  }
  if (lane == 0) { sm[w] = mx; si[w] = mi; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < (int)(blockDim.x >> 5); ++i) if (sm[i] > sm[0] || (sm[i] == sm[0] && si[i] < si[0])) { sm[0] = sm[i]; si[0] = si[i]; }
  }
  __syncthreads();
  const float bm = sm[0];
  float s = 0.f;
  for (int v = v0 + threadIdx.x; v < v1; v += blockDim.x) s += expf(__bfloat162float(lg[v]) - bm);
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) ss[w] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) t += ss[i];
    part[3 * blockIdx.x] = bm; part[3 * blockIdx.x + 1] = t; part[3 * blockIdx.x + 2] = __int_as_float(si[0]);
  }
}
__global__ void pick_final_kernel(const float* __restrict__ part, int nb, const int32_t* __restrict__ map, int32_t* __restrict__ id,
                                  float* __restrict__ conf) {
  pdl_wait();
  if (threadIdx.x != 0) return;
  float mx = -INFINITY; int mi = 0;
  for (int b = 0; b < nb; ++b) if (part[3 * b] > mx) { mx = part[3 * b]; mi = __float_as_int(part[3 * b + 2]); }
  float s = 0.f;
  for (int b = 0; b < nb; ++b) s += part[3 * b + 1] * expf(part[3 * b] - mx);
  *id = map ? map[mi] : mi;
  conf[0] = mx; conf[1] = s;
}
}  // namespace

int fn_draft_pick_parts(int V) { return (V + kPickPer - 1) / kPickPer; }
void fn_draft_pick(const __nv_bfloat16* lg, int V, const int32_t* map, int32_t* id, float* conf, float* part, cudaStream_t st) {
  const int nb = fn_draft_pick_parts(V);
  launch_pdl(pick_part_kernel, dim3(nb), dim3(256), 0, st, lg, V, part);
  launch_pdl(pick_final_kernel, dim3(1), dim3(32), 0, st, part, nb, map, id, conf);
  CUDA_CHECK(cudaGetLastError());
}

void fn_quant_fp8b(const __nv_bfloat16* w, int N, int K, uint8_t* q, float* s, cudaStream_t st) {
  quant_fp8b_kernel<<<dim3((K + 127) / 128, (N + 127) / 128), 256, 0, st>>>(w, N, K, q, s);
  CUDA_CHECK(cudaGetLastError());
}
void fn_gather_rows(const __nv_bfloat16* src, const int32_t* rows, int n, int K, __nv_bfloat16* dst, cudaStream_t st) {
  if (n <= 0) return;
  gather_rows_kernel<<<n, 256, 0, st>>>(src, rows, K, dst);
  CUDA_CHECK(cudaGetLastError());
}
void fn_bf16_to_f32(const __nv_bfloat16* a, size_t n, float* b, cudaStream_t st) {
  bf16_to_f32_kernel<<<(unsigned)std::min<size_t>((n + 255) / 256, 4096), 256, 0, st>>>(a, n, b);
  CUDA_CHECK(cudaGetLastError());
}
}  // namespace hive::qwen
