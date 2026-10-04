// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// E3 HIVE_HC_DECODE_FUSED — decode (M ≤ 8) hc sinkhorn moved into the mixing-coefficient kernel (solved by the last block) · 16-lane parallel sinkhorn.
//   Included by model_kernels.cu (hc_mix_pre_norm_sk) · gemv_decode.cu (hc_mix_pre_norm2_sk) · hc_decode_fused.cu (standalone
//   parallel sinkhorn) (the build has no rdc, so device functions live in a header — each translation unit compiles the same
//   source with the same flags).
//
// Numeric contract (bit-identical — test_hc_decode): **same formula and same order per element** as model_kernels.cu
//   sinkhorn_one<4> (one thread = one token, 4×4 serial).
//   Lane l (0..15) holds one matrix element (j, kk) = (l/4, l%4) in a register. When a row/column sum is needed, it gathers the
//   four values with 16-lane shuffles and every lane adds them **in the same order as the serial version** (s = 0.f; s += v0;
//   s += v1; s += v2; s += v3 — rows in ascending kk, columns in ascending j).
//   The maximum likewise starts at -FLT_MAX and applies fmaxf in ascending kk. The pre/post/comb initial values, softmax and
//   c/(s+eps) are copied statement by statement from sinkhorn_one (same expressions → same nvcc fmul/fma contraction). Iteration
//   count (iters) and order (columns once first → (iters-1)×(rows, columns)) are unchanged. Possible differences: none.
//   (Shuffles only move values. For odd M, where half a warp is used, the spare lanes only read token M-1 and never write — so
//   the shuffle mask can be the full warp.)
//
// Last-block pattern (same shape as threadFenceReduction in the CUDA programming guide): thread 0 of each mixing block
//   (m, j ≤ mix_hc) writes its result, then __threadfence → atomicAdd(cnt). Only the block whose add reaches M·(mix_hc+1) reads
//   mixes·rsq from L2 (__ldcg — values written by other CTAs, no L1/nc path), solves sinkhorn as M tokens × 16 lanes and resets
//   cnt to 0 (for the next launch and graph replay). No spin wait → no residency guarantee or cooperative launch needed, graph
//   capture works (a single kernel). In the PDL version (gemv_decode.cu) this tail comes after the body's griddepcontrol.wait,
//   so it does not overlap the previous grid, and the next grid's wait waits for completion of this whole grid (tail included).
#pragma once
#include <cfloat>

namespace hive::k {
namespace hcdf {

constexpr int SK_LANES = 16;  // hc 4 → 4×4 elements

// Lane l = threadIdx.x % 16 holds element (l/4, l%4). mx and r are already in registers (loaded by the caller). wr = whether this lane writes the result.
// Shuffles use the full-warp mask — callers must enter per warp (all 32 lanes of the warp call this function).
template <int HC>
__device__ __forceinline__ void sinkhorn_par(const float* __restrict__ mx, float r, const float* __restrict__ scale, const float* __restrict__ base,
                                             int iters, float eps, float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb, bool wr) {
  static_assert(HC * HC == SK_LANES, "hc 4 only");
  constexpr int hc = HC;
  const int l = threadIdx.x & (SK_LANES - 1), j = l / hc, kk = l % hc;
  // sinkhorn_one: pre[j] = 1/(1+exp(-(mx[j]·r·scale0 + base[j]))) + eps · post[j] = 2/(1+exp(-(mx[j+hc]·r·scale1 + base[j+hc])))
  if (wr && l < hc) pre[l] = 1.f / (1.f + expf(-(__ldcg(mx + l) * r * scale[0] + base[l]))) + eps;
  if (wr && l >= hc && l < 2 * hc) post[l - hc] = 2.f / (1.f + expf(-(__ldcg(mx + l) * r * scale[1] + base[l])));
  // c[j*hc+kk] = mx[j*hc+kk+2hc]·r·scale2 + base[j*hc+kk+2hc]
  float c = __ldcg(mx + l + 2 * hc) * r * scale[2] + base[l + 2 * hc];
  // softmax(-1) + eps: row max (fmaxf in ascending kk) → exp → row sum (ascending kk) → c/s + eps
  {
    float mxv = -FLT_MAX;
#pragma unroll
    for (int q = 0; q < hc; ++q) mxv = fmaxf(mxv, __shfl_sync(0xffffffffu, c, j * hc + q, SK_LANES));
    c = expf(c - mxv);
    float s = 0.f;
#pragma unroll
    for (int q = 0; q < hc; ++q) s += __shfl_sync(0xffffffffu, c, j * hc + q, SK_LANES);
    c = c / s + eps;
  }
  // column normalization (sum in ascending j) once
  {
    float s = 0.f;
#pragma unroll
    for (int q = 0; q < hc; ++q) s += __shfl_sync(0xffffffffu, c, q * hc + kk, SK_LANES);
    c = c / (s + eps);
  }
  for (int it = 0; it < iters - 1; ++it) {
    float s = 0.f;
#pragma unroll
    for (int q = 0; q < hc; ++q) s += __shfl_sync(0xffffffffu, c, j * hc + q, SK_LANES);
    c = c / (s + eps);
    s = 0.f;
#pragma unroll
    for (int q = 0; q < hc; ++q) s += __shfl_sync(0xffffffffu, c, q * hc + kk, SK_LANES);
    c = c / (s + eps);
  }
  if (wr) comb[l] = c;
}

// M tokens in one block: thread t → token t/16 · element t%16. Entered per warp (ceil(M·16/32) warps). mixes and rsq are read with __ldcg.
template <int HC>
__device__ __forceinline__ void sinkhorn_block(const float* mixes, const float* rsq, const float* __restrict__ scale, const float* __restrict__ base, int M,
                                               int iters, float eps, float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb) {
  constexpr int mix_hc = (2 + HC) * HC;
  const int nthr = ((M * SK_LANES + 31) / 32) * 32;
  if ((int)threadIdx.x >= nthr) return;
  const int t = threadIdx.x / SK_LANES, m = t < M ? t : M - 1;
  sinkhorn_par<HC>(mixes + (size_t)m * mix_hc, __ldcg(rsq + m), scale, base, iters, eps, pre + m * HC, post + m * HC, comb + (size_t)m * HC * HC, t < M);
}

// Tail of a mixing block (called by every thread of the block · after the body's thread 0 has written mixes/rsq). nblk = M·(mix_hc+1).
template <int HC>
__device__ __forceinline__ void sk_tail(const float* mixes, const float* rsq, const float* __restrict__ scale, const float* __restrict__ base, int M, int iters,
                                        float eps, float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb, int* cnt, int nblk, int* last) {
  if (threadIdx.x == 0) {
    __threadfence();  // make this block's mixes/rsq writes visible before the counter
    *last = atomicAdd(cnt, 1) == nblk - 1;
  }
  __syncthreads();
  if (!*last) return;
  __threadfence();  // read only after seeing the other blocks' writes (each increments the counter after its fence)
  if (threadIdx.x == 0) *cnt = 0;  // for the next launch and graph replay (no other block in this grid increments it further)
  sinkhorn_block<HC>(mixes, rsq, scale, base, M, iters, eps, pre, post, comb);
}

}  // namespace hcdf
}  // namespace hive::k
