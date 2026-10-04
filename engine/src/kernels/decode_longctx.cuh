// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Long-context decode — device code (split top-k, new indexer score kernel). Contract and switches: header comment of hive/decode_longctx.h.
//   Included by decode_longctx.cu; the CPU test (engine/tests/test_decode_longctx_cpu.cpp) runs **the same source** on a SIMT emulation
//   header (hence no inline asm and no launch syntax (<<<>>>) in this file, and warp operations are only called where the whole warp has converged).
#pragma once
#include <cstdint>

#include "hive/attn_decode_fused.h"
#include "hive/kv_pack.h"

namespace hive::k::lctx {

constexpr int TPK2 = 1024;                     // top-k block threads (same as the baseline TPK — the prefix-sum layout must match for identical output positions)
constexpr int IDX2_T = 128, IHI2 = 32, IDI2 = 128;  // score block threads, indexer heads, dimension (baseline IDX_T, IHI, IDI)

// ---- Sort key (same formula as the baseline ord_key) ------------------------------------------------------------------------------
// TOPK2 f32 NaN rule: keys are read and compared **only as integer bits**, and the **actual (compiled) f32 +NaN behavior** of the
//   baseline kernel is reproduced explicitly with integers.
//   Measured: with f32 kind 3 (half -inf + ±NaN), k 2048, 2 chunks, only the four cases T_in 5120/10240/12800/16384 differed from the
//   baseline. Detailed rerun: the baseline does not emit +NaN (0x7FC00000) columns and pads the end with that many -1 instead (-1 slots
//   43/99/131/166 = the +NaN count of each row), whereas the new version emitted +NaN.
//   Cause (checked in the PTX/SASS of the baseline dec_topk_kernel<float>, nvcc 13.0 -O3 sm_120a): the source formula is a bit-sort key
//   (+NaN = highest), but in the two selection loops (counting nsel, emit) the compiler turns `u | 0x80000000` into
//   `FADD R, -|f|, -RZ`, so the key of +NaN becomes 0xFFC00000 → 0x7FFFFFFF (canonical NaN bits).
//   The histogram and equal-count loops are integer (LOP3) and count +NaN as highest → threshold (thr) and want put +NaN into selected
//   slots, while the selection loop does not emit +NaN when thr > 0x7FFFFFFF (it only occupies a slot → the end is -1). -NaN, ±inf and
//   denormals keep their bits through the FADD (sign 1 takes the integer `not` branch; FADD has no FTZ).
//   The bf16 baseline kernel is integer-only (no FADD in SASS), so it follows the specification. Before this fix the new version's
//   selection loop had the same FADD, but it also dropped +NaN at the chunk stage, leaving holes in the union (first FAIL: 2488–4589 bytes).
//   → Rule of this version (f32, final selection stage only; integer arithmetic, so independent of the compiler): pick the top keff per
//     the specification (bit-sort key, ties → smaller index), then drop -inf and **also +NaN when thr > 0x7FFFFFFF (threshold key at or
//     above +0.0)** — same output as the baseline. The chunk stage follows the specification (keeps union ⊇ global selection).
//   One case not reproduced: a row with +NaN while thr == 0x7FFFFFFF (threshold exactly at a group of -0.0 ties) — in the baseline, +NaN
//   shares the -0.0 tie slots depending on the per-thread range layout (C = ⌈T/1024⌉), so the result depends on T. Here +NaN is emitted
//   in that case (specification). +NaN together with a -0.0 threshold is assumed not to occur on the score path.
template <class KeyT> struct KeyRaw;
template <> struct KeyRaw<float> {
  using R = unsigned int;
  static constexpr R NEG_INF = 0xFF800000u;  // -inf bits
  // Actual rule of the baseline f32 kernel (see header): if the threshold key is at or above +0.0 (thr > 0x7FFFFFFF), +NaN (sign 0, exponent all ones, mantissa != 0) is not emitted
  __device__ __forceinline__ static bool old_drop(R u, uint32_t thr) { return thr > 0x7FFFFFFFu && u > 0x7F800000u && u < 0x80000000u; }
};
template <> struct KeyRaw<bf16> {
  using R = unsigned short;
  static constexpr R NEG_INF = 0xFF80u;
  __device__ __forceinline__ static bool old_drop(R, uint32_t) { return false; }  // bf16 baseline = specification (integer path only)
};
__device__ __forceinline__ uint32_t okey_raw(unsigned int u) { return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
__device__ __forceinline__ uint32_t okey_raw(unsigned short s) {
  const uint32_t u = s;
  return ((u & 0x8000u) ? ~u : (u | 0x8000u)) & 0xFFFFu;  // == okey_raw(bf16 bits << 16) >> 16
}
// Compatibility (host reference and other tests) — not used by the kernels
__device__ __forceinline__ uint32_t okey(float f) { return okey_raw((unsigned int)__float_as_uint(f)); }
__device__ __forceinline__ uint32_t okey(bf16 v) { return okey_raw((unsigned short)__bfloat16_as_ushort(v)); }
// CG = values written by other CTAs within this launch (the union) — read from L2, bypassing L1
template <bool CG, class R> __device__ __forceinline__ R ldr(const R* p) { return CG ? __ldcg(p) : *p; }

struct Topk2Smem {
  int hist[32][256];
  int wsum[32];
  uint32_t prefix;
  int need, cgt;
};

// Block exclusive prefix sum (1024 threads) — same statements as the baseline block_excl_scan
// Inclusive warp prefix sum: lane i ends with the sum of lanes 0..i (distances 1, 2, 4, 8, 16).
__device__ __forceinline__ int warp_incl_sum(int x, int lane) {
#pragma unroll
  for (int d = 1; d <= 16; d *= 2) {
    const int up = __shfl_up_sync(0xffffffffu, x, d);
    if (lane >= d) x += up;
  }
  return x;
}

__device__ __forceinline__ int excl_scan1024(int v, Topk2Smem& s, int* total) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int x = warp_incl_sum(v, lane);
  if (lane == 31) s.wsum[warp] = x;
  __syncthreads();
  if (warp == 0) s.wsum[lane] = warp_incl_sum(s.wsum[lane], lane);
  __syncthreads();
  const int before = (warp ? s.wsum[warp - 1] : 0) + x - v;
  *total = s.wsum[31];
  __syncthreads();
  return before;
}

// Radix select: over kr[0, n), the boundary thr (sort key) of the top keff by (key desc, index asc), and want = how many of the keys equal to thr to take.
//   Produces the same thr / need / cgt as the pass loop of the baseline dec_topk_kernel (bin sums are integers, so aggregation order does not matter; the bin search finds the unique bin satisfying the same condition).
template <class KeyT, int NBITS, bool CG>
__device__ __forceinline__ void radix_thr(const KeyT* kr, int n, int keff, Topk2Smem& s, uint32_t& thr, int& want) {
  using R = typename KeyRaw<KeyT>::R;
  const R* rr = reinterpret_cast<const R*>(kr);  // bits only (see header)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  __syncthreads();  // after the previous selection's smem reads are done
  if (tid == 0) { s.prefix = 0; s.need = keff - 1; s.cgt = 0; }
  __syncthreads();
  constexpr uint32_t FULL = NBITS == 32 ? 0xFFFFFFFFu : 0xFFFFu;
  for (int pass = 0; pass < NBITS / 8; ++pass) {
    const int shift = NBITS - 8 - pass * 8;
    const uint32_t mask = pass == 0 ? 0u : ((FULL << (NBITS - pass * 8)) & FULL);
    for (int i = tid; i < 32 * 256; i += TPK2) (&s.hist[0][0])[i] = 0;
    __syncthreads();
    const uint32_t pf = s.prefix;
    for (int base = 0; base < n; base += TPK2) {  // iteration count is uniform across the block (warp convergence)
      const int t = base + tid;
      uint32_t u = 0;
      bool act = false;
      if (t < n) { u = okey_raw(ldr<CG>(rr + t)); act = (u & mask) == pf; }
      const uint32_t bin = (u >> shift) & 0xFFu;
      // Warp aggregation: among lanes with the same bin the lowest lane adds the count once (inactive lanes use a private value 256+lane — never grouped with anyone)
      const unsigned peers = __match_any_sync(0xffffffffu, act ? bin : (0x100u + (unsigned)lane));
      if (act && lane == __ffs(peers) - 1) atomicAdd(&s.hist[warp][bin], __popc(peers));
    }
    __syncthreads();
    if (tid < 256) {
      int sum = 0;
      for (int w = 0; w < 32; ++w) sum += s.hist[w][tid];
      s.hist[0][tid] = sum;
    }
    __syncthreads();
    if (warp == 0) {
      // Baseline: thread 0 walks b = 255 → 0 accumulating acc (sum of bins above b) and takes the first b with acc + hist[b] > need. Here: each lane covers 8 bins (255−8·lane … 248−8·lane),
      //   a warp prefix sum gives the total of the bins before the lane (higher bins), and the same condition is solved. Adding (b == 255 || acc <= need) makes the satisfying bin unique
      //   (the [acc, acc+h) intervals do not overlap; at b = 255, need = −1 (keff 0) also yields 255 like the baseline). If none, bin 0 with acc = total, like the baseline.
      const int need = s.need;
      int h[8], ls = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) { h[i] = s.hist[0][255 - 8 * lane - i]; ls += h[i]; }
      const int incl = warp_incl_sum(ls, lane);
      const int total = __shfl_sync(0xffffffffu, incl, 31);
      int fb = -1, facc = 0, acc = incl - ls;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int b = 255 - 8 * lane - i;
        if (fb < 0 && acc + h[i] > need && (b == 255 || acc <= need)) { fb = b; facc = acc; }
        acc += h[i];
      }
      const unsigned fm = __ballot_sync(0xffffffffu, fb >= 0);
      int bin = 0, accv = total;
      if (fm) {
        const int src = __ffs(fm) - 1;
        bin = __shfl_sync(0xffffffffu, fb, src);
        accv = __shfl_sync(0xffffffffu, facc, src);
      }
      if (lane == 0) { s.need = need - accv; s.cgt += accv; s.prefix |= ((uint32_t)bin) << shift; }
    }
    __syncthreads();
  }
  thr = s.prefix;
  want = keff - s.cgt;
}

// emit(j, t, v) the selected elements in ascending index order — same layout as the 3 contiguous-range passes of the baseline dec_topk_kernel (thread = contiguous range, tie rank → selection rank).
//   drop = the baseline final rule (exclude -inf, plus KeyRaw::old_drop for f32 — see header). The chunk stage uses drop = false (exactly keff selected — the union slot-count contract).
template <class KeyT, bool CG, class Emit>
__device__ __forceinline__ int emit_sel(const KeyT* kr, int n, uint32_t thr, int want, bool drop, Topk2Smem& s, Emit emit) {
  using R = typename KeyRaw<KeyT>::R;
  constexpr R NEG_INF = KeyRaw<KeyT>::NEG_INF;
  const R* rr = reinterpret_cast<const R*>(kr);  // bits only — all three loops see the same integer keys as the histogram (-inf is a bit comparison too: f == -inf ⟺ bits == NEG_INF)
  const int tid = threadIdx.x;
  const int C = (n + TPK2 - 1) / TPK2;
  const int t0 = min(n, tid * C), t1 = min(n, t0 + C);
  int neq = 0;
  for (int t = t0; t < t1; ++t) neq += okey_raw(ldr<CG>(rr + t)) == thr;
  int tot;
  const int eq_base = excl_scan1024(neq, s, &tot);
  int nsel = 0;
  {
    int r = eq_base;
    for (int t = t0; t < t1; ++t) {
      const R v = ldr<CG>(rr + t);
      const uint32_t u = okey_raw(v);
      bool sel = u > thr;
      if (u == thr) { sel = r < want; ++r; }
      nsel += (sel && !(drop && (v == NEG_INF || KeyRaw<KeyT>::old_drop(v, thr))));
    }
  }
  int nall;
  int pos = excl_scan1024(nsel, s, &nall);
  {
    int r = eq_base;
    for (int t = t0; t < t1; ++t) {
      const R v = ldr<CG>(rr + t);
      const uint32_t u = okey_raw(v);
      bool sel = u > thr;
      if (u == thr) { sel = r < want; ++r; }
      if (sel && !(drop && (v == NEG_INF || KeyRaw<KeyT>::old_drop(v, thr)))) emit(pos++, t, v);  // v = original key bits (copied into the union unchanged — byte-identical)
    }
  }
  return nall;
}

// threadFenceReduction last block (same shape as the baseline last_of; resets the counter to 0)
__device__ __forceinline__ bool last_block(int* counter, int n, int* flag) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) *flag = (atomicAdd(counter, 1) == n - 1);
  __syncthreads();
  const bool last = *flag != 0;
  if (last) { __threadfence(); if (threadIdx.x == 0) *counter = 0; }
  return last;
}

// ---- (1) Split top-k. grid = (ncmax, M), block 1024. Argument contract = baseline dec_topk_kernel + (CH, union ukeys/uidx [M][ustride], counters cnt[M]).
template <class KeyT, int NBITS>
__global__ void __launch_bounds__(TPK2) topk2_kernel(const KeyT* __restrict__ keys, int T_in, bool use_trows, const DecRow* __restrict__ tab, int k,
                                                     int row_stride, int32_t* __restrict__ out, int out_stride, const int32_t* __restrict__ cand,
                                                     int cand_stride, int bs, int offset, bool map_offset, int CH, KeyT* ukeys, int32_t* uidx, int ustride,
                                                     int* cnt) {
  using R = typename KeyRaw<KeyT>::R;  // key elements as bits only (see header)
  __shared__ Topk2Smem s;
  __shared__ int flag;
  const int row = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
  const int T = use_trows ? min(tab[row].trows, T_in) : T_in;
  const int nc = T > CH ? (T + CH - 1) / CH : 1;
  if (c >= nc) return;  // this row has fewer chunks (per-row trows) — the whole block exits together
  const KeyT* kr = keys + (size_t)row * row_stride;
  int32_t* orow = out + (size_t)row * out_stride;
  auto final_emit = [&](int j, int32_t p) {  // same mapping as the baseline emit
    if (cand && p >= 0) { const int32_t b = cand[(size_t)row * cand_stride + p / bs]; p = b >= 0 ? b * bs + p % bs : -1; }
    if (map_offset) p = (p >= 0 && p < tab[row].visible) ? p + offset : -1;
    orow[j] = p;
  };
  uint32_t thr;
  int want;
  if (nc == 1) {  // single chunk: one selection, same as the baseline
    const int keff = min(k, T);
    radix_thr<KeyT, NBITS, false>(kr, T, keff, s, thr, want);
    const int nall = emit_sel<KeyT, false>(kr, T, thr, want, true, s, [&](int j, int t, R) { final_emit(j, t); });
    for (int j = nall + tid; j < k; j += TPK2) final_emit(j, -1);
    return;
  }
  // Chunk c: the top min(k, n) as (key, original column) into union slots [c·k, c·k + keff) — every earlier chunk has length CH >= 2k, so its slots are full (no holes)
  const int c0 = c * CH, n = min(T - c0, CH), keff = min(k, n);
  R* uk = reinterpret_cast<R*>(ukeys + (size_t)row * ustride + (size_t)c * k);  // copy key bits verbatim (never through a float register)
  int32_t* ui = uidx + (size_t)row * ustride + (size_t)c * k;
  radix_thr<KeyT, NBITS, false>(kr + c0, n, keff, s, thr, want);
  emit_sel<KeyT, false>(kr + c0, n, thr, want, false, s, [&](int j, int t, R v) { uk[j] = v; ui[j] = c0 + t; });
  if (!last_block(cnt + row, nc, &flag)) return;
  // Last chunk block of the row: final selection from the union (original column ascending) with the baseline rule
  const int U = (nc - 1) * k + min(k, T - (nc - 1) * CH);
  const KeyT* ur = ukeys + (size_t)row * ustride;
  const int32_t* uir = uidx + (size_t)row * ustride;
  radix_thr<KeyT, NBITS, true>(ur, U, min(k, U), s, thr, want);
  const int nall = emit_sel<KeyT, true>(ur, U, thr, want, true, s, [&](int j, int t, R) { final_emit(j, __ldcg(uir + t)); });
  for (int j = nall + tid; j < k; j += TPK2) final_emit(j, -1);
}

// ---- (2) New indexer score kernel (Hi 32, Di 128). grid = (⌈ncols / (128·KPT)⌉, M), block 128. Column ci = block base + j·128 + tid (j < KPT).
//   Per-key arithmetic uses the same statements as the baseline idx_score32 (per-head fmaf over ascending d, kv = e2m1 × scale, bf16 → relu → ×w → bf16 → sum over ascending h).
template <bool CAND, int KPT>
__global__ void __launch_bounds__(IDX2_T) idx_scores2_kernel(const bf16* __restrict__ q, const bf16* __restrict__ w, const DecRow* __restrict__ tab,
                                                             int ncols, const int32_t* __restrict__ cand, int cand_stride, int bs, bf16* __restrict__ score) {
  __shared__ __align__(16) float qT[IDI2 * IHI2];
  __shared__ float wsm[IHI2];
  const int m = blockIdx.y, tid = threadIdx.x;
  const int vis = min(tab[m].visible, tab[m].trows);
  const uint8_t* kbase = tab[m].kptr;
  const int cb = blockIdx.x * (IDX2_T * KPT);
  int tj[KPT];
  bool any = false;
#pragma unroll
  for (int j = 0; j < KPT; ++j) {
    const int ci = cb + j * IDX2_T + tid;
    int t = -1;
    if (ci < ncols) {
      t = ci;
      if (CAND) { const int32_t b = cand[(size_t)m * cand_stride + ci / bs]; t = b >= 0 ? b * bs + ci % bs : -1; }
    }
    tj[j] = (ci < ncols && t >= 0 && t < vis) ? t : -1;
    any = any || tj[j] >= 0;
  }
  bf16* srow = score + (size_t)m * ncols;
  if (!__syncthreads_or(any)) {  // the whole block is outside the visible columns — only -inf, same as the baseline
#pragma unroll
    for (int j = 0; j < KPT; ++j) { const int ci = cb + j * IDX2_T + tid; if (ci < ncols) srow[ci] = f2bf(-INFINITY); }
    return;
  }
  {  // transpose q (lane = head h, each warp handles 8 dimensions): qT[d][h] = bf2f(q[h][d]) — write bank = h (conflict-free)
    const int h = tid & 31;
    for (int d0 = (tid >> 5) * 8; d0 < IDI2; d0 += (IDX2_T / 32) * 8) {
      const uint4 raw = *reinterpret_cast<const uint4*>(q + (size_t)m * IHI2 * IDI2 + (size_t)h * IDI2 + d0);
      const bf16* e = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) qT[(d0 + jj) * IHI2 + h] = bf2f(e[jj]);
    }
  }
  if (tid < IHI2) wsm[tid] = bf2f(w[(size_t)m * IHI2 + tid]);
  __syncthreads();
  const uint8_t* kr[KPT];
#pragma unroll
  for (int j = 0; j < KPT; ++j) kr[j] = kbase + (size_t)(tj[j] >= 0 ? tj[j] : 0) * kvp::IDX_ROW;  // invalid columns read row 0 (any → vis >= 1) and discard it
  float acc[KPT][IHI2];
#pragma unroll
  for (int j = 0; j < KPT; ++j)
#pragma unroll
    for (int hh = 0; hh < IHI2; ++hh) acc[j][hh] = 0.f;
  for (int qd = 0; qd < IDI2 / 32; ++qd) {
    float sc[KPT];
    uint32_t wv[KPT][4];
#pragma unroll
    for (int j = 0; j < KPT; ++j) {
      sc[j] = e8m0_to_f32(kr[j][qd]);
      const uint4 u = reinterpret_cast<const uint4*>(kr[j] + kvp::IDX_HDR)[qd];
      wv[j][0] = u.x; wv[j][1] = u.y; wv[j][2] = u.z; wv[j][3] = u.w;
    }
#pragma unroll
    for (int w4 = 0; w4 < 4; ++w4) {
#pragma unroll
      for (int jb = 0; jb < 8; ++jb) {
        const int d = qd * 32 + w4 * 8 + jb;
        float kv[KPT];
#pragma unroll
        for (int j = 0; j < KPT; ++j) kv[j] = kvp::e2m1_val((wv[j][w4] >> (4 * jb)) & 0xFu) * sc[j];
        const float4* qd4 = reinterpret_cast<const float4*>(qT + d * IHI2);
#pragma unroll
        for (int jj = 0; jj < IHI2 / 4; ++jj) {
          const float4 qq = qd4[jj];
#pragma unroll
          for (int j = 0; j < KPT; ++j) {
            acc[j][4 * jj] = fmaf(qq.x, kv[j], acc[j][4 * jj]);
            acc[j][4 * jj + 1] = fmaf(qq.y, kv[j], acc[j][4 * jj + 1]);
            acc[j][4 * jj + 2] = fmaf(qq.z, kv[j], acc[j][4 * jj + 2]);
            acc[j][4 * jj + 3] = fmaf(qq.w, kv[j], acc[j][4 * jj + 3]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int j = 0; j < KPT; ++j) {
    const int ci = cb + j * IDX2_T + tid;
    if (ci >= ncols) continue;
    float total = 0.f;
#pragma unroll
    for (int hh = 0; hh < IHI2; ++hh) {
      float sb = bf2f(f2bf(acc[j][hh]));
      sb = fmaxf(sb, 0.f);
      sb = bf2f(f2bf(sb * wsm[hh]));
      total += sb;
    }
    srow[ci] = tj[j] >= 0 ? f2bf(total) : f2bf(-INFINITY);
  }
}

}  // namespace hive::k::lctx
