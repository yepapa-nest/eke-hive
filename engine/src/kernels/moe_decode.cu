// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Fused decode experts (HIVE_DECODE_FUSED) — all GPU experts of a layer (resident group or DMA group) in **one launch**:
//   w1‖w3 GEMV → swiglu·rw → 32-block quantization (yq/ys) → w2 GEMV → fixed-order accumulation into acc.
//   Unfused path (the fuse_ && use_mx path of runtime.cpp moe_decode_experts): mx_grouped_w13 (+quantization) → mx_grouped_w2 → accum_bf16_rows_seq, 3 launches,
//   grids = (I/64 = 36 blocks) × groups and (dim/64 = 80) × groups → for an M=1 layer (~5 groups) the w13 grid is smaller than the 188 SMs (~4 warps per SM), and weight
//   loads were 4 bytes per lane (8 rows × 16 B pieces) — measured (HIVE_PROFILE): moe.gpu_experts 9.58 ms/step vs a bandwidth floor of ~2.1 ms.
// What changes here:
//   · Work item = (group, column slice) — w13 = 32 columns (4 warps × 8 columns), w2 = 128 columns (4 warps × 4 tiles × 8 columns). 5 groups → 560 items → several CTAs per SM.
//   · Weight loads = 16 B per lane (ld.global.cs, read-once streaming) — lane (r,q) reads block kb0+q of row n0+r whole, and four lanes
//     transpose 4×4 with 4 shuffles to get the fragment mma needs (bytes [4q,4q+4) of block kb0+u). Scales for 4 blocks travel as one u32.
//   · Order: the work list (atomic tickets) is all w13 items → w2 items (group order). A w2 item waits until all w13 items of its group are done (grp_done).
//     Ticket order makes this deadlock-free: every w13 item a w2 item waits for has a smaller ticket = taken by an **already running** CTA (if a CTA's prefetched
//     next ticket is also w13, its current item is w13 too — it does not wait). Independent of grid size and residency.
//   · Accumulation: each finished 128-column w2 slice increments that slice's arrival count (col_done); the **last** arriving CTA (all groups) adds the slice into acc
//     in the same order as accum_bf16_rows_seq (per row m, fp32 additions in ascending r).
//   · One counter buffer per stream (launches on a stream are serialized) — the last CTA to finish resets it to 0 (no memset per launch).
// Values (bit-identical — checked by test_decode_moe): every 8-column output tile uses **the same mma instruction order** as the unfused path (blocks kb ascending, one accumulator),
//   so y·yq·ys·eout are equal; the swiglu and quantization formulas are taken verbatim from mx_grouped_w13_kernel (one 32-block = one CTA), and accumulation uses the same fp32 addition order as accum_bf16_rows_seq.
//   No split-K (it would change the accumulation order). Differences = none (by design). y is still written (scratch — later stages read only yq/ys).
// Scope: M ≤ 8 (decode batch, MTP verify rows) · dim, I multiples of 128 · expert pointers aligned to 16 B (weights) / 4 B (scales) — otherwise the caller uses the unfused path.
//
// HIVE_DECODE_FUSED2 (env_on, default off — when on, both FUSED variants (host table, DEV step graph) use the FUSED2 body below):
//   nsys (decode, 128 steps): moe_decode_fused ≈ 46 launches/step × ~125 µs = 5.8 ms/step, effective ~750 GB/s (HBM ~1.8 TB/s).
//   test_decode_moe (L2 flushed): M=1 445 · M=2 647 · M=4 826 · M=6 871 · M=8 872 GB/s. Structural causes:
//   ① too few bytes in flight — a warp issues only 16 B × (NM·T) per lane per iteration (weights go to registers and straight into mma) and nothing new until the next iteration.
//      CTAs hold 1–2 items each, 8–12 warps per SM → a few KB per SM — far below latency (~1 µs under load) × 1.8 TB/s ≈ 2 MB (≥ ~10 KB per SM).
//   ② the w13 → w2 boundary: a w2 CTA reads nothing while it waits for grp_done (its slot idles meanwhile).
//   FUSED2 changes (values bit-identical to FUSED — see below):
//   · weights and weight scales are prefetched with cp.async (16 B .cg / 4 B .ca, **no hints**) into a **per-warp** smem ring (S stages).
//     Cache-hinted forms (.L2::cache_hint with createpolicy evict_first, plus .L2::256B) died with "illegal instruction" on sm_120a (RTX PRO 6000, driver 580;
//     test_decode_moe and test_decode_step_graph). SASS from the same ptxas (13.0) shows that only the cache_hint form makes LDGSTS use UR0/UR1, which were never
//     written, as the smem base and memory descriptor (`LDGSTS.E.BYPASS.128 [R7+UR0], desc[UR1]` — the descriptor is loaded from c[0x358] into UR4/UR6 but the
//     instruction reads UR1) → prime suspect. Plain cp.async.cg 16 B / .ca 4 B produce the same `desc[UR6]` form as FUSED and other kernels, so all hints were
//     dropped (no evict-first — weights enter L2 at normal priority; no effect on values).
//     A warp writes and reads only its own region, so there is no CTA synchronization (cp.async.wait_group + __syncwarp). One stage = 4 blocks (64 B per row).
//   · the 4×4 transpose is gone (xpose4: 4 shuffles + sel4 with a per-lane index, which compiles into BSSY/BRA branch chains in SASS and diverges the warp —
//     cuobjdump of FUSED: 140 BSSY · 48 SHFL per kernel) — pieces are stored in swizzled slots (4r + (q ^ (r>>1))), and a lane reading one word each (4 × LDS.32, zero bank conflicts)
//     gets the mma fragment directly (the same words xpose4 produced).
//     In flight = (S−1) stages × 16 B·(NM·T) per lane × 128 — with S2=4, w2 24 KB and w13 (twice the stages) 28 KB per CTA, 2 CTAs per SM (40 KB smem per CTA).
//   · a w2 item issues its first S−1 weight stages **before** waiting for grp_done (weights do not depend on the w13 results) — HBM reads continue across the boundary.
//   · activations (A: xq or yq, small and L1/L2 resident) are loaded into registers one stage ahead (so A latency does not show behind the weight wait).
//   · grid = min(items, resident CTAs (occupancy with dynamic smem)) — same ticket order as FUSED (all w13 → w2, group order), so no deadlock (same proof as above).
//   Values: per 8-column tile the mma order (stages = kb0 ascending, u ascending within, one accumulator), the per-lane mma operand words (a0/a2/b0/b1/sfa/sfb), swiglu/quantization
//     and acc accumulation order are the same as FUSED — only the path weight bytes take into registers changes (global → smem → registers). Differences = none (by design).
//     Checked: a CPU emulation (ring, wait counts, swizzle → 1.6 M operand tuples == the FUSED formula, zero WAR/RAW, zero bank conflicts, 3 mutants FAIL); on the GPU test_decode_moe
//     compares FUSED, FUSED2 (S2 3/4/6) and every DEV variant bit for bit against the unfused chain. The FUSED kernel is untouched (FUSED2 is a separate kernel function, moe_decode_fused2_kernel).
//   Stage count = HIVE_DECODE_FUSED2_STAGES (w2 stages S2 ∈ {3, 4, 6}, default 4; w13 uses 2·S2 — same smem): empty/0/unparsable → 4, ≤3 → 3, 4·5 → 4, ≥6 → 6 (normalized, never rejected).
//     smem/CTA = S2 × 10 KB + 528 B static (S2=6 is 60 KB → raises the dynamic-smem limit attribute, 1 CTA per SM). test_decode_moe measures all three values on the GPU.
//
// HIVE_DECODE_FUSED3 (env_on, default off — takes precedence over FUSED2: both FUSED variants (host table, DEV step graph) use the FUSED3 body below):
//   FUSED2 measured (test_decode_moe, L2 flushed): FUSED2 S=4 M=4 1021 · M=6 1049 · M=8 1120 · M=8/pool 12 859 GB/s (FUSED 826/876/871/715) — 57–62 % of HBM ~1.8 TB/s.
//   Structural analysis (real shapes dim 5120 · I 2304 · expert 18.87 MB = w1·w3·w2 5.90 MB each + 0.37 MB scales; FUSED2 S=4 = 41.5 KB smem/CTA → 2 CTAs per SM, grid ≤ 376 CTAs):
//   ① work granularity = one CTA (4 warps) = 32 w13 columns (164 KB) / 128 w2 columns (147 KB) → 112 items per group (72 + 40). Each layer has two launches, resident share + DMA share:
//      M=1: 6 groups → 5 resident (560 items = 1.49 waves) + 1 DMA (112 items — 30 % of 376 CTAs; its w2 phase is 40 CTAs = only 40 SMs reading),
//      M=2: ~12 → 9 (1008 = 2.7 waves) + 3 (336) · M=4: 21 → 16 (1792 = 4.8 waves) + 5 (560) · M=8: 36 → 27 (3024 = 8.0 waves) + 9 (1008 = 2.7 waves).
//      In small launches (especially the DMA share) and in each launch's w2 tail fewer CTAs read than there are SMs — the CTA count caps the bytes in flight (FUSED2: 28 KB w13 / 24 KB w2 per CTA).
//   ② load shape: one warp instruction = 8 rows × 64 B (row stride 2560/1152 B) — half an L2 line (128 B) at a time; the next 64 B of the same row arrive with the next stage (µs later).
//   FUSED3 changes (values bit-identical to FUSED — see below):
//   · work item = one **warp** (no CTA synchronization): w13 = (group, 8-column tile) — 8 rows each of w1 and w3 = 40 KB; w2 = (group, 16 columns = 2 tiles) — 18 KB. 608 items per group (288 + 320) →
//     5.4× as many concurrent read streams per launch (112 CTAs → 608 warps; w2 phase of a 1-group DMA share = 320 warps vs FUSED2's 40 CTAs × 4 warps = 160).
//   · stage = per piece (8-row tile) 8 rows × **8 blocks (128 B contiguous per row = one L2 line)** — one warp instruction = 4 rows × 128 B. 2 pieces/stage = 2 KB per warp per stage.
//     smem slot = row·8 + (block ^ row) swizzle → zero read bank conflicts (checked by CPU emulation). Scales = per piece 8 rows × 8 B (lanes 0..15, 4 B each).
//     In flight = (S−1) × 2 KB per warp; smem = S × 8.5 KB per CTA (4 warps): S=3 25.5 KB (3 CTAs/SM) · S=4 34 KB (2 CTAs) · S=5 42.5 KB (2 CTAs) → 48–64 KB per SM.
//   · a 32-column quantization block = four 8-column tiles — the fourth warp to arrive (qdone atomic counter) re-reads y from L2 (bf16, written by the four warps = the same values as FUSED's ysm),
//     writes yq/ys with the FUSED formula and increments grp_done (I/32 times per group — the same target as FUSED's number of w13 CTAs). As in FUSED2, w2 issues its first S−1 weight
//     stages before waiting for grp_done. The last warp to arrive (all groups) at a 16-column slice accumulates that slice into acc in the FUSED order.
//   · tickets per warp (all w13 → w2, group order) — deadlock-free (same proof as FUSED: only w2 waits, and only on smaller tickets = running warps). qdone and col_done
//     are reset to 0 by their last arrival; ticket, grp_done and the exit count by the last warp to exit (no memset per launch; FUSED3 has its own counter buffer — different size).
//   · grid = min(items, resident CTAs) — surplus warps exit immediately.
//   Values: per 8-column tile the mma order (blocks kb ascending — stage st holds blocks 8st..8st+7, one accumulator), the per-lane operand words (a0/a2/b0/b1/sfa/sfb), the swiglu and quantization formulas
//     and the acc accumulation order (per column, fp32 additions in ascending row r) are the same as FUSED. Only the path weight bytes take into registers and the work split change.
//     **No fp32 accumulation-order difference** (no split-K — one warp accumulates a tile's whole K in one accumulator). Bit identity of y·yq·ys·eout·acc is the design goal, and
//     test_decode_moe compares bytes against the unfused chain, FUSED and FUSED2 S=4. CPU emulation: operand comparison · cp.async group counts (RAW) and overwrites (WAR) · zero bank conflicts ·
//     3 mutants detected (no swizzle → bank conflicts, one wait fewer → RAW, issuing one stage early → value mismatch) · ticket/counter protocol (6 random interleavings — zero deadlocks,
//     quantization and accumulation exactly once, all counters 0 afterwards).
//   cp.async forms = the same plain .cg 16 B / .ca 4 B as FUSED2 (no hints) — on sm_120a SASS only the two forms `LDGSTS.E.BYPASS.128 [R], desc[UR][R.64]` and `LDGSTS.E`, as in FUSED2.
//   Conditions (on top of FUSED2's): dim and I multiples of 256 · groups ≤ 128 · I/32 ≤ 128 · dim/16 ≤ 1024 — otherwise the variant that would be chosen without FUSED3 (env FUSED2 or FUSED) is used (absorbed, not rejected).
//   Stage count = HIVE_DECODE_FUSED3_STAGES (S ∈ {3, 4, 5}, default 4): empty/0/negative/unparsable → 4, ≤3 → 3, 4 → 4, ≥5 → 5 (normalized, never rejected).
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "hive/kernels.h"

namespace hive::k {

namespace {

constexpr int DF_WARPS = 4;         // CTA = 4 warps (128 threads)
constexpr int DF_W13_COLS = 32;     // w13 item = 32 columns (one 8-column tile per warp × w1·w3) = one quantization block
constexpr int DF_W2_T = 4;          // w2: 4 tiles per warp
constexpr int DF_W2_COLS = DF_WARPS * DF_W2_T * 8;  // 128
constexpr int DF_MAX_M = 8;
constexpr int DF_MAX_GROUPS = 1024;
constexpr int DF_MAX_CHUNKS = 1024;
// Counter layout: [0] ticket · [1] exits · [2, 2+MAX_GROUPS) grp_done · then [MAX_CHUNKS) col_done
constexpr int DF_CNT_INTS = 2 + DF_MAX_GROUPS + DF_MAX_CHUNKS;

__device__ __forceinline__ void mma_e4m3_e2m1(float (&d)[4], uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  // same instruction as gemm_mx.cu mma_mx_e4m3_e2m1 (rows 8..15 = a1/a3 = 0)
  const uint16_t zero = 0;
  const uint32_t z = 0u;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(z), "r"(a2), "r"(z), "r"(b0), "r"(b1), "r"(sfa), "h"(zero), "h"(zero), "r"(sfb), "h"(zero), "h"(zero));
}

__device__ __forceinline__ uint32_t e2m1x4_bytes(uint32_t P) {  // same formula as gemm_mx.cu e2m1x4_to_bytes
  return ((P & 0x000Fu) << 2) | ((P & 0x00F0u) << 6) | ((P & 0x0F00u) << 10) | ((P & 0xF000u) << 14);
}

__device__ __forceinline__ uint32_t sel4(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, int i) {
  return i == 0 ? a0 : i == 1 ? a1 : i == 2 ? a2 : a3;
}

// 4-lane transpose (q = 0..3, same r): lane q holds the 16 B (words 0..3) of block kb0+q → o[u] = word q of block kb0+u.
//   Round s: lane q sends its word (q+s)&3 and receives from lane (q-s)&3 → the received value = word q of block kb0+((q-s)&3).
__device__ __forceinline__ void xpose4(const uint4 v, int lane, uint32_t (&o)[4]) {
  const int q = lane & 3;
  uint32_t g[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const uint32_t send = sel4(v.x, v.y, v.z, v.w, (q + s) & 3);
    g[s] = __shfl_sync(0xffffffffu, send, (lane & ~3) | ((q - s) & 3));
  }
#pragma unroll
  for (int u = 0; u < 4; ++u) o[u] = sel4(g[0], g[1], g[2], g[3], (q - u) & 3);
}

template <bool CG> __device__ __forceinline__ uint2 ld_a8(const uint8_t* p) {
  if constexpr (CG) return __ldcg(reinterpret_cast<const uint2*>(p));
  else return __ldg(reinterpret_cast<const uint2*>(p));
}
template <bool CG> __device__ __forceinline__ uint32_t ld_s4(const uint8_t* p) {
  if constexpr (CG) return __ldcg(reinterpret_cast<const unsigned int*>(p));
  else return __ldg(reinterpret_cast<const unsigned int*>(p));
}

// One warp: ≤ 8 rows × T 8-column tiles × NM matrices, the whole K (blocks ascending, one accumulator — same mma order as mx_warp_tiles).
// A_CG: load A·sa bypassing L1 (ld.global.cg) — for reading yq/ys written by other CTAs of the same launch (w2).
template <int NM, int T, bool A_CG>
__device__ __forceinline__ void df_warp_tiles(const uint8_t* A, const uint8_t* sa, const int32_t* __restrict__ rows, int row0, int nrows, int K,
                                              const uint8_t* const (&W)[NM], const uint8_t* const (&SW)[NM], int n0, float (&acc)[NM][T][4]) {
  const int lane = threadIdx.x & 31, q = lane & 3, r = lane >> 2;
  const int nb = K / 32;
#pragma unroll
  for (int j = 0; j < NM; ++j)
#pragma unroll
    for (int i = 0; i < T; ++i) acc[j][i][0] = acc[j][i][1] = acc[j][i][2] = acc[j][i][3] = 0.f;
  const bool have_a = r < nrows;
  const int sfa_row = r + 8 * (lane & 1);  // same scale-provider row selection as mx_warp_tiles
  const int sfa_src = sfa_row < nrows ? sfa_row : 0;
  const uint8_t* sa_row = sa + (size_t)(rows ? rows[row0 + sfa_src] : row0 + sfa_src) * nb;
  const uint8_t* a_row = A + (size_t)(rows ? rows[row0 + (have_a ? r : 0)] : row0 + r) * K;
  const uint8_t* wrow[NM][T];
  const uint8_t* srow[NM][T];
#pragma unroll
  for (int j = 0; j < NM; ++j)
#pragma unroll
    for (int i = 0; i < T; ++i) {
      const size_t n = (size_t)(n0 + i * 8 + r);
      wrow[j][i] = W[j] + n * (K / 2) + 16 * q;
      srow[j][i] = SW[j] + n * nb;
    }
#pragma unroll 2
  for (int kb0 = 0; kb0 < nb; kb0 += 4) {
    uint4 braw[NM][T];
    uint32_t sb[NM][T];
#pragma unroll
    for (int j = 0; j < NM; ++j)
#pragma unroll
      for (int i = 0; i < T; ++i) {
        braw[j][i] = __ldcs(reinterpret_cast<const uint4*>(wrow[j][i] + (size_t)kb0 * 16));
        sb[j][i] = __ldcs(reinterpret_cast<const unsigned int*>(srow[j][i] + kb0));
      }
    uint32_t a0[4], a2[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (have_a) { const uint2 av = ld_a8<A_CG>(a_row + (size_t)(kb0 + u) * 32 + 8 * q); a0[u] = av.x; a2[u] = av.y; }
      else { a0[u] = 0u; a2[u] = 0u; }
    }
    const uint32_t sfa4 = ld_s4<A_CG>(sa_row + kb0);
#pragma unroll
    for (int j = 0; j < NM; ++j)
#pragma unroll
      for (int i = 0; i < T; ++i) {
        uint32_t wv[4];
        xpose4(braw[j][i], lane, wv);
#pragma unroll
        for (int u = 0; u < 4; ++u)
          mma_e4m3_e2m1(acc[j][i], a0[u], a2[u], e2m1x4_bytes(wv[u] & 0xFFFFu), e2m1x4_bytes(wv[u] >> 16), (sfa4 >> (8 * u)) & 0xFFu,
                        (sb[j][i] >> (8 * u)) & 0xFFu);
      }
  }
}

// ---- FUSED2: per-warp cp.async ring
constexpr int DF_THREADS = DF_WARPS * 32;
constexpr int DF2_P = 4;  // pieces per stage (w2: NM·T = 1·4; w13: 2·1 with twice the stages — same bytes)
constexpr int df2_smem_bytes(int S2) { return S2 * DF2_P * DF_THREADS * (16 + 4); }

__device__ __forceinline__ uint32_t df2_su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
// The most conservative form (no hints, no prefetch — see the FUSED2 notes in the file header). Same family as gemm_mx.cu tg_cp_async16.
__device__ __forceinline__ void df2_cp16(void* s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(df2_su32(s)), "l"(g) : "memory");
}
__device__ __forceinline__ void df2_cp4(void* s, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(df2_su32(s)), "l"(g) : "memory");
}
__device__ __forceinline__ void df2_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void df2_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// The df_warp_tiles computation through an S-stage cp.async ring. Stage st = blocks [4·st, 4·st+4) — lane (r,q) receives block 4·st+q (16 B) of row n0+i·8+r and that row's 4 scale bytes
//   (the same bytes as braw·sb in df_warp_tiles). Weight slot = wb[(slot·P + p)·128 + warp·32 + swizzled position] · scale slot = sb[(slot·P + p)·128 + tid] (per lane).
template <int NM, int T, bool A_CG, int S>
struct Df2Tile {
  static constexpr int P = NM * T;
  const uint8_t* w[NM][T];
  const uint8_t* s[NM][T];
  int nst;
  __device__ __forceinline__ void init(const uint8_t* const (&W)[NM], const uint8_t* const (&SW)[NM], int n0, int K) {
    const int lane = threadIdx.x & 31, q = lane & 3, r = lane >> 2;
    const int nb = K / 32;
    nst = K / 128;
#pragma unroll
    for (int j = 0; j < NM; ++j)
#pragma unroll
      for (int i = 0; i < T; ++i) {
        const size_t n = (size_t)(n0 + i * 8 + r);
        w[j][i] = W[j] + n * (K / 2) + 16 * q;
        s[j][i] = SW[j] + n * nb;
      }
  }
  // Slot of a 16 B weight piece = 4r + (q ^ (r>>1)) within the warp region (32 slots) — a swizzle that makes the reader (lane (r,q) reads word q of slot 4r + (u ^ (r>>1)) for u = 0..3)
  //   hit each of the 32 banks exactly once per u (row even/odd × 4 values of (u ^ r>>1) × 4 values of q = 32 banks). The 4 B scale goes to a per-lane slot (tid).
  __device__ __forceinline__ void issue(int st, uint4* wb, uint32_t* sb) const {
    if (st < nst) {
      const int slot = st % S, tid = threadIdx.x, lane = tid & 31, q = lane & 3, r = lane >> 2;
      const int wpos = (tid & ~31) + 4 * r + (q ^ (r >> 1));
#pragma unroll
      for (int j = 0; j < NM; ++j)
#pragma unroll
        for (int i = 0; i < T; ++i) {
          const int base = (slot * P + j * T + i) * DF_THREADS;
          df2_cp16(wb + base + wpos, w[j][i] + (size_t)st * 64);
          df2_cp4(sb + base + tid, s[j][i] + (size_t)st * 4);
        }
    }
    df2_commit();  // commit empty groups too (keeps the wait_group count aligned with the stage number)
  }
  __device__ __forceinline__ void prologue(uint4* wb, uint32_t* sb) const {
#pragma unroll
    for (int st = 0; st < S - 1; ++st) issue(st, wb, sb);
  }
  // Called after the prologue. Accumulation = same mma order as df_warp_tiles.
  __device__ __forceinline__ void run(const uint8_t* A, const uint8_t* sa, const int32_t* __restrict__ rows, int row0, int nrows, int K, float (&acc)[NM][T][4],
                                      uint4* wb, uint32_t* sb) const {
    const int tid = threadIdx.x, lane = tid & 31, q = lane & 3, r = lane >> 2;
    const int nb = K / 32;
#pragma unroll
    for (int j = 0; j < NM; ++j)
#pragma unroll
      for (int i = 0; i < T; ++i) acc[j][i][0] = acc[j][i][1] = acc[j][i][2] = acc[j][i][3] = 0.f;
    const bool have_a = r < nrows;
    const int sfa_row = r + 8 * (lane & 1);  // same scale-provider row as df_warp_tiles
    const int sfa_src = sfa_row < nrows ? sfa_row : 0;
    const uint8_t* sa_row = sa + (size_t)(rows ? rows[row0 + sfa_src] : row0 + sfa_src) * nb;
    const uint8_t* a_row = A + (size_t)(rows ? rows[row0 + (have_a ? r : 0)] : row0 + r) * K;
    auto load_a = [&](int st, uint32_t (&a0)[4], uint32_t (&a2)[4], uint32_t& sfa4) {
      const int kb0 = st * 4;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        if (have_a) { const uint2 av = ld_a8<A_CG>(a_row + (size_t)(kb0 + u) * 32 + 8 * q); a0[u] = av.x; a2[u] = av.y; }
        else { a0[u] = 0u; a2[u] = 0u; }
      }
      sfa4 = ld_s4<A_CG>(sa_row + kb0);
    };
    uint32_t a0[4], a2[4], sfa4;
    load_a(0, a0, a2, sfa4);
    const uint32_t* const wb32 = reinterpret_cast<const uint32_t*>(wb);
    const int wrd = (tid & ~31) + 4 * r;  // the four slots of row r this lane reads (start before swizzling)
    for (int st = 0; st < nst; ++st) {
      df2_wait<S - 2>();  // this lane's share of stage st has arrived
      __syncwarp();       // make pieces received by other lanes of the warp visible (+ the code below overwrites slot (st−1)%S only after all lanes finished reading it in the previous iteration)
      issue(st + S - 1, wb, sb);
      uint32_t n0v[4], n2v[4], nsf;
      load_a(st + 1 < nst ? st + 1 : st, n0v, n2v, nsf);  // activations for the next stage, one stage ahead
      const int slot = st % S;
#pragma unroll
      for (int j = 0; j < NM; ++j)
#pragma unroll
        for (int i = 0; i < T; ++i) {
          const int base = (slot * P + j * T + i) * DF_THREADS;
          // wv[u] = word q (row r) of block 4·st+u — the same value as the xpose4 result in df_warp_tiles (transposed via smem addresses instead of shuffles)
          uint32_t wv[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) wv[u] = wb32[(size_t)(base + wrd + (u ^ (r >> 1))) * 4 + q];
          const uint32_t sbw = sb[base + tid];
#pragma unroll
          for (int u = 0; u < 4; ++u)
            mma_e4m3_e2m1(acc[j][i], a0[u], a2[u], e2m1x4_bytes(wv[u] & 0xFFFFu), e2m1x4_bytes(wv[u] >> 16), (sfa4 >> (8 * u)) & 0xFFu,
                          (sbw >> (8 * u)) & 0xFFu);
        }
#pragma unroll
      for (int u = 0; u < 4; ++u) { a0[u] = n0v[u]; a2[u] = n2v[u]; }
      sfa4 = nsf;
    }
    df2_wait<0>();
  }
};

struct DfArgs {
  const GroupDesc* g;
  int ng;
  const uint8_t *xq, *xs;
  const int32_t* rows;
  const float* rw;
  int r0, nrows, M, dim, I;
  float limit;
  bf16* y;
  uint8_t *yq, *ys;
  bf16* eout;
  float* acc;
  int* cnt;
  DfDevCount dc;  // read by the DEV variant only (group count and row range from device-side counts)
};

// HIVE_DECODE_STEP_GRAPH: the DEV = true variant reads ng·r0·nrows from device-side counts written by the preceding kernel (hs_plan) (the step graph fixes the grid).
//   DEV = false is the host-table kernel (same body — uses ng/r0/nrows from a as given).
template <bool DEV>
__global__ void __launch_bounds__(DF_WARPS * 32) moe_decode_fused_kernel(const DfArgs a_in) {
  __shared__ int s_task, s_last;
  __shared__ bf16 ysm[8][DF_W13_COLS];
  DfArgs a = a_in;
  if constexpr (DEV) {
    const int* b = a.dc.base;
    a.ng = b[a.dc.ng];
    a.r0 = a.dc.r0 >= 0 ? b[a.dc.r0] : 0;
    a.nrows = b[a.dc.nrows];
  }
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, q = lane & 3, r = lane >> 2;
  const int c13 = a.I / DF_W13_COLS, c2 = a.dim / DF_W2_COLS;
  const int t13 = a.ng * c13, total = t13 + a.ng * c2;
  int* const ticket = a.cnt;
  int* const exitc = a.cnt + 1;
  int* const grp_done = a.cnt + 2;
  int* const col_done = a.cnt + 2 + DF_MAX_GROUPS;
  if (tid == 0) s_task = atomicAdd(ticket, 1);
  __syncthreads();
  int t = s_task;
  while (t < total) {
    int nxt = 0;
    if (tid == 0) nxt = atomicAdd(ticket, 1);  // fetch the next ticket early (hides the atomic latency behind the work)
    if (t < t13) {
      // ---- w13 + swiglu + 32-block quantization (formula of mx_grouped_w13_kernel)
      const int gi = t / c13, cc = t % c13;
      const GroupDesc d = a.g[gi];
      const int nc0 = cc * DF_W13_COLS, n0 = nc0 + warp * 8;
      const uint8_t* const W[2] = {d.w1, d.w3};
      const uint8_t* const SW[2] = {d.s1, d.s3};
      float acc[2][1][4];
      df_warp_tiles<2, 1, false>(a.xq, a.xs, a.rows, d.row0, d.n, a.dim, W, SW, n0, acc);
      if (r < d.n) {
        const float rwv = a.rw[d.row0 + r];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const int n = n0 + 2 * q + c;
          float gv = bf2f(f2bf(acc[0][0][c])), uv = bf2f(f2bf(acc[1][0][c]));
          if (a.limit > 0.f) { uv = fminf(fmaxf(uv, -a.limit), a.limit); gv = fminf(gv, a.limit); }
          float v = gv / (1.f + expf(-gv)) * uv * rwv;
          const bf16 vb = f2bf(v);
          a.y[(size_t)(d.row0 + r) * a.I + n] = vb;
          ysm[r][n - nc0] = vb;
        }
      }
      __syncthreads();
      if (tid < d.n) {  // one row = one 32-block
        float v[32];
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) { v[i] = bf2f(ysm[tid][i]); amax = fmaxf(amax, fabsf(v[i])); }
        amax = fmaxf(amax, 1e-4f);
        const uint8_t code = f32_ceil_pow2_e8m0(amax * (1.0f / 448.0f));
        const float sc = e8m0_to_f32(code);
        uint8_t* qp = a.yq + (size_t)(d.row0 + tid) * a.I + nc0;
#pragma unroll
        for (int i = 0; i < 32; ++i) qp[i] = f32_to_e4m3(fminf(fmaxf(v[i] / sc, -448.f), 448.f));
        a.ys[(size_t)(d.row0 + tid) * (a.I / 32) + nc0 / 32] = code;
        __threadfence();
      }
      __syncthreads();
      if (tid == 0) { __threadfence(); atomicAdd(grp_done + gi, 1); }
    } else {
      // ---- w2 (after waiting for all w13 items of the group) → eout · the last arrival at a 128-column slice accumulates into acc
      const int u2 = t - t13, gi = u2 / c2, cc = u2 % c2;
      const GroupDesc d = a.g[gi];
      if (tid == 0) {
        while (*reinterpret_cast<volatile int*>(grp_done + gi) < c13) __nanosleep(64);
        __threadfence();
      }
      __syncthreads();
      const int nc0 = cc * DF_W2_COLS, n0 = nc0 + warp * (DF_W2_T * 8);
      const uint8_t* const W[1] = {d.w2};
      const uint8_t* const SW[1] = {d.s2};
      float acc[1][DF_W2_T][4];
      df_warp_tiles<1, DF_W2_T, true>(a.yq, a.ys, nullptr, d.row0, d.n, a.I, W, SW, n0, acc);
      if (r < d.n) {
#pragma unroll
        for (int i = 0; i < DF_W2_T; ++i)
#pragma unroll
          for (int c = 0; c < 2; ++c) a.eout[(size_t)(d.row0 + r) * a.dim + n0 + i * 8 + 2 * q + c] = f2bf(acc[0][i][c]);
      }
      __threadfence();
      __syncthreads();
      if (tid == 0) s_last = atomicAdd(col_done + cc, 1) == a.ng - 1;
      __syncthreads();
      if (s_last) {
        __threadfence();
        // same order as accum_bf16_rows_seq: add the bf16 values with row_map[r] == m to acc[m, n] in fp32, r ascending
        const int n = nc0 + tid;  // blockDim = 128 = DF_W2_COLS
        float s[DF_MAX_M];
#pragma unroll
        for (int m = 0; m < DF_MAX_M; ++m) s[m] = m < a.M ? a.acc[(size_t)m * a.dim + n] : 0.f;
        const int r1 = a.r0 + a.nrows;
        for (int rb = a.r0; rb < r1; rb += 8) {
          int mr[8];
          float v[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int rr = rb + j;
            mr[j] = rr < r1 ? __ldg(a.rows + rr) : -1;
            v[j] = rr < r1 ? __bfloat162float(__ushort_as_bfloat16(__ldcg(reinterpret_cast<const unsigned short*>(a.eout + (size_t)rr * a.dim + n)))) : 0.f;
          }
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int m = 0; m < DF_MAX_M; ++m)
              if (mr[j] == m) s[m] += v[j];
        }
#pragma unroll
        for (int m = 0; m < DF_MAX_M; ++m) if (m < a.M) a.acc[(size_t)m * a.dim + n] = s[m];
      }
    }
    __syncthreads();  // before s_task·ysm·s_last are reused
    if (tid == 0) s_task = nxt;
    __syncthreads();
    t = s_task;
  }
  // the last CTA to exit resets the counters to 0 (the next launch starts after this one ends, in stream order)
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(exitc, 1) == (int)gridDim.x - 1) {
      *ticket = 0;
      for (int i = 0; i < a.ng; ++i) grp_done[i] = 0;
      for (int i = 0; i < c2; ++i) col_done[i] = 0;
      *exitc = 0;
      __threadfence();
    }
  }
}

// FUSED2 (HIVE_DECODE_FUSED2): a copy of moe_decode_fused_kernel<DEV> — identical elsewhere; only the w13/w2 tile products (Df2Tile) and
//   the early weight issue before the w2 wait differ (the FUSED kernel is left untouched so the default path cannot regress). Dynamic smem = df2_smem_bytes(S2).
template <bool DEV, int S2>
__global__ void __launch_bounds__(DF_WARPS * 32) moe_decode_fused2_kernel(const DfArgs a_in) {
  static_assert(S2 >= 2, "F2 ring stage count");
  __shared__ int s_task, s_last;
  __shared__ bf16 ysm[8][DF_W13_COLS];
  extern __shared__ uint4 df2_sm[];  // [S2·4·128] uint4 weight slots, then [S2·4·128] u32 scale slots
  uint4* const wb = df2_sm;
  uint32_t* const sbm = reinterpret_cast<uint32_t*>(df2_sm + S2 * DF2_P * DF_THREADS);
  DfArgs a = a_in;
  if constexpr (DEV) {
    const int* b = a.dc.base;
    a.ng = b[a.dc.ng];
    a.r0 = a.dc.r0 >= 0 ? b[a.dc.r0] : 0;
    a.nrows = b[a.dc.nrows];
  }
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, q = lane & 3, r = lane >> 2;
  const int c13 = a.I / DF_W13_COLS, c2 = a.dim / DF_W2_COLS;
  const int t13 = a.ng * c13, total = t13 + a.ng * c2;
  int* const ticket = a.cnt;
  int* const exitc = a.cnt + 1;
  int* const grp_done = a.cnt + 2;
  int* const col_done = a.cnt + 2 + DF_MAX_GROUPS;
  if (tid == 0) s_task = atomicAdd(ticket, 1);
  __syncthreads();
  int t = s_task;
  while (t < total) {
    int nxt = 0;
    if (tid == 0) nxt = atomicAdd(ticket, 1);  // fetch the next ticket early (hides the atomic latency behind the work)
    if (t < t13) {
      // ---- w13 + swiglu + 32-block quantization (formula of mx_grouped_w13_kernel)
      const int gi = t / c13, cc = t % c13;
      const GroupDesc d = a.g[gi];
      const int nc0 = cc * DF_W13_COLS, n0 = nc0 + warp * 8;
      const uint8_t* const W[2] = {d.w1, d.w3};
      const uint8_t* const SW[2] = {d.s1, d.s3};
      float acc[2][1][4];
      Df2Tile<2, 1, false, 2 * S2> tl;
      tl.init(W, SW, n0, a.dim);
      tl.prologue(wb, sbm);
      tl.run(a.xq, a.xs, a.rows, d.row0, d.n, a.dim, acc, wb, sbm);
      if (r < d.n) {
        const float rwv = a.rw[d.row0 + r];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const int n = n0 + 2 * q + c;
          float gv = bf2f(f2bf(acc[0][0][c])), uv = bf2f(f2bf(acc[1][0][c]));
          if (a.limit > 0.f) { uv = fminf(fmaxf(uv, -a.limit), a.limit); gv = fminf(gv, a.limit); }
          float v = gv / (1.f + expf(-gv)) * uv * rwv;
          const bf16 vb = f2bf(v);
          a.y[(size_t)(d.row0 + r) * a.I + n] = vb;
          ysm[r][n - nc0] = vb;
        }
      }
      __syncthreads();
      if (tid < d.n) {  // one row = one 32-block
        float v[32];
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) { v[i] = bf2f(ysm[tid][i]); amax = fmaxf(amax, fabsf(v[i])); }
        amax = fmaxf(amax, 1e-4f);
        const uint8_t code = f32_ceil_pow2_e8m0(amax * (1.0f / 448.0f));
        const float sc = e8m0_to_f32(code);
        uint8_t* qp = a.yq + (size_t)(d.row0 + tid) * a.I + nc0;
#pragma unroll
        for (int i = 0; i < 32; ++i) qp[i] = f32_to_e4m3(fminf(fmaxf(v[i] / sc, -448.f), 448.f));
        a.ys[(size_t)(d.row0 + tid) * (a.I / 32) + nc0 / 32] = code;
        __threadfence();
      }
      __syncthreads();
      if (tid == 0) { __threadfence(); atomicAdd(grp_done + gi, 1); }
    } else {
      // ---- w2 (after waiting for all w13 items of the group) → eout · the last arrival at a 128-column slice accumulates into acc
      const int u2 = t - t13, gi = u2 / c2, cc = u2 % c2;
      const GroupDesc d = a.g[gi];
      const int nc0 = cc * DF_W2_COLS, n0 = nc0 + warp * (DF_W2_T * 8);
      const uint8_t* const W[1] = {d.w2};
      const uint8_t* const SW[1] = {d.s2};
      Df2Tile<1, DF_W2_T, true, S2> tl;  // issue the first S2−1 w2 weight stages before waiting (weights do not depend on the w13 results — A (yq/ys) is read after the wait)
      tl.init(W, SW, n0, a.I);
      tl.prologue(wb, sbm);
      if (tid == 0) {
        while (*reinterpret_cast<volatile int*>(grp_done + gi) < c13) __nanosleep(64);
        __threadfence();
      }
      __syncthreads();
      float acc[1][DF_W2_T][4];
      tl.run(a.yq, a.ys, nullptr, d.row0, d.n, a.I, acc, wb, sbm);
      if (r < d.n) {
#pragma unroll
        for (int i = 0; i < DF_W2_T; ++i)
#pragma unroll
          for (int c = 0; c < 2; ++c) a.eout[(size_t)(d.row0 + r) * a.dim + n0 + i * 8 + 2 * q + c] = f2bf(acc[0][i][c]);
      }
      __threadfence();
      __syncthreads();
      if (tid == 0) s_last = atomicAdd(col_done + cc, 1) == a.ng - 1;
      __syncthreads();
      if (s_last) {
        __threadfence();
        // same order as accum_bf16_rows_seq: add the bf16 values with row_map[r] == m to acc[m, n] in fp32, r ascending
        const int n = nc0 + tid;  // blockDim = 128 = DF_W2_COLS
        float s[DF_MAX_M];
#pragma unroll
        for (int m = 0; m < DF_MAX_M; ++m) s[m] = m < a.M ? a.acc[(size_t)m * a.dim + n] : 0.f;
        const int r1 = a.r0 + a.nrows;
        for (int rb = a.r0; rb < r1; rb += 8) {
          int mr[8];
          float v[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int rr = rb + j;
            mr[j] = rr < r1 ? __ldg(a.rows + rr) : -1;
            v[j] = rr < r1 ? __bfloat162float(__ushort_as_bfloat16(__ldcg(reinterpret_cast<const unsigned short*>(a.eout + (size_t)rr * a.dim + n)))) : 0.f;
          }
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int m = 0; m < DF_MAX_M; ++m)
              if (mr[j] == m) s[m] += v[j];
        }
#pragma unroll
        for (int m = 0; m < DF_MAX_M; ++m) if (m < a.M) a.acc[(size_t)m * a.dim + n] = s[m];
      }
    }
    __syncthreads();  // before s_task·ysm·s_last are reused
    if (tid == 0) s_task = nxt;
    __syncthreads();
    t = s_task;
  }
  // the last CTA to exit resets the counters to 0 (the next launch starts after this one ends, in stream order)
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(exitc, 1) == (int)gridDim.x - 1) {
      *ticket = 0;
      for (int i = 0; i < a.ng; ++i) grp_done[i] = 0;
      for (int i = 0; i < c2; ++i) col_done[i] = 0;
      *exitc = 0;
      __threadfence();
    }
  }
}

// ---- FUSED3 (HIVE_DECODE_FUSED3): per-warp work items · stages of 128 B contiguous per row · per-warp cp.async ring
//   (design rationale and value identity: the HIVE_DECODE_FUSED3 section of the file header). piece = one 8-row tile · stage = per piece 8 rows × 8 blocks (128 B contiguous per row).
#ifndef HIVE_DF3_AD
#define HIVE_DF3_AD 1  // activation prefetch distance in stages (Df3Tile::run) — 1 measured best (bench_moe_lowm: 2 and 3 no faster)
#endif
constexpr int DF3_WARPS = 4;
constexpr int DF3_CB = 8;                 // blocks per stage (128 B per row = one L2 line)
constexpr int DF3_P = 2;                  // pieces per stage: w13 = one tile of w1·w3 · w2 = 2 tiles (DF3_W2_T)
constexpr int DF3_W2_T = 2;
constexpr int DF3_W2_COLS = DF3_W2_T * 8;  // w2 item = 16 columns
constexpr int DF3_WPIECE = 8 * DF3_CB;     // uint4 slots per piece (8 rows × 8 blocks)
constexpr int DF3_SPIECE = 8 * DF3_CB / 4; // u32 scale slots per piece (8 rows × 8 bytes)
constexpr int DF3_MAX_GROUPS = 128;        // M ≤ 8 · top-6 → ≤ 48 groups in practice. Above this the caller uses FUSED2/FUSED (absorbed, not rejected)
constexpr int DF3_MAX_QB = 128;            // I/32 cap (actual 72)
constexpr int DF3_MAX_COLS = 1024;         // dim/16 cap (actual 320)
// Counters: [0] ticket · [1] exits (warps) · grp_done[MAX_GROUPS] · col_done[MAX_COLS] · qdone[MAX_GROUPS·MAX_QB]
constexpr int DF3_CNT_INTS = 2 + DF3_MAX_GROUPS + DF3_MAX_COLS + DF3_MAX_GROUPS * DF3_MAX_QB;
constexpr int df3_smem_bytes(int S) { return DF3_WARPS * S * DF3_P * (DF3_WPIECE * 16 + DF3_SPIECE * 4); }  // S × 8704 B

// The df_warp_tiles / Df2Tile computation (per 8-column tile blocks kb ascending, one accumulator, same operand words) with stages of 128 B contiguous per row.
//   Loads: 64 slots per piece (8 rows × 8 blocks of 16 B) — lane l loads slot idx = it·32 + l (it = 0,1) → row idx>>3, block idx&7: one warp instruction = 4 rows × 128 B contiguous.
//   smem slot = row·8 + (block ^ row) — the reader (lane (r,q) reads word q of block u) hits each of the 32 banks once per u (row r selects the 4-bank group (u^r)·4, q the bank within it).
//   Scales: per piece 8 rows × 8 bytes = 16 u32 slots — lanes 0..15 load 4 B each (row l>>1, half l&1). Lane (r,q) reads word h of row r (blocks 4h..4h+3) as a broadcast.
template <int NM, int T, bool A_CG, int S, int AD = HIVE_DF3_AD>
struct Df3Tile {
  static constexpr int P = NM * T;
  static_assert(P == DF3_P, "F3 stage size assumes 2 pieces");
  const uint8_t* w[NM][T];  // start of the weight row for tile row 0
  const uint8_t* s[NM][T];  // start of the scale row for tile row 0
  int nst, rowb, nb;
  __device__ __forceinline__ void init(const uint8_t* const (&W)[NM], const uint8_t* const (&SW)[NM], int n0, int K) {
    nb = K / 32;
    rowb = K / 2;
    nst = nb / DF3_CB;
#pragma unroll
    for (int j = 0; j < NM; ++j)
#pragma unroll
      for (int i = 0; i < T; ++i) {
        const size_t n = (size_t)(n0 + i * 8);
        w[j][i] = W[j] + n * rowb;
        s[j][i] = SW[j] + n * nb;
      }
  }
  __device__ __forceinline__ void issue(int st, uint4* wb, uint32_t* sb) const {
    if (st < nst) {
      const int slot = st % S, lane = threadIdx.x & 31;
#pragma unroll
      for (int j = 0; j < NM; ++j)
#pragma unroll
        for (int i = 0; i < T; ++i) {
          const int pc = slot * P + j * T + i;
#pragma unroll
          for (int it = 0; it < 2; ++it) {
            const int idx = it * 32 + lane, row = idx >> 3, c = idx & 7;
            df2_cp16(wb + pc * DF3_WPIECE + row * 8 + (c ^ row), w[j][i] + (size_t)row * rowb + (size_t)st * (DF3_CB * 16) + c * 16);
          }
          if (lane < 16) {
            const int row = lane >> 1, h = lane & 1;
            df2_cp4(sb + pc * DF3_SPIECE + row * 2 + h, s[j][i] + (size_t)row * nb + (size_t)st * DF3_CB + 4 * h);
          }
        }
    }
    df2_commit();  // commit empty groups too (keeps the wait_group count aligned with the stage number)
  }
  __device__ __forceinline__ void prologue(uint4* wb, uint32_t* sb) const {
#pragma unroll
    for (int st = 0; st < S - 1; ++st) issue(st, wb, sb);
  }
  __device__ __forceinline__ void run(const uint8_t* A, const uint8_t* sa, const int32_t* __restrict__ rows, int row0, int nrows, int K, float (&acc)[NM][T][4],
                                      uint4* wb, uint32_t* sb) const {
    const int lane = threadIdx.x & 31, q = lane & 3, r = lane >> 2;
#pragma unroll
    for (int j = 0; j < NM; ++j)
#pragma unroll
      for (int i = 0; i < T; ++i) acc[j][i][0] = acc[j][i][1] = acc[j][i][2] = acc[j][i][3] = 0.f;
    const bool have_a = r < nrows;
    const int sfa_row = r + 8 * (lane & 1);  // same scale-provider row as df_warp_tiles
    const int sfa_src = sfa_row < nrows ? sfa_row : 0;
    const uint8_t* sa_row = sa + (size_t)(rows ? rows[row0 + sfa_src] : row0 + sfa_src) * nb;
    const uint8_t* a_row = A + (size_t)(rows ? rows[row0 + (have_a ? r : 0)] : row0 + r) * K;
    const int nq4 = nb / 4;  // number of 4-block steps
    auto load_a = [&](int g, uint32_t (&a0)[4], uint32_t (&a2)[4], uint32_t& sfa4) {
      const int kb0 = g * 4;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        if (have_a) { const uint2 av = ld_a8<A_CG>(a_row + (size_t)(kb0 + u) * 32 + 8 * q); a0[u] = av.x; a2[u] = av.y; }
        else { a0[u] = 0u; a2[u] = 0u; }
      }
      sfa4 = ld_s4<A_CG>(sa_row + kb0);
    };
    // Activations run AD stages (2·AD 4-block steps) ahead of their use: at the start of stage st the two steps of stage st+AD are loaded into the
    //   last ring entry, and the ring rotates after the stage. Same addresses and values as loading them one step ahead — only the issue time moves
    //   (the dependent L2 round trip per step was the critical path of a work item when a launch has few items). Measured (S = 4, L2 flushed):
    //   bench_moe_lowm one expert 53.0 → 37.8 µs, five 114.6 → 94.2 µs; test_decode_moe M=1 811 → 912 GB/s, M=8 1349 → 1372 GB/s; AD 2 and 3
    //   no faster than 1; deeper weight rings (S 6/8/10) did not help with AD 0 or 1. Values are bit-identical (test_decode_moe).
    uint32_t c0[AD + 1][2][4], c2[AD + 1][2][4], cs[AD + 1][2];
#pragma unroll
    for (int d = 0; d < AD; ++d)
#pragma unroll
      for (int h = 0; h < 2; ++h) { const int g = d * 2 + h; load_a(g < nq4 ? g : nq4 - 1, c0[d][h], c2[d][h], cs[d][h]); }
    const uint32_t* const wb32 = reinterpret_cast<const uint32_t*>(wb);
    for (int st = 0; st < nst; ++st) {
#pragma unroll
      for (int h = 0; h < 2; ++h) { const int g = (st + AD) * 2 + h; load_a(g < nq4 ? g : nq4 - 1, c0[AD][h], c2[AD][h], cs[AD][h]); }
      df2_wait<S - 2>();  // this lane's share of stage st has arrived
      __syncwarp();       // make slots received by other lanes visible + the issue below overwrites slot (st−1)%S only after the previous iteration finished reading it
      issue(st + S - 1, wb, sb);
      const int slot = st % S;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int j = 0; j < NM; ++j)
#pragma unroll
          for (int i = 0; i < T; ++i) {
            const int pc = slot * P + j * T + i;
            const uint32_t sbw = sb[pc * DF3_SPIECE + r * 2 + h];  // scale bytes of blocks 8st+4h..+3 of row r
#pragma unroll
            for (int u = 0; u < 4; ++u) {
              const int ub = 4 * h + u;  // block within the stage = 8st + ub
              const uint32_t wv = wb32[(size_t)(pc * DF3_WPIECE + r * 8 + (ub ^ r)) * 4 + q];
              mma_e4m3_e2m1(acc[j][i], c0[0][h][u], c2[0][h][u], e2m1x4_bytes(wv & 0xFFFFu), e2m1x4_bytes(wv >> 16), (cs[0][h] >> (8 * u)) & 0xFFu,
                            (sbw >> (8 * u)) & 0xFFu);
            }
          }
      }
#pragma unroll
      for (int d = 0; d < AD; ++d)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
          for (int u = 0; u < 4; ++u) { c0[d][h][u] = c0[d + 1][h][u]; c2[d][h][u] = c2[d + 1][h][u]; }
          cs[d][h] = cs[d + 1][h];
        }
    }
    df2_wait<0>();
  }
};

// Work item = one warp: w13 (group, 8-column tile) — w1·w3 · swiglu · y / the last warp to arrive at a 32-column quantization block quantizes that block
//                   w2  (group, 16 columns) — eout / the last warp to arrive (all groups) at a 16-column slice accumulates it into acc.
//   Ticket order = all w13 (group order) → w2 (group order). Only w2 waits, and what it waits for (w13 of the same group) all have smaller tickets = taken by warps
//   already running (if the prefetched next ticket is w13, the current item is w13 too — it does not wait). No CTA synchronization (per-warp smem). Deadlock-free (same structure as the FUSED proof).
template <bool DEV, int S>
__global__ void __launch_bounds__(DF3_WARPS * 32) moe_decode_fused3_kernel(const DfArgs a_in) {
  static_assert(S >= 2, "F3 ring stage count");
  extern __shared__ uint4 df3_sm[];  // [warp][S·P·64] uint4 weight slots, then [warp][S·P·16] u32 scale slots
  DfArgs a = a_in;
  if constexpr (DEV) {
    const int* b = a.dc.base;
    a.ng = b[a.dc.ng];
    a.r0 = a.dc.r0 >= 0 ? b[a.dc.r0] : 0;
    a.nrows = b[a.dc.nrows];
  }
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, q = lane & 3, r = lane >> 2;
  uint4* const wb = df3_sm + warp * (S * DF3_P * DF3_WPIECE);
  uint32_t* const sbm = reinterpret_cast<uint32_t*>(df3_sm + DF3_WARPS * S * DF3_P * DF3_WPIECE) + warp * (S * DF3_P * DF3_SPIECE);
  const int n13 = a.I / 8, nq = a.I / 32, c2 = a.dim / DF3_W2_COLS;
  const int t13 = a.ng * n13, total = t13 + a.ng * c2;
  int* const ticket = a.cnt;
  int* const exitc = a.cnt + 1;
  int* const grp_done = a.cnt + 2;
  int* const col_done = grp_done + DF3_MAX_GROUPS;
  int* const qdone = col_done + DF3_MAX_COLS;
  int t = 0;
  if (lane == 0) t = atomicAdd(ticket, 1);
  t = __shfl_sync(0xffffffffu, t, 0);
  while (t < total) {
    int nxt = 0;
    if (lane == 0) nxt = atomicAdd(ticket, 1);  // fetch the next ticket early
    if (t < t13) {
      // ---- w13 + swiglu (formula of mx_grouped_w13_kernel) — 8 columns
      const int gi = t / n13, ti = t % n13;
      const GroupDesc d = a.g[gi];
      const int n0 = ti * 8;
      const uint8_t* const W[2] = {d.w1, d.w3};
      const uint8_t* const SW[2] = {d.s1, d.s3};
      float acc[2][1][4];
      Df3Tile<2, 1, false, S> tl;
      tl.init(W, SW, n0, a.dim);
      tl.prologue(wb, sbm);
      tl.run(a.xq, a.xs, a.rows, d.row0, d.n, a.dim, acc, wb, sbm);
      if (r < d.n) {
        const float rwv = a.rw[d.row0 + r];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const int n = n0 + 2 * q + c;
          float gv = bf2f(f2bf(acc[0][0][c])), uv = bf2f(f2bf(acc[1][0][c]));
          if (a.limit > 0.f) { uv = fminf(fmaxf(uv, -a.limit), a.limit); gv = fminf(gv, a.limit); }
          float v = gv / (1.f + expf(-gv)) * uv * rwv;
          a.y[(size_t)(d.row0 + r) * a.I + n] = f2bf(v);
        }
      }
      __threadfence();
      __syncwarp();
      int last = 0;
      if (lane == 0) {
        int* const qc = qdone + gi * DF3_MAX_QB + (ti >> 2);
        last = atomicAdd(qc, 1) == 3;  // 32-column block = four 8-column tiles
        if (last) *qc = 0;             // no further use in this launch (the next launch follows in stream order)
      }
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        __threadfence();
        const int nc0 = (ti >> 2) * 32;
        if (lane < d.n) {  // one row = one 32-block — the FUSED quantization formula verbatim (input = the bf16 y values written by the four warps = the same values as FUSED's ysm)
          const unsigned short* yp = reinterpret_cast<const unsigned short*>(a.y + (size_t)(d.row0 + lane) * a.I + nc0);
          float v[32];
          float amax = 0.f;
#pragma unroll
          for (int i = 0; i < 32; ++i) { v[i] = bf2f(__ushort_as_bfloat16(__ldcg(yp + i))); amax = fmaxf(amax, fabsf(v[i])); }
          amax = fmaxf(amax, 1e-4f);
          const uint8_t code = f32_ceil_pow2_e8m0(amax * (1.0f / 448.0f));
          const float sc = e8m0_to_f32(code);
          uint8_t* qp = a.yq + (size_t)(d.row0 + lane) * a.I + nc0;
#pragma unroll
          for (int i = 0; i < 32; ++i) qp[i] = f32_to_e4m3(fminf(fmaxf(v[i] / sc, -448.f), 448.f));
          a.ys[(size_t)(d.row0 + lane) * (a.I / 32) + nc0 / 32] = code;
          __threadfence();
        }
        __syncwarp();
        if (lane == 0) { __threadfence(); atomicAdd(grp_done + gi, 1); }
      }
    } else {
      // ---- w2 (after waiting for all quantization blocks of the group) → eout · the last arrival at a 16-column slice accumulates into acc
      const int u2 = t - t13, gi = u2 / c2, cc = u2 % c2;
      const GroupDesc d = a.g[gi];
      const int n0 = cc * DF3_W2_COLS;
      const uint8_t* const W[1] = {d.w2};
      const uint8_t* const SW[1] = {d.s2};
      Df3Tile<1, DF3_W2_T, true, S> tl;  // issue the first S−1 w2 weight stages before waiting (weights do not depend on the w13 results)
      tl.init(W, SW, n0, a.I);
      tl.prologue(wb, sbm);
      if (lane == 0) {
        while (*reinterpret_cast<volatile int*>(grp_done + gi) < nq) __nanosleep(64);
        __threadfence();
      }
      __syncwarp();
      float acc[1][DF3_W2_T][4];
      tl.run(a.yq, a.ys, nullptr, d.row0, d.n, a.I, acc, wb, sbm);
      if (r < d.n) {
#pragma unroll
        for (int i = 0; i < DF3_W2_T; ++i)
#pragma unroll
          for (int c = 0; c < 2; ++c) a.eout[(size_t)(d.row0 + r) * a.dim + n0 + i * 8 + 2 * q + c] = f2bf(acc[0][i][c]);
      }
      __threadfence();
      __syncwarp();
      int last = 0;
      if (lane == 0) {
        last = atomicAdd(col_done + cc, 1) == a.ng - 1;
        if (last) col_done[cc] = 0;
      }
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last && lane < DF3_W2_COLS) {
        __threadfence();
        // same order as accum_bf16_rows_seq (verbatim from FUSED): add the bf16 values with row_map[r] == m to acc[m, n] in fp32, r ascending
        const int n = n0 + lane;
        float s[DF_MAX_M];
#pragma unroll
        for (int m = 0; m < DF_MAX_M; ++m) s[m] = m < a.M ? a.acc[(size_t)m * a.dim + n] : 0.f;
        const int r1 = a.r0 + a.nrows;
        for (int rb = a.r0; rb < r1; rb += 8) {
          int mr[8];
          float v[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int rr = rb + j;
            mr[j] = rr < r1 ? __ldg(a.rows + rr) : -1;
            v[j] = rr < r1 ? __bfloat162float(__ushort_as_bfloat16(__ldcg(reinterpret_cast<const unsigned short*>(a.eout + (size_t)rr * a.dim + n)))) : 0.f;
          }
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int m = 0; m < DF_MAX_M; ++m)
              if (mr[j] == m) s[m] += v[j];
        }
#pragma unroll
        for (int m = 0; m < DF_MAX_M; ++m) if (m < a.M) a.acc[(size_t)m * a.dim + n] = s[m];
      }
    }
    __syncwarp();
    t = __shfl_sync(0xffffffffu, nxt, 0);
  }
  // the last warp to exit resets ticket, grp_done and the exit count to 0 (qdone and col_done were already reset by their last arrival)
  if (lane == 0) {
    __threadfence();
    if (atomicAdd(exitc, 1) == (int)gridDim.x * DF3_WARPS - 1) {
      *ticket = 0;
      for (int i = 0; i < a.ng; ++i) grp_done[i] = 0;
      *exitc = 0;
      __threadfence();
    }
  }
}

// One counter buffer per stream (launches on a stream are serialized; different streams use different buffers). Allocated and zeroed once (stream-ordered memset).
int* counters_for(cudaStream_t st) {
  static std::mutex mu;
  static std::unordered_map<cudaStream_t, int*> m;
  std::lock_guard<std::mutex> lk(mu);
  int*& p = m[st];
  if (!p) {
    CUDA_CHECK(cudaMalloc((void**)&p, sizeof(int) * DF_CNT_INTS));
    CUDA_CHECK(cudaMemsetAsync(p, 0, sizeof(int) * DF_CNT_INTS, st));
  }
  return p;
}

int resident_ctas() {
  static int n = [] {
    int dev = 0, sms = 0, occ = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, moe_decode_fused_kernel<false>, DF_WARPS * 32, 0));
    return std::max(1, sms * std::max(1, occ));
  }();
  return n;
}

// ---- FUSED2 host side: variant selection · resident CTAs (with dynamic smem) · launch
template <int S2> int resident_ctas2() {  // also sets the dynamic-smem limit attribute (S2=6 exceeds 48 KB; outside capture — preload calls this first)
  static int n = [] {
    int dev = 0, sms = 0, occ = 0;
    const int smem = df2_smem_bytes(S2);
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaFuncSetAttribute(moe_decode_fused2_kernel<false, S2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaFuncSetAttribute(moe_decode_fused2_kernel<true, S2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, moe_decode_fused2_kernel<false, S2>, DF_WARPS * 32, smem));
    return std::max(1, sms * std::max(1, occ));
  }();
  return n;
}
std::atomic<int> g_force{-1};  // for tests (moe_decode_fused_force) — -1 = from the environment
inline bool md_env_on(const char* name) { const char* v = getenv(name); return v && *v && strcmp(v, "0") != 0; }  // same formula as runtime.cpp env_on
int env_variant() {  // 0 = FUSED · 3/4/6 = FUSED2 with that S2
  static const bool fused2 = md_env_on("HIVE_DECODE_FUSED2");
  static const int v = [] {
    if (!fused2) return 0;
    const char* t = getenv("HIVE_DECODE_FUSED2_STAGES");  // empty/0/negative/unparsable → 4 · ≤3 → 3 · 4..5 → 4 · ≥6 → 6
    const int n = t && *t ? atoi(t) : 0;
    return n <= 0 ? 4 : n <= 3 ? 3 : n <= 5 ? 4 : 6;
  }();
  return v;
}
int env_variant3();
int variant() {  // FUSED3 if enabled (takes precedence over FUSED2) — a forced value (tests) wins
  const int f = g_force.load(std::memory_order_relaxed);
  if (f >= 0) return f;
  const int v3 = env_variant3();
  return v3 ? v3 : env_variant();
}
template <bool DEV, int S2> void launch2_v(const DfArgs& a, int total, cudaStream_t st) {
  const int grid = std::min(total, resident_ctas2<S2>());
  moe_decode_fused2_kernel<DEV, S2><<<grid, DF_WARPS * 32, df2_smem_bytes(S2), st>>>(a);
}
template <bool DEV> void launch2(const DfArgs& a, int total, int v, cudaStream_t st) {
  if (v == 3) launch2_v<DEV, 3>(a, total, st);
  else if (v == 6) launch2_v<DEV, 6>(a, total, st);
  else launch2_v<DEV, 4>(a, total, st);
}

// ---- FUSED3 host side: variant number = 100 + S (3/4/5) · counters (separate per stream — different size) · resident CTAs · launch
constexpr int DF3_VARIANT = 100;
int env_variant3() {  // 0 = off (→ env_variant: FUSED2 or FUSED) · 103/104/105 = FUSED3 with that S
  static const bool fused3 = md_env_on("HIVE_DECODE_FUSED3");
  static const int v = [] {
    if (!fused3) return 0;
    const char* t = getenv("HIVE_DECODE_FUSED3_STAGES");  // empty/0/negative/unparsable → 4 · ≤3 → 3 · 4 → 4 · ≥5 → 5
    const int n = t && *t ? atoi(t) : 0;
    return DF3_VARIANT + (n <= 0 ? 4 : n <= 3 ? 3 : n <= 4 ? 4 : 5);
  }();
  return v;
}
int* counters3_for(cudaStream_t st) {
  static std::mutex mu;
  static std::unordered_map<cudaStream_t, int*> m;
  std::lock_guard<std::mutex> lk(mu);
  int*& p = m[st];
  if (!p) {
    CUDA_CHECK(cudaMalloc((void**)&p, sizeof(int) * DF3_CNT_INTS));
    CUDA_CHECK(cudaMemsetAsync(p, 0, sizeof(int) * DF3_CNT_INTS, st));
  }
  return p;
}
template <int S> int resident_ctas3() {  // also sets the dynamic-smem limit attribute (outside capture — preload calls this first)
  static int n = [] {
    int dev = 0, sms = 0, occ = 0;
    const int smem = df3_smem_bytes(S);
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaFuncSetAttribute(moe_decode_fused3_kernel<false, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaFuncSetAttribute(moe_decode_fused3_kernel<true, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, moe_decode_fused3_kernel<false, S>, DF3_WARPS * 32, smem));
    return std::max(1, sms * std::max(1, occ));
  }();
  return n;
}
// FUSED3 shape conditions (on top of moe_decode_fused_ok): a stage is 8 blocks, so K must be a multiple of 256 · counter table sizes. Otherwise FUSED3 is skipped (absorbed — only the variant choice changes).
bool f3_ok(int dim, int I, int ng) {
  return dim % 256 == 0 && I % 256 == 0 && ng <= DF3_MAX_GROUPS && I / 32 <= DF3_MAX_QB && dim / DF3_W2_COLS <= DF3_MAX_COLS;
}
template <bool DEV, int S> void launch3_v(DfArgs a, int ng, cudaStream_t st) {
  a.cnt = counters3_for(st);
  const int total = ng * (a.I / 8 + a.dim / DF3_W2_COLS);  // number of warp work items
  // grid = min(items, resident CTAs) — not items/4: even in launches with fewer items than resident warps (DMA share, 1–3 groups) the first tickets then spread over
  //   all SMs instead of piling into one CTA (surplus warps get a first ticket ≥ total and exit immediately).
  const int grid = std::min(total, resident_ctas3<S>());
  moe_decode_fused3_kernel<DEV, S><<<grid, DF3_WARPS * 32, df3_smem_bytes(S), st>>>(a);
}
// Launch variant v (variant()) — true if launched (false = use the FUSED kernel). If the FUSED3 conditions fail, the variant chosen without FUSED3 (FUSED when forced by a test, otherwise env FUSED2/FUSED).
template <bool DEV> bool launch_variant(const DfArgs& a, int ng, int total2, int v, cudaStream_t st) {
  if (v > DF3_VARIANT) {
    if (f3_ok(a.dim, a.I, ng)) {
      if (v == DF3_VARIANT + 3) launch3_v<DEV, 3>(a, ng, st);
      else if (v == DF3_VARIANT + 5) launch3_v<DEV, 5>(a, ng, st);
      else launch3_v<DEV, 4>(a, ng, st);
      return true;
    }
    v = g_force.load(std::memory_order_relaxed) >= 0 ? 0 : env_variant();
  }
  if (!v) return false;
  launch2<DEV>(a, total2, v, st);
  return true;
}

}  // namespace

bool moe_decode_fused_ok(int dim, int I, int M, int ngroups) {
  return M >= 1 && M <= DF_MAX_M && ngroups >= 1 && ngroups <= DF_MAX_GROUPS && dim > 0 && I > 0 && dim % 128 == 0 && I % 128 == 0 &&
         dim / DF_W2_COLS <= DF_MAX_CHUNKS;
}

bool moe_decode_desc_aligned(const GroupDesc& d) {
  auto al = [](const void* p, uintptr_t a) { return ((uintptr_t)p & (a - 1)) == 0; };
  return al(d.w1, 16) && al(d.w3, 16) && al(d.w2, 16) && al(d.s1, 4) && al(d.s3, 4) && al(d.s2, 4) && d.n >= 1 && d.n <= 8;
}

void moe_decode_fused(const GroupDesc* g, int ngroups, const uint8_t* xq, const uint8_t* xs, const int32_t* rows, const float* rw, int r0, int n_rows,
                      int M, int dim, int I, float limit, bf16* y, uint8_t* yq, uint8_t* ys, bf16* eout, float* acc, cudaStream_t st) {
  if (ngroups <= 0) return;
  HIVE_CHECK(moe_decode_fused_ok(dim, I, M, ngroups), "moe_decode_fused shape");
  const DfArgs a{g, ngroups, xq, xs, rows, rw, r0, n_rows, M, dim, I, limit, y, yq, ys, eout, acc, counters_for(st), DfDevCount{}};
  const int total = ngroups * (I / DF_W13_COLS + dim / DF_W2_COLS);
  if (const int v = variant(); v && launch_variant<false>(a, ngroups, total, v, st)) return;  // FUSED2 · FUSED3
  const int grid = std::min(total, resident_ctas());
  moe_decode_fused_kernel<false><<<grid, DF_WARPS * 32, 0, st>>>(a);
}

// Group count and row range from device-side counts (step graph). Grid sized for the maximum group count (with tickets, CTAs beyond the actual work exit immediately —
//   ng = 0 means zero items; the last CTA to exit resets the counters). Conditions (shape, alignment) are checked once by the caller (moe_decode_fused_ok(…, max_groups), record alignment).
void moe_decode_fused_dev(const GroupDesc* g, int max_groups, DfDevCount dc, const uint8_t* xq, const uint8_t* xs, const int32_t* rows, const float* rw,
                          int M, int dim, int I, float limit, bf16* y, uint8_t* yq, uint8_t* ys, bf16* eout, float* acc, cudaStream_t st) {
  HIVE_CHECK(max_groups >= 1 && moe_decode_fused_ok(dim, I, M, max_groups), "moe_decode_fused_dev shape");
  const DfArgs a{g, max_groups, xq, xs, rows, rw, 0, 0, M, dim, I, limit, y, yq, ys, eout, acc, counters_for(st), dc};
  const int total = max_groups * (I / DF_W13_COLS + dim / DF_W2_COLS);
  if (const int v = variant(); v && launch_variant<true>(a, max_groups, total, v, st)) return;  // FUSED2 · FUSED3
  const int grid = std::min(total, resident_ctas());
  moe_decode_fused_kernel<true><<<grid, DF_WARPS * 32, 0, st>>>(a);
}
// Finish lazy module loading (CUDA_MODULE_LOADING=LAZY) outside capture and spin waits + create the stream counters up front (no cudaMalloc during capture)
void moe_decode_fused_preload(cudaStream_t st) {
  cudaFuncAttributes fa;
  CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused_kernel<false>));
  CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused_kernel<true>));
  (void)counters_for(st);
  (void)resident_ctas();
  // also load the FUSED2/FUSED3 variant (if enabled) outside capture + the dynamic-smem attribute (raised by resident_ctas2/3)
  switch (variant()) {
    case 3: CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused2_kernel<false, 3>)); CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused2_kernel<true, 3>)); (void)resident_ctas2<3>(); break;
    case 4: CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused2_kernel<false, 4>)); CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused2_kernel<true, 4>)); (void)resident_ctas2<4>(); break;
    case 6: CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused2_kernel<false, 6>)); CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused2_kernel<true, 6>)); (void)resident_ctas2<6>(); break;
    case DF3_VARIANT + 3: CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused3_kernel<false, 3>)); CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused3_kernel<true, 3>)); (void)resident_ctas3<3>(); (void)counters3_for(st); break;
    case DF3_VARIANT + 4: CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused3_kernel<false, 4>)); CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused3_kernel<true, 4>)); (void)resident_ctas3<4>(); (void)counters3_for(st); break;
    case DF3_VARIANT + 5: CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused3_kernel<false, 5>)); CUDA_CHECK(cudaFuncGetAttributes(&fa, moe_decode_fused3_kernel<true, 5>)); (void)resident_ctas3<5>(); (void)counters3_for(st); break;
    default: break;
  }
  // also preload the FUSED2 variant used when FUSED3 drops out on shape conditions (env_variant — if FUSED2 is enabled together with FUSED3)
  if (variant() > DF3_VARIANT) switch (env_variant()) {
    case 3: (void)resident_ctas2<3>(); break;
    case 4: (void)resident_ctas2<4>(); break;
    case 6: (void)resident_ctas2<6>(); break;
    default: break;
  }
}

void moe_decode_fused_force(int stages) {
  g_force.store(stages < 0 ? -1 : stages == 0 ? 0 : stages <= 3 ? 3 : stages <= 5 ? 4 : 6, std::memory_order_relaxed);
}
// FUSED3 forcing for tests: -1 = from the environment · otherwise FUSED3 with S (≤3 → 3 · 4 → 4 · ≥5 → 5). Shares the slot with moe_decode_fused_force (the last call wins).
void moe_decode_fused_force3(int stages) {
  g_force.store(stages < 0 ? -1 : DF3_VARIANT + (stages <= 3 ? 3 : stages == 4 ? 4 : 5), std::memory_order_relaxed);
}
int moe_decode_fused_variant() { return variant(); }

}  // namespace hive::k
