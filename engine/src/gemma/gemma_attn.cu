// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Paged fp8 KV cache and attention for Gemma 4 (hive/gemma/gemma_kernels.h): attn_paged, kv_write.
//
// Work unit = (sequence, kv head). The query heads that share the kv head are folded into rows: unit row r is query qi = r / ratio of
//   head g · ratio + r % ratio (ratio = H / Hkv), so a full layer (ratio 16) at decode is exactly one 16-row mma tile and its 16 heads
//   share one pass over K / V. A block owns M unit rows and walks key tiles of KT keys (a tile never crosses a page).
//
// Tensor cores: mma.sync m16n8k16 with f16 operands and fp32 accumulation. Every e4m3 value is exactly representable in f16, so K / V
//   are converted with one cvt.rn.f16x2.e4m3x2 per pair while the B fragments are built from the fp8 bytes in shared memory (no f16
//   copy of the tile). q (bf16) is converted to f16 once per block (exact for |q| in [2^-14, 65504]; larger values saturate). P is rounded
//   to f16 for P·V (11-bit mantissa, finer than bf16) and the row sums use the same rounded values.
//   Fragment layout (g = lane / 4, c = lane % 4): A a0 = (g, k 2c..2c+1) a1 = (g+8, ..) a2 = (g, k 2c+8..) a3 = (g+8, 2c+8..),
//   B b0 = (k 2c..2c+1, n g) b1 = (k 2c+8.., n g), C c0, c1 = (g, n 2c..2c+1) c2, c3 = (g+8, ..).
//   · Q·K^T: the reduction order over d inside a k16 step is free, so a lane loads 8 contiguous K bytes (d 8c .. 8c+7 of a 32-d group)
//     and uses bytes 0..3 as logical k {2c, 2c+1, 2c+8, 2c+9} of the first k16 step and bytes 4..7 for the second; the q A fragments use
//     the same permutation (one 16-byte load of q[row][8c .. 8c+7] per row per 32-d group).
//   · P·V: a lane loads one 32-bit word (d 4g .. 4g+3 of a 32-d group) from each of the key rows 2c, 2c+1, 2c+8, 2c+9; byte i of the
//     four words is the B fragment of output column 4g + i, so four n-tiles come out of four loads (byte interleave with prmt). The
//     output column of C element (n-tile i, column n) is therefore 8 (n / 2) + 4 (n % 2) + i — a lane ends up owning 8 contiguous columns.
//   Shared-memory row strides are padded (K: D + 32 B, V: D + 16 B, q: 2D + 64 B, P: 2 KT + 16 B) so all of these loads are conflict-free.
// Warps: (M / 16 row groups) × DS. In Q·K^T the first QDS (≤ DS) warps of a row group take KT / QDS keys each; the tile row max is exchanged through
//   shared memory (max is order-free, so all DS warps hold the same running max); each writes its exp2(s - m) as f16 to a P tile, and in
//   P·V the DS warps take D / DS output columns each over all KT keys. Softmax runs in the log2 domain (scores · log2 e, exp2).
// Pipeline: K and V tiles stream through ST stages each with cp.async; K of tile i + ST is issued as soon as Q·K^T of tile i is done
//   (overlapping P·V), V of tile i + ST after P·V of tile i. Key slots beyond the block's last query position are zero-filled (they may hold
//   stale bytes, even NaN codes) and every invisible key is masked to -inf before the max, so only visible keys reach the sums.
//   Page pointers are read only for tiles that intersect some row's visible range, so released (null) sliding pages are never touched.
//
// Two paths, chosen per sequence from its own n_q only (so a sequence's result never depends on the rest of the call):
//   · decode / verify (n_q ≤ 16): M = 16 rows per block, 8 warps, keys split over a thread-block cluster of SPLIT blocks (sliding
//     layers 8: D 256, KT 64, 2 stages; full layers 16 — the non-portable cluster maximum — D 512, KT 64, 1 stage). Block rank b takes
//     the tiles t with t % SPLIT == b (absolute tile index) in increasing order and writes its partial (fp32 O, running max, row sum) to
//     a scratch slot; after the cluster barrier block b combines output columns [32 b, 32 b + 32) over the SPLIT partials in fixed rank
//     order. (Exchanging the partials through distributed shared memory measured 4–50x slower on sm_120 than going through L2.)
//     Because the split is by absolute tile index and an invisible tile adds exact zeros (rescale factor exactly 1), a row's result
//     depends on neither n_q nor the other rows of its tile: decode at position p and the verify row at p are bitwise identical.
//     The scratch is a static device array (no allocation per call); a call needing more slots than it holds runs as several launches
//     over disjoint sequence ranges (the launch count depends only on n_seqs, max_q_per_seq and the layer type, so it is fixed under
//     CUDA graph capture; nothing in either path reads device data on the host).
//   · prefill chunks (n_q > 16): no key split; D 256: M = 64, DS = 2, KT = 64; D 512: M = 32, DS = 4, KT = 32 (dynamic smem ≤ 99 KB per
//     block on sm_120). Row tiles are scheduled latest-first (they have the most keys).
//   Q·K^T with one 8-key n-tile per warp accumulates over D / 16 dependent mmas; it is split into NA interleaved accumulators (added
//   in fixed order) so the tensor pipe is not latency-bound.
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "hive/common.h"
#include "hive/gemma/gemma_kernels.h"
#include "gemma_pdl.cuh"

namespace hive::gemma::k {

namespace {

namespace cg = cooperative_groups;

// decode / verify key split = cluster size, fixed per layer type (never chosen from the batch): sliding layers (8 kv heads) already give
//   batch × 8 units, full layers (1 kv head) need the wider split (16 = the non-portable cluster maximum on sm_120) to fill the GPU.
constexpr int kSplitSliding = 8, kSplitFull = 16;
constexpr int kDecodeMaxQ = 16;   // sequences with n_q ≤ this take the split path
constexpr float kLog2e = 1.4426950408889634f;

// Split-K scratch for the decode / verify path: one slot per cluster in flight, statically allocated (no allocation per call, safe under
//   graph capture). A call whose clusters exceed it runs as several launches (waves) over disjoint sequence ranges, reusing the slots —
//   the numerics never depend on the wave a sequence lands in.
constexpr size_t kWsFloats = (size_t)4 << 20;   // 16 MiB
__device__ float g_split_ws[kWsFloats];

struct Args {
  const bf16* q;
  bf16* out;
  const AttnSeq* seqs;   // decode / verify launches: already offset to the wave's first sequence
  int H, Hkv, ratio, window;
  size_t tok;    // bytes between consecutive tokens of a page
  size_t koff;   // byte offset of this layer's K (head 0) inside a token
  size_t voff;   // same for V
};

// D head dim · M rows per block · DS warps per row group (P·V column split) · QDS of them compute Q·K^T (key split) · KT keys per tile ·
//   ST pipeline stages · SPLIT cluster size (1 = no key split)
template <int D, int M, int DS, int QDS, int KT, int ST, int SPLIT>
struct Cfg {
  static constexpr int RG = M / 16, NW = RG * DS, NT = NW * 32;
  static constexpr int KS = D + 32, VS = D + 16, QS = 2 * D + 64, PS = 2 * KT + 16;   // smem row strides (bytes)
  static constexpr int NQ = KT / QDS / 8;  // Q·K^T n-tiles per Q·K^T warp
  static constexpr int NA = NQ >= 4 ? 1 : NQ == 2 ? 2 : 4;   // Q·K^T accumulator chains per n-tile
  static constexpr int DW = D / DS;        // P·V output columns per warp
  static constexpr int NG = DW / 32;       // 32-column groups per warp
  static constexpr size_t K_BYTES = (size_t)ST * KT * KS, V_BYTES = (size_t)ST * KT * VS;
  static constexpr size_t Q_BYTES = (size_t)M * QS, P_BYTES = (size_t)M * PS, RED_BYTES = (size_t)RG * DS * 16 * 4;
  static constexpr size_t SMEM = K_BYTES + V_BYTES + Q_BYTES + P_BYTES + 2 * RED_BYTES;
  static constexpr size_t SLOT = (size_t)SPLIT * 16 * (D + 2);   // split scratch floats per cluster: O [SPLIT][16][D] + m, l [SPLIT][16]
  static_assert(QDS <= DS && KT % QDS == 0 && (KT / QDS) % 8 == 0 && D % (DS * 32) == 0 && kPage % KT == 0 && KT % 16 == 0, "tile shape");
  static_assert(SMEM <= 101376 && SLOT > 0, "sm_120: 99 KB of dynamic shared memory per block");
  static_assert(P_BYTES >= (size_t)16 * (SPLIT + 1) * 4, "combine weights alias the P tile");
  static_assert(SPLIT == 1 || (M == 16 && D == 32 * SPLIT), "split path: one 16-row tile, 32 output columns per rank");
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(smem)), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_f16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// four e4m3 bytes → two f16x2 (bytes 0, 1 → lo; bytes 2, 3 → hi; the lower byte lands in the lower half)
__device__ __forceinline__ void e4m3x4_to_f16x4(uint32_t w, uint32_t& lo, uint32_t& hi) {
  asm("{\n .reg .b16 l, h;\n mov.b32 {l, h}, %2;\n cvt.rn.f16x2.e4m3x2 %0, l;\n cvt.rn.f16x2.e4m3x2 %1, h;\n}\n" : "=r"(lo), "=r"(hi) : "r"(w));
}
__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t w) {   // saturating at ±65504
  const float a = fminf(fmaxf(__uint_as_float(w << 16), -65504.f), 65504.f);
  const float b = fminf(fmaxf(__uint_as_float(w & 0xFFFF0000u), -65504.f), 65504.f);
  const __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// SPLIT = 1: prefill path (no key split); SPLIT > 1: decode / verify path, cluster of SPLIT blocks
template <int D, int M, int DS, int QDS, int KT, int ST, int SPLIT>
__global__ void __launch_bounds__(Cfg<D, M, DS, QDS, KT, ST, SPLIT>::NT, 1) attn_kernel(const Args a) {
  using C = Cfg<D, M, DS, QDS, KT, ST, SPLIT>;
  constexpr int kSplit = SPLIT;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sK = smem;
  uint8_t* sV = sK + C::K_BYTES;
  uint8_t* sQ = sV + C::V_BYTES;
  uint8_t* sP = sQ + C::Q_BYTES;
  float* red = reinterpret_cast<float*>(sP + C::P_BYTES);   // [RG][DS][16] tile row max partials (first QDS used)
  float* lred = red + C::RG * DS * 16;                      // [RG][DS][16] row sum partials
  pdl_trigger();
  pdl_wait();   // q / KV pages / the partial buffer belong to the chain from here on

  int seq_i, g, mtile, rank = 0;
  if constexpr (SPLIT > 1) {
    rank = blockIdx.x; mtile = blockIdx.y; seq_i = blockIdx.z / a.Hkv; g = blockIdx.z % a.Hkv;
  } else {
    mtile = gridDim.x - 1 - blockIdx.x; g = blockIdx.y; seq_i = blockIdx.z;
  }
  const AttnSeq sq = a.seqs[seq_i];
  // every block of a cluster shares (sequence, kv head, row tile), so these exits are cluster-uniform
  if (SPLIT > 1 ? (sq.n_q <= 0 || sq.n_q > kDecodeMaxQ) : sq.n_q <= kDecodeMaxQ) return;
  const int ratio = a.ratio, rows = sq.n_q * ratio, r0 = mtile * M;
  if (r0 >= rows) return;
  const int rlast = min(rows, r0 + M) - 1;
  const int pmin = sq.q_pos0 + r0 / ratio, pmax = sq.q_pos0 + rlast / ratio;
  const int lo = a.window > 0 ? max(0, pmin - a.window + 1) : 0, hi = pmax;
  const int tlo = lo / KT, thi = hi / KT;
  int t_first, t_step, n_t;
  if constexpr (SPLIT > 1) {
    t_first = tlo + ((rank - tlo % kSplit) + kSplit) % kSplit;
    t_step = kSplit;
    n_t = t_first > thi ? 0 : (thi - t_first) / kSplit + 1;
  } else {
    t_first = tlo; t_step = 1; n_t = thi - tlo + 1;
  }

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rg = warp / DS, ds = warp % DS, gq = lane >> 2, c = lane & 3;
  const uint8_t* const* pages = sq.pages;
  const size_t kv_head = (size_t)g * D;

  auto load = [&](int i, bool v) {
    if (i < n_t) {
      const int j0 = (t_first + i * t_step) * KT;
      const uint8_t* pg = pages[j0 / kPage] + (v ? a.voff : a.koff) + kv_head;
      const int rs = v ? C::VS : C::KS;
      uint8_t* dst = (v ? sV : sK) + (size_t)(i % ST) * KT * rs;
      for (int x = tid; x < KT * (D / 16); x += C::NT) {
        const int kk = x / (D / 16), ch = x % (D / 16), j = j0 + kk;
        uint8_t* d = dst + kk * rs + ch * 16;
        if (j <= hi) cp_async16(d, pg + (size_t)(j % kPage) * a.tok + ch * 16);
        else *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    cp_async_commit();
  };
#pragma unroll
  for (int s = 0; s < ST; ++s) { load(s, false); load(s, true); }

  // q rows → f16 in smem (zero rows past the unit)
  for (int x = tid; x < M * (D / 8); x += C::NT) {
    const int row = x / (D / 8), ch = x % (D / 8), r = r0 + row;
    uint4 w = make_uint4(0u, 0u, 0u, 0u);
    if (r < rows) {
      const int qi = r / ratio, h = g * ratio + r % ratio;
      w = *reinterpret_cast<const uint4*>(a.q + ((size_t)(sq.q_row0 + qi) * a.H + h) * D + ch * 8);
      w.x = bf16x2_to_f16x2(w.x); w.y = bf16x2_to_f16x2(w.y); w.z = bf16x2_to_f16x2(w.z); w.w = bf16x2_to_f16x2(w.w);
    }
    *reinterpret_cast<uint4*>(sQ + row * C::QS + ch * 16) = w;
  }

  const int rowA = rg * 16 + gq;   // this lane's rows: rowA, rowA + 8
  const int posA = sq.q_pos0 + (r0 + rowA) / ratio, posB = sq.q_pos0 + (r0 + rowA + 8) / ratio;
  float mA = -INFINITY, mB = -INFINITY, lA = 0.f, lB = 0.f;
  float o[C::NG][4][4];
#pragma unroll
  for (int G = 0; G < C::NG; ++G)
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[G][i][e] = 0.f;

  const uint8_t* qbase = sQ + rowA * C::QS + 16 * c;
  const bool qk = ds < QDS;         // this warp computes Q·K^T (warp-uniform)
  const int kq0 = ds * (KT / QDS);  // its first key of a tile
  const int dbase = ds * C::DW;     // this warp's first output column in P·V

  for (int i = 0; i < n_t; ++i) {
    const int stage = i % ST;
    const int jt = (t_first + i * t_step) * KT;
    cp_async_wait<2 * ST - 1>();
    __syncthreads();
    // ---- S = q · K^T over this warp's keys
    // NA independent accumulators over interleaved 32-d groups (one 8-key n-tile per warp would otherwise be one serial chain of D / 16
    //   dependent mmas), added in fixed order
    float s[C::NQ][4];
    float xA = -INFINITY, xB = -INFINITY;
    if (qk) {
    float sa[C::NA][C::NQ][4];
#pragma unroll
    for (int u = 0; u < C::NA; ++u)
#pragma unroll
      for (int nt = 0; nt < C::NQ; ++nt) sa[u][nt][0] = sa[u][nt][1] = sa[u][nt][2] = sa[u][nt][3] = 0.f;
    const uint8_t* kbase = sK + (size_t)stage * KT * C::KS + (kq0 + gq) * C::KS + 8 * c;
#pragma unroll 2
    for (int G0 = 0; G0 < D / 32; G0 += C::NA) {
#pragma unroll
      for (int u = 0; u < C::NA; ++u) {
        const int G = G0 + u;
        const uint4 qa = *reinterpret_cast<const uint4*>(qbase + G * 64);
        const uint4 qb = *reinterpret_cast<const uint4*>(qbase + 8 * C::QS + G * 64);
#pragma unroll
        for (int nt = 0; nt < C::NQ; ++nt) {
          const uint2 kw = *reinterpret_cast<const uint2*>(kbase + nt * 8 * C::KS + G * 32);
          uint32_t b0, b1, b2, b3;
          e4m3x4_to_f16x4(kw.x, b0, b1);
          e4m3x4_to_f16x4(kw.y, b2, b3);
          mma_f16(sa[u][nt], qa.x, qb.x, qa.y, qb.y, b0, b1);
          mma_f16(sa[u][nt], qa.z, qb.z, qa.w, qb.w, b2, b3);
        }
      }
    }
#pragma unroll
    for (int nt = 0; nt < C::NQ; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float v = sa[0][nt][e];
#pragma unroll
        for (int u = 1; u < C::NA; ++u) v += sa[u][nt][e];
        s[nt][e] = v;
      }
    // ---- mask, tile row max
    const bool full = jt + KT - 1 <= pmin && (a.window <= 0 || jt > pmax - a.window);
#pragma unroll
    for (int nt = 0; nt < C::NQ; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float v = s[nt][e] * kLog2e;
        if (!full) {
          const int j = jt + kq0 + nt * 8 + 2 * c + (e & 1), p = e < 2 ? posA : posB;
          if (j > p || (a.window > 0 && j <= p - a.window)) v = -INFINITY;
        }
        s[nt][e] = v;
        if (e < 2) xA = fmaxf(xA, v); else xB = fmaxf(xB, v);
      }
    xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 1)); xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 2));
    xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 1)); xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 2));
    }
    if constexpr (DS > 1) {
      if (qk && c == 0) { red[(rg * DS + ds) * 16 + gq] = xA; red[(rg * DS + ds) * 16 + gq + 8] = xB; }
      __syncthreads();
#pragma unroll
      for (int d2 = 0; d2 < QDS; ++d2) { xA = fmaxf(xA, red[(rg * DS + d2) * 16 + gq]); xB = fmaxf(xB, red[(rg * DS + d2) * 16 + gq + 8]); }
    }
    const float nA = fmaxf(mA, xA), nB = fmaxf(mB, xB);
    const float alA = nA == -INFINITY ? 1.f : exp2f(mA - nA), alB = nB == -INFINITY ? 1.f : exp2f(mB - nB);
    const float bA = nA == -INFINITY ? 0.f : nA, bB = nB == -INFINITY ? 0.f : nB;
    mA = nA; mB = nB;
    // ---- P = exp2(s - m) as f16 → smem; row sums of the rounded values
    float sumA = 0.f, sumB = 0.f;
    uint8_t* prow = sP + rowA * C::PS + (kq0 + 2 * c) * 2;
#pragma unroll
    for (int nt = 0; nt < C::NQ; ++nt) {
      if (!qk) break;
      const __half2 pa = __floats2half2_rn(exp2f(s[nt][0] - bA), exp2f(s[nt][1] - bA));
      const __half2 pb = __floats2half2_rn(exp2f(s[nt][2] - bB), exp2f(s[nt][3] - bB));
      const float2 fa = __half22float2(pa), fb = __half22float2(pb);
      sumA += fa.x; sumA += fa.y; sumB += fb.x; sumB += fb.y;
      *reinterpret_cast<__half2*>(prow + nt * 16) = pa;
      *reinterpret_cast<__half2*>(prow + 8 * C::PS + nt * 16) = pb;
    }
    lA = lA * alA + sumA;
    lB = lB * alB + sumB;
#pragma unroll
    for (int G = 0; G < C::NG; ++G)
#pragma unroll
      for (int n = 0; n < 4; ++n) { o[G][n][0] *= alA; o[G][n][1] *= alA; o[G][n][2] *= alB; o[G][n][3] *= alB; }
    cp_async_wait<2 * ST - 2>();
    __syncthreads();      // P visible, V(i) landed, K stage free
    load(i + ST, false);
    // ---- O += P · V over this warp's columns
#pragma unroll
    for (int kc = 0; kc < KT / 16; ++kc) {
      uint32_t pa[4];
      ldmatrix_x4(pa, sP + (rg * 16 + (lane & 15)) * C::PS + (kc * 16 + (lane >> 4) * 8) * 2);
      const uint8_t* vrow = sV + (size_t)stage * KT * C::VS + (kc * 16 + 2 * c) * C::VS + dbase + 4 * gq;
#pragma unroll
      for (int G = 0; G < C::NG; ++G) {
        const uint32_t w0 = *reinterpret_cast<const uint32_t*>(vrow + G * 32);
        const uint32_t w1 = *reinterpret_cast<const uint32_t*>(vrow + C::VS + G * 32);
        const uint32_t w2 = *reinterpret_cast<const uint32_t*>(vrow + 8 * C::VS + G * 32);
        const uint32_t w3 = *reinterpret_cast<const uint32_t*>(vrow + 9 * C::VS + G * 32);
        uint32_t b[4][2];
        e4m3x4_to_f16x4(__byte_perm(w0, w1, 0x5140), b[0][0], b[1][0]);
        e4m3x4_to_f16x4(__byte_perm(w0, w1, 0x7362), b[2][0], b[3][0]);
        e4m3x4_to_f16x4(__byte_perm(w2, w3, 0x5140), b[0][1], b[1][1]);
        e4m3x4_to_f16x4(__byte_perm(w2, w3, 0x7362), b[2][1], b[3][1]);
#pragma unroll
        for (int n = 0; n < 4; ++n) mma_f16(o[G][n], pa[0], pa[1], pa[2], pa[3], b[n][0], b[n][1]);
      }
    }
    __syncthreads();      // V stage and P tile free
    load(i + ST, true);
  }

  // ---- row sums over the quad, then over the DS warps (fixed order)
  lA += __shfl_xor_sync(0xffffffffu, lA, 1); lA += __shfl_xor_sync(0xffffffffu, lA, 2);
  lB += __shfl_xor_sync(0xffffffffu, lB, 1); lB += __shfl_xor_sync(0xffffffffu, lB, 2);
  cp_async_wait<0>();
  if (c == 0) { lred[(rg * DS + ds) * 16 + gq] = lA; lred[(rg * DS + ds) * 16 + gq + 8] = lB; }
  __syncthreads();
  float tA = 0.f, tB = 0.f;
#pragma unroll
  for (int d2 = 0; d2 < DS; ++d2) { tA += lred[(rg * DS + d2) * 16 + gq]; tB += lred[(rg * DS + d2) * 16 + gq + 8]; }

  if constexpr (SPLIT == 1) {
    const float iA = tA > 0.f ? 1.f / tA : 0.f, iB = tB > 0.f ? 1.f / tB : 0.f;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = r0 + rowA + half * 8;
      if (r >= rows) continue;
      const float inv = half ? iB : iA;
      bf16* orow = a.out + ((size_t)(sq.q_row0 + r / ratio) * a.H + g * ratio + r % ratio) * D + dbase + 8 * c;
#pragma unroll
      for (int G = 0; G < C::NG; ++G) {
        const int e0 = half * 2, e1 = half * 2 + 1;
        uint4 w;
        w.x = pack_bf16x2(o[G][0][e0] * inv, o[G][1][e0] * inv);
        w.y = pack_bf16x2(o[G][2][e0] * inv, o[G][3][e0] * inv);
        w.z = pack_bf16x2(o[G][0][e1] * inv, o[G][1][e1] * inv);
        w.w = pack_bf16x2(o[G][2][e1] * inv, o[G][3][e1] * inv);
        *reinterpret_cast<uint4*>(orow + G * 32) = w;
      }
    }
  } else {
    // ---- split: each block writes its partial (unnormalized fp32 O [16][D], row max, row sum) to its cluster's scratch slot; after the
    //   cluster barrier (release / acquire at cluster scope) block `rank` combines output columns [32 rank, 32 rank + 32) over the
    //   SPLIT partials in fixed rank order.
    float* slot = g_split_ws + (size_t)(blockIdx.z * gridDim.y + blockIdx.y) * C::SLOT;
    float* Wo = slot + (size_t)rank * 16 * D;            // [SPLIT][16][D]
    float* Wm = slot + (size_t)kSplit * 16 * D;          // [SPLIT][16]
    float* Wl = Wm + kSplit * 16;
#pragma unroll
    for (int G = 0; G < C::NG; ++G) {
      float* d0 = Wo + rowA * D + dbase + G * 32 + 8 * c;
      __stcg(reinterpret_cast<float4*>(d0), make_float4(o[G][0][0], o[G][1][0], o[G][2][0], o[G][3][0]));
      __stcg(reinterpret_cast<float4*>(d0 + 4), make_float4(o[G][0][1], o[G][1][1], o[G][2][1], o[G][3][1]));
      __stcg(reinterpret_cast<float4*>(d0 + 8 * D), make_float4(o[G][0][2], o[G][1][2], o[G][2][2], o[G][3][2]));
      __stcg(reinterpret_cast<float4*>(d0 + 8 * D + 4), make_float4(o[G][0][3], o[G][1][3], o[G][2][3], o[G][3][3]));
    }
    if (ds == 0 && c == 0) {
      __stcg(Wm + rank * 16 + gq, mA); __stcg(Wm + rank * 16 + gq + 8, mB);
      __stcg(Wl + rank * 16 + gq, tA); __stcg(Wl + rank * 16 + gq + 8, tB);
    }
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    float* wts = reinterpret_cast<float*>(sP);   // [16][kSplit + 1]: per-rank weights, 1 / L
    if (tid < 16) {
      float mk[kSplit], lk[kSplit], mx = -INFINITY;
#pragma unroll
      for (int k = 0; k < kSplit; ++k) { mk[k] = __ldcg(Wm + k * 16 + tid); lk[k] = __ldcg(Wl + k * 16 + tid); }
#pragma unroll
      for (int k = 0; k < kSplit; ++k) mx = fmaxf(mx, mk[k]);
      float L = 0.f;
#pragma unroll
      for (int k = 0; k < kSplit; ++k) {
        const float w = mk[k] == -INFINITY ? 0.f : exp2f(mk[k] - mx);
        wts[tid * (kSplit + 1) + k] = w;
        L += w * lk[k];
      }
      wts[tid * (kSplit + 1) + kSplit] = L > 0.f ? 1.f / L : 0.f;
    }
    __syncthreads();
    for (int x = tid; x < 16 * 8; x += C::NT) {   // 16 rows × 8 float4 columns
      const int row = x / 8, col = rank * 32 + 4 * (x % 8), r = r0 + row;
      if (r >= rows) continue;
      float4 v[kSplit];
#pragma unroll
      for (int k = 0; k < kSplit; ++k) v[k] = __ldcg(reinterpret_cast<const float4*>(slot + ((size_t)k * 16 + row) * D + col));
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int k = 0; k < kSplit; ++k) {
        const float w = wts[row * (kSplit + 1) + k];
        acc.x += w * v[k].x; acc.y += w * v[k].y; acc.z += w * v[k].z; acc.w += w * v[k].w;
      }
      const float inv = wts[row * (kSplit + 1) + kSplit];
      uint2 ov;
      ov.x = pack_bf16x2(acc.x * inv, acc.y * inv);
      ov.y = pack_bf16x2(acc.z * inv, acc.w * inv);
      *reinterpret_cast<uint2*>(a.out + ((size_t)(sq.q_row0 + r / ratio) * a.H + g * ratio + r % ratio) * D + col) = ov;
    }
  }
}

template <int D, int M, int DS, int QDS, int KT, int ST, int SPLIT>
void launch(const Args& a, dim3 grid, cudaStream_t st) {
  using C = Cfg<D, M, DS, QDS, KT, ST, SPLIT>;
  auto kern = attn_kernel<D, M, DS, QDS, KT, ST, SPLIT>;
  static bool init = [&] {
    CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)C::SMEM));
    if (SPLIT > 8) CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    return true;
  }();
  (void)init;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(C::NT);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = st;
  cudaLaunchAttribute attr[2];
  int na = 0;
  if (SPLIT > 1) {
    attr[na].id = cudaLaunchAttributeClusterDimension;
    attr[na].val.clusterDim.x = SPLIT;
    attr[na].val.clusterDim.y = 1;
    attr[na].val.clusterDim.z = 1;
    ++na;
  }
  if (!pdl_off()) {
    attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = attr;
  cfg.numAttrs = na;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, a));
}

// one thread = 16 values of one K or V row of one kv head
__global__ void kv_write_kernel(const bf16* __restrict__ k, const bf16* __restrict__ v, int rows, const int32_t* __restrict__ seq_of,
                                const int32_t* __restrict__ pos, const AttnSeq* __restrict__ seqs, int Hkv, int D, size_t tok, size_t koff,
                                size_t voff) {
  pdl_trigger();
  pdl_wait();
  const int per_row = Hkv * D / 16;
  const long long n = (long long)rows * per_row * 2;
  for (long long x = blockIdx.x * (long long)blockDim.x + threadIdx.x; x < n; x += (long long)gridDim.x * blockDim.x) {
    const int isv = (int)(x / ((long long)rows * per_row));
    const long long y = x % ((long long)rows * per_row);
    const int r = (int)(y / per_row), ch = (int)(y % per_row);
    const int p = pos[r];
    const uint8_t* pg = seqs[seq_of[r]].pages[p / kPage];
    uint8_t* dst = const_cast<uint8_t*>(pg) + (size_t)(p % kPage) * tok + (isv ? voff : koff) + (size_t)ch * 16;
    const uint4* src = reinterpret_cast<const uint4*>((isv ? v : k) + (size_t)r * Hkv * D + (size_t)ch * 16);
    const uint4 s0 = src[0], s1 = src[1];
    const uint32_t in[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
    uint32_t outw[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f0 = make_float2(__uint_as_float(in[2 * i] << 16), __uint_as_float(in[2 * i] & 0xFFFF0000u));
      const float2 f1 = make_float2(__uint_as_float(in[2 * i + 1] << 16), __uint_as_float(in[2 * i + 1] & 0xFFFF0000u));
      const uint32_t lo = __nv_cvt_float2_to_fp8x2(f0, __NV_SATFINITE, __NV_E4M3);
      const uint32_t hi = __nv_cvt_float2_to_fp8x2(f1, __NV_SATFINITE, __NV_E4M3);
      outw[i] = (lo & 0xFFFFu) | (hi << 16);
    }
    *reinterpret_cast<uint4*>(dst) = make_uint4(outw[0], outw[1], outw[2], outw[3]);
  }
}

}  // namespace

void attn_paged(const bf16* q, bf16* out, int H, const KvLayout& L, int layer, const AttnSeq* seqs, int n_seqs, int max_q_per_seq, int window,
                cudaStream_t st) {
  HIVE_CHECK(L.D == 256 || L.D == 512, "attn_paged: D must be 256 or 512");
  HIVE_CHECK(L.Hkv > 0 && H % L.Hkv == 0, "attn_paged: H must be a multiple of Hkv");
  HIVE_CHECK(layer >= 0 && layer < L.n_layers, "attn_paged: layer out of range");
  if (n_seqs <= 0 || max_q_per_seq <= 0) return;
  Args a;
  a.q = q; a.out = out; a.seqs = seqs;
  a.H = H; a.Hkv = L.Hkv; a.ratio = H / L.Hkv; a.window = window;
  a.tok = L.token_bytes();
  a.koff = L.k_off(0, layer);
  a.voff = L.v_off(0, layer);
  const int dq = max_q_per_seq < kDecodeMaxQ ? max_q_per_seq : kDecodeMaxQ;
  const int mt = (dq * a.ratio + 15) / 16;
  // decode / verify path in waves of whole sequences whose scratch slots fit
  auto split_waves = [&](auto launch_fn, size_t slot) {
    const int per_wave = (int)(kWsFloats / slot / ((size_t)mt * L.Hkv));
    HIVE_CHECK(per_wave >= 1, "attn_paged: split scratch too small for one sequence");
    for (int s0 = 0; s0 < n_seqs; s0 += per_wave) {
      Args w = a;
      w.seqs = seqs + s0;
      const int ns = n_seqs - s0 < per_wave ? n_seqs - s0 : per_wave;
      launch_fn(w, ns);
    }
  };
  if (L.D == 256) {
    split_waves([&](const Args& w, int ns) { launch<256, 16, 8, 8, 64, 2, kSplitSliding>(w, dim3(kSplitSliding, mt, ns * L.Hkv), st); },
                Cfg<256, 16, 8, 8, 64, 2, kSplitSliding>::SLOT);
    if (max_q_per_seq > kDecodeMaxQ) launch<256, 64, 2, 2, 64, 1, 1>(a, dim3((max_q_per_seq * a.ratio + 63) / 64, L.Hkv, n_seqs), st);
  } else {
    split_waves([&](const Args& w, int ns) { launch<512, 16, 8, 8, 64, 1, kSplitFull>(w, dim3(kSplitFull, mt, ns * L.Hkv), st); },
                Cfg<512, 16, 8, 8, 64, 1, kSplitFull>::SLOT);
    if (max_q_per_seq > kDecodeMaxQ) launch<512, 32, 4, 4, 32, 1, 1>(a, dim3((max_q_per_seq * a.ratio + 31) / 32, L.Hkv, n_seqs), st);
  }
}

void kv_write(const bf16* k, const bf16* v, int rows, const int32_t* seq_of, const int32_t* pos, const AttnSeq* seqs, const KvLayout& L, int layer,
              cudaStream_t st) {
  HIVE_CHECK(L.D % 16 == 0, "kv_write: D must be a multiple of 16");
  if (rows <= 0) return;
  const long long n = (long long)rows * L.Hkv * L.D / 16 * 2;
  const int blocks = (int)((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096);
  launch_pdl(kv_write_kernel, dim3(blocks), dim3(256), 0, st, k, v, rows, seq_of, pos, seqs, L.Hkv, L.D, L.token_bytes(), L.k_off(0, layer), L.v_off(0, layer));
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hive::gemma::k
