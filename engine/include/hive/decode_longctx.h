// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Long-context decode — new versions of the two decode indexer stages whose cost grows with context length (scores and top-k).
// Both are default-off switches (env_on).
//
// Decode kernels proportional to context (M <= 8 fused front end dec_attention_front step 4, per layer — T = number of compressed rows):
//   - indexer scores, dec_idx_scores_kernel: layers 2/8/14 (ratio 2, T = L/2) and 20 (ratio 1, T = L) score every row; 24/28/32/36 score the
//     candidate pool (min(2048, ceil(L/8))*8 <= 16,384 columns — indexer_scores_cand_tc when HIVE_IDX_TC is on). One key = Hi 32 x Di 128 =
//     4,096 fmaf + 80 B. Blocks are launched up to the graph bucket (Tb = max(1024, next_pow2(T))).
//   - top-k, dec_topk_kernel: **one CTA (one SM)** per row scans all of T with a 2-pass (bf16) / 4-pass (f32 block top-k) 8-bit histogram
//     plus 3 contiguous-range passes. The histogram uses smem atomics — scores are positive with a narrow exponent range, so in the first
//     pass the 32 lanes of a warp hit nearly the same bin and serialize. The bin search (256 bins) runs serially on thread 0.
//   - everything else (q, kv, compressor, window, sparse attention over win+512 columns, wo) is independent of context (constant).
//
// (1) HIVE_DECODE_TOPK2 — split top-k (bit-identical: same selected set, same output order, same -1 padding).
//   A row is cut into chunks of CH (= 1024*2^j >= max(4096, 2k, sqrt(T_in*k))) columns. One CTA per chunk writes the **top min(k, chunk
//   length)** of its chunk (keys are kept even when -inf) as (key, original column) in ascending order into a union buffer; the last CTA of
//   the row (threadFenceReduction counter) then makes the final selection from the union U = (nc-1)*k + min(k, last chunk) with the same
//   rule as the single-stage kernel. Correctness: every element of the global top keff is also in its chunk's top k (fewer than k elements
//   precede it globally, hence fewer than k in its chunk) -> union contains the global selection; the union is laid out chunk order x
//   ascending within the chunk = ascending original column, so the "ties go to the smaller index" rule is preserved; keff = min(k, T) = min(k, U).
//   The per-chunk and final selections use the same radix selection as the single-stage kernel (same thr, want, prefix-sum layout). What
//   differs: histogram atomics are warp-aggregated (__match_any_sync adds the lane count of a bin at once — integer sums, so values are
//   identical); the bin search runs in parallel on warp 0 (8 bins per lane + warp prefix sum — same formula, same bin); chunks are short, so
//   the range passes stay in L1.
//   If T <= CH there is one chunk and its CTA writes the final output directly (no union). The union buffer is caller-provided scratch
//   (in production: Work::iscore_f — unused on the fused path); the counter is DecAttnArgs::attn_cnt[row] (starts at 0, the kernel resets it
//   to 0 — separated from the following sparse attention by stream order). If the buffer is too small, the single-stage kernel is used (absorbed).
// (2) HIVE_DECODE_IDXSCORE2 — new indexer scores (bit-identical: per key, per head an ascending-d fmaf chain, bf16 rounding, relu, x w, and
//   an ascending-h sum — the same statements as idx_score32).
//   What differs: the transposed q load uses lane = head (lane = dimension caused 32-way smem bank conflicts, 4,096 wavefronts per block);
//   KPT keys per thread (one q smem load shared by KPT keys); if a whole block lies outside the visible columns (graph bucket tail) it writes
//   only -inf without loading q (the same value the old kernel writes).
//   HIVE_DECODE_IDXSCORE2_KPT = 1, 2 or 4 (default 2; any other value -> 2) — keys per thread (performance tuning only, no effect on values).
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "hive/common.h"

namespace hive::k {

struct DecRow;

// Switches (env_on: unset/""/"0" = off) — decode_longctx.cu
bool decode_topk2_on();       // HIVE_DECODE_TOPK2
bool decode_idxscore2_on();   // HIVE_DECODE_IDXSCORE2
int decode_idxscore2_kpt();   // HIVE_DECODE_IDXSCORE2_KPT (1/2/4, default 2)

// Chunk plan (host — the CPU test uses the same function). T_in = number of columns (the bucket under graphs — fixed with the graph key),
// k = selection width, M = number of rows.
struct Topk2Plan {
  int CH = 0;         // chunk length (columns)
  int ncmax = 1;      // max chunks per row = ceil(T_in / CH) (1 = no union)
  int ustride = 0;    // union slots per row = ncmax*k
  size_t idx_off = 0; // start of the union index array inside scratch (bytes, 16-aligned)
  size_t bytes = 0;   // scratch bytes required (0 when ncmax is 1)
};
inline Topk2Plan topk2_plan(int T_in, int k, int M, size_t key_bytes) {
  Topk2Plan p;
  const double bal = std::sqrt((double)(T_in > 0 ? T_in : 0) * (double)(k > 0 ? k : 0));
  int CH = 4096;
  while ((CH < 2 * k || (double)CH < bal) && CH < (1 << 24)) CH <<= 1;
  p.CH = CH;
  p.ncmax = T_in > CH ? (T_in + CH - 1) / CH : 1;
  if (p.ncmax > 1) {
    p.ustride = p.ncmax * k;
    const size_t keys = (size_t)M * p.ustride * key_bytes;
    p.idx_off = (keys + 15) / 16 * 16;
    p.bytes = p.idx_off + (size_t)M * p.ustride * 4;
  }
  return p;
}

// (1) Split top-k — argument and output contract identical to dec_topk_bf16/f32 (attn_decode_fused.h). cnt = per-row int counters [M] (zeroed),
//   scratch = union buffer. Returns false = shape/buffer not accepted (nothing was launched — the caller uses the single-stage kernel).
bool dec_topk2_bf16(const bf16* keys, int M, int T_in, bool use_trows, const DecRow* tab, int k, int row_stride, int32_t* out, int out_stride,
                    const int32_t* cand, int cand_stride, int bs, int offset, bool map_offset, void* scratch, size_t scratch_bytes, int* cnt,
                    cudaStream_t st);
bool dec_topk2_f32(const float* keys, int M, int T_in, bool use_trows, const DecRow* tab, int k, int row_stride, int32_t* out, int out_stride,
                   const int32_t* cand, int cand_stride, int bs, int offset, bool map_offset, void* scratch, size_t scratch_bytes, int* cnt,
                   cudaStream_t st);
// (2) New indexer scores (Hi 32, Di 128) — same arguments and same output as dec_idx_scores_kernel<CAND> (attn_decode_fused.cu). kpt <= 0 = switch value.
//   Returns false = not accepted (e.g. misaligned q — the caller uses the old kernel).
bool dec_idx_scores2(const bf16* q, const bf16* w, const DecRow* tab, int M, int ncols, bool cand_mode, const int32_t* cand, int cand_stride, int bs,
                     bf16* score, int kpt, cudaStream_t st);

}  // namespace hive::k
