// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_ATTN_FUSED — the decode (M <= 8) attention front (q, kv, compressor, indexer, sparse attention) in fewer launches.
//   Numeric contract: **bit-identical** to the baseline path (the fuse_ variant of runtime attention_decode_dev — fused_q_proj, fused_kv_proj, gemm_bs, rope_last, fp4_quant_roundtrip,
//   cuBLAS, compressor_step_rows, rmsnorm, fp4_pack_rows, indexer_scores_*, bf16_rows_to_f32, topk_select_rows(_t), cand_to_pos, offset_idxs,
//   sparse_attn_decode_rows). Same formulas and same reduction order; only launch boundaries, load placement and the row-table location change:
//   - The row table (mapped pinned — a PCIe round trip per block) is copied once into a device copy (DecRow) by the last kv block of the q_a||kv kernel; later kernels read the copy.
//   - top-k uses the same selection rule (value descending, ties to the smaller index, -inf excluded, ascending output, -1 padding) as an ordered scan — same set and order without sorting.
//   The cuBLAS calls (compressor projection, idx_wk, idx_weights_proj) are the same calls as before (the algorithm must match for bits to match).
#pragma once
#include <cstdint>

#include "hive/common.h"
#include "hive/model_kernels.h"

namespace hive {
class Blas;
}

namespace hive::k {

// Device row table (row = batched-decode sequence). One layer's copy of the mapped pinned tables (Work::*_h/_d).
struct DecRow {
  KvRow kv;                // ring, compressed cache (source layer), comp_len, topk
  const uint8_t* kptr;     // index-key cache (source layer, read)
  uint8_t* dstp;           // compressed KV cache (this layer, write — kv source layers only)
  uint8_t* dstk;           // index-key cache (this layer, write — index source layers only)
  float* skv;              // compressor state kv [ratio, D]
  float* ssc;              // compressor state score [ratio, D]
  int32_t pos, gpos, visible, trows, dsti, pad_[3];
};
// Column-array copy of the same table (rows <= 8) — passed to existing kernels that take array arguments (indexer_scores_cand_tc)
struct DecSoA {
  const uint8_t* kptr[8];
  int32_t trows[8];
  int32_t visible[8];
};
// Mapped pinned sources (device pointers — Work's *_d). Missing tables are nullptr (not copied).
struct DecRowSrc {
  const KvRow* kv;
  const uint8_t* const* kptr;
  uint8_t* const* dstp;
  uint8_t* const* dstk;
  float* const* skv;
  float* const* ssc;
  const int32_t *pos, *gpos, *visible, *trows, *dsti;
};

struct DecAttnArgs {
  // Shapes and layer kind
  int M = 0, dim = 0, H = 0, D = 0, rd = 0, win = 0, q_lora = 0, Hi = 0, Di = 0, index_topk = 0;
  int ratio = 0, layer = 0, cand_source_layer = -1, cand_block = 8, cand_topk_blocks = 0;
  bool kv_source = false, index_source = false, have_candidates = false;
  int idx_mode = 0;      // 0 = bf16 scores (default), 1 = HIVE_IDX_TC (candidate-pool layers 24/28/32/36 use packed queries + indexer_scores_cand_tc — same kernel as before)
  int Tmax = 0;          // indexer column count (graph bucket — clipped per row by trows)
  int splits = 1;        // attention split count (the caller passes it using the attn_splits_ rule)
  float eps = 0.f;
  // Weights
  const uint8_t *wqa = nullptr, *sqa = nullptr, *wqb = nullptr, *sqb = nullptr, *wkv = nullptr, *skv = nullptr;
  const bf16 *q_norm = nullptr, *kv_norm = nullptr;
  const float* sink = nullptr;
  const void* comp_wkv = nullptr;   // ratio>1: f32, ratio==1: bf16
  const float* comp_wgate = nullptr;
  const bf16* comp_norm = nullptr;
  const uint8_t *idx_wqb = nullptr, *idx_sqb = nullptr;
  const bf16 *idx_wproj = nullptr, *idx_wk = nullptr, *idx_k_norm = nullptr;
  const float2* freqs = nullptr;      // ratio ? rope_compress : rope_window
  const float2* freqs_idx = nullptr;  // rope_compress (indexer query)
  // Work buffers (Work)
  const bf16* xn = nullptr;
  float* xf = nullptr;
  bf16 *qr = nullptr, *qrn = nullptr, *q = nullptr, *kv = nullptr;
  uint8_t *qrq = nullptr, *qrs = nullptr;
  float *ckv = nullptr, *cscore = nullptr, *cout = nullptr;
  bf16* latent = nullptr;
  uint8_t* valid = nullptr;
  bf16 *ik = nullptr, *iq = nullptr, *iw = nullptr, *iscore = nullptr;
  uint8_t* iqp = nullptr;    // idx_mode 1: packed query [M*Hi, IDX_ROW]
  float* bmax = nullptr;
  int32_t* cand = nullptr;
  int32_t* idx = nullptr;
  int idx_stride = 0;
  bf16* o = nullptr;
  float *pacc = nullptr, *pm = nullptr, *ps = nullptr;
  int* counters = nullptr;   // 2 counters (last q_a block, last kv block) — zero-initialised; the kernel resets them to 0
  int* attn_cnt = nullptr;   // [M*H] split-merge counters — zero-initialised; the kernel resets them to 0
  DecRow* tab = nullptr;     // [M] device row table (filled by this function)
  DecSoA* soa = nullptr;     // column arrays of the same table (filled by this function)
  DecRowSrc src{};
  bf16* const* ring_ptrs = nullptr;  // mapped pinned row -> ring (write)
  // (decode_attn3.h): -1 = follow the switches (HIVE_DECODE_QKV3 -> bit 1, HIVE_DECODE_SPARSE3 -> bit 2); >= 0 = this bitmask (tests).
  //   v3_g = heads per block for SPARSE3 (0 = HIVE_DECODE_SPARSE3_G / auto)
  int v3 = -1;
  int v3_g = 0;
  // (hive/decode_longctx.h): HIVE_DECODE_TOPK2 union buffer (in production Work::iscore_f — unused on the fused path). Missing or too small -> baseline top-k.
  void* scratch = nullptr;
  size_t scratch_bytes = 0;
  // -1 = follow the switches (HIVE_DECODE_IDXSCORE2 -> bit 1, HIVE_DECODE_TOPK2 -> bit 2); >= 0 = this bitmask (tests)
  int lctx = -1;
  // HIVE_MTP_VERIFY2_FUSED — verify variant (rows = consecutive positions of the same sequence, several parts). vgrp != nullptr selects it (the same three sites as (1)-(3) in verify_decode.h):
  //   (1) window indices use the same formula as window_idxs_verify (vgrp[m] = first row of that row's part, positions from the row table) — inside the q_a||kv kernel tail (baseline and QKV3)
  //   (2) ratio>1 compressor = compressor_step_seq (the verify kernel as is — row order) -> the compression tail reads cout/valid instead of rewriting state
  //   (3) the ring is written not by q_a||kv but **after** sparse attention by ring_write_rows (same kernel as the verify path)
  //   vsave != nullptr (ratio>1 kv source layers) = keep compressor inputs for rollback: [0, M)*D <- ckv, from vsave_rows*D <- cscore (same layout as attention_verify_dev).
  //   nullptr (default) = plain decode (same launches and values as before).
  const int32_t* vgrp = nullptr;
  float* vsave = nullptr;
  size_t vsave_rows = 0;
};

// Whole front (q/kv -> [compressor] -> [indexer] -> sparse attention (+ inverse RoPE) -> [verify variant: ring write]). Returns whether this call built the candidate table (cand) (layer 20).
bool dec_attention_front(const DecAttnArgs& a, Blas* blas, cudaStream_t st);

// Individual calls for tests (test_decode_attn): the same kernels directly
void dec_topk_bf16(const bf16* keys, int M, int T_in, bool use_trows, const DecRow* tab, int k, int row_stride, int32_t* out, int out_stride,
                   const int32_t* cand, int cand_stride, int bs, int offset, bool map_offset, cudaStream_t st);
void dec_topk_f32(const float* keys, int M, int T_in, bool use_trows, const DecRow* tab, int k, int row_stride, int32_t* out, int out_stride,
                  const int32_t* cand, int cand_stride, int bs, int offset, bool map_offset, cudaStream_t st);
// Test only: a single front stage — stage 0 = q_a||kv (including row table, window indices, ring), 1 = sparse attention.
//   variant < 0 = baseline kernel, >= 0 = decode_attn3 variant (for stage 1, variant = G: 0 auto, 1/2/4/8). Returns whether the new variant accepted the shape (baseline = true)
bool dec_front_stage(const DecAttnArgs& a, int stage, int variant, cudaStream_t st);
// Test only (test_decode_longctx): just front step (4), the indexer (scores -> [block max, block top-k] -> top-k -> mapping) — the same function as in dec_attention_front.
//   a.lctx bits pick the new variants (-1 = switches). Returns whether the candidate table was built (layer 20). Does nothing unless a.Tmax > 0 and this is an index source layer.
bool dec_indexer_stage(const DecAttnArgs& a, cudaStream_t st);
//   One launch of the baseline score kernel dec_idx_scores_kernel<cand_mode> (test_decode_longctx times it next to the new dec_idx_scores2)
void dec_idx_scores_ref(const bf16* q, const bf16* w, const DecRow* tab, int M, int ncols, bool cand_mode, const int32_t* cand, int cand_stride, int bs,
                        bf16* score, cudaStream_t st);
// Fill only the row table (when a test compares top-k separately)
void dec_stage_rows(const DecRowSrc& src, int M, DecRow* tab, DecSoA* soa, cudaStream_t st);

}  // namespace hive::k
