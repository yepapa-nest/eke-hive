// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_MTP_VERIFY2 / HIVE_MTP_BATCH — three kernels (verify_decode.cu) for running DSpark verify rows through the decode path (decode_layer: graph front end, decode attention kernels, moe_decode_experts/F2).
//   The decode path assumes every row is a **different** sequence, so three places go wrong on consecutive rows of the same sequence (verify [tok, d1..dk]);
//   only those three are replaced by these kernels (runtime.cpp attention_verify_dev):
//   ① window indices: decode (window_idxs_rows, idx_out of fused_kv_proj) treats every earlier position of row m as a ring slot — the earlier verify rows of the same sequence
//      (positions p0..p0+m−1) are not in the ring yet (and writing the ring first would let later rows overwrite the oldest slot of row m's window — a future leak, the same defect
//      as the one guarded against in runtime.cpp attention()). window_idxs_verify = the same values as prefill window_idxs (start_pos = first position of that row group): before the group's first position
//      = ring slot (src % win) · inside the group = chunk row (win + group first row + (src − p0)) · src < 0 = −1. Positions and group first rows are read from device (mapped) tables (graph-capturable).
//      A one-row group (ordinary decode row) gives the same values as window_idxs_rows (self = win + m).
//   ② compressor (ratio > 1 kv source layers): compressor_step_rows uses block = row, so rows sharing a state race each other. compressor_step_seq uses thread = dimension d
//      and walks rows m = 0..M−1 **in order** (each row with its own state pointer) — per-row formula and operation order match compressor_step_rows (= bit-identical to calling the 1-row
//      decode step M times; test_verify_decode).
//   ③ ring write: ring_write_rows **after** attention (called by runtime.cpp — no extra kernel).
//   Compressor state restore on rollback: compressor_apply_rows = after restoring the snapshot, rewrite the accepted rows' (kv, score) into their slots — the very values used during verification,
//   so the state is bit-identical to having decoded only the accepted rows (re-projecting the accepted rows with cuBLAS could change low bits depending on the row count).
#pragma once
#include <cstdint>

#include "hive/common.h"

namespace hive::k {

// out[m·stride + j], j ∈ [0, win): window columns of row m. pos[m] = position, grp[m] = first row index of that row's group (group = consecutive rows of the same sequence).
void window_idxs_verify(int M, int win, const int32_t* pos, const int32_t* grp, int32_t* out, int out_stride, cudaStream_t st);
// same arguments and outputs as compressor_step_rows (out[m] only for rows that complete a group, valid[m]) — rows in order.
void compressor_step_seq(const float* kv, const float* score, int M, int D, int ratio, const int32_t* pos, float* const* state_kv,
                         float* const* state_score, float* out, uint8_t* valid, cudaStream_t st);
// state only: kv/score of row i (position pos0+i) into slot (pos0+i) % ratio (in row order — the last row wins). kv/score are contiguous [n, D].
void compressor_apply_rows(const float* kv, const float* score, int n, int D, int ratio, int64_t pos0, float* state_kv, float* state_score, cudaStream_t st);

// HIVE_MTP_VERIFY_ROWIND: acc[rows[r], :] += src[r, :] in order of r (thread = dimension d, no atomics — result independent of the run and of other rows). Same arguments as accum_f32_rows.
void accum_f32_rows_ordered(const float* src, const int32_t* rows, int R, int dim, float* acc, cudaStream_t st);

}  // namespace hive::k
