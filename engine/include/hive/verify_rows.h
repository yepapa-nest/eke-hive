// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// R1 HIVE_MTP_VERIFY2 / HIVE_MTP_BATCH — formulas of the verification row tables (shared by host and device — defined in one place only).
//   verify_decode.cu (window_idxs_verify kernel) and runtime.cpp (attention_verify_host) call these functions directly, and the
//   CPU test engine/tests/test_verify_rows_cpu.cpp uses the same functions to check "verification row = sequential decode at
//   that position" against a toy state (ring, compressor, compressed cache).
#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define HIVE_VR_HD __host__ __device__ __forceinline__
#else
#define HIVE_VR_HD inline
#endif

namespace hive::vrows {

// Window column j ∈ [0, win) of a row (position p; p0 = first position of its part, g = first row number): source position src = p − (win−1) + j.
//   src < 0 → −1 · src < p0 → ring slot src % win (position before the verification — the ring is written only after attention)
//   · otherwise → chunk row win + g + (src − p0) (a kv row of this call).
//   = the value of prefill window_idxs (start_pos = p0, min_src 0) with only the chunk columns shifted by g · for a one-row part
//   (p = p0, g = m) it equals window_idxs_rows.
//   Precondition: rows per part ≤ win (a ring slot must not be written twice within one verification, so that rollback's slot
//   restore does not overwrite accepted rows) — verification rows ≤ 8 < win 128.
HIVE_VR_HD int32_t window_col(int64_t p, int64_t p0, int g, int win, int j) {
  const int64_t src = p - (win - 1) + j;
  if (src < 0) return -1;
  if (src < p0) return (int32_t)(src % win);
  return (int32_t)(win + g + (src - p0));
}

// Compressed table of a row (position p) — exactly the formula attention_decode_host uses when seqs[m]->pos = p (only layers with ratio > 0).
struct Tab {
  int comp_len;  // visible compressed groups = (p+1)/ratio (including the group this row completes)
  int visible;   // indexer visible (same value)
  int trows;     // indexer key count (same value)
  int gpos;      // kv source layer: first position of the group (RoPE) = p + 1 − ratio
  int dsti;      // kv source layer: compressed cache row = p / ratio (written only by the row that completes the group — valid = (p+1) % ratio == 0)
  bool valid;
};
HIVE_VR_HD Tab tab(int64_t p, int ratio) {
  Tab t;
  t.comp_len = (int)((p + 1) / ratio);
  t.visible = t.comp_len;
  t.trows = t.comp_len;
  t.gpos = (int)(p + 1 - ratio);
  t.dsti = (int)(p / ratio);
  t.valid = ((p + 1) % ratio) == 0;
  return t;
}

}  // namespace hive::vrows
