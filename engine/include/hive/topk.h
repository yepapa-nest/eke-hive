// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Per-row top-k (CUB segmented radix sort) — indexer topk 512 · candidate block topk 2048.
// v1: correctness first (full sort). Decode (M=1, large T) is to be replaced by a single-block radix-select in v2.
#pragma once
#include <cstdint>

#include "hive/common.h"
#include "hive/model.h"

namespace hive {

class TopK {
 public:
  // Reserve workspace for up to M rows × T columns
  void reserve(int M, int T);
  // keys [M,T] fp32 (−inf allowed) → indices of each row's top k in ascending order into out[M,k] (if k exceeds the number of valid
  //   entries the remainder holds indices from the tail of the sort, i.e. invalid values, so the caller filters by visible)
  void topk_rows(const float* keys, int M, int T, int k, int32_t* out, cudaStream_t st);

 private:
  DevBuf idx_in_, idx_out_, keys_out_, offsets_, tmp_, sel_keys_, sel_vals_, sel_out_;
  size_t tmp_bytes_ = 0;
  int capM_ = 0, capT_ = 0;
};

}  // namespace hive
