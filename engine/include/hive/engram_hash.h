// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Engram n-gram hashing (host) — a direct port of NgramHashState from the reference engram.py. The compressed token
// map, primes and multipliers are read from engram_hash.json + token_map.bin (int32 [vocab]) exported by the oracle,
// so the tokenizer normalizer does not have to be ported to C++.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace hive {

class EngramHash {
 public:
  void load(const std::string& json_path, const std::string& token_map_path);
  int n_layers() const { return (int)layer_ids_.size(); }
  int n_cols() const { return (max_ngram_ - 1) * n_heads_; }
  int64_t pad_id() const { return pad_id_; }
  // ids[M] (raw token ids at positions start_pos..); compressed ids of earlier tokens come from history. out[M, n_layers, n_cols] int64.
  // history: all compressed ids of this sequence (M entries are appended by the call). dead (-1) = image token (n-grams do not cross it).
  void compute(const int32_t* ids, const int8_t* is_image, int M, std::vector<int64_t>& history, std::vector<int64_t>& out) const;
  // HIVE_ENGRAM_SSD prefetch: produces the same out as compute without modifying the history. tail = the last min(start, max_ngram−1)
  //   history entries (start = history length — earlier positions are out of n-gram reach). Same formula, same values (the CPU test test_engram_ssd_cpu checks it against compute).
  void peek(const int32_t* ids, const int8_t* is_image, int M, const int64_t* tail, int tail_n, int64_t start, std::vector<int64_t>& out) const;
  int max_ngram() const { return max_ngram_; }
  // Compressed id (the value stored in the history): image = -1
  int64_t compressed(int32_t id, bool image) const { return image ? -1 : (int64_t)token_map_[id]; }

 private:
  std::vector<int> layer_ids_;
  int max_ngram_ = 4, n_heads_ = 8;
  int64_t pad_id_ = 0;
  std::vector<std::vector<std::vector<int64_t>>> primes_;  // [layer][ngram-1][head]
  std::vector<std::vector<int64_t>> offsets_;               // [layer][col]
  std::vector<std::vector<int64_t>> multipliers_;           // [layer][max_ngram]
  std::vector<int32_t> token_map_;
};

}  // namespace hive
