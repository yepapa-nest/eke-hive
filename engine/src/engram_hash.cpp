// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#include "hive/engram_hash.h"

#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace hive {

void EngramHash::load(const std::string& json_path, const std::string& token_map_path) {
  std::ifstream f(json_path);
  if (!f) throw std::runtime_error("cannot open " + json_path);
  nlohmann::json j = nlohmann::json::parse(f);
  layer_ids_ = j["layer_ids"].get<std::vector<int>>();
  max_ngram_ = j["max_ngram"].get<int>();
  n_heads_ = j["n_heads"].get<int>();
  pad_id_ = j["pad_id"].get<int64_t>();
  primes_ = j["primes"].get<std::vector<std::vector<std::vector<int64_t>>>>();
  offsets_ = j["offsets"].get<std::vector<std::vector<int64_t>>>();
  multipliers_ = j["multipliers"].get<std::vector<std::vector<int64_t>>>();
  std::ifstream tm(token_map_path, std::ios::binary);
  if (!tm) throw std::runtime_error("cannot open " + token_map_path);
  tm.seekg(0, std::ios::end);
  size_t n = (size_t)tm.tellg() / 4;
  tm.seekg(0);
  token_map_.resize(n);
  tm.read(reinterpret_cast<char*>(token_map_.data()), n * 4);
}

void EngramHash::compute(const int32_t* ids, const int8_t* is_image, int M, std::vector<int64_t>& history,
                         std::vector<int64_t>& out) const {
  const int L = n_layers(), C = n_cols();
  const size_t start = history.size();
  for (int m = 0; m < M; ++m) history.push_back((is_image && is_image[m]) ? -1 : (int64_t)token_map_[ids[m]]);
  out.assign((size_t)M * L * C, 0);
  std::vector<int64_t> toks(max_ngram_);
  for (int m = 0; m < M; ++m) {
    const int64_t pos = (int64_t)start + m;
    bool blocked = false;
    for (int s = 0; s < max_ngram_; ++s) {
      int64_t src = pos - s < 0 ? history[0] : history[pos - s];
      blocked = blocked || (pos < s) || (src == -1);
      toks[s] = blocked ? pad_id_ : src;
    }
    for (int l = 0; l < L; ++l) {
      int64_t rolling = toks[0] * multipliers_[l][0];
      for (int i = 1; i < max_ngram_; ++i) {
        rolling ^= toks[i] * multipliers_[l][i];
        for (int h = 0; h < n_heads_; ++h) {
          int col = (i - 1) * n_heads_ + h;
          int64_t p = primes_[l][i - 1][h];
          int64_t r = rolling % p;            // torch: the result takes the sign of the (positive) divisor (Python %)
          if (r < 0) r += p;
          out[((size_t)m * L + l) * C + col] = r + offsets_[l][col];
        }
      }
    }
  }
}

void EngramHash::peek(const int32_t* ids, const int8_t* is_image, int M, const int64_t* tail, int tail_n, int64_t start,
                      std::vector<int64_t>& out) const {
  const int L = n_layers(), C = n_cols();
  // Virtual history = [tail (positions start − tail_n ..)] + the M new entries. Returns the same value as compute's history[q] for q ≥ start − tail_n.
  std::vector<int64_t> local((size_t)tail_n + M);
  for (int i = 0; i < tail_n; ++i) local[i] = tail[i];
  for (int m = 0; m < M; ++m) local[(size_t)tail_n + m] = (is_image && is_image[m]) ? -1 : (int64_t)token_map_[ids[m]];
  const int64_t base = start - tail_n;
  out.assign((size_t)M * L * C, 0);
  std::vector<int64_t> toks(max_ngram_);
  for (int m = 0; m < M; ++m) {
    const int64_t pos = start + m;
    bool blocked = false;
    for (int s = 0; s < max_ngram_; ++s) {
      // compute: src = pos−s < 0 ? history[0] : history[pos−s] — for pos < s the entry is blocked, so src is unused. A position with pos−s ≥ 0 outside the tail (< base)
      // would be farther than max_ngram−1 — this cannot happen (tail_n = min(start, max_ngram−1)).
      const int64_t q = pos - s;
      const int64_t src = (q < 0 || q < base) ? 0 : local[(size_t)(q - base)];
      blocked = blocked || (pos < s) || (q >= base && q >= 0 && src == -1);
      toks[s] = blocked ? pad_id_ : src;
    }
    for (int l = 0; l < L; ++l) {
      int64_t rolling = toks[0] * multipliers_[l][0];
      for (int i = 1; i < max_ngram_; ++i) {
        rolling ^= toks[i] * multipliers_[l][i];
        for (int h = 0; h < n_heads_; ++h) {
          int col = (i - 1) * n_heads_ + h;
          int64_t p = primes_[l][i - 1][h];
          int64_t r = rolling % p;
          if (r < 0) r += p;
          out[((size_t)m * L + l) * C + col] = r + offsets_[l][col];
        }
      }
    }
  }
}

}  // namespace hive
