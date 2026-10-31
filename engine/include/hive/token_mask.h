// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Answer-span token penalty (FN "hanja filter", docs/qwen.md): a request may ask that a fixed set of vocabulary tokens (the runtime's list —
//   HIVE_QWEN_TOKEN_MASK, tools/build_cjk_mask.py: multi-ideograph and kana tokens) is penalized in the answer span, i.e. after </think> (the
//   whole output of a non-thinking request): their logits are lowered by the request's penalty Δ (∞ = never sampled). The sampled
//   distribution is the model's own with those logits shifted, under the request's temperature / top_k / top_p / min_p: the shift is applied
//   before the candidate selection (the runtime, on the device) or before the host sampler (rows the device did not touch — the first token
//   after a prefill). Speculative verification samples each verify row from its shifted distribution and accepts a draft only when it equals
//   that sample, so the output distribution is the shifted one with or without MTP; the drafts themselves are not shifted.
// Host side only — no CUDA (tests/test_token_mask_cpu.cpp).
#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "hive/sampler.h"

namespace hive::tmask {

// Per-request state: whether the request asked for the filter and whether its next token lies inside a thinking span.
struct State {
  bool on = false;         // the request asked for the filter (and the runtime has a list)
  float penalty = INFINITY;   // Δ subtracted from the listed logits (∞ = removed)
  bool in_think = false;   // inside <think> … </think>
  int32_t end = -1, start = -1;   // </think> and <think> ids (start −1: a later <think> does not reopen the span)
  void feed(int32_t t) {
    if (!on) return;
    if (in_think) { if (t == end) in_think = false; }
    else if (start >= 0 && t == start) in_think = true;
  }
  bool active() const { return on && !in_think; }
  float row() const { return active() ? penalty : 0.f; }   // the row's shift (0 = untouched)
};

// Row shifts of a verify call: row 0 predicts the token after `inputs[0]` (already fed into s), row i the token after inputs[i] —
//   the state after feeding inputs[1..i]. Appends inputs.size() values (0 = untouched).
inline void verify_flags(State s, const int32_t* inputs, int n, std::vector<float>& out) {
  for (int i = 0; i < n; ++i) {
    if (i > 0) s.feed(inputs[i]);
    out.push_back(s.row());
  }
}

// Host path: the request's rule over the full logit row with the listed logits lowered by `penalty` (∞ → −inf; the row is copied).
inline int32_t sample_masked(const std::vector<float>& row, const std::vector<int32_t>& masked, float penalty, float temperature, float top_p,
                             int top_k, float min_p, std::mt19937_64& rng) {
  static thread_local std::vector<float> l;
  l = row;
  const int V = (int)l.size();
  for (int32_t t : masked) if (t >= 0 && t < V) l[t] = std::isinf(penalty) ? -INFINITY : l[t] - penalty;
  return sampling::sample(l, temperature, top_p, top_k, min_p, rng);
}

}  // namespace hive::tmask
