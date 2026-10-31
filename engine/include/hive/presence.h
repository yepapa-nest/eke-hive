// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// presence_penalty (OpenAI semantics): the logit of every token that already appears in this response is lowered by the penalty (raised when
//   it is negative) before the request's temperature / top_k / top_p / min_p rule. Applied by hived at every row it samples — the step's
//   row and each speculative verify row (the drafts accepted before a row are emitted before that row is sampled, so they count) — so the
//   output distribution is the penalized one with or without MTP. Host side, on the GPU's candidate snapshot (values shifted, the row's Σexp
//   corrected with the shifted tokens' own logits) and on the full row when the request has it; exact: the candidates are the row's top NC,
//   a penalty only moves the listed tokens, and the paths that could need a token outside the candidates (a negative penalty lifting one
//   above the smallest candidate, or a candidate set with no greedy winner left) take the full row.
// No CUDA (tests/test_presence_cpu.cpp).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "hive/sampler.h"

namespace hive::presence {

struct State {
  float pp = 0.f;                  // the request's presence_penalty (0 = off)
  std::vector<uint8_t> flag;       // [vocab] token already emitted
  std::vector<int32_t> seen;       // the emitted tokens, each once
  void feed(int32_t t, int V) {
    if (pp == 0.f || t < 0 || t >= V) return;
    if (flag.empty()) flag.assign((size_t)V, 0);
    if (!flag[t]) { flag[t] = 1; seen.push_back(t); }
  }
  bool active() const { return pp != 0.f && !seen.empty(); }
};

// The request's rule over the penalized row. c: the GPU candidates (may be invalid); full: the full row (may be empty for greedy rows —
//   then the candidates decide). full is modified and restored in place.
inline int32_t sample(const sampling::Cands& c, std::vector<float>& full, const State& P, float temperature, float top_p, int top_k, float min_p,
                      std::mt19937_64& rng) {
  const int V = (int)full.size();
  // shift the full row (restored below, bit for bit)
  std::vector<float> keep;
  if (V > 0) {
    keep.reserve(P.seen.size());
    for (int32_t t : P.seen) { keep.push_back(full[t]); full[t] -= P.pp; }
  }
  auto restore = [&] { if (V > 0) for (size_t i = 0; i < P.seen.size(); ++i) full[P.seen[i]] = keep[i]; };
  int32_t out;
  if (!c.valid) {
    out = sampling::sample(full, temperature, top_p, top_k, min_p, rng);
    restore();
    return out;
  }
  sampling::Cands a = c;
  const int NC = (int)a.idx.size();
  float vmin = INFINITY;
  for (int i = 0; i < NC; ++i) if (a.idx[i] >= 0) vmin = std::min(vmin, c.val[i]);
  bool need_full = false;
  const float it = temperature > 0.f ? 1.f / temperature : 1.f;
  double dsum = 0;
  std::vector<char> in_cand;
  for (int i = 0; i < NC; ++i)
    if (a.idx[i] >= 0 && a.idx[i] < (int)P.flag.size() && P.flag[a.idx[i]]) {
      dsum += std::exp(((double)c.val[i] - c.mx) * it) * (1.0 - std::exp(-(double)P.pp * it));
      a.val[i] -= P.pp;
    }
  // seen tokens outside the candidates: their Σexp share (full row) and, for a negative penalty, whether one now beats the smallest candidate
  if (V > 0) {
    std::vector<int32_t> ci;
    for (int i = 0; i < NC; ++i) if (a.idx[i] >= 0) ci.push_back(a.idx[i]);
    std::sort(ci.begin(), ci.end());
    for (size_t k = 0; k < P.seen.size(); ++k) {
      const int32_t t = P.seen[k];
      if (std::binary_search(ci.begin(), ci.end(), t)) continue;
      dsum += std::exp(((double)keep[k] - c.mx) * it) * (1.0 - std::exp(-(double)P.pp * it));
      if (P.pp < 0.f && full[t] > vmin) need_full = true;
    }
  } else if (P.pp < 0.f) {
    need_full = false;   // greedy rows without the full row: the candidates are all we have (a raised token outside them stays unseen)
  }
  a.sum = (float)std::max(1e-30, (double)c.sum - dsum);
  // greedy winner among the shifted candidates; when it fell below the smallest unshifted candidate a token outside could win
  int best = -1;
  for (int i = 0; i < NC; ++i) if (a.idx[i] >= 0 && (best < 0 || a.val[i] > a.val[best])) best = i;
  if (best >= 0) a.argmax = a.idx[best];
  if (best < 0 || a.val[best] < vmin) need_full = V > 0;
  if (need_full) {
    out = sampling::sample(full, temperature, top_p, top_k, min_p, rng);
  } else {
    out = sampling::sample_cands(a, full, temperature, top_p, top_k, min_p, rng);
  }
  restore();
  return out;
}

}  // namespace hive::presence
