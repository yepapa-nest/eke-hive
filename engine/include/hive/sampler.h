// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hived host sampler — moved verbatim from engine/src/hived.cpp (the CPU test tests/test_sampler_cpu.cpp measures the
//   distributions of these same functions).
//   Rule: temperature (≤ 0 = argmax) → top_k → top_p → min_p → multinomial over the remaining mass. sample_cands applies the
//   same rule within the top NC candidates picked by the GPU (values, indices + row max and Σexp with temperature applied), and
//   falls back to the full-logit path (sample) when the candidate mass falls short of top_p or top_k > NC — both paths have the
//   same distribution (tested).
//   request_seed: the request's seed (integer ≠ 0 = that value · 0 or absent = a different seed per request).
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "hive/clock.h"

namespace hive {
namespace sampling {

// Orders (prob, token) pairs by descending probability.
struct ByProbDesc {
  template <class P> bool operator()(const P& a, const P& b) const { return a.first > b.first; }
};

// Host sampling: temperature → top_k → top_p → min_p → multinomial
inline int32_t sample(std::vector<float>& logits, float temperature, float top_p, int top_k, float min_p, std::mt19937_64& rng) {
  const int V = (int)logits.size();
  if (temperature <= 0.f) return (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
  float mx = -INFINITY;
  for (float v : logits) mx = std::max(mx, v);
  std::vector<std::pair<float, int>> p(V);
  double sum = 0;
  for (int i = 0; i < V; ++i) { float e = expf((logits[i] - mx) / temperature); p[i] = {e, i}; sum += e; }
  for (auto& x : p) x.first = (float)(x.first / sum);
  // Partial sort of the top candidates instead of a full sort (vocabulary 129K, ~10 ms per token): top_k of them if set, else 1024 (falls back to a full sort if their mass falls short of top_p)
  {
    int cand = top_k > 0 ? std::min(V, top_k) : std::min(V, 1024);
    std::partial_sort(p.begin(), p.begin() + cand, p.end(), ByProbDesc{});
    double mass = 0;
    for (int i = 0; i < cand; ++i) mass += p[i].first;
    const double need = (top_p > 0.f && top_p < 1.f) ? top_p : 1.0;
    if (top_k <= 0 && mass < need && cand < V) std::sort(p.begin(), p.end(), ByProbDesc{});
    else if (top_k <= 0) p.resize(cand);
    else p.resize(cand);
  }
  int keep = (int)p.size();
  if (top_k > 0) keep = std::min(keep, top_k);
  if (min_p > 0.f) { float thr = p[0].first * min_p; int k2 = 0; while (k2 < keep && p[k2].first >= thr) ++k2; keep = std::max(1, k2); }
  if (top_p > 0.f && top_p < 1.f) { double c = 0; int k2 = 0; while (k2 < keep) { c += p[k2].first; ++k2; if (c >= top_p) break; } keep = std::max(1, k2); }
  double tot = 0;
  for (int i = 0; i < keep; ++i) tot += p[i].first;
  std::uniform_real_distribution<double> U(0.0, tot);
  double r = U(rng), c = 0;
  for (int i = 0; i < keep; ++i) { c += p[i].first; if (r <= c) return p[i].second; }
  return p[keep - 1].second;
}

// Apply the same rule (temperature → top_k → top_p → min_p → multinomial) within the GPU-selected candidates (top NC values and
// indices + row max and Σexp). Removes the cost of the host partial sort over 129K (~1 ms per row, 6 rows for MTP verification).
// Falls back to the full-logit path when the candidate mass falls short of top_p or top_k > NC.
struct Cands {
  std::vector<int32_t> idx;
  std::vector<float> val;
  float mx = 0.f, sum = 1.f;
  int32_t argmax = -1;
  bool valid = false;
};
inline int32_t sample_cands(const Cands& c, std::vector<float>& full_logits, float temperature, float top_p, int top_k, float min_p, std::mt19937_64& rng) {
  if (!c.valid) return sample(full_logits, temperature, top_p, top_k, min_p, rng);
  if (temperature <= 0.f) return c.argmax;
  const int NC = (int)c.idx.size();
  if (top_k > NC) return sample(full_logits, temperature, top_p, top_k, min_p, rng);
  std::vector<std::pair<float, int>> p;
  p.reserve(NC);
  const float it = 1.f / temperature;
  for (int i = 0; i < NC; ++i) if (c.idx[i] >= 0) p.push_back({(float)(expf((c.val[i] - c.mx) * it) / (double)c.sum), c.idx[i]});
  std::sort(p.begin(), p.end(), ByProbDesc{});
  int cand = top_k > 0 ? std::min((int)p.size(), top_k) : (int)p.size();
  double mass = 0;
  for (int i = 0; i < cand; ++i) mass += p[i].first;
  const double need = (top_p > 0.f && top_p < 1.f) ? top_p : 1.0;
  if (top_k <= 0 && mass < need) return sample(full_logits, temperature, top_p, top_k, min_p, rng);  // mass outside the candidates is needed — full path
  p.resize(cand);
  int keep = cand;
  if (min_p > 0.f) { float thr = p[0].first * min_p; int k2 = 0; while (k2 < keep && p[k2].first >= thr) ++k2; keep = std::max(1, k2); }
  if (top_p > 0.f && top_p < 1.f) { double cum = 0; int k2 = 0; while (k2 < keep) { cum += p[k2].first; ++k2; if (cum >= top_p) break; } keep = std::max(1, k2); }
  double tot = 0;
  for (int i = 0; i < keep; ++i) tot += p[i].first;
  std::uniform_real_distribution<double> U(0.0, tot);
  double r = U(rng), cum = 0;
  for (int i = 0; i < keep; ++i) { cum += p[i].first; if (r <= cum) return p[i].second; }
  return p[keep - 1].second;
}
// Seed: a seed derived only from the clock (`seed ? seed : (uint64_t)now_ms()`) gave requests received in the same millisecond
//   (batched prefill admit_batch accepts several requests in one loop) the same seed = the same random stream — parallel
//   requests without a seed and with the same prompt produced the same answer (the server sends an unspecified seed as 0).
//   For 0, mix a per-process random value (one random_device draw) ⊕ the request counter ⊕ nanoseconds with splitmix64.
//   A non-zero seed is used as is (for reproducibility).
inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
inline uint64_t request_seed(int64_t seed) {
  if (seed != 0) return (uint64_t)seed;
  static const uint64_t base = [] { std::random_device rd; return ((uint64_t)rd() << 32) ^ (uint64_t)rd(); }();
  static std::atomic<uint64_t> counter{0};
  const uint64_t ns = (uint64_t)hive::SteadyClock::now().time_since_epoch().count();
  return splitmix64(base ^ splitmix64(counter.fetch_add(1, std::memory_order_relaxed) + 1) ^ ns);
}

}  // namespace sampling
}  // namespace hive
