// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Speculative sampling at temperature > 0 (HIVE_MTP_SAMPLE_DRAFTS) — the draft distribution and the acceptance rule.
//   Before: a draft was the draft head's argmax and was accepted when the target row's sample equalled it, so at temperature T a draft survived
//   with probability p_T(argmax q) — the target's own probability of that one token (production 2026-11-04, DeepSeek: thinking requests at T 1
//   accepted 73 % of their drafts, chat requests 82 %). Here the draft is drawn from the draft head's distribution q (the request's temperature,
//   top_k, top_p and min_p applied — the rule of hive/sampler.h) and verified with the standard rule (Leviathan et al. 2023; Chen et al. 2023):
//   x ~ q is accepted with probability min(1, p(x) / q(x)); on a rejection the emitted token is drawn from norm(max(0, p − q)); when every
//   draft is accepted the last row's sample is the bonus. The emitted tokens follow the target distribution p exactly, whatever q is
//   (tests/test_sampler_cpu.cpp (6)), and a draft survives with probability Σ_x min(p(x), q(x)) ≥ p(argmax q) whenever q is at least as
//   peaked as p at its argmax — the usual case for a draft head trained on the same model.
//   Dist is exactly the distribution hive/sampler.h draws from (temperature → top_k → top_p → min_p): sparse = the kept tokens with their
//   normalised probabilities (nucleus or top_k, at most 1024 — the candidates — or the full sort fallback), dense = every token (no top_k and no
//   nucleus: the order-free rule), read lazily from the logits row so that building p for a verify row costs no vocabulary pass; only a rejection
//   pays one (the correction sample paid the same before).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "hive/sampler.h"

namespace hive {
namespace spec {

using hive::sampling::ByProbDesc;
using hive::sampling::Cands;

struct Dist {
  bool dense = false;
  // sparse: kept (token, probability) sorted by token id; total = Σ probability (the kept mass before renormalisation)
  std::vector<std::pair<int32_t, float>> kept;
  double total = 0;
  // dense: e(t) = exp((l_t − mx)·it), 0 when min_p > 0 and e(t) < min_p (e_max = 1); probability = e(t) / total
  const float* row = nullptr;   // non-owning: a verify row lives for the step
  std::vector<float> own;       // owning copy (draft rows — the device buffer is reused by the next draft)
  int V = 0;
  float mx = 0.f, it = 1.f, min_p = 0.f;
  Cands cands;                  // the device candidates of a dense row — the fast path of draw()
  int32_t argmax = -1;

  const float* logits() const { return own.empty() ? row : own.data(); }
  bool valid() const { return total > 0 && (dense ? logits() != nullptr && V > 0 : !kept.empty()); }
  void take_copy() { if (dense && own.empty() && row) { own.assign(row, row + V); row = nullptr; } }
  double e_dense(int32_t t) const {
    const float x = expf((logits()[t] - mx) * it);
    return (min_p > 0.f && x < min_p) ? 0.0 : (double)x;
  }
  double prob(int32_t t) const {
    if (!valid()) return 0.0;
    if (dense) return (t < 0 || t >= V) ? 0.0 : e_dense(t) / total;
    const auto itr = std::lower_bound(kept.begin(), kept.end(), t, [](const std::pair<int32_t, float>& a, int32_t b) { return a.first < b; });
    return (itr != kept.end() && itr->first == t) ? itr->second / total : 0.0;
  }
};

namespace detail {
inline void finish_sparse(Dist& d, std::vector<std::pair<float, int>>& p, int keep) {
  d.dense = false;
  d.kept.clear();
  d.kept.reserve(keep);
  d.total = 0;
  d.argmax = keep > 0 ? p[0].second : -1;
  for (int i = 0; i < keep; ++i) { d.kept.push_back({(int32_t)p[i].second, p[i].first}); d.total += p[i].first; }
  std::sort(d.kept.begin(), d.kept.end(), [](const std::pair<int32_t, float>& a, const std::pair<int32_t, float>& b) { return a.first < b.first; });
}
// min_p then top_p cuts over p (descending), as sampling::sample / sample_cands apply them
inline int cut(const std::vector<std::pair<float, int>>& p, int keep, float top_p, float min_p) {
  if (min_p > 0.f) { const float thr = p[0].first * min_p; int k2 = 0; while (k2 < keep && p[k2].first >= thr) ++k2; keep = std::max(1, k2); }
  if (top_p > 0.f && top_p < 1.f) { double c = 0; int k2 = 0; while (k2 < keep) { c += p[k2].first; ++k2; if (c >= top_p) break; } keep = std::max(1, k2); }
  return keep;
}
}  // namespace detail

// The distribution sampling::sample(logits, T, top_p, top_k, min_p) draws from (T > 0). Dense rows reference `logits` (not copied).
inline Dist dist_full(const float* logits, int V, float temperature, float top_p, int top_k, float min_p) {
  Dist d;
  d.V = V; d.min_p = min_p;
  if (!(temperature > 0.f) || V <= 0) return d;
  d.it = 1.f / temperature;
  float mx = -INFINITY;
  for (int i = 0; i < V; ++i) mx = std::max(mx, logits[i]);
  d.mx = mx;
  if (top_k <= 0 && !(top_p > 0.f && top_p < 1.f)) {   // order-free rule: every token with e ≥ min_p
    d.dense = true; d.row = logits;
    double sum = 0; float best = -INFINITY; int32_t arg = -1;
    for (int i = 0; i < V; ++i) {
      const float x = expf((logits[i] - mx) * d.it);
      if (!(min_p > 0.f && x < min_p)) sum += x;
      if (logits[i] > best) { best = logits[i]; arg = i; }
    }
    d.total = sum; d.argmax = arg;
    return d;
  }
  std::vector<std::pair<float, int>> p(V);
  double sum = 0;
  for (int i = 0; i < V; ++i) { const float e = expf((logits[i] - mx) * d.it); p[i] = {e, i}; sum += e; }
  for (auto& x : p) x.first = (float)(x.first / sum);
  int cand = top_k > 0 ? std::min(V, top_k) : std::min(V, 1024);
  std::partial_sort(p.begin(), p.begin() + cand, p.end(), ByProbDesc{});
  double mass = 0;
  for (int i = 0; i < cand; ++i) mass += p[i].first;
  const double need = (top_p > 0.f && top_p < 1.f) ? top_p : 1.0;
  if (top_k <= 0 && mass < need && cand < V) std::sort(p.begin(), p.end(), ByProbDesc{});
  else p.resize(cand);
  int keep = (int)p.size();
  if (top_k > 0) keep = std::min(keep, top_k);
  keep = detail::cut(p, keep, top_p, min_p);
  detail::finish_sparse(d, p, keep);
  return d;
}

// The distribution sampling::sample_cands(c, logits, …) draws from: the same rule within the device candidates, falling back to dist_full
//   when the candidate mass falls short of top_p, top_k exceeds the candidates, or min_p needs the full row. A dense result references `logits`.
inline Dist dist_cands(const Cands& c, const float* logits, int V, float temperature, float top_p, int top_k, float min_p) {
  if (!c.valid || !(temperature > 0.f)) return dist_full(logits, V, temperature, top_p, top_k, min_p);
  const int NC = (int)c.idx.size();
  if (top_k > NC) return dist_full(logits, V, temperature, top_p, top_k, min_p);
  const float it = 1.f / temperature;
  if (top_k <= 0 && !(top_p > 0.f && top_p < 1.f)) {
    double ec = 0, emin = INFINITY;
    int nvalid = 0;
    for (int i = 0; i < NC; ++i) if (c.idx[i] >= 0) { const double e = expf((c.val[i] - c.mx) * it); ec += e; emin = std::min(emin, e); ++nvalid; }
    const bool all_in = nvalid >= V;
    if (min_p > 0.f) {
      if (!(all_in || emin < (double)min_p)) return dist_full(logits, V, temperature, top_p, top_k, min_p);
      std::vector<std::pair<float, int>> p;
      p.reserve(NC);
      for (int i = 0; i < NC; ++i) if (c.idx[i] >= 0) { const double e = expf((c.val[i] - c.mx) * it); if (e >= (double)min_p) p.push_back({(float)e, c.idx[i]}); }
      std::sort(p.begin(), p.end(), ByProbDesc{});
      Dist d; d.V = V; d.it = it; d.mx = c.mx; d.min_p = min_p;
      if (p.empty()) return dist_full(logits, V, temperature, top_p, top_k, min_p);
      detail::finish_sparse(d, p, (int)p.size());
      return d;
    }
    if (!(ec > 0) || !std::isfinite(c.sum) || nvalid == 0) return dist_full(logits, V, temperature, top_p, top_k, min_p);
    Dist d;
    d.dense = true; d.row = logits; d.V = V; d.it = it; d.mx = c.mx; d.min_p = 0.f;
    d.total = std::max((double)c.sum, ec);   // sample_cands: fp32 rounding can leave c.sum a hair below the candidates' own sum
    d.cands = c; d.argmax = c.argmax;
    return d;
  }
  std::vector<std::pair<float, int>> p;
  p.reserve(NC);
  for (int i = 0; i < NC; ++i) if (c.idx[i] >= 0) p.push_back({(float)(expf((c.val[i] - c.mx) * it) / (double)c.sum), c.idx[i]});
  std::sort(p.begin(), p.end(), ByProbDesc{});
  int cand = top_k > 0 ? std::min((int)p.size(), top_k) : (int)p.size();
  double mass = 0;
  for (int i = 0; i < cand; ++i) mass += p[i].first;
  const double need = (top_p > 0.f && top_p < 1.f) ? top_p : 1.0;
  if (top_k <= 0 && mass < need) return dist_full(logits, V, temperature, top_p, top_k, min_p);  // mass outside the candidates is needed — full path
  if (cand <= 0) return dist_full(logits, V, temperature, top_p, top_k, min_p);
  p.resize(cand);
  const int keep = detail::cut(p, cand, top_p, min_p);
  Dist d; d.V = V; d.it = it; d.mx = c.mx; d.min_p = min_p;
  detail::finish_sparse(d, p, keep);
  return d;
}

// One token from the distribution. Dense: the two-stage draw of sample_cands (candidates first, the rest of the row only when the draw falls
//   outside their mass) or, without candidates, one order-free pass over the row.
inline int32_t draw(const Dist& d, std::mt19937_64& rng) {
  if (!d.valid()) return d.argmax;
  if (!d.dense) {
    std::uniform_real_distribution<double> U(0.0, d.total);
    const double r = U(rng);
    double c = 0;
    for (const auto& kv : d.kept) { c += kv.second; if (r <= c) return kv.first; }
    return d.kept.back().first;
  }
  const float* l = d.logits();
  const int V = d.V;
  if (d.cands.valid && d.min_p <= 0.f) {
    const Cands& c = d.cands;
    const int NC = (int)c.idx.size();
    double ec = 0;
    int nvalid = 0;
    for (int i = 0; i < NC; ++i) if (c.idx[i] >= 0) { ec += expf((c.val[i] - d.mx) * d.it); ++nvalid; }
    std::uniform_real_distribution<double> U(0.0, d.total);
    const double u = U(rng);
    if (u < ec || nvalid >= V) {
      double cum = 0;
      int32_t last = d.argmax;
      for (int i = 0; i < NC; ++i) if (c.idx[i] >= 0) { cum += expf((c.val[i] - d.mx) * d.it); last = c.idx[i]; if (u <= cum) return c.idx[i]; }
      return last;
    }
    static thread_local std::vector<float> e;
    e.resize(V);
    double rest = 0;
    for (int i = 0, j = 0; i < V; ++i) {
      while (j < NC && (c.idx[j] < 0 || c.idx[j] < i)) ++j;
      const bool cand = j < NC && c.idx[j] == i;
      e[i] = cand ? 0.f : expf((l[i] - d.mx) * d.it);
      rest += e[i];
    }
    if (!(rest > 0)) return d.argmax;
    std::uniform_real_distribution<double> U2(0.0, rest);
    const double r = U2(rng);
    double cum = 0;
    int32_t last = -1;
    for (int i = 0; i < V; ++i) if (e[i] > 0.f) { cum += e[i]; last = i; if (r <= cum) return i; }
    return last >= 0 ? last : d.argmax;
  }
  std::uniform_real_distribution<double> U(0.0, d.total);
  const double r = U(rng);
  double cum = 0;
  int32_t last = -1;
  for (int i = 0; i < V; ++i) { const double x = d.e_dense(i); if (x > 0) { cum += x; last = i; if (r <= cum) return i; } }
  return last >= 0 ? last : d.argmax;
}

// Acceptance of a draft x ~ q against the target p: with probability min(1, p(x) / q(x)).
inline bool accept(const Dist& p, const Dist& q, int32_t x, std::mt19937_64& rng) {
  const double px = p.prob(x), qx = q.prob(x);
  if (!(px > 0)) return false;
  if (!(qx > 0) || px >= qx) return true;
  return std::uniform_real_distribution<double>(0.0, 1.0)(rng) < px / qx;
}

// The correction token after a rejection: from norm(max(0, p − q)). An empty residual (p ≤ q everywhere — p = q up to rounding) draws from p.
inline int32_t residual_draw(const Dist& p, const Dist& q, std::mt19937_64& rng) {
  if (!p.valid()) return p.argmax;
  if (!p.dense) {
    static thread_local std::vector<double> r;
    r.resize(p.kept.size());
    double sum = 0;
    for (size_t i = 0; i < p.kept.size(); ++i) { r[i] = std::max(0.0, p.kept[i].second / p.total - q.prob(p.kept[i].first)); sum += r[i]; }
    if (!(sum > 0)) return draw(p, rng);
    std::uniform_real_distribution<double> U(0.0, sum);
    const double u = U(rng);
    double cum = 0;
    for (size_t i = 0; i < p.kept.size(); ++i) { cum += r[i]; if (r[i] > 0 && u <= cum) return p.kept[i].first; }
    for (size_t i = p.kept.size(); i-- > 0;) if (r[i] > 0) return p.kept[i].first;
    return draw(p, rng);
  }
  // dense p: one pass over the row; q sparse = a merge walk over its kept list (sorted by token), q dense = its own row
  static thread_local std::vector<float> r;
  const int V = p.V;
  r.resize(V);
  double sum = 0;
  size_t j = 0;
  for (int t = 0; t < V; ++t) {
    double qt;
    if (q.dense) qt = q.prob(t);
    else { while (j < q.kept.size() && q.kept[j].first < t) ++j; qt = (j < q.kept.size() && q.kept[j].first == t) ? q.kept[j].second / q.total : 0.0; }
    const double pt = p.e_dense(t) / p.total;
    r[t] = (float)std::max(0.0, pt - qt);
    sum += r[t];
  }
  if (!(sum > 0)) return draw(p, rng);
  std::uniform_real_distribution<double> U(0.0, sum);
  const double u = U(rng);
  double cum = 0;
  int32_t last = -1;
  for (int t = 0; t < V; ++t) if (r[t] > 0.f) { cum += r[t]; last = t; if (u <= cum) return t; }
  return last >= 0 ? last : draw(p, rng);
}

}  // namespace spec
}  // namespace hive
