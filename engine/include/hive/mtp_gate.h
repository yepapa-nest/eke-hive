// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_MTP_GATE2 — DSpark (MTP) speculation gate v2: picks k from "the cost measured now" (used by decode_step in hived.cpp; CPU test engine/tests/test_mtp_gate_cpu.cpp).
//   Problems of the v1 gate (hived.cpp, still the default path): ① the verify cost table ema_verify_ms[2..8] starts from early measurements of 66–152 ms (when a step took ~38 ms) and **only the slot
//   of the chosen row count** is EMA-updated (α 0.1) → once it settles on k=1 the k≥2 slots keep the old values (4–8× a step) and are never chosen again. ② the acceptance table is calibrated on random-token prompts (6–90%),
//   lower than real text (serving log 49/51 = 96%). ③ the draft cost D stays in the denominator after the draft was made (a sunk cost — irrelevant to the decision). ④ a decline always pauses 8 steps.
//   v2: ⓐ a measured table per row count M (1 = ordinary decode step) + freshness (only slots measured within kStale decisions are trusted); unmeasured slots are interpolated/extrapolated **from measured slots** (no old constants; only when a single
//   slot is measured, the prior slope kPriorSlope — GPU layer chain M=1 201 µs, M=4 288, M=8 413 µs/layer → (413/201−1)/7 = 0.151 per row) · ⓑ acceptance rate per confidence bucket learned online
//   (prior = the v1 table, weight kPriorW samples), separately for temperature 0 / >0 · ⓒ the post-draft decision is E(k)/T(k+1) > 1/T(1) (the draft cost is already spent) · ⓓ the value of drafting at all is judged per request by
//   a net-gain EMA (saved ms = emitted tokens×T(1) − verify − draft); consecutive declines back off 1, 2, 4, 8 steps.
//   Losslessness: the gate only decides **how many drafts go into verification**. The acceptance rule (row i sample from the target distribution == draft) and sampling are the existing hived.cpp code unchanged.
#pragma once
#include <algorithm>
#include <cmath>

namespace hive::mtpg {

constexpr int kMaxRows = 8;            // forward_verify limit (runtime.cpp: M ≤ 8)
constexpr long kStale = 512;           // only slots measured within this many decisions count as "current cost"
constexpr double kPriorSlope = 0.151;  // per-row increment / T(1) when only one slot is measured (layer-chain measurement above)
constexpr double kAlpha = 0.2;         // cost EMA
constexpr double kPriorW = 4.0;        // weight of the acceptance prior (v1 table)
constexpr int kBuckets = 8;

inline int bucket(float c) { return c < 0.f ? 0 : c < 1.f ? 1 : c < 2.f ? 2 : c < 3.f ? 3 : c < 4.f ? 4 : c < 5.f ? 5 : c < 7.f ? 6 : 7; }
// v1 table (hived.cpp p_accept, measured over 3,727 steps) = prior
inline double prior_p(int b) { static const double p[kBuckets] = {0.06, 0.19, 0.36, 0.55, 0.65, 0.75, 0.85, 0.90}; return p[b]; }

struct Gate {
  double t[kMaxRows + 1] = {};  // step cost EMA per row count (ms)
  long n[kMaxRows + 1] = {};
  long last[kMaxRows + 1] = {};
  long clock = 0;               // decision count (ticks every single-stream decode step)
  double draft_ms = 0; long n_draft = 0;
  double hit[2][kBuckets] = {}, tried[2][kBuckets] = {};

  void tick() { ++clock; }
  void observe_step(int M, double ms) {
    if (M < 1 || M > kMaxRows || !(ms > 0) || !std::isfinite(ms)) return;  // drop negative/NaN samples (absorbs clock going backwards)
    t[M] = n[M] == 0 || clock - last[M] > kStale ? ms : t[M] * (1 - kAlpha) + ms * kAlpha;  // a stale slot is replaced by the new sample
    ++n[M]; last[M] = clock;
  }
  void observe_draft(double ms) { if (ms > 0 && std::isfinite(ms)) { draft_ms = n_draft ? draft_ms * (1 - kAlpha) + ms * kAlpha : ms; ++n_draft; } }
  bool fresh(int M) const { return M >= 1 && M <= kMaxRows && n[M] > 0 && clock - last[M] <= kStale; }
  bool known() const { for (int m = 1; m <= kMaxRows; ++m) if (n[m] > 0) return true; return false; }
  // Current cost of row count M (ms). Measured slot → its value · in between → linear interpolation of the neighbouring measured slots · outside → extrapolation with the slope of two measured slots on that side (prior slope if none).
  //   Only when two or more slots are fresh is the estimate derived from fresh slots only — with just one (usually T(1)) stale slots are used too: a measured slope, even stale, beats the prior constant
  //   (measured in the synthetic "rows are expensive" world: fresh T(1) + prior slope retried a losing speculation every 512 decisions). −1 if nothing is known.
  double cost(int M) const {
    if (M < 1 || M > kMaxRows) return -1;
    int n_fresh = 0;
    for (int m = 1; m <= kMaxRows; ++m) if (fresh(m)) ++n_fresh;
    const bool use_fresh = n_fresh >= 2;
    auto ok = [&](int m) { return n[m] > 0 && (!use_fresh || fresh(m)); };
    if (ok(M)) return t[M];
    int lo = -1, hi = -1, lo2 = -1, hi2 = -1;
    for (int m = M - 1; m >= 1; --m) if (ok(m)) { if (lo < 0) lo = m; else { lo2 = m; break; } }
    for (int m = M + 1; m <= kMaxRows; ++m) if (ok(m)) { if (hi < 0) hi = m; else { hi2 = m; break; } }
    if (lo > 0 && hi > 0) return t[lo] + (t[hi] - t[lo]) * (M - lo) / (double)(hi - lo);
    auto prior_slope = [&](int m0) { return kPriorSlope * t[m0] / (1.0 + kPriorSlope * (m0 - 1)); };
    if (lo > 0) {
      const double s = lo2 > 0 ? std::max(0.0, (t[lo] - t[lo2]) / (lo - lo2)) : prior_slope(lo);
      return t[lo] + s * (M - lo);
    }
    if (hi > 0) {
      const double s = hi2 > 0 ? std::max(0.0, (t[hi2] - t[hi]) / (hi2 - hi)) : prior_slope(hi);
      return std::max(0.25 * t[hi], t[hi] - s * (hi - M));
    }
    return -1;
  }
  // Verify cost (only for choose_k and can_profit — the batch gate choose_batch keeps using cost: all of its slots run the same decode_layer path).
  //   T(1) = ordinary step (forward_batch) · T(M ≥ 2) = verify step (forward_verify + row sampling + rollback). When the two paths differ (HIVE_MTP_VERIFY2 off = verification on the older layer_forward path)
  //   verification has a large fixed cost (serving log: T1 20.0 · T2 51.9 · T6 112.3 ms — first row +31.9 ms, then ~15 ms per row). But cost() joins T(1) and
  //   the verify slots on one line: with T2 stale (kStale) and another verify slot (T6) fresh, T2 = interpolation T1..T6 = 38.5 ms, so (1+p)/38.5 > 1/20 (p > 0.92) —
  //   it picks k=1 although the measured T2 = 51.9 > 2·T1. Here only verify slots are interpolated/extrapolated among themselves (freshness rule as in cost — if two or more verify slots are fresh, only those).
  //   With no verify slot measured yet: T(1)·(1 + kPriorSlope·(M−1)) as before (exploration to get the first sample — once per process). M = 1 is cost(1).
  double vcost(int M) const {
    if (M < 1 || M > kMaxRows) return -1;
    if (M == 1) return cost(1);
    int n_fresh = 0;
    for (int m = 2; m <= kMaxRows; ++m) if (fresh(m)) ++n_fresh;
    const bool use_fresh = n_fresh >= 2;
    auto ok = [&](int m) { return m >= 2 && n[m] > 0 && (!use_fresh || fresh(m)); };
    if (ok(M)) return t[M];
    int lo = -1, hi = -1, lo2 = -1, hi2 = -1;
    for (int m = M - 1; m >= 2; --m) if (ok(m)) { if (lo < 0) lo = m; else { lo2 = m; break; } }
    for (int m = M + 1; m <= kMaxRows; ++m) if (ok(m)) { if (hi < 0) hi = m; else { hi2 = m; break; } }
    if (lo > 0 && hi > 0) return t[lo] + (t[hi] - t[lo]) * (M - lo) / (double)(hi - lo);
    auto prior_slope = [&](int m0) { return kPriorSlope * t[m0] / (1.0 + kPriorSlope * (m0 - 1)); };  // same prior slope as cost() (when one slot is measured)
    if (lo > 0) {
      const double s = lo2 > 0 ? std::max(0.0, (t[lo] - t[lo2]) / (lo - lo2)) : prior_slope(lo);
      return t[lo] + s * (M - lo);
    }
    if (hi > 0) {
      const double s = hi2 > 0 ? std::max(0.0, (t[hi2] - t[hi]) / (hi2 - hi)) : prior_slope(hi);
      return std::max(0.25 * t[hi], t[hi] - s * (hi - M));
    }
    const double t1 = cost(1);
    return t1 > 0 ? t1 * (1.0 + kPriorSlope * (M - 1)) : -1;
  }
  // There are verify slots but none is fresh (no verification for kStale decisions) — is this a window where one re-measurement (k = 1) is allowed (see the comment above choose_k)?
  bool probe_due() const {
    bool any = false;
    for (int m = 2; m <= kMaxRows; ++m) { if (fresh(m)) return false; if (n[m] > 0) any = true; }
    return any && clock - probe_at > kStale;
  }
  mutable long probe_at = -(kStale + 1);  // decision time of the last re-measurement (written by choose_k)
  double p_accept(bool sampled, float c) const {
    const int b = bucket(c), s = sampled ? 1 : 0;
    return (hit[s][b] + kPriorW * prior_p(b)) / (tried[s][b] + kPriorW);
  }
  // Verification compared the first n_cmp drafts and the first n_acc of them were accepted (n_acc ≤ n_cmp). Each compared draft i is a sample of "acceptance given all earlier ones were accepted".
  void observe_accept(bool sampled, const float* conf, int n_cmp, int n_acc) {
    const int s = sampled ? 1 : 0;
    for (int i = 0; i < n_cmp; ++i) {
      const int b = bucket(conf[i]);
      tried[s][b] += 1; if (i < n_acc) hit[s][b] += 1;
      if (tried[s][b] > 512) { tried[s][b] *= 0.5; hit[s][b] *= 0.5; }  // halve old samples (text properties change from request to request)
    }
  }
  // Post-draft decision: k (0 = no speculation). Drafts from the front while conf ≥ conf_min, k ≤ max_k · k ≤ remaining (= max_tokens − tokens emitted; the condition A.n + i < max_tokens) · k+1 ≤ kMaxRows.
  //   E(k) = 1 + Σ_{i<k} Π_{j≤i} p(c_j) · rate = E(k)/T(k+1) · baseline = 1/T(1). The draft cost is already spent and is not included.
  //   Verify-slot cost = vcost (not joined with T(1) — see the vcost comment). If the table says "cannot win", the drafts are not verified. One exception = re-measurement (probe_due):
  //   if every verify slot is stale (no verification for kStale decisions) T(2) is re-measured once per window with k = 1 — so an inflated first sample (graph capture etc.) cannot block a slot forever.
  //   Loss bound (numbers): one re-measurement = D + T2 − (1 + a)·T1 (a = acceptance 0/1 of that draft). With the serving values above (T1 20, T2 51.9, draft D ~3 ms): a=1 13 ms, a=0 35 ms,
  //   at most once per kStale = 512 decisions (≥ 512 × T1 = 10.2 s) → 0.13–0.34% of decode time. The first exploration with no verify slot measured is unchanged.
  int choose_k(bool sampled, const float* conf, int n_drafts, int max_k, float conf_min, int remaining, double* e_out = nullptr, double* ms_out = nullptr) const {
    const double t1 = cost(1);
    if (e_out) *e_out = 1;
    if (ms_out) *ms_out = t1;
    if (!(t1 > 0)) return 0;
    double best = 1.0 / t1, chain = 1.0, e = 1.0;
    int k = 0;
    for (int i = 0; i < n_drafts && i < max_k && i < remaining && i + 2 <= kMaxRows; ++i) {
      if (conf[i] < conf_min) break;
      chain *= p_accept(sampled, conf[i]);
      e += chain;
      const double tv = vcost(i + 2);
      if (!(tv > 0)) break;
      if (e / tv > best) { best = e / tv; k = i + 1; if (e_out) *e_out = e; if (ms_out) *ms_out = tv; }
    }
    if (k == 0 && n_drafts >= 1 && max_k >= 1 && remaining >= 1 && conf[0] >= conf_min && probe_due()) {
      probe_at = clock;
      if (e_out) *e_out = 1.0 + p_accept(sampled, conf[0]);
      if (ms_out) *ms_out = vcost(2);
      return 1;
    }
    return k;
  }
  // Is drafting worth anything at all: even if the best learned bucket's acceptance applied to every draft, E(k)/T(k+1) ≤ 1/T(1) means no (→ long pause).
  bool can_profit(bool sampled, int max_k) const {
    const double t1 = cost(1);
    if (!(t1 > 0)) return false;
    //   Bucket representative: only buckets with comparison samples (all buckets — the prior — if none). So the prior (0.90) of a bucket never seen cannot keep "hope" on forever.
    const int s = sampled ? 1 : 0;
    bool any = false; for (int b = 0; b < kBuckets; ++b) if (tried[s][b] > 0) any = true;
    double pmax = 0;
    for (int b = 0; b < kBuckets; ++b) if (!any || tried[s][b] > 0) pmax = std::max(pmax, (hit[s][b] + kPriorW * prior_p(b)) / (tried[s][b] + kPriorW));
    double chain = 1, e = 1;
    for (int k = 1; k <= max_k && k + 1 <= kMaxRows; ++k) { chain *= pmax; e += chain; const double tv = vcost(k + 1); if (tv > 0 && e / tv > 1.0 / t1) return true; }  // vcost
    return false;
  }
};

// Per-request backoff (is drafting itself a loss): net-gain EMA (ms per attempt). Consecutive declines → pause 1, 2, 4, 8 steps · hopeless (Gate::can_profit false) → pause 32 steps ·
//   net-gain (expected tokens×T(1) − verify − draft) EMA < 0 (after ≥ 4 attempts) → pause 8 steps.
struct Backoff {
  int skip = 0, declines = 0; long attempts = 0; double net = 0;
  bool want() { if (skip > 0) { --skip; return false; } return true; }
  void on_decline(double draft_ms, bool hopeless = false) {
    ++attempts; net = net * 0.8 + (-draft_ms) * 0.2;
    ++declines; skip = hopeless ? 32 : std::min(8, 1 << std::min(3, declines - 1));
  }
  // e_tok/v_ms = **expected** tokens and verify cost of the chosen k (choose_k output — realized values swing with luck and cut healthy speculation for 8 steps at a time: synthetic world p 0.5, x1.09 → x1.19 with expected values)
  void on_spec(double e_tok, double v_ms, double draft_ms, double t1) {
    ++attempts; declines = 0;
    const double saved = e_tok * t1 - v_ms - draft_ms;
    net = attempts == 1 ? saved : net * 0.8 + saved * 0.2;
    if (net < 0 && attempts >= 4) skip = 8;
  }
};

// HIVE_MTP_BATCH — speculation gate for batched decode with S ≥ 2 active (uses one Gate instance dedicated to batching — tables are not shared with the single-stream gate):
//   slot = **total row count** of one forward (ordinary batch step = S rows · verify = Σ_s (k_s + 1) rows — both run the same decode path decode_layer) · acceptance table = this instance's.
//   Post-draft decision (choose_batch): for each K = Σ k_s, the allocation maximizing expected tokens E(K) = greedy by largest marginal gain (next draft of part s = current chain × p(c)) —
//   the marginal gain within a part only decreases (p ≤ 1), so this is optimal for each K. rate = E(K)/T(S+K) · baseline = S/T(S) (ordinary batch step) · draft cost already spent (same principle as Gate::choose_k).
//   Pre-draft hope (can_profit_batch): even if the best learned bucket acceptance applied to every draft, max_K E(K)/(S·D + T(S+K)) ≤ S/T(S) means hopeless.
//   Losslessness: the gate only decides how many drafts per part go into verification (acceptance rule and sampling as in the hived.cpp single-stream path).
struct BatchChoice {
  int k[kMaxRows] = {};  // drafts per part
  int K = 0;             // Σ k
  double e = 0, ms = 0;  // expected tokens of the chosen allocation (sum over parts) · cost T(S+K)
};
// Batch re-measurement window (batch version of Gate::probe_due): this S's verify slots (rows S+1..max_rows) have been measured but none is fresh, and kStale decisions
//   have passed since the last re-measurement. Reason (measured, c4): after a single first 5-row verify sample of 47.5 ms (T(4) 33 ms) entered its slot, only "no draft · skip 32" repeated —
//   that slot was never chosen again and so never updated. The ordinary batch-step slot (row S) is always fresh, so Gate::probe_due (which looks at rows ≥ 2) is
//   always false in batch mode — hence S is passed in. Loss bound: at most one K = 1 verification per window (kStale decisions), after the draft cost is already paid.
inline bool batch_probe_due(const Gate& g, int S, int max_rows = kMaxRows) {
  if (S < 1 || S >= kMaxRows) return false;
  max_rows = std::min(max_rows, kMaxRows);
  bool any = false;
  for (int m = S + 1; m <= max_rows; ++m) { if (g.fresh(m)) return false; if (g.n[m] > 0) any = true; }
  return any && g.clock - g.probe_at > kStale;
}
// conf[s] = confidence column of part s (n_drafts[s] entries) · remaining[s] = tokens that request may still emit · sampled[s] = temperature > 0 · max_rows = row limit of one verification (≤ kMaxRows)
//   If the table says "cannot win" (K = 0) but batch_probe_due holds, T(S+1) is re-measured with one draft (K = 1) for the part with the largest marginal gain (once per window, records probe_at).
inline BatchChoice choose_batch(const Gate& g, int S, const bool* sampled, const float* const* conf, const int* n_drafts, int max_k, float conf_min,
                                const int* remaining, int max_rows) {
  BatchChoice best;
  if (S < 1 || S > kMaxRows) return best;
  max_rows = std::min(max_rows, kMaxRows);
  const double tS = g.cost(S);
  if (!(tS > 0)) return best;
  double best_rate = S / tS, e = S, chain[kMaxRows];
  int k[kMaxRows];
  for (int s = 0; s < S; ++s) { chain[s] = 1.0; k[s] = 0; }
  for (int K = 1; S + K <= max_rows; ++K) {
    int pick = -1;
    double gain = -1;
    for (int s = 0; s < S; ++s) {
      if (k[s] >= n_drafts[s] || k[s] >= max_k || k[s] >= remaining[s] || conf[s][k[s]] < conf_min) continue;
      const double m = chain[s] * g.p_accept(sampled[s], conf[s][k[s]]);
      if (m > gain) { gain = m; pick = s; }
    }
    if (pick < 0) break;
    chain[pick] *= g.p_accept(sampled[pick], conf[pick][k[pick]]);
    e += chain[pick];
    ++k[pick];
    const double tv = g.cost(S + K);
    if (!(tv > 0)) break;
    if (e / tv > best_rate) {
      best_rate = e / tv;
      for (int s = 0; s < S; ++s) best.k[s] = k[s];
      best.K = K; best.e = e; best.ms = tv;
    }
  }
  if (best.K == 0 && S + 1 <= max_rows && batch_probe_due(g, S, max_rows)) {  // re-measurement (see the batch_probe_due comment above)
    int pick = -1;
    double gain = -1;
    for (int s = 0; s < S; ++s) {
      if (n_drafts[s] < 1 || max_k < 1 || remaining[s] < 1 || conf[s][0] < conf_min) continue;
      const double m = g.p_accept(sampled[s], conf[s][0]);
      if (m > gain) { gain = m; pick = s; }
    }
    if (pick >= 0) {
      g.probe_at = g.clock;
      best.k[pick] = 1; best.K = 1; best.e = S + gain;
      const double tv = g.cost(S + 1);
      best.ms = tv > 0 ? tv : tS;
    }
  }
  return best;
}
// Pre-draft hope (hived.cpp checks it **before drafting**, at the same place as Gate::can_profit; if false, drafts only when batch_probe_due): draft_ms = draft cost of one part (Gate::draft_ms — drafts are made per part). Best acceptance among buckets with samples (all buckets of the prior if none — same rule as Gate::can_profit).
inline bool can_profit_batch(const Gate& g, int S, bool sampled, int max_k, int max_rows) {
  if (S < 1 || S > kMaxRows) return false;
  max_rows = std::min(max_rows, kMaxRows);
  const double tS = g.cost(S);
  if (!(tS > 0)) return false;
  const int si = sampled ? 1 : 0;
  bool any = false; for (int b = 0; b < kBuckets; ++b) if (g.tried[si][b] > 0) any = true;
  double pmax = 0;
  for (int b = 0; b < kBuckets; ++b) if (!any || g.tried[si][b] > 0) pmax = std::max(pmax, (g.hit[si][b] + kPriorW * prior_p(b)) / (g.tried[si][b] + kPriorW));
  const double D = g.n_draft > 0 ? g.draft_ms : 0.0;
  double e = S;
  int k[kMaxRows] = {};
  for (int K = 1; S + K <= max_rows; ++K) {  // with equal p, the part with the largest marginal gain = the part with the fewest drafts (round robin)
    int s = 0;
    for (int t = 1; t < S; ++t) if (k[t] < k[s]) s = t;
    if (k[s] >= max_k) break;
    ++k[s];
    e += std::pow(pmax, k[s]);
    const double tv = g.cost(S + K);
    if (tv > 0 && e / (S * D + tv) > S / tS) return true;
  }
  return false;
}

}  // namespace hive::mtpg
