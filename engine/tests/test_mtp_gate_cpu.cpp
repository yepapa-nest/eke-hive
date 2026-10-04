// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU test for the DSpark speculation gate v2 (HIVE_MTP_GATE2, hive/mtp_gate.h). No GPU.
//   (1) cost table: measured cells · interpolation · extrapolation (measured slope / prior slope) · freshness (stale
//       cells are re-derived from measured ones) · dropped samples · stale cells replaced by a new sample
//   (2) choose_k == exhaustive argmax (independent implementation) · conf threshold · max_k · remaining tokens ·
//       row cap 8 · unknown T(1) -> 0
//   (3) reproduces the trap of the previous gate: with only 2 rows measured the old formula can never pick k >= 2,
//       v2 does
//   (4) acceptance learning (prior -> observed) · separate temperature classes · (5) backoff 1·2·4·8 · net gain < 0 -> 8
//   (6) synthetic world: tokens/ms under cost T(M) = a + b·(M−1) and acceptance p — speculation speeds up a good
//       world, and in a bad world (low p) it is never more than 3% slower than plain decode.
//   (7) HIVE_MTP_BATCH: choose_batch == exhaustive maximum rate · edges (unknown T(S) · row cap) · can_profit_batch ·
//       synthetic batch worlds (assumed c2/c3/c4 scales)
//   (8) F1: verify cells are not joined to T(1) (vcost) · re-measure probes · loss bound in an old-verify-path world
//   (9) MB1 batch re-measure and first-use samples: batch_probe_due semantics (per S · once per window · none while a
//       verify cell is fresh) · reproduces the v2rfb trap (T(4) 33 · first 5-row step 47.5 ms -> K = 0 forever without
//       re-measuring) · synthetic c2 world (service scale T(2) 29.2 ms · draft 3.6 ms/part · first use +40 · capture
//       +40 ms — assumed): the MB1 rule gains, the previous rule (filters only captures · no re-measure · profit check
//       only after drafting) locks up on the first sample (negative control) · in the c4 (v2rfb scale) and costly-row
//       worlds MB1 loses <= 1% (profit check before drafting) · the previous rule does worse than MB1.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <tuple>
#include <random>
#include <vector>

#include "hive/mtp_gate.h"

using namespace hive::mtpg;

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)
static bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }

// Independent implementation: evaluate E(k)/T(k+1) for every k and take the maximum (smaller k on ties). F1: the
//   verify-cell cost is vcost (equal to choose_k outside a re-measure window) · old = true uses the pre-F1 formula
//   (cost — T(1) and the verify cells on one line), to reproduce the previous decisions
static int brute_k(const Gate& g, bool smp, const std::vector<float>& conf, int max_k, float cmin, int remaining, bool old = false) {
  const double t1 = g.cost(1);
  if (!(t1 > 0)) return 0;
  double best = 1.0 / t1; int bk = 0;
  for (int k = 1; k <= (int)conf.size(); ++k) {
    if (k > max_k || k > remaining || k + 1 > kMaxRows) break;
    bool ok = true; for (int i = 0; i < k; ++i) if (conf[i] < cmin) ok = false;
    if (!ok) break;
    double e = 1, ch = 1; for (int i = 0; i < k; ++i) { ch *= g.p_accept(smp, conf[i]); e += ch; }
    const double r = e / (old ? g.cost(k + 1) : g.vcost(k + 1));
    if (r > best) { best = r; bk = k; }
  }
  return bk;
}

int main() {
  // (1) cost table
  {
    Gate g;
    EXPECT(g.cost(1) < 0 && !g.known(), "empty table must be unknown");
    g.observe_step(1, 16.4);
    EXPECT(g.known() && near(g.cost(1), 16.4), "measured M=1");
    // one measured cell -> prior slope: T(M) = T1·(1 + 0.151·(M−1))
    EXPECT(near(g.cost(6), 16.4 * (1 + kPriorSlope * 5), 1e-9), "prior slope extrapolation %.3f", g.cost(6));
    g.observe_step(4, 23.5);
    EXPECT(near(g.cost(2), 16.4 + (23.5 - 16.4) / 3), "interpolation");
    EXPECT(near(g.cost(8), 23.5 + (23.5 - 16.4) / 3 * 4), "extrapolation with measured slope %.3f", g.cost(8));
    g.observe_step(4, -1); g.observe_step(4, NAN);
    EXPECT(g.n[4] == 1, "negative/NaN samples dropped");
    g.observe_step(4, 33.5);
    EXPECT(near(g.cost(4), 23.5 * (1 - kAlpha) + 33.5 * kAlpha), "EMA");
    // downward extrapolation (M=1 missing, e.g. dropped as stale)
    Gate h; h.observe_step(3, 20); h.observe_step(5, 24);
    EXPECT(near(h.cost(1), 16) && near(h.cost(2), 18), "downward extrapolation %.3f", h.cost(1));
    // freshness: T6 = 60 measured once, then only T1/T2 for 600 decisions -> T6 goes stale and is re-derived from the T1/T2 slope
    Gate s; s.observe_step(6, 60);
    for (int i = 0; i < 600; ++i) { s.tick(); s.observe_step(1, 16); if (i % 3 == 0) s.observe_step(2, 18); }
    EXPECT(!s.fresh(6) && near(s.cost(6), 18 + 2 * 4, 1e-6), "stale entry must be re-derived from fresh ones %.3f", s.cost(6));
    s.observe_step(6, 30);  // new sample on a stale cell -> replaced, not EMA-blended
    EXPECT(near(s.cost(6), 30), "stale entry replaced %.3f", s.cost(6));
    // with only one fresh cell (T1), use the slope of the stale measured cell (not the prior constant)
    Gate o; o.observe_step(4, 52); for (int i = 0; i < 600; ++i) { o.tick(); o.observe_step(1, 16); }
    EXPECT(near(o.cost(2), 16 + 12) && near(o.cost(4), 52), "one fresh point + stale measured slope %.3f", o.cost(2));
    // if everything is stale, stale cells are still used
    Gate a; a.observe_step(1, 10); for (int i = 0; i < 1000; ++i) a.tick();
    EXPECT(near(a.cost(1), 10) && near(a.cost(2), 10 * (1 + kPriorSlope)), "all-stale table still usable");
  }
  // (2) choose_k == exhaustive search
  {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-2.f, 9.f);
    for (int it = 0; it < 4000; ++it) {
      Gate g;
      const double t1 = 5 + (rng() % 300) / 10.0;
      g.observe_step(1, t1);
      if (rng() % 2) g.observe_step(2 + rng() % 6, t1 * (1 + (rng() % 200) / 100.0));
      if (rng() % 2) g.observe_step(2 + rng() % 6, t1 * (1 + (rng() % 200) / 100.0));
      const bool smp = rng() % 2;
      if (rng() % 2) { float c[5] = {U(rng), U(rng), U(rng), U(rng), U(rng)}; g.observe_accept(smp, c, 1 + rng() % 5, rng() % 2); }
      const int B = 1 + rng() % 7;
      std::vector<float> conf(B); for (auto& c : conf) c = U(rng);
      const int max_k = 1 + rng() % 8, remaining = 1 + rng() % 8;
      const float cmin = (rng() % 3) ? -1.f : 2.f;
      const int k = g.choose_k(smp, conf.data(), B, max_k, cmin, remaining);
      EXPECT(k == brute_k(g, smp, conf, max_k, cmin, remaining), "choose_k %d != brute %d", k, brute_k(g, smp, conf, max_k, cmin, remaining));
      EXPECT(k <= max_k && k <= remaining && k + 1 <= kMaxRows && k <= B, "bounds");
    }
    Gate g0; float c[3] = {9, 9, 9};
    EXPECT(g0.choose_k(false, c, 3, 3, -1, 10) == 0, "no T(1) → no speculation");
  }
  // (3) trap of the previous gate: service scale (T1 16.4 ms · draft 3 ms · measured 2-row verify 19 ms), old table
  //     ema_verify_ms[3..6] = 81/93/108/124 (its initial values)
  {
    const double T1 = 16.4, D = 3.0, V2 = 19.0;
    const double old_v[9] = {0, 0, V2, 81, 93, 108, 124, 138, 152};
    const float conf[5] = {7.5f, 7.5f, 7.5f, 7.5f, 7.5f};  // old table p = 0.90
    int k_old = 0; { double best = 1 / T1, ch = 1, e = 1; for (int i = 0; i < 5; ++i) { ch *= 0.90; e += ch; const double r = e / (D + old_v[i + 2]); if (r > best) { best = r; k_old = i + 1; } } }
    Gate g; g.observe_step(1, T1); g.observe_step(2, V2);
    const int k_new = g.choose_k(false, conf, 5, 5, -1, 100);
    EXPECT(k_old == 1, "old gate trapped at k=1 (got %d)", k_old);
    EXPECT(k_new == 5, "gate2 extrapolates from measured T1/T2 and picks k=5 (got %d, T6 %.1f)", k_new, g.cost(6));
    fprintf(stderr, "[trap] old k=%d  gate2 k=%d (T6 est %.1f ms vs stale 124)\n", k_old, k_new, g.cost(6));
    // the old formula gives k=0 at conf [1,2) (p 0.36) -> 8-step pause. v2 treats the draft cost as sunk, so
    // 1.36/T2 > 1/T1 <=> T2 < 22.3 ms gives k=1.
    const float lowc[1] = {1.5f};
    EXPECT(g.choose_k(false, lowc, 1, 5, -1, 100) == 1, "sunk draft cost: k=1 at p=0.36 when T2 < 1.36·T1");
  }
  // (4) acceptance learning
  {
    Gate g;
    EXPECT(near(g.p_accept(false, 1.5f), 0.36), "prior");
    float c[5] = {1.5f, 1.5f, 1.5f, 1.5f, 1.5f};
    for (int i = 0; i < 100; ++i) g.observe_accept(false, c, 5, 5);
    EXPECT(g.p_accept(false, 1.5f) > 0.97 && near(g.p_accept(true, 1.5f), 0.36), "learned greedy, sampled untouched %.3f", g.p_accept(false, 1.5f));
    for (int i = 0; i < 2000; ++i) g.observe_accept(true, c, 1, 0);
    EXPECT(g.p_accept(true, 1.5f) < 0.02, "rejections learned %.3f", g.p_accept(true, 1.5f));
    EXPECT(g.tried[1][2] <= 512, "counts halved");
    // conditional samples: n_cmp 3, n_acc 2 -> first two accepted, third rejected, fourth and fifth not compared
    Gate h; float d[5] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f}; h.observe_accept(false, d, 3, 2);
    EXPECT(h.hit[0][1] == 1 && h.hit[0][2] == 1 && h.tried[0][3] == 1 && h.hit[0][3] == 0 && h.tried[0][4] == 0 && h.tried[0][5] == 0, "conditional samples");
  }
  // (5) backoff
  {
    Backoff b;
    std::vector<int> skips;
    for (int i = 0; i < 5; ++i) { b.on_decline(2); skips.push_back(b.skip); while (!b.want()) {} }
    EXPECT(skips == std::vector<int>({1, 2, 4, 8, 8}), "decline backoff");
    b.on_spec(4, 30, 3, 16);  // expected saving 64 − 33 > 0
    EXPECT(b.declines == 0 && b.skip == 0, "spec resets");
    Backoff n; for (int i = 0; i < 4; ++i) n.on_spec(1.1, 20, 3, 16);  // expected loss already
    EXPECT(n.net < 0 && n.skip == 8, "negative net → skip 8");
  }
  // (6) synthetic world: generate 2000 tokens of one request through the gate, compare tokens/ms with plain decode
  {
    struct World { double t1, b, draft, p; };
    auto run = [&](const World& w, bool spec, unsigned seed) {
      std::mt19937 rng(seed);
      std::normal_distribution<double> nz(0, 0.03);
      std::uniform_real_distribution<double> U(0, 1);
      auto T = [&](int M) { return (w.t1 + w.b * (M - 1)) * (1 + nz(rng)); };
      Gate g; Backoff bo;
      double ms = 0; int tok = 0;
      while (tok < 2000) {
        g.tick();
        if (spec && g.known() && bo.want()) {
          ms += w.draft; g.observe_draft(w.draft);
          std::vector<float> conf(5, 4.5f);
          double ek = 0, ev = 0;
          const int k = g.choose_k(false, conf.data(), 5, 5, -1, 2000 - tok, &ek, &ev);
          if (k == 0) { bo.on_decline(w.draft, !g.can_profit(false, 5)); const double t = T(1); ms += t; g.observe_step(1, t); ++tok; continue; }
          int acc = 0; while (acc < k && U(rng) < w.p) ++acc;
          const int cmp = std::min(k, acc + 1);
          const double t = T(k + 1); ms += t;
          g.observe_step(k + 1, t); g.observe_accept(false, conf.data(), cmp, acc); bo.on_spec(ek, ev, w.draft, g.cost(1));
          tok += acc + 1;
        } else { const double t = T(1); ms += t; g.observe_step(1, t); ++tok; }
      }
      return tok / ms;
    };
    // service scale (assumed): T1 16.4 ms, +2.4 ms per row (layer chain 0.151 × 16.4 ≈ 2.5), draft 3 ms
    const World good{16.4, 2.4, 3.0, 0.8}, mid{16.4, 2.4, 3.0, 0.5}, bad{16.4, 2.4, 3.0, 0.05}, costly{16.4, 12.0, 6.0, 0.6};
    for (auto [w, name, min_gain] : {std::tuple{good, "good p0.8", 1.6}, std::tuple{mid, "mid p0.5", 1.15}, std::tuple{bad, "bad p0.05", 0.97}, std::tuple{costly, "costly rows", 0.97}}) {
      const double base = run(w, false, 1), sp = run(w, true, 1);
      fprintf(stderr, "[world %s] base %.1f tok/s · gate2 %.1f tok/s (x%.2f)\n", name, base * 1000, sp * 1000, sp / base);
      EXPECT(sp / base >= min_gain, "world %s gain %.3f < %.2f", name, sp / base, min_gain);
    }
  }
  // (7) HIVE_MTP_BATCH: choose_batch == exhaustive maximum rate (every k vector — prefix within each part · conf
  //     threshold · max_k · remaining tokens · row cap) · unknown T(S) -> 0 · row cap == S -> 0 · can_profit_batch
  //     (good/bad acceptance) · synthetic batch world (T(M) = a + b·(M−1), draft D per part, acceptance p)
  {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uc(-1.f, 8.f);
    for (int trial = 0; trial < 400; ++trial) {
      Gate g;
      const double a = 10 + (trial % 7) * 5, b = 0.5 + (trial % 5) * 1.5;
      for (int m = 1 + trial % 3; m <= kMaxRows; m += 2 + trial % 2) g.observe_step(m, a + b * (m - 1));
      const int S = 2 + trial % 3, max_rows = std::min(kMaxRows, S + 1 + trial % 6), max_k = 1 + trial % 5;
      const float cmin = (trial % 4 == 0) ? 2.f : -1.f;
      std::vector<std::vector<float>> cf(S);
      std::vector<const float*> cp(S);
      std::vector<int> nd(S), rem(S);
      bool smp[kMaxRows];
      for (int s = 0; s < S; ++s) {
        nd[s] = 1 + (trial + s) % 5; cf[s].resize(nd[s]); for (auto& x : cf[s]) x = uc(rng); cp[s] = cf[s].data();
        rem[s] = 1 + (trial * 3 + s) % 6; smp[s] = (trial + s) % 2 == 0;
      }
      const BatchChoice ch = choose_batch(g, S, smp, cp.data(), nd.data(), max_k, cmin, rem.data(), max_rows);
      // exhaustive search
      double best = S / g.cost(S);
      std::vector<int> k(S, 0);
      std::function<void(int)> rec = [&](int s) {
        if (s == S) {
          int K = 0; double e = 0;
          for (int t = 0; t < S; ++t) { K += k[t]; double c = 1; e += 1; for (int i = 0; i < k[t]; ++i) { c *= g.p_accept(smp[t], cf[t][i]); e += c; } }
          if (K > 0 && S + K <= max_rows) best = std::max(best, e / g.cost(S + K));
          return;
        }
        for (int v = 0; v <= std::min({nd[s], max_k, rem[s]}); ++v) {
          bool ok = true; for (int i = 0; i < v; ++i) if (cf[s][i] < cmin) ok = false;
          if (!ok) break;
          k[s] = v; rec(s + 1);
        }
        k[s] = 0;
      };
      rec(0);
      const double got = ch.K > 0 ? ch.e / ch.ms : S / g.cost(S);
      EXPECT(near(got, best, 1e-12 * std::max(1.0, best)), "choose_batch trial %d: rate %.9g != brute %.9g (K %d)", trial, got, best, ch.K);
      int ks = 0; for (int s = 0; s < S; ++s) { ks += ch.k[s]; EXPECT(ch.k[s] <= std::min({nd[s], max_k, rem[s]}), "k bound"); }
      EXPECT(ks == ch.K && S + ch.K <= max_rows, "K sum / rows cap");
    }
    Gate g0;
    bool smp2[2] = {false, false};
    const float c2[1] = {5.f};
    const float* cp2[2] = {c2, c2};
    int nd2[2] = {1, 1}, rem2[2] = {9, 9};
    EXPECT(choose_batch(g0, 2, smp2, cp2, nd2, 5, -1.f, rem2, 8).K == 0, "unknown T(S) → no speculation");
    g0.observe_step(2, 20);
    EXPECT(choose_batch(g0, 2, smp2, cp2, nd2, 5, -1.f, rem2, 2).K == 0, "rows cap == S → no speculation");
    EXPECT(choose_batch(g0, 2, smp2, cp2, nd2, 5, -1.f, rem2, 8).K > 0, "cheap rows (prior slope) + high conf → speculate");
    Gate gb; gb.observe_step(2, 20); gb.observe_step(3, 45); gb.observe_step(4, 70); gb.observe_draft(2);
    EXPECT(!can_profit_batch(gb, 2, false, 5, 4), "rows cost ≥ step each → no profit even at prior p");
    Gate gh; gh.observe_step(2, 20); gh.observe_step(4, 22); gh.observe_draft(1);
    EXPECT(can_profit_batch(gh, 2, false, 5, 8), "cheap verify + prior p → profitable");
    for (int i = 0; i < 200; ++i) { const float c[1] = {5.f}; gh.observe_accept(false, c, 1, 0); }  // every bucket sample is a rejection
    EXPECT(!can_profit_batch(gh, 2, false, 5, 8), "learned p ≈ 0 → hopeless");
    // synthetic batch world: each step S parts draft -> choose_batch -> verify (S+K rows) or plain step (S rows).
    // compare tokens/ms with plain batch decode.
    struct BW { int S; double a, b, draft, p; };
    auto run = [&](const BW& w, bool spec) {
      Gate g; Backoff bo; std::mt19937 r2(11); std::uniform_real_distribution<double> u(0, 1);
      auto T = [&](int M) { return w.a + w.b * (M - 1); };
      double ms = 0, tok = 0;
      for (int step = 0; step < 3000; ++step) {
        g.tick();
        if (!spec || !(g.cost(w.S) > 0) || !bo.want()) { const double t = T(w.S); ms += t; g.observe_step(w.S, t); tok += w.S; continue; }
        std::vector<std::vector<float>> cf(w.S, std::vector<float>(5, 5.f));
        std::vector<const float*> cp(w.S); for (int s = 0; s < w.S; ++s) cp[s] = cf[s].data();
        std::vector<int> nd(w.S, 5), rem(w.S, 1 << 20); bool sm[kMaxRows] = {};
        ms += w.draft * w.S; g.observe_draft(w.draft);
        const BatchChoice ch = choose_batch(g, w.S, sm, cp.data(), nd.data(), 5, -1.f, rem.data(), 8);
        if (ch.K == 0) { bo.on_decline(w.draft * w.S, !can_profit_batch(g, w.S, false, 5, 8)); const double t = T(w.S); ms += t; g.observe_step(w.S, t); tok += w.S; continue; }
        const double t = T(w.S + ch.K); ms += t; g.observe_step(w.S + ch.K, t);
        for (int s = 0; s < w.S; ++s) {
          int acc = 0; while (acc < ch.k[s] && u(r2) < w.p) ++acc;
          const int cmp = std::min(ch.k[s], acc + 1);
          g.observe_accept(false, cp[s], cmp, acc);
          tok += acc + 1;
        }
        bo.on_spec(ch.e, ch.ms, w.draft * w.S, g.cost(w.S) / w.S);
      }
      return tok / ms;
    };
    // assumed scale: c2 T(2) ≈ 28 ms · +6 ms per row (batch step c4 43 -> c8 72 ms gives (72−43)/4 ≈ 7) · draft 4 ms/part
    for (auto [w, name, min_gain] : {std::tuple{BW{2, 28, 6, 4, 0.8}, "c2 p0.8", 1.10}, std::tuple{BW{2, 28, 6, 4, 0.05}, "c2 p0.05", 0.95},
                                     std::tuple{BW{4, 43, 7, 4, 0.8}, "c4 p0.8", 0.95}, std::tuple{BW{3, 35, 7, 4, 0.6}, "c3 p0.6", 0.95}}) {
      const double base = run(w, false), sp = run(w, true);
      fprintf(stderr, "[batch world %s] base %.1f tok/s · batch gate %.1f tok/s (x%.2f)\n", name, base * 1000, sp * 1000, sp / base);
      EXPECT(sp / base >= min_gain, "batch world %s gain %.3f < %.2f", name, sp / base, min_gain);
    }
  }
  // pre-F1 can_profit: the same formula using cost (T(1) and the verify cells on one line)
  auto old_can_profit = [](const Gate& g, int max_k) {
    const double t1 = g.cost(1);
    if (!(t1 > 0)) return false;
    bool any = false; for (int b = 0; b < kBuckets; ++b) if (g.tried[0][b] > 0) any = true;
    double pm = 0; for (int b = 0; b < kBuckets; ++b) if (!any || g.tried[0][b] > 0) pm = std::max(pm, (g.hit[0][b] + kPriorW * prior_p(b)) / (g.tried[0][b] + kPriorW));
    double ch = 1, e = 1;
    for (int k = 1; k <= max_k && k + 1 <= kMaxRows; ++k) { ch *= pm; e += ch; const double tv = g.cost(k + 1); if (tv > 0 && e / tv > 1.0 / t1) return true; }
    return false;
  };
  // (8) F1: verify cells are not joined to T(1) (vcost) · re-measure once per window when every verify cell is stale ·
  //     loss bound in a world with the old verify path (large fixed cost)
  {
    // service log (multi-row verify path off): T1 20.0 · T2 51.9 · T6 112.3 ms. T2 was measured once at the start and
    // went stale; T1 and T6 are fresh.
    Gate g;
    g.observe_step(1, 20.0); g.observe_step(2, 51.9);
    for (int i = 0; i < 600; ++i) { g.tick(); g.observe_step(1, 20.0); if (i % 50 == 0) g.observe_step(6, 112.3); }
    float hc[1] = {7.5f};  // high-confidence bucket — learns 96/100 accepted (p ≈ 0.958)
    for (int i = 0; i < 100; ++i) g.observe_accept(false, hc, 1, i % 25 == 0 ? 0 : 1);
    EXPECT(!g.fresh(2) && g.fresh(6) && g.fresh(1), "scenario freshness");
    EXPECT(near(g.vcost(2), 51.9) && near(g.cost(2), 20.0 + (112.3 - 20.0) / 5, 1e-9), "vcost keeps measured T2 %.2f (cost interpolates T1..T6 %.2f)", g.vcost(2), g.cost(2));
    const std::vector<float> one{7.5f};
    const int k_old = brute_k(g, false, one, 5, -1.f, 100, true);
    const int k_new = g.choose_k(false, one.data(), 1, 5, -1.f, 100);
    EXPECT(k_old == 1, "scenario must reproduce the pre-F1 draft (old k %d)", k_old);
    EXPECT(k_new == 0, "measured T2 %.1f > (1+p)·T1 — must not draft (got k %d)", g.vcost(2), k_new);
    // the profit check uses the same cost: the old formula interpolates T2 = 38.5 -> "profitable" (backoff 8) ·
    // F1 says "hopeless" (backoff 32)
    const bool old_profit = old_can_profit(g, 5);
    EXPECT(old_profit && !g.can_profit(false, 5), "can_profit must use vcost (old %d new %d)", (int)old_profit, (int)g.can_profit(false, 5));
    // re-measure: when every verify cell is stale (only T1 measured), k = 1 once per window · none while any verify cell is fresh
    Gate q;
    q.observe_step(1, 20.0); q.observe_step(2, 51.9);
    for (int i = 0; i < 600; ++i) { q.tick(); q.observe_step(1, 20.0); }
    EXPECT(brute_k(q, false, one, 5, -1.f, 100, true) == 0, "pre-F1 gate never re-measures a stale-only table");
    EXPECT(q.probe_due() && q.choose_k(false, one.data(), 1, 5, -1.f, 100) == 1, "one probe when every verify cell is stale");
    EXPECT(q.choose_k(false, one.data(), 1, 5, -1.f, 100) == 0, "at most one probe per window");
    const float lowc[1] = {-0.5f};
    EXPECT(q.choose_k(false, lowc, 1, 5, 0.f, 100) == 0, "probe respects conf_min");
    q.observe_step(2, 51.9);
    for (int i = 0; i < 300; ++i) { q.tick(); q.observe_step(1, 20.0); }
    EXPECT(!q.probe_due() && q.choose_k(false, one.data(), 1, 5, -1.f, 100) == 0, "fresh T2 → no probe");
    for (int i = 0; i < 300; ++i) { q.tick(); q.observe_step(1, 20.0); }
    EXPECT(q.probe_due(), "next window after T2 goes stale again");
    // with no verify cell measured, the same prior-based exploration as before (T(1)·(1+0.151·(M−1)))
    Gate f; f.observe_step(1, 20.0);
    EXPECT(near(f.vcost(2), f.cost(2)) && near(f.vcost(6), f.cost(6)), "no verify cell → same prior seed as before");
    // synthetic world (old verify path): T(1) = 20 · T(M >= 2) = 51.9 + 15·(M−2) (first row +31.9 and ~15 per row, from
    //   the log above) · acceptance p · draft 3 ms. In this world no k beats a plain step (p <= 0.96: E(k)/T(k+1) < 1/T1)
    //   -> the gate's loss is only exploration, re-measure and draft cost. Bound: −0.5% vs plain decode.
    auto run = [&](double p, bool spec, bool old_rule, long& n_verify) {
      std::mt19937 rng(3);
      std::normal_distribution<double> nz(0, 0.03);
      std::uniform_real_distribution<double> U(0, 1);
      auto T = [&](int M) { return (M == 1 ? 20.0 : 51.9 + 15.0 * (M - 2)) * (1 + nz(rng)); };
      Gate gg; Backoff bo;
      double ms = 0; long tok = 0; n_verify = 0;
      const std::vector<float> cf(5, 7.5f);
      while (tok < 40000) {
        gg.tick();
        // same order as hived.cpp: backoff -> (F1) draft only when profitable or in a re-measure window · the old rule
        // checked profit only after drafting (always paid the draft cost)
        if (spec && gg.known() && bo.want() && (old_rule || gg.can_profit(false, 5) || gg.probe_due())) {
          ms += 3.0; gg.observe_draft(3.0);
          double ek = 0, ev = 0;
          const int k = old_rule ? brute_k(gg, false, cf, 5, -1.f, 1 << 20, true) : gg.choose_k(false, cf.data(), 5, 5, -1.f, 1 << 20, &ek, &ev);
          if (old_rule && k > 0) { double ch = 1; ek = 1; for (int i = 0; i < k; ++i) { ch *= gg.p_accept(false, 7.5f); ek += ch; } ev = gg.cost(k + 1); }
          if (k == 0) { bo.on_decline(3.0, !(old_rule ? old_can_profit(gg, 5) : gg.can_profit(false, 5))); const double t = T(1); ms += t; gg.observe_step(1, t); ++tok; continue; }
          int acc = 0; while (acc < k && U(rng) < p) ++acc;
          const double t = T(k + 1); ms += t; ++n_verify;
          gg.observe_step(k + 1, t); gg.observe_accept(false, cf.data(), std::min(k, acc + 1), acc); bo.on_spec(ek, ev, 3.0, gg.cost(1));
          tok += acc + 1;
        } else { const double t = T(1); ms += t; gg.observe_step(1, t); ++tok; }
      }
      return tok / ms;
    };
    for (double p : {0.96, 0.80, 0.30}) {
      long nv_base = 0, nv_new = 0, nv_old = 0;
      const double base = run(p, false, false, nv_base), nw = run(p, true, false, nv_new), od = run(p, true, true, nv_old);
      fprintf(stderr, "[F1 old-verify world p%.2f] base %.2f tok/s · gate2 %.2f (x%.4f, %ld verify) · pre-F1 rule %.2f (x%.4f, %ld verify)\n", p, base * 1000, nw * 1000, nw / base,
              nv_new, od * 1000, od / base, nv_old);
      EXPECT(nw / base >= 0.995, "old-verify world p%.2f: gate2 loses %.2f%% (> 0.5%%)", p, (1 - nw / base) * 100);
      EXPECT(nv_new <= 2 + 40000 / kStale * 2, "old-verify world p%.2f: %ld verify steps (bounded by one probe per window)", p, nv_new);
    }
  }
  // (9) MB1
  {
    // batch_probe_due: S = 4 · plain cell T(4) always fresh · verify cell T(5) measured once
    Gate g;
    g.observe_step(4, 33.0); g.observe_step(5, 47.5);
    EXPECT(!batch_probe_due(g, 4), "fresh verify cell → no probe");
    for (int i = 0; i < 600; ++i) { g.tick(); g.observe_step(4, 33.0); }
    EXPECT(!g.probe_due(), "Gate::probe_due is blind in a batch table (plain cell S=4 ≥ 2 is fresh) — why batch_probe_due takes S");
    EXPECT(batch_probe_due(g, 4), "every verify cell of S stale → probe due");
    EXPECT(!batch_probe_due(g, 5) && !batch_probe_due(g, 8), "S=5: no verify cell measured (5 is its plain cell) · S=8: no verify rows");
    EXPECT(!batch_probe_due(g, 4, 4), "rows cap = S → nothing to probe");
    // v2rfb trap: 4 parts · high draft confidence · learned acceptance <= 0.9 -> 5 rows 47.5 > (4+0.9)/4·33 = 40.4
    // -> the table alone gives K = 0
    bool smp[4] = {};
    const float hi[1] = {7.5f};
    const float* cp[4] = {hi, hi, hi, hi};
    int nd[4] = {1, 1, 1, 1}, rem[4] = {99, 99, 99, 99};
    Gate locked = g; locked.probe_at = locked.clock;  // this window's probe already used = every decision of the previous (no re-measure) rule
    EXPECT(choose_batch(locked, 4, smp, cp, nd, 5, -1.f, rem, 8).K == 0, "trap: table alone never drafts (T(5) 47.5)");
    EXPECT(!can_profit_batch(locked, 4, false, 5, 8), "trap: hopeless before drafting");
    const BatchChoice pr = choose_batch(g, 4, smp, cp, nd, 5, -1.f, rem, 8);
    EXPECT(pr.K == 1 && pr.ms == g.cost(5), "probe: K = 1 re-measures T(5) (K %d)", pr.K);
    EXPECT(choose_batch(g, 4, smp, cp, nd, 5, -1.f, rem, 8).K == 0, "probe at most once per window");
    const float lo[1] = {-0.5f};
    const float* cpl[4] = {lo, lo, lo, lo};
    Gate g2 = g; g2.probe_at = -(kStale + 1);
    EXPECT(choose_batch(g2, 4, smp, cpl, nd, 5, 0.f, rem, 8).K == 0, "probe respects conf_min");
    int rem0[4] = {0, 0, 0, 0};
    EXPECT(choose_batch(g2, 4, smp, cp, nd, 5, -1.f, rem0, 8).K == 0, "probe respects remaining tokens");
    // probe picks the part with the best first-draft acceptance
    const float mid[1] = {0.5f};
    const float* cpm[2] = {mid, hi};
    Gate g3; g3.observe_step(2, 29.2); g3.observe_step(3, 80.0);
    for (int i = 0; i < 600; ++i) { g3.tick(); g3.observe_step(2, 29.2); }
    const BatchChoice p3 = choose_batch(g3, 2, smp, cpm, nd, 5, -1.f, rem, 8);
    EXPECT(p3.K == 1 && p3.k[1] == 1 && p3.k[0] == 0, "probe on the highest-p part (k %d,%d)", p3.k[0], p3.k[1]);

    // synthetic c2 world: plain c2 step T(2) = a · verify T(2+K) = a + b·K · first use of a graph key (row count)
    //   +first (eager run) · second use +cap (capture) · draft D per part · acceptance p.
    //   new = MB1 (drop first-use and capture samples · re-measure · profit check before drafting) · old = previous rule
    //   (drop only captures · no re-measure · profit check only after drafting).
    struct W { int S; double a, b, D, p, first, cap; };
    auto run = [&](const W& w, int mode /*0 base · 1 new · 2 old*/, long* n_spec = nullptr) {
      Gate gg; Backoff bo; std::mt19937 r2(5); std::uniform_real_distribution<double> u(0, 1);
      int uses[kMaxRows + 1] = {};
      auto T = [&](int M, bool& dirty) { dirty = false; double t = M == w.S ? w.a : w.a + w.b * (M - w.S); if (M > w.S) { if (uses[M] == 0) { t += w.first; dirty = true; } else if (uses[M] == 1) { t += w.cap; dirty = mode == 1 || mode == 2; } ++uses[M]; } return t; };
      // dirty: only MB1 recognizes the first use (eager run); the old rule only captures — split by mode below
      double ms = 0, tok = 0; long ns = 0;
      std::vector<float> cf(5, 7.5f);
      for (int step = 0; step < 6000; ++step) {
        gg.tick();
        bool dirty;
        const int S = w.S;
        bool go = mode != 0 && gg.cost(S) > 0;
        if (go && mode == 1) go = can_profit_batch(gg, S, false, 5, 8) || batch_probe_due(gg, S, 8);
        if (go) go = bo.want();
        if (!go) { const double t = T(S, dirty); ms += t; gg.observe_step(S, t); tok += S; continue; }
        ms += w.D * S; gg.observe_draft(w.D);
        const float* cp2[kMaxRows]; int nd2[kMaxRows], rm2[kMaxRows]; bool sm2[kMaxRows] = {};
        for (int s = 0; s < S; ++s) { cp2[s] = cf.data(); nd2[s] = 5; rm2[s] = 1 << 20; }
        Gate* gsel = &gg;
        if (mode == 2) gg.probe_at = gg.clock;  // old rule: no re-measure
        const BatchChoice ch = choose_batch(*gsel, S, sm2, cp2, nd2, 5, -1.f, rm2, 8);
        if (ch.K == 0) { bo.on_decline(w.D * S, !can_profit_batch(gg, S, false, 5, 8)); const double t = T(S, dirty); ms += t; gg.observe_step(S, t); tok += S; continue; }
        const int M = S + ch.K;
        const bool first = uses[M] == 0;
        const double t = T(M, dirty); ms += t; ++ns;
        const bool drop = mode == 1 ? (dirty || first) : (dirty && !first);  // MB1: first use and capture · old rule: capture only
        if (!drop) gg.observe_step(M, t);
        for (int s = 0; s < S; ++s) {
          int acc = 0; while (acc < ch.k[s] && u(r2) < w.p) ++acc;
          gg.observe_accept(false, cf.data(), std::min(ch.k[s], acc + 1), acc);
          tok += acc + 1;
        }
        bo.on_spec(ch.e, ch.ms, w.D * S, gg.cost(S) / S);
      }
      if (n_spec) *n_spec = ns;
      return tok / ms;
    };
    // service scale (measured): c2 plain step 29.2 ms (median of 20,782 steps) · draft 3.6 ms (256,379 ms / 70,645
    //   drafts) · single-sequence verify +5.7–7 ms per row (T2 27.8 · T4 39.2) -> b = 6.
    //   p = 0.85 (high-confidence drafts) — the overall service acceptance of 75% is an average over the k the gate
    //   picked. First use +25 · capture +40 ms are assumptions (not measured on GPU — only the direction matters).
    {
      const W good{2, 29.2, 6.0, 3.6, 0.85, 40, 40};
      long ns_new = 0, ns_old = 0;
      const double base = run(good, 0), nw = run(good, 1, &ns_new), od = run(good, 2, &ns_old);
      fprintf(stderr, "[MB1 c2 world p0.85] base %.1f tok/s · MB1 %.1f (x%.3f, %ld verify) · pre-MB1 %.1f (x%.3f, %ld verify)\n", base * 1000, nw * 1000, nw / base, ns_new,
              od * 1000, od / base, ns_old);
      EXPECT(nw / base >= 1.08, "MB1 c2 world gain %.3f < 1.08", nw / base);
      // negative control: the old rule takes the first 8-row verify (eager run +40) as a sample, sees T(3..8) as the
      // T(2)..T(8)=105 interpolation (+12.6 per row) and locks up
      EXPECT(ns_old <= 3 && od / base < 1.0, "pre-MB1 rule must lock out after the first-use sample (verify %ld · x%.3f)", ns_old, od / base);
    }
    {
      // v2rfb scale (c4): T(4) 33 ms · +5 per row · first use +10 · capture +40 · draft 3.7 ms/part (4 parts = 14.8 ms —
      //   45% of a step). Because of the draft cost no K beats a plain c4 step in this world (7.4 tok / (14.8 + 53) ms
      //   < 4 / 33) -> MB1 loses <= 1% (the pre-draft profit check stops it); the old rule drafts after every backoff
      //   and loses more.
      const W v2rfb{4, 33.0, 5.0, 3.7, 0.85, 10, 40};
      long ns_new = 0, ns_old = 0;
      const double base = run(v2rfb, 0), nw = run(v2rfb, 1, &ns_new), od = run(v2rfb, 2, &ns_old);
      fprintf(stderr, "[MB1 c4 v2rfb world] base %.1f tok/s · MB1 %.1f (x%.3f, %ld verify) · pre-MB1 %.1f (x%.3f, %ld verify)\n", base * 1000, nw * 1000, nw / base, ns_new,
              od * 1000, od / base, ns_old);
      EXPECT(nw / base >= 0.99, "MB1 c4 world loses %.2f%% (> 1%%)", (1 - nw / base) * 100);
      EXPECT(od <= nw, "pre-MB1 rule does not beat MB1 (%.4f vs %.4f)", od / base, nw / base);
    }
    {
      const W costly{2, 29.2, 16.0, 3.6, 0.85, 25, 40};  // rows are expensive, no K wins -> loss is only exploration, re-measure and draft cost
      long ns_new = 0, ns_old = 0;
      const double base = run(costly, 0), nw = run(costly, 1, &ns_new), od = run(costly, 2, &ns_old);
      fprintf(stderr, "[MB1 costly-rows world] base %.1f tok/s · MB1 %.1f (x%.4f, %ld verify) · pre-MB1 %.1f (x%.4f, %ld verify)\n", base * 1000, nw * 1000, nw / base, ns_new,
              od * 1000, od / base, ns_old);
      EXPECT(nw / base >= 0.99, "MB1 costly world loses %.2f%% (> 1%%)", (1 - nw / base) * 100);
      EXPECT(od < nw, "pre-MB1 rule (drafts after every backoff) loses more than MB1 (%.4f vs %.4f)", od / base, nw / base);
    }
  }
  if (fails) { fprintf(stderr, "test_mtp_gate_cpu: %d FAILED\n", fails); return 1; }
  fprintf(stderr, "test_mtp_gate_cpu: all passed\n");
  return 0;
}
