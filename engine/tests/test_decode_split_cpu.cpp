// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_SPLIT=balance — CPU test of the decode miss split cost model (hive/decode_split.h). No GPU.
//   (1) switch parsing, (2) table (interpolation, extrapolation direction, EMA, discard), (3) cold start / failed validation = exactly the baseline rule's
//   count, (4) on synthetic timings the chosen n = exhaustive minimum (ties → more DMA), never above the cap (staging) or the miss count, (5) the error
//   margin only pushes toward DMA, (6) synthetic-world simulation: once the model has learned, the mean layer end is shorter than the baseline rule;
//   with a G2-style failure (CPU superlinear in job count) it does not over-shift to the CPU; with heavy noise it falls back to the baseline rule,
//   (7) skew learning, negative samples discarded.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "hive/decode_split.h"

using namespace hive::dsplit;

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)
static bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }

// Synthetic world (mimics the measured scale on the target workstation — numbers for testing the model only): CPU 1/2/3 jobs 0.33/0.6/0.8 ms, staging DMA ~0.67 ms per expert (+ wait)
struct World {
  double cpu0 = 0.10, cpu1 = 0.235, cpu_pow = 1.0;       // cpu(j) = cpu0 + cpu1 · j^pow
  double cp0 = 0.05, cp1 = 0.67, dg1 = 0.02, res0 = 0.03, res1 = 0.016, skew = 0.02;
  double noise = 0.0;
  double cpu(int j) const { return j > 0 ? cpu0 + cpu1 * std::pow((double)j, cpu_pow) : 0.0; }
  double gpu(int ng_hit, int n, int ngs) const { const double r = res0 + res1 * ng_hit; return n > 0 ? std::max(r, cp0 + cp1 * n) + dg1 * ngs : r; }
};
// Baseline rule (runtime.cpp: misses >= 3 → min(cap, max(decode_gpu_share = 1, round(frac × misses)))), frac = serving default 0.25 (kDmaFracDefault[kDmaDecode])
static int base_rule(int N, int cap, float frac = 0.25f) { return N >= 3 ? std::min(cap, std::max(1, (int)(frac * (float)N + 0.5f))) : 0; }

// Result of executing split n in the world (multiplicative noise) → Obs
static Obs run(const World& w, const Input& in, const Decision& d, std::mt19937& rng) {
  std::normal_distribution<double> nz(0.0, w.noise);
  auto f = [&](double x) { return w.noise > 0 ? std::max(0.01, x * (1.0 + nz(rng))) : x; };
  Obs o;
  const int n = d.n_dma;
  o.ng_hit = in.ng_hit; o.n_str = n; o.ng_str = groups_of(in, n); o.jobs = jobs_of(in, n);
  o.res_ms = f(w.res0 + w.res1 * in.ng_hit);
  if (n > 0) { o.cp_ms = f(w.cp0 + w.cp1 * n); o.gpu_ms = std::max(o.res_ms, o.cp_ms) + f(w.dg1 * o.ng_str); }
  else o.gpu_ms = o.res_ms;
  if (o.jobs > 0) { o.cpu_ms = f(w.cpu(o.jobs)); o.end_ms = std::max(o.cpu_ms + w.skew, o.gpu_ms); }
  o.has_pred = d.has_pred; o.pred = d.pred;
  return o;
}
static Input rnd_input(std::mt19937& rng, int M, int cap) {
  Input in;
  std::uniform_int_distribution<int> nm(0, M == 1 ? 4 : 12), rows(1, std::max(1, M / 2));
  const int N = nm(rng);
  for (int i = 0; i < N; ++i) in.rows.push_back(rows(rng));
  std::sort(in.rows.begin(), in.rows.end(), std::greater<int>());  // baseline order (rows descending)
  in.ng_hit = 6 * M - N > 0 ? std::min(6 * M - N, 6 * M) : 0;
  in.cap = cap; in.base_n = std::min(base_rule(N, cap), N); in.max_rows = 8;
  return in;
}
struct SimOut { double end_model = 0, end_base = 0, tail_model = 0, tail_base = 0; int why[3] = {0, 0, 0}; int dma_model = 0, dma_base = 0; };
// Same input sequence through (a) the model path (decide → execute → observe) and (b) the baseline rule (same world) side by side — only the second half is counted (after learning)
static SimOut simulate(const World& w, int M, int layers, unsigned seed) {
  std::mt19937 rng(seed), rng_b(seed + 1), rng_in(seed + 2);
  Model m;
  SimOut s;
  for (int i = 0; i < layers; ++i) {
    const Input in = rnd_input(rng_in, M, 8);
    if (in.rows.empty()) continue;
    const Decision d = decide(m, in);
    const Obs o = run(w, in, d, rng);
    add_cpu(m, o.jobs, o.cpu_ms);
    observe(m, o);
    Decision db; db.n_dma = in.base_n;
    const Obs ob = run(w, in, db, rng_b);
    if (i >= layers / 2) {
      const double e = o.end_ms >= 0 ? o.end_ms : o.gpu_ms, eb = ob.end_ms >= 0 ? ob.end_ms : ob.gpu_ms;
      s.end_model += e; s.end_base += eb; s.tail_model += std::max(0.0, e - o.gpu_ms); s.tail_base += std::max(0.0, eb - ob.gpu_ms);
      ++s.why[d.why]; s.dma_model += d.n_dma; s.dma_base += in.base_n;
    }
  }
  return s;
}

int main() {
  // (1) switch
  EXPECT(parse_mode("balance") == kBalance && parse_mode("BaLaNcE") == kBalance, "balance");
  for (const char* v : {(const char*)nullptr, "", "0", "1", "bal", "balanced", "on"}) EXPECT(parse_mode(v) == kOff, "off for %s", v ? v : "(null)");
  EXPECT(cls_of(1) == 0 && cls_of(8) == 7 && cls_of(9) == 8 && cls_of(64) == 8 && cls_of(0) == 0, "cls_of");

  // (2) table
  {
    Tab t;
    EXPECT(!t.any() && t.at(3, true) == 0.0, "empty tab");
    t.add(2, 1.0);
    EXPECT(t.any() && near(t.at(2, true), 1.0), "exact");
    EXPECT(near(t.at(4, true), 2.0) && near(t.at(4, false), 2.0), "one point above: proportional");
    EXPECT(near(t.at(1, true), 1.0) && near(t.at(1, false), 0.5), "one point below: high flat, low proportional");
    t.add(4, 1.4);  // two points: slope 0.2
    EXPECT(near(t.at(3, true), 1.2) && near(t.at(3, false), 1.2), "interp");
    EXPECT(near(t.at(6, false), 1.8) && near(t.at(6, true), 1.4 * 6 / 4), "above: low = linear 1.8, high = max(linear, prop 2.1)");
    EXPECT(near(t.at(1, false), 0.8) && near(t.at(1, true), 1.0), "below: low = linear 0.8, high = flat 1.0");
    EXPECT(near(t.at(0, false), 0.6), "below to 0: linear");
    EXPECT(t.at(-1, true) == 0.0, "negative n");
    EXPECT(near(t.at(200, false), 1.4 + 0.2 * 196), "beyond table: extrapolate");
    t.add(2, 2.0);  // EMA: 1 + 0.25·(2 − 1)
    EXPECT(near(t.v[2], 1.25), "ema %.4f", t.v[2]);
    t.add(3, -1.0); t.add(3, NAN); t.add(3, INFINITY); t.add(kN, 1.0); t.add(-1, 1.0);
    EXPECT(!t.has[3] && t.seen == 2, "reject negative/NaN/inf/out-of-range");
  }

  // (3) cold start / failed validation
  {
    Model m;
    Input in; in.rows = {2, 1, 1, 1}; in.ng_hit = 4; in.cap = 8; in.base_n = 1; in.max_rows = 8;
    Decision d = decide(m, in);
    EXPECT(d.why == kCold && d.n_dma == 1 && !d.has_pred, "cold = base");
    in.base_n = 20; in.cap = 3;
    d = decide(m, in);
    EXPECT(d.n_dma == 3, "cold base clamped to cap: %d", d.n_dma);
    in.cap = 8; in.base_n = 20;
    EXPECT(decide(m, in).n_dma == 4, "cold base clamped to N");
    m.cpu.add(1, 0.33); m.res.add(4, 0.1); m.cp.add(1, 0.7);
    EXPECT(decide(m, in).why == kCold, "warm needs all four tables");
    m.dg.add(1, 0.02);
    in.base_n = 1;
    d = decide(m, in);
    EXPECT(d.why == kUnsure && d.n_dma == 1 && d.has_pred && d.pred.n == 1, "warm, unvalidated = base with prediction");
    m.n_val_c = m.n_val_g = kMinVal; m.err_c = 0.1; m.err_g = kErrMax + 0.01;
    EXPECT(decide(m, in).why == kUnsure, "gpu error too large = base");
    m.err_g = 0.1;
    EXPECT(decide(m, in).why == kModel, "validated = model");
  }

  // (4) synthetic timings: chosen n = exhaustive minimum over [0, cap], ties → more DMA, cap
  {
    World w;
    Model m;
    for (int j = 1; j <= 8; ++j) m.cpu.add(j, w.cpu(j));
    for (int g = 0; g <= 48; ++g) m.res.add(g, w.res0 + w.res1 * g);
    for (int n = 1; n <= 8; ++n) m.cp.add(n, w.cp0 + w.cp1 * n);
    for (int g = 1; g <= 8; ++g) m.dg.add(g, w.dg1 * g);
    m.n_val_c = m.n_val_g = kMinVal;
    Input in; in.max_rows = 8; in.cap = 8;
    in.rows = {1}; in.ng_hit = 5; in.base_n = 0;
    Decision d = decide(m, in);
    EXPECT(d.why == kModel && d.n_dma == 0, "1 miss → CPU (0.33 < DMA 0.72+): n=%d", d.n_dma);
    std::mt19937 rng(5);
    for (int it = 0; it < 500; ++it) {
      in = rnd_input(rng, 1 + (int)(rng() % 8), 1 + (int)(rng() % 8));
      if (in.rows.empty()) continue;
      d = decide(m, in);
      const int N = (int)in.rows.size(), cap = std::min(in.cap, N);
      EXPECT(d.n_dma >= 0 && d.n_dma <= cap, "n=%d outside [0,%d]", d.n_dma, cap);
      double best = 1e30; int best_n = -1;
      for (int n = 0; n <= cap; ++n) { const double e = predict(m, in, n).end; if (e < best || (e == best && n > best_n)) { best = e; best_n = n; } }
      EXPECT(d.n_dma == best_n && near(d.pred.end, best), "argmin: got %d want %d", d.n_dma, best_n);
    }
    // tie: a table with both CPU and GPU near 0 → identical end times everywhere → maximum DMA
    Model z;
    z.cpu.add(1, 0.0); z.res.add(0, 1.0); z.cp.add(1, 0.0); z.dg.add(1, 0.0);
    z.n_val_c = z.n_val_g = kMinVal;
    Input tz; tz.rows = {1, 1, 1}; tz.ng_hit = 0; tz.cap = 8; tz.base_n = 0;
    EXPECT(decide(z, tz).n_dma == 3, "tie → most DMA");
    // rows > 8 (batch 64): job and group counting
    Input big; big.rows = {10, 9, 1}; big.max_rows = 8;
    EXPECT(jobs_of(big, 0) == 5 && jobs_of(big, 1) == 3 && jobs_of(big, 3) == 0 && groups_of(big, 2) == 4, "jobs/groups for rows > 8");
    // (5) the error margin (err_c) never reduces the DMA count (monotone) — same input with err_c 0 → 0.3
    for (int it = 0; it < 300; ++it) {
      in = rnd_input(rng, 1 + (int)(rng() % 8), 8);
      if (in.rows.empty()) continue;
      Model a = m, b = m;
      a.err_c = 0.0; b.err_c = 0.3;
      EXPECT(decide(b, in).n_dma >= decide(a, in).n_dma, "err margin must push toward DMA");
    }
  }

  // (6) world simulation
  {
    World w;
    for (int M : {1, 4, 8}) {
      const SimOut s = simulate(w, M, 4000, 100 + M);
      printf("decode_split sim M=%d: mean layer end model %.3f vs base %.3f ms · tail %.3f vs %.3f · dma/layer %.2f vs %.2f · model %d cold %d unsure %d\n", M,
             s.end_model / 2000, s.end_base / 2000, s.tail_model / 2000, s.tail_base / 2000, (double)s.dma_model / 2000, (double)s.dma_base / 2000, s.why[kModel],
             s.why[kCold], s.why[kUnsure]);
      EXPECT(s.why[kModel] > 0, "model never engaged M=%d", M);
      EXPECT(s.end_model <= s.end_base * 1.001, "model worse than base M=%d: %.3f vs %.3f", M, s.end_model, s.end_base);
    }
    // Reproduce the G2-style failure: CPU superlinear in job count (bandwidth contention) — the table is measured per count, so the mean slope does not underestimate the CPU
    World g2 = w;
    g2.cpu_pow = 1.5;
    const SimOut s = simulate(g2, 8, 4000, 7);
    printf("decode_split sim G2-like (superlinear CPU) M=8: model %.3f vs base %.3f ms · dma/layer %.2f vs %.2f\n", s.end_model / 2000, s.end_base / 2000,
           (double)s.dma_model / 2000, (double)s.dma_base / 2000);
    EXPECT(s.end_model <= s.end_base * 1.001, "superlinear CPU: model worse than base");
    // Noisy world: validation error exceeds kErrMax, so it mostly falls back to the baseline rule (and is not much worse than base)
    World nz = w;
    nz.noise = 0.8;
    const SimOut sn = simulate(nz, 4, 4000, 9);
    printf("decode_split sim noise 0.8 M=4: model %d unsure %d · model %.3f vs base %.3f ms\n", sn.why[kModel], sn.why[kUnsure], sn.end_model / 2000, sn.end_base / 2000);
    EXPECT(sn.why[kUnsure] > 4 * sn.why[kModel], "noisy world should fall back: model %d unsure %d", sn.why[kModel], sn.why[kUnsure]);
  }

  // (7) skew, negative samples
  {
    Model m;
    Obs o; o.ng_hit = 3; o.n_str = 1; o.ng_str = 1; o.jobs = 1; o.res_ms = 0.1; o.cp_ms = 0.2; o.gpu_ms = 0.15;  // gpu < max(res, cp) → negative dg → discarded
    o.cpu_ms = 0.3; o.end_ms = 0.35;
    observe(m, o);
    EXPECT(!m.dg.any() && m.cp.has[1] && m.res.has[3], "negative dg dropped");
    EXPECT(m.has_skew && near(m.skew, 0.05), "skew from CPU-last layer: %.4f", m.skew);
    Obs q = o; q.end_ms = q.gpu_ms + kTailEps / 2; q.gpu_ms = 0.3;
    observe(m, q);
    EXPECT(near(m.skew, 0.05), "GPU-last layer does not move skew");
    EXPECT(m.n_val_c == 0 && m.n_val_g == 0, "no prediction → no validation");
  }
  printf("test_decode_split_cpu: %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
