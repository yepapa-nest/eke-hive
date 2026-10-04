// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU test of the HIVE_PREFILL_SPLIT=balance cost model (hive/prefill_split.h) — no GPU.
//   ① switch parsing (only "balance" turns it on) ② cold start = 0.7 (the fixed-fraction rounding formula) ③ share limits [0.5, 0.98] ④ row cap = the fixed CPU-share loop ⑤ EMA sample absorption
//   ⑥ iterating layers on synthetic times (true cost + noise) converges the decision near the true optimum with prediction ≈ actual ⑦ decision = minimum of an exhaustive search over the allowed range.
//   Build: g++ -std=c++20 -O1 -Iengine/include engine/tests/test_prefill_split_cpu.cpp -o /tmp/t && /tmp/t
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "hive/prefill_split.h"

using namespace hive::ps;

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { if (fails < 30) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } ++fails; } } while (0)

// synthetic layer: N misses (rows ascending), job count = ceil(rows/8) (one sub-chunk)
static std::vector<Miss> make_layer(std::mt19937& rng, int N) {
  std::vector<Miss> m((size_t)N);
  std::geometric_distribution<int> g(0.08);
  for (auto& x : m) { x.rows = 1 + g(rng); x.jobs = (x.rows + 7) / 8; }
  std::sort(m.begin(), m.end(), [](const Miss& a, const Miss& b) { return a.rows < b.rows; });
  return m;
}
struct Truth { double job, rec, row; };
static void actual(const Truth& t, const std::vector<Miss>& m, int gpu_fixed, int n_cpu, double& cpu, double& gpu) {
  double jobs = 0, rows_gpu = gpu_fixed;
  for (int i = 0; i < (int)m.size(); ++i) { if (i < n_cpu) jobs += m[(size_t)i].jobs; else rows_gpu += m[(size_t)i].rows; }
  cpu = t.job * jobs;
  gpu = std::max(t.rec * (double)((int)m.size() - n_cpu), t.row * rows_gpu);
}

int main() {
  // ① switch — unset / empty / "0" / anything else = off (absorbed), only "balance" turns it on (case-insensitive)
  EXPECT(parse_mode(nullptr) == kOff, "null");
  EXPECT(parse_mode("") == kOff, "empty");
  EXPECT(parse_mode("0") == kOff, "0");
  EXPECT(parse_mode("1") == kOff, "1 is not a mode name");
  EXPECT(parse_mode("balanced") == kOff, "balanced");
  EXPECT(parse_mode("balance ") == kOff, "trailing space");
  EXPECT(parse_mode("fixed") == kOff, "fixed");
  EXPECT(parse_mode("balance") == kBalance, "balance");
  EXPECT(parse_mode("BALANCE") == kBalance, "BALANCE");
  EXPECT(parse_mode("Balance") == kBalance, "Balance");

  std::mt19937 rng(20260930);
  // ② cold start: 0.7 if any of the three costs is missing — the fixed formula (int)(0.7·N + 0.5)
  for (int N : {1, 2, 3, 7, 10, 99, 100, 235, 400}) {
    auto m = make_layer(rng, N);
    for (int mask = 0; mask < 7; ++mask) {  // 7 = all three present → excluded
      Costs c; c.job_ms = 1; c.rec_ms = 1; c.row_ms = 1;
      c.has_job = mask & 1; c.has_rec = mask & 2; c.has_row = mask & 4;
      const Decision d = decide(c, m, 1000, 1 << 30);
      EXPECT(d.cold, "cold N=%d mask=%d", N, mask);
      EXPECT(d.n_dma == n_dma_of(0.7f, N) && d.n_cpu == N - d.n_dma, "cold split N=%d: n_dma %d want %d", N, d.n_dma, n_dma_of(0.7f, N));
    }
  }
  { Decision d = decide(Costs{}, {}, 10, 100); EXPECT(d.cold && d.share == kCold && d.n_dma == 0 && d.n_cpu == 0, "N=0"); }

  // ③ limits: a very slow CPU reaches 0.98, a very fast one 0.5
  for (int N : {20, 50, 100, 235}) {
    auto m = make_layer(rng, N);
    Costs slow_cpu{1e3, 0.6, 0.001, true, true, true};
    Decision d = decide(slow_cpu, m, 500, 1 << 30);
    EXPECT(!d.cold && d.n_dma == n_dma_of(kHi, N), "hi clamp N=%d n_dma %d want %d", N, d.n_dma, n_dma_of(kHi, N));
    Costs fast_cpu{1e-6, 0.6, 0.001, true, true, true};
    d = decide(fast_cpu, m, 500, 1 << 30);
    EXPECT(!d.cold && d.n_dma == n_dma_of(kLo, N), "lo clamp N=%d n_dma %d want %d", N, d.n_dma, n_dma_of(kLo, N));
    EXPECT(d.share >= kLo - 1e-6f && d.share <= 1.f, "share range");
  }

  // ④ row cap: the CPU share is the prefix with cumulative rows ≤ cap (the break of the fixed loop) — the rest goes to DMA
  {
    std::vector<Miss> m;
    for (int i = 0; i < 40; ++i) m.push_back({10 + i, 2});
    Costs c{1e-6, 1.0, 0.001, true, true, true};  // free CPU → wants the maximum CPU share (share 0.5 → 20 on the CPU)
    Decision d = decide(c, m, 0, 55);  // 10+11+12+13 = 46 ≤ 55 < 60
    EXPECT(d.n_cpu == 4 && d.n_dma == 36, "row cap: n_cpu %d n_dma %d", d.n_cpu, d.n_dma);
    EXPECT(capped_cpu(m, 20, 55) == 4 && capped_cpu(m, 3, 55) == 3 && capped_cpu(m, 0, 55) == 0, "capped_cpu");
  }

  // ⑤ EMA: first sample = its value, 0 / negative / NaN / inf are dropped, then kAlpha weighting
  {
    double v = 0; bool h = false;
    ema(v, h, 0.0); ema(v, h, -1.0); ema(v, h, NAN); ema(v, h, INFINITY);
    EXPECT(!h && v == 0, "ema rejects");
    ema(v, h, 2.0); EXPECT(h && v == 2.0, "ema first");
    ema(v, h, 4.0); EXPECT(std::fabs(v - (2.0 + kAlpha * 2.0)) < 1e-12, "ema step");
    for (int i = 0; i < 200; ++i) ema(v, h, 1.0);
    EXPECT(std::fabs(v - 1.0) < 1e-9, "ema converges");
  }

  // ⑦ decision = minimum predicted max(CPU, GPU) from an exhaustive search over [lo, hi] · the CPU prediction increases monotonically in n_cpu, the GPU one decreases (crossover)
  for (int it = 0; it < 3000; ++it) {
    const int N = 1 + (int)(rng() % 300);
    auto m = make_layer(rng, N);
    std::uniform_real_distribution<double> u(0.01, 2.0);
    Costs c{u(rng), u(rng), u(rng) * 1e-3, true, true, true};
    const int gf = (int)(rng() % 3000), cap = (it % 5 == 0) ? (int)(rng() % 200) : (1 << 30);
    const Decision d = decide(c, m, gf, cap);
    double best = 1e300;
    for (int nd = n_dma_of(kLo, N); nd <= n_dma_of(kHi, N); ++nd) {
      const int nc = capped_cpu(m, N - nd, cap);
      double pc, pg;
      actual(Truth{c.job_ms, c.rec_ms, c.row_ms}, m, gf, nc, pc, pg);
      best = std::min(best, std::max(pc, pg));
    }
    EXPECT(std::fabs(std::max(d.pred_cpu_ms, d.pred_gpu_ms) - best) <= 1e-9 * std::max(1.0, best), "argmin it=%d: got %.6f best %.6f", it, std::max(d.pred_cpu_ms, d.pred_gpu_ms), best);
    double pc, pg;
    actual(Truth{c.job_ms, c.rec_ms, c.row_ms}, m, gf, d.n_cpu, pc, pg);
    EXPECT(std::fabs(pc - d.pred_cpu_ms) < 1e-9 * std::max(1.0, pc) && std::fabs(pg - d.pred_gpu_ms) < 1e-9 * std::max(1.0, pg), "pred matches model it=%d", it);
    EXPECT(d.n_cpu + d.n_dma == N && d.n_cpu == capped_cpu(m, d.n_cpu, cap), "consistent it=%d", it);
  }

  // ⑥ convergence: cold start without knowing the true costs (two layer types with different values) → decide per layer → feed synthetic actual times (±5 % noise) as samples → decisions approach the true optimum, prediction ≈ actual.
  //   Keeping separate costs per type (runtime.cpp split_*_[type]) prevents cross-contamination — mixing them into one model shifts the converged value (checked as a control).
  {
    const Truth truth[2] = {{1.20, 0.64, 0.0020}, {3.00, 0.64, 0.0040}};  // synthetic values (not measured) — CPU and compute costs differ per type
    Costs model[2], mixed;
    std::normal_distribution<double> noise(0.0, 0.05);
    double worst_late_gap = 0, worst_late_pred = 0, mixed_late_gap = 0;
    int cold_layers = 0;
    for (int layer = 0; layer < 400; ++layer) {
      const int t = layer % 5 == 4 ? 1 : 0;
      const int N = t == 0 ? 180 + (int)(rng() % 80) : 30 + (int)(rng() % 30);
      auto m = make_layer(rng, N);
      const int gf = t == 0 ? 3000 + (int)(rng() % 1500) : 200 + (int)(rng() % 100);
      for (int pass = 0; pass < 2; ++pass) {
        Costs& c = pass == 0 ? model[t] : mixed;
        const Decision d = decide(c, m, gf, 1 << 30);
        if (pass == 0 && d.cold) ++cold_layers;
        if (pass == 0 && (layer == 0 || layer == 4)) EXPECT(d.cold && d.n_dma == n_dma_of(kCold, N), "first layer of each type is cold");  // the first sample fills all three
        double cpu, gpu;
        actual(truth[t], m, gf, d.n_cpu, cpu, gpu);
        cpu *= 1 + noise(rng); gpu *= 1 + noise(rng);
        double jobs = 0; for (int i = 0; i < d.n_cpu; ++i) jobs += m[(size_t)i].jobs;
        if (jobs > 0) ema(c.job_ms, c.has_job, cpu / jobs);
        if (d.n_dma > 0) ema(c.rec_ms, c.has_rec, truth[t].rec * (1 + noise(rng)));
        ema(c.row_ms, c.has_row, truth[t].row * (1 + noise(rng)));
        if (layer >= 300) {
          // true time of the chosen split relative to the true optimum (noise-free)
          double best = 1e300;
          for (int nd = n_dma_of(kLo, N); nd <= n_dma_of(kHi, N); ++nd) { double a, b; actual(truth[t], m, gf, N - nd, a, b); best = std::min(best, std::max(a, b)); }
          double a, b; actual(truth[t], m, gf, d.n_cpu, a, b);
          const double gap = std::max(a, b) / best - 1;
          if (pass == 0) {
            worst_late_gap = std::max(worst_late_gap, gap);
            if (!d.cold) worst_late_pred = std::max(worst_late_pred, std::fabs(std::max(d.pred_cpu_ms, d.pred_gpu_ms) / std::max(a, b) - 1));
          } else mixed_late_gap = std::max(mixed_late_gap, gap);
        }
      }
    }
    printf("convergence: cold layers %d · worst late gap vs optimum %.3f (typed) / %.3f (one mixed model) · worst late |pred/actual-1| %.3f\n", cold_layers,
           worst_late_gap, mixed_late_gap, worst_late_pred);
    EXPECT(cold_layers == 2, "cold layers %d (want one per type)", cold_layers);
    EXPECT(worst_late_gap < 0.12, "typed model gap %.3f", worst_late_gap);
    EXPECT(worst_late_pred < 0.20, "pred error %.3f", worst_late_pred);
    EXPECT(mixed_late_gap > worst_late_gap, "mixed model should be worse (control) %.3f vs %.3f", mixed_late_gap, worst_late_gap);
  }

  if (fails) { fprintf(stderr, "prefill_split: %d FAIL\n", fails); return 1; }
  printf("prefill_split: all PASS\n");
  return 0;
}
