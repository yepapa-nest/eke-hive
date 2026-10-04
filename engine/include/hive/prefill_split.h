// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_PREFILL_SPLIT=balance — choose the CPU/DMA split of missed experts in streaming prefill (moe_experts_multi) with a cost model.
//   Host only (no CUDA) — tested on the CPU by engine/tests/test_prefill_split_cpu.cpp. runtime.cpp feeds measurements and only calls decide().
//   Background (measured on a service build with grouped prefill, host tiles and multi-sequence prefill on): with a fixed streaming-prefill DMA share (HIVE_DMA_FRAC_PREFILL),
//   42K 10.39 s (adaptive +-0.05) -> 7.90 (0.7) / 8.22 (0.85) / 8.61 (0.95), 80K 18.16 -> 14.88/15.04/15.51, 15K 6.54 -> 5.60/6.11/6.63 — the best share differs per layer
//   (row distribution and residency differ per layer and chunk) yet one value was fixed. 0.7 is the service default and is also the cold-start value here.
//   Model: sort the N misses by rows ascending (the existing CPU share rule — fewest rows go to the CPU) and, sending the first n_cpu to the CPU and the rest to DMA,
//     predicted CPU finish = job_ms x (CPU jobs)        — job = one (expert, sub-chunk, rows <= kMaxRows). job_ms = (pool finish time (jobs_done_ms) - start) / jobs.
//                                                   The pool shares one memory bandwidth (each job reads the whole expert weight), so it is taken as linear in jobs.
//     predicted GPU finish = max(rec_ms x DMA records, row_ms x GPU rows) — copy (side stream) and compute (st_) overlap through the staging ring, so the slower one decides.
//                                                   rec_ms = staging copy event pair (pure time of one copy), row_ms = resident batch compute event pair / rows.
//   evaluate all of them and pick the n with the smallest max(CPU, GPU) (CPU increases monotonically in n, GPU decreases -> where they meet). Share = DMA count / N in [0.5, 0.98].
//   Costs are EMAs kept per layer type (encoder layers [0, tail layer), decoder tail [tail layer, end)) — tail layers hold only tail rows, so job/row distributions differ.
//   If any of a type's three costs has no sample yet: cold start with share 0.7 (same rounding as the fixed value).
//   Numerics: the expert compute paths (CPU and GPU) are unchanged. Only **which missed experts take the CPU path** changes — the CPU path accumulates fp32 outputs separately
//   (accum_f32_rows) while the GPU path accumulates bf16 outputs, so changing the selection changes fp32 accumulation order/rounding (low bits). Since the choice depends on measured time,
//   **the same input may differ in low bits between runs** (the fixed 0.7 was deterministic for the same cache state).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <strings.h>
#include <vector>

namespace hive::ps {

enum Mode { kOff = 0, kBalance = 1 };
// Switch parsing: only "balance" (case-insensitive) enables it. Unset, empty, "0" or any other value = off (baseline path — absorbed, not rejected).
inline int parse_mode(const char* v) { return (v && *v && strcasecmp(v, "balance") == 0) ? kBalance : kOff; }

constexpr float kCold = 0.7f;   // cold-start share = service default HIVE_DMA_FRAC_PREFILL=0.7
constexpr float kLo = 0.5f, kHi = 0.98f;  // share limits (specified, not measured — revisit if the share distribution in chunk logs sticks to a limit)
constexpr double kAlpha = 0.3;  // EMA weight (new sample) — not tuned; judged by predicted vs actual in chunk logs
constexpr int kTypes = 2;       // 0 = encoder layer, 1 = decoder tail layer

struct Costs {
  double job_ms = 0, rec_ms = 0, row_ms = 0;
  bool has_job = false, has_rec = false, has_row = false;
  bool warm() const { return has_job && has_rec && has_row; }
};
// Sample update: finite positive values only (0, negative and NaN are dropped — e.g. events not recorded; absorbed)
inline void ema(double& v, bool& has, double x) {
  if (!(x > 0.0) || !std::isfinite(x)) return;
  v = has ? v + kAlpha * (x - v) : x;
  has = true;
}

struct Miss { int rows, jobs; };  // passed sorted by rows ascending (same order as the CPU share rule)
struct Decision {
  float share = kCold;  // actual DMA share = n_dma / N (kCold if N = 0)
  int n_dma = 0, n_cpu = 0;  // n_cpu = number of leading misses sent to the CPU (after the row cap)
  double pred_cpu_ms = 0, pred_gpu_ms = 0;
  bool cold = true;
};

// Same rounding as the share -> DMA count formula in runtime.cpp: (int)(frac x N + 0.5f)
inline int n_dma_of(float frac, int N) { return (int)(frac * (float)N + 0.5f); }
// CPU count -> actual CPU count honouring the row cap (cpu_row_cap = Work M) (the loop stops where cumulative rows would exceed it)
inline int capped_cpu(const std::vector<Miss>& miss, int n_cpu, int cpu_row_cap) {
  int rows = 0, i = 0;
  for (; i < n_cpu; ++i) { if (rows + miss[(size_t)i].rows > cpu_row_cap) break; rows += miss[(size_t)i].rows; }
  return i;
}

// gpu_rows_fixed = rows the GPU computes on this layer anyway (resident hits + prefetched)
inline Decision decide(const Costs& c, const std::vector<Miss>& miss, int gpu_rows_fixed, int cpu_row_cap) {
  Decision d;
  const int N = (int)miss.size();
  std::vector<double> rows_pre((size_t)N + 1, 0), jobs_pre((size_t)N + 1, 0);
  for (int i = 0; i < N; ++i) { rows_pre[(size_t)i + 1] = rows_pre[(size_t)i] + miss[(size_t)i].rows; jobs_pre[(size_t)i + 1] = jobs_pre[(size_t)i] + miss[(size_t)i].jobs; }
  auto eval = [&](int n_cpu, double& pc, double& pg) {
    pc = c.job_ms * jobs_pre[(size_t)n_cpu];
    pg = std::max(c.rec_ms * (double)(N - n_cpu), c.row_ms * ((double)gpu_rows_fixed + rows_pre[(size_t)N] - rows_pre[(size_t)n_cpu]));
  };
  if (N == 0) { d.cold = !c.warm(); eval(0, d.pred_cpu_ms, d.pred_gpu_ms); return d; }
  if (!c.warm()) {
    d.n_cpu = capped_cpu(miss, N - n_dma_of(kCold, N), cpu_row_cap);
    d.n_dma = N - d.n_cpu;
    d.share = (float)d.n_dma / (float)N;
    eval(d.n_cpu, d.pred_cpu_ms, d.pred_gpu_ms);
    return d;
  }
  d.cold = false;
  const int lo = n_dma_of(kLo, N), hi = n_dma_of(kHi, N);  // DMA count candidates [lo, hi] = share [0.5, 0.98] with the same rounding
  double best = -1;
  for (int nd = hi; nd >= lo; --nd) {  // most DMA first — on equal predictions prefer less CPU (the pool competes with host prep for bandwidth)
    const int nc = capped_cpu(miss, N - nd, cpu_row_cap);
    double pc, pg;
    eval(nc, pc, pg);
    const double t = std::max(pc, pg);
    if (best < 0 || t < best) { best = t; d.n_cpu = nc; d.pred_cpu_ms = pc; d.pred_gpu_ms = pg; }
  }
  d.n_dma = N - d.n_cpu;
  d.share = (float)d.n_dma / (float)N;
  return d;
}

}  // namespace hive::ps
