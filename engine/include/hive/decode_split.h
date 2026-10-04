// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_SPLIT=balance — picks, per layer and with a cost model, the DMA/CPU split of the missed experts in decode (moe_decode_experts, kind kDmaDecode).
//   Host only (no CUDA) — tested on the CPU by engine/tests/test_decode_split_cpu.cpp. runtime.cpp feeds measurements and only calls decide()/observe().
//
//   Background (measured with the [decode-host] profile on the serving build, M=1): CPU expert phase 4.2–7.4 ms per step vs GPU expert phase ~4.4 ms; in 16–27 of 40 layers the CPU
//   finishes late and the GPU waits (tail 2.2–3.7 ms/step; batched-decode tail up to 7.2 ms, 35/40 layers). Base rule: misses < 3 → all on CPU; ≥ 3 → DMA count = min(8, staging,
//   max(decode_gpu_share, round(frac × misses))), with frac moved ±0.05 per sample (bounds [0.1, 0.8]) — one value tracking per-layer miss counts and row distributions that differ.
//
//   ⚠️ Lesson from the failed HIVE_PREFILL_SPLIT=balance: the prefill model treated CPU cost as "linear in job count, one average value" and shifted too much work to the CPU,
//   15K prefill 5.8 → 7.9 s. Therefore this model
//     (1) measures cost not as one slope but as **per-count tables** (an EMA per job count, copy count and group count) — counts already seen use the measured value, counts in between interpolate linearly.
//         For counts never seen it extrapolates **pessimistically (high) for the CPU** and **optimistically (low) for the GPU side** (resident compute, copies, DMA groups) — both lean "towards DMA".
//     (2) compares **predicted vs actual for the chosen split** (CPU pool phase, GPU expert phase, layer end) on every sampled layer and keeps a relative-error EMA; with too few samples (kMinVal)
//         or a large error (kErrMax) it falls back to the **base rule** (meanwhile it keeps predicting and checking the base rule's split, so it recovers by itself).
//     (3) multiplies the CPU prediction by a margin equal to that error (pc × (1 + err_c)) and on a tie prefers more DMA — "if the CPU looks like it will finish last, use DMA".
//     (4) cold start (empty tables) = the base rule unchanged (same split as without the switch).
//   Prediction (one layer; misses ordered as in the base rule = rows descending, the first n go to DMA):
//     CPU(n)  = cpu[job count(n)]                       — host clock: pool start (start_jobs) → completion stamped by the pool (jobs_done_ms). Job = (expert, rows ≤ kMaxRows).
//     GPU(n)  = n == 0 ? res[resident group count] : max(res[resident group count], cp[n]) + dg[DMA group count(n)]
//               GPU clock, origin g0 = just before the expert-table H2D (st_ — same point as the [decode-host] kExp mark). res = g0 → end of resident groups · cp = g0 → end of this layer's last demand copy (side_ —
//               includes waits on earlier promotion copies and ring back-pressure) · dg = DMA group compute (from st_ waiting on the copies to the end).
//     end(n)  = max(CPU(n) × (1 + err_c) + skew, GPU(n))  — skew = measured (g0 → accumulation launch) − CPU phase on layers where the CPU is last (host→GPU clock offset + accumulation launch delay).
//   n ranges exhaustively over [0, min(cap, misses)] (cap = min(8, staging slots) — staging slots are never reused within a layer: that would overwrite the layer's earlier copies).
//   Numerics: the expert compute paths (CPU and GPU) are unchanged. The only thing that changes is **which device computes which missed expert** — CPU-path output is accumulated separately in fp32
//   (accum_f32_rows) while the GPU path accumulates bf16 output (and the grouped kernel's mma accumulation order differs from the CPU GEMV), so rows whose choice changes may differ in the low bits.
//   Since the choice depends on measured times, the low bits may differ between runs on the same input (the adaptive frac of the base rule also depended on sample times — same property).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <strings.h>
#include <vector>

namespace hive::dsplit {

enum Mode { kOff = 0, kBalance = 1 };
// Switch parsing: only "balance" (case-insensitive) enables it. Unset, empty, "0" or any other value = off (base rule — absorbed, not rejected). Same form as HIVE_PREFILL_SPLIT.
inline int parse_mode(const char* v) { return (v && *v && strcasecmp(v, "balance") == 0) ? kBalance : kOff; }

// The constants below are untuned starting values (not set from measurement) — judge and adjust them by predicted vs actual in the [decode-split] lines.
constexpr int kN = 65;             // table slots 0..64 (job/group counts — even batch 64 × 6 experts usually stays within this per layer). Larger counts extrapolate
constexpr int kCls = 9;            // table classes: M = 1..8 each, M ≥ 9 one class (row distribution and group size differ per batch composition)
constexpr double kAlpha = 0.25;    // EMA weight of a new sample
constexpr int kMinVal = 16;        // fewer validation samples (sampled layers that had a prediction) than this → base rule
constexpr double kErrMax = 0.35;   // relative-error EMA (CPU or GPU) above this → base rule
constexpr double kTailEps = 0.005; // ms — threshold for "the CPU is last" on a layer (same 5 µs as the cpu-bound criterion of the [decode-host] profile)
constexpr int kGroupRows = 8;      // GPU group = expert × rows ≤ 8 (moe_decode_experts add_group — kernel limit)

inline int cls_of(int M) { return std::min(std::max(M, 1), kCls) - 1; }

// Count → time (ms) table, one EMA per slot. at(): a seen slot returns its value · between two seen slots interpolate linearly · outside extrapolate (high = pessimistically large, !high = optimistically small).
struct Tab {
  double v[kN] = {};
  bool has[kN] = {};
  int seen = 0;
  void add(int n, double x) {
    if (n < 0 || n >= kN || !(x >= 0.0) || !std::isfinite(x)) return;  // drop negatives and NaN (unrecorded events etc. — absorbed); counts outside the table are not stored (answered by extrapolation)
    if (!has[n]) { v[n] = x; has[n] = true; ++seen; }
    else v[n] += kAlpha * (x - v[n]);
  }
  bool any() const { return seen > 0; }
  double at(int n, bool high) const {
    if (n < 0) return 0.0;
    if (n < kN && has[n]) return v[n];
    int lo = -1, hi = -1;
    for (int i = std::min(n, kN) - 1; i >= 0; --i) if (has[i]) { lo = i; break; }
    for (int i = std::max(n + 1, 0); i < kN; ++i) if (has[i]) { hi = i; break; }
    if (lo < 0 && hi < 0) return 0.0;
    if (lo >= 0 && hi >= 0) return v[lo] + (v[hi] - v[lo]) * (double)(n - lo) / (double)(hi - lo);
    if (lo < 0) {  // n is smaller than every seen slot
      const int h = hi;
      int h2 = -1;
      for (int i = h + 1; i < kN; ++i) if (has[i]) { h2 = i; break; }
      if (high) return v[h];  // pessimistic: a smaller count is assumed to take as long as the smallest seen slot
      if (h2 >= 0) { const double s = std::max(0.0, (v[h2] - v[h]) / (double)(h2 - h)); return std::max(0.0, v[h] - s * (double)(h - n)); }
      return h > 0 ? v[h] * (double)n / (double)h : v[h];
    }
    const int h = lo;  // n is larger than every seen slot
    int h2 = -1;
    for (int i = h - 1; i >= 0; --i) if (has[i]) { h2 = i; break; }
    const double prop = h > 0 ? v[h] * (double)n / (double)h : -1.0;
    const double lin = h2 >= 0 ? v[h] + std::max(0.0, (v[h] - v[h2]) / (double)(h - h2)) * (double)(n - h) : (prop >= 0 ? prop : v[h]);
    if (prop < 0) return lin;
    return high ? std::max(lin, prop) : std::min(lin, prop);
  }
};

struct Model {
  Tab cpu, res, cp, dg;
  double skew = 0.0;
  bool has_skew = false;
  double err_c = 0.0, err_g = 0.0;  // relative-error EMA (|predicted − actual| / actual)
  int n_val_c = 0, n_val_g = 0;
  bool warm() const { return cpu.any() && res.any() && cp.any() && dg.any(); }
  bool reliable() const { return n_val_c >= kMinVal && n_val_g >= kMinVal && err_c <= kErrMax && err_g <= kErrMax; }
};

// Layer input: rows per missed expert (base-rule DMA priority order = rows descending — exactly the miss_e sort in runtime.cpp), resident group count, cap, base-rule DMA count
struct Input {
  std::vector<int> rows;
  int ng_hit = 0;
  int cap = 8;
  int base_n = 0;
  int max_rows = 8;  // CPU job row limit (ExpertStore::kMaxRows)
};
enum Why { kModel = 0, kCold = 1, kUnsure = 2 };
struct Pred { int n = 0, jobs = 0, ngs = 0; double cpu = 0, gpu = 0, end = 0; };
struct Decision {
  int n_dma = 0;
  int why = kCold;
  bool has_pred = false;  // a prediction exists (tables are filled) — even when the base rule picks, its split's prediction is kept for validation
  Pred pred;
};

inline int jobs_of(const Input& in, int n) {  // CPU job count: the misses after the DMA ones, ceil(rows / max_rows) per expert
  int j = 0;
  for (size_t i = (size_t)n; i < in.rows.size(); ++i) j += (in.rows[i] + in.max_rows - 1) / in.max_rows;
  return j;
}
inline int groups_of(const Input& in, int n) {  // DMA group count: the first n misses, ceil(rows / 8) per expert
  int g = 0;
  for (int i = 0; i < n && i < (int)in.rows.size(); ++i) g += (in.rows[(size_t)i] + kGroupRows - 1) / kGroupRows;
  return g;
}
inline Pred predict(const Model& m, const Input& in, int n) {
  Pred p;
  p.n = n; p.jobs = jobs_of(in, n); p.ngs = groups_of(in, n);
  p.cpu = p.jobs > 0 ? m.cpu.at(p.jobs, /*high=*/true) : 0.0;
  p.gpu = m.res.at(in.ng_hit, /*high=*/false);
  if (n > 0) p.gpu = std::max(p.gpu, m.cp.at(n, false)) + m.dg.at(p.ngs, false);
  const double cpu_eff = p.jobs > 0 ? p.cpu * (1.0 + m.err_c) + (m.has_skew ? m.skew : 0.0) : 0.0;
  p.end = std::max(cpu_eff, p.gpu);
  return p;
}

inline Decision decide(const Model& m, const Input& in) {
  Decision d;
  const int N = (int)in.rows.size();
  const int cap = std::max(0, std::min(in.cap, N));
  const int base = std::max(0, std::min(in.base_n, cap));  // base-rule DMA count (clamped to the miss count and the cap, as the loop stops at the miss count)
  d.n_dma = base;
  if (!m.warm()) { d.why = kCold; return d; }
  d.has_pred = true;
  d.pred = predict(m, in, base);
  if (!m.reliable()) { d.why = kUnsure; return d; }
  d.why = kModel;
  for (int n = cap; n >= 0; --n) {  // from most DMA down — on a tie the side with more DMA wins (reduces the risk of the CPU finishing last)
    const Pred p = predict(m, in, n);
    if (n == cap || p.end < d.pred.end) { d.pred = p; d.n_dma = n; }
  }
  return d;
}

// One sampled layer (after the GPU events completed) — negative = that phase did not exist (no CPU share, no DMA)
struct Obs {
  int ng_hit = 0, n_str = 0, ng_str = 0, jobs = 0;
  double cpu_ms = -1, res_ms = -1, cp_ms = -1, gpu_ms = -1, end_ms = -1;
  bool has_pred = false;
  Pred pred;
};
// Updates the GPU-side tables, skew and validation errors (the CPU table is updated per layer by add_cpu — host clock, no sampled events needed)
inline void observe(Model& m, const Obs& o) {
  if (o.res_ms >= 0) m.res.add(o.ng_hit, o.res_ms);
  if (o.n_str > 0 && o.cp_ms >= 0 && o.gpu_ms >= 0) {
    m.cp.add(o.n_str, o.cp_ms);
    m.dg.add(o.ng_str, o.gpu_ms - std::max(o.res_ms, o.cp_ms));  // add() drops negatives (event order mismatch)
  }
  if (o.jobs > 0 && o.cpu_ms > 0 && o.end_ms >= 0 && o.gpu_ms >= 0 && o.end_ms - o.gpu_ms > kTailEps) {  // CPU last: accumulation launch = CPU end + delay
    const double s = o.end_ms - o.cpu_ms;
    if (std::isfinite(s)) { m.skew = m.has_skew ? m.skew + kAlpha * (s - m.skew) : s; m.has_skew = true; }
  }
  if (!o.has_pred) return;
  if (o.jobs > 0 && o.cpu_ms > 0) { const double e = std::fabs(o.pred.cpu - o.cpu_ms) / o.cpu_ms; m.err_c = m.n_val_c ? m.err_c + kAlpha * (e - m.err_c) : e; ++m.n_val_c; }
  if (o.gpu_ms > 0) { const double e = std::fabs(o.pred.gpu - o.gpu_ms) / o.gpu_ms; m.err_g = m.n_val_g ? m.err_g + kAlpha * (e - m.err_g) : e; ++m.n_val_g; }
}
inline void add_cpu(Model& m, int jobs, double cpu_ms) { if (jobs > 0 && cpu_ms > 0) m.cpu.add(jobs, cpu_ms); }

// [decode-split] line totals (sum over sampled layers) — runtime.cpp prints and clears them on HIVE_PROFILE sample steps
struct Acc {
  int layers = 0, why[3] = {0, 0, 0}, dma = 0, base = 0, n_pred = 0;
  double pc = 0, ac = 0, pg = 0, ag = 0, pe = 0, ae = 0, tail = 0, abs_c = 0, abs_g = 0, abs_e = 0;
  void add(const Decision& d, int base_n, const Obs& o) {
    ++layers; ++why[d.why]; dma += d.n_dma; base += base_n;
    const double cpu = std::max(0.0, o.cpu_ms), gpu = std::max(0.0, o.gpu_ms), end = o.end_ms >= 0 ? o.end_ms : gpu;
    ac += cpu; ag += gpu; ae += end; tail += std::max(0.0, end - gpu);
    if (o.has_pred) {
      ++n_pred; pc += o.pred.cpu; pg += o.pred.gpu; pe += o.pred.end;
      abs_c += std::fabs(o.pred.cpu - cpu); abs_g += std::fabs(o.pred.gpu - gpu); abs_e += std::fabs(o.pred.end - end);
    }
  }
};

}  // namespace hive::dsplit
