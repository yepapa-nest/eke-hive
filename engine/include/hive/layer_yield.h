// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Layer-level yield (HIVE_LAYER_YIELD) — host-only decisions and ordering skeleton. runtime.cpp (Runtime::layer_yield_point), hived.cpp
//   and the fake runtime (tools/cpu_fake/fake_runtime.inc) use this same code — the CPU test (tools/test_layer_yield_cpu.py) checks
//   the order on top of the real hived.cpp.
//
// Problem (measured): once a single forward of a long prefill starts (98,304 rows = one forward including host tiles ≈ 14 s), the
//   engine thread does not return until all 40 layers are done — hived's yield points (decode_between · HIVE_PREFILL_YIELD) all sit
//   *between* forwards (chunk/round boundaries), so a single-chunk prompt has no boundary at all.
//   Measured: HIVE_PREFILL_YIELD failed here — "85K is one chunk, so there is no boundary to cut in at · in-flight decode also stalls
//   14 s during a long prefill". In service: 5 stalls in 70 minutes, up to 4.9 s.
// Fix: at a layer boundary inside the forward (the expert stage has finished, GPU and CPU pool are idle), if the period has elapsed
//   and there is work (want), park the outer prefill's inter-layer state, run hived's yield body (run: admit short requests + decode
//   steps), then restore it (unpark). Chunk boundaries, chunk size and forward grouping stay unchanged
//   (this does not touch what HIVE_PREFILL_YIELD avoided — "splitting a 98K forward changes the expert CPU/DMA shares and the fp32
//   accumulation order" — the same forward is merely paused between layers).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <vector>

namespace hive::ly {

// ---- Switch parsing (absorb, never reject — unknown values fall back to the default) ----------------------------------------------------------------------------
// HIVE_LAYER_YIELD: unset · "" · "0" = off (default) · fully numeric and ≥ 50 = that period (ms, capped at 60000) · anything else
//   ("1" · "on" · below 50 · non-numeric) = default period.
//   Why 500 ms by default: the target is ~0.5–1 s for a short request's first token and for in-flight decode stalls — one encoder
//   layer of a 98K forward ≈ 0.6 s (estimate: most of the 14 s is the 20 encoder layers), so a 500 ms period is effectively once per
//   encoder layer. To be re-chosen from measurement (period sweep of the validation command).
constexpr double kDefaultPeriodMs = 500.0;
inline double parse_period(const char* v) {
  if (!v || !*v) return 0.0;
  char* end = nullptr;
  const double x = strtod(v, &end);
  const bool num = end != v && end && *end == 0 && std::isfinite(x);
  if (num && x == 0.0) return 0.0;  // "0" = off (env_on convention)
  if (num && x >= 50.0) return std::min(x, 60000.0);
  return kDefaultPeriodMs;
}
// HIVE_LAYER_YIELD_SHARE: decode budget of one yield = (preceding prefill interval) × share/(1−share) — same formula as
//   decode_between (decode_share).
//   unset · "" · non-numeric · out of range = 0.05 (share for the target "long prefill +5 % at most" — an estimate: one decode step
//   always runs).
constexpr double kDefaultShare = 0.05;
inline double parse_share(const char* v) {
  if (!v || !*v) return kDefaultShare;
  char* end = nullptr;
  const double x = strtod(v, &end);
  return end != v && end && *end == 0 && std::isfinite(x) && x > 0.0 && x <= 0.9 ? x : kDefaultShare;
}
// HIVE_LAYER_YIELD_STEPS: max decode steps per yield (together with the budget — whichever is hit first). unset · non-numeric ·
//   below 1 = 4 · capped at 64.
constexpr int kDefaultMaxSteps = 4;
inline int parse_steps(const char* v) {
  if (!v || !*v) return kDefaultMaxSteps;
  char* end = nullptr;
  const long x = strtol(v, &end, 10);
  return end != v && end && *end == 0 && x >= 1 ? (int)std::min<long>(x, 64) : kDefaultMaxSteps;
}
inline double decode_budget_ms(double prefill_ms, double share) {
  return share > 0.0 && share < 1.0 && prefill_ms > 0.0 ? prefill_ms * share / (1.0 - share) : 0.0;
}

// ---- Rows to park ---------------------------------------------------------------------------------------------------------------------------------
// want = max rows of the short prefill hived will run inside this yield (1 for decode only · 0 = nothing to do). floor = max rows
// of Work used by decode · verify · draft · MTP sync (runtime: max(window, max_batch, dspark_block+1) — same formula as the small
// HIVE_CACHE_ELASTIC version). cap = Work row capacity (max_chunk). 0 = do not yield.
inline int park_rows(int want, int floor, int cap) { return want <= 0 || cap <= 0 ? 0 : std::min(cap, std::max({1, floor, want})); }

// Parking a row range: for each buffer, append the first R rows (row_bytes × min(R, cap_rows)) to the stash — restored in the same
//   order. copy(dst, src, n) is supplied by the caller (runtime = cudaMemcpyAsync D2H/H2D(st_) · host tables = memcpy · fake = memcpy).
struct RowBuf { void* p = nullptr; size_t row_bytes = 0; size_t cap_rows = 0; };
inline size_t park_bytes(const std::vector<RowBuf>& v, int R) {
  size_t n = 0;
  for (const RowBuf& b : v) if (b.p && b.row_bytes) n += b.row_bytes * std::min<size_t>((size_t)std::max(0, R), b.cap_rows);
  return n;
}
template <class Copy> size_t park(const std::vector<RowBuf>& v, int R, uint8_t* stash, Copy&& copy) {
  size_t off = 0;
  for (const RowBuf& b : v) {
    if (!b.p || !b.row_bytes) continue;
    const size_t n = b.row_bytes * std::min<size_t>((size_t)std::max(0, R), b.cap_rows);
    if (n) copy(stash + off, b.p, n);
    off += n;
  }
  return off;
}
template <class Copy> size_t unpark(const std::vector<RowBuf>& v, int R, const uint8_t* stash, Copy&& copy) {
  size_t off = 0;
  for (const RowBuf& b : v) {
    if (!b.p || !b.row_bytes) continue;
    const size_t n = b.row_bytes * std::min<size_t>((size_t)std::max(0, R), b.cap_rows);
    if (n) copy(b.p, stash + off, n);
    off += n;
  }
  return off;
}

// ---- Yield point ordering (skeleton) ------------------------------------------------------------------------------------------------------------------------
// Called by the caller (outer prefill forward) at every layer boundary. Order: ① clock (nothing happens before the period elapses —
//   the cost is one time read) ② want (hived — looks at the queue and active set; 0 = nothing happens · the clock is left as is so
//   the next boundary asks again) ③ R = park_rows ④ park(R) (failure = skip only this yield — absorbed) ⑤ depth++ →
//   run(R, preceding prefill ms) → depth-- (even on exception) ⑥ unpark (even on exception — the outer forward's BrokenGuard receives
//   the exception) ⑦ restart clock · stats.
//   No recursion: if a forward inside the yield (short prefill) reaches a layer boundary, depth > 0 ends it at ①.
struct Hooks {
  std::function<int()> want;                 // hived: 0 = nothing to do · n ≥ 1 = max short-prefill rows (1 for decode only)
  std::function<void(int, double)> run;      // hived: (usable rows R, preceding prefill ms)
  double period_ms = 0;                      // 0 = off
  // Yield levels: 1 = forwards run inside a yield never yield themselves (default). 2 = they may yield once more — hived uses it with
  //   HIVE_LAYER_YIELD_MID so a mid-size prompt admitted at a yield (seconds of prefill) still gives the running decoders their steps; its
  //   want() then reports decode work only.
  int max_depth = 1;
};
struct Stats { long yields = 0, skipped_park = 0; double inner_ms = 0, park_ms = 0, max_gap_ms = 0; };
struct State {
  Hooks h;
  int depth = 0;
  int floor = 1, cap = 0;
  double last = 0;  // time of the last resume (or start of the outer forward)
  Stats st;
  bool on() const { return h.run && h.want && h.period_ms > 0; }
};
// park(R) → bool (parked?) · unpark(R) · now() are supplied by the caller (runtime/fake). Returns whether it yielded.
template <class Park, class Unpark, class Now> bool yield_point(State& s, Park&& park_fn, Unpark&& unpark_fn, Now&& now) {
  if (!s.on() || s.depth >= std::max(1, s.h.max_depth)) return false;
  const double t = now();
  if (t - s.last < s.h.period_ms) return false;
  const int want = s.h.want();
  if (want <= 0) return false;
  const int R = park_rows(want, s.floor, s.cap);
  if (R <= 0) return false;
  const double gap = t - s.last;
  if (!park_fn(R)) { ++s.st.skipped_park; s.last = now(); return false; }
  const double t_parked = now();
  struct Guard {
    State& s; int R; Unpark& un; Now& now; double t0;
    ~Guard() { --s.depth; un(R); const double t1 = now(); s.st.inner_ms += t1 - t0; s.last = t1; }
  } guard{s, R, unpark_fn, now, t_parked};
  ++s.depth;
  ++s.st.yields;
  s.st.park_ms += t_parked - t;
  s.st.max_gap_ms = std::max(s.st.max_gap_ms, gap);
  s.h.run(R, gap);
  return true;
}

}  // namespace hive::ly
