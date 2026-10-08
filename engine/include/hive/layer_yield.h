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
struct Stats { long yields = 0, skipped_park = 0, intra_yields = 0; double inner_ms = 0, park_ms = 0, max_gap_ms = 0; };

// ---- Intra-layer points (HIVE_LAYER_YIELD_INTRA) --------------------------------------------------------------------------------------------------------------
// Problem (measured 2026-10-08, service log): with layer-boundary yields only, 78 % of the decode stall tail came from ≥ 49K-row forwards
//   (forward_multi with host tiles): an ordinary encoder layer ≈ 460 ms, the kv/index source layers (2, 8, 14) at positions 140–239K 1.95–5.5 s
//   (indexer), the first gap (embedding + layer 0) 1.0 s — a running decoder waits for the whole layer whatever the period is.
// Fix: yield points inside a layer, decode only (no admission, no short prefill — only the decode floor of rows is parked there):
//   kEmbed = after a unit's embedding + layer-0 front (forward_multi layer-0 pass) · kUnit = after a unit's / sub-chunk's front (forward_multi
//   passes, forward's tiled front loop; after the last unit = right before the expert stage) · kMoe = after the router sync, right before the
//   expert stage (forward's single-sequence streaming layers) · kIndex = between indexer row batches (inside attention).
//   Runtime side: runtime.cpp "T11b" comment above Runtime::layer_yield_point. Off (default) = none of this runs (the points return at once).
enum Kind : int { kLayer = 0, kEmbed = 1, kUnit = 2, kMoe = 3, kIndex = 4, kKinds = 5 };
inline const char* kind_name(int k) {
  static const char* const n[kKinds] = {"layer", "embed", "unit", "moe", "index"};
  return k >= 0 && k < kKinds ? n[k] : "?";
}
// HIVE_LAYER_YIELD_INTRA: switch convention — unset · "" · "0" = off, anything else = on.
inline bool parse_intra(const char* v) { return v && *v && !(v[0] == '0' && v[1] == 0); }
// HIVE_LAYER_YIELD_CAP (ms, look-ahead bound of intra points): unset · "" · non-numeric · below 50 = 200 · "0" = look-ahead off · ≥ 50 = that
//   value (capped at 60000). 200 ms = the target stall on the reference machine (a decode step is ~30–50 ms; the measured stall after the
//   150 ms period was 0.36–0.42 s).
constexpr double kDefaultCapMs = 200.0;
inline double parse_cap(const char* v) {
  if (!v || !*v) return kDefaultCapMs;
  char* end = nullptr;
  const double x = strtod(v, &end);
  const bool num = end != v && end && *end == 0 && std::isfinite(x);
  if (num && x == 0.0) return 0.0;
  if (num && x >= 50.0) return std::min(x, 60000.0);
  return kDefaultCapMs;
}
// Decision at an intra point: yield when the period has elapsed, or (look-ahead) when the next indivisible segment (EMA of the time from a
//   point with this key to the next point) would end beyond the cap — a stall of elapsed + next becomes one of next. The look-ahead only
//   fires when the stall it removes (elapsed) is at least what a yield costs (EMA of earlier intra yields) — right after a resume it would
//   insert a decode step for nothing.
inline bool intra_due(double elapsed, double period, double cap, double next_ms, double cost_ms) {
  if (elapsed >= period) return true;
  return cap > 0 && next_ms > 0 && elapsed + next_ms > cap && elapsed >= cost_ms;
}
inline void ema(double& v, double x, double a = 0.1) { v = v <= 0 ? x : v * (1 - a) + x * a; }  // same form as hived's ema (first sample = value)
// Segment key: (point kind, layer, whether it is the last point of its run — the last unit of a pass and the last indexer batch are
//   followed by a different kind of work than the others).
inline int seg_key(int kind, int layer, bool last) { return (std::max(0, layer) * kKinds + kind) * 2 + (last ? 1 : 0); }
struct Segs {
  std::vector<double> ms;  // EMA per key (0 = no sample)
  double t0 = 0;           // start of the open segment (a point's time, or the resume after a yield)
  int key = -1;            // key of the point that opened it (-1 = none — the first point of a forward closes nothing)
  void close(int next_key, double t) {
    if (key >= 0 && t >= t0) {
      if ((size_t)key >= ms.size()) ms.resize((size_t)key + 1, 0.0);
      ema(ms[(size_t)key], t - t0);
    }
    key = next_key; t0 = t;
  }
  double next(int k) const { return k >= 0 && (size_t)k < ms.size() ? ms[(size_t)k] : 0.0; }
};

struct State {
  Hooks h;
  int depth = 0;
  int floor = 1, cap = 0;
  double last = 0;  // time of the last resume (or start of the outer forward)
  Stats st;
  // HIVE_LAYER_YIELD_INTRA (off: intra false and none of the fields below is touched)
  bool intra = false;
  double cap_ms = kDefaultCapMs;
  int kind = kLayer;  // kind of the point currently yielding (hived reads it — intra points run decode only)
  Segs seg;
  double cost_ms = 0;  // EMA of the inner time of intra yields
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

// Intra-layer point (HIVE_LAYER_YIELD_INTRA): same order as yield_point with three differences — ① the clock rule is intra_due (period or
//   look-ahead) and every call closes the open segment (segment EMA) ② R = the decode floor (park_rows(1, …)): nothing larger than decode
//   runs inside (hived's want/run see s.kind ≠ kLayer and neither admit nor prefill) ③ the yield's inner time feeds cost_ms and, after the
//   resume, a new segment starts. Exceptions: the guard restores (unpark, depth, kind) exactly like yield_point.
template <class Park, class Unpark, class Now> bool yield_point_intra(State& s, int kind, int key, Park&& park_fn, Unpark&& unpark_fn, Now&& now) {
  if (!s.on() || !s.intra || s.depth >= std::max(1, s.h.max_depth)) return false;
  const double t = now();
  s.seg.close(key, t);
  const double gap = t - s.last;
  if (!intra_due(gap, s.h.period_ms, s.cap_ms, s.seg.next(key), s.cost_ms)) return false;
  struct KindGuard { State& s; int was; ~KindGuard() { s.kind = was; } } kind_guard{s, s.kind};
  s.kind = kind;
  if (s.h.want() <= 0) return false;
  const int R = park_rows(1, s.floor, s.cap);
  if (R <= 0) return false;
  if (!park_fn(R)) { ++s.st.skipped_park; s.last = now(); s.seg.t0 = s.last; return false; }
  const double t_parked = now();
  struct Guard {
    State& s; int R; Unpark& un; Now& now; double t0;
    ~Guard() {
      --s.depth; un(R); const double t1 = now(); s.st.inner_ms += t1 - t0; s.last = t1;
      ema(s.cost_ms, t1 - t0); s.seg.t0 = t1;  // the next segment starts at the resume (the yield is not prefill work)
    }
  } guard{s, R, unpark_fn, now, t_parked};
  ++s.depth;
  ++s.st.yields;
  ++s.st.intra_yields;
  s.st.park_ms += t_parked - t;
  s.st.max_gap_ms = std::max(s.st.max_gap_ms, gap);
  s.h.run(R, gap);
  return true;
}

}  // namespace hive::ly
