// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#pragma once
// Short-prefill (TTFT) helpers — pure decision functions used by runtime.cpp (host only, no CUDA). CPU test: engine/tests/test_short_prefill_cpu.cpp.
//   Switches (runtime.h ShortPrefillOpts, read by the constructor — unset / "" / "0" = off = default path):
//     HIVE_PREFILL_TAIL_SHORT   route tail layers (rows < prefill_threshold inside the prefill forward) to the short-prefill expert path (moe_decode_experts)
//     HIVE_PREFETCH_SCALE       predicted prefetch rows = previous chunk rows × (rows now / previous rows)
//     HIVE_PREFILL_SHORT_ADAPT  DMA share of streaming layers with rows ≤ N = its own adaptive state (instead of the fixed HIVE_DMA_FRAC_PREFILL)
//     HIVE_PREFILL_PROF         [prefill-prof] phase breakdown
//   Same convention (off = default path):
//     HIVE_PREFILL_MULTI_TAIL_SHORT  route upper-layer units of forward_multi (host tiles / multiple sequences) with rows < prefill_threshold to the short-prefill path (the forward_multi counterpart of TAIL_SHORT)
//     HIVE_PREFILL_SMALL             send the experts of chunks below prefill_threshold (all layers, all rows) through the streaming path (moe_experts — staging-ring DMA + CPU for misses with few rows).
//                                    Tail mode stays off there (only for rows ≥ prefill_threshold) — decoder replay is not extended below that
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace hive {
namespace sp {

// HIVE_PREFILL_SHORT_ADAPT value → row cap N (0 = off). Off = unset / "" / "0". On = an integer ≥ 2 is used as is; anything else ("1", "yes", negative, overflow) = 4096 (absorbed, not rejected).
//   4096 = the upper end of the target range (1K–4K prompts) — a range definition, not a measured value.
inline int parse_short_adapt(const char* v) {
  if (!v || !*v || !std::strcmp(v, "0")) return 0;
  char* end{nullptr};
  const long n = std::strtol(v, &end, 10);
  return end != v && end && *end == 0 && n >= 2 && n <= (1L << 30) ? (int)n : 4096;
}

// HIVE_PREFILL_SMALL value → row floor (0 = off). Off = unset / "" / "0". On = an integer N ≥ 2 gives N; anything else ("1", non-numeric, negative) = no floor —
//   either way the result is raised to max(·, max_batch + 1, 9) (absorbed, not rejected). These are two existing boundaries, not new constants:
//   · max_batch + 1 = where moe_decode_experts already counts rows as "short prefill" (kDmaShort, M > max_batch) — below that is the decode kind.
//   · 9 = the fused decode kernel boundary (the M ≤ 8 branch in moe_router_shared / layer_forward) — router/shared-expert results at those row counts have never been wired to the streaming expert path.
inline int parse_small(const char* v, int max_batch) {
  if (!v || !*v || !std::strcmp(v, "0")) return 0;
  char* end{nullptr};
  const long n = std::strtol(v, &end, 10);
  const long floor = std::max<long>(9, (long)max_batch + 1);
  const long want = end != v && end && *end == 0 && n >= 2 && n <= (1L << 30) ? n : floor;
  return (int)std::max(want, floor);
}
// Is this forward (single sequence, not tiled) a HIVE_PREFILL_SMALL chunk: floor ≤ M < prefill_threshold, and not verify / batch decode
inline bool small_forward(int floor, bool verify, bool batch, bool tiled, int M, int prefill_threshold) {
  return floor > 0 && !verify && !batch && !tiled && M >= floor && M < prefill_threshold;
}
// Should one upper-layer unit of forward_multi (l ≥ last kv-source layer) take the short-prefill path (same row condition as short_moe for TAIL_SHORT — forward_multi is always prefill, never verify or batch)
inline bool multi_short(bool on, bool cpu_for_misses, int M, int prefill_threshold) { return on && cpu_for_misses && M < prefill_threshold; }

// Does this layer take the short tail path (same formula as Runtime::prefill_short_moe in runtime.cpp)
inline bool short_moe(bool on, bool in_prefill, bool verify, bool batch, bool cpu_for_misses, int M, int prefill_threshold) {
  return on && in_prefill && !verify && !batch && cpu_for_misses && M < prefill_threshold;
}

// Prefill isolation — the DMA-share kind used by moe_decode_experts: true = kDmaShort (short-prefill state), false = kDmaDecode (decode state).
//   Looking at the row count alone (M > max_batch → short prefill) is not enough: **prefill** layers (in_prefill) routed here by TAIL_SHORT / MULTI_TAIL_SHORT
//   can have ≤ max_batch rows (--decoder-tail ≤ max_batch, small forward_multi units). They would then count as decode, read and modify dma_frac_[kDmaDecode]
//   (adaptation / rollback), record decode samples and feed the HIVE_DECODE_SPLIT cost-model samples and CPU table. Prefill layers always use the prefill state.
//   With in_prefill = false the formula is the row-count rule (with those switches off no call arrives here with in_prefill = true — moe() and forward_multi take the streaming path then).
inline bool decode_path_short_kind(int M, int max_batch, bool in_prefill) { return M > max_batch || in_prefill; }

// Prefill isolation — usage observations for the upper layers of forward_multi (HIVE_PREFILL_MULTI_TAIL_SHORT), with the same formula and order as a single moe_experts_multi(subs) call:
//   weight = phase_score ? 1/(sum of sub-chunk rows) : 1 · row = 0 (that sequence) for one sub-chunk, -1 (no owner) for several · sub-chunk order → routing order (row, j) · hist per sub-chunk (trace 'H').
//   Elements of subs provide .M and .route_ids_h (M rows × k). observe(e, w, row) and hist(sub) are supplied by the caller (runtime.cpp: store_.observe, xtrace_hist).
template <class Subs, class Obs, class Hist>
inline void observe_like_multi(const Subs& subs, int k, bool phase_score, Obs&& observe, Hist&& hist) {
  int total = 0;
  for (const auto& sc : subs) total += sc.M;
  const float w = phase_score ? 1.f / std::max(1, total) : 1.f;
  const int row = subs.size() == 1 ? 0 : -1;
  for (const auto& sc : subs) {
    for (int i = 0; i < sc.M * k; ++i) observe(sc.route_ids_h[i], w, row);
    hist(sc);
  }
}

// DMA-share kind for a streaming layer: true (= kDmaStreamShort) for a short-chunk layer — always false (= kDmaStream) when adapt_rows = 0 (off)
inline bool short_stream_kind(int adapt_rows, int total_rows) { return adapt_rows > 0 && total_rows <= adapt_rows; }

// Prefetch prediction scaling ratio num/den: predicted rows = lr[e]·num/den. Off, rows_now ≤ 0 or previous sum 0 = 1/1 (unscaled rule).
//   Previous chunk rows = Σ lr / kact (top-k routings per row) → prediction = lr·rows_now/(Σlr/kact) = lr·(rows_now·kact)/Σlr.
inline void prefetch_ratio(bool on, int rows_now, int kact, const std::vector<int>& lr, int64_t& num, int64_t& den) {
  num = 1; den = 1;
  if (!on || rows_now <= 0 || kact <= 0) return;
  int64_t prev = 0;
  for (int v : lr) prev += v;
  if (prev > 0) { num = (int64_t)rows_now * kact; den = prev; }
}
// Candidate threshold: lr·num/den ≥ max(1, min_rows) (integer arithmetic, no rounding from division). With num = den = 1 this is the unscaled rule lr ≥ max(1, min_rows).
inline bool prefetch_pass(int lr_e, int64_t num, int64_t den, int min_rows) {
  return (int64_t)lr_e * num >= (int64_t)(min_rows > 1 ? min_rows : 1) * den;
}

}  // namespace sp
}  // namespace hive
