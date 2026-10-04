// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Miss handling for batched decode (forward_batch, M >= 2) — host-only logic (no CUDA). Called from runtime.cpp
//   moe_decode_experts and tested on the CPU by engine/tests/test_batch_miss_cpu.cpp. Three switches (all unset / "" / "0" = off =
//   the baseline path unchanged):
//
//   HIVE_DECODE_CPU_FIRST  — issue order only: classify → start CPU jobs → expert table H2D and resident-group launch → (only then)
//       issue the on-demand DMA copies → DMA group. Without it the on-demand DMA copies (12 cudaMemcpyAsync pieces per record plus
//       event waits/records) are issued *before* the CPU start and resident launch, so the CPU pool and the resident group start
//       late by that API time. Same experts, same device, same kernels, same accumulation order → identical values (only the host
//       issue time changes).
//   HIVE_DECODE_STAGE_HIT  — reuse records still intact in the staging ring (HIVE_STAGING slots), written by an earlier step's or
//       layer's on-demand DMA or by the prefetch below: a miss whose (layer, expert) record is in the ring with its copy finished goes
//       to the GPU group without any copy (neither CPU job nor DMA). If the copy is still in flight it becomes a DMA-share candidate
//       (waits on that copy instead of issuing a new one). Slot integrity = StageTrack::find (ring position distance — every staging
//       writer advances stage_next_ in turn, so a slot can only be rewritten S allocations later). The [decode-miss] line
//       (HIVE_PROFILE sample steps) counts reuse and prediction hits.
//   HIVE_DECODE_PREFETCH=N — (implies STAGE_HIT) cross-step prefetch: the misses layer l sent to the CPU = prediction (Pred) for the
//       same layer in the next step. In the next step, after layer l-1's experts are launched and only when the link is idle (side_
//       and promo_ streams empty, no pending promotion pieces), N of the predictions (ordered by consecutive-step streak, then rows)
//       are copied into staging on promo_ (cache slots are untouched — no eviction). If layer l uses such a record it is consumed via
//       the STAGE_HIT path.
//       N: positive integer (capped at kMaxPrefetch); any other "on" value ("on" etc.) = 1.
//
//   Numerics: when STAGE_HIT/PREFETCH moves a miss from the CPU to the GPU, the low bits of that expert's output may differ (rounding
//   of CPU fp32 block-accumulated GEMV → bf16 vs GPU mma accumulation → bf16; accumulation order: the GPU group goes first via
//   accum_bf16_rows_seq, CPU rows afterwards via accum_f32_rows). A reused record is a byte copy of the same host record, so the
//   weights are identical. A layer with no staging reuse (nothing in the ring) classifies exactly as the baseline (assign invariant
//   — tested). Because the choice depends on ring state and copy completion time, low bits may vary between runs on the same input
//   (the same property as the adaptive frac).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace hive::bmiss {

constexpr int kMaxPrefetch = 4;   // per-layer prefetch cap (record 18.8 MB, PCIe ~0.67 ms — the link cannot carry more within one layer's ~0.74 ms: 29.6 ms / 40 layers, batch sample)
constexpr int kLayerAlloc = 8;    // per-layer on-demand DMA cap (moe_decode_experts dma_cap — 8 for decode)

inline bool on(const char* v) { return v && *v && strcmp(v, "0") != 0; }
// HIVE_DECODE_PREFETCH: off = 0, positive integer = that value (capped at kMaxPrefetch), any other "on" value = 1 (absorbed, never rejected)
inline int parse_prefetch(const char* v) {
  if (!on(v)) return 0;
  char* end = nullptr;
  const long n = strtol(v, &end, 10);
  if (end != v && end && *end == 0 && n >= 1) return (int)std::min<long>(n, kMaxPrefetch);
  return 1;
}

// Staging ring tracking: key (= layer · n_routed + expert) → ring position that wrote the record + 1 (0 = none). Ring position p maps
//   to slot p % S. Every staging writer (decode DMA, prefill streaming/prefetch, G1 dispatcher) advances stage_next_ in turn to obtain
//   a slot, so the slot written at p is only rewritten once stage_next_ reaches p + S → if the current next plus the number this layer
//   may still allocate (margin) stays at or below p + S, the record is intact for the rest of this layer.
struct StageTrack {
  std::vector<uint32_t> pos1;
  std::vector<uint8_t> pf;  // record arrived via prefetch (cleared on consumption — the prefetch hit in [decode-miss])
  void put(size_t key, uint32_t pos, bool prefetched) {
    if (key >= pos1.size()) { pos1.resize(key + 1, 0); pf.resize(key + 1, 0); }
    pos1[key] = pos + 1; pf[key] = prefetched ? 1 : 0;
  }
  // Slot index if intact, else -1. next = current stage_next_, margin = how many more slots this layer may allocate after the check.
  int find(size_t key, uint32_t next, int S, int margin) const {
    if (S <= 0 || key >= pos1.size() || pos1[key] == 0) return -1;
    const uint32_t pos = pos1[key] - 1;
    const uint32_t age = next - pos;  // allocations since the write + 1 (>= 1; unsigned subtraction — the difference stays correct across 2^32 wrap)
    if (age == 0 || (uint64_t)age + (uint64_t)std::max(0, margin) > (uint64_t)S) return -1;
    return (int)(pos % (uint32_t)S);
  }
  bool take_pf(size_t key) {  // mark consumed (was it a prefetch?)
    if (key >= pf.size() || !pf[key]) return false;
    pf[key] = 0;
    return true;
  }
};

// Classification input for one miss (baseline order = after sorting by rows descending). stage: -1 = not in ring, 0 = in ring with copy in flight, 1 = copy done
struct MissIn { int e; int rows; int stage; };
enum Where : uint8_t { kCpu = 0, kDmaCopy = 1, kDmaReuse = 2, kStaged = 3 };
// share(n) = the baseline DMA share formula (miss count n → number of DMAs — exactly moe_decode_experts' gpu_share_left; supplied by the caller).
// Rule: stage 1 → kStaged (uses no share — needs neither copy nor CPU). The remaining n' misses get share(n') DMA slots: first stage 0
//   (reuse the in-flight copy, no new copy) in row order, then the leftover slots go to misses not in the ring, in row order (kDmaCopy —
//   the baseline "largest row count first to DMA"). Everything else is kCpu.
//   Invariant: if every stage is -1 the result equals the baseline (first share(n) DMA, rest CPU).
template <class Share>
inline void assign(const MissIn* m, int n, Share share, uint8_t* where) {
  int rest = 0;
  for (int i = 0; i < n; ++i) rest += m[i].stage != 1 ? 1 : 0;
  int left = std::max(0, std::min(rest, (int)share(rest)));
  for (int i = 0; i < n; ++i) where[i] = m[i].stage == 1 ? kStaged : kCpu;
  for (int i = 0; i < n && left > 0; ++i) if (m[i].stage == 0) { where[i] = kDmaReuse; --left; }
  for (int i = 0; i < n && left > 0; ++i) if (m[i].stage < 0) { where[i] = kDmaCopy; --left; }
}

// Cross-step prediction: per layer, the misses that went to the CPU in the previous batch step (expert, rows, consecutive-step streak).
struct Ent { int e, rows, streak; };
struct Pred {
  std::vector<std::vector<Ent>> layer;
  std::vector<Ent> tmp;
  const std::vector<Ent>* at(int l) const { return l >= 0 && (size_t)l < layer.size() ? &layer[(size_t)l] : nullptr; }
  // Replace with this step's CPU share for layer l (e[i], rows[i]) — experts present in the previous step get streak + 1
  void update(int l, const int* e, const int* rows, int n) {
    if (l < 0) return;
    if ((size_t)l >= layer.size()) layer.resize((size_t)l + 1);
    std::vector<Ent>& old = layer[(size_t)l];
    tmp.clear();
    for (int i = 0; i < n; ++i) {
      int s = 1;
      for (const Ent& o : old) if (o.e == e[i]) { s = std::min(o.streak + 1, 1 << 20); break; }
      tmp.push_back({e[i], rows[i], s});
    }
    old.swap(tmp);
  }
  // Prefetch candidates from the prediction: drop those where skip(e) (resident, or already intact in the ring), then sort by streak desc → rows desc → expert asc, at most n. Returns the count.
  template <class Skip>
  int pick(int l, int n, Skip skip, int* out) const {
    const std::vector<Ent>* p = at(l);
    if (!p || n <= 0) return 0;
    Ent c[64];
    int k = 0;
    for (const Ent& x : *p) if (k < 64 && !skip(x.e)) c[k++] = x;
    std::sort(c, c + k, [](const Ent& a, const Ent& b) { return a.streak != b.streak ? a.streak > b.streak : a.rows != b.rows ? a.rows > b.rows : a.e < b.e; });
    const int m = std::min(k, n);
    for (int i = 0; i < m; ++i) out[i] = c[i].e;
    return m;
  }
};

// [decode-miss] running totals (all batched layers since the last line)
struct Acc {
  long layers = 0, miss = 0, staged = 0, reuse = 0, copy = 0, cpu = 0, cpu_jobs = 0;
  long pred = 0, pred_used = 0;          // sum of prediction sizes, and how many of them actually missed this step (routed to non-resident) → precision = used/pred, recall = used/miss
  long pf_issued = 0, pf_used = 0, pf_busy = 0;  // prefetches issued, consumed (used via STAGE_HIT), skipped because the link was busy (layer count)
};

}  // namespace hive::bmiss
