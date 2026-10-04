// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_PREGATE=K — reads the CPU-missed experts of a decode layer ahead of time while attention runs (pre-gate + owned-item prefetch).
//
// Rationale (measured in real use; see docs/performance.md):
//   · CPU expert time is linear in the number of (expert, row) items — [decode-host] regression over 9,753 steps: verify M=2..4 = 0.18 ms/item (R² 0.98–0.99), M=1 = 0.20 ms/item.
//     17.9 MB record / 0.185 ms ≈ 97 GB/s ≈ the DRAM bandwidth of both nodes (61 GB/s measured per node) — the CPU share is bound by DRAM reads.
//   · Those DRAM reads only start **after** routing is known. During the attention phase before it (0.25–0.42 ms per layer = [decode-host] front 10.1–16.9 ms / 40 layers)
//     the CPU pool is idle. GPU idle time (tail = how much later the CPU finishes than the GPU experts) is 2.9 ms at M=1 and 4.7 ms at verify M=2 per step (out of 19.0 / 24.8 ms).
//   · The routing of layer l is predicted fairly well **before** attention: by the reference structure the FFN input = ffn_norm(hc_pre(h_after_attn, attn_pre)), and attn_pre is already
//     computed from h_in at the layer head → prediction = gate_l(ffn_norm(hc_pre(h_in, attn_pre))) (only the attention contribution is missing). Recall measured on engine dumps (17 tokens × layers 1–28):
//     top 6 = 0.800 · 8 = 0.874 · 10 = 0.903 · 12 = 0.925 (higher in deeper layers: layer 1 0.41, layer 15 0.97).
// Behaviour (when on — K = number of top experts to predict, 1..16):
//   GPU: right after hc_attn_pre at the layer head (after hc_join_[0], which produces attn_pre), on a separate stream: hc_pre_norm(h_in) → router (top K) → mapped host post (same kernel as er_route_post).
//        h is overwritten by hc_post after attention, so st_ waits before that for the prediction's h reads to finish (pg_read), and at the end of the front end for the whole prediction (pg_done) (shorter than attention, so free).
//   host: right after launching the front end it waits for the post (it arrives before the routing post) and hands the non-resident predicted experts to the CPU pool as prefetch.
//   CPU pool: owned mode — the worker for each item (a job's phase-1 block / phase-2 block) is fixed as owner(e, idx), and each worker is pinned to one physical core. In prefetch each worker
//        reads **only the rows it will later compute itself** (they stay in L2 → the L3 victim cache of the same CCX; on Zen2 L3 is per CCX, so if another worker computed them the gain would be bound by IF bandwidth).
//        When a new job batch arrives (start_jobs) prefetch stops (routing is known, so compute comes first).
// Lossless: computation uses the same items, same functions, same inputs — only which worker does it differs (independent accumulation per row; phase-2 after the phase-1 barrier). Prefetch only reads.
//   However, the DMA/CPU share adaptation (dma_frac) looks at CPU completion time, so a faster CPU may change which device gets a miss (low bits that already varied with timing — checked with the quality suite).
//   A wrong prediction only wastes DRAM reads in an idle phase. The GPU side writes only to separate buffers (the real routing and h are only read).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace hive::pg {

constexpr int kMaxK = 16;     // limit of the router kernel (fused_router)
constexpr int kMaxPf = 64;    // per-layer prefetch expert limit (only non-resident ones out of M ≤ 8 × K ≤ 16 = 128 candidates — only ~2 per node fit in the window, so this is ample)

// Switch parsing (option convention — unset/""/"0" = off, an on-value is taken as is): unset, empty, "0" = 0 (off) · "1", non-numeric, negative = on with default kDefaultK · 2..16 = that value ·
//   above that 16 (kernel limit — absorbed, not rejected). Default 8 = where dump recall (top 6 0.800 · 8 0.874 · 10 0.903 · 12 0.925) flattens to about +0.03 per +2 beyond 8 —
//   while the prediction list grows by 1/3 per +2 (wrong prefetch = DRAM reads in the idle phase). Other values are chosen by GPU A/B.
constexpr int kDefaultK = 8;
inline int parse_k(const char* v) {
  if (!v || !*v || !strcmp(v, "0")) return 0;
  char* end = nullptr;
  const long x = strtol(v, &end, 10);
  if (end == v || *end != 0 || x <= 1) return kDefaultK;
  return (int)std::min<long>(x, kMaxK);
}

// Worker responsible for an item. idx = item number within the job (phase-1 block b → b, phase-2 block r → nblk + r). The start worker is rotated per expert (× 7 — coprime with 16)
//   so that with several jobs in a layer the same worker does not always get the remainder. T = workers per node.
inline int owner(int e, int idx, int T) { return T <= 1 ? 0 : (int)(((unsigned)idx + (unsigned)e * 7u) % (unsigned)T); }

// Row ranges (half-record local rows) worker t owns in expert e — same formula as the item layout of start_jobs (P1, P2, row split).
struct Span { int phase, r0, r1; };  // phase 1 = w1/w3 (rows r0..r1 of I2) · 2 = w2 (rows of D2)
template <class F>
inline void owned_spans(int e, int t, int T, int I2, int D2, int P1, int P2, F&& f) {
  const int nblk = (I2 + P1 - 1) / P1;
  int idx = 0;
  for (int r = 0; r < I2; r += P1, ++idx) if (owner(e, idx, T) == t) f(Span{1, r, std::min(r + P1, I2)});
  idx = nblk;
  for (int r = 0; r < D2; r += P2, ++idx) if (owner(e, idx, T) == t) f(Span{2, r, std::min(r + P2, D2)});
}

// Candidate cleanup: ids[M×K] (top K per row in score order) → non-resident experts into out, ordered by (predicted row count desc, best rank asc, id asc).
//   resident(e) = true if currently VRAM-resident (or already going to the GPU) — not a prefetch target. Returns the count (≤ cap).
template <class Resident>
inline int plan(const int32_t* ids, int M, int K, int E, Resident&& resident, int* out, int cap) {
  struct C { int e, rows, rank; };
  std::vector<C> c;
  c.reserve((size_t)M * K);
  for (int m = 0; m < M; ++m)
    for (int j = 0; j < K; ++j) {
      const int e = ids[(size_t)m * K + j];
      if (e < 0 || e >= E) continue;  // invalid values are absorbed (only the prefetch is skipped)
      bool dup_row = false;           // duplicates within a row (the kernel never emits them; defensive)
      for (int q = 0; q < j; ++q) if (ids[(size_t)m * K + q] == e) dup_row = true;
      if (dup_row) continue;
      auto it = std::find_if(c.begin(), c.end(), [&](const C& x) { return x.e == e; });
      if (it == c.end()) c.push_back({e, 1, j});
      else { ++it->rows; it->rank = std::min(it->rank, j); }
    }
  std::sort(c.begin(), c.end(), [](const C& a, const C& b) {
    if (a.rows != b.rows) return a.rows > b.rows;
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.e < b.e;
  });
  int n = 0;
  for (const C& x : c) {
    if (n >= cap) break;
    if (resident(x.e)) continue;
    out[n++] = x.e;
  }
  return n;
}

// [pregate] totals — prediction (handed over as non-resident) vs the actual CPU job experts
struct Acc {
  long layers = 0, posted = 0, late = 0, untrusted = 0;  // late = layers where the front end finished first and the post could not be used (absorbed)
  long pred = 0, actual = 0, covered = 0;              // experts handed to prefetch · actual CPU experts · of those, the handed ones
  void add_layer(const int* pf, int n_pf, const int* cpu_e, int n_cpu) {
    ++layers; pred += n_pf; actual += n_cpu;
    for (int i = 0; i < n_cpu; ++i)
      for (int j = 0; j < n_pf; ++j) if (pf[j] == cpu_e[i]) { ++covered; break; }
  }
  double recall() const { return actual ? (double)covered / (double)actual : 0.0; }
  double precision() const { return pred ? (double)covered / (double)pred : 0.0; }
};

}  // namespace hive::pg
