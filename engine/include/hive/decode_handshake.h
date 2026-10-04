// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_DECODE_STEP_GRAPH — one decode step (40 layers + head) as a single CUDA graph: the per-layer host sync becomes a GPU<->CPU handshake.
//
// Baseline (decode_layer): per layer replay graph A -> cudaStreamSynchronize -> the host reads route_ids_h and (a) classifies resident/DMA/CPU (b) observe (c) CPU jobs and activation conversion
//   (d) group-table H2D -> launch the expert kernels eagerly. Measured (perf on a service build): 59 % of the engine thread's 100 % is that cuStreamSynchronize; CPU package 175 W.
// This variant: (a)(d) on the GPU (hs_plan — same rules with a device copy of the slot table), (c) by a dispatcher thread on the CPU pool side (mapped pinned mailbox), (b) replayed
//   by the host in the same order after the step (nothing inside the step reads observe results, so cache decisions match the baseline — see the replay comment in runtime.cpp).
//
// This file = POD types and planning functions shared by device and host (including the sort) + kernel launch declarations (decode_handshake.cu). The dispatcher (host thread) is in decode_dispatch.h.
#pragma once
#include <cstddef>
#include <cstdint>

#include "hive/common.h"
#include "hive/kernels.h"  // k::GroupDesc

#ifndef HS_HD
#define HS_HD __host__ __device__
#endif

namespace hive {
namespace hs {

constexpr int kMaxE = 512;      // same bound as HIVE_CHECK(E <= 512) in moe_decode_experts
constexpr int kGroupRows = 8;   // max rows per group / CPU job (= mx_grouped kernel, ExpertStore::kMaxRows)
constexpr int kMaxDma = 8;      // max DMA per decode layer (dma_cap 8)
constexpr int kMaxR = 1024;     // row bound for the plan kernel's shared memory (M*k — max_batch 64 x k 6 = 384)

// ---- Mapped pinned mailbox (written by the GPU and read by the dispatcher, or vice versa) ----------------------------------------------------
struct alignas(64) Ctrl {
  uint32_t post_seq;      // GPU -> CPU: number of the last posted layer (seq = seq_base + l + 1). Written after all contents, following __threadfence_system
  uint32_t pad0[15];
  uint32_t done_seq;      // CPU -> GPU: post number finished (release after all of cpu_out is written)
  uint32_t pad1[15];
  uint32_t err;           // GPU: 1 = CPU wait timeout, 2 = DMA wait timeout (bitwise OR)
  uint32_t err_seq;       // post number of the first timeout
  uint32_t cpu_err;       // dispatcher: exception while handling a post (1)
  uint32_t pad2[13];
};
struct DmaItem { int32_t e, si; };                 // expert to upload into staging ring slot si
struct JobItem {                                    // one CPU job (= one ExpertStore::Job — same order and row grouping)
  int32_t e, R, row0;                              // row0 = first row in cpu_out (job_rows)
  int32_t m[kGroupRows];                           // row (sequence) indices
  float rw[kGroupRows];                            // routing weights
};
struct PostHdr {
  uint32_t seq;
  int32_t layer, M, n_jobs, n_dma, job_rows;
  DmaItem dma[kMaxDma];
};
// One layer's classification result (the device copy is read by kernels; the log copy is checked by the host after the step)
struct LayerCounts { int32_t ng_hit, ng_str, R_hit, R_str, n_jobs, job_rows, n_miss, n_dma; };
static_assert(offsetof(LayerCounts, ng_hit) == 0 && offsetof(LayerCounts, ng_str) == 4 && offsetof(LayerCounts, R_hit) == 8 && offsetof(LayerCounts, R_str) == 12,
              "hs_layer_experts reads LayerCounts as int[] (DfDevCount indices 0..3)");
// Per step host -> device (H2D on the same stream before the graph launch)
struct Params {
  uint32_t seq_base;                // post number base for this step (layer l = seq_base + l + 1)
  int32_t stage_base;               // next staging ring slot (stage_next_) — the GPU advances it by the DMA count per layer
  int32_t n_staging;
  int32_t pad;
  unsigned long long timeout_ns;    // GPU wait limit (%globaltimer)
  int32_t share[kMaxE + 1];         // miss count n -> DMA share (computed by the host with the host formula — the floating-point formula is not recomputed on the device)
};
struct DevMisc { int32_t dev_err, stage_ctr; uint32_t dma_flag, pad; };  // device only (first 8 bytes zeroed at step start — dma_flag is monotonic and not cleared)

// Matrix offsets inside an expert record (same values as ExpertLayout)
struct RecOff { size_t w1, s1, w3, s3, w2, s2, total; };

// Layout inside one mailbox block (mapped pinned) — host and device pointers use the same offsets
struct BoxLayout {
  size_t ctrl, hdr, jobs, xq, xs, route_log, counts_log, time_log, total;
  int Jcap = 0, Mb = 0, k = 0, dim = 0, nL = 0;
  static BoxLayout make(int Mb, int k, int dim, int nL) {
    BoxLayout b{};
    auto al = [](size_t x) { return (x + 63) / 64 * 64; };
    b.Mb = Mb; b.k = k; b.dim = dim; b.nL = nL; b.Jcap = Mb * k;
    size_t o = 0;
    b.ctrl = o; o = al(o + sizeof(Ctrl));
    b.hdr = o; o = al(o + sizeof(PostHdr));
    b.jobs = o; o = al(o + sizeof(JobItem) * (size_t)b.Jcap);
    b.xq = o; o = al(o + (size_t)Mb * dim);
    b.xs = o; o = al(o + (size_t)Mb * dim / 32);
    b.route_log = o; o = al(o + sizeof(int32_t) * (size_t)nL * Mb * k);
    b.counts_log = o; o = al(o + sizeof(LayerCounts) * (size_t)nL);
    b.time_log = o; o = al(o + sizeof(unsigned long long) * 2 * (size_t)nL);
    b.total = o;
    return b;
  }
};

// ---- Sort: a port of the libstdc++ (GCC 13) std::sort algorithm (introsort -> insertion sort at <= 16, heap sort at depth limit = 2*floor(log2 n)) ----------
//   The host code orders missed experts by row count descending with std::sort(miss_e, ..., n_of(a) > n_of(b)). std::sort is not stable, so
//   the order of equal counts depends on the algorithm, and that order decides which experts go to DMA (GPU) or CPU and the accumulation order. To get the same result on the device
//   it must be the same algorithm — the device plan and the host replay both use this port, and the CPU test (test_decode_handshake_cpu) compares it with std::sort on random inputs.
//   Comparison: a before b <=> cnt[a] > cnt[b].
namespace sortimpl {
HS_HD inline bool lt(int a, int b, const int32_t* cnt) { return cnt[a] > cnt[b]; }
HS_HD inline void swp(int* x, int* y) { int t = *x; *x = *y; *y = t; }
HS_HD inline void unguarded_linear_insert(int* last, const int32_t* cnt) {
  int val = *last; int* next = last - 1;
  while (lt(val, *next, cnt)) { *last = *next; last = next; --next; }
  *last = val;
}
HS_HD inline void insertion_sort(int* first, int* last, const int32_t* cnt) {
  if (first == last) return;
  for (int* i = first + 1; i != last; ++i) {
    if (lt(*i, *first, cnt)) { int val = *i; for (int* p = i; p != first; --p) *p = *(p - 1); *first = val; }
    else unguarded_linear_insert(i, cnt);
  }
}
HS_HD inline void push_heap(int* first, long hole, long top, int value, const int32_t* cnt) {
  long parent = (hole - 1) / 2;
  while (hole > top && lt(first[parent], value, cnt)) { first[hole] = first[parent]; hole = parent; parent = (hole - 1) / 2; }
  first[hole] = value;
}
HS_HD inline void adjust_heap(int* first, long hole, long len, int value, const int32_t* cnt) {
  const long top = hole;
  long second = hole;
  while (second < (len - 1) / 2) {
    second = 2 * (second + 1);
    if (lt(first[second], first[second - 1], cnt)) second--;
    first[hole] = first[second];
    hole = second;
  }
  if ((len & 1) == 0 && second == (len - 2) / 2) {
    second = 2 * (second + 1);
    first[hole] = first[second - 1];
    hole = second - 1;
  }
  push_heap(first, hole, top, value, cnt);
}
HS_HD inline void make_heap(int* first, int* last, const int32_t* cnt) {
  const long len = last - first;
  if (len < 2) return;
  long parent = (len - 2) / 2;
  for (;;) {
    int value = first[parent];
    adjust_heap(first, parent, len, value, cnt);
    if (parent == 0) return;
    parent--;
  }
}
HS_HD inline void pop_heap(int* first, int* last, int* result, const int32_t* cnt) {
  int value = *result; *result = *first;
  adjust_heap(first, 0, last - first, value, cnt);
}
HS_HD inline void heap_sort_all(int* first, int* last, const int32_t* cnt) {  // __partial_sort(first, last, last) = heap_select (empty loop) + sort_heap
  make_heap(first, last, cnt);
  while (last - first > 1) { --last; pop_heap(first, last, last, cnt); }
}
HS_HD inline void move_median_to_first(int* result, int* a, int* b, int* c, const int32_t* cnt) {
  if (lt(*a, *b, cnt)) {
    if (lt(*b, *c, cnt)) swp(result, b);
    else if (lt(*a, *c, cnt)) swp(result, c);
    else swp(result, a);
  } else if (lt(*a, *c, cnt)) swp(result, a);
  else if (lt(*b, *c, cnt)) swp(result, c);
  else swp(result, b);
}
HS_HD inline int* unguarded_partition(int* first, int* last, int* pivot, const int32_t* cnt) {
  for (;;) {
    while (lt(*first, *pivot, cnt)) ++first;
    --last;
    while (lt(*pivot, *last, cnt)) --last;
    if (!(first < last)) return first;
    swp(first, last);
    ++first;
  }
}
HS_HD inline int lg(long n) { int r = 0; while (n > 1) { n >>= 1; ++r; } return r; }  // std::__lg = floor(log2 n)
// __introsort_loop's recursion (right part) as an explicit stack — processing order (right first, left iterated) matches the original. n <= kMaxE so depth <= 2*9.
HS_HD inline void introsort_loop(int* first, int* last, int depth, const int32_t* cnt) {
  struct Fr { int* f; int* l; int d; };
  Fr stk[64];
  int sp = 0;
  stk[sp++] = {first, last, depth};
  while (sp > 0) {
    Fr fr = stk[--sp];
    int* f = fr.f; int* l = fr.l; int d = fr.d;
    while (l - f > 16) {
      if (d == 0) { heap_sort_all(f, l, cnt); break; }
      --d;
      int* mid = f + (l - f) / 2;
      move_median_to_first(f, f + 1, mid, l - 1, cnt);
      int* cut = unguarded_partition(f + 1, l, f, cnt);
      // Original: run introsort_loop(cut, l, d) to completion first, then continue with l = cut — the two parts do not overlap, so deferring the right one on the stack
      // and doing the left first gives the same result (the array inside each part). Processing one part never touches the other's elements.
      stk[sp++] = {cut, l, d};
      l = cut;
    }
  }
}
}  // namespace sortimpl
HS_HD inline void sort_desc_by_count(int* a, int n, const int32_t* cnt) {
  if (n <= 0) return;
  sortimpl::introsort_loop(a, a + n, sortimpl::lg(n) * 2, cnt);
  if (n > 16) {
    sortimpl::insertion_sort(a, a + 16, cnt);
    for (int* i = a + 16; i != a + n; ++i) sortimpl::unguarded_linear_insert(i, cnt);
  } else sortimpl::insertion_sort(a, a + n, cnt);
}

// ---- Plan ----------------------------------------------------------------------------------------------------------------
// (1) Lists: cnt (rows per expert), off (counting-sort offsets), rows_by_e (ascending i), resident list (ascending expert id, slot), miss list (ascending).
//     Host version = plan_lists (same formulas as the host loop), device version = the hs_plan kernel computes the same values in parallel.
// (2) Tables: build_tables — sort misses -> first `share` go to DMA (staging slots), rest to CPU -> group table (resident -> DMA, row0 continues), g_rows/g_rw, CPU job table.
//     **Thread 0 of the device hs_plan and the host (replay, CPU tests) share this one function** — same order as add_group and the CPU job loop in moe_decode_experts.
HS_HD inline LayerCounts build_tables(const int32_t* cnt, const int32_t* off, const int32_t* rbe, const float* rw, int k, const int32_t* hit_e,
                                      const int32_t* hit_slot, int nh, int32_t* miss, int nm, int share, const uint8_t* slots_base,
                                      const uint8_t* staging_base, RecOff rec, int stage_base, int32_t* stage_ctr, int n_staging, k::GroupDesc* gd_hit,
                                      k::GroupDesc* gd_dma, int32_t* g_rows, float* g_rw, int32_t* cpu_rows, JobItem* jobs, DmaItem* dma) {
  sort_desc_by_count(miss, nm, cnt);  // = std::sort(miss_e, ..., n_of(a) > n_of(b))
  int goff = 0;
  auto add_group = [&](k::GroupDesc* gdst, int e, const uint8_t* recp) {  // as add_group: group = expert x rows <= 8; more rows continue in further groups
    const int e0 = off[e], e1 = off[e] + cnt[e];
    for (int i0 = e0; i0 < e1; i0 += kGroupRows) {
      k::GroupDesc d;
      d.w1 = recp + rec.w1; d.s1 = recp + rec.s1; d.w3 = recp + rec.w3; d.s3 = recp + rec.s3; d.w2 = recp + rec.w2; d.s2 = recp + rec.s2;
      d.row0 = goff; d.n = (e1 - i0) < kGroupRows ? (e1 - i0) : kGroupRows;
      for (int i = i0; i < i0 + d.n; ++i) { const int mj = rbe[i]; g_rows[goff] = mj / k; g_rw[goff] = rw[mj]; ++goff; }
      *gdst++ = d;
    }
    return gdst;
  };
  LayerCounts c{};
  k::GroupDesc* gp = gd_hit;
  for (int x = 0; x < nh; ++x) gp = add_group(gp, hit_e[x], slots_base + (size_t)hit_slot[x] * rec.total);
  c.ng_hit = (int)(gp - gd_hit); c.R_hit = goff;
  int share_left = share, n_str = 0, n_jobs = 0, job_rows = 0;
  k::GroupDesc* gq = gd_dma;
  for (int x = 0; x < nm; ++x) {
    const int e = miss[x];
    if (share_left > 0) {
      --share_left;
      const int si = (stage_base + (*stage_ctr)++) % n_staging;  // = stage_next_++ % slot count
      gq = add_group(gq, e, staging_base + (size_t)si * rec.total);
      dma[n_str].e = e; dma[n_str].si = si;
      ++n_str;
    } else {  // CPU jobs: per expert, batches of rows <= kMaxRows (8); cpu_out rows in job_rows order
      const int e0 = off[e], e1 = off[e] + cnt[e];
      for (int i0 = e0; i0 < e1; i0 += kGroupRows) {
        JobItem* jb = jobs + n_jobs++;
        const int R8 = (e1 - i0) < kGroupRows ? (e1 - i0) : kGroupRows;
        jb->e = e; jb->R = R8; jb->row0 = job_rows;
        for (int r = 0; r < R8; ++r) {
          const int mj = rbe[i0 + r], m = mj / k;
          jb->m[r] = m; jb->rw[r] = rw[mj];
          cpu_rows[job_rows++] = m;
        }
      }
    }
  }
  c.ng_str = (int)(gq - gd_dma); c.R_str = goff - c.R_hit;
  c.n_jobs = n_jobs; c.job_rows = job_rows; c.n_miss = nm; c.n_dma = n_str;
  return c;
}

// Whole host version (replay, tests). The struct owns the buffers (no allocation — ~90 KB; the caller creates it once).
struct HostPlan {
  int32_t cnt[kMaxE];
  int32_t off[kMaxE + 1];
  int32_t rows_by_e[kMaxR];
  int32_t hit_e[kMaxE], hit_slot[kMaxE], miss_e[kMaxE], miss_sorted[kMaxE];
  int n_hit_e = 0, n_miss_e = 0, n_str_e = 0, n_cpu_e = 0, n_miss = 0;
  const int32_t* str_e() const { return miss_sorted; }             // [0, n_str_e)
  const int32_t* cpu_e() const { return miss_sorted + n_str_e; }   // [0, n_cpu_e)
  k::GroupDesc gd_hit[kMaxR], gd_dma[kMaxR];
  int32_t g_rows[kMaxR], cpu_rows[kMaxR];
  float g_rw[kMaxR];
  JobItem jobs[kMaxR];
  DmaItem dma[kMaxDma];
  int32_t stage_ctr = 0;
  LayerCounts counts{};
};
// (1) Lists — same formulas as the counting-sort/classification loop in moe_decode_experts
inline void plan_lists(const int32_t* ids, int M, int k, int E, const int32_t* slot_row, HostPlan& p) {
  const int R = M * k;
  for (int e = 0; e < E; ++e) p.cnt[e] = 0;
  for (int i = 0; i < R; ++i) ++p.cnt[ids[i]];
  p.off[0] = 0;
  for (int e = 0; e < E; ++e) p.off[e + 1] = p.off[e] + p.cnt[e];
  {
    int fill[kMaxE];
    for (int e = 0; e < E; ++e) fill[e] = p.off[e];
    for (int i = 0; i < R; ++i) p.rows_by_e[fill[ids[i]]++] = i;
  }
  p.n_hit_e = p.n_miss_e = 0;
  for (int e = 0; e < E; ++e) {
    if (p.cnt[e] == 0) continue;
    if (slot_row[e] >= 0) { p.hit_slot[p.n_hit_e] = slot_row[e]; p.hit_e[p.n_hit_e++] = e; }
    else p.miss_e[p.n_miss_e++] = e;
  }
  p.n_miss = p.n_miss_e;
}
// (1)+(2). ids[M*k], rw[M*k] (0 if absent), slot_row[E] (-1 = not resident), share[n] = DMA share for n misses. Record pointers are fake bases (address arithmetic only).
inline void plan_host(const int32_t* ids, const float* rw, int M, int k, int E, const int32_t* slot_row, const int32_t* share, HostPlan& p,
                      const uint8_t* slots_base = nullptr, const uint8_t* staging_base = nullptr, RecOff rec = RecOff{}, int stage_base = 0,
                      int n_staging = 1) {
  plan_lists(ids, M, k, E, slot_row, p);
  for (int i = 0; i < p.n_miss_e; ++i) p.miss_sorted[i] = p.miss_e[i];
  static const float zero_rw[kMaxR] = {};
  p.stage_ctr = 0;
  p.counts = build_tables(p.cnt, p.off, p.rows_by_e, rw ? rw : zero_rw, k, p.hit_e, p.hit_slot, p.n_hit_e, p.miss_sorted, p.n_miss_e,
                          share[p.n_miss < kMaxE ? p.n_miss : kMaxE], slots_base, staging_base, rec, stage_base, &p.stage_ctr, n_staging, p.gd_hit,
                          p.gd_dma, p.g_rows, p.g_rw, p.cpu_rows, p.jobs, p.dma);
  p.n_str_e = p.counts.n_dma;
  p.n_cpu_e = p.n_miss_e - p.n_str_e;
}

}  // namespace hs

// ---- Kernel launches (decode_handshake.cu) — all graph-capture safe (no host sync, no host callbacks) ----------------------------------------
namespace k {
struct HsPlanArgs {
  // Inputs (device)
  const int32_t* ids; const float* rw; int M, k, E, l;
  const int32_t* slot_row;            // row l of the device slot table [E]
  const uint8_t* slots_base; const uint8_t* staging_base; hs::RecOff rec;
  const uint8_t* xq; const uint8_t* xs; int dim;
  const hs::Params* prm; hs::DevMisc* misc;
  // Outputs (device)
  hs::LayerCounts* cnt; GroupDesc* gd; int Gh, Gd; int32_t* g_rows; float* g_rw; int32_t* cpu_rows;
  // Mailbox (device pointers into mapped pinned memory)
  hs::Ctrl* ctrl; hs::PostHdr* hdr; hs::JobItem* jobs; uint8_t* bxq; uint8_t* bxs;
  int32_t* route_log_l; hs::LayerCounts* counts_log_l; unsigned long long* time_log_l;
};
void hs_plan(const HsPlanArgs& a, cudaStream_t st);
// Step start: zero DevMisc's dev_err and stage_ctr (memset node inside the graph)
void hs_step_reset(hs::DevMisc* misc, cudaStream_t st);
// All experts of a layer (plan -> resident groups -> DMA wait -> DMA groups -> CPU wait -> CPU accumulate). Same kernels and order as the fuse_ && use_mx chain (mx_grouped_w13 (row map + quantize) ->
//   mx_grouped_w2 -> accum_bf16_rows_seq). The group table is filled with Gh resident entries + Gd DMA entries (empty entries n = 0) so the grid is fixed.
struct HsLayerArgs {
  HsPlanArgs plan;
  int I; float limit;
  bf16* y; uint8_t* yq; uint8_t* ys; bf16* eout; float* acc;
  const float* cpu_out_d;            // device pointer of the mapped pinned CPU results [job_rows, dim]
  bool fused = false;                // HIVE_DECODE_FUSED variant (moe_decode_fused_dev — bit-identical to the unfused chain, verified on GPU). The caller checks shape/alignment once
};
void hs_layer_experts(const HsLayerArgs& a, cudaStream_t st);
// Called by the dispatcher on the side stream: after the DMA copy, flag <- value (monotonic)
void hs_signal(uint32_t* flag, uint32_t value, cudaStream_t st);
// Load every kernel of this path **before the first step** (cudaFuncGetAttributes + one synchronous hs_signal on side).
//   Cause (observed in test_decode_step_graph): only the very first step of a run (M=1 step 0) hit 0x2 (DMA wait timeout), with plan mismatches on that step's last 5 layers
//   (after a timeout hs_plan does not handle misses) — the first capture for each M (M=2,4,8) was bit-identical. CUDA lazy module loading (CUDA_MODULE_LOADING=LAZY, the default) tried to load
//   the dispatcher's first hs_signal on the spot; loading may need a context synchronisation and blocked behind the hs_wait_dma spin kernel
//   (the "Concurrent execution" deadlock case in the CUDA Lazy Loading docs). Once loading had completed (from the second step) it never reproduced.
//   side = the stream the dispatcher will use (launched and synchronised once here). Also preloads the fused kernels and the stream counter (fused_st).
void hs_preload(cudaStream_t side, cudaStream_t fused_st);
// HIVE_DECODE_EARLY_ROUTE (hive/early_route.h): write routing ids/rw[n] to mapped host memory (ids_h, rw_h — same copy and values as route_to_host), then
//   increment the device counter *ctr and write the value to the mapped u32 *flag_h (write -> __threadfence_system -> __syncthreads -> number). One block — graph-capture safe.
void er_route_post(const int32_t* ids, const float* rw, int n, int32_t* ids_h, float* rw_h, uint32_t* ctr, uint32_t* flag_h, cudaStream_t st);
}  // namespace k
}  // namespace hive
