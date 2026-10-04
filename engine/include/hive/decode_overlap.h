// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Decode GPU idle-time reduction (used by runtime.cpp decode_layer / moe_decode_launch·finish / forward_batch). Pure
// helpers that compile without CUDA: the CPU test (engine/tests/test_decode_ubatch_cpu.cpp — fake CUDA stream threads
// + TSAN) and the GPU test (engine/tests/test_decode_ubatch.cu) call **this same production code**.
//
// (1) group_routes_sparse — expert grouping for HIVE_DECODE_HOST_FAST. The previous path (runtime.cpp moe_decode_experts)
//     did an E-slot (384) counting sort plus two E-slot scans. Here only the R (<= 48) routes are sorted to obtain the
//     used experts in ascending order, and cnt[e], cnt[e+1] (for used e only) and rows_by_e are filled with **the same
//     values** as before (experts ascending; within an expert, i = m*k+j ascending). cnt slots of unused e are left
//     untouched (the caller does not read them). The CPU test compares every slot against the previous formula on random
//     routings.
//
// (2) run_ubatch — layer pipeline for HIVE_DECODE_UBATCH (batch split into two halves, A=0 and B=1). Uses only the order
//     of a single stream (st_) plus event waits:
//        GPU: front A(l) · front B(l) · experts A(l) · experts B(l) · accum A(l) · front A(l+1) · accum B(l) · front B(l+1) · ...
//        CPU pool: misses of A(l) <- during front B(l) and the GPU experts; misses of B(l) <- during accum A and front A(l+1)
//     What the host order (exactly as in the body below) guarantees:
//       · launch(h,l) happens after wait_front(h,l) (routing and activations have reached mapped host memory)
//       · prep(h,l+1) (writing the mapped row table) happens after wait_front(h,l) — front(h,l), which reads that table,
//         has finished. The halves use different row ranges, so they never overlap the other half's table
//       · the CPU pool runs one batch at a time: start_cpu(B,l) after wait_cpu(A,l) (or directly in launch when A has no
//         CPU share)
//       · front(h,l+1) (= tail of layer l + head of layer l+1) comes after accum(h,l) (stream order)
//       · launch(h,l+1) rewrites half h's host table (tbl_h region) only after wait_front(h,l+1), i.e. after the expert
//         copy (h,l) that read it has finished on the stream
//     GPU scratch shared by the halves (xn, q, y, yq, eout, ...) is never used concurrently because every launch is on
//     one stream.
//     WARNING: changing this order is caught by the CPU test (fake GPU threads check the (half, layer) tags of the host
//     and mapped tables at execution time) and by the mutation negative controls.
#pragma once
#include <algorithm>
#include <cstdint>

namespace hive {
namespace dov {

// Routes ids[0..R) (each in [0,E)) -> used[0..nu) = used experts ascending; cnt[e] = total rows of preceding experts;
// cnt[e+1] = cnt[e] + n(e) (used e only); rows_by_e[cnt[e] + t] = the t-th i routed to e (ascending). tmp and fill are
// caller scratch (tmp >= R, fill >= E). Returns nu.
inline int group_routes_sparse(const int32_t* ids, int R, int* cnt, int* rows_by_e, int* used, int* tmp, int* fill) {
  for (int i = 0; i < R; ++i) tmp[i] = ids[i];
  std::sort(tmp, tmp + R);
  int nu = 0, run = 0;
  for (int i = 0; i < R;) {
    const int e = tmp[i];
    int j = i;
    while (j < R && tmp[j] == e) ++j;
    used[nu++] = e;
    cnt[e] = run; fill[e] = run;
    run += j - i;
    cnt[e + 1] = run;  // if the next used expert is e+1 it is rewritten with the same value (run) — consistent, since it is a prefix sum
    i = j;
  }
  for (int i = 0; i < R; ++i) rows_by_e[fill[ids[i]]++] = i;
  return nu;
}

// Half row ranges: A = [0, (M+1)/2), B = [(M+1)/2, M)
inline int half_rows(int M, int h) { return h == 0 ? (M + 1) / 2 : M / 2; }
inline int half_row0(int M, int h) { return h == 0 ? 0 : (M + 1) / 2; }

// Required Ops (all called on the host — async work is enqueued on st_ and the call returns):
//   prep(h,l)        host tables for half h, layer l (engram and attention row tables — mapped pinned)
//   front(h,l)       launch graph A (tail of layer l-1 + head of layer l) and record its completion event
//   wait_front(h,l)  host waits on that event (routing has arrived)
//   launch(h,l,go)   classification, CPU work setup (start the pool immediately if go), staging DMA, table H2D, GPU expert launch
//   cpu_idle(h,l)    whether half h's CPU share for layer l is empty or already done (can the pool be handed over now — non-blocking query)
//   start_cpu(h,l)   start the deferred CPU batch (no-op if already started or empty)
//   wait_cpu(h,l)    wait for the CPU batch to complete (returns immediately if none)
//   accum(h,l)       launch accumulation of the CPU results (st_)
template <class Ops>
void run_ubatch(int nL, Ops& o) {
  if (nL <= 0) return;
  o.prep(0, 0); o.front(0, 0);
  o.prep(1, 0); o.front(1, 0);
  for (int l = 0; l < nL; ++l) {
    const bool more = l + 1 < nL;
    o.wait_front(0, l); o.launch(0, l, true);
    o.wait_front(1, l); o.launch(1, l, o.cpu_idle(0, l));
    if (more) o.prep(0, l + 1);  // front A(l) is done (wait above) — switch A's tables to the next layer while CPU A runs
    o.wait_cpu(0, l); o.start_cpu(1, l); o.accum(0, l);
    if (more) o.front(0, l + 1);
    if (more) o.prep(1, l + 1);
    o.wait_cpu(1, l); o.accum(1, l);
    if (more) o.front(1, l + 1);
  }
}

}  // namespace dov
}  // namespace hive
