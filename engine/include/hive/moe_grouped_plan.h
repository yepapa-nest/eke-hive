// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// H1 grouped prefill batch planning (HIVE_GROUPED_PREFILL) — host only (no CUDA). Tested on the CPU by engine/tests/test_grouped_plan_cpu.cpp.
//   Input = the expert list in exactly the order of the previous per-expert loop (resident, staged with copies already issued
//   (pre-copied / pre-issued), streaming) + per-expert seg row counts.
//   Output = an action list: issue copy (kCopy), add seg (kSeg), run batch (kFlush). If the executor performs them in this order:
//   (1) staging slot numbers match the previous loop (streaming experts take next++ % S in order); (2) when a copy is issued to
//   slot si, the slot's previous occupant is in an already executed batch (release event recorded) — the previous loop recorded
//   the release per expert before issuing the next copy; since a batch gathers several experts, the same invariant is kept by
//   "if the slot belongs to the batch being built, run that batch first"; (3) batch rows <= cap_rows (work buffer capacity);
//   (4) streaming experts per batch <= str_cap (half the ring — if one batch held the whole ring, the next batch's copies could
//   not be issued until its compute finished); (5) seg order = input order (accumulation order unchanged).
#pragma once
#include <cstdint>
#include <vector>

namespace hive::gp {

enum { kResident = 0, kPrecopied = 1, kStream = 2 };
struct Expert { int kind; int si; int seg_begin, seg_end; };  // si: input only for kPrecopied (ignored otherwise); seg [seg_begin, seg_end) = range in seg_rows
enum { kCopy = 0, kSeg = 1, kFlush = 2 };
struct Action { int type; int x; int si; int seg; bool last; };  // x = index into ex; last = that expert's last seg (slot released after this batch runs)

inline void plan_batches(const std::vector<Expert>& ex, const std::vector<int>& seg_rows, int cap_rows, int str_cap, int S, uint32_t& next,
                         std::vector<Action>& out) {
  out.clear();
  if (S < 1) S = 1;
  if (str_cap < 1) str_cap = 1;
  std::vector<char> busy((size_t)S, 0);  // slots held by the batch being built
  std::vector<int> busy_list;
  int rows = 0, n_str = 0, n_seg = 0;
  auto flush = [&] {
    if (n_seg == 0) return;
    out.push_back({kFlush, -1, -1, -1, false});
    for (int si : busy_list) busy[(size_t)si] = 0;
    busy_list.clear();
    rows = 0; n_str = 0; n_seg = 0;
  };
  for (int x = 0; x < (int)ex.size(); ++x) {
    const Expert& X = ex[(size_t)x];
    if (X.seg_end <= X.seg_begin) continue;
    const bool str = X.kind != kResident;
    int si = -1;
    if (str) {
      if (n_str >= str_cap) flush();
      if (X.kind == kStream) {
        si = (int)(next % (uint32_t)S);
        if (busy[(size_t)si]) flush();  // the previous occupant is in this batch — run it first so its release event is recorded
        ++next;
        out.push_back({kCopy, x, si, -1, false});
      } else {
        si = ((X.si % S) + S) % S;
        if (busy[(size_t)si]) flush();  // pre-copied slots are distinct, so this should not happen (absorbed if it does)
      }
    }
    for (int g = X.seg_begin; g < X.seg_end; ++g) {
      if (n_seg > 0 && rows + seg_rows[(size_t)g] > cap_rows) flush();  // may split mid-expert (the next batch re-acquires the slot — release happens after the batch of the last seg)
      if (str && !busy[(size_t)si]) { busy[(size_t)si] = 1; busy_list.push_back(si); ++n_str; }
      out.push_back({kSeg, x, si, g, g + 1 == X.seg_end});
      rows += seg_rows[(size_t)g];
      ++n_seg;
    }
  }
  flush();
}

}  // namespace hive::gp
