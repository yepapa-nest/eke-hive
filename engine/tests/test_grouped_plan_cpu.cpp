// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU test (no GPU) for the H1 HIVE_GROUPED_PREFILL batch plan (hive/moe_grouped_plan.h).
//   Builds random layers (pre-copied, L1 pre-issued, resident, streaming, multi-seg experts), simulates executing the plan and checks the invariants:
//   (1) slot numbers = the per-expert loop (next++ % S per streaming expert); (2) a copy is issued only after the slot's previous occupant is released (= the batch holding its last seg ran);
//   (3) until the batch consuming a seg runs, the slot still belongs to that expert (not overwritten); (4) batch rows <= cap, streaming <= str_cap; (5) seg order = input order, no omissions or duplicates.
//   Build: g++ -std=c++20 -O1 -Iengine/include engine/tests/test_grouped_plan_cpu.cpp -o /tmp/test_grouped_plan_cpu && /tmp/test_grouped_plan_cpu
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "hive/moe_grouped_plan.h"

using namespace hive::gp;

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { if (fails < 20) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } ++fails; } } while (0)

int main() {
  std::mt19937 rng(20260930);
  int cases = 0, total_batches = 0;
  for (int it = 0; it < 4000; ++it) {
    const int S = (it % 3 == 0) ? 8 : (it % 3 == 1 ? 48 : 1 + (int)(rng() % 64));
    const int cap = 16 + (int)(rng() % 2000);
    const int str_cap = std::max(1, S / 2);
    const int n_ex = 1 + (int)(rng() % 400);
    uint32_t next0 = rng() % 1000;
    // Pre-copied (pf): a few distinct slots -> front of order. L1 pre-issue: slots within one turn of the ring (stops at a pf slot) — mimics runtime.cpp build_order/plan_early_stream
    std::vector<Expert> ex;
    std::vector<int> seg_rows;
    std::vector<int> occupant(S, -1);  // current data owner of each slot (ex index); -1 = previous layer (release recorded)
    std::vector<char> freed(S, 1);
    std::vector<char> pf_busy(S, 0);
    const int n_pf = (int)(rng() % (S + 1)) / 2;
    std::vector<int> slots(S);
    for (int i = 0; i < S; ++i) slots[i] = i;
    std::shuffle(slots.begin(), slots.end(), rng);
    auto add_segs = [&](Expert& X) {
      X.seg_begin = (int)seg_rows.size();
      const int ns = 1 + (rng() % 5 == 0 ? (int)(rng() % 4) : 0);
      for (int g = 0; g < ns; ++g) seg_rows.push_back(1 + (int)(rng() % std::max(1, (rng() % 4 == 0) ? cap : cap / 8 + 1)));
      X.seg_end = (int)seg_rows.size();
    };
    for (int i = 0; i < n_pf && (int)ex.size() < n_ex; ++i) {
      Expert X{kPrecopied, slots[i], 0, 0}; add_segs(X);
      occupant[slots[i]] = (int)ex.size(); freed[slots[i]] = 0; pf_busy[slots[i]] = 1;
      ex.push_back(X);
    }
    std::vector<int> kinds;
    while ((int)ex.size() < n_ex) { Expert X{(int)(rng() % 2 ? kResident : kStream), -1, 0, 0}; add_segs(X); ex.push_back(X); }
    uint32_t next = next0;
    if (it % 2 == 0) {  // L1 pre-issue (same rule as runtime plan_early_stream)
      int taken = 0;
      for (int x = 0; x < (int)ex.size() && taken < S; ++x) {
        if (ex[x].kind != kStream) continue;
        const int si = (int)(next % (uint32_t)S);
        if (pf_busy[si]) break;
        ++next; ++taken;
        ex[x].kind = kPrecopied; ex[x].si = si;
        EXPECT(freed[si] || occupant[si] == -1, "early copy over unfreed slot (test setup)");
        occupant[si] = x; freed[si] = 0;
      }
    }
    // Expected slots (per-expert loop): remaining streaming experts take next++ % S in order
    std::vector<int> want_si(ex.size(), -1);
    {
      uint32_t nx = next;
      for (int x = 0; x < (int)ex.size(); ++x) if (ex[x].kind == kStream) want_si[x] = (int)(nx++ % (uint32_t)S);
    }
    std::vector<Action> acts;
    uint32_t pnext = next;
    plan_batches(ex, seg_rows, cap, str_cap, S, pnext, acts);
    int n_stream = 0;
    for (auto& X : ex) n_stream += X.kind == kStream;
    EXPECT(pnext == next + (uint32_t)n_stream, "next advance %u vs %u", pnext, next + n_stream);
    // Simulated execution
    std::vector<char> in_batch(S, 0);
    std::vector<int> batch_segs;
    int rows = 0, nstr = 0, expect_seg = 0;
    std::vector<int> x_si(ex.size(), -1);
    for (int x = 0; x < (int)ex.size(); ++x) if (ex[x].kind == kPrecopied) x_si[x] = ex[x].si;
    std::vector<char> slot_seen(S, 0);
    for (const Action& a : acts) {
      if (a.type == kCopy) {
        EXPECT(ex[a.x].kind == kStream, "copy for non-stream");
        EXPECT(a.si == want_si[a.x], "slot %d vs legacy %d", a.si, want_si[a.x]);
        EXPECT(!in_batch[a.si], "copy into slot held by the open batch");
        EXPECT(freed[a.si] || occupant[a.si] == -1, "copy before previous occupant freed (slot %d)", a.si);
        occupant[a.si] = a.x; freed[a.si] = 0; x_si[a.x] = a.si;
      } else if (a.type == kSeg) {
        EXPECT(a.seg == expect_seg, "seg order %d vs %d", a.seg, expect_seg);
        ++expect_seg;
        const Expert& X = ex[a.x];
        EXPECT(a.seg >= X.seg_begin && a.seg < X.seg_end, "seg owner");
        EXPECT(a.last == (a.seg + 1 == X.seg_end), "last flag");
        if (X.kind != kResident) {
          EXPECT(a.si == x_si[a.x] && a.si >= 0, "seg slot");
          EXPECT(occupant[a.si] == a.x, "slot data owner %d vs %d", occupant[a.si], a.x);
          if (!in_batch[a.si]) { in_batch[a.si] = 1; ++nstr; }
        }
        batch_segs.push_back(a.seg);
        rows += seg_rows[a.seg];
      } else {
        EXPECT(!batch_segs.empty(), "empty flush");
        EXPECT(rows <= cap || batch_segs.size() == 1, "batch rows %d > cap %d", rows, cap);
        EXPECT(nstr <= str_cap, "stream %d > cap %d", nstr, str_cap);
        for (int g : batch_segs) {  // the slot owner is still the same at execution time -> release
          int owner = -1;
          for (int x = 0; x < (int)ex.size(); ++x) if (g >= ex[x].seg_begin && g < ex[x].seg_end) owner = x;
          if (ex[owner].kind == kResident) continue;
          EXPECT(occupant[x_si[owner]] == owner, "slot overwritten before its batch ran");
          if (g + 1 == ex[owner].seg_end) freed[x_si[owner]] = 1;
        }
        for (int s = 0; s < S; ++s) in_batch[s] = 0;
        batch_segs.clear(); rows = 0; nstr = 0; ++total_batches;
      }
    }
    EXPECT(batch_segs.empty(), "unflushed tail");
    EXPECT(expect_seg == (int)seg_rows.size(), "segs %d vs %d", expect_seg, (int)seg_rows.size());
    ++cases;
  }
  printf("test_grouped_plan_cpu: %d cases, %d batches, %d failures\n", cases, total_batches, fails);
  printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
