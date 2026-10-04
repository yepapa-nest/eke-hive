// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// CPU test for hive/batch_miss.h (batched decode misses: STAGE_HIT ring tracking · classification · cross-step prediction). No GPU.
//   tools/test_batch_miss_cpu.py builds and runs it, and separately checks the runtime.cpp text contracts (switch parsing · off = baseline classification loop).
//   1. parse_prefetch: unset · "" · "0" = 0 · positive integer = that value (capped at kMaxPrefetch) · any other "on" value = 1 (absorbed)
//   2. StageTrack::find: in a world that mimics the staging ring (tracked writers + untracked writers that only advance stage_next_, like prefill and the G1 dispatcher)
//      (a) the content of the returned slot == that key — still that key even after margin more allocations following the decision (same-layer demand DMA · pre-copy) (safety)
//      (b) a tracked write within the margin is always found (completeness)
//   3. assign: with nothing on the ring (stage all -1) identical to the baseline loop slot by slot (first share(n) via DMA · rest on CPU) — 100k random layers · real share formula (frac · cap · gpu_share)
//      with ring entries mixed in: stage 1 is always kStaged (uses no share) · total DMA share = min(rest, share(rest)) · in flight (stage 0) first · row order within each class
//   4. Pred: streak (consecutive steps) · pick order (streak → row → expert) · skip · count cap
//   5. synthetic batched decode (routing with temporal locality): in the STAGE_HIT world reuse actually happens, the CPU share drops, and the safety check from 2 never fails.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <vector>

#include "hive/batch_miss.h"

using namespace hive::bmiss;

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

// baseline moe_decode_experts classification (misses in descending row order) — gpu_share_left goes to DMA from the front
static void legacy(int n, int share_n, std::vector<uint8_t>& where) {
  where.assign((size_t)n, kCpu);
  int left = share_n;
  for (int i = 0; i < n; ++i) if (left > 0) { --left; where[(size_t)i] = kDmaCopy; }
}

int main() {
  // 1) switch parsing
  struct { const char* v; int want; } cases[] = {{nullptr, 0}, {"", 0}, {"0", 0}, {"1", 1}, {"2", 2}, {"4", 4}, {"9", kMaxPrefetch},
                                                  {"on", 1}, {"true", 1}, {"-2", 1}, {"00", 1}, {"2x", 1}};
  for (auto& cs : cases) EXPECT(parse_prefetch(cs.v) == cs.want, "parse_prefetch(%s) = %d want %d", cs.v ? cs.v : "(unset)", parse_prefetch(cs.v), cs.want);
  EXPECT(!on(nullptr) && !on("") && !on("0") && on("1") && on("x"), "on()");

  std::mt19937 rng(20260930);
  // 2) ring world
  for (int S : {8, 16, 24, 96}) {
    for (int margin : {0, 4, 8, 12}) {
      StageTrack t;
      std::vector<long> content((size_t)S, -1);  // slot → key it holds (-1 = written by an untracked writer)
      std::vector<uint32_t> last_put(4096, 0);   // key → last tracked write position + 1
      uint32_t next = (uint32_t)(0xFFFFFF00u);   // also exercises 2^32 wrap-around
      long found = 0, checked = 0;
      for (int it = 0; it < 20000; ++it) {
        const int op = (int)(rng() % 10);
        if (op < 5) {  // tracked write (decode DMA · pre-copy)
          const long key = (long)(rng() % 64);
          const uint32_t pos = next++;
          content[pos % (uint32_t)S] = key;
          t.put((size_t)key, pos, rng() % 2);
          last_put[(size_t)key] = pos + 1;
        } else if (op < 7) {  // untracked writer (prefill streaming · G1): takes a slot and only writes
          const int n = 1 + (int)(rng() % (S / 2 + 1));
          for (int j = 0; j < n; ++j) content[(next++) % (uint32_t)S] = -1;
        } else {  // lookup → margin allocations after the decision (same layer) → still that key?
          const long key = (long)(rng() % 64);
          const int si = t.find((size_t)key, next, S, margin);
          ++checked;
          if (si >= 0) {
            ++found;
            EXPECT(content[(size_t)si] == key, "S=%d margin=%d: find slot %d holds %ld not %ld", S, margin, si, content[(size_t)si], key);
            std::vector<long> after = content;
            for (int j = 0; j < margin; ++j) after[(next + (uint32_t)j) % (uint32_t)S] = -2;
            EXPECT(after[(size_t)si] == key, "S=%d margin=%d: slot %d overwritten within the layer margin", S, margin, si);
          } else if (last_put[(size_t)key]) {
            const uint32_t pos = last_put[(size_t)key] - 1;
            const uint64_t age = (uint32_t)(next - pos);
            EXPECT(age + (uint64_t)margin > (uint64_t)S, "S=%d margin=%d: intact record (age %llu) not found", S, margin, (unsigned long long)age);
          }
        }
      }
      if (S == 96 && margin == 12) printf("batch_miss: ring world S=%d margin=%d — lookups %ld · found %ld (safety/completeness violations %d)\n", S, margin, checked, found, fails);
    }
  }
  {  // take_pf
    StageTrack t;
    t.put(5, 10, true); t.put(6, 11, false);
    EXPECT(t.take_pf(5) && !t.take_pf(5) && !t.take_pf(6) && !t.take_pf(999), "take_pf");
    EXPECT(t.find(5, 12, 96, 8) == 10 && t.find(6, 12, 96, 8) == 11 && t.find(7, 12, 96, 8) == -1, "find basic");
    EXPECT(t.find(5, 10 + 96 - 8 + 1, 96, 8) == -1 && t.find(5, 10 + 96 - 8, 96, 8) == 10, "find edge");
    EXPECT(t.find(5, 12, 0, 0) == -1 && t.find(5, 12, 8, 9) == -1, "find tiny ring");
  }

  // 3) assign
  long n_layers = 0, n_eq = 0;
  for (int it = 0; it < 100000; ++it) {
    const int n = (int)(rng() % 20);
    std::vector<MissIn> m((size_t)n);
    std::vector<int> rows((size_t)n);
    for (auto& r : rows) r = 1 + (int)(rng() % 8);
    std::sort(rows.begin(), rows.end(), std::greater<int>());
    const float frac = 0.1f + 0.7f * (float)(rng() % 1000) / 1000.f;
    const int gshare = (int)(rng() % 3), cap = 8, S = 8 + (int)(rng() % 90);
    auto share = [&](int k) { return k >= 3 ? std::min(std::min(cap, S), std::max(gshare, (int)(frac * k + 0.5f))) : 0; };
    const bool mixed = rng() % 2;
    for (int i = 0; i < n; ++i) m[(size_t)i] = {i, rows[(size_t)i], mixed ? (int)(rng() % 3) - 1 : -1};
    std::vector<uint8_t> w((size_t)n + 1), ref;
    assign(m.data(), n, share, w.data());
    ++n_layers;
    if (!mixed) {
      legacy(n, share(n), ref);
      bool eq = true;
      for (int i = 0; i < n; ++i) eq &= w[(size_t)i] == ref[(size_t)i];
      n_eq += eq;
      EXPECT(eq, "assign != legacy for n=%d share=%d", n, share(n));
      continue;
    }
    int rest = 0, dma = 0, reuse = 0, pend = 0;
    for (int i = 0; i < n; ++i) {
      if (m[(size_t)i].stage == 1) EXPECT(w[(size_t)i] == kStaged, "stage 1 not staged");
      else { ++rest; EXPECT(w[(size_t)i] != kStaged, "staged without done copy"); }
      if (m[(size_t)i].stage == 0) ++pend;
      if (w[(size_t)i] == kDmaCopy) { ++dma; EXPECT(m[(size_t)i].stage < 0, "copy of a ring record"); }
      if (w[(size_t)i] == kDmaReuse) { ++dma; ++reuse; EXPECT(m[(size_t)i].stage == 0, "reuse without pending copy"); }
    }
    const int want = std::min(rest, share(rest));
    EXPECT(dma == want, "dma %d want %d", dma, want);
    EXPECT(reuse == std::min(pend, want), "pending records must take the DMA share first (%d vs %d)", reuse, std::min(pend, want));
    // row order within each class: a copy must sit in an earlier slot (row ≥) than an off-ring miss sent to the CPU
    int last_copy = -1, first_cpu = n;
    for (int i = 0; i < n; ++i) {
      if (m[(size_t)i].stage >= 0) continue;
      if (w[(size_t)i] == kDmaCopy) last_copy = i;
      else if (first_cpu == n) first_cpu = i;
    }
    EXPECT(last_copy < first_cpu, "copies must be the largest-row ring misses");
  }
  printf("batch_miss: assign %ld layers · empty-ring layers %ld identical to baseline in every slot\n", n_layers, n_eq);

  // 4) Pred
  {
    Pred p;
    int e1[] = {3, 7, 9}, r1[] = {2, 1, 1};
    p.update(2, e1, r1, 3);
    int e2[] = {7, 9, 11}, r2[] = {1, 3, 1};
    p.update(2, e2, r2, 3);
    const auto* v = p.at(2);
    EXPECT(v && v->size() == 3, "pred size");
    for (const Ent& x : *v) {
      if (x.e == 7 || x.e == 9) EXPECT(x.streak == 2, "streak of %d = %d", x.e, x.streak);
      if (x.e == 11) EXPECT(x.streak == 1, "streak of 11");
    }
    EXPECT(p.at(0) && p.at(0)->empty() && !p.at(5) && !p.at(-1), "pred other layers");
    int out[4];
    int n = p.pick(2, 4, [](int) { return false; }, out);
    EXPECT(n == 3 && out[0] == 9 && out[1] == 7 && out[2] == 11, "pick order %d %d %d %d", n, out[0], out[1], out[2]);  // streak 2: 9 (row 3) → 7 (row 1) · streak 1: 11
    n = p.pick(2, 1, [](int e) { return e == 9; }, out);
    EXPECT(n == 1 && out[0] == 7, "pick skip/cap");
    EXPECT(p.pick(3, 4, [](int) { return false; }, out) == 0 && p.pick(2, 0, [](int) { return false; }, out) == 0, "pick empty");
  }

  // 5) synthetic batched decode: rows M=4 · layers L=6 · experts E=64 · k=6 · resident 0..15 (fixed) · each row reuses the previous token's experts with probability 0.5
  {
    const int M = 4, L = 6, E = 64, K = 6, S = 96, margin = kLayerAlloc + 1;
    StageTrack t;
    std::vector<long> content(S, -1);
    uint32_t next = 0;
    std::vector<std::vector<std::vector<int>>> prev(L, std::vector<std::vector<int>>(M));
    long staged = 0, cpu = 0, miss = 0, bad = 0;
    for (int step = 0; step < 400; ++step)
      for (int l = 0; l < L; ++l) {
        std::vector<int> cntv(E, 0);
        for (int m = 0; m < M; ++m) {
          std::set<int> pick;
          for (int e : prev[l][m]) if (rng() % 2 && (int)pick.size() < K) pick.insert(e);
          while ((int)pick.size() < K) pick.insert((int)(rng() % E));
          prev[l][m].assign(pick.begin(), pick.end());
          for (int e : pick) ++cntv[e];
        }
        std::vector<MissIn> mi;
        std::vector<int> si_of;
        for (int e = 16; e < E; ++e) if (cntv[e]) {
          const long key = (long)l * E + e;
          const int si = t.find((size_t)key, next, S, margin);
          if (si >= 0 && content[(size_t)si] != key) ++bad;
          mi.push_back({e, cntv[e], si < 0 ? -1 : 1});
          si_of.push_back(si);
        }
        std::vector<size_t> ord(mi.size());
        for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
        std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return mi[a].rows > mi[b].rows; });
        std::vector<MissIn> ms;
        for (size_t i : ord) ms.push_back(mi[i]);
        std::vector<uint8_t> w(ms.size() + 1);
        assign(ms.data(), (int)ms.size(), [&](int n) { return n >= 3 ? std::min(8, (int)(0.25f * n + 0.5f)) : 0; }, w.data());
        for (size_t i = 0; i < ms.size(); ++i) {
          ++miss;
          if (w[i] == kStaged) ++staged;
          if (w[i] == kCpu) ++cpu;
          if (w[i] == kDmaCopy) { const uint32_t pos = next++; content[pos % S] = (long)l * E + ms[i].e; t.put((size_t)l * E + ms[i].e, pos, false); }
        }
      }
    EXPECT(bad == 0, "stale ring reuse %ld", bad);
    EXPECT(staged > 0, "no ring reuse in a temporally local world");
    printf("batch_miss: synthetic batch (M=%d · reuse probability 0.5): misses %ld · ring reuse %ld (%.1f%%) · CPU %ld · stale reuse %ld\n", M, miss, staged,
           100.0 * staged / std::max(1L, miss), cpu, bad);
  }
  printf("test_batch_miss_cpu: %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
