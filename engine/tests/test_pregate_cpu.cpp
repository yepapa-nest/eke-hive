// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// P1 HIVE_DECODE_PREGATE host rules (hive/pregate.h) — no GPU.
//  1. parse_k: off values (unset, "", "0") = 0; "1" and non-numbers = default 8; 2..16 as is; above that 16
//  2. owned_spans: the spans of workers 0..T-1 cover phase-1 rows [0, I2) and phase-2 rows [0, D2) exactly once (no gaps or overlaps);
//     owner(e, idx) == that worker for every span; per-worker row sets equal those of the start_jobs item layout (a list rebuilt with the same formula)
//  3. plan: non-resident only; ordered by (row count desc, rank asc, id asc); cap; out-of-range ids absorbed; duplicates within a row counted once
//  4. Acc: recall and precision counting
//  Negative control (argv[1] == "neg"): an item list built with a deliberately wrong owner (no start rotation) must be caught diverging from the spans — check_layout below must fail in that case.
#include "hive/pregate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <random>
#include <set>
#include <vector>

using namespace hive;
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// Builds per-worker (phase, r0) sets with the same item layout as start_jobs (phase-1 blocks -> phase-2 blocks). own_fn = owner formula.
template <class Own>
static std::vector<std::set<std::pair<int, int>>> layout(int e, int T, int I2, int D2, int P1, int P2, Own own_fn) {
  std::vector<std::set<std::pair<int, int>>> by(T);
  const int nblk = (I2 + P1 - 1) / P1;
  for (int r = 0; r < I2; r += P1) by[own_fn(e, r / P1, T)].insert({1, r});
  for (int r = 0; r < D2; r += P2) by[own_fn(e, nblk + r / P2, T)].insert({2, r});
  return by;
}

static bool check_layout(int e, int T, int I2, int D2, int P1, int P2, bool neg) {
  auto want = layout(e, T, I2, D2, P1, P2, neg ? [](int, int idx, int T) { return idx % T; } : [](int e, int idx, int T) { return pg::owner(e, idx, T); });
  bool ok = true;
  std::vector<int> c1(I2, 0), c2(D2, 0);
  for (int t = 0; t < T; t++) {
    std::set<std::pair<int, int>> got;
    pg::owned_spans(e, t, T, I2, D2, P1, P2, [&](const pg::Span& s) {
      got.insert({s.phase, s.r0});
      for (int r = s.r0; r < s.r1; ++r) ++(s.phase == 1 ? c1 : c2)[(size_t)r];
      if (s.r1 <= s.r0 || s.r1 > (s.phase == 1 ? I2 : D2)) ok = false;
    });
    if (got != want[(size_t)t]) ok = false;
  }
  for (int x : c1) if (x != 1) ok = false;
  for (int x : c2) if (x != 1) ok = false;
  return ok;
}

int main(int argc, char** argv) {
  const bool neg = argc > 1 && !strcmp(argv[1], "neg");
  if (neg) {  // negative control: an owner formula without rotation must diverge from the spans for e != 0
    int caught = 0;
    for (int e = 1; e < 40; ++e) caught += !check_layout(e, 16, 1152, 2560, 32, 32, true);
    printf("pregate CPU negative control: %d/39 layouts with a wrong owner rule detected\n", caught);
    return caught > 0 ? 1 : 0;  // 1 = caught (the caller expects failure)
  }
  // 1
  EXPECT(pg::parse_k(nullptr) == 0 && pg::parse_k("") == 0 && pg::parse_k("0") == 0, "off values");
  EXPECT(pg::parse_k("1") == 8 && pg::parse_k("on") == 8 && pg::parse_k("-3") == 8 && pg::parse_k("10x") == 8, "on with the default K");
  EXPECT(pg::parse_k("2") == 2 && pg::parse_k("6") == 6 && pg::parse_k("16") == 16 && pg::parse_k("40") == 16, "explicit K (capped at 16)");
  // 2 — production dimensions (I2 1152, D2 2560, P1/P2 32), production T (16) and various boundaries
  int n_layout = 0;
  for (int T : {1, 2, 3, 4, 7, 16, 17, 32})
    for (int e : {0, 1, 5, 127, 383})
      for (auto [I2, D2, P1, P2] : {std::tuple{1152, 2560, 32, 32}, std::tuple{1152, 2560, 64, 17}, std::tuple{96, 70, 32, 9}, std::tuple{32, 5, 32, 64}}) {
        EXPECT(check_layout(e, T, I2, D2, P1, P2, false), "owned_spans != start_jobs layout (e %d T %d I2 %d D2 %d P1 %d P2 %d)", e, T, I2, D2, P1, P2);
        ++n_layout;
      }
  // At production dimensions the per-worker shares are even (one job: 116 items / 16 = 7–8) — combining several experts stays balanced thanks to the rotation
  {
    std::vector<int> load(16, 0);
    for (int e = 0; e < 32; ++e) for (int t = 0; t < 16; ++t) pg::owned_spans(e, t, 16, 1152, 2560, 32, 32, [&](const pg::Span&) { ++load[(size_t)t]; });
    int mn = 1 << 30, mx = 0;
    for (int x : load) { mn = std::min(mn, x); mx = std::max(mx, x); }
    EXPECT(mx - mn <= 4, "owner rotation leaves an imbalance %d..%d over 32 experts", mn, mx);
    printf("pregate CPU: per-worker items over 32 experts (T 16): %d..%d\n", mn, mx);
  }
  // 3
  {
    const int M = 3, K = 4, E = 10;
    const int32_t ids[M * K] = {5, 2, 9, 1,   2, 5, 7, 7,   3, 2, 42, -1};  // 7 duplicated in row 1; 42 and -1 in row 2 out of range
    int out[16];
    const int n = pg::plan(ids, M, K, E, [](int e) { return e == 9; }, out, 16);
    // Row counts: 2 -> 3 rows (0,1,2); 5 -> 2 rows (rank 0); 1 (row 0, rank 3); 7 (row 1, rank 2); 3 (row 2, rank 0); 9 is resident -> dropped
    const int want[] = {2, 5, 3, 7, 1};
    EXPECT(n == 5, "plan count %d", n);
    for (int i = 0; i < std::min(n, 5); ++i) EXPECT(out[i] == want[i], "plan order [%d] = %d want %d", i, out[i], want[i]);
    EXPECT(pg::plan(ids, M, K, E, [](int) { return false; }, out, 2) == 2 && out[0] == 2 && out[1] == 5, "plan cap");
    EXPECT(pg::plan(ids, M, K, E, [](int) { return true; }, out, 16) == 0, "all resident");
  }
  // 4
  {
    pg::Acc a;
    const int pf[] = {1, 2, 3}, cpu[] = {2, 3, 4, 5};
    a.add_layer(pf, 3, cpu, 4);
    a.add_layer(pf, 0, cpu, 0);
    EXPECT(a.layers == 2 && a.pred == 3 && a.actual == 4 && a.covered == 2, "acc counts");
    EXPECT(a.recall() == 0.5 && std::abs(a.precision() - 2.0 / 3.0) < 1e-12, "acc ratios");
  }
  // Random: plan results are always non-resident, duplicate-free and within the cap
  std::mt19937 rng(7);
  for (int it = 0; it < 2000; ++it) {
    const int M = 1 + rng() % 8, K = 1 + rng() % 16, E = 384;
    std::vector<int32_t> ids((size_t)M * K);
    for (auto& x : ids) x = (int)(rng() % (E + 8)) - 4;
    std::vector<char> res(E);
    for (auto& r : res) r = rng() % 3 == 0;
    int out[pg::kMaxPf];
    const int cap = 1 + rng() % pg::kMaxPf;
    const int n = pg::plan(ids.data(), M, K, E, [&](int e) { return res[(size_t)e] != 0; }, out, cap);
    std::set<int> seen;
    bool ok = n <= cap;
    for (int i = 0; i < n; ++i) { ok &= out[i] >= 0 && out[i] < E && !res[(size_t)out[i]] && seen.insert(out[i]).second; }
    EXPECT(ok, "random plan invariant broken at %d", it);
    if (fails > 5) break;
  }
  printf("pregate CPU: parse_k · %d owned layouts == start_jobs items (exact cover) · plan order/cap/absorb · acc — %s\n", n_layout, fails ? "FAIL" : "ok");
  return fails ? 1 : 0;
}
