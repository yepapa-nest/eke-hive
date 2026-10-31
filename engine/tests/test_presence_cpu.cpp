// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hive/presence.h on the CPU: sampling on the GPU-candidate snapshot with the presence penalty == the rule over the fully penalized row
//   (per-token counts within 5 σ of their Poisson spread, 100K draws) for greedy, top-k / top-p, min_p, plain temperature and the order-free path, with
//   positive and negative penalties, seen tokens inside and outside the candidates, with and without the full row (greedy); the full row
//   is restored bit for bit. Negative control: ignoring the penalty moves the frequencies of the seen tokens.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "hive/presence.h"

using namespace hive;
static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, m); ++fails; } } while (0)

// the GPU's candidate snapshot of a row: top NC by value, row max, Σexp((l − mx)/T) over the row
static sampling::Cands cands_of(const std::vector<float>& row, int NC, float T) {
  sampling::Cands c;
  std::vector<int> o(row.size());
  for (size_t i = 0; i < o.size(); ++i) o[i] = (int)i;
  std::partial_sort(o.begin(), o.begin() + NC, o.end(), [&](int a, int b) { return row[a] > row[b]; });
  c.mx = row[o[0]];
  double s = 0;
  const float it = T > 0 ? 1.f / T : 1.f;
  for (float v : row) s += std::exp(((double)v - c.mx) * it);
  c.sum = (float)s;
  std::sort(o.begin(), o.begin() + NC);   // the device's candidate list is in ascending token order (sample_cands merges on it)
  for (int i = 0; i < NC; ++i) { c.idx.push_back(o[i]); c.val.push_back(row[o[i]]); }
  c.argmax = (int)(std::max_element(row.begin(), row.end()) - row.begin());
  c.valid = true;
  return c;
}

int main() {
  const int V = 3000, NC = 256;
  std::mt19937_64 g(5);
  std::normal_distribution<float> nd(0.f, 2.5f);
  std::vector<float> row(V);
  for (auto& x : row) x = nd(g);
  std::vector<int> ord(V);
  for (int i = 0; i < V; ++i) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](int a, int b) { return row[a] > row[b]; });
  struct Cfg { float T, top_p, min_p; int top_k; const char* name; };
  const Cfg cfgs[] = {{0.f, 1.f, 0.f, 0, "greedy"}, {0.6f, 0.95f, 0.f, 20, "T0.6 p0.95 k20"}, {0.6f, 0.8f, 0.f, 20, "T0.6 p0.8 k20"},
                      {0.6f, 0.95f, 0.1f, 20, "T0.6 k20 min_p0.1"}, {1.f, 1.f, 0.f, 0, "T1 order-free"}, {0.8f, 0.9f, 0.f, 0, "T0.8 p0.9"}};
  for (float pp : {1.5f, -1.0f})
    for (int variant = 0; variant < 2; ++variant) {
      presence::State P;
      P.pp = pp;
      // seen: the top two, a mid candidate and two tokens outside the candidates (variant 1: also a token just outside, near the edge)
      for (int t : {ord[0], ord[1], ord[40], ord[NC + 5], ord[V - 3]}) P.feed(t, V);
      if (variant == 1) P.feed(ord[NC], V);
      for (const Cfg& c : cfgs) {
        std::vector<float> pen = row;
        for (int32_t t : P.seen) pen[t] -= pp;
        const sampling::Cands cd = cands_of(row, NC, c.T);
        const int N = c.T <= 0.f ? 50 : 100000;
        std::vector<int> got(V, 0), ref(V, 0), raw(V, 0);
        std::mt19937_64 r1(9), r2(9), r3(9);
        std::vector<float> full = row;
        for (int n = 0; n < N; ++n) ++got[presence::sample(cd, full, P, c.T, c.top_p, c.top_k, c.min_p, r1)];
        EXPECT(full == row, "full row restored bit for bit");
        for (int n = 0; n < N; ++n) { std::vector<float> l = pen; ++ref[sampling::sample(l, c.T, c.top_p, c.top_k, c.min_p, r2)]; }
        for (int n = 0; n < N; ++n) { std::vector<float> l = row; ++raw[sampling::sample_cands(cd, l, c.T, c.top_p, c.top_k, c.min_p, r3)]; }
        bool close = true, moved = false;
        for (int i = 0; i < V; ++i) {
          // two independent multinomial counts of one token: |a − b| within 5 σ of their Poisson spread (+3 for tiny counts)
          if (std::abs(got[i] - ref[i]) > 5 * std::sqrt((double)std::max(1, got[i] + ref[i])) + 3) close = false;
          if (std::abs(ref[i] - raw[i]) > 6 * std::sqrt((double)std::max(1, ref[i] + raw[i])) + 3) moved = true;
        }
        if (c.T <= 0.f) {   // greedy: also without the full row (the candidates decide)
          std::vector<float> none;
          std::mt19937_64 r4(1);
          EXPECT(presence::sample(cd, none, P, 0.f, 1.f, 0, 0.f, r4) == (int32_t)(std::max_element(pen.begin(), pen.end()) - pen.begin()),
                 "greedy without the full row");
        }
        EXPECT(close, "penalized candidates != the penalized row");
        {   // negative control: whenever the penalized rule differs from the unpenalized one, ignoring the penalty must show it
          std::vector<int> un(V, 0);
          std::mt19937_64 r5(9);
          for (int n = 0; n < N; ++n) { std::vector<float> l = row; ++un[sampling::sample(l, c.T, c.top_p, c.top_k, c.min_p, r5)]; }
          bool differs = false;
          for (int i = 0; i < V; ++i) differs |= std::abs(un[i] - ref[i]) > 6 * std::sqrt(std::max(1, ref[i])) + 2;
          EXPECT(!differs || moved, "negative control: the unpenalized candidates should not match the penalized row");
        }
        printf("presence pp %+.1f seen %zu %-18s: == the penalized row (5 σ) %s · unpenalized differs %s\n", pp, P.seen.size(), c.name, close ? "yes" : "NO",
               moved ? "yes" : "no");

      }
    }
  if (fails) { fprintf(stderr, "test_presence_cpu: %d failure(s)\n", fails); return 1; }
  printf("test_presence_cpu: passed\n");
  return 0;
}
