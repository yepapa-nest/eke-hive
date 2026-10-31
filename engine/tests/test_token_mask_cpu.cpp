// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hive/token_mask.h on the CPU: the thinking-span state machine, the verify-row flags (row i = state after inputs[1..i]), and the host
//   masked sampler: never a listed token, and the distribution of the rest = the unmasked rule's renormalized (chi-square-free check:
//   frequencies within 4 standard errors) for greedy, top-k, top-p and plain temperature sampling. Negative control: the unmasked sampler
//   does draw listed tokens on the same rows.
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <map>
#include <random>
#include <vector>

#include "hive/token_mask.h"

using namespace hive;
static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, m); ++fails; } } while (0)

int main() {
  // 1. state machine: thinking span closed by </think> (9), reopened by <think> (8)
  tmask::State s; s.on = true; s.in_think = true; s.end = 9; s.start = 8;
  EXPECT(!s.active(), "inside the thinking span");
  s.feed(5); EXPECT(!s.active(), "still thinking");
  s.feed(9); EXPECT(s.active(), "answer after </think>");
  s.feed(8); EXPECT(!s.active(), "<think> reopens");
  tmask::State off; off.feed(9); EXPECT(!off.active(), "a request without the filter is never active");
  // 2. verify flags: inputs [tok, d0, d1, d2] with d1 = </think>
  tmask::State v; v.on = true; v.in_think = true; v.end = 9; v.start = 8;
  const int32_t in[4] = {3, 4, 9, 7};
  std::vector<float> f;
  tmask::verify_flags(v, in, 4, f);
  EXPECT(f.size() == 4 && f[0] == 0 && f[1] == 0 && std::isinf(f[2]) && std::isinf(f[3]), "verify flags follow the drafts");
  tmask::State a; a.on = true; a.in_think = false;
  f.clear(); tmask::verify_flags(a, in, 4, f);
  EXPECT(f.size() == 4 && f[0] && f[1] && f[2] && f[3], "non-thinking: every row");
  a.penalty = 5.f; f.clear(); tmask::verify_flags(a, in, 4, f);
  EXPECT(f.size() == 4 && f[0] == 5.f && f[3] == 5.f, "finite penalty carried per row");
  // 3. masked sampling
  const int V = 64;
  std::mt19937_64 rng(11);
  std::normal_distribution<float> nd(0.f, 2.f);
  std::vector<float> row(V);
  for (auto& x : row) x = nd(rng);
  // mask the three most likely tokens and a few others
  std::vector<int> order(V);
  for (int i = 0; i < V; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int x, int y) { return row[x] > row[y]; });
  std::vector<int32_t> masked = {order[0], order[1], order[2], 17, 33};
  std::vector<char> is_m(V, 0);
  for (int32_t t : masked) is_m[t] = 1;
  struct Cfg { float T, top_p; int top_k; const char* name; };
  for (float pen : {INFINITY, 3.f})
  for (const Cfg& c : {Cfg{0.f, 1.f, 0, "greedy"}, Cfg{0.6f, 0.95f, 20, "T0.6 top_p0.95 top_k20"}, Cfg{1.f, 1.f, 0, "T1"}, Cfg{0.8f, 0.9f, 0, "top_p0.9"}}) {
    // reference: the unmasked rule over the row with the masked logits at −inf, computed exactly
    std::vector<double> p(V, 0.0);
    {
      std::vector<std::pair<double, int>> e;
      double mx = -1e30;
      std::vector<double> sh(V);   // the shifted row (−inf for removed)
      for (int i = 0; i < V; ++i) sh[i] = is_m[i] ? (std::isinf(pen) ? -1e300 : (double)(row[i] - pen)) : (double)row[i];
      for (int i = 0; i < V; ++i) mx = std::max(mx, sh[i]);
      if (c.T <= 0.f) { for (int i = 0; i < V; ++i) if (sh[i] == mx) { p[i] = 1; break; } }
      else {
        double sum = 0;
        for (int i = 0; i < V; ++i) if (sh[i] > -1e299) { const double x = std::exp((sh[i] - mx) / c.T); e.push_back({x, i}); sum += x; }
        std::sort(e.begin(), e.end(), [](auto& x, auto& y) { return x.first > y.first; });
        int keep = (int)e.size();
        if (c.top_k > 0) keep = std::min(keep, c.top_k);
        if (c.top_p < 1.f) { double cum = 0; int k2 = 0; while (k2 < keep) { cum += e[k2].first / sum; ++k2; if (cum >= c.top_p) break; } keep = k2; }
        double tot = 0;
        for (int i = 0; i < keep; ++i) tot += e[i].first;
        for (int i = 0; i < keep; ++i) p[e[i].second] = e[i].first / tot;
      }
    }
    const int N = 200000;
    std::vector<int> cnt(V, 0), cnt_plain(V, 0);
    std::mt19937_64 r2(99);
    for (int n = 0; n < N; ++n) ++cnt[tmask::sample_masked(row, masked, pen, c.T, c.top_p, c.top_k, 0.f, r2)];
    for (int n = 0; n < N; ++n) { std::vector<float> l = row; ++cnt_plain[sampling::sample(l, c.T, c.top_p, c.top_k, 0.f, r2)]; }
    bool none = true, close = true;
    for (int i = 0; i < V; ++i) {
      if (std::isinf(pen) && is_m[i] && cnt[i]) none = false;
      const double se = std::sqrt(std::max(p[i] * (1 - p[i]), 1e-12) / N);
      if (std::fabs(cnt[i] / (double)N - p[i]) > 4 * se + 1e-9) close = false;
    }
    int plain_masked = 0;
    for (int t : masked) plain_masked += cnt_plain[t];
    EXPECT(none, "removed tokens drawn");
    EXPECT(close, "shifted distribution != the rule over the shifted row");
    EXPECT(plain_masked > 0, "negative control: the unmasked sampler should draw masked tokens on this row");
    int listed = 0;
    for (int t : masked) listed += cnt[t];
    printf("token_mask Δ %-4s %-24s: listed draws %d · frequencies within 4 SE of the shifted rule · unshifted sampler drew listed tokens %d/%d times\n",
           std::isinf(pen) ? "inf" : "3", c.name, listed, plain_masked, N);
  }
  if (fails) { fprintf(stderr, "test_token_mask_cpu: %d failure(s)\n", fails); return 1; }
  printf("test_token_mask_cpu: passed\n");
  return 0;
}
