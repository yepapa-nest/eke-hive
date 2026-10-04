// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hive/sampler.h CPU test — checks the distributions (lossless). Built and run by tools/test_sampler_cpu.py (including negative controls).
//   (1) sample(): TV distance between the theoretical distribution computed by an independent implementation (full sort) and the sampled distribution —
//       combinations of temperature, top_p, top_k and min_p; V = 3000 so both the 1024-candidate partial-sort path and the "candidate mass < top_p →
//       full sort" fallback path are exercised.
//   (2) sample_cands(): GPU candidates (top NC values and indices + row max and Σexp with temperature) built on the CPU give the same theoretical
//       distribution — including the fallbacks (insufficient mass, top_k > NC).
//   (3) Speculative (MTP) acceptance rule — the same rule as the single and batched paths of hived.cpp decode_step (accept if row i's sample t equals the
//       draft d_{i+1}; otherwise t is the correction token; if all are accepted, row k's sample is the bonus): the joint distribution of the emitted token
//       sequence == sampling the target model one token at a time (whatever the draft — correct, wrong or context-dependent drafts). Negative control:
//       the rule "accept the draft without sampling if the target probability is above a threshold" is caught by the same TV test (test sensitivity).
//   (4) Explains sample collapse: if the top token's probability after temperature is >= top_p, top_p truncation keeps only the top token
//       (deterministic) — the mechanism behind 64/64 identical answers in an evaluation sampled at T 0.7, top_p 0.9. With the same logits the serving
//       settings (T 1.0, top_p 0.95) give varied answers.
//   (5) request_seed: 0 = a different seed per call (even for back-to-back calls), non-zero seed = that value (reproducible).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "hive/sampler.h"

using namespace hive::sampling;
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

// Independent implementation (straight from the definition): softmax(l/T) → full descending sort → top_k → min_p (p >= p_max·min_p) → top_p (up to where the cumulative sum first reaches top_p) → renormalize
static std::vector<double> theory(const std::vector<float>& l, float T, float top_p, int top_k, float min_p) {
  const int V = (int)l.size();
  std::vector<double> out(V, 0.0);
  if (T <= 0.f) { out[std::max_element(l.begin(), l.end()) - l.begin()] = 1.0; return out; }
  double mx = -1e300;
  for (float v : l) mx = std::max(mx, (double)v);
  std::vector<std::pair<double, int>> p(V);
  double s = 0;
  for (int i = 0; i < V; ++i) { p[i] = {std::exp(((double)l[i] - mx) / T), i}; s += p[i].first; }
  for (auto& x : p) x.first /= s;
  std::stable_sort(p.begin(), p.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  int keep = V;
  if (top_k > 0) keep = std::min(keep, top_k);
  if (min_p > 0.f) { int k2 = 0; while (k2 < keep && p[k2].first >= p[0].first * min_p) ++k2; keep = std::max(1, k2); }
  if (top_p > 0.f && top_p < 1.f) { double c = 0; int k2 = 0; while (k2 < keep) { c += p[k2].first; ++k2; if (c >= top_p) break; } keep = std::max(1, k2); }
  double t = 0;
  for (int i = 0; i < keep; ++i) t += p[i].first;
  for (int i = 0; i < keep; ++i) out[p[i].second] = p[i].first / t;
  return out;
}
// Rank-bucket TV: split the tokens by theoretical probability rank into B buckets (within the support, equal rank counts) and compare bucket mass — noise
//   relative to sample count stays small even for flat distributions with thousands of supported tokens, and it is sensitive to the "mass per rank" that
//   temperature and truncation change. Samples outside the support are counted separately (must be 0).
static double tv_binned(const std::vector<double>& a, const std::map<int, long>& cnt, long n, long* outside) {
  std::vector<int> ord;
  for (int i = 0; i < (int)a.size(); ++i) if (a[i] > 0) ord.push_back(i);
  std::stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return a[x] > a[y]; });
  const int K = (int)ord.size(), B = std::min(K, 40);
  std::vector<int> bin(a.size(), -1);
  for (int r = 0; r < K; ++r) bin[ord[r]] = (int)((long)r * B / K);
  std::vector<double> pa(B, 0.0), pe(B, 0.0);
  for (int i = 0; i < (int)a.size(); ++i) if (bin[i] >= 0) pa[bin[i]] += a[i];
  *outside = 0;
  for (auto& [k, c] : cnt) { if (bin[k] >= 0) pe[bin[k]] += (double)c / n; else *outside += c; }
  double d = 0;
  for (int b = 0; b < B; ++b) d += std::fabs(pa[b] - pe[b]);
  return d / 2 + (double)*outside / n / 2;
}
// GPU candidates on the CPU: top NC (ascending index — runtime convention), row max, Σexp((l − max)·inv_temp)
static Cands make_cands(const std::vector<float>& l, int NC, float T) {
  Cands c;
  const int V = (int)l.size();
  std::vector<int> idx(V);
  for (int i = 0; i < V; ++i) idx[i] = i;
  std::partial_sort(idx.begin(), idx.begin() + std::min(NC, V), idx.end(), [&l](int a, int b) { return l[a] > l[b]; });
  idx.resize(std::min(NC, V));
  std::sort(std::begin(idx), std::end(idx));
  c.idx.assign(NC, -1); c.val.assign(NC, -INFINITY);
  for (size_t i = 0; i < idx.size(); ++i) { c.idx[i] = idx[i]; c.val[i] = l[idx[i]]; }
  c.mx = *std::max_element(l.begin(), l.end());
  const float it = T > 0.f ? 1.f / T : 1.f;
  double s = 0;
  for (float v : l) s += std::exp((double)(v - c.mx) * it);
  c.sum = (float)s;
  c.argmax = (int)(std::max_element(l.begin(), l.end()) - l.begin());
  c.valid = true;
  return c;
}
static std::vector<float> logits_peaked(int V, uint64_t seed, float spread) {
  std::mt19937_64 g(seed);
  std::normal_distribution<float> N(0.f, spread);
  std::vector<float> l(V);
  for (auto& x : l) x = N(g);
  return l;
}

static void test_sample_paths() {
  struct Case { float T, top_p; int top_k; float min_p; float spread; const char* what; };
  const Case cases[] = {
      {1.0f, 0.95f, 0, 0.f, 3.0f, "service T1.0 top_p 0.95 (1024-candidate path)"},
      {0.7f, 0.90f, 0, 0.f, 3.0f, "gap_eval T0.7 top_p 0.9"},
      {1.0f, 0.95f, 0, 0.f, 1.2f, "flat logits — candidate mass < top_p → full-sort fallback"},
      {1.3f, 1.0f, 40, 0.f, 2.0f, "top_k 40"},
      {0.8f, 0.99f, 0, 0.05f, 2.5f, "min_p 0.05"},
      {1.0f, 1.0f, 0, 0.f, 1.2f, "no truncation, wide"},
  };
  const int V = 3000;
  const long N = 80000;
  for (const auto& cs : cases) {
    const auto l = logits_peaked(V, 1234 + (uint64_t)(cs.T * 100) + cs.top_k, cs.spread);
    const auto th = theory(l, cs.T, cs.top_p, cs.top_k, cs.min_p);
    std::mt19937_64 rng(7);
    std::map<int, long> a, b;
    const Cands c = make_cands(l, 1024, cs.T);
    if (strstr(cs.what, "fallback")) {  // in this case the mass of the 1024 candidates must fall short of top_p (the fallback path is really taken)
      double m = 0;
      for (int i = 0; i < 1024; ++i) if (c.idx[i] >= 0) m += std::exp((double)(c.val[i] - c.mx) / cs.T) / c.sum;
      EXPECT(m < cs.top_p, "fallback case: candidate mass %.3f >= top_p — the full-sort path is not exercised", m);
    }
    for (long i = 0; i < N; ++i) {
      std::vector<float> lc = l;
      ++a[sample(lc, cs.T, cs.top_p, cs.top_k, cs.min_p, rng)];
      ++b[sample_cands(c, lc, cs.T, cs.top_p, cs.top_k, cs.min_p, rng)];
    }
    long na = 0, nb = 0;
    for (auto& [k, v] : a) na += v;
    for (auto& [k, v] : b) nb += v;
    long oa = 0, ob = 0, orf = 0, obad = 0;
    const double da = tv_binned(th, a, na, &oa), db = tv_binned(th, b, nb, &ob);
    // Noise floor: the same metric for samples drawn directly from the theoretical distribution
    std::discrete_distribution<int> ref(th.begin(), th.end());
    std::mt19937_64 rr(99);
    std::map<int, long> rc;
    for (long i = 0; i < na; ++i) ++rc[ref(rr)];
    const double floor_tv = tv_binned(th, rc, na, &orf);
    // Negative control: a distribution with a 30 % wrong temperature (same sample count) — evidence that the test sees a difference of that size
    const auto th_bad = theory(l, cs.T * 1.3f, cs.top_p, cs.top_k, cs.min_p);
    std::discrete_distribution<int> bad(th_bad.begin(), th_bad.end());
    std::map<int, long> bc;
    for (long i = 0; i < na; ++i) ++bc[bad(rr)];
    const double dbad = tv_binned(th, bc, na, &obad);
    const double tol = 2.0 * floor_tv + 0.004;
    int support = 0;
    for (double x : th) support += x > 0;
    EXPECT(oa == 0 && ob == 0, "%s: %ld / %ld samples outside the truncated support", cs.what, oa, ob);
    EXPECT(da < tol && db < tol, "%s: TV(sample) %.4f · TV(sample_cands) %.4f > %.4f (noise floor %.4f, support %d)", cs.what, da, db, tol, floor_tv, support);
    EXPECT(dbad > tol, "%s: negative control (temperature x1.3) TV %.4f not above tol %.4f — test too weak", cs.what, dbad, tol);
    printf("  %-58s support %4d · binned TV sample %.4f · cands %.4f · noise floor %.4f · wrong-T control %.4f\n", cs.what, support, da, db, floor_tv, dbad);
  }
  // Temperature 0: argmax (both paths)
  {
    auto l = logits_peaked(V, 99, 2.f);
    std::mt19937_64 rng(1);
    const Cands c = make_cands(l, 1024, 0.f);
    const int am = (int)(std::max_element(l.begin(), l.end()) - l.begin());
    EXPECT(sample(l, 0.f, 0.95f, 0, 0.f, rng) == am && sample_cands(c, l, 0.f, 0.95f, 0, 0.f, rng) == am, "temperature 0 must be argmax");
  }
}

// (3) Speculative acceptance rule — target model: a small-vocabulary (V=7) distribution determined by the context (preceding tokens). Draft model: a deterministic function (context-dependent, sometimes right).
static std::vector<float> target_logits(const std::vector<int>& ctx) {
  uint64_t h = 1469598103934665603ull;
  for (int t : ctx) { h ^= (uint64_t)t + 0x9E37; h *= 1099511628211ull; }
  std::mt19937_64 g(h);
  std::normal_distribution<float> N(0.f, 1.6f);
  std::vector<float> l(7);
  for (auto& x : l) x = N(g);
  return l;
}
static int draft_of(const std::vector<int>& ctx, int mode) {  // 0 = target argmax (a good draft), 1 = a fixed token that is usually wrong, 2 = context hash
  if (mode == 0) { auto l = target_logits(ctx); return (int)(std::max_element(l.begin(), l.end()) - l.begin()); }
  if (mode == 1) return 6;
  return (int)((ctx.empty() ? 3 : ctx.back() * 5 + (int)ctx.size()) % 7);
}
// Same rule as the acceptance loop in hived.cpp (single: for i<k { t = sample_row(i); if (t != drafts[i]) { nxt = t; break; } emit(t) }; if all are accepted, nxt = sample_row(k)).
//   wrong = negative control (no sampling: "accept the draft if the target probability >= 0.25").
static std::vector<int> spec_generate(int L, int k, int mode, float T, float top_p, std::mt19937_64& rng, bool wrong) {
  std::vector<int> out;
  int pending = -1;
  auto row = [&](const std::vector<int>& ctx) { auto l = target_logits(ctx); return l; };
  // First token (after prefill, normally sampled)
  { auto l = row(out); out.push_back(sample(l, T, top_p, 0, 0.f, rng)); }
  while ((int)out.size() < L) {
    if (pending >= 0) { out.push_back(pending); pending = -1; if ((int)out.size() >= L) break; }
    // Drafts d1..dk (the draft model chains its own predictions)
    std::vector<int> ctx = out, drafts;
    for (int i = 0; i < k; ++i) { const int d = draft_of(ctx, mode); drafts.push_back(d); ctx.push_back(d); }
    int nxt = -1, n_keep = 1;
    bool cont = true;
    std::vector<int> pre = out;
    for (int i = 0; i < k; ++i) {
      auto l = row(pre);  // row i = logits after seeing [last token, d1..di]
      int t;
      if (wrong) {
        const auto th = theory(l, T, top_p, 0, 0.f);
        t = th[drafts[i]] >= 0.25 ? drafts[i] : sample(l, T, top_p, 0, 0.f, rng);
      } else {
        t = sample(l, T, top_p, 0, 0.f, rng);
      }
      if (t != drafts[i]) { nxt = t; break; }
      ++n_keep;
      out.push_back(t); pre.push_back(t);
      cont = (int)out.size() < L;
      if (!cont) break;
    }
    if (nxt < 0 && cont && n_keep == k + 1) { auto l = row(pre); nxt = sample(l, T, top_p, 0, 0.f, rng); }
    pending = cont ? nxt : -1;
  }
  out.resize(L);
  return out;
}
static std::vector<int> ar_generate(int L, float T, float top_p, std::mt19937_64& rng) {
  std::vector<int> out;
  while ((int)out.size() < L) { auto l = target_logits(out); out.push_back(sample(l, T, top_p, 0, 0.f, rng)); }
  return out;
}
static double seq_tv(const std::map<std::vector<int>, long>& a, const std::map<std::vector<int>, long>& b, long n) {
  std::set<std::vector<int>> keys;
  for (auto& [k, v] : a) keys.insert(k);
  for (auto& [k, v] : b) keys.insert(k);
  double d = 0;
  for (auto& k : keys) {
    const auto ia = a.find(k), ib = b.find(k);
    d += std::fabs((ia == a.end() ? 0.0 : (double)ia->second) - (ib == b.end() ? 0.0 : (double)ib->second)) / n;
  }
  return d / 2;
}
static void test_speculative() {
  const int L = 4;
  const long N = 400000;
  struct Case { int k, mode; float T, top_p; };
  const Case cases[] = {{3, 0, 1.0f, 0.95f}, {3, 1, 1.0f, 0.95f}, {2, 2, 0.7f, 0.9f}, {5, 0, 0.7f, 1.0f}, {1, 2, 1.0f, 1.0f}};
  for (const auto& cs : cases) {
    std::mt19937_64 r1(11), r2(22), r3(33), r4(44);
    std::map<std::vector<int>, long> spec, ar, ar2, bad;
    for (long i = 0; i < N; ++i) {
      ++spec[spec_generate(L, cs.k, cs.mode, cs.T, cs.top_p, r1, false)];
      ++ar[ar_generate(L, cs.T, cs.top_p, r2)];
      ++ar2[ar_generate(L, cs.T, cs.top_p, r3)];
      if (i < N / 4) ++bad[spec_generate(L, cs.k, cs.mode, cs.T, cs.top_p, r4, true)];
    }
    const double noise = seq_tv(ar, ar2, N), d = seq_tv(spec, ar, N);
    std::map<std::vector<int>, long> ar_q;  // the negative control uses N/4 samples — compared against an AR sample of the same size
    { long c = 0; std::mt19937_64 r5(55); for (; c < N / 4; ++c) ++ar_q[ar_generate(L, cs.T, cs.top_p, r5)]; }
    const double dbad = seq_tv(bad, ar_q, N / 4);
    std::map<std::vector<int>, long> ar_q2;
    { std::mt19937_64 r6(66); for (long c = 0; c < N / 4; ++c) ++ar_q2[ar_generate(L, cs.T, cs.top_p, r6)]; }
    const double noise_q = seq_tv(ar_q2, ar_q, N / 4);
    // Verdict: speculative ≈ AR (within 1.5× the noise between two same-size AR samples + 0.003); the negative control must be >= 3× the noise (the test can see the difference)
    EXPECT(d < 1.5 * noise + 0.003, "spec k %d draft mode %d T %.1f top_p %.2f: TV(spec, AR) %.4f vs noise %.4f", cs.k, cs.mode, cs.T, cs.top_p, d, noise);
    EXPECT(cs.mode == 1 || dbad > 3 * noise_q, "negative control not detected (k %d mode %d): TV %.4f vs noise %.4f", cs.k, cs.mode, dbad, noise_q);
    printf("  spec k %d draft %-14s T %.1f top_p %.2f: TV(spec,AR) %.4f · AR noise %.4f · wrong rule TV %.4f (noise %.4f)\n", cs.k,
           cs.mode == 0 ? "target-argmax" : cs.mode == 1 ? "fixed token" : "context hash", cs.T, cs.top_p, d, noise, dbad, noise_q);
  }
}

static void test_peaked_truncation() {
  // Top probability 0.80 (T=1), the other 9 share 0.2 (vocabulary 10 — answers like "one of 1–10")
  std::vector<float> l(10, 0.f);
  const double p1 = 0.80, rest = (1 - p1) / 9;
  for (int i = 0; i < 10; ++i) l[i] = (float)std::log(i == 6 ? p1 : rest);
  const auto a = theory(l, 0.7f, 0.9f, 0, 0.f), b = theory(l, 1.0f, 0.95f, 0, 0.f);
  int sa = 0, sb = 0;
  for (double x : a) sa += x > 0;
  for (double x : b) sb += x > 0;
  EXPECT(sa == 1 && a[6] == 1.0, "T0.7/top_p0.9 on a 0.80-peaked distribution should keep only the top token (kept %d)", sa);
  EXPECT(sb > 1 && b[6] < 0.9, "T1.0/top_p0.95 should keep several tokens (kept %d, p_top %.3f)", sb, b[6]);
  std::mt19937_64 rng(5);
  std::set<int> seen;
  for (int i = 0; i < 2000; ++i) { auto x = l; seen.insert(sample(x, 0.7f, 0.9f, 0, 0.f, rng)); }
  EXPECT(seen.size() == 1, "T0.7/top_p0.9 sampled %zu distinct tokens", seen.size());
  // Threshold: the minimum p1 at which only the top token survives at T=0.7, top_p 0.9 (other 9 uniform) — reported in the table
  double lo = 0.3, hi = 1.0;
  for (int it = 0; it < 50; ++it) {
    const double m = (lo + hi) / 2, r = (1 - m) / 9;
    std::vector<float> q(10);
    for (int i = 0; i < 10; ++i) q[i] = (float)std::log(i == 6 ? m : r);
    const auto t = theory(q, 0.7f, 0.9f, 0, 0.f);
    int s = 0;
    for (double x : t) s += x > 0;
    (s == 1 ? hi : lo) = m;
  }
  printf("  peaked: T0.7/top_p0.9 keeps only the top token once p_top(T=1) >= %.3f (9 equal rivals) — same logits at T1.0/top_p0.95 keep %d tokens\n", hi, sb);
}

static void test_seed() {
  std::set<uint64_t> s;
  for (int i = 0; i < 1000; ++i) s.insert(request_seed(0));
  EXPECT(s.size() == 1000, "request_seed(0) repeated within the same instant: %zu distinct of 1000", s.size());
  EXPECT(request_seed(42) == 42 && request_seed(-1) == (uint64_t)-1, "explicit seeds must be used as given");
}

int main(int argc, char** argv) {
  const bool quick = argc > 1 && !strcmp(argv[1], "--quick");  // for mutation checks (skips the speculative test)
  test_sample_paths();
  if (!quick) test_speculative();
  test_peaked_truncation();
  test_seed();
  if (fails) { fprintf(stderr, "sampler CPU: %d failures\n", fails); return 1; }
  printf("sampler CPU: sample/sample_cands match the definition · speculative acceptance preserves the target distribution · top_p truncation explains "
         "identical samples · per-request seeds distinct\n");
  return 0;
}
