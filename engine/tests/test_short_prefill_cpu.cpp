// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// R2 short-prefill (TTFT) decision formulas — CPU test (hive/short_prefill.h, called directly by runtime.cpp) + runtime.cpp text contracts.
//   ① HIVE_PREFILL_SHORT_ADAPT value parsing (switch convention: unset, "" or "0" = off) ② truth table of the short tail path
//   ③ short streaming kind (off = always the default kind)
//   ④ copy-ahead rescaling: off · rows_now 0 · same size = the same candidate set as the default decision (20K random cases) ·
//      smaller → the threshold rises proportionally · larger → the top depth of the default candidates stay as they are (ranking
//      unchanged) ⑤ whether runtime.cpp uses these formulas in place (text contract — replacing the calls fails here).
//   ⑥ Q3: HIVE_PREFILL_SMALL parsing (floor = max(N, max_batch+1, 9)) · small_forward truth table · multi_short truth table + text
//      contracts (encoder layers keep the default call).
//   Run: g++ -std=c++20 -O1 -Iengine/include engine/tests/test_short_prefill_cpu.cpp -o /tmp/t && (cd <repo> && /tmp/t)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "hive/short_prefill.h"

using namespace hive;

static int fails = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// Copy-ahead candidates (same order as the default prefetch_layer_experts: threshold pass → partial sort by (-rows, e) → top depth)
static std::vector<int> pick(const std::vector<int>& lr, int64_t num, int64_t den, int min_rows, int depth) {
  std::vector<std::pair<int, int>> cand;
  for (int e = 0; e < (int)lr.size(); ++e) if (sp::prefetch_pass(lr[e], num, den, min_rows)) cand.push_back({-lr[e], e});
  const size_t n = std::min<size_t>((size_t)depth, cand.size());
  std::partial_sort(cand.begin(), cand.begin() + n, cand.end());
  std::vector<int> out;
  for (size_t i = 0; i < n; ++i) out.push_back(cand[i].second);
  return out;
}
static std::vector<int> pick_old(const std::vector<int>& lr, int min_rows, int depth) {  // the default formula as is (lr ≥ max(1, min_rows))
  std::vector<std::pair<int, int>> cand;
  for (int e = 0; e < (int)lr.size(); ++e) if (lr[e] >= std::max(1, min_rows)) cand.push_back({-lr[e], e});
  const size_t n = std::min<size_t>((size_t)depth, cand.size());
  std::partial_sort(cand.begin(), cand.begin() + n, cand.end());
  std::vector<int> out;
  for (size_t i = 0; i < n; ++i) out.push_back(cand[i].second);
  return out;
}
// Rows per expert of a previous chunk with `rows` rows · top-k routing (skewed distribution — the first experts are popular)
static std::vector<int> route_rows(std::mt19937& rng, int E, int rows, int k) {
  std::vector<int> lr(E, 0);
  std::vector<double> w(E);
  for (int e = 0; e < E; ++e) w[e] = 1.0 / (1.0 + e * 0.05) * (0.5 + (rng() % 1000) / 1000.0);
  std::discrete_distribution<int> d(w.begin(), w.end());
  for (int m = 0; m < rows; ++m) {
    std::vector<int> used;
    while ((int)used.size() < k) { const int e = d(rng); if (std::find(used.begin(), used.end(), e) == used.end()) used.push_back(e); }
    for (int e : used) ++lr[e];
  }
  return lr;
}

int main() {
  // ① value parsing
  struct { const char* v; int want; } tbl[] = {{nullptr, 0}, {"", 0}, {"0", 0}, {"1", 4096}, {"2", 2}, {"1024", 1024}, {"4096", 4096}, {"8192", 8192},
                                               {"x", 4096}, {"-5", 4096}, {"3.5", 4096}, {"99999999999", 4096}};
  for (auto& t : tbl) { const int got = sp::parse_short_adapt(t.v); EXPECT(got == t.want, "parse_short_adapt(%s) = %d, want %d", t.v ? t.v : "null", got, t.want); }
  // ② short tail path: only when all six conditions hold
  int n_true = 0;
  for (int mask = 0; mask < 64; ++mask) {
    const bool on = mask & 1, inp = mask & 2, ver = mask & 4, bat = mask & 8, cpu = mask & 16, small = mask & 32;
    const bool got = sp::short_moe(on, inp, ver, bat, cpu, small ? 128 : 1024, 1024);
    const bool want = on && inp && !ver && !bat && cpu && small;
    EXPECT(got == want, "short_moe mask %d = %d", mask, got);
    n_true += got;
  }
  EXPECT(n_true == 1, "short_moe true in %d cases", n_true);
  EXPECT(!sp::short_moe(true, true, false, false, true, 1023 + 1, 1024) && sp::short_moe(true, true, false, false, true, 1023, 1024), "short_moe threshold edge");
  // ③ short streaming kind
  for (int rows : {1, 128, 1024, 4096, 4097, 16384, 49152}) EXPECT(!sp::short_stream_kind(0, rows), "adapt off must keep stream kind (rows %d)", rows);
  EXPECT(sp::short_stream_kind(4096, 128) && sp::short_stream_kind(4096, 4096) && !sp::short_stream_kind(4096, 4097), "short_stream_kind 4096 edges");
  // ④ copy-ahead rescaling
  std::mt19937 rng(12345);
  const int E = 384, K = 6;
  long n_cases = 0, n_shrunk = 0;
  for (int it = 0; it < 20000; ++it) {
    const int prev_rows = (int)(64 + rng() % 16384), min_rows = (int)(rng() % 3 == 0 ? 0 : 1 + rng() % 64), depth = (int)(1 + rng() % 96);
    std::vector<int> lr = it % 50 == 0 ? route_rows(rng, E, prev_rows, K) : std::vector<int>(E);
    if (it % 50 != 0) {
      int left = prev_rows * K;  // fast synthesis: random distribution so the sum is prev_rows·K
      for (int e = 0; e < E && left > 0; ++e) { const int v = (int)std::min<long>(left, (long)(rng() % (2 * prev_rows * K / E + 2))); lr[e] = v; left -= v; }
      lr[rng() % E] += left;
    }
    int64_t num, den;
    // off · rows_now 0 = default
    sp::prefetch_ratio(false, (int)(1 + rng() % 20000), K, lr, num, den);
    EXPECT(num == 1 && den == 1 && pick(lr, num, den, min_rows, depth) == pick_old(lr, min_rows, depth), "off must equal old (it %d)", it);
    sp::prefetch_ratio(true, 0, K, lr, num, den);
    EXPECT(num == 1 && den == 1, "rows_now 0 must be 1/1");
    // same size (rows now = previous rows) = the same candidates as the default
    sp::prefetch_ratio(true, prev_rows, K, lr, num, den);
    EXPECT(pick(lr, num, den, min_rows, depth) == pick_old(lr, min_rows, depth), "same size must equal old (it %d prev %d)", it, prev_rows);
    // larger: if the default has ≥ depth candidates, the top depth are the same (ranking unchanged · only the threshold drops)
    const int grow = prev_rows * (int)(2 + rng() % 3);
    sp::prefetch_ratio(true, grow, K, lr, num, den);
    const auto old = pick_old(lr, min_rows, depth);
    if ((int)old.size() == depth) EXPECT(pick(lr, num, den, min_rows, depth) == old, "grow must keep the old top-depth (it %d)", it);
    // smaller: pass ⇔ lr·rows_now·K ≥ max(1,min_rows)·Σlr (exact integer comparison) · candidates are a subset of the default ones
    const int shrink = std::max(1, prev_rows / (int)(2 + rng() % 30));
    sp::prefetch_ratio(true, shrink, K, lr, num, den);
    const auto now = pick(lr, num, den, min_rows, E);
    const auto was = pick_old(lr, min_rows, E);
    for (int e : now) EXPECT(std::find(was.begin(), was.end(), e) != was.end(), "shrink must be a subset (it %d e %d)", it, e);
    int64_t sum = 0;
    for (int v : lr) sum += v;
    for (int e = 0; e < E; ++e) {
      const bool want = (int64_t)lr[e] * shrink * K >= (int64_t)std::max(1, min_rows) * sum;
      EXPECT(sp::prefetch_pass(lr[e], num, den, min_rows) == want, "shrink pass e %d", e);
    }
    n_shrunk += (long)(was.size() - now.size());
    ++n_cases;
  }
  // Representative scene: a 1K chunk after a 16K chunk — copy-ahead candidates shrink a lot (threshold 32 rows → 512 rows relative to the previous chunk)
  {
    std::mt19937 r2(7);
    const auto lr = route_rows(r2, E, 16384, K);
    int64_t num, den;
    sp::prefetch_ratio(true, 1024, K, lr, num, den);
    const size_t old_n = pick_old(lr, 32, 96).size(), new_n = pick(lr, num, den, 32, 96).size();
    printf("short_prefill: 16K→1K pre-copy candidates (depth 96) baseline %zu → scaled %zu\n", old_n, new_n);
    EXPECT(new_n < old_n, "16K→1K must prefetch fewer");
  }
  // ⑥ Q3 HIVE_PREFILL_SMALL parsing: off = unset, "" or "0" · on = max(N or no floor, max_batch+1, 9)
  {
    struct { const char* v; int mb; int want; } t2[] = {{nullptr, 8, 0}, {"", 8, 0}, {"0", 8, 0}, {"1", 8, 9}, {"1", 16, 17}, {"1", 1, 9}, {"x", 8, 9}, {"-3", 8, 9},
                                                        {"2", 8, 9}, {"9", 8, 9}, {"256", 8, 256}, {"512", 64, 512}, {"32", 64, 65}, {"1000", 8, 1000}, {"3.5", 8, 9},
                                                        {"99999999999", 8, 9}};
    for (auto& t : t2) { const int got = sp::parse_small(t.v, t.mb); EXPECT(got == t.want, "parse_small(%s, %d) = %d, want %d", t.v ? t.v : "null", t.mb, got, t.want); }
    // small_forward: only when floor ≤ M < threshold and not verify, batch or tiled · when off (floor 0) never, for any M
    for (int M : {1, 2, 8, 9, 100, 1023, 1024, 4096}) EXPECT(!sp::small_forward(0, false, false, false, M, 1024), "small off must be false (M %d)", M);
    int nt = 0;
    for (int mask = 0; mask < 8; ++mask) {
      const bool ver = mask & 1, bat = mask & 2, til = mask & 4;
      const bool got = sp::small_forward(9, ver, bat, til, 500, 1024);
      EXPECT(got == (!ver && !bat && !til), "small_forward mask %d = %d", mask, got);
      nt += got;
    }
    EXPECT(nt == 1, "small_forward true in %d cases", nt);
    EXPECT(!sp::small_forward(256, false, false, false, 255, 1024) && sp::small_forward(256, false, false, false, 256, 1024) &&
           sp::small_forward(256, false, false, false, 1023, 1024) && !sp::small_forward(256, false, false, false, 1024, 1024), "small_forward edges");
    // if the floor is ≥ threshold no chunk qualifies (absorbed — no rejection)
    for (int M : {9, 1023, 1024, 5000}) EXPECT(!sp::small_forward(2048, false, false, false, M, 1024), "floor ≥ threshold must be empty (M %d)", M);
    // multi_short: on · CPU misses on · rows < threshold
    for (int mask = 0; mask < 8; ++mask) {
      const bool on = mask & 1, cpu = mask & 2, small = mask & 4;
      EXPECT(sp::multi_short(on, cpu, small ? 128 : 1024, 1024) == (on && cpu && small), "multi_short mask %d", mask);
    }
    EXPECT(sp::multi_short(true, true, 1023, 1024) && !sp::multi_short(true, true, 1024, 1024), "multi_short edge");
  }
  // ⑤ runtime.cpp text contracts (these formulas are used in place)
  {
    std::ifstream f("engine/src/runtime.cpp");
    std::stringstream ss; ss << f.rdbuf();
    const std::string rt = ss.str();
    EXPECT(!rt.empty(), "cannot read engine/src/runtime.cpp (run from the repo root)");
    for (const char* needle : {
             "spo_.tail_short = env_on(\"HIVE_PREFILL_TAIL_SHORT\");",
             "spo_.prefetch_scale = env_on(\"HIVE_PREFETCH_SCALE\");",
             "spo_.prof = env_on(\"HIVE_PREFILL_PROF\");",
             "spo_.adapt_rows = env_on(\"HIVE_PREFILL_SHORT_ADAPT\") ? sp::parse_short_adapt(getenv(\"HIVE_PREFILL_SHORT_ADAPT\")) : 0;",
             "return sp::short_moe(spo_.tail_short, in_prefill_, verify_, batch_ != nullptr, opt_.cpu_for_misses, M, opt_.prefill_threshold);",
             "const bool prefill = (M >= opt_.prefill_threshold || in_prefill_) && !short_tail;",
             "const int dma_kind = sp::short_stream_kind(spo_.adapt_rows, total_rows) ? kDmaStreamShort : kDmaStream;",
             "const bool fixed_prefill = dma_frac_prefill > 0.f && kind == kDmaStream;",
             "if (sp::prefetch_pass(lr[e], pf_num, pf_den, min_rows) && store_.slot_of(l, e) < 0) cand.push_back({-lr[e], e});",
             "if ((M >= opt_.prefill_threshold || in_prefill_) && !verify_ && !short_moe) prefetch_layer_experts(l, M);",
             // Q3 — switch parsing · SMALL decision and prefill handling · TAIL_SHORT exclusion · forward_multi upper-layer branch (off = default call) · tail-mode condition unchanged (replay is not widened)
             "spo_.multi_tail_short = env_on(\"HIVE_PREFILL_MULTI_TAIL_SHORT\");",
             "spo_.small_rows = env_on(\"HIVE_PREFILL_SMALL\") ? sp::parse_small(getenv(\"HIVE_PREFILL_SMALL\"), opt_.max_batch) : 0;",
             "const bool small_fwd = sp::small_forward(spo_.small_rows, verify_, batch_ != nullptr, tiled, M, opt_.prefill_threshold);",
             "if (M >= opt_.prefill_threshold || small_fwd) store_.flush_promotions();",
             "in_prefill_ = (M_orig >= opt_.prefill_threshold || small_fwd) && !verify_;",
             "if (small_fwd_) return false;",
             "if (small_fwd_ && opt_.promote_misses > 0) for (int e = 0; e < E; ++e) if (!by_e[e].empty() && store_.slot_of(l, e) < 0) step_miss_.push_back(l * c.n_routed + e);",
             "auto mshort = [&](int rows) { return sp::multi_short(spo_.multi_tail_short, opt_.cpu_for_misses, rows, opt_.prefill_threshold); };",
             "if (n_short == 0) { moe_experts_multi(model_.layer(l), l, subs, stats); return; }",
             "return tail_len > 0 && M >= opt_.prefill_threshold && M > tail_len && tail_layer >= 0 && tail_layer < model_.n_loaded_layers();"})
      EXPECT(rt.find(needle) != std::string::npos, "runtime.cpp contract missing: %s", needle);
    // Q3: forward_multi encoder layers [0, L) keep the default call (no short-path branch) · only the two upper-layer sites (L · L+1..) use up_experts
    const size_t a = rt.find("// ---- Encoder layers [0, L): all units"), b = rt.find("// ---- Tail layer L (last kv source)"), z = rt.find("// ---- Last layer tail → head (logits)");
    EXPECT(a != std::string::npos && b != std::string::npos && z != std::string::npos && a < b && b < z, "forward_multi section markers");
    if (a != std::string::npos && b != std::string::npos && z != std::string::npos && a < b && b < z) {
      const std::string enc = rt.substr(a, b - a), upper = rt.substr(b, z - b);
      EXPECT(enc.find("moe_experts_multi(model_.layer(l), l, subs, stats);") != std::string::npos && enc.find("up_experts") == std::string::npos, "encoder layers must keep moe_experts_multi");
      size_t n_up = 0;
      for (size_t q = upper.find("up_experts("); q != std::string::npos; q = upper.find("up_experts(", q + 1)) ++n_up;
      EXPECT(n_up == 2 && upper.find("moe_experts_multi(model_.layer(") == std::string::npos, "upper layers: up_experts at L and L+1.. only (%zu)", n_up);
    }
  }
  // ⑦ F1 prefill isolation: (a) DMA share kind in moe_decode_experts — prefill layers always use short prefill regardless of row
  //     count (outside the decode state dma_frac_[kDmaDecode] and the C1 samples) · non-prefill = the default formula (M > max_batch)
  //     unchanged. (b) forward_multi upper-layer observations = the same order, weights and rows as the observations of the single
  //     moe_experts_multi(subs) call (the per-unit variant = 1/Mu per unit · row i/k → the difference is shown numerically here).
  //     (c) runtime.cpp text contracts (call and disable sites).
  {
    for (int mb : {1, 4, 8, 16})
      for (int M = 1; M <= 2048; ++M) {
        EXPECT(sp::decode_path_short_kind(M, mb, false) == (M > mb), "not prefill: old formula (M %d mb %d)", M, mb);
        EXPECT(sp::decode_path_short_kind(M, mb, true), "prefill layer must use prefill-owned kind (M %d mb %d)", M, mb);
      }
    struct Sub { int M; std::vector<int32_t> ids; const int32_t* route_ids_h; };
    struct Ob { int e; float w; int row; bool operator==(const Ob& o) const { return e == o.e && w == o.w && row == o.row; } };
    std::mt19937 rng(5);
    long n_obs = 0;
    for (int trial = 0; trial < 200; ++trial) {
      const int k = 6, P = 1 + trial % 4;
      std::vector<Sub> subs(P);
      for (auto& sc : subs) { sc.M = 1 + (int)(rng() % 200); sc.ids.resize((size_t)sc.M * k); for (auto& v : sc.ids) v = (int32_t)(rng() % 384); sc.route_ids_h = sc.ids.data(); }
      for (bool ps : {false, true}) {
        // reference = the moe_experts_multi observation loop (formula from runtime.cpp: weight phase_score ? 1/Σrows : 1 · row S==1 ? 0 : -1 · s → m → j · hist per sub-chunk)
        std::vector<Ob> ref, got, q3; std::vector<int> ref_h, got_h;
        int total = 0; for (auto& sc : subs) total += sc.M;
        const float sw = ps ? 1.f / std::max(1, total) : 1.f;
        const int S = (int)subs.size();
        for (int si = 0; si < S; ++si) { for (int m = 0; m < subs[si].M; ++m) for (int j = 0; j < k; ++j) ref.push_back({subs[si].route_ids_h[m * k + j], sw, S == 1 ? 0 : -1}); ref_h.push_back(si); }
        int hi = 0;
        sp::observe_like_multi(subs, k, ps, [&](int e, float w, int row) { got.push_back({e, w, row}); }, [&](const Sub& sc) { got_h.push_back((int)(&sc - subs.data())); ++hi; });
        EXPECT(got == ref && got_h == ref_h, "observe_like_multi != moe_experts_multi observation (trial %d P %d ps %d)", trial, P, (int)ps);
        n_obs += (long)got.size();
        // per-unit variant (moe_decode_experts observations per unit: 1/Mu · row i/k) — with 2+ units the weight sum is P times larger (negative control: this formula differs from the reference)
        double w_ref = 0, w_q3 = 0;
        for (auto& o : ref) w_ref += o.w;
        for (auto& sc : subs) { const float wu = ps ? 1.f / std::max(1, sc.M) : 1.f; for (int i = 0; i < sc.M * k; ++i) { q3.push_back({sc.route_ids_h[i], wu, i / k}); w_q3 += wu; } }
        if (ps) EXPECT(std::fabs(w_ref - k) < 1e-3 && std::fabs(w_q3 - (double)k * P) < 1e-3 * P, "weight sums: streaming %.4f (= k) · per-unit %.4f (= k·P)", w_ref, w_q3);
        if (P >= 2 || !ps) EXPECT(!(q3 == ref) || (!ps && P == 1), "per-unit observation must differ from the streaming call (trial %d)", trial);
      }
    }
    std::ifstream f("engine/src/runtime.cpp");
    std::stringstream ss; ss << f.rdbuf();
    const std::string rt = ss.str();
    for (const char* needle : {
             "const int kind = sp::decode_path_short_kind(M, opt_.max_batch, in_prefill_) ? kDmaShort : kDmaDecode;",
             "sp::observe_like_multi(subs, kk, opt_.phase_score, [&](int e, float wt, int row) { store_.observe(l, e, wt, row); },",
             "dov_->obs_off = true;",
             "if (D.obs_off) return;",
             "if (!D.obs_off) xtrace_layer(l, M, k, w.route_ids_h);",
             "const bool obs = !dov_->obs_off;",
             "if (obs) store_.observe(l, e, score_weight, S == 1 ? 0 : -1);",
             "if (obs) xtrace_hist(l, sc.M, k, E, sc.route_ids_h);"})
      EXPECT(rt.find(needle) != std::string::npos, "runtime.cpp F1 contract missing: %s", needle);
    // the disable site is only inside the short branch of up_experts (after the n_short == 0 early return · before the unit loop) — every other call always observes
    const size_t u0 = rt.find("auto up_experts = [&](int l) {"), u1 = rt.find("auto tail_prev = [&](int l, int M) {");
    EXPECT(u0 != std::string::npos && u1 != std::string::npos && u0 < u1, "up_experts markers");
    if (u0 != std::string::npos && u1 != std::string::npos && u0 < u1) {
      const std::string ue = rt.substr(u0, u1 - u0);
      const size_t a = ue.find("if (n_short == 0) { moe_experts_multi(model_.layer(l), l, subs, stats); return; }"), b = ue.find("sp::observe_like_multi("),
                   c = ue.find("dov_->obs_off = true;"), d = ue.find("moe_decode_experts(model_.layer(l), l, Mu, stats);");
      EXPECT(a != std::string::npos && a < b && b < c && c < d, "up_experts order: early return → observe once → obs_off → unit calls");
    }
    size_t n_off = 0;
    for (size_t q = rt.find("obs_off = true"); q != std::string::npos; q = rt.find("obs_off = true", q + 1)) ++n_off;
    EXPECT(n_off == 1, "obs_off set in exactly one place (%zu)", n_off);
    printf("short_prefill F1: kind truth table · %ld observations = moe_experts_multi observations · text contract ok\n", n_obs);
  }
  if (fails) { fprintf(stderr, "short_prefill: %d failure(s)\n", fails); return 1; }
  printf("short_prefill: %ld random cases (off/same size = baseline candidates · larger = baseline top depth · smaller = subset, %.1f fewer on average) · parsing, truth table, text contract ok\n", n_cases,
         (double)n_shrunk / std::max(1L, n_cases));
  return 0;
}
