// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Cache replayer — plays an expert trace (HIVE_EXPERT_TRACE, format = runtime.h xtrace_*) back **in file order (= engine execution order)** and runs the
//   cache policy with **the engine's own code**. tools/cache_replay.py prepends a generated prelude to this file and builds it:
//     · engine/src/expert_store.cpp (whole, unmodified) — observe/decay/promote/promote_keys/commit/seed/policy (HIVE_CACHE_POLICY) are all production code
//     · runtime.h's struct RuntimeOptions (with defaults) and the bodies of runtime.cpp's Runtime::promote_after_step / warm_cache / warm_from_keys (text slices)
//   The only thing this file imitates is **the order in which** Runtime calls the store (observe/classify/touch in moe_decode_experts, observe/touch_at in
//   moe_experts_multi, commit_pending → … → promote_after_step in forward/forward_batch, hived's warm_after_prefill condition, warm_from_keys at startup).
//   That call convention (weight formula etc.) is checked against the runtime.cpp source by tools/test_cache_replay_cpu.py (the test breaks if the source changes).
// Fake CUDA (synchronous mode): promotion copies complete as soon as they are issued → residency takes effect at commit_pending at the start of the next
//   forward = "issued at step end, hits from the next step" (1-step delay).
//   Measured basis: 8 promotions × 18.8 MB = 150 MB ÷ ~25 GB/s ≈ 6 ms < a 26.9 ms decode step (HIVE_PROFILE) — they finish within one step.
// Not modeled: the DMA share (staging, cache unchanged), copy bandwidth contention, time (only steps are counted). Routing does not depend on the cache
//   (lossless), so a trace recorded under any policy can be replayed.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace hive;

namespace {
// Also builds against an expert_store from before the policy API (for the test's "policy unset = same decisions as without policy" comparison): falls back to the old calls when the new API is absent
template <class S> auto set_owners(S& s, const uint32_t* u, int n, bool p, int) -> decltype(s.set_row_owners(u, n, p), void()) { s.set_row_owners(u, n, p); }
template <class S> void set_owners(S&, const uint32_t*, int, bool, long) {}
template <class S> auto observe_row(S& s, int l, int e, float w, int row, int) -> decltype(s.observe(l, e, w, row), void()) { s.observe(l, e, w, row); }
template <class S> void observe_row(S& s, int l, int e, float w, int, long) { s.observe(l, e, w); }
struct LRec { int l, k, M; std::vector<uint16_t> ids; };
struct HRec { int l; uint32_t M; std::vector<uint16_t> cnt; };
struct Step { int kind = 0, M = 0; std::vector<uint32_t> uids; std::vector<LRec> L; std::vector<HRec> H; };

std::vector<Step> parse(const char* path, size_t& bytes, size_t& parsed) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
  std::vector<uint8_t> b;
  { uint8_t tmp[1 << 16]; size_t n; while ((n = fread(tmp, 1, sizeof tmp, f)) > 0) b.insert(b.end(), tmp, tmp + n); }
  fclose(f);
  bytes = b.size();
  std::vector<Step> steps;
  size_t off = 0;
  auto u16 = [&](size_t o) { uint16_t v; memcpy(&v, &b[o], 2); return v; };
  auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, &b[o], 4); return v; };
  while (off < b.size()) {
    const uint8_t tag = b[off];
    if (tag == 'S') {
      if (off + 14 > b.size()) break;
      const int nt = u16(off + 4);
      if (off + 14 + 8 * (size_t)nt > b.size()) break;
      Step s; s.kind = b[off + 1]; s.M = u16(off + 2);
      int nreal = nt;
      if (s.M == 0xFFFF && nt >= 1 && u32(off + 14 + 8 * (nt - 1)) == 0xFFFFFFFFu) { s.M = (int)u32(off + 14 + 8 * (nt - 1) + 4); --nreal; }  // B9 large M
      for (int i = 0; i < nreal; ++i) s.uids.push_back(u32(off + 14 + 8 * i));
      steps.push_back(std::move(s));
      off += 14 + 8 * (size_t)nt;
    } else if (tag == 'L') {
      if (off + 6 > b.size()) break;
      LRec r; r.l = b[off + 1]; r.k = u16(off + 2); r.M = u16(off + 4);
      const size_t n = (size_t)r.k * r.M;
      if (off + 6 + 2 * n > b.size()) break;
      r.ids.resize(n); memcpy(r.ids.data(), &b[off + 6], 2 * n);
      off += 6 + 2 * n;
      if (!steps.empty()) steps.back().L.push_back(std::move(r));
    } else if (tag == 'H') {
      if (off + 8 > b.size()) break;
      HRec r; r.l = b[off + 1]; const int E = u16(off + 2); r.M = u32(off + 4);
      if (off + 8 + 2 * (size_t)E > b.size()) break;
      r.cnt.resize(E); memcpy(r.cnt.data(), &b[off + 8], 2 * (size_t)E);
      off += 8 + 2 * (size_t)E;
      if (!steps.empty()) steps.back().H.push_back(std::move(r));
    } else {
      fprintf(stderr, "unknown record 0x%02x @ %zu — stop\n", tag, off);
      break;
    }
  }
  parsed = off;
  // Drop a truncated last step (daemon shutdown): a main-model step needs all of layers 0..39, a draft step layers 40.. (a prefill chunk has 40 layers either way, H or L)
  while (!steps.empty()) {
    const Step& s = steps.back();
    const size_t nl = s.L.size() + s.H.size();
    if ((s.kind == 3 && nl >= 1) || nl >= 40 || (s.kind == 1 && nl == 0)) break;
    steps.pop_back();
  }
  return steps;
}

struct Acc {  // per-kind totals (evaluation window only); cpu_ms = estimated CPU time of miss jobs (0.30 ms per expert + 0.05 ms per miss row — the measurement noted at runtime.h prefill_threshold)
  uint64_t rows = 0, hit = 0, jobs = 0, steps = 0;
  double cpu_ms = 0;
};

// Budgeted Belady (reference upper bound — not an engine policy): at every promotion point, swap up to B non-residents that will be **reused soonest** with
//   residents that will be **reused latest** (only when the incoming one is used earlier). Same slot count, promotion count and 1-step delay — how much more a policy could hit if it knew the future.
struct Oracle {
  int K, slots, B;
  std::vector<std::vector<int>> occ;  // key → step numbers at which the decode path uses it (ascending)
  std::vector<size_t> ptr;
  std::vector<char> res;
  int n_res = 0;
  Oracle(const std::vector<Step>& steps, int K_, int E, int slots_, int B_) : K(K_), slots(slots_), B(B_), occ(K_), ptr(K_, 0), res(K_, 0) {
    for (size_t i = 0; i < steps.size(); ++i)
      for (const LRec& r : steps[i].L)
        for (uint16_t e : r.ids) { const int k = r.l * E + e; if (k < K && (occ[k].empty() || occ[k].back() != (int)i)) occ[k].push_back((int)i); }
  }
  int next_use(int k, int after) {
    auto& v = occ[k]; size_t& p = ptr[k];
    while (p < v.size() && v[p] <= after) ++p;
    return p < v.size() ? v[p] : INT32_MAX;
  }
  int window = 0;  // > 0: value = number of use steps within the next `window` steps (more first, ties → used sooner) — an upper bound that does not spend bandwidth on promotions used once and dropped
  long value(int k, int now, int nu) {
    if (window <= 0) return nu == INT32_MAX ? INT64_MAX / 4 : nu;
    auto& v = occ[k];
    const long c = std::upper_bound(v.begin() + ptr[k], v.end(), now + window) - (v.begin() + ptr[k]);
    return c == 0 ? INT64_MAX / 4 : -c * (1L << 32) + nu;
  }
  void promote(int now, int budget) {
    std::vector<std::pair<long, int>> cand, vict;
    for (int k = 0; k < K; ++k) {
      const int nu = next_use(k, now);
      const long v = value(k, now, nu);
      if (res[k]) vict.push_back({-v, k});
      else if (v != INT64_MAX / 4) cand.push_back({v, k});
    }
    const size_t nc = std::min<size_t>(budget, cand.size());
    std::partial_sort(cand.begin(), cand.begin() + nc, cand.end());
    std::sort(vict.begin(), vict.end());
    size_t vi = 0;
    for (size_t i = 0; i < nc; ++i) {
      if (n_res < slots) { res[cand[i].second] = 1; ++n_res; continue; }
      if (vi >= vict.size() || -vict[vi].first <= cand[i].first) break;
      res[vict[vi].second] = 0; res[cand[i].second] = 1; ++vi;
    }
  }
};
}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) { fprintf(stderr, "usage: cache_replay TRACE [--slots N] [--promote N] [--promote-misses N] [--phase-score 0|1] [--cache-state F] [--eval-from X] [--oracle] [--label S]\n"); return 2; }
  const char* trace = argv[1];
  int slots = 3792, promote = 8, promote_misses = 0, phase = 1, oracle = 0, mtp_cache = 0, oracle_window = 0;
  double eval_from = 0.0;
  std::string cache_state, label;
  for (int i = 2; i < argc; i++) {
    const std::string a(argv[i]);
    auto next = [&] { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
    if (a == "--slots") slots = atoi(next().c_str());
    else if (a == "--promote") promote = atoi(next().c_str());
    else if (a == "--promote-misses") promote_misses = atoi(next().c_str());
    else if (a == "--phase-score") phase = atoi(next().c_str());
    else if (a == "--mtp-cache") mtp_cache = atoi(next().c_str());
    else if (a == "--cache-state") cache_state = next();
    else if (a == "--eval-from") eval_from = atof(next().c_str());
    else if (a == "--oracle") oracle = 1;
    else if (a == "--oracle-window") { oracle = 1; oracle_window = atoi(next().c_str()); }
    else if (a == "--label") label = next();
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  size_t bytes = 0, parsed = 0;
  std::vector<Step> steps = parse(trace, bytes, parsed);
  Config cfg;  // config with only the record size reduced (dim/inter 64 → 8 KiB records): slot count, key convention (l·384 + e) and draft-layer expert count (128) match production
  cfg.dim = 64; cfg.moe_inter = 64; cfg.n_routed = 384; cfg.n_act = 6; cfg.dspark_experts = 128; cfg.n_layers = 40;
  const int n_back = 40, n_mtp = 3, E = cfg.n_routed;
  const size_t rec = ExpertLayout::make(cfg.dim, cfg.moe_inter).total;
  ExpertStore store(cfg, n_back, (size_t)slots * rec, 1, n_mtp);
  RuntimeOptions opt;
  opt.promote_per_token = promote;
  opt.promote_misses = promote_misses;
  opt.phase_score = phase != 0;
  opt.mtp_cache = mtp_cache != 0;
  Runtime rt(store, opt);
  const int K = (n_back + n_mtp) * E;
  int warm_start = 0;
  if (!cache_state.empty()) {  // same header check as the hived warm start
    FILE* f = fopen(cache_state.c_str(), "rb");
    int32_t hdr[4] = {0, 0, 0, 0};
    std::vector<int32_t> keys;
    if (f && fread(hdr, 4, 4, f) == 4 && hdr[0] == 0x48564353 && hdr[1] == store.n_layers() && hdr[2] == store.E() && hdr[3] > 0 && hdr[3] < 100000) {
      keys.resize(hdr[3]);
      if (fread(keys.data(), 4, keys.size(), f) != keys.size()) keys.clear();
    }
    if (f) fclose(f);
    if (keys.empty()) fprintf(stderr, "cache-state %s ignored (header mismatch)\n", cache_state.c_str());
    else warm_start = rt.warm_from_keys(keys);
  }
  // hived warm_after_prefill condition: right after a request's prefill (consecutive kind 1 steps) ends, warm_cache() if new tokens >= 64 and residency < 90 %
  std::vector<char> warm_after(steps.size(), 0);
  {
    std::vector<int> last_kind1_step;  // uid → last step of the ongoing prefill run (-1 = none), accumulated M
    std::vector<long> run_m;
    auto grow = [&](uint32_t u) { if (u >= last_kind1_step.size()) { last_kind1_step.resize(u + 1, -1); run_m.resize(u + 1, 0); } };
    for (size_t i = 0; i < steps.size(); ++i) {
      const Step& s = steps[i];
      for (uint32_t u : s.uids) {
        if (u == 0xFFFFFFFFu) continue;
        grow(u);
        if (s.kind == 1) { last_kind1_step[u] = (int)i; run_m[u] += s.M; }
        else if (last_kind1_step[u] >= 0) { if (run_m[u] >= 64) warm_after[last_kind1_step[u]] = 1; last_kind1_step[u] = -1; run_m[u] = 0; }
      }
    }
  }
  size_t n_dec = 0;
  for (const Step& s : steps) n_dec += !s.L.empty();
  const size_t eval_t0 = (size_t)(eval_from * (double)n_dec);
  Oracle* orc = oracle ? new Oracle(steps, K, E, store.n_slots(), promote) : nullptr;
  if (orc) orc->window = oracle_window;
  Acc acc[4];
  std::vector<uint64_t> layer_rows(n_back + n_mtp, 0), layer_hit(n_back + n_mtp, 0), layer_jobs(n_back + n_mtp, 0);
  uint64_t promoted = 0, warm_n = 0, promote_points = 0;
  size_t dec_t = 0;
  std::vector<int> ecount(512);
  std::vector<char> was_res(K, 0);
  std::vector<uint32_t> res_hits(K, 0);
  uint64_t entered = 0, left = 0, wasted = 0;
  for (size_t i = 0; i < steps.size(); ++i) {
    const Step& s = steps[i];
    const bool has_l = !s.L.empty();
    if (has_l) ++dec_t;
    const bool counting = has_l && dec_t > eval_t0;
    store.commit_pending();  // start of forward/forward_batch/mtp_draft
    if (!orc) {  // resident entry/exit tracking: entered and left without a single hit = wasted promotion (bandwidth only)
      for (int k = 0; k < K; ++k) {
        const bool r = store.slot_of(k / E, k % E) >= 0;
        if (r && !was_res[k]) { was_res[k] = 1; res_hits[k] = 0; ++entered; }
        else if (!r && was_res[k]) { was_res[k] = 0; ++left; if (res_hits[k] == 0) ++wasted; }
      }
    }
    if (counting) ++acc[s.kind & 3].steps;
    if (!orc && !s.uids.empty()) set_owners(store, s.uids.data(), (int)s.uids.size(), s.kind == 1, 0);  // start of Runtime::xtrace_step (every forward calls it)
    for (const LRec& r : s.L) {  // store call order of Runtime::moe_decode_experts
      // score_weight = opt_.phase_score && score_prefill_ ? 1/M : 1 — score_prefill_ = !verify_ in forward() (kind 1); false for forward_batch/draft (kind 0/3) and verify (kind 2)
      const float w = opt.phase_score && s.kind == 1 ? 1.f / std::max(1, r.M) : 1.f;
      const int R = r.M * r.k;
      std::fill(ecount.begin(), ecount.end(), 0);
      for (int j = 0; j < R; ++j) { ++ecount[r.ids[j]]; if (!orc) observe_row(store, r.l, r.ids[j], w, j / r.k, 0); }  // row = j / k (with n = 1, set_row_owners attributes every row to that sequence)
      for (int e = 0; e < 512; ++e) {
        if (!ecount[e]) continue;
        const int key = r.l * E + e;
        const bool hit = orc ? orc->res[key] != 0 : store.slot_of(r.l, e) >= 0;
        if (hit) { if (!orc) { store.touch(store.slot_of(r.l, e)); ++res_hits[key]; } }
        else if (!orc && opt.promote_misses > 0) rt.step_miss_.push_back(key);
        if (counting) {
          acc[s.kind & 3].rows += ecount[e];
          if (hit) { acc[s.kind & 3].hit += ecount[e]; layer_hit[r.l] += ecount[e]; }
          else { ++acc[s.kind & 3].jobs; ++layer_jobs[r.l]; acc[s.kind & 3].cpu_ms += 0.30 + 0.05 * ecount[e]; }
          layer_rows[r.l] += ecount[e];
        }
      }
    }
    // Runtime::moe_experts_multi (streaming prefill): all sub-chunks of a layer in one call — total_rows = the summed rows of consecutive H records of the same layer
    for (size_t h = 0; h < s.H.size();) {
      size_t h1 = h;
      uint64_t total_rows = 0;
      while (h1 < s.H.size() && s.H[h1].l == s.H[h].l) total_rows += s.H[h1++].M;
      const int l = s.H[h].l;
      const float w = opt.phase_score ? 1.f / (float)std::max<uint64_t>(1, total_rows) : 1.f;
      if (!orc) {
        for (size_t q = h; q < h1; ++q)
          for (size_t e = 0; e < s.H[q].cnt.size(); ++e)
            for (int c = 0; c < s.H[q].cnt[e]; ++c) observe_row(store, l, (int)e, w, h1 - h == 1 ? 0 : -1, 0);  // moe_experts_multi: row 0 (that sequence) with one sub-chunk, -1 with several
        const uint64_t stamp = store.lru_stamp();
        for (size_t e = 0; e < s.H[h].cnt.size(); ++e) {
          bool used = false;
          for (size_t q = h; q < h1; ++q) used |= s.H[q].cnt[e] > 0;
          if (used && store.slot_of(l, (int)e) >= 0) store.touch_at(store.slot_of(l, (int)e), stamp);
        }
      }
      h = h1;
    }
    if (s.kind == 3) {  // MissClear of mtp_draft: with mtp_cache, draft misses go to draft_miss_, otherwise dropped (the next main-model step promotes)
      if (!orc && opt.mtp_cache) rt.draft_miss_ = std::move(rt.step_miss_);
      rt.step_miss_.clear();
    } else if (has_l) {  // promotion at the end of forward (M < prefill_threshold) / forward_batch
      if (opt.promote_per_token > 0 || opt.promote_misses > 0) {
        ++promote_points;
        if (orc) orc->promote((int)i, std::max(opt.promote_per_token, opt.promote_misses));
        else promoted += rt.promote_after_step();
      }
    } else {
      rt.step_miss_.clear();
    }
    if (warm_after[i] && !orc && store.n_resident() < store.n_slots() * 9 / 10) warm_n += rt.warm_cache();
    if (warm_after[i] && orc && orc->n_res < store.n_slots() * 9 / 10) orc->promote((int)i, opt.warm_cap);
  }
  store.commit_all();
  const char* pol = getenv("HIVE_CACHE_POLICY");
  Acc all, dec;  // dec = decode (batch decode 0, verify 2, draft 3) — the token generation path, excluding prompt chunks (1)
  for (int k : {0, 1, 2, 3}) {
    all.rows += acc[k].rows; all.hit += acc[k].hit; all.jobs += acc[k].jobs; all.steps += acc[k].steps; all.cpu_ms += acc[k].cpu_ms;
    if (k != 1) { dec.rows += acc[k].rows; dec.hit += acc[k].hit; dec.jobs += acc[k].jobs; dec.cpu_ms += acc[k].cpu_ms; }
  }
  dec.steps = acc[0].steps + acc[2].steps;  // token steps (drafts belong to verify steps)
  auto pct = [](uint64_t a, uint64_t b) { return b ? 100.0 * (double)a / (double)b : 0.0; };
  uint64_t brows = 0, bhit = 0, bjobs = 0;
  for (int l = 0; l < n_back; ++l) { brows += layer_rows[l]; bhit += layer_hit[l]; bjobs += layer_jobs[l]; }
  printf("{\"label\": \"%s\", \"policy\": \"%s\", \"trace\": \"%s\", \"bytes\": %zu, \"parsed_bytes\": %zu, \"steps\": %zu, \"decode_path_steps\": %zu, "
         "\"eval_from\": %.3f, \"slots\": %d, \"promote\": %d, \"promote_misses\": %d, \"phase_score\": %d, \"warm_start\": %d, \"warm_cache\": %" PRIu64 ", "
         "\"hit_pct\": %.3f, \"backbone_hit_pct\": %.3f, \"draft_hit_pct\": %.3f, \"miss_jobs_per_step\": %.3f, \"backbone_miss_jobs\": %" PRIu64 ", "
         "\"promotions_per_point\": %.3f, \"resident_end\": %d, \"decode_hit_pct\": %.3f, \"decode_jobs_per_step\": %.3f, \"decode_cpu_ms_per_step\": %.3f, "
         "\"chunk_hit_pct\": %.3f, \"chunk_cpu_ms_total\": %.1f",
         label.c_str(), orc ? (oracle_window > 0 ? "oracle(next-window uses)" : "oracle(budgeted Belady)") : (pol && *pol ? pol : "current"), trace, bytes, parsed, steps.size(), n_dec, eval_from, store.n_slots(),
         promote, promote_misses, phase, warm_start, warm_n, pct(all.hit, all.rows), pct(bhit, brows),
         pct(layer_hit[40] + layer_hit[41] + layer_hit[42], layer_rows[40] + layer_rows[41] + layer_rows[42]),
         all.steps ? (double)all.jobs / (double)all.steps : 0.0, bjobs, promote_points ? (double)promoted / (double)promote_points : 0.0,
         orc ? orc->n_res : store.n_resident(), pct(dec.hit, dec.rows), dec.steps ? (double)dec.jobs / (double)dec.steps : 0.0,
         dec.steps ? dec.cpu_ms / (double)dec.steps : 0.0, pct(acc[1].hit, acc[1].rows), acc[1].cpu_ms);
  const char* names[4] = {"decode", "chunk", "verify", "draft"};
  printf(", \"entered\": %" PRIu64 ", \"evicted\": %" PRIu64 ", \"evicted_unhit\": %" PRIu64, entered, left, wasted);
  printf(", \"kinds\": {");
  for (int k = 0; k < 4; ++k)
    printf("%s\"%s\": {\"steps\": %" PRIu64 ", \"rows\": %" PRIu64 ", \"hit_pct\": %.3f, \"miss_jobs_per_step\": %.3f}", k ? ", " : "", names[k], acc[k].steps, acc[k].rows,
           pct(acc[k].hit, acc[k].rows), acc[k].steps ? (double)acc[k].jobs / (double)acc[k].steps : 0.0);
  printf("}, \"layer_miss_jobs\": [");
  for (int l = 0; l < n_back + n_mtp; ++l) printf("%s%" PRIu64, l ? ", " : "", layer_jobs[l]);
  printf("]}\n");
  delete orc;
  return 0;
}
