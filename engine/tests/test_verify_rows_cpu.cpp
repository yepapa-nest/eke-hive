// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// HIVE_MTP_VERIFY2 / HIVE_MTP_BATCH — CPU test of the verify-row design (no GPU). A toy state machine checks that "what verify row m sees == what sequential decode
//   sees at that position". Window columns and the compressed table use **the same functions** as production (hive/verify_rows.h — called by the window_idxs_verify kernel and attention_verify_host).
//   State: per sequence a ring[win] (content c(q) of position q) · compressor state[ratio] · compressed cache (group g = hash of slot contents). Sequential decode (M=1 — the semantics of the service decode path:
//   self = chunk row, earlier positions = ring, one compressor row, ring write) is the reference. Verify (several parts · rows = part order) follows the order of runtime.cpp attention_verify_dev:
//   ① compressor (row order) → cache write (row that ends a group) ② per row window columns (window_col) → ring/chunk · compressed columns comp_len(tab) ③ ring write (after attention).
//   rollback_batch: restore ring slots (snapshot) · compressor state = snapshot + re-apply accepted rows (compressor_apply_rows) · rewind position/tokens.
//   Checks: (a) every verify row output == sequential · (b) future independence (changing later rows' tokens leaves earlier rows' outputs unchanged) · (c) continuation after rollback at every keep position == sequential ·
//   (d) the output of one part (including length 1) is independent of the other parts in the same batch. All 5 negative controls (mutations inside the test — production code is unchanged) must be caught:
//   decode-style window columns (all earlier positions from the ring) · ring written before attention · parallel compressor (rows see only the step-head state) · comp_len from the part's first row · chunk columns missing the part's first row (g).
//   Batched MTP: (e) multi-step schedule — 2–4 sequences per step under a row budget Σ(k_s+1) ≤ 8 (same cap as hived choose_batch · k_s = 0 = a plain one-row decode)
//   verified with shuffled part order and rollback_batch repeated with a different keep per part → per sequence "accepted row outputs == that sequence decoded alone sequentially" · identical final continuation (sequence independence:
//   independent of other sequences, part order and row positions in the same batch). Two more negative controls: rollback uses another part's snapshot · all parts rewound to the first part's pos0 (leftover v_pos0_ of single verify).
//   What this test proves = table formulas and state order (the design). Kernel values = test_verify_decode (GPU) · model numerics = hive --verify2-test (GPU).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "hive/verify_rows.h"

using namespace hive;

namespace {
int fails = 0;
long checks = 0;
uint64_t mix(uint64_t h, uint64_t v) { h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2); return h * 0x100000001B3ull; }
uint64_t content(int32_t tok, int64_t q) { return mix(mix(1469598103934665603ull, (uint64_t)(uint32_t)tok), (uint64_t)q); }
constexpr uint64_t kUnset = 0xDEADDEADDEADDEADull;

struct Mut { bool decode_window = false, ring_before = false, comp_parallel = false, comp_len_first = false, chunk_no_g = false, snap_other = false, pos0_first = false; };

struct SeqS {
  int win = 0, ratio = 1;
  std::vector<int32_t> toks;
  std::vector<uint64_t> ring, state, cache;
  int64_t pos() const { return (int64_t)toks.size(); }
  void init(int w, int r) { win = w; ratio = r; ring.assign(w, kUnset); state.assign(r, kUnset); cache.clear(); toks.clear(); }
  uint64_t group_of_state() const { uint64_t h = 7; for (uint64_t v : state) h = mix(h, v); return h; }
  void put_cache(int64_t g, uint64_t v) { if ((int64_t)cache.size() <= g) cache.resize(g + 1, kUnset); cache[g] = v; }
};

// sequential decode of one token (reference): output = hash of what it sees (window column order · compressed group order)
uint64_t decode_one(SeqS& s, int32_t tok) {
  const int64_t p = s.pos();
  const uint64_t kv = content(tok, p);
  {  // compressor (ratio 1 uses the same formula — group = one position)
    s.state[p % s.ratio] = kv;
    if ((p + 1) % s.ratio == 0) s.put_cache(p / s.ratio, s.group_of_state());
  }
  uint64_t h = 11;
  for (int j = 0; j < s.win; ++j) {
    const int64_t src = p - (s.win - 1) + j;
    h = mix(h, src < 0 ? 0 : src < p ? s.ring[src % s.win] : kv);
  }
  const int cl = (int)((p + 1) / s.ratio);
  for (int g = 0; g < cl; ++g) h = mix(h, g < (int)s.cache.size() ? s.cache[g] : kUnset);  // (reads stay safe even if a mutation rewinds the position wrongly — caught as a value mismatch)
  s.ring[p % s.win] = kv;
  s.toks.push_back(tok);
  return h;
}

struct Part { SeqS* s; std::vector<int32_t> ids; };
struct Snap { int64_t pos0; std::vector<uint64_t> ring_slots, state; int M, row0; };

// verify (several parts): output[row] · snapshots (for rollback)
std::vector<uint64_t> verify(std::vector<Part>& parts, std::vector<Snap>& snaps, const Mut& mu) {
  std::vector<uint64_t> kv, out;
  std::vector<int64_t> pos;
  std::vector<int> grp, part_of;
  snaps.clear();
  for (size_t pi = 0; pi < parts.size(); ++pi) {
    SeqS& s = *parts[pi].s;
    Snap sn{s.pos(), {}, s.state, (int)parts[pi].ids.size(), (int)kv.size()};
    for (int i = 0; i < sn.M; ++i) sn.ring_slots.push_back(s.ring[(sn.pos0 + i) % s.win]);
    snaps.push_back(sn);
    for (int i = 0; i < sn.M; ++i) {
      kv.push_back(content(parts[pi].ids[i], sn.pos0 + i)); pos.push_back(sn.pos0 + i); grp.push_back(sn.row0); part_of.push_back((int)pi);
    }
  }
  const int M = (int)kv.size();
  // ① compressor (row order) — mutation: parallel (step-head state + own slot only)
  std::vector<std::vector<uint64_t>> head_state(parts.size());
  for (size_t pi = 0; pi < parts.size(); ++pi) head_state[pi] = parts[pi].s->state;
  for (int m = 0; m < M; ++m) {
    SeqS& s = *parts[part_of[m]].s;
    const vrows::Tab t = vrows::tab(pos[m], s.ratio);
    if (mu.comp_parallel) {
      std::vector<uint64_t> st = head_state[part_of[m]];
      st[pos[m] % s.ratio] = kv[m];
      s.state[pos[m] % s.ratio] = kv[m];
      if (t.valid) { uint64_t h = 7; for (uint64_t v : st) h = mix(h, v); s.put_cache(t.dsti, h); }
    } else {
      s.state[pos[m] % s.ratio] = kv[m];
      if (t.valid) s.put_cache(t.dsti, s.group_of_state());
    }
  }
  if (mu.ring_before) for (int m = 0; m < M; ++m) { SeqS& s = *parts[part_of[m]].s; s.ring[pos[m] % s.win] = kv[m]; }
  // ② attention
  for (int m = 0; m < M; ++m) {
    SeqS& s = *parts[part_of[m]].s;
    const int64_t p0 = pos[grp[m]];
    uint64_t h = 11;
    for (int j = 0; j < s.win; ++j) {
      int32_t col;
      if (mu.decode_window) {  // window_idxs_rows formula (assumes a different sequence per row)
        const int64_t src = pos[m] - (s.win - 1) + j;
        col = src < 0 ? -1 : src < pos[m] ? (int32_t)(src % s.win) : (int32_t)(s.win + m);
      } else {
        col = vrows::window_col(pos[m], p0, mu.chunk_no_g ? 0 : grp[m], s.win, j);
      }
      h = mix(h, col < 0 ? 0 : col < s.win ? s.ring[col] : (col - s.win < M ? kv[col - s.win] : kUnset));
    }
    const int cl = vrows::tab(mu.comp_len_first ? p0 : pos[m], s.ratio).comp_len;
    for (int g = 0; g < cl; ++g) h = mix(h, g < (int)s.cache.size() ? s.cache[g] : kUnset);
    out.push_back(h);
  }
  // ③ ring (after attention) · advance position
  if (!mu.ring_before) for (int m = 0; m < M; ++m) { SeqS& s = *parts[part_of[m]].s; s.ring[pos[m] % s.win] = kv[m]; }
  for (auto& pt : parts) for (int32_t t : pt.ids) pt.s->toks.push_back(t);
  return out;
}

void rollback(std::vector<Part>& parts, const std::vector<Snap>& snaps, const std::vector<int>& keep, const Mut& mu = Mut{}) {
  for (size_t pi = 0; pi < parts.size(); ++pi) {
    SeqS& s = *parts[pi].s;
    const Snap& sn = snaps[mu.snap_other && snaps.size() > 1 ? (pi + 1) % snaps.size() : pi];  // batched-MTP mutation: another part's snapshot
    const int64_t p0 = mu.pos0_first ? snaps[0].pos0 : sn.pos0;                               // batched-MTP mutation: the first part's pos0
    const int nk = std::min(keep[pi], sn.M);
    if (nk < snaps[pi].M) {
      s.toks.resize((size_t)(p0 + nk));
      for (int i = nk; i < sn.M; ++i) s.ring[(p0 + i) % s.win] = sn.ring_slots[i];
      s.state = sn.state;
      for (int i = 0; i < nk; ++i) s.state[(p0 + i) % s.ratio] = content(parts[pi].ids[i], p0 + i);  // compressor_apply_rows
    }
  }
}

void expect(bool c, const char* what, int trial) {
  ++checks;
  if (!c) { if (fails < 20) fprintf(stderr, "FAIL trial %d: %s\n", trial, what); ++fails; }
}

// one test world: random shapes · prompts · part batches · all checks. With a mutation enabled it only counts failures (the caller decides whether it was caught).
int run_world(int trial, const Mut& mu, bool quiet) {
  std::mt19937 rng(1000 + trial);
  const int wins[] = {6, 7, 8, 16}, ratios[] = {1, 2, 3, 4};  // win ≥ part row count (≤ 6) — production win 128 ≥ 8 rows (a ring slot is never written twice within one verify)
  const int win = wins[rng() % 4], ratio = ratios[rng() % 4];
  const int P = 1 + (int)(rng() % 4);
  std::vector<int> Ms;
  int tot = 0;
  for (int p = 0; p < P; ++p) { int m = 1 + (int)(rng() % 6); if (tot + m > 8) m = std::max(1, 8 - tot); if (tot + m > 8) break; Ms.push_back(m); tot += m; }
  auto rtok = [&] { return (int32_t)(20 + rng() % 4000); };
  std::vector<std::vector<int32_t>> prompts(Ms.size()), drafts(Ms.size()), follow(Ms.size());
  for (size_t p = 0; p < Ms.size(); ++p) {
    const int L = (int)(rng() % 41);
    for (int i = 0; i < L; ++i) prompts[p].push_back(rtok());
    for (int i = 0; i < Ms[p]; ++i) drafts[p].push_back(rtok());
    for (int i = 0; i < 5; ++i) follow[p].push_back(rtok());
  }
  int local_fail = 0;
  auto chk = [&](bool c, const char* w) { if (!c) ++local_fail; if (!quiet) expect(c, w, trial); };
  auto fresh = [&](std::vector<SeqS>& ss) {
    ss.assign(Ms.size(), {});
    for (size_t p = 0; p < Ms.size(); ++p) { ss[p].init(win, ratio); for (int32_t t : prompts[p]) decode_one(ss[p], t); }
  };
  // reference: sequential decode output per part
  std::vector<std::vector<uint64_t>> ref(Ms.size());
  {
    std::vector<SeqS> ss; fresh(ss);
    for (size_t p = 0; p < Ms.size(); ++p) for (int32_t t : drafts[p]) ref[p].push_back(decode_one(ss[p], t));
  }
  // (a) every row == sequential
  std::vector<uint64_t> base_out;
  {
    std::vector<SeqS> ss; fresh(ss);
    std::vector<Part> parts; for (size_t p = 0; p < Ms.size(); ++p) parts.push_back({&ss[p], drafts[p]});
    std::vector<Snap> sn;
    base_out = verify(parts, sn, mu);
    int r = 0;
    for (size_t p = 0; p < Ms.size(); ++p) for (int i = 0; i < Ms[p]; ++i, ++r) chk(base_out[r] == ref[p][i], "verify row == sequential");
  }
  // (b) future independence: change the tokens after cut in part p → earlier rows (including other rows of every part) unchanged
  for (size_t p = 0; p < Ms.size(); ++p)
    for (int cut = 0; cut < Ms[p]; ++cut) {
      std::vector<SeqS> ss; fresh(ss);
      std::vector<Part> parts; for (size_t q = 0; q < Ms.size(); ++q) parts.push_back({&ss[q], drafts[q]});
      for (int i = cut; i < Ms[p]; ++i) parts[p].ids[i] = (parts[p].ids[i] + 97) % 4000 + 20;
      std::vector<Snap> sn;
      const std::vector<uint64_t> out = verify(parts, sn, mu);
      int r = 0;
      for (size_t q = 0; q < Ms.size(); ++q) for (int i = 0; i < Ms[q]; ++i, ++r) if (q != p || i < cut) chk(out[r] == base_out[r], "future independence");
    }
  // (c) keep combinations (random keep per part + accept all + accept one) → rollback → continuation == sequential (accepted prefix + continuation)
  for (int rep = 0; rep < 4; ++rep) {
    std::vector<int> keep(Ms.size());
    for (size_t p = 0; p < Ms.size(); ++p) keep[p] = rep == 0 ? Ms[p] : rep == 1 ? 1 : 1 + (int)(rng() % Ms[p]);
    std::vector<SeqS> ss; fresh(ss);
    std::vector<Part> parts; for (size_t p = 0; p < Ms.size(); ++p) parts.push_back({&ss[p], drafts[p]});
    std::vector<Snap> sn;
    verify(parts, sn, mu);
    rollback(parts, sn, keep, mu);
    for (size_t p = 0; p < Ms.size(); ++p) {
      SeqS r0; r0.init(win, ratio);
      for (int32_t t : prompts[p]) decode_one(r0, t);
      for (int i = 0; i < keep[p]; ++i) decode_one(r0, drafts[p][i]);
      chk(ss[p].pos() == r0.pos(), "rollback pos");
      for (int32_t t : follow[p]) chk(decode_one(ss[p], t) == decode_one(r0, t), "rollback continuation == sequential");
    }
  }
  // (e) batched-MTP multi-step schedule: row budget 8 · shuffled part order · k_s = 0 parts · different keep per part → each sequence equals its own sequential decode
  {
    const int NS = 2 + (int)(rng() % 3);
    std::vector<SeqS> ss(NS), solo(NS);
    std::vector<std::vector<int32_t>> pr(NS);
    for (int q = 0; q < NS; ++q) {
      ss[q].init(win, ratio); solo[q].init(win, ratio);
      const int L = (int)(rng() % 30);
      for (int i = 0; i < L; ++i) pr[q].push_back(rtok());
      for (int32_t t : pr[q]) { decode_one(ss[q], t); decode_one(solo[q], t); }
    }
    for (int step = 0; step < 12; ++step) {
      std::vector<int> order(NS);
      for (int q = 0; q < NS; ++q) order[q] = q;
      std::shuffle(std::begin(order), std::end(order), rng);
      int budget = 8 - NS;  // at least one row per sequence
      std::vector<Part> parts;
      std::vector<int> who;
      for (int q : order) {
        const int k = budget > 0 ? (int)(rng() % (std::min(budget, 5) + 1)) : 0;
        budget -= k;
        Part pt{&ss[q], {}};
        for (int i = 0; i <= k; ++i) pt.ids.push_back(rtok());
        parts.push_back(pt); who.push_back(q);
      }
      std::vector<Snap> sn;
      const std::vector<uint64_t> out = verify(parts, sn, mu);
      std::vector<int> keep(parts.size());
      for (size_t pi = 0; pi < parts.size(); ++pi) keep[pi] = 1 + (int)(rng() % parts[pi].ids.size());
      int r = 0;
      for (size_t pi = 0; pi < parts.size(); ++pi) {
        const int q = who[pi];
        for (int i = 0; i < (int)parts[pi].ids.size(); ++i, ++r)
          if (i < keep[pi]) chk(out[r] == decode_one(solo[q], parts[pi].ids[i]), "batch schedule: kept row == solo sequential");
      }
      rollback(parts, sn, keep, mu);
      for (int q = 0; q < NS; ++q) chk(ss[q].pos() == solo[q].pos(), "batch schedule: position after rollback");
    }
    for (int q = 0; q < NS; ++q)
      for (int i = 0; i < 4; ++i) { const int32_t t = rtok(); chk(decode_one(ss[q], t) == decode_one(solo[q], t), "batch schedule: continuation == solo"); }
  }
  return local_fail;
}
}  // namespace

int main() {
  const int kTrials = 3000;
  for (int t = 0; t < kTrials; ++t) run_world(t, Mut{}, false);
  printf("verify rows: %d worlds · %ld checks · %d failures\n", kTrials, checks, fails);
  // negative controls: each mutation must be caught in at least one world
  struct { const char* name; Mut m; } muts[] = {{"decode-style window (all earlier positions from the ring)", {true, false, false, false, false}},
                                                {"ring written before attention", {false, true, false, false, false}},
                                                {"parallel compressor (rows see the step-head state)", {false, false, true, false, false}},
                                                {"comp_len of the part's first row", {false, false, false, true, false}},
                                                {"chunk column without the part's first row", {false, false, false, false, true}},
                                                {"MB1: rollback uses another part's snapshot", {false, false, false, false, false, true, false}},
                                                {"MB1: rollback rewinds every part to the first part's pos0", {false, false, false, false, false, false, true}}};
  bool all = true;
  for (auto& mt : muts) {
    int caught = 0;
    for (int t = 0; t < kTrials; ++t) caught += run_world(t, mt.m, true) > 0 ? 1 : 0;
    printf("  mutant [%s]: %s (%d/%d worlds)\n", mt.name, caught ? "DETECTED" : "NOT DETECTED", caught, kTrials);
    all &= caught > 0;
  }
  if (fails || !all) { printf("test_verify_rows_cpu: FAIL\n"); return 1; }
  printf("test_verify_rows_cpu: all passed\n");
  return 0;
}
