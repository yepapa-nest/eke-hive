#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""GlmExperts step bookkeeping (the routed-expert cache of the GLM and Qwen families) — CPU checks (no GPU).

The cache-policy functions of engine/src/glm/glm_experts.cpp (after_step, promote_key, commit_ready, next_victim, prefetch,
warm_from_keys, lend_tail_slots, return_tail_slots) and note_used of glm_experts.h are sliced VERBATIM from the production
sources into a small class with the same members, on the fake CUDA runtime (tools/cpu_fake/include — copies are host memcpys,
a promotion stays "in flight" until the next commit_ready).

* after_step == its eager definition (REF_AFTER_STEP below: the victim candidates and the prefetch victim list built on every
  step — the code before 2026-11-11). Random worlds (cache smaller than / equal to / larger than the key count; routing with
  misses, prefetch, tail lending, warm quota, warm starts): after every operation the two caches hold the same slot map,
  scores (bit for bit), pending slots, promotions in flight, counters, warm quota and return values. The lazy paths must have
  been taken (steps without a candidate, steps with every key resident) and so must the eager ones.
* warm_from_keys with a promotion in flight: every key ends in exactly one slot (slot_of_ / key_of_slot_ stay a bijection,
  n_resident == keys with a slot), slot bytes == the key's host record. Negative control: the body without the in-flight
  test (the code before 2026-11-11) MUST leave a key in two slots.
* Negative control for the differential: an after_step that skips the prefetch victim list whenever the step had no candidate
  (instead of: every key resident) MUST diverge from the definition.
* Text contracts: every function that takes a key out of the cache outside after_step / prefetch clears the victim list.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'engine/src/glm/glm_experts.cpp'
HDR = ROOT / 'engine/include/hive/glm/glm_experts.h'
FAKE = ROOT / 'tools/cpu_fake/include'

# The definition: after_step as it was before 2026-11-11 (both lists built on every step).
REF_AFTER_STEP = r'''
void GlmExperts::after_step(int tokens) {
  for (auto& s : score_) s *= 0.97f;
  commit_ready();
  std::sort(step_used_.begin(), step_used_.end());
  step_used_.erase(std::unique(step_used_.begin(), step_used_.end()), step_used_.end());
  std::sort(step_used_.begin(), step_used_.end(), [&](int a, int b) { return score_[a] > score_[b]; });
  int budget = promote_per_step_ * std::max(1, tokens) + (warm_left_ > 0 ? std::min(warm_left_, warm_per_step_) : 0);
  vic_.clear();
  for (int s2 = 0; s2 < n_slots_; ++s2) if (!pending_[s2]) vic_.push_back(s2);
  auto vsc = [&](int s2) { const int k2 = key_of_slot_[s2]; return k2 < 0 ? -1.f : score_[k2]; };
  const size_t vk = std::min<size_t>(vic_.size(), (size_t)std::max(0, budget));
  std::partial_sort(vic_.begin(), vic_.begin() + vk, vic_.end(), [&](int a, int b) { const float x = vsc(a), y = vsc(b); return x < y || (x == y && a < b); });
  vic_.resize(vk);
  size_t vp = 0;
  int issued = 0;
  for (int k : step_used_) { if (issued >= budget) break; const uint64_t before = stats_.promoted; promote_key(k, vic_, vp); issued += stats_.promoted > before; }
  if (warm_left_ > 0 && issued < budget) {
    std::vector<int> cand;
    for (int k = 0; k < n_keys(); ++k) if (slot_of_[k] < 0 && score_[k] >= 1.f) cand.push_back(k);
    const int need = budget - issued;
    if ((int)cand.size() > need) std::partial_sort(cand.begin(), cand.begin() + need, cand.end(), [&](int a, int b) { return score_[a] > score_[b]; });
    int n = 0;
    for (int k : cand) { if (n >= need) break; const uint64_t before = stats_.promoted; promote_key(k, vic_, vp); n += stats_.promoted > before; }
    warm_left_ = n == 0 ? 0 : warm_left_ - n;
  }
  step_used_.clear();
  victims_.resize(n_slots_);
  for (int s2 = 0; s2 < n_slots_; ++s2) victims_[s2] = s2;
  auto sc = [&](int s2) { const int k2 = key_of_slot_[s2]; return k2 < 0 ? -1.f : score_[k2]; };
  const size_t keep = std::min<size_t>(512, victims_.size());
  std::partial_sort(victims_.begin(), victims_.begin() + keep, victims_.end(), [&](int a, int b) { return sc(a) < sc(b); });
  victims_.resize(keep);
  vpos_ = 0;
}
'''

HARNESS = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <random>
#include <set>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { if ((x) != cudaSuccess) { fprintf(stderr, "CUDA_CHECK failed: %s\n", #x); abort(); } } while (0)
#define HIVE_CHECK(c, m) do { if (!(c)) { fprintf(stderr, "HIVE_CHECK failed: %s\n", #c); abort(); } } while (0)

struct GlmCacheStats { uint64_t routed = 0, hit = 0, cpu = 0, streamed = 0, promoted = 0, evicted = 0, prefetched = 0, skipped = 0, deferred = 0; };
struct RecLayout { size_t total = 64; };

// The members the sliced functions use, as in hive/glm/glm_experts.h.
#define DECL_CLASS(NAME)                                                                                                         \
  struct NAME {                                                                                                                  \
    int E_ = 0, K_ = 0, n_moe_ = 0;                                                                                              \
    RecLayout L_;                                                                                                                \
    uint8_t* host_ = nullptr;                                                                                                    \
    uint8_t* slots_ = nullptr; int n_slots_ = 0; int lent_ = 0;                                                                  \
    std::vector<int> slot_of_, key_of_slot_;                                                                                     \
    std::vector<float> score_;                                                                                                   \
    std::vector<uint8_t> pending_;                                                                                               \
    struct Promo { int slot, key; cudaEvent_t ev; };                                                                             \
    std::deque<Promo> promos_;                                                                                                   \
    std::vector<cudaEvent_t> ev_pool_;                                                                                           \
    cudaStream_t side_ = nullptr;                                                                                                \
    cudaEvent_t pf_ev_ = nullptr;                                                                                                \
    std::vector<int> victims_; size_t vpos_ = 0;                                                                                 \
    std::vector<int> vic_;                                                                                                       \
    int promote_per_step_ = 8, warm_left_ = 0, warm_per_step_ = 32;                                                              \
    std::vector<int> step_used_;                                                                                                 \
    GlmCacheStats stats_;                                                                                                        \
    int n_keys() const { return n_moe_ * E_; }                                                                                   \
    const uint8_t* host_rec(int key) const { return host_ + (size_t)key * L_.total; }                                            \
    void warm_quota(int n) { warm_left_ = n; }                                                                                   \
    int n_resident() const { int n = 0; for (int k : key_of_slot_) n += k >= 0; return n; }                                      \
    void init(int layers, int experts, int k, int slots, int per_step) {                                                         \
      n_moe_ = layers; E_ = experts; K_ = k; n_slots_ = slots; promote_per_step_ = per_step;                                      \
      host_ = (uint8_t*)malloc((size_t)n_keys() * L_.total);                                                                     \
      for (int key = 0; key < n_keys(); ++key) for (size_t b = 0; b < L_.total; ++b) host_[(size_t)key * L_.total + b] = (uint8_t)(key * 131 + b * 7 + 3); \
      slots_ = (uint8_t*)malloc((size_t)std::max(1, slots) * L_.total);                                                          \
      memset(slots_, 0xEE, (size_t)std::max(1, slots) * L_.total);                                                               \
      slot_of_.assign(n_keys(), -1); key_of_slot_.assign(slots, -1); pending_.assign(slots, 0); score_.assign(n_keys(), 0.f);     \
      CUDA_CHECK(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking));                                                      \
    }                                                                                                                            \
    NOTE_USED                                                                                                                    \
    void after_step(int tokens = 1);                                                                                             \
    int warm_from_keys(const std::vector<int32_t>& keys);                                                                        \
    void commit_ready();                                                                                                         \
    void promote_key(int key, const std::vector<int>& victims, size_t& next);                                                    \
    int next_victim(int target_layer);                                                                                           \
    int prefetch(int li, const int32_t* ids, const float* w, int rows, int max_n, cudaStream_t compute);                         \
    uint8_t* lend_tail_slots(int n);                                                                                             \
    void return_tail_slots();                                                                                                    \
  };

@@CLASSES@@

template <class A, class B> bool same(const A& a, const B& b, const char* where, long step) {
  bool ok = a.slot_of_ == b.slot_of_ && a.key_of_slot_ == b.key_of_slot_ && a.pending_ == b.pending_ && a.step_used_ == b.step_used_ &&
            a.warm_left_ == b.warm_left_ && a.n_slots_ == b.n_slots_ && a.lent_ == b.lent_ && a.promos_.size() == b.promos_.size() &&
            a.score_.size() == b.score_.size() && memcmp(a.score_.data(), b.score_.data(), a.score_.size() * 4) == 0 &&
            a.stats_.routed == b.stats_.routed && a.stats_.hit == b.stats_.hit && a.stats_.streamed == b.stats_.streamed &&
            a.stats_.promoted == b.stats_.promoted && a.stats_.evicted == b.stats_.evicted && a.stats_.prefetched == b.stats_.prefetched;
  for (size_t i = 0; ok && i < a.promos_.size(); ++i) ok = a.promos_[i].slot == b.promos_[i].slot && a.promos_[i].key == b.promos_[i].key;
  if (!ok) fprintf(stderr, "  state differs after %s (operation %ld)\n", where, step);
  return ok;
}

// every key in at most one slot, the two maps agree, resident slots hold their key's record
template <class A> bool consistent(const A& a, const char* where) {
  std::set<int> seen;
  int with_slot = 0;
  for (int k = 0; k < a.n_keys(); ++k) with_slot += a.slot_of_[k] >= 0;
  bool ok = true;
  for (int s = 0; s < (int)a.key_of_slot_.size(); ++s) {
    const int k = a.key_of_slot_[s];
    if (k < 0) continue;
    if (!seen.insert(k).second) { fprintf(stderr, "  %s: key %d sits in two slots\n", where, k); ok = false; }
    if (a.slot_of_[k] != s) { fprintf(stderr, "  %s: slot %d holds key %d but slot_of_ says %d\n", where, s, k, a.slot_of_[k]); ok = false; }
    if (s < a.n_slots_ && memcmp(a.slots_ + (size_t)s * a.L_.total, a.host_rec(k), a.L_.total) != 0) { fprintf(stderr, "  %s: slot %d bytes are not key %d's record\n", where, s, k); ok = false; }
  }
  if (a.n_resident() != with_slot) { fprintf(stderr, "  %s: n_resident %d but %d keys have a slot\n", where, a.n_resident(), with_slot); ok = false; }
  return ok;
}

struct World { int layers, experts, k, slots, per_step; const char* name; };

// Random world on the pair (A = production after_step, B = the definition). Returns false on the first difference.
template <class A, class B> bool run_world(const World& w, unsigned seed, long ops, long* lazy_vic, long* eager_vic, long* lazy_victims, long* eager_victims) {
  A a; B b;
  a.init(w.layers, w.experts, w.k, w.slots, w.per_step); b.init(w.layers, w.experts, w.k, w.slots, w.per_step);
  std::mt19937 rng(seed);
  auto rnd = [&](int n) { return (int)(rng() % (unsigned)std::max(1, n)); };
  const int NK = a.n_keys();
  int hot = std::max(2, w.experts / 3);   // routing locality: mostly the first `hot` experts of a layer
  for (long step = 0; step < ops; ++step) {
    const int op = rnd(100);
    if (op < 38) {   // a routed step: note_used on a few layers, then after_step
      const int rows = 1 + rnd(4);
      for (int li = 0; li < w.layers; ++li) {
        if (rnd(4) == 0) continue;
        std::vector<int32_t> ids((size_t)rows * w.k);
        for (auto& x : ids) x = rnd(10) < 8 ? rnd(hot) : rnd(w.experts);
        if (rnd(16) == 0) ids[0] = -1;
        a.note_used(li, ids.data(), (int)ids.size()); b.note_used(li, ids.data(), (int)ids.size());
      }
      const bool cand = !a.step_used_.empty() || a.warm_left_ > 0;
      a.after_step(rows); b.after_step(rows);
      // which paths the production code took: the candidate list is built only on need; the prefetch list is dropped when every key is resident
      bool all_res = true;
      for (int k = 0; k < NK; ++k) all_res = all_res && a.slot_of_[k] >= 0;
      ++*(cand ? eager_vic : lazy_vic);
      ++*(all_res ? lazy_victims : eager_victims);
      if (all_res && !a.victims_.empty()) { fprintf(stderr, "  every key resident but the prefetch victim list was built\n"); return false; }
      if (!same(a, b, "after_step", step)) return false;
    } else if (op < 52) {   // a device-routed step: nothing used
      const int t = 1 + rnd(5);
      const bool cand = a.warm_left_ > 0;
      a.after_step(t); b.after_step(t);
      bool all_res = true;
      for (int k = 0; k < NK; ++k) all_res = all_res && a.slot_of_[k] >= 0;
      ++*(cand ? eager_vic : lazy_vic);
      ++*(all_res ? lazy_victims : eager_victims);
      if (!same(a, b, "after_step (nothing used)", step)) return false;
    } else if (op < 72) {   // prediction prefetch for one layer
      const int li = rnd(w.layers), rows = 1 + rnd(3), max_n = rnd(5);
      std::vector<int32_t> ids((size_t)rows * w.k);
      std::vector<float> wt(ids.size());
      for (auto& x : ids) x = rnd(10) < 7 ? rnd(hot) : rnd(w.experts);
      for (auto& x : wt) x = (float)(1 + rnd(1000)) / 1000.f;
      const int na = a.prefetch(li, ids.data(), wt.data(), rows, max_n, nullptr), nb = b.prefetch(li, ids.data(), wt.data(), rows, max_n, nullptr);
      if (na != nb) { fprintf(stderr, "  prefetch issued %d vs %d (operation %ld)\n", na, nb, step); return false; }
      if (!same(a, b, "prefetch", step)) return false;
    } else if (op < 80) {   // the tail lent to a prefill and returned (keys leave the cache outside after_step)
      if (a.lent_ == 0 && a.n_slots_ > 2) { const int n = 1 + rnd(std::max(1, a.n_slots_ / 3)); a.lend_tail_slots(n); b.lend_tail_slots(n); }
      else { a.return_tail_slots(); b.return_tail_slots(); }
      if (!same(a, b, "lend / return", step)) return false;
    } else if (op < 86) {
      const int q = rnd(3) == 0 ? 0 : 1 + rnd(64);
      a.warm_quota(q); b.warm_quota(q);
    } else if (op < 94) {   // warm start from a key list (missing keys in some order, duplicates and bad ids included)
      std::vector<int32_t> keys;
      for (int k = 0; k < NK; ++k) if (a.slot_of_[k] < 0 && rnd(3) != 0) keys.push_back(k);
      std::shuffle(keys.begin(), keys.end(), rng);
      if (!keys.empty() && rnd(2)) keys.push_back(keys[0]);
      if (rnd(4) == 0) { keys.push_back(-1); keys.push_back(NK + 3); }
      const int na = a.warm_from_keys(keys), nb = b.warm_from_keys(keys);
      if (na != nb) { fprintf(stderr, "  warm_from_keys loaded %d vs %d (operation %ld)\n", na, nb, step); return false; }
      if (!same(a, b, "warm_from_keys", step)) return false;
      if (!consistent(a, "after warm_from_keys")) return false;
    } else {   // the routing moves
      hot = std::max(2, 1 + rnd(w.experts));
    }
    if ((step & 63) == 0 && !consistent(a, "periodic check")) return false;
  }
  return consistent(a, "end of world") && consistent(b, "end of world (definition)");
}

// A promotion in flight when a warm start lists its key: afterwards every key sits in exactly one slot.
template <class A> bool warm_in_flight() {
  A a;
  a.init(2, 8, 2, 16, 4);   // 16 keys, 16 slots
  std::vector<int32_t> first;
  for (int k = 0; k < 12; ++k) first.push_back(k);
  a.warm_from_keys(first);                       // 12 resident, 4 free slots
  const int32_t ids[2] = {4, 5};                 // layer 1, experts 4 and 5 = keys 12 and 13: misses
  a.note_used(1, ids, 2);
  a.after_step(1);                               // promotions issued: in flight until the next commit_ready
  if (a.promos_.size() != 2) { fprintf(stderr, "  setup: expected 2 promotions in flight, have %zu\n", a.promos_.size()); return false; }
  std::vector<int32_t> miss;
  for (int k = 0; k < a.n_keys(); ++k) if (a.slot_of_[k] < 0) miss.push_back(k);   // 12..15, two of them in flight
  a.warm_from_keys(miss);
  a.after_step(1);
  bool ok = consistent(a, "warm start with promotions in flight");
  for (int k = 0; k < a.n_keys(); ++k) if (a.slot_of_[k] < 0) { fprintf(stderr, "  key %d is not resident after the warm start (16 slots for 16 keys)\n", k); ok = false; }
  return ok;
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "all";
  if (!strcmp(mode, "warm")) return warm_in_flight<New>() ? 0 : 1;
  const World worlds[] = {
      {3, 12, 3, 10, 2, "cache smaller than the keys"},
      {2, 8, 2, 16, 4, "cache = keys"},
      {2, 8, 2, 24, 3, "cache larger than the keys"},
      {4, 16, 4, 64, 8, "cache = keys, wider"},
      {1, 700, 4, 600, 8, "one layer, more than 512 slots"},
  };
  long lazy_vic = 0, eager_vic = 0, lazy_victims = 0, eager_victims = 0;
  for (const World& w : worlds)
    for (unsigned seed = 1; seed <= 6; ++seed)
      if (!run_world<New, Ref>(w, seed * 7919u, w.experts > 100 ? 1500 : 6000, &lazy_vic, &eager_vic, &lazy_victims, &eager_victims)) {
        fprintf(stderr, "FAIL: world \"%s\" seed %u\n", w.name, seed);
        return 1;
      }
  printf("after_step == its eager definition over %zu worlds x 6 seeds: steps without a promotion candidate %ld (with %ld) · steps ending with every key resident %ld (not %ld)\n",
         sizeof(worlds) / sizeof(worlds[0]), lazy_vic, eager_vic, lazy_victims, eager_victims);
  if (lazy_vic < 100 || eager_vic < 100 || lazy_victims < 100 || eager_victims < 100) { fprintf(stderr, "FAIL: a path was not exercised\n"); return 1; }
  if (!warm_in_flight<New>()) { fprintf(stderr, "FAIL: warm start with promotions in flight\n"); return 1; }
  printf("warm start with promotions in flight: every key in exactly one slot\n");
  return 0;
}
'''


def slice_fn(text, start):
    """One top-level function body: from its signature line to the first line that is just '}'."""
    i = text.index(start)
    j = text.index('\n}\n', i)
    return text[i:j + 3]


def slice_note_used(hdr):
    i = hdr.index('  void note_used(int li, const int32_t* ids, int n) {')
    j = hdr.index('\n  }\n', i)
    return hdr[i:j + 5]


SLICED = ['int GlmExperts::warm_from_keys(', 'void GlmExperts::commit_ready() {', 'void GlmExperts::promote_key(', 'int GlmExperts::next_victim(',
          'int GlmExperts::prefetch(', 'void GlmExperts::after_step(', 'uint8_t* GlmExperts::lend_tail_slots(', 'void GlmExperts::return_tail_slots() {']


def program(src, hdr, new_after_step=None, new_warm=None):
    fns = {s: slice_fn(src, s) for s in SLICED}
    if new_after_step is not None:
        fns['void GlmExperts::after_step('] = new_after_step
    if new_warm is not None:
        fns['int GlmExperts::warm_from_keys('] = new_warm
    new = '\n'.join(fns.values()).replace('GlmExperts::', 'New::')
    ref_fns = dict(fns)
    ref_fns['void GlmExperts::after_step('] = REF_AFTER_STEP
    ref = '\n'.join(ref_fns.values()).replace('GlmExperts::', 'Ref::')
    note = ' \\\n'.join(slice_note_used(hdr).rstrip('\n').split('\n')) + ' \\'
    harness = HARNESS.replace('    NOTE_USED                                                                                                                    \\', note)
    assert harness != HARNESS, 'note_used was not placed'
    return harness.replace('@@CLASSES@@', 'DECL_CLASS(New)\nDECL_CLASS(Ref)\n' + new + '\n' + ref)


def build(td, name, code):
    cpp = Path(td) / (name + '.cpp')
    cpp.write_text(code)
    exe = Path(td) / name
    subprocess.run(['g++', '-std=c++20', '-O1', '-pthread', '-I' + str(FAKE), str(cpp), '-o', str(exe)], check=True)
    return exe


def contracts(src):
    # keys leave the cache outside after_step / prefetch only here, and each of these drops the prefetch victim list — what lets
    #   after_step skip building it while every key is resident
    for start in ('size_t GlmExperts::shrink_cache(', 'uint8_t* GlmExperts::lend_tail_slots(', 'size_t GlmExperts::sleep_release() {'):
        body = slice_fn(src, start)
        assert 'victims_.clear()' in body and 'vpos_ = 0' in body, start + ' must clear the prefetch victim list'
    writers = [ln for ln in src.split('\n') if ('slot_of_[' in ln and '= -1' in ln)]
    assert len(writers) == 5, 'a new place takes keys out of the cache (%d lines set slot_of_ to -1): check that it clears victims_ or runs in after_step / prefetch, then update this count' % len(writers)
    pf = slice_fn(src, 'int GlmExperts::prefetch(')
    assert pf.index('if (slot_of_[key] >= 0) continue;') < pf.index('if (cand.empty()) return 0;') < pf.index('next_victim(li)'), 'prefetch must return before next_victim when every predicted key is resident'
    assert src.count('= next_victim(') == 1 and '= next_victim(li)' in pf, 'next_victim has a caller other than prefetch'
    step = slice_fn(src, 'void GlmExperts::after_step(')
    assert step.count('need_vic(); const uint64_t before = stats_.promoted; promote_key(k, vic_, vp);') == 2 and step.count('promote_key(') == 2, \
        'after_step: every promote_key call must build the candidate list first (need_vic)'


def main():
    src, hdr = SRC.read_text(), HDR.read_text()
    contracts(src)
    with tempfile.TemporaryDirectory() as td:
        exe = build(td, 'glm_experts_step', program(src, hdr))
        subprocess.run([str(exe)], check=True)
        # negative control 1: warm_from_keys without the in-flight test leaves a key in two slots
        warm = slice_fn(src, 'int GlmExperts::warm_from_keys(')
        needle = '    if (!promos_.empty() && in_flight(k)) continue;\n'
        assert needle in warm, 'warm_from_keys no longer skips keys whose promotion is in flight'
        mut = build(td, 'mut_warm', program(src, hdr, new_warm=warm.replace(needle, '')))
        r = subprocess.run([str(mut), 'warm'], capture_output=True, text=True)
        assert r.returncode != 0 and 'sits in two slots' in r.stderr, 'negative control (warm start without the in-flight test) was not detected:\n' + r.stderr
        # negative control 2: skipping the prefetch victim list on another condition than "every key resident" diverges
        step = slice_fn(src, 'void GlmExperts::after_step(')
        needle = '    if (any_missing >= 0) { victims_.clear(); vpos_ = 0; return; }\n'
        assert needle in step, 'after_step: the every-key-resident test changed'
        mut = build(td, 'mut_step', program(src, hdr, new_after_step=step.replace(needle, '    if (!vic_built) { victims_.clear(); vpos_ = 0; return; }\n')))
        r = subprocess.run([str(mut)], capture_output=True, text=True)
        assert r.returncode != 0 and 'differs' in r.stderr + r.stdout, 'negative control (victim list skipped without the residency test) was not detected:\n' + r.stderr
    print('GlmExperts step bookkeeping: differential + warm start + 2 negative controls + text contracts passed')


if __name__ == '__main__':
    sys.exit(main())
