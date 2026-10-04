#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""D4 cache replayer (tools/cache_replay.py) fidelity + HIVE_CACHE_POLICY tests — no GPU.

1) Call-convention check: the store calls of Runtime/hived that the replayer (tools/cache_replay.cpp) imitates (observation weight
   formula · row argument · promotion points · warm condition · set_row_owners sites) must appear verbatim in runtime.cpp / hived.cpp
   — if the source changes this breaks, forcing the replayer to be updated with it.
2) Independent re-implementation: the default policy (score observation · decay 0.97 · promote (at most 8, threshold 2,
   lowest-score victim, down to the libstdc++ partial_sort tie order) · commit on the next step · warm_cache) is written separately
   in Python, and on synthetic traces (batched decode · verification · drafts · short/long prompts · request boundaries) it must
   match the replayer **exactly in per-row hits, promotions and final residents** (phase-score 0/1 · two slot counts).
3) Negative control: building with a mutation of the promote threshold in expert_store.cpp changes the result (evidence that the
   replayer runs that file's code).
4) Policy unset = default: the JSON of a replayer built from HEAD's expert_store (before the policy) equals the working-tree
   version (only when HEAD has no policy block).
   Switch decision: unset · "" · "0" · unknown name = the same decisions as the default · "seq" = policy.
5) seq policy unit test: on a real ExpertStore (fake CUDA), the prio = Σ_o (c+α·p)/(n+α) values · reset on a new request after a
   prompt · retirement after idle · argument parsing.
"""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'tools')]
import cache_replay  # noqa: E402

RT = (ROOT / 'engine/src/runtime.cpp').read_text()
HIVED = (ROOT / 'engine/src/hived.cpp').read_text()


def fn_body(text: str, sig: str) -> str:
    i = text.index(sig)
    j = text.index('\n}\n', i)
    return text[i:j]


def contracts():
    dec = fn_body(RT, 'void Runtime::moe_decode_experts(')
    assert 'const float score_weight = opt_.phase_score && score_prefill_ ? 1.f / std::max(1, M) : 1.f;' in dec
    assert 'store_.observe(l, w.route_ids_h[i], score_weight, i / k);' in dec
    assert 'store_.touch(slot)' in dec and dec.count('store_.observe(') == 1
    assert 'if (opt_.promote_misses > 0) step_miss_.push_back(l * c.n_routed + e);' in dec
    mul = fn_body(RT, 'void Runtime::moe_experts_multi(')
    assert 'const float score_weight = opt_.phase_score ? 1.f / std::max(1, total_rows) : 1.f;' in mul
    assert 'store_.observe(l, e, score_weight, S == 1 ? 0 : -1);' in mul and mul.count('store_.observe(') == 1
    assert 'const uint64_t lru_stamp = store_.lru_stamp();' in mul and 'store_.touch_at(slot, lru_stamp);' in mul
    # score_prefill_: forward() = !verify_ · forward_multi = true · otherwise (forward_batch, drafts) false (restored by the destructor)
    #   T11: layer_yield_point hands off and restores it inside the outer prefill forward (inside the yield = off · restored = the outer value) — its body is excluded from the count
    ly = fn_body(RT, 'void Runtime::layer_yield_point(')
    assert 'score_prefill_ = false;' in ly and 'score_prefill_ = P.score_prefill;' in ly
    assert len(re.findall(r'score_prefill_ = ', RT.replace(ly, ''))) == 2
    fwd = fn_body(RT, 'int32_t Runtime::forward(')
    assert 'score_prefill_ = !verify_;' in fwd and 'xtrace_step(verify_ ? 2 : 1, M, &sp, 1);' in fwd and 'store_.commit_pending();' in fwd
    assert 'if (M_orig < opt_.prefill_threshold && (opt_.promote_per_token > 0 || opt_.promote_misses > 0)) {' in fwd
    assert '} else step_miss_.clear();' in fwd
    bat = fn_body(RT, 'void Runtime::forward_batch(')
    assert 'xtrace_step(0, M, seqs.data(), M);' in bat and 'store_.commit_pending();' in bat
    assert 'if (opt_.promote_per_token > 0 || opt_.promote_misses > 0) {\n    int n = promote_after_step();' in bat
    mul_fwd = fn_body(RT, 'void Runtime::forward_multi(')
    assert 'score_prefill_ = true;' in mul_fwd and 'xtrace_step(1, p.M, &sp, 1);' in mul_fwd and 'promote_after_step' not in mul_fwd
    drf = fn_body(RT, 'void Runtime::mtp_draft(')
    assert 'xtrace_step(3, model_.cfg().dspark_block, &sp, 1);' in drf and 'promote_after_step' not in drf
    assert 'if (keep && std::uncaught_exceptions() == ex) *draft = std::move(*v);' in drf
    xs = fn_body(RT, 'void Runtime::xtrace_step(')
    assert xs.index('store_.set_row_owners(u, m, kind == 1);') < xs.index('if (!xtrace_) return;')
    assert HIVED.count('if (ids.size() - common >= 64 && store.n_resident() < store.n_slots() * 9 / 10) {') == 1
    assert 'const int n = rt.warm_from_keys(keys);' in HIVED
    print('contracts: runtime.cpp/hived.cpp store call order == tools/cache_replay.cpp OK')


# ---------------- synthetic traces ----------------
def write_trace(path: Path, seed: int = 7, n_req: int = 14, E_used: int = 28) -> None:
    rng = np.random.default_rng(seed)
    out = bytearray()
    L, K = 40, 6

    def S(kind, M, tags):
        nonlocal out
        out += struct.pack('<BBHHd', 0x53, kind, M, len(tags), 0.0)
        for u, p in tags: out += struct.pack('<II', u, p)

    def Lrec(l, k, ids):
        nonlocal out
        out += struct.pack('<BBHH', 0x4C, l, k, len(ids) // k) + np.asarray(ids, '<u2').tobytes()

    def Hrec(l, M, cnt):
        nonlocal out
        out += struct.pack('<BBHI', 0x48, l, len(cnt), M) + np.asarray(cnt, '<u2').tobytes()

    def route(pref, M, k, E):
        rows = []
        for _ in range(M):
            p = pref / pref.sum()
            rows.extend(rng.choice(E, size=k, replace=False, p=p).tolist())
        return rows

    active = []  # [uid, pos, left, pref[L,E], dpref[3,16]]
    uid = 0
    pending = n_req
    while pending or active:
        if pending and (len(active) < 3 and rng.random() < 0.3 or not active):
            uid += 1; pending -= 1
            pref = rng.gamma(0.3, 1.0, (L, E_used)) + 0.02
            dpref = rng.gamma(0.5, 1.0, (3, 16)) + 0.05
            M = int(rng.integers(6, 60)) if rng.random() < 0.75 else 1100
            pos = 0
            for c0 in range(0, M, 400 if M > 1000 else M):  # long prompts are several chunks (H) · short ones a single L
                m = min(400 if M > 1000 else M, M - c0)
                if M > 1000:
                    S(1, m, [(uid, pos)])
                    for l in range(L):
                        cnt = np.bincount(route(pref[l], m, K, E_used), minlength=384)
                        Hrec(l, m, cnt)
                else:
                    S(1, m, [(uid, pos)])
                    for l in range(L): Lrec(l, K, route(pref[l], m, K, E_used))
                pos += m
            active.append([uid, pos, int(rng.integers(30, 120)), pref, dpref])
        r = rng.random()
        if r < 0.15 and active:  # draft + verification (one sequence)
            a = active[int(rng.integers(len(active)))]
            S(3, 5, [(a[0], a[1])])
            for l in range(3): Lrec(40 + l, 3, route(a[4][l], 5, 3, 16))
            mv = int(rng.integers(2, 6))
            S(2, mv, [(a[0], a[1])])
            for l in range(L): Lrec(l, K, route(a[3][l], mv, K, E_used))
            a[1] += mv; a[2] -= mv
        elif active:  # batched decode (all active)
            S(0, len(active), [(a[0], a[1]) for a in active])
            for l in range(L):
                ids = []
                for a in active: ids.extend(route(a[3][l], 1, K, E_used))
                Lrec(l, K, ids)
            for a in active: a[1] += 1; a[2] -= 1
        active = [a for a in active if a[2] > 0]
    out += bytes([0x53, 0, 1])  # truncated final record (daemon exit) — the replayer uses only the preceding part
    path.write_bytes(bytes(out))


def parse_py(path: Path):
    b = path.read_bytes(); off = 0; steps = []
    while off < len(b):
        t = b[off]
        if t == 0x53:
            if off + 14 > len(b): break
            kind = b[off + 1]; M, nt = struct.unpack_from('<HH', b, off + 2)
            tags = [struct.unpack_from('<II', b, off + 14 + 8 * i) for i in range(nt)]
            steps.append({'kind': kind, 'M': M, 'uids': [u for u, _ in tags], 'L': [], 'H': []}); off += 14 + 8 * nt
        elif t == 0x4C:
            l = b[off + 1]; k, M = struct.unpack_from('<HH', b, off + 2)
            steps[-1]['L'].append((l, k, M, np.frombuffer(b, '<u2', M * k, off + 6).astype(np.int64))); off += 6 + 2 * M * k
        elif t == 0x48:
            l = b[off + 1]; (E,) = struct.unpack_from('<H', b, off + 2); (M,) = struct.unpack_from('<I', b, off + 4)
            steps[-1]['H'].append((l, M, np.frombuffer(b, '<u2', E, off + 8).astype(np.int64))); off += 8 + 2 * E
        else: break
    while steps:
        s = steps[-1]; nl = len(s['L']) + len(s['H'])
        if (s['kind'] == 3 and nl >= 1) or nl >= 40 or (s['kind'] == 1 and nl == 0): break
        steps.pop()
    return steps


# libstdc++ std::partial_sort(first, middle, last, comp) — identical down to tie order (heap_select + sort_heap)
def _adjust_heap(a, first, hole, length, value, comp):
    top = hole; child = hole
    while child < (length - 1) // 2:
        child = 2 * (child + 1)
        if comp(a[first + child], a[first + child - 1]): child -= 1
        a[first + hole] = a[first + child]; hole = child
    if (length & 1) == 0 and child == (length - 2) // 2:
        child = 2 * (child + 1)
        a[first + hole] = a[first + child - 1]; hole = child - 1
    parent = (hole - 1) // 2
    while hole > top and comp(a[first + parent], value):
        a[first + hole] = a[first + parent]; hole = parent; parent = (hole - 1) // 2
    a[first + hole] = value


def partial_sort(a, mid, comp):
    n = mid
    if n >= 2:
        parent = (n - 2) // 2
        while True:
            _adjust_heap(a, 0, parent, n, a[parent], comp)
            if parent == 0: break
            parent -= 1
    for i in range(mid, len(a)):
        if comp(a[i], a[0]):
            v = a[i]; a[i] = a[0]; _adjust_heap(a, 0, 0, n, v, comp)
    while n > 1:
        n -= 1
        v = a[n]; a[n] = a[0]; _adjust_heap(a, 0, 0, n, v, comp)


class Mirror:
    """Independent re-implementation of the default policy (spec: expert_store.h/.cpp comments + runtime promote_after_step/warm_cache + hived warm condition)."""
    def __init__(self, slots, promote=8, phase=True):
        self.K = 43 * 384; self.slots = slots; self.promote_n = promote; self.phase = phase
        self.score = np.zeros(self.K, np.float32)
        self.slot_of = np.full(self.K, -1, np.int64); self.key_of = [-1] * slots
        self.pending = [-1] * slots; self.pending_key = {}; self.batches = []
        self.promotions = 0

    def commit(self):
        for b in self.batches:
            for s in b:
                k = self.pending[s]
                self.key_of[s] = k; self.slot_of[k] = s; self.pending[s] = -1; self.pending_key.pop(k, None)
        self.batches = []

    def promote(self, max_n, min_score):
        max_n = min(max_n, max(1, self.slots // 4))
        if sum(1 for p in self.pending if p >= 0) >= 2 * max_n: return 0
        cand = [(float(self.score[k]), k) for k in np.flatnonzero((self.slot_of < 0) & (self.score >= np.float32(min_score))).tolist()
                if k not in self.pending_key]
        if not cand: return 0
        partial_sort(cand, min(max_n, len(cand)), lambda a, b: a[0] > b[0])
        batch = []
        for sc_c, key in cand:
            if len(batch) >= max_n: break
            victim, vs = -1, 1e30
            for s in range(self.slots):
                if self.pending[s] >= 0: continue
                kk = self.key_of[s]; sc = -1.0 if kk < 0 else float(self.score[kk])
                if sc < vs: vs, victim = sc, s
            if victim < 0 or vs >= sc_c: break
            old = self.key_of[victim]
            if old >= 0 and self.slot_of[old] == victim: self.slot_of[old] = -1
            self.key_of[victim] = -1
            self.pending[victim] = key; self.pending_key[key] = victim; batch.append(victim)
        if batch: self.batches.append(batch); self.promotions += len(batch)
        return len(batch)

    def warm(self, cap=2048):
        total = 0
        for _ in range(64):
            if total >= cap: break
            n = self.promote(min(max(1, self.slots // 4), cap - total), 1.0)
            if n == 0: break
            self.commit(); total += n
        return total

    def run(self, steps):
        # hived warm_after_prefill: right after a run of consecutive kind-1 steps of one uid (sum M ≥ 64) ends, if residents < 90%
        warm_after = [False] * len(steps); last, runm = {}, {}
        for i, s in enumerate(steps):
            for u in s['uids']:
                if s['kind'] == 1: last[u] = i; runm[u] = runm.get(u, 0) + s['M']
                elif last.get(u, -1) >= 0:
                    if runm[u] >= 64: warm_after[last[u]] = True
                    last[u] = -1; runm[u] = 0
        acc = {k: [0, 0] for k in range(4)}
        for i, s in enumerate(steps):
            self.commit()
            for l, k, M, ids in s['L']:
                w = np.float32(1.0) / np.float32(max(1, M)) if (self.phase and s['kind'] == 1) else np.float32(1.0)
                for e in ids.tolist(): self.score[l * 384 + e] += w
                for e, c in zip(*np.unique(ids, return_counts=True)):
                    acc[s['kind']][0] += int(c)
                    if self.slot_of[l * 384 + e] >= 0: acc[s['kind']][1] += int(c)
            j = 0
            while j < len(s['H']):
                j1 = j; tot = 0
                while j1 < len(s['H']) and s['H'][j1][0] == s['H'][j][0]: tot += s['H'][j1][1]; j1 += 1
                w = np.float32(1.0) / np.float32(max(1, tot)) if self.phase else np.float32(1.0)
                for q in range(j, j1):
                    l, _, cnt = s['H'][q]
                    for e in np.flatnonzero(cnt).tolist():
                        for _ in range(int(cnt[e])): self.score[l * 384 + e] += w
                j = j1
            if s['kind'] != 3 and s['L'] and self.promote_n > 0:
                self.score *= np.float32(0.97)
                self.promote(self.promote_n, 2.0)
            if warm_after[i] and sum(1 for k in self.key_of if k >= 0) < self.slots * 9 // 10: self.warm()
        self.commit()
        return acc, sum(1 for k in self.key_of if k >= 0)


def run_exe(exe, trace, env_policy=None, args=()):
    env = dict(os.environ); env.pop('HIVE_CACHE_POLICY', None); env.pop('HIVE_CACHE_EVENTS', None)
    if env_policy is not None: env['HIVE_CACHE_POLICY'] = env_policy
    p = subprocess.run([str(exe), str(trace), *args], env=env, capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1]), p.stderr


UNIT = r'''
#include "hive/expert_store.h"
#include <cassert>
#include <cmath>
#include <cstdio>
using namespace hive;
static bool near(float a, float b) { return std::fabs(a - b) <= 1e-5f * std::max(1.f, std::fabs(b)); }
int main() {
  Config cfg; cfg.dim = 64; cfg.moe_inter = 64; cfg.n_routed = 384; cfg.n_act = 6; cfg.dspark_experts = 128; cfg.n_layers = 40;
  const size_t rec = ExpertLayout::make(64, 64).total;
  ExpertStore s(cfg, 40, 64 * rec, 1, 3);
  assert(s.cache_policy() == 1);
  // global prior: on layer 0, expert 1 three times and expert 2 once (decode weight 1) — no row owner (row -1) → global only
  s.observe(0, 1, 1.f); s.observe(0, 1, 1.f); s.observe(0, 1, 1.f); s.observe(0, 2, 1.f);
  s.decay_scores(0.97f);  // no active sequence → prio = 0 (the prior share is attached per sequence)
  assert(s.prio_of(1) == 0.f && s.prio_of(2) == 0.f);
  // sequence 7: one decode row, observes expert 1 on layer 0 (row 0) — α 8
  const uint32_t u7 = 7; s.set_row_owners(&u7, 1, false);
  s.observe(0, 1, 1.f, 0); s.observe(0, 5, 1.f, 0);
  s.decay_scores(0.97f);
  // the global prior decays by gd 0.999 per step: the earlier 4 observations are one step old (0.999) · p_k = g_k / Σ_layer g × 6 · n = 2/(6·40) tokens
  const float d = 0.999f, gl = 4.f * d + 2.f, n7 = 2.f / 6.f / 40.f, a = 8.f;
  const float p1 = (3.f * d + 1.f) / gl * 6.f, p5 = 1.f / gl * 6.f, p2 = d / gl * 6.f;
  assert(near(s.prio_of(1), (1.f + a * p1) / (n7 + a)));
  assert(near(s.prio_of(5), (1.f + a * p5) / (n7 + a)));
  assert(near(s.prio_of(2), a * p2 / (n7 + a)));
  // a prompt with the same uid (after decode) = a new request: this request's share is cleared → prio = prior share only (p is a ratio, so the decay cancels)
  s.set_row_owners(&u7, 1, true);
  s.decay_scores(0.97f);
  assert(near(s.prio_of(5), a * p5 / (0.f + a)));
  // not seen for idle (8 steps) → retired → contribution 0
  for (int i = 0; i < 9; ++i) s.decay_scores(0.97f);
  assert(s.prio_of(5) == 0.f && s.prio_of(1) == 0.f);
  // per-row owners (batch decode): row 0 → uid 1, row 1 → uid 2 — their evidence does not mix
  const uint32_t u12[2] = {1, 2}; s.set_row_owners(u12, 2, false);
  s.observe(3, 9, 1.f, 0); s.observe(3, 9, 1.f, 0); s.observe(3, 10, 1.f, 1);
  s.decay_scores(0.97f);
  const float n1 = 2.f / 6.f / 40.f, n2 = 1.f / 6.f / 40.f;
  const float p9 = 2.f / 3.f * 6.f, p10 = 1.f / 3.f * 6.f;
  assert(near(s.prio_of(3 * 384 + 9), (2.f + a * p9) / (n1 + a) + (0.f + a * p9) / (n2 + a)));
  assert(near(s.prio_of(3 * 384 + 10), (0.f + a * p10) / (n1 + a) + (1.f + a * p10) / (n2 + a)));
  // promotion: threshold = policy min (0.025) — the caller's 2.f is not used
  assert(s.promote(4, 2.f, nullptr) > 0);
  s.commit_all();
  assert(s.slot_of(3, 9) >= 0 && s.slot_of(3, 10) >= 0);
  puts("seq policy unit: prior/owner formula, new-request reset, idle retire, per-row owners, policy threshold OK");
}
'''


def main():
    contracts()
    with tempfile.TemporaryDirectory(prefix='hive-cache-replay-test-') as td:
        td = Path(td)
        trace = td / 't.bin'; write_trace(trace)
        (td / 'cur').mkdir()
        exe = cache_replay.build(td / 'cur', sanitize='undefined', opt_level='-O1')
        steps = parse_py(trace)
        kinds = {k: sum(1 for s in steps if s['kind'] == k) for k in range(4)}
        assert all(kinds[k] > 0 for k in range(4)) and any(s['H'] for s in steps), kinds
        # 2) independent re-implementation
        for slots, phase in ((96, 1), (96, 0), (240, 1)):
            r, _ = run_exe(exe, trace, None, ['--slots', str(slots), '--phase-score', str(phase)])
            m = Mirror(slots, 8, bool(phase))
            acc, res_end = m.run(steps)
            for k, name in enumerate(('decode', 'chunk', 'verify', 'draft')):
                rows, hit = acc[k]
                got = r['kinds'][name]
                assert got['rows'] == rows, (slots, phase, name, got, rows)
                assert abs(got['hit_pct'] - (100.0 * hit / rows if rows else 0.0)) < 5e-4, (slots, phase, name, got, hit, rows)
            n_points = sum(1 for s in steps if s['kind'] != 3 and s['L'])
            assert abs(r['promotions_per_point'] * n_points - (m.promotions - r['warm_cache'])) < 0.6, (r['promotions_per_point'], m.promotions, r['warm_cache'])
            assert r['resident_end'] == res_end, (r['resident_end'], res_end)
            print(f'mirror == replay: slots {slots} phase {phase} · decode hit {r["kinds"]["decode"]["hit_pct"]:.2f}% · '
                  f'chunk {r["kinds"]["chunk"]["hit_pct"]:.2f}% · verify {r["kinds"]["verify"]["hit_pct"]:.2f}% · draft {r["kinds"]["draft"]["hit_pct"]:.2f}% OK')
        base, _ = run_exe(exe, trace, None, ['--slots', '96'])
        # 3) negative control: a mutation ignoring the promote threshold → the result must differ
        src = (ROOT / 'engine/src/expert_store.cpp').read_text()
        needle = 'if (slot_of_[k] < 0 && sv[k] >= min_score) {'
        assert src.count(needle) == 1
        mut = td / 'expert_store_mutant.cpp'; mut.write_text(src.replace(needle, 'if (slot_of_[k] < 0 && sv[k] >= 0.f) {'))
        (td / 'mut').mkdir()
        mexe = cache_replay.build(td / 'mut', sanitize='undefined', opt_level='-O1', store_source=mut)
        mr, _ = run_exe(mexe, trace, None, ['--slots', '96'])
        assert (mr['hit_pct'], mr['promotions_per_point']) != (base['hit_pct'], base['promotions_per_point']), 'mutant not detected'
        print('negative control: mutated promote threshold changes the replay (it runs expert_store.cpp) OK')
        # 4) policy unset = default · switch decision
        strip = lambda d: {k: v for k, v in d.items() if k not in ('policy', 'label')}
        for v in ('', '0', 'lfu9', 'SEQ'):
            rv, err = run_exe(exe, trace, v, ['--slots', '96'])
            assert strip(rv) == strip(base), v
            if v not in ('', '0'): assert 'unknown policy' in err, err
        head = subprocess.run(['git', '-C', str(ROOT), 'show', 'HEAD:engine/src/expert_store.cpp'], capture_output=True, text=True)
        if head.returncode == 0 and 'D4 cache policy BEGIN' not in head.stdout:
            hh = subprocess.run(['git', '-C', str(ROOT), 'show', 'HEAD:engine/include/hive/expert_store.h'], capture_output=True, text=True, check=True)
            (td / 'head').mkdir(); (td / 'head/expert_store.cpp').write_text(head.stdout); (td / 'head/expert_store.h').write_text(hh.stdout)
            hexe = cache_replay.build(td / 'head', sanitize='undefined', opt_level='-O1', store_source=td / 'head/expert_store.cpp',
                                      store_header=td / 'head/expert_store.h')
            for args in (['--slots', '96'], ['--slots', '240', '--phase-score', '0'], ['--slots', '96', '--promote-misses', '4']):
                a, _ = run_exe(exe, trace, None, args); b, _ = run_exe(hexe, trace, None, args)
                assert strip(a) == strip(b), args
            print('policy unset == pre-policy HEAD expert_store (identical replay JSON, incl. --promote-misses) OK')
        else:
            print('policy unset vs HEAD: skipped (HEAD already contains the D4 policy block)')
        rs, err = run_exe(exe, trace, 'seq', ['--slots', '96'])
        assert rs['policy'] == 'seq' and 'cache policy seq (alpha 8' in err
        rs2, err2 = run_exe(exe, trace, 'seq:alpha=4:min=x:bogus=1', ['--slots', '96'])
        assert 'alpha 4' in err2 and "ignored 'min=x'" in err2 and "ignored 'bogus=1'" in err2
        print(f'switch: ""/"0"/unknown = default decisions · seq active (synthetic decode hit {base["decode_hit_pct"]:.2f} → {rs["decode_hit_pct"]:.2f}%) · bad args ignored OK')
        # 5) seq unit test (real ExpertStore)
        (td / 'unit').mkdir()
        cpp = td / 'unit/unit.cpp'; cpp.write_text(UNIT)
        cmd = ['g++', '-std=c++20', '-O1', '-g', '-pthread', '-mavx2', '-mfma', '-mf16c', '-fsanitize=undefined', '-fno-sanitize-recover=all',
               '-I' + str(ROOT / 'tools/cpu_fake/include'), '-I' + str(ROOT / 'engine/include'), '-I' + str(ROOT / 'engine/third_party'),
               str(cpp), str(ROOT / 'engine/src/expert_store.cpp'), str(ROOT / 'engine/src/cpu/expert_cpu.cpp'), str(ROOT / 'tools/cpu_fake/fake_checkpoint.cpp'),
               '-o', str(td / 'unit/unit')]
        subprocess.run(cmd, check=True)
        env = dict(os.environ); env['HIVE_CACHE_POLICY'] = 'seq'; env.pop('HIVE_CACHE_EVENTS', None)
        p = subprocess.run([str(td / 'unit/unit')], env=env, capture_output=True, text=True)
        sys.stdout.write(p.stdout)
        if p.returncode != 0: sys.stderr.write(p.stderr); raise SystemExit('seq unit test failed')
    print('cache replay CPU: contracts, mirror parity, negative control, default identity, seq unit OK')


if __name__ == '__main__':
    main()
