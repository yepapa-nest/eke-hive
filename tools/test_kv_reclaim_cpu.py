#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""hived's KV reclaim hook (the families whose KV growth can evict idle sessions — GLM, Qwen, 27B) on the REAL hived.cpp over the
CPU fake runtime (tools/test_daemon_cpu.py). The fake DeepSeek runtime has no reclaim member, so the build routes hived's
rt_set_kv_reclaim / rt_kv_reclaimable probes to the fake: a forward whose ids contain TRIGGER calls the reclaim callback up to
FAKE_RECLAIM_N times (as GlmEngine::kv_reserve does under VRAM pressure), and a sequence counts as reclaimable when its position is
above FAKE_RECLAIM_MIN_POS (as GlmEngine::kv_above_base). Checked:
  * the least recently used idle session that holds KV is evicted; one holding none is skipped; the session whose request runs
    the reclaim is never evicted — also when it is being restored from a host image;
  * an evicted conversation continues exactly from its host image (state oracle tokens, cached prefix = its previous length).
Negative controls (HIVE_KV_RECLAIM_NEGATIVE=1): without the reclaimable filter the short session is evicted; without the
admitting protection the session being restored evicts itself.
"""
import os
from pathlib import Path
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent / 'cpu_fake')]
import harness  # noqa: E402
import test_daemon_cpu as T  # noqa: E402

TRIGGER = 9          # never produced by T.tokens (20..3999) nor by the oracle (>= 10 + ... but 9 < 10)
MIN_POS = 100

HOOK_DECL = ('namespace hive { void fake_kv_reclaim_set(std::function<bool()> f); bool fake_kv_reclaimable(int64_t pos); }\n')
PROBE = 'template <class R> concept RtHasKvReclaim = requires(R& r, std::function<bool()> f) { r.set_kv_reclaim(f); };'
SETTER = 'template <class R> void rt_set_kv_reclaim(R& r, std::function<bool()> f) { if constexpr (RtHasKvReclaim<R>) r.set_kv_reclaim(std::move(f)); }'
FILTER = 'template <class R, class S> bool rt_kv_reclaimable(const R& r, const S& q) { if constexpr (RtHasKvReclaimable<R, S>) return r.kv_reclaimable(q); else return true; }'
ADMIT = '!admitting_sids.count(jt->first) &&\n            rt_kv_reclaimable(rt, *jt->second.seq)'
FWD_ANCHOR = '  HIVE_CHECK(seq.pos + M <= seq.cap, "context overflow");\n'
FWD_HOOK = FWD_ANCHOR + '''  for (int m = 0; m < M; ++m)
    if (ids[m] == fake_env("FAKE_RECLAIM_TRIGGER", -1)) {
      int n = 0;
      while (n < fake_env("FAKE_RECLAIM_N", 1) && g_fake_kv_reclaim && g_fake_kv_reclaim()) ++n;
      fprintf(stderr, "[fake] kv reclaim: %d session(s)\\n", n);
      break;
    }
'''
RT_HOOKS = '''
namespace hive {
std::function<bool()> g_fake_kv_reclaim;
void fake_kv_reclaim_set(std::function<bool()> f) { g_fake_kv_reclaim = std::move(f); }
bool fake_kv_reclaimable(int64_t pos) { return pos > fake_env("FAKE_RECLAIM_MIN_POS", 0); }
}  // namespace hive
'''


def hived_text(filter_on=True, admit_on=True):
    src = (harness.ROOT / 'engine/src/hived.cpp').read_text()
    for a in (PROBE, SETTER, FILTER, ADMIT):
        assert src.count(a) == 1, 'hived.cpp anchor drifted: ' + a[:60]
    src = src.replace('\nnamespace {\n', '\n' + HOOK_DECL + 'namespace {\n', 1)
    src = src.replace(PROBE, 'template <class R> concept RtHasKvReclaim = true;  // test: the fake takes the hook')
    src = src.replace(SETTER, 'template <class R> void rt_set_kv_reclaim(R&, std::function<bool()> f) { hive::fake_kv_reclaim_set(std::move(f)); }')
    src = src.replace(FILTER, 'template <class R, class S> bool rt_kv_reclaimable(const R&, const S& q) { return '
                      + ('hive::fake_kv_reclaimable(q.pos)' if filter_on else 'true') + '; }')
    if not admit_on:
        src = src.replace(ADMIT, 'rt_kv_reclaimable(rt, *jt->second.seq)')
    return src


def build(td, name, hived_source):
    rt = harness.runtime_source()
    assert rt.count(FWD_ANCHOR) == 1, 'fake forward anchor drifted'
    rt = rt.replace('static int fake_env(const char* n, int d) { const char* v = getenv(n); return v && *v ? atoi(v) : d; }\n',
                    'static int fake_env(const char* n, int d) { const char* v = getenv(n); return v && *v ? atoi(v) : d; }\n'
                    'extern std::function<bool()> g_fake_kv_reclaim;\n', 1)
    rt = rt.replace(FWD_ANCHOR, FWD_HOOK, 1) + RT_HOOKS
    gen = Path(td) / f'{name}_rt.cpp'
    gen.write_text(rt)
    src = Path(td) / f'{name}_hived.cpp'
    src.write_text(hived_source)
    objs = harness.objects(td, [src, gen, harness.ROOT / 'engine/src/expert_store.cpp', harness.ROOT / 'engine/src/cpu/expert_cpu.cpp',
                                harness.FAKE / 'fake_checkpoint.cpp'])
    return harness.build(td, name, '', objs)


def evicted(h):
    return [l.split('idle session ')[1].split(' ')[0] for l in T.log_lines(h, 'KV reclaim: idle session')]


def scenario(exe, td, ck):
    env = {'FAKE_RECLAIM_TRIGGER': str(TRIGGER), 'FAKE_RECLAIM_MIN_POS': str(MIN_POS), 'FAKE_RECLAIM_N': '8'}
    h = T.Hived(exe, td, env, args=('--max-sessions', '6', '--max-batch', '1'), name='kvr')
    try:
        S, A, C = T.tokens(40, 301), T.tokens(300, 302), T.tokens(300, 303)
        T.done_ok(ck, 'short idle session', h.generate('sshort', S, 3), S, 3, 0)          # oldest, no KV above the base
        ra = T.done_ok(ck, 'long session A', h.generate('along', A, 5), A, 5, 0)
        rc = T.done_ok(ck, 'long session C', h.generate('clong', C, 5), C, 5, 0)
        D = T.tokens(150, 304) + [TRIGGER] + T.tokens(20, 305)
        T.done_ok(ck, 'D runs the reclaim', h.generate('dnew', D, 3), D, 3, 0)
        ck.eq('D: idle sessions holding KV evicted, least recently used first', evicted(h), ['along', 'clong'])
        # A returns: restored from its host image; its own new rows run the reclaim again — D (idle, holding KV) goes, A stays
        A2 = A + ra['tokens'] + T.tokens(30, 306) + [TRIGGER]
        r = T.done_ok(ck, 'A returns after eviction', h.generate('along', A2, 4), A2, 4, None)
        ck.true('A: continued from its host image', r['done'] and r['done']['cached_prefix'] >= len(A), repr(r['done']))
        ck.eq('A2: evictions so far', evicted(h), ['along', 'clong', 'dnew'])
        C2 = C + rc['tokens'] + T.tokens(10, 307)
        r = T.done_ok(ck, 'C returns after eviction', h.generate('clong', C2, 4), C2, 4, None)
        ck.true('C: continued from its host image', r['done'] and r['done']['cached_prefix'] >= len(C), repr(r['done']))
        ck.true('reclaim calls logged', len(T.log_lines(h, '[fake] kv reclaim:')) == 2)
    finally:
        T.stop_clean(h, ck)


def main():
    with tempfile.TemporaryDirectory(prefix='hive-kvr-') as td:
        if os.environ.get('HIVE_KV_RECLAIM_NEGATIVE'):
            bad = 0
            for name, src in (('nofilter', hived_text(filter_on=False)), ('noadmit', hived_text(admit_on=False))):
                ck = T.Checker(name)
                try:
                    scenario(build(td, 'kvr_' + name, src), td, ck)
                except AssertionError as e:   # the daemon died — also a detection
                    ck.failures.append(str(e)[:200])
                print(f'  mutant [{name}]: {"DETECTED" if ck.failures else "NOT DETECTED"} — {(ck.failures or [""])[0][:160]}')
                bad += not ck.failures
            sys.exit(1 if bad else 0)
        ck = T.Checker('kv-reclaim')
        scenario(build(td, 'kvr', hived_text()), td, ck)
        for f in ck.failures:
            print('  FAIL', f)
        if ck.failures:
            sys.exit(1)
        print(f'KV reclaim (real hived.cpp on the fake runtime): {ck.checks} checks passed — LRU idle victims with KV, short and '
              'admitting sessions kept, evicted conversations continue from their host images')


if __name__ == '__main__':
    main()
