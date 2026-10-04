#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""T3 HIVE_PREFILL_YIELD on the REAL engine/src/hived.cpp (fake CUDA/runtime, tools/test_daemon_cpu.py harness).

A short request arriving while a longer prefill runs is admitted at the long prompt's next EXISTING chunk/round boundary
(single path and H3 batch path) and streams its first token before the long prefill continues. Checked:
  * every token = the state oracle of its own prompt (scheduling only — nothing numeric changes),
  * the long prompt's chunk sizes are the same with and without the switch (no new chunk boundaries),
  * the short request's first token precedes the long one's by at least one fake forward (control: it does not),
  * size limit (HIVE_PREFILL_YIELD_MAX), busy sessions and max_batch are respected,
  * C1 (upper-layer skip): with the switch, a chunk that skipped the upper layers is always followed by a tail-mode chunk even
    when a yielded request becomes active between chunks under HIVE_PREFILL_BUDGET_MS (mutant without the c1_floor line: detected).
No GPU. Run: nice -n 19 taskset -c 4-5 python3 tools/test_prefill_yield_cpu.py  (HIVE_TEST_SANITIZER=thread for TSAN)
"""
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import test_daemon_cpu as t  # noqa: E402

SLOW = t.SLOW
TILE3 = t.TILE3                      # max_chunk 256 × tile 3 = 768-row chunks · prefill_threshold 128
YIELD = {'HIVE_PREFILL_YIELD': '1'}
YIELD_RE = re.compile(r'^\[hived\] prefill yield: (\d+) short requests? admitted')
FWD_MS = 300


def yields(h):
    return sum(int(m[1]) for m in (YIELD_RE.match(l) for l in h.tail(100000).splitlines()) if m)


def staggered(h, long_ids, shorts, long_sid='L', n=3, delay=.12):
    """long prompt first; `shorts` [(sid, ids)] sent `delay` s later (inside the long prompt's first forward)."""
    res, first, ths = {}, {}, []
    ths.append(t.launch(h, long_sid, long_ids, n, res, long_sid, on_token=lambda k: first.setdefault(long_sid, time.monotonic())))
    time.sleep(delay * SLOW)
    for sid, ids in shorts:
        ths.append(t.launch(h, sid, ids, n, res, sid, on_token=lambda k, sid=sid: first.setdefault(sid, time.monotonic())))
        time.sleep(.005)
    for th in ths:
        th.join(60 * SLOW)
    return res, first


def sc_basic(h, ck, on, tag):
    L, S = t.tokens(2000, 900), t.tokens(60, 901)
    res, first = staggered(h, L, [('S', S)])
    t.done_ok(ck, f'{tag} long', res['L'], L, 3, 0)
    t.done_ok(ck, f'{tag} short', res['S'], S, 3, 0)
    ck.eq(f'{tag}: long chunks unchanged (768/768/464)', [m for m, _ in t.chunks(h, 'L')], [768, 768, 464])
    gap = first.get('L', 0) - first.get('S', 1e9)
    if on:
        ck.true(f'{tag}: short streams before the long prefill ends', gap > .25 * SLOW, f'{gap:.3f}s {first!r}')
        ck.eq(f'{tag}: one yield', yields(h), 1)
    else:
        ck.true(f'{tag} control: short waits for the long prefill', gap < .25 * SLOW, f'{gap:.3f}s')
        ck.eq(f'{tag} control: no yield line', yields(h), 0)
    # the yielded short request's session continues normally (live state)
    nxt = S + res['S']['tokens'] + t.tokens(10, 902)
    t.done_ok(ck, f'{tag}: short turn 2 reuses its live state', h.generate('S', nxt, 3), nxt, 3, len(S) + 2)


def sc_limits(h, ck, tag, mid_yields):
    """200-row request (> default max 127) and a request of the long prompt's own session are not yielded."""
    n0 = yields(h)
    L, M = t.tokens(2000, 910), t.tokens(200, 911)
    res, first = staggered(h, L, [('M', M)], long_sid='L2')
    t.done_ok(ck, f'{tag} long', res['L2'], L, 3, 0)
    t.done_ok(ck, f'{tag} 200-row request', res['M'], M, 3, 0)
    ck.eq(f'{tag}: 200-row request yielded = {mid_yields}', yields(h) - n0, 1 if mid_yields else 0)
    # same session as the running long prefill: must stay queued (the long prefill owns that sequence)
    n1 = yields(h)
    L3, Q = t.tokens(2000, 912), t.tokens(40, 913)
    res = {}
    ths = [t.launch(h, 'L3', L3, 3, res, 'long')]
    time.sleep(.12 * SLOW)
    ths.append(t.launch(h, 'L3', Q, 3, res, 'follower'))
    for th in ths:
        th.join(60 * SLOW)
    t.done_ok(ck, f'{tag} long (same-session follower)', res['long'], L3, 3, 0)
    f = res['follower']
    ck.true(f'{tag}: same-session follower = session busy or its own oracle', f['error'] == 'session busy' or
            (f['error'] is None and f['tokens'] == t.oracle(Q, 3)), repr(f))
    ck.eq(f'{tag}: same-session request not yielded', yields(h) - n1, 0)


def sc_decoder(h, ck, tag):
    """A running decoder keeps its tokens correct and keeps stepping between chunks while a short request is yielded."""
    D = t.tokens(100, 920)
    t_tok, res = [], {}
    th = t.launch(h, 'D', D, 3000, res, 'D', on_token=lambda k: t_tok.append(time.monotonic()))
    deadline = time.monotonic() + 10 * SLOW
    while len(t_tok) < 5 and time.monotonic() < deadline:
        time.sleep(.01)
    L, S = t.tokens(2000, 921), t.tokens(60, 922)
    n0 = yields(h)
    r2, first = staggered(h, L, [('S4', S)], long_sid='L4')
    h.cancel('D')
    th.join(60 * SLOW)
    t.done_ok(ck, f'{tag} long with a decoder', r2['L4'], L, 3, 0)
    t.done_ok(ck, f'{tag} short with a decoder', r2['S4'], S, 3, 0)
    ck.eq(f'{tag}: decoder tokens follow its oracle', res['D']['tokens'], t.oracle(D, len(res['D']['tokens'])))
    ck.eq(f'{tag}: one yield with a decoder', yields(h) - n0, 1)
    ck.true(f'{tag}: short first token before the long one', first.get('L4', 0) - first.get('S4', 1e9) > .25 * SLOW, repr(first))


def c1_violations(h, tail):
    """A chunk that skipped the upper layers must be followed (same request) by a tail-mode chunk (M >= threshold 128 and > tail)."""
    by = {}
    for m in (t.CHUNK_RE.match(l) for l in h.tail(100000).splitlines()):
        if m:
            by.setdefault(m[1], []).append((int(m[3]), m[7]))
    bad = []
    for sid, cs in by.items():
        for (m0, up0), (m1, _) in zip(cs, cs[1:]):
            if up0 == 'skip' and not (m1 >= 128 and m1 > tail):
                bad.append((sid, m0, m1))
    return bad, by


def sc_c1(h, ck, tag, want_ok=True):
    L, S = t.tokens(1500, 930), t.tokens(40, 931)
    res, _ = staggered(h, L, [('S5', S)], long_sid='L5', n=40)
    t.done_ok(ck, f'{tag} long', res['L5'], L, 40, 0)
    t.done_ok(ck, f'{tag} short', res['S5'], S, 40, 0)
    ck.true(f'{tag}: a request was yielded between chunks', yields(h) >= 1, h.tail(20))
    bad, by = c1_violations(h, 200)
    if want_ok:
        ck.true(f'{tag}: C1 — every skipped chunk is followed by a tail-mode chunk', not bad, f'{bad} {by}')
    return bad


def run(exe, td):
    ck = t.Checker('t3-prefill-yield')
    slow = {'FAKE_PREFILL_MS': str(FWD_MS * SLOW)}
    batch = {**slow, **t.BATCH}
    c1 = {**slow, **YIELD, 'FAKE_KV_SOURCE': '1', 'HIVE_PREFILL_BUDGET_MS': '1'}
    c1_args = ('--prefill-tile', '2', '--prefill-threshold', '128', '--decoder-tail', '200')
    cases = (
        ('single-on', {**slow, **YIELD}, TILE3, lambda h: (sc_basic(h, ck, True, 'single'), sc_limits(h, ck, 'single', False), sc_decoder(h, ck, 'single'))),
        ('single-off', slow, TILE3, lambda h: sc_basic(h, ck, False, 'single-off')),
        ('batch-on', {**batch, **YIELD}, TILE3, lambda h: (sc_basic(h, ck, True, 'batch'), sc_limits(h, ck, 'batch', False), sc_decoder(h, ck, 'batch'))),
        ('batch-off', batch, TILE3, lambda h: sc_basic(h, ck, False, 'batch-off')),
        ('batch-sjf-on', {**batch, **YIELD, 'HIVE_BATCH_SJF': '1', 'HIVE_PREFILL_FAIR': '1'}, TILE3, lambda h: sc_basic(h, ck, True, 'batch+P1')),
        ('max300', {**slow, **YIELD, 'HIVE_PREFILL_YIELD_MAX': '300'}, TILE3, lambda h: sc_limits(h, ck, 'max300', True)),
        ('maxjunk', {**slow, **YIELD, 'HIVE_PREFILL_YIELD_MAX': '12x'}, TILE3, lambda h: sc_limits(h, ck, 'maxjunk', False)),
        ('c1', c1, c1_args, lambda h: sc_c1(h, ck, 'c1')),
    )
    only = [x for x in os.environ.get('T3_CASES', '').split(',') if x]
    for name, env, args, fn in cases:
        if only and name not in only:
            continue
        h = t.Hived(exe, td, env, args=args, name='t3' + name)
        try:
            fn(h)
            on = 'HIVE_PREFILL_YIELD' in env
            ck.true(f'{name}: startup line iff switch on', ('[hived] prefill yield on:' in h.tail(100000)) == on)
        finally:
            t.stop_clean(h, ck)
        print(f'  {name}: {ck.checks} checks so far, {len(ck.failures)} failures', flush=True)
    return ck


def run_mutant(td):
    """Negative control: without the single-path C1 floor line the C1 check must fail."""
    src = (t.harness.ROOT / 'engine/src/hived.cpp').read_text()
    line = '    else if (prefill_yield || layer_yield) next_cap = std::min(next_cap, c1_floor_j(J));'  # T11 added layer_yield (same floor)
    assert src.count(line) == 1, 'C1 floor line drifted'
    exe = t.build_daemon(td, name='hived_mut_c1', hived_source=src.replace(line, '    else if (prefill_yield || layer_yield) {}'))
    ck = t.Checker('t3-mutant')
    slow = {'FAKE_PREFILL_MS': str(FWD_MS * SLOW)}
    h = t.Hived(exe, td, {**slow, **YIELD, 'FAKE_KV_SOURCE': '1', 'HIVE_PREFILL_BUDGET_MS': '1'},
                args=('--prefill-tile', '2', '--prefill-threshold', '128', '--decoder-tail', '200'), name='t3mut')
    try:
        bad = sc_c1(h, ck, 'mutant', want_ok=False)
    finally:
        h.stop()
    return bool(bad), bad


def main():
    with tempfile.TemporaryDirectory(prefix='hive-t3-') as td:
        t0 = time.monotonic()
        exe = t.build_daemon(td)
        print(f'built real hived.cpp on fake CUDA/runtime in {time.monotonic() - t0:.1f}s (sanitizer {t.harness.sanitizer()})', flush=True)
        ck = run(exe, td)
        if os.environ.get('T3_MUTANT', '1') != '0' and not os.environ.get('T3_CASES'):
            detected, bad = run_mutant(td)
            ck.checks += 1
            if not detected:
                ck.failures.append('mutant without the C1 floor was NOT detected')
            print(f'  mutant (no single-path C1 floor): {"DETECTED" if detected else "NOT DETECTED"} {bad}')
    for f in ck.failures:
        print('FAIL', f)
    print(f't3 prefill yield: {ck.checks} checks, {len(ck.failures)} failures')
    sys.exit(1 if ck.failures else 0)


if __name__ == '__main__':
    main()
