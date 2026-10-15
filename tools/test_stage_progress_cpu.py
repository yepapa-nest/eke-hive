#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Read progress inside a prefill forward (HIVE_STAGE_PROGRESS_MS, hive/layer_yield.h Progress) on the REAL engine/src/hived.cpp
(fake CUDA/runtime, tools/test_daemon_cpu.py harness).

Before it, a request that asked for stages got {"progress":i} only after each chunk — a long prompt sat at 0 % for a whole chunk
(measured 2026-10-15: GLM ~25 s, DS ~20 s, FN ~9 s). Now the layer boundaries of the running forward report the fraction of layers done
at most once per period. Checked:
  * a one-chunk prompt (700 rows, 8 fake layers × 100 ms) with "stages" gets several in-forward lines strictly inside (0, 700), increasing,
    spaced about one period apart, then the chunk notice 700 — with the layer yield off (FN runs without it) and on,
  * the same prompt without "stages" gets only the chunk notice; HIVE_STAGE_PROGRESS_MS=0 turns the in-forward lines off,
  * a multi-chunk prompt: in-forward values stay inside their own chunk and the sequence never goes back,
  * the H3 batch path (forward_multi, two prompts in one round): each request gets its own rows,
  * tokens = the state oracle of the prompt every time (reporting changes nothing else),
  * negative control: with the fake runtime's progress call removed (FAKE_NO_PROGRESS) no in-forward line appears (the check would catch
    a runtime that drops the hook).
No GPU. Run: nice -n 19 python3 tools/test_stage_progress_cpu.py
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import test_daemon_cpu as t  # noqa: E402

SLOW = t.SLOW
LAYERS, FWD_MS, PERIOD = 8, 800, 150
BASE = {'FAKE_LAYERS': str(LAYERS), 'FAKE_PREFILL_MS': str(FWD_MS * SLOW), 'HIVE_STAGE_PROGRESS_MS': str(PERIOD * SLOW)}


def gen_lines(h, sid, ids, n, stages):
    """One generate request; returns (tokens, done, [(t, progress)]) — every progress line with its arrival time."""
    req = {'op': 'generate', 'session': sid, 'ids': ids, 'temperature': 0, 'max_tokens': n}
    if stages:
        req['stages'] = True
    s = h.connect()
    toks, done, prog = [], None, []
    try:
        s.sendall((json.dumps(req) + '\n').encode())
        for line in s.makefile():
            m = json.loads(line)
            if 'id' in m:
                toks.append(m['id'])
            elif 'progress' in m:
                prog.append((time.monotonic(), m['progress'], m.get('total')))
            elif m.get('done') or 'error' in m:
                done = m
                break
    finally:
        s.close()
    return toks, done, prog


def check_oracle(ck, what, toks, done, ids, n):
    ck.true(f'{what}: done', done is not None and done.get('done'), repr(done))
    ck.eq(f'{what}: tokens (state oracle)', toks, t.oracle(ids, n))


def inner(prog, chunk_ends):
    return [p for _, p, _ in prog if p not in chunk_ends]


def sc_one_chunk(h, ck, tag, expect_inner=True):
    ids = t.tokens(700, 1200 + len(tag))
    toks, done, prog = gen_lines(h, tag + 'A', ids, 3, True)
    check_oracle(ck, f'{tag} stages', toks, done, ids, 3)
    vals = [p for _, p, _ in prog]
    ins = inner(prog, {700})
    if expect_inner:
        ck.true(f'{tag}: several in-forward lines (8 layers × 100 ms, period {PERIOD} ms)', len(ins) >= 3, repr(vals))
        ck.true(f'{tag}: in-forward values inside (0, 700)', all(0 < p < 700 for p in ins), repr(ins))
        ck.true(f'{tag}: values never go back', vals == sorted(vals), repr(vals))
        gaps = [b[0] - a[0] for a, b in zip(prog, prog[1:]) if b[1] in ins and a[1] in ins]
        ck.true(f'{tag}: lines about one period apart', all(g >= PERIOD * SLOW / 1000 * .8 for g in gaps), repr([round(g, 3) for g in gaps]))
        ck.true(f'{tag}: total sent with every line', all(tot == 700 for _, _, tot in prog), repr(prog))
    else:
        ck.eq(f'{tag}: no in-forward line', ins, [])
    ck.eq(f'{tag}: chunk notice last', vals[-1:] if vals else [], [700])
    ids2 = t.tokens(700, 1300 + len(tag))
    toks, done, prog = gen_lines(h, tag + 'B', ids2, 3, False)
    check_oracle(ck, f'{tag} no stages', toks, done, ids2, 3)
    ck.eq(f'{tag}: without stages only the chunk notice', [p for _, p, _ in prog], [700])
    return len(ins)


def sc_multi_chunk(h, ck, tag):
    ids = t.tokens(1800, 1400)  # 768-row chunks (TILE3): 768, 1536, 1800
    toks, done, prog = gen_lines(h, tag + 'M', ids, 3, True)
    check_oracle(ck, f'{tag} multi', toks, done, ids, 3)
    vals = [p for _, p, _ in prog]
    ends = [768, 1536, 1800]
    ck.true(f'{tag}: chunk notices present', all(e in vals for e in ends), repr(vals))
    ck.true(f'{tag}: never goes back', vals == sorted(vals), repr(vals))
    for lo, hi in zip([0] + ends, ends):
        own = [p for p in vals if lo < p < hi]
        ck.true(f'{tag}: in-forward lines inside chunk ({lo}, {hi})', len(own) >= 1 or hi - lo < 300, repr(vals))


def sc_batch(h, ck, tag):
    """H3: two prompts in one forward_multi (queued behind a blocker so one round takes both — test_layer_yield_cpu.py sc_batch)."""
    a, b, blocker = t.tokens(400, 1500), t.tokens(380, 1501), t.tokens(150, 1502)
    res = {}
    def run(sid, ids, stages):
        res[sid] = gen_lines(h, sid, ids, 3, stages)
    ths = [threading.Thread(target=run, args=('bz', blocker, False))]
    ths[0].start()
    time.sleep(.15 * SLOW)
    for sid, ids in (('PA', a), ('PB', b)):
        th = threading.Thread(target=run, args=(sid, ids, True)); th.start(); ths.append(th)
    for th in ths:
        th.join(60 * SLOW)
    for sid, ids in (('PA', a), ('PB', b)):
        toks, done, prog = res[sid]
        check_oracle(ck, f'{tag} {sid}', toks, done, ids, 3)
        vals = [p for _, p, _ in prog]
        ck.true(f'{tag} {sid}: own rows only (inside (0, {len(ids)}], never back)', all(0 < p <= len(ids) for p in vals) and vals == sorted(vals), repr(vals))
        ck.true(f'{tag} {sid}: in-forward lines', len([p for p in vals if p < len(ids)]) >= 2, repr(vals))
    ck.true(f'{tag}: the two prompts shared one forward_multi', any(l.startswith('[fake] forward_multi parts 2 ') for l in h.tail(1000000).splitlines()), '')


def run(exe, td):
    ck = t.Checker('stage-progress')
    cases = (
        ('yield-off', BASE, t.TILE3, lambda h: (sc_one_chunk(h, ck, 'off'), sc_multi_chunk(h, ck, 'off'))),
        ('yield-on', {**BASE, 'HIVE_LAYER_YIELD': '100'}, t.TILE3, lambda h: sc_one_chunk(h, ck, 'on')),
        ('batch', {**BASE, **t.BATCH}, t.TILE3, lambda h: sc_batch(h, ck, 'batch')),
        ('progress-off', {**BASE, 'HIVE_STAGE_PROGRESS_MS': '0'}, t.TILE3, lambda h: sc_one_chunk(h, ck, 'pg0', expect_inner=False)),
    )
    for name, env, args, fn in cases:
        h = t.Hived(exe, td, env, args=args, name='pg' + name)
        try:
            out = fn(h)
            ck.true(f'{name}: no fake-state error in the log', 'fake state' not in h.tail(100000), '')
        finally:
            t.stop_clean(h, ck)
        print(f'  {name}: {ck.checks} checks so far, {len(ck.failures)} failures · {out!r}', flush=True)
    return ck


def mutant(exe, td):
    """FAKE_NO_PROGRESS: the fake runtime drops its progress call — the in-forward check must fail."""
    ck = t.Checker('stage-progress-mutant')
    h = t.Hived(exe, td, {**BASE, 'FAKE_NO_PROGRESS': '1'}, args=t.TILE3, name='pgmut')
    try:
        sc_one_chunk(h, ck, 'mut')
    finally:
        t.stop_clean(h, ck)
    return bool(ck.failures), ck.failures[:2]


def main():
    with tempfile.TemporaryDirectory(prefix='hive-pg-') as td:
        t0 = time.monotonic()
        exe = t.build_daemon(td)
        print(f'built real hived.cpp on fake CUDA/runtime in {time.monotonic() - t0:.1f}s (sanitizer {t.harness.sanitizer()})', flush=True)
        ck = run(exe, td)
        detected, why = mutant(exe, td)
        ck.checks += 1
        if not detected:
            ck.failures.append('mutant "no progress call" was NOT detected')
        print(f'  mutant (no progress call): {"DETECTED" if detected else "NOT DETECTED"} {why}', flush=True)
    for f in ck.failures:
        print('FAIL', f)
    print(f'stage progress: {ck.checks} checks, {len(ck.failures)} failures')
    sys.exit(1 if ck.failures else 0)


if __name__ == '__main__':
    main()
