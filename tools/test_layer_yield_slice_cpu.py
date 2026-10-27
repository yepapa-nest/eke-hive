#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HIVE_LAYER_YIELD_SLICE · HIVE_PRIORITY_SHARED · image requests at a layer yield, on the REAL engine/src/hived.cpp (fake CUDA/runtime —
the tools/test_layer_yield_cpu.py setup: 8 fake layers × 100 ms per prefill forward, a yield at every layer boundary when there is work).

Checked (every token = the state oracle of its own prompt — scheduling only):
  * slices: a 600-row request of the same priority arriving inside a 700-row forward starts at a layer yield (its first slice runs
    inside that forward, at most HIVE_LAYER_YIELD_SLICE rows), and the one left when the forward ends continues as the main prefill;
    the long prompt's chunk sizes are unchanged. Control (switch off): the request's first chunk comes after the long forward.
  * slices of a better class: a priority -1 request inside a priority 1 forward gets a slice at every yield and all its tokens come out
    before the long prompt's first token.
  * a worse class is not sliced: a priority 1 request inside a priority -1 forward waits for it.
  * cancel while sliced: the request ends with finish "cancel", the long prompt is unaffected.
  * HIVE_PRIORITY_SHARED=0: a priority 0 request is admitted and decodes while a priority -1 one decodes. Control (off): it waits for it.
  * images: a follow-up turn whose image lies inside the reusable prefix is admitted at a yield (short rule) and answers before the long
    prompt; the same turn with different image bytes (nothing reusable — an image to encode) waits for the forward.
  * HIVE_PREFILL_PRIORITY_HOLD: a priority 1 decoder gets no decode step while a priority -1 prompt prefills (three chunks — no decode
    share between them nor at their layer yields), then resumes and follows its oracle; a decoder of the same class keeps its share.
    Control (off): the priority 1 decoder steps between and inside the chunks.
  * negative controls (each must be DETECTED): slices that ignore the parked rows R · slices that keep the tail-layer skip (C1) on ·
    a hold that skips the layer yields.
No GPU. Run: nice -n 19 taskset -c 4-5 python3 tools/test_layer_yield_slice_cpu.py
"""
from pathlib import Path
import re
import sys
import tempfile
import time

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import test_daemon_cpu as t  # noqa: E402
import test_layer_yield_cpu as ly  # noqa: E402

SLOW = t.SLOW
TILE3 = t.TILE3
SLICE = 256
SLICE_ON = {**ly.ON, 'HIVE_LAYER_YIELD_SLICE': str(SLICE)}
STARTED_RE = re.compile(r'^\[hived\] (\S+): sliced prefill started at a layer yield · (\d+) rows to prefill')
CONT_RE = re.compile(r'^\[hived\] (\S+): sliced prefill continues as the main prefill')


def first_line(h, pred):
    for i, l in enumerate(ly.lines(h)):
        if pred(l):
            return i
    return None


def chunk_lines(h, sid):
    return [(i, int(m[3])) for i, m in ((i, ly.CHUNK_RE.match(l)) for i, l in enumerate(ly.lines(h))) if m and m[1] == sid]


def pair(h, L_sid, L, lp, B_sid, B, bp, n_b=3, delay=.25):
    res, first, last = {}, {}, {}
    def on(key):
        return lambda k: (first.setdefault(key, time.monotonic()), last.__setitem__(key, time.monotonic()))
    a = t.launch(h, L_sid, L, 3, res, 'L', on_token=on('L'), priority=lp)
    time.sleep(delay * SLOW)
    b = t.launch(h, B_sid, B, n_b, res, 'B', on_token=on('B'), priority=bp)
    a.join(60 * SLOW)
    b.join(60 * SLOW)
    return res, first, last


def sc_slice_same(h, ck, on, tag):
    L, B = t.tokens(700, 2001), t.tokens(600, 2002)
    res, first, _ = pair(h, tag + 'L', L, None, tag + 'B', B, None)
    t.done_ok(ck, f'{tag} long', res['L'], L, 3, 0)
    t.done_ok(ck, f'{tag} 600-row request', res['B'], B, 3, 0)
    lc, bc = chunk_lines(h, tag + 'L'), chunk_lines(h, tag + 'B')
    ck.eq(f'{tag}: long prompt chunk sizes unchanged', [m for _, m in lc], [700])
    started = [m for m in (STARTED_RE.match(l) for l in ly.lines(h)) if m and m[1] == tag + 'B']
    if on:
        ck.eq(f'{tag}: started as a sliced prefill once', len(started), 1)
        ck.true(f'{tag}: its first chunk is a slice (<= {SLICE} rows)', bool(bc) and bc[0][1] <= SLICE, repr(bc))
        ck.true(f'{tag}: its first slice ran inside the long forward', bool(bc) and bool(lc) and bc[0][0] < lc[0][0], repr((bc, lc)))
        ck.true(f'{tag}: the rest continued as the main prefill', any((m := CONT_RE.match(l)) and m[1] == tag + 'B' for l in ly.lines(h)), '')
        ck.eq(f'{tag}: all rows prefilled', sum(m for _, m in bc), len(B))
    else:
        ck.eq(f'{tag} control: not sliced', len(started), 0)
        ck.true(f'{tag} control: its first chunk after the long forward', bool(bc) and bool(lc) and bc[0][0] > lc[0][0], repr((bc, lc)))
    return first


def sc_slice_better(h, ck, tag):
    L, B = t.tokens(700, 2011), t.tokens(600, 2012)
    res, first, last = pair(h, tag + 'L', L, 1, tag + 'B', B, -1, n_b=6)
    t.done_ok(ck, f'{tag} long (priority 1)', res['L'], L, 3, 0)
    t.done_ok(ck, f'{tag} request (priority -1)', res['B'], B, 6, 0)
    bc = chunk_lines(h, tag + 'B')
    ck.true(f'{tag}: every chunk of the better request is a slice', bool(bc) and all(m <= SLICE for _, m in bc), repr(bc))
    ck.true(f'{tag}: all its tokens before the long prompt\'s first', last.get('B', 1e9) < first.get('L', 0), repr((first, last)))


def sc_slice_worse(h, ck, tag):
    L, B = t.tokens(700, 2021), t.tokens(600, 2022)
    res, first, _ = pair(h, tag + 'L', L, -1, tag + 'B', B, 1)
    t.done_ok(ck, f'{tag} long (priority -1)', res['L'], L, 3, 0)
    t.done_ok(ck, f'{tag} request (priority 1)', res['B'], B, 3, 0)
    started = [m for m in (STARTED_RE.match(l) for l in ly.lines(h)) if m and m[1] == tag + 'B']
    ck.eq(f'{tag}: a worse class is not sliced', len(started), 0)
    ck.true(f'{tag}: it answers after the long prompt', first.get('B', 0) > first.get('L', 1e9), repr(first))


def sc_slice_cancel(h, ck, tag):
    # a 3-forward long prompt (2,000 rows · 768-row chunks) and a 1,200-row request of the same class: the request is sliced and cancelled
    #   between its slices; the long prompt is unaffected
    L, B = t.tokens(2000, 2031), t.tokens(1200, 2032)
    res = {}
    a = t.launch(h, tag + 'L', L, 3, res, 'L')
    time.sleep(.25 * SLOW)
    b = t.launch(h, tag + 'B', B, 3, res, 'B')
    deadline = time.monotonic() + 20 * SLOW
    while time.monotonic() < deadline and not chunk_lines(h, tag + 'B'):
        time.sleep(.02)
    h.cancel(tag + 'B')
    a.join(60 * SLOW)
    b.join(60 * SLOW)
    t.done_ok(ck, f'{tag} long', res['L'], L, 3, 0)
    ck.true(f'{tag}: the cancelled request ends with finish cancel', res['B']['done'] is not None and res['B']['done'].get('finish') == 'cancel',
            repr(res['B']))
    ck.true(f'{tag}: it was sliced before the cancel', bool(chunk_lines(h, tag + 'B')), '')


def sc_same_session(h, ck, tag):
    # a short request on the session of a sliced prefill, sent between its slices, must not run on that session's half-written state
    L, B, B2 = t.tokens(700, 2061), t.tokens(600, 2062), t.tokens(50, 2063)
    res = {}
    a = t.launch(h, tag + 'L', L, 3, res, 'L')
    time.sleep(.25 * SLOW)
    b = t.launch(h, tag + 'B', B, 3, res, 'B')
    time.sleep(.25 * SLOW)
    c = t.launch(h, tag + 'B', B2, 3, res, 'B2')
    for th in (a, b, c):
        th.join(60 * SLOW)
    t.done_ok(ck, f'{tag} long', res['L'], L, 3, 0)
    t.done_ok(ck, f'{tag} sliced request', res['B'], B, 3, 0)


def sc_shared(h, ck, on, tag):
    D1, D2 = t.tokens(100, 2041), t.tokens(100, 2042)
    res, first, last = {}, {}, {}
    def on_tok(key):
        return lambda k: (first.setdefault(key, time.monotonic()), last.__setitem__(key, time.monotonic()))
    a = t.launch(h, tag + 'D1', D1, 120, res, 'D1', on_token=on_tok('D1'), priority=-1)
    deadline = time.monotonic() + 10 * SLOW
    while 'D1' not in first and time.monotonic() < deadline:
        time.sleep(.01)
    b = t.launch(h, tag + 'D2', D2, 10, res, 'D2', on_token=on_tok('D2'), priority=0)
    a.join(60 * SLOW)
    b.join(60 * SLOW)
    t.done_ok(ck, f'{tag} priority -1 decoder', res['D1'], D1, 120, 0)
    t.done_ok(ck, f'{tag} priority 0 decoder', res['D2'], D2, 10, 0)
    if on:
        ck.true(f'{tag}: the priority 0 request finishes while the -1 one decodes', last.get('D2', 1e9) < last.get('D1', 0), repr((first, last)))
    else:
        ck.true(f'{tag} control: the priority 0 request waits for the -1 one', first.get('D2', 0) > last.get('D1', 1e9), repr((first, last)))


def sc_prefill_hold(h, ck, on, tag, dp=1):
    """A decoder (priority dp) is streaming; a priority -1 prompt of three chunks arrives. Counts the decoder's tokens while that prompt
    prefills (from its first chunk line to its first token)."""
    D, L = t.tokens(100, 2061), t.tokens(2000, 2062)
    res, stamps, first = {}, [], {}
    n0 = len(ly.lines(h))
    a = t.launch(h, tag + 'D', D, 1900, res, 'D', on_token=lambda k: stamps.append(time.monotonic()), priority=dp)
    deadline = time.monotonic() + 10 * SLOW
    while len(stamps) < 5 and time.monotonic() < deadline:
        time.sleep(.01)
    t0 = time.monotonic()
    b = t.launch(h, tag + 'L', L, 3, res, 'L', on_token=lambda k: first.setdefault('L', time.monotonic()), priority=-1)
    b.join(60 * SLOW)
    deadline = time.monotonic() + 10 * SLOW  # let the decoder step once after the prompt (cancelling at once raced it on a slow runner)
    while not any(x > first.get('L', 1e9) for x in stamps) and time.monotonic() < deadline:
        time.sleep(.01)
    h.cancel(tag + 'D')
    a.join(60 * SLOW)
    t.done_ok(ck, f'{tag} prompt (priority -1)', res['L'], L, 3, 0)
    ck.true(f'{tag}: decoder ended by cancel', (res['D'].get('done') or {}).get('finish') == 'cancel', repr(res['D'].get('done')))
    ck.eq(f'{tag}: decoder tokens follow its oracle', res['D']['tokens'], t.oracle(D, len(res['D']['tokens'])))
    lc = chunk_lines(h, tag + 'L')
    ck.eq(f'{tag}: prompt chunk sizes unchanged', [m for _, m in lc], [768, 768, 464])
    during = sum(1 for x in stamps if t0 + .2 * SLOW < x < first.get('L', 0))
    held = sum(1 for l in ly.lines(h)[n0:] if l.startswith('[hived] decode share held: prefilling priority -1'))
    held_ly = sum(1 for l in ly.lines(h)[n0:] if l.startswith('[hived] layer yield done:') and l.find(' · held by priority') > 0)
    if on and dp > -1:
        ck.true(f'{tag}: the worse-class decoder waits for the whole prefill', during == 0, f'{during} tokens during the prefill')
        ck.eq(f'{tag}: decode share held between the chunks', held, 2)
        ck.true(f'{tag}: layer yields hold it too', held_ly > 0, f'{held_ly} held yields')
        ck.true(f'{tag}: it resumes after the prompt', any(x > first.get('L', 1e9) for x in stamps), '')
    else:
        ck.true(f'{tag} control: the decoder steps during the prefill', during >= 10, f'{during} tokens during the prefill')
        ck.eq(f'{tag} control: nothing held', (held, held_ly), (0, 0))


def sc_image_prefix(h, ck, tag):
    P = t.tokens(300, 2051)
    im = t.image(50, 3, 3, 2052)
    r1 = t.done_ok(ck, f'{tag} image turn 1', h.generate(tag + 'I', P, 3, [im]), P, 3, 0, [im])
    T2 = P + r1['tokens'] + t.tokens(20, 2053)
    a0 = ly.n_admits(h)
    L = t.tokens(700, 2054)
    res, first, _ = {}, {}, {}
    a = t.launch(h, tag + 'L', L, 3, res, 'L', on_token=lambda k: first.setdefault('L', time.monotonic()))
    time.sleep(.25 * SLOW)
    b = t.launch(h, tag + 'I', T2, 3, res, 'I', images=[im], on_token=lambda k: first.setdefault('I', time.monotonic()))
    a.join(60 * SLOW)
    b.join(60 * SLOW)
    t.done_ok(ck, f'{tag} long', res['L'], L, 3, 0)
    t.done_ok(ck, f'{tag} image turn 2 (image in the reusable prefix)', res['I'], T2, 3, len(P) + 2, [im])
    ck.eq(f'{tag}: admitted at a yield', ly.n_admits(h) - a0, 1)
    ck.true(f'{tag}: it answers before the long prompt', first.get('I', 1e9) < first.get('L', 0), repr(first))
    # control: different image bytes — nothing reusable, the image must be encoded: it waits for the forward
    im2 = (im[0], bytes(reversed(im[1])))
    a1 = ly.n_admits(h)
    L2 = t.tokens(700, 2055)
    res, first = {}, {}
    a = t.launch(h, tag + 'L2', L2, 3, res, 'L', on_token=lambda k: first.setdefault('L', time.monotonic()))
    time.sleep(.25 * SLOW)
    b = t.launch(h, tag + 'I', T2, 3, res, 'I', images=[im2], on_token=lambda k: first.setdefault('I', time.monotonic()))
    a.join(60 * SLOW)
    b.join(60 * SLOW)
    t.done_ok(ck, f'{tag} control long', res['L'], L2, 3, 0)
    t.done_ok(ck, f'{tag} control image turn (changed image)', res['I'], T2, 3, 0, [im2])
    ck.eq(f'{tag} control: not admitted at a yield', ly.n_admits(h) - a1, 0)
    ck.true(f'{tag} control: it answers after the long prompt', first.get('I', 0) > first.get('L', 1e9), repr(first))


def run(exe, td):
    ck = t.Checker('layer-yield-slice')
    cases = (
        ('slice-on', SLICE_ON, lambda h: (sc_slice_same(h, ck, True, 'ss'), sc_slice_better(h, ck, 'sb'), sc_slice_worse(h, ck, 'sw'),
                                          sc_slice_cancel(h, ck, 'sc'), sc_same_session(h, ck, 'sm'))),
        ('slice-off', ly.ON, lambda h: sc_slice_same(h, ck, False, 'so')),
        ('shared-on', {**ly.ON, 'HIVE_PRIORITY_SHARED': '0'}, lambda h: sc_shared(h, ck, True, 'p1')),
        ('shared-off', ly.ON, lambda h: sc_shared(h, ck, False, 'p0')),
        ('image-prefix', ly.ON, lambda h: sc_image_prefix(h, ck, 'ip')),
        ('hold-on', {**ly.ON, 'HIVE_PREFILL_PRIORITY_HOLD': '1'}, lambda h: (sc_prefill_hold(h, ck, True, 'ph'), sc_prefill_hold(h, ck, True, 'pq', dp=-1))),
        ('hold-off', ly.ON, lambda h: sc_prefill_hold(h, ck, False, 'p_')),
    )
    for name, env, fn in cases:
        h = t.Hived(exe, td, env, args=TILE3, name='lys' + name)
        try:
            fn(h)
            ck.true(f'{name}: slices startup line iff switch on', ('[hived] layer yield slices on:' in h.tail(100000)) == ('HIVE_LAYER_YIELD_SLICE' in env), '')
            ck.true(f'{name}: hold startup line iff switch on', ('[hived] prefill priority hold on:' in h.tail(100000)) == ('HIVE_PREFILL_PRIORITY_HOLD' in env), '')
            ck.true(f'{name}: no fake-state error in the log', 'fake state' not in h.tail(100000), '')
        finally:
            t.stop_clean(h, ck)
        print(f'  {name}: {ck.checks} checks so far, {len(ck.failures)} failures', flush=True)
    return ck


def mutant(exe, td, env, fn):
    ck = t.Checker('mutant')
    h = t.Hived(exe, td, env, args=TILE3, name='lysmut')
    try:
        fn(h, ck)
        log = h.tail(100000)
    finally:
        try:
            h.stop()
        except Exception:  # noqa: BLE001 — a mutant may crash the daemon; that also counts as detected
            pass
    return bool(ck.failures) or 'fake state' in log, ck.failures[:3]


def run_mutants(exe, td):
    src = (t.harness.ROOT / 'engine/src/hived.cpp').read_text()
    out = []
    # 1) a slice that ignores the rows the runtime parked: it overwrites the paused forward's Work rows beyond R
    lim = "const size_t rem = J.ids.size() - J.i, lim = std::min((size_t)std::max(0, R), ly_slice);"
    want = "          need = std::max(need, (int)std::min(sliced[(size_t)k]->ids.size() - sliced[(size_t)k]->i, ly_slice));"
    assert src.count(lim) == 1 and src.count(want) == 1, 'slice lines drifted'
    m1 = src.replace(lim, "const size_t rem = J.ids.size() - J.i, lim = ly_slice;").replace(want, "          (void)k;")
    exe1 = t.build_daemon(td, name='hived_mut_slice_r', hived_source=m1)
    out.append(('slice beyond the parked rows', *mutant(exe1, td, SLICE_ON, lambda h, ck: sc_slice_better(h, ck, 'mr'))))
    # 2) the sliced request is not marked prefilling: its own follow-up could be admitted on top of the paused session
    mark = "              prefilling_sids.insert(Jp->A->sid);\n              ly_mark(Jp->A->sid, Jp->ids.size(), Jp->A->prio);\n              if (sliced.empty()) slice_credit = 0;"
    assert src.count(mark) == 1, 'slice mark drifted'
    m2 = src.replace(mark, "              ly_mark(Jp->A->sid, Jp->ids.size(), Jp->A->prio);\n              if (sliced.empty()) slice_credit = 0;")
    exe2 = t.build_daemon(td, name='hived_mut_slice_mark', hived_source=m2)
    out.append(('sliced session not marked prefilling', *mutant(exe2, td, SLICE_ON, lambda h, ck: sc_same_session(h, ck, 'mm'))))
    # 3) HIVE_PREFILL_PRIORITY_HOLD only between chunks: the layer yields still give the worse-class decoder its share
    ly_hold = "      if (!active.empty() && !held) {"
    assert src.count(ly_hold) == 1, 'layer-yield hold drifted'
    exe3 = t.build_daemon(td, name='hived_mut_hold_ly', hived_source=src.replace(ly_hold, "      if (!active.empty()) {"))
    out.append(('hold skips the layer yields', *mutant(exe3, td, {**ly.ON, 'HIVE_PREFILL_PRIORITY_HOLD': '1'}, lambda h, ck: sc_prefill_hold(h, ck, True, 'mh'))))
    return out


def main():
    with tempfile.TemporaryDirectory(prefix='hive-lys-') as td:
        t0 = time.time()
        exe = t.build_daemon(td)
        print(f'built real hived.cpp on fake CUDA/runtime in {time.time() - t0:.1f}s (sanitizer {t.harness.sanitizer() or "undefined"})', flush=True)
        ck = run(exe, td)
        muts = run_mutants(exe, td)
        for name, detected, why in muts:
            print(f'  mutant {name}: {"DETECTED" if detected else "MISSED"} {why!r}')
            ck.true(f'mutant detected: {name}', detected, repr(why))
        print(f'{ck.checks} checks, {len(ck.failures)} failures')
        for f in ck.failures:
            print('FAIL', f)
        sys.exit(1 if ck.failures else 0)


if __name__ == '__main__':
    main()
