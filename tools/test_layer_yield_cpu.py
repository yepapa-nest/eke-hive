#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""T11 HIVE_LAYER_YIELD on the REAL engine/src/hived.cpp (fake CUDA/runtime, tools/test_daemon_cpu.py harness).

A long prefill forward that has started no longer runs all its layers before anything else: at layer boundaries (the fake runtime
models FAKE_LAYERS layers and calls the production hive/layer_yield.h yield_point), hived's yield body admits waiting short text
requests (first token inside the long forward) and gives running decoders steps. Checked:
  * every token = the state oracle of its own prompt (scheduling only),
  * the long prompt's chunk sizes are the same with and without the switch (no new chunk boundaries — T3's constraint),
  * a short request sent during a one-chunk long prefill gets its first token before the long one, within ~period + layer (control: after),
  * a running decoder's longest token gap during the long prefill is ~one layer (control: the whole forward),
  * the long forward's state (fake Work rows [0, R)) survives every yield (park/unpark — the fake checks its rows after each yield),
  * size limit (HIVE_LAYER_YIELD_MAX ∩ runtime rows R), busy/same session, image requests, max_batch are respected,
  * H3 batch path (forward_multi with two long prompts) yields too; MTP decoder (FAKE_MTP) steps inside yields keep their oracle,
  * switch parsing (period/share/steps/max) absorbs junk values,
  * negative controls (each must be DETECTED): fake park disabled (FAKE_LY_NO_PARK) · hived admits beyond R (want under-reports and
    run ignores R) · long prompt's session not marked prefilling (same-session request admitted mid-forward) · want() without the
    queue scan (short request no longer admitted mid-forward),
  * source contract: runtime.cpp's park list covers tile_xchg's swap list (the "lives between layers" set) + cand + mh.
No GPU. Run: nice -n 19 taskset -c 4-5 python3 tools/test_layer_yield_cpu.py  (HIVE_TEST_SANITIZER=thread for TSAN)
"""
import os
from pathlib import Path
import re
import sys
import tempfile
import time

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import test_daemon_cpu as t  # noqa: E402

SLOW = t.SLOW
TILE3 = t.TILE3                      # max_chunk 256 × tile 3 = 768-row chunks · prefill_threshold 128
LAYERS, FWD_MS, PERIOD = 8, 800, 100  # fake forward = 8 layers × 100 ms; yield period 100 ms → every boundary when there is work
BASE = {'FAKE_LAYERS': str(LAYERS), 'FAKE_PREFILL_MS': str(FWD_MS * SLOW), 'FAKE_PREFILL_SHORT_MS': str(20 * SLOW)}
ON = {**BASE, 'HIVE_LAYER_YIELD': str(PERIOD)}
LY_RE = re.compile(r'^\[hived\] layer yield: (\S+) · prefill ([\d.]+) ms since resume · rows (\d+) · active (\d+)$')
LYDONE_RE = re.compile(r'^\[hived\] layer yield done: admitted (\d+) · decode steps (\d+) · ([\d.]+) ms$')
ADMIT_RE = re.compile(r'^\[hived\] prefill yield: (\d+) short requests? admitted in ([\d.]+) ms \(active \d+ · prefilling \d+\) · layer(?: · (\d+) by the mid rule)?$')
CHUNK_RE = re.compile(r'^\[hived\] (\S+): prefill chunk (\d+) M (\d+) · .* · ([\d.]+) ms(?: · layer yields (\d+) last ([\d.]+) ms)?$')


def lines(h):
    return h.tail(1000000).splitlines()


def n_yields(h):
    return sum(1 for l in lines(h) if LY_RE.match(l))


def n_admits(h):
    return sum(int(m[1]) for m in (ADMIT_RE.match(l) for l in lines(h)) if m)


def chunk_sizes(h, sid):
    return [int(m[3]) for m in (CHUNK_RE.match(l) for l in lines(h)) if m and m[1] == sid]


def staggered(h, long_ids, shorts, long_sid='L', n=3, delay=.25):
    """long prompt first; `shorts` [(sid, ids, kw)] sent `delay` s later (inside the long prompt's first forward). Returns results, first-token and send times."""
    res, first, sent, ths = {}, {}, {}, []
    sent[long_sid] = time.monotonic()
    ths.append(t.launch(h, long_sid, long_ids, n, res, long_sid, on_token=lambda k: first.setdefault(long_sid, time.monotonic())))
    time.sleep(delay * SLOW)
    for sid, ids, kw in shorts:
        sent[sid] = time.monotonic()
        ths.append(t.launch(h, sid, ids, n, res, sid, on_token=lambda k, sid=sid: first.setdefault(sid, time.monotonic()), **kw))
        time.sleep(.005)
    for th in ths:
        th.join(60 * SLOW)
    return res, first, sent


def sc_short(h, ck, on, tag):
    """One-chunk long prompt (700 rows ≤ 768) — the service shape (98K = one forward). A 60-row request sent 0.25 s in."""
    L, S = t.tokens(700, 1100), t.tokens(60, 1101)
    a0 = n_admits(h)
    res, first, sent = staggered(h, L, [('S', S, {})])
    t.done_ok(ck, f'{tag} long', res['L'], L, 3, 0)
    t.done_ok(ck, f'{tag} short', res['S'], S, 3, 0)
    ck.eq(f'{tag}: long chunk sizes unchanged (one 700-row forward)', chunk_sizes(h, 'L'), [700])
    ttft = first.get('S', 1e9) - sent['S']
    before_long = first.get('L', 0) - first.get('S', 1e9)
    if on:
        ck.true(f'{tag}: short first token inside the long forward', before_long > .2 * SLOW, f'{before_long:.3f}s {first!r}')
        ck.true(f'{tag}: short TTFT ≤ period + layer + slack', ttft < (PERIOD + FWD_MS / LAYERS + 250) / 1000 * SLOW, f'{ttft:.3f}s')
        ck.eq(f'{tag}: one layer-yield admission', n_admits(h) - a0, 1)
    else:
        ck.true(f'{tag} control: short waits for the long forward', before_long < .05 * SLOW, f'{before_long:.3f}s')
        ck.true(f'{tag} control: TTFT ≥ rest of the long forward', ttft > (FWD_MS * .5) / 1000 * SLOW, f'{ttft:.3f}s')
        ck.eq(f'{tag} control: no layer yield line', n_yields(h), 0)
    nxt = S + res['S']['tokens'] + t.tokens(10, 1102)
    t.done_ok(ck, f'{tag}: short turn 2 reuses its live state', h.generate('S', nxt, 3), nxt, 3, len(S) + 2)
    return ttft


def sc_decoder(h, ck, on, tag, sid='D', mtp=False):
    """A running decoder: its longest token gap while a one-forward long prefill runs."""
    D = t.tokens(100, 1110 + len(sid))
    times, res = [], {}
    th = t.launch(h, sid, D, 3000, res, sid, on_token=lambda k: times.append(time.monotonic()))
    deadline = time.monotonic() + 10 * SLOW
    while len(times) < 5 and time.monotonic() < deadline:
        time.sleep(.01)
    L = t.tokens(700, 1120 + len(sid))
    t0 = time.monotonic()
    r = t.launch(h, sid + 'L', L, 3, res, 'L')
    r.join(60 * SLOW)
    t1 = time.monotonic()
    h.cancel(sid)
    th.join(60 * SLOW)
    t.done_ok(ck, f'{tag} long with a decoder', res['L'], L, 3, 0)
    ck.true(f'{tag}: decoder ok', res[sid]['error'] is None, repr(res[sid]['error']))
    ck.eq(f'{tag}: decoder tokens follow its oracle', res[sid]['tokens'], t.oracle(D, len(res[sid]['tokens'])))
    win = [x for x in times if t0 <= x <= t1]
    gaps = [b - a for a, b in zip([t0] + win, win + [t1])]
    gmax = max(gaps) if gaps else t1 - t0
    if on:
        ck.true(f'{tag}: decoder kept stepping inside the long forward', len(win) >= LAYERS // 2, f'{len(win)} tokens')
        ck.true(f'{tag}: longest decoder gap ≈ one layer (not the forward)', gmax < (FWD_MS / LAYERS + PERIOD + 200) / 1000 * SLOW, f'{gmax:.3f}s')
    else:
        ck.true(f'{tag} control: decoder stalls for the whole forward', gmax > FWD_MS * .8 / 1000 * SLOW, f'{gmax:.3f}s')
    return gmax, t1 - t0


def sc_mid(h, ck, tag, nested=True):
    """HIVE_LAYER_YIELD_MID (300 here; default max 127): a 250-row request arriving 0.25 s into a 700-row forward is admitted at a yield
    (2 x 250 <= 700 remaining rows); its own 800 ms forward yields once more for decode only, so a running decoder's longest gap stays about one
    layer. Control: behind a 400-row forward the same request waits (2 x 250 > 400)."""
    D = t.tokens(100, 1170)
    times, res = [], {}
    th = t.launch(h, 'D', D, 3000, res, 'D', on_token=lambda k: times.append(time.monotonic()))
    deadline = time.monotonic() + 10 * SLOW
    while len(times) < 5 and time.monotonic() < deadline:
        time.sleep(.01)
    mid0 = sum(int(m[3] or 0) for m in (ADMIT_RE.match(l) for l in lines(h)) if m)
    L, M = t.tokens(700, 1171), t.tokens(250, 1172)
    t0 = time.monotonic()
    r2, first, sent = staggered(h, L, [('M', M, {})], long_sid='ML')
    t1 = time.monotonic()
    res.update(r2)
    t.done_ok(ck, f'{tag} long', res['ML'], L, 3, 0)
    t.done_ok(ck, f'{tag} 250-row request', res['M'], M, 3, 0)
    mid1 = sum(int(m[3] or 0) for m in (ADMIT_RE.match(l) for l in lines(h)) if m)
    ck.eq(f'{tag}: one admission by the mid rule', mid1 - mid0, 1)
    ck.true(f'{tag}: 250-row request first token before the long one', first.get('M', 1e9) < first.get('ML', 0), repr(first))
    win = [x for x in times if t0 <= x <= t1]
    gaps = [b - a for a, b in zip([t0] + win, win + [t1])]
    gmax = max(gaps) if gaps else t1 - t0
    lim = (FWD_MS / LAYERS + PERIOD + 250) / 1000 * SLOW
    if nested:
        ck.true(f'{tag}: decoder gap stays about one layer while the admitted request prefills (level-2 yield)', gmax < lim, f'{gmax:.3f}s')
        ck.true(f'{tag}: level-2 (decode only) yield lines', any('level 2 (decode only)' in l for l in lines(h)), '')
    # control: a shorter paused forward — the same-size request waits for it
    L2, M2 = t.tokens(400, 1173), t.tokens(250, 1174)
    r3, f3, _ = staggered(h, L2, [('M2', M2, {})], long_sid='ML2')
    t.done_ok(ck, f'{tag} control long', r3['ML2'], L2, 3, 0)
    t.done_ok(ck, f'{tag} control request', r3['M2'], M2, 3, 0)
    mid2 = sum(int(m[3] or 0) for m in (ADMIT_RE.match(l) for l in lines(h)) if m)
    ck.eq(f'{tag} control: not admitted (2 x 250 > 400 remaining rows)', mid2 - mid1, 0)
    h.cancel('D')
    th.join(60 * SLOW)
    ck.true(f'{tag}: decoder ok', res['D']['error'] is None, repr(res['D']['error']))
    ck.eq(f'{tag}: decoder tokens follow its oracle', res['D']['tokens'], t.oracle(D, len(res['D']['tokens'])))
    return gmax


def sc_reuse(h, ck, tag):
    """A follow-up turn of a cached conversation: 300 ids + its 3-token answer + 50 new ids = 353 prompt ids (> default max 127) but only
    51 rows to prefill (the live state covers 302) → admitted mid-forward. Control: the same 353 ids in a fresh session → waits."""
    X = t.tokens(300, 1140)
    r0 = h.generate('X', X, 3)
    t.done_ok(ck, f'{tag} warm turn', r0, X, 3, 0)
    follow = X + r0['tokens'] + t.tokens(50, 1141)
    a0 = n_admits(h)
    L = t.tokens(700, 1142)
    res, first, sent = staggered(h, L, [('X', follow, {})], long_sid='L5')
    t.done_ok(ck, f'{tag} long', res['L5'], L, 3, 0)
    t.done_ok(ck, f'{tag} follow-up turn reuses the live state', res['X'], follow, 3, len(X) + 2)
    ck.eq(f'{tag}: follow-up turn admitted mid-forward (353 ids, 51 rows to prefill)', n_admits(h) - a0, 1)
    ck.true(f'{tag}: follow-up first token inside the long forward', first.get('L5', 0) - first.get('X', 1e9) > .2 * SLOW, repr(first))
    a1 = n_admits(h)
    L2 = t.tokens(700, 1143)
    res, first, sent = staggered(h, L2, [('Y', follow, {})], long_sid='L6')
    t.done_ok(ck, f'{tag} long (control)', res['L6'], L2, 3, 0)
    t.done_ok(ck, f'{tag} same ids in a fresh session', res['Y'], follow, 3, 0)
    ck.eq(f'{tag} control: fresh session (353 rows) not admitted mid-forward', n_admits(h) - a1, 0)


def sc_small_fwd(h, ck, on, tag, sid='E'):
    """HIVE_LAYER_YIELD_SMALL: a running decoder's longest gap while a 100-row prompt (below the 128 threshold; FAKE_PREFILL_SHORT_MS =
    FWD_MS here) is prefilled — on: ≈ one layer; off (switch unset, layer yield still on): the whole forward."""
    D = t.tokens(100, 1150 + len(sid))
    times, res = [], {}
    th = t.launch(h, sid, D, 3000, res, sid, on_token=lambda k: times.append(time.monotonic()))
    deadline = time.monotonic() + 10 * SLOW
    while len(times) < 5 and time.monotonic() < deadline:
        time.sleep(.01)
    P = t.tokens(100, 1160 + len(sid))
    t0 = time.monotonic()
    r = t.launch(h, sid + 'P', P, 3, res, 'P')
    r.join(60 * SLOW)
    t1 = time.monotonic()
    h.cancel(sid)
    th.join(60 * SLOW)
    t.done_ok(ck, f'{tag} short prompt with a decoder', res['P'], P, 3, 0)
    ck.eq(f'{tag}: short prompt = one 100-row forward', chunk_sizes(h, sid + 'P'), [100])
    ck.true(f'{tag}: decoder ok', res[sid]['error'] is None, repr(res[sid]['error']))
    ck.eq(f'{tag}: decoder tokens follow its oracle', res[sid]['tokens'], t.oracle(D, len(res[sid]['tokens'])))
    win = [x for x in times if t0 <= x <= t1]
    gaps = [b - a for a, b in zip([t0] + win, win + [t1])]
    gmax = max(gaps) if gaps else t1 - t0
    if on:
        ck.true(f'{tag}: decoder kept stepping inside the short forward', len(win) >= LAYERS // 2, f'{len(win)} tokens')
        ck.true(f'{tag}: longest decoder gap ≈ one layer', gmax < (FWD_MS / LAYERS + PERIOD + 200) / 1000 * SLOW, f'{gmax:.3f}s')
    else:
        ck.true(f'{tag} control: decoder stalls for the whole short forward', gmax > FWD_MS * .8 / 1000 * SLOW, f'{gmax:.3f}s')
    return gmax


def sc_limits(h, ck, tag, admit_200):
    """200-row request (> default max 127): admitted only with HIVE_LAYER_YIELD_MAX ≥ 200 (and R covers it). Same session: never mid-forward.
    Image request: never mid-forward."""
    a0 = n_admits(h)
    L, M = t.tokens(700, 1130), t.tokens(200, 1131)
    res, first, sent = staggered(h, L, [('M', M, {})], long_sid='L2')
    t.done_ok(ck, f'{tag} long', res['L2'], L, 3, 0)
    t.done_ok(ck, f'{tag} 200-row request', res['M'], M, 3, 0)
    ck.eq(f'{tag}: 200-row request admitted mid-forward = {admit_200}', n_admits(h) - a0, 1 if admit_200 else 0)
    if admit_200:
        rows = [int(m[3]) for m in (LY_RE.match(l) for l in lines(h)) if m]
        ck.true(f'{tag}: runtime parked ≥ 200 rows for that yield', any(r >= 200 for r in rows), repr(rows[-5:]))
    # same session as the running long prefill
    a1 = n_admits(h)
    L3, Q = t.tokens(700, 1132), t.tokens(40, 1133)
    res = {}
    ths = [t.launch(h, 'L3', L3, 3, res, 'long')]
    time.sleep(.25 * SLOW)
    ths.append(t.launch(h, 'L3', Q, 3, res, 'follower'))
    for th in ths:
        th.join(60 * SLOW)
    t.done_ok(ck, f'{tag} long (same-session follower)', res['long'], L3, 3, 0)
    f = res['follower']
    ck.true(f'{tag}: same-session follower = session busy or its own oracle', f['error'] == 'session busy' or
            (f['error'] is None and f['tokens'] == t.oracle(Q, 3)), repr(f))
    ck.eq(f'{tag}: same-session request not admitted mid-forward', n_admits(h) - a1, 0)
    # image request: waits (vision lazy load would allocate VRAM mid-prefill in production)
    a2 = n_admits(h)
    L4 = t.tokens(700, 1134)
    im = t.image(2, 2, 3, 1135)
    I = t.tokens(30, 1136)
    res, first, sent = staggered(h, L4, [('I', I, {'images': [im]})], long_sid='L4')
    t.done_ok(ck, f'{tag} long (image follower)', res['L4'], L4, 3, 0)
    t.done_ok(ck, f'{tag} image request', res['I'], I, 3, 0, images=[im])
    ck.eq(f'{tag}: image request not admitted mid-forward', n_admits(h) - a2, 0)


def sc_batch(h, ck, tag):
    """H3: two long prompts in one forward_multi + a short one mid-round."""
    A, B, S = t.tokens(400, 1140), t.tokens(380, 1141), t.tokens(50, 1142)
    a0 = n_admits(h)
    res, first, ths = {}, {}, []
    blocker = t.tokens(150, 1143)
    ths.append(t.launch(h, 'bz', blocker, 3, res, 'bz'))   # queue A and B behind a forward so one round takes both
    time.sleep(.15 * SLOW)
    ths.append(t.launch(h, 'A', A, 3, res, 'A', on_token=lambda k: first.setdefault('A', time.monotonic())))
    ths.append(t.launch(h, 'B', B, 3, res, 'B'))
    time.sleep(1.2 * SLOW)  # blocker done, A+B round started
    ths.append(t.launch(h, 'S', S, 3, res, 'S', on_token=lambda k: first.setdefault('S', time.monotonic())))
    for th in ths:
        th.join(60 * SLOW)
    for k, ids in (('bz', blocker), ('A', A), ('B', B), ('S', S)):
        t.done_ok(ck, f'{tag} {k}', res[k], ids, 3, 0)
    multi = [l for l in lines(h) if l.startswith('[fake] forward_multi parts 2 ')]
    ck.true(f'{tag}: A and B prefilled in one forward_multi', bool(multi), '')
    ck.eq(f'{tag}: short admitted inside the forward_multi', n_admits(h) - a0, 1)
    ck.true(f'{tag}: short first token before A', first.get('A', 0) > first.get('S', 1e9), repr(first))
    rl = [l for l in lines(h) if l.startswith('[hived] batch prefill round') and ' · layer yields ' in l]
    ck.true(f'{tag}: round line carries the layer-yield tail', bool(rl), '')


WINDOW_RE = re.compile(r'^\[hived\] batch window: \+(\d+) requests in ([\d.]+) ms \((\S+) · (\d+) jobs · slots left (-?\d+)\)$')


def burst(h, n=3, rows=200, gap=.010, seed=1170):
    """n streaming-size requests (rows ≥ threshold 128, one slot each) sent `gap` s apart — the server hands a concurrent burst to hived ~10 ms apart
    (observed in a production hived.log: arrivals 8·10·8 ms / 11·9·10 ms apart). Returns results and the round sizes of the first rounds."""
    res, ths, ids = {}, [], {}
    r0 = len([l for l in lines(h) if l.startswith('[hived] batch prefill round')])
    for k in range(n):
        ids[f'B{seed}{k}'] = t.tokens(rows + 3 * k, seed + k)
    for sid, v in ids.items():
        ths.append(t.launch(h, sid, v, 1, res, sid))
        time.sleep(gap)
    for th in ths:
        th.join(60 * SLOW)
    rounds = [int(m[1]) for m in (t.BATCH_ROUND_RE.match(l) for l in lines(h)[0:]) if m][r0:]
    return res, ids, rounds


def sc_burst(h, ck, tag, want_batched):
    """T11b: a concurrent burst must be prefilled in ONE forward_multi round (as layer-yield-off happened to do when its first add was slow).
    want_batched False = reproduces the pre-fix race (first request alone, the rest one forward later)."""
    res, ids, rounds = burst(h)
    for sid, v in ids.items():
        t.done_ok(ck, f'{tag} burst {sid}', res[sid], v, 1, 0)
    solo = [l for l in lines(h) if l.startswith('[hived] B1170') and ': prefill chunk' in l]
    batched = bool(rounds) and rounds[0] == len(ids)
    if want_batched:
        ck.true(f'{tag}: burst of {len(ids)} prefilled in one round', batched, f'rounds {rounds}')
        w = [m for m in (WINDOW_RE.match(l) for l in lines(h)) if m]
        ck.true(f'{tag}: coalesce window logged (+{len(ids) - 1}, quiet/full)', any(int(m[1]) == len(ids) - 1 and m[3] in ('quiet', 'full') for m in w), repr([m[0] for m in w][-3:]))
        # a lone streaming request pays at most ~the quiet gap
        L = t.tokens(300, 1180)
        t0 = time.monotonic()
        t.done_ok(ck, f'{tag} lone streaming request', h.generate('lone', L, 1), L, 1, 0)
        w = [m for m in (WINDOW_RE.match(l) for l in lines(h)) if m]
        ck.true(f'{tag}: lone request waited ≈ quiet gap only', w and w[-1][3] == 'quiet' and int(w[-1][1]) == 0 and float(w[-1][2]) < 60 * SLOW, repr(w[-1][0] if w else None))
    else:
        ck.true(f'{tag} control: burst NOT prefilled in one round (pre-fix race)', not batched, f'rounds {rounds} {len(solo)}')
    return rounds


def sc_parse(exe, td, ck):
    """Switch parsing: junk values are absorbed (defaults), "0" is off."""
    cases = (('0', None), ('1', '500'), ('on', '500'), ('49', '500'), ('75', '75'), ('1e9', '60000'))
    for v, want in cases:
        h = t.Hived(exe, td, {'HIVE_LAYER_YIELD': v, 'HIVE_LAYER_YIELD_SHARE': 'x', 'HIVE_LAYER_YIELD_STEPS': '0', 'HIVE_LAYER_YIELD_MAX': '9q'},
                    name='lyparse')
        try:
            log = h.tail(100000)
            m = re.search(r'\[hived\] layer yield on: every (\d+) ms .* decode share ([\d.]+) · <= (\d+) steps · admits requests with <= (\d+) rows to prefill', log)
            if want is None:
                ck.true(f'parse {v!r}: off (no startup line)', m is None, log[-300:])
            else:
                ck.true(f'parse {v!r}: on', m is not None, log[-300:])
                if m:
                    ck.eq(f'parse {v!r}: period', m[1], want)
                    ck.eq(f'parse {v!r}: share junk → 0.05', m[2], '0.05')
                    ck.eq(f'parse {v!r}: steps 0 → 4', m[3], '4')
                    ck.eq(f'parse {v!r}: max junk → threshold − 1', m[4], '1023')
        finally:
            t.stop_clean(h, ck)


def source_contract(ck):
    """runtime.cpp park list ⊇ tile_xchg swap list (+ cand, mh) and the scalar list covers tile_xchg's globals and UnitCtx's."""
    rt = (t.harness.ROOT / 'engine/src/runtime.cpp').read_text()
    x = rt[rt.index('void Runtime::tile_xchg(int s) {'):]
    x = x[:x.index('\n}\n')]
    swapped = set(re.findall(r'std::swap\(w\.(\w+), t\.\w+\)', x))
    g_swapped = set(re.findall(r'std::swap\((\w+_), t\.\w+\)', x))
    p = rt[rt.index('void Runtime::layer_yield_point(int kind, int layer, bool last) {'):]
    p = p[:p.index('\n}\n')]
    park = p[:p.index('auto unpark')]
    parked = set(re.findall(r'\b(?:dv|hb)\(w\.(\w+)', park)) | set(re.findall(r'P\.dev\.push_back\(\{w\.(\w+)\.p', park))
    # device pointer twins (route_ids_d …) alias the mapped host table — parking the host side covers them
    need = {s for s in swapped if not s.endswith('_d')} | {'cand', 'mh'}
    ck.true('contract: park list ⊇ tile_xchg swap list + cand + mh', need <= parked, f'missing {sorted(need - parked)}')
    for g in sorted(g_swapped) + ['shared_comp_kv_', 'shared_index_k_', 'have_candidates_', 'win_min_pos_', 'mh_n_', 'mh_pos0_', 'in_prefill_',
                                  'score_prefill_', 'small_fwd_', 'prefill_pending_', 'last_rows_']:
        ck.true(f'contract: {g} saved and restored', f'= {g};' in park and re.search(rf'\b{g} = (std::move\()?P\.', p) is not None, '')
    lend = rt[rt.index('static void lend_if_idle(Runtime& r) {'):]
    lend = lend[:lend.index('\n  }\n')]
    ck.true('contract: no elastic lending inside a layer yield', 'r.ly_.depth > 0' in lend, '')
    hv = (t.harness.ROOT / 'engine/src/hived.cpp').read_text()
    w = hv[hv.index('auto warm_after_prefill = [&](size_t self_active) {'):]  # self_active: HIVE_WARM_BUSY_CAP (the deferred warm counts its own request in active)
    ck.true('contract: no warm_cache inside a layer yield', 'if (rt.in_layer_yield()) return;' in w[:w.index('};')], '')


def run(exe, td):
    ck = t.Checker('t11-layer-yield')
    source_contract(ck)
    batch = {**t.BATCH}
    cases = (
        ('single-on', ON, TILE3, lambda h: (sc_short(h, ck, True, 'single'), sc_decoder(h, ck, True, 'single'), sc_limits(h, ck, 'single', False), sc_reuse(h, ck, 'single'))),
        ('small-on', {**ON, 'HIVE_LAYER_YIELD_SMALL': '1', 'FAKE_PREFILL_SHORT_MS': str(FWD_MS * SLOW)}, TILE3, lambda h: (sc_small_fwd(h, ck, True, 'small'), sc_decoder(h, ck, True, 'small+long'))),
        ('small-off', {**ON, 'FAKE_PREFILL_SHORT_MS': str(FWD_MS * SLOW)}, TILE3, lambda h: sc_small_fwd(h, ck, False, 'small-off')),
        ('single-off', BASE, TILE3, lambda h: (sc_short(h, ck, False, 'single-off'), sc_decoder(h, ck, False, 'single-off'))),
        ('max300', {**ON, 'HIVE_LAYER_YIELD_MAX': '300'}, TILE3, lambda h: sc_limits(h, ck, 'max300', True)),
        ('mid300', {**ON, 'HIVE_LAYER_YIELD_MID': '300'}, TILE3, lambda h: (sc_mid(h, ck, 'mid'), sc_short(h, ck, True, 'mid+short'))),
        ('batch-on', {**ON, **batch}, TILE3, lambda h: (sc_short(h, ck, True, 'batch'), sc_batch(h, ck, 'batch'), sc_decoder(h, ck, True, 'batch'))),
        ('mtp-on', {**ON, 'FAKE_MTP': '3', 'FAKE_MTP_DRAFT_MS': '1'}, TILE3, lambda h: sc_decoder(h, ck, True, 'mtp', mtp=True)),
        ('t3+t11', {**ON, 'HIVE_PREFILL_YIELD': '1'}, TILE3, lambda h: (sc_short(h, ck, True, 't3+t11'), sc_decoder(h, ck, True, 't3+t11'))),
        # T11b simultaneous-arrival coalescing: on (default 25 ms) = one round; HIVE_BATCH_COALESCE_MS=0 = reproduces the old contention (negative control)
        ('burst-on', {**ON, **batch}, TILE3, lambda h: (sc_burst(h, ck, 'burst', True), sc_short(h, ck, True, 'burst+short'))),
        ('burst-coalesce0', {**ON, **batch, 'HIVE_BATCH_COALESCE_MS': '0'}, TILE3, lambda h: sc_burst(h, ck, 'burst0', False)),
    )
    only = [x for x in os.environ.get('T11_CASES', '').split(',') if x]
    summary = {}
    for name, env, args, fn in cases:
        if only and name not in only:
            continue
        h = t.Hived(exe, td, env, args=args, name='t11' + name)
        try:
            out = fn(h)
            summary[name] = out
            on = 'HIVE_LAYER_YIELD' in env
            ck.true(f'{name}: startup line iff switch on', ('[hived] layer yield on:' in h.tail(100000)) == on)
            ck.true(f'{name}: no fake-state error in the log', 'fake state' not in h.tail(100000), '')
        finally:
            t.stop_clean(h, ck)
        print(f'  {name}: {ck.checks} checks so far, {len(ck.failures)} failures · {summary.get(name)!r}', flush=True)
    if not only:
        sc_parse(exe, td, ck)
    return ck


# ---- negative controls --------------------------------------------------------------------------------------------------
def mutant_case(exe, td, env, fn):
    ck = t.Checker('mutant')
    h = t.Hived(exe, td, env, args=TILE3, name='t11mut')
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
    # 1) park/unpark disabled in the fake runtime: the decoder (rows [0, floor)) clobbers the long forward's rows
    out.append(('fake park disabled', *mutant_case(exe, td, {**ON, 'FAKE_LY_NO_PARK': '1'}, lambda h, ck: sc_decoder(h, ck, True, 'np'))))
    # 2) hived admits beyond R: want() reports only decoders (R = floor 64) and run() ignores R → 200-row admission overwrites rows 64..199
    want_scan = "          for (const auto& r : queue) need = std::max(need, (int)std::min<size_t>(ly_rows_of(r, std::max(ly_max, ly_mid)), (size_t)INT32_MAX));"
    cap_line = "      const size_t cap = nested || kind != ly::kLayer ? 0 : std::min(std::max(ly_max, ly_mid), (size_t)std::max(0, R));"
    assert src.count(want_scan) == 1 and src.count(cap_line) == 1, 'hived T11 lines drifted'
    m2 = src.replace(want_scan, "          (void)ly_rows_of;").replace(cap_line, "      const size_t cap = nested || kind != ly::kLayer ? 0 : std::max(ly_max, ly_mid);")
    exe2 = t.build_daemon(td, name='hived_mut_rows', hived_source=m2)

    def beyond_r(h, ck):
        D = t.tokens(100, 1150)
        res = {}
        th = t.launch(h, 'D', D, 3000, res, 'D')
        time.sleep(.3 * SLOW)
        L, M = t.tokens(700, 1151), t.tokens(200, 1152)
        r2, _, _ = staggered(h, L, [('M', M, {})], long_sid='L')
        h.cancel('D')
        th.join(60 * SLOW)
        t.done_ok(ck, 'mut long', r2['L'], L, 3, 0)
        t.done_ok(ck, 'mut 200-row', r2['M'], M, 3, 0)
    out.append(('admission beyond parked rows', *mutant_case(exe2, td, {**ON, 'HIVE_LAYER_YIELD_MAX': '300'}, beyond_r)))
    # 3) the long prompt's session not marked prefilling: a same-session request is admitted mid-forward
    guard = "if (prefill_yield || layer_yield) prefilling_sids.insert(J.A->sid);"
    assert src.count(guard) == 1
    m3 = src.replace(guard, "if (prefill_yield) prefilling_sids.insert(J.A->sid);")
    exe3 = t.build_daemon(td, name='hived_mut_sid', hived_source=m3)

    def same_session(h, ck):
        L3, Q = t.tokens(700, 1160), t.tokens(40, 1161)
        res = {}
        ths = [t.launch(h, 'L3', L3, 3, res, 'long')]
        time.sleep(.25 * SLOW)
        ths.append(t.launch(h, 'L3', Q, 3, res, 'follower'))
        for th in ths:
            th.join(60 * SLOW)
        t.done_ok(ck, 'mut long', res['long'], L3, 3, 0)
        ck.eq('mut: same-session not admitted mid-forward', n_admits(h), 0)
    out.append(('long session not marked prefilling', *mutant_case(exe3, td, ON, same_session)))
    # 4) want() without the queue scan: with no decoder there is nothing "to do" → the short request waits for the whole forward
    m4 = src.replace(want_scan, "          (void)ly_rows_of;")
    exe4 = t.build_daemon(td, name='hived_mut_want', hived_source=m4)
    out.append(('want() ignores the queue', *mutant_case(exe4, td, ON, lambda h, ck: sc_short(h, ck, True, 'mut-want'))))
    # 5) T11b removed (the window loop never runs for coalescing): a concurrent burst is split again
    w_line = "    bool window_done = !(batch_window_ms > 0 || batch_quiet_ms > 0);  // arrival coalescing uses the same window loop"
    assert src.count(w_line) == 1
    exe5 = t.build_daemon(td, name='hived_mut_burst', hived_source=src.replace(w_line, "    bool window_done = !(batch_window_ms > 0);"))
    out.append(('no arrival coalescing', *mutant_case(exe5, td, {**ON, **t.BATCH}, lambda h, ck: sc_burst(h, ck, 'mut-burst', True))))
    # 6) no level-2 yield (HIVE_LAYER_YIELD_MID admits, but the admitted forward never yields): the decoder stalls for that whole forward
    d_line = "    h.max_depth = ly_mid > ly_max ? 2 : 1;"
    assert src.count(d_line) == 1
    exe6 = t.build_daemon(td, name='hived_mut_depth', hived_source=src.replace(d_line, "    h.max_depth = 1;"))
    out.append(('no level-2 yield under the mid rule', *mutant_case(exe6, td, {**ON, 'HIVE_LAYER_YIELD_MID': '300'}, lambda h, ck: sc_mid(h, ck, 'mut-depth'))))
    return out


def main():
    with tempfile.TemporaryDirectory(prefix='hive-t11-') as td:
        t0 = time.monotonic()
        exe = t.build_daemon(td)
        print(f'built real hived.cpp on fake CUDA/runtime in {time.monotonic() - t0:.1f}s (sanitizer {t.harness.sanitizer()})', flush=True)
        ck = run(exe, td)
        if os.environ.get('T11_MUTANT', '1') != '0' and not os.environ.get('T11_CASES'):
            for name, detected, why in run_mutants(exe, td):
                ck.checks += 1
                if not detected:
                    ck.failures.append(f'mutant "{name}" was NOT detected')
                print(f'  mutant ({name}): {"DETECTED" if detected else "NOT DETECTED"} {why}', flush=True)
    for f in ck.failures:
        print('FAIL', f)
    print(f't11 layer yield: {ck.checks} checks, {len(ck.failures)} failures')
    sys.exit(1 if ck.failures else 0)


if __name__ == '__main__':
    main()
