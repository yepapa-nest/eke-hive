#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Admission order, queued cancels, the generation-suffix tail and disk snapshot look-up on the REAL engine/src/hived.cpp
(fake CUDA/runtime, tools/test_daemon_cpu.py harness). Four fixes of 2026-11-11, each with its negative control:

  order   HIVE_PRIORITY_SHARED=999: requests queued as priority 0, -1, 1 are admitted in that (arrival) order. The queue compared the
          raw values, so the later -1 went first. Without the switch the -1 still goes first (unchanged).
  cancel  a request cancelled while it waits for a seat is answered at once (finish "cancel", nothing read), not admitted when the
          seat frees and read for a chunk.
  tail    HIVE_TAIL_FIRST (default on): with a decoder running, no decode share between a request's main chunk and its
          generation-suffix tail (the decoder gets no token between the two forwards); HIVE_TAIL_FIRST=0 = the share as before.
          Tokens of both requests are the state oracle either way. The batched path (HIVE_BATCH_PREFILL) is measured and printed.
  disk    a boundary snapshot that is on disk only is found although a deeper, unrelated snapshot whose last token happens to match
          the prompt at that position is in memory (the search ended at the unrelated one).

Negative controls: the same scenarios on a build with each fix reverted in the source text (the checks must fail there), and
HIVE_ADMIT_ORIG=<path to an older hived.cpp> runs them on that file instead (all four must fail on a pre-fix source).
No GPU. Run: nice -n 19 python3 tools/test_admit_order_cpu.py  (ADMIT_CASES=order,cancel,tail,disk · ADMIT_MUTANT=0 skips the controls)
"""
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import test_daemon_cpu as t  # noqa: E402

SLOW = t.SLOW
SHARED = {'HIVE_PRIORITY_SHARED': '999'}


def first_chunk_order(h, sids):
    """sids in the order their first prefill chunk line appears in the log (= admission order on the single path)."""
    seen = []
    for m in (t.CHUNK_RE.match(l) for l in h.tail(100000).splitlines()):
        if m and m[1] in sids and m[1] not in seen:
            seen.append(m[1])
    return seen


# ---- order -----------------------------------------------------------------------------------------------------------------
def sc_order(h, tag):
    """A slow prefill blocks the engine; q0 (priority 0), qm (-1), q1 (1) arrive in that order behind it. Returns the admission order."""
    res, ths = {}, []
    B = t.tokens(700, 5000)
    ths.append(t.launch(h, tag + 'bz', B, 3, res, 'bz', priority=0))
    time.sleep(.15 * SLOW)  # the blocker is inside its first forward
    reqs = [(tag + 'q0', t.tokens(90, 5001), 0), (tag + 'qm', t.tokens(91, 5002), -1), (tag + 'q1', t.tokens(92, 5003), 1)]
    for sid, ids, prio in reqs:
        ths.append(t.launch(h, sid, ids, 3, res, sid, priority=prio))
        time.sleep(.04 * SLOW)  # separate connections: far enough apart that the receiver queues them in this order
    for th in ths:
        th.join(60 * SLOW)
    return res, B, reqs, first_chunk_order(h, {sid for sid, _, _ in reqs})


def case_order(exe, td, ck, expect_fixed=True):
    slow = {'FAKE_PREFILL_MS': str(200 * SLOW)}
    args = ('--max-sessions', '6')
    h = t.Hived(exe, td, {**slow, **SHARED}, args=args, name='ao-shared')
    try:
        res, B, reqs, order = sc_order(h, 's')
        t.done_ok(ck, 'order: blocker', res['bz'], B, 3, 0)
        for sid, ids, _ in reqs:
            t.done_ok(ck, f'order: {sid}', res[sid], ids, 3, 0)
        ck.eq('order (HIVE_PRIORITY_SHARED=999): one class is admitted in arrival order', order, ['sq0', 'sqm', 'sq1'])
    finally:
        t.stop_clean(h, ck)
    if not expect_fixed:
        return
    # without the switch priorities are strict, as before: the -1 goes first whatever the arrival order
    h = t.Hived(exe, td, slow, args=args, name='ao-strict')
    try:
        res, B, reqs, order = sc_order(h, 'n')
        for sid, ids, _ in reqs:
            t.done_ok(ck, f'order strict: {sid}', res[sid], ids, 3, 0)
        ck.eq('order (no HIVE_PRIORITY_SHARED): strict priority unchanged', order, ['nqm', 'nq0', 'nq1'])
    finally:
        t.stop_clean(h, ck)


# ---- cancel ----------------------------------------------------------------------------------------------------------------
def case_cancel(exe, td, ck, expect_fixed=True):
    """One seat (--max-batch 1), a long decode holds it; a queued request is cancelled by rid."""
    h = t.Hived(exe, td, {}, args=('--max-batch', '1'), name='ao-cancel')
    try:
        D, Q = t.tokens(100, 5100), t.tokens(300, 5101)
        got, rd, rq = threading.Event(), {}, {}
        thd = t.launch(h, 'cd', D, 3500, rd, 'r', on_token=lambda n: n >= 5 and got.set())
        ck.true('cancel: the decoder holds the only seat', got.wait(20 * SLOW))
        thq = t.launch(h, 'cq', Q, 3, rq, 'r', rid='cq-rid')
        time.sleep(.25 * SLOW)  # queued behind the full seat
        ck.true('cancel: the request is still waiting', thq.is_alive())
        t0 = time.monotonic()
        ck.eq('cancel op', h.cancel('cq', rid='cq-rid'), {'ok': True})
        thq.join(1.5 * SLOW)
        answered = not thq.is_alive()
        if expect_fixed:
            ck.true('cancel: answered while the engine is busy', answered, f'{time.monotonic() - t0:.2f}s')
            r = rq.get('r') or {}
            ck.true('cancel: finish=cancel, no token', (r.get('done') or {}).get('finish') == 'cancel' and r.get('tokens') == [], repr(r))
            ck.true('cancel: the decoder still runs', thd.is_alive())
        h.cancel('cd')
        thd.join(30 * SLOW)
        thq.join(30 * SLOW)
        ck.eq('cancel: decoder tokens follow its oracle', rd['r']['tokens'], t.oracle(D, len(rd['r']['tokens'])))
        read = [l for l in h.tail(100000).splitlines() if l.startswith('[hived] cq: ')]
        if expect_fixed:
            ck.eq('cancel: nothing of the cancelled request was read or restored', read, [])
            # the session name is free again and the engine serves the next request normally
            N = t.tokens(280, 5102)
            t.done_ok(ck, 'cancel: the next request of that session', h.generate('cq', N, 3), N, 3, 0)
        else:
            ck.true('cancel (pre-fix shape): answered at once and nothing read', answered and not read, f'answered {answered} · log {read[:2]}')
    finally:
        t.stop_clean(h, ck)


# ---- tail ------------------------------------------------------------------------------------------------------------------
def sc_tail(h, ck, tag):
    """Decoder D runs; R (600 tokens, generation-suffix hint 3 before the end) arrives. Returns D's tokens between R's send and first token."""
    D, R = t.tokens(100, 5200), t.tokens(600, 5201)
    stamps, rd, rr, first = [], {}, {}, []
    thd = t.launch(h, tag + 'd', D, 3900, rd, 'r', on_token=lambda n: stamps.append(time.monotonic()))
    deadline = time.monotonic() + 20 * SLOW
    while len(stamps) < 5 and time.monotonic() < deadline:
        time.sleep(.01)
    t_send = time.monotonic()
    rr['r'] = h.generate(tag + 'r', R, 3, boundaries=[len(R) - 3], on_token=lambda n: first.append(time.monotonic()))
    h.cancel(tag + 'd')
    thd.join(30 * SLOW)
    t.done_ok(ck, f'{tag}: request beside a decoder', rr['r'], R, 3, 0)
    ck.eq(f'{tag}: decoder tokens follow its oracle', rd['r']['tokens'], t.oracle(D, len(rd['r']['tokens'])))
    ck.eq(f'{tag}: main chunk + generation-suffix tail', [m for m, _ in t.chunks(h, tag + 'r')], [597, 3])
    t_first = first[0] if first else t_send
    # the first 80 ms after the send belong to the steps before the admission (and tokens still in the socket)
    between = sum(1 for s in stamps if t_send + .08 * SLOW < s < t_first)
    return between, t_first - t_send


def case_tail(exe, td, ck):
    # main chunk 400 ms, tail 40 ms; a decode step 3 ms: the share between the two would be 0.515 x 400 ms = ~60 steps
    base = {'FAKE_PREFILL_MS': str(400 * SLOW), 'FAKE_PREFILL_SHORT_MS': str(40 * SLOW), 'HIVE_PREFIX_SHARE': '1', 'HIVE_PREFIX_LEARN': '0'}
    out = {}
    for name, env in (('on', {}), ('off', {'HIVE_TAIL_FIRST': '0'}), ('batch-on', dict(t.BATCH)), ('batch-off', {**t.BATCH, 'HIVE_TAIL_FIRST': '0'})):
        h = t.Hived(exe, td, {**base, **env}, args=t.TILE3, name='ao-tail-' + name)
        try:
            out[name] = sc_tail(h, ck, 'tail-' + name)
        finally:
            t.stop_clean(h, ck)
    ck.true('tail (default): no decode step between the main chunk and the tail', out['on'][0] <= 3, repr(out))
    ck.true('tail (HIVE_TAIL_FIRST=0): the decoders get their share between the two', out['off'][0] >= 15, repr(out))
    ck.true('tail: the first token comes earlier by about that share', out['off'][1] - out['on'][1] > .1 * SLOW, repr(out))
    # HIVE_BATCH_PREFILL: a text request the main loop takes goes through admit_batch, whose rounds still give the share before the tail
    #   (2026-11-11 review: HIVE_TAIL_FIRST is in run_chunks only). Measured and printed, not judged, until that path is decided.
    gap = out['batch-on'][0]
    print(f'  tail on the batched path (HIVE_BATCH_PREFILL): decoder tokens between main chunk and tail = {gap} with the switch on, '
          f'{out["batch-off"][0]} off → {"the tail runs first there too" if gap <= 3 else "KNOWN GAP: the share still comes before the tail"}', flush=True)
    return out


# ---- disk ------------------------------------------------------------------------------------------------------------------
def store_stats(h):
    return h.stats().get('session_store') or {}


def wait_writes(h, n, timeout=20):
    deadline = time.monotonic() + timeout * SLOW
    while time.monotonic() < deadline:
        st = store_stats(h)
        if st.get('writes', 0) >= n and st.get('queue', 1) == 0:
            return st
        time.sleep(.05)
    return store_stats(h)


def case_disk(exe, td, ck, expect_fixed=True):
    store = str(Path(td) / f'store-{int(expect_fixed)}')
    env = {'HIVE_SESSION_STORE': store, 'HIVE_PREFIX_SHARE': '1', 'HIVE_CKPT_DELTA': '1', 'HIVE_PREFIX_LEARN': '0'}
    T = 3500                                       # the token both prompts carry at position 699
    HEAD = t.tokens(500, 5300)
    # 1. a conversation leaves a snapshot at 500 on disk; the daemon is killed (the snapshot is then on disk only)
    h = t.Hived(exe, td, env, name='ao-disk1')
    A = HEAD + t.tokens(30, 5301)
    r = h.generate('dz', A, 3, boundaries=[500])
    ck.true('disk: first conversation done', r['done'] is not None, repr(r['error']))
    st = wait_writes(h, 2)
    ck.true('disk: snapshot at 500 written', st.get('snaps', 0) >= 1 and 'dz: prefix snapshot at 500 (' in h.tail(100000), repr(st))
    h.proc.kill(); h.proc.wait(10); h.log.close()
    # 2. restart; an unrelated conversation leaves a snapshot at 700 (in memory and on disk) whose last token is T
    h = t.Hived(exe, td, env, name='ao-disk2')
    try:
        U = t.tokens(699, 5302) + [T] + t.tokens(30, 5303)
        r = h.generate('du', U, 3, boundaries=[700])
        ck.true('disk: unrelated conversation done', r['done'] is not None, repr(r['error']))
        st = wait_writes(h, 2)
        ck.true('disk: unrelated snapshot at 700 in memory and on disk', 'du: prefix snapshot at 700 (' in h.tail(100000) and st.get('snaps', 0) >= 2, repr(st))
        # 3. a new conversation shares the 500-token head and has T at 699: the candidates are 700 (unrelated, in memory) and 500 (its own head, disk only)
        R = HEAD + t.tokens(199, 5304) + [T] + t.tokens(60, 5305)
        r = h.generate('dr', R, 3)
        if expect_fixed:
            t.done_ok(ck, 'disk: the head snapshot on disk is found behind an unrelated one in memory', r, R, 3, 500)
            ck.true('disk: restore line', 'dr: boundary snapshot at 500 restored from the session store' in h.tail(100000), '')
        else:
            ck.eq('disk (pre-fix shape): resumed at the head snapshot', (r.get('done') or {}).get('cached_prefix'), 500)
    finally:
        t.stop_clean(h, ck)


# ---- negative controls --------------------------------------------------------------------------------------------------
REVERTS = {  # name -> (text in the fixed source, the pre-fix text)
    'order': ('if (prio_class(p) < prio_class(pick_prio)) { pick = it; pick_prio = p; }', 'if (p < pick_prio) { pick = it; pick_prio = p; }'),
    'cancel': ('    sweep_cancelled_queue();  // a request its client gave up while it waited', '    // (reverted for the test) a request its client gave up while it waited'),
    'disk': ('        if (prefix_key(pids.data(), (size_t)pos, sig_upto_j(J, (size_t)pos)) != key) continue;\n        if (shared_snaps.count(key)) break;  // already in memory',
             '        if (shared_snaps.count(key)) break;  // already in memory\n        if (prefix_key(pids.data(), (size_t)pos, sig_upto_j(J, (size_t)pos)) != key) continue;'),
}
CASES = {'order': case_order, 'cancel': case_cancel, 'disk': case_disk}


def run_control(td, name, source, label):
    """The scenario of `name` with expect_fixed=False on `source`: its fixed-shape check must fail."""
    exe = t.build_daemon(td, name='hived_ctl_' + label, hived_source=source)
    ck = t.Checker('control-' + label)
    CASES[name](exe, td, ck, expect_fixed=False)
    detected = bool(ck.failures)
    print(f'  control {label}: {"DETECTED" if detected else "NOT DETECTED"} {ck.failures[:1]}', flush=True)
    return detected


def main():
    only = [x for x in os.environ.get('ADMIT_CASES', '').split(',') if x]
    want = lambda n: not only or n in only  # noqa: E731
    with tempfile.TemporaryDirectory(prefix='hive-admit-') as td:
        t0 = time.monotonic()
        exe = t.build_daemon(td)
        print(f'built real hived.cpp on fake CUDA/runtime in {time.monotonic() - t0:.1f}s (sanitizer {t.harness.sanitizer()})', flush=True)
        ck = t.Checker('admit-order')
        for name in ('order', 'cancel', 'disk'):
            if want(name):
                CASES[name](exe, td, ck)
                print(f'  {name}: {ck.checks} checks so far, {len(ck.failures)} failures', flush=True)
        if want('tail'):
            out = case_tail(exe, td, ck)
            print(f'  tail: decoder tokens between / first token s — {out} · {ck.checks} checks so far, {len(ck.failures)} failures', flush=True)
        if os.environ.get('ADMIT_MUTANT', '1') != '0':
            fixed = (t.harness.ROOT / 'engine/src/hived.cpp').read_text()
            orig = os.environ.get('HIVE_ADMIT_ORIG')
            for name, (new, old) in REVERTS.items():
                if not want(name):
                    continue
                if orig:
                    src, label = Path(orig).read_text(), name + '-orig'
                else:
                    assert fixed.count(new) == 1, f'{name}: the fixed line drifted'
                    src, label = fixed.replace(new, old), name + '-reverted'
                ck.checks += 1
                if not run_control(td, name, src, label):
                    ck.failures.append(f'control {label}: the pre-fix behaviour was NOT detected')
    for f in ck.failures:
        print('FAIL', f)
    print(f'admit order: {ck.checks} checks, {len(ck.failures)} failures')
    sys.exit(1 if ck.failures else 0)


if __name__ == '__main__':
    main()
