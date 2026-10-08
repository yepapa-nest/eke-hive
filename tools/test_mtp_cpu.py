#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HIVE_MTP_GATE2 — the REAL engine/src/hived.cpp DSpark scheduling on fake CUDA/runtime.

tools/cpu_fake/fake_runtime.inc: FAKE_MTP=B turns on a fake DSpark whose drafts are the state oracle's own continuation
(FAKE_MTP_WRONG=N corrupts some), forward_verify = M single-row oracle forwards, rollback restores pos/tokens/history/ring
(a wrong rollback raises "fake state" or yields a wrong token). Timing is synthetic: FAKE_STEP_MS per normal decode step,
FAKE_VERIFY_MS + FAKE_VERIFY_ROW_MS·(M−1) per verify, FAKE_MTP_DRAFT_MS per draft.

Checks: (1) every config emits exactly the oracle tokens (MTP on/off, gate 1/2, wrong drafts, two concurrent sessions);
(2) gate 2 engages (most tokens come from speculative steps) where the default gate — stale verify table — does not;
(3) switch parsing: HIVE_MTP_GATE2 unset / "" / "0" = default gate (no gate-2 trace lines), "1" = gate 2;
(4) batch (two active sequences) never speculates, and speculation resumes once one sequence remains.
(5) HIVE_MTP_BATCH=1 — two/three concurrent sessions speculate TOGETHER (forward_verify_batch, one part per session; the fake
checks the contract: distinct sequences, rows <= mtp_batch_rows()) and still emit exactly the oracle tokens, with exact and corrupted drafts;
unset/""/"0" = no batch speculation (switch parsing); HIVE_MTP_GATE3 on top of gate 2 keeps oracle tokens (the fake never captures graphs, so
the filter is a no-op there — its effect is a GPU measurement) and GATE3 alone does not turn gate 2 on.
(6) first-use verify samples (production: run_graph's first use of a graph key runs eagerly — FAKE_VERIFY_FIRST_MS adds that cost to the
first batch verify of each row count and bumps graph_eager_) must not lock batch speculation out: with the filter the pair keeps speculating; a runtime
that hides first uses (FAKE_VERIFY_FIRST_SILENT=1 — what the pre-MB1 gate saw) locks out after one verify (negative control, observed through the same
real hived.cpp). Mutants (HIVE_MTP_NEGATIVE=1) add: batch rollback one row short · the first-use filter removed.
Certifies host logic only; model numerics / real verify costs are GPU checks (see the M1/R1 reports).
"""
import os
from pathlib import Path
import re
import sys
import tempfile
import threading

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import test_daemon_cpu as T  # noqa: E402

TIMING = {'FAKE_STEP_MS': '20', 'FAKE_VERIFY_MS': '20', 'FAKE_VERIFY_ROW_MS': '3', 'FAKE_MTP_DRAFT_MS': '2'}


def run(exe, td, name, env, ck, prompts, n_tok=60, args=()):
    h = T.Hived(exe, td, {**TIMING, 'HIVE_TRACE_MTP': '1', **env}, args=args, name=name)
    res = []
    try:
        for i, ids in enumerate(prompts):
            r = h.generate(f'{name}-s{i}', ids, n_tok)
            T.done_ok(ck, f'{name}[{i}]', r, ids, n_tok, None)
            res.append(r)
        stats = h.stats()
    finally:
        T.stop_clean(h, ck)
    log = h.log_path.read_text(errors='replace')
    return res, stats, log


def mtp_tokens(res):
    return sum(r['done']['mtp_steps'] + r['done']['mtp_accepted'] for r in res if r['done'])


MUTANTS = [('rollback keeps one row too few', 'rt.rollback(seq, n_keep);', 'rt.rollback(seq, n_keep > 1 ? n_keep - 1 : n_keep);'),
           ('accept every draft', 'if (t != drafts[i]) { nxt = t; break; }', 'if (false) { nxt = t; break; }'),
           # batch path (HIVE_MTP_BATCH): the negative controls run with two concurrent sessions (run_mutants)
           ('batch: accept every draft', 'if (t != bdr[s][i]) { bnxt[s] = t; break; }', 'if (false) { bnxt[s] = t; break; }'),
           ('batch: rows of the wrong part', 'const int32_t t = sample_at(brow0[s] + i);', 'const int32_t t = sample_at(brow0[S - 1 - s] + i);'),
           # batch rollback and first-use filter
           ('batch: rollback keeps one row too few', 'rt.rollback_batch(bkeep);',
            '{ std::vector<int> kk = bkeep; for (int& x : kk) if (x > 1) --x; rt.rollback_batch(kk); }'),
           ('batch first-use: filter removed', 'if (rt.graph_captures() == cap_v && rt.graph_eager_runs() == eag_v) gate_b.observe_step(row, tvr);',
            'if (rt.graph_captures() == cap_v) gate_b.observe_step(row, tvr);')]


def first_use_env(silent=False):
    """First-use world: first batch verify of each row count costs +300 ms (eager graph run) — kept as a sample it makes every verify look hopeless."""
    e = {**TIMING, 'FAKE_MTP': '5', 'HIVE_MTP_BATCH': '1', 'HIVE_MTP_GATE2': '1', 'HIVE_TRACE_MTP': '1', 'FAKE_VERIFY_FIRST_MS': '300'}
    if silent:
        e['FAKE_VERIFY_FIRST_SILENT'] = '1'
    return e


FIRST_USE_REQS = [(T.tokens(50, 61), 120), (T.tokens(46, 62), 120)]


def batch_verifies(log):
    return log.count('[mtp] batch S ') - len(re.findall(r'\[mtp\] batch S \d+ draft', log))


def run_mutants(td):
    """Negative controls (HIVE_MTP_NEGATIVE=1): hived.cpp mutants of the verify/rollback code must be caught by the oracle."""
    src = (T.harness.ROOT / 'engine/src/hived.cpp').read_text()
    ok = True
    for i, (name, a, b) in enumerate(MUTANTS):
        assert src.count(a) == 1, name
        exe = T.build_daemon(td, name=f'mtpmut{i}', hived_source=src.replace(a, b))
        batch = name.startswith('batch')
        first_use = name.startswith('batch first-use')
        env = {'FAKE_MTP': '5', 'HIVE_MTP_GATE2': '1', 'FAKE_MTP_WRONG': '3', 'FAKE_STEP_MS': '5'}
        if batch:
            env.update({'HIVE_MTP_BATCH': '1', 'FAKE_STEP_MS': '20', 'FAKE_VERIFY_MS': '20', 'FAKE_VERIFY_ROW_MS': '3', 'FAKE_MTP_DRAFT_MS': '2'})
        if first_use:
            env = first_use_env()
        h = T.Hived(exe, td, env, name=f'mtpmut{i}')
        ids = T.tokens(40, 1)
        try:
            if first_use:  # detected = the pair stops speculating (oracle tokens stay exact — the filter is a speed property)
                h.generate('warm', T.tokens(30, 7), 4)
                concurrent(h, FIRST_USE_REQS)
                h.stop()
                nv = batch_verifies(h.log_path.read_text(errors='replace'))
                print(f'    (first-use mutant: batch verifies {nv})')
                detected = nv <= 3
            elif batch:
                got = concurrent(h, [(T.tokens(40, 21), 60), (T.tokens(45, 22), 60)])
                detected = any(bool(r['error']) or r['tokens'] != T.oracle(x, n) for (x, n), r in got)
            else:
                r = h.generate('x', ids, 40)
                detected = bool(r['error']) or r['tokens'] != T.oracle(ids, 40)
        except Exception:  # noqa: BLE001 — a dead daemon is a detection too
            detected = True
        try:
            h.stop()
        except Exception:  # noqa: BLE001
            pass
        print(f'  mutant [{name}]: {"DETECTED" if detected else "NOT DETECTED"}')
        ok &= detected
    return ok


def concurrent(h, reqs):
    """Start every (ids, n) request at once (one thread each); returns [((ids, n), result)] in request order."""
    got = [None] * len(reqs)

    def one(i):
        got[i] = h.generate(f'c{i}', reqs[i][0], reqs[i][1])
    ths = [threading.Thread(target=one, args=(i,)) for i in range(len(reqs))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return list(zip(reqs, got))


def batch_checks(exe, td, ck):
    """(5): HIVE_MTP_BATCH — concurrent sessions speculate together; oracle tokens; switch parsing; GATE3."""
    reqs2 = [(T.tokens(50, 31), 90), (T.tokens(44, 32), 90)]
    reqs3 = [(T.tokens(40, 41), 70), (T.tokens(52, 42), 70), (T.tokens(47, 43), 70)]
    out = {}
    for name, env, reqs in [('batch_off', {}, reqs2), ('batch_zero', {'HIVE_MTP_BATCH': '0'}, reqs2), ('batch_empty', {'HIVE_MTP_BATCH': ''}, reqs2),
                            ('batch', {'HIVE_MTP_BATCH': '1'}, reqs2), ('batch_wrong', {'HIVE_MTP_BATCH': '1', 'FAKE_MTP_WRONG': '3'}, reqs2),
                            ('batch3', {'HIVE_MTP_BATCH': '1', 'HIVE_MTP_GATE2': '1'}, reqs3)]:
        h = T.Hived(exe, td, {**TIMING, 'FAKE_MTP': '5', 'HIVE_TRACE_MTP': '1', **env}, name=name)
        try:
            h.generate('warm', T.tokens(30, 7), 4)
            got = concurrent(h, reqs)
            for i, ((ids, n), r) in enumerate(got):
                T.done_ok(ck, f'{name}[{i}]', r, ids, n, None)
        finally:
            T.stop_clean(h, ck)
        log = h.log_path.read_text(errors='replace')
        out[name] = (got, log)
        nb = log.count('[mtp] batch S ')
        rows = [int(m) for m in re.findall(r'\[mtp\] batch S \d+ rows (\d+)', log)]
        print(f'  {name}: batch trace lines {nb} · verify rows max {max(rows) if rows else 0} · spec tokens {mtp_tokens([r for _, r in got])}')
    for name in ('batch_off', 'batch_zero', 'batch_empty'):
        ck.true(f'{name}: no batch speculation', '[mtp] batch S ' not in out[name][1])
    for name in ('batch', 'batch_wrong', 'batch3'):
        ck.true(f'{name}: batch verify ran', '[mtp] batch S ' in out[name][1] and ' rows ' in out[name][1], out[name][1][-1500:])
        rows = [int(m) for m in re.findall(r'\[mtp\] batch S \d+ rows (\d+)', out[name][1])]
        ck.true(f'{name}: rows within cap 8', rows and max(rows) <= 8, repr(rows[:20]))
    ck.true('batch: most tokens speculative while concurrent', mtp_tokens([r for _, r in out['batch'][0]]) >= 0.5 * sum(n for _, n in reqs2),
            f"{mtp_tokens([r for _, r in out['batch'][0]])}")
    ck.true('batch: faster than batch_off', sum(r['done']['decode_ms'] for _, r in out['batch'][0]) < 0.9 * sum(r['done']['decode_ms'] for _, r in out['batch_off'][0]))
    w = [r for _, r in out['batch_wrong'][0]]
    ck.true('batch_wrong: rejections exercised', sum(r['done']['mtp_drafted'] - r['done']['mtp_accepted'] for r in w) > 0)
    ck.true('batch3: three parts in one verify', '[mtp] batch S 3 ' in out['batch3'][1])
    # run-time switch {"op":"set","mtp_batch":...} (2026-10-09): on over a startup-off daemon whose verify path is allocated (HIVE_MTP_VERIFY2),
    #   off over HIVE_MTP_BATCH=1, and on without the allocation (absorbed: no batch speculation, oracle tokens) — then back to the startup value (-1)
    for name, env, val, want in [('set_on', {'HIVE_MTP_VERIFY2': '1'}, 1, True), ('set_off', {'HIVE_MTP_BATCH': '1'}, 0, False),
                                 ('set_on_unallocated', {}, 1, False), ('set_reset', {'HIVE_MTP_BATCH': '1'}, -1, True)]:
        h = T.Hived(exe, td, {**TIMING, 'FAKE_MTP': '5', 'HIVE_TRACE_MTP': '1', **env}, name=name)
        try:
            if name == 'set_reset':
                ck.true(f'{name}: set 0 acknowledged', h.op({'op': 'set', 'mtp_batch': 0}).get('mtp_batch') == 0)
            r = h.op({'op': 'set', 'mtp_batch': val})
            ck.true(f'{name}: set acknowledged', r.get('ok') is True and r.get('mtp_batch') == val, repr(r))
            h.generate('warm', T.tokens(30, 7), 4)
            got = concurrent(h, reqs2)
            for i, ((ids, n), rr) in enumerate(got):
                T.done_ok(ck, f'{name}[{i}]', rr, ids, n, None)
        finally:
            T.stop_clean(h, ck)
        log = h.log_path.read_text(errors='replace')
        print(f'  {name}: batch trace lines {log.count("[mtp] batch S ")}')
        ck.true(f'{name}: batch speculation {"ran" if want else "absent"}', ('[mtp] batch S ' in log) == want, log[-1500:])
    # (6): first-use verify samples are filtered (speculation continues); a runtime hiding first uses locks out (negative control)
    fu = {}
    for name, silent in [('first_use', False), ('first_use_silent', True)]:
        h = T.Hived(exe, td, first_use_env(silent), name=name)
        try:
            h.generate('warm', T.tokens(30, 7), 4)
            got = concurrent(h, FIRST_USE_REQS)
            for i, ((ids, n), r) in enumerate(got):
                T.done_ok(ck, f'{name}[{i}]', r, ids, n, None)
        finally:
            T.stop_clean(h, ck)
        log = h.log_path.read_text(errors='replace')
        fu[name] = (batch_verifies(log), mtp_tokens([r for _, r in got]), sum(r['done']['decode_ms'] for _, r in got))
        print(f'  {name}: batch verifies {fu[name][0]} · spec tokens {fu[name][1]} · decode {fu[name][2]:.0f} ms')
    ck.true('first_use: batch speculation survives first-use samples', fu['first_use'][0] >= 15, repr(fu))
    ck.true('first_use_silent (negative control): unfiltered first-use sample locks batch speculation out', fu['first_use_silent'][0] <= 3, repr(fu))
    # GATE3: on top of gate 2 (oracle tokens, gate-2 trace) · alone = default gate
    for name, env in [('gate3', {'HIVE_MTP_GATE2': '1', 'HIVE_MTP_GATE3': '1'}), ('gate3_alone', {'HIVE_MTP_GATE3': '1'})]:
        res, _, log = run(exe, td, name, {'FAKE_MTP': '5', **env}, ck, [T.tokens(40, 51)], 60)
        print(f'  {name}: spec tokens {mtp_tokens(res)}/60')
        if name == 'gate3':
            ck.true('gate3: gate-2 trace + speculation', 'gate2 k' in log and mtp_tokens(res) >= 0.7 * 60)
        else:
            ck.true('gate3 alone: default gate (no gate-2 trace)', 'gate2' not in log)


def main():
    ck = T.Checker('mtp')
    prompts = [T.tokens(40, 1), T.tokens(70, 2), T.tokens(33, 3)]
    n_tok = 60
    with tempfile.TemporaryDirectory(prefix='hive-mtp-test-') as td:
        exe = T.build_daemon(td)
        out = {}
        for name, env, args in [('nomtp', {'FAKE_MTP': '5'}, ('--no-mtp',)),
                                ('gate1', {'FAKE_MTP': '5'}, ()),
                                ('gate1_zero', {'FAKE_MTP': '5', 'HIVE_MTP_GATE2': '0'}, ()),
                                ('gate1_empty', {'FAKE_MTP': '5', 'HIVE_MTP_GATE2': ''}, ()),
                                ('gate2', {'FAKE_MTP': '5', 'HIVE_MTP_GATE2': '1'}, ()),
                                ('gate2_wrong', {'FAKE_MTP': '5', 'HIVE_MTP_GATE2': '1', 'FAKE_MTP_WRONG': '3'}, ()),
                                ('gate2_max2', {'FAKE_MTP': '5', 'HIVE_MTP_GATE2': '1'}, ('--mtp-max', '2')),
                                ('trap1', {'FAKE_MTP': '5', 'FAKE_MTP_LOWCONF_DRAFTS': '4'}, ()),
                                ('trap2', {'FAKE_MTP': '5', 'FAKE_MTP_LOWCONF_DRAFTS': '4', 'HIVE_MTP_GATE2': '1'}, ())]:
            ps = prompts[1:] if name.startswith('trap') else prompts  # trap: 1st request pays the low-confidence phase, 2nd is measured
            res, stats, log = run(exe, td, name, env, ck, ps, n_tok, args)
            if name.startswith('trap'):
                res = res[1:]
            out[name] = (res, stats, log)
            steps = sum(r['done']['mtp_steps'] for r in res if r['done'])
            drafted = sum(r['done']['mtp_drafted'] for r in res if r['done'])
            acc = sum(r['done']['mtp_accepted'] for r in res if r['done'])
            dms = sum(r['done']['decode_ms'] for r in res if r['done'])
            print(f'  {name}: mtp {acc}/{drafted} in {steps} steps · spec tokens {mtp_tokens(res)}/{len(res) * n_tok} · decode {dms:.0f} ms')
        # (1) oracle tokens are checked inside run() for every config. (3) switch parsing
        for name in ('gate1', 'gate1_zero', 'gate1_empty', 'nomtp'):
            ck.true(f'{name}: no gate-2 trace', 'gate2' not in out[name][2])
        ck.true('gate2: gate-2 trace present', 'gate2 k' in out['gate2'][2], out['gate2'][2][-1500:])
        ck.eq('nomtp: no speculation', sum(r['done']['mtp_steps'] for r in out['nomtp'][0]), 0)
        # (2) gate 2 engages: exact drafts, verify of 6 rows = 35 ms vs 20 ms step → ≥ 70% of tokens from speculative steps
        g2 = mtp_tokens(out['gate2'][0])
        ck.true('gate2: most tokens speculative', g2 >= 0.7 * len(prompts) * n_tok, f'{g2}')
        # root cause reproduced: after normal steps were measured (first 4 drafts low-confidence → declined), the default gate compares
        # against its stale verify table (66..152 ms) and never speculates again; gate 2 uses the measured T(1) and speculates.
        ck.true('trap: default gate locked out', mtp_tokens(out['trap1'][0]) <= 0.1 * n_tok, f'{mtp_tokens(out["trap1"][0])}')
        ck.true('trap: gate 2 speculates', mtp_tokens(out['trap2'][0]) >= 0.6 * n_tok, f'{mtp_tokens(out["trap2"][0])}')
        ck.true('gate2: faster decode than no-MTP',
                sum(r['done']['decode_ms'] for r in out['gate2'][0]) < 0.8 * sum(r['done']['decode_ms'] for r in out['nomtp'][0]))
        w = out['gate2_wrong'][0]
        ck.true('gate2_wrong: rejections exercised', sum(r['done']['mtp_drafted'] - r['done']['mtp_accepted'] for r in w) > 0)
        ck.true('gate2_max2: k <= 2', all(r['done']['mtp_drafted'] <= 2 * r['done']['mtp_steps'] for r in out['gate2_max2'][0]))
        # (4) batch: two concurrent sessions — no speculation while both are active; lone survivor speculates again
        h = T.Hived(exe, td, {**TIMING, 'FAKE_MTP': '5', 'HIVE_MTP_GATE2': '1', 'HIVE_TRACE_MTP': '1'}, name='batch')
        try:
            h.generate('warm', prompts[0], 4)  # T(1) measured
            a_ids, b_ids = T.tokens(50, 11), T.tokens(50, 12)
            got = {}
            th = threading.Thread(target=lambda: got.__setitem__('a', h.generate('a', a_ids, 120)))
            th.start()
            got['b'] = h.generate('b', b_ids, 20)
            th.join()
            T.done_ok(ck, 'batch a', got['a'], a_ids, 120, None)
            T.done_ok(ck, 'batch b', got['b'], b_ids, 20, None)
            ck.true('batch: rows > 1 seen', got['b']['done']['batch_rows'] >= 1)
            ck.true('batch: a speculates after b ends', got['a']['done']['mtp_steps'] > 0, repr(got['a']['done']))
        finally:
            T.stop_clean(h, ck)
        batch_checks(exe, td, ck)
        if os.environ.get('HIVE_MTP_NEGATIVE', '') not in ('', '0'):
            ck.true('negative controls detected', run_mutants(td))
    print(f'mtp: {ck.checks} checks, {len(ck.failures)} failures')
    for f in ck.failures:
        print('  FAIL', f[:600])
    sys.exit(1 if ck.failures else 0)


if __name__ == '__main__':
    main()
