#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Thinking cap (think_cap) — the REAL engine/src/hived.cpp on fake CUDA/runtime + server/hive_server.py field handling. No GPU, no service.

Daemon (tools/cpu_fake oracle: the next token is a hash of the whole history, so a forced token changes everything after it):
  (1) no think fields → tokens == oracle, done has no think_* keys (default off = unchanged)
  (2) think_cap N (+ think_end_id E, think_open true) → tokens == oracle(prompt)[:N] + [E] + oracle(prompt + those + [E]) — exactly N thinking tokens,
      then </think>, then the continuation from the forced state; done.think_forced true · think_tokens N · think_cap N
  (3) the same with DSpark speculation (FAKE_MTP=5, gate 2, exact and corrupted drafts) — accepted drafts never cross the cap
  (4) two concurrent sessions with HIVE_MTP_BATCH=1 (batch speculation) and different caps
  (5) think_open false and no <think> emitted → nothing forced · cap larger than the answer → nothing forced (think_forced false)
  (6) absorbed values: think_cap 0 / -3 / "x" / think_end_id out of vocab → feature off (tokens == oracle, no think_* in done)
  Negative controls (hived.cpp mutants) MUST fail: no head override · single-path draft clamp removed · batch clamp removed · counting the forced token.
Server: think_cap_from_request field names/normalization · sampling dict unchanged without the field · think fields only in thinking mode ·
  requests-server record carries think_cap (null by default) and think_forced/think_tokens from done.
"""
import asyncio
import importlib.util
import json
import re
from pathlib import Path
import sys
import tempfile
import threading

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import test_daemon_cpu as T  # noqa: E402

END, START = 5, 6  # oracle tokens are 10..4009 — no overlap
FAILS = []
QUIET = [False]  # during mutant runs, failure lines are not printed (only one line saying whether it was detected)


def check(what, cond, detail=''):
    if not cond:
        FAILS.append(f'{what} {detail}')
        if not QUIET[0]:
            print(f'FAIL {what} {detail}')


def gen(h, session, ids, n, extra=None):
    req = {'op': 'generate', 'session': session, 'ids': ids, 'temperature': 0, 'max_tokens': n, **(extra or {})}
    s = h.connect()
    try:
        s.sendall((json.dumps(req) + '\n').encode())
        out = {'tokens': [], 'done': None, 'error': None}
        for line in s.makefile():
            m = json.loads(line)
            if 'id' in m:
                out['tokens'].append(m['id'])
            elif m.get('done'):
                out['done'] = m
                break
            elif 'error' in m:
                out['error'] = m['error']
                break
        return out
    finally:
        s.close()


def expect_capped(ids, n, cap):
    head = T.oracle(ids, min(cap, n))
    if cap >= n:
        return head
    rest = T.oracle(ids + head + [END], n - cap - 1)
    return head + [END] + rest


CAP = lambda c, **kw: {'think_cap': c, 'think_end_id': END, 'think_start_id': START, 'think_open': True, **kw}  # noqa: E731
EXIT = [7, 8, 9]  # exit phrase (does not overlap oracle tokens 10..4009 · END 5 · START 6)


def expect_exit(ids, n, cap, ex=EXIT):
    """On reaching cap: the exit phrase ex token by token → </think> → the oracle continuation from that forced state."""
    head = T.oracle(ids, min(cap, n))
    forced = (ex + [END])[:max(0, n - len(head))]
    out = head + forced
    if len(out) < n:
        out += T.oracle(ids + out, n - len(out))
    return out


def daemon_checks(exe, td, tag, env):
    h = T.Hived(exe, td, env, name=f'tc{tag}')
    try:
        ids = T.tokens(40, 3)
        r = gen(h, 'plain', ids, 30)
        check(f'[{tag}] no fields: oracle tokens', r['tokens'] == T.oracle(ids, 30), str(r['error']))
        check(f'[{tag}] no fields: done has no think keys', r['done'] is not None and not any(k.startswith('think') for k in r['done']), str(r['done']))
        for cap in (1, 7, 13):
            ids = T.tokens(35 + cap, 10 + cap)
            r = gen(h, f'cap{cap}', ids, 40, CAP(cap))
            want = expect_capped(ids, 40, cap)
            check(f'[{tag}] cap {cap}: tokens', r['tokens'] == want, f"err {r['error']} got {r['tokens'][:cap + 3]} want {want[:cap + 3]}")
            d = r['done'] or {}
            check(f'[{tag}] cap {cap}: done', d.get('think_forced') is True and d.get('think_tokens') == cap and d.get('think_cap') == cap, str(d))
        for cap in (1, 6):  # exit phrase: after cap tokens, EXIT one token at a time → </think> · think_tokens = cap + len(EXIT) (the phrase counts as thinking)
            ids = T.tokens(33 + cap, 50 + cap)
            r = gen(h, f'exit{cap}', ids, 40, CAP(cap, think_exit_ids=EXIT))
            want = expect_exit(ids, 40, cap)
            check(f'[{tag}] exit cap {cap}: tokens', r['tokens'] == want, f"err {r['error']} got {r['tokens'][:cap + 6]} want {want[:cap + 6]}")
            d = r['done'] or {}
            check(f'[{tag}] exit cap {cap}: done', d.get('think_forced') is True and d.get('think_tokens') == cap + len(EXIT), str(d))
        ids = T.tokens(31, 88)
        for bad_ex in ([7, 999999], 'x', [7, -1], [7.5]):  # absorbed: out of vocab · negative · non-integer · not an array → no phrase, plain behaviour (</think> only)
            r = gen(h, 'badexit', ids, 30, CAP(4, think_exit_ids=bad_ex))
            check(f'[{tag}] exit absorbed {bad_ex!r}', r['tokens'] == expect_capped(ids, 30, 4) and r['error'] is None, f"{r['error']} {r['tokens'][:8]}")
        ids = T.tokens(30, 77)
        r = gen(h, 'closed', ids, 25, CAP(5, think_open=False))
        check(f'[{tag}] think_open false: nothing forced', r['tokens'] == T.oracle(ids, 25) and (r['done'] or {}).get('think_forced') is False, str(r['done']))
        r = gen(h, 'big', ids, 20, CAP(500))
        check(f'[{tag}] cap beyond the answer: nothing forced', r['tokens'] == T.oracle(ids, 20) and (r['done'] or {}).get('think_tokens') == 20, str(r['done']))
        for bad in ({'think_cap': 0, 'think_end_id': END}, {'think_cap': -3, 'think_end_id': END}, {'think_cap': 'x', 'think_end_id': END},
                    {'think_cap': 5, 'think_end_id': 999999}, {'think_cap': 5}):
            r = gen(h, 'bad', ids, 20, bad)
            check(f'[{tag}] absorbed {bad}', r['tokens'] == T.oracle(ids, 20) and r['error'] is None and not any(k.startswith('think') for k in (r['done'] or {})),
                  f"{r['error']} {r['done']}")
    finally:
        T.stop_clean(h, T.Checker(tag))


def batch_checks(exe, td, tag, env):
    h = T.Hived(exe, td, env, name=f'tcb{tag}')
    try:
        jobs = [(T.tokens(40, 31), 60, 9), (T.tokens(44, 32), 60, -17), (T.tokens(38, 33), 60, None)]  # negative = cap with exit phrase (adding a 4th job reduces batch speculation and defeats the scenario — measured)
        out = [None] * len(jobs)

        def run(i):
            ids, n, cap = jobs[i]
            out[i] = gen(h, f'b{i}', ids, n, (CAP(-cap, think_exit_ids=EXIT) if cap < 0 else CAP(cap)) if cap else None)
        th = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
        for t in th:
            t.start()
        for t in th:
            t.join()
        for i, (ids, n, cap) in enumerate(jobs):
            want = (expect_exit(ids, n, -cap) if cap < 0 else expect_capped(ids, n, cap)) if cap else T.oracle(ids, n)
            check(f'[{tag}] concurrent {i} cap {cap}: tokens', out[i]['tokens'] == want, str(out[i]['error']))
        log = h.log_path.read_text(errors='replace')
        return log
    finally:
        T.stop_clean(h, T.Checker(tag))


MUTANTS = [
    ('no head override', '        tok = A->think_exit_pos < A->think_exit.size() ? A->think_exit[A->think_exit_pos++] : A->think_end;\n        ++A->think_forced;\n', '        ++A->think_forced;\n', 'plain'),
    ('single-path draft clamp removed', '      if (A.think_cap > 0 && A.in_think) k = std::min(k, std::max(0, A.think_cap - A.think_n));', '', 'mtp'),
    ('batch clamp removed', 'if (stepping[s]->think_cap > 0 && stepping[s]->in_think) rem[s] = std::min(', 'if (false) rem[s] = std::min(', 'batch'),
    ('exit phrase skipped', 'A->think_exit_pos < A->think_exit.size() ? A->think_exit[A->think_exit_pos++] : A->think_end', 'A->think_end', 'plain'),
    ('exit ids ignored', 'if (ok) A->think_exit = std::move(ex);', '', 'plain'),
    ('forced token counted as thinking', 'if (tok == A.think_end) A.in_think = false; else ++A.think_n;', '++A.think_n; if (tok == A.think_end) A.in_think = false;', 'plain'),
]
MTP_ENV = {'FAKE_MTP': '5', 'HIVE_MTP_GATE2': '1', 'HIVE_TRACE_MTP': '1', 'FAKE_STEP_MS': '20', 'FAKE_VERIFY_MS': '20', 'FAKE_VERIFY_ROW_MS': '3', 'FAKE_MTP_DRAFT_MS': '2'}
# The batch speculation gate (drops the first sample · batch_probe_due · can_profit_batch before drafting · 8-row budget) speculated in
#   batch only 2 steps per run under the MTP_ENV costs (verify 20 + 3 ms per row · draft 2 ms) — never at a cap boundary, so the
#   "batch clamp removed" mutant went undetected (diagnosed with the mutant hived: 2 [mtp] batch lines). The batch scenario sets
#   row and draft costs to 0 so speculation always pays and runs every step (44 lines · mutant detected).
BATCH_ENV = {**MTP_ENV, 'HIVE_MTP_BATCH': '1', 'FAKE_VERIFY_ROW_MS': '0', 'FAKE_MTP_DRAFT_MS': '0'}


def mutant_checks(td):
    src = (T.harness.ROOT / 'engine/src/hived.cpp').read_text()
    for i, (name, a, b, mode) in enumerate(MUTANTS):
        if src.count(a) != 1:
            check(f'mutant anchor [{name}]', False, 'anchor drifted')
            continue
        exe = T.build_daemon(td, name=f'tcmut{i}', hived_source=src.replace(a, b))
        before = len(FAILS)
        QUIET[0] = True
        try:
            if mode == 'plain':
                daemon_checks(exe, td, f'mut{i}', {})
            elif mode == 'mtp':
                daemon_checks(exe, td, f'mut{i}', MTP_ENV)
            else:
                # Whether a cap boundary lands on a batch-speculation step depends on how the three requests overlap, which is
                #   timing-dependent: on a 2-vCPU CI runner one run missed it (mutant not detected). Up to 4 runs —
                #   the mutant is detected if any run shows a wrong token; the unit scenario above still has to pass on its own.
                for attempt in range(4):
                    batch_checks(exe, td, f'mut{i}r{attempt}', BATCH_ENV)
                    if len(FAILS) > before:
                        break
        except Exception as e:  # noqa: BLE001 — a dead daemon is a detection too
            FAILS.append(f'mutant crash {e}')
        QUIET[0] = False
        detected = len(FAILS) > before
        del FAILS[before:]
        print(f'  think-cap mutant [{name}]: {"DETECTED" if detected else "NOT DETECTED"}')
        check(f'mutant [{name}] detected', detected)


# ---------------------------------------------------------------------------------------------------------- server
def server_checks():
    spec = importlib.util.spec_from_file_location('server_tc', T.harness.ROOT / 'server/hive_server.py')
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    f = s.think_cap_from_request
    check('server: no field', f({}) is None)
    check('server: max_thinking_tokens', f({'max_thinking_tokens': 16384}) == 16384)
    check('server: thinking_token_budget', f({'thinking_token_budget': 512.0}) == 512)
    check('server: kwargs', f({'chat_template_kwargs': {'max_thinking_tokens': '300'}}) == 300)
    check('server: kwargs thinking_token_budget', f({'chat_template_kwargs': {'thinking_token_budget': 77}}) == 77)
    check('server: top-level wins', f({'max_thinking_tokens': 10, 'chat_template_kwargs': {'max_thinking_tokens': 99}}) == 10)
    for bad in (0, -1, True, None, 'x', float('nan'), [], {}):
        check(f'server: absorbed {bad!r}', f({'max_thinking_tokens': bad}) is None)
    check('server: invalid first, valid second', f({'max_thinking_tokens': 'x', 'thinking_token_budget': 40}) == 40)
    check('server: _num', (s._num(None, 1.0), s._num('0.7', 1.0), s._num(0, 1.0), s._num(True, 2), s._num(float('inf'), 3)) == (1.0, 0.7, 0, 2, 3))

    class Tok:
        eos_token_id = 99
        eos_token = '<EOS>'
        VOC = {1: 'a', 2: 'b', 5: '</think>', 6: '<think>', 99: '<EOS>'}

        def decode(self, ids, **kw):
            return ''.join(self.VOC[i] for i in ids)

        def convert_tokens_to_ids(self, t):
            return {'<think>': 6, '</think>': 5}.get(t, 0)

        def encode(self, text, add_special_tokens=True):
            return [1, 2] if text else []

    class Enc:
        def parse_message_from_completion_text(self, text, thinking_mode=None, **kw):
            text = text.replace('<EOS>', '')
            if thinking_mode == 'thinking' and '</think>' in text:
                a, b = text.split('</think>', 1)
                return {'reasoning_content': a, 'content': b}
            return {'content': text}

    class D:
        def __init__(self, done_extra):
            self.reqs = []
            self.done_extra = done_extra

        async def generate(self, session, ids, images, blob, rid='', boundaries=None, **sampling):
            self.reqs.append(sampling)
            for t in (1, 5, 2):
                yield {'id': t}
            yield {'done': True, 'n': 3, 'finish': 'stop', **self.done_extra}

        def cancel(self, session, rid=''):
            pass

    class Req:
        def __init__(self, body):
            self.body = body

        async def json(self):
            return self.body

        async def is_disconnected(self):
            return False

    recs = []
    s.log_request = lambda r: recs.append(r)
    s.TOK, s.ENC = Tok(), Enc()
    s.encode_prompt = lambda *a: ('prompt', [])
    s.tokenize_with_images = lambda *a: ([10, 11, 6], [], b'')
    s.daemon_max_ctx = lambda: asyncio.sleep(0, result=0)
    msgs = [{'role': 'user', 'content': 'hi'}]

    async def go():
        s.DAEMON = D({})
        await s.chat(Req({'messages': msgs}))
        base = dict(s.DAEMON.reqs[-1])
        check('server: default request has no think fields', not any(k.startswith('think') for k in base), str(base))
        check('server: default record think_cap null', recs[-1].get('think_cap') is None and 'think_forced' not in recs[-1], str(recs[-1]))
        s.DAEMON = D({'think_cap': 1, 'think_tokens': 1, 'think_forced': True})
        out = await s.chat(Req({'messages': msgs, 'max_thinking_tokens': 1}))
        sent = s.DAEMON.reqs[-1]
        check('server: thinking request carries the cap', sent.get('think_cap') == 1 and sent.get('think_end_id') == 5 and sent.get('think_start_id') == 6
              and sent.get('think_open') is True, str(sent))
        check('server: exit phrase ids by default', sent.get('think_exit_ids') == [1, 2], str(sent))
        import os
        os.environ['HIVE_THINK_EXIT_TEXT'] = ''
        try:
            await s.chat(Req({'messages': msgs, 'max_thinking_tokens': 1}))
            check('server: HIVE_THINK_EXIT_TEXT empty = no exit ids (previous behaviour)', 'think_exit_ids' not in s.DAEMON.reqs[-1] and s.DAEMON.reqs[-1].get('think_cap') == 1, str(s.DAEMON.reqs[-1]))
        finally:
            del os.environ['HIVE_THINK_EXIT_TEXT']
        check('server: other sampling fields unchanged', {k: v for k, v in sent.items() if not k.startswith('think')} == base, f'{sent} vs {base}')
        check('server: record think_forced', recs[-1].get('think_cap') == 1 and recs[-1].get('think_forced') is True and recs[-1].get('think_tokens') == 1, str(recs[-1]))
        check('server: response hive think fields', out['hive'].get('think_forced') is True and out['hive'].get('think_cap') == 1, str(out.get('hive')))
        check('server: thinking mode / effort still logged', recs[-1].get('thinking_mode') == 'thinking' and recs[-1].get('effort') == 75, str(recs[-1]))
        s.DAEMON = D({})
        await s.chat(Req({'messages': msgs, 'max_thinking_tokens': 50, 'reasoning_effort': 'none'}))
        check('server: chat mode ignores the cap', not any(k.startswith('think') for k in s.DAEMON.reqs[-1]) and recs[-1].get('think_cap') is None, str(s.DAEMON.reqs[-1]))
        await s.chat(Req({'messages': msgs, 'temperature': None, 'top_p': None, 'seed': None}))
        q = s.DAEMON.reqs[-1]
        check('server: null sampling fields normalized', (q['temperature'], q['top_p'], q['seed']) == (1.0, 0.95, 0), str(q))
    asyncio.run(go())


def main():
    server_checks()
    with tempfile.TemporaryDirectory() as td:
        exe = T.build_daemon(td, name='hived_tc')
        daemon_checks(exe, td, 'plain', {})
        daemon_checks(exe, td, 'mtp', MTP_ENV)
        daemon_checks(exe, td, 'mtp-wrong', {**MTP_ENV, 'FAKE_MTP_WRONG': '3'})
        log = batch_checks(exe, td, 'batch', BATCH_ENV)
        acc = sum(1 for l in log.splitlines() if re.search(r'\[mtp\] batch .*accepted \[[1-9]', l))
        check('batch speculation actually ran (accepted drafts in the first part at many steps)', acc >= 10, f'{acc} batch steps with accepted drafts')
        mutant_checks(td)
    if FAILS:
        print(f'think cap CPU: {len(FAILS)} failures')
        sys.exit(1)
    print('think cap CPU: default off unchanged · exact cap (plain · MTP · wrong drafts · batch MTP) · absorbed values · server fields/record · mutants detected')


if __name__ == '__main__':
    main()
