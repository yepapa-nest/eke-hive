#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""allowed_tools — the tool-name constraint (hive/tool_mask.h) through the real hived.cpp on the fake runtime, and through the server.

Daemon: the fake model's row favours its oracle token; every other logit is 0, so a constrained row picks the first allowed token unless the oracle's token is
allowed. Plain decode, MTP with right and wrong drafts, gate 2 and batched MTP must all emit [tags] + [the allowed path] + the oracle's continuation of what
was emitted, and exactly the oracle's tokens when the model already writes an allowed name (drafts survive). A request without the field and a malformed field
are served as before. Mutants of each sampling path are detected.

Server: compile_tool_trie reaches every spelling of a name and nothing else, the field's absorption rules, the record fields — and the field changes nothing
but the daemon's sampling: same rendered prompt, same token ids, same session, same boundaries, same other sampling fields.
"""
import asyncio
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import threading

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import test_daemon_cpu as T  # noqa: E402

FAILS = []
QUIET = [False]
A, B, C = 2, 3, 4   # outside the oracle's 10..4009
CALLS, PARAM = 5, 6
MTP_ENV = {'FAKE_MTP': '5', 'HIVE_MTP_GATE2': '1', 'HIVE_TRACE_MTP': '1', 'FAKE_STEP_MS': '20', 'FAKE_VERIFY_MS': '20', 'FAKE_VERIFY_ROW_MS': '3', 'FAKE_MTP_DRAFT_MS': '2'}
BATCH_ENV = {**MTP_ENV, 'HIVE_MTP_BATCH': '1', 'FAKE_VERIFY_ROW_MS': '0', 'FAKE_MTP_DRAFT_MS': '0'}


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


def mask(ids, first=None, tok=None, nxt=None, **kw):
    """The oracle's first four tokens are the tags ｜DSML｜ invoke name =" of a prompt that ended inside a block (mode 1), so the fifth row starts a name.
    Default automaton: node 0 {A → node 1, B → closes} · node 1 {C → closes}."""
    o = T.oracle(ids, 4)
    return {'tags': [o[0], CALLS, o[1], PARAM, o[2], o[3]], 'first': first or [0, 2, 3], 'tok': tok or [A, B, C], 'next': nxt or [1, -1, -1], 'mode': 1, **kw}


def expect_path(ids, n, path):
    head = T.oracle(ids, 4) + list(path)
    return (head + T.oracle(list(ids) + head, max(0, n - len(head))))[:n]


def daemon_checks(exe, td, tag, env):
    h = T.Hived(exe, td, env, name=f'tm{tag}')
    try:
        ids = T.tokens(40, 3)
        r = gen(h, 'plain', ids, 30)
        check(f'[{tag}] no field: oracle tokens', r['tokens'] == T.oracle(ids, 30), str(r['error']))
        check(f'[{tag}] no field: done has no tool_mask', 'tool_mask' not in (r['done'] or {}), str(r['done']))
        for k in range(6):
            ids = T.tokens(35 + k, 20 + k)
            r = gen(h, f'path{k}', ids, 40, {'tool_mask': mask(ids)})
            want = expect_path(ids, 40, [A, C])
            check(f'[{tag}] {k}: name constrained, then the model again', r['tokens'] == want, f"err {r['error']} got {r['tokens'][:8]} want {want[:8]}")
            d = (r['done'] or {}).get('tool_mask') or {}
            check(f'[{tag}] {k}: done counts', d.get('rows') == 2 and d.get('spans') == 1 and d.get('left') == 0, str(r['done']))
        # the model's own token is allowed: output = the oracle's (drafts survive verification)
        for k in range(4):
            ids = T.tokens(36 + k, 60 + k)
            o = T.oracle(ids, 8)
            # node 0 {A → closes, the model's token → node 1} · node 1 {the model's next token → closes}
            r = gen(h, f'same{k}', ids, 40, {'tool_mask': mask(ids, first=[0, 2, 3], tok=[A, o[4], o[5]], nxt=[-1, 1, -1])})
            check(f'[{tag}] {k}: allowed model token: oracle tokens', r['tokens'] == T.oracle(ids, 40), f"err {r['error']} got {r['tokens'][:8]} want {T.oracle(ids, 8)}")
        # one token closes the name (the second edge of node 0)
        ids = T.tokens(41, 90)
        r = gen(h, 'close', ids, 30, {'tool_mask': mask(ids, first=[0, 1], tok=[B], nxt=[-1])})
        check(f'[{tag}] closing token', r['tokens'] == expect_path(ids, 30, [B]), str(r['tokens'][:7]))
        # not a block: tags in plain text do nothing
        ids = T.tokens(42, 91)
        r = gen(h, 'text', ids, 30, {'tool_mask': {**mask(ids), 'mode': 0}})
        check(f'[{tag}] outside a block: oracle tokens', r['tokens'] == T.oracle(ids, 30) and (r['done'].get('tool_mask') or {}).get('spans') == 0, str(r['done']))
        # thinking: the tags are text until </think>
        ids = T.tokens(43, 92)
        r = gen(h, 'think', ids, 30, {'tool_mask': {**mask(ids), 'think_end': 7, 'in_think': True}})
        check(f'[{tag}] inside thinking: oracle tokens', r['tokens'] == T.oracle(ids, 30), str(r['tokens'][:6]))
        # absorption: not a valid automaton = served as if absent, never an error
        ids = T.tokens(44, 93)
        good = mask(ids)
        for name, bad in (('not an object', 'x'), ('no tags', {k: v for k, v in good.items() if k != 'tags'}), ('short tags', {**good, 'tags': [1, 2]}),
                          ('token outside the vocabulary', {**good, 'tok': [A, B, 10 ** 6]}), ('next past the end', {**good, 'next': [9, -1, -1]}),
                          ('first broken', {**good, 'first': [0, 9, 3]}), ('not integers', {**good, 'tok': [A, B, 1.5]}), ('tag outside the vocabulary', {**good, 'tags': [-1] * 6})):
            r = gen(h, f'bad-{name}', ids, 20, {'tool_mask': bad})
            check(f'[{tag}] absorbed: {name}', r['tokens'] == T.oracle(ids, 20) and r['error'] is None and 'tool_mask' not in (r['done'] or {}), f"{r['error']} {r['done']}")
    finally:
        T.stop_clean(h, T.Checker(tag))


def batch_checks(exe, td, tag, env):
    h = T.Hived(exe, td, env, name=f'tmb{tag}')
    try:
        jobs = [(T.tokens(40, 31), True), (T.tokens(44, 32), False), (T.tokens(38, 33), True)]
        out = [None] * len(jobs)

        def run(i):
            ids, on = jobs[i]
            out[i] = gen(h, f'b{i}', ids, 60, {'tool_mask': mask(ids)} if on else None)
        th = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
        for t in th:
            t.start()
        for t in th:
            t.join()
        for i, (ids, on) in enumerate(jobs):
            want = expect_path(ids, 60, [A, C]) if on else T.oracle(ids, 60)
            check(f'[{tag}] concurrent {i}: tokens', out[i]['tokens'] == want, f"err {out[i]['error']} got {out[i]['tokens'][:8]} want {want[:8]}")
        return h.log_path.read_text(errors='replace')
    finally:
        T.stop_clean(h, T.Checker(tag))


MUTANTS = [
    ('plain path samples unconstrained', 'else if (A->tool.in_span() && !A->logits.empty()) tok = A->tool.sample(', 'else if (false) tok = A->tool.sample(', 'plain'),
    ('verify path samples unconstrained', 'if (A.tool.in_span() && !rows.empty()) return A.tool.sample(', 'if (false) return A.tool.sample(', 'mtp'),
    ('batch verify path samples unconstrained', 'if (A.tool.in_span() && !brows.empty()) return A.tool.sample(', 'if (false) return A.tool.sample(', 'batch'),
    ('the state never sees the emitted tokens', '    A.tool.feed(tok);\n', '', 'plain'),
    ('every field accepted without validation', 't.trie.valid(c.vocab) &&', 'true &&', 'plain'),
]


def mutant_checks(td):
    src = (T.harness.ROOT / 'engine/src/hived.cpp').read_text()
    for i, (name, a, b, mode) in enumerate(MUTANTS):
        if src.count(a) != 1:
            check(f'mutant anchor [{name}]', False, 'anchor drifted')
            continue
        exe = T.build_daemon(td, name=f'tmmut{i}', hived_source=src.replace(a, b))
        before = len(FAILS)
        QUIET[0] = True
        try:
            if mode == 'plain':
                daemon_checks(exe, td, f'mut{i}', {})
            elif mode == 'mtp':
                daemon_checks(exe, td, f'mut{i}', MTP_ENV)
            else:
                for attempt in range(4):
                    batch_checks(exe, td, f'mut{i}r{attempt}', BATCH_ENV)
                    if len(FAILS) > before:
                        break
        except Exception as e:  # noqa: BLE001 — a dead daemon is a detection too
            FAILS.append(f'mutant crash {e}')
        QUIET[0] = False
        detected = len(FAILS) > before
        del FAILS[before:]
        print(f'  tool-mask mutant [{name}]: {"DETECTED" if detected else "NOT DETECTED"}')
        check(f'mutant [{name}] detected', detected)


# ---------------------------------------------------------------------------------------------------------- server
VOCAB = {'shell': 10, 'sh': 11, 'ell': 12, 's': 13, 'h': 14, 'e': 15, 'l': 16, 'll': 17, 'files': 20, '_write': 21, 'f': 22, 'i': 23, '_': 24, 'w': 25, 'r': 26, 't': 27,
         'write': 28, 'browser': 30, 'b': 31, 'rowser': 32, '_task': 33, 'o': 34, 'a': 35, 'k': 36, 'u': 37, 'c': 38, 'n': 39, 'p': 40,
         '"': 50, '">': 51, '">Ċ': 52, 'e">': 53, '"><': 54,
         '｜DSML｜': 60, 'Ġcalls': 61, 'Ġinvoke': 62, 'Ġparameter': 63, 'Ġname': 64, '="': 65, '</think>': 70, '<think>': 71}


def walk(trie, tokens):
    """Node after these tokens, None if a token has no edge."""
    node = 0
    for t in tokens:
        for e in range(trie['first'][node], trie['first'][node + 1]):
            if trie['tok'][e] == t:
                node = trie['next'][e]
                break
        else:
            return None
        if node < 0:
            return node
    return node


def server_checks():
    spec = importlib.util.spec_from_file_location('server_tm', T.harness.ROOT / 'server/hive_server.py')
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)

    t = s.compile_tool_trie(['browser', 'browser_task', 'files_write', 'shell'], VOCAB)
    check('trie: canonical spelling', walk(t, [10, 52]) == -1 and walk(t, [20, 21, 52]) == -1 and walk(t, [30, 52]) == -1 and walk(t, [30, 33, 52]) == -1, str(t))
    check('trie: other spellings of a name', walk(t, [11, 12, 52]) == -1 and walk(t, [31, 32, 50]) == -1 and walk(t, [13, 14, 15, 16, 16, 51]) == -1, str(t))
    check('trie: a token finishing the name and closing it', walk(t, [11, 15 + 0, 16, 16]) is not None and walk(t, [13, 14, 53]) is None, str(t))
    check('trie: a prefix alone is not closed', walk(t, [11, 52]) is None and walk(t, [13, 52]) is None and walk(t, [31, 52]) is None, str(t))
    check('trie: other names are out', walk(t, [28]) is None and walk(t, [26]) is None and walk(t, [40]) is None, str(t))
    check('trie: closers only as the grammar writes them', walk(t, [10, 54]) is None, str(t))
    tg = s.compile_tool_trie(['te'], {'t': 1, 'e': 2, 'te': 3, 'e">': 4, '">Ċ': 5})
    check('trie: last piece + closer in one token', walk(tg, [1, 4]) == -1 and walk(tg, [3, 5]) == -1, str(tg))
    check('trie: names it cannot spell = none', s.compile_tool_trie(['\ud55c\uae00'], VOCAB) is None and s.compile_tool_trie(['a b'], VOCAB) is None
          and s.compile_tool_trie(['zzz'], {'">': 1}) is None and s.compile_tool_trie(['a"b'], VOCAB) is None)

    TOOLS = [{'type': 'function', 'function': {'name': n, 'parameters': {}}} for n in ('shell', 'files_write', 'browser', 'browser_task')]
    f = s.allowed_tool_names
    check('field: subset, sorted', f({'allowed_tools': ['shell', 'browser']}, TOOLS, []) == ['browser', 'shell'])
    for name, body in (('absent', {}), ('null', {'allowed_tools': None}), ('empty', {'allowed_tools': []}), ('not a list', {'allowed_tools': 'shell'}),
                       ('tool_choice none', {'allowed_tools': ['shell'], 'tool_choice': 'none'}), ('only unknown names', {'allowed_tools': ['nope', 5]}),
                       ('every tool', {'allowed_tools': ['shell', 'files_write', 'browser', 'browser_task']})):
        check(f'field: {name} = no restriction', f(body, TOOLS, []) == [], str(f(body, TOOLS, [])))
    check('field: unknown names dropped', f({'allowed_tools': ['shell', 'nope', None]}, TOOLS, []) == ['shell'])
    late = [{'role': 'system', 'tools': [{'type': 'function', 'function': {'name': 'later'}}]}]
    check('field: a tool announced in a later system message', f({'allowed_tools': ['later']}, TOOLS, late) == ['later'])
    ns = [{'type': 'function', 'namespace': 'mail', 'function': {'name': 'send'}}, {'type': 'function', 'function': {'name': 'x'}}]
    check('field: namespaced tool, either form', f({'allowed_tools': ['send']}, ns, []) == ['mail::send'] and f({'allowed_tools': ['mail::send']}, ns, []) == ['mail::send'])

    class Tok:
        eos_token_id = 99
        eos_token = '<EOS>'
        VOC = {1: 'a', 2: 'b', 70: '</think>', 71: '<think>', 99: '<EOS>'}

        def get_vocab(self):
            return dict(VOCAB)

        def decode(self, ids, **kw):
            return ''.join(self.VOC.get(i, '?') for i in ids)

        def convert_tokens_to_ids(self, x):
            return VOCAB.get(x, 0)

        def encode(self, text, add_special_tokens=True):
            k = text.replace(' ', 'Ġ')
            return [VOCAB[k]] if k in VOCAB else [1, 2]

    class Enc:
        calls = None

        def parse_message_from_completion_text(self, text, thinking_mode=None, **kw):
            return {'content': '', 'tool_calls': Enc.calls} if Enc.calls else {'content': text.replace('<EOS>', '')}

    class D:
        def __init__(self, done_extra):
            self.reqs = []
            self.done_extra = done_extra

        async def generate(self, session, ids, images, blob, rid='', boundaries=None, **sampling):
            self.reqs.append({'session': session, 'ids': ids, 'boundaries': boundaries, **sampling})
            yield {'id': 1}
            yield {'done': True, 'n': 1, 'finish': 'stop', **self.done_extra}

        def cancel(self, session, rid=''):
            pass

    class Req:
        def __init__(self, body):
            self.body = body

        async def json(self):
            return json.loads(json.dumps(self.body))

        async def is_disconnected(self):
            return False

    recs = []
    prompts = []
    s.log_request = lambda r: recs.append(r)
    s.TOK, s.ENC = Tok(), Enc()
    real_encode = lambda messages, tools, mode, effort: (prompts.append(json.dumps([messages, tools, mode, effort])) or ('prompt', []))
    s.encode_prompt = real_encode
    s.tokenize_with_images = lambda *a: ([10, 11, 71], [], b'')
    s.daemon_max_ctx = lambda: asyncio.sleep(0, result=0)
    msgs = [{'role': 'user', 'content': 'hi'}]
    base = {'messages': msgs, 'tools': TOOLS}

    async def go():
        s.DAEMON = D({})
        await s.chat(Req(base))
        plain, plain_p = dict(s.DAEMON.reqs[-1]), prompts[-1]
        check('server: no field: no tool_mask, no record fields', 'tool_mask' not in plain and 'allowed_tools_n' not in recs[-1] and 'allowed_violation' not in recs[-1], str(recs[-1]))
        s.DAEMON = D({'tool_mask': {'rows': 3, 'spans': 1, 'left': 0}})
        await s.chat(Req({**base, 'allowed_tools': ['shell']}))
        q = dict(s.DAEMON.reqs[-1])
        tm = q.pop('tool_mask', None)
        check('server: field: daemon gets the automaton', tm and tm['tags'] == [60, 61, 62, 63, 64, 65] and walk(tm, [10, 52]) == -1 and walk(tm, [20]) is None, str(tm))
        check('server: thinking span passed', tm and tm.get('think_end') == 70 and tm.get('think_start') == 71, str(tm))
        check('server: prompt open in <think>', tm and tm.get('in_think') is True, str(tm))
        check('server: the prompt is the same with and without the field', prompts[-1] == plain_p, prompts[-1])
        same = {k: q.get(k) for k in plain} == plain and set(q) == set(plain)
        check('server: ids, session, boundaries and every other daemon field unchanged', same, f'{q} vs {plain}')
        r = recs[-1]
        check('server: record', r.get('allowed_tools_n') == 1 and r.get('mask_applied') is True and r.get('allowed_violation') is False, str(r))
        # served without the constraint (daemon without the field in its done message), a call outside the list: returned as generated, flagged
        Enc.calls = [{'type': 'function', 'id': 'c', 'function': {'name': 'files_write', 'arguments': '{}'}}]
        s.DAEMON = D({})
        out = await s.chat(Req({**base, 'allowed_tools': ['shell']}))
        r = recs[-1]
        check('server: violation flagged, call returned', r.get('allowed_violation') is True and r.get('mask_applied') is False
              and out['choices'][0]['message']['tool_calls'][0]['function']['name'] == 'files_write', str(r))
        Enc.calls = [{'type': 'function', 'id': 'c', 'function': {'name': 'shell', 'arguments': '{}'}}]
        await s.chat(Req({**base, 'allowed_tools': ['shell']}))
        check('server: a call on the list is no violation', recs[-1].get('allowed_violation') is False, str(recs[-1]))
        Enc.calls = None
        for name, body in (('null', {'allowed_tools': None}), ('empty', {'allowed_tools': []}), ('tool_choice none', {'allowed_tools': ['shell'], 'tool_choice': 'none'}),
                           ('unknown', {'allowed_tools': ['nope']})):
            await s.chat(Req({**base, **body}))
            check(f'server: {name}: no mask', 'tool_mask' not in s.DAEMON.reqs[-1] and 'allowed_tools_n' not in recs[-1], str(s.DAEMON.reqs[-1]))
        import os
        os.environ['HIVE_TOOL_MASK'] = '0'
        try:
            await s.chat(Req({**base, 'allowed_tools': ['shell']}))
            check('server: HIVE_TOOL_MASK=0: unconstrained, still recorded', 'tool_mask' not in s.DAEMON.reqs[-1] and recs[-1].get('allowed_tools_n') == 1 and recs[-1].get('mask_applied') is False, str(recs[-1]))
        finally:
            del os.environ['HIVE_TOOL_MASK']
        s.FAMILY = 'glm5_next'
        check('server: another family: no automaton', s.tool_mask_for(['shell'], [1]) is None)
        s.FAMILY = 'deepseek_v4'
        check('server: this family again', s.tool_mask_for(['shell'], [1]) is not None)
        Enc.tool_parameter_tag_name = ' unknown tag'   # a tag piece that is not one token: off (the tag table is read once per tokenizer)
        s.TOK = Tok()
        check('server: a tag piece that is not one token: off', s.tool_mask_for(['shell'], [1]) is None)
        del Enc.tool_parameter_tag_name
        s.TOK = Tok()
        check('server: tags back', s.tool_mask_for(['shell'], [1]) is not None)
    asyncio.run(go())


def main():
    server_checks()
    with tempfile.TemporaryDirectory() as td:
        exe = T.build_daemon(td, name='hived_tm')
        daemon_checks(exe, td, 'plain', {})
        daemon_checks(exe, td, 'mtp', MTP_ENV)
        daemon_checks(exe, td, 'mtp-wrong', {**MTP_ENV, 'FAKE_MTP_WRONG': '3'})
        daemon_checks(exe, td, 'gate1', {'FAKE_MTP': '5', 'HIVE_TRACE_MTP': '1', 'FAKE_STEP_MS': '20', 'FAKE_VERIFY_MS': '20', 'FAKE_VERIFY_ROW_MS': '3', 'FAKE_MTP_DRAFT_MS': '2'})
        log = batch_checks(exe, td, 'batch', BATCH_ENV)
        acc = sum(1 for l in log.splitlines() if re.search(r'\[mtp\] batch .*accepted \[[1-9]', l))
        check('batch speculation actually ran', acc >= 10, f'{acc} batch steps')
        mutant_checks(td)
    if FAILS:
        print(f'tool mask CPU: {len(FAILS)} failures')
        sys.exit(1)
    print('tool mask CPU: OK')


if __name__ == '__main__':
    main()
