#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Server-level session tests: production server/hive_server.py handlers (in-process ASGI-free
calls, as tools/test_server_cpu.py does) talking over a real Unix socket to the REAL hived.cpp on
CPU (tools/test_daemon_cpu.py harness, deterministic state oracle). No model, no GPU.

Covers: session reuse across turns through session_id_for, editing the latest / the first user
message (no mismatched prefix reuse), image change invalidation end to end, stream cancellation
releasing the server reservation AND the daemon session, concurrent conversations, two requests
of one conversation at once (server splits the session), Anthropic stream close.
Fake tokenizer/template: role marker + one token per character; assistant text maps back to the
exact generated ids, so turn N+1's prompt extends turn N's prompt + reply (as the real template
does when there is no dropped reasoning).
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
sys.path[:0] = [str(HERE / 'cpu_fake')]
import harness  # noqa: E402
import test_daemon_cpu as d  # noqa: E402

spec = importlib.util.spec_from_file_location('server_under_test', HERE.parent / 'server/hive_server.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

ROLE = {'system': 11, 'user': 12, 'assistant': 13, 'tool': 14}
BASE = 0x4E00


class Tokenizer:
    eos_token_id = 5  # the fake daemon never emits < 10
    eos_token = '<EOS>'

    def decode(self, ids, **kw):
        return ''.join(chr(BASE + i) for i in ids)


def char_ids(text):
    # ids stay inside the fake checkpoint's vocabulary (4096 — the daemon rejects ids outside [0, vocab)): chars that decode() produced
    #   round-trip exactly (BASE + id); other wide chars (Hangul in the prompts) fold into the vocabulary; ASCII as before
    out = []
    for c in text:
        o = ord(c)
        out.append(o - BASE if BASE + 10 <= o < BASE + 4096 else 20 + (o - BASE) % 4000 if o >= BASE + 4096 else 20 + o % 3000)
    return out


class Encoder:
    def parse_message_from_completion_text(self, text, **kw):
        return {'content': text.replace('<EOS>', '')}


def fake_encode(messages, tools, thinking_mode, effort):
    ids, images = [], []
    for m in messages:
        ids.append(ROLE[m['role']])
        content = m.get('content') or ''
        parts = content if isinstance(content, list) else [{'type': 'text', 'text': content}]
        for p in parts:
            if p['type'] == 'text':
                ids += char_ids(p['text'])
            else:  # fake image: 3x3 patches, 9 placeholder positions
                meta, blob = d.image(len(ids), 3, 3, p['seed'])
                ids += [s.CFG.get('image_token_id', 3999)] * 9
                images.append((meta, blob))
    ids.append(ROLE['assistant'])
    return json.dumps({'ids': ids, 'images': [[m, b.hex()] for m, b in images]}), images


def fake_tokenize(prompt, media):
    p = json.loads(prompt)
    return p['ids'], [m for m, _ in p['images']], b''.join(bytes.fromhex(b) for _, b in p['images'])


def prompt_of(messages):
    p = json.loads(fake_encode(messages, None, 'chat', None)[0])
    return p['ids'], [(m, bytes.fromhex(b)) for m, b in p['images']]


class Request:
    def __init__(self, body): self.body = body
    async def json(self): return self.body
    async def is_disconnected(self): return False


def body(messages, n=6, **kw):
    return {'messages': messages, 'reasoning_effort': 'none', 'max_tokens': n, 'temperature': 0, **kw}


def text_of(ids):
    return Tokenizer().decode(ids)


class Tests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory(prefix='hive-server-session-')
        cls.exe = harness.build_daemon(cls.td.name)
        cls.h = d.Hived(cls.exe, cls.td.name, {'FAKE_STEP_MS': '10'}, name='srv')

    @classmethod
    def tearDownClass(cls):
        log = cls.h.log_path.read_text(errors='replace')
        rc = cls.h.stop()
        cls.td.cleanup()
        assert rc == -15 and 'ThreadSanitizer' not in log, log[-3000:]

    def setUp(self):
        s.TOK, s.ENC, s.INFLIGHT = Tokenizer(), Encoder(), {}
        s.encode_prompt, s.tokenize_with_images = fake_encode, fake_tokenize
        s.DAEMON = s.Daemon(self.h.sock)

    async def turn(self, messages, n=6, **kw):
        r = await s.chat(Request(body(messages, n, **kw)))
        self.assertIsInstance(r, dict, getattr(r, 'body', r))
        ids, images = prompt_of(messages)
        self.assertEqual(r['choices'][0]['message']['content'], text_of(d.oracle(ids, n, images)), 'server text != state oracle')
        return r, ids

    async def test_reuse_edit_latest_and_first(self):
        u1 = [{'role': 'user', 'content': '첫 질문 ' + 'x' * 300}]
        r1, p1 = await self.turn(u1)
        self.assertEqual(r1['hive']['cached_prefix'], 0)
        a1 = {'role': 'assistant', 'content': r1['choices'][0]['message']['content']}
        u2 = u1 + [a1, {'role': 'user', 'content': '두 번째 ' + 'y' * 40}]
        r2, p2 = await self.turn(u2)
        self.assertEqual(r2['hive']['cached_prefix'], len(p1) + 6 - 1, 'second turn must continue the live state')
        edited = u1 + [a1, {'role': 'user', 'content': '두 번째 ' + 'z' * 40}]  # edit the latest user message
        r3, p3 = await self.turn(edited)
        self.assertEqual(r3['hive']['cached_prefix'], 0, 'edited turn: no mismatched prefix reuse (default HIVE_CKPT_HISTORY=1)')
        first_edit = [{'role': 'user', 'content': '첫 질문 수정 ' + 'x' * 300}]
        r4, _ = await self.turn(first_edit)
        self.assertEqual(r4['hive']['cached_prefix'], 0, 'edited first message is a new conversation')
        r5, _ = await self.turn(u2)  # the original conversation's session was not disturbed by the other one
        self.assertIn(r5['hive']['cached_prefix'], (0, len(p2)))
        self.assertFalse(s.INFLIGHT)

    async def test_image_change_invalidates(self):
        msgs = lambda seed, extra='': [{'role': 'user', 'content': [{'type': 'text', 'text': '사진 ' + 'p' * 280}, {'type': 'image', 'seed': seed},
                                                                    {'type': 'text', 'text': '설명해 ' + extra}]}]
        r1, _ = await self.turn(msgs(1))
        r2, _ = await self.turn(msgs(1) + [{'role': 'assistant', 'content': r1['choices'][0]['message']['content']}, {'role': 'user', 'content': '더'}])
        self.assertGreater(r2['hive']['cached_prefix'], 0, 'same image: continuation reuses the live state')
        # same conversation text, different picture (same size, same placeholder ids): must not reuse
        m3 = msgs(2) + [{'role': 'assistant', 'content': r1['choices'][0]['message']['content']}, {'role': 'user', 'content': '더'}]
        r3, _ = await self.turn(m3)
        self.assertEqual(r3['hive']['cached_prefix'], 0)

    async def test_stream_cancel_releases_server_and_daemon_session(self):
        msgs = [{'role': 'user', 'content': '길게 ' + 'c' * 270}]
        resp = await s.chat(Request(body(msgs, 400, stream=True)))
        session = next(iter(s.INFLIGHT))
        seen = 0
        async for chunk in resp.body_iterator:
            if '"content"' in chunk:
                seen += 1
                if seen == 3:
                    break
        await resp.body_iterator.aclose()
        self.assertFalse(s.INFLIGHT, 'server reservation released on close')
        r, _ = await self.turn(msgs, 4)  # same conversation right away: same session id, daemon must accept it
        self.assertNotIn('~', session)
        self.assertIsNone(r.get('error'))
        st = self.h.stats()
        self.assertEqual(st['pending_requests'], 0)

    async def test_concurrent_conversations_and_same_conversation_twice(self):
        async def convo(k):
            u = [{'role': 'user', 'content': f'대화 {k} ' + 'q' * (260 + k)}]
            r1, p1 = await self.turn(u)
            u2 = u + [{'role': 'assistant', 'content': r1['choices'][0]['message']['content']}, {'role': 'user', 'content': '계속'}]
            r2, _ = await self.turn(u2)
            return r2['hive']['cached_prefix'], len(p1)
        out = await asyncio.gather(*(convo(k) for k in range(4)))
        for cached, p1 in out:
            self.assertEqual(cached, p1 + 5)
        same = [{'role': 'user', 'content': '동시 ' + 'w' * 270}]
        a, b = await asyncio.gather(self.turn(same), self.turn(same))  # second is split into "<session>~1" by the server (B8 stable slot)
        self.assertFalse(s.INFLIGHT)

    async def test_d5_late_cancel_does_not_kill_next_request(self):
        # D5 against the REAL hived: the stream's client drops and its cancel is slow (100 ms, like a loaded box);
        # the same conversation's next request arrives inside that window and must finish with the oracle text.
        msgs = [{'role': 'user', 'content': '늦은 취소 ' + 'd' * 270}]
        real = s.DAEMON.cancel
        def slow_cancel(session, rid=''):
            import time; time.sleep(.1); real(session, rid)
        s.DAEMON.cancel = slow_cancel
        resp = await s.chat(Request(body(msgs, 400, stream=True)))
        seen = 0
        async for chunk in resp.body_iterator:
            if '"content"' in chunk:
                seen += 1
                if seen == 2:
                    break
        closing = asyncio.create_task(resp.body_iterator.aclose())
        await asyncio.sleep(.02)
        r, _ = await self.turn(msgs, 40)  # 40 steps x 10 ms: a late cancel (~100 ms) would land inside this request
        await closing
        self.assertEqual(r['choices'][0]['finish_reason'], 'length')
        self.assertFalse(s.INFLIGHT)
        self.assertEqual(self.h.stats()['pending_requests'], 0)

    async def test_anthropic_stream_close_releases(self):
        oa = {'messages': [{'role': 'user', 'content': 'anthropic ' + 'a' * 270}], 'max_tokens': 400, 'stream': True,
              'thinking': {'type': 'disabled'}, 'temperature': 0}
        resp = await s.anthropic_messages(Request(oa))
        n = 0
        async for chunk in resp.body_iterator:
            if 'text_delta' in chunk:
                n += 1
                if n == 2:
                    break
        await resp.close()
        self.assertFalse(s.INFLIGHT)
        r, _ = await self.turn(oa['messages'], 3)


# H5 end to end: the server computes boundary hints from a concatenative template (exact token offsets via the
# tokenizer's offset mapping) and the REAL hived (HIVE_PREFIX_SHARE=1) cuts/snapshots there; another conversation with the
# same system prompt and an edit of the latest message reuse them. Tokens round-trip: decode(id) re-encodes to id.
SHARE_MARKERS = {'<|system|>': 11, '<|user|>': 12, '<|assistant|>': 13}


class ShareTok(Tokenizer):
    def _split(self, text):
        out, i = [], 0
        while i < len(text):
            m = next((m for m in SHARE_MARKERS if text.startswith(m, i)), None)
            if m:
                out.append((SHARE_MARKERS[m], i, i + len(m))); i += len(m)
            else:
                out.append((char_ids(text[i])[0], i, i + 1)); i += 1
        return out

    def encode(self, text):
        return [t for t, _, _ in self._split(text)]

    def __call__(self, text, return_offsets_mapping=False):
        sp = self._split(text)
        return {'input_ids': [t for t, _, _ in sp], 'offset_mapping': [(a, b) for _, a, b in sp]}


def share_encode(messages, tools, thinking_mode, effort):
    return ''.join('<|' + m['role'] + '|>' + (m.get('content') or '') for m in messages) + '<|assistant|>', []


class ShareTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory(prefix='hive-server-share-')
        cls.exe = harness.build_daemon(cls.td.name)
        cls.h = d.Hived(cls.exe, cls.td.name, {'FAKE_STEP_MS': '10', 'HIVE_PREFIX_SHARE': '1'}, name='srvshare')

    @classmethod
    def tearDownClass(cls):
        log = cls.h.log_path.read_text(errors='replace')
        rc = cls.h.stop()
        cls.td.cleanup()
        assert rc == -15 and 'ThreadSanitizer' not in log, log[-3000:]

    def setUp(self):
        s.TOK, s.ENC, s.INFLIGHT = ShareTok(), Encoder(), {}
        s.encode_prompt = share_encode
        s.tokenize_with_images = lambda prompt, media: (s.TOK.encode(prompt), [], b'')
        s.DAEMON = s.Daemon(self.h.sock)
        self.saved = os.environ.get('HIVE_PREFIX_SHARE')
        os.environ['HIVE_PREFIX_SHARE'] = '1'

    def tearDown(self):
        if self.saved is None: os.environ.pop('HIVE_PREFIX_SHARE', None)
        else: os.environ['HIVE_PREFIX_SHARE'] = self.saved

    async def turn(self, messages, n=6):
        r = await s.chat(Request(body(messages, n)))
        self.assertIsInstance(r, dict, getattr(r, 'body', r))
        ids = s.TOK.encode(share_encode(messages, None, 'chat', None)[0])
        self.assertEqual(r['choices'][0]['message']['content'], text_of(d.oracle(ids, n)), 'server text != state oracle')
        return r, ids

    async def test_shared_system_prefix_and_edit_reuse_boundaries(self):
        SYS = {'role': 'system', 'content': '시스템 ' + 's' * 300}
        uA = {'role': 'user', 'content': 'A 질문 ' + 'a' * 40}
        r1, p1 = await self.turn([SYS, uA])
        self.assertEqual(r1['hive']['cached_prefix'], 0)
        sys_end = len(s.TOK.encode('<|system|>' + SYS['content'] + '<|user|>'))
        r2, _ = await self.turn([SYS, {'role': 'user', 'content': 'B 다른 질문 ' + 'b' * 50}])  # another conversation, same system prompt
        self.assertEqual(r2['hive']['cached_prefix'], sys_end, 'cross-session reuse of the system boundary snapshot')
        a1 = {'role': 'assistant', 'content': r1['choices'][0]['message']['content']}
        r3, _ = await self.turn([SYS, uA, a1, {'role': 'user', 'content': '이어서 ' + 'c' * 30}])
        self.assertEqual(r3['hive']['cached_prefix'], len(p1) + 5, 'turn 2 continues the live state')
        turn_end = len(p1) + 6 + 1  # prompt + 6 reply tokens + the next user marker
        r4, _ = await self.turn([SYS, uA, a1, {'role': 'user', 'content': '고쳐서 ' + 'd' * 30}])  # edit the latest user message
        self.assertEqual(r4['hive']['cached_prefix'], turn_end, 'edited latest message reuses the completed-turn boundary')
        self.assertFalse(s.INFLIGHT)


if __name__ == '__main__':
    unittest.main()
