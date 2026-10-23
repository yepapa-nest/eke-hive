#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Session cache location endpoints of server/hive_server.py (GET /v1/hive/cache, POST /v1/hive/cache/load) and the conversation-key ->
session table behind them (ConvSessions), against a fake daemon on a real Unix socket (the production Daemon client talks to it).

Checked: a chat request with a conversation key records key -> session (session_id_for of that key and head; a changed head moves the key
to the new session; a request split to "~k" by a concurrent one keeps the base session); GET resolves several keys in one daemon "where"
call (unknown keys: session null, where none, no daemon call for them); POST load sends the latest session to the daemon "load" and
returns its answer; unknown key / missing key / daemon down / slow load (202 + "loading"); the table survives a restart (file next to the
socket, keys hashed — the raw key is not in the file). No model, no GPU.
Run: python3 tools/test_cache_server_cpu.py
"""
import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from starlette.datastructures import QueryParams

spec = importlib.util.spec_from_file_location('server_under_test', Path(__file__).resolve().parents[1] / 'server/hive_server.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class Request:
    def __init__(self, body=None, query=''):
        self.body, self.query_params = body, QueryParams(query)

    async def json(self): return self.body

    async def is_disconnected(self): return False


class Tokenizer:
    eos_token_id = 99
    eos_token = '<EOS>'

    def decode(self, ids, **kw): return ''.join({1: 'a', 2: 'b', 99: '<EOS>'}[i] for i in ids)


class Encoder:
    def parse_message_from_completion_text(self, text, **kw): return {'content': text.replace('<EOS>', '')}


class FakeHived:
    """A Unix-socket peer speaking the hived line protocol for generate / where / load. state: session -> (where, tokens, bytes)."""

    def __init__(self, path, load_delay=0.0):
        self.path, self.load_delay, self.ops, self.state = path, load_delay, [], {}

    async def peer(self, reader, writer):
        req = json.loads(await reader.readline())
        op = req.get('op', 'generate')
        self.ops.append(req)
        if op == 'generate':
            self.state[req['session']] = ('vram', len(req['ids']) + 1, 0)
            for line in ({'id': 1}, {'id': 2}, {'done': True, 'n': 2, 'finish': 'length', 'cached_prefix': 0}):
                writer.write((json.dumps(line) + '\n').encode())
        elif op == 'where':
            out = {sid: self.row(sid) for sid in req['sessions']}
            writer.write((json.dumps({'ok': True, 'sessions': out}) + '\n').encode())
        elif op == 'load':
            await asyncio.sleep(self.load_delay)
            sid = req['session']
            at, tok, b = self.state.get(sid, ('none', 0, 0))
            loaded = at == 'ssd'
            if loaded:
                self.state[sid] = ('ram', tok, b)
            writer.write((json.dumps({'ok': True, 'session': sid, 'loaded': loaded, **self.row(sid)}) + '\n').encode())
        try:
            await writer.drain()
        except ConnectionError:  # the client gave up (the slow-load case)
            pass
        writer.close()

    def row(self, sid):
        at, tok, b = self.state.get(sid, ('none', 0, 0))
        return {'where': at, 'tokens': tok, 'bytes': b, 'tiers': {} if at == 'none' else {at: {'tokens': tok, 'bytes': b}}}


def body(first, key=None, **kw):
    b = {'messages': [{'role': 'user', 'content': first}], 'reasoning_effort': 'none', 'max_tokens': 2, **kw}
    if key is not None:
        b['prompt_cache_key'] = key
    return b


class Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.td = tempfile.TemporaryDirectory(prefix='hive-cache-srv-')
        self.sock = str(Path(self.td.name) / 'hive.sock')
        self.fake = FakeHived(self.sock)
        self.server = await asyncio.start_unix_server(self.fake.peer, self.sock)
        s.TOK, s.ENC, s.INFLIGHT = Tokenizer(), Encoder(), {}
        s.encode_prompt = lambda *a: ('prompt', [])
        s.tokenize_with_images = lambda *a: ([10, 11, 12], [], b'')
        s.DAEMON = s.Daemon(self.sock)
        s.CONV_SESSIONS = s.ConvSessions()
        s.CONV_SESSIONS.load(str(Path(self.td.name) / 'conv-sessions.json'))

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        self.td.cleanup()

    async def get(self, *keys):
        r = await s.cache_where(Request(query='&'.join('key=' + k for k in keys)))
        return r if isinstance(r, dict) else json.loads(r.body)

    async def test_key_to_session_and_where(self):
        await s.chat(Request(body('hello', key='room-1')))
        sid = s.session_id_for([{'role': 'user', 'content': 'hello'}], 'room-1')
        self.assertEqual(s.CONV_SESSIONS.get('room-1'), sid)
        self.fake.ops.clear()
        r = await self.get('room-1', 'never-seen')
        self.assertEqual([o['op'] for o in self.fake.ops], ['where'])
        self.assertEqual(self.fake.ops[0]['sessions'], [sid])  # only the known session is asked for
        self.assertEqual(r['sessions'][0], {'key': 'room-1', 'session': sid, 'where': 'vram', 'tokens': 4, 'bytes': 0,
                                            'tiers': {'vram': {'tokens': 4, 'bytes': 0}}})
        self.assertEqual(r['sessions'][1], {'key': 'never-seen', 'session': None, 'where': 'none', 'tokens': 0, 'bytes': 0, 'tiers': {}})
        # a changed head (other first user message) is another session for the same key: the key follows the latest
        await s.chat(Request(body('hello again', key='room-1')))
        sid2 = s.session_id_for([{'role': 'user', 'content': 'hello again'}], 'room-1')
        self.assertNotEqual(sid, sid2)
        self.assertEqual((await self.get('room-1'))['sessions'][0]['session'], sid2)
        # no key: nothing recorded · no keys asked: no daemon call
        n = len(s.CONV_SESSIONS.entries)
        await s.chat(Request(body('anonymous')))
        self.assertEqual(len(s.CONV_SESSIONS.entries), n)
        self.fake.ops.clear()
        self.assertEqual(await self.get(), {'sessions': []})
        self.assertEqual(self.fake.ops, [])

    async def test_split_request_keeps_the_base_session(self):
        sid = s.session_id_for([{'role': 'user', 'content': 'hi'}], 'room-2')
        s.INFLIGHT[sid] = 1  # another request of this conversation is running: this one is sent as "<sid>~1"
        await s.chat(Request(body('hi', key='room-2')))
        self.assertEqual([o['session'] for o in self.fake.ops if o.get('op', 'generate') == 'generate'], [sid + '~1'])
        self.assertEqual(s.CONV_SESSIONS.get('room-2'), sid)

    async def test_hive_session_id(self):
        await s.chat(Request(body('x', key='room-3', hive_session_id='abc')))
        self.assertEqual(s.CONV_SESSIONS.get('room-3'), s.hashlib.sha256(b'abc').hexdigest()[:32])

    async def test_load(self):
        await s.chat(Request(body('hello', key='room-4')))
        sid = s.CONV_SESSIONS.get('room-4')
        self.fake.state[sid] = ('ssd', 700, 1234)
        self.fake.ops.clear()
        r = await s.cache_load(Request({'key': 'room-4'}))
        self.assertEqual(self.fake.ops, [{'op': 'load', 'session': sid}])
        self.assertEqual((r['key'], r['session'], r['loaded'], r['where'], r['tokens'], r['bytes']), ('room-4', sid, True, 'ram', 700, 1234))
        self.assertNotIn('ok', r)
        r = await s.cache_load(Request({'key': 'room-4'}))  # already in RAM: the daemon's answer, loaded false
        self.assertEqual((r['loaded'], r['where']), (False, 'ram'))
        r = await s.cache_load(Request({'key': 'nobody'}))
        self.assertEqual((r['session'], r['loaded'], r['where'], r['reason']), (None, False, 'none', 'unknown key'))
        for bad in ({}, {'key': 5}, {'key': ''}):
            self.assertEqual((await s.cache_load(Request(bad))).status_code, 400)

    async def test_slow_load_and_daemon_down(self):
        await s.chat(Request(body('hello', key='room-5')))
        sid = s.CONV_SESSIONS.get('room-5')
        self.fake.state[sid] = ('ssd', 300, 10)
        self.fake.load_delay = 0.5
        old, s.CACHE_LOAD_TIMEOUT_S = s.CACHE_LOAD_TIMEOUT_S, 0.1
        try:
            r = await s.cache_load(Request({'key': 'room-5'}))
        finally:
            s.CACHE_LOAD_TIMEOUT_S = old
        self.assertEqual(r.status_code, 202)
        d = json.loads(r.body)
        self.assertEqual((d['loading'], d['loaded'], d['where']), (True, False, 'ssd'))
        s.DAEMON = s.Daemon(str(Path(self.td.name) / 'missing.sock'))
        self.assertEqual((await s.cache_where(Request(query='key=room-5'))).status_code, 503)
        self.assertEqual((await s.cache_load(Request({'key': 'room-5'}))).status_code, 503)

    async def test_persisted_across_restart(self):
        await s.chat(Request(body('hello', key='room-secret-name')))
        sid = s.CONV_SESSIONS.get('room-secret-name')
        await asyncio.sleep(s.ConvSessions.SAVE_DELAY_S + 0.5)  # the timer thread writes the file
        path = Path(self.td.name) / 'conv-sessions.json'
        text = path.read_text()
        self.assertIn(sid, text)
        self.assertNotIn('room-secret-name', text)  # keys are kept hashed
        again = s.ConvSessions()
        again.load(str(path))
        self.assertEqual(again.get('room-secret-name'), sid)
        # a damaged file is ignored (learnt again from requests)
        path.write_text('{not json')
        third = s.ConvSessions()
        third.load(str(path))
        self.assertIsNone(third.get('room-secret-name'))

    def test_lru_cap(self):
        c = s.ConvSessions(cap=3)
        for i in range(5):
            c.note(f'k{i}', f's{i}')
        self.assertEqual([c.get(f'k{i}') for i in range(5)], [None, None, 's2', 's3', 's4'])


if __name__ == '__main__':
    unittest.main()
