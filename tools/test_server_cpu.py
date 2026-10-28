#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Production handlers + fake token source and real Unix EOF. No model/GPU."""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from array import array

spec = importlib.util.spec_from_file_location('server_under_test', Path(__file__).resolve().parents[1] / 'server/hive_server.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
ORIG_ENCODE_PROMPT = s.encode_prompt  # the real encode_prompt (some tests below replace the module's)

class Request:
    def __init__(self, body): self.body = body
    async def json(self): return self.body
    async def is_disconnected(self): return False

class Tokenizer:
    eos_token_id = 99
    eos_token = '<EOS>'
    def decode(self, ids, **kw): return ''.join({1:'a', 2:'b', 3:'c', 99:'<EOS>'}[i] for i in ids)

class Encoder:
    def parse_message_from_completion_text(self, text, **kw): return {'content':text.replace('<EOS>', '')}

class Daemon:
    def __init__(self, eof=False, delay=0): self.eof=eof; self.delay=delay; self.cancelled=[]; self.closed=0; self.sessions=[]; self.emitted=0
    async def generate(self, session, *args, **kw):
        self.sessions.append(session)
        try:
            for i in (1,2,3):
                await asyncio.sleep(self.delay)
                self.emitted += 1
                yield {'id':i}
                if self.eof: return
            yield {'done':True, 'n':3, 'finish':'stop'}
        finally: self.closed += 1
    def cancel(self, session, rid=''): self.cancelled.append(session)
    async def stats(self): return {'build_id':'fake', 'unique_resident':2}

BODY = {'messages':[{'role':'user','content':'hello'}], 'reasoning_effort':'none'}

class Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        s.TOK=Tokenizer(); s.ENC=Encoder(); s.DAEMON=Daemon(); s.INFLIGHT.clear()
        s.encode_prompt=lambda *a: ('prompt', [])
        s.tokenize_with_images=lambda *a: ([10,11], [], b'')

    async def test_stop_and_usage(self):
        response=await s.chat(Request({**BODY, 'stop':'bc'}))
        self.assertEqual(response['choices'][0]['message']['content'], 'a')
        self.assertEqual(response['usage']['completion_tokens'], 3)
        self.assertEqual(len(s.DAEMON.cancelled), 1)
        self.assertFalse(s.INFLIGHT)

    async def test_eof_never_success(self):
        s.DAEMON=Daemon(eof=True)
        response=await s.chat(Request(BODY))
        self.assertEqual(response.status_code, 502)
        self.assertEqual(json.loads(response.body)['partial_content'], 'a')
        response=await s.chat(Request({**BODY,'stream':True}))
        chunks=[x async for x in response.body_iterator]
        self.assertTrue(any('"error"' in x for x in chunks))
        self.assertFalse(any('"finish_reason": "' in x for x in chunks))
        self.assertFalse(s.INFLIGHT)

    async def test_stream_is_live(self):
        for handler in (s.chat, s.anthropic_messages):
            s.DAEMON=Daemon(delay=.025)
            start=time.monotonic()
            response=await handler(Request({**BODY,'stream':True,'thinking':{'type':'disabled'}}))
            first=None
            async for chunk in response.body_iterator:
                if '"content": "a"' in chunk or '"text": "a"' in chunk:
                    first=time.monotonic()-start
                    self.assertEqual(s.DAEMON.emitted, 1, 'first content buffered until completion')
            self.assertIsNotNone(first)
            print(f'{handler.__name__}: first content {first*1000:.2f}ms, source 3x25ms')
        self.assertFalse(s.INFLIGHT)

    async def test_close_cancels_and_releases(self):
        response=await s.chat(Request({**BODY,'stream':True}))
        await anext(response.body_iterator)  # role
        await anext(response.body_iterator)  # token
        await response.body_iterator.aclose()
        self.assertEqual(len(s.DAEMON.cancelled),1)
        self.assertEqual(s.DAEMON.closed,1)
        self.assertFalse(s.INFLIGHT)

    async def test_close_before_source_and_parallel_reservation(self):
        a=await s.chat(Request({**BODY,'stream':True}))
        b=await s.chat(Request({**BODY,'stream':True}))
        self.assertEqual(len(s.INFLIGHT),2)
        for response in (a,b):
            await anext(response.body_iterator)
            await response.body_iterator.aclose()
        self.assertFalse(s.INFLIGHT)

    async def test_close_before_first_iteration(self):
        for handler in (s.chat,s.anthropic_messages):
            response=await handler(Request({**BODY,'stream':True}))
            self.assertTrue(s.INFLIGHT)
            await response.close()
            self.assertFalse(s.INFLIGHT)

    async def test_anthropic_thinking_tool_usage(self):
        async def chunks():
            for delta,finish in [({'reasoning_content':'reason'},None),({'content':'text'},None),
                ({'tool_calls':[{'id':'call1','function':{'name':'f','arguments':'{"x":1}'}}]},'tool_calls')]:
                yield 'data: '+json.dumps({'choices':[{'delta':delta,'finish_reason':finish}],
                                           'usage':{'prompt_tokens':7,'completion_tokens':5}})+'\n\n'
        response=s.ClosingStreamingResponse(chunks());response.prompt_tokens=7
        out=''.join([x async for x in s.anthropic_stream(response)])
        for expected in ('thinking_delta','text_delta','input_json_delta','"stop_reason": "tool_use"','"input_tokens": 7','"output_tokens": 5','event: message_stop'):
            self.assertIn(expected,out)

    async def test_cached_prefix_usage(self):
        # the daemon's cached_prefix → OpenAI prompt_tokens_details.cached_tokens (capped at the prompt) and the Anthropic split
        class CachedDaemon(Daemon):
            def __init__(self, cached): super().__init__(); self.cached = cached
            async def generate(self, session, *args, **kw):
                async for m in super().generate(session, *args, **kw):
                    yield {**m, 'cached_prefix': self.cached} if m.get('done') else m
        for cached, want in ((1, 1), (0, 0), (9, 2)):  # prompt = 2 tokens ([10, 11])
            s.DAEMON = CachedDaemon(cached)
            response = await s.chat(Request(BODY))
            self.assertEqual(response['usage']['prompt_tokens'], 2)
            self.assertEqual(response['usage']['prompt_tokens_details']['cached_tokens'], want)
        self.assertEqual(s.anthropic_usage({'prompt_tokens': 7, 'completion_tokens': 5, 'prompt_tokens_details': {'cached_tokens': 4}}),
                         {'input_tokens': 3, 'output_tokens': 5, 'cache_read_input_tokens': 4, 'cache_creation_input_tokens': 0})
        self.assertEqual(s.anthropic_usage({'prompt_tokens': 7, 'completion_tokens': 5})['input_tokens'], 7)  # no details = nothing cached
        async def chunks():
            yield 'data: ' + json.dumps({'choices': [{'delta': {'content': 'x'}, 'finish_reason': 'stop'}],
                                         'usage': {'prompt_tokens': 7, 'completion_tokens': 1, 'prompt_tokens_details': {'cached_tokens': 6}}}) + '\n\n'
        response = s.ClosingStreamingResponse(chunks()); response.prompt_tokens = 7
        out = ''.join([x async for x in s.anthropic_stream(response)])
        self.assertIn('"cache_read_input_tokens": 6', out)
        self.assertIn('"input_tokens": 1', out)

    async def test_anthropic_error(self):
        s.DAEMON=Daemon(eof=True)
        response=await s.anthropic_messages(Request({**BODY,'stream':True}))
        chunks=''.join([x async for x in response.body_iterator])
        self.assertIn('event: error',chunks)
        self.assertNotIn('event: message_stop',chunks)

    async def test_actual_unix_eof(self):
        async def peer(reader, writer):
            await reader.readline()
            writer.write(b'{"id":1}\n'); await writer.drain()
            writer.close(); await writer.wait_closed()
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'test.sock')
            server=await asyncio.start_unix_server(peer,path)
            async with server:
                out=[x async for x in s.Daemon(path).generate('test',[1],[],b'')]
            self.assertEqual(out[0], {'id':1})
            self.assertIn('error',out[-1])

    async def test_actual_unix_large_payload(self):
        # a request of >= Daemon.BLOCKING_SEND_BYTES is written by a worker thread (one blocking sendall) and then read through asyncio:
        #   the daemon must receive the header and every payload byte in order, and the reply must still stream back
        blob = os.urandom(3 * s.Daemon.BLOCKING_SEND_BYTES + 12345)
        got = {}
        async def peer(reader, writer):
            head = json.loads(await reader.readline())
            got['bin'] = head['bin']
            got['blob'] = await reader.readexactly(head['bin'])
            writer.write(b'{"id":7}\n{"done":true,"n":1}\n'); await writer.drain()
            writer.close(); await writer.wait_closed()
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'test.sock')
            server=await asyncio.start_unix_server(peer,path)
            async with server:
                out=[x async for x in s.Daemon(path).generate('test',[1,2,3],[{'start':0}],blob)]
        self.assertEqual(got['bin'], len(blob))
        self.assertEqual(hashlib.sha256(got['blob']).digest(), hashlib.sha256(blob).digest())
        self.assertEqual(out, [{'id':7}, {'done':True,'n':1}])
        with self.assertRaises(OSError):
            [x async for x in s.Daemon('/nonexistent-hive-test-socket').generate('test',[1],[],blob)]

    async def test_health(self):
        self.assertTrue((await s.health())['ok'])
        s.DAEMON=s.Daemon('/nonexistent-hive-test-socket')
        self.assertEqual((await s.health()).status_code,503)

    def test_stop_fragments(self):
        for fragments, stops, expected in [(['a','b','c'],['bc'],'a'), (['ab','x'],['bc'],'abx'),
                                            (['abc'],['b','bc'],'a'), (['a','ab','a'],['aba'],'a')]:
            b=s.StopBuffer(stops)
            result=''
            for f in fragments:
                result+=b.push(f)
                if b.stopped: break
            result+=b.finish()
            self.assertEqual(result,expected)

# ---- regression tests (cancellation, stop sequences inside reasoning/tool blocks, parallel slots) ------------------------------------------
VOC = {1:'a', 2:'b', 3:'c', 4:'.', 5:'</think>', 6:'X', 7:'Y', 8:'\n\n<｜DSML｜', 9:'invoke name="f">', 99:'<EOS>'}

class VocTokenizer(Tokenizer):
    def decode(self, ids, **kw): return ''.join(VOC[i] for i in ids)

class ThinkEncoder:
    """Reference-parser stand-in: thinking splits at </think>; a DSML block becomes one tool call."""
    def parse_message_from_completion_text(self, text, thinking_mode=None, **kw):
        text = text.replace('<EOS>', '')
        out = {}
        if thinking_mode == 'thinking':
            if '</think>' not in text: raise ValueError('no end think')
            out['reasoning_content'], text = text.split('</think>', 1)
        if '<｜DSML｜' in text:
            text, tool = text.split('\n\n<｜DSML｜', 1)
            if 'name=' not in tool: raise ValueError('bad dsml')
            out['tool_calls'] = [{'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}}]
        out['content'] = text
        return out

class FlagDaemon:
    """Models hived cancel_flags (hived.cpp op=="cancel"): cancel(session) flags the NEWEST request of that session.
    `silent` = seconds of prefill-like silence before the first token (flag still honoured)."""
    def __init__(self, seq, delay=0.0, cancel_delay=0.0, silent=0.0):
        self.seq=seq; self.delay=delay; self.cancel_delay=cancel_delay; self.silent=silent
        self.flags={}; self.log=[]; self.cancelled=[]; self.rids=[]; self.cancel_rids=[]
    async def generate(self, session, *a, **kw):
        flag={'c':False}; self.flags[session]=flag; self.log.append(('gen', session)); self.rids.append(kw.get('rid', ''))
        t0=time.monotonic()
        while time.monotonic()-t0 < self.silent and not flag['c']: await asyncio.sleep(.005)
        n=0
        for tid in self.seq:
            if flag['c']: yield {'done':True, 'n':n, 'finish':'cancel'}; return
            await asyncio.sleep(self.delay)
            n+=1; yield {'id':tid}
        yield {'done':True, 'n':n, 'finish':'stop'}
    def cancel(self, session, rid=''):
        # rid is only recorded — the flag is set on the session's latest request (old daemon rule), so the server's cancel→release order itself is tested
        time.sleep(self.cancel_delay)
        self.log.append(('cancel', session)); self.cancelled.append(session); self.cancel_rids.append(rid)
        f=self.flags.get(session)
        if f: f['c']=True

def sse_json(chunks):
    return [json.loads(c[6:]) for c in chunks if c.startswith('data: {')]

class StageTokenizer(Tokenizer):
    PIECES = {1: '<think>', 2: 'hm', 3: '</think>', 4: '<tool_call>', 5: 'read_file', 6: '<arg_key>', 7: 'p', 8: '</arg_key>', 9: '</tool_call>',
              21: 'x', 22: '<｜DSML｜function_calls>\n<｜DSML｜invoke name="web_search">', 99: '<EOS>'}
    def decode(self, ids, **kw): return ''.join(self.PIECES[i] for i in ids)

class StageDaemon(Daemon):
    """admitted (only when asked) → progress → generated ids; records the request's keyword arguments and the stage seen after each id."""
    def __init__(self, ids, seen): super().__init__(); self.ids = ids; self.seen = seen; self.kw = None
    async def generate(self, session, *args, **kw):
        self.kw = kw
        if kw.get('stages'):
            yield {'admitted': True, 'cached': 900, 'total': 1000}
        yield {'progress': 1000, 'total': 1000}
        for i in self.ids:
            await asyncio.sleep(0)
            yield {'id': i}
            self.seen.append(dict(s.STAGES.get(next(iter(s.STAGES)), {})) if s.STAGES else None)
        yield {'done': True, 'n': len(self.ids), 'finish': 'stop'}

def strip_ids(chunks):
    return [json.dumps({k: v for k, v in json.loads(c[6:]).items() if k not in ('id', 'created')}) if c.startswith('data: {') else c for c in chunks]

class StageTests(unittest.IsolatedAsyncioTestCase):
    """HIVE_STAGE_STATUS: off = nothing changes (no "stages" field, the route answers 404); on = the same response bytes (apart from the random
    id) plus a status entry that follows queued → prefill (reused / to read / read) → thinking → tool_call (name) → done."""
    def setUp(self):
        s.TOK = StageTokenizer(); s.ENC = Encoder(); s.INFLIGHT.clear(); s.STAGES.clear()
        s.encode_prompt = lambda *a: ('prompt', [])
        s.tokenize_with_images = lambda *a: (list(range(1000)), [], b'')
        self.saved = (s.STAGE_STATUS, s.DSML_START)
    def tearDown(self):
        s.STAGE_STATUS, s.DSML_START = self.saved; s.STAGES.clear()

    async def stream(self, ids, on):
        s.STAGE_STATUS = on
        seen = []
        s.DAEMON = StageDaemon(ids, seen)
        response = await s.chat(Request({**BODY, 'stream': True, 'reasoning_effort': 'high'}))
        chunks = [x async for x in response.body_iterator]
        return chunks, seen, s.DAEMON.kw

    async def test_off_changes_nothing(self):
        s.DSML_START = '<tool_call>'
        chunks_off, seen, kw = await self.stream([1, 2, 3, 4, 5, 6, 7, 8, 9], False)
        self.assertNotIn('stages', kw)
        self.assertFalse(s.STAGES)
        rid = json.loads(chunks_off[0][6:])['id']
        self.assertEqual((await s.stage_status(rid)).status_code, 404)
        s.STAGES.clear()
        chunks_on, seen, kw = await self.stream([1, 2, 3, 4, 5, 6, 7, 8, 9], True)
        self.assertTrue(kw.get('stages'))
        self.assertEqual(strip_ids(chunks_on), strip_ids(chunks_off))  # the status is a side channel: same stream either way

    async def test_glm_stages(self):
        s.DSML_START = '<tool_call>'
        chunks, seen, kw = await self.stream([1, 2, 3, 4, 5, 6, 7, 8, 9], True)
        rid = json.loads(chunks[0][6:])['id']
        stages = [x and x['stage'] for x in seen]
        self.assertEqual(seen[0]['cached_tokens'], 900); self.assertEqual(seen[0]['prefill_total'], 100); self.assertEqual(seen[0]['prefill_done'], 100)
        self.assertEqual(stages[:2], ['thinking', 'thinking'])
        self.assertIn('tool_call', stages)
        self.assertEqual(seen[-1].get('tool_name'), 'read_file')
        final = await s.stage_status(rid)
        self.assertEqual(final['stage'], 'done'); self.assertEqual(final['id'], rid)

    async def test_ds_tool_name_and_answer(self):
        s.DSML_START = '<｜DSML｜'
        chunks, seen, kw = await self.stream([3, 21, 22], True)
        self.assertEqual([x['stage'] for x in seen], ['answer', 'answer', 'tool_call'])
        self.assertEqual(seen[-1].get('tool_name'), 'web_search')

    async def test_prefill_rate_eta_and_seen(self):
        """Time left from a cost per token that grows with the position (a dense model slows ~4x over a long prompt, measured
        2026-10-15): fitted to this request's chunks, the prior fit before the first chunk ends, kept across a restart in a file.
        seen_before = the server served the conversation key before."""
        s.STAGE_STATUS = True
        saved_time, saved_cost = s.time.time, dict(s.PREFILL_COST)
        clock = [1000.0]
        s.time.time = lambda: clock[0]
        A, B = 1 / 5000, 4e-9                                                     # 5,000 tok/s at 0 -> 1,000 tok/s at 200K
        true = lambda p0, p1: A * (p1 - p0) + B * (p1 * p1 - p0 * p0) / 2
        def read(rid, cached, total, step):
            etas = []
            for done in range(step, total + 1, step):
                clock[0] += true(cached + done - step, cached + done); s.stage_prefill(rid, done)
                etas.append((s.STAGES[rid]['prefill_eta_s'], true(cached + done, cached + total)))
            return etas
        try:
            s.PREFILL_COST.update(a=None, b=None); s.STAGE_SEEN.clear()
            s.stage_open('a', 200000, 'conv1')
            self.assertFalse(s.STAGES['a']['seen_before'])
            s.stage_prefill('a', 0, total=200000, cached=0)
            self.assertIsNone(s.STAGES['a']['prefill_eta_s'])                    # nothing known yet
            etas = read('a', 0, 200000, 20000)
            for got, want in etas[3:]:                                            # once the chunks span a quarter of the read: within 2 %
                self.assertAlmostEqual(got, want, delta=max(0.2, 0.02 * want))
            self.assertEqual(s.STAGES['a']['prefill_rate'], 1000)
            s.stage_open('b', 1000, 'conv2'); s.stage_prefill('b', 0, total=1000, cached=0)
            clock[0] += 2.0; s.stage_prefill('b', 1000)                           # a short read does not replace the prior
            self.assertAlmostEqual(s.PREFILL_COST['b'], B, delta=B * 0.02)
            s.stage_open('c', 300000, 'conv1')                                    # the same conversation again
            self.assertTrue(s.STAGES['c']['seen_before'])
            s.stage_prefill('c', 0, total=150000, cached=50000)                   # before its first chunk: the prior fit
            self.assertAlmostEqual(s.STAGES['c']['prefill_eta_s'], true(50000, 200000), delta=0.02 * true(50000, 200000))
            s.stage_prefill('zzz', 5)                                             # unknown request: ignored
            with tempfile.TemporaryDirectory() as d:                             # the fit survives a restart (file next to the socket)
                f = os.path.join(d, 'prefill-rate.json')
                s.prefill_rate_load(f)                                            # no file yet: the fit is kept
                self.assertAlmostEqual(s.PREFILL_COST['b'], B, delta=B * 0.02)
                s.stage_open('d', 100000, 'conv3'); s.stage_prefill('d', 0, total=100000, cached=0)
                read('d', 0, 100000, 25000)                                       # saved
                s.PREFILL_COST.update(a=None, b=None)
                s.prefill_rate_load(f)
                self.assertAlmostEqual(s.PREFILL_COST['a'], A, delta=A * 0.02)
        finally:
            s.time.time = saved_time; s.PREFILL_COST.update(saved_cost); s.STAGE_SEEN.clear(); s.PREFILL_RATE_FILE = None

    async def test_partial_points_show_progress_but_not_the_fit(self):
        """In-chunk points (daemon HIVE_STAGE_PROGRESS_MS, "partial") move the shown progress; the fit uses whole chunks only (a layer
        fraction is not a position — fitting it swung the time left to 997 s on GLM, 2026-10-15). Before any fit they give the level."""
        s.STAGE_STATUS = True
        saved_time, saved_cost = s.time.time, dict(s.PREFILL_COST)
        clock = [1000.0]
        s.time.time = lambda: clock[0]
        try:
            s.PREFILL_COST.update(a=None, b=None)
            s.stage_open('p', 200000, 'convp'); s.stage_prefill('p', 0, total=200000, cached=0)
            clock[0] += 2.0; s.stage_prefill('p', 10000, partial=True)            # no fit yet: level = 5,000 tok/s
            st = s.STAGES['p']
            self.assertEqual((st['prefill_done'], st['prefill_rate'], st['prefill_eta_s']), (10000, 5000, 38.0))
            clock[0] += 0.5; s.stage_prefill('p', 30000, partial=True)            # a fast-looking layer stretch: no slope from it
            self.assertEqual(st['_chunks'], [])
            self.assertEqual(st['prefill_done'], 30000)
            s.stage_prefill('p', 20000, partial=True)                             # never goes back
            self.assertEqual(st['prefill_done'], 30000)
            clock[0] += 9.5; s.stage_prefill('p', 50000)                          # chunk end: 50,000 in 12 s
            self.assertEqual(len(st['_chunks']), 1)
            self.assertAlmostEqual(st['prefill_eta_s'], 150000 * 12 / 50000, delta=0.2)
            eta = st['prefill_eta_s']
            clock[0] += 6.0; s.stage_prefill('p', 75000, partial=True)            # inside the next chunk: same fit, less left
            self.assertAlmostEqual(st['prefill_eta_s'], eta - 25000 * 12 / 50000, delta=0.2)
            self.assertEqual(len(st['_chunks']), 1)
        finally:
            s.time.time = saved_time; s.PREFILL_COST.update(saved_cost)

    def test_prefill_fit_fallbacks(self):
        """One chunk keeps the prior slope; a fit that is not positive over the read falls back to the plain average."""
        a, b = s.prefill_fit([(1000, 500.0, 1.0)], 1e-9)
        self.assertEqual(b, 1e-9); self.assertAlmostEqual(a + b * 500, 1e-3)
        a, b = s.prefill_fit([(1000, 500.0, 1.0), (1000, 1500.0, 0.5)], None)   # faster later: no negative slope
        self.assertEqual(b, 0.0); self.assertAlmostEqual(a, 0.75e-3)
        a, b = s.prefill_fit([(1000, 500.0, 0.2), (1000, 1500.0, 0.6)], 1e-9, min_span=5000)   # too narrow a stretch: prior slope
        self.assertEqual(b, 1e-9)

    async def test_early_estimate_keeps_the_prior(self):
        """With a prior fit: it stands until STAGE_FIT_MIN tokens are read, then the slope stays the prior one until the chunks span a
        quarter of the read — small early chunks no longer swing the time left (27B, 2026-10-15: 756 s for a 114 s read)."""
        s.STAGE_STATUS = True
        saved_time, saved_cost = s.time.time, dict(s.PREFILL_COST)
        clock = [1000.0]
        s.time.time = lambda: clock[0]
        A, B = 1 / 5000, 4e-9
        true = lambda p0, p1: A * (p1 - p0) + B * (p1 * p1 - p0 * p0) / 2
        try:
            s.PREFILL_COST.update(a=A, b=B)
            s.stage_open('e', 200000, 'conve'); s.stage_prefill('e', 0, total=200000, cached=0)
            worst, prev = 0.0, 0
            for done, noise in ((2048, 3.0), (4096, .2), (6144, 1.5), (8192, .3), (16384, 1.0), (24576, 1.0), (40000, 1.0), (60000, 1.0)):
                clock[0] += true(prev, done) * noise; prev = done                 # noisy small early chunks (admission, layer mix)
                s.stage_prefill('e', done)
                want = true(done, 200000)
                worst = max(worst, abs(s.STAGES['e']['prefill_eta_s'] - want) / want)
            self.assertLess(worst, 0.25)
        finally:
            s.time.time = saved_time; s.PREFILL_COST.update(saved_cost)

    # The long reads (16,384+ tokens) a DeepSeek service did between two restarts, 2026-10-19 13:33–18:12, from its daemon log:
    #   (chunk start, chunk end, ms) absolute positions — the first start is the reused prefix. Short reads in between are left out
    #   (they never replace the prior). Reads 2, 3, 6 and 9 resume deep in a long conversation; most reads end with 1–8-row tail chunks.
    DAY_READS = (
        ((0, 98304, 17966), (98304, 195073, 20124), (195073, 197285, 1071), (197285, 199497, 1242), (199497, 199505, 181), (199505, 199508, 97)),
        ((98304, 195075, 20132), (195075, 197287, 1096), (197287, 199499, 1234), (199499, 199507, 182), (199507, 199510, 98)),
        ((98304, 195078, 20201), (195078, 197290, 1041), (197290, 199502, 1207), (199502, 199510, 175), (199510, 199513, 95)),
        ((0, 19806, 5561), (19806, 19807, 32), (19807, 19809, 67)),
        ((0, 22757, 5827), (22757, 22758, 25), (22758, 22760, 58)),
        ((145683, 179699, 8572), (179699, 179747, 361), (179747, 179750, 75)),
        ((0, 17170, 5057), (17170, 17173, 80)),
        ((0, 23034, 5702), (23034, 23037, 102)),
        ((302975, 340418, 10931), (340418, 342070, 2517), (342070, 342073, 103)),
        ((0, 20878, 5505), (20878, 20880, 80)),
        ((0, 89305, 21520), (89305, 92704, 2962), (92704, 92706, 78)),   # 18:12, a new conversation: 24.7 s in all
    )

    async def test_prefill_eta_after_a_day_of_reads(self):
        """The 2026-10-19 bug: after the reads above the kept fit was a = -0.00037, b = 1.37e-8 (a < 0 passed the old check, which only
        looked at the chunks' mid positions; tail chunks and deep resumed reads bent the slope), and the 18:12 read of 92,706 tokens
        (24.7 s) was told ~215 minutes left at 8,037 tokens (222 on screen). Now every kept prior is sane and that point gets about 22 s left."""
        s.STAGE_STATUS = True
        saved_time, saved_cost = s.time.time, dict(s.PREFILL_COST)
        clock = [1000.0]
        s.time.time = lambda: clock[0]
        try:
            s.PREFILL_COST.update(a=None, b=None, chunks=None)
            for i, read in enumerate(self.DAY_READS):
                cached, total = read[0][0], read[-1][1]
                rid, t0 = f'day{i}', clock[0]
                s.stage_open(rid, total, f'conv-day{i}'); s.stage_prefill(rid, 0, total=total - cached, cached=cached)
                if i == len(self.DAY_READS) - 1:                                  # the 18:12 read: a point inside its first chunk
                    start, end, ms = read[0]
                    dt = ms / 1000 * 8037 / (end - start)
                    clock[0] += dt; s.stage_prefill(rid, 8037, partial=True)
                    eta = s.STAGES[rid]['prefill_eta_s']
                    left = sum(ms for _, _, ms in read) / 1000 - dt                   # 22.3 s really left
                    self.assertIsNotNone(eta)
                    self.assertAlmostEqual(eta, left, delta=0.25 * left)
                    self.assertAlmostEqual(dt + eta, 24.7, delta=0.25 * 24.7)
                    clock[0] = t0
                before = dict(s.PREFILL_COST)
                for start, end, ms in read:
                    clock[0] += ms / 1000; s.stage_prefill(rid, end - cached)
                    if s.PREFILL_COST['a'] is not None:                         # every kept prior is sane on its own chunks
                        self.assertTrue(s.prefill_cost_ok(s.PREFILL_COST['a'], s.PREFILL_COST['b'], s.PREFILL_COST['chunks']))
                if cached:                                                        # deep resumed reads never replace the prior
                    self.assertEqual(s.PREFILL_COST, before)
            self.assertGreater(s.PREFILL_COST['a'], 0); self.assertGreaterEqual(s.PREFILL_COST['b'], 0)
        finally:
            s.time.time = saved_time; s.PREFILL_COST.update(saved_cost)

    def test_prefill_fit_rejects_the_bent_slopes(self):
        """The fits that went wrong on 2026-10-19: tail chunks (1–8 rows, 4–32 ms per token) and a slower tail-layer chunk after a long
        one gave a < 0; deep resumed reads gave a < 0 too. Now the slope is checked (prefill_cost_ok) and the level is the read's own."""
        def chunks(read):
            return [(e - b, b + (e - b) / 2.0, ms / 1000) for b, e, ms in read]
        for read in (self.DAY_READS[1], self.DAY_READS[3], self.DAY_READS[-1]):   # old fits: a = -0.00072, -0.033, -0.00037
            ch = chunks(read)
            a, b = s.prefill_fit(ch, None, 0.25 * (read[-1][1] - read[0][0]))
            self.assertGreater(a, 0); self.assertGreaterEqual(b, 0)
            self.assertTrue(s.prefill_cost_ok(a, b, s.fit_chunks(ch)))
            t = sum(c[2] for c in ch)
            self.assertAlmostEqual(sum(n * (a + b * m) for n, m, _ in ch), t, delta=0.1 * t)
        self.assertFalse(s.prefill_cost_ok(-0.00037066, 1.3719e-08, chunks(self.DAY_READS[-1])))
        self.assertFalse(s.prefill_cost_ok(1e-3, 0.0, chunks(self.DAY_READS[-1])))         # ~4x the measured time
        self.assertFalse(s.prefill_cost_ok(1e-4, -1e-9, chunks(self.DAY_READS[-1])))
        self.assertFalse(s.prefill_cost_ok(2.6e-4, 0.0, []))

    def test_prefill_rate_load_checks_the_file(self):
        """A kept fit is loaded only when it is sane on the chunks stored with it: the file the 2026-10-19 service kept (a < 0, no
        chunks), a file without chunks and one that does not reproduce its chunks are ignored."""
        saved_cost = dict(s.PREFILL_COST)
        good = [[20878, 10439.0, 5.505]]
        try:
            with tempfile.TemporaryDirectory() as d:
                f = os.path.join(d, 'prefill-rate.json')
                for bad in ({'a': -0.0003706596098990203, 'b': 1.371871275846038e-08, 'at': 1792401148},
                            {'a': 2.6e-4, 'b': 2.6e-10},                                          # sane numbers, no chunks to check
                            {'a': 2.6e-3, 'b': 2.6e-10, 'chunks': good},                          # 10x the measured time
                            {'a': 2.6e-4, 'b': 2.6e-10, 'chunks': [[20878, 10439.0]]}):          # malformed chunk
                    with open(f, 'w') as fh: json.dump(bad, fh)
                    s.PREFILL_COST.update(a=None, b=None, chunks=None)
                    s.prefill_rate_load(f)
                    self.assertIsNone(s.PREFILL_COST['a'], repr(bad))
                    self.assertIsNone(s.prior_cost())
                with open(f, 'w') as fh: json.dump({'a': 2.6e-4, 'b': 2.6e-10, 'chunks': good}, fh)
                s.prefill_rate_load(f)
                self.assertEqual((s.PREFILL_COST['a'], s.PREFILL_COST['b']), (2.6e-4, 2.6e-10))
                self.assertEqual(s.PREFILL_COST['chunks'], [(20878.0, 10439.0, 5.505)])
        finally:
            s.PREFILL_COST.update(saved_cost); s.PREFILL_RATE_FILE = None

    async def test_partial_eta_null_when_implausible(self):
        """Inside the first chunk the time left comes from the prior. A prior with a < 0 (what the 2026-10-19 server held in memory before
        the 18:12 read: a = -0.0395, b = 3.8e-6 → 12,891 s left at 8,037 of 92,706) is not used: the chunk's own speed is. A sane prior
        that extrapolates far past what it measured (a steep slope from a 33K read, applied to a 1M read) gives a speed more than
        ETA_SANE times below its slowest chunk: the time left is null (clients hide it) until this read's own chunks give a sane one."""
        s.STAGE_STATUS = True
        saved_time, saved_cost = s.time.time, dict(s.PREFILL_COST)
        clock = [1000.0]
        s.time.time = lambda: clock[0]
        try:
            s.PREFILL_COST.update(a=-0.03946884546421123, b=3.806161524202415e-06, chunks=None)
            s.stage_open('n1', 92706, 'convn1'); s.stage_prefill('n1', 0, total=92706, cached=0)
            st = s.STAGES['n1']
            self.assertIsNone(st['prefill_eta_s'])
            clock[0] += 1.937; s.stage_prefill('n1', 8037, partial=True)            # 89,305 rows in 21.52 s
            self.assertAlmostEqual(st['prefill_eta_s'], (92706 - 8037) * 1.937 / 8037, delta=0.5)   # ~20 s, the chunk's own pace

            steep = [(16384, 8192.0, 4.096), (16384, 24576.0, 8.192)]                 # 4,000 then 2,000 tok/s
            a, b = s.prefill_fit(steep, None)
            self.assertTrue(s.prefill_cost_ok(a, b, steep))
            s.PREFILL_COST.update(a=a, b=b, chunks=steep)
            s.stage_open('n2', 50000, 'convn2'); s.stage_prefill('n2', 0, total=50000, cached=0)
            self.assertIsNotNone(s.STAGES['n2']['prefill_eta_s'])                    # a read like the measured one: shown
            s.stage_open('n3', 1000000, 'convn3'); s.stage_prefill('n3', 0, total=1000000, cached=0)
            st = s.STAGES['n3']
            self.assertIsNone(st['prefill_eta_s']); self.assertIsNone(st['prefill_rate'])   # 7,755 s = 129 tok/s implied
            clock[0] += 2.0; s.stage_prefill('n3', 8000, partial=True)
            self.assertIsNone(st['prefill_eta_s'])
            A, B = 6e-4, 1e-10                                                      # its own chunks, ~1,600 tok/s falling slowly
            true = lambda p0, p1: A * (p1 - p0) + B * (p1 * p1 - p0 * p0) / 2
            for done in range(65536, 327681, 65536):
                clock[0] += true(done - 65536, done) - (2.0 if done == 65536 else 0); s.stage_prefill('n3', done)
            self.assertIsNotNone(st['prefill_eta_s'])                               # shown again, from its own fit
            self.assertAlmostEqual(st['prefill_eta_s'], true(327680, 1000000), delta=0.02 * true(327680, 1000000))
        finally:
            s.time.time = saved_time; s.PREFILL_COST.update(saved_cost)

    async def test_prerender_scheduled_on_final_answers_only(self):
        """HIVE_TURN_PRERENDER hook: both paths hand the generated text to schedule_prerender when the turn ends with an answer, never for
        a tool call (the turn goes on)."""
        calls = []
        saved = s.schedule_prerender
        s.schedule_prerender = lambda *a: calls.append(a[5])
        try:
            s.DSML_START = '<tool_call>'
            await self.stream([1, 2, 3, 21], False)                             # <think>hm</think>x — an answer
            self.assertEqual(len(calls), 1); self.assertIn('</think>', calls[0])
            await self.stream([1, 2, 3, 4, 5, 6, 7, 8, 9], False)               # a tool call
            self.assertEqual(len(calls), 1)
        finally:
            s.schedule_prerender = saved

    async def test_entries_expire(self):
        s.STAGE_STATUS = True
        s.stage_open('old', 1); s.stage_update('old', stage='done', ended=True)
        s.STAGES['old']['_ended'] -= s.STAGE_KEEP_S + 1
        s.stage_open('new', 1)
        self.assertNotIn('old', s.STAGES); self.assertIn('new', s.STAGES)

class ReviewTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        s.TOK=VocTokenizer(); s.ENC=ThinkEncoder(); s.INFLIGHT.clear()
        s.encode_prompt=lambda *a: ('prompt', [])
        s.tokenize_with_images=lambda *a: ([10,11], [], b'')

    async def test_d5_late_cancel_does_not_kill_next_request(self):
        # A streams, client drops; its cancel takes 50 ms. B (same conversation, its client never cancels) arrives in that window.
        s.DAEMON=FlagDaemon([1]*400, delay=.002, cancel_delay=.05)
        a=await s.chat(Request({**BODY, 'stream':True}))
        session=next(iter(s.INFLIGHT))
        await anext(a.body_iterator); await anext(a.body_iterator)
        closing=asyncio.create_task(a.body_iterator.aclose())
        await asyncio.sleep(.01)
        self.assertIn(session, s.INFLIGHT, 'reservation must be held until the cancel reached the daemon (D5)')
        b=await s.chat(Request({**BODY, 'stream':True}))
        bchunks=[]
        async def drain():
            async for x in b.body_iterator: bchunks.append(x)
        dt=asyncio.create_task(drain())
        await closing; await asyncio.wait_for(dt, 10)
        fin=[c['choices'][0]['finish_reason'] for c in sse_json(bchunks) if c.get('choices') and c['choices'][0]['finish_reason']]
        self.assertEqual(fin, ['stop'], s.DAEMON.log[:4])
        self.assertEqual(sum(1 for c in sse_json(bchunks) if c.get('choices') and c['choices'][0]['delta'].get('content')), 400)
        self.assertFalse(any('error' in c for c in sse_json(bchunks)))
        self.assertTrue(all(c['id'].startswith('chatcmpl-') for c in sse_json(bchunks) if 'id' in c), 'response id must stay the OpenAI chatcmpl id, not the daemon rid')
        self.assertEqual(s.DAEMON.cancelled, [session])
        # request ids: A's cancel carries the rid A sent with generate, which differs from B's rid (hived rid_flags flags only that request)
        self.assertEqual(len(s.DAEMON.rids), 2); self.assertTrue(all(s.DAEMON.rids)); self.assertNotEqual(s.DAEMON.rids[0], s.DAEMON.rids[1])
        self.assertEqual(s.DAEMON.cancel_rids, [s.DAEMON.rids[0]])
        self.assertFalse(s.INFLIGHT)

    async def test_d11_anthropic_stop_sequence(self):
        body={**BODY, 'stop_sequences':['zz', 'bc'], 'thinking':{'type':'disabled'}, 'max_tokens':10}
        s.DAEMON=FlagDaemon([1,2,3,99])
        out=await s.anthropic_messages(Request(body))
        self.assertEqual((out['stop_reason'], out['stop_sequence']), ('stop_sequence', 'bc'))
        self.assertEqual(out['content'], [{'type':'text', 'text':'a'}])
        s.DAEMON=FlagDaemon([1,2,3,99])
        txt=''.join([x async for x in (await s.anthropic_messages(Request({**body, 'stream':True}))).body_iterator])
        self.assertIn('"stop_reason": "stop_sequence", "stop_sequence": "bc"', txt)
        # no hit: end_turn with stop_sequence null, field present in both modes
        s.DAEMON=FlagDaemon([1,2,3,99])
        out=await s.anthropic_messages(Request({**body, 'stop_sequences':['q']}))
        self.assertEqual(out['stop_reason'], 'end_turn'); self.assertIn('stop_sequence', out); self.assertIsNone(out['stop_sequence'])
        s.DAEMON=FlagDaemon([1,2,3,99])
        txt=''.join([x async for x in (await s.anthropic_messages(Request({**body, 'stop_sequences':['q'], 'stream':True}))).body_iterator])
        self.assertIn('"stop_reason": "end_turn", "stop_sequence": null', txt)
        # OpenAI: finish stop plus the matched string (vLLM stop_reason convention) only on a hit
        s.DAEMON=FlagDaemon([1,2,3,99])
        r=await s.chat(Request({**BODY, 'stop':['bc']}))
        self.assertEqual((r['choices'][0]['finish_reason'], r['choices'][0].get('stop_reason')), ('stop', 'bc'))
        s.DAEMON=FlagDaemon([1,2,3,99])
        r=await s.chat(Request(BODY))
        self.assertNotIn('stop_reason', r['choices'][0])

    async def test_d12_stop_inside_thinking_is_ignored(self):
        body={'messages':BODY['messages'], 'stop':['.']}  # thinking mode (default); '.' only inside reasoning
        s.DAEMON=FlagDaemon([1,4,2,5,6,7,99])
        m=(await s.chat(Request(body)))['choices'][0]
        self.assertEqual((m['finish_reason'], m['message']['content'], m['message']['reasoning_content']), ('stop', 'XY', 'a.b'))
        s.DAEMON=FlagDaemon([1,4,2,5,6,4,7,99])  # a '.' in the visible content still stops
        m=(await s.chat(Request(body)))['choices'][0]
        self.assertEqual((m['message']['content'], m['message']['reasoning_content'], m.get('stop_reason')), ('X', 'a.b', '.'))
        s.DAEMON=FlagDaemon([1,4,2,5,6,7,99])
        ch=sse_json([x async for x in (await s.chat(Request({**body, 'stream':True}))).body_iterator])
        self.assertEqual(''.join(c['choices'][0]['delta'].get('content') or '' for c in ch if c.get('choices')), 'XY')
        self.assertEqual(''.join(c['choices'][0]['delta'].get('reasoning_content') or '' for c in ch if c.get('choices')), 'a.b')

    async def test_d12_stop_inside_tool_block_keeps_the_call(self):
        body={**BODY, 'stop':['name=']}
        for stream in (False, True):
            s.DAEMON=FlagDaemon([1,8,9,6,99])
            r=await s.chat(Request({**body, 'stream':stream}))
            if not stream:
                m=r['choices'][0]
                self.assertEqual((m['finish_reason'], m['message']['content'], len(m['message']['tool_calls'])), ('tool_calls', 'a', 1))
                continue
            ch=sse_json([x async for x in r.body_iterator])
            content=''.join(c['choices'][0]['delta'].get('content') or '' for c in ch if c.get('choices'))
            self.assertEqual(content, 'a', 'DSML must not leak into content')
            self.assertTrue(any(c['choices'][0]['delta'].get('tool_calls') for c in ch if c.get('choices')))
            self.assertEqual([c['choices'][0]['finish_reason'] for c in ch if c.get('choices') and c['choices'][0]['finish_reason']], ['tool_calls'])
            self.assertEqual(s.DAEMON.cancelled, [])

    async def test_d13_cancel_is_not_length_and_error_shape(self):
        class CancelDaemon(Daemon):
            async def generate(self, session, *a, **kw):
                yield {'id':1}; yield {'done':True, 'n':1, 'finish':'cancel'}
        s.DAEMON=CancelDaemon()
        r=await s.chat(Request(BODY))
        self.assertEqual(r.status_code, 502)
        err=json.loads(r.body)
        self.assertEqual(err['error']['type'], 'server_error'); self.assertEqual(err['partial_content'], 'a')
        s.DAEMON=CancelDaemon()
        ch=[x async for x in (await s.chat(Request({**BODY, 'stream':True}))).body_iterator]
        errs=[c['error'] for c in sse_json(ch) if 'error' in c]
        self.assertEqual(len(errs), 1); self.assertEqual(errs[0]['type'], 'server_error'); self.assertIsInstance(errs[0]['message'], str)
        self.assertFalse(any(c.get('choices') and c['choices'][0]['finish_reason'] for c in sse_json(ch)), 'cancel must not masquerade as length')
        s.DAEMON=CancelDaemon()
        txt=''.join([x async for x in (await s.anthropic_messages(Request({**BODY, 'stream':True}))).body_iterator])
        self.assertIn('event: error', txt); self.assertIn('generation cancelled', txt); self.assertNotIn('max_tokens', txt)
        for err in ({'message':'dict shape', 'type':'server_error'}, 'string shape'):
            async def chunks():
                yield 'data: '+json.dumps({'error':err})+'\n\n'
            resp=s.ClosingStreamingResponse(chunks())
            out=''.join([x async for x in s.anthropic_stream(resp)])
            self.assertIn('"message": "%s"' % (err['message'] if isinstance(err, dict) else err), out)
        self.assertFalse(s.INFLIGHT)

    async def test_stream_disconnect_sends_cancel_during_silence(self):
        class Gone(Request):
            def __init__(self, body, after): super().__init__(body); self.t=time.monotonic()+after
            async def is_disconnected(self): return time.monotonic() >= self.t
        for handler in (s.chat, s.anthropic_messages):
            s.DAEMON=FlagDaemon([1,2,3], silent=5.0)  # long prefill: nothing is sent, so ASGI send never fails
            t0=time.monotonic()
            r=await handler(Gone({**BODY, 'stream':True}, .15))
            chunks=[x async for x in r.body_iterator]
            dt=time.monotonic()-t0
            self.assertLess(dt, 2.0, 'cancel was not sent during the silent phase')
            self.assertEqual(len(s.DAEMON.cancelled), 1, 'exactly one cancel (monitor + teardown must not double-cancel)')
            self.assertFalse(any('"finish_reason": "length"' in c or 'max_tokens"' in c for c in chunks))
            self.assertFalse(s.INFLIGHT)

    async def test_stream_ends_when_disconnect_probe_swallows_cancel(self):
        # Reproduces a stall seen on the GPU host: if the anyio CancelScope in Starlette's is_disconnected() swallows monitor.cancel() (emulated here
        #   — eats CancelledError and returns False), a plain `await monitor` waits forever and the final stream chunk is never sent.
        class Swallow(Request):
            async def is_disconnected(self):
                try:
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    pass
                return False
        for handler in (s.chat, s.anthropic_messages):
            s.DAEMON=FlagDaemon([1,2,3], delay=.001)
            r=await handler(Swallow({**BODY, 'stream':True}))
            chunks=await asyncio.wait_for(self._drain(r), 5.0)
            self.assertTrue(any('[DONE]' in c or 'message_stop' in c for c in chunks), chunks[-2:])
            self.assertFalse(s.INFLIGHT)

    async def _drain(self, r):
        return [x async for x in r.body_iterator]

    async def test_b8_parallel_uses_stable_overflow_slots(self):
        s.DAEMON=FlagDaemon([1]*50, delay=.002)
        rs=[await s.chat(Request({**BODY, 'stream':True})) for _ in range(3)]
        base=min(s.INFLIGHT, key=len)
        self.assertEqual(sorted(s.INFLIGHT), sorted([base, base+'~1', base+'~2']))
        for r in rs: [x async for x in r.body_iterator]
        self.assertFalse(s.INFLIGHT)
        a=await s.chat(Request({**BODY, 'stream':True})); b=await s.chat(Request({**BODY, 'stream':True}))
        self.assertIn(base+'~1', s.INFLIGHT, 'the freed slot is reused (prefix reuse, bounded session count)')
        for r in (a, b): [x async for x in r.body_iterator]

    def test_visible_stop_fragments(self):
        # stop matched only in content; think / tool text passes through verbatim, across fragment boundaries
        for frags, thinking, expected, matched in [
                (['a.b</th', 'ink>X.Y'], True, 'a.b</think>X', '.'),
                (['a.b</think>XY'], True, 'a.b</think>XY', None),
                (['X\n\n<｜DS', 'ML｜invoke name="f">.'], False, 'X\n\n<｜DSML｜invoke name="f">.', None),
                (['X', '.', 'Y'], False, 'X', '.')]:
            v=s.VisibleStop(['.'], thinking); out=''
            for f in frags:
                out+=v.push(f)
                if v.stopped: break
            if not v.stopped: out+=v.finish()
            self.assertEqual((out, v.matched), (expected, matched))
        v=s.VisibleStop(None, True); self.assertEqual(v.push('a.b'), 'a.b')  # no stops: untouched pass-through

# DSML tool-call streaming (HIVE_TOOL_STREAM): the streamed pieces concatenate to exactly the reference arguments string.
D='｜DSML｜'
def dsml_block(calls):
    """The reference grammar (encoding.encode_arguments_to_dsml + tool_call_template + the calls block)."""
    inv=[]
    for name, args in calls:
        ps=[f'<{D} parameter name="{k}" string="{"true" if isinstance(v, str) else "false"}">{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</{D} parameter>' for k, v in args.items()]
        inv.append(f'<{D} invoke name="{name}">\n' + '\n'.join(ps) + f'\n</{D} invoke>')
    return f'<{D} calls>\n' + '\n'.join(inv) + f'\n</{D} calls>'
def ref_args(args):
    """encoding.decode_dsml_to_arguments: "{" + ", ".join(json(k) + ": " + (json(v) if string else v)) + "}"."""
    return '{' + ', '.join(json.dumps(k, ensure_ascii=False) + ': ' + (json.dumps(v, ensure_ascii=False) if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for k, v in args.items()) + '}'

class ToolStreamTests(unittest.TestCase):
    CASES=[
        [('write_file', {'path': 'a/b.py', 'content': 'line "one"\n\ttab \\ back <tag> </｜DSML｜ par ω ✓ \u0001'})],
        [('calc', {'x': 3, 'opts': {'deep': [1, 2, {'k': None}]}, 'flag': True})],
        [('noargs', {})],
        [('first', {'q': 'a'}), ('second', {'n': 1.5, 's': ''})],
        [('ns::tool', {'v': '<<<>>>'})],
    ]
    def run_stream(self, text, sizes):
        t=s.DsmlToolStream(type('Enc', (), {'_split_tool_name': staticmethod(lambda n: (n.split('::')[0], n.split('::')[1]) if '::' in n else (None, n))})())
        pieces, i, k=[], 0, 0
        while i < len(text):
            n=sizes[k % len(sizes)]; pieces.append(text[i:i+n]); i+=n; k+=1
        calls={}
        for p in pieces:
            for d in t.feed(p):
                c=calls.setdefault(d['index'], {'id': None, 'name': None, 'args': ''})
                if 'id' in d:
                    self.assertIsNone(c['id'], 'id only on the first piece'); c['id']=d['id']; c['name']=d['function']['name']
                    self.assertEqual(d['type'], 'function')
                c['args']+=d['function']['arguments']
        return t, calls
    def test_pieces_equal_reference(self):
        for case in self.CASES:
            text=dsml_block(case)
            for sizes in ([1], [2, 7], [3, 1, 11], [len(text)]):
                t, calls=self.run_stream(text, sizes)
                self.assertFalse(t.broken, (case, sizes))
                self.assertTrue(t.complete())
                want=[(n.split('::')[-1], ref_args(a)) for n, a in case]
                self.assertEqual([(calls[i]['name'], calls[i]['args']) for i in sorted(calls)], want, (case, sizes))
                self.assertEqual(t.result(), want)
                for i in sorted(calls): json.loads(calls[i]['args'])  # valid JSON
    def test_arguments_stream_before_the_call_ends(self):
        text=dsml_block([('write_file', {'content': 'x'*500})])
        t=s.DsmlToolStream(object())
        got=''.join(d['function']['arguments'] for d in t.feed(text[:text.index('x'*500)+300]))
        self.assertTrue(got.startswith('{"content": "') and got.count('x') >= 280, got[:40])  # most of the value is out before the end tag
        self.assertFalse(t.complete())
    def test_malformed_stops_streaming(self):
        for bad in (f'<{D} calls>X\n<{D} invoke name="f">\n', f'<{D} calls>\n<{D} invoke nam="f">\n<{D} parameter', f'<{D} callz>'):
            t=s.DsmlToolStream(object())
            t.feed(bad + 'tail' * 3)
            self.assertTrue(t.broken, bad)
    def test_truncated_call_is_incomplete(self):
        text=dsml_block([('f', {'a': 'long value'})])
        t=s.DsmlToolStream(object())
        t.feed(text[:text.index('value')])
        self.assertTrue(t.calls and not t.complete())

class CharTok(Tokenizer):
    """One token per character of a fixed text (ids 100.., clear of <EOS>=99), plus <EOS>."""
    def __init__(self, text): self.chars=list(text)
    def decode(self, ids, **kw): return ''.join('<EOS>' if i == 99 else self.chars[i-100] for i in ids)

class RefLikeEncoder:
    """Stand-in for the reference parse of one DSML block (name and arguments built by the reference rule)."""
    def __init__(self, calls): self.calls=calls
    def parse_message_from_completion_text(self, text, **kw):
        head=text.replace('<EOS>', '').split('\n\n<｜DSML｜', 1)[0]
        return {'content': head, 'tool_calls': [{'type': 'function', 'function': {'name': n, 'arguments': ref_args(a)}} for n, a in self.calls]}

class FlushEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_flush_cache_status(self):
        import types
        class CtlDaemon:
            def __init__(self, *replies): self.replies=list(replies); self.ops=[]
            async def control(self, op, **kw):
                self.ops.append((op, kw)); return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        req=lambda **q: types.SimpleNamespace(query_params=q)
        busy={'error': 'not idle', 'running': 1, 'retryable': True}
        saved=s.DAEMON
        try:
            s.DAEMON=CtlDaemon({'ok': True, 'sessions': 3, 'archived': 1, 'snapshots': 2})
            r=await s.flush_cache(req())
            self.assertEqual(r['ok'], True); self.assertEqual(s.DAEMON.ops, [('flush', {})])
            s.DAEMON=CtlDaemon(busy)
            r=await s.flush_cache(req())
            self.assertEqual((r.status_code, len(s.DAEMON.ops)), (400, 1), 'busy without timeout -> 400 at once, like SGLang')
            s.DAEMON=CtlDaemon(busy, busy, {'ok': True})
            t0=time.monotonic(); r=await s.flush_cache(req(timeout='2'))
            self.assertEqual((r['ok'], len(s.DAEMON.ops)), (True, 3), 'busy twice then ok within the timeout')
            self.assertLess(time.monotonic()-t0, 1.5)
            s.DAEMON=CtlDaemon(busy)
            r=await s.flush_cache(req(timeout='0.5'))
            self.assertEqual(r.status_code, 400, 'still busy at the deadline -> 400')
            self.assertGreaterEqual(len(s.DAEMON.ops), 2)
            s.DAEMON=CtlDaemon({'ok': True})
            self.assertEqual((await s.flush_cache(req(timeout='abc')))['ok'], True, 'bad timeout absorbed')
        finally:
            s.DAEMON=saved

class ToolStreamSseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.saved=(s.TOK, s.ENC, s.DAEMON, s.log_request)
        self.recs=[]; s.log_request=lambda rec: self.recs.append(dict(rec)); s.INFLIGHT.clear()
    async def asyncTearDown(self):
        s.TOK, s.ENC, s.DAEMON, s.log_request=self.saved
        os.environ.pop('HIVE_TOOL_STREAM', None)
    def setup_call(self, calls, prefix='ok'):
        text=prefix + '\n\n' + dsml_block(calls)
        s.TOK=CharTok(text); s.ENC=RefLikeEncoder(calls)
        s.DAEMON=FlagDaemon(list(range(100, len(text) + 100)) + [99])
    async def test_streamed_pieces_and_finish(self):
        calls=[('write_file', {'path': 'x.txt', 'content': 'body "q"\n' * 20}), ('noargs', {})]
        self.setup_call(calls)
        r=await s.chat(Request({**BODY, 'stream': True}))
        ch=sse_json([x async for x in r.body_iterator])
        deltas=[d for c in ch if c.get('choices') for d in (c['choices'][0]['delta'].get('tool_calls') or [])]
        self.assertGreater(len(deltas), 3, 'arguments arrive in several pieces')
        got={}
        for d in deltas:
            g=got.setdefault(d['index'], {'name': None, 'args': ''})
            if 'id' in d: g['name']=d['function']['name']
            g['args']+=d['function'].get('arguments') or ''
        self.assertEqual([(got[i]['name'], got[i]['args']) for i in sorted(got)], [(n, ref_args(a)) for n, a in calls])
        content=''.join(c['choices'][0]['delta'].get('content') or '' for c in ch if c.get('choices'))
        self.assertEqual(content, 'ok', 'DSML must not leak into content')
        self.assertEqual([c['choices'][0]['finish_reason'] for c in ch if c.get('choices') and c['choices'][0]['finish_reason']], ['tool_calls'])
        self.assertEqual(self.recs[-1].get('tool_stream'), 'match')
        self.assertEqual(len(self.recs), 1, 'one request-log line per stream')
        self.assertNotIn('tool_args_invalid', self.recs[-1], 'valid arguments: no field')
        self.assertNotIn('_defer_log', self.recs[-1])
    def setup_mislabeled(self):
        """A string value the model marked string="false": the reference rule copies it as written -> {"job_id": 3e382151}."""
        text='ok\n\n' + f'<{D} calls>\n<{D} invoke name="job_wait">\n<{D} parameter name="job_id" string="false">3e382151</{D} parameter>\n</{D} invoke>\n</{D} calls>'
        s.TOK=CharTok(text)
        s.ENC=type('Enc', (), {'parse_message_from_completion_text': staticmethod(lambda t, **kw: {'content': 'ok', 'tool_calls': [
            {'type': 'function', 'function': {'name': 'job_wait', 'arguments': '{"job_id": 3e382151}'}}]})})()
        s.DAEMON=FlagDaemon(list(range(100, len(text) + 100)) + [99])
    async def test_invalid_arguments_recorded_stream(self):
        self.setup_mislabeled()
        r=await s.chat(Request({**BODY, 'stream': True}))
        ch=sse_json([x async for x in r.body_iterator])
        args=''.join(d['function'].get('arguments') or '' for c in ch if c.get('choices') for d in (c['choices'][0]['delta'].get('tool_calls') or []))
        self.assertEqual(args, '{"job_id": 3e382151}', 'recorded only: the call is sent unchanged')
        self.assertEqual((len(self.recs), self.recs[-1].get('tool_args_invalid')), (1, ['job_wait']))
    async def test_invalid_arguments_recorded_once_without_stream(self):
        self.setup_mislabeled()
        r=await s.chat(Request({**BODY, 'stream': False}))
        self.assertEqual(r['choices'][0]['message']['tool_calls'][0]['function']['arguments'], '{"job_id": 3e382151}')
        self.assertEqual((len(self.recs), self.recs[-1].get('tool_args_invalid')), (1, ['job_wait']), 'one log line, after the parse')
        self.assertNotIn('_defer_log', self.recs[-1])
    def test_invalid_tool_args_rule(self):
        f=s.invalid_tool_args
        self.assertEqual(f([('a', '{"x": 1}'), ('b', '{}'), ('c', '{"s": "3e382151"}')]), [])
        self.assertEqual(f([('a', '{"x": 3e382151}'), ('b', '{"x": NaN}'), ('c', '{"x": [Infinity]}'), ('d', '{x}'), ('e', '[1]'), ('f', None)]),
                         ['a', 'b', 'c', 'd', 'e', 'f'])
    async def test_disconnect_still_logs_once(self):
        self.setup_call([('f', {'a': 'z' * 200})])
        r=await s.chat(Request({**BODY, 'stream': True}))
        it=r.body_iterator
        for _ in range(5): await it.__anext__()
        await it.aclose()  # client went away mid-stream
        await asyncio.sleep(0.05)
        self.assertEqual(len(self.recs), 1)
        self.assertNotIn('_defer_log', self.recs[-1])
    async def test_off_switch_keeps_the_old_single_chunk(self):
        os.environ['HIVE_TOOL_STREAM']='0'
        calls=[('f', {'a': 'v' * 50})]
        self.setup_call(calls)
        r=await s.chat(Request({**BODY, 'stream': True}))
        ch=sse_json([x async for x in r.body_iterator])
        deltas=[d for c in ch if c.get('choices') for d in (c['choices'][0]['delta'].get('tool_calls') or [])]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]['function']['arguments'], ref_args(calls[0][1]))
        self.assertNotIn('tool_stream', self.recs[-1])
    async def test_anthropic_tool_use_from_pieces(self):
        calls=[('f', {'a': 'piece ' * 30})]
        self.setup_call(calls)
        r=await s.chat(Request({**BODY, 'stream': True}))
        out=''.join([x async for x in s.anthropic_stream(r)])
        ev=[json.loads(l[6:]) for l in out.splitlines() if l.startswith('data: ')]
        starts=[e for e in ev if e['type'] == 'content_block_start' and e['content_block']['type'] == 'tool_use']
        self.assertEqual(len(starts), 1)
        idx=starts[0]['index']
        joined=''.join(e['delta']['partial_json'] for e in ev if e['type'] == 'content_block_delta' and e['index'] == idx)
        self.assertEqual(json.loads(joined), json.loads(ref_args(calls[0][1])))
        self.assertEqual(sum(1 for e in ev if e['type'] == 'content_block_stop' and e['index'] == idx), 1)
        self.assertIn('"stop_reason": "tool_use"', out)

# Boundary hints: a concatenative fake template + marker-aware tokenizer with offset mapping.
MARKERS={'<|bos|>':1, '<|system|>':2, '<|user|>':3, '<|assistant|>':4, '<|tools|>':5, '<|eos|>':6}

class HintTokenizer:
    eos_token_id=6; eos_token='<|eos|>'
    def _split(self, text):
        out, i = [(1, 0, 0)], 0  # BOS: zero-width at 0
        while i < len(text):
            m=next((m for m in MARKERS if text.startswith(m, i)), None)
            if m: out.append((MARKERS[m], i, i+len(m))); i+=len(m)
            else: out.append((100+ord(text[i]) % 3000, i, i+1)); i+=1
        return out
    def encode(self, text): return [t for t, _, _ in self._split(text)]
    def __call__(self, text, return_offsets_mapping=False):
        sp=self._split(text)
        return {'input_ids':[t for t, _, _ in sp], 'offset_mapping':[(a, b) for _, a, b in sp]}
    def decode(self, ids, **kw): return ''.join(chr(0x4E00+i) for i in ids)

def hint_encode(messages, tools, thinking_mode, effort):
    messages=[dict(m) for m in messages]
    if tools and messages[0].get('role')!='system': messages.insert(0, {'role':'system', 'content':''})
    text=''
    for i, m in enumerate(messages):
        text+='<|'+m['role']+'|>'+(m.get('content') or '')
        if i==0 and tools: text+='<|tools|>'+json.dumps(tools)
        if m['role']=='assistant': text+='<|eos|>'
    return text+'<|assistant|>', []

class RecDaemon(Daemon):
    def __init__(self): super().__init__(); self.kw=[]
    async def generate(self, session, *args, **kw):
        self.kw.append(kw)
        async for m in super().generate(session, *args, **kw): yield m

class HintTests(unittest.IsolatedAsyncioTestCase):
    SYS='system prompt '*10
    def setUp(self):
        s.TOK=HintTokenizer(); s.ENC=Encoder(); s.INFLIGHT.clear(); s.DAEMON=RecDaemon()
        s.encode_prompt=hint_encode
        s.tokenize_with_images=lambda prompt, media: (s.TOK.encode(prompt), [], b'')
        self.env=os.environ.pop('HIVE_PREFIX_SHARE', None)
    def tearDown(self):
        os.environ.pop('HIVE_PREFIX_SHARE', None)
        if self.env is not None: os.environ['HIVE_PREFIX_SHARE']=self.env

    def conv(self):
        return [{'role':'system','content':self.SYS}, {'role':'user','content':'first question'},
                {'role':'assistant','content':'first answer'}, {'role':'user','content':'second question'}]

    def hints_for(self, msgs, tools=None):
        prompt,_=hint_encode(msgs, tools, 'chat', None)
        ids=s.TOK.encode(prompt)
        return prompt, ids, s.boundary_hints(msgs, tools, 'chat', None, prompt, ids)

    def test_exact_offsets(self):
        prompt, ids, hints=self.hints_for(self.conv())
        # system block (+ the next role marker, shared by every conversation whose next message is a user turn)
        sys_end=len(s.TOK.encode('<|system|>'+self.SYS+'<|user|>'))
        turn_end=len(s.TOK.encode('<|system|>'+self.SYS+'<|user|>first question<|assistant|>first answer<|eos|><|user|>'))
        self.assertEqual(hints, [sys_end, turn_end])
        for h in hints:  # each hint is an exact token prefix of the encoded prompt, ending on a role marker token
            self.assertEqual(ids[h-1], MARKERS['<|user|>'])
            self.assertEqual(ids[:h], s.TOK.encode(prompt[:s.TOK(prompt, True)['offset_mapping'][h-1][1]]))
        # another conversation with the same system prompt shares the first hint's token prefix exactly
        other=[{'role':'system','content':self.SYS}, {'role':'user','content':'something else'}]
        p2, ids2, h2=self.hints_for(other)
        self.assertEqual(h2, [sys_end]); self.assertEqual(ids2[:sys_end], ids[:sys_end])

    def test_leading_user_block_boundary(self):
        """2026-10-14: a client sent its fixed per-account instructions as the first user message, then the user's turn — new
        conversations shared that whole block, but the only boundary was the system block's end."""
        msgs=[{'role':'system','content':self.SYS}, {'role':'user','content':'fixed block'}, {'role':'user','content':'question one'}]
        prompt, ids, hints=self.hints_for(msgs)
        sys_end=len(s.TOK.encode('<|system|>'+self.SYS+'<|user|>'))
        block_end=len(s.TOK.encode('<|system|>'+self.SYS+'<|user|>fixed block<|user|>'))
        self.assertEqual(hints, [sys_end, block_end])
        other=msgs[:2]+[{'role':'user','content':'a different question'}]
        _, ids2, h2=self.hints_for(other)
        self.assertIn(block_end, h2); self.assertEqual(ids2[:block_end], ids[:block_end])   # another conversation resumes there
        # an ordinary conversation (first question, then its answer) gets no extra boundary
        _, _, h3=self.hints_for(self.conv())
        self.assertNotIn(len(s.TOK.encode('<|system|>'+self.SYS+'<|user|>first question')), h3)

    def test_leading_user_block_then_room_notice(self):
        """2026-10-19: a chat client sent [system + tools] [user: per-person block] [system: per-room notice] [user: ...] — the person's
        block had no boundary because a system message, not a user message, followed it."""
        msgs=[{'role':'system','content':self.SYS}, {'role':'user','content':'person block'}, {'role':'system','content':'room notice'},
              {'role':'user','content':'question one'}]
        prompt, ids, hints=self.hints_for(msgs)
        sys_end=len(s.TOK.encode('<|system|>'+self.SYS+'<|user|>'))
        block_end=len(s.TOK.encode('<|system|>'+self.SYS+'<|user|>person block'))  # before the next role marker: rooms with and without a notice share it
        notice_end=len(s.TOK.encode('<|system|>'+self.SYS+'<|user|>person block<|system|>room notice<|user|>'))
        self.assertEqual(hints, [sys_end, block_end, notice_end])
        # another room of the same person without the notice shares the block; one with the same notice shares the notice too
        _, ids2, h2=self.hints_for(msgs[:2]+[{'role':'user','content':'other room'}])
        self.assertEqual(ids2[:block_end], ids[:block_end])  # a snapshot at block_end serves it (its own boundary includes the next marker)
        _, ids3, h3=self.hints_for(msgs[:3]+[{'role':'user','content':'third room'}])
        self.assertIn(notice_end, h3); self.assertEqual(ids3[:notice_end], ids[:notice_end])
        # several notices in a row: one boundary at the end of the last; a block followed by its answer stays an ordinary first turn
        _, _, h4=self.hints_for(msgs[:3]+[{'role':'developer','content':'more'}, {'role':'user','content':'q'}])
        self.assertIn(block_end, h4); self.assertNotIn(notice_end, h4)
        _, _, h5=self.hints_for(msgs[:3]+[{'role':'assistant','content':'a'}])
        self.assertEqual(h5, [sys_end])

    def test_several_leading_user_blocks(self):
        """2026-10-27: an analysis client sent [system] [user: fixed note] [user: owner's index] [user: item] — only the note's end was a
        boundary, so every item of the owner re-read the index (prefill median 3.9 → 6.8 s)."""
        msgs=[{'role':'system','content':self.SYS}, {'role':'user','content':'note'}, {'role':'user','content':'owner index'},
              {'role':'user','content':'item one'}]
        prompt, ids, hints=self.hints_for(msgs)
        sys_end=len(s.TOK.encode('<|system|>'+self.SYS+'<|user|>'))
        note_end=len(s.TOK.encode('<|system|>'+self.SYS+'<|user|>note<|user|>'))
        index_end=len(s.TOK.encode('<|system|>'+self.SYS+'<|user|>note<|user|>owner index<|user|>'))
        self.assertEqual(hints, [sys_end, note_end, index_end])
        # the next item of the same owner resumes at the index's end; another owner (another index) at the note's end
        _, ids2, h2=self.hints_for(msgs[:3]+[{'role':'user','content':'item two'}])
        self.assertIn(index_end, h2); self.assertEqual(ids2[:index_end], ids[:index_end])
        _, ids3, h3=self.hints_for(msgs[:2]+[{'role':'user','content':'other index'}, {'role':'user','content':'item'}])
        self.assertIn(note_end, h3); self.assertEqual(ids3[:note_end], ids[:note_end])
        # a run ending in its answer: blocks before the last user message keep their boundaries, the question gets none
        _, _, h4=self.hints_for(msgs+[{'role':'assistant','content':'a'}])
        self.assertEqual(h4, [sys_end, note_end, index_end])
        # at most HINT_LEAD_BLOCKS blocks
        many=[{'role':'system','content':self.SYS}]+[{'role':'user','content':'b%d'%i} for i in range(s.HINT_LEAD_BLOCKS+3)]
        _, _, h5=self.hints_for(many)
        self.assertEqual(len(h5), 1+s.HINT_LEAD_BLOCKS)

    def test_boundary_stops_where_the_next_content_starts(self):
        """2026-10-08: a next message starting with "<" (a per-turn "<info-msg>" block) shared "<" with the empty probe's
        "<|assistant|>" and the boundary took that token — the next turn (user's words there) could not resume it."""
        msgs=self.conv()[:3]+[{'role':'user','content':'<info>t</info>'}, {'role':'user','content':'second question'}]
        prompt, ids, hints=self.hints_for(msgs)
        turn_end=len(s.TOK.encode('<|system|>'+self.SYS+'<|user|>first question<|assistant|>first answer<|eos|><|user|>'))
        self.assertIn(turn_end, hints)
        self.assertEqual(ids[turn_end-1], MARKERS['<|user|>'])
        nxt=self.conv()[:3]+[{'role':'user','content':'second question'}, {'role':'assistant','content':'a2'},
                             {'role':'user','content':'<info>u</info>'}, {'role':'user','content':'third'}]
        _, ids2, _=self.hints_for(nxt)
        self.assertEqual(ids2[:turn_end], ids[:turn_end])   # the saved boundary is a prefix of the next turn

    def test_generation_suffix_hint_is_where_the_next_turn_continues(self):
        """2026-10-08: with reasoning on, the prompt ends in a reasoning-start token while the next turn renders this answer without
        it ("</think>answer") — the prompt-end checkpoint never matched. The hint sits where the two renderings part."""
        def think_encode(messages, tools, thinking_mode, effort):
            text=''
            for m in messages:
                text+='<|'+m['role']+'|>'+('</think>' if m['role']=='assistant' else '')+(m.get('content') or '')
                if m['role']=='assistant': text+='<|eos|>'
            return text+'<|assistant|><think>', []
        added={'<think>':7, '</think>':8}
        MARKERS.update(added); s.encode_prompt=think_encode
        try:
            msgs=self.conv()
            prompt,_=think_encode(msgs, None, 'chat', None); ids=s.TOK.encode(prompt)
            self.assertNotIn(len(ids)-1, s.boundary_hints(msgs, None, 'chat', None, prompt, ids))            # off unless asked
            hints=s.boundary_hints(msgs, None, 'chat', None, prompt, ids, gen_cut=True)
            self.assertIn(len(ids)-1, hints)                                                                # before <think>
            nxt=msgs+[{'role':'assistant','content':'second answer'}, {'role':'user','content':'third'}]
            p2,_=think_encode(nxt, None, 'chat', None)
            self.assertEqual(s.TOK.encode(p2)[:len(ids)-1], ids[:len(ids)-1])                                # a prefix of the next turn
        finally:
            for k in added: MARKERS.pop(k)
            s.encode_prompt=hint_encode

    def test_generation_suffix_hint_text_prefix(self):
        """2026-10-14: a Qwen-style prompt ends "<think>\n" and the next turn has "<think>\n\n</think>\n\n" there — the whole prompt is
        a text prefix of the next turn, but its last token is not (it merges into "\n\n"). The hint is one token before the end."""
        def qwen_encode(messages, tools, thinking_mode, effort):
            text=''
            for m in messages:
                text+='<|'+m['role']+'|>'+('<think>\n\n</think>\n\n' if m['role']=='assistant' else '')+(m.get('content') or '')
                if m['role']=='assistant': text+='<|eos|>'
            return text+'<|assistant|><think>\n', []
        added={'<think>':7, '</think>':8, '\n\n':9}   # "\n\n" is one token, as in the Qwen vocabulary
        MARKERS.update(added); s.encode_prompt=qwen_encode
        try:
            msgs=self.conv()
            prompt,_=qwen_encode(msgs, None, 'chat', None); ids=s.TOK.encode(prompt)
            self.assertIn(len(ids)-1, s.boundary_hints(msgs, None, 'chat', None, prompt, ids, gen_cut=True))
        finally:
            for k in added: MARKERS.pop(k)
            s.encode_prompt=hint_encode

    def test_generation_suffix_hint_for_an_added_message(self):
        """2026-10-19: a background job sent each item twice — the second prompt was the first one up to the generation prompt plus a
        further user message (no answer in between), so it parted from the prompt-end checkpoint before it. A hint sits at the end
        of the last message too, and it is a prefix of that second request; a first turn without a conversation key gets it."""
        def gen_encode(messages, tools, thinking_mode, effort):
            text=''
            for m in messages:
                text+='<|'+m['role']+'|>'+(m.get('content') or '')
                if m['role']=='assistant': text+='<|eos|>'
            return text+'<|assistant|><think>', []
        added={'<think>':7}
        MARKERS.update(added); s.encode_prompt=gen_encode
        try:
            msgs=[{'role':'system','content':self.SYS}, {'role':'user','content':'a long document and the task'}]
            prompt,_=gen_encode(msgs, None, 'chat', None); ids=s.TOK.encode(prompt)
            hints=s.boundary_hints(msgs, None, 'chat', None, prompt, ids, gen_cut=True)
            self.assertIn(len(ids)-2, hints)                                                                 # before <|assistant|>
            p2,_=gen_encode(msgs+[{'role':'user','content':'answer again in JSON only'}], None, 'chat', None)
            self.assertEqual(s.TOK.encode(p2)[:len(ids)-2], ids[:len(ids)-2])                                # a prefix of that request
        finally:
            for k in added: MARKERS.pop(k)
            s.encode_prompt=hint_encode
        # the same, when the template joins a further user message with "\n\n" and the last message's closing "." merges with it
        #   (measured 2026-10-19 on DeepSeek: the request parted one token before the hint)
        def join_encode(messages, tools, thinking_mode, effort):
            text, prev=('', None)
            for m in messages:
                if m['role']=='user' and prev=='user': text+='\n\n'+(m.get('content') or '')
                else: text+='<|'+m['role']+'|>'+(m.get('content') or '')
                if m['role']=='assistant': text+='<|eos|>'
                prev=m['role']
            return text+'<|assistant|><think>', []
        added={'<think>':7, '.\n\n':10}
        MARKERS.update(added); s.encode_prompt=join_encode
        try:
            msgs=[{'role':'system','content':self.SYS}, {'role':'user','content':'a long document and the task.'}]
            prompt,_=join_encode(msgs, None, 'chat', None); ids=s.TOK.encode(prompt)
            hints=s.boundary_hints(msgs, None, 'chat', None, prompt, ids, gen_cut=True)
            p2,_=join_encode(msgs+[{'role':'user','content':'again in JSON only'}], None, 'chat', None); ids2=s.TOK.encode(p2)
            k=next(i for i,(a,b) in enumerate(zip(ids, ids2)) if a!=b)
            self.assertEqual(k, len(ids)-3)                                                                  # the "." token differs
            self.assertIn(k, hints)                                                                          # the hint is a prefix of it
        finally:
            for k in added: MARKERS.pop(k)
            s.encode_prompt=hint_encode

    def test_turn_prerender_plan_matches_next_request(self):
        """HIVE_TURN_PRERENDER (2026-10-15): the planned ids are the next request's tokens up to where its user message starts — that
        request drops this turn's reasoning (templates that keep reasoning only after the last user message; clients that send it back
        only within a turn) — and the boundary sits exactly there. No plan without reasoning, for a tool call, or for a short prefix.
        2026-10-26: no plan either when the next prompt only appends to this request and its answer (the session holds them), and a
        conversation of one user message takes its client's observed habit — none observed, no plan."""
        def think_encode(messages, tools, thinking_mode, effort):  # keeps reasoning only after the last user message (Qwen / DeepSeek style)
            last = max(i for i, m in enumerate(messages) if m['role'] == 'user')
            text = ''
            for i, m in enumerate(messages):
                text += '<|' + m['role'] + '|>'
                if m['role'] == 'assistant':
                    r = m.get('reasoning_content') or ''
                    text += ('<think>' + r + '</think>' if i > last and r else '') + (m.get('content') or '') + ''.join('[call]' for _ in m.get('tool_calls') or []) + '<|eos|>'
                else:
                    text += m.get('content') or ''
            return text + '<|assistant|><think>', []
        class Parse:
            def parse_message_from_completion_text(self, text, thinking_mode=None, **kw):
                text = text.replace('<|eos|>', '')
                r, c = text.split('</think>', 1) if '</think>' in text else ('', text)
                out = {'reasoning_content': r, 'content': c}
                if '[call]' in c: out = {'reasoning_content': r, 'content': c.replace('[call]', ''), 'tool_calls': [{'name': 'f', 'arguments': {}}]}
                return out
        added = {'<think>': 7, '</think>': 8}
        MARKERS.update(added); saved = (s.encode_prompt, s.ENC, s.PRERENDER_MIN, dict(s.PRERENDER_CLIENT_KEEPS))
        s.encode_prompt = think_encode; s.ENC = Parse(); s.PRERENDER_MIN = 16
        s.PRERENDER_CLIENT_KEEPS.clear()
        try:
            msgs = [{'role': 'system', 'content': self.SYS}, {'role': 'user', 'content': 'fix the build'},
                    {'role': 'assistant', 'content': '', 'reasoning_content': 'look at the log', 'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}}]},
                    {'role': 'tool', 'content': 'error at line 3'}]
            ans = 'line 3 needs a semicolon</think>Added the semicolon.'
            # one user message and nothing observed of the client: no plan (measured 2026-10-22..26 — guessing "drops" there read
            #   33 rooms again in the background for prompts their client never sent)
            self.assertIsNone(s.prerender_plan(msgs, None, 'thinking', None, ans, 'dropper'))
            s.PRERENDER_CLIENT_KEEPS['dropper'] = False                              # seen dropping earlier turns' reasoning
            info = {}
            plan = s.prerender_plan(msgs, None, 'thinking', None, ans, 'dropper', info)
            self.assertEqual(info, {'basis': 'client'})
            self.assertIsNotNone(plan)
            ids, bounds, text = plan
            nxt = [dict(m) for m in msgs]
            nxt[2].pop('reasoning_content')
            nxt += [{'role': 'assistant', 'content': 'Added the semicolon.'}, {'role': 'user', 'content': 'thanks, run the tests'}]
            prompt, _ = think_encode(nxt, None, 'thinking', None)
            nids = s.TOK.encode(prompt)
            b = bounds[-1]
            self.assertEqual(b, len(ids) - 1)
            self.assertEqual(ids[:b], nids[:b])                                      # every token before the boundary is the next request's
            self.assertEqual(nids[b - 1], MARKERS['<|user|>'])                       # the boundary is where the next user message's content starts
            self.assertTrue(prompt.startswith(text) and text.endswith('<|user|>'))  # the kept text = the next prompt up to the user's content
            # the conversation as sent now (reasoning kept in this turn) differs earlier — that is the re-read the plan saves
            now, _ = think_encode(msgs + [{'role': 'assistant', 'content': 'Added the semicolon.', 'reasoning_content': 'x'}], None, 'thinking', None)
            self.assertNotEqual(s.TOK.encode(now)[:b], nids[:b])
            # GLM's official template keeps every assistant message's reasoning (clear_thinking false) and the client does not send it back
            #   after the turn: the plan must drop it itself
            def keep_all(messages, tools, thinking_mode, effort):
                text = ''
                for m in messages:
                    text += '<|' + m['role'] + '|>'
                    if m['role'] == 'assistant':
                        r = m.get('reasoning_content') or ''
                        text += ('<think>' + r + '</think>' if r else '') + (m.get('content') or '') + ''.join('[call]' for _ in m.get('tool_calls') or []) + '<|eos|>'
                    else:
                        text += m.get('content') or ''
                return text + '<|assistant|><think>', []
            s.encode_prompt = keep_all
            ids, bounds, _ = s.prerender_plan(msgs, None, 'thinking', None, ans, 'dropper')
            nids = s.TOK.encode(keep_all(nxt, None, 'thinking', None)[0])
            b = bounds[-1]
            self.assertEqual(ids[:b], nids[:b]); self.assertEqual(nids[b - 1], MARKERS['<|user|>'])
            # a client that sends earlier turns' reasoning back (keeps it on every turn) sends this turn's too: with a template that keeps
            #   it as well the next prompt only appends to this request and its answer — the session holds them, no plan
            kept = [{'role': 'system', 'content': self.SYS}, {'role': 'user', 'content': 'hello'},
                    {'role': 'assistant', 'content': 'Hi.', 'reasoning_content': 'greet back'}, *msgs[1:]]
            info = {}
            self.assertIsNone(s.prerender_plan(kept, None, 'thinking', None, ans, 'keeper', info))
            self.assertEqual(info, {'basis': 'turns'}); self.assertIs(s.PRERENDER_CLIENT_KEEPS['keeper'], True)   # the habit is learned
            # ... and its one-message conversations (a background job's tool loop) take that habit: an append again, no plan
            self.assertIsNone(s.prerender_plan(msgs, None, 'thinking', None, ans, 'keeper'))
            # a template that drops earlier turns' reasoning rewrites the turn even for that client: the plan keeps what the client sends
            s.encode_prompt = think_encode
            ids, bounds, _ = s.prerender_plan(kept, None, 'thinking', None, ans, 'keeper')
            nxt_kept = [*kept, {'role': 'assistant', 'content': 'Added the semicolon.', 'reasoning_content': 'line 3 needs a semicolon'},
                        {'role': 'user', 'content': 'thanks, run the tests'}]
            nids = s.TOK.encode(think_encode(nxt_kept, None, 'thinking', None)[0])
            b = bounds[-1]
            self.assertEqual(ids[:b], nids[:b]); self.assertEqual(nids[b - 1], MARKERS['<|user|>'])
            # an answer whose re-read would be shorter than PRERENDER_MIN is not read ahead (what the session misses is that tail only)
            s.PRERENDER_MIN = b - 2
            self.assertIsNone(s.prerender_plan(kept, None, 'thinking', None, ans, 'keeper'))
            s.PRERENDER_MIN = 16
            self.assertIsNone(s.prerender_plan(msgs[:2], None, 'thinking', None, 'Done.', 'dropper'))            # no reasoning in the turn
            self.assertIsNone(s.prerender_plan(msgs, None, 'thinking', None, 'again</think>[call]', 'dropper'))  # the turn goes on (a tool call)
            s.PRERENDER_MIN = 10 ** 6
            self.assertIsNone(s.prerender_plan(msgs, None, 'thinking', None, 'r</think>Added.', 'dropper'))      # too short to be worth it
        finally:
            for k in added: MARKERS.pop(k)
            s.encode_prompt, s.ENC, s.PRERENDER_MIN = saved[:3]
            s.PRERENDER_CLIENT_KEEPS.clear(); s.PRERENDER_CLIENT_KEEPS.update(saved[3])

    def test_prerender_match_records_where_the_next_prompt_differs(self):
        """HIVE_TURN_PRERENDER follow-up (2026-10-16): the next request of a conversation compares its prompt with the read-ahead text
        once — kept length, and on a mismatch the token position and both texts around it; a read-ahead that finishes after the next
        request arrived is not kept (it would be compared with the turn after)."""
        saved = (s.TURN_PRERENDER, dict(s.PRERENDER_TEXT), dict(s.PRERENDER_SEEN), dict(s.PRERENDER_OFF), dict(s.PRERENDER_BASIS),
                 dict(s.PRERENDER_CLIENT_KEEPS))
        s.TURN_PRERENDER = True
        try:
            s.PRERENDER_TEXT.clear(); s.PRERENDER_SEEN.clear(); s.PRERENDER_OFF.clear(); s.PRERENDER_BASIS.clear()
            self.assertEqual(s.prerender_match('c', 'anything', None), {})
            s.PRERENDER_TEXT['c'] = 'sys|user a|asst ok|user '
            r = s.prerender_match('c', 'sys|user a|asst ok|user next question', None)
            self.assertEqual(r, {'prerender_chars': 24, 'prerender_kept_chars': 24})
            self.assertEqual(s.prerender_match('c', 'sys|user a|asst ok|user next question', None), {})   # one use
            s.PRERENDER_TEXT['c'] = 'sys|user a|asst ok|user '
            prompt = 'sys|user a|asst OK|user next'
            ends = array('l', range(1, len(prompt) + 1))                                                # one token per character
            r = s.prerender_match('c', prompt, ends)
            self.assertEqual((r['prerender_kept_chars'], r['prerender_miss']['at_token']), (16, 16))
            self.assertTrue(r['prerender_miss']['pre'].endswith('ok|user ') and r['prerender_miss']['got'].endswith('OK|user next'))
            # 2026-10-26: a miss stops this conversation's read-aheads, and forgets the client habit the read-ahead relied on
            self.assertIn('c', s.PRERENDER_OFF)
            s.PRERENDER_CLIENT_KEEPS['ua'] = False
            s.PRERENDER_TEXT['d'] = 'sys|user a|asst ok|user '; s.PRERENDER_BASIS['d'] = ('ua', 'client')
            s.prerender_match('d', prompt, ends)
            self.assertNotIn('ua', s.PRERENDER_CLIENT_KEEPS)
            s.PRERENDER_CLIENT_KEEPS['ua'] = True                                     # a habit seen in the request's own turns stays
            s.PRERENDER_TEXT['e'] = 'sys|user a|asst ok|user '; s.PRERENDER_BASIS['e'] = ('ua', 'turns')
            s.prerender_match('e', prompt, ends)
            self.assertIs(s.PRERENDER_CLIENT_KEEPS['ua'], True)
            s.PRERENDER_OFF.clear()
            # a read-ahead started before the conversation's latest request is dropped when it finishes
            class D:
                async def generate(self, *a, **k):
                    yield {'done': True}
            saved_d, saved_log = s.DAEMON, s.log_request
            s.DAEMON, s.log_request = D(), (lambda rec: None)
            try:
                s.PRERENDER_SEEN['c'] = time.time() + 5
                asyncio.run(s.turn_prerender('c', ([1, 2], [1], 'txt')))
                self.assertNotIn('c', s.PRERENDER_TEXT)
                s.PRERENDER_SEEN['c'] = 0.0
                asyncio.run(s.turn_prerender('c', ([1, 2], [1], 'txt')))
                self.assertEqual(s.PRERENDER_TEXT.get('c'), 'txt')
            finally:
                s.DAEMON, s.log_request = saved_d, saved_log
        finally:
            s.TURN_PRERENDER = saved[0]
            s.PRERENDER_TEXT.clear(); s.PRERENDER_TEXT.update(saved[1]); s.PRERENDER_SEEN.clear(); s.PRERENDER_SEEN.update(saved[2])
            for d, v in zip((s.PRERENDER_OFF, s.PRERENDER_BASIS, s.PRERENDER_CLIENT_KEEPS), saved[3:]):
                d.clear(); d.update(v)

    def test_prerender_task_is_kept_until_done(self):
        """The read-ahead task is strongly referenced until it ends (2026-10-15/16: 13 of 59 were garbage-collected mid-request)."""
        saved = (s.TURN_PRERENDER, s.prerender_plan, s.turn_prerender)
        seen = []
        async def slow(session, plan, basis=None):
            import gc
            gc.collect()
            await asyncio.sleep(0.05)
            seen.append(session)
        async def main():
            s.schedule_prerender([], None, 'thinking', None, 'c', 'x')
            await asyncio.sleep(0)
            self.assertEqual(len(s.PRERENDER_TASKS), 1)
            for _ in range(50):
                if seen: break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0)
        s.TURN_PRERENDER, s.prerender_plan, s.turn_prerender = True, (lambda *a: ([1], [1], 't')), slow
        try:
            asyncio.run(main())
            self.assertEqual(seen, ['c'])
            self.assertEqual(len(s.PRERENDER_TASKS), 0)
            # a conversation whose read-ahead missed is not read ahead again (2026-10-26)
            seen.clear(); s.PRERENDER_OFF['c'] = True
            async def off():
                s.schedule_prerender([], None, 'thinking', None, 'c', 'x')
                self.assertEqual(len(s.PRERENDER_TASKS), 0)                     # no task at all
                await asyncio.sleep(0.1)
            asyncio.run(off())
            self.assertEqual(seen, [])
            s.PRERENDER_OFF.pop('c')
        finally:
            s.TURN_PRERENDER, s.prerender_plan, s.turn_prerender = saved

    def test_request_priority(self):
        self.assertEqual(s.request_priority({}), 0)
        self.assertEqual(s.request_priority({'priority': 3}), 3)
        self.assertEqual(s.request_priority({'priority': -5000}), -1000)
        self.assertEqual(s.request_priority({'priority': True}), 0)                    # bool is not a number here
        self.assertEqual(s.request_priority({'priority': 'high'}), 0)
        self.assertEqual(s.request_priority({'service_tier': 'flex'}), 1)
        self.assertEqual(s.request_priority({'service_tier': 'priority'}), -1)
        self.assertEqual(s.request_priority({'service_tier': 'flex', 'priority': 0}), 0)  # an explicit priority wins

    def test_conversation_key_fields(self):
        class R:
            def __init__(self, h): self.headers=h
        self.assertEqual(s.conversation_key({'prompt_cache_key':'a', 'session_id':'b', 'user':'u'}, R({'x-session-id':'c'})), 'a')
        self.assertEqual(s.conversation_key({'session_id':'b', 'user':'u'}, R({'x-session-id':'c'})), 'b')
        self.assertEqual(s.conversation_key({'user':'u'}, R({'x-session-id':'c'})), 'c')
        self.assertEqual(s.conversation_key({'user':'u'}, R({})), 'u')
        self.assertEqual(s.conversation_key({'prompt_cache_key':'  '}, None), '')
        head=[{'role':'system','content':'S'}, {'role':'user','content':'fixed instructions'}]
        self.assertNotEqual(s.session_id_for(head, 'room-1'), s.session_id_for(head, 'room-2'))           # same head, different rooms
        self.assertNotEqual(s.session_id_for(head, 'room-1'), s.session_id_for(head+[], 'room-1x'))

    def test_tools_without_system_and_turn_cap(self):
        tools=[{'type':'function','function':{'name':'f'}}]
        msgs=[{'role':'user','content':'q'}]
        prompt, ids, hints=self.hints_for(msgs, tools)
        self.assertEqual(hints, [len(s.TOK.encode('<|system|><|tools|>'+json.dumps(tools)+'<|user|>'))])
        long=[{'role':'user','content':'u0'}]
        for k in range(8): long+=[{'role':'assistant','content':f'a{k}'}, {'role':'user','content':f'u{k+1}'}]
        _, ids, hints=self.hints_for(long)
        self.assertEqual(len(hints), s.HINT_TURNS, 'only the last completed turns')
        self.assertEqual(hints, sorted(hints)); self.assertTrue(all(0<h<len(ids) for h in hints))

    def test_mismatched_tokenizer_or_error_gives_no_hints(self):
        msgs=self.conv(); prompt,_=hint_encode(msgs, None, 'chat', None)
        self.assertEqual(s.boundary_hints(msgs, None, 'chat', None, prompt, s.TOK.encode(prompt)[:-1]), [])
        class NoOffsets(HintTokenizer):
            def __call__(self, text, **kw): raise TypeError('no offsets')
        s.TOK=NoOffsets()
        self.assertEqual(s.boundary_hints(msgs, None, 'chat', None, prompt, s.TOK.encode(prompt)), [])

    async def test_adaptive_extra_chunk_for_changing_tails(self):
        """HIVE_PREFIX_ADAPTIVE: prefix_extra=1 only when the previous request was not a prefix of this one but shared one of its
        boundaries (a block near the end is replaced each turn), or when a conversation that already has turns is seen for the first
        time (2026-10-08). Append-only conversations and a first turn (no assistant message yet) never get it."""
        os.environ['HIVE_PREFIX_SHARE']='1'
        hist=[{'role':'system','content':self.SYS}, {'role':'user','content':'first question'}, {'role':'assistant','content':'first answer'}]
        def turn(ctx, q='second question'):  # fresh context placed just before the newest user message (the client pattern)
            return hist+[{'role':'user','content':'[context '+ctx+']'}, {'role':'user','content':q}]
        def extra(): return s.DAEMON.kw[-1].get('prefix_extra')
        try:
            s.TAILS=s.TailChangeTracker()
            await s.chat(Request({'messages':turn('A'), 'reasoning_effort':'none'}))
            self.assertIsNone(extra(), 'off by default')
            os.environ['HIVE_PREFIX_ADAPTIVE']='1'
            s.TAILS=s.TailChangeTracker()
            await s.chat(Request({'messages':turn('A'), 'reasoning_effort':'none'}))
            self.assertEqual(extra(), 1, 'first sight of a conversation that already has turns (a restart): snapshot once')
            await s.chat(Request({'messages':turn('B'), 'reasoning_effort':'none'}))
            self.assertEqual(extra(), 1, 'tail replaced after a shared boundary')
            self.assertIn('boundaries', s.DAEMON.kw[-1])
            await s.chat(Request({'messages':turn('C'), 'reasoning_effort':'none'}))
            self.assertEqual(extra(), 1, 'still changing')
            # append-only: the next request extends the previous one (answer + new question) -> no extra chunk
            nxt=turn('C')+[{'role':'assistant','content':'answer C'}, {'role':'user','content':'third question'}]
            await s.chat(Request({'messages':nxt, 'reasoning_effort':'none'}))
            self.assertIsNone(extra(), 'append-only turn')
            # an unrelated conversation (different head) is tracked separately and starts unflagged
            other=[{'role':'system','content':self.SYS}, {'role':'user','content':'unrelated'}]
            await s.chat(Request({'messages':other, 'reasoning_effort':'none'}))
            self.assertIsNone(extra())
            # the tracker keeps only hashes and stays bounded
            t=s.TailChangeTracker(cap=2)
            for k in range(5): t.observe(f'k{k}', list(range(100)), [10, 50])
            self.assertEqual(list(t.entries), ['k3', 'k4'])
            self.assertTrue(all(isinstance(e[1], bytes) and len(e[1]) == 16 for e in t.entries.values()))
            # a different prefix before every boundary is a new conversation shape, not a changing tail
            t2=s.TailChangeTracker()
            self.assertFalse(t2.observe('c', [1]*60, [20, 40]))
            self.assertTrue(s.TailChangeTracker().observe('h', [1]*60, [20, 40], history=True))   # unknown, with history
            # 2026-10-08: the next turn continues at the previous request's generation-suffix hint (59), not at its end — append-only
            t3=s.TailChangeTracker()
            self.assertFalse(t3.observe('g', [1]*60, [20, 59]))
            self.assertFalse(t3.observe('g', [1]*59+[5]*20, [20, 59, 78]))
            t4=s.TailChangeTracker()   # the same pair without the suffix hint reads as a changed tail
            self.assertFalse(t4.observe('g', [1]*60, [20]))
            self.assertTrue(t4.observe('g', [1]*59+[5]*20, [20]))
            self.assertFalse(t2.observe('c', [2]*60, [20, 40]))
            self.assertTrue(t2.observe('c', [2]*40+[3]*30, [20, 40]))
            # 2026-10-14: a conversation with turns whose head changed (re-read from the top) still saves a boundary for its next turn
            t5=s.TailChangeTracker()
            t5.observe('r', [1]*60, [20, 40, 59], history=True)   # first sight with history (flagged; covered above)
            self.assertTrue(t5.observe('r', [7]*80, [20, 40, 79], history=True))
            self.assertFalse(t5.observe('r', [7]*79+[4]*10, [20, 40, 79, 88], history=True), 'append-only stays unflagged')
        finally:
            os.environ.pop('HIVE_PREFIX_ADAPTIVE', None)
            s.TAILS=s.TailChangeTracker()

    async def test_first_turn_seen(self):
        """HIVE_PREFIX_FIRST_TURN=seen: a first turn asks for the boundary cut only when another conversation already sent the same
        system block; =1 asks on every first turn; later turns never get it from this switch."""
        os.environ['HIVE_PREFIX_SHARE']='1'
        def first(q, sys_text=None): return [{'role':'system','content':sys_text or self.SYS}, {'role':'user','content':q}]
        def extra(): return s.DAEMON.kw[-1].get('prefix_extra')
        try:
            os.environ['HIVE_PREFIX_FIRST_TURN']='1'
            s.SEEN=s.SeenPrefixes()
            await s.chat(Request({'messages':first('q1'), 'reasoning_effort':'none'}))
            self.assertEqual(extra(), 1, '=1: every first turn')
            os.environ['HIVE_PREFIX_FIRST_TURN']='seen'
            s.SEEN=s.SeenPrefixes()
            await s.chat(Request({'messages':first('q1'), 'reasoning_effort':'none'}))
            self.assertIsNone(extra(), 'first conversation with this block: no cut')
            await s.chat(Request({'messages':first('q1'), 'reasoning_effort':'none'}))
            self.assertIsNone(extra(), 'the same conversation again is not evidence')
            await s.chat(Request({'messages':first('q2'), 'reasoning_effort':'none'}))
            self.assertEqual(extra(), 1, 'second conversation with the same block: cut')
            self.assertIn('boundaries', s.DAEMON.kw[-1])
            await s.chat(Request({'messages':first('q3', self.SYS+' (dated 2026-10-07)'), 'reasoning_effort':'none'}))
            self.assertIsNone(extra(), 'a block that differs at its end is a new block')
            # 2026-10-18: a room's own context block after the shared system block — the cut goes to the shared boundary
            def room(ctx, q): return [{'role':'system','content':self.SYS}, {'role':'user','content':ctx}, {'role':'user','content':q}]
            s.SEEN=s.SeenPrefixes()
            await s.chat(Request({'messages':room('room A context '*20, 'question about room A '*8), 'reasoning_effort':'none'}))
            self.assertIsNone(extra(), 'first room: no cut')
            hints_a=s.DAEMON.kw[-1].get('boundaries') or []
            self.assertEqual(len(hints_a), 3, hints_a)  # system block, room block, end of the last message (generation suffix)
            await s.chat(Request({'messages':room('room B context, other words '*20, 'question about room B '*8), 'reasoning_effort':'none'}))
            self.assertEqual(extra(), 1, 'second room: cut')
            self.assertEqual(s.DAEMON.kw[-1].get('boundaries')[:1], [hints_a[0]], 'only the shared system-block boundary is sent')
            self.assertTrue(all(h > hints_a[1] for h in s.DAEMON.kw[-1].get('boundaries')[1:]), 'besides it, only suffix hints')
            later=first('q2')+[{'role':'assistant','content':'a'}, {'role':'user','content':'q2b'}]
            await s.chat(Request({'messages':later, 'reasoning_effort':'none'}))
            self.assertIsNone(extra(), 'not a first turn')
            t=s.SeenPrefixes(cap=2)
            for k in range(4): t.observe('c', [k]*30, [10])
            self.assertEqual(len(t.entries), 2)
            self.assertFalse(t.observe('d', [0]*30, [10]), 'evicted prefix counts as new')
            self.assertTrue(t.observe('e', [3]*30, [10]))
        finally:
            os.environ.pop('HIVE_PREFIX_FIRST_TURN', None)
            s.SEEN=s.SeenPrefixes()

    def test_rewinding_family_gets_no_suffix_hints(self):
        """2026-10-19: a family that rewinds its live KV (Gemma) continues a repeated prompt from its live state — the suffix hints of a
        first turn only left snapshots (two per short request, the copies that stalled the engine on 10-18). The server asks for them
        only from families that cannot rewind (gen_cut=False: no suffix hint, test_generation_suffix_hint_is_where_the_next_turn_continues)."""
        fam, had, rw=s.FAMILY, hasattr(s.ENC, 'REWINDS'), getattr(s.ENC, 'REWINDS', None)
        try:
            s.FAMILY=None
            self.assertFalse(s.rewinding_family(), 'DeepSeek: suffix hints')
            s.FAMILY=next(iter(s.ADAPTER_FAMILIES)); s.ENC.REWINDS=False
            self.assertFalse(s.rewinding_family(), 'an adapter family without rewind: suffix hints')
            s.ENC.REWINDS=True
            self.assertTrue(s.rewinding_family(), 'Gemma: none')
        finally:
            s.FAMILY=fam
            if had: s.ENC.REWINDS=rw
            elif hasattr(s.ENC, 'REWINDS'): del s.ENC.REWINDS

    async def test_sent_only_when_enabled(self):
        await s.chat(Request({'messages':self.conv(), 'reasoning_effort':'none'}))
        self.assertNotIn('boundaries', s.DAEMON.kw[-1], 'default: no field, no hint cost')
        os.environ['HIVE_PREFIX_SHARE']='0'
        await s.chat(Request({'messages':self.conv(), 'reasoning_effort':'none'}))
        self.assertNotIn('boundaries', s.DAEMON.kw[-1])
        os.environ['HIVE_PREFIX_SHARE']='1'
        await s.chat(Request({'messages':self.conv(), 'reasoning_effort':'none'}))
        prompt, ids, _=self.hints_for(self.conv())
        want=s.boundary_hints(self.conv(), None, 'chat', None, prompt, ids, gen_cut=True)  # every text request gets the suffix hints (2026-10-19)
        self.assertEqual(s.DAEMON.kw[-1].get('boundaries'), want)
        s.tokenize_with_images=lambda prompt, media: (s.TOK.encode(prompt), [{'start':0}], b'x')  # image request: no text offsets
        s.encode_prompt=lambda *a: (hint_encode(*a)[0], [{'image':1}])
        await s.chat(Request({'messages':self.conv(), 'reasoning_effort':'none'}))
        self.assertNotIn('boundaries', s.DAEMON.kw[-1])
        self.assertFalse(s.INFLIGHT)



def rich_encode(messages, tools, thinking_mode, effort):
    """A template whose output depends on everything the real ones read: thinking mode and effort, tools, a reasoning field
    shown only after the last user message (DeepSeek drops earlier reasoning), tool-call arguments rendered as JSON (types and key
    order matter), image parts (media returned, a placeholder in the text)."""
    s.check_image_parts(messages)
    messages=[dict(m) for m in messages]
    if tools and messages and messages[0].get('role')!='system': messages.insert(0, {'role':'system', 'content':''})
    last_user=max([i for i, m in enumerate(messages) if m.get('role')=='user'], default=-1)
    media, text=[], '<|bos|>'
    if thinking_mode=='thinking' and effort is not None: text+=f'[effort {effort!r}]'
    for i, m in enumerate(messages):
        c=m.get('content')
        if isinstance(c, list):
            parts=[]
            for p in c:
                if p.get('type')=='image_url': media.append(p); parts.append('[img]')
                else: parts.append(p.get('text', ''))
            c=''.join(parts)
        text+='<|'+m['role']+'|>'+(c or '')
        if i==0 and tools: text+='<|tools|>'+json.dumps(tools)
        if m['role']=='assistant':
            if thinking_mode=='thinking' and i>last_user: text+='(r:'+(m.get('reasoning_content') or '')+')'
            for tc in m.get('tool_calls') or []: text+='{call '+json.dumps(tc['function'])+'}'
            text+='<|eos|>'
    return text+'<|assistant|>'+('(think' if thinking_mode=='thinking' else ''), media

class HintCacheTests(unittest.TestCase):
    """HIVE_HINT_CACHE: hints with the probe cache == hints rendered from scratch (HIVE_HINT_CACHE=0), request after request."""
    TOOLS=[{'type':'function','function':{'name':'read','parameters':{'type':'object','properties':{'path':{'type':'string'}}}}}]
    def setUp(self):
        self.saved=(s.TOK, s.encode_prompt, s.HINT_CACHE)
        s.TOK=HintTokenizer(); s.encode_prompt=self.counting(rich_encode); s.HINT_CACHE=s.HintProbeCache()
        self.env={k: os.environ.pop(k, None) for k in ('HIVE_HINT_CACHE', 'HIVE_HINT_CACHE_MB')}
    def tearDown(self):
        s.TOK, s.encode_prompt, s.HINT_CACHE=self.saved
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None: os.environ[k]=v
    def counting(self, enc):
        self.renders=0
        def f(*a):
            self.renders+=1
            return enc(*a)
        return f

    def both(self, msgs, tools=None, mode='chat', effort=None, gen_cut=True):
        """Cached hints (offsets from a tokenization, and as TokenCache's int64 array) checked against the uncached path."""
        prompt, media=s.encode_prompt(msgs, tools, mode, effort)
        ids=s.TOK.encode(prompt)
        os.environ['HIVE_HINT_CACHE']='0'
        want=s.boundary_hints(msgs, tools, mode, effort, prompt, ids, gen_cut=gen_cut)
        os.environ.pop('HIVE_HINT_CACHE')
        r0=self.renders
        got=s.boundary_hints(msgs, tools, mode, effort, prompt, ids, gen_cut=gen_cut)
        self.last_renders=self.renders-r0
        ends=array('l', [e for _, e in s.TOK(prompt, True)['offset_mapping']])
        got2=s.boundary_hints(msgs, tools, mode, effort, prompt, ids, ends, gen_cut=gen_cut)
        self.assertEqual(got, want); self.assertEqual(got2, want)
        return want

    @staticmethod
    def agent(turns, args=None):
        m=[{'role':'system','content':'system prompt '*20}, {'role':'user','content':'task'}]
        for k in range(turns):
            m.append({'role':'assistant','content':'' if k%2 else f'step {k}','reasoning_content':f'why {k}',
                      'tool_calls':[{'id':f'c{k}','type':'function','function':{'name':'read','arguments':args if args is not None else {'path':f'/f{k}'}}}]})
            m.append({'role':'tool','tool_call_id':f'c{k}','content':f'result {k} '*5})
            if k%3==2: m.append({'role':'user','content':f'<info-msg>t{k}</info-msg> go on'})
        return m

    def test_growing_agent_conversation(self):
        for mode, effort in (('chat', None), ('thinking', 'high'), ('thinking', 50)):
            s.HINT_CACHE=s.HintProbeCache()
            for k in range(1, 12):
                hints=self.both(self.agent(k), self.TOOLS, mode, effort)
                self.assertTrue(hints)
                if k>=2:  # system block and older turn ends come from the cache: the newest turn end + the suffix probes (pair + added message) only
                    self.assertEqual(self.last_renders, 5, (mode, k))
            self.assertGreater(s.HINT_CACHE.hits, 0)

    def test_edited_history_misses(self):
        base=self.agent(6)
        self.both(base, self.TOOLS)
        def edit(f):
            e=json.loads(json.dumps(base)); f(e); return e
        def args(v):
            def f(e): e[2]['tool_calls'][0]['function']['arguments']=v
            return f
        groups=[[edit(lambda e: e[1].update(content='a much longer task description'))],   # first user message
                [edit(lambda e: e[2].update(content='step 0 rewritten'))],                   # an older assistant turn
                [edit(lambda e: e[0].update(content='system prompt '*21))],                  # the system block
                [base[:3]+base[5:]],                                                         # a dropped turn
                # run back to back, so each one meets the entry the previous one left: 1 / True / 1.0 render differently, so do
                #   0.0 / -0.0 and two key orders, yet plain == calls each pair equal
                [edit(args({'path':1})), edit(args({'path':True})), edit(args({'path':1.0})), edit(args({'path':1}))],
                [edit(args({'path':0.0})), edit(args({'path':-0.0}))],
                [edit(args({'a':1,'b':2})), edit(args({'b':2,'a':1}))]]
        for group in groups:
            for e in group:
                self.both(e, self.TOOLS)
            self.both(base, self.TOOLS)   # and back: the result is still exact

    def test_tools_mode_and_effort_changes(self):
        m=self.agent(5)
        other=[{'type':'function','function':{'name':'write','parameters':{}}}]
        for tools, mode, effort in ((self.TOOLS, 'chat', None), (other, 'chat', None), (None, 'chat', None), (self.TOOLS, 'thinking', None),
                                    (self.TOOLS, 'thinking', 'high'), (self.TOOLS, 'thinking', 'max'), (self.TOOLS, 'thinking', 75),
                                    (self.TOOLS, 'thinking', 75.0), (self.TOOLS, 'thinking', True), (self.TOOLS, 'chat', None)):
            self.both(m, tools, mode, effort)
            self.both(m, tools, mode, effort)

    def test_image_messages_emit_nothing_there(self):
        img={'type':'image_url','image_url':{'url':'data:image/png;base64,AAAA'}}
        m=[{'role':'system','content':'sys'}, {'role':'user','content':[{'type':'text','text':'look'}, img]},
           {'role':'assistant','content':'a cat'}, {'role':'user','content':'and now?'}]
        for _ in range(2):
            prompt, media=rich_encode(m, None, 'chat', None)
            self.assertTrue(media)
            hints=self.both(m)
            self.assertEqual(hints, [len(s.TOK.encode('<|bos|><|system|>sys<|user|>'))])   # only the block before the image

    def test_info_msg_boundary_stays_before_the_content(self):
        """0ab95c3 (2026-10-08): the boundary must stop where the next message's content starts even when it begins with "<"."""
        s.encode_prompt=self.counting(hint_encode)
        conv=[{'role':'system','content':'S '*10}, {'role':'user','content':'first question'}, {'role':'assistant','content':'first answer'}]
        turn_end=len(s.TOK.encode('<|system|>'+'S '*10+'<|user|>first question<|assistant|>first answer<|eos|><|user|>'))
        for nxt in ([{'role':'user','content':'<info>t</info>'}, {'role':'user','content':'q'}],
                    [{'role':'user','content':'<info>u</info>'}, {'role':'user','content':'q2'}]):
            hints=self.both(conv+nxt, gen_cut=False)
            self.assertIn(turn_end, hints)
            self.assertEqual(s.TOK.encode(hint_encode(conv+nxt, None, 'chat', None)[0])[turn_end-1], MARKERS['<|user|>'])
        self.assertGreater(s.HINT_CACHE.hits, 0)

    def test_off_switch_uncacheable_inputs_budget_and_rebinding(self):
        m=self.agent(4)
        os.environ['HIVE_HINT_CACHE']='0'
        prompt,_=s.encode_prompt(m, self.TOOLS, 'chat', None); ids=s.TOK.encode(prompt)
        s.boundary_hints(m, self.TOOLS, 'chat', None, prompt, ids)
        self.assertEqual(len(s.HINT_CACHE.entries), 0)
        os.environ.pop('HIVE_HINT_CACHE')
        class Odd(str): pass
        odd=[dict(x) for x in m]; odd[1]['content']=Odd('task')   # not JSON-like: rendered, never cached
        self.both(odd, self.TOOLS); self.assertEqual(len(s.HINT_CACHE.entries), 0)
        os.environ['HIVE_HINT_CACHE_MB']='0.0001'                  # ~100 bytes: every entry is evicted at once
        self.both(m, self.TOOLS); self.both(m, self.TOOLS)
        self.assertLessEqual(len(s.HINT_CACHE.entries), 1)
        os.environ.pop('HIVE_HINT_CACHE_MB')
        self.both(m, self.TOOLS); self.assertTrue(s.HINT_CACHE.entries)
        s.encode_prompt=self.counting(hint_encode)                  # another encoder: the kept renders are dropped
        self.both(m[:2]+[{'role':'assistant','content':'x'}, {'role':'user','content':'y'}], self.TOOLS)
        self.assertTrue(all(e[2] is None or not e[2].startswith('<|bos|>') for e in s.HINT_CACHE.entries.values()))

    def test_clock_reading_template_is_not_cached(self):
        s.TOK.chat_template="{{ strftime_now('%Y-%m-%d') }}"
        self.both(self.agent(3), self.TOOLS); self.both(self.agent(3), self.TOOLS)
        self.assertEqual(len(s.HINT_CACHE.entries), 0)

    def test_non_monotonic_offsets(self):
        m=self.agent(2); prompt,_=s.encode_prompt(m, None, 'chat', None); ids=s.TOK.encode(prompt)
        ends=array('l', [e for _, e in s.TOK(prompt, True)['offset_mapping']]); ends[3], ends[4]=ends[4], ends[3]-1
        self.assertEqual(s.boundary_hints(m, None, 'chat', None, prompt, ids, ends), [])
        os.environ['HIVE_HINT_CACHE']='0'
        self.assertEqual(s.boundary_hints(m, None, 'chat', None, prompt, ids, ends), [])
        self.assertIsNone(s._ends_monotonic([1, 2])); self.assertTrue(s._ends_monotonic(array('l', [0, 0, 3])))

    def test_negative_control_a_lossy_snapshot_is_caught(self):
        """The comparison above detects a cache that matches on too little: snapshots that keep only the type."""
        real=s._freeze
        s._freeze=lambda x: type(x).__name__
        try:
            base=self.agent(6); self.both(base, self.TOOLS)
            e=json.loads(json.dumps(base)); e[1]['content']='a much longer task description'
            with self.assertRaises(AssertionError):
                self.both(e, self.TOOLS)
        finally:
            s._freeze=real


@unittest.skipUnless(os.environ.get('HIVE_TEST_CKPT'), 'HIVE_TEST_CKPT (a checkpoint with encoding/ and the tokenizer) not set')
class HintCacheReferenceTests(unittest.TestCase):
    """The same comparison with the reference DeepSeek encoding and tokenizer (optional: HIVE_TEST_CKPT=<checkpoint dir>)."""
    def test_reference_encoding(self):
        saved=(s.TOK, s.ENC, s.encode_prompt, s.HINT_CACHE, s.TOKCACHE)
        glob={k: getattr(s, k) for k in ('IMG', 'FAMILY', 'STOP_IDS', 'EOS_TEXTS', 'DSML_START')}
        s.encode_prompt=ORIG_ENCODE_PROMPT
        s.load_modules(os.environ['HIVE_TEST_CKPT']); s.HINT_CACHE=s.HintProbeCache(); s.TOKCACHE=s.TokenCache()
        try:
            tools=HintCacheTests.TOOLS
            for mode, effort in (('chat', None), ('thinking', 'high')):
                m=[{'role':'system','content':'You are a coding agent. '*50}, {'role':'user','content':'<info-msg>ctx</info-msg> fix the bug'}]
                for k in range(8):
                    m=m+[{'role':'assistant','content':'' if k%2 else f'step {k}','reasoning_content':f'think {k}',
                          'tool_calls':[{'id':f'c{k}','type':'function','function':{'name':'read','arguments':json.dumps({'path':f'/f{k}'})}}]},
                         {'role':'tool','tool_call_id':f'c{k}','content':f'contents {k} '*40}]
                    prompt,_=s.encode_prompt(m, tools, mode, effort)
                    ids, ends=s.TOKCACHE.tokenize(mode, prompt)
                    os.environ['HIVE_HINT_CACHE']='0'
                    want=s.boundary_hints(m, tools, mode, effort, prompt, ids, ends, gen_cut=True)
                    os.environ.pop('HIVE_HINT_CACHE')
                    self.assertEqual(s.boundary_hints(m, tools, mode, effort, prompt, ids, ends, gen_cut=True), want)
                    self.assertTrue(want)
            self.assertGreater(s.HINT_CACHE.hits, 0)
        finally:
            os.environ.pop('HIVE_HINT_CACHE', None)
            s.TOK, s.ENC, s.encode_prompt, s.HINT_CACHE, s.TOKCACHE=saved
            for k, v in glob.items(): setattr(s, k, v)


class ExposureTests(unittest.IsolatedAsyncioTestCase):
    """Public-exposure rules of the server: image inputs, credential-shaped headers, malformed bodies, Anthropic turn conversion."""
    def test_image_url_rules(self):
        s._check_image_url("data:image/png;base64,AAAA")
        saved = s.IMAGE_FETCH
        try:
            s.IMAGE_FETCH = True
            s._check_image_url("https://example.com/a.png")
            for bad in ("/etc/passwd", "file:///etc/passwd", "../x.png", "", None, 7):
                with self.assertRaises(ValueError, msg=repr(bad)):
                    s._check_image_url(bad)
            s.IMAGE_FETCH = False
            with self.assertRaises(ValueError):
                s._check_image_url("https://example.com/a.png")
        finally:
            s.IMAGE_FETCH = saved
        # every shape the two APIs use for image parts is checked; a base64 Anthropic source needs no URL
        s.check_image_parts([{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "AAAA"}}, {"type": "text", "text": "hi"}]}])
        with self.assertRaises(ValueError):
            s.check_image_parts([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "/out/hive.sock"}}]}])
        with self.assertRaises(ValueError):
            s.check_image_parts([{"role": "user", "content": [{"type": "image_url", "image_url": "/etc/hostname"}]}])
        with self.assertRaises(ValueError):
            s.check_image_parts([{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "/etc/hostname"}}]}])

    def test_client_tags_never_log_credentials(self):
        import types
        headers = {"x-hive-feature": "chat", "x-hive-key": "secret1", "authorization": "Bearer t", "x-api-key": "k", "cookie": "a=b",
                   "x-client-token": "t2", "user-agent": "ua"}
        req = types.SimpleNamespace(headers=headers)
        tags = s.client_tags(req, {})
        self.assertEqual(sorted(tags), ["user-agent", "x-hive-feature"])

    async def test_malformed_bodies(self):
        class R:
            def __init__(self, headers, payload=None, raise_=False): self.headers = headers; self.payload = payload; self.raise_ = raise_
            async def json(self):
                if self.raise_: raise ValueError("bad json")
                return self.payload
        body, bad = await s.read_body(R({"content-length": str(s.MAX_BODY_BYTES + 1)}, {}))
        self.assertEqual((body, bad.status_code), (None, 413))
        body, bad = await s.read_body(R({}, raise_=True))
        self.assertEqual((body, bad.status_code), (None, 400))
        body, bad = await s.read_body(R({}, [1, 2]))
        self.assertEqual((body, bad.status_code), (None, 400))
        body, bad = await s.read_body(R({"content-length": "12"}, {"messages": []}))
        self.assertEqual((body, bad), ({"messages": []}, None))

    def test_effort_tolerates_non_object_kwargs(self):
        self.assertEqual(s.effort_from_request({"chat_template_kwargs": "x", "reasoning_effort": "none"})[0], "chat")  # thinking off = mode "chat"
        self.assertEqual(s.effort_from_request({"chat_template_kwargs": [1]})[0], "thinking")

    def test_anthropic_turn_keeps_text_and_tool_calls_together(self):
        body = {"messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "Let me check."},
                                              {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}},
                                              {"type": "tool_use", "id": "t2", "name": "get_time", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "sunny"}]},
                                         {"type": "tool_result", "tool_use_id": "t2", "content": "noon"},
                                         {"type": "text", "text": "thanks"}]}]}
        msgs, tools = s.anthropic_to_openai(body)
        self.assertEqual(tools, None)
        self.assertEqual([m["role"] for m in msgs], ["assistant", "user", "tool", "tool"])
        self.assertEqual(msgs[0]["content"], "Let me check.")
        self.assertEqual([c["function"]["name"] for c in msgs[0]["tool_calls"]], ["get_weather", "get_time"])
        self.assertEqual(json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"]), {"city": "Paris"})
        self.assertEqual((msgs[2]["tool_call_id"], msgs[2]["content"]), ("t1", "sunny"))
        self.assertEqual((msgs[3]["tool_call_id"], msgs[3]["content"]), ("t2", "noon"))
        self.assertEqual(msgs[1]["content"], "thanks")


class ImageSpanHintTests(unittest.TestCase):
    def test_in_span(self):
        spans = [(10, 5), (40, 3)]
        self.assertEqual([t for t in range(50) if s._in_span(t, spans)], [10, 11, 12, 13, 14, 40, 41, 42])
        self.assertFalse(s._in_span(12, None)); self.assertFalse(s._in_span(12, []))
        self.assertEqual([s._out_of_spans(t, spans) for t in (10, 12, 15, 41)], [10, 10, 15, 40])


class DisconnectTests(unittest.IsolatedAsyncioTestCase):
    """2026-10-27: behind Starlette's @app.middleware("http") a non-streaming request never saw its client leave (a dropped batch
    request generated 982K tokens for two hours). ApiKeyGuard is a plain ASGI middleware: the endpoint gets the server's receive."""

    async def _run(self, wrap, path="/probe", headers=()):
        from fastapi import FastAPI, Request as FRequest
        inner = FastAPI()
        seen = {}

        @inner.post("/probe")
        @inner.post("/health")
        async def probe(request: FRequest):
            await request.json()
            for _ in range(50):
                if await request.is_disconnected():
                    seen["disconnect"] = True
                    break
                await asyncio.sleep(.01)
            return s.JSONResponse({})
        app = wrap(inner)
        msgs = [{"type": "http.request", "body": b"{}", "more_body": False}]
        gone = time.monotonic() + .05   # the client leaves 50 ms after its body

        async def receive():   # like uvicorn's: returns at once once the client is gone, else waits (is_disconnected cancels the wait)
            if msgs:
                return msgs.pop(0)
            while time.monotonic() < gone:
                await asyncio.sleep(.01)
            return {"type": "http.disconnect"}
        sent = []

        async def send(m):
            sent.append(m)
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1", "method": "POST",
                 "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
                 "headers": [(b"content-type", b"application/json"), *headers], "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 80)}
        await app(scope, receive, send)
        return seen.get("disconnect", False), next((m["status"] for m in sent if m["type"] == "http.response.start"), None)

    async def test_guard_passes_the_disconnect(self):
        saved = s.API_KEY
        try:
            for key in (None, "k1"):
                s.API_KEY = key
                hdr = ((b"authorization", b"Bearer k1"),) if key else ()
                seen, _ = await self._run(s.ApiKeyGuard, headers=hdr)
                self.assertTrue(seen, key)
        finally:
            s.API_KEY = saved

    async def test_guard_rejects_a_wrong_key(self):
        saved = s.API_KEY
        try:
            s.API_KEY = "k1"
            self.assertEqual((await self._run(s.ApiKeyGuard))[1], 401)
            self.assertEqual((await self._run(s.ApiKeyGuard, headers=((b"x-api-key", b"nope"),)))[1], 401)
            self.assertEqual((await self._run(s.ApiKeyGuard, headers=((b"x-api-key", b"k1"),)))[1], 200)
            self.assertEqual((await self._run(s.ApiKeyGuard, path="/health"))[1], 200)
        finally:
            s.API_KEY = saved

    def test_server_app_uses_the_guard(self):
        self.assertTrue(any(getattr(m, "cls", None) is s.ApiKeyGuard for m in s.app.user_middleware))
        from starlette.middleware.base import BaseHTTPMiddleware
        self.assertFalse(any(getattr(m, "cls", None) is BaseHTTPMiddleware for m in s.app.user_middleware))


def bpe_tokenizer():
    """A small real fast tokenizer (byte-level BPE trained on the fly) with role markers as added special tokens — the
    structure TokenCache relies on, without any model files."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    corpus = ['the engine streams experts over PCIe and computes misses on the CPU',
              '\uc11c\ubc84 \uc900\ube44 \ub2e8\uacc4\uc758 \uc2dc\uac04\uc744 \uc7ac\uae30 \uc704\ud55c \uae34 \ub300\ud654\uc785\ub2c8\ub2e4', 'def f(x):\n    return [i * i for i in range(x)]'] * 50
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=['<bos>', '<|user|>', '<|assistant|>', '<|system|>', '<pad>']))
    tk.post_processor = processors.TemplateProcessing(single='<bos> $A', special_tokens=[('<bos>', tk.token_to_id('<bos>'))])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token='<bos>', pad_token='<pad>')
    tok.add_special_tokens({'additional_special_tokens': ['<|user|>', '<|assistant|>', '<|system|>']})
    return tok


class TokenCacheTests(unittest.TestCase):
    """HIVE_TOKEN_CACHE: cached ids (prefix reused up to a role marker + tail tokenized) == whole-prompt tokenization."""
    def setUp(self):
        self.tok_before = s.TOK
        s.TOK = bpe_tokenizer()
        self.cache = s.TokenCache()
        self.env = {k: os.environ.pop(k, None) for k in ('HIVE_TOKEN_CACHE', 'HIVE_TOKEN_CACHE_VERIFY', 'HIVE_TOKEN_CACHE_SESSIONS', 'HIVE_TOKEN_CACHE_TOKENS')}
    def tearDown(self):
        s.TOK = self.tok_before
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None: os.environ[k] = v

    @staticmethod
    def render(turns):
        return '<|system|>you are helpful. ' + ''.join(f'<|{r}|>{t}' for r, t in turns) + '<|assistant|>'

    def check(self, key, prompt):
        ids, ends = self.cache.tokenize(key, prompt)
        enc = s.TOK(prompt, return_offsets_mapping=True)
        self.assertEqual(ids, list(enc['input_ids']))
        self.assertEqual(list(ends), [e for _, e in enc['offset_mapping']])
        return ids

    def test_growing_conversation_reuses_and_matches(self):
        os.environ['HIVE_TOKEN_CACHE_VERIFY'] = '0'
        turns = []
        for k in range(12):
            turns += [('user', f'question {k}: the engine streams experts \uc11c\ubc84 {k * 7}'), ('assistant', f'answer {k}: def f(x): return x*{k}')]
            self.check('c1', self.render(turns + [('user', 'next')]))
        self.assertGreaterEqual(self.cache.hits, 10)  # every turn after the first reused its prefix

    def test_edited_middle_and_unrelated_sessions_match(self):
        os.environ['HIVE_TOKEN_CACHE_VERIFY'] = '0'
        base = [('user', 'a ' * 30), ('assistant', 'b ' * 30), ('user', 'c ' * 30), ('assistant', 'd ' * 30)]
        self.check('c2', self.render(base + [('user', 'q')]))
        edited = [base[0], ('assistant', 'B CHANGED ' * 5)] + base[2:]
        self.check('c2', self.render(edited + [('user', 'q')]))          # divergence inside an earlier message
        self.check('c2', self.render(edited[:1] + [('user', 'q2')]))      # shorter prompt (rewritten history)
        self.check('c3', self.render(base))                                # another session: whole tokenization
        self.check('c2', 'no role markers at all ' * 20)                   # nothing to cut at: whole tokenization

    def test_new_key_reuses_a_kept_prompt_by_content(self):
        """2026-10-08: a conversation under a new key (a client began sending a key, a window moved) shares its prompt with a kept
        one — the longest common prefix is taken, and the ids still equal a whole tokenization."""
        os.environ['HIVE_TOKEN_CACHE_VERIFY'] = '0'
        long_sys = [('user', 'the engine streams experts over PCIe ' * 200), ('assistant', 'ok')]
        self.check('old-key', self.render(long_sys + [('user', 'first')]))
        h0 = self.cache.hits
        self.check('new-key', self.render(long_sys + [('user', 'first'), ('assistant', 'answer'), ('user', 'second')]))
        self.assertEqual(self.cache.hits, h0 + 1)
        h1 = self.cache.hits
        self.check('short-key', self.render([('user', 'unrelated and short')]))   # below the 4,096-character floor: whole
        self.assertEqual(self.cache.hits, h1)

    def test_short_conversations_do_not_push_out_long_ones(self):
        """2026-10-28: kept by tokens, not by count — 200 short conversations after a long one leave the long one in the cache (its next
        turn reuses the prefix); the token budget evicts the least recently used once exceeded; the kept-token count stays exact."""
        os.environ['HIVE_TOKEN_CACHE_VERIFY'] = '0'
        long_turns = [('user', 'the engine streams experts over PCIe ' * 300), ('assistant', 'ok')]
        n_long = len(self.check('long', self.render(long_turns + [('user', 'q1')])))
        for k in range(200):
            self.check(f'short{k}', self.render([('user', f'caption {k}')]))
        self.assertIn('long', self.cache.entries)
        h0 = self.cache.hits
        self.check('long', self.render(long_turns + [('user', 'q1'), ('assistant', 'a1'), ('user', 'q2')]))
        self.assertEqual(self.cache.hits, h0 + 1)
        self.assertEqual(self.cache.kept_tokens, sum(len(e[1]) for e in self.cache.entries.values()))
        # budget: just above the long conversation — the next long one evicts it (least recently used first), short ones go first
        os.environ['HIVE_TOKEN_CACHE_TOKENS'] = str(n_long + 600)
        self.check('long2', self.render([('user', 'another talk ' * 100)]))
        self.assertNotIn('long', self.cache.entries)
        self.assertIn('long2', self.cache.entries)
        self.assertLessEqual(self.cache.kept_tokens, n_long + 600)
        self.assertEqual(self.cache.kept_tokens, sum(len(e[1]) for e in self.cache.entries.values()))
        # a single conversation larger than the budget is still kept (one entry always stays)
        os.environ['HIVE_TOKEN_CACHE_TOKENS'] = '10'
        self.check('long3', self.render(long_turns + [('user', 'q3')]))
        self.assertEqual(list(self.cache.entries), ['long3'])
        # HIVE_TOKEN_CACHE_SESSIONS set: a count cap again
        os.environ.pop('HIVE_TOKEN_CACHE_TOKENS')
        os.environ['HIVE_TOKEN_CACHE_SESSIONS'] = '3'
        for k in range(5):
            self.check(f'c{k}', self.render([('user', f'short {k}')]))
        self.assertEqual(list(self.cache.entries), ['c2', 'c3', 'c4'])
        self.assertEqual(s.TokenCache.limits(), (3, 32_000_000))

    def test_divergence_record(self):
        """last_diverge: the previous prompt's length and how many of its tokens this prompt keeps (tools/reread_report.py) — only
        for the conversation's own previous prompt, not for a prompt borrowed by content under another key."""
        os.environ['HIVE_TOKEN_CACHE_VERIFY'] = '0'
        base = [('user', 'a ' * 30), ('assistant', 'b ' * 30), ('user', 'c ' * 30), ('assistant', 'd ' * 30)]
        first = self.check('d1', self.render(base + [('user', 'q')]))
        self.assertNotIn('d1', self.cache.last_diverge)                       # a first prompt: nothing to compare
        grown = self.render(base + [('user', 'q'), ('assistant', 'r'), ('user', 'q2')])
        self.check('d1', grown)
        prev, at = self.cache.last_diverge.pop('d1')
        self.assertEqual(prev, len(first))
        self.assertGreaterEqual(at, len(first) - 2)                            # an append keeps (nearly) the whole previous prompt
        edited = self.render([base[0], ('assistant', 'B CHANGED ' * 5)] + base[2:] + [('user', 'q')])
        self.check('d1', grown)                                                # re-seed the kept prompt
        self.cache.last_diverge.pop('d1')
        self.check('d1', edited)
        prev, at = self.cache.last_diverge.pop('d1')
        whole = s.TOK(edited, return_offsets_mapping=True)['input_ids']
        self.assertLess(at, prev // 2)                                         # rewritten early history: diverges in the second message
        self.assertEqual(list(whole[:at]), list(s.TOK(grown)['input_ids'][:at]))  # the shared tokens really are shared
        long_sys = [('user', 'the engine streams experts over PCIe ' * 200), ('assistant', 'ok')]
        self.check('d2', self.render(long_sys + [('user', 'first')]))
        self.check('d3', self.render(long_sys + [('user', 'other')]))         # new key served by content: not its own history
        self.assertNotIn('d3', self.cache.last_diverge)

    def test_off_switch_and_lstrip_markers(self):
        os.environ['HIVE_TOKEN_CACHE'] = '0'
        p = self.render([('user', 'x'), ('assistant', 'y')])
        self.check('c4', p); self.check('c4', p + '<|user|>z')
        self.assertEqual(self.cache.hits, 0)
        os.environ.pop('HIVE_TOKEN_CACHE')
        from tokenizers import AddedToken
        s.TOK.add_special_tokens({'additional_special_tokens': [AddedToken('<|tool|>', lstrip=True)]})
        self.cache._bind(s.TOK)
        self.assertNotIn(s.TOK.convert_tokens_to_ids('<|tool|>'), self.cache.cut_ids)
        self.assertIn(s.TOK.convert_tokens_to_ids('<|user|>'), self.cache.cut_ids)

    def test_verify_mismatch_turns_the_cache_off(self):
        os.environ['HIVE_TOKEN_CACHE_VERIFY'] = '1'
        logged = []
        import threading
        real_thread = threading.Thread
        class Inline:
            def __init__(self, target, daemon=None): self.target = target
            def start(self): self.target()
        threading.Thread = Inline
        try:
            p = self.render([('user', 'x'), ('assistant', 'y')])
            self.check('c5', p)
            self.check('c5', p + '<|user|>z')                 # verified inline: ok
            self.assertFalse(self.cache.disabled)
            real_tok = s.TOK
            class Wrong:  # whole tokenization disagrees with the cached ids
                def __call__(self, text, **kw):
                    out = dict(real_tok(text, **kw)); out['input_ids'] = list(out['input_ids'])[:-1]; return out
                def __getattr__(self, n): return getattr(real_tok, n)
            self.cache.tok = s.TOK = Wrong()
            self.cache._verify_later(p, [1, 2, 3])
            self.assertTrue(self.cache.disabled)
        finally:
            threading.Thread = real_thread


class AnnouncedToolsTests(unittest.TestCase):
    """Tools announced in a later system message: DeepSeek keeps them in place, the adapter families merge them into the head."""

    def _run(self, family):
        seen = {}

        class Enc:
            def encode_messages(self, messages, **kw):
                seen['messages'] = messages
                return 'p', {}

        old = (s.FAMILY, s.ENC)
        s.FAMILY, s.ENC = family, Enc()
        try:
            tool = lambda n: {'type': 'function', 'function': {'name': n}}
            msgs = [{'role': 'system', 'content': 'head'}, {'role': 'user', 'content': 'q'},
                    {'role': 'system', 'content': 'new', 'tools': [tool('later'), tool('a')]}, {'role': 'user', 'content': 'q2'}]
            ORIG_ENCODE_PROMPT(msgs, [tool('a')], 'chat', None)
            return seen['messages'], msgs
        finally:
            s.FAMILY, s.ENC = old

    def test_deepseek_keeps_announced_tools_in_place(self):
        out, given = self._run('deepseek')
        self.assertEqual([t['function']['name'] for t in out[0]['tools']], ['a'])
        self.assertEqual([t['function']['name'] for t in out[2]['tools']], ['later', 'a'])

    def test_adapter_family_merges_them_into_the_head(self):
        family = next(iter(s.ADAPTER_FAMILIES))
        out, given = self._run(family)
        self.assertEqual([t['function']['name'] for t in out[0]['tools']], ['a', 'later'])
        self.assertNotIn('tools', out[2])
        self.assertIn('tools', given[2])   # the caller's messages are not mutated


# ---------------------------------------------------------------------------------------------------------- HIVE_HINT_PARALLEL
# boundary_hints as it was before the renders were listed first and handed to the render pool (main ec451e3, verbatim) — the
#   reference the pooled and the serial path are compared with (the leading-block rule follows the server's: several leading user blocks, 2026-10-27)
OLD_BOUNDARY_HINTS = r'''
def boundary_hints(messages: list[dict], tools, thinking_mode: str, effort, prompt: str, ids: list[int], ends=None,
                   gen_cut: bool = False) -> list[int]:
    """H5 (HIVE_PREFIX_SHARE): **exact token offsets** in the encoded prompt — end of the system/tool block, end of a leading
    user message followed by another user message, and ends of completed assistant turns.
    Makes no assumption about the reference template's internals: for each boundary, render "the messages before it + an
    empty user message" with the same encode_prompt, measure the common-prefix length c against the full prompt string
    (past that point the next message's content / generation prompt diverges), tokenize the full prompt once more with an
    offset mapping, and emit the number of tokens that end at or before c. If the offset tokenization's ids differ from
    the ids actually sent, emit nothing.
    Results are always prefix positions of ids (a wrong boundary only loses a reuse opportunity — the daemon compares the
    token prefix itself). Image requests emit nothing, since text offsets do not match the placeholder expansion.
    Cost: two template renders per boundary (<= 1 + HINT_TURNS) + string compares, plus one extra full-prompt tokenization
    unless the caller passes the token end offsets of ids (`ends`, from TokenCache) — only when enabled. With HIVE_HINT_CACHE
    (HintProbeCache) a boundary whose messages were probed on an earlier request costs a compare instead of its renders."""
    if ends is None or len(ends) != len(ids):
        try:
            enc = TOK(prompt, return_offsets_mapping=True)
            if list(enc["input_ids"]) != list(ids):
                return []
            ends = [int(e) for _, e in enc["offset_mapping"]]
        except Exception:
            return []
    import itertools, operator
    mono = _ends_monotonic(ends) if HintProbeCache.enabled() else None  # HIVE_HINT_CACHE: the same check vectorized (~3 ms at 150K tokens)
    if mono is None:
        mono = all(map(operator.le, ends, itertools.islice(ends, 1, None)))
    if not mono:
        return []  # non-monotonic mapping (unknown tokenizer) — no boundaries
    import bisect
    probes = []
    n = 0
    while n < len(messages) and messages[n].get("role") in ("system", "developer"):
        n += 1
    if n or tools:
        probes.append(messages[:n])
    # A client's fixed context sent as the first user message, ahead of the conversation (measured 2026-10-14: a chat client put
    #   ~2,000 tokens of per-account instructions there, so its new conversations shared 8,879 leading tokens, but the only
    #   boundary sat at the end of the system/tools block (6,922) and every new conversation prefilled the rest again). Only when the
    #   next message is another user message — that is the shape of a fixed block; an ordinary first question is followed by its answer.
    #   A client may put a per-room notice (system/developer messages) between that block and the conversation (measured 2026-10-19, a
    #   chat client: [system + tools] [user: per-person block] [system: this room's client-tool notice] [user: ...] — the boundary at the end
    #   of the person's block was missing, so rooms of the same person shared only the system/tools block, 15.8K tokens). Then both the
    #   block's end (every room of that person) and the notice's end (that person's rooms with the same notice) are boundaries.
    #   The leading run may hold several such blocks, each followed by another user message (measured 2026-10-27, an analysis client:
    #   [system] [user: a fixed one-line note] [user: the owner's index, ~24K] [user: the item] — the note keeps one conversation key per
    #   owner, but only its end (4,576) was a boundary, so every item re-read the owner's index: prefill median 3.9 → 6.8 s, 58 requests
    #   in 16 hours). Every block of the run ends a boundary, up to HINT_LEAD_BLOCKS; the run's last user message is the question.
    u = n
    for _ in range(HINT_LEAD_BLOCKS):
        if not (u < len(messages) - 1 and messages[u].get("role") == "user"):
            break
        m = u + 1
        while m < len(messages) and messages[m].get("role") in ("system", "developer"):
            m += 1
        if not (m < len(messages) and messages[m].get("role") == "user"):
            break
        probes.append(messages[: u + 1])
        if m > u + 1:
            probes.append(messages[:m])
        u = m
    turn_ends = [k for k in range(len(messages) - 1) if messages[k].get("role") == "assistant"]
    for k in turn_ends[-HINT_TURNS:]:
        probes.append(messages[: k + 1])
    out = set()
    if gen_cut and messages:
        # Where the next turn's prompt leaves this one: the generation prompt ends in a reasoning-start token (DeepSeek
        #   "<｜Assistant｜><think>", GLM "<|assistant|><think>"), while the next turn renders this answer without its reasoning
        #   ("...</think>answer"). A checkpoint at the prompt end therefore never matches the next turn — measured 2026-10-08: the
        #   previous prompt ended "<｜Assistant｜><think>" and the next one had "<｜Assistant｜></think>a1" there, so every follow-up turn
        #   with reasoning on re-read the whole conversation. Rendering this conversation plus a finished assistant turn finds the
        #   point without knowing the template (llama.cpp checkpoints a few tokens before the prompt end for the same reason).
        try:
            # the next turn = this answer followed by a user message (the DeepSeek template keeps the reasoning markers of a final
            #   assistant message and drops them only once a user message follows — a probe ending in the answer misses the cut)
            done, media = encode_prompt(list(messages) + [{"role": "assistant", "content": "\x00"}, {"role": "user", "content": "\x02"}],
                                        tools, thinking_mode, effort)
            other, _ = encode_prompt(list(messages) + [{"role": "assistant", "content": "\x01"}, {"role": "user", "content": "\x02"}],
                                     tools, thinking_mode, effort)
        except Exception:
            done, media = None, None
        if done is not None and not media:
            c = min(_common_prefix_len(done, prompt), _common_prefix_len(done, other))
            t = bisect.bisect_right(ends, c)
            # The whole prompt is a text prefix of the next turn (Qwen: the prompt ends "<think>\n", the next turn has "<think>\n\n</think>"
            #   there) — but not a token prefix: the last "\n" merges with the next one into a "\n\n" token. Measured 2026-10-14, 27B: the
            #   next turn diverged one token before the prompt end, so neither the prompt checkpoint nor any snapshot matched and every
            #   follow-up turn after a long one re-read it all (62K tokens, 15.4 s). Cut one token earlier — only when the next turn
            #   really tokenizes that place differently (a few dozen characters re-tokenized from the start of the last token).
            if t == len(ids) and c >= len(prompt) and len(ids) >= 2:
                try:
                    s0 = ends[-2]
                    enc = TOK(done[s0:s0 + 64], return_offsets_mapping=True)
                    first = next((i for i, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]) if b > a), None)  # skip a zero-width BOS
                    if first != ids[-1]:
                        t -= 1
                except Exception:
                    pass
            if 0 < t < len(ids) and len(ids) - t <= GEN_TAIL_MAX:
                out.add(t)
        # Where a request that adds a message without this answer leaves this one: the end of the last message, before the generation
        #   prompt. Measured 2026-10-18/19 (DeepSeek, chat mode): a background job sent each item twice — the second prompt was the first
        #   one up to "<｜Assistant｜></think>" plus ~52 tokens of a further user message. The prompt-end checkpoint sat 2 tokens past the
        #   divergence and a rewind of the live state is impossible (window ring, compressor state), so 54 second requests re-read
        #   23-44K tokens each (1.63M tokens over three days, 6-10 s apiece). The daemon cuts every generation-suffix hint (each piece
        #   after one is about one decode step).
        try:
            more, media2 = encode_prompt(list(messages) + [{"role": "user", "content": "\x02"}], tools, thinking_mode, effort)
        except Exception:
            more, media2 = None, None
        if more is not None and not media2:
            c = _common_prefix_len(more, prompt)
            t = bisect.bisect_right(ends, c)
            # The last token before that point can merge with what the further message brings (measured 2026-10-19, DeepSeek: a second
            #   user message is joined to the first with "\n\n", the first one's closing "." merged with it, and the request parted one
            #   token before the hint) — cut one token earlier when the further rendering tokenizes that place differently.
            if 2 <= t < len(ids):
                try:
                    s0 = ends[t - 2]
                    enc = TOK(more[s0:s0 + 64], return_offsets_mapping=True)
                    first = next((i for i, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]) if b > a), None)  # skip a zero-width BOS
                    if first != ids[t - 1]:
                        t -= 1
                except Exception:
                    pass
            if 0 < t < len(ids) and len(ids) - t <= GEN_TAIL_MAX:
                out.add(t)
    # HIVE_HINT_CACHE: probe renders seen on an earlier request are reused (HintProbeCache). Off (or inputs that cannot be
    #   snapshotted) = the loop below without a cache, unchanged.
    cache = HINT_CACHE if probes and HintProbeCache.enabled() and HINT_CACHE.usable() else None
    if cache is not None:
        try:
            fmsgs = [_freeze(m) for m in messages[: max(len(p) for p in probes)]]
            ftools, fmode, feff = _freeze(tools), _freeze(thinking_mode), _freeze(effort)
        except Exception:  # noqa: BLE001 — not JSON-like (or too deep): this request renders every probe
            cache = None
    for pre in probes:
        if cache is not None:
            fpre = tuple(fmsgs[: len(pre)])
            key = (len(pre), fmode, feff, hash(fpre[-1]) if fpre else None)
            hit, kept = cache.get(key, fpre, ftools)
            if hit:
                if kept is None:  # that probe rendered media: no boundary
                    continue
                # kept = text[:c_other], so its common prefix with the prompt is min(cpl(text, prompt), cpl(text, other)) — the
                #   same cap at the next message's content as below (the "<info-msg>" fix, 0ab95c3)
                t = bisect.bisect_right(ends, _common_prefix_len(kept, prompt))
                if 0 < t < len(ids):
                    out.add(t)
                continue
            try:
                text, media = encode_prompt(list(pre) + [{"role": "user", "content": ""}], tools, thinking_mode, effort)
                other, _ = encode_prompt(list(pre) + [{"role": "user", "content": "\x00"}], tools, thinking_mode, effort)
            except Exception:
                continue
            if media:
                cache.put(key, fpre, ftools, None)
                continue
            c_other = _common_prefix_len(text, other)
            cache.put(key, fpre, ftools, text[:c_other])
            c = min(_common_prefix_len(text, prompt), c_other)
            t = bisect.bisect_right(ends, c)
            if 0 < t < len(ids):
                out.add(t)
            continue
        try:
            text, media = encode_prompt(list(pre) + [{"role": "user", "content": ""}], tools, thinking_mode, effort)
            # Where the next message's own content starts: the same probe with a different content diverges exactly there.
            #   Without this cap the boundary ran into the content whenever it began with a character the template's next text
            #   also begins with — a per-turn info block "<info-msg>" placed before the user's message against the empty
            #   probe's "<｜Assistant｜>" shares "<", so the boundary took that "<" token. The next turn has the user's words
            #   there, never matched the snapshot, and re-read the whole conversation (measured 2026-10-08: "reuse 0 via reset ·
            #   common with live 24731" with the snapshot at 24732, every follow-up turn cached 0).
            other, _ = encode_prompt(list(pre) + [{"role": "user", "content": "\x00"}], tools, thinking_mode, effort)
        except Exception:
            continue
        if media:
            continue
        c = min(_common_prefix_len(text, prompt), _common_prefix_len(text, other))
        t = bisect.bisect_right(ends, c)  # number of leading tokens ending at or before c (a prefix, since offsets are monotonic)
        if 0 < t < len(ids):
            out.add(t)
    return sorted(out)


'''

# A checkpoint for load_modules: an encoder in the style of the reference one (reasoning only after the last user message, tools in
#   system messages, tool calls, mid-conversation system messages, a generation prompt after a user / tool / system message) and a
#   small BPE tokenizer with the role markers as special tokens. Both written by this test.
POOL_ENCODING = r'''
import copy, json

def encode_messages(messages, thinking_mode="chat", return_multi_modal_data=False, reasoning_effort=None):
    msgs = copy.deepcopy(messages)
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
    out = ["<|bos|>"]
    if thinking_mode == "thinking" and reasoning_effort is not None:
        out.append("[effort %s]" % (reasoning_effort,))
    for i, m in enumerate(msgs):
        r = m.get("role")
        if r in ("system", "developer"):
            out.append("<|system|>" + str(m.get("content") or ""))
            if m.get("tools"):
                out.append("<|tools|>" + json.dumps(m["tools"], sort_keys=True))
        elif r == "user":
            out.append("<|user|>" + str(m.get("content") or ""))
        elif r == "assistant":
            out.append("<|assistant|>")
            if thinking_mode == "thinking":
                out.append("(think" + m["reasoning_content"] + ")" if i > last_user and m.get("reasoning_content") else ")")
            out.append(str(m.get("content") or ""))
            for c in m.get("tool_calls") or []:
                a = c["function"]["arguments"]
                out.append("<|call|>" + c["function"]["name"] + ":" + (a if isinstance(a, str) else json.dumps(a)))
            out.append("<|eos|>")
        elif r == "tool":
            out.append("<|tool|>" + str(m.get("content") or ""))
    if msgs and msgs[-1].get("role") in ("user", "tool", "system", "developer"):
        out.append("<|assistant|>" + ("(think" if thinking_mode == "thinking" else ""))
    text = "".join(out)
    return (text, {}) if return_multi_modal_data else text
'''
POOL_SPECIAL = ['<|bos|>', '<|system|>', '<|tools|>', '<|user|>', '<|assistant|>', '<|call|>', '<|eos|>', '<|tool|>']
POOL_WORDS = 'the of and to in is that for it as with < > <info> </info> ( ) . , ! ? : 1 2 3 x y z think go read path / .. " { }'.split(' ') + ['\n', '\n\n', ' ']


def pool_ckpt(root):
    """A checkpoint directory load_modules can read: config.json, encoding/encoding.py, a fast tokenizer."""
    from tokenizers import Tokenizer as RTok, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    import random
    os.makedirs(os.path.join(root, 'encoding'))
    with open(os.path.join(root, 'config.json'), 'w') as f: json.dump({}, f)
    with open(os.path.join(root, 'encoding', 'encoding.py'), 'w') as f: f.write(POOL_ENCODING)
    rnd = random.Random(3)
    corpus = [''.join(rnd.choice(POOL_WORDS) for _ in range(60)) for _ in range(3000)]
    tok = RTok(models.BPE(unk_token=None, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    from tokenizers import decoders
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=POOL_SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    PreTrainedTokenizerFast(tokenizer_object=tok, bos_token='<|bos|>', eos_token='<|eos|>').save_pretrained(root)
    return root


def pool_conversation(rnd, scale=1):
    """A random conversation for the pool comparison (tools, tool results, fixed leading block, notices, reasoning, long turns)."""
    W = POOL_WORDS
    def txt(a, b): return ''.join(rnd.choice(W) for _ in range(rnd.randint(a, b)))
    tools = None if rnd.random() < 0.3 else [{'type': 'function', 'function': {'name': f't{i}', 'description': txt(2, 10),
                                              'parameters': {'type': 'object', 'properties': {'q': {'type': 'string'}}}}} for i in range(rnd.randint(1, 3))]
    m = []
    if rnd.random() < 0.8: m.append({'role': 'system', 'content': txt(3, 40 * scale)})
    if rnd.random() < 0.3: m.append({'role': 'user', 'content': txt(3, 30)})
    if rnd.random() < 0.3: m.append({'role': rnd.choice(['system', 'developer']), 'content': txt(1, 10)})
    for k in range(rnd.randint(1, 7)):
        m.append({'role': 'user', 'content': txt(0, 30 * scale)})
        if rnd.random() < 0.2: m.append({'role': 'system', 'content': txt(1, 8)})
        a = {'role': 'assistant', 'content': txt(0, 30 * scale)}
        if rnd.random() < 0.5: a['reasoning_content'] = txt(1, 20)
        if tools and rnd.random() < 0.4:
            a['tool_calls'] = [{'id': f'c{k}', 'type': 'function', 'function': {'name': tools[0]['function']['name'],
                                'arguments': rnd.choice([json.dumps({'q': txt(1, 5)}), {'q': txt(1, 5)}])}}]
            m.append(a); m.append({'role': 'tool', 'tool_call_id': f'c{k}', 'content': txt(1, 40 * scale)})
            m.append({'role': 'assistant', 'content': txt(1, 20)})
        else:
            m.append(a)
    last = rnd.random()
    if last < 0.8: m.append({'role': 'user', 'content': txt(0, 20)})
    elif last < 0.9: m.append({'role': 'tool', 'tool_call_id': 'x', 'content': txt(0, 20)})
    return m, tools, rnd.choice(['thinking', 'chat']), rnd.choice([None, 'high', 50, 50.0, True])


class HintPoolTests(unittest.TestCase):
    """HIVE_HINT_PARALLEL: the hints with the renders in worker processes == the serial renders == boundary_hints before the change
    (verbatim), on random conversations, with the probe cache off, filling and hit; the request-path halves; a dead worker; the
    dropped batch; and negative controls (mutants of the new code must be caught)."""
    N = int(os.environ.get('HIVE_HINT_POOL_CASES') or 400)
    GLOBALS = ('TOK', 'ENC', 'IMG', 'FAMILY', 'STOP_IDS', 'EOS_TEXTS', 'DSML_START', 'LOADED_CKPT', 'LOADED_BIND', 'HINT_CACHE', 'RENDER_POOL')

    @classmethod
    def setUpClass(cls):
        cls.saved = {k: getattr(s, k) for k in cls.GLOBALS}
        cls.saved_encode = s.encode_prompt
        cls.saved_path, cls.saved_enc_mod = list(sys.path), sys.modules.pop('encoding', None)
        cls.env = {k: os.environ.pop(k, None) for k in ('HIVE_HINT_CACHE', 'HIVE_HINT_PARALLEL')}
        cls.td = tempfile.TemporaryDirectory()
        # the worker functions pickle by module name: as a script the server is "__main__"; loaded from its path here, its name must be
        #   importable for pickle (the workers load it from its path under that name)
        cls.had_mod = sys.modules.get(s.__name__)
        sys.modules[s.__name__] = s
        s.encode_prompt = ORIG_ENCODE_PROMPT
        s.load_modules(pool_ckpt(cls.td.name))
        s.RENDER_POOL = s.RenderPool()
        ns = dict(vars(s))
        exec(OLD_BOUNDARY_HINTS, ns)
        cls.old_ns = ns
        import random
        rnd = random.Random(11)
        cls.cases = []
        while len(cls.cases) < cls.N:   # conversations of 1-4 turns, each turn a case: later turns meet the probe cache entries of earlier ones
            m, tools, mode, eff = pool_conversation(rnd, 1 if len(cls.cases) % 8 else 30)
            for turn in range(rnd.randint(1, 4)):
                prompt, media = s.encode_prompt(m, tools, mode, eff)
                enc = s.TOK(prompt, return_offsets_mapping=True)
                cls.cases.append((list(m), tools, mode, eff, prompt, list(enc['input_ids']), array('q', [e for _, e in enc['offset_mapping']]), rnd.random() < 0.8))
                more, *_ = pool_conversation(rnd)
                nxt = [x for x in more if x['role'] != 'system'][-4:]
                ends_at = [k for k, x in enumerate(m) if x['role'] == 'assistant']
                if ends_at and rnd.random() < 0.3:   # an edited history: cut after an earlier answer and ask something else
                    m = m[: rnd.choice(ends_at) + 1] + [{'role': 'user', 'content': 'instead %d' % turn}]
                else:
                    m = m + [{'role': 'assistant', 'content': 'answer %d' % turn, 'reasoning_content': 'r'}] + (
                        nxt if nxt and nxt[-1]['role'] == 'user' else [{'role': 'user', 'content': 'next'}])

    @classmethod
    def tearDownClass(cls):
        if s.RENDER_POOL.ex is not None:
            s.RENDER_POOL.ex.shutdown(wait=True, cancel_futures=True)
        for k, v in cls.saved.items(): setattr(s, k, v)
        if cls.had_mod is None: sys.modules.pop(s.__name__, None)
        s.encode_prompt = cls.saved_encode
        sys.path[:] = cls.saved_path
        sys.modules.pop('encoding', None)
        if cls.saved_enc_mod is not None: sys.modules['encoding'] = cls.saved_enc_mod
        for k, v in cls.env.items():
            os.environ.pop(k, None)
            if v is not None: os.environ[k] = v
        cls.td.cleanup()

    def setUp(self):
        os.environ.pop('HIVE_HINT_CACHE', None); os.environ.pop('HIVE_HINT_PARALLEL', None)

    def old(self, c):
        m, tools, mode, eff, prompt, ids, ends, gc = c
        os.environ['HIVE_HINT_CACHE'] = '0'
        try:
            return self.old_ns['boundary_hints'](m, tools, mode, eff, prompt, ids, ends, gen_cut=gc)
        finally:
            os.environ.pop('HIVE_HINT_CACHE')

    def new(self, c, cache=True, workers='0', halves=False):
        m, tools, mode, eff, prompt, ids, ends, gc = c
        os.environ['HIVE_HINT_PARALLEL'] = workers
        if not cache: os.environ['HIVE_HINT_CACHE'] = '0'
        try:
            if halves:   # the request path: start before the tokenization, finish after it
                st = s.boundary_hints_start(m, tools, mode, eff, prompt, gen_cut=gc)
                s.TOK(prompt, return_offsets_mapping=True)
                return s.boundary_hints_finish(st, ids, ends)
            return s.boundary_hints(m, tools, mode, eff, prompt, ids, ends, gen_cut=gc)
        finally:
            os.environ.pop('HIVE_HINT_PARALLEL', None); os.environ.pop('HIVE_HINT_CACHE', None)

    def mismatches(self, limit=None):
        """Cases where any path differs from the old function (stops after `limit` mismatches)."""
        bad = []
        s.HINT_CACHE = s.HintProbeCache()
        for i, c in enumerate(self.cases):
            want = self.old(c)
            got = [self.new(c, cache=False), self.new(c), self.new(c),                     # serial: no cache, cache filling, cache hit
                   self.new(c, cache=False, workers='1'), self.new(c, workers='1', halves=True)]
            if any(g != want for g in got):
                bad.append((i, want, got))
                if limit and len(bad) >= limit: break
        return bad

    def test_pool_serial_and_old_agree(self):
        bad = self.mismatches()
        self.assertEqual(bad[:3], [])
        print(f'hint pool: {len(self.cases)} cases × 5 paths == the old function · {sum(1 for c in self.cases if self.old(c))} with hints · '
              f'{s.RENDER_POOL.batches} pooled batches')
        self.assertGreater(s.RENDER_POOL.batches, self.N // 2)
        self.assertFalse(s.RENDER_POOL.failed)
        self.assertGreater(sum(1 for c in self.cases if self.old(c)), self.N * 3 // 4)   # most cases do have hints

    def test_off_switch_renders_on_the_thread(self):
        b0 = s.RENDER_POOL.batches
        for v in ('', '0'):
            os.environ['HIVE_HINT_PARALLEL'] = v
            self.assertFalse(s.RENDER_POOL.usable())
            c = self.cases[1]
            self.assertEqual(s.boundary_hints(*c[:7], gen_cut=c[7]), self.old(c))
        os.environ.pop('HIVE_HINT_PARALLEL')
        self.assertEqual(s.RENDER_POOL.batches, b0)

    def test_swapped_encoder_is_not_pooled(self):
        b0 = s.RENDER_POOL.batches
        s.encode_prompt = lambda *a: ORIG_ENCODE_PROMPT(*a)   # a test or family swap: the workers would not render what this thread renders
        try:
            c = self.cases[2]
            self.assertEqual(self.new(c, workers='1'), self.old(c))
            self.assertEqual(s.RENDER_POOL.batches, b0)
        finally:
            s.encode_prompt = ORIG_ENCODE_PROMPT

    def test_dropped_batch_waits_and_closes(self):
        c = max(self.cases, key=lambda c: len(c[4]))
        self.assertEqual(self.new(c, workers='1'), self.old(c))   # the pool (and its pipes) up before counting
        fds = len(os.listdir('/proc/self/fd'))
        os.environ['HIVE_HINT_PARALLEL'] = '1'
        try:
            for _ in range(5):
                st = s.boundary_hints_start(c[0], c[1], c[2], c[3], c[4], gen_cut=True)
                self.assertTrue(st[-1][6], 'renders were handed to the pool')
                self.assertEqual(s.boundary_hints_finish(st, c[5][:-1]), [])   # ids that do not match the prompt: no hints
        finally:
            os.environ.pop('HIVE_HINT_PARALLEL')
        self.assertEqual(len(os.listdir('/proc/self/fd')), fds)

    def test_zy_dead_worker_is_absorbed(self):
        """Last (it turns the pool off for the process): a killed worker → that batch renders on the thread, same hints, one log line."""
        import signal
        c = self.cases[3]
        self.assertEqual(self.new(c, workers='1'), self.old(c))   # pool up
        ex = s.RENDER_POOL.ex
        for pid in list(ex._processes): os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        for c in self.cases[:20]:
            self.assertEqual(self.new(c, workers='1'), self.old(c))
        self.assertTrue(s.RENDER_POOL.failed)
        s.RENDER_POOL = s.RenderPool()   # a fresh pool for any test that runs after this one

    def test_zz_negative_controls(self):
        """Mutants of the new code (comparison helpers, the job list, the split halves) must make the comparison fail."""
        import inspect
        mutants = [
            ('_probe_cpl', 'return min(a[1], b[1])', 'return min(a[1], b[1]) - 1'),
            ('_probe_cpl', 'return a[1] + _common_prefix_len(a[2], b[2])', 'return a[1]'),
            ('_probe_slice', 'return prompt[i:c] + tail[:j - c]', 'return prompt[i:c] + tail[:j - c - 1]'),
            ('_probe_result', 'return bool(media), c, text[c:]', 'return bool(media), c, text[c + 1:]'),
            ('boundary_hints_start', 'jobs.append((len(pre), [{"role": "user", "content": ""}]))',
             'jobs.append((len(pre) - 1, [{"role": "user", "content": ""}]))'),
            ('boundary_hints_start', 'jobs.append((len(messages), [{"role": "user", "content": "\\x02"}]))',
             'jobs.append((len(messages) - 1, [{"role": "user", "content": "\\x02"}]))'),
            ('boundary_hints_finish', 'r_more = rendered[j_done + 2]', 'r_more = rendered[j_done]'),
            ('boundary_hints_finish', 'c = min(r_text[1], c_other)', 'c = r_text[1]'),
            ('boundary_hints_finish', 'if r_text[0] and not media_ok:', 'if not r_text[0] and not media_ok:'),
            ('boundary_hints_finish', 't = bisect.bisect_right(ends, _common_prefix_len(kept, prompt))',
             't = bisect.bisect_right(ends, _common_prefix_len(kept, prompt) - 1)'),
        ]
        caught = []
        for name, a, b in mutants:
            orig = getattr(s, name)
            src = inspect.getsource(orig)
            self.assertEqual(src.count(a), 1, (name, a))
            ns = vars(s)
            exec(src.replace(a, b), ns)
            try:
                bad = self.mismatches(limit=1)
            finally:
                ns[name] = orig
            caught.append(bool(bad))
            self.assertTrue(bad, f'mutant not caught: {name}: {a!r} -> {b!r}')
        print(f'hint pool: {len(caught)} mutants, {sum(caught)} caught')


class DivergeRecordTests(unittest.IsolatedAsyncioTestCase):
    """MessageTrail: request records name the first changed message (index, roles, counts, prefix hashes — no content), follow a
    conversation across a session-key change by its hashed conversation key, and give image requests prev_prompt_tokens / diverge_at
    in the daemon's ids; diverge_hint reuses the computed hints. Mutants of the new code must be caught."""
    SYS = 'system prompt ' * 10

    def setUp(self):
        self.saved = (s.TOK, s.ENC, s.DAEMON, s.log_request, s.encode_prompt, s.tokenize_with_images, s.MSGTRAIL, s.TOKCACHE)
        s.TOK = HintTokenizer(); s.ENC = Encoder(); s.INFLIGHT.clear(); s.DAEMON = RecDaemon()
        s.encode_prompt = hint_encode
        s.tokenize_with_images = lambda prompt, media, key=None: (s.TOK.encode(prompt), [], b'')
        s.MSGTRAIL = s.MessageTrail(); s.TOKCACHE = s.TokenCache()
        self.recs = []; s.log_request = lambda rec: self.recs.append(dict(rec))
        self.env = {k: os.environ.pop(k, None) for k in ('HIVE_PREFIX_SHARE',)}

    def tearDown(self):
        s.TOK, s.ENC, s.DAEMON, s.log_request, s.encode_prompt, s.tokenize_with_images, s.MSGTRAIL, s.TOKCACHE = self.saved
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None: os.environ[k] = v

    def conv(self, *turns):
        m = [{'role': 'system', 'content': self.SYS}, {'role': 'user', 'content': 'first question'}]
        for k, (a, u) in enumerate(turns):
            m += [{'role': 'assistant', 'content': a}, {'role': 'user', 'content': u}]
        return m

    async def ask(self, msgs, **extra):
        await s.chat(Request({'messages': msgs, 'reasoning_effort': 'none', **extra}))
        return self.recs[-1]

    async def scenario(self):
        """The checks, as a list of failures (the mutants below must produce at least one)."""
        bad = []
        def want(cond, what):
            if not cond: bad.append(what)
        r1 = await self.ask(self.conv(), prompt_cache_key='room-1')
        want('msg_diverge' not in r1 and r1.get('head') and r1.get('conv'), 'first request: identity, no divergence')
        r2 = await self.ask(self.conv(('a1', 'q2')), prompt_cache_key='room-1')
        want('msg_diverge' not in r2 and r2['session'] == r1['session'], 'an extension: no divergence')
        r3 = await self.ask(self.conv(('a1 rewritten', 'q2'), ('a2', 'q3')), prompt_cache_key='room-1')
        d = r3.get('msg_diverge') or {}
        want(d.get('at') == 2 and d.get('prev') == 4 and d.get('new') == 6 and d.get('role') == 'assistant' and d.get('prev_role') == 'assistant',
             f'an edited answer: message 2 of 4 -> 6 ({d})')
        want(d.get('head') == s._short_hash('a1 rewritten') and d.get('prev_head') == s._short_hash('a1'), 'prefix hashes of the changed message')
        want('from_session' not in d, 'same session: no from_session')
        # a new head (the first user message changed) moves the session key; the conversation key links it to the previous one
        m4 = self.conv(('a1', 'q2')); m4[1] = {'role': 'user', 'content': 'first question, edited'}
        r4 = await self.ask(m4, prompt_cache_key='room-1')
        d = r4.get('msg_diverge') or {}
        want(r4['session'] != r3['session'] and r4.get('conv') == r3.get('conv') and r4.get('head') != r3.get('head'), 'new session key, same conversation')
        want(d.get('from_session') == r3['session'] and d.get('at') == 1 and d.get('prev') == 6 and d.get('new') == 4, f'session change followed ({d})')
        # another conversation key with the same head: a different conversation, nothing compared
        r5 = await self.ask(self.conv(), prompt_cache_key='room-2')
        want('msg_diverge' not in r5 and r5.get('conv') != r4.get('conv'), 'another conversation key: no divergence')
        # image requests: the token fields come from the daemon's ids
        img = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}
        def media_encode(m, *a):
            t, _ = hint_encode([dict(x, content=x['content'] if isinstance(x['content'], str) else 'IMG') for x in m], *a)
            return t, [{'image': 1}]
        s.encode_prompt = media_encode
        s.tokenize_with_images = lambda prompt, media, key=None: (s.TOK.encode(prompt) + [7] * 5, [{'start': 0}], b'x')
        mi = self.conv(('a1', 'q2')); mi[3] = {'role': 'user', 'content': [img, {'type': 'text', 'text': 'q2'}]}
        r6 = await self.ask(mi, prompt_cache_key='room-3')
        r7 = await self.ask(mi + [{'role': 'assistant', 'content': 'a3'}, {'role': 'user', 'content': 'q4'}], prompt_cache_key='room-3')
        want(r7.get('prev_prompt_tokens') == r6['prompt_tokens'], f"image request: prev_prompt_tokens ({r7.get('prev_prompt_tokens')} vs {r6['prompt_tokens']})")
        want(isinstance(r7.get('diverge_at'), int) and 0 < r7['diverge_at'] < r6['prompt_tokens'], f"image request: diverge_at ({r7.get('diverge_at')})")
        ids6 = s.TOK.encode(media_encode(mi, None, 'chat', None)[0]) + [7] * 5
        ids7 = s.TOK.encode(media_encode(mi + [{'role': 'assistant', 'content': 'a3'}, {'role': 'user', 'content': 'q4'}], None, 'chat', None)[0]) + [7] * 5
        k = next((i for i, (x, y) in enumerate(zip(ids6, ids7)) if x != y), min(len(ids6), len(ids7)))
        want(r7.get('diverge_at') == k, f"image request: diverge_at = common ids ({r7.get('diverge_at')} vs {k})")
        # nothing of the messages' text in any record
        text = json.dumps(self.recs)
        want(not any(w in text for w in ('first question', 'a1 rewritten', 'q2', 'q3')), 'no message content in the records')
        return bad

    async def test_records(self):
        self.assertEqual(await self.scenario(), [])

    class _TokenCacheLike:
        """The token cache's contract for the record (ids, ends; last_diverge per key) over the test tokenizer."""
        def __init__(self): self.lock = __import__('threading').Lock(); self.last_diverge = {}; self.prev = {}; self.hits = 0; self.entries = {}; self.sent_len = {}
        def tokenize(self, key, prompt):
            ids = s.TOK.encode(prompt); ends = [e for _, e in s.TOK(prompt, True)['offset_mapping']]
            if key in self.prev:
                old = self.prev[key]
                k = next((i for i, (x, y) in enumerate(zip(old, ids)) if x != y), min(len(old), len(ids)))
                self.last_diverge[key] = (len(old), k)
            self.prev[key] = ids
            return ids, ends
        def usable(self): return False

    async def diverge_hint_case(self):
        os.environ['HIVE_PREFIX_SHARE'] = '1'
        s.TOKCACHE = self._TokenCacheLike()
        await self.ask(self.conv(('a1', 'q2'), ('a2', 'q3')), prompt_cache_key='h')
        r = await self.ask(self.conv(('a1', 'q2'), ('a2 changed', 'q3')), prompt_cache_key='h')
        hints = s.DAEMON.kw[-1].get('boundaries') or []
        below = [h for h in hints if h <= r['diverge_at']]
        return r, hints, below

    async def test_diverge_hint_reuses_the_hints(self):
        r, hints, below = await self.diverge_hint_case()
        self.assertTrue(len(below) >= 2, (r.get('diverge_at'), hints))   # more than one boundary before the change: the deepest is meant
        self.assertEqual(r.get('diverge_hint'), max(below))
        self.assertEqual((r.get('msg_diverge') or {}).get('at'), 4)

    async def test_zz_negative_controls(self):
        import inspect
        mutants = [('MessageTrail', 'while k < n and old[k][0] == cur[k][0]:', 'while k < n - 1 and old[k][0] == cur[k][0]:'),
                   ('MessageTrail', 'if prev is None and conv and self.by_conv.get(conv) in self.entries:', 'if False:'),
                   ('MessageTrail', '"head": cur[k][2] if k < len(cur) else None', '"head": cur[k - 1][2] if k < len(cur) else None'),
                   ('MessageTrail', 'out.update(prev_prompt_tokens=len(prev[2]), diverge_at=_common_ids_len(prev[2], mids))',
                    'out.update(prev_prompt_tokens=len(mids), diverge_at=_common_ids_len(prev[2], mids))'),
                   ('_common_ids_len', 'return int(d[0]) if d.size else n', 'return int(d[0]) + 1 if d.size else n'),
                   ('_message_text', 'return c', 'return c[1:]'),
                   ('_diverge_hint', 'return max(below) if below else None', 'return min(below) if below else None')]
        caught = 0
        had = sys.modules.get(s.__name__)
        sys.modules[s.__name__] = s   # inspect finds a class's source through its module
        self.addCleanup(lambda: sys.modules.pop(s.__name__, None) if had is None else None)
        for name, a, b in mutants:
            orig = getattr(s, name)
            src = inspect.getsource(orig)
            self.assertEqual(src.count(a), 1, (name, a))
            exec(src.replace(a, b), vars(s))
            try:
                s.MSGTRAIL = s.MessageTrail(); s.TOKCACHE = s.TokenCache(); self.recs.clear()
                s.encode_prompt = hint_encode
                s.tokenize_with_images = lambda prompt, media, key=None: (s.TOK.encode(prompt), [], b'')
                bad = await self.scenario()
                if name == '_diverge_hint':
                    s.encode_prompt = hint_encode
                    s.tokenize_with_images = lambda prompt, media, key=None: (s.TOK.encode(prompt), [], b'')
                    r, hints, below = await self.diverge_hint_case()
                    if r.get('diverge_hint') != max(below): bad.append('diverge_hint')
            finally:
                vars(s)[name] = orig
            self.assertTrue(bad, f'mutant not caught: {name}: {a!r} -> {b!r}')
            caught += 1
        print(f'diverge record: {len(mutants)} mutants, {caught} caught')


class InertPlaceholderTests(unittest.TestCase):
    """The image placeholder quoted as text is read as text (2026-10-21: a tool result holding the DS chat template made every later
    request of that conversation a 400); image blocks and requests without the placeholder are untouched."""
    PH = "<\uff5cdeepseek_image\uff5c>"

    def setUp(self):
        self.enc = s.ENC
        s.ENC = type("E", (), {"IMAGE_PLACEHOLDER": self.PH})()

    def tearDown(self):
        s.ENC = self.enc

    def test_text_everywhere_and_images_kept(self):
        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD" + self.PH}}
        msgs = [{"role": "system", "content": "sys " + self.PH},
                {"role": "user", "content": [{"type": "text", "text": "a" + self.PH + "b"}, img]},
                {"role": "assistant", "content": "", "reasoning_content": "r" + self.PH,
                 "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": '{"x": "' + self.PH + '"}'}}]},
                {"role": "tool", "tool_call_id": "c", "content": [{"type": "text", "text": "{%- set IMG = \"" + self.PH + "\" -%}"}, img]}]
        tools = [{"type": "function", "function": {"name": "f", "description": "d" + self.PH, "parameters": {}}}]
        m2, t2 = s.inert_image_placeholders(msgs, tools)
        text = json.dumps([m2, t2], ensure_ascii=False)
        self.assertEqual(text.count(self.PH), 2, text)               # only the two image URLs keep it (references, not text)
        self.assertEqual(text.count("<\u200b\uff5cdeepseek_image\uff5c>"), 6, text)
        self.assertEqual(m2[1]["content"][1], img)                   # image blocks unchanged
        self.assertIn(self.PH, msgs[0]["content"])                    # the caller's messages are not modified

    def test_escaped_in_arguments(self):
        args = json.dumps({"note": "x" + self.PH})                  # ensure_ascii: the placeholder hides as \\uff5c escapes
        self.assertNotIn(self.PH, args)
        m2, _ = s.inert_image_placeholders([{"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function",
                                            "function": {"name": "f", "arguments": args}}]}], None)
        got = json.loads(m2[0]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(got, {"note": "x<\u200b\uff5cdeepseek_image\uff5c>"})
        plain = json.dumps({"a": 1})
        m3, _ = s.inert_image_placeholders([{"role": "assistant", "tool_calls": [{"function": {"arguments": plain}}]}], None)
        self.assertIs(m3[0]["tool_calls"][0]["function"]["arguments"], plain)   # arguments without it keep their bytes

    def test_untouched_without_placeholder(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "plain <\uff5cUser\uff5c> text"}]}, {"role": "assistant", "content": "ok"}]
        tools = [{"type": "function", "function": {"name": "f"}}]
        m2, t2 = s.inert_image_placeholders(msgs, tools)
        self.assertIs(m2, msgs); self.assertIs(t2, tools)            # same objects = same bytes for every request accepted before

    def test_no_placeholder_in_encoder(self):
        s.ENC = type("E", (), {})()
        msgs = [{"role": "user", "content": self.PH}]
        self.assertIs(s.inert_image_placeholders(msgs, None)[0], msgs)


class HeldMedia:
    """MediaTokens stand-in: the image request of a conversation with 1 or 2 images (one-token spans, so text and sent positions agree);
    the second request extends the first (diverge_at = its length), as TokenCache records it."""
    def __init__(self): self.calls = 0
    def usable(self): return True
    def prepare(self, key, prompt, media, keys_out=None):
        self.calls += 1
        n = len(media)
        L = 40 if n == 1 else 60
        ids = list(range(1000, 1000 + L))
        metas = [{'start': 10, 'n_vit_h': 1, 'n_vit_w': 1, 'types': [1], 'nbytes': 4}]
        if n == 2:
            metas.append({'start': 45, 'n_vit_h': 1, 'n_vit_w': 1, 'types': [1], 'nbytes': 4})
            with s.TOKCACHE.lock:
                s.TOKCACHE.last_diverge[key] = (40, 40)
        if keys_out is not None:
            keys_out.extend(['key-a', 'key-b'][:n])
        return ids, array('l', range(1, L + 1)), metas, b'AAAABBBB'[:4 * n], [(m['start'], 1) for m in metas], [m['start'] for m in metas]


class HoldDaemon(Daemon):
    """Records what each generate call carried; with missing=True a request holding an image is answered "images missing"."""
    def __init__(self, missing=False): super().__init__(); self.missing = missing; self.calls = []
    async def generate(self, session, ids, images, blob, **kw):
        self.calls.append(([dict(m) for m in images], bytes(blob)))
        if self.missing and any(m.get('held') for m in images):
            yield {'error': 'images missing'}
            return
        async for m in super().generate(session, ids, images, blob, **kw): yield m


class HeldImageTests(unittest.IsolatedAsyncioTestCase):
    """HIVE_IMAGE_HOLD (DaemonImages): images hived accepted earlier and lying inside the part shared with the conversation's previous
    request go without their bytes; "images missing" sends the request again with every image's bytes; "0" = the images as before."""
    IMG = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}

    def setUp(self):
        self.saved = (s.TOK, s.ENC, s.DAEMON, s.log_request, s.encode_prompt, s.MEDIATOK, s.TOKCACHE, s.DAEMON_IMAGES)
        s.TOK = Tokenizer(); s.ENC = Encoder(); s.INFLIGHT.clear()
        s.encode_prompt = lambda m, *a: ('prompt', [{'image': 1}] * sum(isinstance(x['content'], list) for x in m))
        s.MEDIATOK = HeldMedia(); s.TOKCACHE = s.TokenCache(); s.DAEMON_IMAGES = s.DaemonImages()
        self.recs = []; s.log_request = lambda rec: self.recs.append(dict(rec))
        self.env = {k: os.environ.pop(k, None) for k in ('HIVE_PREFIX_SHARE', 'HIVE_IMAGE_HOLD')}

    def tearDown(self):
        s.TOK, s.ENC, s.DAEMON, s.log_request, s.encode_prompt, s.MEDIATOK, s.TOKCACHE, s.DAEMON_IMAGES = self.saved
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None: os.environ[k] = v

    def msgs(self, n):
        m = [{'role': 'user', 'content': [self.IMG, {'type': 'text', 'text': 'look'}]}]
        if n == 2:
            m += [{'role': 'assistant', 'content': 'seen'}, {'role': 'user', 'content': [self.IMG, {'type': 'text', 'text': 'and this'}]}]
        return m

    async def ask(self, n):
        r = await s.chat(Request({'messages': self.msgs(n), 'reasoning_effort': 'none', 'prompt_cache_key': 'room'}))
        return r, self.recs[-1]

    async def test_held_then_resent(self):
        s.DAEMON = HoldDaemon()
        r, rec = await self.ask(1)
        self.assertEqual(r['choices'][0]['message']['content'], 'abc')
        (metas, blob), = s.DAEMON.calls
        self.assertEqual([(m.get('key'), m.get('held')) for m in metas], [('key-a', None)])   # first request: bytes + key
        self.assertEqual(blob, b'AAAA')
        self.assertIn('key-a', s.DAEMON_IMAGES.known)
        r, rec = await self.ask(2)
        metas, blob = s.DAEMON.calls[-1]
        self.assertEqual([(m['key'], m.get('held', False)) for m in metas], [('key-a', True), ('key-b', False)])
        self.assertEqual(blob, b'BBBB')                                  # only the new image's bytes
        self.assertEqual((rec.get('images_held'), rec.get('image_bytes'), rec.get('images')), (1, 4, 2))
        # hived lacks a held image: the same request once more with every image's bytes, the answer unchanged
        s.DAEMON = HoldDaemon(missing=True)
        r, rec = await self.ask(2)
        self.assertEqual(r['choices'][0]['message']['content'], 'abc')
        (m1, b1), (m2, b2) = s.DAEMON.calls
        self.assertEqual(([x.get('held', False) for x in m1], b1), ([True, False], b'BBBB'))
        self.assertEqual(([(x['key'], x.get('held', False)) for x in m2], b2), ([('key-a', False), ('key-b', False)], b'AAAABBBB'))
        self.assertTrue(rec.get('images_resent'))
        self.assertEqual(rec.get('image_bytes'), 8)
        self.assertEqual(list(s.DAEMON_IMAGES.known), ['key-a', 'key-b'])  # forgotten, then learned again from the accepted request

    async def test_nothing_held_before_accepted(self):
        s.DAEMON = HoldDaemon()
        s.DAEMON_IMAGES.known.clear()
        await self.ask(2)  # diverge_at 40, but hived has not accepted key-a from this process: its bytes go along
        metas, blob = s.DAEMON.calls[-1]
        self.assertEqual(([m.get('held', False) for m in metas], blob), ([False, False], b'AAAABBBB'))

    async def test_off(self):
        os.environ['HIVE_IMAGE_HOLD'] = '0'
        s.DAEMON = HoldDaemon()
        await self.ask(1); await self.ask(2)
        for metas, blob in s.DAEMON.calls:
            self.assertTrue(all('key' not in m and 'held' not in m for m in metas))
        self.assertEqual(s.DAEMON.calls[-1][1], b'AAAABBBB')

    def test_plan_bounds(self):
        d = s.DaemonImages(cap=2)
        d.learn(['a', 'b', 'c'])
        self.assertEqual(list(d.known), ['b', 'c'])                     # least recently used dropped past the cap
        im = [{'start': 0, 'types': [1, 1], 'nbytes': 2}, {'start': 5, 'types': [1], 'nbytes': 3}]
        metas, payload, held = d.plan(im, b'xxyyy', ['b', 'c'], 6)
        self.assertEqual((held, payload), (2, b''))
        metas, payload, held = d.plan(im, b'xxyyy', ['b', 'c'], 5)   # the second image ends at 6 > 5: sent
        self.assertEqual((held, payload, [m.get('held', False) for m in metas]), (1, b'yyy', [True, False]))
        self.assertEqual(d.plan(im, b'xxyyy', ['b', 'c'], None)[1:], (b'xxyyy', 0))


if __name__=='__main__': unittest.main()
