#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Production handlers + fake token source and real Unix EOF. No model/GPU."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

spec = importlib.util.spec_from_file_location('server_under_test', Path(__file__).resolve().parents[1] / 'server/hive_server.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

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

    async def test_sent_only_when_enabled(self):
        await s.chat(Request({'messages':self.conv(), 'reasoning_effort':'none'}))
        self.assertNotIn('boundaries', s.DAEMON.kw[-1], 'default: no field, no hint cost')
        os.environ['HIVE_PREFIX_SHARE']='0'
        await s.chat(Request({'messages':self.conv(), 'reasoning_effort':'none'}))
        self.assertNotIn('boundaries', s.DAEMON.kw[-1])
        os.environ['HIVE_PREFIX_SHARE']='1'
        await s.chat(Request({'messages':self.conv(), 'reasoning_effort':'none'}))
        _, _, want=self.hints_for(self.conv())
        self.assertEqual(s.DAEMON.kw[-1].get('boundaries'), want)
        s.tokenize_with_images=lambda prompt, media: (s.TOK.encode(prompt), [{'start':0}], b'x')  # image request: no text offsets
        s.encode_prompt=lambda *a: (hint_encode(*a)[0], [{'image':1}])
        await s.chat(Request({'messages':self.conv(), 'reasoning_effort':'none'}))
        self.assertNotIn('boundaries', s.DAEMON.kw[-1])
        self.assertFalse(s.INFLIGHT)



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
        self.env = {k: os.environ.pop(k, None) for k in ('HIVE_TOKEN_CACHE', 'HIVE_TOKEN_CACHE_VERIFY')}
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


if __name__=='__main__': unittest.main()
