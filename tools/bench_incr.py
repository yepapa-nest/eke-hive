#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Multi-turn incremental prefill benchmark — appends turns of a few hundred to a few thousand new tokens after a long
conversation and measures TTFT, chunk count and CPU wait.

In production monitoring, conversational turns with 300–4000 new tokens had a TTFT of 1.8–2.6 s, 60–70 % of it CPU wait, and
follow-up turns almost always took 2 chunks (H5 boundary cut — the previous assistant reply of ~120 tokens as a separate chunk).
The hive log does not contain prompt text, so real requests cannot be replayed; this script builds the same shape instead:
  system + user (long context) → [assistant (fixed text) + user (N new tokens + question)] × turns.
The assistant content is fixed text rather than generated output — it diverges from the live state and therefore takes the same
'prompt-ckpt' reuse path as production (conversations with thinking enabled take the same path because the next turn's template
drops the reasoning text).

Appends one JSON line to --out (per-turn rows + per-size summary). Engine-side numbers are read only from this script's sessions'
lines in hived.log.
"""
import argparse, hashlib, json, os, re, statistics as st, time, uuid
from pathlib import Path
import requests

# hived.log of scripts/hive-start.sh: $HIVE_STATE_DIR/logs, otherwise run/logs in the repository
DEFAULT_LOG = os.path.join(os.environ.get('HIVE_STATE_DIR') or str(Path(__file__).resolve().parents[1] / 'run'), 'logs', 'hived.log')

P = argparse.ArgumentParser()
P.add_argument('--name', required=True)
P.add_argument('--out', required=True, help='JSONL file to append the result line to')
P.add_argument('--base', default='http://127.0.0.1:8430')
P.add_argument('--log', default=DEFAULT_LOG, help='hived.log (default: $HIVE_STATE_DIR/logs/hived.log or run/logs/hived.log)')
P.add_argument('--corpus', required=True, help='long text file the contexts and new turns are cut from')
P.add_argument('--ctx', default='80000,150000', help='target token counts of the long contexts (comma-separated) — one session per context')
P.add_argument('--sizes', default='300,1000,3000', help='new tokens appended per turn (comma-separated) — cycled turn by turn')
P.add_argument('--reps', type=int, default=10, help='turns per size')
P.add_argument('--max-tokens', type=int, default=16)
A = P.parse_args()

CH_PER_TOK = 2.5  # characters per token of this corpus (bench_tune: 200,000 chars ≈ 80K tokens) — actual token counts are recorded from the response usage
corpus = open(A.corpus, encoding='utf-8', errors='replace').read()
ASK = '\n\n위 내용을 한 문장으로 요약해 줘.'
SYSTEM = '너는 코드와 문서를 읽고 짧게 답하는 도우미다.'


def assistant_text(k):
    # ~120–150 tokens — same size as the first of the two production chunks (M 116–158)
    return (f'[턴 {k}] 요약: 앞 내용은 엔진의 프리필·디코드 경로와 캐시 정책, 세션 재사용 규칙, 로깅 형식을 설명한다. '
            'The text describes the prefill and decode paths, the expert cache policy, session reuse rules and the log format. '
            '핵심은 청크 단위 전송 비용과 CPU 전문가 계산의 균형이며, 긴 문맥에서는 재사용 체크포인트가 첫 토큰 시간을 좌우한다. '
            'Key points: per-chunk transfer cost, CPU expert balance, and checkpoint reuse for long contexts.')


def stream(messages, sid_key):
    body = {'model': 'hive', 'messages': messages, 'max_tokens': A.max_tokens, 'temperature': 0.0, 'reasoning_effort': 'none',
            'stream': True, 'stream_options': {'include_usage': True}, 'hive_session_id': sid_key}
    t0 = time.time(); first = None; usage = {}
    r = requests.post(A.base + '/v1/chat/completions', json=body, stream=True, timeout=1800)
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data: ') or line == 'data: [DONE]':
            continue
        ev = json.loads(line[6:])
        d = (ev.get('choices') or [{}])[0].get('delta', {})
        if (d.get('content') or d.get('reasoning_content')) and first is None:
            first = time.time() - t0
        if ev.get('usage'):
            usage = ev['usage']
    return {'ttft': round(first, 3) if first else None, 'wall': round(time.time() - t0, 3), 'prompt': usage.get('prompt_tokens')}


RE_REUSE = re.compile(r'^\[hived\] (\S+): reuse (\d+)/(\d+) via (\S+)')
RE_CHUNK = re.compile(r'^\[hived\] (\S+): prefill chunk (\d+) M (\d+) · (\d+)→(\d+)/(\d+) .*cpu wait (\d+) ms span (\d+) ms · ([\d.]+) ms')
RE_FINAL = re.compile(r'^\[hived\] (\S+): prefill (\d+) tok (\d+) ms · decode')
RE_ANY = re.compile(r'^\[hived\] (\S+): prefill \d+ tok \d+ ms · decode')


def engine_turns(text, sid):
    """Per-request engine numbers for this session (log order = request order). A reuse line can appear twice per request (reset → shared-prefix)."""
    out, cur = [], None
    for line in text.split('\n'):
        if not line.startswith('[hived] ' + sid + ':'):
            continue
        m = RE_REUSE.match(line)
        if m:
            if cur is None or cur.get('chunks'):
                cur = {'chunks': []}
            cur['reuse'], cur['via'] = int(m[2]), m[4]
            continue
        m = RE_CHUNK.match(line)
        if m:
            cur = cur if cur is not None else {'chunks': []}
            cur['chunks'].append({'M': int(m[3]), 'cpu_wait': int(m[7]), 'ms': float(m[9])})
            continue
        m = RE_FINAL.match(line)
        if m:
            cur = cur if cur is not None else {'chunks': []}
            cur['prefill_tok'], cur['prefill_ms'] = int(m[2]), int(m[3])
            out.append(cur); cur = None
    return out


os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
log_off = os.path.getsize(A.log)
res = {'name': A.name, 'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'env': {k: v for k, v in os.environ.items() if k.startswith('HIVE_')}}
sizes = [int(x) for x in A.sizes.split(',')]
turns, sids = [], []
off = 0
for ci, ctx in enumerate(int(x) for x in A.ctx.split(',')):
    key = f'bench-incr-{A.name}-{ctx}-{uuid.uuid4().hex[:8]}'
    sid = hashlib.sha256(key.encode()).hexdigest()[:32]; sids.append(sid)
    nch = int(ctx * CH_PER_TOK)
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': corpus[off:off + nch] + ASK}]
    off += nch
    s = stream(msgs, key)
    res[f'ctx{ctx}_first'] = s
    k = 0
    for rep in range(A.reps):
        for n in sizes:
            k += 1
            nc = int(n * CH_PER_TOK)
            if off + nc > len(corpus):
                off = 0  # wrap around when the corpus is used up (overlapping another session's earlier text is fine: the session prefixes differ)
            msgs = msgs + [{'role': 'assistant', 'content': assistant_text(k)}, {'role': 'user', 'content': corpus[off:off + nc] + ASK}]
            off += nc
            s = stream(msgs, key)
            turns.append({'ctx': ctx, 'size': n, 'k': k, **s})
time.sleep(1.0)
with open(A.log, 'rb') as f:
    f.seek(log_off); text = f.read().decode('utf-8', errors='replace')
# Attach engine numbers to turns (per session: first request = long context → following turns in order)
for ci, sid in enumerate(sids):
    ctx = int(A.ctx.split(',')[ci])
    eng = engine_turns(text, sid)
    mine = [t for t in turns if t['ctx'] == ctx]
    if len(eng) == len(mine) + 1:
        res[f'ctx{ctx}_first_engine'] = eng[0]
        for t, e in zip(mine, eng[1:]):
            t['new_tok'] = e.get('prefill_tok'); t['prefill_ms'] = e.get('prefill_ms'); t['via'] = e.get('via'); t['reuse'] = e.get('reuse')
            t['n_chunks'] = len(e['chunks']); t['chunk_M'] = [c['M'] for c in e['chunks']]
            t['cpu_wait'] = sum(c['cpu_wait'] for c in e['chunks'])
    else:
        res[f'ctx{ctx}_engine_mismatch'] = [len(eng), len(mine) + 1]
med = lambda xs: round(st.median(xs), 3) if xs else None
summ = {}
for ctx in (int(x) for x in A.ctx.split(',')):
    for n in sizes:
        v = [t for t in turns if t['ctx'] == ctx and t['size'] == n]
        summ[f'{ctx}/{n}'] = {'n': len(v), 'ttft_p50': med([t['ttft'] for t in v if t['ttft']]),
                              'ttft_max': max((t['ttft'] for t in v if t['ttft']), default=None),
                              'prefill_ms_p50': med([t['prefill_ms'] for t in v if t.get('prefill_ms') is not None]),
                              'cpu_wait_p50': med([t['cpu_wait'] for t in v if t.get('cpu_wait') is not None]),
                              'chunks_p50': med([t['n_chunks'] for t in v if t.get('n_chunks') is not None]),
                              'new_tok_p50': med([t['new_tok'] for t in v if t.get('new_tok') is not None])}
res['summary'] = summ
res['ttft_p50_all'] = med([t['ttft'] for t in turns if t['ttft']])
# Did requests outside the benchmark interleave? (count of other sessions' request-end lines · their decode mixed into this window perturbs TTFT)
foreign = [m[1] for m in (RE_ANY.match(l) for l in text.split('\n')) if m and m[1] not in sids]
res['foreign_requests'] = len(foreign)
res['contaminated'] = len(foreign) > 0
res['turns'] = turns
open(A.out, 'a').write(json.dumps(res, ensure_ascii=False) + '\n')
print(json.dumps({k: v for k, v in res.items() if k != 'turns'}, ensure_ascii=False, indent=1))
