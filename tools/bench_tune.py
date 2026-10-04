#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""bench_tune.py — the development benchmark behind the step-by-step history in docs/performance.md (fixed decode prompts, prefill on a fixed text).
One run = 3 warm-up requests, 3 answer checks, decode with 1 (x2) / 4 / 8 concurrent streams, then prefill of 15K / 42K / 80K
(and 1K / 4K) tokens of natural text plus decode right after an 80K prompt. Appends one JSON line to --out and prints it.
Corpus: the repository's own sources and documents (--corpus FILE to use a fixed text instead — the published numbers used a
frozen snapshot of this tree, so prompt token counts differ slightly from run to run of a changed tree).
Reads the daemon log for ready time, VRAM, errors and foreign requests; package energy/temperature are reported when readable."""

import argparse, glob, json, os, re, threading, time
import requests

P = argparse.ArgumentParser()
P.add_argument('--name', required=True)
P.add_argument('--out', default=os.path.join(os.environ.get('HIVE_STATE_DIR', 'run'), 'bench', 'tune.jsonl'))
P.add_argument('--corpus', default='', help='text file used for the prefill prompts (default: this repository)')
P.add_argument('--log', default=os.path.join(os.environ.get('HIVE_STATE_DIR', 'run'), 'logs', 'hived.log'))
P.add_argument('--base', default='http://127.0.0.1:8430')
A = P.parse_args()
LOG = A.log
os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)


def stream(prompt, max_tokens, temperature=0.0):
    body = {'model': 'hive', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tokens, 'temperature': temperature,
            'reasoning_effort': 'none', 'stream': True}
    t0 = time.time(); first = None; usage = {}; text = ''
    r = requests.post(A.base + '/v1/chat/completions', json=body, stream=True, timeout=900)
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data: ') or line == 'data: [DONE]':
            continue
        ev = json.loads(line[6:])
        d = (ev.get('choices') or [{}])[0].get('delta', {})
        piece = d.get('content') or d.get('reasoning_content')
        if piece:
            first = first or time.time() - t0; text += piece
        if ev.get('usage'):
            usage = ev['usage']
    wall = time.time() - t0
    n = usage.get('completion_tokens') or 0
    return {'ttft': first, 'wall': wall, 'n': n, 'prompt': usage.get('prompt_tokens'), 'tps': (n - 1) / (wall - first) if first and n > 1 and wall > first else None, 'text': text}


import subprocess
def energy_uj():
    try: return int(open('/sys/class/powercap/intel-rapl:0/energy_uj').read())  # package energy (readable as root or with relaxed permissions); None otherwise
    except Exception: return None
def tctl():
    import glob as g
    for h in g.glob('/sys/class/hwmon/hwmon*'):
        try:
            if open(h + '/name').read().strip() == 'k10temp': return int(open(h + '/temp1_input').read()) / 1000
        except Exception: pass
class Meter:
    def __enter__(self): self.e0 = energy_uj(); self.t0 = time.time(); self.peak = 0; self.stop = False; threading.Thread(target=self.poll, daemon=True).start(); return self
    def poll(self):
        while not self.stop: self.peak = max(self.peak, tctl() or 0); time.sleep(0.5)
    def __exit__(self, *a):
        self.stop = True; e1 = energy_uj(); dt = time.time() - self.t0
        self.watts = round((e1 - self.e0) / 1e6 / dt, 1) if self.e0 and e1 and e1 > self.e0 else None
res = {'name': A.name, 'ts': time.strftime('%H:%M:%S')}
log = open(LOG, errors='replace').read()
ready = [m for m in re.finditer(r'\[hived\] ready in ([\d.]+)s', log)]
vf = re.findall(r'session pool .* VRAM free (\d+) MiB', log)
res['ready_s'] = float(ready[-1].group(1)) if ready else None
res['vram_free_mib'] = int(vf[-1]) if vf else None
log_start = ready[-1].end() if ready else 0
for i in range(3):
    stream(f'Warmup {i}: explain in detail how a refrigerator works, then write a short poem about winter, then list 10 programming languages with one fact each.', 300, 0.7)
ok = []
ok.append('paris' in stream('What is the capital of France? Answer in one word.', 16)['text'].lower())
ok.append('1591' in stream('Compute 37*43. Reply with just the number.', 24)['text'])
ok.append('blue' in stream('What colour is a clear daytime sky? One word.', 16)['text'].lower())
res['answers_ok'] = sum(ok)
with Meter() as m:
    res['c1'] = [round(stream(f'Run {i}: write a long story about a lighthouse keeper and a storm. ' * 2, 256)['tps'] or 0, 2) for i in range(2)]
res['c1_watts'], res['c1_tctl_peak'] = m.watts, m.peak
for c in (4, 8):
    sink = {}
    def one(k):
        sink[k] = stream(f'Stream {k} of {c}: tell a long story about city number {k}. ', 256)
    th = [threading.Thread(target=one, args=(k,)) for k in range(c)]
    with Meter() as m:
        t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
        res[f'c{c}_agg'] = round(sum(v['n'] for v in sink.values()) / (time.time() - t0), 2)
    res[f'c{c}_watts'], res[f'c{c}_tctl_peak'] = m.watts, m.peak
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
srcs = sorted(glob.glob(os.path.join(ROOT, 'engine/src/**/*.c*'), recursive=True)) + sorted(glob.glob(os.path.join(ROOT, 'server/*.py'))) \
     + sorted(glob.glob(os.path.join(ROOT, 'docs/*.md'))) + sorted(glob.glob(os.path.join(ROOT, 'tools/*.py')))
blob = ''.join('\n\n### FILE ' + os.path.relpath(f, ROOT) + '\n' + open(f, encoding='utf-8', errors='replace').read() for f in srcs)
# A fixed corpus keeps the prompt token counts (and chunk boundaries) identical between runs of different trees.
if A.corpus: blob = open(A.corpus, encoding='utf-8', errors='replace').read()
for chars, tag, off in ((40000, 'pf15k', 60000), (40000, 'pf15k_2nd', 520000), (100000, 'pf42k', 170000), (200000, 'pf80k', 280000)) + (((245000, 'pf100k', 500000),) if os.environ.get('BENCH_PF100K') else ()):
    s = stream(blob[off:off + chars] + '\n\nSummarize the text above in one sentence.', 16)
    res[tag] = round(s['ttft'], 2) if s['ttft'] else None
    res[tag + '_prompt'] = s['prompt']
# Short prompts (follow-up turns) and decode after a long (80K) context
for chars, tag, off in ((3000, 'pf1k', 900000), (12000, 'pf4k', 910000)):
    s = stream(blob[off:off + chars] + '\n\nSummarize the text above in one sentence.', 16)
    res[tag] = round(s['ttft'], 2) if s['ttft'] else None
    res[tag + '_prompt'] = s['prompt']
s = stream(blob[280000:480000] + '\n\nList 20 function names used in the code above with a one-line description each.', 192)
res['ld80k_tps'] = round(s['tps'], 2) if s['tps'] else None
res['ld80k_ttft'] = round(s['ttft'], 2) if s['ttft'] else None
res['ld80k_prompt'] = s['prompt']
log = open(LOG, errors='replace').read()[log_start:]
tot = re.findall(r'prefill total (\d+) rows in (\d+) chunks .* cpu wait (\d+) ms span (\d+) ms · (\d+) ms', log)
big = [t for t in tot if 70000 < int(t[0]) < 90000][:1]
huge = [t for t in tot if int(t[0]) >= 90000]
if huge: res['pf100k_chunks'] = int(huge[-1][1])
if big:
    res['pf80k_chunks'], res['pf80k_cpu_wait_ms'] = int(big[-1][1]), int(big[-1][2])
res['errors'] = len(re.findall(r'(?i)\b(error|failed|exception|abort)\b', log))
# Foreign traffic during the run: this script sends warm-up 3 + answers 3 + c1 2 + c4 4 + c8 8 + prefill 7 (+1 with BENCH_PF100K)
res['requests_seen'] = len(re.findall(r'^\[hived\] [0-9a-f~]+: prefill \d+ tok', log, re.M))
res['contaminated'] = res['requests_seen'] > 27 + (1 if os.environ.get('BENCH_PF100K') else 0)
res['foreign_batch'] = len(re.findall(r'^\[hived\] [0-9a-f~]+: prefill \d+ tok .*batch [2-9]', log, re.M))
res['resident'] = (re.findall(r'resident (\d+)/(\d+)', log) or [None])[-1]
open(A.out, 'a').write(json.dumps(res, ensure_ascii=False) + '\n')
print(json.dumps(res, ensure_ascii=False))
