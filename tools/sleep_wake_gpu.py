#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""hive sleep/wake — GPU check on a host with a running hive service (hived + hive_server :8430).

Talks only to the public HTTP API (stdlib urllib — no extra packages). Greedy (temperature 0) requests, fixed prompts:
  before sleep : fresh X1 · fresh X2 (noise floor — same prompt, new session = full prefill + decode) · control conversation C (turn 1 + turn 2,
                 never slept) · conversation S turn 1
  sleep        : POST /release_memory_occupation — wall time, hived sleep report (drain/ms/freed/kept table), VRAM via nvidia-smi and /health
  [--hold-s N] : stays asleep N seconds (the media window); a request sent while asleep (--probe) must wait and be served after wake
  wake         : POST /resume_memory_occupation — wall time, hived wake report (alloc/warm ms, slots, warmed), VRAM
  after wake   : fresh X3 (== X1?) · conversation S turn 2 (prefix reuse across the sleep: cached_prefix > 0, and == control C turn 2?)
Verdict lines: "same answer bytes" for X3 vs X1 and S2 vs C2, read against the X1 vs X2 noise floor (if X1 != X2 the engine itself is not
bit-deterministic for this prompt under the service settings — CPU/DMA split and promotions — and the byte comparison is inconclusive;
re-run the daemon with --no-cpu --promote 0 / HIVE_DMA_FRAC fixed for a deterministic comparison, see docs/validation.md).
--level 3 (hived started with HIVE_SLEEP_VMM=1): also prints hived's per-process VRAM while asleep (nvidia-smi --query-compute-apps) and a bare-context
baseline (engine/tests/test_devmem --context-only via scripts/hive-run.sh) — the difference is what hived holds beyond an empty CUDA context.
Exit 0 = sleep/wake worked and every comparison that the noise floor allows matched; 1 = a failure; 2 = inconclusive (noise floor differs).
"""
import argparse
import json
import os
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request

PROMPT = ('List the first twelve prime numbers, then explain in three short sentences why there are infinitely many primes. '
          'Answer in plain text without markdown.')
TURN2 = 'Now give the sum of those twelve primes and show the addition step by step.'


def http(base, method, path, body=None, timeout=86400):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b'{}')
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b'{}')
        except Exception:  # noqa: BLE001
            return e.code, {}


def chat(base, session, messages, max_tokens):
    t0 = time.monotonic()
    code, r = http(base, 'POST', '/v1/chat/completions', {'model': 'hive', 'messages': messages, 'temperature': 0, 'max_tokens': max_tokens,
                                                         'reasoning_effort': 'none', 'hive_session_id': session, 'stream': False})
    dt = time.monotonic() - t0
    if code != 200:
        raise RuntimeError(f'chat {session}: HTTP {code} {r}')
    msg = r['choices'][0]['message']
    return {'text': msg.get('content') or '', 'n': r['usage']['completion_tokens'], 'hive': r.get('hive', {}), 's': dt}


def nvsmi():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True,
                             timeout=10).stdout.strip().splitlines()[0]
        used, tot = (int(x) for x in out.split(','))
        return used, tot
    except Exception:  # noqa: BLE001
        return None, None


def proc_vram(name_part):
    """Per-process VRAM (MiB) from nvidia-smi — the first process whose name contains name_part (context + all of that process's allocations). None if unreadable."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], capture_output=True,
                             text=True, timeout=10).stdout
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) == 3 and name_part in parts[1]:
                return int(parts[2])
    except Exception:  # noqa: BLE001
        pass
    return None


def bare_context(cmd, timeout=60):
    """VRAM baseline of an empty CUDA context: launches test_devmem --context-only and measures that process with nvidia-smi."""
    pr = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    v, t0 = None, time.monotonic()
    while time.monotonic() - t0 < timeout and pr.poll() is None and v is None:
        time.sleep(1)
        v = proc_vram('test_devmem')
    try:
        pr.wait(timeout=timeout)
    except Exception:  # noqa: BLE001
        pr.kill()
    return v


def health(base):
    code, h = http(base, 'GET', '/health', timeout=10)
    return h.get('state'), (h.get('vram') or {}).get('used_mib')


def first_diff(a, b):
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a[k] != b[k]), None)
    if i is not None:
        return i
    return None if len(a) == len(b) else n


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument('--base', default='http://127.0.0.1:8430')
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--hold-s', type=float, default=0, help='stay asleep this long (media window stand-in)')
    ap.add_argument('--probe', action='store_true', help='send a request while asleep; it must be served after wake')
    ap.add_argument('--target-gib', type=float, default=None, help='asleep VRAM target (report only · default 14 for level 1/2, 1 for level 3)')
    ap.add_argument('--level', type=int, default=1, help='sleep level 1 | 2 (+ session KV to RAM) | 3 (+ dense/work via VMM — hived started with HIVE_SLEEP_VMM=1)')
    hive_run = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'hive-run.sh')
    ap.add_argument('--context-cmd', default=f'bash {shlex.quote(hive_run)} "build/test_devmem --context-only --hold 15"',
                    help='command that starts the bare-context probe (engine/tests/test_devmem.cu); "" = skip')
    a = ap.parse_args()
    if a.target_gib is None:
        a.target_gib = 1.0 if a.level >= 3 else 14.0
    base, mt = a.base, a.max_tokens
    tag = str(int(time.time()))
    st, _ = health(base)
    if st != 'ready':
        raise SystemExit(f'hive is not ready (state {st})')
    u = [{'role': 'user', 'content': PROMPT}]
    print('[before] warm-up + noise floor …', flush=True)
    chat(base, f'sw-warm-{tag}', u, 16)
    x1 = chat(base, f'sw-x1-{tag}', u, mt)
    x2 = chat(base, f'sw-x2-{tag}', u, mt)
    c1 = chat(base, f'sw-c-{tag}', u, mt)
    c2 = chat(base, f'sw-c-{tag}', u + [{'role': 'assistant', 'content': c1['text']}, {'role': 'user', 'content': TURN2}], mt)
    s1 = chat(base, f'sw-s-{tag}', u, mt)
    used0, tot = nvsmi()
    print(f'[before] VRAM {used0}/{tot} MiB (nvidia-smi) · X1 {x1["n"]} tok {x1["s"]:.1f}s · X2 {x2["s"]:.1f}s · C2 cached_prefix {c2["hive"].get("cached_prefix")}')

    t0 = time.monotonic()
    code, sr = http(base, 'POST', '/release_memory_occupation', {'level': a.level})
    t_sleep = time.monotonic() - t0
    used1, _ = nvsmi()
    st1, vram1 = health(base)
    hived_vram = proc_vram('hived')  # VRAM held by this process (including its context) — the reference for the asleep verdict
    sl = sr.get('sleep', {})
    print(f'[sleep] HTTP {code} · wall {t_sleep * 1000:.0f} ms · hived {sl.get("ms", 0):.0f} ms (drain {sl.get("drain_ms", 0):.0f}) · state {st1} · '
          f'VRAM {used1} MiB (nvidia-smi) / {vram1} MiB (hived) · freed {sl.get("freed_mib", 0):.0f} MiB · slots {sl.get("slots_released")} · keys {sl.get("resident_keys")}')
    print(f'[sleep] level {sl.get("level")} (asked {a.level}){" · note: " + sl["note"] if sl.get("note") else ""} · hived per-process VRAM {hived_vram} MiB '
          f'(nvidia-smi --query-compute-apps) · vmm {sl.get("vmm", {})} · host {sl.get("host_memory", {})}')
    ctx = bare_context(a.context_cmd) if a.context_cmd else None
    if ctx is not None:
        print(f'[context] bare CUDA context (test_devmem --context-only) {ctx} MiB → hived asleep beyond a bare context: '
              f'{(hived_vram - ctx) if hived_vram is not None else "?"} MiB (loaded kernel modules · cuBLAS handle · graphs · small cudaMalloc)')
    kept = sl.get('vram_kept', {})
    print('[sleep] VRAM kept while asleep (MiB): ' + ' · '.join(f'{k} {v:.0f}' for k, v in kept.items()))
    probe = {}
    if a.probe:
        def run_probe():
            probe['t0'] = time.monotonic()
            probe['r'] = chat(base, f'sw-probe-{tag}', [{'role': 'user', 'content': 'Say OK.'}], 8)
            probe['t1'] = time.monotonic()
        th = threading.Thread(target=run_probe, daemon=True)
        th.start()
    if a.hold_s > 0:
        time.sleep(a.hold_s)
    t0 = time.monotonic()
    code_w, wr = http(base, 'POST', '/resume_memory_occupation', {})
    t_wake = time.monotonic() - t0
    t_wake_end = time.monotonic()
    used2, _ = nvsmi()
    st2, _ = health(base)
    wk = wr.get('wake', {})
    print(f'[wake] HTTP {code_w} · wall {t_wake * 1000:.0f} ms · hived {wk.get("ms", 0):.0f} ms (alloc {wk.get("alloc_ms", 0):.0f} · warm {wk.get("warm_ms", 0):.0f}) · '
          f'slots {wk.get("slots")}/{wk.get("slots_before_sleep")} · warmed {wk.get("warmed")}/{wk.get("resident_keys")} · state {st2} · VRAM {used2} MiB')
    if a.probe:
        th.join(600)
        ok_probe = 'r' in probe and probe['t1'] >= t_wake_end - 0.5
        print(f'[probe] request sent while asleep: {"served after wake" if ok_probe else "NOT OK"} ({probe.get("r", {}).get("text", "")[:40]!r})')

    x3 = chat(base, f'sw-x3-{tag}', u, mt)
    s2 = chat(base, f'sw-s-{tag}', u + [{'role': 'assistant', 'content': s1['text']}, {'role': 'user', 'content': TURN2}], mt)
    floor = x1['text'] == x2['text']
    res = {
        'noise floor X1 == X2 (no sleep)': floor,
        'fresh X3 (after wake) == X1': x3['text'] == x1['text'],
        'turn 1 S1 == C1': s1['text'] == c1['text'],
        'turn 2 S2 (across sleep) == C2 (no sleep)': s2['text'] == c2['text'],
    }
    print(f'[after] X3 {x3["n"]} tok {x3["s"]:.1f}s · S2 cached_prefix {s2["hive"].get("cached_prefix")} (C2 {c2["hive"].get("cached_prefix")}) · '
          f'first diff X3/X1 {first_diff(x3["text"], x1["text"])} · S2/C2 {first_diff(s2["text"], c2["text"])}')
    for k, v in res.items():
        print(f'  {k}: {"SAME BYTES" if v else "DIFFERENT"}')
    ok = code == 200 and code_w == 200 and st1 == 'sleeping' and st2 == 'ready' and (s2['hive'].get('cached_prefix') or 0) > 0
    asleep_mib = hived_vram if hived_vram is not None else used1
    if asleep_mib is not None:
        print(f'[verdict] asleep VRAM {asleep_mib / 1024:.2f} GiB ({"hived per-process" if hived_vram is not None else "device-wide"}) vs target {a.target_gib} GiB: '
              f'{"within" if asleep_mib / 1024 <= a.target_gib else "ABOVE"} · sleep {t_sleep:.2f}s · wake {t_wake:.2f}s (level {a.level})')
    if not ok:
        print('[verdict] FAIL: sleep/wake state or prefix reuse')
        raise SystemExit(1)
    if not floor:
        print('[verdict] INCONCLUSIVE: the engine is not bit-deterministic for this prompt even without sleep (X1 != X2)')
        raise SystemExit(2)
    same = all(res.values())
    print(f'[verdict] {"PASS: same answer bytes across sleep/wake" if same else "FAIL: answer bytes differ across sleep/wake while the noise floor is clean"}')
    raise SystemExit(0 if same else 1)


if __name__ == '__main__':
    main()
