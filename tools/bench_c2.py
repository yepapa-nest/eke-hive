#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""bench_c2.py — two concurrent streams measured the way bench_tune.py measures 4 and 8 (the C2 row of docs/benchmarks.md): 3 warm-up requests, then two runs of 2 streams x 256 tokens. Prints one JSON line."""

import json, threading, time, requests
BASE = "http://127.0.0.1:8430"
def stream(prompt, max_tokens, temperature=0.0):
    body = {"model": "hive", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature,
            "reasoning_effort": "none", "stream": True}
    t0 = time.time(); first = None; usage = {}
    r = requests.post(BASE + "/v1/chat/completions", json=body, stream=True, timeout=900)
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: ") or line == "data: [DONE]": continue
        ev = json.loads(line[6:]); d = (ev.get("choices") or [{}])[0].get("delta", {})
        if d.get("content") or d.get("reasoning_content"): first = first or time.time() - t0
        if ev.get("usage"): usage = ev["usage"]
    wall = time.time() - t0; n = usage.get("completion_tokens") or 0
    return {"ttft": first, "wall": wall, "n": n, "tps": (n - 1) / (wall - first) if first and n > 1 and wall > first else None}
for i in range(3):
    stream(f"Warmup {i}: explain in detail how a refrigerator works, then write a short poem about winter, then list 10 programming languages with one fact each.", 300, 0.7)
res = {"ts": time.strftime("%H:%M:%S"), "runs": []}
for run in range(2):
    c = 2; sink = {}
    def one(k): sink[k] = stream(f"Stream {k} of {c}: tell a long story about city number {k + 10 * run}. ", 256)
    th = [threading.Thread(target=one, args=(k,)) for k in range(c)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
    res["runs"].append({"agg": round(sum(v["n"] for v in sink.values()) / wall, 2), "per_stream_tps": [round(v["tps"] or 0, 2) for v in sink.values()],
                        "ttft": [round(v["ttft"] or 0, 2) for v in sink.values()], "n": [v["n"] for v in sink.values()]})
print(json.dumps(res))
