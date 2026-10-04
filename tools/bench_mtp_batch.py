#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Batched MTP (HIVE_MTP_BATCH) GPU measurement — runs against a live hive server (OpenAI API). Does not start the GPU or the server.

Two modes:
  perf   — concurrent decode with c streams (default 1,2,3,4). For each c, --repeat times: send c streams at once (each a new session with a
           distinct prefix line — no prefix reuse); aggregate = sum(completion_tokens) / wall clock (the same definition as the c4_agg/c8_agg
           tuning metrics — includes the short prompt prefill); per-stream decode tok/s = median of (n-1)/(wall - TTFT). With --log, the lines
           added to hived.log during the run are counted: batched verifications ([mtp] batch S ... rows), batch rejections ([mtp] batch S ...
           no draft) and single verifications ([mtp] gate2 k / [mtp] pos ... verify).
  parity — greedy equivalence at temperature 0 with thinking off. Six prompts are run alone twice (A and B — the noise floor of the same
           server), then the same prompts are run concurrently as c2 pairs and a c4 group, and each output is compared character by character
           with solo run A (equal, position of the first differing character). The verdict comes from **running the same measurement on a
           HIVE_MTP_BATCH=0 server as well**: batched decode (even without MTP) can differ from solo decode in low-order bits depending on CPU/GPU
           split timing and the row set, so tokens may diverge (docs/performance.md: "sequential decode also shows rel 0.039 on the same input
           run twice"). Lossless batched MTP means "match rate and divergence positions with it on have the same distribution as with it off";
           bit-level losslessness (row argmax, greedy accepted count, rollback continuation) is covered by the model test `hive --verify2-test`
           (HIVE_MTP_BATCH=1, partial batch case).

Usage (the server is started separately for each configuration):
  bench_mtp_batch.py --base http://127.0.0.1:8430 --label batch1 --out <out-dir>/mtpbatch/bench.jsonl \\
      --log <log-dir>/hived.log [--conc 1,2,3,4] [--tokens 256] [--repeat 2] [--parity] [--skip-perf]
  bench_mtp_batch.py --self-test   # CPU self-test (comparison and aggregation functions)
"""
import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import uuid

PARITY_PROMPTS = [
    "Explain step by step how a refrigerator moves heat from inside to outside. Use plain language.",
    "대한민국의 사계절이 생활에 주는 영향을 다섯 문단으로 자세히 설명해 줘.",
    "Write a Python function that merges two sorted lists into one sorted list, then explain its complexity.",
    "Tell a long story about a lighthouse keeper who finds a message in a bottle during a storm.",
    "List ten practical tips for learning a new language as an adult, with one sentence of explanation each.",
    "커피 원두의 로스팅 단계별 맛 차이를 표 없이 문장으로 정리해 줘.",
]
RE_BATCH_VERIFY = re.compile(r"^\[mtp\] batch S (\d+) rows (\d+) ")
RE_BATCH_NODRAFT = re.compile(r"^\[mtp\] batch S (\d+) draft [\d.]+ ms · no draft")
RE_SINGLE_VERIFY = re.compile(r"^\[mtp\] (gate2 k \d+|pos \d+ draft [\d.]+ ms · verify)")


def stream(base, prompt, max_tokens, timeout=1800):
    import requests  # the self-test runs without requests
    msgs = [{"role": "user", "content": prompt}]
    body = {"model": "hive", "messages": msgs, "max_tokens": max_tokens, "temperature": 0.0,
            "reasoning_effort": "none", "stream": True, "user": "mtpbatch-" + uuid.uuid4().hex[:12]}
    t0 = time.time()
    first = None
    usage = {}
    text = ""
    err = None
    try:
        r = requests.post(base + "/v1/chat/completions", json=body, stream=True, timeout=timeout)
        if r.status_code != 200:
            return {"ttft": None, "wall": time.time() - t0, "n": 0, "tps": None, "text": "", "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            d = (ev.get("choices") or [{}])[0].get("delta", {})
            piece = d.get("content") or d.get("reasoning_content")
            if piece:
                first = first or time.time() - t0
                text += piece
            if ev.get("usage"):
                usage = ev["usage"]
    except Exception as e:  # noqa: BLE001 — measurement tool: a failure is still recorded as one line
        err = repr(e)
    wall = time.time() - t0
    n = usage.get("completion_tokens") or 0
    tps = (n - 1) / (wall - first) if first and n > 1 and wall > first else None
    return {"ttft": first, "wall": wall, "n": n, "tps": tps, "text": text, "error": err}


def run_group(base, prompts, max_tokens):
    """Run the prompts concurrently (one thread each) — returns (result list, wall clock)."""
    out = [None] * len(prompts)

    def one(i):
        out[i] = stream(base, prompts[i], max_tokens)
    ths = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return out, time.time() - t0


def log_mark(path):
    try:
        return os.path.getsize(path) if path else None
    except OSError:
        return None


def log_counts(path, off):
    """MTP counts from the lines added to hived.log after offset off (from the start if the log rotated and shrank)."""
    c = {"batch_verify": 0, "batch_nodraft": 0, "single_verify": 0, "batch_rows": {}}
    if not path or off is None:
        return c
    try:
        size = os.path.getsize(path)
        start = off if size >= off else 0
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return c
    return count_lines(data.splitlines(), c)


def count_lines(lines, c=None):
    c = c or {"batch_verify": 0, "batch_nodraft": 0, "single_verify": 0, "batch_rows": {}}
    for ln in lines:
        m = RE_BATCH_VERIFY.match(ln)
        if m:
            c["batch_verify"] += 1
            k = f"S{m[1]}r{m[2]}"
            c["batch_rows"][k] = c["batch_rows"].get(k, 0) + 1
        elif RE_BATCH_NODRAFT.match(ln):
            c["batch_nodraft"] += 1
        elif RE_SINGLE_VERIFY.match(ln):
            c["single_verify"] += 1
    return c


def first_diff(a, b):
    """Position of the first differing character (-1 if equal)."""
    if a == b:
        return -1
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def perf(a, emit):
    for c in [int(x) for x in a.conc.split(",") if x.strip()]:
        for rep in range(a.repeat):
            tag = uuid.uuid4().hex[:6]
            prompts = [f"[{tag}] Stream {k} of {c}: tell a long story about city number {k}. " for k in range(c)]
            off = log_mark(a.log)
            res, wall = run_group(a.base, prompts, a.tokens)
            time.sleep(0.5)  # until the last [mtp]/[cache] lines reach the log
            lc = log_counts(a.log, off)
            tps = [r["tps"] for r in res if r["tps"]]
            row = {"kind": "perf", "label": a.label, "c": c, "rep": rep, "agg_tok_s": round(sum(r["n"] for r in res) / wall, 2) if wall > 0 else None,
                   "stream_tps_med": round(statistics.median(tps), 2) if tps else None, "stream_tps": [round(x, 2) for x in tps],
                   "ttft_max": round(max((r["ttft"] or 0) for r in res), 3), "wall": round(wall, 2), "tokens": [r["n"] for r in res],
                   "errors": [r["error"] for r in res if r["error"]], **lc}
            emit(row)


def parity(a, emit):
    P = PARITY_PROMPTS
    solo_a = [stream(a.base, p, a.parity_tokens) for p in P]
    solo_b = [stream(a.base, p, a.parity_tokens) for p in P]
    groups = {"c2": [[0, 1], [2, 3], [4, 5]], "c4": [[0, 1, 2, 3], [2, 3, 4, 5]]}
    res = {"solo_b": [(i, solo_b[i]) for i in range(len(P))]}
    logc = {}
    for name, gs in groups.items():
        off = log_mark(a.log)
        res[name] = []
        for g in gs:
            out, _ = run_group(a.base, [P[i] for i in g], a.parity_tokens)
            res[name] += list(zip(g, out))
        time.sleep(0.5)
        logc[name] = log_counts(a.log, off)
    summary = summarize_parity(solo_a, res)
    for name in res:
        emit({"kind": "parity", "label": a.label, "set": name, **summary[name], **(logc.get(name) or {})})


def summarize_parity(solo_a, res):
    out = {}
    for name, pairs in res.items():
        same, diffs, errs = 0, [], 0
        for i, r in pairs:
            if r["error"] or solo_a[i]["error"]:
                errs += 1
                continue
            d = first_diff(r["text"], solo_a[i]["text"])
            if d < 0:
                same += 1
            else:
                diffs.append({"prompt": i, "first_diff_char": d, "len_ref": len(solo_a[i]["text"]), "len": len(r["text"])})
        out[name] = {"n": len(pairs), "identical": same, "errors": errs, "diffs": diffs}
    return out


def self_test():
    ok = True

    def check(c, what):
        nonlocal ok
        print(("PASS " if c else "FAIL ") + what)
        ok &= bool(c)
    check(first_diff("abc", "abc") == -1, "first_diff identical")
    check(first_diff("abcd", "abXd") == 2, "first_diff middle")
    check(first_diff("ab", "abc") == 2, "first_diff prefix")
    lines = ["[mtp] batch S 2 rows 5 · accepted [1/1,2/2] · expect 4.10 tok / 40.1 ms (T(S) 29.0) · got 5 tok / 41.0 ms · draft 7.0 ms · net 3.0 ms",
             "[mtp] batch S 2 draft 7.1 ms · no draft (T(S) 29.0 · T(S+1) 60.0 ms · skip 1)",
             "[mtp] gate2 k 2 · expect 2.34 tok / 35.2 ms (T1 20.5) · got 2 tok / 30.5 ms · net 13.5 ms",
             "[mtp] pos 100 draft 3.0 ms · gate2 no draft (conf0 1.00 · T1 20.0 T2 30.0 T6 60.0 ms · skip 1)",
             "[cache] step 1 M=2 routed 480 hit 400 cpu 80 streamed 0 · resident 1 pending 0 / 2"]
    c = count_lines(lines)
    check(c["batch_verify"] == 1 and c["batch_rows"] == {"S2r5": 1}, f"count batch verify {c}")
    check(c["batch_nodraft"] == 1 and c["single_verify"] == 1, f"count nodraft/single {c}")
    sa = [{"text": "hello world", "error": None}, {"text": "x", "error": None}]
    s = summarize_parity(sa, {"c2": [(0, {"text": "hello world", "error": None}), (1, {"text": "y", "error": None})]})
    check(s["c2"]["identical"] == 1 and s["c2"]["diffs"][0]["first_diff_char"] == 0, f"summarize {s}")
    # Log rotation: if off is larger than the current size, start from the beginning
    import tempfile
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(lines[0] + "\n")
    try:
        check(log_counts(f.name, 10 ** 9)["batch_verify"] == 1, "log rotated → read from start")
        check(log_counts(f.name, os.path.getsize(f.name))["batch_verify"] == 0, "log offset → only new lines")
    finally:
        os.unlink(f.name)
    print("bench_mtp_batch self-test:", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8430")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="")
    ap.add_argument("--log", default="", help="hived.log path (counts batched/single verify lines — needs a server with HIVE_TRACE_MTP=1, the service launcher default)")
    ap.add_argument("--conc", default="1,2,3,4")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--parity-tokens", type=int, default=160)
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    fo = open(a.out, "a") if a.out else None

    def emit(row):
        row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if fo:
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
    stream(a.base, "Warm up: say hello in three languages.", 32)
    if not a.skip_perf:
        perf(a, emit)
    if a.parity:
        parity(a, emit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
