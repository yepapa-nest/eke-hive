#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Decode performance benchmark — sends (1) a single stream and (2) N concurrent streams to the hive server (:8430) and prints
tok/s and cache hits.
Usage: bench_decode.py [--base http://127.0.0.1:8430] [--tokens 256] [--streams 4] [--prompt-words 300]"""
import argparse
import json
import threading
import time

import requests

WORDS = ("The river bends past the old mill where the miller once ground wheat for the whole valley , and every spring the water rises "
         "until the stones of the bridge disappear under a brown torrent that carries branches , leaves and the occasional lost boot . ")


def run(base, prompt, tokens, tag, out):
    msgs = [{"role": "user", "content": prompt}]
    body = {"model": "hive", "messages": msgs, "max_tokens": tokens, "temperature": 0.0,
            "reasoning_effort": "none"}
    t0 = time.time()
    r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=3600)
    dt = time.time() - t0
    if r.status_code != 200:
        out[tag] = {"wall": dt, "n": 0, "prefill_ms": 0, "decode_ms": 0, "tps": 0.0, "hit": None, "cpu": None, "cpu_wait_ms": 0, "batch_rows": None,
                    "text": f"HTTP {r.status_code}: {r.text[:120]}"}
        return
    j = r.json()
    u, h = j.get("usage", {}) or {}, {k: (v if v is not None else 0) for k, v in (j.get("hive", {}) or {}).items()}
    n = u.get("completion_tokens", 0)
    tps = (n - 1) * 1000 / max(1.0, h.get("decode_ms", 1)) if n > 1 else 0.0
    msg = j["choices"][0]["message"]
    out[tag] = {"wall": dt, "n": n, "prefill_ms": h.get("prefill_ms", 0), "decode_ms": h.get("decode_ms", 0), "tps": tps,
                "hit": h.get("decode_hit"), "cpu": h.get("decode_cpu"), "cpu_wait_ms": h.get("cpu_wait_ms", 0), "batch_rows": h.get("batch_rows"),
                "text": (msg.get("content") or "")[:80].replace("\n", " ")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8430")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--prompt-words", type=int, default=300)
    a = ap.parse_args()
    prompt = (WORDS * 20)[: a.prompt_words * 6] + "\n\nContinue this story for a long while."
    print("== single stream (2 runs: the second one after the cache has warmed up)")
    for rep in range(2):
        out = {}
        run(a.base, prompt + f" ({rep})", a.tokens, "s", out)
        o = out["s"]
        print(f"  #{rep}: {o['n']} tok · prefill {o['prefill_ms']:.0f} ms · decode {o['decode_ms']:.0f} ms → {o['tps']:.1f} tok/s · "
              f"hit {o['hit']} cpu {o['cpu']} (cpu wait {o['cpu_wait_ms']:.0f} ms) · {o['text']!r}")
    print(f"== {a.streams} concurrent streams")
    out = {}
    ths = [threading.Thread(target=run, args=(a.base, prompt + f" [{i}]", a.tokens, f"c{i}", out)) for i in range(a.streams)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    tot = 0
    for i in range(a.streams):
        o = out[f"c{i}"]
        tot += o["n"]
        print(f"  c{i}: {o['n']} tok · prefill {o['prefill_ms']:.0f} ms · decode {o['decode_ms']:.0f} ms → {o['tps']:.1f} tok/s · "
              f"batch {o['batch_rows']} · hit {o['hit']} cpu {o['cpu']}")
    print(f"  total {tot} tok / {wall:.1f} s = {tot / wall:.1f} tok/s (wall clock, prefill included)")


if __name__ == "__main__":
    main()
