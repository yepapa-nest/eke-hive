#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Benchmark of short-request and in-flight decode latency during a long prefill (T3) — measured against a live hive server (OpenAI API).

Scenario (one run = one JSONL line):
  1) (unless --decoder 0) start one streaming decoder first and wait for its first token (ignore_eos, long max_tokens).
  2) at t=0 send a long prompt (--long-tokens; text cut from a fixed corpus + a per-run unique header — prevents session/prefix reuse).
  3) at t=+1 s, +3 s, ... (--short-at) send short requests one at a time (each in a new session).
  4) when everything has finished, cut the decoder.
Recorded: per-request TTFT (send time -> first content/reasoning chunk), prompt_tokens (usage), total time, hive.prefill_ms of the long request,
      decoder chunk gaps P50/P95/P99/max (overall, and during the long prefill [long request sent -> its first token]), number of gaps over 1 s, chunks received in that window.
      One server chunk may bundle several tokens (tag buffer) — gaps are "chunk gaps". daemon.build_id from /health is recorded too.
Baseline (--sequential): first measures, under the same conditions, the TTFT of a short request alone and of the long request alone (no decoder).

Switch matrix (server environment variables — requires a hive restart; start the server for each configuration and run this tool with --label): --print-matrix
Usage: bench_staggered.py --base http://127.0.0.1:8430 --corpus <corpus.txt> --long-tokens 100000 \
        --short-at 1,3,5,8 --label base --out staggered.jsonl [--repeat 2] [--sequential]
"""
import argparse
import json
import random
import statistics
import threading
import time
import uuid

import requests

MATRIX = [  # (label, server environment variables — everything else stays at the service launcher defaults)
    ("base", {}),
    ("p1a-window", {"HIVE_BATCH_WINDOW": "1"}),
    ("p1b-sjf", {"HIVE_BATCH_SJF": "1"}),
    ("p1c-fair", {"HIVE_PREFILL_FAIR": "1"}),
    ("p1-all", {"HIVE_BATCH_WINDOW": "1", "HIVE_BATCH_SJF": "1", "HIVE_PREFILL_FAIR": "1"}),
    ("t3-yield", {"HIVE_PREFILL_YIELD": "1"}),
    ("t3-yield-4k", {"HIVE_PREFILL_YIELD": "1", "HIVE_PREFILL_YIELD_MAX": "4096"}),
    ("t3-yield+p1", {"HIVE_PREFILL_YIELD": "1", "HIVE_BATCH_WINDOW": "1", "HIVE_BATCH_SJF": "1", "HIVE_PREFILL_FAIR": "1"}),
    # T11 per-layer yield (inside a long prefill forward) — period sweep (ms), also combined with T3
    ("t11-layer", {"HIVE_LAYER_YIELD": "1"}),
    ("t11-layer-250", {"HIVE_LAYER_YIELD": "250"}),
    ("t11-layer-1000", {"HIVE_LAYER_YIELD": "1000"}),
    ("t11-layer+t3", {"HIVE_LAYER_YIELD": "1", "HIVE_PREFILL_YIELD": "1"}),
]
SHORT_Q = ["What is 17*19? Return only the integer.", "Name the capital of France in one word.", "대한민국의 수도를 한 단어로 답해줘.",
           "Write one sentence about the sea.", "List three prime numbers.", "2의 10제곱은? 숫자만."]


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def gap_stats(ts):
    g = [(b - a) * 1000.0 for a, b in zip(ts, ts[1:])]
    return {"n_chunks": len(ts), "p50_ms": pct(g, 50), "p95_ms": pct(g, 95), "p99_ms": pct(g, 99), "max_ms": max(g) if g else None,
            "gaps_over_1s": sum(1 for x in g if x > 1000.0)}


class Stream:
    """One streaming request (thread). t_send, t_first (first content/reasoning/tool chunk), chunk times, usage, hive, error."""

    def __init__(self, base, messages, max_tokens, session, ignore_eos=False, timeout=3600):
        self.body = {"model": "hive", "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
                     "reasoning_effort": "none", "hive_session_id": session, "ignore_eos": ignore_eos}
        self.base, self.timeout = base, timeout
        self.t_send = self.t_first = self.t_end = None
        self.chunks, self.usage, self.error, self.finish = [], None, None, None
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.th.start()
        return self

    def _run(self):
        self.t_send = time.monotonic()
        try:
            with requests.post(f"{self.base}/v1/chat/completions", json=self.body, stream=True, timeout=self.timeout) as r:
                if r.status_code != 200:
                    self.error = f"HTTP {r.status_code}: {r.text[:300]}"
                    return
                for line in r.iter_lines(decode_unicode=True):
                    if self.stop.is_set():
                        break
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    d = json.loads(data)
                    if d.get("error"):
                        self.error = json.dumps(d["error"], ensure_ascii=False)[:300]
                        continue
                    if d.get("usage"):
                        self.usage = d["usage"]
                    for c in d.get("choices") or []:
                        dl = c.get("delta") or {}
                        if dl.get("content") or dl.get("reasoning_content") or dl.get("tool_calls"):
                            now = time.monotonic()
                            self.chunks.append(now)
                            if self.t_first is None:
                                self.t_first = now
                        if c.get("finish_reason"):
                            self.finish = c["finish_reason"]
        except Exception as e:  # noqa: BLE001 — benchmark: errors are only recorded
            if not self.stop.is_set():
                self.error = repr(e)[:300]
        finally:
            self.t_end = time.monotonic()

    def join(self, t=None):
        self.th.join(t)

    def ttft(self):
        return None if self.t_first is None or self.t_send is None else self.t_first - self.t_send


def long_text(corpus, n_chars, rng):
    c = corpus
    if len(c) < n_chars + 1:
        c = c * (n_chars // max(1, len(c)) + 2)
    off = rng.randrange(0, max(1, len(c) - n_chars))
    return c[off:off + n_chars]


def calibrate(base, corpus, rng):
    """Characters per token: sends 8,000 corpus characters once with max_tokens 1 and reads usage.prompt_tokens (short prefill, ~1–2 s)."""
    txt = long_text(corpus, 8000, rng)
    s = Stream(base, [{"role": "user", "content": f"[{uuid.uuid4().hex}]\n{txt}\n\nReply OK."}], 1, "cal-" + uuid.uuid4().hex).start()
    s.join()
    if s.error or not s.usage:
        raise SystemExit(f"calibration failed: {s.error or 'no usage in stream'}")
    return len(txt) / max(1, s.usage["prompt_tokens"] - 20)


def long_messages(corpus, n_tokens, cpt, rng):
    txt = long_text(corpus, int(n_tokens * cpt), rng)
    return [{"role": "user", "content": f"[run {uuid.uuid4().hex}]\n{txt}\n\nIn one sentence, what is the text above mainly about?"}]


def one_run(a, corpus, cpt, rng, run_idx):
    res = {"label": a.label, "run": run_idx, "long_tokens_target": a.long_tokens, "short_at_s": a.short_at, "decoder": bool(a.decoder)}
    try:
        res["build_id"] = requests.get(f"{a.base}/health", timeout=5).json().get("daemon", {}).get("build_id")
    except Exception:  # noqa: BLE001
        res["build_id"] = None
    dec = None
    if a.decoder:
        dec = Stream(a.base, [{"role": "user", "content": f"[dec {uuid.uuid4().hex}] Count upward from 1, one number per line, forever."}],
                     a.decoder_max_tokens, "dec-" + uuid.uuid4().hex, ignore_eos=True).start()
        deadline = time.monotonic() + 120
        while dec.t_first is None and dec.error is None and time.monotonic() < deadline:
            time.sleep(.05)
        time.sleep(a.decoder_warm)  # let a few steps accumulate as the baseline
        if dec.t_first is None:
            res["decoder_error"] = dec.error or "decoder produced no token in 120 s"
    msgs = long_messages(corpus, a.long_tokens, cpt, rng)
    t0 = time.monotonic()
    lng = Stream(a.base, msgs, a.long_max_tokens, "long-" + uuid.uuid4().hex).start()
    shorts = []
    for k, at in enumerate(a.short_at):
        while time.monotonic() < t0 + at:
            time.sleep(.005)
        q = SHORT_Q[k % len(SHORT_Q)]
        shorts.append((at, Stream(a.base, [{"role": "user", "content": f"[{uuid.uuid4().hex[:8]}] {q}"}], a.short_max_tokens,
                                  f"short{k}-" + uuid.uuid4().hex).start()))
    lng.join()
    for _, s in shorts:
        s.join()
    if dec:
        time.sleep(a.decoder_tail)
        dec.stop.set()
        dec.join(10)
    res["long"] = {"ttft_s": lng.ttft(), "total_s": (lng.t_end - lng.t_send) if lng.t_end else None,
                   "prompt_tokens": (lng.usage or {}).get("prompt_tokens"), "error": lng.error}
    res["shorts"] = [{"at_s": at, "ttft_s": s.ttft(), "first_token_at_s": (s.t_first - t0) if s.t_first else None,
                      "before_long_first": (s.t_first is not None and lng.t_first is not None and s.t_first < lng.t_first),
                      "prompt_tokens": (s.usage or {}).get("prompt_tokens"), "error": s.error} for at, s in shorts]
    if dec and dec.chunks:
        lo, hi = lng.t_send, lng.t_first or lng.t_end
        res["decoder_all"] = gap_stats(dec.chunks)
        res["decoder_during_long_prefill"] = gap_stats([x for x in dec.chunks if lo is not None and hi is not None and lo <= x <= hi])
        before = [x for x in dec.chunks if x < lo]
        inside = [x for x in dec.chunks if lo <= x <= hi]
        if before and hi:  # the gap straddling the window start + the gap remaining to the window end (if no chunk falls inside the window, this is the maximum stall)
            pts = [before[-1]] + inside + [hi]
            res["decoder_during_long_prefill"]["max_stall_ms"] = max((b - x) * 1000.0 for x, b in zip(pts, pts[1:]))
        res["decoder_error"] = res.get("decoder_error") or (dec.error if dec.error and "Read timed out" not in dec.error else None)
    return res


def fmt(r):
    L = r["long"]
    s = " · ".join(f"+{x['at_s']:g}s {x['ttft_s']:.2f}s" if x["ttft_s"] is not None else f"+{x['at_s']:g}s ERR" for x in r["shorts"])
    d = r.get("decoder_during_long_prefill") or {}
    dd = (f" · decode in long prefill: {d.get('n_chunks')} chunks p50 {d.get('p50_ms') or 0:.0f} p95 {d.get('p95_ms') or 0:.0f} "
          f"p99 {d.get('p99_ms') or 0:.0f} ms · max stall {d.get('max_stall_ms') or 0:.0f} ms") if d else ""
    return (f"[{r['label']} #{r['run']}] long {L['prompt_tokens']} tok TTFT {L['ttft_s'] or float('nan'):.2f}s · shorts TTFT {s}{dd}")


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--base", default="http://127.0.0.1:8430")
    cli.add_argument("--corpus", default=None, help="long text file the prompts are cut from (required unless --print-matrix)")
    cli.add_argument("--long-tokens", type=int, default=100000)
    cli.add_argument("--long-max-tokens", type=int, default=16)
    cli.add_argument("--short-at", default="1,3,5,8", help="times to send the short requests (seconds after the long request, comma-separated)")
    cli.add_argument("--short-max-tokens", type=int, default=16)
    cli.add_argument("--decoder", type=int, default=1, help="1 = start a streaming decoder first · 0 = none")
    cli.add_argument("--decoder-max-tokens", type=int, default=20000)
    cli.add_argument("--decoder-warm", type=float, default=2.0, help="seconds to wait after the decoder's first token before the long request")
    cli.add_argument("--decoder-tail", type=float, default=2.0, help="seconds to keep the decoder running after everything has finished")
    cli.add_argument("--chars-per-token", type=float, default=0.0, help="0 = calibrate once against the server (8,000 chars · max_tokens 1)")
    cli.add_argument("--repeat", type=int, default=1)
    cli.add_argument("--sequential", action="store_true", help="baseline first: short request alone · long request alone (no decoder)")
    cli.add_argument("--seed", type=int, default=20261001)
    cli.add_argument("--label", default="run")
    cli.add_argument("--out", default="")
    cli.add_argument("--print-matrix", action="store_true", help="print the server switch matrix (environment variables) and exit")
    a = cli.parse_args()
    if a.print_matrix:
        for label, env in MATRIX:
            print(f"{label:14s} " + (" ".join(f"{k}={v}" for k, v in env.items()) or "(service defaults unchanged)"))
        return
    if not a.corpus:
        cli.error("--corpus FILE is required")
    a.short_at = [float(x) for x in a.short_at.split(",") if x.strip()]
    with open(a.corpus, encoding="utf-8", errors="replace") as f:
        corpus = f.read()
    rng = random.Random(a.seed)
    cpt = a.chars_per_token or calibrate(a.base, corpus, rng)
    print(f"chars/token {cpt:.3f}", flush=True)
    rows = []
    if a.sequential:
        s = Stream(a.base, [{"role": "user", "content": f"[{uuid.uuid4().hex[:8]}] {SHORT_Q[0]}"}], a.short_max_tokens, "seqs-" + uuid.uuid4().hex).start()
        s.join()
        lg = Stream(a.base, long_messages(corpus, a.long_tokens, cpt, rng), a.long_max_tokens, "seql-" + uuid.uuid4().hex).start()
        lg.join()
        r = {"label": a.label, "run": "sequential", "short_ttft_s": s.ttft(), "long_ttft_s": lg.ttft(),
             "long_prompt_tokens": (lg.usage or {}).get("prompt_tokens"), "errors": [x for x in (s.error, lg.error) if x]}
        print(f"[{a.label} sequential] short TTFT {s.ttft() or float('nan'):.2f}s · long {r['long_prompt_tokens']} tok TTFT {lg.ttft() or float('nan'):.2f}s", flush=True)
        rows.append(r)
    for i in range(a.repeat):
        r = one_run(a, corpus, cpt, rng, i)
        print(fmt(r), flush=True)
        rows.append(r)
    runs = [r for r in rows if isinstance(r.get("run"), int)]
    if len(runs) > 1:
        med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731
        print(f"[{a.label} median of {len(runs)}] long TTFT {med([r['long']['ttft_s'] for r in runs if r['long']['ttft_s']]):.2f}s · "
              f"short TTFT {med([x['ttft_s'] for r in runs for x in r['shorts'] if x['ttft_s']]):.2f}s")
    if a.out:
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
