#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""bench_chat.py — the real-chat benchmark behind the headline numbers of README.md (DeepSeek and GLM alike).

Twelve different chat prompts (5 Korean explanations, 3 English explanations, 2 code, 2 creative — the kind of traffic the
maintainers send), answers up to --max-tokens (600), temperature 0, thinking off (reasoning_effort none; GLM maps it to its lowest
effort). Decode speed counts only the tokens after the first one over the time after the first token, so prompt processing is not
mixed in. Steps:
  c1: the twelve prompts one after another — total decode tokens / total decode time (also per kind), median per request, time to first token
  cN (N = 2, 4, 8, ...): N streams at once, each walking the prompt list from a different start — total tok/s over the wall time of the
      decode phase, per-stream median, time to first token median / max (queueing included: a server serves at most HIVE_MAX_BATCH)
  long prompts (--long): one request each with ~T tokens of documentation text — time to first token, prefill tok/s, decode after it
Prints one line per step and appends one JSON line to --out.

  python3 tools/bench_chat.py --base http://127.0.0.1:8430 --name ds --out run/bench/chat.jsonl
"""
import argparse, json, os, statistics, threading, time, urllib.request
from pathlib import Path

PROMPTS = [
    "파이썬으로 LRU 캐시를 구현하고 설명해줘.", "Explain how TCP congestion control works in detail.",
    "조선 시대 과거 제도의 구조와 변화를 설명해줘.", "Write a short story about a lighthouse keeper.",
    "행렬 곱셈을 GPU에서 빠르게 하는 방법을 설명해줘.", "Summarize the causes of the French Revolution.",
    "김치찌개 맛있게 끓이는 법 알려줘.", "Write a Rust function that parses a CSV line with quotes.",
    "양자역학의 불확정성 원리를 쉽게 설명해줘.", "Describe the water cycle for a 10-year-old.",
    "SQL 인덱스가 느려지는 경우를 예시로 설명해줘.", "Write a haiku sequence about autumn in Seoul.",
]
# category of each prompt (c1 also reports decode speed per category; the headline number is the total over all twelve)
CATEGORY = ["code", "en", "ko", "creative", "ko", "en", "ko", "code", "ko", "en", "ko", "creative"]

P = argparse.ArgumentParser()
P.add_argument("--base", default="http://127.0.0.1:8430")
P.add_argument("--name", required=True)
P.add_argument("--out", default=os.path.join(os.environ.get("HIVE_STATE_DIR", "run"), "bench", "chat.jsonl"))
P.add_argument("--max-tokens", type=int, default=600)
P.add_argument("--conc", default="1,2,4,8,16,32", help="concurrency levels")
P.add_argument("--long", default="17000,42000,54000", help="approximate prompt sizes in tokens for the long-prompt step ('' = skip)")
A = P.parse_args()


def stream(prompt, max_tokens):
    body = json.dumps({"model": "hive", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
                       "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}).encode()
    t0 = time.time(); first = None; n = 0; usage = None
    req = urllib.request.Request(A.base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    first = first or time.time()
                    n += 1
            usage = d.get("usage") or usage
    t1 = time.time()
    if usage:
        n = usage.get("completion_tokens", n)
    first = first or t1
    return {"ttft": first - t0, "n": n, "t_first": first, "t_end": t1, "dec_tps": (n - 1) / (t1 - first) if n > 1 and t1 > first else 0.0,
            "prompt": (usage or {}).get("prompt_tokens")}


res = {"name": A.name, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "max_tokens": A.max_tokens}
for N in [int(x) for x in A.conc.split(",") if x.strip()]:
    out = []
    lock = threading.Lock()

    def worker(start, count):
        for j in range(count):
            i = (start + j) % len(PROMPTS)
            r = stream(PROMPTS[i], A.max_tokens)
            r["cat"] = CATEGORY[i]
            with lock:
                out.append(r)

    per = len(PROMPTS) if N == 1 else max(1, len(PROMPTS) // N + (1 if N < len(PROMPTS) else 0))
    th = [threading.Thread(target=worker, args=(i * 3, per)) for i in range(N)]
    [t.start() for t in th]
    [t.join() for t in th]
    dec_tok = sum(max(0, r["n"] - 1) for r in out)
    if N == 1:
        total = dec_tok / sum(r["t_end"] - r["t_first"] for r in out)
    else:  # wall time of the decode phase: first token of the earliest stream to the end of the last
        total = dec_tok / (max(r["t_end"] for r in out) - min(r["t_first"] for r in out))
    row = {"total_tps": round(total, 1), "per_stream_median": round(statistics.median(r["dec_tps"] for r in out), 1),
           "ttft_median": round(statistics.median(r["ttft"] for r in out), 2), "ttft_max": round(max(r["ttft"] for r in out), 2), "requests": len(out)}
    if N == 1:
        for cat in ("ko", "en", "code", "creative"):
            rs = [r for r in out if r["cat"] == cat]
            row[cat] = round(sum(max(0, r["n"] - 1) for r in rs) / sum(r["t_end"] - r["t_first"] for r in rs), 1)
    res[f"c{N}"] = row
    print(f"c{N}: total {row['total_tps']} tok/s · per stream {row['per_stream_median']} · TTFT median {row['ttft_median']} s (max {row['ttft_max']}) · {len(out)} requests"
          + (f" · by kind: Korean {row['ko']} · English {row['en']} · code {row['code']} · creative {row['creative']}" if N == 1 else ""), flush=True)
if A.long.strip():
    corpus = "".join(p.read_text() for p in sorted((Path(__file__).resolve().parents[1] / "docs").glob("*.md"))) * 20
    for T in [int(x) for x in A.long.split(",") if x.strip()]:
        r = stream("Read the following and then summarize its first section in two sentences.\n\n" + corpus[: int(T * 3.5)], 256)
        res[f"long{T}"] = {"prompt_tokens": r["prompt"], "ttft": round(r["ttft"], 2), "prefill_tps": round((r["prompt"] or 0) / r["ttft"]) if r["ttft"] else None,
                           "decode_tps": round(r["dec_tps"], 1)}
        print(f"long ~{T}: prompt {r['prompt']} tokens · TTFT {r['ttft']:.2f} s · prefill {res[f'long{T}']['prefill_tps']} tok/s · decode after it {r['dec_tps']:.1f} tok/s", flush=True)
Path(A.out).parent.mkdir(parents=True, exist_ok=True)
with open(A.out, "a") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
