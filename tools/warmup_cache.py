#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Expert VRAM cache warm-up — right after a restart (resident 0), generate from varied prompts to fill the cache (a step before a full benchmark battery).

The hived cache fills only through score-based promotion (promote N per token · warm_cache ≤2048 after a prefill ≥64). Feeding a single kind of text fills only
the experts that text calls, so languages, domains and thinking modes are mixed. Stop condition = the last `[cache] … resident R` in hived.log reaches the target (+ the hit rate of the last 200 steps with --target-hit) or the time limit.
Measured: after a prefill, warm_cache brings resident to 4,139 within 1.5 minutes — that means the slots are full, not that the hot set has been gathered
(decode hit rate at that point 0.55 · ~33 tok/s summed over 4 rows). Warmth is judged by hit rate (--target-hit).

Usage: python3 warmup_cache.py [--base http://127.0.0.1:8430] [--log <log dir>/hived.log]
          [--target 3900] [--max-min 45] [--conc 4] [--tokens 320]
Output: one line per request (hit rate, tok/s) + total time and final resident at the end. JSON summary via --out.
"""
import argparse
import itertools
import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

# hived.log of scripts/hive-start.sh: $HIVE_STATE_DIR/logs, otherwise run/logs in the repository
DEFAULT_LOG = os.path.join(os.environ.get("HIVE_STATE_DIR") or str(Path(__file__).resolve().parents[1] / "run"), "logs", "hived.log")

PROMPTS = [
    "서울의 대중교통 체계를 역사·노선·요금 관점에서 자세히 설명해 줘.",
    "한국 전통 음식 김치의 종류와 지역별 차이를 표로 정리하고 각 특징을 설명해 줘.",
    "Explain how a B-tree index works in a relational database, including insertion and node splitting.",
    "Write a Python function that parses an nginx access log line with a regex and aggregates status codes per hour. Include tests.",
    "다음 연립방정식을 풀어라: 3x + 2y = 16, 5x - y = 5. 풀이 과정을 단계별로 보여라.",
    "Summarize the causes and consequences of the 2008 global financial crisis for a university economics class.",
    "Rust로 스레드 안전한 LRU 캐시를 구현하고 동작 원리를 한국어로 설명해 줘.",
    "请用中文介绍长城的历史，以及它在今天的旅游意义。",
    "日本の四季と、それぞれの季節の代表的な行事について説明してください。",
    "Write a short science-fiction story about a lighthouse keeper on Europa who receives a signal from Earth.",
    "SQL: 고객 테이블과 주문 테이블이 있을 때 월별 재구매율을 계산하는 쿼리를 작성하고 인덱스 설계를 제안해.",
    "Derive the formula for the sum of a geometric series and prove convergence for |r|<1.",
    "Translate into natural English and explain nuances: '눈치가 빠르다', '정이 들다', '한이 맺히다'.",
    "Write a bash script that rotates logs older than 7 days, compresses them with zstd, and keeps at most 30 archives.",
    "양자컴퓨터의 큐비트, 중첩, 얽힘을 고등학생이 이해할 수 있게 비유로 설명해 줘.",
    "Create a JSON schema for a library management system with books, members, loans and fines, then give an example document.",
    "Explain the difference between TCP congestion control algorithms Reno, CUBIC and BBR.",
    "조선 시대 과거 제도의 구조와 사회적 영향을 논술형으로 서술하라.",
    "Write a JavaScript React component for a sortable, filterable table with pagination.",
    "What are the main mechanisms of antibiotic resistance in bacteria? Give examples.",
    "한국어 맞춤법: '되/돼', '안/않', '로서/로써'의 구분을 예문과 함께 설명해 줘.",
    "Solve: A train leaves at 9:40 at 72 km/h, another at 10:10 at 90 km/h from the same station. When does the second catch up?",
    "Write a C++ template for a fixed-size ring buffer with lock-free single-producer single-consumer semantics.",
    "제주도 3박 4일 가족 여행 일정을 날짜별로 짜 주고 예산도 추정해 줘.",
    "Explain transformers' attention mechanism, including multi-head attention and positional encodings, with equations.",
    "Écris un court poème en français sur l'automne à Paris, puis explique les images utilisées.",
    "Kubernetes에서 파드가 CrashLoopBackOff 상태일 때 원인 진단 절차를 단계별로 알려 줘.",
    "Describe the plot and themes of Dostoevsky's 'Crime and Punishment'.",
    "Go 언어로 HTTP 서버를 만들고 요청마다 지연 시간을 측정해 프로메테우스 지표로 내보내는 코드를 작성해.",
    "List 10 interview questions for a senior backend engineer and give model answers for three of them.",
]


def _log_tail(log_path, nbytes):
    """Last nbytes of the log decoded, cut to the latest daemon start; None if unreadable."""
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - nbytes))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    start = tail.rfind("[hived] listening on")   # if the daemon restarted, keep only the part after it
    return tail[start:] if start >= 0 else tail


def last_resident(log_path):
    tail = _log_tail(log_path, 200_000)
    if tail is None:
        return None
    # the last of the [cache] lines (trace) or request-completion lines (`resident R/S`) — the MTP path does not print [cache] lines
    m = [(x.start(), int(x.group(1)), int(x.group(2))) for x in re.finditer(r"resident (\d+) pending \d+ / (\d+)", tail)]
    m += [(x.start(), int(x.group(1)), int(x.group(2))) for x in re.finditer(r"resident (\d+)/(\d+) · ", tail)]
    if not m:
        return None
    _, r, n = max(m)
    return (r, n)


def recent_hit(log_path, steps=200):
    """Instantaneous hit rate from the last N `[cache] step … routed R hit H` lines at the end of hived.log (decode steps only — summed regardless of M)."""
    tail = _log_tail(log_path, 400_000)
    if tail is None:
        return None
    m = re.findall(r"\[cache\] step \d+ M=\d+ routed (\d+) hit (\d+)", tail)[-steps:]
    r = sum(int(a) for a, _ in m)
    if r:
        return sum(int(b) for _, b in m) / r
    mm = re.findall(r"^\[mtp\] .* hit (\d+) cpu (\d+)$", tail, re.M)[-steps:]   # MTP path: hit/cpu of the verify steps
    t = sum(int(h) + int(c) for h, c in mm)
    return sum(int(h) for h, _ in mm) / t if t else None


def one(base, prompt, tokens, think, out, lock):
    msgs = [{"role": "user", "content": prompt}]
    body = {"model": "hive", "messages": msgs, "max_tokens": tokens, "temperature": 0.7,
            "top_p": 0.95, "seed": int(time.time() * 1000) % 100000}
    body["reasoning_effort"] = "low" if think else "none"
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        j = json.load(urllib.request.urlopen(req, timeout=1800))
    except Exception as e:  # noqa
        with lock:
            out.append({"err": str(e)[:200]})
            print(f"  ERR {e}", flush=True)
        return
    h = j.get("hive") or {}
    n = (j.get("usage") or {}).get("completion_tokens", 0)
    hit, cpu = h.get("decode_hit") or 0, h.get("decode_cpu") or 0
    rate = hit / (hit + cpu) if hit + cpu else 0
    tps = (n - 1) * 1000 / h["decode_ms"] if n > 1 and h.get("decode_ms") else 0
    rec = {"wall": time.time() - t0, "n": n, "hit_rate": rate, "tps": tps, "prefill_ms": h.get("prefill_ms"), "rows": h.get("batch_rows"), "think": think}
    with lock:
        out.append(rec)
        print(f"  {time.strftime('%H:%M:%S')} n={n} hit={rate:.3f} tps={tps:.1f} rows={h.get('batch_rows')} think={think} · {prompt[:30]!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8430")
    ap.add_argument("--log", default=DEFAULT_LOG, help="hived.log (default: $HIVE_STATE_DIR/logs/hived.log or run/logs/hived.log)")
    ap.add_argument("--target", type=int, default=0, help="resident experts to reach (0 = 95 %% of the cache slots reported by the daemon log)")
    ap.add_argument("--max-min", type=float, default=45)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=320)
    ap.add_argument("--out", default="")
    ap.add_argument("--target-hit", type=float, default=0.0, help="also require the hit rate of the last 200 steps to reach this value (0 = only resident counts)")
    ap.add_argument("--min-rounds", type=int, default=0)
    a = ap.parse_args()
    if a.target <= 0:
        r0 = last_resident(a.log)
        a.target = int(0.95 * r0[1]) if r0 and r0[1] else 3900
    t0 = time.time()
    out, lock = [], threading.Lock()
    print(f"start resident={last_resident(a.log)}", flush=True)
    cyc = itertools.cycle(enumerate(PROMPTS * 4))
    rnd = 0
    while True:
        r = last_resident(a.log)
        h = recent_hit(a.log)
        if r and r[0] >= a.target and (a.target_hit <= 0 or (h or 0) >= a.target_hit) and rnd >= a.min_rounds:
            print(f"target reached: resident {r[0]}/{r[1]} · recent hit {h}", flush=True)
            break
        if (time.time() - t0) / 60 > a.max_min:
            print(f"time cap: resident {r}", flush=True)
            break
        ths = []
        for _ in range(a.conc):
            i, p = next(cyc)
            ths.append(threading.Thread(target=one, args=(a.base, p, a.tokens, (i + rnd) % 3 == 0, out, lock)))
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        rnd += 1
        print(f"[round {rnd}] {(time.time() - t0) / 60:.1f} min · resident {last_resident(a.log)} · recent hit {recent_hit(a.log)}", flush=True)
    summ = {"minutes": (time.time() - t0) / 60, "final_resident": last_resident(a.log), "recent_hit": recent_hit(a.log), "requests": len(out),
            "last8_hit_rate": [round(x.get("hit_rate", 0), 3) for x in out[-8:]], "last8_tps": [round(x.get("tps", 0), 1) for x in out[-8:]]}
    print(json.dumps(summ, ensure_ascii=False), flush=True)
    if a.out:
        json.dump({"summary": summ, "requests": out}, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
