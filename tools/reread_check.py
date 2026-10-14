#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Re-read scenario (synthetic content): does the turn AFTER a long re-read resume, or re-read everything again?

Usage: reread_check.py PORT LABEL [N_TURNS] [OUT_DIR]   -> OUT_DIR/LABEL.json (default ./reread-results)
N_TURNS 150 makes a ~165-190K-token conversation (more than one prefill chunk on every family).
Case A (head change): an agent turn whose tool JSON keys come back in another order (head changes -> full re-read), then the next
  user message (the template drops the reasoning of that turn -> the tail changes at the previous user boundary).
Case B (engine lost the state): /flush_cache between two append-only steps of an agent turn (the server tracker still sees an
  append-only conversation), then the next user message.
Measured per request: TTFT and the server cached tokens (usage.prompt_tokens_details.cached_tokens).
"""
import json, os, random, sys, time, urllib.request, uuid

PORT, LABEL = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 150
OUT_DIR = sys.argv[4] if len(sys.argv) > 4 else "reread-results"
BASE = f"http://127.0.0.1:{PORT}"
WORDS = ("system cache prefix token engine session layer expert memory budget prompt window answer question context model family schedule "
         "weather report summary document table chart market price stock news search result page file").split()


def text(n, seed):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) + (str(r.randint(0, 99)) if r.random() < 0.2 else "") for _ in range(n))


def tools(order):
    out = []
    for i in range(12):
        p = {"type": "object", "properties": {"q": {"type": "string", "description": text(10, 300 + i)}}, "required": ["q"],
             "additionalProperties": False}
        if order:
            p = dict(reversed(list(p.items())))
        out.append({"type": "function", "function": {"name": f"tool_{i}", "description": text(30, 100 + i), "parameters": p}})
    return out


def ask(msgs, key, order=0):
    body = {"model": "x", "messages": msgs, "tools": tools(order), "max_tokens": 12, "stream": True, "reasoning_effort": "low",
            "prompt_cache_key": key, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = ((d.get("choices") or [{}])[0].get("delta") or {})
            if ttft is None and (ch.get("content") or ch.get("reasoning_content") or ch.get("tool_calls")):
                ttft = time.time() - t0
    return {"ttft_s": round(ttft if ttft is not None else time.time() - t0, 2), "prompt_tokens": usage.get("prompt_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


def history(seed):
    h = [{"role": "system", "content": f"[run {uuid.uuid4().hex[:12]}]\n" + text(600, seed)}]
    for k in range(N):
        h += [{"role": "user", "content": text(400, seed * 1000 + 2 * k)}, {"role": "assistant", "content": text(500, seed * 1000 + 2 * k + 1)}]
    return h


def step(i):  # one agent step: reasoning + a tool call, then the tool result
    return [{"role": "assistant", "content": "", "reasoning_content": text(60, 9000 + i),
             "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "tool_1", "arguments": json.dumps({"q": text(4, 9100 + i)})}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": text(300, 9200 + i)}]


res = {"label": LABEL, "port": PORT, "n_turns": N, "steps": []}


def rec(case, name, r, note=""):
    r = {"case": case, "step": name, **r, "note": note}
    res["steps"].append(r)
    print(f"{case} {name:26s} ttft {r['ttft_s']:7.2f} s  prompt {r['prompt_tokens']}  cached {r['cached']}  {note}", flush=True)


def flush():
    for _ in range(30):
        try:
            req = urllib.request.Request(BASE + "/flush_cache?timeout=60", method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode()[:120]
        except Exception as e:  # not idle yet
            last = str(e)
            time.sleep(2)
    return last


q1 = {"role": "user", "content": "question one " + text(20, 1)}
# ---- Case A: head change
key = "reread-a-" + uuid.uuid4().hex[:8]
H = history(11)
rec("A", "1 first request", ask(H + [q1], key))
rec("A", "2 agent step (append)", ask(H + [q1] + step(1), key))
rec("A", "3 head changed (re-read)", ask(H + [q1] + step(1) + step(2), key, order=1), "tool keys reordered")
done = H + [q1] + step(1) + step(2) + [{"role": "assistant", "content": text(80, 77)}]
rec("A", "4 next user message", ask(done + [{"role": "user", "content": "question two " + text(20, 2)}], key, order=1), "expect resume")
# ---- Case B: the engine lost the state (append-only for the server)
key = "reread-b-" + uuid.uuid4().hex[:8]
H = history(22)
rec("B", "1 first request", ask(H + [q1], key))
rec("B", "2 agent step (append)", ask(H + [q1] + step(1), key))
print("flush:", flush(), flush=True)
rec("B", "3 append after flush", ask(H + [q1] + step(1) + step(2), key), "engine state dropped")
done = H + [q1] + step(1) + step(2) + [{"role": "assistant", "content": text(80, 78)}]
rec("B", "4 next user message", ask(done + [{"role": "user", "content": "question two " + text(20, 3)}], key), "expect resume")
os.makedirs(OUT_DIR, exist_ok=True)
json.dump(res, open(os.path.join(OUT_DIR, f"{LABEL}.json"), "w"), indent=1)
