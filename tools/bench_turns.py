#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""bench_turns.py — the chat-traffic shape (docs/performance.md steps 18–20) that stalls decoders: N streaming decoders are running, then follow-up turns of long,
already-cached conversations arrive (a few hundred new tokens on a 60K+ prompt) and one fresh mid-size prompt. Measured per run:
  decoders: tok/s over the run, chunk-gap p50/p95/max while the turns are in flight;
  turns: total time (send -> complete, 12 output tokens), engine prefill_ms, cached_prefix, the answer (a word only that turn knows).
Usage: bench_turns.py --label NAME [--decoders 4] [--turns 3] [--new-tokens 420] [--history-tokens 60000] [--out tune.jsonl]
Run against the live server; each conversation's first request prefills its history once (warm) before the measured part.
"""
import argparse, json, os, random, threading, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument("--base", default="http://127.0.0.1:8430")
P.add_argument("--label", required=True)
P.add_argument("--decoders", type=int, default=4)
P.add_argument("--decoder-tokens", type=int, default=400)
P.add_argument("--turns", type=int, default=3)
P.add_argument("--turn-gap", type=float, default=6.0)
P.add_argument("--new-tokens", type=int, default=420)
P.add_argument("--history-tokens", type=int, default=60000)
P.add_argument("--fresh-tokens", type=int, default=3000, help="one fresh (uncached) prompt sent after the turns; 0 = none")
P.add_argument("--behind-tokens", type=int, default=30000, help="phase 2: a fresh long prompt; a follow-up turn is sent 1 s after it (0 = skip)")
P.add_argument("--out", default=os.path.join(os.environ.get("HIVE_STATE_DIR", "run"), "bench", "turns.jsonl"))
A = P.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split()
def text(n_tokens, seed):
    r = random.Random(seed); return " ".join(r.choice(WORDS) for _ in range(n_tokens))

def post(body):
    req = urllib.request.Request(A.base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", "x-client-test": "bench_turns"})
    return urllib.request.urlopen(req, timeout=1800)

def stream_chat(messages, max_tokens, session, prog, ignore_eos=False):
    body = {"model": "hive", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "none", "stream": True,
            "hive_session_id": session, **({"ignore_eos": True} if ignore_eos else {})}
    prog.update({"t0": time.time(), "first": None, "times": [], "usage": {}, "wall": None, "error": None})
    try:
        with post(body) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]": continue
                ev = json.loads(line[6:])
                for ch in ev.get("choices", []):
                    if ch.get("delta", {}).get("content"):
                        now = time.time(); prog["first"] = prog["first"] or now; prog["times"].append(now)
                if ev.get("usage"): prog["usage"] = ev["usage"]
    except Exception as e:
        prog["error"] = str(e)[:200]
    prog["wall"] = time.time() - prog["t0"]

def nonstream(messages, max_tokens, session):
    body = {"model": "hive", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "none", "stream": False, "hive_session_id": session}
    t0 = time.time()
    with post(body) as r:
        j = json.loads(r.read())
    return j, time.time() - t0

# --- conversations: history + first turn (warm the session state once) ---------------------------------------------------------------
convs = []
for i in range(A.turns):
    hist = [{"role": "system", "content": "Answer with one word."},
            {"role": "user", "content": "Notes: " + text(A.history_tokens, 100 + i)}, {"role": "assistant", "content": "Noted."}]
    sid = f"turns-{A.label}-{i}"
    j, el = nonstream(hist + [{"role": "user", "content": "Say READY."}], 8, sid)
    ans = (j.get("choices") or [{}])[0].get("message", {}).get("content", "")
    convs.append({"sid": sid, "hist": hist + [{"role": "user", "content": "Say READY."}, {"role": "assistant", "content": ans}], "warm_s": round(el, 2)})
print(json.dumps({"warm": [(c["sid"], c["warm_s"]) for c in convs]}), flush=True)
time.sleep(2)

# --- decoders -------------------------------------------------------------------------------------------------------------------------
dec = [{} for _ in range(A.decoders)]
th = [threading.Thread(target=stream_chat, args=([{"role": "user", "content": f"Decoder {k}: tell a very long story about city number {k}."}], A.decoder_tokens, f"dec-{A.label}-{k}", dec[k], True)) for k in range(A.decoders)]
[t.start() for t in th]
deadline = time.time() + 120
while time.time() < deadline and not all(d.get("first") for d in dec): time.sleep(0.2)
time.sleep(3.0)

# --- follow-up turns (new context block + question), then one fresh prompt ----------------------------------------------------------
turns = []
t_turn0 = time.time()
for i, c in enumerate(convs):
    secret = f"{WORDS[i]}-{i}"
    msgs = c["hist"] + [{"role": "user", "content": "[Context] " + text(A.new_tokens - 20, 500 + i) + f" The secret word is {secret}."}, {"role": "user", "content": "What is the secret word? One word."}]
    t0 = time.time(); j, el = nonstream(msgs, 12, c["sid"])
    h = j.get("hive") or {}
    turns.append({"i": i, "sent_at": round(t0 - t_turn0, 2), "total_s": round(el, 2), "prefill_ms": round(h.get("prefill_ms") or 0), "cached_prefix": h.get("cached_prefix"),
                  "prompt_tokens": (j.get("usage") or {}).get("prompt_tokens"), "answer": (j.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()[:30], "want": secret})
    time.sleep(max(0.0, A.turn_gap - el))
fresh = None
if A.fresh_tokens:
    t0 = time.time(); j, el = nonstream([{"role": "user", "content": "Summarize in one word: " + text(A.fresh_tokens, 999)}], 12, f"fresh-{A.label}")
    fresh = {"total_s": round(el, 2), "prefill_ms": round((j.get("hive") or {}).get("prefill_ms") or 0), "prompt_tokens": (j.get("usage") or {}).get("prompt_tokens")}
t_turn1 = time.time()
# --- phase 2: a follow-up turn that arrives 1 s after a long fresh prompt started (it waits for that prefill unless admitted by a yield) --
behind = None
if A.behind_tokens:
    c = convs[0]
    msgs = c["hist"] + [{"role": "user", "content": "[Context] " + text(A.new_tokens - 20, 777) + " The secret word is behind-0."}, {"role": "user", "content": "What is the secret word? One word."}]
    longp = {}
    tl = threading.Thread(target=stream_chat, args=([{"role": "user", "content": "Summarize in one word: " + text(A.behind_tokens, 888)}], 12, f"behind-long-{A.label}", longp))
    t0 = time.time(); tl.start(); time.sleep(1.0)
    t1 = time.time(); j, el = nonstream(msgs, 12, c["sid"]); tl.join()
    h = j.get("hive") or {}
    behind = {"long_prompt_tokens": (longp.get("usage") or {}).get("prompt_tokens"), "long_ttft_s": round((longp["first"] - longp["t0"]), 2) if longp.get("first") else None,
              "turn_sent_after_s": round(t1 - t0, 2), "turn_total_s": round(el, 2), "turn_prefill_ms": round(h.get("prefill_ms") or 0), "turn_cached": h.get("cached_prefix"),
              "turn_answer": (j.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()[:30]}
[t.join() for t in th]

# --- decoder metrics ----------------------------------------------------------------------------------------------------------------
def gaps(times, a, b):
    g = sorted(t2 - t1 for t1, t2 in zip(times, times[1:]) if a <= t2 <= b)
    return {"n": len(g), "p50": round(g[len(g)//2], 3) if g else None, "p95": round(g[int(.95*len(g))-1], 3) if len(g) >= 20 else None, "max": round(g[-1], 3) if g else None}
def tps(d):
    n = (d.get("usage") or {}).get("completion_tokens") or len(d.get("times", []))
    return round((n - 1) / (d["wall"] - (d["first"] - d["t0"])), 1) if d.get("first") and d.get("wall") and n > 1 else None
out = {"label": A.label, "decoders": A.decoders, "turns": turns, "fresh": fresh, "behind": behind,
       "decoder_tps": [tps(d) for d in dec], "decoder_errors": [d.get("error") for d in dec if d.get("error")],
       "decoder_gaps_during_turns": [gaps(d.get("times", []), t_turn0, t_turn1) for d in dec],
       "decoder_gaps_overall": [gaps(d.get("times", []), 0, 1e18) for d in dec]}
open(A.out, "a").write(json.dumps(out, ensure_ascii=False) + "\n")
print(json.dumps(out, ensure_ascii=False, indent=1))
