#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""reread_report.py — how much of each conversation was read again, and why, from the server request log (requests-server.jsonl).

A follow-up request of a conversation should only read what is new: the engine still holds the previous prompt (and its answer), so
everything the two prompts share is reused. The server records, per request (hive_server.py, TokenCache):
  prev_prompt_tokens = tokens of the same conversation's previous prompt
  diverge_at         = how many of those tokens this prompt keeps unchanged (a pure append keeps all of them)
and the engine reports done.cached_prefix (tokens reused). From them:
  re-read        = min(prev_prompt_tokens, prompt_tokens) − cached_prefix   (tokens already read once and read again)
  client rewrite = prev_prompt_tokens − diverge_at   — the client changed an earlier part of the conversation (history trimmed,
                   a message edited, reasoning dropped from an earlier step); the engine cannot reuse what was rewritten
  engine loss    = diverge_at − cached_prefix        — the prompt still shared that much, but the engine no longer held it
                   (evicted, restarted without the session store, another conversation took the slot)
Both are clamped at 0; up to SLACK tokens of rewrite are template noise (the last token of a prompt merging with the next message).
Read-ahead requests (endpoint turn-prerender, HIVE_TURN_PRERENDER) are not conversation turns; their own prefill is listed as the
cost of reading ahead.

usage: reread_report.py [--root DIR] [--day YYYY-MM-DD] [--top 10] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

SLACK = 8  # rewrite tokens counted as template noise, not as a client rewrite


def _num(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def classify(r):
    """One server record → None (not a follow-up turn with the divergence fields) or its re-read split."""
    prev, at, pt = _num(r.get("prev_prompt_tokens")), _num(r.get("diverge_at")), _num(r.get("prompt_tokens"))
    cached = _num((r.get("done") or {}).get("cached_prefix"))
    if prev is None or at is None or pt is None or cached is None:
        return None
    reread = max(0, min(prev, pt) - cached)
    rewrite = max(0, prev - at)
    loss = max(0, min(at, pt) - cached)
    if loss > SLACK:
        cause = "engine"
    elif rewrite > SLACK and reread > SLACK:
        cause = "client"
    else:
        cause = "ok"
    return {"session": r.get("session"), "prompt": pt, "prev": prev, "diverge_at": at, "cached": cached,
            "reread": reread, "rewrite": rewrite, "loss": loss, "cause": cause,
            "prefill_ms": (r.get("done") or {}).get("prefill_ms"), "t_recv": r.get("t_recv")}


def summarize_reread(server, top=10):
    """Per conversation: follow-up turns, tokens read again and the cause; plus what reading ahead cost."""
    turns = [c for c in (classify(r) for r in server if r.get("endpoint") != "turn-prerender") if c]
    pre = [r for r in server if r.get("endpoint") == "turn-prerender" and r.get("outcome") == "done"]
    by = {}
    for c in turns:
        s = by.setdefault(c["session"], {"session": c["session"], "turns": 0, "reread": 0, "client": 0, "engine": 0,
                                         "client_tokens": 0, "engine_tokens": 0, "max_prompt": 0})
        s["turns"] += 1
        s["reread"] += c["reread"]
        s["max_prompt"] = max(s["max_prompt"], c["prompt"])
        if c["cause"] == "client":
            s["client"] += 1; s["client_tokens"] += c["reread"]
        elif c["cause"] == "engine":
            s["engine"] += 1; s["engine_tokens"] += c["reread"]
    rooms = sorted(by.values(), key=lambda s: -s["reread"])
    worst = sorted((c for c in turns if c["cause"] != "ok"), key=lambda c: -c["reread"])[:top]
    return {"followups": len(turns), "reread_tokens": sum(c["reread"] for c in turns),
            "client_turns": sum(1 for c in turns if c["cause"] == "client"), "engine_turns": sum(1 for c in turns if c["cause"] == "engine"),
            "client_tokens": sum(c["reread"] for c in turns if c["cause"] == "client"),
            "engine_tokens": sum(c["reread"] for c in turns if c["cause"] == "engine"),
            "conversations": len(by), "rooms": rooms[:top], "worst": worst,
            "prerender": {"n": len(pre), "tokens": sum(_num((r.get("done") or {}).get("prefill_tokens")) or 0 for r in pre),
                          "prefill_s": round(sum(float((r.get("done") or {}).get("prefill_ms") or 0) for r in pre) / 1000.0, 1)}}


def render_md(R, f=lambda x: "–" if x is None else f"{x:,}"):
    L = ["## Re-read per conversation (server records)", ""]
    if not R["followups"]:
        return L + ["- no follow-up turns with divergence fields (prev_prompt_tokens / diverge_at)", ""]
    L += [f"- follow-up turns {f(R['followups'])} in {f(R['conversations'])} conversations · read again {f(R['reread_tokens'])} tokens · "
          f"client rewrite {f(R['client_turns'])} turns ({f(R['client_tokens'])} tokens) · engine loss {f(R['engine_turns'])} turns ({f(R['engine_tokens'])} tokens)",
          f"- reading ahead (turn-prerender): {f(R['prerender']['n'])} requests · {f(R['prerender']['tokens'])} tokens · {R['prerender']['prefill_s']} s prefill", "",
          "| conversation | turns | read again | client turns | client tokens | engine turns | engine tokens | largest prompt |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    L += [f"| `{s['session']}` | {s['turns']} | {f(s['reread'])} | {s['client']} | {f(s['client_tokens'])} | {s['engine']} | {f(s['engine_tokens'])} | {f(s['max_prompt'])} |"
          for s in R["rooms"]]
    if R["worst"]:
        L += ["", "| largest re-reads | cause | read again | prompt | previous | kept | reused | prefill ms |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        L += [f"| `{c['session']}` | {c['cause']} | {f(c['reread'])} | {f(c['prompt'])} | {f(c['prev'])} | {f(c['diverge_at'])} | {f(c['cached'])} | "
              f"{f(round(c['prefill_ms'])) if c['prefill_ms'] is not None else '–'} |" for c in R["worst"]]
    return L + [""]


def main(argv=None):
    sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
    from hive_daily_report import iter_server, load_server_day  # noqa: E402
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--root", default=os.path.join(os.environ.get("HIVE_STATE_DIR") or os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir)), "run"), "logs"),
                     help="directory holding requests-server.jsonl (default: $HIVE_STATE_DIR/logs, else <repo>/run/logs)")
    cli.add_argument("--day", default=None, help="only that day (local date of t_done); default: every record")
    cli.add_argument("--top", type=int, default=10)
    cli.add_argument("--json", action="store_true")
    a = cli.parse_args(argv)
    server = load_server_day(a.root, a.day) if a.day else list(iter_server(a.root))
    R = summarize_reread(server, a.top)
    sys.stdout.write(json.dumps(R, ensure_ascii=False, indent=1) + "\n" if a.json else "\n".join(render_md(R)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
