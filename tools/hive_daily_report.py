#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""hive_daily_report.py — daily summary (markdown + JSON) from hive_monitor.py output (requests/ and metrics/ daily JSONL), with comparison to the previous day.

usage:
  hive_daily_report.py [--root DIR] [--day YYYY-MM-DD] [--prev YYYY-MM-DD] [--out DIR] [--top 10] [--clients PREFIX,...]
    root = the same directory as hive_monitor.py --out (default: see --root). Default day = yesterday (local time).
    output = <root>/reports/YYYY-MM-DD.md · .json  (changeable with --out)

Definitions (all values come from hived.log — engine clock):
  TTFT       = prefill ms of the request's final line (engine t0 = prefill start after admission → t1 = first token). Queue wait, HTTP and tokenization are excluded.
  prompt     = total prompt tokens (including the reused prefix). Buckets: <1K · 1–4K · 4–16K · 16–64K · 64K+ (K = 1024)
  prefill/s  = newly computed tokens (prompt − reused) / prefill ms
  decode/s   = tok/s of the final line ((n−1)/decode ms). Batch = mean [cache]-step M seen during that request's decode (rounded), else the batch at finish.
  stall      = while a ≥16K prefill runs, the sum of prefill compute ms inserted between two consecutive decode steps of another request that was decoding (lower bound)
  hit/cpu %  = share of routed experts in decode steps ([cache]) that hit VRAM / were computed on the CPU
  MTP net    = Σ(accepted+1)·T1 − Σ(draft+verify ms) (T1 = single-step cost measured by the gate) — positive means MTP reduced decode time
  thinking (reasoning) = from the server log <root>/requests-server.jsonl (hive_server.py log_request · per request thinking_mode, effort, think_cap; requests that
             received a cap also carry think_forced and think_tokens), requests completed that day (local date of t_done) by mode and effort — output tokens, latency, cap-hit rate.
             effort number → name follows DeepSeek's official mapping (50 low · 75 medium · 100 xhigh).
  traffic filter (--clients or HIVE_REPORT_CLIENTS = comma-separated user-agent prefixes; empty = all traffic) = keep only engine requests whose
             server record (joined by daemon request id) came from one of those clients and carries no x-client-test tag — benchmark and test clients
             drop out of the per-request sections. Engine-wide sections built from minute metrics (concurrency, engine steps, expert cache, MTP,
             pre-gate) cannot be split by client and still count all traffic.
  TTFT+queue = TTFT plus the wait between arrival at the daemon and admission (requests that waited for a free decode slot).
  pre-gate (HIVE_DECODE_PREGATE) = sum of per-minute-bucket differences of the [pregate M=…] totals (hive_monitor) — recall = covered/actual · precision = covered/pred.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from hive_monitor import hist_merge, hist_quantile  # noqa: E402

EFFORT_NAME = {50: "low", 75: "medium", 100: "xhigh"}  # DeepSeek's official effort levels (hive_server effort_from_request)
BUCKETS = [("<1K", 0, 1024), ("1-4K", 1024, 4096), ("4-16K", 4096, 16384), ("16-64K", 16384, 65536), ("64K+", 65536, 1 << 62)]


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 2)


def load_jsonl(path):
    out = []
    try:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        out.append(json.loads(ln))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return out


def load_day(root, day):
    reqs, seen = [], set()
    for r in load_jsonl(os.path.join(root, "requests", f"{day}.jsonl")):
        k = (r.get("id"), r.get("loff_end"))  # the monitor is at-least-once (rewrites if it dies after output but before saving state) — deduplicate here
        if k in seen:
            continue
        seen.add(k)
        reqs.append(r)
    mets, seen = [], set()
    for m in load_jsonl(os.path.join(root, "metrics", f"{day}.jsonl")):
        k = (m.get("minute"), m.get("lines"), m.get("decode_steps"))
        if k in seen:
            continue
        seen.add(k)
        mets.append(m)
    return reqs, mets


def _ms(v):
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def iter_server(root):
    """Every server request record (current requests-server.jsonl plus archived archive/requests-server*.jsonl[.gz]), deduplicated by (rid, t_recv)."""
    import glob
    import gzip
    paths = [os.path.join(root, "requests-server.jsonl")] + sorted(glob.glob(os.path.join(root, "archive", "requests-server*.jsonl*")))
    seen = set()
    for path in paths:
        op = gzip.open if path.endswith(".gz") else open
        try:
            with op(path, "rt") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    k = (r.get("rid"), r.get("t_recv"))
                    if k in seen:
                        continue
                    seen.add(k)
                    yield r
        except FileNotFoundError:
            pass


def load_server_day(root, day):
    """Server request records of that day (local date of t_done, else t_recv) — the file is neither split by date nor rotated, so it is read in full and filtered
    (measured: 3,476 records, 2.9 MB per day)."""
    from datetime import datetime
    out = []
    for r in iter_server(root):
        t = _ms(r.get("t_done")) or _ms(r.get("t_recv"))
        if t is not None and datetime.fromtimestamp(t / 1000.0).date().isoformat() == day:
            out.append(r)
    return out


def client_kept(rec, prefixes):
    """True when a server record came from one of the user-agent prefixes and is not tagged as a test (x-client-test)."""
    if not rec:
        return False
    c = rec.get("client") or {}
    if c.get("x-client-test"):
        return False
    ua = c.get("user-agent") or ""
    return any(ua.startswith(p) for p in prefixes)


def filter_traffic(reqs, server, index, prefixes):
    """Apply the traffic filter to one day: engine requests are joined to server records by daemon request id (monitor `rid` = server `daemon_rid`)."""
    from collections import Counter
    kept, dropped = [], Counter()
    for r in reqs:
        rec = index.get(r.get("rid"))
        if client_kept(rec, prefixes):
            kept.append(r)
        else:
            c = (rec or {}).get("client") or {}
            dropped["(no server record)" if rec is None else ("test-tagged" if c.get("x-client-test") else (c.get("user-agent") or "(empty)")[:32])] += 1
    info = {"clients": prefixes, "kept": len(kept), "total": len(reqs), "excluded": dict(dropped.most_common())}
    return kept, [r for r in server if client_kept(r, prefixes)], info


def effort_key(r):
    mode = r.get("thinking_mode")
    if mode != "thinking":
        return mode or "?"
    e = r.get("effort")
    try:
        e = int(float(e))
    except (TypeError, ValueError):
        return f"thinking/{e}"
    return f"thinking/{e}" + (f"({EFFORT_NAME[e]})" if e in EFFORT_NAME else "")


def summarize_thinking(server, top=10):
    done = [r for r in server if (r.get("done") or {}).get("n") is not None]
    by = {}
    for r in done:
        by.setdefault(effort_key(r), []).append(r)
    out = {}
    for key, rs in sorted(by.items()):
        ntok = [(r.get("done") or {}).get("n") for r in rs]
        lat = [(_ms(r.get("t_done")) - _ms(r.get("t_recv"))) / 1000.0 for r in rs if _ms(r.get("t_done")) and _ms(r.get("t_recv"))]
        capped = [r for r in rs if r.get("think_cap") not in (None, "", "None")]
        forced = [r for r in capped if r.get("think_forced") is True]
        caps = sorted({int(float(r["think_cap"])) for r in capped})
        out[key] = {"n": len(rs), "completion_p50": pct(ntok, .5), "completion_p95": pct(ntok, .95), "completion_max": max(ntok),
                    "completion_sum": sum(ntok), "latency_s_p50": pct(lat, .5), "latency_s_p95": pct(lat, .95),
                    "finish": dict(sorted({f: sum(1 for r in rs if (r.get("done") or {}).get("finish") == f)
                                           for f in {(r.get("done") or {}).get("finish") for r in rs}}.items(), key=lambda kv: str(kv[0]))),
                    "with_cap": len(capped), "caps": caps, "forced": len(forced),
                    "forced_share": round(len(forced) / len(capped), 4) if capped else None,
                    "think_tokens_p50": pct([r.get("think_tokens") for r in capped], .5),
                    "think_tokens_p95": pct([r.get("think_tokens") for r in capped], .95)}
    allcap = [r for r in done if r.get("think_cap") not in (None, "", "None")]
    longest = sorted(done, key=lambda r: -((r.get("done") or {}).get("n") or 0))[:top]
    return {"by_effort": out, "requests": len(server), "completed": len(done), "with_cap": len(allcap),
            "forced": sum(1 for r in allcap if r.get("think_forced") is True),
            "capped_share": round(sum(1 for r in allcap if r.get("think_forced") is True) / len(allcap), 4) if allcap else None,
            "longest": [{"effort": effort_key(r), "completion": (r.get("done") or {}).get("n"), "think_tokens": r.get("think_tokens"),
                         "forced": r.get("think_forced"), "finish": (r.get("done") or {}).get("finish"), "session": r.get("session"),
                         "prompt": r.get("prompt_tokens"), "latency_s": round((_ms(r.get("t_done")) - _ms(r.get("t_recv"))) / 1000.0, 1)
                         if _ms(r.get("t_done")) and _ms(r.get("t_recv")) else None} for r in longest],
            "note": "think_tokens and forced cover only requests that received a cap (think_cap) — uncapped thinking requests show only their output tokens (completion)"}


def bucket_of(pt):
    for name, lo, hi in BUCKETS:
        if lo <= (pt or 0) < hi:
            return name
    return BUCKETS[-1][0]


def summarize(reqs, mets, top=10, server=None):
    S: dict = {}
    done = [r for r in reqs if r.get("prefill_ms") is not None]
    S["requests"] = {"total": len(reqs), "completed": len(done),
                     "finish": dict(sorted({f: sum(1 for r in reqs if r["finish"] == f) for f in {r["finish"] for r in reqs}}.items())),
                     "ts_quality": {q: sum(1 for r in reqs if r.get("ts_quality") == q) for q in {r.get("ts_quality") for r in reqs}},
                     "builds": sorted({r.get("build") for r in reqs if r.get("build")}),
                     "prompt_tokens_sum": sum(r.get("prompt_tokens") or 0 for r in reqs),
                     "decode_tokens_sum": sum(r.get("decode_tokens") or 0 for r in done)}
    # per prompt bucket
    pb = {}
    for name, _, _ in BUCKETS:
        rs = [r for r in done if bucket_of(r.get("prompt_tokens")) == name]
        if not rs:
            continue
        ttft = [r["ttft_ms"] for r in rs]
        tq = [r["ttft_with_queue_ms"] for r in rs if r.get("ttft_with_queue_ms") is not None]
        pb[name] = {"n": len(rs), "ttft_p50_ms": pct(ttft, .5), "ttft_p95_ms": pct(ttft, .95), "ttft_p99_ms": pct(ttft, .99),
                    "ttft_max_ms": max(ttft), "ttft_with_queue_p50_ms": pct(tq, .5), "ttft_with_queue_p95_ms": pct(tq, .95), "prefill_tok_s_p50": pct([r.get("prefill_tok_s") for r in rs], .5),
                    "prefill_tok_s_p5": pct([r.get("prefill_tok_s") for r in rs], .05),
                    "prompt_p50": pct([r.get("prompt_tokens") for r in rs], .5),
                    "cached_share": round(sum(r.get("cached_prefix_tokens") or 0 for r in rs) / max(1, sum(r.get("prompt_tokens") or 0 for r in rs)), 4),
                    "decode_tok_s_p50": pct([r["decode_tok_s"] for r in rs if r.get("decode_tokens", 0) >= 16], .5)}
    S["by_prompt_bucket"] = pb
    # decode speed per batch
    bb: dict = {}
    for r in done:
        if r.get("decode_tokens", 0) < 16:
            continue
        b = r.get("decode_batch_avg")
        b = int(round(b)) if b else r.get("batch_at_end")
        bb.setdefault(str(b), []).append(r["decode_tok_s"])
    S["decode_by_batch"] = {k: {"n": len(v), "tok_s_p50": pct(v, .5), "tok_s_p5": pct(v, .05), "tok_s_p95": pct(v, .95)}
                            for k, v in sorted(bb.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 99)}
    # sum of engine minute buckets
    agg = {k: 0 for k in ("decode_steps", "routed", "hit", "cpu", "streamed", "reclaims", "reclaim_evicted", "snapshots", "warm_n",
                          "mtp_steps", "mtp_nodraft", "mtp_drafted", "mtp_accepted", "mtp_tokens", "mtp_batch_n", "mtp_batch_nodraft",
                          "stall_n", "prefill_chunks", "batch_rounds", "batch_windows", "requests_done", "warnings", "dh_n",
                          "dh_cpu_bound_layers", "dh_layers", "dsync_n", "dma_samples", "dma_wait_samples", "dh_promo_wait_n")}
    fl = {k: 0.0 for k in ("dma_span_ms", "dma_wait_ms", "mtp_draft_ms", "mtp_verify_ms", "mtp_saved_ms", "stall_ms_sum", "reclaim_ms",
                           "dh_gpu_idle_ms", "dh_of_ms", "dh_cpu_ms", "dh_gpu_ms", "dsync_wait_ms", "warm_ms", "prefill_chunk_ms", "promo_mib",
                           "dh_tail_ms", "dh_promo_wait_ms", "dh_defer_wait_ms")}
    conc, batch, step_all, verify, stall_h, gidle = {}, {}, {}, {}, {}, {}
    step_m: dict = {}
    wall_m: dict = {}
    errors, events, unparsed = [], [], {}
    pg = {"n": 0, "layers": 0, "posted": 0, "late": 0, "untrusted": 0, "pred": 0, "actual": 0, "covered": 0, "pool_req": 0, "pool_experts": 0,
          "read_mib": 0.0, "finished": 0, "aborted": 0}
    stall_max = 0.0
    vram_min = None
    for m in mets:
        for k in agg:
            agg[k] += m.get(k, 0) or 0
        for k in fl:
            fl[k] += m.get(k, 0.0) or 0.0
        hist_merge(conc, m.get("conc_hist", {})); hist_merge(batch, m.get("batch_hist", {}))
        hist_merge(step_all, m.get("step_ms_hist", {})); hist_merge(verify, m.get("verify_ms_hist", {}))
        hist_merge(stall_h, m.get("stall_hist", {})); hist_merge(gidle, m.get("gpu_idle_pct_hist", {}))
        for M, h in (m.get("step_ms_by_m") or {}).items():
            hist_merge(step_m.setdefault(M, {}), h)
        for M, h in (m.get("step_wall_by_m") or {}).items():  # wall clock of every step (available when the engine stamps wall-clock times on request lines)
            hist_merge(wall_m.setdefault(M, {}), h)
        stall_max = max(stall_max, m.get("stall_ms_max", 0) or 0)
        if m.get("vram_free_min") is not None:
            vram_min = m["vram_free_min"] if vram_min is None else min(vram_min, m["vram_free_min"])
        errors += [{**e, "minute": m["minute"]} for e in m.get("errors", [])]
        events += [{**e, "minute": m["minute"]} for e in m.get("events", [])]
        for k, v in (m.get("unparsed") or {}).items():
            unparsed[k] = unparsed.get(k, 0) + v
        for k, v in (m.get("pregate") or {}).items():
            pg[k] = pg.get(k, 0) + (v or 0)
    tot_steps = sum(conc.values()) or 1
    S["concurrency"] = {"open_requests_at_decode_step": {k: round(v / tot_steps, 4) for k, v in sorted(conc.items(), key=lambda kv: int(kv[0]))},
                        "decode_batch_M": {k: round(v / max(1, sum(batch.values())), 4) for k, v in sorted(batch.items(), key=lambda kv: int(kv[0]))},
                        "per_request_conc_max_p50": pct([r.get("conc_max") for r in reqs], .5),
                        "per_request_conc_max_max": max([r.get("conc_max") or 0 for r in reqs], default=None)}
    S["engine_steps"] = {"decode_steps": agg["decode_steps"],
                         "step_ms_by_M": {M: {"n": sum(h.values()), "p50": hist_quantile(h, .5), "p95": hist_quantile(h, .95)}
                                          for M, h in sorted(step_m.items(), key=lambda kv: int(kv[0]))},
                         "step_wall_ms_by_M": {M: {"n": sum(h.values()), "p50": hist_quantile(h, .5), "p95": hist_quantile(h, .95)}
                                               for M, h in sorted(wall_m.items(), key=lambda kv: int(kv[0]))},
                         "step_ms_all": {"n": sum(step_all.values()), "p50": hist_quantile(step_all, .5), "p95": hist_quantile(step_all, .95)},
                         "mtp_verify_ms": {"n": sum(verify.values()), "p50": hist_quantile(verify, .5), "p95": hist_quantile(verify, .95)},
                         "gpu_idle_pct": {"n": sum(gidle.values()), "p50": hist_quantile(gidle, .5), "p95": hist_quantile(gidle, .95),
                                          "sum_ratio": round(100 * fl["dh_gpu_idle_ms"] / fl["dh_of_ms"], 2) if fl["dh_of_ms"] else None},
                         "cpu_bound_layer_share": round(agg["dh_cpu_bound_layers"] / agg["dh_layers"], 4) if agg["dh_layers"] else None,
                         "cpu_vs_gpu_ms": [round(fl["dh_cpu_ms"], 1), round(fl["dh_gpu_ms"], 1)],
                         "decode_sync_wait_ms_mean": round(fl["dsync_wait_ms"] / agg["dsync_n"], 2) if agg["dsync_n"] else None,
                         "cpu_tail_ms_mean": round(fl["dh_tail_ms"] / agg["dh_n"], 3) if agg["dh_n"] else None,
                         "promo_wait_ms_per_step": round(fl["dh_promo_wait_ms"] / agg["dh_promo_wait_n"], 3) if agg["dh_promo_wait_n"] else None,
                         "defer_wait_ms_mean": round(fl["dh_defer_wait_ms"] / agg["dh_n"], 3) if agg["dh_n"] and fl["dh_defer_wait_ms"] else None,
                         "note": "step ms covers only HIVE_PROFILE sample steps ([cache] after [profile]) — the sampling period is the HIVE_PROFILE value"}
    R = agg["routed"] or 1
    dec = [r for r in done if (r.get("decode_hit", 0) + r.get("decode_cpu", 0)) > 0]
    S["cache"] = {"decode_hit_pct": round(100 * agg["hit"] / R, 2) if agg["routed"] else None,
                  "decode_cpu_pct": round(100 * agg["cpu"] / R, 2) if agg["routed"] else None,
                  "decode_streamed_pct": round(100 * agg["streamed"] / R, 2) if agg["routed"] else None,
                  "request_cpu_share_p50": pct([r.get("decode_cpu_share") for r in dec], .5),
                  "request_cpu_share_p95": pct([r.get("decode_cpu_share") for r in dec], .95),
                  "prefill_cpu_share": round(sum(r.get("prefill_cpu", 0) for r in done) / max(1, sum(r.get("prefill_cpu", 0) + r.get("prefill_hit", 0) + r.get("prefill_streamed", 0) for r in done)), 4),
                  "dma_wait_ms_per_step": round(fl["dma_wait_ms"] / agg["dma_wait_samples"], 2) if agg["dma_wait_samples"] else None,
                  "dma_span_ms_per_step": round(fl["dma_span_ms"] / agg["dma_samples"], 2) if agg["dma_samples"] else None,
                  "promo_gib": round(fl["promo_mib"] / 1024, 1), "elastic_reclaims": agg["reclaims"], "reclaim_evicted": agg["reclaim_evicted"],
                  "reclaim_ms": round(fl["reclaim_ms"], 1), "warm_cache_runs": agg["warm_n"], "warm_cache_ms": round(fl["warm_ms"]),
                  "vram_free_min_mib": vram_min}
    mt_req = [r for r in done if r.get("mtp_drafted")]
    S["mtp"] = {"accept_rate": round(agg["mtp_accepted"] / agg["mtp_drafted"], 4) if agg["mtp_drafted"] else None,
                "verify_steps": agg["mtp_steps"] + agg["mtp_batch_n"], "no_draft_steps": agg["mtp_nodraft"] + agg["mtp_batch_nodraft"],
                "tokens_per_verify": round(agg["mtp_tokens"] / (agg["mtp_steps"] + agg["mtp_batch_n"]), 3) if (agg["mtp_steps"] + agg["mtp_batch_n"]) else None,
                "draft_ms": round(fl["mtp_draft_ms"]), "verify_ms": round(fl["mtp_verify_ms"]),
                "net_saved_ms": round(fl["mtp_saved_ms"]),
                "speedup_x": round((fl["mtp_saved_ms"] + fl["mtp_draft_ms"] + fl["mtp_verify_ms"]) / (fl["mtp_draft_ms"] + fl["mtp_verify_ms"]), 3) if (fl["mtp_draft_ms"] + fl["mtp_verify_ms"]) else None,
                "request_accept_p50": pct([r.get("mtp_accept_rate") for r in mt_req], .5),
                "requests_with_mtp": len(mt_req)}
    longp = [r for r in reqs if (r.get("prompt_tokens") or 0) >= 16384]
    S["long_prefill_stalls"] = {"events": agg["stall_n"], "sum_ms": round(fl["stall_ms_sum"]), "max_ms": stall_max,
                                "p50_ms": hist_quantile(stall_h, .5), "p95_ms": hist_quantile(stall_h, .95),
                                "long_prefill_requests": len(longp),
                                "top": [{"id": r["id"], "prompt": r.get("prompt_tokens"), "ttft_ms": r.get("ttft_ms"), "stalls": r.get("stall_caused_n"),
                                         "stall_max_ms": r.get("stall_caused_max_ms"), "stall_sum_ms": r.get("stall_caused_ms"), "end": r.get("end")}
                                        for r in sorted(longp, key=lambda r: -(r.get("stall_caused_max_ms") or 0))[:top] if r.get("stall_caused_n")],
                                "victim_stalled_max_ms_p95": pct([r.get("stalled_max_ms") for r in done if r.get("stalled_max_ms")], .95),
                                "note": "gap = sum of prefill compute ms between decode steps (lower bound — excludes the step time itself). Gaps from short (<16K) prefills are excluded from hist"}
    pg_on = [e for e in events if e.get("kind") == "pregate_on"]
    S["pregate"] = {"on": bool(pg_on) or pg["n"] > 0, "k": pg_on[-1].get("k") if pg_on else None, "samples": pg["n"],
                    "layers": pg["layers"], "posted": pg["posted"], "late": pg["late"], "untrusted": pg["untrusted"],
                    "posted_share": round(pg["posted"] / (pg["posted"] + pg["late"] + pg["untrusted"]), 4) if (pg["posted"] + pg["late"] + pg["untrusted"]) else None,
                    "recall": round(pg["covered"] / pg["actual"], 4) if pg["actual"] else None,
                    "precision": round(pg["covered"] / pg["pred"], 4) if pg["pred"] else None,
                    "prefetch_experts": pg["pred"], "cpu_experts": pg["actual"], "covered": pg["covered"],
                    "pool_requests": pg["pool_req"], "pool_experts": pg["pool_experts"], "read_mib": round(pg["read_mib"], 1),
                    "finished": pg["finished"], "aborted": pg["aborted"],
                    "aborted_share": round(pg["aborted"] / (pg["finished"] + pg["aborted"]), 4) if (pg["finished"] + pg["aborted"]) else None,
                    "note": "sum of total differences over sample steps ([pregate] lines = HIVE_PROFILE cadence) — a sum over sampled intervals, not a full count"}
    S["thinking"] = summarize_thinking(server or [], top)
    errs: dict = {}
    for e in errors:
        k = (e.get("kind"), (e.get("msg") or "")[:120])
        errs.setdefault(k, {"kind": k[0], "msg": k[1], "n": 0, "first": e["minute"], "last": e["minute"]})
        errs[k]["n"] += 1; errs[k]["last"] = e["minute"]
    S["errors"] = {"engine": sorted(errs.values(), key=lambda e: -e["n"]),
                   "requests_error": [r["id"] for r in reqs if r["finish"] in ("error", "lost")][:50],
                   "cancelled": sum(1 for r in reqs if r["finish"] == "cancel"),
                   "length_capped": sum(1 for r in reqs if r["finish"] == "length"),
                   "daemon_restart_cut": sum(1 for r in reqs if r["finish"] == "daemon_restart"),
                   "restarts": sum(1 for e in events if e.get("kind") == "container_start"),
                   "sleep_wake": [e for e in events if e.get("kind") in ("sleep", "wake")][:50],
                   "warnings": agg["warnings"], "unparsed_shapes": dict(sorted(unparsed.items(), key=lambda kv: -kv[1])[:10]),
                   "note": "the engine logs no timeout lines — server/client timeouts show up as cancel"}
    S["top_slow"] = {
        "ttft": [{"id": r["id"], "ttft_ms": r["ttft_ms"], "prompt": r.get("prompt_tokens"), "cached": r.get("cached_prefix_tokens"),
                  "conc_max": r.get("conc_max"), "end": r.get("end")} for r in sorted(done, key=lambda r: -r["ttft_ms"])[:top]],
        "ttft_per_1k_new": [{"id": r["id"], "ms_per_1k": round(r["ttft_ms"] * 1000 / max(1, r["prefill_tokens"]), 1), "new_tokens": r["prefill_tokens"],
                             "ttft_ms": r["ttft_ms"], "end": r.get("end")}
                            for r in sorted([r for r in done if r["prefill_tokens"] >= 256], key=lambda r: -r["ttft_ms"] / r["prefill_tokens"])[:top]],
        "decode_slowest": [{"id": r["id"], "tok_s": r["decode_tok_s"], "tokens": r["decode_tokens"], "batch_avg": r.get("decode_batch_avg"),
                            "cpu_share": r.get("decode_cpu_share"), "end": r.get("end")}
                           for r in sorted([r for r in done if r.get("decode_tokens", 0) >= 32], key=lambda r: r["decode_tok_s"])[:top]],
    }
    return S


KEYS_CMP = [("requests.completed", "completed requests"), ("cache.decode_hit_pct", "decode hit %"), ("cache.decode_cpu_pct", "decode CPU %"),
            ("mtp.accept_rate", "MTP accept rate"), ("mtp.net_saved_ms", "MTP net saved ms"), ("mtp.speedup_x", "MTP speedup"), ("long_prefill_stalls.max_ms", "max stall ms"),
            ("engine_steps.step_ms_all.p50", "decode step p50 ms"), ("engine_steps.gpu_idle_pct.sum_ratio", "GPU idle %"),
            ("thinking.capped_share", "thinking cap hit rate"), ("pregate.recall", "pre-gate recall"), ("pregate.precision", "pre-gate precision")]


def dig(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def compare(cur, prev):
    out = {}
    for k, _ in KEYS_CMP:
        a, b = dig(cur, k), dig(prev, k) if prev else None
        out[k] = {"cur": a, "prev": b, "delta": round(a - b, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None}
    if prev:
        for name, _, _ in BUCKETS:
            a, b = dig(cur, f"by_prompt_bucket.{name}.ttft_p50_ms"), dig(prev, f"by_prompt_bucket.{name}.ttft_p50_ms")
            if a is not None or b is not None:
                out[f"ttft_p50.{name}"] = {"cur": a, "prev": b, "delta": round(a - b, 2) if a is not None and b is not None else None}
        for bs in set(cur.get("decode_by_batch", {})) | set(prev.get("decode_by_batch", {})):
            a, b = dig(cur, f"decode_by_batch.{bs}.tok_s_p50"), dig(prev, f"decode_by_batch.{bs}.tok_s_p50")
            out[f"decode_tok_s_p50.batch{bs}"] = {"cur": a, "prev": b, "delta": round(a - b, 2) if a is not None and b is not None else None}
    return out


def f(x, nd=1):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def render_md(day, S, cmp_, prev_day):
    L = [f"# hive daily report — {day}", ""]
    q = S["requests"]
    L += [f"- requests {q['total']} (completed {q['completed']}) · finish {q['finish']} · timestamp quality {q['ts_quality']}",
          f"- prompt tokens total {f(q['prompt_tokens_sum'])} · decode tokens total {f(q['decode_tokens_sum'])} · builds {', '.join(q['builds']) or '–'}",
          f"- restarts {S['errors']['restarts']} · engine errors {sum(e['n'] for e in S['errors']['engine'])} · warnings {S['errors']['warnings']}", ""]
    tr = S.get("traffic")
    if tr:
        L[-1:-1] = [f"- traffic filter (user-agent prefixes {', '.join(tr['clients'])}): kept {tr['kept']} of {tr['total']} engine requests · excluded {tr['excluded']}",
                    "  · per-request sections count only the kept requests; concurrency, engine steps, expert cache, MTP and pre-gate count all traffic"]
    L += ["## TTFT and prefill by prompt length", "",
          "| bucket | n | TTFT p50 | p95 | p99 | max | TTFT+queue p50 | p95 | prefill tok/s p50 | p5 | reused share | decode tok/s p50 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, v in S["by_prompt_bucket"].items():
        L.append(f"| {name} | {v['n']} | {f(v['ttft_p50_ms'],0)} | {f(v['ttft_p95_ms'],0)} | {f(v['ttft_p99_ms'],0)} | {f(v['ttft_max_ms'],0)} | "
                 f"{f(v.get('ttft_with_queue_p50_ms'),0)} | {f(v.get('ttft_with_queue_p95_ms'),0)} | "
                 f"{f(v['prefill_tok_s_p50'])} | {f(v['prefill_tok_s_p5'])} | {f(100*v['cached_share'])}% | {f(v['decode_tok_s_p50'])} |")
    L += ["", "TTFT = engine prefill start → first token (ms). TTFT+queue adds the wait for admission (a free decode slot).", "", "## Decode speed by batch (requests, ≥16 tokens)", "",
          "| batch | n | tok/s p50 | p5 | p95 | engine step ms p50 | p95 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    sm = S["engine_steps"]["step_ms_by_M"]
    for bs, v in S["decode_by_batch"].items():
        e = S["engine_steps"]["step_wall_ms_by_M"].get(bs) or sm.get(bs, {})  # full-step wall clock if present (engine stamps wall-clock times on request lines), otherwise the sampled [profile] steps
        L.append(f"| {bs} | {v['n']} | {f(v['tok_s_p50'])} | {f(v['tok_s_p5'])} | {f(v['tok_s_p95'])} | {f(e.get('p50'),2)} | {f(e.get('p95'),2)} |")
    es = S["engine_steps"]
    L += ["", f"- engine decode steps {f(es['decode_steps'])} · sampled step ms p50 {f(es['step_ms_all']['p50'],2)} / p95 {f(es['step_ms_all']['p95'],2)} "
              f"· MTP verify step p50 {f(es['mtp_verify_ms']['p50'],2)} ms",
          f"- GPU idle (decode-host) total ratio {f(es['gpu_idle_pct']['sum_ratio'],1)}% (step p50 {f(es['gpu_idle_pct']['p50'],1)}% · p95 {f(es['gpu_idle_pct']['p95'],1)}%) · "
          f"CPU-bound layer share {f(es['cpu_bound_layer_share'] and 100*es['cpu_bound_layer_share'],1)}% · cpu/gpu ms {es['cpu_vs_gpu_ms']} · decode-sync wait mean {f(es['decode_sync_wait_ms_mean'],2)} ms",
          f"- per sampled step: wait for CPU experts {f(es['cpu_tail_ms_mean'],3)} ms · promotion commit wait {f(es['promo_wait_ms_per_step'],3)} ms/step · "
          f"deferral wait {f(es['defer_wait_ms_mean'],3)} ms", ""]
    c = S["concurrency"]
    L += ["## Concurrency", "", "| open requests (at decode step) | share |", "|---:|---:|"]
    L += [f"| {k} | {f(100*v,1)}% |" for k, v in c["open_requests_at_decode_step"].items()]
    L += ["", "| decode batch M | share |", "|---:|---:|"] + [f"| {k} | {f(100*v,1)}% |" for k, v in c["decode_batch_M"].items()]
    L += ["", f"- per-request max concurrency p50 {f(c['per_request_conc_max_p50'])} · max {f(c['per_request_conc_max_max'])}", ""]
    st = S["long_prefill_stalls"]
    L += ["## Decode gaps during long prefills (≥16K)", "",
          f"- ≥16K requests {st['long_prefill_requests']} · gaps {st['events']} · total {f(st['sum_ms'])} ms · p50 {f(st['p50_ms'],0)} · p95 {f(st['p95_ms'],0)} · max {f(st['max_ms'],0)} ms"
          f" · affected requests' max gap p95 {f(st['victim_stalled_max_ms_p95'],0)} ms", ""]
    if st["top"]:
        L += ["| request | prompt | TTFT ms | gaps | max ms | total ms | end |", "|---|---:|---:|---:|---:|---:|---|"]
        L += [f"| `{t['id']}` | {f(t['prompt'])} | {f(t['ttft_ms'],0)} | {t['stalls']} | {f(t['stall_max_ms'],0)} | {f(t['stall_sum_ms'],0)} | {t['end']} |" for t in st["top"]]
        L.append("")
    ca = S["cache"]
    L += ["## Expert cache", "",
          f"- decode hit {f(ca['decode_hit_pct'],2)}% · CPU {f(ca['decode_cpu_pct'],2)}% · streamed {f(ca['decode_streamed_pct'],2)}% · request CPU share p50 {f(ca['request_cpu_share_p50'],3)} / p95 {f(ca['request_cpu_share_p95'],3)}",
          f"- prefill CPU share {f(ca['prefill_cpu_share'],3)} · DMA wait {f(ca['dma_wait_ms_per_step'],2)} ms/step (measured steps) · span {f(ca['dma_span_ms_per_step'],2)} ms · promoted {f(ca['promo_gib'])} GiB",
          f"- elastic reclaim {ca['elastic_reclaims']} (evicted {f(ca['reclaim_evicted'])} · {f(ca['reclaim_ms'])} ms) · warm cache {ca['warm_cache_runs']} runs {f(ca['warm_cache_ms'])} ms · VRAM free min {f(ca['vram_free_min_mib'])} MiB", ""]
    pg = S["pregate"]
    L += ["## Decode pre-gate (HIVE_DECODE_PREGATE)", ""]
    if not pg["on"]:
        L += ["- off (no startup banner or [pregate] lines)", ""]
    else:
        L += [f"- on (top-{f(pg['k'])}) · samples {f(pg['samples'])} · layers {f(pg['layers'])} · posted {f(pg['posted'])} / late {f(pg['late'])} / untrusted {f(pg['untrusted'])}"
              f" (posted share {f(pg['posted_share'] and 100*pg['posted_share'],1)}%)",
              f"- **recall {f(pg['recall'] and 100*pg['recall'],1)}% · precision {f(pg['precision'] and 100*pg['precision'],1)}%** "
              f"(prefetched {f(pg['prefetch_experts'])} · actual CPU {f(pg['cpu_experts'])} · covered {f(pg['covered'])})",
              f"- pool requests {f(pg['pool_requests'])} · experts {f(pg['pool_experts'])} · read {f(pg['read_mib'])} MiB · finished {f(pg['finished'])} / aborted {f(pg['aborted'])}"
              f" (aborted share {f(pg['aborted_share'] and 100*pg['aborted_share'],1)}%)", ""]
    th = S["thinking"]
    L += ["## Thinking (reasoning) — server records", ""]
    if not th["completed"]:
        L += ["- no server records for the day (requests-server.jsonl)", ""]
    else:
        L += [f"- completed {f(th['completed'])} · requests with a cap {f(th['with_cap'])} · cap reached {f(th['forced'])}"
              f" (hit rate {f(th['capped_share'] and 100*th['capped_share'],1)}%)", "",
              "| mode/effort | n | output tokens p50 | p95 | max | total | latency s p50 | p95 | cap | reached | hit rate | think tokens p50 | p95 | finish |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"]
        for key, v in th["by_effort"].items():
            L.append(f"| {key} | {v['n']} | {f(v['completion_p50'],0)} | {f(v['completion_p95'],0)} | {f(v['completion_max'])} | {f(v['completion_sum'])} | "
                     f"{f(v['latency_s_p50'])} | {f(v['latency_s_p95'])} | {','.join(map(str, v['caps'])) or '–'} | {v['forced']} | "
                     f"{f(v['forced_share'] and 100*v['forced_share'],1)}% | {f(v['think_tokens_p50'],0)} | {f(v['think_tokens_p95'],0)} | {v['finish']} |")
        L += ["", "| top output tokens | mode/effort | output | think tokens | reached | finish | latency s | prompt | session |", "|---:|---|---:|---:|---|---|---:|---:|---|"]
        L += [f"| {i+1} | {t['effort']} | {f(t['completion'])} | {f(t['think_tokens'])} | {t['forced'] if t['forced'] is not None else '–'} | {t['finish']} | "
              f"{f(t['latency_s'])} | {f(t['prompt'])} | `{t['session']}` |" for i, t in enumerate(th["longest"])]
        L += ["", f"- {th['note']}", ""]
    mt = S["mtp"]
    L += ["## MTP", "",
          f"- accept rate {f(mt['accept_rate'] and 100*mt['accept_rate'],1)}% · verify steps {f(mt['verify_steps'])} (no draft {f(mt['no_draft_steps'])}) · tokens per verify {f(mt['tokens_per_verify'],2)}",
          f"- draft {f(mt['draft_ms'])} ms + verify {f(mt['verify_ms'])} ms · **net saved {f(mt['net_saved_ms'])} ms** (MTP spans {f(mt['speedup_x'],2)}x vs single-step T1) · request accept rate p50 {f(mt['request_accept_p50'],3)}", ""]
    er = S["errors"]
    L += ["## Errors and finishes", "", f"- cancel {er['cancelled']} · length {er['length_capped']} · cut by restart {er['daemon_restart_cut']} · error/lost {len(er['requests_error'])}"]
    for e in er["engine"][:15]:
        L.append(f"- `{e['kind']}` ×{e['n']} ({e['first']} → {e['last']}): {e['msg']}")
    if er["unparsed_shapes"]:
        L.append(f"- unparsed line shapes (new printf?): {er['unparsed_shapes']}")
    L += ["", "## Slowest requests", "", "| top TTFT | TTFT ms | prompt | reused | concurrent | end |", "|---|---:|---:|---:|---:|---|"]
    L += [f"| `{t['id']}` | {f(t['ttft_ms'],0)} | {f(t['prompt'])} | {f(t['cached'])} | {t['conc_max']} | {t['end']} |" for t in S["top_slow"]["ttft"]]
    L += ["", "| top prefill ms per 1K | ms/1K | new tokens | TTFT ms | end |", "|---|---:|---:|---:|---|"]
    L += [f"| `{t['id']}` | {f(t['ms_per_1k'])} | {f(t['new_tokens'])} | {f(t['ttft_ms'],0)} | {t['end']} |" for t in S["top_slow"]["ttft_per_1k_new"]]
    L += ["", "| slowest decode | tok/s | tokens | batch mean | CPU share | end |", "|---|---:|---:|---:|---:|---|"]
    L += [f"| `{t['id']}` | {f(t['tok_s'])} | {t['tokens']} | {f(t['batch_avg'],2)} | {f(t['cpu_share'],3)} | {t['end']} |" for t in S["top_slow"]["decode_slowest"]]
    L += ["", f"## Compared with the previous day ({prev_day})", ""]
    if not any(v["prev"] is not None for v in cmp_.values()):
        L.append("- no data for the previous day")
    else:
        L += ["| metric | today | previous day | delta |", "|---|---:|---:|---:|"]
        names = dict(KEYS_CMP)
        for k, v in cmp_.items():
            L.append(f"| {names.get(k, k)} | {f(v['cur'],2)} | {f(v['prev'],2)} | {f(v['delta'],2)} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--root", default=os.path.join(os.environ.get("HIVE_STATE_DIR") or os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir)), "run"), "logs"),
                     help="hive_monitor.py output root (default: $HIVE_STATE_DIR/logs, else <repo>/run/logs)")
    cli.add_argument("--day", default=None)
    cli.add_argument("--prev", default=None)
    cli.add_argument("--out", default=None)
    cli.add_argument("--top", type=int, default=10)
    cli.add_argument("--stdout", action="store_true", help="also write the markdown to stdout")
    cli.add_argument("--clients", default=None,
                     help="count only requests from these user-agent prefixes (comma-separated; default $HIVE_REPORT_CLIENTS; empty = all traffic)")
    a = cli.parse_args(argv)
    day = a.day or (date.today() - timedelta(days=1)).isoformat()
    prev_day = a.prev or (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    raw = a.clients if a.clients is not None else os.environ.get("HIVE_REPORT_CLIENTS", "")
    prefixes = [p.strip() for p in raw.split(",") if p.strip()]
    index = {r.get("daemon_rid"): r for r in iter_server(a.root) if r.get("daemon_rid")} if prefixes else {}

    def one(d):
        reqs, mets = load_day(a.root, d)
        server = load_server_day(a.root, d)
        info = None
        if prefixes:
            reqs, server, info = filter_traffic(reqs, server, index, prefixes)
        return reqs, mets, server, info

    reqs, mets, server, info = one(day)
    S = summarize(reqs, mets, a.top, server)
    if info:
        S["traffic"] = info
    preqs, pmets, pserver, _ = one(prev_day)
    P = summarize(preqs, pmets, a.top, pserver) if (preqs or pmets) else None
    cmp_ = compare(S, P)
    out = a.out or os.path.join(a.root, "reports")
    os.makedirs(out, exist_ok=True)
    md = render_md(day, S, cmp_, prev_day)
    with open(os.path.join(out, f"{day}.json"), "w") as fh:
        json.dump({"day": day, "prev_day": prev_day, "summary": S, "compare": cmp_}, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out, f"{day}.md"), "w") as fh:
        fh.write(md)
    if a.stdout:
        sys.stdout.write(md)
    else:
        print(os.path.join(out, f"{day}.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
