#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""analyze_hived_log.py — aggregator for hived.log (HIVE_PROFILE=1, HIVE_TRACE_CACHE=1). Also used for continuous tuning from serving logs.

hived.log line kinds (as emitted by the engine — runtime.cpp preport, end of forward_batch, hived.cpp finish_active):
  [profile M=k] total T ms: name ms name ms …   phase times of one forward (cudaEvent). Phases run as a CUDA graph are measured as a whole as segA.*.
  [cache] step S M=k routed R hit H cpu C streamed X · resident N pending P / SLOTS   expert hits of one decode step (forward_batch)
  [hived] <sid>: prefill P tok X ms · decode N tok Y ms (Z tok/s, batch B) · hit H cpu C · resident R/S · <finish>   one request completed
  [hived] warm cache: N experts in T ms   · [hived] request failed: …
Classification: a profile line **immediately followed by a [cache] line is a decode step** (forward_batch), otherwise a prefill chunk (forward). The log has no timestamps, so ranges
are given as line numbers (--lines A:B) — to cut by time use the CSV of log_line_sampler.sh (epoch → line count) with --window (--axes converts the axis timestamps in the run.log
of a bench result directory into line ranges via that CSV and builds per-axis tables).

usage:
  analyze_hived_log.py hived.log [--lines A:B] [--sampler loglines.csv --window EPOCH0:EPOCH1] [--axes RUN_DIR ...] [--json out.json] [--top 6]
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import json
import os
import re
import statistics as stt
import sys
from collections import defaultdict

RE_PROF = re.compile(r"^\[profile M=(\d+)\] total ([\d.]+) ms:(.*)$")
RE_CACHE = re.compile(r"^\[cache\] step (\d+) M=(\d+) routed (\d+) hit (\d+) cpu (\d+) streamed (\d+) · resident (\d+) pending (\d+) / (\d+)")
RE_REQ = re.compile(r"^\[hived\] (\S+): prefill (\d+) tok ([\d.]+) ms · decode (\d+) tok ([\d.]+) ms \(([\d.]+) tok/s, batch (\d+)\) · hit (\d+) cpu (\d+) · resident (\d+)/(\d+) · (?:mtp (\d+)/(\d+) in (\d+) steps · )?(\S+)")
# MTP (HIVE_TRACE_MTP=1) line: draft profile (phase names mtp.*) → verify profile (M = verify rows) → this line. No [cache] line is printed on this path.
RE_MTP = re.compile(r"^\[mtp\] pos (\d+) draft ([\d.]+) ms · verify (\d+) rows ([\d.]+) ms · accepted (\d+)/(\d+) · conf \[[^\]]*\] · hit (\d+) cpu (\d+)")
RE_WARM = re.compile(r"^\[hived\] warm cache: (\d+) experts in ([\d.]+) ms")
RE_FAIL = re.compile(r"^\[hived\] (?:\S+: )?request failed: (.*)$")  # hived prefixes the session id (the older format is still accepted)


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def parse_stages(rest: str) -> dict[str, float]:
    toks = rest.split()
    out = {}
    for i in range(0, len(toks) - 1, 2):
        try:
            out[toks[i]] = out.get(toks[i], 0.0) + float(toks[i + 1])
        except ValueError:
            pass
    return out


def read_lines(path, a=None, b=None):
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if a is not None and i < a:
                continue
            if b is not None and i > b:
                break
            yield i, line.rstrip("\n")


def analyze(path, a=None, b=None):
    dec = defaultdict(list)          # M -> [(total, stages)]
    pre = []                         # (M, total, stages)
    cache = []                       # (M, routed, hit, cpu, streamed, resident)
    reqs, warms, fails = [], [], []
    mtp_draft, mtp_steps = [], []    # draft (total, stages) · verify step dicts
    pending_prof = None
    for _, line in read_lines(path, a, b):
        if line.startswith("[profile"):
            m = RE_PROF.match(line)
            prof = (int(m.group(1)), float(m.group(2)), parse_stages(m.group(3))) if m else None
            if prof and any(k.startswith("mtp.") for k in prof[2]):   # MTP draft (before verification)
                mtp_draft.append((prof[1], prof[2]))
                continue
            if pending_prof is not None:   # the previous profile was not followed by cache/mtp = prefill chunk
                pre.append(pending_prof)
            pending_prof = prof
            continue
        if line.startswith("[mtp]"):
            m = RE_MTP.match(line)
            if m:
                st = {"draft_ms": float(m.group(2)), "rows": int(m.group(3)), "verify_ms": float(m.group(4)), "acc": int(m.group(5)),
                      "drafted": int(m.group(6)), "hit": int(m.group(7)), "cpu": int(m.group(8)), "verify_stages": pending_prof[2] if pending_prof else {}}
                mtp_steps.append(st)
                pending_prof = None
            continue
        if line.startswith("[cache]"):
            m = RE_CACHE.match(line)
            if m:
                M = int(m.group(2))
                cache.append((M, int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7))))
                if pending_prof is not None:
                    dec[pending_prof[0]].append((pending_prof[1], pending_prof[2]))
                    pending_prof = None
            continue
        if pending_prof is not None:
            pre.append(pending_prof)
            pending_prof = None
        if line.startswith("[hived]"):
            m = RE_REQ.match(line)
            if m:
                reqs.append({"sid": m.group(1), "prefill_tok": int(m.group(2)), "prefill_ms": float(m.group(3)), "decode_tok": int(m.group(4)),
                             "decode_ms": float(m.group(5)), "tps": float(m.group(6)), "batch": int(m.group(7)), "hit": int(m.group(8)),
                             "cpu": int(m.group(9)), "finish": m.group(15),
                             "mtp_acc": int(m.group(12)) if m.group(12) else None, "mtp_draft": int(m.group(13)) if m.group(13) else None,
                             "mtp_steps": int(m.group(14)) if m.group(14) else None})
                continue
            m = RE_WARM.match(line)
            if m:
                warms.append((int(m.group(1)), float(m.group(2))))
                continue
            m = RE_FAIL.match(line)
            if m:
                fails.append(m.group(1)[:160])
    if pending_prof is not None:
        pre.append(pending_prof)
    return {"dec": dec, "pre": pre, "cache": cache, "reqs": reqs, "warms": warms, "fails": fails, "mtp_draft": mtp_draft, "mtp": mtp_steps}


def summarize(r, top=6):
    S = {}
    # decode steps: total ms per M · top phases
    dsum = {}
    for M in sorted(r["dec"]):
        rows = r["dec"][M]
        tot = [t for t, _ in rows]
        stage_vals = defaultdict(list)
        for _, st in rows:
            for k, v in st.items():
                stage_vals[k].append(v)
        n = len(rows)
        # phase averages are over **all steps** (0 when absent), not just steps where the phase was printed — so the sum matches the total average even when graph and eager steps mix
        means = {k: sum(v) / n for k, v in stage_vals.items()}
        order = sorted(means, key=lambda k: -means[k])[:top]
        dsum[M] = {"steps": n, "total_mean": stt.mean(tot), "total_p50": pct(tot, .5), "total_p95": pct(tot, .95),
                   "tok_s_per_row": 1000.0 / stt.mean(tot) if tot else None, "tok_s_total": M * 1000.0 / stt.mean(tot) if tot else None,
                   "top": [{"stage": k, "mean": means[k], "p95": pct(stage_vals[k] + [0.0] * (n - len(stage_vals[k])), .95),
                            "share": means[k] / stt.mean(tot)} for k in order],
                   "graph_steps": sum(1 for _, st in rows if any(k.startswith("segA.") for k in st))}
    S["decode_by_M"] = dsum
    ms_ = r.get("mtp") or []
    if ms_:
        acc = sum(x["acc"] for x in ms_); dr = sum(x["drafted"] for x in ms_)
        byrows = defaultdict(list)
        for x in ms_:
            byrows[x["rows"]].append(x["verify_ms"])
        vst = defaultdict(float)
        for x in ms_:
            for k, v in x["verify_stages"].items():
                vst[k] += v
        vtot = sum(x["verify_ms"] for x in ms_) or 1
        S["mtp"] = {"steps": len(ms_), "accepted": acc, "drafted": dr, "accept_rate": acc / dr if dr else None,
                    "draft_ms_mean": stt.mean([x["draft_ms"] for x in ms_]), "verify_ms_by_rows": {k: stt.mean(v) for k, v in sorted(byrows.items())},
                    "verify_top": sorted(((k, v / vtot) for k, v in vst.items()), key=lambda kv: -kv[1])[:top],
                    "hit_rate": sum(x["hit"] for x in ms_) / max(1, sum(x["hit"] + x["cpu"] for x in ms_))}
    # cache
    c = r["cache"]
    if c:
        R = sum(x[1] for x in c); H = sum(x[2] for x in c); C = sum(x[3] for x in c)
        byM = defaultdict(lambda: [0, 0, 0])
        for M, rr, hh, cc, *_ in c:
            byM[M][0] += rr; byM[M][1] += hh; byM[M][2] += cc
        chunks = []
        step = max(1, len(c) // 10)
        for i in range(0, len(c), step):
            seg = c[i:i + step]
            rr = sum(x[1] for x in seg)
            chunks.append(round(sum(x[2] for x in seg) / rr, 3) if rr else None)
        S["cache"] = {"steps": len(c), "routed": R, "hit": H, "cpu": C, "hit_rate": H / R if R else None,
                      "hit_rate_by_M": {M: v[1] / v[0] for M, v in sorted(byM.items()) if v[0]},
                      "hit_rate_trend_10": chunks, "resident_last": c[-1][5], "streamed_total": sum(x[4] for x in c)}
    # prefill chunks
    buckets = defaultdict(list)
    for M, t, st in r["pre"]:
        # ⚠️ M=2688 = decoder tail (decoder_tail): for chunks larger than 2688 the profile prints only the tail row count as M (measured: 8016- and 16016-token chunks both show M=2688)
        #   → token counts and tok/s of this bucket come out lower than reality. Long-prefill throughput is read from the request lines (prefill_by_len).
        key = ("<64" if M < 64 else "64-1023" if M < 1024 else "2688(tail, chunk≥2688)" if M == 2688 else "1024-2687" if M < 2688 else ">2688")
        buckets[key].append((M, t, st))
    psum = {}
    for key, rows in buckets.items():
        toks = sum(M for M, _, _ in rows); ms = sum(t for _, t, _ in rows)
        stage_tot = defaultdict(float)
        for _, _, st in rows:
            for k, v in st.items():
                stage_tot[k] += v
        order = sorted(stage_tot, key=lambda k: -stage_tot[k])[:top]
        psum[key] = {"chunks": len(rows), "tokens": toks, "ms": ms, "tok_s": toks / ms * 1000 if ms else None,
                     "top": [{"stage": k, "share": stage_tot[k] / ms if ms else None, "ms_per_chunk": stage_tot[k] / len(rows)} for k in order]}
    S["prefill_chunks"] = psum
    # requests
    q = r["reqs"]
    if q:
        pb = defaultdict(list)
        for x in q:
            p = x["prefill_tok"]
            key = ("<64" if p < 64 else "64-1023" if p < 1024 else "1024-8191" if p < 8192 else "8192-32767" if p < 32768 else ">=32768")
            pb[key].append(x)
        S["requests"] = {"n": len(q), "finish": dict(sorted(((f, sum(1 for x in q if x["finish"] == f)) for f in {x["finish"] for x in q}))),
                         "prefill_by_len": {k: {"n": len(v), "tok_s": sum(x["prefill_tok"] for x in v) / max(1e-9, sum(x["prefill_ms"] for x in v)) * 1000,
                                                "ms_p50": pct([x["prefill_ms"] for x in v], .5)} for k, v in sorted(pb.items())},
                         "decode_tps_by_batch": {b: {"n": len(v), "tps_p50": pct([x["tps"] for x in v], .5)} for b, v in
                                                 sorted({bb: [x for x in q if x["batch"] == bb and x["decode_tok"] > 8] for bb in {x["batch"] for x in q}}.items()) if v},
                         "req_hit_rate": sum(x["hit"] for x in q) / max(1, sum(x["hit"] + x["cpu"] for x in q))}
    # engine time budget (sum of profile totals — sums of event-timed GPU stream phases, so some host waiting is missing): decode steps vs prefill chunks vs cache warm-up
    dec_ms = sum(t for rows in r["dec"].values() for t, _ in rows) + sum(t for t, _ in r.get("mtp_draft") or []) + sum(x["verify_ms"] for x in r.get("mtp") or [])
    pre_ms = sum(t for _, t, _ in r["pre"])
    warm_ms = sum(w[1] for w in r["warms"])
    tot_ms = dec_ms + pre_ms + warm_ms
    S["time_ms"] = {"decode": dec_ms, "prefill": pre_ms, "warm_cache": warm_ms,
                    "share": {k: (v / tot_ms if tot_ms else None) for k, v in (("decode", dec_ms), ("prefill", pre_ms), ("warm_cache", warm_ms))}}
    S["warm_cache"] = {"n": len(r["warms"]), "experts_mean": stt.mean([w[0] for w in r["warms"]]) if r["warms"] else None,
                       "ms_mean": stt.mean([w[1] for w in r["warms"]]) if r["warms"] else None}
    S["fails"] = {"n": len(r["fails"]), "samples": r["fails"][:5]}
    return S


def fmt(S, title=""):
    L = [f"## {title}" if title else "## hived.log summary"]
    d = S.get("decode_by_M", {})
    if d:
        L += ["", "### Decode steps (M = batch rows)", "", "| M | steps | total ms mean | p50 | p95 | tok/s per row | total tok/s | graph steps | top phases (mean ms · share) |", "| --: | --: | --: | --: | --: | --: | --: | --: | --- |"]
        for M, v in d.items():
            top = " · ".join(f"{t['stage']} {t['mean']:.2f}({t['share']*100:.0f}%)" for t in v["top"])
            L.append(f"| {M} | {v['steps']} | {v['total_mean']:.2f} | {v['total_p50']:.2f} | {v['total_p95']:.2f} | {v['tok_s_per_row']:.1f} | {v['tok_s_total']:.1f} | {v['graph_steps']} | {top} |")
    mt = S.get("mtp")
    if mt:
        L += ["", f"### MTP speculative steps {mt['steps']} · accepted {mt['accepted']}/{mt['drafted']} = {mt['accept_rate']*100:.1f}% · draft mean {mt['draft_ms_mean']:.1f} ms · hit {mt['hit_rate']*100:.1f}%",
              "- verify ms (mean by row count): " + " · ".join(f"{k} rows {v:.1f}" for k, v in mt["verify_ms_by_rows"].items()),
              "- verify phase shares: " + " · ".join(f"{k} {v*100:.0f}%" for k, v in mt["verify_top"])]
    c = S.get("cache")
    if c:
        L += ["", f"### Expert cache: decode steps {c['steps']} · hit {c['hit_rate']*100:.1f}% (hit {c['hit']:,} / routed {c['routed']:,} · CPU {c['cpu']:,}) · resident {c['resident_last']}",
              "- hit by M: " + " · ".join(f"M={M} {v*100:.1f}%" for M, v in c["hit_rate_by_M"].items()),
              "- trend over 10 equal spans: " + " → ".join(f"{x*100:.0f}%" if x is not None else "-" for x in c["hit_rate_trend_10"])]
    p = S.get("prefill_chunks")
    if p:
        L += ["", "### Prefill chunks (one forward = one chunk · the 2688-row decoder tail is printed as M)", "", "| chunk size | chunks | tokens | tok/s | top phases (share · ms per chunk) |", "| --- | --: | --: | --: | --- |"]
        for k in ("<64", "64-1023", "1024-2687", "2688(tail, chunk≥2688)", ">2688"):
            if k in p:
                v = p[k]
                top = " · ".join(f"{t['stage']} {t['share']*100:.0f}%·{t['ms_per_chunk']:.0f}" for t in v["top"])
                L.append(f"| {k} | {v['chunks']} | {v['tokens']:,} | {v['tok_s']:.0f} | {top} |")
    q = S.get("requests")
    if q:
        L += ["", f"### Requests {q['n']} · finish {q['finish']} · per-request hit {q['req_hit_rate']*100:.1f}%",
              "- prefill (per request, tokens excluding the cached prefix): " + " · ".join(f"{k}: {v['n']} req {v['tok_s']:.0f} tok/s(p50 {v['ms_p50']:.0f} ms)" for k, v in q["prefill_by_len"].items()),
              "- decode tok/s (p50 by batch rows at request end): " + " · ".join(f"b{b}: {v['tps_p50']:.1f}(n={v['n']})" for b, v in q["decode_tps_by_batch"].items())]
    tm = S.get("time_ms")
    if tm and (tm["decode"] + tm["prefill"]):
        L.append(f"- engine time (profile sum): decode {tm['decode']/1000:.0f}s({tm['share']['decode']*100:.0f}%) · prefill {tm['prefill']/1000:.0f}s({tm['share']['prefill']*100:.0f}%) · warm_cache {tm['warm_cache']/1000:.0f}s({tm['share']['warm_cache']*100:.0f}%)")
    w = S.get("warm_cache")
    if w and w["n"]:
        L.append(f"- warm_cache {w['n']} runs · mean {w['experts_mean']:.0f} experts · {w['ms_mean']:.0f} ms")
    f = S.get("fails")
    if f and f["n"]:
        L.append(f"- 🔴 request failed {f['n']}: {f['samples']}")
    return "\n".join(L)


def load_sampler(p):
    ts, hl = [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            ts.append(int(row["ts"])); hl.append(int(row["hived_lines"]))
    return ts, hl


def line_at(ts, hl, t):
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        return hl[0]
    if i >= len(ts):
        return hl[-1]
    return hl[i]


def axes_windows(run_dir):
    """Axis marks (`  ▶ <axis>`) in a bench run.log → [(axis, start epoch, end epoch)]. End = next axis start or the end-of-measurement line (see the regex below)."""
    rl = os.path.join(run_dir, "run.log")
    year = dt.date.today().year
    marks = []
    for line in open(rl, errors="replace"):
        m = re.match(r"^\[(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\]\s+▶ (\S+)", line)
        if m:
            t = dt.datetime(year, int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))).timestamp()
            marks.append((m.group(6), int(t)))
    out = []
    for (ax, t0), (_, t1) in zip(marks, marks[1:]):
        if ax != "END":
            out.append((ax, t0, t1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--lines", default="")
    ap.add_argument("--sampler", default="")
    ap.add_argument("--window", default="")
    ap.add_argument("--axes", nargs="*", default=[])
    ap.add_argument("--json", default="")
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    out = {}
    if a.axes:
        ts, hl = load_sampler(a.sampler)
        for rd in a.axes:
            for ax, t0, t1 in axes_windows(rd):
                l0, l1 = line_at(ts, hl, t0), line_at(ts, hl, t1)
                S = summarize(analyze(a.log, l0, l1), a.top)
                S["lines"] = [l0, l1]
                out[f"{os.path.basename(rd.rstrip('/'))}/{ax}"] = S
                print(fmt(S, f"{os.path.basename(rd.rstrip('/'))} · {ax} (hived.log {l0}~{l1})"), "\n", flush=True)
    else:
        l0 = l1 = None
        if a.lines:
            x, y = a.lines.split(":")
            l0, l1 = (int(x) if x else None), (int(y) if y else None)
        elif a.window:
            ts, hl = load_sampler(a.sampler)
            x, y = a.window.split(":")
            l0, l1 = line_at(ts, hl, int(x)), line_at(ts, hl, int(y))
        S = summarize(analyze(a.log, l0, l1), a.top)
        S["lines"] = [l0, l1]
        out["all"] = S
        print(fmt(S, f"hived.log {l0 or 1}~{l1 or 'end'}"))
    if a.json:
        json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
