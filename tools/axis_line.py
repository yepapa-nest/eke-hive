#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""axis_line.py <bench run directory> <axis> [--sampler CSV] [--log hived.log] — one-line summary of one axis (for interim progress reports).
Values: TTFT median/P95 · TPOT median (→ tok/s per stream) · system output tok/s · input tok/s (prefill axes) from <axis>.txt (sglang.benchmark.serving output)
    + expert hit rate · MTP acceptance (accepted/drafted · tokens per step) over that axis's span of hived.log (time → line via the log_line_sampler CSV)."""
import argparse, os, re, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import analyze_hived_log as A

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("axis")
ap.add_argument("--sampler", default=None, help="log_line_sampler.sh CSV (time → hived.log line); required when the run has axis windows")
ap.add_argument("--log", default=os.path.join(os.environ.get("HIVE_STATE_DIR") or os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir)), "run"), "logs", "hived.log"),
                help="hived.log (default: $HIVE_STATE_DIR/logs/hived.log or run/logs/hived.log)")
a = ap.parse_args()
t = open(os.path.join(a.run, a.axis + ".txt"), errors="ignore").read()
g = lambda k: (re.search(rf"{re.escape(k)}:\s+([\d.]+)", t) or [None, None])[1]
ok = g("Successful requests"); ttft, ttft95, tpot, otp, itp = g("Median TTFT (ms)"), g("P95 TTFT (ms)"), g("Median TPOT (ms)"), g("Output token throughput (tok/s)"), g("Input token throughput (tok/s)")
win = [w for w in A.axes_windows(a.run) if w[0] == a.axis]
extra = ""
if win:
    if not a.sampler:
        ap.error("--sampler CSV is required to map this axis to its hived.log span")
    ts, hl = A.load_sampler(a.sampler)
    l0, l1 = A.line_at(ts, hl, win[0][1]), A.line_at(ts, hl, win[0][2])
    r = A.analyze(a.log, l0, l1); S = A.summarize(r)
    hits = []
    if S.get("cache"): hits.append(f"step hit {S['cache']['hit_rate']*100:.1f}%")
    if S.get("mtp"):
        m = r["mtp"]; toks = sum(x["acc"] + 1 for x in m)
        hits.append(f"MTP accepted {S['mtp']['accepted']}/{S['mtp']['drafted']}={S['mtp']['accept_rate']*100:.1f}% · {toks/len(m):.2f} tokens/step · verify hit {S['mtp']['hit_rate']*100:.1f}%")
    if S.get("requests"): hits.append(f"request hit {S['requests']['req_hit_rate']*100:.1f}%")
    extra = " · " + " · ".join(hits) + f" (hived.log {l0}~{l1})"
dec = f"{1000/float(tpot):.1f}" if tpot and float(tpot) > 0 else "-"
line = f"{a.axis}: ok {ok} · TTFT median/P95 {float(ttft):,.0f}/{float(ttft95):,.0f} ms · TPOT median {float(tpot):.2f} ms ({dec} tok/s per stream) · total output {float(otp):.1f} tok/s" + (f" · input {float(itp):,.0f} tok/s" if a.axis.startswith("prefill") and itp else "") + extra
print(line)
