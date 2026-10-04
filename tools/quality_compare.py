#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Compare two quality_eval.py results (BASELINE, FINAL) — per-suite pass rates, deltas, per-item flips, exact McNemar test p.

No verdict thresholds: only the numbers (both pass rates, b = pass→fail, c = fail→pass, two-sided exact binomial p) and the flipped items are reported.
Decoding is greedy (temperature 0), so for a change claimed to be "bit-identical" the outputs themselves must be equal — the number of items whose output hash differs is reported too.
Usage: quality_compare.py baseline.json final.json [--json out.json]
"""
import argparse
import json
import math
import statistics
import sys


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: binomial (0.5) probability of ≤ min(b,c) among n=b+c discordant pairs, ×2 (capped at 1). b=c=0 → 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _load(p):
    with open(p) as f:
        return json.load(f)


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def compare(base, final):
    bi = {r["id"]: r for r in base["items"]}
    fi = {r["id"]: r for r in final["items"]}
    suites = []
    for r in base["items"] + final["items"]:
        if r["suite"] not in suites:
            suites.append(r["suite"])
    out = {"suites": {}, "only_in_baseline": sorted(set(bi) - set(fi)), "only_in_final": sorted(set(fi) - set(bi)),
           "prompt_mismatch": sorted(i for i in set(bi) & set(fi) if bi[i].get("prompt_sha") and bi[i].get("prompt_sha") != fi[i].get("prompt_sha"))}
    for s in suites:
        ids = [i for i in bi if bi[i]["suite"] == s and i in fi]
        paired = [i for i in ids if bi[i].get("pass") is not None and fi[i].get("pass") is not None]
        skipped = [i for i in ids if i not in paired]
        b_pass = sum(1 for i in paired if bi[i]["pass"])
        f_pass = sum(1 for i in paired if fi[i]["pass"])
        p2f = [i for i in paired if bi[i]["pass"] and not fi[i]["pass"]]
        f2p = [i for i in paired if not bi[i]["pass"] and fi[i]["pass"]]
        diff_out = [i for i in paired if bi[i].get("output_sha") != fi[i].get("output_sha")]
        n = len(paired)

        def lat(side, key):
            return _med([side[i].get(key) for i in paired])
        out["suites"][s] = {
            "n": n, "skipped": skipped, "baseline_pass": b_pass, "final_pass": f_pass,
            "baseline_rate": b_pass / n if n else None, "final_rate": f_pass / n if n else None,
            "delta": (f_pass - b_pass) / n if n else None,
            "pass_to_fail": p2f, "fail_to_pass": f2p, "mcnemar_b": len(p2f), "mcnemar_c": len(f2p), "p": mcnemar_exact(len(p2f), len(f2p)),
            "output_differs": diff_out,
            "latency_med_s": [lat(bi, "latency_s"), lat(fi, "latency_s")], "ttft_med_s": [lat(bi, "ttft_s"), lat(fi, "ttft_s")],
        }
    return out


def _f(x, pct=False):
    if x is None:
        return "-"
    return f"{x * 100:.1f}%" if pct else (f"{x:.3f}" if isinstance(x, float) else str(x))


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("baseline")
    cli.add_argument("final")
    cli.add_argument("--json", default=None)
    a = cli.parse_args(argv)
    base, final = _load(a.baseline), _load(a.final)
    res = compare(base, final)
    bl, fl = base["meta"].get("label") or "baseline", final["meta"].get("label") or "final"
    print(f"baseline = {a.baseline} ({bl}, seed {base['meta'].get('seed')}) · final = {a.final} ({fl}, seed {final['meta'].get('seed')})")
    if base["meta"].get("seed") != final["meta"].get("seed"):
        print("  ⚠ seeds differ — items are not the same")
    if base["meta"].get("chars_per_token") != final["meta"].get("chars_per_token"):
        print(f"  ⚠ chars_per_token differ: {base['meta'].get('chars_per_token')} vs {final['meta'].get('chars_per_token')}")
    for k in ("only_in_baseline", "only_in_final", "prompt_mismatch"):
        if res[k]:
            print(f"  ⚠ {k}: {res[k]}")
    print()
    print(f"{'suite':<6} {'n':>4} {'baseline':>14} {'final':>14} {'delta':>8} {'p→f':>4} {'f→p':>4} {'McNemar p':>10} {'out≠':>5}  lat med b/f (s)   ttft med b/f (s)")
    for s, v in res["suites"].items():
        print(f"{s:<6} {v['n']:>4} {v['baseline_pass']:>4} ({_f(v['baseline_rate'], True):>6}) {v['final_pass']:>4} ({_f(v['final_rate'], True):>6}) "
              f"{(v['delta'] or 0) * 100:>+7.1f}% {v['mcnemar_b']:>4} {v['mcnemar_c']:>4} {v['p']:>10.4f} {len(v['output_differs']):>5}  "
              f"{_f(v['latency_med_s'][0])}/{_f(v['latency_med_s'][1])}   {_f(v['ttft_med_s'][0])}/{_f(v['ttft_med_s'][1])}")
    print()
    bi = {r["id"]: r for r in base["items"]}
    fi = {r["id"]: r for r in final["items"]}
    for s, v in res["suites"].items():
        print(f"[{s}] verdict: baseline {v['baseline_pass']}/{v['n']} · final {v['final_pass']}/{v['n']} · pass→fail {v['mcnemar_b']} · fail→pass {v['mcnemar_c']}"
              f" · exact McNemar p = {v['p']:.4g} · outputs differing {len(v['output_differs'])}/{v['n']}" + (f" · skipped {v['skipped']}" if v["skipped"] else ""))
        for tag, lst in (("pass→fail", v["pass_to_fail"]), ("fail→pass", v["fail_to_pass"])):
            for i in lst:
                print(f"    {tag} {i}\n        baseline: {(bi[i].get('output') or '')[:160]!r}\n        final   : {(fi[i].get('output') or '')[:160]!r}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
