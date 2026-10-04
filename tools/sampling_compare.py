#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Compare sample distributions of two gap_eval.py --suite sampling results (e.g. MTP off vs on) — numbers only, no thresholds.

Per prompt: chi-square homogeneity test of the first-word distribution and the first stream-chunk distribution (categories with expected count < 5 merged into OTHER, p),
mean output length (tokens) with a permutation test p (10,000 rounds, seed 0), accuracy with Fisher exact test p, and the number of identical outputs for the same seed.
With lossless sampling (same RNG consumption) per-seed outputs may be identical; otherwise only the distributions should match — read both numbers together.
Usage: sampling_compare.py A.json B.json [--json out.json]
"""
import argparse
import json
import math
import random
import statistics
import sys


def _gammainc_q(s, x):
    """Regularised upper incomplete gamma Q(s, x) (Numerical Recipes: series / continued fraction)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(s)
    if x < s + 1:
        ap, tot, d = s, 1.0 / s, 1.0 / s
        for _ in range(1000):
            ap += 1
            d *= x / ap
            tot += d
            if abs(d) < abs(tot) * 1e-15:
                break
        return max(0.0, 1.0 - tot * math.exp(-x + s * math.log(x) - gln))
    b, c, d = x + 1 - s, 1e300, 1.0 / (x + 1 - s)
    hh = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        hh *= de
        if abs(de - 1) < 1e-15:
            break
    return min(1.0, math.exp(-x + s * math.log(x) - gln) * hh)


def chi2_sf(x, df):
    return _gammainc_q(df / 2.0, x / 2.0) if df > 0 else 1.0


def chi2_homogeneity(ca, cb):
    """2xk homogeneity. Small categories (expected < 5) are merged into OTHER. -> (chi2, df, p, merged table)."""
    na, nb = sum(ca.values()), sum(cb.values())
    if na == 0 or nb == 0:
        return None, 0, None, {}
    n = na + nb
    keys = sorted(set(ca) | set(cb), key=lambda k: -(ca.get(k, 0) + cb.get(k, 0)))
    table, other = {}, [0, 0]
    for k in keys:
        a, b = ca.get(k, 0), cb.get(k, 0)
        if min(na, nb) * (a + b) / n < 5:
            other[0] += a
            other[1] += b
        else:
            table[k] = [a, b]
    if sum(other):
        table["<OTHER>"] = other
    # If OTHER itself is still small, merge it into the smallest category
    if "<OTHER>" in table and len(table) > 1 and min(na, nb) * sum(table["<OTHER>"]) / n < 5:
        o = table.pop("<OTHER>")
        k = min(table, key=lambda kk: sum(table[kk]))
        table[k] = [table[k][0] + o[0], table[k][1] + o[1]]
    if len(table) < 2:
        return 0.0, 0, 1.0, table
    x = 0.0
    for a, b in table.values():
        t = a + b
        for obs, nn in ((a, na), (b, nb)):
            e = t * nn / n
            x += (obs - e) ** 2 / e
    df = len(table) - 1
    return x, df, chi2_sf(x, df), table


def perm_test_mean(xa, xb, iters=10000, seed=0):
    if not xa or not xb:
        return None
    obs = abs(statistics.mean(xa) - statistics.mean(xb))
    pool = list(xa) + list(xb)
    rng = random.Random(seed)
    na, hit = len(xa), 0
    for _ in range(iters):
        rng.shuffle(pool)
        if abs(statistics.mean(pool[:na]) - statistics.mean(pool[na:])) >= obs - 1e-12:
            hit += 1
    return (hit + 1) / (iters + 1)


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact test p for 2x2 [[a,b],[c,d]]."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d

    def pr(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p0 = pr(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9)))


def compare(A, B):
    sa, sb = A["sampling"], B["sampling"]
    out = {"params": [sa["params"], sb["params"]], "prompts": {}}
    for pid in sa["samples"]:
        if pid not in sb["samples"]:
            continue
        xa = [x for x in sa["samples"][pid] if x]
        xb = [x for x in sb["samples"][pid] if x]
        res = {"n": [len(xa), len(xb)], "errors": [sum(1 for x in xa if x.get("error")), sum(1 for x in xb if x.get("error"))]}
        for key in ("first_word", "first_piece"):
            ca, cb = {}, {}
            for x in xa:
                ca[x.get(key) or ""] = ca.get(x.get(key) or "", 0) + 1
            for x in xb:
                cb[x.get(key) or ""] = cb.get(x.get(key) or "", 0) + 1
            chi, df, p, table = chi2_homogeneity(ca, cb)
            res[key] = {"chi2": None if chi is None else round(chi, 4), "df": df, "p": None if p is None else round(p, 6), "table": table}
        la = [x["completion_tokens"] for x in xa if x.get("completion_tokens") is not None]
        lb = [x["completion_tokens"] for x in xb if x.get("completion_tokens") is not None]
        res["length"] = {"mean": [round(statistics.mean(la), 3) if la else None, round(statistics.mean(lb), 3) if lb else None],
                         "sd": [round(statistics.pstdev(la), 3) if la else None, round(statistics.pstdev(lb), 3) if lb else None],
                         "perm_p": perm_test_mean(la, lb)}
        if any(x.get("correct") is not None for x in xa):
            ka, kb = sum(1 for x in xa if x.get("correct")), sum(1 for x in xb if x.get("correct"))
            res["accuracy"] = {"correct": [ka, kb], "rate": [ka / len(xa) if xa else None, kb / len(xb) if xb else None],
                               "fisher_p": fisher_exact(ka, len(xa) - ka, kb, len(xb) - kb)}
        by_seed_b = {x["seed"]: x for x in xb}
        same = sum(1 for x in xa if x["seed"] in by_seed_b and x.get("text") == by_seed_b[x["seed"]].get("text"))
        res["identical_by_seed"] = [same, len(xa)]
        out["prompts"][pid] = res
    return out


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("a")
    cli.add_argument("b")
    cli.add_argument("--json", default=None)
    x = cli.parse_args(argv)
    A, B = json.load(open(x.a)), json.load(open(x.b))
    la, lb = A["meta"].get("label") or "A", B["meta"].get("label") or "B"
    res = compare(A, B)
    if res["params"][0] != res["params"][1]:
        print(f"⚠ sampling params differ: {res['params'][0]} vs {res['params'][1]}")
    print(f"A = {x.a} ({la}) · B = {x.b} ({lb}) · params {res['params'][0]}")
    print(f"{'prompt':<10} {'n':>7} {'1st-word χ²/df p':>20} {'1st-piece χ²/df p':>20} {'len A/B':>13} {'perm p':>7} {'acc A/B':>9} {'Fisher p':>8} {'same@seed':>9}")
    for pid, r in res["prompts"].items():
        fw, fp, ln = r["first_word"], r["first_piece"], r["length"]
        acc = r.get("accuracy")
        print(f"{pid:<10} {r['n'][0]:>3}/{r['n'][1]:<3} {(fw['chi2'] or 0):>8.2f}/{fw['df']:<2} {fw['p'] if fw['p'] is not None else '-':>8} "
              f"{(fp['chi2'] or 0):>8.2f}/{fp['df']:<2} {fp['p'] if fp['p'] is not None else '-':>8} {ln['mean'][0]!s:>6}/{ln['mean'][1]!s:<6} "
              f"{ln['perm_p'] if ln['perm_p'] is not None else '-':>7.4} "
              + (f"{acc['correct'][0]:>4}/{acc['correct'][1]:<4} {acc['fisher_p']:>8.4f}" if acc else f"{'-':>9} {'-':>8}")
              + f" {r['identical_by_seed'][0]:>4}/{r['identical_by_seed'][1]}" + (f"  errors {r['errors']}" if any(r["errors"]) else ""))
    print("\nfirst-word tables (A, B):")
    for pid, r in res["prompts"].items():
        print(f"  {pid}: " + " · ".join(f"{k!r} {v[0]}/{v[1]}" for k, v in r["first_word"]["table"].items()))
    if x.json:
        with open(x.json, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
