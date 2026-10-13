#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Compare an engine dump (q27_cpu_check / q27_check) with the q27 oracle (oracle/q27_oracle.py) — per layer residual stream and logits
(adapted from tools/compare_qwen.py; the same format). vision.npy (encoder output) is compared too when both directories have it.
Per-layer injection results (q27_cpu_check with Q27_CHECK_INJECT) are compared by passing ENGINE_DIR/inject.

Both directories hold h_LNN.npy ([rows, 5120] float32, hidden stream after layer NN) and logits.npy ([rows, V] float32). Rows are
compared up to the shorter of the two. Prints per layer: cosine (min over rows), max |diff|, relative RMS; for logits: top-1 agreement and
top-k overlap per row. Exit code 1 when the lowest layer cosine is below --min-cos, the top-1 agreement is below --min-top1 or the mean
top-k overlap is below --min-topk.

  python3 tools/compare_q27.py ORACLE_DIR ENGINE_DIR [--min-cos 0.999] [--min-top1 1.0] [--topk 20 --min-topk 0.8]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.linalg.norm(a64, axis=-1), np.linalg.norm(b64, axis=-1)
    return (a64 * b64).sum(-1) / np.maximum(na * nb, 1e-30)


def main() -> int:
    ap = argparse.ArgumentParser(description="Engine vs oracle comparison for Qwen3.8-27B.")
    ap.add_argument("oracle")
    ap.add_argument("engine")
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--min-top1", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--min-topk", type=float, default=0.8)
    a = ap.parse_args()
    fail = False
    layers = sorted(int(re.search(r"h_L(\d+)\.npy$", p).group(1)) for p in glob.glob(os.path.join(a.oracle, "h_L*.npy")))
    worst = (2.0, -1)
    for l in layers:
        pe = os.path.join(a.engine, f"h_L{l:02d}.npy")
        if not os.path.exists(pe):
            print(f"layer {l:2d}: missing in engine dump")
            continue
        o, e = np.load(os.path.join(a.oracle, f"h_L{l:02d}.npy")), np.load(pe)
        n = min(len(o), len(e))
        o, e = o[:n].reshape(n, -1), e[:n].reshape(n, -1)
        c = cos_rows(o, e)
        d = np.abs(o - e)
        rel = np.sqrt(((o - e) ** 2).mean()) / max(np.sqrt((o ** 2).mean()), 1e-30)
        print(f"layer {l:2d}: rows {n} · cos min {c.min():.6f} mean {c.mean():.6f} · max|diff| {d.max():.4g} · rel rms {rel:.3e}")
        if c.min() < worst[0]:
            worst = (float(c.min()), l)
    if layers:
        print(f"lowest layer cosine {worst[0]:.6f} (layer {worst[1]})")
        if worst[0] < a.min_cos:
            fail = True
    vo, ve = os.path.join(a.oracle, "vision.npy"), os.path.join(a.engine, "vision.npy")
    if os.path.exists(vo) and os.path.exists(ve):
        o, e = np.load(vo), np.load(ve)
        c = cos_rows(o, e)
        rel = np.sqrt(((o - e) ** 2).mean()) / max(np.sqrt((o ** 2).mean()), 1e-30)
        print(f"vision: rows {len(o)} · cos min {c.min():.6f} mean {c.mean():.6f} · max|diff| {np.abs(o - e).max():.4g} · rel rms {rel:.3e}")
        if c.min() < a.min_cos:
            fail = True
    po, pe = os.path.join(a.oracle, "logits.npy"), os.path.join(a.engine, "logits.npy")
    if os.path.exists(po) and os.path.exists(pe):
        o, e = np.load(po), np.load(pe)
        # an oracle run with --all-logits holds every prompt row: align its last prompt row with the engine's first row
        dj = os.path.join(a.oracle, "decode.json")
        if os.path.exists(dj):
            import json
            d = json.load(open(dj))
            T, nd = len(d.get("prompt", [])), len(d.get("decoded", []))
            if T > 1 and len(o) == T + nd:
                o = o[T - 1:]
        n = min(len(o), len(e))
        top1, ov = [], []
        for r in range(n):
            to, te = np.argsort(-o[r])[: a.topk], np.argsort(-e[r])[: a.topk]
            top1.append(to[0] == te[0])
            ov.append(len(set(to.tolist()) & set(te.tolist())) / a.topk)
            print(f"logits row {r}: top1 {to[0]} vs {te[0]} {'ok' if to[0] == te[0] else 'DIFF'} · top-{a.topk} overlap {ov[-1]:.2f} · "
                  f"cos {cos_rows(o[r:r + 1], e[r:r + 1])[0]:.6f}")
        t1, mo = float(np.mean(top1)), float(np.mean(ov))
        print(f"logits: top-1 agreement {t1:.3f} · mean top-{a.topk} overlap {mo:.3f} over {n} rows")
        if t1 < a.min_top1 or mo < a.min_topk:
            fail = True
    else:
        print("logits.npy missing in one of the directories")
    mo_, me_ = os.path.join(a.oracle, "mtp_top.json"), os.path.join(a.engine, "mtp_last.npy")
    if os.path.exists(mo_) and os.path.exists(me_):
        import json
        top = json.load(open(mo_))
        oi, ol = top["ids"][-1], np.array(top["logits"][-1])
        el = np.load(me_)[0]
        ei = np.argsort(-el)[: len(oi)]
        ov = len(set(oi) & set(ei.tolist())) / len(oi)
        d = np.abs(el[oi] - ol).max()
        print(f"MTP last entry: top1 {oi[0]} vs {ei[0]} {'ok' if oi[0] == ei[0] else 'DIFF'} · top-{len(oi)} overlap {ov:.2f} · max|logit diff| over the "
              f"oracle's top-{len(oi)} {d:.4g}")
        if oi[0] != ei[0]:
            fail = True
    print("FAIL" if fail else "PASS")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
