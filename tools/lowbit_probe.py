#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Low-bit (2/3-bit) cold-expert quantisation error probe — re-quantises real experts (MXFP4 dequantised) with group-32 fp16-scale RTN and
measures relative weight error and relative output error (with fp8 activations). Per group the scale is the min-squared-error choice among a few candidates (simple search).
Usage: lowbit_probe.py [--layer 0] [--experts 0,7,100] [--bits 2,3]"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle"))
import dsv41_oracle as O  # noqa: E402


def rtn_group(w: np.ndarray, bits: int, group: int = 32, search: int = 6):
    """w [N,K] fp32 -> symmetric uniform grid {+-0.5, +-1.5, ...}*s; per-group s is the min-squared-error choice among `search` candidates scaled from max. Returns the dequantised w'"""
    N, K = w.shape
    g = w.reshape(N, K // group, group)
    levels = 2 ** (bits - 1)  # 2bit: 2 → {±0.5,±1.5}, 3bit: 4 → {±0.5..±3.5}
    qmax = levels - 0.5
    amax = np.abs(g).max(-1, keepdims=True) + 1e-12
    best = None
    best_err = None
    for f in np.linspace(0.6, 1.0, search):  # candidates that clip the max to shrink the scale
        s = (amax * f) / qmax
        q = np.clip(np.round(g / s - 0.5) + 0.5, -qmax, qmax)
        deq = (q * s).astype(np.float32)
        err = ((deq - g) ** 2).sum(-1, keepdims=True)
        if best is None:
            best, best_err = deq, err
        else:
            m = err < best_err
            best = np.where(m, deq, best)
            best_err = np.where(m, err, best_err)
    return best.reshape(N, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("HIVE_CKPT"), help="checkpoint directory (default: $HIVE_CKPT)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", default="0,7,100,255,383")
    ap.add_argument("--bits", default="2,3")
    a = ap.parse_args()
    if not a.ckpt:
        ap.error("--ckpt DIR (or HIVE_CKPT) is required")
    ck = O.Ckpt(a.ckpt)
    rng = np.random.default_rng(0)
    bits_list = [int(b) for b in a.bits.split(",")]
    print(f"layer {a.layer} · experts {a.experts} · relative RMS weight error / relative RMS output error (64 random fp8 activation rows, w1·w3·w2 each)")
    for e in [int(x) for x in a.experts.split(",")]:
        p = f"layers.{a.layer}.ffn.experts.{e}"
        row = [f"e{e:3d}"]
        for name, K in (("w1", 5120), ("w3", 5120), ("w2", 2304)):
            w = O.deq_fp4(ck.get(f"{p}.{name}.weight"), ck.get(f"{p}.{name}.scale")).float().numpy()  # [N,K]
            x = rng.standard_normal((64, w.shape[1])).astype(np.float32)
            y0 = x @ w.T
            cell = []
            for b in bits_list:
                wq = rtn_group(w, b)
                werr = np.sqrt(((wq - w) ** 2).mean()) / (np.sqrt((w ** 2).mean()) + 1e-12)
                y = x @ wq.T
                yerr = np.sqrt(((y - y0) ** 2).mean()) / (np.sqrt((y0 ** 2).mean()) + 1e-12)
                cell.append(f"{b}b w{werr:.3f}/y{yerr:.3f}")
            row.append(f"{name}: " + " ".join(cell))
        print(" · ".join(row))
    print("note: the MXFP4 grid itself (e2m1, group-32 e8m0) is {0,.5,1,1.5,2,3,4,6}·2^k — a uniform 3-bit grid is denser for small values and coarser for large ones.")


if __name__ == "__main__":
    main()
