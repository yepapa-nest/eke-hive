#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Compare the DSpark draft golden (oracle --spec: spec.npz/spec.json) with engine dumps (hive --decode 1 --spec 1 --dump DIR).
Usage: compare_spec.py --golden /out/golden/spec1 --dump /out/dump/spec1
Compared: main_x of the decode step (target-layer hidden -> main_proj -> norm), per-stage routing, per-stage output h/pre_mix, head input x, markov embedding, logits (after bias), draft ids, confidences"""
import argparse
import glob
import json
import os
import re

import numpy as np


def bf16_file(path, shape):
    raw = np.fromfile(path, dtype=np.uint16)
    f = (raw.astype(np.uint32) << 16).view(np.float32)
    return f.reshape(shape)


def stats(a, b):
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('inf'), float('nan'), 'FAIL: shape/empty/nonfinite'
    d = np.abs(a - b)
    ref = np.abs(b).max() + 1e-30
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return d.max() / ref, cos, f"max|Δ|={d.max():.3e} mean|Δ|={d.mean():.3e} rel={d.max() / ref:.3e} cos={cos:.6f} (ref max {np.abs(b).max():.3e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--rel", type=float, default=2e-2, help="pass threshold (rel)")
    a = ap.parse_args()
    g = np.load(os.path.join(a.golden, "spec.npz"))
    gj = json.load(open(os.path.join(a.golden, "spec.json")))
    ids_files = sorted(glob.glob(os.path.join(a.dump, "s*_mtp_ids.i32")), key=lambda p: int(re.search(r"s(\d+)_", os.path.basename(p)).group(1)))
    assert ids_files, "no s*_mtp_ids.i32 in the engine dump (was it run with --spec 1 --dump?)"
    step = int(re.search(r"s(\d+)_", os.path.basename(ids_files[-1])).group(1))
    pre = os.path.join(a.dump, f"s{step}_")
    dec_pre = os.path.join(a.dump, f"s{step - 1}_")  # mtp_sync dump of the previous forward (one decode token)
    B, V = g["spec_logits"].shape
    dim = g["spec_x"].shape[1]
    rank = g["spec_markov"].shape[1]
    worst = 0.0
    missing = False
    routing_ok = True
    print(f"== spec step s{step}: block {B} · tok {gj['tok']} · pos {gj['pos']}")
    # main_x of the decode step (one row)
    p = dec_pre + "mtp_main_x.bf16"
    if os.path.exists(p):
        eng = bf16_file(p, (-1, dim))[-1:]
        rel, cos, msg = stats(eng, g["spec_main_x"][-1:])
        worst = max(worst, rel)
        print(f"main_x(decode)      {msg}")
    else:
        missing = True
        print("main_x(decode)      (missing)")
    # Per-stage routing and outputs
    stages = sorted(int(k[len("spec_h_M"):]) for k in g.files if k.startswith("spec_h_M"))
    if not stages: missing = True
    n_layers = 40
    for s in stages:
        l = n_layers + s
        rp = pre + f"route_ids_L{l}.i32"
        if os.path.exists(rp) and f"L{l:02d}" in gj.get("routing", {}):
            eng_ids = np.fromfile(rp, dtype=np.int32).reshape(B, -1)
            ref_ids = np.array(gj["routing"][f"L{l:02d}"]["ids"])
            same = sum(set(eng_ids[m].tolist()) == set(ref_ids[m].tolist()) for m in range(B))
            routing_ok &= same == B
            print(f"stage {s} routing     {same}/{B} rows identical" + ("" if same == B else f"  eng={eng_ids.tolist()} ref={ref_ids.tolist()}"))
        else:
            missing = True
        hp = pre + f"h_L{l}.bf16"
        if os.path.exists(hp):
            ref = g[f"spec_h_M{s}"]
            eng = bf16_file(hp, ref.shape)
            rel, cos, msg = stats(eng, ref)
            worst = max(worst, rel)
            print(f"stage {s} h           {msg}")
            pm = np.fromfile(pre + f"premix_L{l}.f32", dtype=np.float32).reshape(g[f"spec_premix_M{s}"].shape)
            rel, cos, msg = stats(pm, g[f"spec_premix_M{s}"])
            worst = max(worst, rel)
            print(f"stage {s} pre_mix     {msg}")
        else:
            missing = True
            print(f"stage {s} h           (no dump: {hp})")
    for name, key, shape, kind in [("head x", "spec_x", (B, dim), "bf16"), ("markov emb", "spec_markov", (B, rank), "bf16"),
                                   ("logits", "spec_logits", (B, V), "f32"), ("conf", "spec_conf", (B,), "f32")]:
        p = pre + {"spec_x": "mtp_x.bf16", "spec_markov": "mtp_markov.bf16", "spec_logits": "mtp_logits.f32", "spec_conf": "mtp_conf.f32"}[key]
        if not os.path.exists(p):
            missing = True
            print(f"{name:<20}(missing)")
            continue
        eng = bf16_file(p, shape) if kind == "bf16" else np.fromfile(p, dtype=np.float32).reshape(shape)
        rel, cos, msg = stats(eng, g[key])
        worst = max(worst, rel)
        print(f"{name:<20}{msg}")
    eng_ids = np.fromfile(pre + "mtp_ids.i32", dtype=np.int32)
    ref_ids = g["spec_ids"]
    same = int((eng_ids == ref_ids).sum())
    print(f"draft ids           eng={eng_ids.tolist()} ref={ref_ids.tolist()} → {same}/{len(ref_ids)} identical")
    lg = np.fromfile(pre + "mtp_logits.f32", dtype=np.float32).reshape(B, V)
    top_eng = lg.argmax(-1); top_ref = g["spec_logits"].argmax(-1)
    print(f"row argmax          eng={top_eng.tolist()} ref={top_ref.tolist()}")
    ok = not missing and routing_ok and worst <= a.rel and eng_ids.shape == ref_ids.shape and same == len(ref_ids)
    print(f"== {'OK' if ok else 'FAIL'} (worst rel {worst:.3e} ≤ {a.rel}, ids {same}/{len(ref_ids)})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
