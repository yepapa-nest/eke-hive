#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Compares the dumps of two engine runs (graph path vs eager path, etc.).
   route_ids must match bit for bit; h_final/logits are checked by cos and max|Δ| (they vary with the atomic-add order of the CPU expert sums).
Usage: compare_runs.py A_DIR B_DIR"""
import glob
import os
import sys

import numpy as np

a_dir, b_dir = sys.argv[1], sys.argv[2]
names_a = {os.path.basename(p) for p in glob.glob(os.path.join(a_dir, 's*_*'))}
names_b = {os.path.basename(p) for p in glob.glob(os.path.join(b_dir, 's*_*'))}
ok = bool(names_a) and names_a == names_b
if not ok: print('FAIL: empty or mismatched dump inventory')


def load(path):
    if path.endswith(".bf16"):
        u = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32)
    if path.endswith(".f32"):
        return np.fromfile(path, dtype=np.float32)
    return np.fromfile(path, dtype=np.int32)


for pa in sorted(glob.glob(os.path.join(a_dir, "s*_*"))):
    name = os.path.basename(pa)
    pb = os.path.join(b_dir, name)
    if not os.path.exists(pb):
        print(f"{name}: missing in B")
        ok = False
        continue
    x, y = load(pa), load(pb)
    if not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        print(f'{name}: empty/nonfinite FAIL')
        ok = False
        continue
    if x.shape != y.shape:
        print(f"{name}: shape differs {x.shape} vs {y.shape}")
        ok = False
        continue
    if x.dtype == np.int32:
        neq = int((x != y).sum())
        print(f"{name}: mismatches {neq}/{x.size} {'ok' if neq == 0 else 'FAIL'}")
        ok &= neq == 0
    else:
        xf, yf = x.astype(np.float64), y.astype(np.float64)
        cos = float(xf @ yf / (np.linalg.norm(xf) * np.linalg.norm(yf) + 1e-30))
        md = float(np.abs(xf - yf).max())
        good = np.array_equal(x,y) or (cos > 0.99999 and md < 1e-2 * max(1.0, float(np.abs(xf).max())))
        print(f"{name}: cos={cos:.6f} max|Δ|={md:.3e} (ref max {np.abs(xf).max():.3e}) {'ok' if good else 'FAIL'}")
        ok &= good
print("ALL OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
