#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Unpack the per-layer h and premix arrays of a golden (npz) into the raw f32 files the engine reads (for per-layer injection checks — engine --inject DIR).
File names follow the engine dump convention: s0_ = prefill, s1_.. = decode steps. Usage: export_golden_raw.py GOLDEN_DIR OUT_DIR"""
import glob
import os
import sys

import numpy as np

gdir, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
steps = [("prefill.npz", 0)] + [(os.path.basename(p), i + 1) for i, p in enumerate(sorted(glob.glob(os.path.join(gdir, "decode_*.npz"))))]
n = 0
for name, s in steps:
    g = np.load(os.path.join(gdir, name))
    for k in g.files:
        if k.startswith("h_L") or k.startswith("premix_L"):
            l = int(k.split("L")[1])
            base = "h_L" if k.startswith("h_") else "premix_L"
            g[k].astype(np.float32).tofile(os.path.join(out, f"s{s}_{base}{l}.f32"))
            n += 1
print(f"exported {n} files → {out}")
