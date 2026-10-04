#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Per-stage ViT comparison: oracle bf16 reference (vit_<start>_*), fp32 reference (vit32_<start>_*), engine (HIVE_VISION_DUMP)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from compare_golden import bf16_file, stats
g = np.load(os.path.join(sys.argv[1], "prefill.npz")); dump = sys.argv[2]
start = sorted(int(k.split("_")[1]) for k in g.files if k.startswith("vit_"))[0]
for stage in ["patch", "blk0", "blk1", "blk7", "blk15", "blk31", "norm"]:
    ref = g[f"vit_{start}_{stage}"]; ref32 = g[f"vit32_{start}_{stage}"]
    path = os.path.join(dump, f"vit_{stage}.bf16")
    if not os.path.exists(path): continue
    eng = bf16_file(path, ref.shape)
    print(f"{stage:6s} engine↔bf16ref {stats(eng, ref)}")
    print(f"{'':6s} engine↔fp32ref {stats(eng, ref32)}")
    print(f"{'':6s} bf16ref↔fp32ref {stats(ref, ref32)}")
