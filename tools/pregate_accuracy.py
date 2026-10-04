#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Pre-gate accuracy: predicts the routing of layer l **before** attention (layer input h + hc_attn pre coefficients -> ffn_norm
-> the real router) and compares it with the actual routing.

Input = an engine layer dump directory (HIVE_DUMP — s*_h_L{l-1}.bf16, s*_route_ids_L{l}.i32, s*_engram_L{l}.bf16 for engram layers) + the
checkpoint (only gate, hc and norm weights are read).
CPU, single thread (OMP/OPENBLAS 1 recommended, nice). Result (17 tokens x layers 1-28): recall top-6 0.800, top-8 0.874, top-10 0.903,
top-12 0.925.
Usage: HIVE_CKPT=<checkpoint dir> python3 tools/pregate_accuracy.py <dump-dir>
"""
import json, struct, os, sys, numpy as np
CK = os.environ.get("HIVE_CKPT")
if not CK or len(sys.argv) < 2:
    raise SystemExit("usage: HIVE_CKPT=<checkpoint dir> pregate_accuracy.py <dump-dir>  (HIVE_CKPT is required)")
DUMP = sys.argv[1]  # e.g. <out-dir>/dump/l28
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
_hdr = {}
def _bf16_to_f32(u16): return (u16.astype(np.uint32) << 16).view(np.float32)
def tensor(name):
    fn = f"{CK}/{idx[name]}"
    if fn not in _hdr:
        with open(fn, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8)); _hdr[fn] = (json.loads(f.read(n)), 8 + n)
    h, base = _hdr[fn]; m = h[name]; a, b = m["data_offsets"]
    with open(fn, "rb") as f:
        f.seek(base + a); raw = f.read(b - a)
    dt = m["dtype"]
    if dt == "BF16": arr = _bf16_to_f32(np.frombuffer(raw, np.uint16))
    elif dt == "F32": arr = np.frombuffer(raw, np.float32)
    else: raise SystemExit(f"dtype {dt} {name}")
    return arr.reshape(m["shape"]).astype(np.float32)
def bf16file(p, shape): return _bf16_to_f32(np.fromfile(p, np.uint16)).reshape(shape)
HC, D, E, K, EPS = 4, 5120, 384, 6, 1e-20
steps = sorted({f.split("_")[0] for f in os.listdir(DUMP) if f.startswith("s")})
layers = sorted({int(f.split("_L")[1].split(".")[0]) for f in os.listdir(DUMP) if "_route_ids_L" in f})
res = {k: [0, 0] for k in ("top6", "top8", "top10", "top12")}
per_layer = {}
for l in layers:
    if l == 0: continue
    W = tensor(f"layers.{l}.ffn.gate.weight"); bias = tensor(f"layers.{l}.ffn.gate.bias")
    fn = tensor(f"layers.{l}.hc_attn_fn"); sc = tensor(f"layers.{l}.hc_attn_scale"); bs = tensor(f"layers.{l}.hc_attn_base")
    nw = tensor(f"layers.{l}.ffn_norm.weight")
    hit6 = tot6 = 0
    for s in steps:
        pid = f"{DUMP}/{s}_route_ids_L{l}.i32"; ph = f"{DUMP}/{s}_h_L{l-1}.bf16"
        if not (os.path.exists(pid) and os.path.exists(ph)): continue
        ids = np.fromfile(pid, np.int32).reshape(-1, K); M = ids.shape[0]
        pe = f"{DUMP}/{s}_engram_L{l}.bf16"  # for engram layers the layer input is the state after engram
        h = bf16file(pe if os.path.exists(pe) else ph, (M, HC, D))
        x = h.reshape(M, HC * D)
        mixes = (x @ fn.T) / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
        pre = 1 / (1 + np.exp(-(mixes[:, :HC] * sc[0] + bs[:HC]))) + 1e-6
        y = (pre[:, :, None] * h).sum(1)
        y = y / np.sqrt((y * y).mean(-1, keepdims=True) + EPS) * nw
        z = y @ W.T
        score = np.sqrt(np.logaddexp(0, z)) + bias
        order = np.argsort(-score, 1)
        for m in range(M):
            act = set(ids[m].tolist())
            for kk, key in ((6, "top6"), (8, "top8"), (10, "top10"), (12, "top12")):
                res[key][0] += len(act & set(order[m, :kk].tolist())); res[key][1] += K
            hit6 += len(act & set(order[m, :6].tolist())); tot6 += K
    if tot6: per_layer[l] = round(hit6 / tot6, 3)
print(json.dumps({"dump": DUMP, "steps": steps, "recall": {k: round(a / b, 4) for k, (a, b) in res.items() if b}, "n_pairs": res["top6"][1] // K, "per_layer_top6": per_layer}))
