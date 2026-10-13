#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Rebuild a Qwen3.8-27B NVFP4 checkpoint with some tensors re-encoded from a bf16 checkpoint (e.g. an abliterated variant), keeping the
quantized checkpoint's format split: a tensor that is FP8 (per-row scale) in the base is re-encoded as FP8 per row, a tensor that is NVFP4
(compressed-tensors: e2m1 pairs, e4m3 scale per 16 values, fp32 weight_global_scale) is re-encoded as NVFP4; input_global_scale stays.
Shapes do not change, so the base model.safetensors is copied and the re-encoded tensors are written over their byte ranges; every other
file is hard-linked.

FP8:   s = bf16(max|row| / 448); q = e4m3_rne(w / s)
NVFP4: g = 448·6 / max|W| (stored as weight_global_scale); per 16 values s = e4m3_rne(max|blk| / 6 · g · f) with f from a small set of
       factors (the one with the smallest squared error per block, f = 1 first), c = e2m1_rne(w · g / s) (ties to the even code).

usage: q27_reencode.py --base <quantized dir> --source <bf16 dir> --changed <json with a list "changed" of {"name": ...}> --out <dir>
"""
import argparse
import json
import os
import shutil
import struct

import numpy as np

FACTORS = (1.0, 0.95, 0.9, 0.85, 1.05)


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_tensor(path, hdr, base, name):
    t = hdr[name]
    a, b = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    if t["dtype"] == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(t["shape"])
    if t["dtype"] == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(t["shape"])
    raise ValueError(f"{name}: unsupported source dtype {t['dtype']}")


def to_bf16_bits(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_round(x):
    return (to_bf16_bits(x).astype(np.uint32) << 16).view(np.float32)


# e4m3 (fn): codes 0..126 are the finite non-negative values (0x7f = NaN)
def _e4m3_table():
    v = []
    for c in range(127):
        e, m = c >> 3, c & 7
        v.append(m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return np.array(v, dtype=np.float64)


E4M3 = _e4m3_table()
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def rne_codes(a, table):
    """|a| → code of the nearest table value (ties to the even code), saturating at the largest."""
    a = np.minimum(np.abs(a).astype(np.float64), table[-1])
    hi = np.clip(np.searchsorted(table, a, side="left"), 1, len(table) - 1)
    lo = hi - 1
    dlo, dhi = a - table[lo], table[hi] - a
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (hi % 2 == 0))
    return np.where(pick_hi, hi, lo)


def fp8_row(w):
    s = bf16_round(np.max(np.abs(w), axis=1, keepdims=True) / 448.0)
    s = np.where(s > 0, s, 1.0).astype(np.float32)
    x = w / s
    code = rne_codes(x, E4M3).astype(np.uint8) | (np.signbit(x).astype(np.uint8) << 7)
    return code, to_bf16_bits(s).reshape(w.shape[0], 1)


def nvfp4(w):
    N, K = w.shape
    amax = float(np.max(np.abs(w)))
    g = np.float32(448.0 * 6.0 / amax) if amax > 0 else np.float32(1.0)
    blk = w.reshape(N, K // 16, 16).astype(np.float32) * g
    bmax = np.max(np.abs(blk), axis=2)
    best_err = best_sc = best_c = None
    for f in FACTORS:
        sc = rne_codes(bmax / 6.0 * f, E4M3)
        sv = E4M3[sc][..., None]
        y = np.divide(blk, sv, out=np.zeros_like(blk, dtype=np.float64), where=sv > 0)
        c = rne_codes(y, E2M1)
        rec = np.sign(y) * E2M1[c] * sv
        err = np.sum((rec - blk) ** 2, axis=2)
        if best_err is None:
            best_err, best_sc, best_c = err, sc, np.where(np.signbit(blk), c | 8, c)
        else:
            m = err < best_err
            best_err = np.where(m, err, best_err)
            best_sc = np.where(m, sc, best_sc)
            best_c = np.where(m[..., None], np.where(np.signbit(blk), c | 8, c), best_c)
    codes = best_c.reshape(N, K).astype(np.uint8)
    packed = (codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)
    return packed.astype(np.uint8), best_sc.astype(np.uint8), np.array([g], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--changed", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = [c["name"] for c in json.load(open(a.changed))["changed"]]
    os.makedirs(a.out, exist_ok=True)
    for fn in os.listdir(a.base):
        src, dst = os.path.join(a.base, fn), os.path.join(a.out, fn)
        if os.path.isfile(src) and fn != "model.safetensors" and not os.path.exists(dst):
            os.link(src, dst)
    bpath, opath = os.path.join(a.base, "model.safetensors"), os.path.join(a.out, "model.safetensors")
    if not os.path.exists(opath):
        shutil.copyfile(bpath, opath)
    bh, bbase = header(bpath)
    sidx = json.load(open(os.path.join(a.source, "model.safetensors.index.json")))["weight_map"]
    shdr = {}
    counts = {"fp8": 0, "nvfp4": 0}
    with open(opath, "r+b") as out:
        def put(name, arr):
            t = bh[name]
            lo, hi = t["data_offsets"]
            raw = np.ascontiguousarray(arr).tobytes()
            assert len(raw) == hi - lo, f"{name}: {len(raw)} bytes, slot {hi - lo}"
            out.seek(bbase + lo)
            out.write(raw)
        for name in names:
            stem = name[: -len(".weight")]
            shard = os.path.join(a.source, sidx[name])
            if shard not in shdr:
                shdr[shard] = header(shard)
            w = read_tensor(shard, shdr[shard][0], shdr[shard][1], name)
            if stem + ".weight_packed" in bh:
                p, s, g = nvfp4(w)
                put(stem + ".weight_packed", p); put(stem + ".weight_scale", s); put(stem + ".weight_global_scale", g)
                counts["nvfp4"] += 1
            elif bh.get(name, {}).get("dtype") == "F8_E4M3":
                q, s = fp8_row(w)
                put(name, q); put(stem + ".weight_scale", s)
                counts["fp8"] += 1
            else:
                raise ValueError(f"{name}: not FP8 / NVFP4 in the base checkpoint")
            print(f"  {name}: {'nvfp4' if stem + '.weight_packed' in bh else 'fp8'}", flush=True)
    print(f"re-encoded {counts} tensors into {opath}")


if __name__ == "__main__":
    main()
