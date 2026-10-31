#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Format check of NVIDIA's Qwen3.8-Flash-Next-FP4-Local checkpoint (no GPU): every compressed (CSF) expert block-scale tensor decoded by an
independent numpy implementation and checked against the invariants hive/ckpt_quant.h enforces (fixed shape, exception index range, zero
delta under an exception, every code a finite non-negative E4M3 byte), the per-row base = the row's smallest non-exception code, and the
MXFP8 / NVFP4 dense tensors' scale bytes. With --cpp BINARY (engine test_ckpt_quant_cpu) a sample of projections is decoded by the engine
code too and compared byte for byte.

    python3 tools/fp4l_check.py /models/nvidia-Qwen3.8-Flash-Next-FP4-Local [--cpp build/test_ckpt_quant_cpu] [--sample 64]
"""
import argparse, json, os, struct, subprocess, sys, tempfile
from collections import Counter, defaultdict

import numpy as np


class Ckpt:
    def __init__(self, d):
        self.d = d
        self.map = json.load(open(os.path.join(d, 'model.safetensors.index.json')))['weight_map']
        self.h = {}

    def hdr(self, shard):
        if shard not in self.h:
            f = open(os.path.join(self.d, shard), 'rb')
            n = struct.unpack('<Q', f.read(8))[0]
            self.h[shard] = (f, json.loads(f.read(n)), 8 + n)
        return self.h[shard]

    def get(self, k):
        f, h, base = self.hdr(self.map[k])
        v = h[k]
        a, b = v['data_offsets']
        f.seek(base + a)
        dt = {'U8': np.uint8, 'U32': np.uint32, 'F32': np.float32, 'F8_E4M3': np.uint8, 'BF16': np.uint16, 'I64': np.int64}[v['dtype']]
        return v['dtype'], np.frombuffer(f.read(b - a), dt).reshape(v['shape'])


def csf_decode(fixed, exc, R, C):
    """returns (codes [R, C] int64, nibbles [R, C]) — the format as described in hive/ckpt_quant.h"""
    G = R // 16
    base = fixed[:, :16].astype(np.int64)                      # [G, 16]
    body = fixed[:, 16:]
    nib = np.empty((G, body.shape[1] * 2), np.int64)
    nib[:, 0::2] = body & 15
    nib[:, 1::2] = body >> 4
    nib = nib.reshape(G, 16, C)
    codes = (base[:, :, None] + nib).reshape(R, C)
    flat = codes.reshape(-1)
    idx = (exc & 0xFFFFFF).astype(np.int64)
    flat[idx] = (exc >> 24).astype(np.int64)
    return codes, nib.reshape(R, C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--cpp', help='engine test_ckpt_quant_cpu binary (byte comparison of a sample)')
    ap.add_argument('--sample', type=int, default=64)
    a = ap.parse_args()
    ck = Ckpt(a.ckpt)
    bases = sorted({k[:-len('weight_scale.nvfp4_csf_fixed')] for k in ck.map if k.endswith('weight_scale.nvfp4_csf_fixed')})
    print(f'CSF projections: {len(bases)}')
    bad = Counter()
    n_exc = []
    rowmin_ok = rows = 0
    code_hist = np.zeros(256, np.int64)
    sample = set(bases[:: max(1, len(bases) // max(1, a.sample))][: a.sample])
    decoded = {}
    for i, b in enumerate(bases):
        _, w = ck.get(b + 'weight')
        _, fx = ck.get(b + 'weight_scale.nvfp4_csf_fixed')
        _, ex = ck.get(b + 'weight_scale.nvfp4_csf_exceptions')
        R, C = w.shape[0], w.shape[1] * 2 // 16
        if fx.shape != (R // 16, 16 + 8 * C) or R % 16:
            bad['fixed shape'] += 1
            continue
        idx = (ex & 0xFFFFFF).astype(np.int64)
        if len(idx) and idx.max() >= R * C:
            bad['index range'] += 1
            continue
        codes, nib = csf_decode(fx, ex, R, C)
        if len(idx) and nib.reshape(-1)[idx].max() != 0:
            bad['delta under exception'] += 1
        if len(idx) and (ex >> 24).max() > 0x7E:
            bad['exception value > 0x7E'] += 1
        if codes.max() > 0x7E or codes.min() < 0:
            bad['code outside 0..0x7E'] += 1
        if len(np.unique(idx)) != len(idx):
            bad['duplicate exception index'] += 1
        # base = smallest non-exception code of its row?
        m = np.ones(R * C, bool)
        m[idx] = False
        nm = np.where(m.reshape(R, C), nib, 99).min(axis=1)
        rowmin_ok += int((nm == 0).sum())
        rows += R
        n_exc.append(len(ex))
        code_hist += np.bincount(codes.reshape(-1).clip(0, 255), minlength=256)
        if b in sample:
            decoded[b] = codes.astype(np.uint8)
        if i % 10000 == 0:
            print(f'  {i}/{len(bases)}', flush=True)
    ne = np.array(n_exc)
    print(f'invariant violations: {dict(bad) if bad else "none"}')
    print(f'exceptions per projection: min {ne.min()} median {int(np.median(ne))} max {ne.max()} · total {ne.sum()}')
    print(f'rows whose smallest non-exception delta is 0: {rowmin_ok}/{rows}')
    nz = np.nonzero(code_hist)[0]
    print(f'scale codes used: {nz.min():#x}..{nz.max():#x}')
    # dense tensors
    st = defaultdict(lambda: [0, 255, 0, 0])   # count, min, max, NaN codes in the e4m3 weights
    for k, v in ck.map.items():
        if not k.endswith('.weight') or 'experts' in k or 'ngram_embedding' in k:
            continue
        b = k[:-len('weight')]
        if b + 'weight_scale' not in ck.map:
            continue
        dt, s = ck.get(b + 'weight_scale')
        wdt, w = ck.get(k)
        kind = 'MXFP8' if wdt == 'F8_E4M3' and dt == 'U8' else 'NVFP4' if wdt == 'U8' and dt == 'F8_E4M3' else f'{wdt}/{dt}'
        e = st[kind]
        e[0] += 1
        e[1] = min(e[1], int(s.min()))
        e[2] = max(e[2], int(s.max()))
        if wdt == 'F8_E4M3':
            e[3] += int(((w & 0x7F) == 0x7F).sum())
    for kind, (n, mn, mx, nan) in st.items():
        print(f'{kind}: {n} matrices · scale bytes {mn}..{mx} · NaN weight codes {nan}')
    if a.cpp and decoded:
        with tempfile.TemporaryDirectory() as d:
            names = sorted(decoded)
            subprocess.run([a.cpp, '--dump', a.ckpt, d] + names, check=True)
            same = sum(np.array_equal(np.fromfile(os.path.join(d, f'{i}.bin'), np.uint8), decoded[b].reshape(-1)) for i, b in enumerate(names))
            print(f'engine decoder == numpy decoder: {same}/{len(names)} projections byte-identical')
            if same != len(names):
                sys.exit(1)
    if bad:
        sys.exit(1)


if __name__ == '__main__':
    main()
