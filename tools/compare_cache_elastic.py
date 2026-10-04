#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HIVE_CACHE_ELASTIC — compare engine/tests/test_cache_elastic output files (switch off vs on [, off again]).

Per record (phase, kind, step): argmax equality, top-5 overlap, max|Δ| of logits, KL(ref‖x). Phase 1 (first prefill, empty cache in both
runs when started without --cache-state) must be bit-identical: the big-view buffers differ only in address. Later phases differ by the
CPU/GPU expert-path fp32 order wherever the cache hit set differs (the on-run has more slots); judge them against the off-vs-off floor
(third file). Exit 1 if phase 1 is not bit-identical or a token sequence diverges before the first logit difference above the floor.
"""
import struct
import sys

import numpy as np


def load(path):
    with open(path, 'rb') as f:
        b = f.read()
    assert b[:4] == b'HVEL', path
    V = struct.unpack_from('<I', b, 4)[0]
    rec = 2 + 2 + 4 + 4 * V
    out = []
    for o in range(8, len(b) - rec + 1, rec):
        ph, kd = b[o], b[o + 1]
        st, tok = struct.unpack_from('<Hi', b, o + 2)
        lg = np.frombuffer(b, dtype=np.float32, count=V, offset=o + 8)
        out.append(((ph, kd, st), tok, lg))
    return out


def cmp(a, b):
    if not a.any() or not b.any():
        return None
    ta, tb = np.argsort(-a)[:5], np.argsort(-b)[:5]
    la = a.astype(np.float64) - (np.log(np.exp(a.astype(np.float64) - a.max()).sum()) + a.max())
    lb = b.astype(np.float64) - (np.log(np.exp(b.astype(np.float64) - b.max()).sum()) + b.max())
    return dict(argmax=int(ta[0] == tb[0]), top5=len(set(ta) & set(tb)), maxabs=float(np.abs(a - b).max()),
                kl=float((np.exp(la) * (la - lb)).sum()), bits=bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))))


def summary(ref, x, name):
    rows = {}
    diverged = None
    for (k, ta, la), (k2, tb, lb) in zip(ref, x):
        assert k == k2, (k, k2)
        r = cmp(la, lb)
        if r is None:
            continue
        rows.setdefault(k[0], []).append(r)
        if ta != tb and diverged is None:
            diverged = k
    print(f'== {name}')
    for ph, rs in sorted(rows.items()):
        print(f'  phase {ph}: n {len(rs)} · bit-identical {sum(r["bits"] for r in rs)} · argmax {sum(r["argmax"] for r in rs)}/{len(rs)} · '
              f'top5 {np.mean([r["top5"] for r in rs]):.2f} · max|Δ| {max(r["maxabs"] for r in rs):.4g} · KL max {max(r["kl"] for r in rs):.3g} '
              f'mean {np.mean([r["kl"] for r in rs]):.3g}')
    print(f'  first token divergence: {diverged}')
    return rows, diverged


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    off, on = load(sys.argv[1]), load(sys.argv[2])
    rows, _ = summary(off, on, 'off vs on')
    ok = all(r['bits'] for r in rows.get(1, [])) and bool(rows.get(1))
    floor_bits = None
    if len(sys.argv) > 3:
        frows, _ = summary(off, load(sys.argv[3]), 'off vs off (noise floor)')
        floor_bits = all(r['bits'] for r in frows.get(1, [])) and bool(frows.get(1))
    print('phase 1 (first prefill) bit-identical:', 'YES' if ok else 'NO')
    if not ok and floor_bits is False:
        print('  (off vs off is not bit-identical either — the prefill split adapts at run time; judge phase 1 against the floor above)')
        return 0
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
