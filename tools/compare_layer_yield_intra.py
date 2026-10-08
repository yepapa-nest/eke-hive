#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HIVE_LAYER_YIELD_INTRA — compare engine/tests/test_layer_yield_intra outputs: OFF ON [OFF2] (prefixes given to --out).

Per run: <p>.bin (L's logits: last prefill row + every decode step, raw f32), <p>.img (L's SeqImage bytes after the prefill), <p>.json (tokens,
yield counters). Reported: L logits bit equality per record · image sha256 · L tokens · decoder streams on their common prefix · yield counts.
Verdict: PASS when L's logits and image are bit-identical OFF vs ON. When they differ, the OFF vs OFF2 floor decides: if OFF2 also differs from OFF
the run is not deterministic (fix the shares/promotion settings — see the test's header) → INCONCLUSIVE (exit 2); otherwise FAIL (exit 1).
"""
import hashlib
import json
import struct
import sys


def load(prefix):
    with open(prefix + '.bin', 'rb') as f:
        b = f.read()
    assert b[:4] == b'HVLY', prefix
    V = struct.unpack_from('<I', b, 4)[0]
    rec = 4 + 4 * V
    recs = [(struct.unpack_from('<i', b, o)[0], b[o + 4:o + rec]) for o in range(8, len(b) - rec + 1, rec)]
    with open(prefix + '.img', 'rb') as f:
        img = hashlib.sha256(f.read()).hexdigest()
    with open(prefix + '.json') as f:
        js = json.load(f)
    return dict(recs=recs, img=img, js=js)


def diff(a, b):
    out = {}
    out['logits_records'] = len(a['recs'])
    bad = [i for i, (x, y) in enumerate(zip(a['recs'], b['recs'])) if x[1] != y[1]]
    out['logits_identical'] = not bad and len(a['recs']) == len(b['recs'])
    out['first_logit_diff'] = bad[0] if bad else None
    out['image_identical'] = a['img'] == b['img']
    out['L_tokens_identical'] = a['js']['L'] == b['js']['L']
    dec = []
    for x, y in zip(a['js']['decoders'], b['js']['decoders']):
        n = min(len(x), len(y))
        div = next((i for i in range(n) if x[i] != y[i]), None)
        dec.append('same' if div is None else f'diverge@{div}/{n}')
    out['decoders'] = dec
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    runs = [load(p) for p in sys.argv[1:]]
    for p, r in zip(sys.argv[1:], runs):
        j = r['js']
        print(f"{p}: {j['label']} · prefill {j['prefill_ms']:.0f} ms · yields {j['yields']} (intra {j['intra_yields']}) · inner {j['yield_inner_ms']:.0f} ms · "
              f"max gap {j['max_gap_ms']:.0f} ms · decoder steps in yields {j['steps_in_yields']} · image sha256 {r['img'][:16]}…")
    on = diff(runs[0], runs[1])
    print('OFF vs ON :', json.dumps(on))
    floor = diff(runs[0], runs[2]) if len(runs) > 2 else None
    if floor:
        print('OFF vs OFF2:', json.dumps(floor))
    if on['logits_identical'] and on['image_identical']:
        print('PASS: L logits and image bit-identical with intra-layer yields')
        sys.exit(0)
    if floor and not (floor['logits_identical'] and floor['image_identical']):
        print('INCONCLUSIVE: OFF vs OFF2 differs too (run is not deterministic — fixed DMA shares, --promote 0)')
        sys.exit(2)
    print('FAIL: intra-layer yields changed the paused prefill')
    sys.exit(1)


if __name__ == '__main__':
    main()
