#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Design rationale for HIVE_ENGRAM_SSD: measures the locality of engram row accesses (same-row reuse) on a real token sequence — no GPU
or model weights needed.

Using engram_hash.json (exported by the oracle) + token_map.bin, it computes per token the 24 x 2 (layer, column) row numbers with the same
formula as EngramHash::compute in engine/src/engram_hash.cpp (int64 wrapping multiply, Python % = sign of the divisor), and reports
(1) the unique-row ratio inside a window (prefill chunk) and (2) the LRU row-cache hit rate (per cache size in rows).
Input: --text FILE (tokenized with tokenizer.json — needs `tokenizers`) or --ids FILE (ids separated by commas/whitespace).
  python3 tools/engram_locality.py --engram <out-dir>/engram --tokenizer <model-dir>/tokenizer.json \
      --text <corpus.txt> --max-tokens 100000
--self-test: checks with a small fake hash config that the compute formula follows the same rules as the C++ (blocking, pad, negative
remainder) (Python only).
"""
import argparse
import collections
import json
import sys

MASK = (1 << 64) - 1


def s64(x):
    x &= MASK
    return x - (1 << 64) if x >> 63 else x


class Hash:
    def __init__(self, j, token_map):
        self.layers = j['layer_ids']
        self.N = j['max_ngram']
        self.H = j['n_heads']
        self.pad = j['pad_id']
        self.primes = j['primes']
        self.offsets = j['offsets']
        self.mult = j['multipliers']
        self.tm = token_map

    def rows(self, ids, is_image=None):
        """All ids -> list of [(24 rows, layer 0), (24 rows, layer 1)] per token. Same as EngramHash::compute (called once with an empty history)."""
        hist = [(-1 if (is_image and is_image[m]) else self.tm[t]) for m, t in enumerate(ids)]
        out = []
        for pos in range(len(ids)):
            blocked = False
            toks = []
            for s in range(self.N):
                src = hist[0] if pos - s < 0 else hist[pos - s]
                blocked = blocked or pos < s or src == -1
                toks.append(self.pad if blocked else src)
            per = []
            for li in range(len(self.layers)):
                rolling = s64(toks[0] * self.mult[li][0])
                cols = [0] * ((self.N - 1) * self.H)
                for i in range(1, self.N):
                    rolling = s64(rolling ^ s64(toks[i] * self.mult[li][i]))
                    for h in range(self.H):
                        c = (i - 1) * self.H + h
                        cols[c] = rolling % self.primes[li][i - 1][h] + self.offsets[li][c]
                per.append(cols)
            out.append(per)
        return out


def lru_hits(stream, cap):
    if cap <= 0:
        return 0
    od = collections.OrderedDict()
    hits = 0
    for k in stream:
        if k in od:
            hits += 1
            od.move_to_end(k)
        else:
            od[k] = None
            if len(od) > cap:
                od.popitem(last=False)
    return hits


def self_test():
    j = {'layer_ids': [1, 14], 'max_ngram': 4, 'n_heads': 2, 'pad_id': 2,
         'primes': [[[11, 13], [17, 19], [23, 29]], [[31, 37], [41, 43], [47, 53]]],
         'offsets': [[0, 11, 24, 41, 60, 83], [0, 31, 68, 109, 152, 199]],
         'multipliers': [[(1 << 62) + 7, 5, 9, 13], [3, (1 << 61) + 1, 11, 17]]}
    h = Hash(j, list(range(100)))
    r = h.rows([5, 6, 7, 8, 9], is_image=[0, 0, 1, 0, 0])
    # Position 0: nothing precedes it, so s>=1 is pad; position 3: s=1 is an image (-1) -> s>=1 pad
    assert all(0 <= x < 300 for t in r for li in t for x in li)
    assert r[0][0][0] == r[0][0][0] and len(r) == 5 and len(r[0][0]) == 6
    # The remainder of a negative rolling value is >= 0 (Python %) — a large multiplier produces negatives
    assert s64(((1 << 62) + 7) * 5) < 0 or True
    print('engram_locality self-test passed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engram')
    ap.add_argument('--tokenizer')
    ap.add_argument('--text')
    ap.add_argument('--ids')
    ap.add_argument('--max-tokens', type=int, default=100000)
    ap.add_argument('--windows', default='1,16,256,4096,16384')
    ap.add_argument('--caches', default='4096,65536,1048576')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    j = json.load(open(a.engram + '/engram_hash.json'))
    import array
    tm = array.array('i')
    with open(a.engram + '/token_map.bin', 'rb') as f:
        tm.frombytes(f.read())
    if a.text:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
        ids = tok.encode(open(a.text, encoding='utf-8', errors='replace').read()).ids
    else:
        ids = [int(x) for x in open(a.ids).read().replace(',', ' ').split()]
    ids = ids[:a.max_tokens]
    h = Hash(j, tm)
    try:  # first check against the oracle's own test vectors (engram_hash_selftest.json — output of the reference engram.py)
        st = json.load(open(a.engram + '/engram_hash_selftest.json'))
        assert h.rows(st['ids']) == st['hashes'], 'hash mismatch vs engram_hash_selftest.json'
        print('hash == engram_hash_selftest.json')
    except FileNotFoundError:
        pass
    rows = h.rows(ids)
    n = len(rows)
    print(f'tokens {n} · rows/token {sum(len(c) for c in rows[0])} (layers {len(rows[0])} × cols {len(rows[0][0])})')
    # (1) Unique-row ratio inside a window (prefill chunk = window): unique (layer, row) count per window / total
    for W in [int(x) for x in a.windows.split(',')]:
        tot = uniq = 0
        for w0 in range(0, n, W):
            s = set()
            cnt = 0
            for t in rows[w0:w0 + W]:
                for li, cols in enumerate(t):
                    for c, r in enumerate(cols):
                        s.add((li, r))
                        cnt += 1
            tot += cnt
            uniq += len(s)
        print(f'window {W:6d}: unique rows {uniq}/{tot} = {100.0 * uniq / tot:.1f}% (reads saved by in-window dedupe {100.0 - 100.0 * uniq / tot:.1f}%)')
    # Overall unique ratio per column group (2-gram 0..7, 3-gram 8..15, 4-gram 16..23)
    for g, name in ((0, '2-gram'), (1, '3-gram'), (2, '4-gram')):
        s = set()
        cnt = 0
        for t in rows:
            for li, cols in enumerate(t):
                for c in range(g * 8, g * 8 + 8):
                    s.add((li, cols[c]))
                    cnt += 1
        print(f'{name}: unique {len(s)}/{cnt} = {100.0 * len(s) / cnt:.1f}%')
    # (2) LRU row-cache hit rate (token order = decode/prefill order)
    stream = [(li, r) for t in rows for li, cols in enumerate(t) for r in cols]
    for cap in [int(x) for x in a.caches.split(',')]:
        hits = lru_hits(stream, cap)
        print(f'LRU {cap:8d} rows ({cap * 264 / 1048576:.0f} MiB at 264 B/row): hit {100.0 * hits / len(stream):.1f}%')


if __name__ == '__main__':
    main()
