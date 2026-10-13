#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""PLE n-gram hash test vectors for the engine's CPU test (hive/qwen/ple.h PleHash::rows_of).

Hash (per token x_t with predecessors x_{t−1}, x_{t−2}; EOS before the sequence start):
  b = x_{t−1}; c = EOS if x_{t−1} == EOS else x_{t−2}
  mix2 = x_t·m0 ⊕ b·m1 ; mix3 = mix2 ⊕ c·m2   (non-negative int64, ⊕ = xor)
  head j < 8 : row = mix2 mod size[j] + offset[j] ; head 8 ≤ j < 16 : row = mix3 mod size[j] + offset[j]
No int64 overflow: token ids < 2^18 and every multiplier m < (2^63 − 1) // vocab (how the checkpoint's multipliers are built, checked
here), so each product < 2^63; xor of non-negative values below 2^63 stays below 2^63. Python ints are exact; the check makes the claim
explicit for the given constants.

  python3 oracle/qwen4_ple_vectors.py --ckpt /models/nvidia-Qwen3.8-Flash-Next-NVFP4 --out ple_vectors.json
  python3 oracle/qwen4_ple_vectors.py --selftest [--out selftest.json]     (made-up constants, no checkpoint)
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

EOS = 248044
VOCAB = 248320
INT64_MAX = (1 << 63) - 1
# fixed token list: start of sequence, repeats, EOS in the middle (resets the context), the largest ids
TOKENS = [248045, 872, 198, 872, 198, 248044, 1, 248044, 248044, 0, 248319, 151643, 248046, 198, 3837, 99, 248044, 5, 6, 7]


def rows_of(tokens, mult, size, offset, eos=EOS, heads_per=8, prev2=-1, prev1=-1):
    p2 = eos if prev2 < 0 else prev2
    p1 = eos if prev1 < 0 else prev1
    out = []
    for x in tokens:
        b = p1
        c = eos if p1 == eos else p2
        mix2 = (x * mult[0]) ^ (b * mult[1])
        mix3 = mix2 ^ (c * mult[2])
        for v in (x * mult[0], b * mult[1], c * mult[2], mix2, mix3):
            assert 0 <= v <= INT64_MAX, "int64 overflow"
        r = [mix2 % size[j] + offset[j] for j in range(heads_per)] + [mix3 % size[j] + offset[j] for j in range(heads_per, 2 * heads_per)]
        out.append(r)
        p2, p1 = p1, x
    return out


def read_small(ckpt: str, names: list[str]) -> dict:
    """Read a few small tensors straight from the safetensors files (header + bytes), any shard."""
    want = set(names)
    got = {}
    for fn in sorted(os.listdir(ckpt)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(ckpt, fn)
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(n))
            for k in want & set(hdr):
                m = hdr[k]
                a, b = m["data_offsets"]
                f.seek(8 + n + a)
                raw = f.read(b - a)
                if m["dtype"] == "I64":
                    got[k] = np.frombuffer(raw, dtype=np.int64).tolist()
                elif m["dtype"] == "BF16":
                    got[k] = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).tolist()
                else:
                    raise ValueError(f"{k}: unexpected dtype {m['dtype']}")
        if len(got) == len(want):
            break
    missing = want - set(got)
    if missing:
        raise SystemExit(f"tensors not found: {sorted(missing)}")
    return got


def main():
    ap = argparse.ArgumentParser(description="PLE hash test vectors (hive/qwen/ple.h).")
    ap.add_argument("--ckpt")
    ap.add_argument("--layer", type=int, default=1, help="decoder layer holding the PLE (config ple_layer_ids − 1)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        # made-up constants (odd multipliers below the int64 bound, primes near 20M) — the engine CPU test carries the same numbers
        mult = [1000003, 2000000011 * 2 + 1, 999999937 * 4 + 1]
        size = [20000003, 20000023, 20000033, 20000047, 20000059, 20000101, 20000107, 20000141,
                20000143, 20000149, 20000177, 20000213, 20000231, 20000243, 20000249, 20000251]
        offset, acc = [], 0
        for s in size:
            offset.append(acc)
            acc += s
        scale = 1.0
    else:
        if not a.ckpt:
            raise SystemExit("--ckpt or --selftest")
        p = f"model.language_model.layers.{a.layer}.ple.ple_embedding."
        g = read_small(a.ckpt, [p + "layer_multipliers", p + "ngram_heads_vocab_sizes", p + "ngram_heads_offsets", p + "ngram_embedding.weight_scale"])
        mult, size, offset = g[p + "layer_multipliers"], g[p + "ngram_heads_vocab_sizes"], g[p + "ngram_heads_offsets"]
        scale = g[p + "ngram_embedding.weight_scale"][0]
        for m in mult:
            assert 0 < m <= INT64_MAX // VOCAB, f"multiplier {m} could overflow int64"
    rows = rows_of(TOKENS, mult, size, offset)
    doc = {"eos": EOS, "mult": mult, "size": size, "offset": offset, "weight_scale": scale, "tokens": TOKENS, "rows": rows,
           "table_rows": offset[-1] + size[-1]}
    txt = json.dumps(doc, indent=1)
    if a.out:
        open(a.out, "w").write(txt)
        print(f"wrote {a.out}: {len(TOKENS)} tokens × 16 rows", file=sys.stderr)
    else:
        print(txt)


if __name__ == "__main__":
    main()
