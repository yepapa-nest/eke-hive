#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Portions ported from the DeepSeek-V4.1-Flash reference implementation
# (inference/model.py, inference/kernel.py), Copyright (c) 2023 DeepSeek, MIT License.
"""Export the engram hash constants — writes token_map.bin + engram_hash.json so the engine (C++) produces the same hashes without the tokenizer normalizer.
Usage: python3 export_engram_hash.py --ckpt /path/to/DeepSeek-V4.1-Flash --out /path/to/out/engram"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import dsv41_oracle as O  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.environ.get("HIVE_CKPT"), help="checkpoint directory (default: $HIVE_CKPT)")
    parser.add_argument("--out", required=True)
    a = parser.parse_args()
    if not a.ckpt:
        parser.error("--ckpt DIR (or HIVE_CKPT) is required")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(a.ckpt, "tokenizer.json"))
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))["text_config"]
    eh = O.EngramHash(cfg, tok, 64)
    os.makedirs(a.out, exist_ok=True)
    np.asarray(eh.token_map.numpy(), dtype=np.int32).tofile(os.path.join(a.out, "token_map.bin"))
    j = {
        "layer_ids": eh.layer_ids,
        "max_ngram": eh.max_ngram,
        "n_heads": cfg["engram_n_heads"],
        "pad_id": int(eh.pad_id),
        "primes": eh.primes.tolist(),
        "offsets": eh.offsets.tolist(),
        "multipliers": eh.multipliers.tolist(),
    }
    json.dump(j, open(os.path.join(a.out, "engram_hash.json"), "w"))
    # self-check: also store oracle hashes for short sequences (for engine unit tests)
    import torch

    ids = torch.tensor([0, 671, 6102, 294, 8760, 344, 13, 1052, 5])
    h = eh(ids, 0)
    json.dump({"ids": ids.tolist(), "hashes": h.tolist()}, open(os.path.join(a.out, "engram_hash_selftest.json"), "w"))
    print("wrote", a.out, "vocab", len(eh.token_map), "pad", int(eh.pad_id))


if __name__ == "__main__":
    main()
