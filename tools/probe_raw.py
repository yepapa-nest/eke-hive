#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Raw token probe — feeds text to the daemon socket without a chat template and decodes the continuation (to separate engine
correctness from template/tokenizer issues).
Usage (in the container): probe_raw.py [--sock /out/hive.sock] [--n 12] ["prompt" ...]"""
import argparse
import json
import os
import socket

from tokenizers import Tokenizer


def gen(sock_path, ids, n, sid="probe"):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    s.sendall((json.dumps({"op": "generate", "session": sid, "ids": ids, "max_tokens": n, "temperature": 0.0, "stop_ids": [-1]}) + "\n").encode())
    buf, out = b"", []
    while True:
        c = s.recv(65536)
        if not c:
            break
        buf += c
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            j = json.loads(line)
            if "id" in j:
                out.append(j["id"])
            if "done" in j or "error" in j:
                s.close()
                return out, j
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/out/hive.sock")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ckpt", default=os.environ.get("HIVE_CKPT"), help="checkpoint directory (default: $HIVE_CKPT)")
    ap.add_argument("prompts", nargs="*", default=["The capital of France is", "1, 2, 3, 4, 5,", "def fibonacci(n):\n    "])
    a = ap.parse_args()
    if not a.ckpt:
        ap.error("--ckpt DIR (or HIVE_CKPT) is required")
    tok = Tokenizer.from_file(f"{a.ckpt}/tokenizer.json")
    for text in a.prompts:
        ids = [0] + tok.encode(text, add_special_tokens=False).ids
        out, d = gen(a.sock, ids, a.n)
        info = {k: d.get(k) for k in ("finish", "decode_hit", "decode_cpu", "decode_ms", "error")} if d else None
        print(f"{text!r} -> {tok.decode(out)!r} | ids {out[:8]} | {info}")


if __name__ == "__main__":
    main()
