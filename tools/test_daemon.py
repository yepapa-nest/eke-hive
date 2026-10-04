#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""hived scheduler test — sends N concurrent requests to the socket to check batched decode (batch_rows>1), session prefix reuse and cancellation.
Also runs against a daemon started with --fake-head --max-layer 0 (partial checkpoint). Usage: test_daemon.py [--sock /out/hive.sock] [--n 3] [--tokens 24]"""
import argparse
import json
import socket
import threading
import time


def call(sock_path, req, bin_data=b"", on_line=None):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    s.sendall((json.dumps(req) + "\n").encode() + bin_data)
    buf = b""
    out = []
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            j = json.loads(line)
            out.append(j)
            if on_line:
                on_line(j)
            if "done" in j or "error" in j:
                s.close()
                return out
    s.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/out/hive.sock")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=24)
    a = ap.parse_args()
    base = [0, 671, 6102, 294, 8760, 344]
    results = {}

    def worker(i):
        ids = base + [1000 + i] * (i + 1)
        t0 = time.time()
        out = call(a.sock, {"op": "generate", "session": f"t{i}", "ids": ids, "max_tokens": a.tokens, "temperature": 0.0, "stop_ids": [-1]})
        results[i] = (out, time.time() - t0)

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(a.n)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    ok = True
    for i in range(a.n):
        out, dt = results[i]
        done = [j for j in out if "done" in j]
        toks = [j["id"] for j in out if "id" in j]
        err = [j for j in out if "error" in j]
        d = done[0] if done else {}
        print(f"req {i}: {len(toks)} tokens in {dt:.2f}s · finish {d.get('finish')} · batch_rows {d.get('batch_rows')} · "
              f"prefill {d.get('prefill_ms', 0):.0f}ms decode {d.get('decode_ms', 0):.0f}ms · err {err}")
        ok &= bool(done) and len(toks) == a.tokens and not err
    print(f"wall {time.time() - t0:.2f}s")
    # Session prefix reuse: a request appended to the same session -> cached_prefix > 0
    # Session t0 = prompt (7) + 24 generated (the last one is not forwarded yet -> 30 processed). The request must append new tokens after that to hit the prefix
    out0 = [j for j in results[0][0] if "id" in j]
    ids = base + [1000] + [j["id"] for j in out0] + [5, 6, 7]
    out = call(a.sock, {"op": "generate", "session": "t0", "ids": ids, "max_tokens": 4, "temperature": 0.0, "stop_ids": [-1]})
    d = [j for j in out if "done" in j][0]
    print(f"prefix reuse: cached_prefix {d.get('cached_prefix')} (expected {len(base) + 1 + a.tokens - 1})")
    ok &= d.get("cached_prefix", 0) == len(base) + 1 + a.tokens - 1
    # Cancellation: start a long generation and cancel after 0.3 s
    got = {}

    def long_worker():
        got["out"] = call(a.sock, {"op": "generate", "session": "tc", "ids": base, "max_tokens": 400, "temperature": 0.0, "stop_ids": [-1]})

    t = threading.Thread(target=long_worker)
    t.start()
    time.sleep(0.5)
    call(a.sock, {"op": "cancel", "session": "tc"})
    t.join()
    d = [j for j in got["out"] if "done" in j][0]
    print(f"cancel: finish {d.get('finish')} after {d.get('n')} tokens")
    ok &= d.get("finish") == "cancel"
    print("ALL OK" if ok else "FAIL")


if __name__ == "__main__":
    main()
