#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HIVE_LAYER_YIELD GPU correctness and latency probe — drives a live hived socket with raw token ids (no template, no server)
and writes the result as JSON. Restart hived for each setting (yield off / on / off again = noise floor), run the same command,
then compare tokens with --compare.

One run = decoder D (started first, receiving tokens) → long prompt L (--long rows, at most the service chunk cap of 98,304 so
it is a single forward) → after sending L, a short request S_i (--short rows) every --short-at seconds. All use temperature 0 ·
stop_ids [-1] (never stops at EOS) · prompts = random token ids from a fixed seed (identical across settings).
Recorded: tokens per stream · first-token time (relative to send) · prefill_ms from done · D's token gaps (overall · max over
the interval from sending L to L's first token).
--compare A B [C …]: whether each stream's tokens match · first divergence position. Judge A (off) ↔ B (on) side by side with
  the A (off) ↔ C (off again) noise floor (off ↔ off can diverge too because the CPU/GPU split depends on timing —
  measured: even sequential decode of the same input twice differs by rel 0.039).
Usage (in the container, via scripts/hive-run.sh): python3 /hive/tools/layer_yield_check.py --sock /out/hive.sock --label off --out /out/ly-off.json
                                     python3 /hive/tools/layer_yield_check.py --compare /out/ly-off.json /out/ly-on.json /out/ly-off2.json
"""
import argparse
import json
import random
import socket
import threading
import time


def ids_of(n, seed, lo=1000, hi=120000):
    r = random.Random(seed)
    return [r.randrange(lo, hi) for _ in range(n)]


class Gen(threading.Thread):
    def __init__(self, sock, session, ids, n):
        super().__init__(daemon=True)
        self.sock, self.session, self.ids, self.n = sock, session, ids, n
        self.tokens, self.times, self.done, self.error, self.t_send = [], [], None, None, None
        self.stop_flag = False

    def run(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.sock)
        self.t_send = time.monotonic()
        s.sendall((json.dumps({"op": "generate", "session": self.session, "ids": self.ids, "max_tokens": self.n, "temperature": 0.0,
                               "stop_ids": [-1]}) + "\n").encode())
        buf = b""
        try:
            while not self.stop_flag:
                c = s.recv(65536)
                if not c:
                    break
                buf += c
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    j = json.loads(line)
                    if "id" in j:
                        self.tokens.append(j["id"]); self.times.append(time.monotonic())
                    elif j.get("done"):
                        self.done = j; return
                    elif "error" in j:
                        self.error = j["error"]; return
        finally:
            s.close()


def cancel(sock, session):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock)
    s.sendall((json.dumps({"op": "cancel", "session": session}) + "\n").encode())
    s.recv(4096)
    s.close()


def run_once(a):
    tag = a.label
    D = Gen(a.sock, "lyD", ids_of(a.decoder_len, a.seed + 1), a.decoder_max) if a.decoder else None
    if D:
        D.start()
        while not D.tokens and D.is_alive():
            time.sleep(.01)
        time.sleep(a.decoder_warm)
    L = Gen(a.sock, "lyL", ids_of(a.long, a.seed + 2), a.n)
    L.start()
    shorts = []
    t0 = time.monotonic()
    for k, at in enumerate(float(x) for x in a.short_at.split(",") if x):
        while time.monotonic() - t0 < at:
            time.sleep(.005)
        g = Gen(a.sock, f"lyS{k}", ids_of(a.short, a.seed + 10 + k), a.n)
        g.start()
        shorts.append(g)
    L.join()
    for g in shorts:
        g.join()
    if D:
        time.sleep(a.decoder_tail)
        cancel(a.sock, "lyD")
        D.join(60)
    out = {"label": tag, "args": vars(a), "streams": {}}

    def rec(name, g):
        out["streams"][name] = {"tokens": g.tokens, "error": g.error, "ttft_s": (g.times[0] - g.t_send) if g.times else None,
                                "prefill_ms": g.done.get("prefill_ms") if g.done else None, "n": len(g.tokens)}
    rec("L", L)
    for k, g in enumerate(shorts):
        rec(f"S{k}", g)
    if D:
        rec("D", D)
        lo, hi = L.t_send, (L.times[0] if L.times else time.monotonic())
        w = [t for t in D.times if lo <= t <= hi]
        gaps = [b - x for x, b in zip([lo] + w, w + [hi])]
        out["decoder_during_long"] = {"tokens": len(w), "max_gap_s": max(gaps) if gaps else None, "window_s": hi - lo}
        # gaps are daemon token gaps (raw socket), not server chunks
    return out


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    base = runs[0]
    print(f"{'stream':8} " + " ".join(f"{r['label']:>18}" for r in runs[1:]))
    for name, s0 in base["streams"].items():
        cells = []
        for r in runs[1:]:
            s1 = r["streams"].get(name)
            if not s1:
                cells.append("missing"); continue
            a, b = s0["tokens"], s1["tokens"]
            n = min(len(a), len(b)) if name == "D" else max(len(a), len(b))  # D is cancelled at different times so lengths differ — compare the common prefix only
            div = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
            same = div is None and (name == "D" or len(a) == len(b))
            cells.append("same" if same else f"diverge@{div if div is not None else min(len(a), len(b))}/{n}")
        print(f"{name:8} " + " ".join(f"{c:>18}" for c in cells))
    print("timing:")
    for r in runs:
        st = r["streams"]
        sh = [v["ttft_s"] for k, v in st.items() if k.startswith("S") and v["ttft_s"] is not None]
        d = r.get("decoder_during_long") or {}
        print(f"  {r['label']:>10}: L ttft {st['L']['ttft_s']:.2f}s prefill {st['L']['prefill_ms']} ms · short ttft max {max(sh) if sh else None} · "
              f"decoder max gap during long {d.get('max_gap_s')} s ({d.get('tokens')} tokens)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/out/hive.sock")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="")
    ap.add_argument("--long", type=int, default=85000)
    ap.add_argument("--short", type=int, default=64)
    ap.add_argument("--short-at", default="1,3,5,8")
    ap.add_argument("--n", type=int, default=48, help="generated tokens of L·S (greedy)")
    ap.add_argument("--decoder", type=int, default=1)
    ap.add_argument("--decoder-len", type=int, default=2000)
    ap.add_argument("--decoder-max", type=int, default=4000)
    ap.add_argument("--decoder-warm", type=float, default=2.0)
    ap.add_argument("--decoder-tail", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare)
        return
    out = run_once(a)
    s = json.dumps(out)
    if a.out:
        open(a.out, "w").write(s)
    print(json.dumps({k: v for k, v in out.items() if k != "args"}, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
