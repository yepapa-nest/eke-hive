#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""CPU test — runs tools/gap_eval.py and sampling_compare.py against a fake hive server (no GPU or model needed).
Fake server: word-level tokenizer (usage.prompt_tokens) · streaming/non-streaming · stop · images (color of the first PNG pixel) · client-disconnect detection · /health decode counter ·
seeded sampling when temperature>0. Mode "bug" plants ignored image replacement (stale cache), ignored stop and a wrong answer at the 1024 boundary → checks that the comparators catch them.
Usage: nice -n 19 taskset -c 6-7 python3 tools/test_gap_eval_cpu.py"""
import base64
import json
import os
import random
import re
import struct
import sys
import tempfile
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import gap_eval as ge  # noqa: E402
import quality_compare as qc  # noqa: E402
import sampling_compare as sc  # noqa: E402

FAILS = []
STATE = {"mode": "good", "sampling": "uniform", "decode": 0, "img_cache": {}}
LOCK = threading.Lock()


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def toks(text):
    return len(re.findall(r"\w+|[^\w\s]", text))


def msg_text(m):
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if p.get("type") == "text")
    return c or ""


def png_color(url):
    data = base64.b64decode(url.split(",", 1)[1])
    i, idat = 8, b""
    while i < len(data):
        n = struct.unpack(">I", data[i:i + 4])[0]
        tag = data[i + 4:i + 8]
        if tag == b"IDAT":
            idat += data[i + 8:i + 8 + n]
        i += 12 + n
    raw = zlib.decompress(idat)
    r, g, b = raw[1], raw[2], raw[3]
    return "Red" if r > b else "Blue"


def images(m):
    c = m.get("content")
    return [p["image_url"]["url"] for p in c if p.get("type") == "image_url"] if isinstance(c, list) else []


def prompt_tokens(msgs):
    return 4 + sum(toks(msg_text(m)) + 100 * len(images(m)) for m in msgs)


SAMPLE_OPTS = {"rand10": [str(i) for i in range(1, 11)], "mul": ["391", "391", "391", "392"], "color": ["Red", "Blue", "Yellow"],
               "canberra": ["Canberra", "Canberra", "Sydney"], "seoul_ko": ["서울", "서울입니다"], "story": ["A cat slept.", "The cat ran far away today."],
               "prime97": ["Yes", "Yes", "No"], "rhyme": ["Night", "Bright", "Kite", "Fight"]}


def reply_for(body):
    """→ (text, is_count) — with is_count, slowly streams 1..600."""
    msgs = body.get("messages", [])
    last = msg_text(msgs[-1]) if msgs else ""
    mode = STATE["mode"]
    t = float(body.get("temperature", 0) or 0)
    if t > 0:
        for pid, p, _ in ge.SAMPLING_PROMPTS:
            if last == p:
                rng = random.Random(body.get("seed", 0))
                if pid == "rand10" and STATE["sampling"] == "skew":
                    return ("7" if rng.random() < 0.8 else rng.choice(SAMPLE_OPTS[pid])), False
                return rng.choice(SAMPLE_OPTS[pid]), False
    if "Count from 1 to 600" in last:
        return " ".join(str(i) for i in range(1, 601)), True
    m = re.search(r"secret code word is ([A-Z]+-\d+)", msg_text(msgs[0]))
    if m and "secret code word stated" in last:
        if mode == "bug" and prompt_tokens(msgs) == 1024:
            return "UNKNOWN", False
        return m.group(1), False
    if last.endswith(ge.TINY_Q):
        return "12", False
    if "harbour master's surname" in last:
        return "Sorensen", False
    if "Which city do I live in" in last:
        return re.findall(r"I live in (\w+)", " ".join(msg_text(x) for x in msgs))[-1], False
    if "pet's name" in last and "?" in last:
        return re.findall(r"My pet's name is (\w+)", " ".join(msg_text(x) for x in msgs))[-1], False
    if "secret password" in last:
        return re.search(r"password is (\w+)", msg_text(msgs[0])).group(1), False
    if "what a lighthouse is for" in last:
        return "A lighthouse guides ships safely at night.", False
    imgs = images(msgs[-1]) if msgs else []
    if imgs:
        col = png_color(imgs[-1])
        sid = body.get("hive_session_id")
        if mode == "bug" and sid:  # stale image cache: keeps returning the answer to the session's first image
            col = STATE["img_cache"].setdefault(sid, col)
        return col, False
    if "alpha beta END gamma delta" in last:
        return "alpha beta END gamma delta", False
    if "Say only the word" in last:
        return "yes", False
    m = re.search(r"(\d+)\s*\+\s*(\d+)", last)
    if m:
        s = int(m.group(1)) + int(m.group(2))
        if last.startswith("[load"):
            return f"{s}\n" + " ".join(["word"] * 100), False
        return str(s), False
    if "Reply 'Noted.'" in last or "Remember this" in last:
        return "Noted.", False
    return "?", False


def apply_stop(text, stop):
    if STATE["mode"] == "bug" or not stop:
        return text, None
    for s in stop:
        j = text.find(s)
        if j >= 0:
            return text[:j].rstrip(), s
    return text, None


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        for k, v in (("Content-Type", "application/json"), ("Content-Length", str(len(payload)))):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._json({"ok": True, "daemon": {"decode_routed": STATE["decode"], "pending_requests": 0}})

    def _chunk(self, s):
        b = s.encode()
        self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
        self.wfile.flush()

    def do_POST(self):
        n_body = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n_body))
        if self.path == "/v1/messages":
            ob = {"messages": body["messages"], "stop": body.get("stop_sequences")}
            text, _ = reply_for(ob)
            text, hit = apply_stop(text, ob["stop"])
            part = {"type": "text", "text": text}
            return self._json({"content": [part], "stop_reason": "stop_sequence" if hit else "end_turn", "stop_sequence": hit})
        msgs = body["messages"]
        if body.get("system"):
            msgs = [{"role": "system", "content": body["system"]}] + msgs
        pt = prompt_tokens(msgs)
        text, is_count = reply_for(body)
        text, hit = apply_stop(text, body.get("stop"))
        pieces = re.findall(r"\s*\S+", text)
        mt = body.get("max_tokens") or 10 ** 9
        finish = "length" if len(pieces) > mt else "stop"
        pieces = pieces[:mt]
        if not body.get("stream"):
            with LOCK:
                STATE["decode"] += len(pieces)
            return self._json({"choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(pieces)}, "finish_reason": finish,
                                            **({"stop_reason": hit} if hit else {})}],
                               "usage": {"prompt_tokens": pt, "completion_tokens": len(pieces)}, "hive": {"prefill_ms": pt * 0.01, "cached_prefix": 0}})
        self.send_response(200)
        for k, v in (("Content-Type", "text/event-stream"), ("Transfer-Encoding", "chunked")):
            self.send_header(k, v)
        self.end_headers()
        try:
            self._chunk("data: " + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n")
            for p in pieces:
                time.sleep(0.01 if is_count else 0.001)
                with LOCK:
                    STATE["decode"] += 1
                self._chunk("data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]}, ensure_ascii=False) + "\n\n")
            last = {"index": 0, "delta": {}, "finish_reason": finish, **({"stop_reason": hit} if hit else {})}
            self._chunk("data: " + json.dumps({"choices": [last], "usage": {"prompt_tokens": pt, "completion_tokens": len(pieces)}}) + "\n\n")
            self._chunk("data: [DONE]\n\n")
            self._chunk("")
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True


class Srv(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass


def main():
    print("== unit: statistics")
    check(abs(sc.chi2_sf(3.841459, 1) - 0.05) < 1e-4, "chi2_sf(3.84,1)=0.05")
    check(abs(sc.chi2_sf(18.307038, 10) - 0.05) < 1e-4, "chi2_sf(18.31,10)=0.05")
    check(sc.chi2_sf(100.0, 3) < 1e-15 and abs(sc.chi2_sf(2.0, 4) - 0.735759) < 1e-5, "chi2_sf tail / body")
    check(abs(sc.fisher_exact(1, 9, 11, 3) - 0.002759) < 1e-5, "fisher [[1,9],[11,3]] = 0.002759")
    check(sc.fisher_exact(5, 5, 5, 5) == 1.0, "fisher equal = 1")
    check(sc.perm_test_mean([1, 2, 3] * 10, [1, 2, 3] * 10) == 1.0 and sc.perm_test_mean([1] * 20, [5] * 20) < 0.001, "permutation test")
    x, df, p, _ = sc.chi2_homogeneity({"a": 30, "b": 30}, {"a": 30, "b": 30})
    check(x == 0.0 and df == 1 and p == 1.0, "chi2 homogeneity identical")
    check(ge.pct([1, 2, 3, 4], 50) == 2.5 and ge.pct([5], 99) == 5, "percentile")
    check(png_color("data:image/png;base64," + base64.b64encode(ge.png_solid(ge.RED)).decode()) == "Red", "png_solid decodes (red)")

    srv = Srv(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    d = tempfile.mkdtemp(prefix="gap-test-")
    good, bug = os.path.join(d, "good.json"), os.path.join(d, "bug.json")
    print("== fake server: good")
    STATE["mode"] = "good"
    rc = ge.main(["--base", base, "--out", good, "--suite", "boundary,session,load,sampling", "--n", "48", "--sampling-conc", "8",
                  "--host-metrics", "--label", "good", "--cpt", "5.0"])
    check(rc == 0, "good rc 0")
    G = json.load(open(good))
    gi = {r["id"]: r for r in G["items"]}
    bnd = [r for r in G["items"] if r["suite"] == "boundary"]
    check(len(bnd) == len(ge.BOUNDARIES) and all(r["exact"] and r["achieved"] == r["target"] for r in bnd),
          f"all {len(ge.BOUNDARIES)} boundaries hit exactly: {[(r['target'], r['achieved']) for r in bnd if not r['exact']]}")
    check(all(r["pass"] for r in bnd), "boundary answers pass")
    check(max(len(r["detail"]["attempts"]) for r in bnd) <= 6, f"fit converges (max attempts {max(len(r['detail']['attempts']) for r in bnd)})")
    check(all(r["ttft_s"] is not None for r in bnd), "boundary ttft recorded")
    ses = [r for r in G["items"] if r["suite"] == "session"]
    check(all(r["pass"] for r in ses), f"session all pass: {[r['id'] for r in ses if not r['pass']]}")
    check(len(ses) == 13, f"session item count {len(ses)}")
    load = G["load"]
    check(all(gi[f"load.c16.{i}"]["pass"] for i in range(16)) and gi["load.after_short"]["pass"], "c16 streams + after-short pass")
    check(load["cancelled_streams"] == [12, 13, 14, 15] and all(gi[f"load.c16.{i}"]["kind"] == "cancelled" for i in (12, 13, 14, 15)), "4 streams cancelled under load")
    check(all(load["chunk_gap_s"][k] is not None for k in ("p50", "p95", "p99")) and load["aggregate_tok_s"], "gap percentiles + throughput")
    q = load["cancel_c1"]["quiet_s"]
    check(len(q) == 3 and all(v is not None and v < 5 for v in q), f"cancel quiet measured {q}")
    hm = G["host_metrics"]
    check(hm["on"] and (hm["notes"] or hm["max_by_phase"]), f"host metrics degrade gracefully: notes {hm['notes']}")
    smp = G["sampling"]
    check(all(len(v) == 48 and all(v) for v in smp["samples"].values()), "sampling 8 × 48 collected")
    check(smp["summary"]["mul"]["accuracy"] is not None and smp["summary"]["rand10"]["accuracy"] is None, "sampling accuracy only for checkable prompts")
    print("== fake server: same distribution once more (= lossless MTP off/on assumption) · skewed distribution")
    same, skew = os.path.join(d, "same.json"), os.path.join(d, "skew.json")
    ge.main(["--base", base, "--out", same, "--suite", "sampling", "--n", "48", "--sampling-conc", "8", "--label", "same"])
    STATE["sampling"] = "skew"
    ge.main(["--base", base, "--out", skew, "--suite", "sampling", "--n", "48", "--sampling-conc", "8", "--label", "skew"])
    r_same = sc.compare(G, json.load(open(same)))
    r_skew = sc.compare(G, json.load(open(skew)))
    check(all(v["identical_by_seed"] == [48, 48] for v in r_same["prompts"].values()), "same server → identical by seed")
    check(all(v["first_word"]["p"] == 1.0 for v in r_same["prompts"].values()), "same → chi2 p = 1")
    check(r_skew["prompts"]["rand10"]["first_word"]["p"] < 1e-4, f"skewed rand10 detected p={r_skew['prompts']['rand10']['first_word']['p']}")
    check(r_skew["prompts"]["mul"]["first_word"]["p"] == 1.0, "unchanged prompt stays p=1 in skew run")
    check(sc.main([good, skew, "--json", os.path.join(d, "sc.json")]) == 0, "sampling_compare CLI")
    print("== fake server: bug (stale image cache · stop ignored · wrong answer at 1024)")
    STATE["mode"], STATE["sampling"] = "bug", "uniform"
    ge.main(["--base", base, "--out", bug, "--suite", "boundary,session", "--targets", "1023,1024,1025", "--label", "bug", "--cpt", "5.0"])
    B = json.load(open(bug))
    bi = {r["id"]: r for r in B["items"]}
    check(bi["boundary.1024"]["pass"] is False and bi["boundary.1023"]["pass"] and bi["boundary.1025"]["pass"], "boundary 1024 failure isolated")
    check(1024 in bi["boundary.1024"]["detail"]["probe_answer_fails_at_tokens"], "probe fail recorded at 1024")
    want = {"session.image_replace", "session.stop_openai_nonstream", "session.stop_openai_stream", "session.stop_anthropic"}
    got = {i for i, r in bi.items() if r["suite"] == "session" and not r["pass"]}
    check(got == want, f"bug mode session fails {sorted(got)}")
    res = qc.compare(G, B)
    p2f = sorted(i for s in res["suites"].values() for i in s["pass_to_fail"])
    check(p2f == sorted(want | {"boundary.1024"}), f"quality_compare reads gap output: {p2f}")
    srv.shutdown()
    print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    for m in FAILS:
        print("  -", m)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
