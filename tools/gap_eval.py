#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Fill hive validation gaps — against a live server (:8430). Measures four things quality_eval.py does not cover.

  boundary : boundary-length prefill — sends prompts sized **exactly** to 15/16, 127/128/129, 1023/1024/1025, 2687/2688/2689, 16383/16384/16385, 32767/32768,
             98303/98304/98305 tokens (server usage.prompt_tokens = tokens the daemon receives, chat template included)
             and asks at the very end about a fact stated at the very beginning. Fitting attempts (probes) are real queries too, so their answers and prefill times are recorded.
             Each attempt starts with a different nonce, so prefix sharing (HIVE_PREFIX_SHARE) is not hit (cached_prefix is recorded).
  session  : continuation, mid-conversation edit (turn 1/turn 2), system prompt change, regeneration, image change (different image next turn / image replaced in the same turn),
             disconnect mid-stream -> next request, stop strings (OpenAI stream/non-stream, Anthropic), EOS.
  load     : c16 concurrent streams (answers checked), inter-chunk gap P50/P95/P99, TTFT distribution, 4 disconnects under load, c1 disconnect response x3,
             (--host-metrics) max hived/server RSS and VRAM (/proc and nvidia-smi on the serving host).
  sampling : temperature>0 (default 0.7, top_p 0.9), 8 prompts x n=64 (seed 1..n) — distribution comparison with tools/sampling_compare.py.

The result JSON's items have the same shape as quality_eval, so two runs (boundary, session, load) can also be compared with tools/quality_compare.py.
Usage: gap_eval.py --base http://127.0.0.1:8430 --out gap.json [--suite boundary,session,load,sampling] [--corpus F] [--host-metrics]
      [--n 64] [--temperature 0.7] [--top-p 0.9] [--sampling-conc 1] [--label mtp-on]
"""
import argparse
import datetime as dt
import glob
import json
import os
import random
import re
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
import zlib

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import quality_eval as qe  # noqa: E402

SUITES = ["boundary", "session", "load", "sampling"]
BOUNDARIES = [15, 16, 127, 128, 129, 1023, 1024, 1025, 2687, 2688, 2689, 16383, 16384, 16385, 32767, 32768, 98303, 98304, 98305]
NOTHINK = {"reasoning_effort": "none"}


# ============================================================================================ common
def pct(xs, q):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 4)


def chat_body(messages, max_tokens, seed=0, stream=False, **kw):
    b = {"model": "hive", "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "seed": seed, "stream": stream, **NOTHINK}
    b.update(kw)
    return b


def nonstream(h, body):
    """Non-streaming OpenAI -> dict(content, finish, stop_reason, prompt_tokens, completion_tokens, prefill_ms, cached_prefix, latency, error)."""
    st, j, lat = h.post("/v1/chat/completions", body)
    out = {"latency": lat, "error": None, "content": "", "finish": None, "stop_reason": None, "prompt_tokens": None,
           "completion_tokens": None, "prefill_ms": None, "cached_prefix": None}
    if st != 200:
        out["error"] = f"HTTP {st}: {json.dumps(j, ensure_ascii=False)[:400]}"
        return out
    ch = (j.get("choices") or [{}])[0]
    u, hv = j.get("usage") or {}, j.get("hive") or {}
    out.update(content=(ch.get("message") or {}).get("content") or "", finish=ch.get("finish_reason"), stop_reason=ch.get("stop_reason"),
               prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
               prefill_ms=hv.get("prefill_ms"), cached_prefix=hv.get("cached_prefix"))
    return out


def stream_raw(h, body, abort_after=None, path="/v1/chat/completions"):
    """Collect an OpenAI stream with chunk timestamps. abort_after=N closes the socket after N body chunks (client cancellation)."""
    c = h._conn()
    t0 = time.time()
    out = {"times": [], "content": "", "reasoning": "", "finish": None, "stop_reason": None, "prompt_tokens": None, "completion_tokens": None,
           "aborted": False, "t_abort": None, "ttft": None, "error": None, "t_start": t0, "first_piece": None}
    try:
        c.request("POST", path, body=json.dumps(body, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
        sock = c.sock
        r = c.getresponse()
        if r.status != 200:
            out["error"] = f"HTTP {r.status}: {r.read()[:400]!r}"
            return out
        while True:
            line = r.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except ValueError:
                continue
            if d.get("error"):
                out["error"] = json.dumps(d["error"], ensure_ascii=False)[:400]
                continue
            if d.get("usage"):
                out["prompt_tokens"] = d["usage"].get("prompt_tokens")
                out["completion_tokens"] = d["usage"].get("completion_tokens")
            for ch in d.get("choices") or []:
                dl = ch.get("delta") or {}
                t = time.time() - t0
                if dl.get("content") or dl.get("reasoning_content"):
                    out["content"] += dl.get("content") or ""
                    out["reasoning"] += dl.get("reasoning_content") or ""
                    out["times"].append(t)
                    if out["ttft"] is None:
                        out["ttft"] = t
                        out["first_piece"] = (dl.get("content") or dl.get("reasoning_content") or "")
                if ch.get("finish_reason"):
                    out["finish"] = ch["finish_reason"]
                    out["stop_reason"] = ch.get("stop_reason")
            if abort_after is not None and len(out["times"]) >= abort_after:
                out["aborted"] = True
                out["t_abort"] = time.time()
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                break
    except (OSError, ValueError) as e:
        if not out["aborted"]:
            out["error"] = f"{type(e).__name__}: {e}"
    finally:
        c.close()
    out["latency"] = time.time() - t0
    return out


def png_solid(rgb, w=448, h=448):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def img_part(rgb):
    import base64
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png_solid(rgb)).decode()}}


RED, BLUE = (220, 20, 20), (20, 40, 220)


# ============================================================================================ host metrics (on the serving host only)
class HostSampler:
    """Every 0.5 s: sum of hived/hive_server RSS (/proc) and GPU memory.used (nvidia-smi) — max per phase."""
    def __init__(self, on):
        self.on, self.phase, self.max, self.stop_ev, self.th, self.notes = on, "idle", {}, threading.Event(), None, []

    def _pids(self):
        out = {}
        for p in glob.glob("/proc/[0-9]*"):
            try:
                comm = open(p + "/comm").read().strip()
                cmd = open(p + "/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if comm == "hived":
                out[int(p[6:])] = "hived"
            elif "hive_server.py" in cmd:
                out[int(p[6:])] = "server"
        return out

    def sample(self):
        s = {}
        for pid, name in self._pids().items():
            try:
                for line in open(f"/proc/{pid}/status"):
                    if line.startswith("VmRSS:"):
                        s[name + "_rss_mb"] = s.get(name + "_rss_mb", 0) + int(line.split()[1]) / 1024
            except OSError:
                pass
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                s["vram_used_mb"] = sum(float(x) for x in r.stdout.split())
        except (OSError, subprocess.SubprocessError):
            pass
        return s

    def _loop(self):
        while not self.stop_ev.is_set():
            s = self.sample()
            m = self.max.setdefault(self.phase, {})
            for k, v in s.items():
                m[k] = round(max(m.get(k, 0), v), 1)
            self.stop_ev.wait(0.5)

    def start(self):
        if self.on:
            first = self.sample()
            if not any(k.endswith("rss_mb") for k in first):
                self.notes.append("no hived/hive_server process visible in /proc (not on the ws host, or other pid namespace)")
            if "vram_used_mb" not in first:
                self.notes.append("nvidia-smi unavailable")
            self.th = threading.Thread(target=self._loop, daemon=True)
            self.th.start()

    def stop(self):
        self.stop_ev.set()
        if self.th:
            self.th.join(2)
        return {"on": self.on, "max_by_phase": self.max, "notes": self.notes}


# ============================================================================================ (1) boundary lengths
TINY_Q, TINY_A = "7+5=?", "12"  # shortest question — the 15-token target must also fit the chat template
FACT_Q = "\n\nWhat is the secret code word stated at the very beginning of this message? Answer with the code word only."


class BoundaryFitter:
    """Prompt = [nonce] fact + filler[:chars] + " x"*pad + question (small targets: " x"*pad + short arithmetic). Fitted exactly using server usage."""
    def __init__(self, h, seed, filler_text, cpt, log, max_attempts=8):
        self.h, self.seed, self.fill, self.cpt, self.log, self.max_attempts = h, seed, filler_text, cpt, log, max_attempts
        self.base = {}

    def prompt(self, kind, chars, pad, nonce, code):
        if kind == "tiny":
            return " x" * pad + (" " if pad else "") + TINY_Q
        f = self.fill[:chars]
        if chars and chars < len(self.fill):
            j = f.rfind(" ")
            f = f[:j] if j > chars * 0.9 else f
        return f"[{nonce}] The secret code word is {code}.\n\n" + f + " x" * pad + FACT_Q

    def ask(self, kind, chars, pad, nonce, code):
        p = self.prompt(kind, chars, pad, nonce, code)
        r = nonstream(self.h, chat_body([{"role": "user", "content": p}], 24, self.seed))
        want = TINY_A if kind == "tiny" else code
        r["pass"] = (not r["error"]) and want.lower() in (r["content"] or "").lower()
        r["chars"], r["pad"] = len(p), pad
        return r

    def fit(self, target):
        rng = random.Random(f"{self.seed}:boundary:{target}")
        code = f"{rng.choice(['AMBER', 'COBALT', 'VELVET', 'LUNAR', 'IVORY', 'TUNDRA'])}-{rng.randrange(1000, 9999)}"
        kind = "tiny" if target < 64 else "fact"
        attempts = []
        if kind not in self.base:
            r0 = self.ask(kind, 0, 0, "00000000", code)
            self.base[kind] = r0["prompt_tokens"]
            self.log(f"  base[{kind}] = {r0['prompt_tokens']} tokens")
        t0 = self.base[kind]
        if t0 is None:
            return {"target": target, "exact": False, "attempts": [], "error": "base probe failed"}
        if target < t0:
            return {"target": target, "exact": False, "attempts": [], "skipped": f"minimum prompt is {t0} tokens"}
        if kind == "tiny" or target - t0 <= 48:
            chars, pad = 0, target - t0
        else:
            pad = 16
            chars = int((target - t0 - pad) * self.cpt)
        for k in range(self.max_attempts):
            nonce = "%08d" % rng.randrange(10 ** 7, 10 ** 8)
            if chars > len(self.fill):
                return {"target": target, "exact": False, "attempts": attempts, "error": f"filler too short ({len(self.fill)} chars)"}
            r = self.ask(kind, chars, pad, nonce, code)
            attempts.append({k2: r.get(k2) for k2 in ("prompt_tokens", "pass", "prefill_ms", "cached_prefix", "latency", "content", "error", "chars", "pad")})
            t = r["prompt_tokens"]
            if t is None:
                break
            if t == target:
                return {"target": target, "exact": True, "attempts": attempts, "final": r, "code": code, "kind": kind}
            d = target - t
            if chars and t - t0 - pad > 0:
                self.cpt = 0.5 * self.cpt + 0.5 * (chars / max(1, t - t0 - pad))  # update the chars/token ratio from the measurement
            if 0 <= pad + d <= 256 and (abs(d) <= 64 or not chars):
                pad += d
            else:
                chars = max(0, chars + int(d * self.cpt))
                pad = 16 if chars else max(0, pad + d)
        return {"target": target, "exact": False, "attempts": attempts, "code": code, "kind": kind}


def run_boundary(h, seed, filler_text, cpt, targets, log):
    bf = BoundaryFitter(h, seed, filler_text, cpt, log)
    recs = []
    for T in targets:
        res = bf.fit(T)
        fin = res.get("final") or {}
        probe_fails = [a["prompt_tokens"] for a in res["attempts"] if a.get("pass") is False]
        rec = {"id": f"boundary.{T}", "suite": "boundary", "kind": res.get("kind"), "target": T, "exact": res["exact"],
               "achieved": fin.get("prompt_tokens") if res["exact"] else (res["attempts"][-1]["prompt_tokens"] if res["attempts"] else None),
               "pass": (bool(fin.get("pass")) if res["exact"] else None), "ttft_s": (fin.get("prefill_ms") or 0) / 1000 if fin.get("prefill_ms") is not None else None,
               "latency_s": fin.get("latency"), "prompt_tokens": fin.get("prompt_tokens"), "completion_tokens": fin.get("completion_tokens"),
               "cached_prefix": fin.get("cached_prefix"), "output": fin.get("content", ""),
               "detail": {"attempts": res["attempts"], "probe_answer_fails_at_tokens": probe_fails,
                          **({"skipped": res["skipped"]} if res.get("skipped") else {}), **({"error": res["error"]} if res.get("error") else {})}}
        if res.get("skipped"):
            rec["skipped"] = res["skipped"]
        recs.append(rec)
        log(f"  {'PASS' if rec['pass'] else ('SKIP' if rec['pass'] is None else 'FAIL')} boundary {T:>6} achieved {rec['achieved']} exact {res['exact']} "
            f"attempts {len(res['attempts'])} ttft {rec['ttft_s']} cached {rec['cached_prefix']}" + (f" · probe fails at {probe_fails}" if probe_fails else "")
            + (f" · {res.get('skipped') or res.get('error')}" if res.get("skipped") or res.get("error") else ""))
    return recs, round(bf.cpt, 4)


# ============================================================================================ (2) session
def _rec(i, ok, kind, r=None, **detail):
    r = r or {}
    return {"id": i, "suite": "session", "kind": kind, "pass": bool(ok), "latency_s": r.get("latency"), "ttft_s": r.get("ttft"),
            "prompt_tokens": r.get("prompt_tokens"), "completion_tokens": r.get("completion_tokens"), "output": r.get("content", ""), "detail": detail}


def _has(r, w):
    return (not r.get("error")) and w.lower() in (r.get("content") or "").lower()


def health_counter(h):
    try:
        st, body = h.get("/health")
        d = json.loads(body).get("daemon") or {}
        return d.get("decode_routed"), d.get("pending_requests")
    except (OSError, ValueError):
        return None, None


def cancel_probe(h, seed, tag, idle_ttft):
    """Cut the stream -> (a) wait until the decode_routed counter is quiet for 1.5 s (snapshot throttled to 1 s, so ~1 s resolution) (b) TTFT of a short request on another session."""
    long_body = chat_body([{"role": "user", "content": f"[{tag}] Count from 1 to 600, separated by spaces."}], 1500, seed, stream=True)
    r = stream_raw(h, long_body, abort_after=30)
    t_abort = r["t_abort"] or time.time()
    last, t_change = health_counter(h)[0], time.time()
    while time.time() - t_abort < 15:
        c = health_counter(h)[0]
        if c != last:
            last, t_change = c, time.time()
        elif time.time() - t_change >= 1.5:
            break
        time.sleep(0.05)
    quiet = round(max(0.0, t_change - t_abort), 3)
    nxt = stream_raw(h, chat_body([{"role": "user", "content": f"[{tag}-next] What is 3 + 4? Answer with the number only."}], 16, seed, stream=True))
    return r, {"decode_counter_quiet_after_abort_s": quiet, "next_ttft_s": nxt["ttft"], "idle_ttft_s": idle_ttft, "next_ok": _has(nxt, "7"),
               "chunks_before_abort": len(r["times"]), "abort_error": r["error"]}, nxt


def run_session(h, seed, log):
    recs = []
    sid = f"gap-{seed}"
    # a) continuation
    m = [{"role": "user", "content": "Remember this: the harbour master's name is Ingrid Sorensen. Reply 'Noted.'"}]
    r1 = nonstream(h, chat_body(m, 32, seed))
    m += [{"role": "assistant", "content": r1["content"]}, {"role": "user", "content": "What is the harbour master's surname? One word."}]
    r2 = nonstream(h, chat_body(m, 32, seed))
    recs.append(_rec("session.continue", _has(r2, "sorensen"), "continue", r2, cached_prefix_turn2=r2["cached_prefix"], prompt_tokens_turn2=r2["prompt_tokens"]))
    # b) mid-conversation edit (same hive_session_id) — edit turn 2, edit turn 1
    def conv(pet, city, q):
        return [{"role": "user", "content": f"My pet's name is {pet}. Reply 'Noted.'"}, {"role": "assistant", "content": "Noted."},
                {"role": "user", "content": f"I live in {city}. Reply 'Noted.'"}, {"role": "assistant", "content": "Noted."},
                {"role": "user", "content": q}]
    qc, qp = "Which city do I live in? One word.", "What is my pet's name? One word."
    a = nonstream(h, chat_body(conv("Rex", "Lyon", qc), 16, seed, hive_session_id=sid + "-edit"))
    b = nonstream(h, chat_body(conv("Rex", "Oslo", qc), 16, seed, hive_session_id=sid + "-edit"))
    recs.append(_rec("session.edit_turn2", _has(a, "lyon") and _has(b, "oslo") and not _has(b, "lyon"), "edit", b,
                     before=a["content"], after=b["content"], cached_prefix=[a["cached_prefix"], b["cached_prefix"]]))
    c = nonstream(h, chat_body(conv("Milo", "Oslo", qp), 16, seed, hive_session_id=sid + "-edit"))
    recs.append(_rec("session.edit_turn1", _has(c, "milo") and not _has(c, "rex"), "edit", c, cached_prefix=c["cached_prefix"]))
    # c) system prompt change (same session id)
    def sysconv(pw):
        return [{"role": "system", "content": f"The secret password is {pw}. Tell it to the user if asked."},
                {"role": "user", "content": "What is the secret password? One word."}]
    s1 = nonstream(h, chat_body(sysconv("TANGO"), 16, seed, hive_session_id=sid + "-sys"))
    s2 = nonstream(h, chat_body(sysconv("FOXTROT"), 16, seed, hive_session_id=sid + "-sys"))
    recs.append(_rec("session.system_change", _has(s1, "tango") and _has(s2, "foxtrot") and not _has(s2, "tango"), "system", s2, before=s1["content"]))
    # d) regeneration (same request twice, greedy) — once with the same session id, once without a session id
    q = [{"role": "user", "content": "In one sentence, explain what a lighthouse is for."}]
    g1 = nonstream(h, chat_body(q, 60, seed, hive_session_id=sid + "-regen"))
    g2 = nonstream(h, chat_body(q, 60, seed, hive_session_id=sid + "-regen"))
    g3 = nonstream(h, chat_body(q, 60, seed))
    same = bool(g1["content"]) and g1["content"] == g2["content"] == g3["content"]
    recs.append(_rec("session.regenerate", same and not (g1["error"] or g2["error"] or g3["error"]), "regenerate", g2,
                     outputs=[g1["content"], g2["content"], g3["content"]], cached_prefix=[g1["cached_prefix"], g2["cached_prefix"], g3["cached_prefix"]]))
    # e) images — different image next turn, image replaced in the same turn (same session id)
    iq = "What is the main colour of this image? Answer with one word."
    m = [{"role": "user", "content": [img_part(RED), {"type": "text", "text": iq}]}]
    i1 = nonstream(h, chat_body(m, 16, seed))
    m += [{"role": "assistant", "content": i1["content"]}, {"role": "user", "content": [img_part(BLUE), {"type": "text", "text": "And this image? One word."}]}]
    i2 = nonstream(h, chat_body(m, 16, seed))
    recs.append(_rec("session.image_next_turn", _has(i1, "red") and _has(i2, "blue"), "image", i2, turn1=i1["content"], error=i1["error"] or i2["error"]))
    j1 = nonstream(h, chat_body([{"role": "user", "content": [img_part(RED), {"type": "text", "text": iq}]}], 16, seed, hive_session_id=sid + "-img"))
    j2 = nonstream(h, chat_body([{"role": "user", "content": [img_part(BLUE), {"type": "text", "text": iq}]}], 16, seed, hive_session_id=sid + "-img"))
    recs.append(_rec("session.image_replace", _has(j1, "red") and _has(j2, "blue") and not _has(j2, "red"), "image", j2, before=j1["content"],
                     cached_prefix=[j1["cached_prefix"], j2["cached_prefix"]], error=j1["error"] or j2["error"]))
    # f) disconnect -> continue the same conversation, other request
    idle = stream_raw(h, chat_body([{"role": "user", "content": "[idle] What is 3 + 4? Answer with the number only."}], 16, seed, stream=True))
    cm = [{"role": "user", "content": "Count from 1 to 600, separated by spaces."}]
    ca = stream_raw(h, chat_body(cm, 1500, seed, stream=True, hive_session_id=sid + "-cancel"), abort_after=25)
    cont = nonstream(h, chat_body(cm + [{"role": "assistant", "content": ca["content"]}, {"role": "user", "content": "Stop counting. What is 6 + 7? Number only."}],
                                  16, seed, hive_session_id=sid + "-cancel"))
    r, info, nxt = cancel_probe(h, seed, "sess", idle["ttft"])
    recs.append(_rec("session.cancel_then_continue", ca["aborted"] and _has(cont, "13"), "cancel", cont, partial_chars=len(ca["content"]),
                     cached_prefix=cont["cached_prefix"]))
    recs.append(_rec("session.cancel_then_other", info["next_ok"], "cancel", nxt, **info))
    # g) stop strings, EOS
    sq = [{"role": "user", "content": "Repeat exactly, with nothing else: alpha beta END gamma delta"}]
    st1 = nonstream(h, chat_body(sq, 40, seed, stop=["END"]))
    ok1 = _has(st1, "alpha") and "gamma" not in st1["content"] and "END" not in st1["content"] and st1["finish"] == "stop" and st1["stop_reason"] == "END"
    recs.append(_rec("session.stop_openai_nonstream", ok1, "stop", st1, finish=st1["finish"], stop_reason=st1["stop_reason"]))
    st2 = stream_raw(h, chat_body(sq, 40, seed, stream=True, stop=["END"]))
    ok2 = _has(st2, "alpha") and "gamma" not in st2["content"] and "END" not in st2["content"] and st2["finish"] == "stop" and st2["stop_reason"] == "END"
    recs.append(_rec("session.stop_openai_stream", ok2, "stop", st2, finish=st2["finish"], stop_reason=st2["stop_reason"]))
    stc, sj, lat = h.post("/v1/messages", {"model": "hive", "max_tokens": 40, "temperature": 0.0, "thinking": {"type": "disabled"},
                                            "stop_sequences": ["END"], "messages": sq})
    txt = "".join(b.get("text", "") for b in (sj.get("content") or []) if b.get("type") == "text") if stc == 200 else ""
    ok3 = stc == 200 and "alpha" in txt.lower() and "gamma" not in txt and sj.get("stop_reason") == "stop_sequence" and sj.get("stop_sequence") == "END"
    recs.append(_rec("session.stop_anthropic", ok3, "stop", {"content": txt, "latency": lat}, stop_reason=sj.get("stop_reason"), stop_sequence=sj.get("stop_sequence"), http=stc))
    eo = nonstream(h, chat_body([{"role": "user", "content": "Say only the word: yes"}], 64, seed))
    recs.append(_rec("session.eos", _has(eo, "yes") and eo["finish"] == "stop" and (eo["completion_tokens"] or 99) < 64, "eos", eo,
                     finish=eo["finish"], completion_tokens=eo["completion_tokens"]))
    for x in recs:
        log(f"  {'PASS' if x['pass'] else 'FAIL'} {x['id']:<34} {json.dumps(x['detail'], ensure_ascii=False)[:150]}")
    return recs


# ============================================================================================ (3) load
TOPICS = ["lighthouses", "bees", "glaciers", "trains", "volcanoes", "libraries", "owls", "bridges", "tea", "comets", "rivers", "bicycles",
          "deserts", "violins", "harbours", "maps"]


def run_load(h, seed, conc, log, sampler):
    rng = random.Random(f"{seed}:load")
    prompts = []
    for i in range(conc):
        a, b = rng.randrange(100, 999), rng.randrange(100, 999)
        prompts.append((f"[load {i}] First line: the value of {a} + {b} as a number. Then write about 120 words about {TOPICS[i % len(TOPICS)]}.", str(a + b)))
    n_cancel = min(4, conc // 4)
    cancel_ids = set(range(conc - n_cancel, conc))
    res = [None] * conc
    sampler.phase = f"load_c{conc}"
    t0 = time.time()

    def one(i):
        res[i] = stream_raw(h, chat_body([{"role": "user", "content": prompts[i][0]}], 256, seed, stream=True), abort_after=20 if i in cancel_ids else None)
    th = [threading.Thread(target=one, args=(i,)) for i in range(conc)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    sampler.phase = "load_after"
    recs, gaps, ttfts, toks = [], [], [], 0
    for i, r in enumerate(res):
        g = [b - a for a, b in zip(r["times"], r["times"][1:])]
        if i not in cancel_ids:
            gaps += g
            ttfts.append(r["ttft"])
            toks += r["completion_tokens"] or 0
            ok = not r["error"] and r["finish"] in ("stop", "length") and prompts[i][1] in r["content"]
        else:
            ok = r["aborted"] and not r["error"]
        recs.append({"id": f"load.c{conc}.{i}", "suite": "load", "kind": "cancelled" if i in cancel_ids else "stream", "pass": bool(ok),
                     "latency_s": r["latency"], "ttft_s": r["ttft"], "completion_tokens": r["completion_tokens"], "output": r["content"],
                     "detail": {"finish": r["finish"], "chunks": len(r["times"]), "gap_p50": pct(g, 50), "gap_max": pct(g, 100), "error": r["error"]}})
    after = stream_raw(h, chat_body([{"role": "user", "content": "[after load] What is 12 + 30? Answer with the number only."}], 16, seed, stream=True))
    recs.append({"id": "load.after_short", "suite": "load", "kind": "after", "pass": _has(after, "42"), "latency_s": after["latency"], "ttft_s": after["ttft"],
                 "output": after["content"], "detail": {"error": after["error"]}})
    metrics = {"conc": conc, "cancelled_streams": sorted(cancel_ids), "wall_s": round(wall, 2), "completion_tokens_finished": toks,
               "aggregate_tok_s": round(toks / wall, 2) if wall else None,
               "chunk_gap_s": {"n": len(gaps), "p50": pct(gaps, 50), "p95": pct(gaps, 95), "p99": pct(gaps, 99), "max": pct(gaps, 100)},
               "ttft_s": {"p50": pct(ttfts, 50), "p95": pct(ttfts, 95), "max": pct(ttfts, 100)},
               "note": "gaps are between SSE content chunks (a chunk can carry >1 token when MTP accepts drafts)"}
    log(f"  c{conc}: {sum(1 for x in recs[:conc] if x['pass'])}/{conc} ok · wall {wall:.1f}s · agg {metrics['aggregate_tok_s']} tok/s · gap p50/p95/p99 "
        f"{metrics['chunk_gap_s']['p50']}/{metrics['chunk_gap_s']['p95']}/{metrics['chunk_gap_s']['p99']} s · ttft p50 {metrics['ttft_s']['p50']}")
    # c1 disconnect response x3
    sampler.phase = "cancel_c1"
    idle = stream_raw(h, chat_body([{"role": "user", "content": "[idle2] What is 3 + 4? Answer with the number only."}], 16, seed, stream=True))
    cr = []
    for k in range(3):
        r, info, nxt = cancel_probe(h, seed, f"load-{k}", idle["ttft"])
        cr.append(info)
        recs.append({"id": f"load.cancel_c1.{k}", "suite": "load", "kind": "cancel", "pass": bool(r["aborted"] and info["next_ok"]), "latency_s": nxt["latency"],
                     "ttft_s": nxt["ttft"], "output": nxt["content"], "detail": info})
    metrics["cancel_c1"] = {"quiet_s": [c["decode_counter_quiet_after_abort_s"] for c in cr], "next_ttft_s": [c["next_ttft_s"] for c in cr],
                            "idle_ttft_s": idle["ttft"], "scope": "quiet_s = decode_routed stats counter unchanged 1.5 s (daemon snapshot throttle ~1 s → coarse)"}
    log(f"  cancel c1: quiet {metrics['cancel_c1']['quiet_s']} s · next ttft {metrics['cancel_c1']['next_ttft_s']} (idle {idle['ttft']})")
    sampler.phase = "idle"
    return recs, metrics


# ============================================================================================ (4) sampling
SAMPLING_PROMPTS = [
    ("rand10", "Pick a random integer between 1 and 10. Answer with the number only.", None),
    ("mul", "What is 17 * 23? Answer with the number only.", "391"),
    ("color", "Name one primary colour. Answer with one word.", None),
    ("canberra", "What is the capital of Australia? Answer with one word.", "canberra"),
    ("seoul_ko", "대한민국의 수도는 어디인가요? 한 단어로 답하세요.", "서울"),
    ("story", "Write a one-sentence story about a cat.", None),
    ("prime97", "Is 97 a prime number? Answer yes or no.", "yes"),
    ("rhyme", "Give one English word that rhymes with 'light'. Answer with the word only.", None),
]


def run_sampling(h, n, temperature, top_p, conc, log):
    jobs = [(pid, p, ans, s) for pid, p, ans in SAMPLING_PROMPTS for s in range(1, n + 1)]
    out = {pid: [None] * n for pid, _, _ in SAMPLING_PROMPTS}
    lock = threading.Lock()
    it = iter(jobs)

    def worker():
        while True:
            with lock:
                try:
                    pid, p, ans, s = next(it)
                except StopIteration:
                    return
            r = stream_raw(h, chat_body([{"role": "user", "content": p}], 64, s, stream=True, temperature=temperature, top_p=top_p))
            # first stream chunk (the server detokenizer may merge tokens, so an approximation of the "first token" — usually 1 token) and the first word (normalised)
            m = re.match(r"\s*([^\s]+)", r["content"] or "")
            first_word = re.sub(r"[^\w가-힣]+", "", m.group(1)).lower() if m else ""
            out[pid][s - 1] = {"seed": s, "text": r["content"], "first_word": first_word, "first_piece": r["first_piece"], "completion_tokens": r["completion_tokens"],
                               "finish": r["finish"], "ttft": r["ttft"], "correct": (None if ans is None else ans.lower() in (r["content"] or "").lower()),
                               "error": r["error"]}
    th = [threading.Thread(target=worker) for _ in range(max(1, conc))]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    summ = {}
    for pid, p, ans in SAMPLING_PROMPTS:
        xs = out[pid]
        fw = {}
        for x in xs:
            fw[x["first_word"]] = fw.get(x["first_word"], 0) + 1
        lens = [x["completion_tokens"] for x in xs if x["completion_tokens"] is not None]
        summ[pid] = {"n": len(xs), "errors": sum(1 for x in xs if x["error"]), "first_word_top": sorted(fw.items(), key=lambda kv: -kv[1])[:6],
                     "mean_len": round(statistics.mean(lens), 2) if lens else None,
                     "accuracy": (round(sum(1 for x in xs if x["correct"]) / len(xs), 4) if ans else None)}
        log(f"  {pid:<9} n {len(xs)} err {summ[pid]['errors']} mean_len {summ[pid]['mean_len']} acc {summ[pid]['accuracy']} top {summ[pid]['first_word_top'][:3]}")
    return {"params": {"n": n, "temperature": temperature, "top_p": top_p, "conc": conc, "max_tokens": 64, "seeds": f"1..{n}", "reasoning_effort": "none"},
            "prompts": {pid: {"prompt": p, "answer": ans} for pid, p, ans in SAMPLING_PROMPTS}, "samples": out, "summary": summ,
            "wall_s": round(time.time() - t0, 1)}


# ============================================================================================ main
def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--base", default="http://127.0.0.1:8430")
    cli.add_argument("--out", required=True)
    cli.add_argument("--suite", default="boundary,session,load,sampling")
    cli.add_argument("--seed", type=int, default=20261001)
    cli.add_argument("--corpus", default=None, help="filler corpus text file (e.g. a bench corpus). Default: generated English prose")
    cli.add_argument("--targets", default=",".join(map(str, BOUNDARIES)))
    cli.add_argument("--cpt", type=float, default=3.5, help="initial filler chars/token estimate (refined from measurements)")
    cli.add_argument("--conc", type=int, default=16)
    cli.add_argument("--host-metrics", action="store_true", help="on the serving host: max hived/hive_server RSS from /proc and VRAM from nvidia-smi")
    cli.add_argument("--n", type=int, default=64)
    cli.add_argument("--temperature", type=float, default=0.7)
    cli.add_argument("--top-p", type=float, default=0.9)
    cli.add_argument("--sampling-conc", type=int, default=1)
    cli.add_argument("--label", default="")
    cli.add_argument("--timeout", type=float, default=1800)
    a = cli.parse_args(argv)
    suites = [s.strip() for s in a.suite.split(",") if s.strip()]
    for s in suites:
        if s not in SUITES:
            cli.error(f"unknown suite {s}")
    h = qe.Http(a.base, a.timeout)

    def log(m):
        print(m, flush=True)
    t_start = time.time()
    meta = {"label": a.label, "base": a.base, "seed": a.seed, "suites": suites, "started": dt.datetime.now().isoformat(timespec="seconds"), "corpus": a.corpus}
    try:
        st, body = h.get("/health")
        meta["health"] = {"status": st, "body": body[:4000]}
    except OSError as e:
        meta["health"] = {"error": str(e)}
    sampler = HostSampler(a.host_metrics)
    sampler.start()
    items, extra = [], {}
    for s in suites:
        t_s = time.time()
        log(f"[{s}]")
        sampler.phase = s
        if s == "boundary":
            targets = [int(x) for x in a.targets.split(",") if x.strip()]
            need = int(max(targets) * max(a.cpt, 5.0) * 1.3) + 1000
            filler = qe.Filler(a.corpus)
            text = filler.get(random.Random(f"{a.seed}:bfill"), need, "en") if not a.corpus else (filler.corpus * (need // max(1, len(filler.corpus)) + 1))[:need]
            recs, cpt = run_boundary(h, a.seed, text, a.cpt, targets, log)
            items += recs
            meta["boundary_cpt_final"] = cpt
        elif s == "session":
            items += run_session(h, a.seed, log)
        elif s == "load":
            recs, metrics = run_load(h, a.seed, a.conc, log, sampler)
            items += recs
            extra["load"] = metrics
        elif s == "sampling":
            extra["sampling"] = run_sampling(h, a.n, a.temperature, a.top_p, a.sampling_conc, log)
        meta.setdefault("suite_wall_s", {})[s] = round(time.time() - t_s, 1)
    extra["host_metrics"] = sampler.stop()
    for r in items:
        o = r.get("output") or ""
        r["output_sha"] = qe.sha(o)
        r["output"] = o[:4000]
    meta["wall_s"] = round(time.time() - t_start, 1)
    summary = {}
    for s in ("boundary", "session", "load"):
        rs = [r for r in items if r["suite"] == s]
        if rs:
            ps = [r for r in rs if r.get("pass") is not None]
            summary[s] = {"n": len(ps), "pass": sum(1 for r in ps if r["pass"]), "skipped": len(rs) - len(ps)}
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "summary": summary, "items": items, **extra}, f, ensure_ascii=False, indent=1)
    print(f"\n== summary ({a.label or a.out}) · wall {meta['wall_s']}s")
    for s, v in summary.items():
        print(f"  {s:<9} {v['pass']}/{v['n']}" + (f" · skipped {v['skipped']}" if v["skipped"] else ""))
        for r in items:
            if r["suite"] == s and r.get("pass") is False:
                print(f"     FAIL {r['id']}: {json.dumps(r.get('detail'), ensure_ascii=False)[:200]}")
    if "load" in extra:
        print(f"  load metrics: {json.dumps({k: v for k, v in extra['load'].items() if k != 'note'}, ensure_ascii=False)}")
    if extra.get("host_metrics", {}).get("on"):
        print(f"  host max: {json.dumps(extra['host_metrics'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
