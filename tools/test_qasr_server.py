#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HTTP surface of server/speech/asr_server.py (a running server), optionally side by side with a reference server of the same API:
  · GET /health, /v1/models
  · POST /generate with the forced-language prompt and with no language (the model writes "language X<asr_text>..."), audio_data as a
    data URL and as a one-element list; response fields (text, output_ids incl. the stop token, meta_info prompt/completion tokens,
    finish_reason) — and, with --ref, the same text / ids / token counts as the reference server for every clip given
  · errors: no JSON, audio without placeholder, placeholder without audio, bad base64, non-WAV audio -> HTTP 400, server still healthy
  · client disconnect: a request whose connection is closed early is cancelled (checked through a request that would otherwise run to
    max_new_tokens), and the server keeps serving
Usage: test_qasr_server.py --url http://127.0.0.1:18440 --wavs <dir> [--ref http://127.0.0.1:8444] [--clips a.wav,b.wav]
Exit code = number of failed checks."""
import argparse
import base64
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

PROMPT = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
          "<|im_start|>assistant\n")
fails = 0


def ok(cond, what):
    global fails
    fails += not cond
    print(f"{'ok  ' if cond else 'FAIL'} {what}", flush=True)


def post(url, body, raw=None, timeout=60):
    data = raw if raw is not None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url + "/generate", data, {"Content-Type": "application/json"}), timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=10) as r:
        return r.status, json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--wavs", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--clips", default="ryan-03.wav,vivian-00.wav,sohee-09.wav")
    a = ap.parse_args()
    st, j = get(a.url, "/health")
    ok(st == 200 and j.get("status") == "ok", "GET /health")
    st, j = get(a.url, "/v1/models")
    ok(st == 200 and j["data"][0]["id"] == "asr", "GET /v1/models lists 'asr'")
    for name in a.clips.split(","):
        wav = base64.b64encode((Path(a.wavs) / name).read_bytes()).decode()
        for lang in ("Korean", ""):
            body = {"text": PROMPT + (f"language {lang}<asr_text>" if lang else ""), "audio_data": "data:audio/wav;base64," + wav,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 256}}
            st, j = post(a.url, body)
            m = j.get("meta_info", {})
            good = st == 200 and isinstance(j.get("text"), str) and m.get("finish_reason", {}).get("type") == "stop" \
                and m["finish_reason"].get("matched") == j["output_ids"][-1] and m.get("completion_tokens") == len(j["output_ids"])
            if not lang:
                good = good and j["text"].startswith("language ") and "<asr_text>" in j["text"]
            ok(good, f"{name} {'forced' if lang else 'auto'}: {j.get('text')!r} ({m.get('prompt_tokens')}+{m.get('completion_tokens')} tok)")
            st2, j2 = post(a.url, dict(body, audio_data=[body["audio_data"]]))
            ok(st2 == 200 and j2["output_ids"] == j["output_ids"], f"{name}: audio_data as a one-element list gives the same result")
            if a.ref:
                rs, rj = post(a.ref, body)
                rm = rj.get("meta_info", {})
                same = rs == 200 and rj["text"] == j["text"] and rj.get("output_ids") == j["output_ids"] \
                    and rm.get("prompt_tokens") == m.get("prompt_tokens") and rm.get("completion_tokens") == m.get("completion_tokens") \
                    and rm.get("finish_reason") == m.get("finish_reason")
                ok(same, f"{name} {'forced' if lang else 'auto'}: same text / ids / token counts / finish_reason as the reference"
                         + ("" if same else f"\n    ref: {rj.get('text')!r} {rm.get('prompt_tokens')}+{rm.get('completion_tokens')} {rm.get('finish_reason')}"))
    wav = base64.b64encode((Path(a.wavs) / a.clips.split(",")[0]).read_bytes()).decode()
    sp = {"temperature": 0, "max_new_tokens": 16}
    cases = {
        "body is not JSON": (None, b"{not json"),
        "audio but no placeholder": ({"text": "<|im_start|>user\nhi<|im_end|>\n", "audio_data": "data:audio/wav;base64," + wav, "sampling_params": sp}, None),
        "placeholder but no audio": ({"text": PROMPT, "sampling_params": sp}, None),
        "bad base64": ({"text": PROMPT, "audio_data": "data:audio/wav;base64,@@@@", "sampling_params": sp}, None),
        "not a WAV": ({"text": PROMPT, "audio_data": "data:audio/wav;base64," + base64.b64encode(b"hello world, not audio").decode(), "sampling_params": sp}, None),
        "two audio inputs": ({"text": PROMPT, "audio_data": ["data:audio/wav;base64," + wav] * 2, "sampling_params": sp}, None),
    }
    for what, (body, raw) in cases.items():
        st, j = post(a.url, body, raw)
        ok(st == 400 and "error" in j, f"error path '{what}' -> HTTP {st}")
    st, j = post(a.url, {"text": "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n", "sampling_params": {"temperature": 0, "max_new_tokens": 8}})
    ok(st == 200 and j["meta_info"]["completion_tokens"] >= 1, f"text-only prompt works: {j.get('text')!r}")
    # client disconnect -> cancelled: a long text-only generation (ignore_eos, 1500 tokens) is abandoned after 0.3 s
    u = urlparse(a.url)
    body = json.dumps({"text": "<|im_start|>user\nCount from one to one thousand.<|im_end|>\n<|im_start|>assistant\n",
                       "sampling_params": {"temperature": 0, "max_new_tokens": 1500, "ignore_eos": True}}).encode()
    s = socket.create_connection((u.hostname, u.port))
    s.sendall(b"POST /generate HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    time.sleep(0.3)
    s.close()
    t0 = time.perf_counter()
    st, j = post(a.url, {"text": PROMPT + "language Korean<asr_text>", "audio_data": "data:audio/wav;base64," + wav, "sampling_params": sp})
    waited = (time.perf_counter() - t0) * 1e3
    # 1500 tokens would hold a slot for seconds; after the cancel the next request runs at its normal speed (< 1 s even under load)
    ok(st == 200 and waited < 1000, f"disconnected request cancelled, next request served in {waited:.0f} ms")
    st, j = get(a.url, "/health")
    ok(st == 200, "server healthy after the error and cancel paths")
    print(f"{'PASS' if fails == 0 else 'FAIL'} ({fails} failed)")
    return fails


if __name__ == "__main__":
    sys.exit(main())
