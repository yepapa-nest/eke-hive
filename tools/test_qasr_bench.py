#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""End-to-end ASR benchmark against an SGLang-style /generate endpoint (this repo's server/speech/asr_server.py or any server with the
same API): every clip of a directory is sent whole, as the reference client does (16-bit WAV, base64 data URL, temperature 0), and the
time from request to full response text is measured on the client.
Reports CER against transcript.tsv (character-weighted, whitespace and punctuation removed), latency p50 / p90 / max and per length band;
with --concurrency 2 the clips are sent two at a time (each request still timed on its own).
Several --url values are measured interleaved, clip by clip, so the servers see the same GPU conditions; --compare prints the clips whose
texts differ between them.
Usage: test_qasr_bench.py --wavs <dir> --url http://127.0.0.1:18440 [--url http://127.0.0.1:8444] [--rounds 3] [--language Korean|auto]
                          [--concurrency 1] [--out result.json]"""
import argparse
import base64
import json
import re
import statistics
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

PROMPT = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
          "<|im_start|>assistant\n")
ASR_TEXT = "<asr_text>"


def norm(s):
    return re.sub(r"[\s\.,\?\!~·'\"“”‘’…:;\-]", "", s)


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def request(url, wav_b64, language):
    prompt = PROMPT + (f"language {language}{ASR_TEXT}" if language and language != "auto" else "")
    body = json.dumps({"text": prompt, "audio_data": "data:audio/wav;base64," + wav_b64,
                       "sampling_params": {"temperature": 0, "max_new_tokens": 256}}).encode()
    t0 = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(url + "/generate", body, {"Content-Type": "application/json"}), timeout=60) as r:
        j = json.loads(r.read())
    ms = (time.perf_counter() - t0) * 1e3
    text = j["text"]
    if ASR_TEXT in text:
        text = text.split(ASR_TEXT, 1)[1]
    return text.strip(), ms, j


def pct(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", required=True)
    ap.add_argument("--url", action="append", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--language", default="Korean")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = Path(a.wavs)
    refs = dict(l.split("\t", 1) for l in (d / "transcript.tsv").read_text(encoding="utf-8").splitlines() if "\t" in l)
    clips = []
    for name in refs:
        raw = (d / name).read_bytes()
        with wave.open(str(d / name)) as w:
            dur = w.getnframes() / w.getframerate()
        clips.append((name, base64.b64encode(raw).decode(), dur))
    for u in a.url:  # warm-up (connection, first-use costs)
        request(u, clips[0][1], a.language)
    res = {u: {} for u in a.url}  # url -> clip -> [ms...], text
    texts = {u: {} for u in a.url}
    for rnd in range(a.rounds):
        for i in range(0, len(clips), a.concurrency):
            group = clips[i:i + a.concurrency]
            for u in a.url:
                out = {}

                def run(c, u=u, out=out):
                    out[c[0]] = request(u, c[1], a.language)

                th = [threading.Thread(target=run, args=(c,)) for c in group]
                for t in th:
                    t.start()
                for t in th:
                    t.join()
                for name, (text, ms, _) in out.items():
                    res[u].setdefault(name, []).append(ms)
                    texts[u][name] = text
    summary = {}
    dur = {c[0]: c[2] for c in clips}
    for u in a.url:
        lat = [ms for name in res[u] for ms in res[u][name]]
        tot_err = sum(cer(refs[n], texts[u][n]) * len(norm(refs[n])) for n in refs)
        tot_len = sum(len(norm(refs[n])) for n in refs)
        s = {"cer_total": round(tot_err / tot_len, 4), "p50_ms": round(statistics.median(lat), 1), "p90_ms": round(pct(lat, 0.9), 1),
             "max_ms": round(max(lat), 1), "n": len(lat), "concurrency": a.concurrency}
        for lo, hi in ((0, 3), (3, 6), (6, 30)):
            xs = [ms for n in res[u] if lo <= dur[n] < hi for ms in res[u][n]]
            if xs:
                s[f"p50_ms_{lo}-{hi}s"] = round(statistics.median(xs), 1)
        summary[u] = s
        print(u, json.dumps(s))
    if len(a.url) > 1:
        base = a.url[0]
        for u in a.url[1:]:
            diff = [n for n in refs if texts[u][n] != texts[base][n]]
            print(f"texts differing {base} vs {u}: {len(diff)}/{len(refs)}")
            for n in diff:
                print(f"  {n}\n    {base}: {texts[base][n]}\n    {u}: {texts[u][n]}\n    ref: {refs[n]}")
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "latency": res, "texts": texts}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
