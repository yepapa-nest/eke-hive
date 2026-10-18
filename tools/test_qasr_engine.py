#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Qwen3-ASR engine (libhive_qasr.so) against the fp32 oracle (oracle/qwen3_asr_oracle.py --dump <dir>), on a GPU:
  · per clip: audio encoder output (cosine, relative L2), first-step logits (cosine, top-1 agreement), greedy ids identical to the oracle
  · CER against transcript.tsv (character-weighted, whitespace and punctuation removed)
  · two concurrent requests give the same ids as alone (batched decode), a cancel flag stops a request, sampling (temperature 0.7,
    fixed seed) is reproducible and differs from greedy only where the distribution allows it
Usage: test_qasr_engine.py --ckpt <dir> --wavs <dir> --oracle <dump dir> [--transcript tsv] [--lib libhive_qasr.so]
Exit code 0 when every check passes."""
import argparse
import ctypes as C
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server" / "speech"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))
import qasr_frontend as fe  # noqa: E402
from qasr_engine import QasrEngine  # noqa: E402
from qwen3_asr_oracle import cer, norm_text  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    fails += not cond
    print(f"{'ok  ' if cond else 'FAIL'} {what}", flush=True)


def cos(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wavs", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--transcript", default="")
    ap.add_argument("--lib", default=None)
    ap.add_argument("--max-new", type=int, default=256, help="must match the oracle run's --max-new")
    ap.add_argument("--min-identical", type=float, default=0.9, help="share of clips whose greedy ids must equal the oracle's")
    a = ap.parse_args()
    orc = json.loads((Path(a.oracle) / "oracle.json").read_text())
    pre, post = orc["pre"], orc["post"]
    t0 = time.time()
    eng = QasrEngine(a.ckpt, lib=a.lib)
    i = eng.info
    print(f"engine up in {time.time() - t0:.1f}s: weights {i.weights_mib:.0f} MiB, KV {i.kv_mib:.0f}, work {i.work_mib:.0f}, context "
          f"{i.context_mib:.0f} ({'measured' if i.context_measured else 'estimate'}), plan {i.plan_mib:.0f}, measured {i.used_mib:.0f} MiB")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    gold = {}
    if a.transcript:
        for line in Path(a.transcript).read_text().splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                gold[k] = v
    clips = [c for c in sorted(Path(a.wavs).glob("*.wav")) if c.name in orc["clips"]]
    same = 0
    errs, lens = 0.0, 0
    mels = {}
    for c in clips:
        mel = fe.log_mel(fe.decode_wav(c.read_bytes()))
        mels[c.name] = mel
        r = eng.transcribe(mel, pre, post, max_new_tokens=a.max_new)
        text = tok.decode(r.ids, skip_special_tokens=True)
        line = f"{c.name}: {r.stats['total_ms']:.1f} ms (queue {r.stats['queue_ms']:.1f} enc {r.stats['encode_ms']:.1f} pf {r.stats['prefill_ms']:.1f} dec {r.stats['decode_ms']:.1f}, " \
               f"{r.stats['n_generated']} tok) {text}"
        oc = orc["clips"].get(c.name)
        if oc is not None:
            ident = oc["ids"] == r.ids
            same += ident
            line += f"  | ids {'==' if ident else '!='} oracle"
            ap_ = Path(a.oracle) / f"{c.name}.audio.npy"
            if ap_.exists():
                ea = eng.debug_encode(mel)
                oa = np.load(ap_)
                rel = float(np.linalg.norm(ea - oa) / np.linalg.norm(oa))
                line += f", audio cos {cos(ea, oa):.6f} rel {rel:.2e}"
                lo = np.load(Path(a.oracle) / f"{c.name}.logits.npy")
                el = eng.debug_prompt_logits(mel, pre, post)
                line += f", logits cos {cos(el, lo):.6f} top1 {'==' if int(el.argmax()) == int(lo.argmax()) else '!='}"
            if not ident:
                line += f"\n    oracle: {oc['text']}"
        if c.name in gold:
            n = len(norm_text(gold[c.name]))
            errs += cer(gold[c.name], text) * n
            lens += n
        print(line, flush=True)
    n_or = sum(1 for c in clips if c.name in orc["clips"])
    ok(same >= a.min_identical * n_or, f"greedy ids identical to the oracle on {same}/{n_or} clips")
    if lens:
        print(f"CER vs transcript: {errs / lens:.4f}")

    # concurrency: two requests at once must give the ids each gives alone
    names = [c.name for c in clips][-2:]
    alone = {n: eng.transcribe(mels[n], pre, post, max_new_tokens=a.max_new).ids for n in names}
    got = {}

    def run(n):
        got[n] = eng.transcribe(mels[n], pre, post, max_new_tokens=a.max_new).ids

    th = [threading.Thread(target=run, args=(n,)) for n in names]
    t1 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    ok(all(got[n] == alone[n] for n in names), f"two concurrent requests = alone ({(time.time() - t1) * 1e3:.0f} ms for both)")

    # cancel: a flag raised from another thread stops the request
    flag = C.c_int32(0)
    res = {}

    def run_c():
        res["r"] = eng.transcribe(mels[names[0]], pre, post, max_new_tokens=256, cancel=flag)

    t = threading.Thread(target=run_c)
    t.start()
    time.sleep(0.03)
    flag.value = 1
    t.join()
    ok(res["r"].stats["finish"] == "abort" and len(res["r"].ids) < 256,
       f"cancel stops the request (finish {res['r'].stats['finish']}, {len(res['r'].ids)} tokens)")

    # sampling
    s1 = eng.transcribe(mels[names[0]], pre, post, temperature=0.7, top_p=0.9, seed=42).ids
    s2 = eng.transcribe(mels[names[0]], pre, post, temperature=0.7, top_p=0.9, seed=42).ids
    s3 = eng.transcribe(mels[names[0]], pre, post, temperature=0.7, top_k=20, seed=7).ids
    ok(s1 == s2 and len(s1) > 0 and len(s3) > 0, "sampling with a fixed seed is reproducible")
    print(f"  sampled: {tok.decode(s1, skip_special_tokens=True)}")
    eng.close()
    print(f"{'PASS' if fails == 0 else 'FAIL'} ({fails} failed)")
    return fails


if __name__ == "__main__":
    sys.exit(main())
