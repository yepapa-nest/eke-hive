#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Reference run for the Qwen3-ASR oracle: transcribes a directory of 16 kHz WAV clips with the model's own reference implementation
(the `qwen_asr` package on top of transformers, fp32 on the CPU, greedy) and writes what the oracle is checked against:
  <out>/ref.json        {clip: {"prompt_ids": [...], "ids": [...], "text": "..."}}
  <out>/<clip>.logits.npy  first-step logits (the distribution after the full prompt) for the first --logits clips
  <out>/<clip>.mel.npy     the reference log-mel features for the same clips
Needs an environment with qwen_asr + transformers installed (this script only calls their public API; nothing is copied).
Usage: test_qasr_ref_dump.py --ckpt <dir> --wavs <dir> --out <dir> [--language Korean] [--logits 4] [--threads 16]"""
import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

PROMPT = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
          "<|im_start|>assistant\n")


def read_wav(p):
    with wave.open(str(p), "rb") as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2, p
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wavs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="Korean", help="forced language ('' = let the model write it)")
    ap.add_argument("--logits", type=int, default=4)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--only", default="", help="comma-separated clip names")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    import qwen_asr  # noqa: F401  (registers the model classes with transformers' Auto* factories)
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(a.ckpt, dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(a.ckpt, fix_mistral_regex=True)
    prompt = PROMPT + (f"language {a.language}<asr_text>" if a.language else "")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "ref.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    clips = sorted(Path(a.wavs).glob("*.wav"))
    if a.only:
        keep = set(a.only.split(","))
        clips = [c for c in clips if c.name in keep]
    for i, c in enumerate(clips):
        if c.name in res:
            continue
        t0 = time.time()
        pcm = read_wav(c)
        inp = proc(text=[prompt], audio=[pcm], return_tensors="pt", padding=True)
        n_in = inp["input_ids"].shape[1]
        with torch.no_grad():
            if i < a.logits:
                lo = model.thinker(**inp).logits[0, -1].float().numpy()
                np.save(out / f"{c.name}.logits.npy", lo)
                n = int(inp["feature_attention_mask"][0].sum())
                np.save(out / f"{c.name}.mel.npy", inp["input_features"][0, :, :n].float().numpy())
            g = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False)
        ids = g.sequences[0, n_in:].tolist()
        text = proc.batch_decode([ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        res[c.name] = {"prompt_ids": inp["input_ids"][0].tolist(), "ids": ids, "text": text}
        res_path.write_text(json.dumps(res, ensure_ascii=False, indent=0))
        print(f"{c.name}: {time.time() - t0:.1f}s {len(ids)} tok  {text}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
