#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Qwen3-ASR front end (server/speech/qasr_frontend.py), CPU only, no checkpoint needed:
  · audio_token_count against the closed form the reference encoder uses, for every frame count 1..4000
  · WAV decoding (16-bit round trip, stereo averaging, 8/24/32-bit and float encodings, extensible format tag)
  · log-mel features against transformers' WhisperFeatureExtractor(feature_size=128) when transformers is installed
    (max abs difference <= 2e-4 on noise, tones, silence-with-click and, with HIVE_TEST_QASR_WAVS=<dir>, real WAV files).
Exit code = number of failed checks."""
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server" / "speech"))
import qasr_frontend as fe  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    fails += not cond
    print(f"{'ok  ' if cond else 'FAIL'} {what}")


# ---- audio token count ----
def closed_form(n):
    leave = n % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (n // 100) * 13


bad = [n for n in range(1, 4001) if fe.audio_token_count(n) != closed_form(n)]
ok(not bad, f"audio_token_count matches the closed form for 1..4000 frames (mismatches: {bad[:5]})")
ok([fe.audio_token_count(n) for n in (100, 200, 1300)] == [13, 26, 169], "whole chunks give 13 tokens each")

# ---- WAV decoding ----
rng = np.random.default_rng(0)
x = (rng.standard_normal(16000) * 0.2).clip(-1, 1).astype(np.float32)
y = fe.decode_wav(fe.wav_bytes(x))
ok(y.dtype == np.float32 and y.shape == x.shape and np.abs(y - x).max() < 1.0 / 16000, "16-bit mono WAV round trip")


def riff(fmt_tag, ch, sr, bits, payload, extensible=False):
    ba = ch * bits // 8
    if extensible:
        fmt = struct.pack("<HHIIHH", 0xFFFE, ch, sr, sr * ba, ba, bits) + struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", fmt_tag) + b"\0" * 14
    else:
        fmt = struct.pack("<HHIIHH", fmt_tag, ch, sr, sr * ba, ba, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"LIST" + struct.pack("<I", 4) + b"INFO" + b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


st = np.stack([x, -x * 0.5], axis=1)
y = fe.decode_wav(riff(1, 2, 16000, 16, (st * 32767).astype("<i2").tobytes()))
ok(np.abs(y - (x - 0.5 * x) / 2).max() < 2e-4, "stereo is averaged (with an extra LIST chunk before data)")
y = fe.decode_wav(riff(3, 1, 16000, 32, x.astype("<f4").tobytes()))
ok(np.array_equal(y, x), "float32 WAV")
y = fe.decode_wav(riff(3, 1, 16000, 32, x.astype("<f4").tobytes(), extensible=True))
ok(np.array_equal(y, x), "WAVE_FORMAT_EXTENSIBLE float32")
v = np.round(x * 8388607).astype(np.int32)
b24 = np.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], axis=1).astype(np.uint8).tobytes()
y = fe.decode_wav(riff(1, 1, 16000, 24, b24))
ok(np.abs(y - x).max() < 1e-6, "24-bit WAV")
y = fe.decode_wav(riff(1, 1, 8000, 16, (x[::2] * 32767).astype("<i2").tobytes()))
ok(y.shape[0] == 16000, "8 kHz input is resampled to 16 kHz")
try:
    fe.decode_wav(b"not a wav")
    ok(False, "garbage is rejected")
except ValueError:
    ok(True, "garbage is rejected")

# ---- log-mel against the reference extractor ----
ok(fe.log_mel(np.zeros(100, np.float32)).shape == (128, 0), "shorter than one hop -> zero frames")
ok(fe.log_mel(x).shape == (128, 100), "1 s -> 100 frames")
try:
    from transformers import WhisperFeatureExtractor
except Exception as e:  # noqa: BLE001
    print(f"skip reference log-mel comparison (transformers not importable: {e})")
    WhisperFeatureExtractor = None

if WhisperFeatureExtractor is not None:
    ref = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, n_fft=400, chunk_length=30, padding_value=0.0)
    t = np.arange(16000 * 3) / 16000.0
    cases = {
        "noise 1.3 s": (rng.standard_normal(20800) * 0.1).astype(np.float32),
        "tone 3 s": (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 3100 * t)).astype(np.float32),
        "odd length 0.73 s": (rng.standard_normal(11713) * 0.05).astype(np.float32),
        "click in silence": np.concatenate([np.zeros(8000), [0.9], np.zeros(8159)]).astype(np.float32),
    }
    wav_dir = os.environ.get("HIVE_TEST_QASR_WAVS")
    if wav_dir:
        for p in sorted(Path(wav_dir).glob("*.wav"))[:12]:
            cases[p.name] = fe.decode_wav(p.read_bytes())
    for name, sig in cases.items():
        r = ref(sig, sampling_rate=16000, padding=True, truncation=False, return_attention_mask=True, return_tensors="np")
        n = int(r["attention_mask"][0].sum())
        want = r["input_features"][0][:, :n]
        got = fe.log_mel(sig)
        same_shape = got.shape == want.shape
        err = float(np.abs(got - want).max()) if same_shape else float("inf")
        ok(same_shape and err <= 2e-4, f"log-mel {name}: shape {got.shape} vs {want.shape}, max |diff| {err:.2e}")

print(f"{'PASS' if fails == 0 else 'FAIL'} ({fails} failed)")
sys.exit(fails)
