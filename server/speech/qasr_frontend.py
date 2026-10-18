# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Qwen3-ASR audio front end: WAV bytes -> 16 kHz mono float32 -> 128-bin log-mel features [128, frames].

Written from the textbook formulas (no code taken from other projects). The feature contract is the one the checkpoint was trained
with (a Whisper-style extractor, preprocessor_config.json: feature_size 128, n_fft 400, hop_length 160, sampling_rate 16000,
dither 0, no normalisation):
  1. frames: centred STFT — the signal is reflect-padded by n_fft/2 on both sides, frame i covers [i*hop, i*hop + n_fft).
     A signal of N samples gives 1 + N//hop frames; the last frame is dropped, so the model sees N//hop frames.
  2. window: periodic Hann, w[n] = 0.5 - 0.5*cos(2*pi*n/n_fft), n = 0..n_fft-1.
  3. power spectrum |FFT|^2 over the n_fft/2+1 = 201 one-sided bins (bin k = k*sr/n_fft Hz).
  4. mel filter bank: 128 triangular filters on the Slaney mel scale (linear below 1 kHz: m = 3f/200; logarithmic above:
     m = 15 + 27*ln(f/1000)/ln(6.4)), band edges equally spaced in mel between 0 Hz and 8 kHz, each triangle scaled by
     2/(f_right - f_left) ("Slaney" area normalisation).
  5. log10(max(mel, 1e-10)), floored at (global max - 8), then (x + 4) / 4.
The number of audio tokens the encoder produces for F frames is audio_token_count(F) (13 tokens per full 100-frame chunk).
tools/test_qasr_frontend_cpu.py checks this module against the reference extractor when one is available.
"""
from __future__ import annotations

import io
import struct
import wave

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 128
F_MAX = 8000.0


def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    lin = 3.0 * f / 200.0
    logm = 15.0 + np.log(np.maximum(f, 1e-12) / 1000.0) * (27.0 / np.log(6.4))
    return np.where(f >= 1000.0, logm, lin)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    lin = 200.0 * m / 3.0
    logf = 1000.0 * np.exp((m - 15.0) * (np.log(6.4) / 27.0))
    return np.where(m >= 15.0, logf, lin)


def mel_filters(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SAMPLE_RATE, fmax: float = F_MAX) -> np.ndarray:
    """[n_fft//2 + 1, n_mels] float64 triangular filters (Slaney scale and area normalisation)."""
    nbins = n_fft // 2 + 1
    fft_hz = np.linspace(0.0, sr / 2.0, nbins)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((nbins, n_mels), dtype=np.float64)
    for m in range(n_mels):
        lo, ce, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (fft_hz - lo) / (ce - lo)
        down = (hi - fft_hz) / (hi - ce)
        fb[:, m] = np.maximum(0.0, np.minimum(up, down)) * (2.0 / (hi - lo))
    return fb


_FB = None
_WIN = None


def log_mel(pcm: np.ndarray) -> np.ndarray:
    """16 kHz mono float signal -> float32 [128, len(pcm)//160] log-mel features."""
    global _FB, _WIN
    if _FB is None:
        _FB = mel_filters().astype(np.float32)
        _WIN = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
    x = np.asarray(pcm, dtype=np.float32).reshape(-1)
    n_out = x.shape[0] // HOP
    if n_out == 0:
        return np.zeros((N_MELS, 0), dtype=np.float32)
    pad = N_FFT // 2
    if x.shape[0] > pad:
        xp = np.pad(x, (pad, pad), mode="reflect")
    else:  # too short to reflect: mirror what is there (only for clips under 13 ms)
        xp = np.pad(x, (pad, pad), mode="symmetric")
    n_frames = 1 + x.shape[0] // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    # fp32 throughout (the extractor the model was trained with computes in fp32 too); 24 s of audio takes ~5 ms on one core
    frames = xp[idx[:-1]] * _WIN[None, :]  # the last frame is dropped anyway
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)
    power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)  # [n_out, 201]
    mel = power @ _FB  # [n_out, 128]
    logm = np.log10(np.maximum(mel, np.float32(1e-10)))
    logm = np.maximum(logm, logm.max() - np.float32(8.0))
    return np.ascontiguousarray(((logm + np.float32(4.0)) * np.float32(0.25)).T)


def audio_token_count(frames: int) -> int:
    """Encoder output length for `frames` mel frames: every full 100-frame chunk gives 13 tokens, a tail of r frames gives
    ceil(ceil(ceil(r/2)/2)/2) (three stride-2 convolutions with padding 1)."""
    full, r = divmod(int(frames), 100)
    n = 13 * full
    if r:
        a = (r - 1) // 2 + 1
        b = (a - 1) // 2 + 1
        n += (b - 1) // 2 + 1
    return n


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    """Linear-interpolation resampler for inputs that are not 16 kHz (the reference client always sends 16 kHz)."""
    if sr == SAMPLE_RATE or x.size == 0:
        return x
    n_out = int(round(x.shape[0] * SAMPLE_RATE / sr))
    t = np.arange(n_out) * (sr / SAMPLE_RATE)
    return np.interp(t, np.arange(x.shape[0]), x).astype(np.float32)


def decode_wav(data: bytes) -> np.ndarray:
    """RIFF/WAVE bytes (PCM 8/16/24/32-bit or IEEE float 32/64, any channel count) -> 16 kHz mono float32 in [-1, 1).
    Integer PCM is scaled by 1/2^(bits-1), as the common audio loaders do; channels are averaged."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("audio is not a RIFF/WAVE file")
    pos, fmt, payload = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
            if fmt[0] == 0xFFFE and len(body) >= 26:  # WAVE_FORMAT_EXTENSIBLE: the sub-format tag is the first 2 bytes of the GUID
                fmt = (struct.unpack("<H", body[24:26])[0],) + fmt[1:]
        elif cid == b"data":
            payload = body
            if fmt is not None:
                break
        pos += 8 + size + (size & 1)
    if fmt is None or payload is None:
        raise ValueError("WAV without fmt/data chunk")
    tag, ch, sr, _, _, bits = fmt
    ch = max(1, ch)
    if tag == 1 and bits == 16:
        x = np.frombuffer(payload[: len(payload) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 8:
        x = (np.frombuffer(payload, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(payload[: len(payload) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / float(1 << 23)
    elif tag == 1 and bits == 32:
        x = np.frombuffer(payload[: len(payload) // 4 * 4], dtype="<i4").astype(np.float32) / 2147483648.0
    elif tag == 3 and bits in (32, 64):
        x = np.frombuffer(payload[: len(payload) // (bits // 8) * (bits // 8)], dtype="<f4" if bits == 32 else "<f8").astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV encoding (format tag {tag}, {bits} bits)")
    if ch > 1:
        x = x[: x.shape[0] // ch * ch].reshape(-1, ch).mean(axis=1)
    return _resample(x, sr)


def wav_bytes(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """float [-1, 1] -> 16-bit mono WAV bytes (the same encoding the reference client sends)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()
