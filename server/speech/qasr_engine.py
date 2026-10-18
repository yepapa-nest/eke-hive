# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""ctypes binding of libhive_qasr.so (engine/include/hive/qasr/qasr.h). ctypes releases the GIL for the duration of every call, so
several Python threads can be inside transcribe() at once; the engine batches them (up to max_seqs) on its own worker thread."""
from __future__ import annotations

import ctypes as C
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class Options(C.Structure):
    _fields_ = [("device", C.c_int), ("max_seqs", C.c_int), ("max_ctx", C.c_int), ("vram_limit_mb", C.c_int), ("prefill_chunk", C.c_int),
                ("encoder_windows", C.c_int), ("warmup", C.c_int)]


class Stats(C.Structure):
    _fields_ = [("n_prompt", C.c_int), ("n_audio", C.c_int), ("n_generated", C.c_int), ("finish", C.c_int), ("stop_token", C.c_int), ("queue_ms", C.c_double),
                ("encode_ms", C.c_double), ("prefill_ms", C.c_double), ("decode_ms", C.c_double), ("total_ms", C.c_double)]


class Info(C.Structure):
    _fields_ = [("weights_mib", C.c_double), ("kv_mib", C.c_double), ("work_mib", C.c_double), ("context_mib", C.c_double),
                ("plan_mib", C.c_double), ("used_mib", C.c_double), ("context_measured", C.c_int), ("max_seqs", C.c_int),
                ("max_ctx", C.c_int), ("vocab", C.c_int), ("hidden", C.c_int), ("n_mels", C.c_int)]


FINISH = {0: "stop", 1: "length", 2: "abort"}


@dataclass
class Result:
    ids: list[int]
    stats: dict


def _find_lib() -> str:
    env = os.environ.get("HIVE_QASR_LIB")
    if env:
        return env
    root = Path(__file__).resolve().parents[2]
    for cand in (root / "engine" / "build-asr" / "libhive_qasr.so", root / "engine" / "build" / "libhive_qasr.so"):
        if cand.exists():
            return str(cand)
    return "libhive_qasr.so"


class QasrEngine:
    def __init__(self, ckpt: str, lib: str | None = None, max_seqs: int = 2, max_ctx: int = 2048, vram_limit_mb: int = 0,
                 prefill_chunk: int = 512, encoder_windows: int = 8, device: int = 0, warmup: bool = True):
        self.lib = C.CDLL(lib or _find_lib())
        L = self.lib
        f32p, i32p = C.POINTER(C.c_float), C.POINTER(C.c_int32)
        L.qasr_default_options.argtypes = [C.POINTER(Options)]
        L.qasr_create.restype = C.c_void_p
        L.qasr_create.argtypes = [C.c_char_p, C.POINTER(Options), C.c_char_p, C.c_int]
        L.qasr_destroy.argtypes = [C.c_void_p]
        L.qasr_get_info.argtypes = [C.c_void_p, C.POINTER(Info)]
        L.qasr_audio_tokens.argtypes = [C.c_int]
        L.qasr_audio_tokens.restype = C.c_int
        L.qasr_transcribe.restype = C.c_int
        L.qasr_transcribe.argtypes = [C.c_void_p, f32p, C.c_int, C.c_int, i32p, C.c_int, i32p, C.c_int, C.c_float, C.c_float, C.c_int,
                                      C.c_uint64, C.c_int, i32p, C.c_int, C.POINTER(C.c_int32), i32p, C.c_int, C.POINTER(Stats), C.c_char_p,
                                      C.c_int]
        L.qasr_debug_encode.restype = C.c_int
        L.qasr_debug_encode.argtypes = [C.c_void_p, f32p, C.c_int, C.c_int, f32p, C.c_int, C.c_char_p, C.c_int]
        L.qasr_debug_prompt_logits.restype = C.c_int
        L.qasr_debug_prompt_logits.argtypes = [C.c_void_p, f32p, C.c_int, C.c_int, i32p, C.c_int, i32p, C.c_int, f32p, C.c_char_p, C.c_int]
        o = Options()
        L.qasr_default_options(C.byref(o))
        o.device, o.max_seqs, o.max_ctx, o.vram_limit_mb = device, max_seqs, max_ctx, vram_limit_mb
        o.prefill_chunk, o.encoder_windows, o.warmup = prefill_chunk, encoder_windows, int(warmup)
        err = C.create_string_buffer(1024)
        self.h = L.qasr_create(str(ckpt).encode(), C.byref(o), err, len(err))
        if not self.h:
            raise RuntimeError(f"qasr_create failed: {err.value.decode(errors='replace')}")
        self.info = Info()
        L.qasr_get_info(self.h, C.byref(self.info))

    def close(self):
        if getattr(self, "h", None):
            self.lib.qasr_destroy(self.h)
            self.h = None

    def audio_tokens(self, n_frames: int) -> int:
        return self.lib.qasr_audio_tokens(n_frames)

    @staticmethod
    def _ids(a) -> tuple:
        arr = np.ascontiguousarray(np.asarray(a, dtype=np.int32).reshape(-1))
        return arr, arr.ctypes.data_as(C.POINTER(C.c_int32)), arr.size

    @staticmethod
    def _mel(mel):
        if mel is None or mel.size == 0:
            return None, None, 128, 0
        m = np.ascontiguousarray(mel, dtype=np.float32)
        return m, m.ctypes.data_as(C.POINTER(C.c_float)), m.shape[0], m.shape[1]

    def transcribe(self, mel, pre, post, max_new_tokens=256, temperature=0.0, top_p=1.0, top_k=0, seed=0, stop_ids=(151643, 151645),
                   cancel: C.c_int32 | None = None) -> Result:
        mk, mp, nm, nf = self._mel(mel)
        pa, pp, pn = self._ids(pre)
        qa, qp, qn = self._ids(post)
        sa, sp, sn = self._ids(list(stop_ids))
        cap = max(1, int(max_new_tokens))
        out = np.zeros(cap, dtype=np.int32)
        st = Stats()
        err = C.create_string_buffer(512)
        flag = cancel if cancel is not None else C.c_int32(0)
        n = self.lib.qasr_transcribe(self.h, mp, nm, nf, pp, pn, qp, qn, float(temperature), float(top_p), int(top_k), int(seed) & (2**64 - 1),
                                     int(max_new_tokens), sp, sn, C.byref(flag), out.ctypes.data_as(C.POINTER(C.c_int32)), cap,
                                     C.byref(st), err, len(err))
        if n < 0:
            raise ValueError(err.value.decode(errors="replace"))
        stats = {k: getattr(st, k) for k, _ in Stats._fields_}
        stats["finish"] = FINISH.get(st.finish, str(st.finish))
        return Result(out[:n].tolist(), stats)

    def debug_encode(self, mel) -> np.ndarray:
        mk, mp, nm, nf = self._mel(mel)
        na = self.audio_tokens(nf)
        out = np.zeros((na, self.info.hidden), dtype=np.float32)
        err = C.create_string_buffer(512)
        n = self.lib.qasr_debug_encode(self.h, mp, nm, nf, out.ctypes.data_as(C.POINTER(C.c_float)), na, err, len(err))
        if n < 0:
            raise ValueError(err.value.decode(errors="replace"))
        return out

    def debug_prompt_logits(self, mel, pre, post) -> np.ndarray:
        mk, mp, nm, nf = self._mel(mel)
        pa, pp, pn = self._ids(pre)
        qa, qp, qn = self._ids(post)
        out = np.zeros(self.info.vocab, dtype=np.float32)
        err = C.create_string_buffer(512)
        n = self.lib.qasr_debug_prompt_logits(self.h, mp, nm, nf, pp, pn, qp, qn, out.ctypes.data_as(C.POINTER(C.c_float)), err, len(err))
        if n < 0:
            raise ValueError(err.value.decode(errors="replace"))
        return out
