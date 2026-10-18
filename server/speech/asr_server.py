#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Eke Hive speech recognition server: Qwen3-ASR (libhive_qasr.so) behind an SGLang-compatible native HTTP API.

Endpoints
  POST /generate   {"text": <prompt>, "audio_data": "data:audio/wav;base64,<WAV>", "sampling_params": {...}}
                   -> {"text": <generated text>, "output_ids": [...], "meta_info": {prompt_tokens, completion_tokens, finish_reason,
                       e2e_latency, timings}}
                   The prompt holds one <|audio_start|><|audio_pad|><|audio_end|> group; the single <|audio_pad|> (or a run of them) is
                   expanded to the audio encoder's token count. "input_ids" may replace "text". audio_data may be a one-element list.
                   sampling_params: temperature (default 1.0; 0 = greedy), top_p (1.0), top_k (-1 = off), max_new_tokens (128),
                   stop_token_ids (added to the model's stop tokens), ignore_eos, seed (`sampling_seed` also accepted).
                   A client that disconnects stops its request (the engine checks the flag before every decode step).
  GET  /health, /health_generate -> 200 {"status": "ok"}
  GET  /v1/models  -> {"object": "list", "data": [{"id": <served name>, ...}]}
  GET  /get_model_info -> {"model_path": ..., "is_generation": true}
Audio: RIFF/WAVE (PCM 8/16/24/32-bit or float, any channel count; non-16 kHz input is resampled linearly), decoded and turned into
  log-mel features on the CPU (qasr_frontend.py), so the GPU only ever sees the model.

Usage: asr_server.py --ckpt <Qwen3-ASR dir> [--host 127.0.0.1] [--port 18440] [--served-model-name asr] [--max-seqs 2]
                     [--max-ctx 2048] [--vram-limit-mb N] [--lib libhive_qasr.so]
HIVE_VRAM_LIMIT_MB caps the engine's startup plan when --vram-limit-mb is not given; startup fails (exit 2) if it does not fit.
"""
from __future__ import annotations

import os

# The log-mel matmul is tiny; a multi-threaded BLAS spawns one thread per host core and, in a CPU-limited container, burns the CFS quota
#   so the whole process (the engine's worker thread included) is throttled — measured 2026-10-18: 60-80 ms stalls on the first request of
#   a new length with --cpus 8. One thread is plenty.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import asyncio
import base64
import binascii
import ctypes
import itertools
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Request  # noqa: E402  (module level: the handlers' annotations are resolved from module globals)
from fastapi.responses import JSONResponse  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qasr_frontend as fe  # noqa: E402
from qasr_engine import QasrEngine  # noqa: E402

log = logging.getLogger("asr_server")

AUDIO_PAD = "<|audio_pad|>"


class Service:
    def __init__(self, a):
        from transformers import AutoTokenizer
        self.ckpt = a.ckpt
        self.name = a.served_model_name
        self.tok = AutoTokenizer.from_pretrained(a.ckpt)
        self.pad_id = self.tok.convert_tokens_to_ids(AUDIO_PAD)
        gen = {}
        p = Path(a.ckpt) / "generation_config.json"
        if p.exists():
            gen = json.loads(p.read_text())
        eos = gen.get("eos_token_id", [151643, 151645])
        self.eos = [int(x) for x in (eos if isinstance(eos, list) else [eos])]
        t0 = time.time()
        self.engine = QasrEngine(a.ckpt, lib=a.lib, max_seqs=a.max_seqs, max_ctx=a.max_ctx, vram_limit_mb=a.vram_limit_mb or 0,
                                 device=a.device)
        i = self.engine.info
        log.info("engine ready in %.1fs: weights %.0f MiB, KV %.0f MiB, work %.0f MiB, context %.0f MiB, plan %.0f MiB, measured %.0f MiB",
                 time.time() - t0, i.weights_mib, i.kv_mib, i.work_mib, i.context_mib, i.plan_mib, i.used_mib)
        self.max_ctx = i.max_ctx
        # one thread per request in flight (the engine batches up to max_seqs of them and queues the rest); mel features use the same pool
        self.pool = ThreadPoolExecutor(max_workers=max(8, 4 * a.max_seqs), thread_name_prefix="asr")
        self.prompt_cache: dict[str, tuple[list[int], list[int], bool]] = {}
        self.rid = itertools.count(1)

    def split_prompt(self, ids: list[int]) -> tuple[list[int], list[int], bool]:
        """ids -> (before the audio, after it, has audio). A run of <|audio_pad|> tokens counts as one audio placeholder."""
        pos = [k for k, t in enumerate(ids) if t == self.pad_id]
        if not pos:
            return ids, [], False
        if pos[-1] - pos[0] + 1 != len(pos):
            raise ValueError("only one audio input per request is supported")
        return ids[: pos[0]], ids[pos[-1] + 1:], True

    def prompt_ids(self, text: str) -> tuple[list[int], list[int], bool]:
        hit = self.prompt_cache.get(text)
        if hit is None:
            hit = self.split_prompt(self.tok(text, add_special_tokens=False)["input_ids"])
            if len(self.prompt_cache) < 256:
                self.prompt_cache[text] = hit
        return hit


def decode_audio_field(v) -> bytes:
    if isinstance(v, list):
        if len(v) != 1:
            raise ValueError("exactly one audio input per request is supported")
        v = v[0]
    if isinstance(v, dict):  # {"url": ...} / {"data": ...}
        v = v.get("url") or v.get("data") or ""
    if not isinstance(v, str) or not v:
        raise ValueError("audio_data must be a base64 string or data URL")
    if v.startswith("data:"):
        v = v.split(",", 1)[1] if "," in v else ""
    try:
        return base64.b64decode(v, validate=False)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"audio_data is not valid base64: {e}") from None


def build_app(svc: Service):
    app = FastAPI(title="Eke Hive ASR")

    def bad(msg, code=400):
        return JSONResponse({"error": {"message": msg, "code": code}}, status_code=code)

    @app.get("/health")
    @app.get("/health_generate")
    async def health():
        return {"status": "ok", "model": svc.name}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": svc.name, "object": "model", "owned_by": "eke-hive", "root": svc.ckpt,
                                            "max_model_len": svc.max_ctx}]}

    @app.get("/get_model_info")
    async def model_info():
        return {"model_path": svc.ckpt, "tokenizer_path": svc.ckpt, "is_generation": True}

    @app.post("/generate")
    async def generate(request: Request):
        t0 = time.perf_counter()
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return bad("request body is not JSON")
        if not isinstance(body, dict):
            return bad("request body must be a JSON object")
        if body.get("stream"):
            return bad("streaming is not supported by this server")
        sp = body.get("sampling_params") or {}
        if isinstance(body.get("text"), list) or isinstance(body.get("input_ids"), list) and body["input_ids"] and isinstance(body["input_ids"][0], list):
            return bad("batched requests are not supported; send one request per prompt")
        loop = asyncio.get_running_loop()
        try:
            if body.get("input_ids") is not None:
                pre, post, has_audio = svc.split_prompt([int(x) for x in body["input_ids"]])
            else:
                pre, post, has_audio = svc.prompt_ids(str(body.get("text") or ""))
            audio = body.get("audio_data")
            mel = None
            if audio:
                if not has_audio:
                    return bad("audio_data given but the prompt has no <|audio_pad|> placeholder")
                raw = decode_audio_field(audio)
                mel = await loop.run_in_executor(svc.pool, lambda: fe.log_mel(fe.decode_wav(raw)))
            elif has_audio:
                return bad("the prompt has an <|audio_pad|> placeholder but no audio_data was given")
            temperature = float(sp.get("temperature", 1.0))
            top_p = float(sp.get("top_p", 1.0))
            top_k = int(sp.get("top_k", -1))
            max_new = int(sp.get("max_new_tokens", 128) if sp.get("max_new_tokens") is not None else 128)
            seed = int(sp.get("seed", sp.get("sampling_seed", 0)) or 0)
            stop = [] if sp.get("ignore_eos") else list(svc.eos)
            stop += [int(x) for x in (sp.get("stop_token_ids") or [])]
        except (ValueError, TypeError) as e:
            return bad(str(e))
        if not pre and not post and mel is None:
            return bad("empty prompt")
        t_prep = time.perf_counter()
        cancel = ctypes.c_int32(0)
        fut = loop.run_in_executor(svc.pool, lambda: svc.engine.transcribe(mel, pre, post, max_new_tokens=max_new, temperature=temperature,
                                                                          top_p=top_p, top_k=top_k, seed=seed, stop_ids=stop, cancel=cancel))
        disconnected = False
        while True:
            done, _ = await asyncio.wait({fut}, timeout=0.05)
            if done:
                break
            if not disconnected and await request.is_disconnected():
                disconnected = True
                cancel.value = 1
        try:
            r = fut.result()
        except ValueError as e:
            return bad(str(e))
        rid = next(svc.rid)
        if disconnected:
            log.info("req %d cancelled by client disconnect after %d tokens", rid, len(r.ids))
            return JSONResponse({"error": {"message": "client disconnected", "code": 499}}, status_code=499)
        # Text: the generated tokens without the stop token, special tokens skipped. <asr_text> is an ordinary added token, so without a
        #   forced language the text reads "language Korean<asr_text>...", exactly what the client splits on (same as SGLang).
        text = svc.tok.decode(r.ids, skip_special_tokens=True)
        st = r.stats
        t1 = time.perf_counter()
        out_ids = list(r.ids)
        fin = {"type": st["finish"]}
        if st["finish"] == "stop" and st["stop_token"] >= 0:  # SGLang reports the matched stop token and counts it as generated
            fin["matched"] = st["stop_token"]
            out_ids.append(st["stop_token"])
        elif st["finish"] == "length":
            fin["length"] = st["n_generated"]
        meta = {
            "id": str(rid), "finish_reason": fin, "prompt_tokens": st["n_prompt"], "completion_tokens": len(out_ids),
            "cached_tokens": 0, "audio_tokens": st["n_audio"], "e2e_latency": t1 - t0,
            "timings_ms": {"preprocess": (t_prep - t0) * 1e3, "queue": st["queue_ms"], "encode": st["encode_ms"], "prefill": st["prefill_ms"],
                           "decode": st["decode_ms"], "engine": st["total_ms"], "total": (t1 - t0) * 1e3},
        }
        log.info("req %d: %d+%d tok, %.1f ms (prep %.1f queue %.1f enc %.1f prefill %.1f decode %.1f)", rid, st["n_prompt"], st["n_generated"],
                 (t1 - t0) * 1e3, (t_prep - t0) * 1e3, st["queue_ms"], st["encode_ms"], st["prefill_ms"], st["decode_ms"])
        return {"text": text, "output_ids": out_ids, "meta_info": meta}

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18440)
    ap.add_argument("--served-model-name", default="asr")
    ap.add_argument("--max-seqs", type=int, default=2)
    ap.add_argument("--max-ctx", type=int, default=2048)
    ap.add_argument("--vram-limit-mb", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--lib", default=None)
    ap.add_argument("--log-level", default="info")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        svc = Service(a)
    except Exception as e:  # noqa: BLE001
        log.error("startup failed: %s", e)
        sys.exit(2)
    import uvicorn
    uvicorn.run(build_app(svc), host=a.host, port=a.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
