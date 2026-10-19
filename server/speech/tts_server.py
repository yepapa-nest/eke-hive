#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Eke Hive speech server — streaming text-to-speech (Qwen3-TTS 12Hz CustomVoice) on the qtts engine (libhive_qtts.so, ctypes).

API (one request per WebSocket connection):
  WS  /ws/speak   client text frame {"type": "speak", "text": "...", "voice": "sohee", "instruct": "", "language": "Korean"}
                  (optional "temperature" 0.05-2.0, "top_k" 1-2048 and "seed" ≥ 0 override the server's sampling for this request;
                   "temperature" / "top_k" set both stages, "sub_temperature" / "sub_top_k" then set the code predictor's alone)
                  -> binary frames: raw PCM16 little-endian mono 24 kHz, 7680 samples (4 codec frames, 320 ms) each, the last one shorter
                  -> text frame {"type": "done", "audio_ms": N, "first_ms": N, "total_ms": N}
                  errors: {"type": "error", "message": "..."} then done. Empty text: done with audio_ms 0 only.
                  Closing the connection stops the generation before the next frame (barge-in): the socket is read while generating.
                  Silence: after speech, silent chunks are held and at most --silence-keep-ms of a pause is sent; a pause of
                  --silence-stop-ms (or that long with no speech at all) ends the take ("done" then carries "stopped": "silence").
  GET /health     {"ok": true, "speakers": [...], ...} — ok once the engine is loaded, its graphs captured and one sentence synthesized.
One generation runs at a time; further connections wait their turn (a client that leaves while waiting is skipped).

The WebSocket layer is a small RFC 6455 server on asyncio streams (no extensions, no subprotocols), so the server needs only the Python
standard library, ctypes and transformers (the checkpoint's Qwen2 tokenizer).

Usage: python3 server/speech/tts_server.py --model /models/Qwen3-TTS-12Hz-1.7B-CustomVoice --port 18440 [--lib .../libhive_qtts.so]
       [--vram-limit-mb N] [--max-seq 2048] [--seed N]
"""
from __future__ import annotations

import argparse
import array
import asyncio
import base64
import concurrent.futures
import ctypes
import glob
import hashlib
import json
import math
import os
import secrets
import struct
import sys
import time

SAMPLE_RATE = 24000
CHUNK_SAMPLES = 4 * 1920
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_MESSAGE = 1 << 20


# ---------------------------------------------------------------------------------------------------------------------------------------
# engine binding
class Sampling(ctypes.Structure):
    _fields_ = [("temperature", ctypes.c_float), ("top_k", ctypes.c_int32), ("sub_temperature", ctypes.c_float), ("sub_top_k", ctypes.c_int32),
                ("rep_penalty", ctypes.c_float), ("min_frames", ctypes.c_int32), ("max_frames", ctypes.c_int32)]


def find_lib(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("HIVE_QTTS_LIB"):
        return os.environ["HIVE_QTTS_LIB"]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hits = sorted(glob.glob(os.path.join(root, "engine", "build*", "libhive_qtts.so")), key=os.path.getmtime, reverse=True)
    if not hits:
        raise FileNotFoundError("libhive_qtts.so not found (build the hive_qtts target or pass --lib / HIVE_QTTS_LIB)")
    return hits[0]


class Engine:
    """ctypes wrapper of hive/qtts/qtts_capi.h. ctypes releases the GIL for the duration of each call."""

    def __init__(self, lib_path: str, model_dir: str, vram_limit_mb: int = 0, max_seq: int = 0, dump_frames: int = 0, flags: int = 0):
        L = ctypes.CDLL(lib_path)
        P, I32, U64 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint64
        pi32 = ctypes.POINTER(ctypes.c_int32)
        L.qtts_create.restype = P
        L.qtts_create.argtypes = [ctypes.c_char_p, I32, I32, I32, I32, ctypes.c_char_p, I32]
        L.qtts_destroy.argtypes = [P]
        L.qtts_error.restype = ctypes.c_char_p
        L.qtts_error.argtypes = [P]
        L.qtts_info.restype = I32
        L.qtts_info.argtypes = [P, ctypes.c_char_p, I32]
        L.qtts_start.restype = I32
        L.qtts_start.argtypes = [P, pi32, I32, pi32, I32, I32, I32, U64, ctypes.POINTER(Sampling), pi32, I32]
        L.qtts_next_chunk.restype = I32
        L.qtts_next_chunk.argtypes = [P, ctypes.POINTER(ctypes.c_int16), I32, pi32]
        for fn in ("qtts_frames", "qtts_prompt_rows"):
            getattr(L, fn).restype = I32
            getattr(L, fn).argtypes = [P]
        L.qtts_get_codes.restype = I32
        L.qtts_get_codes.argtypes = [P, pi32, I32]
        self.L = L
        err = ctypes.create_string_buffer(1024)
        self.h = L.qtts_create(model_dir.encode(), vram_limit_mb, max_seq, dump_frames, flags, err, len(err))
        if not self.h:
            raise RuntimeError(f"qtts_create failed: {err.value.decode(errors='replace')}")
        n = L.qtts_info(self.h, None, 0)
        buf = ctypes.create_string_buffer(n + 1)
        L.qtts_info(self.h, buf, n + 1)
        self.info = json.loads(buf.value.decode())
        self.pcm = (ctypes.c_int16 * CHUNK_SAMPLES)()

    def _err(self) -> str:
        return self.L.qtts_error(self.h).decode(errors="replace")

    def start(self, ids: list[int], instruct: list[int] | None, speaker_id: int, language_id: int, seed: int,
              sampling: Sampling | None = None) -> int:
        a = (ctypes.c_int32 * len(ids))(*ids)
        ins = (ctypes.c_int32 * len(instruct))(*instruct) if instruct else None
        r = self.L.qtts_start(self.h, a, len(ids), ins, len(instruct or []), speaker_id, language_id, seed & (2**64 - 1),
                              ctypes.byref(sampling) if sampling else None, None, 0)
        if r < 0:
            raise RuntimeError(self._err())
        return r

    def next_chunk(self, cancel: ctypes.c_int32) -> bytes | None:
        """PCM16 bytes of the next chunk; None when finished or cancelled."""
        n = self.L.qtts_next_chunk(self.h, self.pcm, CHUNK_SAMPLES, ctypes.byref(cancel))
        if n == -1:
            raise RuntimeError(self._err())
        if n <= 0:
            return None
        return ctypes.string_at(self.pcm, n * 2)

    def frames(self) -> int:
        return self.L.qtts_frames(self.h)

    def codes(self) -> list[list[int]]:
        n = self.frames()
        buf = (ctypes.c_int32 * (16 * max(n, 1)))()
        got = self.L.qtts_get_codes(self.h, buf, n)
        return [list(buf[i * 16:(i + 1) * 16]) for i in range(got)]

    def close(self) -> None:
        if self.h:
            self.L.qtts_destroy(self.h)
            self.h = None


# ---------------------------------------------------------------------------------------------------------------------------------------
# prompt
class Prompter:
    """Tokenizes the CustomVoice templates and maps voice / language names to codec ids (checkpoint config.json)."""

    def __init__(self, model_dir: str):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        tc = cfg["talker_config"]
        self.spk = {k.lower(): v for k, v in tc["spk_id"].items()}
        self.speakers = sorted(tc["spk_id"].keys())  # the order the old server reports
        self.langs = {k.lower(): v for k, v in tc["codec_language_id"].items()}
        self.dialect = {k.lower(): v for k, v in tc.get("spk_is_dialect", {}).items() if v}
        self.tok = self._tokenizer(model_dir)

    @staticmethod
    def _tokenizer(model_dir: str):
        from transformers import AutoTokenizer  # the checkpoint ships vocab.json / merges.txt / tokenizer_config.json (Qwen2 BPE)

        t = AutoTokenizer.from_pretrained(model_dir)
        return lambda s: list(t(s)["input_ids"])

    def resolve(self, voice: str, language: str) -> tuple[int, int]:
        v = (voice or "").strip().lower()
        if v not in self.spk:
            raise ValueError(f"unsupported speaker {voice!r} (supported: {', '.join(self.speakers)})")
        lang = (language or "auto").strip().lower()
        # like the reference: "auto" or a non-dialect language; under chinese / auto a dialect speaker uses its dialect id
        if lang != "auto" and (lang not in self.langs or "dialect" in lang):
            raise ValueError(f"unsupported language {language!r}")
        lid = -1 if lang == "auto" else self.langs[lang]
        if lang in ("chinese", "auto") and v in self.dialect:
            lid = self.langs[self.dialect[v]]
        return self.spk[v], lid

    def ids(self, text: str, instruct: str) -> tuple[list[int], list[int] | None]:
        ids = self.tok(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
        ins = self.tok(f"<|im_start|>user\n{instruct}<|im_end|>\n") if instruct else None
        return ids, ins


# ---------------------------------------------------------------------------------------------------------------------------------------
# minimal HTTP / WebSocket (RFC 6455) on asyncio streams
class WsClosed(Exception):
    pass


class WebSocket:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.r, self.w = reader, writer
        self.closed = False
        self.send_lock = asyncio.Lock()

    async def _frame(self) -> tuple[bool, int, bytes]:
        h = await self.r.readexactly(2)
        fin, op = bool(h[0] & 0x80), h[0] & 0x0F
        masked, n = bool(h[1] & 0x80), h[1] & 0x7F
        if n == 126:
            n = struct.unpack("!H", await self.r.readexactly(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", await self.r.readexactly(8))[0]
        if n > MAX_MESSAGE:
            raise WsClosed("message too large")
        key = await self.r.readexactly(4) if masked else b""
        data = await self.r.readexactly(n) if n else b""
        if masked:
            data = _unmask(data, key)
        return fin, op, data

    async def recv(self) -> str | bytes:
        """Next text (str) or binary (bytes) message; answers pings; raises WsClosed on close / EOF."""
        parts, kind = [], None
        try:
            while True:
                fin, op, data = await self._frame()
                if op == 0x8:
                    await self._send(0x8, data[:2])
                    raise WsClosed("closed by peer")
                if op == 0x9:
                    await self._send(0xA, data)
                    continue
                if op == 0xA:
                    continue
                if op in (0x1, 0x2):
                    kind, parts = op, [data]
                elif op == 0x0 and kind is not None:
                    parts.append(data)
                else:
                    raise WsClosed(f"bad opcode {op}")
                if sum(len(p) for p in parts) > MAX_MESSAGE:
                    raise WsClosed("message too large")
                if fin:
                    msg = b"".join(parts)
                    return msg.decode("utf-8") if kind == 0x1 else msg
        except (asyncio.IncompleteReadError, ConnectionError, OSError) as e:
            self.closed = True
            raise WsClosed(str(e)) from e

    async def _send(self, op: int, data: bytes) -> None:
        if self.closed:
            raise WsClosed("closed")
        n = len(data)
        if n < 126:
            hdr = struct.pack("!BB", 0x80 | op, n)
        elif n < 65536:
            hdr = struct.pack("!BBH", 0x80 | op, 126, n)
        else:
            hdr = struct.pack("!BBQ", 0x80 | op, 127, n)
        async with self.send_lock:
            try:
                self.w.write(hdr + data)
                await self.w.drain()
            except (ConnectionError, OSError) as e:
                self.closed = True
                raise WsClosed(str(e)) from e

    async def send_text(self, s: str) -> None:
        await self._send(0x1, s.encode("utf-8"))

    async def send_bytes(self, b: bytes) -> None:
        await self._send(0x2, b)

    async def close(self, code: int = 1000) -> None:
        if not self.closed:
            try:
                await self._send(0x8, struct.pack("!H", code))
            except WsClosed:
                pass
        self.closed = True


def _unmask(data: bytes, key: bytes) -> bytes:
    n = len(data)
    k = (key * (n // 4 + 1))[:n]
    return (int.from_bytes(data, "big") ^ int.from_bytes(k, "big")).to_bytes(n, "big")


async def read_request(reader: asyncio.StreamReader) -> tuple[str, str, dict[str, str]] | None:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        return None
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) < 2:
        return None
    hdr = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            hdr[k.strip().lower()] = v.strip()
    return parts[0], parts[1].split("?", 1)[0], hdr


async def http_reply(writer: asyncio.StreamWriter, status: str, body: bytes, ctype: str = "application/json") -> None:
    writer.write(f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
    try:
        await writer.drain()
    finally:
        writer.close()


# ---------------------------------------------------------------------------------------------------------------------------------------
class Server:
    def __init__(self, a):
        self.a = a
        self.engine: Engine | None = None
        self.prompter: Prompter | None = None
        self.ready = False
        self.load_error = ""
        self.loaded_s = None
        self.lock = asyncio.Lock()
        self.worker = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="qtts")
        self.sampling = Sampling(a.temperature, a.top_k, a.temperature, a.top_k, a.rep_penalty, 2, 0)

    # ---- loading
    def _load(self) -> None:
        t0 = time.time()
        self.prompter = Prompter(self.a.model)
        self.engine = Engine(find_lib(self.a.lib), self.a.model, self.a.vram_limit_mb, self.a.max_seq)
        # warm-up: one short sentence through the whole path (graphs are captured inside qtts_create)
        sid, lid = self.prompter.resolve(self.default_voice(), "Korean")
        ids, ins = self.prompter.ids("\uc548\ub155\ud558\uc138\uc694.", "")   # "Hello." in Korean
        cancel = ctypes.c_int32(0)
        self.engine.start(ids, ins, sid, lid, 0, self.sampling)
        while self.engine.next_chunk(cancel) is not None:
            pass
        self.loaded_s = round(time.time() - t0, 1)

    async def load(self) -> None:
        try:
            await asyncio.get_running_loop().run_in_executor(self.worker, self._load)
            self.ready = True
            print(f"[tts] ready in {self.loaded_s}s: {json.dumps({k: self.engine.info[k] for k in ('weight_mb', 'buffer_mb', 'context_mb', 'process_mb', 'max_seq')})}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            self.load_error = repr(e)
            print(f"[tts] load failed: {e!r}", file=sys.stderr, flush=True)
            os._exit(1)  # a server that cannot synthesize should not look alive to a supervisor

    def frame_cap(self, n_text: int) -> int:
        """Frames allowed for n_text text tokens (0 = no cap, the KV cache is the limit). The model sometimes misses EOS on very short
        inputs and keeps babbling until max_new_tokens: the reference qwen_tts (its own defaults, fp32 CPU) did so on 5 of 11 seeds for
        "あ。" (ono_anna, Japanese; 63-99+ frames) and 2 of 30 for the Korean "a." (ah; sohee; 67, 70 frames), where normal takes are 4-14 frames.
        Normal speech used at most 4.2 frames per text token on 11 Korean sentences (both servers, 2026-10-18), so the default
        20 + 8 per token keeps >= 2x headroom for real speech and ends a runaway take of a 2-token input after 2.9 s."""
        if self.a.max_frames_per_token <= 0:
            return 0
        return self.a.max_frames_base + self.a.max_frames_per_token * max(n_text, 0)

    def is_voiced(self, pcm: bytes) -> bool:
        """RMS of a PCM16 chunk above --silence-rms (int16 units)."""
        a = array.array("h")
        a.frombytes(pcm)
        return bool(a) and math.sqrt(math.fsum(x * x for x in a) / len(a)) > self.a.silence_rms

    def default_voice(self) -> str:
        sp = self.prompter.speakers if self.prompter else []
        return "sohee" if "sohee" in [s.lower() for s in sp] else (sp[0] if sp else "")

    def health(self) -> dict:
        inf = self.engine.info if self.engine else {}
        return {"ok": self.ready, "role": "speak", "tts_impl": "hive-qtts", "tts": self.a.model,
                "speakers": list(self.prompter.speakers) if self.ready else [], "loaded_s": self.loaded_s, "device": "cuda:0",
                "sample_rate": SAMPLE_RATE, "chunk_samples": CHUNK_SAMPLES, "max_seq": inf.get("max_seq"),
                "vram_process_mb": round(inf.get("process_mb", 0), 1) if inf else None,
                "vram_limit_mb": round(inf.get("limit_mb", 0), 1) if inf else None, "error": self.load_error or None}

    # ---- connections
    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        req = await read_request(reader)
        if req is None:
            writer.close()
            return
        method, path, hdr = req
        if path == "/health" and method in ("GET", "HEAD"):
            await http_reply(writer, "200 OK", json.dumps(self.health(), ensure_ascii=False).encode())
            return
        if path == "/ws/speak" and hdr.get("upgrade", "").lower() == "websocket" and "sec-websocket-key" in hdr:
            accept = base64.b64encode(hashlib.sha1((hdr["sec-websocket-key"] + WS_GUID).encode()).digest()).decode()
            writer.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                          f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
            ws = WebSocket(reader, writer)
            try:
                await writer.drain()
                await self.speak(ws)
            except WsClosed:
                pass
            finally:
                await ws.close()
                writer.close()
            return
        await http_reply(writer, "404 Not Found", b'{"detail":"Not Found"}')

    async def speak(self, ws: WebSocket) -> None:
        msg = await ws.recv()
        t0 = time.perf_counter()
        try:
            req = json.loads(msg if isinstance(msg, str) else msg.decode())
            text = str(req.get("text") or "").strip()
        except (ValueError, AttributeError):
            await ws.send_text(json.dumps({"type": "error", "message": "bad request"}))
            await ws.send_text(json.dumps({"type": "done", "audio_ms": 0}))
            return
        if not text:
            await ws.send_text(json.dumps({"type": "done", "audio_ms": 0}))
            return
        cancel = ctypes.c_int32(0)

        async def watch() -> None:  # the client only listens after its request: any close / EOF means it left
            try:
                while True:
                    await ws.recv()
            except WsClosed:
                cancel.value = 1

        watcher = asyncio.create_task(watch())
        loop = asyncio.get_running_loop()
        audio_samples, first_ms, err, cap = 0, None, None, 0
        made_samples, stopped, dropped_ms = 0, "", 0
        try:
            if not self.ready:
                raise RuntimeError("model is not loaded" + (f": {self.load_error}" if self.load_error else ""))
            sid, lid = self.prompter.resolve(req.get("voice") or self.default_voice(), req.get("language") or "Korean")
            ids, ins = self.prompter.ids(text, str(req.get("instruct") or ""))
            async with self.lock:
                if cancel.value:
                    return
                seed = self.a.seed if self.a.seed >= 0 else secrets.randbits(63)
                cap = self.frame_cap(len(ids) - 8)
                sampling = Sampling(*[getattr(self.sampling, f) for f, _ in Sampling._fields_[:-1]], cap)
                # per-request sampling (optional): a client that wants an even, calm delivery lowers the temperature for its voice;
                #   out-of-range or non-numeric values keep the server's
                t = req.get("temperature")
                if isinstance(t, (int, float)) and not isinstance(t, bool) and 0.05 <= t <= 2.0:
                    sampling.temperature = sampling.sub_temperature = float(t)
                k = req.get("top_k")
                if isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= 2048:
                    sampling.top_k = sampling.sub_top_k = k
                # the code predictor (codebooks 1-15) alone: a low first-codebook temperature evens the delivery, but the same low
                #   temperature on the code predictor made comma lists lose their place — measured 2026-10-19 (a voice client, an added
                #   Korean voice, instruct + 0.4 on both): a comma list of the Korean numbers eleven … twenty (10 items) spoke 2/10 and 8/10 in 2 of 3 takes and filled the rest with silence to the
                #   length cap; the reference keeps both stages at 0.9 (generation_config.json subtalker_temperature)
                t = req.get("sub_temperature")
                if isinstance(t, (int, float)) and not isinstance(t, bool) and 0.05 <= t <= 2.0:
                    sampling.sub_temperature = float(t)
                k = req.get("sub_top_k")
                if isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= 2048:
                    sampling.sub_top_k = k
                if isinstance(req.get("seed"), int) and not isinstance(req.get("seed"), bool) and req["seed"] >= 0:
                    seed = req["seed"]
                await loop.run_in_executor(self.worker, self.engine.start, ids, ins, sid, lid, seed, sampling)
                fut = loop.run_in_executor(self.worker, self.engine.next_chunk, cancel)
                # Silence (2026-10-19, a voice client): when the model loses its place or misses EOS it keeps emitting silent frames until the
                #   length cap — 21.5 s pieces, a 66 s answer for "count to 30". Natural pauses are <= 0.9 s in sentences and <= 1.0 s in
                #   calm comma lists, so silent chunks are held: a pause sends at most silence_keep_ms of them, and silence_stop_ms of
                #   silence (or that long before any speech) ends the take.
                keep_chunks = max(0, self.a.silence_keep_ms * SAMPLE_RATE // 1000 // CHUNK_SAMPLES)
                held: list[bytes] = []
                quiet_samples, spoke = 0, False
                while True:
                    pcm = await fut
                    if pcm is None:
                        break
                    fut = loop.run_in_executor(self.worker, self.engine.next_chunk, cancel)  # the next chunk runs while this one is sent
                    if first_ms is None:
                        first_ms = (time.perf_counter() - t0) * 1000
                    made_samples += len(pcm) // 2
                    out: list[bytes] = []
                    if self.a.silence_stop_ms <= 0 or self.is_voiced(pcm):
                        out, held, quiet_samples, spoke = held + [pcm], [], 0, True
                    else:
                        quiet_samples += len(pcm) // 2
                        if len(held) < keep_chunks:
                            held.append(pcm)
                        else:
                            dropped_ms += round(len(pcm) / 2 / SAMPLE_RATE * 1000)
                        if quiet_samples * 1000 >= self.a.silence_stop_ms * SAMPLE_RATE:
                            stopped = "silence"
                    try:
                        for c in out:
                            await ws.send_bytes(c)
                            audio_samples += len(c) // 2
                    except WsClosed:
                        cancel.value = 1
                        try:
                            await fut  # stops before its next frame
                        except Exception:  # noqa: BLE001
                            pass
                        return
                    if stopped:
                        cancel.value = 1
                        try:
                            await fut
                        except Exception:  # noqa: BLE001
                            pass
                        cancel.value = 0  # the engine stopped for us, not because the client left: still answer "done"
                        held = held if spoke else []
                        break
                for c in held:  # a pause at the end (natural, or the start of a stopped run) — at most silence_keep_ms
                    await ws.send_bytes(c)
                    audio_samples += len(c) // 2
        except WsClosed:
            raise
        except Exception as e:  # noqa: BLE001
            err = e
        finally:
            watcher.cancel()
        if cancel.value:
            return
        if stopped:
            print(f"[tts] silence stop after {round(made_samples / SAMPLE_RATE, 1)} s generated ({text[:40]!r})", flush=True)
        if err is None and cap and made_samples >= cap * 1920:
            print(f"[tts] length cap: {cap} frames for {len(ids) - 8} text tokens ({text[:40]!r})", flush=True)
        if err is not None:
            await ws.send_text(json.dumps({"type": "error", "message": str(err) or repr(err)}, ensure_ascii=False))
        done = {"type": "done", "audio_ms": round(audio_samples / SAMPLE_RATE * 1000), "first_ms": round(first_ms or 0),
                "total_ms": round((time.perf_counter() - t0) * 1000)}
        if stopped:
            done["stopped"] = stopped
        if dropped_ms:
            done["dropped_ms"] = dropped_ms
        await ws.send_text(json.dumps(done))


async def amain(a) -> None:
    srv = Server(a)
    server = await asyncio.start_server(srv.handle, a.host, a.port, limit=1 << 16)
    print(f"[tts] listening on {a.host}:{a.port}", flush=True)
    asyncio.create_task(srv.load())
    async with server:
        await server.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description="Eke Hive streaming TTS server (Qwen3-TTS 12Hz CustomVoice)")
    ap.add_argument("--model", required=True, help="checkpoint directory (config.json, model.safetensors, speech_tokenizer/)")
    ap.add_argument("--lib", default=None, help="path of libhive_qtts.so (default: HIVE_QTTS_LIB or engine/build*/)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18440)
    ap.add_argument("--vram-limit-mb", type=int, default=0, help="0 = HIVE_VRAM_LIMIT_MB or no limit")
    ap.add_argument("--max-seq", type=int, default=0, help="talker positions (prompt + frames); 0 = 2048")
    ap.add_argument("--seed", type=int, default=-1, help=">= 0: fixed sampling seed for every request (tests)")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--rep-penalty", type=float, default=1.05)
    ap.add_argument("--max-frames-base", type=int, default=20, help="length cap: frames allowed beyond the per-token budget")
    ap.add_argument("--max-frames-per-token", type=int, default=8, help="length cap: frames (80 ms) per text token; 0 = no cap")
    ap.add_argument("--silence-rms", type=float, default=328.0, help="chunk RMS (int16) at or below which a chunk counts as silent")
    ap.add_argument("--silence-keep-ms", type=int, default=960, help="longest pause sent; silence beyond it is dropped")
    ap.add_argument("--silence-stop-ms", type=int, default=3200, help="silence that ends the take; 0 = off (send every chunk)")
    a = ap.parse_args()
    try:
        asyncio.run(amain(a))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
