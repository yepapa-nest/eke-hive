#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""hive API server — OpenAI (/v1/chat/completions) and Anthropic (/v1/messages) compatible. Token ids are handed to the daemon
(hived or hived_glm) over a Unix socket.
  DeepSeek-V4.1-Flash: tokenization, the chat template, DSML tool-call parsing and image preprocessing reuse the reference
    encoding/ and inference/image_processor shipped with the checkpoint, unchanged.
  GLM-5.3-Flash: the checkpoint's chat_template.jinja through transformers, plus server/families/glm.py (tool-call parsing,
    image and video preprocessing).

Run: python3 hive_server.py --ckpt /path/to/checkpoint --sock /tmp/hive.sock --port 8430
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing, suppress
import hashlib
import hmac
import json
import logging
import math
import os
import re
import socket
import sys
import time
import threading
import uuid
from array import array
from collections import OrderedDict
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

# ---- public exposure ---------------------------------------------------------------------------
# The server has no accounts: whoever can reach the port can run inference and call the control endpoints (sleep/wake/flush).
#   The entrypoint binds 127.0.0.1 unless HIVE_BIND says otherwise; HIVE_API_KEY (optional) requires `Authorization: Bearer <key>` or
#   `x-api-key: <key>` on every endpoint except /health; HIVE_MAX_BODY_MB (default 64) rejects larger request bodies before parsing;
#   HIVE_IMAGE_FETCH=0 limits image inputs to data: URLs (the reference image loader would otherwise fetch http(s) URLs from the server).
API_KEY = os.environ.get("HIVE_API_KEY") or None
MAX_BODY_BYTES = int(float(os.environ.get("HIVE_MAX_BODY_MB") or 64) * 1024 * 1024)
IMAGE_FETCH = os.environ.get("HIVE_IMAGE_FETCH", "1") not in ("", "0")


def _err(message: str, kind: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": kind}}


@app.middleware("http")
async def api_key_guard(request, call_next):
    if API_KEY and request.url.path != "/health":
        h = request.headers
        auth = h.get("authorization", "")
        key = auth[7:] if auth.lower().startswith("bearer ") else h.get("x-api-key", "")
        if not hmac.compare_digest(key.encode(), API_KEY.encode()):
            return JSONResponse(_err("invalid API key", "authentication_error"), status_code=401)
    return await call_next(request)


async def read_body(request):
    """The request's JSON object, or (None, error response): 413 above HIVE_MAX_BODY_MB, 400 for invalid JSON or a non-object body."""
    try:
        cl = int((getattr(request, "headers", None) or {}).get("content-length") or 0)
    except (TypeError, ValueError):
        cl = 0
    if cl > MAX_BODY_BYTES:
        return None, JSONResponse(_err(f"request body larger than {MAX_BODY_BYTES // (1024 * 1024)} MiB"), status_code=413)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — malformed JSON is a client error
        return None, JSONResponse(_err("invalid JSON body"), status_code=400)
    if not isinstance(body, dict):
        return None, JSONResponse(_err("request body must be a JSON object"), status_code=400)
    return body, None


def _check_image_url(u) -> None:
    """Image inputs are data: URLs (or http(s) URLs when HIVE_IMAGE_FETCH is on). Anything else — in particular a filesystem path, which the
    reference image loader would open as a local file — is a request error."""
    ok = isinstance(u, str) and (u.startswith("data:") or (IMAGE_FETCH and u.startswith(("http://", "https://"))))
    if not ok:
        raise ValueError("image_url must be a data: URL" + (" or an http(s) URL" if IMAGE_FETCH else ""))


def check_image_parts(messages) -> None:
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("type")
            if t == "image_url":
                u = part.get("image_url")
                _check_image_url(u if isinstance(u, str) else (u or {}).get("url", "") if isinstance(u, dict) else "")
            elif t == "image":
                src = part.get("source")
                if isinstance(src, dict) and src.get("data") is None:
                    _check_image_url(src.get("url", ""))
CFG: dict = {}
TOK = None
ENC = None  # reference encoding module
IMG = None  # reference image_processor module
ARGS = None
ENCODE_LOCK = threading.Lock()


class ClosingStreamingResponse(StreamingResponse):
    """Release reservations even when ASGI disconnects before the first iteration."""
    def __init__(self, *args, on_close=None, **kwargs):
        self.on_close = on_close
        super().__init__(*args, **kwargs)

    async def close(self):
        try:
            await self.body_iterator.aclose()
        finally:
            if self.on_close: self.on_close()

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.close()


# ----------------------------------------------------------------------------------------------
FAMILY = "deepseek_v4"   # model family of the checkpoint (config.json model_type); "glm5_next" uses server/families/glm.py
STOP_IDS: list = []      # stop token ids (DeepSeek: the tokenizer's eos; GLM: config eos_token_id list)
EOS_TEXTS: list = []     # their text forms (stripped from completions)


def load_modules(ckpt: str):
    global TOK, ENC, IMG, FAMILY, STOP_IDS, EOS_TEXTS, DSML_START
    try:
        with open(os.path.join(ckpt, "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    if cfg.get("model_type") == "glm5_next":
        from transformers import AutoTokenizer

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from families import glm as fam  # noqa

        TOK = AutoTokenizer.from_pretrained(ckpt)
        fam.init(TOK, cfg)
        ENC, FAMILY, STOP_IDS, EOS_TEXTS, DSML_START = fam, fam.FAMILY, list(fam.STOP_IDS), list(fam.EOS_TEXTS), fam.TOOL_START
        print(f"[server] family {FAMILY}: stop ids {STOP_IDS} · non-thinking requests -> {fam.nothink_mode()}")
        return
    sys.path.insert(0, os.path.join(ckpt, "encoding"))
    sys.path.insert(0, os.path.join(ckpt, "inference"))
    import encoding as enc  # noqa

    ENC = enc
    try:
        import image_processor as ip  # noqa

        IMG = ip
    except Exception as e:  # e.g. no PIL: text-only
        print("[server] image_processor unavailable:", e)
    from transformers import AutoTokenizer

    TOK = AutoTokenizer.from_pretrained(ckpt)
    STOP_IDS, EOS_TEXTS = [TOK.eos_token_id], [TOK.eos_token]


def eos_texts() -> list:
    """Stop-token texts to strip; before load_modules ran (tests that install a tokenizer directly) the tokenizer's own eos."""
    return EOS_TEXTS or [TOK.eos_token]


def stop_ids() -> list:
    return list(STOP_IDS) or [TOK.eos_token_id]


def strip_eos(text: str) -> str:
    for e in eos_texts():
        text = text.replace(e, "")
    return text


class VisionArgs:
    """The args fields image_processor expects (the vision_* names of the reference ModelArgs)."""

    def __init__(self, cfg: dict):
        v = cfg.get("vision_config", {})
        self.vision_patch_size = v.get("patch_size", 14)
        self.vision_downsample_ratio = v.get("downsample_ratio", 3)
        self.vision_max_n_token = v.get("max_image_tokens", 1024)
        self.vision_min_pixels = v.get("min_pixels", 295936)
        self.vision_max_wh_ratio = v.get("max_wh_ratio")
        self.image_token_id = cfg.get("image_token_id", 129264)
        self.vision_enabled = True


# ----------------------------------------------------------------------------------------------
def effort_from_request(body: dict) -> tuple[str, Any]:
    """Extract thinking_mode and reasoning_effort from OpenAI/Anthropic fields. Default = thinking, 75 (DeepSeek's official default "high")."""
    if FAMILY == "glm5_next":
        return ENC.effort_from_request(body)
    kw = body.get("chat_template_kwargs")
    kw = kw if isinstance(kw, dict) else {}
    thinking = kw.get("thinking", kw.get("enable_thinking"))
    effort = kw.get("reasoning_effort", body.get("reasoning_effort"))
    if isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort", effort)
    # Effort levels follow the official mapping: DeepSeek_V41_Tech_Report.pdf §5.3.2-5.3.3 and appendix B.3, and
    #   encoding.py (REASONING_EFFORT_MAPPINGS, DEFAULT "high"). Public API levels low/high/max = 50/75/100, default high (75).
    #   Per the report, "60-80 recovers most of max accuracy with less than half of max's tokens"; 25->100 raises the
    #   8-benchmark reasoning average 67.1->76.3% at ~2.5x output tokens, and 100 costs 1.6-1.8x trajectory length for
    #   marginal gain -> default 75. Client level names low/medium/xhigh map to 50/75/100.
    if effort == "none":
        thinking = False
        effort = None
    effort = {"minimal": 25, "low": 50, "medium": 75, "high": 75, "xhigh": 100, "max": 100}.get(effort, effort)
    # Normalize at the entrance: numeric strings, floats and out-of-range values become an integer in 1..100, unknown strings
    # fall back to the default (the reference encoding asserts on anything else).
    if isinstance(effort, bool):  # bool is a subclass of int — do not let True/False leak through as 1/0
        effort = None
    if effort is not None and not isinstance(effort, int):
        try:
            effort = int(float(effort))
        except (TypeError, ValueError):
            effort = None
    if isinstance(effort, int):
        effort = min(100, max(1, effort))
    if thinking is None:
        thinking = True
    return ("thinking" if thinking else "chat"), (effort if effort is not None else (75 if thinking else None))


# Thinking cap (safety valve, off by default): if the request carries max_thinking_tokens (the name used by SGLang patches)
#   or thinking_token_budget (the vLLM-family name) — top level or inside chat_template_kwargs — then in thinking mode hived
#   forces </think> as soon as the thinking tokens reach N (engine side: the same token flow as streaming, MTP drafts and
#   batched decode — hived.cpp Active::think_cap). Without the field the daemon request is unchanged.
#   Normalization (never rejects): only positive numbers (int, float, numeric string); 0, negatives, bool and non-numbers
#   count as absent. If several names are present, the first valid one in the order top-level max_thinking_tokens ->
#   top-level thinking_token_budget -> kwargs max_thinking_tokens -> kwargs thinking_token_budget wins.
#   WARNING: the model was not trained on truncated thinking — enable it only after checking quality.
THINK_CAP_KEYS = ("max_thinking_tokens", "thinking_token_budget")


def _num(v, default):
    """Normalize a numeric sampling field: a number (int/float, not bool) or numeric string yields its value; anything else (missing, null, non-numeric, NaN, inf) yields default."""
    if v is None or isinstance(v, bool):
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x != x or x in (float("inf"), float("-inf")):
        return default
    return v if isinstance(v, (int, float)) else x


def think_cap_from_request(body: dict):
    kw = body.get("chat_template_kwargs") if isinstance(body.get("chat_template_kwargs"), dict) else {}
    for src in (body, kw):
        for k in THINK_CAP_KEYS:
            v = src.get(k)
            if isinstance(v, bool) or v is None:
                continue
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            if x == x and 1 <= x < 1e9:
                return int(x)
    return None


_THINK_IDS = None


def think_token_ids():
    """(<think> id, </think> id) — only when the tokenizer has each as a single token (DS V4.1: 128821, 128822). Otherwise None, which disables the cap (absorbed, not rejected)."""
    global _THINK_IDS
    if _THINK_IDS is None or _THINK_IDS[0] is not TOK:
        ids = []
        for tok in ("<think>", "</think>"):
            i = None
            try:
                c = TOK.convert_tokens_to_ids(tok)
                if isinstance(c, int) and c >= 0 and TOK.decode([c]) == tok:
                    i = c
            except Exception:  # noqa: BLE001 — tokenizers differ: if the id cannot be obtained, disable
                i = None
            ids.append(i)
        _THINK_IDS = (TOK, ids[0], ids[1])
    return _THINK_IDS[1], _THINK_IDS[2]


# Thinking-cap exit phrase: when the cap is reached, hived forces this phrase's tokens before </think> (hived.cpp Active::think_exit).
#   Rationale (measured on vLLM): forcing only </think> left 3.7% (6/162) of answers continuing to think inside the
#   answer, exposing the reasoning; adding a transition phrase ("I have enough reasoning. I will now write the final
#   answer...") brought it to 0/162. This phrase replaces "final answer" with "response" because in agent loops the next
#   output may be a tool call rather than an answer.
#   HIVE_THINK_EXIT_TEXT: unset = the default below; empty = no phrase (only </think>).
THINK_EXIT_DEFAULT = "\n\nI have enough reasoning. I will now stop thinking and write the response.\n"
_THINK_EXIT = None


def think_exit_ids() -> list[int]:
    global _THINK_EXIT
    text = os.environ.get("HIVE_THINK_EXIT_TEXT", THINK_EXIT_DEFAULT)
    if _THINK_EXIT is None or _THINK_EXIT[0] is not TOK or _THINK_EXIT[1] != text:
        ids: list[int] = []
        if text:
            try:
                ids = [int(i) for i in TOK.encode(text, add_special_tokens=False)]
            except Exception:  # noqa: BLE001 — tokenization failure: no phrase (only </think>)
                ids = []
        _THINK_EXIT = (TOK, text, ids)
    return _THINK_EXIT[2]


def encode_prompt(messages: list[dict], tools, thinking_mode: str, effort) -> tuple[str, list[dict]]:
    """Reference encoding.encode_messages: the tool list is attached to the first message as `tools`. Images come back in media["images"]."""
    check_image_parts(messages)
    messages = [dict(m) for m in messages]
    if tools and messages:
        # The reference encoding only calls render_tools for role=system — requests that start with a user message (e.g.
        #   BFCL function calling) got no tool list in the prompt at all (measured: weather question + get_weather, 0/4
        #   calls). Prepend an empty system message and attach the tools there.
        if messages[0].get("role") != "system":
            messages[:0] = [{"role": "system", "content": ""}]
        messages[0]["tools"] = tools
    kwargs = {"thinking_mode": thinking_mode, "return_multi_modal_data": True}
    if effort is not None and thinking_mode == "thinking":
        kwargs["reasoning_effort"] = effort
    prompt, media = ENC.encode_messages(messages, **kwargs)
    images = (media or {}).get("images") or []
    return prompt, images


def tokenize_with_images(prompt: str, media: list[dict]):
    """Token ids plus, when images are present, patches and token types. Returns (ids, images_meta, bin_bytes)."""
    if not media:
        return TOK.encode(prompt), [], b""
    if FAMILY == "glm5_next":  # GLM: the family adapter preprocesses images and video (server/families/glm.py)
        return ENC.prepare_inputs(prompt, media, TOK)
    if IMG is None:
        raise ValueError("image preprocessing module is not available")
    import numpy as np

    tokens, token_types, image_inputs = IMG.prepare_vl_inputs(prompt, media, TOK, ARGS)
    metas, blobs = [], []
    for im in image_inputs or []:
        patches = im.patches.to(__import__("torch").bfloat16).contiguous()
        raw = patches.view(__import__("torch").int16).numpy().tobytes()
        metas.append({"start": im.start, "n_vit_h": im.n_vit_h, "n_vit_w": im.n_vit_w, "types": im.types.tolist(), "nbytes": len(raw)})
        blobs.append(raw)
    return tokens, metas, b"".join(blobs)


def env_on(name: str) -> bool:
    """Project convention: unset / "" / "0" = off (same as hived env_on)."""
    v = os.environ.get(name)
    return bool(v) and v != "0"


def _common_prefix_len(a: str, b: str) -> int:
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:  # binary search over slice comparisons (C speed) — O(n log n), no per-character Python loop
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class TailChangeTracker:
    """HIVE_PREFIX_ADAPTIVE: spend an extra prefill chunk on a boundary snapshot only for conversations whose tail changes.

    A conversation is append-only when each request starts with the whole previous request — the daemon then resumes from the
    session's own state and boundary snapshots are wasted work (that is why HIVE_PREFIX_EXTRA_CHUNKS=0 sped up follow-up turns).
    Some clients instead replace a block near the end on every turn (for example fresh context placed just before the newest user
    message). Then the previous request is not a prefix of the new one, the daemon finds no saved state at the last stable boundary
    and prefills the whole prompt again. This tracker remembers, per conversation, the previous request's length, a hash of all its
    tokens and a hash of its prefix at each boundary hint. A request is flagged when the previous one was not a prefix of it but did
    share one of its boundaries — exactly the case where a snapshot at that boundary would have been resumed. The flag asks the daemon
    for one extra chunk ("prefix_extra": 1) so this request's deepest boundary is saved for the next turn. Append-only conversations
    are never flagged, so they keep the global budget.
    Only hashes are kept (no token copies); entries are evicted least-recently-used beyond `cap` conversations."""

    def __init__(self, cap: int = 4096):
        self.cap = cap
        self.entries: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def _hash(view: memoryview, n: int) -> bytes:
        return hashlib.blake2b(view[:n], digest_size=16).digest()

    def observe(self, key: str, ids: list[int], hints: list[int]) -> bool:
        """Record this request (call off the event loop: hashing touches every token). True = ask for an extra chunk."""
        view = memoryview(array("i", ids)).cast("B")
        w = array("i").itemsize
        with self.lock:
            prev = self.entries.get(key)
        flag = False
        if prev is not None:
            plen, phash, phints = prev
            append = len(ids) >= plen and self._hash(view, plen * w) == phash
            if not append:
                flag = any(h <= len(ids) and self._hash(view, h * w) == hh for h, hh in phints)
        entry = (len(ids), self._hash(view, len(ids) * w), tuple((h, self._hash(view, h * w)) for h in hints if 0 < h <= len(ids)))
        with self.lock:
            self.entries[key] = entry
            self.entries.move_to_end(key)
            while len(self.entries) > self.cap:
                self.entries.popitem(last=False)
        return flag


TAILS = TailChangeTracker()


HINT_TURNS = 4  # only the last few completed assistant-turn ends (the daemon only uses boundaries inside the newly prefilled span — earlier ones were recorded on earlier turns)


def boundary_hints(messages: list[dict], tools, thinking_mode: str, effort, prompt: str, ids: list[int]) -> list[int]:
    """H5 (HIVE_PREFIX_SHARE): **exact token offsets** in the encoded prompt — end of the system/tool block and ends of
    completed assistant turns.
    Makes no assumption about the reference template's internals: for each boundary, render "the messages before it + an
    empty user message" with the same encode_prompt, measure the common-prefix length c against the full prompt string
    (past that point the next message's content / generation prompt diverges), tokenize the full prompt once more with an
    offset mapping, and emit the number of tokens that end at or before c. If the offset tokenization's ids differ from
    the ids actually sent, emit nothing.
    Results are always prefix positions of ids (a wrong boundary only loses a reuse opportunity — the daemon compares the
    token prefix itself). Image requests emit nothing, since text offsets do not match the placeholder expansion.
    Cost: one extra full-prompt tokenization + one template render per boundary (<= 1 + HINT_TURNS) + string compares —
    only when enabled."""
    try:
        enc = TOK(prompt, return_offsets_mapping=True)
        if list(enc["input_ids"]) != list(ids):
            return []
        ends = [int(e) for _, e in enc["offset_mapping"]]
    except Exception:
        return []
    if any(ends[i] > ends[i + 1] for i in range(len(ends) - 1)):
        return []  # non-monotonic mapping (unknown tokenizer) — no boundaries
    import bisect
    probes = []
    n = 0
    while n < len(messages) and messages[n].get("role") in ("system", "developer"):
        n += 1
    if n or tools:
        probes.append(messages[:n])
    turn_ends = [k for k in range(len(messages) - 1) if messages[k].get("role") == "assistant"]
    for k in turn_ends[-HINT_TURNS:]:
        probes.append(messages[: k + 1])
    out = set()
    for pre in probes:
        try:
            text, media = encode_prompt(list(pre) + [{"role": "user", "content": ""}], tools, thinking_mode, effort)
        except Exception:
            continue
        if media:
            continue
        c = _common_prefix_len(text, prompt)
        t = bisect.bisect_right(ends, c)  # number of leading tokens ending at or before c (a prefix, since offsets are monotonic)
        if 0 < t < len(ids):
            out.add(t)
    return sorted(out)


# ----------------------------------------------------------------------------------------------
class Daemon:
    def __init__(self, sock_path: str):
        self.sock_path = sock_path

    async def generate(self, session: str, ids: list[int], images: list[dict], blob: bytes, rid: str = "", boundaries=None, **sampling):
        """Send a request to hived and yield token ids asynchronously; the last item is a done dict."""
        reader, writer = await asyncio.open_unix_connection(self.sock_path)
        req = {"op": "generate", "session": session, "ids": ids, "images": images, "bin": len(blob), **sampling}
        if rid: req["rid"] = rid  # D5: so a cancel reaches only this request (hived rid_flags)
        if boundaries: req["boundaries"] = boundaries  # H5: only when enabled (a hived with it off ignores the unknown field)
        try:
            writer.write((json.dumps(req) + "\n").encode())
            if blob:
                writer.write(blob)
            await writer.drain()
            while True:
                line = await reader.readline()
                if not line:
                    # An audit reproduced truncated content reported as a successful
                    # length completion. EOF is not a terminal message.
                    yield {"error": "daemon connection ended before completion"}
                    break
                msg = json.loads(line)
                yield msg
                if msg.get("done") or msg.get("error"):
                    break
        finally:
            writer.close()
            await writer.wait_closed()

    async def stats(self):
        reader, writer = await asyncio.open_unix_connection(self.sock_path)
        try:
            writer.write(b'{"op":"stats"}\n')
            await writer.drain()
            return json.loads(await reader.readline())
        finally:
            writer.close()
            await writer.wait_closed()

    async def control(self, op: str, **kw):
        """Z1 sleep/wake — hived answers with one line once done (for sleep: after in-flight requests finish and VRAM is released). No deadline here (the caller sets one)."""
        reader, writer = await asyncio.open_unix_connection(self.sock_path)
        try:
            writer.write((json.dumps({"op": op, **kw}) + "\n").encode())
            await writer.drain()
            line = await reader.readline()
            if not line:
                return {"error": "daemon connection ended before the reply"}
            return json.loads(line)
        finally:
            writer.close()
            await writer.wait_closed()

    def cancel(self, session: str, rid: str = ""):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)  # control connection only, never a generation deadline
                s.connect(self.sock_path)
                s.sendall((json.dumps({"op": "cancel", "session": session, **({"rid": rid} if rid else {})}) + "\n").encode())
                # D5: wait for the ack ({"ok":true}) — hived replies after setting the flag (hived.cpp op=="cancel"). Returning
                #   right after sendall would let the caller release its reservation and send a new generate for the same
                #   session, which the acceptor might process first, so the late cancel could hit the new request.
                buf = b""
                while b"\n" not in buf:
                    chunk = s.recv(256)
                    if not chunk: break
                    buf += chunk
        except Exception:
            pass


DAEMON: Daemon | None = None
# Request log (read by tools/hive_monitor.py): one JSONL line per request — HIVE_REQUEST_LOG (path; "" = off; unset =
#   /out/logs/requests-server.jsonl if /out/logs exists). Times are epoch ms: t_recv (HTTP received), t_first (first daemon
#   token), t_done (finished). Write failures are absorbed (no effect on the request).
_REQ_LOG_LOCK = threading.Lock()


def _req_log_path():
    p = os.environ.get("HIVE_REQUEST_LOG")
    if p is not None:
        return p or None
    return "/out/logs/requests-server.jsonl" if os.path.isdir("/out/logs") else None


def log_request(rec: dict):
    path = _req_log_path()
    if not path:
        return
    try:
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        with _REQ_LOG_LOCK:
            with open(path, mode="a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass


# Context-window overflow = HTTP 400 in OpenAI format (context_length_exceeded). A 5xx makes gateways retry and put the
#   backend on cooldown, while agent clients typically recover (compact and retry) only on a 400 with the overflow wording.
#   The wording follows vLLM so that common client matchers ("maximum context length", "prompt contains at least (\d+)
#   input tokens") both fire. hived clamps max_tokens to the remaining context, so an overflow is only prompt + 1 first token.
_MAX_CTX: int | None = None


def openai_usage(prompt_tokens: int, done: dict) -> dict:
    """OpenAI usage with the reused prompt prefix reported as cached tokens (prompt_tokens_details.cached_tokens — the field OpenAI and
    LiteLLM read for prompt-cache hits). cached = the daemon's cached_prefix (tokens taken from the session state or a shared boundary
    snapshot instead of being prefilled), never more than the prompt."""
    cached = max(0, min(int(done.get("cached_prefix") or 0), prompt_tokens))
    n = done.get("n", 0)
    return {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n, "prompt_tokens_details": {"cached_tokens": cached}}


def anthropic_usage(u: dict) -> dict:
    """Anthropic usage from an OpenAI usage: input_tokens counts only the prompt tokens that were processed, cache_read_input_tokens the
    reused prefix (Anthropic's split — input_tokens + cache_read_input_tokens + cache_creation_input_tokens = whole prompt). The engine
    keeps prefixes without a separate write step, so cache_creation_input_tokens is 0."""
    cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    return {"input_tokens": int(u.get("prompt_tokens", 0)) - cached, "output_tokens": int(u.get("completion_tokens", 0)),
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}


def overflow_body(max_ctx: int, prompt_tokens: int) -> dict:
    total = prompt_tokens + 1
    return {"error": {"message": f"This model's maximum context length is {max_ctx} tokens. However, you requested {total} tokens "
                                 f"({prompt_tokens} in the messages, 1 for the completion): your prompt contains at least {prompt_tokens} input tokens, "
                                 f"for a total of at least {total} tokens. Please reduce the length of the messages.",
                      "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}}


async def daemon_max_ctx() -> int | None:
    """Daemon max_ctx (from stats) — cached after the first read (a restarted daemon normally has the same setting, and if it is wrong the daemon's own overflow reason is mapped as a fallback). None if unreadable (no pre-check)."""
    global _MAX_CTX
    if _MAX_CTX is None:
        try:
            v = (await asyncio.wait_for(DAEMON.stats(), timeout=2.0)).get("max_ctx")
            if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                _MAX_CTX = v
        except Exception:
            pass
    return _MAX_CTX


CLIENT_TAG_PREFIXES_DEFAULT = ("x-hive", "x-client")
CLIENT_TAG_DENY = ("authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key", "api-key")  # never logged, whatever the prefixes


def client_tag_prefixes() -> tuple[str, ...]:
    """Header-name prefixes recorded as client tags: HIVE_CLIENT_TAG_PREFIXES (comma-separated, case-insensitive;
    unset or empty = CLIENT_TAG_PREFIXES_DEFAULT). Add a gateway's own header prefix here to record its tags."""
    v = os.environ.get("HIVE_CLIENT_TAG_PREFIXES", "")
    p = tuple(x.strip().lower() for x in v.split(",") if x.strip())
    return p or CLIENT_TAG_PREFIXES_DEFAULT


def client_tags(request, body: dict) -> dict:
    """Client tags: headers whose name starts with a client_tag_prefixes() prefix, user-agent, the body's user field and metadata (e.g. a gateway's model alias)."""
    tags = {}
    h = getattr(request, "headers", None)
    try:
        prefixes = client_tag_prefixes()
        for k, v in (h.items() if h is not None else []):
            kl = k.lower()
            if kl in CLIENT_TAG_DENY or any(w in kl for w in ("key", "token", "secret", "auth", "cookie")):
                continue  # credential-shaped names are never written to the request log
            if kl.startswith(prefixes) or kl == "user-agent":
                tags[kl] = str(v)[:200]
    except Exception:
        pass
    if body.get("user"):
        tags["user"] = str(body["user"])[:200]
    md = body.get("metadata")
    if isinstance(md, dict):
        tags["metadata"] = {str(k)[:64]: str(v)[:200] for k, v in list(md.items())[:16]}
    return tags
INFLIGHT: dict[str, int] = {}  # session id -> in-flight request count (the daemon rejects concurrent requests on one session with "session busy", so the server splits them)


def session_id_for(messages: list[dict], explicit: str | None) -> str:
    # Conversation cache: the session is keyed on messages up to and including the first user message (with any leading
    #   system/developer messages), so later turns of the same conversation reuse the prefix. (Hashing a fixed messages[:2]
    #   would key turn 2 of a system-less conversation on [u1, a1], a different session from turn 1, forcing a full re-prefill.)
    #   The `user` field may be an account id (several conversations of one user would overwrite one session), so it is
    #   never the sole key — it is mixed with the head hash.
    n = 0
    while n < len(messages) and messages[n].get("role") != "user":
        n += 1
    head = json.dumps(messages[: min(n + 1, len(messages))], ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(((explicit or "") + "\x00" + head).encode()).hexdigest()[:16]


def parse_completion(text: str, thinking_mode: str, tools):
    """The family parser. GLM also takes the request's tools — its tool-call format has no type marker, so argument types come from
    the schemas (server/families/glm.py _arg_value); DeepSeek's DSML marks string arguments itself (string="true")."""
    if FAMILY == "glm5_next":
        return ENC.parse_message_from_completion_text(text, thinking_mode=thinking_mode, tools=tools)
    return ENC.parse_message_from_completion_text(text, thinking_mode=thinking_mode)


def _reject_constant(name: str):
    raise ValueError(name)


def _finite_json(x) -> bool:
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, list):
        return all(_finite_json(y) for y in x)
    if isinstance(x, dict):
        return all(_finite_json(y) for y in x.values())
    return True


def invalid_tool_args(calls) -> list:
    """Names of the tool calls whose arguments the client cannot read as a JSON object (not JSON, NaN / Infinity, non-finite).
    Recorded only (request log field tool_args_invalid), the calls are sent unchanged. DeepSeek copies a value marked
    string="false" into the arguments as written, so a string the model marks false would land here; none was found in 23,029
    production tool calls (2026-09-28..10-05) — this field shows whether it ever happens before a schema correction is added."""
    bad = []
    for name, args in calls:
        try:
            x = json.loads(args, parse_constant=_reject_constant) if isinstance(args, str) else None
        except ValueError:
            x = None
        if not isinstance(x, dict) or not _finite_json(x):
            bad.append(name)
    return bad


def openai_tool_calls(calls) -> list[dict]:
    """tool_calls from the reference parse_message_from_completion_text are already in OpenAI format ({type, function:{name,
    arguments}}) — converting them again raises KeyError 'name'. Here we only add an id and ensure arguments is a string."""
    out = []
    for i, c in enumerate(calls or []):
        fn = c.get("function") or {"name": c.get("name"), "arguments": c.get("arguments")}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args or {}, ensure_ascii=False)
        call_id = c.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        out.append({"id": call_id, "type": "function", "function": dict(name=fn.get("name"), arguments=args)})
    return out


class DsmlToolStream:
    """Stream DSML tool calls as OpenAI tool_call deltas while they are generated (HIVE_TOOL_STREAM, default on; "0" = off).

    Without it the server collected the whole tool block and sent the parsed calls at the end, so the client received no bytes while
    a long call (for example a file body as an argument) was generated — at ~10 tok/s per stream under concurrency, minutes of silence,
    long enough to trip a client's stream-gap timeout (measured: three requests closed after 300-409 s of a 300 s gap timeout).

    The arguments string is built exactly as the reference parser builds it (encoding.decode_dsml_to_arguments): "{" + ", ".join(
    json(key) + ": " + (json(value) if string="true" else value)) + "}". json() of a string escapes each character independently, so
    escaping the value piece by piece and concatenating gives the same bytes. The block grammar follows encoding.parse_tool_calls
    (">\n" between tags, the value ends at the first "</｜DSML｜ parameter"). Any deviation stops streaming (`broken`); the caller then
    falls back to the reference parse at the end — calls already streamed are kept, and the final comparison is recorded."""

    def __init__(self, enc):
        dsml = getattr(enc, "dsml_token", "｜DSML｜")
        blk = getattr(enc, "tool_calls_block_name", " calls")
        inv = getattr(enc, "tool_call_tag_name", " invoke")
        par = getattr(enc, "tool_parameter_tag_name", " parameter")
        self.split = getattr(enc, "_split_tool_name", None)
        self.CALLS = f"<{dsml}{blk}"
        self.CALLS_END = f"</{dsml}{blk}>"
        self.INV = f"<{dsml}{inv}"
        self.INV_END = f"</{dsml}{inv}"
        self.PAR = f"<{dsml}{par}"
        self.PAR_END = f"</{dsml}{par}"
        self.text = ""
        self.pos = 0
        self.state = "start"
        self.calls: list[dict] = []  # {"id", "name", "args": [pieces], "done"}
        self.first_param = True
        self.is_str = False
        self.broken = False

    @staticmethod
    def _escape(piece: str) -> str:
        try:
            return json.dumps(piece, ensure_ascii=False)[1:-1]
        except Exception:
            return json.dumps(piece, ensure_ascii=True)[1:-1]

    def _next(self, tokens):
        best = None
        for t in tokens:
            i = self.text.find(t, self.pos)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, t)
        return best

    def _hold(self, token: str) -> int:
        tail = self.text[self.pos:]
        return max((k for k in range(1, min(len(token), len(tail)) + 1) if tail.endswith(token[:k])), default=0)

    def feed(self, piece: str) -> list[dict]:
        """Add generated text of the tool block; returns OpenAI tool_call delta objects to send now (possibly empty)."""
        self.text += piece
        out: dict[int, dict] = {}

        def add_args(i, frag):
            if not frag:
                return
            self.calls[i]["args"].append(frag)
            d = out.setdefault(i, {"index": i, "function": {"arguments": ""}})
            d["function"]["arguments"] += frag

        while not self.broken:
            if self.state == "start":
                if len(self.text) < len(self.CALLS):
                    break
                if not self.text.startswith(self.CALLS):
                    self.broken = True
                    break
                self.pos = len(self.CALLS)
                self.state = "between"
            elif self.state in ("between", "after_param"):
                nxt = self._next([self.INV, self.CALLS_END] if self.state == "between" else [self.PAR, self.INV_END])
                if nxt is None:
                    break
                i, tok = nxt
                if self.text[self.pos:i] != ">\n":
                    self.broken = True
                    break
                self.pos = i + len(tok)
                if tok == self.CALLS_END:
                    self.state = "end"
                elif tok == self.INV:
                    self.state = "header"
                elif tok == self.PAR:
                    self.state = "param"
                else:  # INV_END
                    add_args(len(self.calls) - 1, "}")
                    self.calls[-1]["done"] = True
                    self.state = "between"
            elif self.state == "header":
                nxt = self._next([self.PAR, self.INV_END])
                if nxt is None:
                    break
                i, tok = nxt
                found = re.findall(r'^\s*name="(.*?)">\n$', self.text[self.pos:i], flags=re.DOTALL)  # the reference pattern and test, verbatim
                if len(found) != 1:
                    self.broken = True
                    break
                name = found[0]
                if self.split is not None:
                    try:
                        name = self.split(name)[1]
                    except Exception:
                        self.broken = True
                        break
                k = len(self.calls)
                self.calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "name": name, "args": [], "done": False})
                out[k] = {"index": k, "id": self.calls[k]["id"], "type": "function", "function": {"name": name, "arguments": ""}}
                self.pos = i + len(tok)
                self.first_param = True
                if tok == self.PAR:
                    self.state = "param"
                else:
                    add_args(k, "{}")
                    self.calls[k]["done"] = True
                    self.state = "between"
            elif self.state == "param":
                m = re.match(r' name="(.*?)" string="(true|false)">', self.text[self.pos:], flags=re.DOTALL)
                if not m:
                    if self.text.find(self.PAR_END, self.pos) >= 0 or self.text.find(self.INV_END, self.pos) >= 0:
                        self.broken = True  # the header never became valid
                    break
                key, self.is_str = m.group(1), m.group(2) == "true"
                add_args(len(self.calls) - 1, ("{" if self.first_param else ", ") + json.dumps(key, ensure_ascii=False) + ": " + ('"' if self.is_str else ""))
                self.first_param = False
                self.pos += m.end()
                self.state = "value"
            elif self.state == "value":
                i = self.text.find(self.PAR_END, self.pos)
                end = i if i >= 0 else len(self.text) - self._hold(self.PAR_END)
                val = self.text[self.pos:end]
                add_args(len(self.calls) - 1, self._escape(val) if self.is_str else val)
                self.pos = end
                if i < 0:
                    break
                if self.is_str:
                    add_args(len(self.calls) - 1, '"')
                self.pos = i + len(self.PAR_END)
                self.state = "after_param"
            else:  # end: nothing more is expected inside the block
                break
        return [out[k] for k in sorted(out)]

    def result(self) -> list[tuple]:
        return [(c["name"], "".join(c["args"])) for c in self.calls]

    def complete(self) -> bool:
        return bool(self.calls) and all(c["done"] for c in self.calls)


class Detok:
    """Receive tokens one at a time and emit only the safe text prefix (protects partial multi-byte sequences).
    vLLM-style incremental decoding: only the last few tokens (the prefix window) are decoded twice and the difference is
    emitted — same output text as re-decoding the whole id sequence per token, which is O(n^2) and kept a CPU core busy on
    long responses."""

    WINDOW = 6  # prefix window (tokens) protecting multi-byte sequences and merges

    def __init__(self):
        self.ids: list[int] = []
        self.prefix_offset = 0
        self.read_offset = 0

    def push(self, tid: int) -> str:
        self.ids.append(tid)
        prefix_text = TOK.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=False)
        new_text = TOK.decode(self.ids[self.prefix_offset:], skip_special_tokens=False)
        if new_text.endswith("�"):
            return ""
        out = new_text[len(prefix_text):]
        self.read_offset = len(self.ids)
        self.prefix_offset = max(0, self.read_offset - self.WINDOW)
        return out


class StopBuffer:
    """Hold only a suffix that could begin a stop string; safe across token boundaries."""
    def __init__(self, stops):
        if isinstance(stops, str): stops = [stops]
        self.stops = [s for s in (stops or []) if isinstance(s, str) and s]
        self.buf = ''
        self.stopped = False
        self.matched = None  # D11: the stop string that matched (Anthropic stop_sequence) — earliest position, ties by list order

    def push(self, piece):
        self.buf += piece
        matches = [(self.buf.find(s), i, s) for i, s in enumerate(self.stops) if s in self.buf]
        if matches:
            at, _, self.matched = min(matches)
            out = self.buf[:at]
            self.buf = ''; self.stopped = True
            return out
        hold = max((n for s in self.stops for n in range(1, min(len(s), len(self.buf)) + 1)
                    if self.buf.endswith(s[:n])), default=0)
        out = self.buf[:-hold] if hold else self.buf
        self.buf = self.buf[-hold:] if hold else ''
        return out

    def finish(self):
        out, self.buf = self.buf, ''
        return out


THINK_END = "</think>"
DSML_START = "<｜DSML｜"


def _tag_hold(buf: str, tag: str) -> int:
    """If buf ends with a prefix of tag, return its length (undecidable until the next piece arrives)."""
    return max((k for k in range(1, len(tag)) if buf.endswith(tag[:k])), default=0)


class VisibleStop:
    """D12: stop strings apply only to the visible body (content). Applied to the whole raw text (including thinking and
    DSML tool blocks), a stop inside the thinking would cut the answer, and a stop inside a tool block would drop the call
    and leak DSML fragments into the content (reproduced with a fake daemon).
    The raw text passes through unchanged (thinking -> </think> -> content -> <｜DSML｜ tool block — the same boundaries as
    sse() and the reference parser); only the content span goes through StopBuffer."""

    def __init__(self, stops, thinking: bool):
        self.sb = StopBuffer(stops)
        self.mode = "think" if thinking else "content"
        self.pend = ""

    @property
    def stopped(self): return self.sb.stopped

    @property
    def matched(self): return self.sb.matched

    def push(self, piece: str) -> str:
        if not self.sb.stops:
            return piece  # no stop strings: plain pass-through
        self.pend += piece
        out = ""
        while self.pend:
            if self.mode == "think":
                j = self.pend.find(THINK_END)
                if j < 0:
                    h = _tag_hold(self.pend, THINK_END)
                    out += self.pend[: len(self.pend) - h]; self.pend = self.pend[len(self.pend) - h:]
                    break
                out += self.pend[: j + len(THINK_END)]; self.pend = self.pend[j + len(THINK_END):]
                self.mode = "content"
                continue
            if self.mode == "tool":
                out += self.pend; self.pend = ""
                break
            j = self.pend.find(DSML_START)
            h = 0 if j >= 0 else _tag_hold(self.pend, DSML_START)
            seg = self.pend[:j] if j >= 0 else self.pend[: len(self.pend) - h]
            self.pend = self.pend[j:] if j >= 0 else self.pend[len(self.pend) - h:]
            out += self.sb.push(seg)
            if self.sb.stopped:
                self.pend = ""
                return out
            if j < 0:
                break
            out += self.sb.finish()  # before a tool block: a held stop-string prefix in the content is content
            self.mode = "tool"
        return out

    def finish(self) -> str:
        if self.mode != "content":
            out, self.pend = self.sb.finish() + self.pend, ""
            return out
        out = self.sb.push(self.pend); self.pend = ""
        return out if self.sb.stopped else out + self.sb.finish()


# ----------------------------------------------------------------------------------------------
@app.get("/health")
async def health():
    # Z1: state is one of loading (daemon socket not there yet — loading), ready, draining (sleep requested: finishing
    #   in-flight requests), sleeping, waking. Returns 200 while asleep too (the daemon is alive and queues requests,
    #   processing them after wake). vram = GPU memory use measured by the daemon (MiB).
    try:
        state = await asyncio.wait_for(DAEMON.stats(), timeout=2.0)
        return {"ok": True, "state": state.get("state", "ready"), "vram": state.get("vram"), "daemon": state}
    except Exception as e:
        return JSONResponse({"ok": False, "state": "loading", "error": str(e)}, status_code=503)


# Z1 sleep/wake — same paths and calls as SGLang --enable-memory-saver (POST, body {}), so existing orchestration
#   scripts can call either engine the same way.
#   /release_memory_occupation (= /admin/sleep): waits until in-flight requests finish (new requests queue in hived),
#     releases VRAM, then 200.
#     Optional body {"level": 2} also moves the session KV (session pool) to a RAM image; {"level": 3} additionally keeps
#     dense weights, Work and cuBLAS as pinned host copies (same VA reservation — only when hived was started with
#     HIVE_SLEEP_VMM=1, otherwise absorbed as 2) — for keeping hived resident while asleep and another process uses the GPU.
#     Optional body {"timeout_s": N}: if in-flight requests do not finish within N s, 400 {"error":"not idle"} — the same
#     shape as SGLang's rejection (retryable).
#   /resume_memory_occupation (= /admin/wake): re-acquires the slots, reloads the experts that were resident before sleep,
#     then 200. 503 if VRAM is insufficient (stays asleep, retryable).
#   Both are idempotent (200 immediately if already in that state). 503 {"state":"loading"} while the daemon is still loading.
_CTL_STATUS = (("not idle", 400), ("sleep cancelled", 409), ("wake failed", 503), ("shutting down", 503))


async def _control(op: str, request: Request | None):
    kw = {}
    if op == "sleep" and request is not None:
        try:
            body = await request.json()
        except Exception:
            body = None  # empty body / not JSON = no options (normalized)
        t = body.get("timeout_s") if isinstance(body, dict) else None
        if isinstance(t, (int, float)) and not isinstance(t, bool) and t > 0:
            kw["timeout_s"] = float(t)
        lv = body.get("level") if isinstance(body, dict) else None  # 2 = session KV to RAM too; 3 = + dense weights and Work (VMM, resident sleep); otherwise 1
        if isinstance(lv, (int, float)) and not isinstance(lv, bool) and lv >= 2:
            kw["level"] = 3 if lv >= 3 else 2
    try:
        res = await DAEMON.control(op, **kw)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        return JSONResponse({"ok": False, "state": "loading", "error": f"daemon not ready: {e}"}, status_code=503)
    err = res.get("error")
    if not err:
        return res
    code = next((c for k, c in _CTL_STATUS if k in err), 500)
    return JSONResponse({"ok": False, **res}, status_code=code)


@app.post("/release_memory_occupation")
@app.post("/admin/sleep")
async def release_memory_occupation(request: Request):
    return await _control("sleep", request)


@app.post("/resume_memory_occupation")
@app.post("/admin/wake")
async def resume_memory_occupation(request: Request):
    return await _control("wake", request)


# /flush_cache (POST, as SGLang): drop every reusable prompt state in hived — session checkpoints and history, archived sessions, shared
#   boundary snapshots — so the next request of any conversation is prefilled from scratch. The expert VRAM cache (weights) is kept.
#   400 {"error":"not idle"} while a request is decoding (SGLang refuses the same way; retry when idle). Benchmarks call it between runs.
@app.post("/flush_cache")
async def flush_cache(request: Request):
    # Optional query `timeout` (seconds, as SGLang's /flush_cache?timeout=N): keep retrying while the daemon answers "not idle" until
    #   it succeeds or the timeout passes. The benchmark client sends the flush right after its warm-up streams end; a flush with no
    #   timeout that lands a moment before the daemon has retired the last request would otherwise fail the whole axis.
    try:
        t = float(request.query_params.get("timeout") or 0)
    except (TypeError, ValueError):
        t = 0.0
    deadline = time.monotonic() + max(0.0, min(t, 3600.0))
    while True:
        res = await _control("flush", None)
        if not isinstance(res, JSONResponse) or res.status_code != 400 or time.monotonic() >= deadline:
            return res
        await asyncio.sleep(0.2)


class _NoHealthAccess(logging.Filter):  # drop /health and /live access-log lines (they were half of the server log)
    def filter(self, record):
        msg = record.getMessage()
        return '"GET /health' not in msg and '"GET /live' not in msg


@app.on_event("startup")
async def _quiet_health_log():
    logging.getLogger("uvicorn.access").addFilter(_NoHealthAccess())


@app.get('/live')
async def live():
    return {'ok': True}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "hive", "object": "model", "owned_by": "eke-hive"}]}


# HIVE_STAGE_STATUS (default off): the engine-side stage of each request, for clients that want to show what a response is waiting on.
#   GET /v1/hive/requests/{id} (id = the response id, "chatcmpl-…", sent in the first stream chunk) →
#   {"id", "stage": queued | prefill | thinking | tool_call | answer | done | error, "prompt_tokens", "cached_tokens", "prefill_total",
#    "prefill_done", "tool_name", "updated"}. queued = waiting for the engine to admit it (other requests run), prefill = reading the
#   prompt (cached_tokens reused, prefill_total to read, prefill_done after each chunk), then the generated text's stage. Off = the route
#   answers 404 and nothing else changes (responses are byte-identical; the daemon line "admitted" is only sent when asked for).
#   Entries live while the request runs and STAGE_KEEP_S after it ends.
STAGE_STATUS = env_on("HIVE_STAGE_STATUS")
STAGE_KEEP_S = 120.0
STAGES: dict = {}


def stage_open(rid: str, prompt_tokens: int) -> None:
    now = time.time()
    for k in [k for k, v in STAGES.items() if v.get("_ended") and now - v["_ended"] > STAGE_KEEP_S]:
        STAGES.pop(k, None)
    STAGES[rid] = {"stage": "queued", "prompt_tokens": prompt_tokens, "updated": round(now, 3)}


def stage_update(rid: str, ended: bool = False, **kw) -> None:
    st = STAGES.get(rid)
    if st is None:
        return
    st.update(kw)
    st["updated"] = round(time.time(), 3)
    if ended:
        st["_ended"] = time.time()


class StageText:
    """Stage of the generated text (HIVE_STAGE_STATUS): thinking until </think> in thinking mode, then answer; tool_call from the family's
    tool-call start on, with the tool name once the format has written it (GLM `<tool_call>name<arg_key>`, DeepSeek DSML `invoke name="…"`).
    feed() returns True when the stage or the tool name changed."""

    _DS_NAME = re.compile(r'invoke name="([^"]+)"')
    _GLM_NAME = re.compile(r"^\s*([^<\s]+)\s*<")

    def __init__(self, thinking: bool):
        self.stage = "thinking" if thinking else "answer"
        self.tool = None
        self.buf = ""

    def feed(self, text: str) -> bool:
        if not text or (self.stage == "tool_call" and self.tool):
            return False
        before = (self.stage, self.tool)
        self.buf = (self.buf + text)[-8192:]
        if self.stage == "thinking":
            j = self.buf.find("</think>")
            if j < 0:
                return False
            self.stage, self.buf = "answer", self.buf[j + len("</think>"):]
        if self.stage == "answer":
            j = self.buf.find(DSML_START)
            if j >= 0:
                self.stage, self.buf = "tool_call", self.buf[j + len(DSML_START):]
        if self.stage == "tool_call" and not self.tool:
            m = self._DS_NAME.search(self.buf) or self._GLM_NAME.match(self.buf)
            if m:
                self.tool = m.group(1)[:80]
        return (self.stage, self.tool) != before


@app.get("/v1/hive/requests/{rid}")
async def stage_status(rid: str):
    st = STAGES.get(rid) if STAGE_STATUS else None
    if st is None:
        return JSONResponse({"error": {"message": "not found", "type": "not_found"}}, status_code=404)
    return {"id": rid, **{k: v for k, v in st.items() if not k.startswith("_")}}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    t_recv = time.time() * 1000.0
    body, bad = await read_body(request)
    if bad is not None:
        return bad
    endpoint = getattr(request, "hive_endpoint", "/v1/chat/completions")
    messages = body.get("messages", [])
    tools = body.get("tools")
    thinking_mode, effort = effort_from_request(body)
    try:
        prep_ms = {}  # request record "prep_ms": where the server's time before the engine goes (lock wait, chat template, tokenizer, boundary hints)
        def prepare():
            t0 = time.perf_counter()
            with ENCODE_LOCK:
                t1 = time.perf_counter()
                prompt, media = encode_prompt(messages, tools, thinking_mode, effort)
                t2 = time.perf_counter()
                ids, images, blob = tokenize_with_images(prompt, media)
                t3 = time.perf_counter()
                # H5: boundary hints are computed only with HIVE_PREFIX_SHARE (same container env as hived — off = zero cost, no field)
                hints = boundary_hints(messages, tools, thinking_mode, effort, prompt, ids) if env_on("HIVE_PREFIX_SHARE") and not media else []
                t4 = time.perf_counter()
            prep_ms.update(lock=round((t1 - t0) * 1000, 1), template=round((t2 - t1) * 1000, 1), tokenize=round((t3 - t2) * 1000, 1), hints=round((t4 - t3) * 1000, 1))
            return ids, images, blob, hints
        t_prep = time.perf_counter()
        ids, images, blob, hints = await asyncio.to_thread(prepare)
        prep_ms["total"] = round((time.perf_counter() - t_prep) * 1000, 1)
    except Exception as e:
        log_request({"t_recv": t_recv, "t_done": time.time() * 1000.0, "endpoint": endpoint, "status": 400, "outcome": "bad_request", "error": str(e)[:300],
                     "stream": bool(body.get("stream")), "client": client_tags(request, body)})
        return JSONResponse(_err(str(e)[:300]), status_code=400)
    mc = await daemon_max_ctx()
    if mc and len(ids) + 1 > mc:  # 400 before the stream starts (same response for streaming requests)
        log_request({"t_recv": t_recv, "t_done": time.time() * 1000.0, "endpoint": endpoint, "status": 400, "outcome": "context_length_exceeded",
                     "prompt_tokens": len(ids), "max_ctx": mc, "stream": bool(body.get("stream")), "client": client_tags(request, body)})
        return JSONResponse(overflow_body(mc, len(ids)), status_code=400)
    session = (hashlib.sha256(str(body['hive_session_id']).encode()).hexdigest()[:32]
               if body.get('hive_session_id') else session_id_for(messages, body.get("user")))
    # HIVE_PREFIX_ADAPTIVE (needs HIVE_PREFIX_SHARE hints): one extra boundary chunk only for conversations whose tail changes (TailChangeTracker)
    prefix_extra = (await asyncio.to_thread(TAILS.observe, session, ids, hints)) if hints and env_on("HIVE_PREFIX_ADAPTIVE") else False
    # HIVE_PREFIX_FIRST_TURN (needs HIVE_PREFIX_SHARE hints): the first turn of a conversation (no assistant message yet) asks for one extra
    #   chunk too, so the daemon saves a shared snapshot at the end of the system/tools block — the next new conversation with the same
    #   system block resumes there instead of prefilling it again. With HIVE_PREFIX_EXTRA_CHUNKS=0 no such snapshot was ever taken
    #   (a cut at that boundary adds a chunk, so it is never "free"). The daemon skips the cut once that snapshot exists, so the extra
    #   chunk is paid only by the first conversation with a given system block; follow-up turns keep the global budget.
    if hints and not prefix_extra and env_on("HIVE_PREFIX_FIRST_TURN") and not any(m.get("role") == "assistant" for m in messages if isinstance(m, dict)):
        prefix_extra = True
    if INFLIGHT.get(session, 0) > 0:  # concurrent request in the same conversation (regenerate, two tabs, parallel bench) — separate session
        # B8: a per-request "~<uuid>" session would fill the daemon session pool and archived budget with single-use
        #   sessions and push real conversations out of the LRU. Use fixed per-conversation auxiliary slots (~1, ~2, ...,
        #   lowest free number) instead — the session count is bounded by peak concurrency, and repeated parallel patterns
        #   (bench repeats) reuse that slot's prefix. Avoiding pool occupancy entirely would need a daemon-side one-shot
        #   request field (not in the protocol).
        k = 1
        while INFLIGHT.get(f"{session}~{k}", 0) > 0:
            k += 1
        session = f"{session}~{k}"
    sampling = {
        # Unset = the full context: 0 lets the daemon use max_ctx - prompt
        "max_tokens": max(0, int(_num(body.get("max_tokens") or body.get("max_completion_tokens"), 0))) if abs(_num(body.get("max_tokens") or body.get("max_completion_tokens"), 0)) < 2 ** 62 else 0,
        # Normalize: null, bool, non-numeric and non-finite values become the default (a raw null would make hived's json value() throw -> request error)
        "temperature": _num(body.get("temperature"), 1.0),
        "top_p": _num(body.get("top_p"), 0.95),
        "top_k": int(_num(body.get("top_k"), 0)),
        "min_p": _num(body.get("min_p"), 0.0),
        "seed": int(_num(body.get("seed"), 0)) if abs(_num(body.get("seed"), 0)) < 2 ** 62 else 0,
        # ignore_eos (benchmark harnesses measure throughput at a fixed output length): no stop token, run to max_tokens
        "stop_ids": [] if body.get("ignore_eos") else stop_ids(),
        **({"prefix_extra": 1} if prefix_extra else {}),
    }
    think_cap = think_cap_from_request(body) if thinking_mode == "thinking" else None
    if think_cap is not None:
        t_start, t_end = think_token_ids()
        if t_end is None:
            think_cap = None  # the tokenizer has no single </think> token — feature off (absorbed)
        else:
            sampling.update({"think_cap": think_cap, "think_end_id": t_end, "think_open": bool(ids) and t_start is not None and ids[-1] == t_start,
                             **({"think_start_id": t_start} if t_start is not None else {}),
                             **({"think_exit_ids": ex} if (ex := think_exit_ids()) else {})})
    rid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    if STAGE_STATUS:  # GET /v1/hive/requests/{rid} (stage_status) — the daemon adds its "admitted" line only to requests that ask
        stage_open(rid, len(ids))
        sampling["stages"] = True
    stream = bool(body.get("stream"))
    # Reserve before yielding the response; two streams may be created before either
    # body iterator starts. The daemon still validates the exact token/image prefix.
    INFLIGHT[session] = INFLIGHT.get(session, 0) + 1
    released = False
    def _release():
        nonlocal released
        if not released:
            released = True
            remaining = INFLIGHT.get(session, 1) - 1
            if remaining: INFLIGHT[session] = remaining
            else: INFLIGHT.pop(session, None)
    # D5: on a non-terminal end (disconnect, exception), releasing the reservation before sending cancel opens a gap in
    #   which a new request of the same conversation takes the same session and the late cancel kills it (the daemon
    #   cancel flags the session's latest request — hived.cpp op=="cancel"; reproduced with a fake daemon).
    #   So cancel is sent first (until the daemon ack, in one thread, once even if several paths call it) and the
    #   reservation is released in its completion callback. The callback still releases if an await is cancelled during
    #   ASGI cleanup, so the reservation can never leak.
    loop = asyncio.get_running_loop()
    daemon_rid = uuid.uuid4().hex  # D5: even if cancel is delayed by its 2 s timeout, it cannot hit the next request on the same session
    cancel_fut = None
    def send_cancel():
        nonlocal cancel_fut
        if cancel_fut is None:
            cancel_fut = loop.run_in_executor(None, DAEMON.cancel, session, daemon_rid)
            cancel_fut.add_done_callback(lambda _f: _release())
        return cancel_fut
    def release_session():
        if cancel_fut is not None and not cancel_fut.done():
            return  # the cancel completion callback releases it
        _release()
    async def wait_cancel():
        # Daemon.cancel has a 2 s connect/send/recv timeout — wait at most 2.5 s here; past that the callback releases the reservation
        with suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(cancel_fut), 2.5)

    req_rec = {"t_recv": t_recv, "t_first": None, "endpoint": endpoint, "stream": stream, "max_tokens": sampling["max_tokens"], "session": session,
               "rid": rid, "daemon_rid": daemon_rid, "prompt_tokens": len(ids), "client": client_tags(request, body), "outcome": "closed", "done": None,
               "prep_ms": prep_ms,
               # record thinking mode and effort so it can be checked whether the requested effort actually arrived
               "thinking_mode": thinking_mode, "effort": effort,
               # thinking cap as received (null if none) — whether it was forced is done.think_forced (daemon), also lifted to the top level below
               "think_cap": think_cap,
               **({"prefix_extra": 1} if prefix_extra else {})}

    async def run():
        det = Detok()
        stage_text = StageText(thinking_mode == "thinking") if STAGE_STATUS else None
        stop_ids = set(sampling["stop_ids"])
        stop_text = VisibleStop(body.get('stop'), thinking_mode == "thinking")  # D12: content only
        terminal = False
        generated = 0
        monitor_stop = asyncio.Event()
        async def watch_disconnect():
            # Measured (2 in ~60 requests): when monitor.cancel() at the end of a response lands inside the anyio CancelScope
            #   of Starlette's is_disconnected(), the cancellation is swallowed as a scope cancel, this loop never stops, and
            #   an unbounded `await monitor` would block forever so the last stream chunk is never sent (daemon done, client
            #   waiting in read_chunked — seen with py-spy). So the loop checks its own stop flag and the waiter is bounded.
            while not monitor_stop.is_set():
                if await request.is_disconnected():
                    await asyncio.shield(send_cancel())
                    return
                await asyncio.sleep(.1)
        # Watch streaming requests too — relying on Starlette's version-specific behaviour (listen_for_disconnect for ASGI
        #   spec < 2.4) fails on 2.4+, where a disconnect is only noticed when a send fails, so no cancel went out during
        #   long quiet stretches (prefill, held tool/stop/think-tag text). Cancel is sent once via send_cancel.
        monitor = (asyncio.create_task(watch_disconnect())
                   if hasattr(request, 'is_disconnected') else None)
        try:
            async with aclosing(DAEMON.generate(session, ids, images, blob, rid=daemon_rid, **({"boundaries": hints} if hints else {}), **sampling)) as source:
                async for msg in source:
                    if "id" in msg:
                        if req_rec["t_first"] is None:
                            req_rec["t_first"] = time.time() * 1000.0
                        generated += 1
                        if msg["id"] in stop_ids:
                            continue
                        raw = det.push(msg["id"])
                        if stage_text is not None and (stage_text.feed(raw) or generated == 1):  # the first id ends the prefill stage
                            stage_update(rid, stage=stage_text.stage, **({"tool_name": stage_text.tool} if stage_text.tool else {}))
                        piece = stop_text.push(raw)
                        if piece: yield ("delta", piece)
                        if stop_text.stopped:
                            terminal = True
                            req_rec["outcome"] = "stop_sequence"
                            await asyncio.shield(send_cancel())
                            yield ('done', {'done': True, 'n': generated, 'finish': 'stop', 'stop_sequence': stop_text.matched})
                            break
                    elif "admitted" in msg or "progress" in msg:  # sent with "stages" (admitted) / after each prefill chunk (progress)
                        if STAGE_STATUS:
                            if "admitted" in msg:
                                stage_update(rid, stage="prefill", cached_tokens=int(msg.get("cached") or 0), prefill_done=0,
                                             prefill_total=max(0, int(msg.get("total") or 0) - int(msg.get("cached") or 0)))
                            elif STAGES.get(rid, {}).get("stage") == "prefill":
                                stage_update(rid, prefill_done=max(0, int(msg.get("progress") or 0) - int(STAGES[rid].get("cached_tokens") or 0)))
                        continue
                    elif msg.get("done"):
                        terminal = True
                        req_rec["done"] = msg
                        req_rec["outcome"] = "cancelled" if msg.get("finish") == "cancel" else "done"
                        if "think_forced" in msg:  # only requests with a thinking cap (the daemon sends it only then) — also at the record's top level for aggregation
                            req_rec["think_forced"] = bool(msg["think_forced"])
                            req_rec["think_tokens"] = msg.get("think_tokens")
                        tail = stop_text.finish()
                        if tail: yield ('delta', tail)
                        if msg.get("finish") == "cancel":
                            # D13: a daemon cancel must not be reported as length (= max_tokens reached) — a cut answer would look
                            #   like a normal truncation. OpenAI has no cancel finish reason. If the client disconnected this never
                            #   arrives anyway; if it is still connected (external cancel), state honestly that the answer is
                            #   incomplete = an error (non-stream: 502 + partial_content; stream: error event).
                            yield ("error", "generation cancelled by the engine")
                        else:
                            yield ("done", msg)
                        break
                    elif msg.get("error"):
                        terminal = True
                        req_rec["outcome"] = "error"
                        req_rec["error"] = str(msg["error"])[:300]
                        if msg["error"] == "context overflow" and isinstance(msg.get("max_ctx"), int):  # missed by the pre-check (e.g. daemon config changed)
                            req_rec["outcome"] = "context_length_exceeded"
                            ob = overflow_body(msg["max_ctx"], int(msg.get("prompt_tokens") or len(ids)))
                            yield ("overflow", ob) if not stream else ("error", ob["error"]["message"])
                            break
                        yield ("error", msg["error"])
                        break
            if not terminal:
                req_rec["outcome"] = "error"
                req_rec["error"] = "daemon ended before completion"
                yield ('error', 'daemon ended before completion')
        except (OSError, ValueError, RuntimeError) as e:
            req_rec["outcome"] = "error"
            req_rec["error"] = str(e)[:300]
            yield ('error', str(e))
        finally:
            if STAGE_STATUS:
                stage_update(rid, stage="done" if req_rec["outcome"] in ("done", "stop_sequence") else "error", ended=True)
            # HTTP status: streams are always 200 (errors are in-stream error events); non-stream: done/stop 200, error/cancel 502, closed midway (client disconnect) 499
            req_rec["t_done"] = time.time() * 1000.0
            req_rec["status"] = 200 if stream else {"done": 200, "stop_sequence": 200, "closed": 499, "context_length_exceeded": 400}.get(req_rec["outcome"], 502)
            if not req_rec.get("_defer_log"):  # streams log once at the very end of sse() (after the tool-call comparison)
                log_request(req_rec)
            try:
                if monitor:
                    monitor_stop.set()
                    monitor.cancel()
                    with suppress(asyncio.CancelledError): await asyncio.wait({monitor}, timeout=1.0)
                if not terminal: send_cancel()  # D5: cancel first, release afterwards (callback)
                if cancel_fut is not None: await wait_cancel()
            finally:
                release_session()

    async def complete_once():
        text_parts, done = [], {}
        async with aclosing(run()) as source:
            async for kind, val in source:
                if hasattr(request, 'is_disconnected') and await request.is_disconnected():
                    return JSONResponse({'error': {'message': 'client disconnected'}}, status_code=499)
                if kind == "delta":
                    text_parts.append(val)
                elif kind == "done":
                    done = val
                elif kind == "overflow":
                    return JSONResponse(val, status_code=400)
                elif kind == "error":
                    return JSONResponse({"error": {"message": val, "type": "server_error"}, 'partial_content': ''.join(text_parts)}, status_code=502)
        text = strip_eos("".join(text_parts))
        try:
            parsed = parse_completion(text + TOK.eos_token if done.get("finish") == "stop" else text, thinking_mode, tools)
        except Exception:
            # If the reference parser rejects the text (e.g. cut at max_tokens): thinking mode without </think> = all thinking; with it, split before/after
            if thinking_mode == "thinking":
                j = text.find("</think>")
                parsed = {"reasoning_content": text, "content": ""} if j < 0 else {"reasoning_content": text[:j], "content": text[j + len("</think>"):]}
            else:
                parsed = {"content": text}
        message = {"role": "assistant", "content": parsed.get("content") or ""}
        if parsed.get("reasoning_content"):
            message["reasoning_content"] = parsed["reasoning_content"]
        if parsed.get("tool_calls"):
            message["tool_calls"] = openai_tool_calls(parsed["tool_calls"])
            bad = invalid_tool_args([(c["function"]["name"], c["function"]["arguments"]) for c in message["tool_calls"]])
            if bad:
                req_rec["tool_args_invalid"] = bad
        return {
            "id": rid, "object": "chat.completion", "created": created, "model": "hive",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else ("stop" if done.get("finish") == "stop" else "length"),
                         # D11: the matched stop string (vLLM's choices[].stop_reason convention) — only when one matched (default response shape unchanged)
                         **({"stop_reason": done["stop_sequence"]} if done.get("stop_sequence") else {})}],
            "usage": openai_usage(len(ids), done),
            "hive": {k: done.get(k) for k in ("prefill_ms", "decode_ms", "cached_prefix", "decode_hit", "decode_cpu", "decode_dma_rows", "cpu_wait_ms", "cpu_span_ms", "timing_scope", "batch_rows", "mtp_steps", "mtp_drafted", "mtp_accepted")
                     + (("think_cap", "think_tokens", "think_forced") if "think_forced" in done else ())},  # only requests with a thinking cap
        }

    if not stream:
        # One request-log line, written after the parse (tool_args_invalid) — also on the early returns.
        req_rec["_defer_log"] = True
        try:
            return await complete_once()
        finally:
            req_rec.pop("_defer_log", None)
            log_request(req_rec)

    async def sse():
        # One request-log line per stream, written after the tail below (tool-call comparison) — also when the client disconnects.
        req_rec["_defer_log"] = True
        try:
            async with aclosing(sse_body()) as body_gen:
                async for piece in body_gen:
                    yield piece
        finally:
            req_rec.pop("_defer_log", None)
            log_request(req_rec)

    async def sse_body():
        # Streaming: text inside <think> goes out as reasoning_content, outside as content. DSML tool calls stream as tool_call deltas
        #   while they are generated (DsmlToolStream; HIVE_TOOL_STREAM=0 = parse at the end and send them in one piece).
        buf = ""
        in_think = thinking_mode == "thinking"
        tool_buf = ""
        in_tool = False
        tstream = DsmlToolStream(ENC) if os.environ.get("HIVE_TOOL_STREAM", "1") != "0" and FAMILY != "glm5_next" else None
        done = {}

        def tool_chunks(piece):
            if tstream is None or tstream.broken:
                return []
            deltas = tstream.feed(piece)
            return [f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': 'hive', 'choices': [{'index': 0, 'delta': {'tool_calls': deltas}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"] if deltas else []
        try:
            yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': 'hive', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            source = run()
            async for kind, val in source:
                if kind == "delta":
                    buf += val
                    while buf:
                        if in_think:
                            j = buf.find("</think>")
                            if j < 0:
                                # protect a partial tag
                                safe = buf if not any(buf.endswith("</think>"[:k]) for k in range(1, 8)) else buf[: len(buf) - max(k for k in range(1, 8) if buf.endswith("</think>"[:k]))]
                                if safe:
                                    yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': 'hive', 'choices': [{'index': 0, 'delta': {'reasoning_content': safe}, 'finish_reason': None}]})}\n\n"
                                    buf = buf[len(safe):]
                                break
                            if j > 0:
                                yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': 'hive', 'choices': [{'index': 0, 'delta': {'reasoning_content': buf[:j]}, 'finish_reason': None}]})}\n\n"
                            buf = buf[j + len("</think>"):]
                            in_think = False
                            continue
                        if in_tool:
                            tool_buf += buf
                            for c in tool_chunks(buf):
                                yield c
                            buf = ""
                            break
                        j = buf.find(DSML_START)
                        if j >= 0:
                            # the reference format puts "\n\n" before the tool block ("{content}\n\n<｜DSML｜ calls>...") — those newlines are not content
                            pre = buf[:j].rstrip("\n")
                            if pre:
                                yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': 'hive', 'choices': [{'index': 0, 'delta': {'content': pre}, 'finish_reason': None}]})}\n\n"
                            tool_buf = buf[j:]
                            for c in tool_chunks(tool_buf):
                                yield c
                            buf = ""
                            in_tool = True
                            break
                        safe = buf if not any(buf.endswith(DSML_START[:k]) for k in range(1, len(DSML_START))) else buf[: len(buf) - max(k for k in range(1, len(DSML_START)) if buf.endswith(DSML_START[:k]))]
                        safe = safe[: len(safe.rstrip("\n"))]  # hold trailing newlines — they may start a "\n\n<｜DSML｜" tool block (emitted with the next piece or the final flush)
                        if safe:
                            yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': 'hive', 'choices': [{'index': 0, 'delta': {'content': safe}, 'finish_reason': None}]})}\n\n"
                            buf = buf[len(safe):]
                        break
                elif kind == "done":
                    done = val
                elif kind == "error":
                    # D13: OpenAI SSE error shape {"error": {"message", "type"}}
                    yield f"data: {json.dumps({'error': {'message': val, 'type': 'server_error'}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
        except asyncio.CancelledError:
            raise
        finally:
            if 'source' in locals(): await source.aclose()
            release_session()
        finish = "stop" if done.get("finish") == "stop" else "length"
        if buf:
            key = "reasoning_content" if in_think else "content"
            yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': 'hive', 'choices': [{'index': 0, 'delta': {key: buf}, 'finish_reason': None}]})}\n\n"
        sent = []
        if tool_buf:
            try:
                # the reference parser's tool start marker is "\n\n<｜DSML｜ calls" — re-attach the newlines stripped from the stream so it finds the block
                parsed = parse_completion(("\n\n" if not tool_buf.startswith("\n") else "") + tool_buf + TOK.eos_token, "chat", tools)
                calls = openai_tool_calls(parsed.get("tool_calls") or [])
            except Exception:
                calls = []
            if tstream is not None and tstream.calls:
                # calls were already streamed: they stay (a client cannot take them back) — record whether they equal the reference parse
                ref = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
                req_rec["tool_stream"] = "match" if ref == tstream.result() else ("incomplete" if not tstream.complete() else "mismatch")
                sent = tstream.result()
                if tstream.complete():
                    finish = "tool_calls"
            elif calls:
                sent = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
                deltas = [{"index": i, "id": c["id"], "type": "function", "function": c["function"]} for i, c in enumerate(calls)]
                yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': 'hive', 'choices': [{'index': 0, 'delta': {'tool_calls': deltas}, 'finish_reason': None}]})}\n\n"
                finish = "tool_calls"
            else:
                yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': 'hive', 'choices': [{'index': 0, 'delta': {'content': tool_buf}, 'finish_reason': None}]})}\n\n"
        bad = invalid_tool_args(sent) if tool_buf and sent else []
        if bad:
            req_rec["tool_args_invalid"] = bad
        last = {'index': 0, 'delta': {}, 'finish_reason': finish, **({'stop_reason': done['stop_sequence']} if done.get('stop_sequence') and finish == 'stop' else {})}  # D11
        yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': 'hive', 'choices': [last], 'usage': openai_usage(len(ids), done)})}\n\n"
        yield "data: [DONE]\n\n"

    response = ClosingStreamingResponse(sse(), media_type="text/event-stream", on_close=release_session)
    response.prompt_tokens = len(ids)
    return response


async def anthropic_stream(response):
    """Translate the live OpenAI event stream; never collect the full completion."""
    def event(kind, **data):
        return f'event: {kind}\ndata: {json.dumps({"type": kind, **data}, ensure_ascii=False)}\n\n'
    mid = 'msg_' + uuid.uuid4().hex[:24]
    index = -1
    current = None
    usage = {}
    finish = None
    stop_seq = None
    source = response.body_iterator
    try:
        yield event('message_start', message={'id': mid, 'type': 'message', 'role': 'assistant', 'model': 'hive',
                    'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': getattr(response,'prompt_tokens',0), 'output_tokens': 0}})
        async for chunk in source:
            if isinstance(chunk, bytes): chunk = chunk.decode()
            for line in chunk.splitlines():
                if not line.startswith('data: '): continue
                payload = line[6:]
                if payload == '[DONE]': continue
                data = json.loads(payload)
                if data.get('error'):
                    err = data['error']  # D13: accept both {"error": {"message": ...}} and a plain string
                    yield event('error', error={'type': 'api_error', 'message': str(err.get('message', err) if isinstance(err, dict) else err)})
                    return
                usage = data.get('usage') or usage
                for choice in data.get('choices', []):
                    delta = choice.get('delta', {})
                    finish = choice.get('finish_reason') or finish
                    stop_seq = choice.get('stop_reason') if isinstance(choice.get('stop_reason'), str) else stop_seq
                    for field, kind in [('reasoning_content', 'thinking'), ('content', 'text')]:
                        text = delta.get(field)
                        if not text: continue
                        if current != kind:
                            if current: yield event('content_block_stop', index=index)
                            index += 1; current = kind
                            yield event('content_block_start', index=index, content_block={'type': kind, kind: ''})
                        yield event('content_block_delta', index=index, delta={'type': kind + '_delta', kind: text})
                    for call in delta.get('tool_calls') or []:
                        fn = call.get('function') or {}
                        if call.get('id'):  # a new call (the first piece carries id and name; later pieces only arguments)
                            if current: yield event('content_block_stop', index=index)
                            index += 1; current = 'tool_use'
                            yield event('content_block_start', index=index, content_block={'type': 'tool_use', 'id': call['id'], 'name': fn.get('name'), 'input': {}})
                        if fn.get('arguments') and current == 'tool_use':
                            yield event('content_block_delta', index=index, delta={'type': 'input_json_delta', 'partial_json': fn['arguments']})
        if current: yield event('content_block_stop', index=index)
        if finish is None:
            yield event('error', error={'type': 'api_error', 'message': 'stream ended before completion'})
            return
        reason = {'tool_calls': 'tool_use', 'stop': 'end_turn', 'length': 'max_tokens'}.get(finish, 'end_turn')
        if reason == 'end_turn' and stop_seq: reason = 'stop_sequence'  # D11
        else: stop_seq = None
        yield event('message_delta', delta={'stop_reason': reason, 'stop_sequence': stop_seq},
                    usage=anthropic_usage(usage))
        yield event('message_stop')
    finally:
        await source.aclose()
        if getattr(response,'on_close',None): response.on_close()


# ---- Anthropic Messages ---------------------------------------------------------------
def anthropic_to_openai(body: dict) -> tuple[list[dict], list | None]:
    """Anthropic messages -> OpenAI chat messages. One turn stays one message: its text (and images) plus every tool_use block as
    tool_calls; tool_result blocks become the tool messages that follow it (their block lists are reduced to their text)."""
    msgs = []
    if body.get("system"):
        sys_text = body["system"] if isinstance(body["system"], str) else "".join(b.get("text", "") for b in body["system"] if isinstance(b, dict))
        msgs.append({"role": "system", "content": sys_text})
    for m in body.get("messages", []):
        if not isinstance(m, dict):
            continue
        role, blocks = m.get("role", "user"), m.get("content")
        if not isinstance(blocks, list):
            msgs.append({"role": role, "content": blocks})
            continue
        parts, tool_calls, after = [], [], []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                parts.append({"type": "text", "text": b.get("text", "")})
            elif t == "image":
                parts.append({"type": "image", "source": b.get("source")})
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = "".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
                after.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": c if isinstance(c, str) else json.dumps(c)})
            elif t == "tool_use":
                tool_calls.append({"id": b.get("id"), "type": "function",
                                   "function": {"name": b.get("name", ""), "arguments": json.dumps(b.get("input") or {})}})
        if parts or tool_calls:
            msg = {"role": role, "content": parts if any(p["type"] == "image" for p in parts) else "".join(p["text"] for p in parts if p["type"] == "text")}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            msgs.append(msg)
        msgs.extend(after)
    tools = None
    if body.get("tools"):
        tools = [_anthropic_tool(t) for t in body["tools"]]
    return msgs, tools


def _anthropic_tool(t: dict) -> dict:
    fn = dict(name=t["name"], description=t.get("description", ""), parameters=t.get("input_schema", {}))
    return {"type": "function", "function": fn}


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body, bad = await read_body(request)
    if bad is not None:
        return bad
    messages, tools = anthropic_to_openai(body)
    oa = {"messages": messages, "tools": tools, "max_tokens": body.get("max_tokens") or 0, "temperature": body.get("temperature", 1.0),
          "top_p": body.get("top_p", 0.95), "stream": bool(body.get('stream')),
          "stop": body.get('stop_sequences'), "user": (body.get('metadata') or {}).get('user_id')}
    if body.get('hive_session_id'): oa['hive_session_id'] = body['hive_session_id']
    th = body.get("thinking")
    if isinstance(th, dict):
        if th.get("type") == "disabled":
            oa["reasoning_effort"] = "none"
        elif th.get("budget_tokens"):
            b = th["budget_tokens"]
            oa["reasoning_effort"] = "low" if b < 2000 else ("medium" if b < 8000 else "high")
    if isinstance(body.get("output_config"), dict) and body["output_config"].get("effort"):
        oa["reasoning_effort"] = body["output_config"]["effort"]

    class _R:
        hive_endpoint = "/v1/messages"  # endpoint field of the request-log JSONL
        headers = getattr(request, "headers", None)

        async def json(self):
            return oa
        async def is_disconnected(self):
            return await request.is_disconnected() if hasattr(request, 'is_disconnected') else False

    resp = await chat(_R())
    if isinstance(resp, JSONResponse):
        return resp
    if isinstance(resp, StreamingResponse):
        return ClosingStreamingResponse(anthropic_stream(resp), media_type='text/event-stream', on_close=resp.on_close)
    msg = resp["choices"][0]["message"]
    content = []
    if msg.get("reasoning_content"):
        content.append({"type": "thinking", "thinking": msg["reasoning_content"]})
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for c in msg.get("tool_calls") or []:
        try:
            args = json.loads(c["function"]["arguments"])
        except Exception:
            args = {"_raw": c["function"]["arguments"]}
        content.append({"type": "tool_use", "id": c.get("id") or "toolu_" + uuid.uuid4().hex[:12], "name": c["function"]["name"], "input": args})
    stop = "tool_use" if msg.get("tool_calls") else ("end_turn" if resp["choices"][0]["finish_reason"] == "stop" else "max_tokens")
    stop_seq = resp["choices"][0].get("stop_reason") if stop == "end_turn" else None
    if stop_seq: stop = "stop_sequence"  # D11: match = stop_sequence + that string, otherwise null (the field is always present — Anthropic response shape)
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    u = resp["usage"]
    out = dict(id=msg_id, type="message", role="assistant", model="hive", content=content,
               stop_reason=stop, stop_sequence=stop_seq or None, usage=anthropic_usage(u))
    return out


# ----------------------------------------------------------------------------------------------
def main():
    global CFG, ARGS, DAEMON
    parser = argparse.ArgumentParser(description="Eke Hive API server (OpenAI chat and Anthropic messages) in front of hived / hived_glm.")
    parser.add_argument("--ckpt", default=os.environ.get("HIVE_CKPT"), help="checkpoint directory (default: $HIVE_CKPT)")
    parser.add_argument("--sock", default="/tmp/hive.sock", help="Unix socket of the running daemon")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8430, help="HTTP port (default: 8430)")
    a = parser.parse_args()
    if not a.ckpt:
        parser.error("--ckpt DIR (or HIVE_CKPT) is required")
    CFG = json.load(open(os.path.join(a.ckpt, "config.json")))
    ARGS = VisionArgs(CFG)
    load_modules(a.ckpt)
    DAEMON = Daemon(a.sock)
    import uvicorn

    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
