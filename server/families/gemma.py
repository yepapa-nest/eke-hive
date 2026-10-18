# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Gemma 4 Unified family adapter (config.json model_type "gemma4_unified", text and images) for hive_server.py — the same surface as
server/families/qwen.py (encode_messages, parse_message_from_completion_text, prepare_inputs, effort_from_request) plus the family facts.

Prompt: the checkpoint's chat_template.jinja through the tokenizer (apply_chat_template), or the file HIVE_GEMMA_CHAT_TEMPLATE. With
  HIVE_GEMMA_TOOL_THOUGHT=1 (default) one branch is added to the template: after a tool response with thinking off, the generation prompt
  closes an empty thought channel ("<|channel>thought\\n<channel|>") as it does after a user turn — without it the model may open a thought
  channel itself and end with an empty answer (reported for Gemma 4 after tool results; the same fix the serving stack used before).
Thinking: `chat_template_kwargs.enable_thinking` (false: the template ends the prompt with an empty thought channel). Thinking on: the
  model writes "<|channel>thought\\n …<channel|>" before the answer; the prompt is opened inside that channel here, so a stream starts in
  the reasoning and THINK_END ends it. Gemma 4 has no effort levels — any thinking request is one mode.
Stop ids: generation_config.json eos_token_id (<eos> 1, <turn|> 106, and 50, the token that would open a tool response).
Tool calls (the template's own rendering of history tool calls, parsed back):
  <|tool_call>call:NAME{key:VALUE,key2:VALUE}<tool_call|>   (several may follow each other)
  VALUE: <|"|>string<|"|> · number · true / false · null · {key:VALUE,…} (keys bare) · [VALUE,…]
  Arguments are returned as a JSON object string. A string arrives without JSON escaping, so it is taken verbatim between the delimiters.
Sampling defaults when a request omits them come from generation_config.json (temperature 1.0 · top_p 0.95 · top_k 64), as the model card
  recommends; `defaults()` hands them to the server.
Images (image / image_url parts, also inside tool results): the template writes one <|image|> per part; prepare_inputs expands it to
  <|image> + N × <|image|> + <image|> and sends the N merged patches to the daemon (the vision embedder runs there). Preprocessing follows
  processor_config.json image_processor (patch 16, pooling 3, 280 soft tokens, /255, no normalization); the code is this project's own:
  1. size: keep the aspect ratio and take the largest height / width, multiples of 48 (patch × pooling), whose area is at most
     280 · 48² — an image is shrunk or enlarged to it (bicubic); a side that rounds to zero becomes 48 and the other side gets the
     aspect ratio (at most 280 · 48);
  2. merged patches: the 48 × 48 blocks in raster order (row of blocks, then column), each the flattened (y, x, channel) values / 255 —
     the order of the embedder's patch_dense weights; N = (height / 48) · (width / 48), the block at grid (x, y) takes the position
     embedding rows x (axis 0) and y (axis 1).
  Audio and video parts are not served (the audio projection is not loaded; video would need frame sampling and timestamps).
"""
from __future__ import annotations

import io
import json
import math
import os
import re
from typing import Any

TOK = None
FAMILY = "gemma4_unified"
THINK_START = "<|channel>thought\n"
THINK_END = "<channel|>"
# the daemon cuts a conversation's live KV back to a new request's common prefix (hived RtHasRewind): no boundary hints after the first turn
REWINDS = True
TOOL_START = "<|tool_call>"
TOOL_END = "<tool_call|>"
STR_DELIM = '<|"|>'
MEDIA_TYPES = ("image", "image_url", "video", "video_url", "audio", "input_audio", "audio_url")
IMAGE_TYPES = ("image", "image_url")
IMAGE_TOKEN, BOI, EOI = "<|image|>", "<|image>", "<image|>"
_PATCH, _POOL, _MAX_TOKENS = 16, 3, 280   # processor_config.json image_processor (read in init)
# Text that itself contains the placeholder string (code, documentation about this model): it is set aside before the template runs, so
#   the placeholders left in the prompt are exactly the template's one per image part, and put back before tokenization (the tokenizer reads
#   it as before). Measured 2026-10-18: a 64K-token documentation prompt without images was refused with "the template rendered 2 image
#   placeholders for 0 image parts".
_LITERAL = "\ue000gemma-image-literal\ue000"


def _hide_literal(x):
    if isinstance(x, str):
        return x.replace(IMAGE_TOKEN, _LITERAL)
    if isinstance(x, list):
        return [_hide_literal(v) for v in x]
    if isinstance(x, dict):
        return {k: _hide_literal(v) for k, v in x.items()}
    return x


def _restore_literal(text: str) -> str:
    return text.replace(_LITERAL, IMAGE_TOKEN)

STOP_IDS: list[int] = []
EOS_TEXTS: list[str] = []
_GEN: dict = {}
_TEMPLATE: str | None = None

_ANCHOR = """    {%- elif ns.prev_message_type == 'tool_response' and enable_thinking -%}
        {{- '<|channel>thought\\n' -}}
    {%- endif -%}"""
_PATCHED = """    {%- elif ns.prev_message_type == 'tool_response' and enable_thinking -%}
        {{- '<|channel>thought\\n' -}}
    {%- elif ns.prev_message_type == 'tool_response' -%}
        {{- '<|channel>thought\\n<channel|>' -}}
    {%- endif -%}"""


def init(tok, cfg: dict, ckpt: str | None = None) -> None:
    global TOK, STOP_IDS, EOS_TEXTS, _GEN, _TEMPLATE
    TOK = tok
    _GEN = {}
    if ckpt:
        try:
            with open(os.path.join(ckpt, "generation_config.json")) as f:
                _GEN = json.load(f)
        except (OSError, ValueError):
            _GEN = {}
    eos = _GEN.get("eos_token_id")
    if eos is None:
        t = cfg.get("text_config", cfg)
        eos = cfg.get("eos_token_id", t.get("eos_token_id"))
    STOP_IDS = [int(x) for x in (eos if isinstance(eos, list) else [eos])] if eos is not None else [tok.eos_token_id]
    EOS_TEXTS = [s for s in (tok.convert_ids_to_tokens(i) for i in STOP_IDS) if isinstance(s, str) and s]
    path = os.environ.get("HIVE_GEMMA_CHAT_TEMPLATE") or (os.path.join(ckpt, "chat_template.jinja") if ckpt else "")
    tpl = None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            tpl = f.read()
    if tpl is not None and os.environ.get("HIVE_GEMMA_TOOL_THOUGHT", "1") != "0":
        if tpl.count(_ANCHOR) == 1:
            tpl = tpl.replace(_ANCHOR, _PATCHED)
        elif _PATCHED not in tpl:
            print("[gemma] chat template: the tool-response branch was not found — template used unchanged")
    _TEMPLATE = tpl
    global _PATCH, _POOL, _MAX_TOKENS
    try:
        with open(os.path.join(ckpt or "", "processor_config.json")) as f:
            ip = json.load(f).get("image_processor", {})
        _PATCH, _POOL, _MAX_TOKENS = int(ip.get("patch_size", _PATCH)), int(ip.get("pooling_kernel_size", _POOL)), int(ip.get("max_soft_tokens", _MAX_TOKENS))
    except (OSError, ValueError):
        pass


def nothink_mode() -> str:
    return "enable_thinking=false (empty thought channel)"


def defaults() -> dict:
    """Sampling defaults of the model (generation_config.json) for fields a request leaves out."""
    return {k: _GEN[k] for k in ("temperature", "top_p", "top_k") if isinstance(_GEN.get(k), (int, float))}


def effort_from_request(body: dict) -> tuple[str, Any]:
    """("chat", None) unless the request turns thinking on (enable_thinking / thinking true, or an effort other than none / minimal)."""
    kw = body.get("chat_template_kwargs")
    kw = kw if isinstance(kw, dict) else {}
    thinking = kw.get("enable_thinking", kw.get("thinking"))
    effort = kw.get("reasoning_effort", body.get("reasoning_effort"))
    if isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort", effort)
    if thinking is True or (thinking is None and isinstance(effort, (str, int, float)) and not isinstance(effort, bool)
                            and str(effort) not in ("none", "minimal", "0")):
        return "thinking", None
    return "chat", None


def _content_text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return "" if c is None else str(c)


def _keep_images(parts: list, media: list) -> list:
    """A content list with media: image parts are collected (in order — the template writes one <|image|> for each) and reduced to their
    type; text parts stay; audio / video parts are refused."""
    out = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") in IMAGE_TYPES:
            media.append(p)
            out.append({"type": "image"})
        elif isinstance(p, dict) and p.get("type") in MEDIA_TYPES:
            raise ValueError("audio / video parts are not served by this model (images only)")
        elif isinstance(p, dict) and p.get("type") in ("text", "input_text"):
            out.append({"type": "text", "text": p.get("text", "")})
        elif isinstance(p, str):
            out.append({"type": "text", "text": p})
    return out


def encode_messages(messages: list[dict], thinking_mode: str = "chat", reasoning_effort=None, return_multi_modal_data: bool = False, **_):
    msgs = []
    tools = None
    media: list[dict] = []
    for m in messages:
        m = dict(m)
        if "tools" in m:
            tools = m.pop("tools")
        if m.get("role") == "system" and not _content_text(m.get("content")) and tools is not None and len(msgs) == 0:
            continue  # the server's tool carrier (empty system message) — the template takes tools as an argument
        if isinstance(m.get("content"), list):
            if any(isinstance(p, dict) and p.get("type") in MEDIA_TYPES for p in m["content"]):
                m["content"] = _keep_images(m["content"], media)   # the template renders text and <|image|> parts in order
            else:
                m["content"] = _content_text(m["content"])
        tcs = []
        for tc in m.get("tool_calls") or []:
            tc = dict(tc)
            f = dict(tc.get("function") or {})
            if isinstance(f.get("arguments"), str):   # the template renders arguments from a mapping only
                try:
                    a = json.loads(f["arguments"]) if f["arguments"].strip() else {}
                except ValueError:
                    a = {}
                f["arguments"] = a if isinstance(a, dict) else {}
            tc["function"] = f
            tcs.append(tc)
        if tcs:
            m["tool_calls"] = tcs
        msgs.append(m)
    msgs = _hide_literal(msgs)
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": thinking_mode == "thinking"}
    if tools:
        kwargs["tools"] = _hide_literal(tools)
    if _TEMPLATE is not None:
        kwargs["chat_template"] = _TEMPLATE
    prompt = TOK.apply_chat_template(msgs, **kwargs)
    if thinking_mode == "thinking" and not prompt.endswith(THINK_START):
        prompt += THINK_START   # the answer starts inside the thought channel (the stream's reasoning part)
    if not media:   # text only: the prompt as the template wrote it (literal placeholder text back in place)
        prompt = _restore_literal(prompt)
        return (prompt, {"images": []}) if return_multi_modal_data else prompt
    if prompt.count(IMAGE_TOKEN) != len(media):
        raise ValueError(f"the template rendered {prompt.count(IMAGE_TOKEN)} image placeholders for {len(media)} image parts")
    if not return_multi_modal_data:
        raise ValueError("image parts need the multimodal path")
    return prompt, {"images": media}   # literal placeholder text stays set aside until prepare_inputs


# ---- Tool-call parsing -----------------------------------------------------------------------------------------------------------------
class _Args:
    """Recursive-descent reader of the template's argument syntax (see the module docstring)."""

    def __init__(self, s: str):
        self.s, self.i = s, 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n":
            self.i += 1

    def peek(self, t: str) -> bool:
        return self.s.startswith(t, self.i)

    def value(self):
        self.ws()
        if self.peek(STR_DELIM):
            j = self.s.find(STR_DELIM, self.i + len(STR_DELIM))
            if j < 0:
                raise ValueError("unterminated string")
            v = self.s[self.i + len(STR_DELIM): j]
            self.i = j + len(STR_DELIM)
            return v
        if self.peek("{"):
            return self.obj()
        if self.peek("["):
            self.i += 1
            out = []
            self.ws()
            if self.peek("]"):
                self.i += 1
                return out
            while True:
                out.append(self.value())
                self.ws()
                if self.peek(","):
                    self.i += 1
                    continue
                if self.peek("]"):
                    self.i += 1
                    return out
                raise ValueError("bad array")
        m = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|true|false|null").match(self.s, self.i)
        if not m:
            # a bare word (the model sometimes drops the string delimiters): up to the next , } ]
            m = re.compile(r"[^,}\]]*").match(self.s, self.i)
            self.i = m.end()
            return m.group(0).strip()
        self.i = m.end()
        t = m.group(0)
        if t in ("true", "false"):
            return t == "true"
        if t == "null":
            return None
        x = float(t) if any(ch in t for ch in ".eE") else int(t)
        return x if not isinstance(x, float) or math.isfinite(x) else t

    def key(self) -> str:
        self.ws()
        if self.peek(STR_DELIM):
            return self.value()
        if self.peek('"'):
            j = self.s.find('"', self.i + 1)
            k = self.s[self.i + 1: j]
            self.i = j + 1
            return k
        m = re.compile(r"[^:{}\[\],\s]+").match(self.s, self.i)
        if not m:
            raise ValueError("bad key")
        self.i = m.end()
        return m.group(0)

    def obj(self) -> dict:
        assert self.peek("{")
        self.i += 1
        out = {}
        self.ws()
        if self.peek("}"):
            self.i += 1
            return out
        while True:
            k = self.key()
            self.ws()
            if not self.peek(":"):
                raise ValueError("missing ':'")
            self.i += 1
            out[k] = self.value()
            self.ws()
            if self.peek(","):
                self.i += 1
                continue
            if self.peek("}"):
                self.i += 1
                return out
            raise ValueError("bad object")


def parse_tool_calls(text: str, tools=None) -> list[dict]:
    calls = []
    pos = 0
    while True:
        a = text.find(TOOL_START, pos)
        if a < 0:
            break
        b = text.find(TOOL_END, a)
        body = text[a + len(TOOL_START): b if b >= 0 else len(text)].strip()
        pos = (b + len(TOOL_END)) if b >= 0 else len(text)
        if body.startswith("call:"):
            body = body[5:]
        j = body.find("{")
        name = (body[:j] if j >= 0 else body).strip()
        args: dict = {}
        if j >= 0:
            try:
                args = _Args(body[j:]).obj()
            except (ValueError, AssertionError, IndexError):
                args = {}
        if name:
            calls.append({"type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


class ToolStream:
    """Stream Gemma tool calls as OpenAI tool_call deltas while they are generated (the server's HIVE_TOOL_STREAM): the call's id and
    name go out as soon as "call:NAME{" is complete, its arguments when the call closes (<tool_call|>) — built by parse_tool_calls on
    that call's text, so the streamed arguments are exactly the end-of-turn parse. A client waiting on a tool call sees it begin about
    one call's length earlier (measured: bench_gemma tool cases' first delta 190-280 ms after the request, the whole call's generation).
    Same interface as the server's DSML stream: feed(piece) -> deltas, result(), complete(), calls, broken."""

    def __init__(self, _enc=None):
        self.text = ""
        self.pos = 0
        self.calls: list[dict] = []   # {"id", "name", "args": [pieces], "done"}
        self.broken = False

    def feed(self, piece: str) -> list[dict]:
        import uuid
        self.text += piece
        out: list[dict] = []
        while True:
            a = self.text.find(TOOL_START, self.pos)
            if a < 0:
                break
            head = self.text[a + len(TOOL_START):]
            open_call = self.calls and not self.calls[-1]["done"] and self.calls[-1]["start"] == a
            if not open_call:
                j = head.find("{")
                if j < 0:
                    break   # the name is not complete yet
                name = head[:j].strip()
                if name.startswith("call:"):
                    name = name[5:].strip()
                if not name:
                    self.broken = True
                    break
                c = {"id": f"call_{uuid.uuid4().hex[:12]}", "name": name, "args": [], "done": False, "start": a}
                self.calls.append(c)
                out.append({"index": len(self.calls) - 1, "id": c["id"], "type": "function", "function": {"name": name, "arguments": ""}})
            b = self.text.find(TOOL_END, a)
            if b < 0:
                break
            c = self.calls[-1]
            parsed = parse_tool_calls(self.text[a: b + len(TOOL_END)])
            args = parsed[0]["function"]["arguments"] if parsed else "{}"
            c["args"].append(args)
            c["done"] = True
            out.append({"index": len(self.calls) - 1, "function": {"arguments": args}})
            self.pos = b + len(TOOL_END)
        return out

    def result(self) -> list[tuple]:
        return [(c["name"], "".join(c["args"])) for c in self.calls]

    def complete(self) -> bool:
        return bool(self.calls) and all(c["done"] for c in self.calls)


def parse_message_from_completion_text(text: str, thinking_mode: str = "chat", tools=None) -> dict:
    for e in EOS_TEXTS:
        text = text.replace(e, "")
    reasoning = ""
    if text.startswith(THINK_START):   # a thought channel the model opened itself
        text = text[len(THINK_START):]
        thinking_mode = "thinking"
    if thinking_mode == "thinking":
        j = text.find(THINK_END)
        if j < 0:
            return {"reasoning_content": text.strip(), "content": ""}
        reasoning, text = text[:j], text[j + len(THINK_END):]
    k = text.find(TOOL_START)
    content, tool_part = (text, "") if k < 0 else (text[:k], text[k:])
    out: dict[str, Any] = {"content": content.strip()}
    if reasoning.strip():
        out["reasoning_content"] = reasoning.strip()
    calls = parse_tool_calls(tool_part, tools) if tool_part else []
    if calls:
        out["tool_calls"] = calls
    return out


# ---- Images ---------------------------------------------------------------------------------------------------------------------
def _target_size(h: int, w: int) -> tuple[int, int]:
    """(height, width): multiples of patch · pooling, aspect ratio kept, at most max_soft_tokens · pooling² patches (module docstring)."""
    side = _PATCH * _POOL
    budget = _MAX_TOKENS * _POOL * _POOL * _PATCH * _PATCH
    k = math.sqrt(budget / (h * w))
    th, tw = int(math.floor(k * h / side)) * side, int(math.floor(k * w / side)) * side
    if th == 0 and tw == 0:
        raise ValueError(f"image {w}x{h} is too small to resize")
    if th == 0:
        th, tw = side, min(int(math.floor(w / h)) * side, _MAX_TOKENS * side)
    elif tw == 0:
        tw, th = side, min(int(math.floor(h / w)) * side, _MAX_TOKENS * side)
    return th, tw


def _image_patches(data: bytes):
    """Image bytes → (bf16 merged patches as bytes, grid height, grid width)."""
    import numpy as np
    import torch
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError as e:
        raise ValueError(f"cannot identify image: {len(data)} bytes, starting {data[:16].hex() or '(empty)'}") from e
    img = img.convert("RGB")
    th, tw = _target_size(img.height, img.width)
    if (th, tw) != (img.height, img.width):
        img = img.resize((tw, th), Image.BICUBIC)
    side = _PATCH * _POOL
    gh, gw = th // side, tw // side
    x = np.asarray(img, dtype=np.float32) / 255.0                       # [th, tw, 3]
    x = x.reshape(gh, side, gw, side, 3).transpose(0, 2, 1, 3, 4)       # [gh, gw, y, x, c]
    x = np.ascontiguousarray(x.reshape(gh * gw, side * side * 3))
    raw = torch.from_numpy(x).to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
    return raw, gh, gw


def prepare_inputs(prompt: str, media: list[dict], tok):
    """Prompt with one <|image|> per image part → (token ids, hived image metas, bf16 patch bytes) — the server's tokenize_with_images."""
    if not media:
        return tok.encode(_restore_literal(prompt), add_special_tokens=False), [], b""
    from families.glm import _fetch, _part_url   # the shared data: / http(s) URL reader

    pieces = [_image_patches(_fetch(_part_url(p))) for p in media]
    parts = prompt.split(IMAGE_TOKEN)
    if len(parts) != len(pieces) + 1:
        raise ValueError("image parts and placeholders do not match")
    text = _restore_literal(parts[0] + "".join(BOI + IMAGE_TOKEN * (gh * gw) + EOI + after for (_, gh, gw), after in zip(pieces, parts[1:])))
    ids = tok.encode(text, add_special_tokens=False)
    img_id, boi_id, eoi_id = (tok.convert_tokens_to_ids(t) for t in (IMAGE_TOKEN, BOI, EOI))
    metas, blobs, i = [], [], 0
    for raw, gh, gw in pieces:
        n = gh * gw
        # the span is <|image> + n × <|image|> + <image|> (a placeholder string inside the text is a lone token, never this shape)
        while i + n + 1 < len(ids) and not (ids[i] == boi_id and ids[i + n + 1] == eoi_id and all(ids[j] == img_id for j in range(i + 1, i + n + 1))):
            i += 1
        if i + n + 1 >= len(ids):
            raise ValueError("image token span not found in the encoded prompt")
        metas.append({"start": i + 1, "n_vit_h": gh, "n_vit_w": gw, "types": [1] * n, "nbytes": len(raw)})
        blobs.append(raw)
        i += n + 2
    return ids, metas, b"".join(blobs)
