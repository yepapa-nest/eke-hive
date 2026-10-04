# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""GLM-5.3-Flash family adapter for hive_server.py — the same surface the server uses from the DeepSeek reference `encoding`
module (encode_messages, parse_message_from_completion_text) plus the family facts the server needs (stop ids, tool-call tag).

Prompt: the checkpoint's own chat_template.jinja through the tokenizer (apply_chat_template), unchanged.
Thinking: GLM-5.3-Flash has no way to turn thinking off — Z.ai documents `thinking.type` as "enabled" only and `reasoning_effort`
  low / high / max (default max) (docs.z.ai/guides/vlm/glm-5.3-flash; the model card; the template itself: the generation prompt
  always ends with `<think>`). A request for non-thinking (reasoning_effort none, enable_thinking false) therefore maps to:
    HIVE_GLM_NOTHINK=low   (default) — the official minimum: reasoning_effort "low", output still starts in the thinking block
    HIVE_GLM_NOTHINK=empty — an empty thinking block is prefilled (`<think></think>`), the model answers directly (not an official mode)
Tool calls: <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call> (the template's own instruction).
  Values are JSON when they parse as JSON (the template writes non-string arguments with tojson), else the raw string.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import urllib.request
from typing import Any

TOK = None          # set by init()
THINK_END = "</think>"
TOOL_START = "<tool_call>"
FAMILY = "glm5_next"

_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def init(tok, cfg: dict) -> None:
    global TOK, STOP_IDS, EOS_TEXTS
    TOK = tok
    t = cfg.get("text_config", cfg)
    eos = cfg.get("eos_token_id", t.get("eos_token_id"))
    STOP_IDS = [int(x) for x in (eos if isinstance(eos, list) else [eos])] if eos is not None else [tok.eos_token_id]
    EOS_TEXTS = [s for s in (tok.convert_ids_to_tokens(i) for i in STOP_IDS) if isinstance(s, str) and s]


STOP_IDS: list[int] = []
EOS_TEXTS: list[str] = []


def nothink_mode() -> str:
    v = (os.environ.get("HIVE_GLM_NOTHINK") or "low").strip().lower()
    return v if v in ("low", "empty") else "low"


def effort_from_request(body: dict) -> tuple[str, Any]:
    """(thinking_mode, reasoning_effort) for GLM. thinking_mode "chat" only with HIVE_GLM_NOTHINK=empty."""
    kw = body.get("chat_template_kwargs")
    kw = kw if isinstance(kw, dict) else {}
    thinking = kw.get("thinking", kw.get("enable_thinking"))
    effort = kw.get("reasoning_effort", body.get("reasoning_effort"))
    if isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort", effort)
    if isinstance(effort, bool):
        effort = None
    if effort in ("none", "minimal") or thinking is False:
        return ("chat", None) if nothink_mode() == "empty" else ("thinking", "low")
    if isinstance(effort, (int, float)) or (isinstance(effort, str) and re.fullmatch(r"\d+(\.\d+)?", effort or "")):
        x = float(effort)
        effort = "low" if x < 50 else ("high" if x < 90 else "max")
    # Unset or unknown → "high", not the template default "max". Community measurements on the official checkpoints
    #   (NVIDIA developer forum topic 381350, posts #496, #573, #604): high answers 36-45 % sooner than max for 1-4 benchmark points;
    #   the chat template itself treats anything other than low/high as max (#456).
    effort = {"low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}.get(effort, "high")
    return "thinking", effort


def _content_text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return "" if c is None else str(c)


def encode_messages(messages: list[dict], thinking_mode: str = "thinking", reasoning_effort=None, return_multi_modal_data: bool = False, **_):
    msgs = []
    tools = None
    media: list[dict] = []
    for m in messages:
        m = dict(m)
        if "tools" in m:
            tools = m.pop("tools")
        if m.get("role") == "system" and not _content_text(m.get("content")) and tools is not None and len(msgs) == 0:
            continue  # the server's tool carrier (empty system message) — the GLM template takes tools as an argument
        if isinstance(m.get("content"), list):
            if any(isinstance(p, dict) and p.get("type") in MEDIA_TYPES for p in m["content"]):
                # keep the list: the template writes one placeholder per image / video part (expanded in prepare_inputs)
                parts = []
                for p in m["content"]:
                    if isinstance(p, dict) and p.get("type") in MEDIA_TYPES:
                        media.append(p)
                        parts.append({"type": "video" if p.get("type") in ("video", "video_url") else "image"})
                    elif isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                        parts.append({"type": "text", "text": p.get("text", "")})
                    elif isinstance(p, str):
                        parts.append({"type": "text", "text": p})
                m["content"] = parts
            else:
                m["content"] = _content_text(m["content"])
        for tc in m.get("tool_calls") or []:
            f = tc.get("function") if isinstance(tc, dict) else None
            if f and isinstance(f.get("arguments"), str):
                try:
                    f["arguments"] = json.loads(f["arguments"])
                except ValueError:
                    f["arguments"] = {}
        msgs.append(m)
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True, "clear_thinking": True}
    if reasoning_effort in ("low", "high", "max"):
        kwargs["reasoning_effort"] = reasoning_effort
    if tools:
        kwargs["tools"] = tools
    prompt = TOK.apply_chat_template(msgs, **kwargs)
    if thinking_mode == "chat":
        prompt += THINK_END  # empty thinking block (HIVE_GLM_NOTHINK=empty)
    if media and not return_multi_modal_data:
        raise ValueError("image/video parts need the multimodal path")
    return (prompt, {"images": media}) if return_multi_modal_data else prompt


def _arg_value(v: str):
    s = v.strip()
    try:
        return json.loads(s)
    except ValueError:
        return v


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for body in _TOOL_RE.findall(text):
        name, _, rest = body.partition("<arg_key>")
        name = name.strip()
        args = {k.strip(): _arg_value(v) for k, v in _ARG_RE.findall("<arg_key>" + rest)} if rest else {}
        if name:
            calls.append({"type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


def parse_message_from_completion_text(text: str, thinking_mode: str = "thinking") -> dict:
    for e in EOS_TEXTS:
        text = text.replace(e, "")
    reasoning = ""
    if thinking_mode == "thinking":
        j = text.find(THINK_END)
        if j < 0:
            return {"reasoning_content": text, "content": ""}
        reasoning, text = text[:j], text[j + len(THINK_END):]
    k = text.find(TOOL_START)
    content, tool_part = (text, "") if k < 0 else (text[:k], text[k:])
    out: dict[str, Any] = {"content": content.strip()}
    if reasoning.strip():
        out["reasoning_content"] = reasoning.strip()
    calls = parse_tool_calls(tool_part) if tool_part else []
    if calls:
        out["tool_calls"] = calls
    return out


# ---- Images and video ----------------------------------------------------------------------------------------------------------
# The checkpoint's vision encoder (hived GlmVision) takes one temporal patch per span: an image repeated twice, or two video frames.
#   Preprocessing follows the model's processor_config.json (patch 14, 2×2 merge, temporal patch 2, CLIP mean/std, image tokens 16-8000,
#   video 2 fps and up to 240,000 tokens); the code is this project's own:
#   1. size: keep the aspect ratio, shrink (never enlarge, unless below the minimum) so that frames · h · w fits the token budget, then
#      pad right/bottom with black to multiples of 28 (patch × merge);
#   2. normalize: /255, (x − mean)/std per channel;
#   3. patchify: rows in 2×2 merge-block order (block row, block column, then the 4 patches of the block), each row the flattened
#      (channel, frame, 14, 14) values — the order of the encoder's 3D convolution weights.
#   Prompt: each image placeholder becomes N image tokens (N = patches / 4); a video becomes, per pair of frames,
#   <|begin_of_image|> N image tokens <|end_of_image|> followed by the pair's start time as "t.t seconds".
MEDIA_TYPES = ("image", "image_url", "video", "video_url")
IMAGE_TOKEN, VIDEO_TOKEN = "<|image|>", "<|video|>"
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)
_PATCH, _MERGE, _TPATCH = 14, 2, 2
_FACTOR = _PATCH * _MERGE
_IMG_TOKENS = (16, 8000)
_VID_TOKENS = (16, 240000)
_VID_FPS = 2.0
_SPAN_MAX_TOKENS = 8000  # per span (hived admits at most 8000 · 4 patches per image span)


def _part_url(p: dict) -> str:
    t = p.get("type")
    v = p.get("image_url") if t == "image_url" else p.get("video_url") if t == "video_url" else p.get(t)
    if isinstance(v, dict):
        v = v.get("url")
    if v is None and isinstance(p.get("source"), dict):  # Anthropic style {"type": "image", "source": {...}}
        src = p["source"]
        if src.get("data") is not None:
            return f"data:{src.get('media_type', 'application/octet-stream')};base64,{src['data']}"
        v = src.get("url")
    if not isinstance(v, str) or not v:
        raise ValueError(f"{t} part without a URL")
    return v


def _fetch(url: str) -> bytes:
    if url.startswith("data:"):
        head, _, data = url.partition(",")
        return base64.b64decode(data) if head.endswith(";base64") else data.encode()
    if url.startswith(("http://", "https://")) and os.environ.get("HIVE_IMAGE_FETCH", "1") not in ("", "0"):
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    raise ValueError("media URL must be a data: URL" + (" or an http(s) URL" if os.environ.get("HIVE_IMAGE_FETCH", "1") not in ("", "0") else ""))


def _target_size(frames: int, h: int, w: int, tokens: tuple[int, int]) -> tuple[int, int]:
    """Canvas (height, width), multiples of 28, whose frames·h·w stays within the token budget."""
    per = _TPATCH * _FACTOR ** 2
    lo, hi = tokens[0] * per, tokens[1] * per
    al = lambda v: math.ceil(v / _FACTOR) * _FACTOR  # noqa: E731
    af = max(_TPATCH, round(frames / _TPATCH) * _TPATCH)
    th, tw = al(h), al(w)
    if af * th * tw < lo:
        k = math.sqrt(lo / (frames * h * w))
        th, tw = al(max(1, math.ceil(h * k))), al(max(1, math.ceil(w * k)))
    if af * th * tw > hi:
        best, a, b = (_FACTOR, _FACTOR), 1, h
        while a <= b:
            ch = (a + b) // 2
            cw = max(1, math.floor(w * ch / h))
            if af * al(ch) * al(cw) <= hi:
                best, a = (al(ch), al(cw)), ch + 1
            else:
                b = ch - 1
        th, tw = best
    return th, tw


def _fit(frames: list, tokens: tuple[int, int]):
    """Resize (BICUBIC, aspect kept) and pad a list of same-size PIL RGB frames → float32 [T, 3, H, W] normalized."""
    import numpy as np
    from PIL import Image

    w, h = frames[0].size
    th, tw = _target_size(len(frames), h, w, tokens)
    per = _TPATCH * _FACTOR ** 2
    k = min(th / h, tw / w)
    if len(frames) * h * w >= per * tokens[0]:
        k = min(1.0, k)
    ch, cw = max(1, min(th, math.floor(h * k))), max(1, min(tw, math.floor(w * k)))
    out = np.zeros((len(frames), 3, th, tw), dtype=np.float32)
    mean = np.array(_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(_STD, dtype=np.float32)[:, None, None]
    pad = (0.0 - mean) / std
    for i, f in enumerate(frames):
        if (cw, ch) != (w, h):
            f = f.resize((cw, ch), Image.BICUBIC)
        a = np.asarray(f, dtype=np.float32).transpose(2, 0, 1) / 255.0
        out[i] = pad
        out[i, :, :ch, :cw] = (a - mean) / std
    return out


def _patchify(pair) -> tuple[Any, int, int]:
    """[2, 3, H, W] → ([gh·gw, 3·2·14·14] float32 in merge-block order, gh, gw)."""
    t, c, H, W = pair.shape
    gh, gw = H // _PATCH, W // _PATCH
    x = pair.reshape(t, c, gh // _MERGE, _MERGE, _PATCH, gw // _MERGE, _MERGE, _PATCH)
    x = x.transpose(2, 5, 3, 6, 1, 0, 4, 7)  # block row, block col, row in block, col in block, channel, frame, py, px
    return x.reshape(gh * gw, c * t * _PATCH * _PATCH), gh, gw


def _image_spans(data: bytes):
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    x = _fit([img], _IMG_TOKENS)
    pair = __import__("numpy").concatenate([x, x], axis=0)  # an image is one temporal patch: the same frame twice
    return [(_patchify(pair), None)]


def _video_frames(data: bytes):
    """Decoded frames sampled at 2 fps (PIL RGB) and their times in seconds. Needs PyAV."""
    try:
        import av  # noqa
    except ImportError as e:
        raise ValueError("video input needs PyAV (pip install av) in the server image") from e
    with av.open(io.BytesIO(data)) as box:
        st = box.streams.video[0]
        fps = float(st.average_rate or st.guessed_rate or 24)
        frames = [f for f in box.decode(st)]
    if not frames:
        raise ValueError("video has no frames")
    times = [float(f.time) if f.time is not None else i / fps for i, f in enumerate(frames)]
    duration = max(times[-1] + 1.0 / fps, 1.0 / fps)
    want = max(1, int(duration * _VID_FPS))
    picked, nxt = [], 0.0
    for i, t in enumerate(times):  # the first frame at or after each 0.5 s step
        if t >= nxt:
            picked.append(i)
            nxt += 1.0 / _VID_FPS
    if len(picked) > want:
        step = len(picked) / want
        picked = [picked[int(j * step)] for j in range(want)]
    return [frames[i].to_image().convert("RGB") for i in picked], [times[i] for i in picked]


def _video_spans(data: bytes):
    import numpy as np

    frames, times = _video_frames(data)
    if len(frames) % 2:
        frames.append(frames[-1]); times.append(times[-1])
    x = _fit(frames, _VID_TOKENS)
    if (x.shape[2] // _PATCH) * (x.shape[3] // _PATCH) // 4 > _SPAN_MAX_TOKENS:  # keep each pair within one span's limit
        x = _fit(frames, (_IMG_TOKENS[0], _SPAN_MAX_TOKENS))
    return [(_patchify(np.ascontiguousarray(x[i:i + 2])), times[i]) for i in range(0, len(frames), 2)]


def prepare_inputs(prompt: str, media: list[dict], tok):
    """Prompt with expanded placeholders → (token ids, hived image metas, bf16 patch bytes) — the server's tokenize_with_images."""
    import torch

    spans_by_part = []
    for p in media:
        data = _fetch(_part_url(p))
        spans_by_part.append(_video_spans(data) if p.get("type") in ("video", "video_url") else _image_spans(data))
    out, pos, k = [], 0, 0
    pat = re.compile(re.escape(IMAGE_TOKEN) + "|" + re.escape(VIDEO_TOKEN))
    for mt in pat.finditer(prompt):
        if k >= len(spans_by_part):
            raise ValueError("more media placeholders than media parts")
        out.append(prompt[pos:mt.start()])
        spans = spans_by_part[k]
        if mt.group(0) == IMAGE_TOKEN:
            (_, gh, gw), _ = spans[0]
            out.append(IMAGE_TOKEN * (gh * gw // 4))
        else:
            for (_, gh, gw), t in spans:
                out.append(f"<|begin_of_image|>{IMAGE_TOKEN * (gh * gw // 4)}<|end_of_image|>{t:.1f} seconds")
        pos, k = mt.end(), k + 1
    if k != len(spans_by_part):
        raise ValueError("media parts and placeholders do not match")
    out.append(prompt[pos:])
    ids = tok.encode("".join(out))
    img_id = tok.convert_tokens_to_ids(IMAGE_TOKEN)
    flat = [s for spans in spans_by_part for s in spans]
    metas, blobs, i = [], [], 0
    for (patches, gh, gw), _ in flat:
        n = gh * gw // 4
        while i < len(ids) and ids[i] != img_id:
            i += 1
        if i + n > len(ids) or any(ids[j] != img_id for j in range(i, i + n)):
            raise ValueError("image token span not found in the encoded prompt")
        raw = torch.from_numpy(patches).to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
        metas.append({"start": i, "n_vit_h": gh, "n_vit_w": gw, "types": [1] * n, "nbytes": len(raw)})
        blobs.append(raw)
        i += n
    return ids, metas, b"".join(blobs)
