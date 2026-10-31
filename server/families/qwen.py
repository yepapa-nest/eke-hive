# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Qwen3.8-Flash-Next family adapter (config.json model_type "qwen4_exp") for hive_server.py — the same surface as server/families/glm.py
(encode_messages, parse_message_from_completion_text, prepare_inputs, effort_from_request) plus the family facts the server needs.

Prompt: the checkpoint's own chat_template.jinja through the tokenizer (apply_chat_template), unchanged.
Thinking: the template takes `enable_thinking` (false → the generation prompt ends with an empty thinking block "<think>\n\n</think>\n\n")
  and `reasoning_effort` in xhigh (template default) / medium / low. Request mapping:
    enable_thinking false, reasoning_effort none / minimal → non-thinking ("chat")
    low → low · medium → medium · high / xhigh / max → xhigh · numbers 0-33 / 34-66 / 67+ → low / medium / xhigh · unset or unknown → xhigh
Stop ids: generation_config.json eos_token_id (<|im_end|>, <|endoftext|>), else config.json's eos.
Tool calls (the template's own instruction):
  <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>
  A value is the text between the parameter tags minus one leading and one trailing newline; typed with the request's schemas as in
  glm.py (_arg_value): a string-only parameter keeps the raw text, otherwise finite JSON when it parses, else the raw text.
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
FAMILY = "qwen4_exp"

STOP_IDS: list[int] = []
EOS_TEXTS: list[str] = []

_TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.S)


def init(tok, cfg: dict, ckpt: str | None = None) -> None:
    global TOK, STOP_IDS, EOS_TEXTS
    TOK = tok
    eos = None
    if ckpt:
        try:
            with open(os.path.join(ckpt, "generation_config.json")) as f:
                eos = json.load(f).get("eos_token_id")
        except (OSError, ValueError):
            eos = None
    if eos is None:
        t = cfg.get("text_config", cfg)
        eos = cfg.get("eos_token_id", t.get("eos_token_id"))
    STOP_IDS = [int(x) for x in (eos if isinstance(eos, list) else [eos])] if eos is not None else [tok.eos_token_id]
    EOS_TEXTS = [s for s in (tok.convert_ids_to_tokens(i) for i in STOP_IDS) if isinstance(s, str) and s]


def nothink_mode() -> str:
    return "enable_thinking=false (empty thinking block)"


def effort_from_request(body: dict) -> tuple[str, Any]:
    """(thinking_mode, reasoning_effort): ("chat", None) or ("thinking", "low" | "medium" | "xhigh")."""
    kw = body.get("chat_template_kwargs")
    kw = kw if isinstance(kw, dict) else {}
    thinking = kw.get("thinking", kw.get("enable_thinking"))
    effort = kw.get("reasoning_effort", body.get("reasoning_effort"))
    if isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort", effort)
    if isinstance(effort, bool):  # a bool is not an effort
        effort = None
    if thinking is False or effort in ("none", "minimal"):
        return "chat", None
    if isinstance(effort, (int, float)) or (isinstance(effort, str) and re.fullmatch(r"\d+(\.\d+)?", effort or "")):
        x = float(effort)
        effort = "low" if x < 34 else ("medium" if x < 67 else "xhigh")
    effort = {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}.get(effort, "xhigh")
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
            continue  # the server's tool carrier (empty system message) — the template takes tools as an argument
        if m.get("role") == "system" and msgs:
            # The template takes a system message only at the start (it raises "System message must be at the beginning."); a
            #   later one — a client's mid-conversation notice, e.g. a tool announcement whose tools encode_prompt already moved
            #   to the head — becomes a user note at its place, as clients do on wires without mid-conversation system messages.
            c = m.get("content")
            if isinstance(c, list):   # keep image / video parts (the template takes them in a user message); prefix the note to the first text
                parts = [dict(p) if isinstance(p, dict) else {"type": "text", "text": str(p)} for p in c]
                texts = [p for p in parts if p.get("type") in ("text", "input_text")]
                if texts:
                    texts[0]["text"] = "[system] " + (texts[0].get("text") or "")
                else:
                    parts.insert(0, {"type": "text", "text": "[system]"})
                m = {"role": "user", "content": parts}
            else:
                text = _content_text(c).strip()
                if not text:
                    continue   # an empty notice carries nothing — the template would get a bare "[system]" line
                m = {"role": "user", "content": "[system] " + text}
        if isinstance(m.get("content"), list):
            if any(isinstance(p, dict) and p.get("type") in MEDIA_TYPES for p in m["content"]):
                # keep the list: the template writes one vision placeholder per image / video part (expanded in prepare_inputs)
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
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if thinking_mode == "chat":
        kwargs["enable_thinking"] = False
    else:
        kwargs["enable_thinking"] = True
        kwargs["reasoning_effort"] = reasoning_effort if reasoning_effort in ("low", "medium", "xhigh") else "xhigh"
    if tools:
        kwargs["tools"] = tools
    prompt = TOK.apply_chat_template(msgs, **kwargs)
    if media and not return_multi_modal_data:
        raise ValueError("image/video parts need the multimodal path")
    return (prompt, {"images": media}) if return_multi_modal_data else prompt


# ---- Completion parsing ----------------------------------------------------------------------------------------------------------------
def _reject_constant(name: str):
    raise ValueError(f"non-finite JSON constant {name}")


def _finite(x) -> bool:
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, list):
        return all(_finite(y) for y in x)
    if isinstance(x, dict):
        return all(_finite(y) for y in x.values())
    return True


def _schema_types(sch) -> set:
    """JSON types a parameter schema allows ("type" as a string or a list, or the branches of anyOf / oneOf); empty = unknown."""
    if not isinstance(sch, dict):
        return set()
    t = sch.get("type")
    out = {t} if isinstance(t, str) else {x for x in t if isinstance(x, str)} if isinstance(t, list) else set()
    for k in ("anyOf", "oneOf"):
        for b in sch.get(k) or []:
            out |= _schema_types(b)
    if not out and isinstance(sch.get("enum"), list) and sch["enum"] and all(isinstance(e, str) for e in sch["enum"]):
        out = {"string"}
    return out


def tool_param_schemas(tools) -> dict:
    """{tool name: {argument name: parameter schema}} from a request's tools (OpenAI {type, function} or bare {name, parameters})."""
    out = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        props = ((fn.get("parameters") or fn.get("input_schema") or {}) if isinstance(fn, dict) else {}).get("properties")
        if isinstance(fn.get("name"), str) and isinstance(props, dict):
            out[fn["name"]] = props
    return out


def _arg_value(v: str, schema=None):
    """One parameter value (raw text without the framing newlines). The format carries no type marker — the template writes strings
    raw and everything else with tojson — so: a string-only parameter keeps the text; otherwise JSON is tried, but NaN / Infinity and
    non-finite numbers keep the text, so the arguments are always valid JSON (the same rule as glm.py, for the same reason)."""
    types = _schema_types(schema) - {"null"}
    if types == {"string"}:
        return v
    s = v.strip()
    try:
        x = json.loads(s, parse_constant=_reject_constant)
    except ValueError:
        return v
    return x if _finite(x) else v


def _unframe(v: str) -> str:
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def parse_tool_calls(text: str, tools=None) -> list[dict]:
    schemas = tool_param_schemas(tools)
    calls = []
    for name, body in _TOOL_RE.findall(text):
        name = name.strip()
        props = schemas.get(name) or {}
        args = {}
        for k, v in _PARAM_RE.findall(body):
            k = k.strip()
            args[k] = _arg_value(_unframe(v), props.get(k))
        if name:
            calls.append({"type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False, allow_nan=False)}})
    return calls


def parse_message_from_completion_text(text: str, thinking_mode: str = "thinking", tools=None) -> dict:
    """tools: the request's tool list — argument types come from its schemas (see _arg_value)."""
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
    calls = parse_tool_calls(tool_part, tools) if tool_part else []
    if calls:
        out["tool_calls"] = calls
    return out


# ---- Images and video ----------------------------------------------------------------------------------------------------------------------
# The vision encoder (hived QwenVision) takes one temporal patch per span: an image repeated twice, or two consecutive video frames.
#   Preprocessing follows the checkpoint's preprocessor_config.json / video_preprocessor_config.json (patch 16, 2×2 merge, temporal
#   patch 2, mean / std 0.5, rescale 1/255, bicubic resize); the code is this project's own:
#   1. size ("smart resize", factor 32 = patch × merge): each side rounded to a multiple of 32 (at least 32); if the area exceeds the
#      maximum, both sides are scaled by sqrt(h·w / max) and floored to multiples of 32; if below the minimum, scaled up and ceiled.
#      Images: min 65,536 px; max = min(16,777,216 (config), HIVE_QWEN_IMAGE_MAX_PIXELS (default 8,388,608), 8,000 merged tokens · 1,024 px)
#      — 8,000 merged tokens per span is hived's admission limit (one merged token = 32 × 32 px), so it is the binding cap by default.
#      Videos: frames sampled at 2 fps (4 to 768 frames, an even count — the last frame repeats), each frame min 4,096 px and
#      max = min(8,000 · 1,024, 25,165,824 / frames) px (the config's video budget spread over the frames, within the span limit).
#      Aspect ratio above 200 → error.
#   2. normalize: /255, then (x − 0.5) / 0.5.
#   3. patchify: rows in 2×2 merge-block order (block row, block column, then the 4 patches of the block row-major), each row the
#      flattened (channel 3, frame 2, 16, 16) values — the order of the encoder's 3D patch-embedding weights.
#   Prompt: the template writes <|vision_start|><|image_pad|><|vision_end|> per image; the single pad becomes N = patches / 4 pads.
#   A video's <|vision_start|><|video_pad|><|vision_end|> becomes, per pair of frames, "<t seconds>" (t = the mean of the pair's two
#   timestamps, one decimal) + <|vision_start|> + N video pads + <|vision_end|>.
MEDIA_TYPES = ("image", "image_url", "video", "video_url")
IMAGE_PAD, VIDEO_PAD = "<|image_pad|>", "<|video_pad|>"
VISION_START, VISION_END = "<|vision_start|>", "<|vision_end|>"
_IMAGE_PH = VISION_START + IMAGE_PAD + VISION_END
_VIDEO_PH = VISION_START + VIDEO_PAD + VISION_END
_PATCH, _MERGE, _TPATCH = 16, 2, 2
_FACTOR = _PATCH * _MERGE
_SPAN_MAX_TOKENS = 8000                   # hived's per-span admission limit (merged tokens)
_IMG_MIN_PX, _IMG_MAX_PX_CFG = 65536, 16777216
_VID_MIN_PX, _VID_TOTAL_PX = 4096, 25165824
_VID_FPS, _VID_MIN_FRAMES, _VID_MAX_FRAMES = 2.0, 4, 768
_MAX_RATIO = 200


def image_max_pixels() -> int:
    try:
        v = int(os.environ.get("HIVE_QWEN_IMAGE_MAX_PIXELS") or 8388608)
    except ValueError:
        v = 8388608
    return max(_FACTOR * _FACTOR, min(_IMG_MAX_PX_CFG, v, _SPAN_MAX_TOKENS * _FACTOR * _FACTOR))


def smart_size(h: int, w: int, min_px: int, max_px: int) -> tuple[int, int]:
    """(height, width) multiples of 32 near (h, w) with min_px ≤ area ≤ max_px (see the rule above)."""
    if min(h, w) <= 0:
        raise ValueError("empty image")
    if max(h, w) / min(h, w) > _MAX_RATIO:
        raise ValueError(f"image aspect ratio must be at most {_MAX_RATIO}, got {max(h, w) / min(h, w):.1f}")
    f = _FACTOR
    hb, wb = max(f, round(h / f) * f), max(f, round(w / f) * f)
    if hb * wb > max_px:
        beta = math.sqrt(h * w / max_px)
        hb, wb = max(f, math.floor(h / beta / f) * f), max(f, math.floor(w / beta / f) * f)
    elif hb * wb < min_px:
        beta = math.sqrt(min_px / (h * w))
        hb, wb = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
    return hb, wb


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


def _normalize(frames: list, th: int, tw: int):
    """PIL RGB frames → float32 [T, 3, th, tw]: bicubic resize to (th, tw), /255, (x − 0.5) / 0.5."""
    import numpy as np
    from PIL import Image

    out = np.empty((len(frames), 3, th, tw), dtype=np.float32)
    for i, f in enumerate(frames):
        if f.size != (tw, th):
            f = f.resize((tw, th), Image.BICUBIC)
        a = np.asarray(f, dtype=np.float32).transpose(2, 0, 1) / 255.0
        out[i] = (a - 0.5) / 0.5
    return out


def patchify(pair) -> tuple[Any, int, int]:
    """[2, 3, H, W] → ([gh·gw, 3·2·16·16] float32 in merge-block order, gh, gw)."""
    t, c, H, W = pair.shape
    gh, gw = H // _PATCH, W // _PATCH
    x = pair.reshape(t, c, gh // _MERGE, _MERGE, _PATCH, gw // _MERGE, _MERGE, _PATCH)
    x = x.transpose(2, 5, 3, 6, 1, 0, 4, 7)  # block row, block col, row in block, col in block, channel, frame, py, px
    return x.reshape(gh * gw, c * t * _PATCH * _PATCH), gh, gw


def _image_spans(data: bytes):
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"cannot identify image: {len(data)} bytes, starting {data[:16].hex() or '(empty)'}") from e
    w, h = img.size
    th, tw = smart_size(h, w, _IMG_MIN_PX, image_max_pixels())
    x = _normalize([img], th, tw)
    pair = np.concatenate([x, x], axis=0)  # an image is one temporal patch: the same frame twice
    return [(patchify(pair), None)]


def sample_frame_indices(times: list[float], fps: float) -> list[int]:
    """Indices of the frames kept at _VID_FPS: the first frame at or after each 0.5 s step, then evenly thinned to at most
    _VID_MAX_FRAMES, padded (last index repeated) to at least _VID_MIN_FRAMES and to an even count."""
    duration = max(times[-1] + 1.0 / fps, 1.0 / fps)
    want = max(1, int(duration * _VID_FPS))
    picked, nxt = [], 0.0
    for i, t in enumerate(times):
        if t >= nxt:
            picked.append(i)
            nxt += 1.0 / _VID_FPS
    if len(picked) > want:
        step = len(picked) / want
        picked = [picked[int(j * step)] for j in range(want)]
    if len(picked) > _VID_MAX_FRAMES:
        step = len(picked) / _VID_MAX_FRAMES
        picked = [picked[int(j * step)] for j in range(_VID_MAX_FRAMES)]
    while len(picked) < _VID_MIN_FRAMES or len(picked) % 2:
        picked.append(picked[-1])
    return picked


def _video_frames(data: bytes):
    """Sampled frames (PIL RGB) and their times in seconds. Needs PyAV."""
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
    picked = sample_frame_indices(times, fps)
    return [frames[i].to_image().convert("RGB") for i in picked], [times[i] for i in picked]


def video_frame_max_pixels(n_frames: int) -> int:
    return max(_VID_MIN_PX, min(_SPAN_MAX_TOKENS * _FACTOR * _FACTOR, _VID_TOTAL_PX // max(1, n_frames)))


def _video_spans(data: bytes):
    import numpy as np

    frames, times = _video_frames(data)
    w, h = frames[0].size
    th, tw = smart_size(h, w, _VID_MIN_PX, video_frame_max_pixels(len(frames)))
    x = _normalize(frames, th, tw)
    return [(patchify(np.ascontiguousarray(x[i:i + 2])), (times[i] + times[i + 1]) / 2.0) for i in range(0, len(frames), 2)]


def expand_prompt(prompt: str, spans_by_part: list) -> str:
    """The prompt with each image / video placeholder expanded for its spans (spans: [((patches, gh, gw), t), ...] per media part)."""
    out, pos, k = [], 0, 0
    pat = re.compile(re.escape(_IMAGE_PH) + "|" + re.escape(_VIDEO_PH))
    for mt in pat.finditer(prompt):
        if k >= len(spans_by_part):
            raise ValueError("more media placeholders than media parts")
        out.append(prompt[pos:mt.start()])
        spans = spans_by_part[k]
        if mt.group(0) == _IMAGE_PH:
            (_, gh, gw), _ = spans[0]
            out.append(VISION_START + IMAGE_PAD * (gh * gw // 4) + VISION_END)
        else:
            for (_, gh, gw), t in spans:
                out.append(f"<{t:.1f} seconds>" + VISION_START + VIDEO_PAD * (gh * gw // 4) + VISION_END)
        pos, k = mt.end(), k + 1
    if k != len(spans_by_part):
        raise ValueError("media parts and placeholders do not match")
    out.append(prompt[pos:])
    return "".join(out)


def span_metas(ids: list[int], spans: list, kinds: list[int]) -> list[dict]:
    """hived metas for the spans in order: kinds[i] = the pad token id of span i; each span = n consecutive pads at its start."""
    metas, i = [], 0
    for ((_, gh, gw), _), pad in zip(spans, kinds):
        n = gh * gw // 4
        while i < len(ids) and ids[i] != pad:
            i += 1
        if i + n > len(ids) or any(ids[j] != pad for j in range(i, i + n)):
            raise ValueError("vision token span not found in the encoded prompt")
        metas.append({"start": i, "n_vit_h": gh, "n_vit_w": gw, "types": [1] * n})
        i += n
    return metas


def _bf16_bytes(patches) -> bytes:
    import torch

    return torch.from_numpy(patches).to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()


# ---- one image at a time (the server's MediaTokens: prompt through its token cache, each image preprocessed once per content) ----
def image_part_bytes(p: dict) -> bytes:
    """The encoded bytes of an image part (data: or http(s) URL)."""
    return _fetch(_part_url(p))


def image_part_patches(data: bytes) -> tuple[bytes, int, int]:
    """(bf16 patch bytes, gh, gw) of one image — the same bytes prepare_inputs sends for it; its span = gh·gw / 4 image pads."""
    (patches, gh, gw), _ = _image_spans(data)[0]
    return _bf16_bytes(patches), gh, gw


def image_cache_key():
    """What besides the image bytes decides its patches (part of the server's per-image cache key)."""
    return ("qwen", image_max_pixels())


def prepare_inputs(prompt: str, media: list[dict], tok):
    """Prompt with expanded placeholders → (token ids, hived image metas, bf16 patch bytes) — the server's tokenize_with_images."""
    spans_by_part, kinds = [], []
    img_id, vid_id = tok.convert_tokens_to_ids(IMAGE_PAD), tok.convert_tokens_to_ids(VIDEO_PAD)
    for p in media:
        data = _fetch(_part_url(p))
        video = p.get("type") in ("video", "video_url")
        spans = _video_spans(data) if video else _image_spans(data)
        spans_by_part.append(spans)
        kinds += [vid_id if video else img_id] * len(spans)
    ids = tok.encode(expand_prompt(prompt, spans_by_part))
    flat = [s for spans in spans_by_part for s in spans]
    metas = span_metas(ids, flat, kinds)
    blobs = []
    for m, ((patches, _, _), _) in zip(metas, flat):
        raw = _bf16_bytes(patches)
        m["nbytes"] = len(raw)
        blobs.append(raw)
    return ids, metas, b"".join(blobs)
