# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Qwen3.8-27B family adapter (config.json model_type "qwen3_5") for hive_server.py — the same surface as server/families/qwen.py.

Qwen3.8-27B ships the same prompt contract as Qwen3.8-Flash-Next (checked 2026-10-09 against both chat_template.jinja files): the template
takes `enable_thinking` and `reasoning_effort` (xhigh default / medium / low; 27B also accepts "high" = xhigh), tool calls use the same
<tool_call><function=NAME><parameter=KEY>…</parameter></function></tool_call> format, images / videos use the same
<|vision_start|><|image_pad|><|vision_end|> placeholders, and preprocessor_config.json has the same values (patch 16, merge 2, temporal 2,
mean / std 0.5, 65,536 … 16,777,216 pixels). Template differences (several leading system / developer messages merged into one; no error
for a conversation without a user turn) are handled by the template itself, which is applied unchanged through the tokenizer.

So this module is server/families/qwen.py loaded as its own module instance (its own TOK / STOP_IDS globals, so both families can be
imported in one process) with FAMILY = "qwen3_5". Stop ids come from generation_config.json (<|im_end|> 248046, <|endoftext|> 248044).
HIVE_QWEN_IMAGE_MAX_PIXELS caps the image size, as for Flash-Next.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location("families._q27_qwen_base", os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen.py"))
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)
_base.FAMILY = "qwen3_5"

FAMILY = "qwen3_5"
THINK_END = _base.THINK_END
TOOL_START = _base.TOOL_START
STOP_IDS: list[int] = []
EOS_TEXTS: list[str] = []


def init(tok, cfg: dict, ckpt: str | None = None) -> None:
    global STOP_IDS, EOS_TEXTS
    _base.init(tok, cfg, ckpt)
    STOP_IDS, EOS_TEXTS = _base.STOP_IDS, _base.EOS_TEXTS


def __getattr__(name):  # everything else (encode_messages, parse_message_from_completion_text, prepare_inputs, …) is qwen.py's
    return getattr(_base, name)
