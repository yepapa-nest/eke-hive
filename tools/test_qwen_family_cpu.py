#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Qwen3.8-Flash-Next family rules (server/families/qwen.py), no model and no tokenizer download: effort mapping, the prompt kwargs
handed to the chat template, tool-call parsing, stop ids, image sizing / patch order / prompt expansion and span metas."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from families import qwen  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    fails += not cond
    print(f"{'ok  ' if cond else 'FAIL'} {what}")


def check(body, want):
    got = qwen.effort_from_request(body)
    ok(got == want, f"{body} -> {got}" + ("" if got == want else f" (want {want})"))


# ---- effort ----
check({}, ("thinking", "xhigh"))
check({"reasoning_effort": "weird"}, ("thinking", "xhigh"))
check({"reasoning_effort": "none"}, ("chat", None))
check({"reasoning_effort": "minimal"}, ("chat", None))
check({"chat_template_kwargs": {"enable_thinking": False}}, ("chat", None))
check({"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}, ("thinking", "low"))
check({"reasoning_effort": "low"}, ("thinking", "low"))
check({"reasoning_effort": "medium"}, ("thinking", "medium"))
check({"reasoning_effort": "high"}, ("thinking", "xhigh"))
check({"reasoning_effort": "max"}, ("thinking", "xhigh"))
check({"reasoning": {"effort": "medium"}}, ("thinking", "medium"))
check({"reasoning_effort": 10}, ("thinking", "low"))
check({"reasoning_effort": "50"}, ("thinking", "medium"))
check({"reasoning_effort": 90}, ("thinking", "xhigh"))
check({"reasoning_effort": True}, ("thinking", "xhigh"))


# ---- fake tokenizer: records the template kwargs; special tokens are single ids, other text one id per character ----
class FakeTok:
    SPECIAL = {"<|endoftext|>": 248044, "<|im_end|>": 248046, "<|vision_start|>": 248053, "<|vision_end|>": 248054,
               "<|image_pad|>": 248056, "<|video_pad|>": 248057}

    def __init__(self):
        self.calls = []
        self.eos_token_id = 248046
        self.rev = {v: k for k, v in self.SPECIAL.items()}

    def apply_chat_template(self, msgs, **kw):
        self.calls.append((msgs, kw))
        return "PROMPT" + "".join(
            ("<|vision_start|><|image_pad|><|vision_end|>" if p.get("type") == "image" else "<|vision_start|><|video_pad|><|vision_end|>")
            for m in msgs if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") in ("image", "video"))

    def convert_ids_to_tokens(self, i):
        return self.rev.get(i)

    def convert_tokens_to_ids(self, t):
        return self.SPECIAL[t]

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            for s, v in self.SPECIAL.items():
                if text.startswith(s, i):
                    out.append(v); i += len(s); break
            else:
                out.append(ord(text[i])); i += 1
        return out


tok = FakeTok()
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "generation_config.json"), "w") as f:
        json.dump({"eos_token_id": [248046, 248044]}, f)
    qwen.init(tok, {"text_config": {"eos_token_id": 248044}}, d)
ok(qwen.STOP_IDS == [248046, 248044], f"stop ids from generation_config.json {qwen.STOP_IDS}")
ok(qwen.EOS_TEXTS == ["<|im_end|>", "<|endoftext|>"], f"eos texts {qwen.EOS_TEXTS}")
qwen.init(tok, {"text_config": {"eos_token_id": 248044}})
ok(qwen.STOP_IDS == [248044], "stop ids fall back to config eos")
qwen.init(tok, {}, None)

tools = [{"type": "function", "function": {"name": "f", "parameters": {"properties": {"a": {"type": "string"}, "n": {"type": "integer"}}}}}]
qwen.encode_messages([{"role": "system", "content": "", "tools": tools}, {"role": "user", "content": "hi"}], thinking_mode="chat")
msgs, kw = tok.calls[-1]
ok(kw.get("enable_thinking") is False and "reasoning_effort" not in kw and kw.get("tools") == tools and msgs[0]["role"] == "user",
   f"chat mode: enable_thinking False, tools as argument, carrier dropped ({kw})")
qwen.encode_messages([{"role": "user", "content": "hi"}], thinking_mode="thinking", reasoning_effort="low")
ok(tok.calls[-1][1].get("enable_thinking") is True and tok.calls[-1][1].get("reasoning_effort") == "low", "thinking low")
qwen.encode_messages([{"role": "user", "content": "hi"}], thinking_mode="thinking", reasoning_effort=75)
ok(tok.calls[-1][1].get("reasoning_effort") == "xhigh", "unknown effort value in encode -> xhigh")
qwen.encode_messages([{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": '{"a": "x"}'}}]},
                      {"role": "user", "content": "go"}])
ok(tok.calls[-1][0][0]["tool_calls"][0]["function"]["arguments"] == {"a": "x"}, "assistant tool-call arguments decoded for the template")
qwen.encode_messages([{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
                      {"role": "system", "content": "New tools are available from this point of the conversation: f."},
                      {"role": "user", "content": "again"}], thinking_mode="chat")
msgs = tok.calls[-1][0]
ok([m["role"] for m in msgs] == ["system", "user", "assistant", "user", "user"] and msgs[0]["content"] == "S" and
   msgs[3]["content"] == "[system] New tools are available from this point of the conversation: f.",
   f"later system message -> user note at its place, first system kept ({[m['role'] for m in msgs]})")

# ---- completion parsing ----
qwen.EOS_TEXTS = ["<|im_end|>"]
text = ("let me see</think>\n\nOK.\n\n<tool_call>\n<function=f>\n<parameter=a>\n3e382151\n</parameter>\n<parameter=n>\n7\n</parameter>\n"
        "</function>\n</tool_call>\n<tool_call>\n<function=g>\n<parameter=body>\nline1\nline2\n</parameter>\n</function>\n</tool_call><|im_end|>")
m = qwen.parse_message_from_completion_text(text, "thinking", tools)
calls = m.get("tool_calls") or []
ok(m.get("reasoning_content") == "let me see" and m.get("content") == "OK.", f"reasoning / content split {m.get('reasoning_content')!r} {m.get('content')!r}")
ok(len(calls) == 2 and json.loads(calls[0]["function"]["arguments"]) == {"a": "3e382151", "n": 7}, f"typed arguments {calls[:1]}")
ok(json.loads(calls[1]["function"]["arguments"]) == {"body": "line1\nline2"}, "multi-line value keeps inner newlines")
m = qwen.parse_message_from_completion_text("<tool_call>\n<function=h>\n<parameter=x>\nNaN\n</parameter>\n</function>\n</tool_call>", "chat", None)
ok(json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"x": "NaN"}, "non-finite JSON stays text")
m = qwen.parse_message_from_completion_text("still thinking", "thinking", None)
ok(m == {"reasoning_content": "still thinking", "content": ""}, "unfinished thinking")

# ---- images ----
mx = qwen.image_max_pixels()
ok(mx == 8000 * 1024, f"default image max pixels = the 8,000-token span limit ({mx})")
h, w = qwen.smart_size(360, 640, 65536, mx)
ok(h % 32 == 0 and w % 32 == 0 and h * w >= 65536 and h * w <= mx, f"360x640 -> {h}x{w}")
h, w = qwen.smart_size(4000, 6000, 65536, mx)
ok(h % 32 == 0 and w % 32 == 0 and h * w <= mx and (h // 16) * (w // 16) // 4 <= 8000, f"4000x6000 -> {h}x{w}")
h, w = qwen.smart_size(10, 20, 65536, mx)
ok(h * w >= 65536 and h % 32 == 0, f"tiny 10x20 -> {h}x{w}")
try:
    qwen.smart_size(10, 3000, 65536, mx)
    ok(False, "aspect ratio > 200 rejected")
except ValueError:
    ok(True, "aspect ratio > 200 rejected")

try:
    import numpy as np
except ImportError:
    np = None
if np is not None:
    H, W = 64, 96
    pair = np.random.default_rng(0).standard_normal((2, 3, H, W)).astype(np.float32)
    rows, gh, gw = qwen.patchify(pair)
    ok(rows.shape == (gh * gw, 1536) and (gh, gw) == (4, 6), f"patch rows {rows.shape}")
    # row r: merge block (br, bc), inner (ir, ic) → patch (py, px) = (2·br + ir, 2·bc + ic); row = (channel, frame, 16, 16)
    r = (1 * (gw // 2) + 2) * 4 + 1 * 2 + 0  # block row 1, block col 2, inner row 1, inner col 0
    py, px = 3, 4
    want = pair[:, :, py * 16:(py + 1) * 16, px * 16:(px + 1) * 16].transpose(1, 0, 2, 3).reshape(-1)
    ok(np.array_equal(rows[r], want), "patch row order (merge block, inner row-major, then channel, frame, py, px)")
    times = [i / 30 for i in range(30 * 3)]  # 3 s at 30 fps
    idx = qwen.sample_frame_indices(times, 30.0)
    ok(len(idx) == 6 and len(idx) % 2 == 0, f"3 s video at 2 fps -> {len(idx)} frames")
    idx = qwen.sample_frame_indices([0.0], 30.0)
    ok(len(idx) == 4, f"one-frame video padded to 4 ({len(idx)})")
    ok(qwen.video_frame_max_pixels(768) == 25165824 // 768, "video per-frame budget")

# ---- prompt expansion and metas ----
spans_img = [((None, 4, 6), None)]
spans_vid = [((None, 2, 4), 0.25), ((None, 2, 4), 1.25)]
p = "A<|vision_start|><|image_pad|><|vision_end|>B<|vision_start|><|video_pad|><|vision_end|>C"
e = qwen.expand_prompt(p, [spans_img, spans_vid])
want = ("A<|vision_start|>" + "<|image_pad|>" * 6 + "<|vision_end|>B<0.2 seconds><|vision_start|>" + "<|video_pad|>" * 2 +
        "<|vision_end|><1.2 seconds><|vision_start|>" + "<|video_pad|>" * 2 + "<|vision_end|>C")
ok(e == want, f"expanded prompt {e!r}")
ids = tok.encode(e)
metas = qwen.span_metas(ids, spans_img + spans_vid, [248056, 248057, 248057])
ok([mm["start"] for mm in metas] == [2, ids.index(248057), ids.index(248057) + 2 + 1 + len("<1.2 seconds>") + 1] and
   [len(mm["types"]) for mm in metas] == [6, 2, 2], f"metas {[(mm['start'], len(mm['types'])) for mm in metas]}")
try:
    qwen.expand_prompt(p, [spans_img])
    ok(False, "placeholder / media count mismatch rejected")
except ValueError:
    ok(True, "placeholder / media count mismatch rejected")

# ---- answer-span Han / kana filter: request switch (server/hive_server.py hanja_filter_wanted) and the token classes of tools/build_cjk_mask.py ----
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("server_under_test", Path(__file__).resolve().parents[1] / "server/hive_server.py")
srv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srv)
srv.FAMILY = "qwen4_exp"
for body, hdr, want in [({}, {}, False), ({"hanja_filter": True}, {}, True), ({"hanja_filter": False}, {"x-hanja-filter": "1"}, False),
                        ({"chat_template_kwargs": {"hanja_filter": 1}}, {}, True), ({}, {"x-hanja-filter": "0"}, False),
                        ({"hanja_filter": "maybe"}, {"x-hanja-filter": "1"}, True), ({"hanja_filter": None}, {}, False)]:
    got = srv.hanja_filter_wanted(body, hdr)
    ok(got == want, f"hanja_filter {body} {hdr} -> {got}")
ok(srv.hanja_filter_mode({"hanja_filter": True, "hanja_penalty": 5}, {}) == ("on", 5.0), "hanja_penalty 5")
ok(srv.hanja_filter_mode({}, {"x-hanja-filter": "1"})[1] == float("inf"), "default penalty inf")
ok(srv.hanja_filter_mode({"hanja_filter": True, "hanja_penalty": -2}, {})[1] == float("inf"), "bad hanja_penalty -> inf")
srv.FAMILY = "deepseek_v4"
ok(srv.hanja_filter_wanted({"hanja_filter": True}, {}) is False, "hanja_filter is FN only")
_m = importlib.util.spec_from_file_location("cjk_mask", Path(__file__).resolve().parents[1] / "tools/build_cjk_mask.py")
cm = importlib.util.module_from_spec(_m)
_m.loader.exec_module(cm)
for text, want in [("您的", "han"), (" 贵公司", "han"), ("開", None), (" 港", None), ("ですね", "kana"), ("東京", "han"), ("ア", "kana"),
                   ("개항", None), ("(開港)", None), ("hello", None), ("中a", None), ("\ufffd", None)]:
    ok(cm.classify(text) == want, f"cjk mask class {text!r} -> {cm.classify(text)}")

print("ALL OK" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
