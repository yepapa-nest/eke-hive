#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Gemma 4 family rules (server/families/gemma.py), no model and no tokenizer download: thinking mapping, the prompt kwargs and template
handed to the chat template (incl. the tool-response branch), history tool-call arguments, tool-call / thought-channel parsing, stop ids
and sampling defaults. HIVE_TEST_GEMMA_DIR=<checkpoint dir> also renders the real template and parses its own tool-call rendering back."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from families import gemma  # noqa: E402

fails = 0


def ok(cond, what):
    global fails
    fails += not cond
    print(f"{'ok  ' if cond else 'FAIL'} {what}")


# ---- thinking ----
for body, want in [({}, "chat"), ({"chat_template_kwargs": {"enable_thinking": False}}, "chat"),
                   ({"chat_template_kwargs": {"enable_thinking": True}}, "thinking"), ({"reasoning_effort": "low"}, "thinking"),
                   ({"reasoning_effort": "none"}, "chat"), ({"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "high"}, "chat"),
                   ({"reasoning": {"effort": "medium"}}, "thinking"), ({"reasoning_effort": True}, "chat")]:
    got = gemma.effort_from_request(body)[0]
    ok(got == want, f"{body} -> {got}" + ("" if got == want else f" (want {want})"))


class FakeTok:
    eos_token_id = 1

    def __init__(self):
        self.last = None

    def convert_ids_to_tokens(self, i):
        return {1: "<eos>", 106: "<turn|>", 50: "<|tool_response>"}.get(i)

    def apply_chat_template(self, msgs, **kw):
        self.last = (msgs, kw)
        return "PROMPT" + ("<|channel>thought\n<channel|>" if not kw.get("enable_thinking") else "")

    def encode(self, s, add_special_tokens=False):
        return [ord(ch) for ch in s]


ANCHOR = gemma._ANCHOR
with tempfile.TemporaryDirectory() as d:
    Path(d, "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 106, 50], "temperature": 1.0, "top_p": 0.95, "top_k": 64}))
    Path(d, "chat_template.jinja").write_text("HEAD\n" + ANCHOR + "\nTAIL")
    tok = FakeTok()
    gemma.init(tok, {"model_type": "gemma4_unified"}, d)
    ok(gemma.STOP_IDS == [1, 106, 50], f"stop ids from generation_config.json: {gemma.STOP_IDS}")
    ok(gemma.EOS_TEXTS == ["<eos>", "<turn|>", "<|tool_response>"], f"eos texts {gemma.EOS_TEXTS}")
    ok(gemma.defaults() == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}, f"sampling defaults {gemma.defaults()}")
    ok(gemma._TEMPLATE is not None and gemma._PATCHED in gemma._TEMPLATE and gemma._TEMPLATE.count("tool_response") == 2,
       "tool-response branch added to the template")
    tools = [{"type": "function", "function": {"name": "delegate_task", "parameters": {"type": "object", "properties": {"task": {"type": "string"}}}}}]
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "delegate_task", "arguments": "{\"task\": \"x\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "done"}]
    p = gemma.encode_messages(msgs + [{"role": "system", "content": "", "tools": tools}][:0] + [], "chat")
    m, kw = tok.last
    ok(kw.get("enable_thinking") is False and kw.get("chat_template") == gemma._TEMPLATE and kw.get("add_generation_prompt"), "chat kwargs")
    ok(m[1]["content"] == "hi", "text parts joined")
    ok(m[2]["tool_calls"][0]["function"]["arguments"] == {"task": "x"}, "history arguments: JSON string -> mapping (template renders mappings)")
    ok(msgs[2]["tool_calls"][0]["function"]["arguments"] == "{\"task\": \"x\"}", "caller's messages untouched")
    p2 = gemma.encode_messages([{"role": "system", "content": "", "tools": tools}, {"role": "user", "content": "q"}], "thinking")
    m2, kw2 = tok.last
    ok(kw2.get("tools") == tools and kw2.get("enable_thinking") is True and m2[0]["role"] == "user", "tool carrier -> tools kwarg, thinking on")
    ok(p2.endswith(gemma.THINK_START), "thinking prompt opened inside the thought channel")
    ok(p.endswith("<|channel>thought\n<channel|>"), "chat prompt keeps the template's empty thought channel")
    try:
        gemma.encode_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}], "chat")
        ok(False, "image part rejected")
    except ValueError:
        ok(True, "image part rejected (text path)")
    os.environ["HIVE_GEMMA_TOOL_THOUGHT"] = "0"
    gemma.init(tok, {}, d)
    ok(gemma._TEMPLATE is not None and gemma._PATCHED not in gemma._TEMPLATE, "HIVE_GEMMA_TOOL_THOUGHT=0 keeps the template unchanged")
    os.environ.pop("HIVE_GEMMA_TOOL_THOUGHT")
    gemma.init(tok, {}, d)

# ---- parsing ----
Q = '<|"|>'
cases = [
    (f'<|tool_call>call:delegate_task{{task:{Q}서울 날씨, 내일{Q},title:{Q}날씨{Q}}}<tool_call|>', [("delegate_task", {"task": "서울 날씨, 내일", "title": "날씨"})]),
    (f'<|tool_call>call:f{{a:1,b:-2.5,c:true,d:false,e:null,g:[1,{Q}x{Q},[2]],h:{{i:{Q}}}{{]{Q}}}}}<tool_call|>',
     [("f", {"a": 1, "b": -2.5, "c": True, "d": False, "e": None, "g": [1, "x", [2]], "h": {"i": "}{]"}})]),
    (f'<|tool_call>call:a{{}}<tool_call|><|tool_call>call:b{{x:{Q}1{Q}}}<tool_call|>', [("a", {}), ("b", {"x": "1"})]),
    ('<|tool_call>call:search{query:weather tomorrow,n:3}<tool_call|>', [("search", {"query": "weather tomorrow", "n": 3})]),
    (f'<|tool_call>call:x{{q:{Q}unterminated', [("x", {})]),
]
for text, want in cases:
    got = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in gemma.parse_tool_calls(text)]
    ok(got == want, f"{text[:60]!r} -> {got}" + ("" if got == want else f" (want {want})"))

r = gemma.parse_message_from_completion_text(f'바로 알아볼게요.<|tool_call>call:delegate_task{{task:{Q}t{Q}}}<tool_call|><|tool_response>', "chat")
ok(r["content"] == "바로 알아볼게요." and r["tool_calls"][0]["function"]["name"] == "delegate_task", f"content + tool call: {r}")
r = gemma.parse_message_from_completion_text("생각 중<channel|>답이에요<turn|>", "thinking")
ok(r == {"content": "답이에요", "reasoning_content": "생각 중"}, f"thinking split: {r}")
r = gemma.parse_message_from_completion_text("<|channel>thought\n혼자 연 생각<channel|>답", "chat")
ok(r == {"content": "답", "reasoning_content": "혼자 연 생각"}, f"self-opened thought channel: {r}")
r = gemma.parse_message_from_completion_text("끝나지 않은 생각", "thinking")
ok(r == {"reasoning_content": "끝나지 않은 생각", "content": ""}, f"unfinished thinking: {r}")
r = gemma.parse_message_from_completion_text("그냥 답<turn|>", "chat")
ok(r == {"content": "그냥 답"}, f"plain answer: {r}")

# ---- streamed tool calls == the end-of-turn parse ----
text = f'앞말<|tool_call>call:delegate_task{{task:{Q}서울 날씨, 내일{Q},n:3}}<tool_call|><|tool_call>call:b{{}}<tool_call|>'
for step in (1, 3, 7, len(text)):
    ts = gemma.ToolStream()
    deltas = []
    for i in range(0, len(text), step):
        deltas += ts.feed(text[i:i + step])
    ref = [(c["function"]["name"], c["function"]["arguments"]) for c in gemma.parse_tool_calls(text)]
    names = [d["function"]["name"] for d in deltas if "name" in d.get("function", {})]
    ok(ts.result() == ref and ts.complete() and names == ["delegate_task", "b"] and not ts.broken,
       f"tool stream (pieces of {step}): names first, arguments = the end parse")
ts = gemma.ToolStream()
d = ts.feed("<|tool_call>call:delegate_ta")
ok(d == [] and not ts.complete(), "tool stream: nothing before the name is complete")
d = ts.feed("sk{task:")
ok(len(d) == 1 and d[0]["function"] == {"name": "delegate_task", "arguments": ""} and not ts.complete(), "tool stream: name delta at '{'")

# ---- the real template's own rendering, parsed back (optional) ----
real = os.environ.get("HIVE_TEST_GEMMA_DIR")
if real:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(real)
    gemma.init(tok, json.load(open(os.path.join(real, "config.json"))), real)
    args = {"task": "서울, 부산 \"날씨\"", "n": 3, "flags": [True, None], "nested": {"k": "v"}}
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": json.dumps(args)}}]},
            {"role": "tool", "tool_call_id": "1", "content": "r"}]
    p = gemma.encode_messages(msgs, "chat")
    a = p.find(gemma.TOOL_START)
    b = p.find(gemma.TOOL_END, a) + len(gemma.TOOL_END)
    got = gemma.parse_tool_calls(p[a:b])
    ok(len(got) == 1 and json.loads(got[0]["function"]["arguments"]) == args, f"real template round trip: {p[a:b]!r}")
    ok(p.endswith("<|channel>thought\n<channel|>"), "real template: empty thought channel after a tool response (patched branch)")
    ids = gemma.prepare_inputs(p, [], tok)[0]
    ok(isinstance(ids, list) and len(ids) > 10, f"prepare_inputs: {len(ids)} ids")

print(f"gemma family CPU: {'ok' if not fails else f'{fails} FAIL'}")
sys.exit(1 if fails else 0)
