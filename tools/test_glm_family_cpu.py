#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""GLM family request rules (server/families/glm.py), no model: effort mapping, the unset default (high) and HIVE_GLM_NOTHINK."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from families import glm  # noqa: E402

fails = 0


def check(body, want, env=None):
    global fails
    old = os.environ.get("HIVE_GLM_NOTHINK")
    if env is None:
        os.environ.pop("HIVE_GLM_NOTHINK", None)
    else:
        os.environ["HIVE_GLM_NOTHINK"] = env
    got = glm.effort_from_request(body)
    if old is None:
        os.environ.pop("HIVE_GLM_NOTHINK", None)
    else:
        os.environ["HIVE_GLM_NOTHINK"] = old
    ok = got == want
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {body} (HIVE_GLM_NOTHINK={env}) -> {got}" + ("" if ok else f" (want {want})"))


check({}, ("thinking", "high"))                                      # unset: high, not the template default max
check({"reasoning_effort": "weird"}, ("thinking", "high"))           # unknown value: high
check({"reasoning_effort": "none"}, ("thinking", "low"))             # thinking cannot be turned off: official minimum
check({"reasoning_effort": "minimal"}, ("thinking", "low"))
check({"chat_template_kwargs": {"enable_thinking": False}}, ("thinking", "low"))
check({"reasoning_effort": "none"}, ("chat", None), env="empty")      # opt-in: prefilled empty thinking block
check({"reasoning_effort": "low"}, ("thinking", "low"))
check({"reasoning_effort": "medium"}, ("thinking", "high"))
check({"reasoning_effort": "high"}, ("thinking", "high"))
check({"reasoning_effort": "xhigh"}, ("thinking", "max"))
check({"reasoning": {"effort": "max"}}, ("thinking", "max"))
check({"reasoning_effort": 25}, ("thinking", "low"))
check({"reasoning_effort": "75"}, ("thinking", "high"))
check({"reasoning_effort": 100}, ("thinking", "max"))
check({"reasoning_effort": True}, ("thinking", "high"))               # a bool is not an effort


# ---- images / video preprocessing (no model) ----
import numpy as np  # noqa: E402

th, tw = glm._target_size(2, 360, 640, glm._IMG_TOKENS)
ok = th % 28 == 0 and tw % 28 == 0 and 2 * th * tw <= glm._IMG_TOKENS[1] * 2 * 28 * 28
fails += not ok
print(f"{'ok  ' if ok else 'FAIL'} target size 360x640 -> {th}x{tw} (multiples of 28, within 8000 tokens)")
th, tw = glm._target_size(2, 4000, 6000, glm._IMG_TOKENS)  # large: shrinks into the budget
ok = (th // 14) * (tw // 14) // 4 <= 8000 and th % 28 == 0 and tw % 28 == 0
fails += not ok
print(f"{'ok  ' if ok else 'FAIL'} target size 4000x6000 -> {th}x{tw} ({(th // 14) * (tw // 14) // 4} tokens)")
# patch order: rows in 2x2 merge-block order, each row (channel, frame, 14, 14)
pair = np.arange(2 * 3 * 56 * 56, dtype=np.float32).reshape(2, 3, 56, 56)
p, gh, gw = glm._patchify(pair)
want = pair[:, :, 14:28, 0:14].transpose(1, 0, 2, 3).reshape(-1)  # third row of block (0,0) = patch (row 1, col 0)
ok = p.shape == (16, 1176) and gh == 4 and gw == 4 and np.array_equal(p[2], want)
fails += not ok
print(f"{'ok  ' if ok else 'FAIL'} patchify 56x56 -> {p.shape}, merge-block order")
# ---- tool-call arguments: schema-typed strings stay raw, output is always valid JSON (no model) ----
import json  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "job_wait", "parameters": {"type": "object", "properties": {
    "job_id": {"type": "string"}, "code": {"type": ["string", "null"]}, "timeout_seconds": {"type": "number"},
    "flag": {"type": "boolean"}, "ids": {"type": "array"}}}}}]


def _strict(s):
    return json.loads(s, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def check_args(args_xml, want, tools=TOOLS, name="job_wait"):
    global fails
    text = f"<tool_call>{name}" + "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in args_xml) + "</tool_call>"
    calls = glm.parse_tool_calls(text, tools)
    try:
        got = _strict(calls[0]["function"]["arguments"])
        ok = got == want and all(type(got[k]) is type(want[k]) for k in want)
    except (ValueError, IndexError) as e:
        got, ok = repr(e), False
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} tool args {args_xml} -> {got}" + ("" if ok else f" (want {want})"))


check_args([("job_id", "3e382151")], {"job_id": "3e382151"})          # 2026-10-05 incident: was inf / Infinity
check_args([("code", "0123")], {"code": "0123"})                      # string|null: leading zero kept
check_args([("job_id", "true")], {"job_id": "true"})
check_args([("job_id", "null")], {"job_id": "null"})
check_args([("timeout_seconds", "600")], {"timeout_seconds": 600})    # number: decoded
check_args([("flag", "true")], {"flag": True})
check_args([("ids", '["a", 1]')], {"ids": ["a", 1]})
check_args([("timeout_seconds", "1e999")], {"timeout_seconds": "1e999"})   # non-finite: raw text, never Infinity
check_args([("x", "1e999")], {"x": "1e999"}, tools=None)              # no schema
check_args([("x", "3e382151")], {"x": "3e382151"}, tools=None)
check_args([("x", "NaN")], {"x": "NaN"}, tools=None)
check_args([("x", "-Infinity")], {"x": "-Infinity"}, tools=None)
check_args([("x", "[1, Infinity]")], {"x": "[1, Infinity]"}, tools=None)
check_args([("x", "[1, 1e999]")], {"x": "[1, 1e999]"}, tools=None)
check_args([("x", "42")], {"x": 42}, tools=None)                      # no schema, finite: decoded as before
check_args([("job_id", "3e382151")], {"job_id": "3e382151"}, tools=[{"name": "job_wait", "parameters": {"properties": {"job_id": {"type": "string"}}}}])
check_args([("job_id", "3e382151")], {"job_id": "3e382151"}, tools=[{"type": "function", "function": {"name": "job_wait", "parameters": {"properties": {"job_id": {"enum": ["3e382151", "x"]}}}}}])
msg = glm.parse_message_from_completion_text("hm</think><tool_call>job_wait<arg_key>job_id</arg_key><arg_value>3e382151</arg_value></tool_call>", tools=TOOLS)
ok = _strict(msg["tool_calls"][0]["function"]["arguments"]) == {"job_id": "3e382151"}
fails += not ok
print(f"{'ok  ' if ok else 'FAIL'} parse_message_from_completion_text passes tools to the argument parser")

print("RESULT:", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
