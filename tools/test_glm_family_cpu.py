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
print("RESULT:", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
