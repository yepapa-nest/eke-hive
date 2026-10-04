#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""End-to-end quality/speed smoke test — sends text, Korean, image and tool-call requests to the hive server (:8430) and prints answers and timings.
Usage: e2e_smoke.py [--base http://127.0.0.1:8430] [--image PATH_TO_IMAGE]  (default image: $HIVE_CKPT/inference/examples/images/corn.jpeg)"""
import argparse
import base64
import json
import os
import time

import requests


def chat(base, messages, **kw):
    body = {"model": "hive", "messages": messages, "max_tokens": kw.pop("max_tokens", 256), "temperature": kw.pop("temperature", 0.0)}
    body.update(kw)
    t0 = time.time()
    r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=1800)
    dt = time.time() - t0
    if r.status_code != 200:
        print(f"--- HTTP {r.status_code}: {r.text[:300]}")
        return {}
    j = r.json()
    m = j["choices"][0]["message"]
    u = j.get("usage", {}) or {}
    h = {k: (v if v is not None else 0) for k, v in (j.get("hive", {}) or {}).items()}
    print(f"--- {dt:.1f}s · prompt {u.get('prompt_tokens')} · completion {u.get('completion_tokens')} · prefill {h.get('prefill_ms', 0):.0f}ms "
          f"decode {h.get('decode_ms', 0):.0f}ms ({(u.get('completion_tokens', 1) - 1) * 1000 / max(1, h.get('decode_ms', 1)):.1f} tok/s) · "
          f"cached {h.get('cached_prefix')} · finish {j['choices'][0]['finish_reason']}")
    if m.get("reasoning_content"):
        print("  [thinking]", m["reasoning_content"][:300].replace("\n", " "), "…" if len(m["reasoning_content"]) > 300 else "")
    print("  [content ]", (m.get("content") or "")[:600])
    if m.get("tool_calls"):
        print("  [tools   ]", json.dumps(m["tool_calls"], ensure_ascii=False)[:400])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8430")
    ckpt = os.environ.get("HIVE_CKPT")
    ap.add_argument("--image", default=os.path.join(ckpt, "inference/examples/images/corn.jpeg") if ckpt else None,
                    help="test image (default: $HIVE_CKPT/inference/examples/images/corn.jpeg)")
    ap.add_argument("--think", action="store_true", help="thinking mode (off by default)")
    a = ap.parse_args()
    if not a.image:
        ap.error("--image PATH (or HIVE_CKPT) is required")
    eff = {} if a.think else {"reasoning_effort": "none"}
    print("== 1. arithmetic")
    chat(a.base, [{"role": "user", "content": "What is 17*19? Return only the integer."}], max_tokens=16, **eff)
    print("== 2. Korean")
    chat(a.base, [{"role": "user", "content": "대한민국의 수도와 그 도시의 유명한 강 이름을 한 문장으로 말해줘."}], max_tokens=80, **eff)
    print("== 3. image")
    with open(a.image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    data_url = f"data:image/jpeg;base64,{b64}"
    img = {"type": "image_url", "image_url": {"url": data_url}}
    chat(a.base, [{"role": "user", "content": [img, {"type": "text", "text": "What is in this picture? Answer in one sentence."}]}], max_tokens=60, **eff)
    print("== 4. tool call")
    weather = dict(name="get_weather", description="Get current weather for a city",
                   parameters=dict(type="object", properties={"city": {"type": "string"}}, required=["city"]))
    tools = [{"type": "function", "function": weather}]
    chat(a.base, [{"role": "user", "content": "서울 날씨 알려줘"}], tools=tools, max_tokens=120, **eff)
    print("== 5. conversation cache (continuing the same conversation)")
    msgs = [{"role": "user", "content": "Name three prime numbers."}]
    m = chat(a.base, msgs, max_tokens=40, **eff)
    msgs += [{"role": "assistant", "content": m.get("content", "")}, {"role": "user", "content": "Now multiply the first two."}]
    chat(a.base, msgs, max_tokens=40, **eff)


if __name__ == "__main__":
    main()
