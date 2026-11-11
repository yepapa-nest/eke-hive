#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Tool-call format gate, against the real checkpoint's tokenizer and reference encoding (no GPU, no model weights read). Run before a build goes into service:

    python3 tools/test_tool_calls_ref.py --ckpt /path/to/DeepSeek-V4.1-Flash [--write]

A local server's tool calling regresses without the model changing — the chat template, the tool-call parser, the streaming parser. This pins them:
  (1) the rendered prompts (tools block, an assistant call, a tool result, in chat and thinking mode) equal tools/fixtures/tool_format_golden.json, byte and token,
      so a template, encoding or tokenizer change cannot pass silently. --write records the fixture (a deliberate step, reviewed in the diff).
  (2) a call rendered by the reference encoding parses back to the same names and arguments (types, Korean, quotes, a value quoting a tool tag, several calls, thinking).
  (3) the streaming parser (DsmlToolStream) fed token by token gives exactly the reference parse.
  (4) hive/tool_mask.h replays the real token ids: each call opens one name span, its tokens stay on the allowed names' automaton, a value quoting an invoke tag opens
      none, reasoning opens none, and the same name is off the automaton of a list without it.
  (5) the automaton (compile_tool_trie) over the real vocabulary: every path spells a name and closes it as the grammar does, and the canonical tokens are a path.
Skipped (exit 0) without a checkpoint unless --ckpt names one: the public CI has none.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
FIXTURE = ROOT / "tools/fixtures/tool_format_golden.json"
FAILS = []


def check(what, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + what + ("" if cond else f"  {detail}"))
    if not cond:
        FAILS.append(what)


def fn(name, props, required=None, **kw):
    return {"type": "function", **kw, "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object", "properties": props, "required": required or list(props)}}}


TOOLS = [fn("shell", {"command": {"type": "string"}, "timeout": {"type": "integer"}}, ["command"]),
         fn("files_write", {"path": {"type": "string"}, "content": {"type": "string"}, "mode": {"type": "string", "enum": ["w", "a"]}}, ["path", "content"]),
         fn("web_search", {"query": {"type": "string"}, "language": {"type": "string"}}, ["query"]),
         fn("web_url_read", {"url": {"type": "string"}}), fn("browser", {"action": {"type": "string"}, "args": {"type": "object"}}, ["action"]),
         fn("browser_task", {"goal": {"type": "string"}, "steps": {"type": "array", "items": {"type": "string"}}}, ["goal"]),
         fn("site_credential", {"site": {"type": "string"}}), fn("send", {"to": {"type": "string"}}, namespace="mail")]
NAMES = ["browser", "browser_task", "files_write", "mail::send", "shell", "site_credential", "web_search", "web_url_read"]
QUOTED = '문서 예시: <｜DSML｜ invoke name="other">\n끝'


def call(name, args):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


CALLS = {
    "one": [call("shell", {"command": "ls -la /tmp && echo \"done\"", "timeout": 30})],
    "korean": [call("web_search", {"query": "오늘 서울 날씨 '비' 예보", "language": "ko"})],
    "object arg": [call("browser", {"action": "click", "args": {"x": 10, "y": 2.5, "on": True, "tags": ["a", "b"], "none": None}})],
    "array arg": [call("browser_task", {"goal": "로그인", "steps": ["열기", "입력"]})],
    "several": [call("shell", {"command": "a"}), call("files_write", {"path": "/x", "content": "line 1\nline 2\n\tindented"}), call("web_url_read", {"url": "https://example.com/?a=1&b=2"})],
    "value quoting a tag": [call("files_write", {"path": "/doc", "content": QUOTED})],
    "site": [call("site_credential", {"site": "example.com"})],
}
CONVERSATION = [{"role": "system", "content": "You are an assistant."}, {"role": "user", "content": "무엇을 할 수 있나요?"},
                {"role": "assistant", "content": "확인합니다.", "tool_calls": CALLS["one"]},
                {"role": "tool", "tool_call_id": "c", "content": "total 0\n"}, {"role": "user", "content": "다음은?"}]


def sha(x):
    return hashlib.sha256(x.encode() if isinstance(x, str) else json.dumps(x).encode()).hexdigest()


def tags(s):
    e = s.ENC
    pieces = [e.dsml_token, e.tool_calls_block_name, e.tool_call_tag_name, e.tool_parameter_tag_name, " name", '="']
    coded = [s.TOK.encode(p, add_special_tokens=False) for p in pieces]
    check("tag pieces are single tokens", all(len(c) == 1 for c in coded), str(coded))
    return [c[0] for c in coded]


def replay(exe, tag, trie, ids, think=(-1, -1, 0)):
    data = " ".join(map(str, [*tag, *think, len(trie["first"]), *trie["first"], len(trie["tok"]), *trie["tok"], len(trie["next"]), *trie["next"], len(ids), *ids]))
    out = subprocess.run([exe], input=data, capture_output=True, text=True, check=True).stdout.splitlines()
    spans, left, mode, in_think = map(int, out[0].split())
    return spans, left, mode, in_think, [list(map(int, l.split())) for l in out[1:]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=os.environ.get("HIVE_CKPT"))
    ap.add_argument("--write", action="store_true", help="record the golden rendering")
    a = ap.parse_args()
    if not a.ckpt or not Path(a.ckpt, "encoding").is_dir():
        print("tool format gate: no checkpoint — skipped")
        return 0
    import hive_server as s
    s.load_modules(a.ckpt)
    eos = s.TOK.eos_token

    # (1) rendering
    golden = {}
    for mode, effort in (("chat", None), ("thinking", 75)):
        prompt, _ = s.encode_prompt(json.loads(json.dumps(CONVERSATION)), json.loads(json.dumps(TOOLS)), mode, effort)
        ids = s.TOK.encode(prompt, add_special_tokens=False)
        golden[mode] = {"prompt_sha256": sha(prompt), "tokens": len(ids), "ids_sha256": sha([int(i) for i in ids])}
    no_field, _ = s.encode_prompt(json.loads(json.dumps(CONVERSATION)), json.loads(json.dumps(TOOLS)), "chat", None)
    if a.write:
        FIXTURE.write_text(json.dumps(golden, indent=1) + "\n")
        print("written", FIXTURE)
    want = json.loads(FIXTURE.read_text())
    for mode in golden:
        check(f"rendering {mode} = golden", golden[mode] == want[mode], f"{golden[mode]} vs {want[mode]}")

    with tempfile.TemporaryDirectory() as td:
        exe = str(Path(td, "replay"))
        subprocess.run(["g++", "-std=c++20", "-O1", f"-I{ROOT}/engine/include", str(ROOT / "engine/tests/tool_mask_replay.cpp"), "-o", exe], check=True)
        tag = tags(s)
        vocab = s.TOK.get_vocab()
        trie_all = s.compile_tool_trie(NAMES, vocab)
        check("automaton compiled over the vocabulary", trie_all is not None)

        # (5) every path of the automaton spells a name and closes as the grammar does
        closers = ('">\n', '">', '"')
        paths, seen = 0, set()
        stack = [(0, "")]
        ok = True
        while stack:
            node, text = stack.pop()
            for e in range(trie_all["first"][node], trie_all["first"][node + 1]):
                t, to = trie_all["tok"][e], trie_all["next"][e]
                piece = s.TOK.decode([t], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                if to < 0:
                    paths += 1
                    if not any((text + piece) == n + c for n in NAMES for c in closers):
                        ok = False
                        print("  bad path", repr(text + piece))
                else:
                    if not any(n.startswith(text + piece) for n in NAMES):
                        ok = False
                        print("  bad prefix", repr(text + piece))
                    elif (to, text + piece) not in seen:
                        seen.add((to, text + piece))
                        stack.append((to, text + piece))
        check(f"automaton: every path spells a name and closes it ({paths} closing edges)", ok and paths > 0)
        for n in NAMES:
            ids = s.TOK.encode(n + '">\n', add_special_tokens=False)
            node = 0
            for t in ids:
                es = range(trie_all["first"][node], trie_all["first"][node + 1])
                e = next((e for e in es if trie_all["tok"][e] == t), None)
                if e is None:
                    node = None
                    break
                node = trie_all["next"][e]
                if node < 0:
                    break
            check(f"automaton: canonical tokens of {n}", node == -1, str(ids))

        # (2) (3) (4) per case, chat and thinking
        t_start, t_end = s.think_token_ids()
        check("thinking tags are single tokens", t_start is not None and t_end is not None)
        for label, calls in CALLS.items():
            msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": json.loads(json.dumps(calls))}, {"role": "user", "content": "y"}]
            full, _ = s.encode_prompt(json.loads(json.dumps(msgs)), json.loads(json.dumps(TOOLS)), "chat", None)
            # the assistant turn as the reference writes it: from the block's start through the closing tag
            start = full.rindex("<" + s.ENC.dsml_token + s.ENC.tool_calls_block_name)   # the last one: the tools block quotes an example
            close = "</" + s.ENC.dsml_token + s.ENC.tool_calls_block_name + ">"
            end = full.rindex(close) + len(close)
            block = full[start:end]
            parsed = s.parse_completion("\n\n" + block + eos, "chat", TOOLS)
            got = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in s.openai_tool_calls(parsed.get("tool_calls") or [])]
            want_calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]
            check(f"[{label}] parse = the calls", got == want_calls, f"{got} vs {want_calls}")

            ids = [int(i) for i in s.TOK.encode(block, add_special_tokens=False)]
            stream = s.DsmlToolStream(s.ENC)
            decoder = s.Detok()   # as the server decodes: a character split across tokens comes out whole
            for i in ids:
                stream.feed(decoder.push(i))
            ref = [(c["function"]["name"], c["function"]["arguments"]) for c in s.openai_tool_calls(parsed.get("tool_calls") or [])]
            check(f"[{label}] stream = parse", stream.complete() and stream.result() == ref, f"{stream.result()} vs {ref}")

            spans, left, mode, _, inside = replay(exe, tag, trie_all, ids)
            check(f"[{label}] one name span per call, all on the automaton", spans == len(calls) and left == 0 and mode == 0, f"{spans} {left} {mode}")
            only = s.compile_tool_trie(["web_search" if calls[0]["function"]["name"] != "web_search" else "shell"], vocab)
            _, left2, _, _, _ = replay(exe, tag, only, ids)
            check(f"[{label}] a list without the name: off the automaton", left2 >= 1, str(left2))

            # reasoning with the same tags opens no span
            quoted = ids[:]
            think = [t_start] + quoted + [t_end] + ids
            spans3, left3, _, it3, _ = replay(exe, tag, trie_all, think, (t_start, t_end, 0))
            check(f"[{label}] tags in the reasoning open nothing, the answer's do", spans3 == len(calls) and left3 == 0 and it3 == 0, f"{spans3} {left3}")
            spans4, _, _, it4, _ = replay(exe, tag, trie_all, ids + [t_end] + ids, (t_start, t_end, 1))
            check(f"[{label}] a prompt open in <think>", spans4 == len(calls) and it4 == 0, str(spans4))

    print(f"tool format gate: {'FAILED ' + str(len(FAILS)) if FAILS else 'OK'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
