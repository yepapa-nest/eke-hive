#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""CPU test — runs tools/quality_eval.py and quality_compare.py against a fake hive server (OpenAI/Anthropic SSE and non-streaming) to check
grading, extraction, the sandbox, tool-call parsing, the Anthropic empty-text diagnosis and the comparison script (flips, McNemar p). No GPU or model needed.
Usage: nice -n 19 taskset -c 0-3 python3 tools/test_quality_eval_cpu.py"""
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import quality_eval as qe  # noqa: E402
import quality_compare as qc  # noqa: E402

SEED = 7
CPT = 3.0
LENGTHS = [1000, 2000]
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL", msg)
    else:
        print("  ok  ", msg)


# ------------------------------------------------------------------ answer table of the fake server
STATE = {"mode": "good", "table": {}}
BROKEN_IN_BAD = {"qa.en.arith.0", "qa.ko.date.3", "ruler.2000.multivalue", "code.rle", "tool.calc_mul", "tool.followup_weather"}
BROKEN_IN_GOOD = {"qa.en.fact.1"}  # item wrong in good and right in bad → one fail→pass


def answer_text(it):
    k, e = it["check"], it["expected"]
    if k == "num":
        return f"Let me think.\nAnswer: {int(e) if float(e).is_integer() else e}"
    if k == "weekday":
        return f"계산하면\n정답: {qe._WD_KO[e]}" if it["lang"] == "ko" else f"So\n**Answer:** {qe._WD_EN[e].title()}."
    if k in ("date", "str"):
        return f"...\n{'정답' if it['lang'] == 'ko' else 'Answer'}: {e}"
    if k == "alias":
        return f"{'정답' if it['lang'] == 'ko' else 'Answer'}: {e[0]}"
    if k == "contains":
        return e
    if k == "contains_all":
        return ", ".join(e)
    if k == "names_exact":
        return ", ".join(e[0])
    if k == "code":
        return "Here:\n```python\n" + it["ref"] + "```"
    raise ValueError(k)


def wrong_text(it):
    k = it["check"]
    if k == "code":
        return "```python\ndef nope():\n    pass\n```"
    if k == "contains_all":
        return ", ".join(it["expected"][:2])
    return "I am not sure.\nAnswer: 999999"


def tool_args(spec):
    a = {}
    for key, how, want in spec:
        a[key] = want[0] if how == "alias" else (" ".join(want) if how == "has" else (str(want) if how == "expr" else want))
    return a


def build_table():
    t = {}
    filler = qe.Filler(None)
    cpt = {"en": CPT, "ko": CPT}
    for it in qe.build_items(["ruler", "qa", "code", "tool"], SEED, filler, cpt, LENGTHS):
        t[it["messages"][-1]["content"]] = it
    for i, (p, v) in enumerate(qe.conc_prompts(SEED)):
        t[p] = {"id": f"ops.conc8.{i}", "reply": f"{v}\nAnswer: {v}"}
    for mode in ("seq", "overlap"):
        it = qe.ruler_item(SEED, 100000, f"single_ops_{mode}", "en", 0.5, filler, cpt)
        t[it["messages"][-1]["content"]] = {"id": f"ops.long100k.{mode}", "reply": it["expected"]}
    t[qe.SHORT_Q[0]] = {"id": "ops.short", "reply": "12+30=42\nAnswer: 42"}
    for q, want in qe.MULTITURN:
        t[q] = {"id": "ops.mt", "reply": "Noted." if want is None else want}
    t["Write two sentences about the ocean."] = {"id": "ops.ocean", "reply": "The ocean is vast. It covers most of Earth."}
    return t


def last_user_text(msgs):
    for m in reversed(msgs):
        if m["role"] == "user":
            c = m["content"]
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if p.get("type") == "text"), True
            return c, False
    return "", False


def respond(body):
    """→ dict(content, reasoning, tool_calls[(name, args)])"""
    msgs = body.get("messages", [])
    mode = STATE["mode"]
    if msgs and msgs[-1]["role"] == "tool":  # follow-up turn
        res = json.loads(msgs[-1]["content"])
        if "temp_c" in res:
            return {"content": "It is sunny." if mode == "bad" else f"It is {res['temp_c']}°C and sunny in Seoul."}
        return {"content": f"The result is {res['result']:,}."}
    text, has_img = last_user_text(msgs)
    if has_img:
        return {"content": "A cob of corn on a table."}
    it = STATE["table"].get(text)
    if it is None:
        if text.endswith("Reply with OK."):
            return {"content": "OK"}
        return {"content": "?"}
    if "reply" in it:
        return {"content": it["reply"], "reasoning": "thinking about it" if body.get("_thinking") else ""}
    broken = (it["id"] in BROKEN_IN_BAD) if mode == "bad" else (it["id"] in BROKEN_IN_GOOD)
    if it["suite"] == "tool":
        exp = it["expected"]
        if it["check"] == "tool_followup":
            return {"tool_calls": [(exp["call"]["name"], exp["call"]["arguments"])]}
        if exp["name"] is None:
            return {"content": "Buenos días"}
        args = tool_args(exp["args"])
        if broken:
            args = {k: "1+1" for k in args}
        return {"tool_calls": [(exp["name"], args)]}
    return {"content": wrong_text(it) if broken else answer_text(it)}


def prompt_tokens(body):
    text, _ = last_user_text(body.get("messages", []))
    return len(text) // 3


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        for k, v in (("Content-Type", "application/json"), ("Content-Length", str(len(b)))):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._json({"ok": True, "daemon": {"fake": True}})

    def _sse_start(self):
        self.send_response(200)
        for k, v in (("Content-Type", "text/event-stream"), ("Transfer-Encoding", "chunked")):
            self.send_header(k, v)
        self.end_headers()

    def _chunk(self, s):
        b = s.encode()
        self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
        self.wfile.flush()

    def do_POST(self):
        n_body = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n_body))
        if self.path == "/v1/chat/completions":
            thinking = body.get("reasoning_effort") != "none"
            body["_thinking"] = thinking
            r = respond(body)
            content, reasoning = r.get("content", ""), (r.get("reasoning") or ("hmm" if thinking else ""))
            calls = [{"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
                     for i, (n, a) in enumerate(r.get("tool_calls") or [])]
            pt = prompt_tokens(body)
            if not body.get("stream"):
                msg = {"role": "assistant", "content": content}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                if calls:
                    msg["tool_calls"] = calls
                n_prev = sum(1 for m in body["messages"] if m["role"] == "assistant")
                choice = {"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}
                return self._json({"choices": [choice],
                                   "usage": {"prompt_tokens": pt, "completion_tokens": 5}, "hive": {"prefill_ms": 3.0, "cached_prefix": 10 * n_prev}})
            self._sse_start()

            def ch(delta, fin=None, usage=None):
                choice = {"index": 0, "delta": delta, "finish_reason": fin}
                d = {"choices": [choice]}
                if usage:
                    d["usage"] = usage
                self._chunk("data: " + json.dumps(d, ensure_ascii=False) + "\n\n")
            ch({"role": "assistant"})
            if reasoning:
                ch({"reasoning_content": reasoning})
            for i in range(0, len(content), 7):
                ch({"content": content[i:i + 7]})
            if calls:
                ch({"tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]})
            ch({}, "tool_calls" if calls else "stop", {"prompt_tokens": pt, "completion_tokens": 5})
            self._chunk("data: [DONE]\n\n")
            self._chunk("")
            return
        if self.path == "/v1/messages":
            thinking = not (isinstance(body.get("thinking"), dict) and body["thinking"].get("type") == "disabled")
            msgs = []
            for m in body["messages"]:
                c = m["content"]
                if isinstance(c, list):
                    c = [{"type": "image_url"} if p["type"] == "image" else p for p in c]
                msgs.append({"role": m["role"], "content": c})
            ob = {"messages": msgs}
            if body.get("tools") and "London" in str(body["messages"]):
                r = {"tool_calls": [("get_weather", {"city": "London"})]}
            else:
                r = respond(ob)
            # reproduce max_tokens ending inside the reasoning under the server default (thinking on): thinking only, no text
            text = "" if thinking else r.get("content", "")
            think = "long thinking " * 20 if thinking else ""
            stop = "max_tokens" if thinking else ("tool_use" if r.get("tool_calls") else "end_turn")
            if not body.get("stream"):
                content = ([{"type": "thinking", "thinking": think}] if think else []) + ([{"type": "text", "text": text}] if text else [])
                content += [{"type": "tool_use", "id": "t1", "name": n, "input": a} for n, a in r.get("tool_calls") or []]
                return self._json({"type": "message", "content": content, "stop_reason": stop, "usage": {"input_tokens": 9, "output_tokens": 5}})
            self._sse_start()

            def ev(kind, **d):
                self._chunk(f"event: {kind}\ndata: {json.dumps({'type': kind, **d}, ensure_ascii=False)}\n\n")
            ev("message_start", message={"usage": {"input_tokens": 9}})
            idx = 0
            for kind, val in (("thinking", think), ("text", text)):
                if not val:
                    continue
                ev("content_block_start", index=idx, content_block={"type": kind, kind: ""})
                for i in range(0, len(val), 10):
                    ev("content_block_delta", index=idx, delta={"type": kind + "_delta", kind: val[i:i + 10]})
                ev("content_block_stop", index=idx)
                idx += 1
            ev("message_delta", delta={"stop_reason": stop}, usage={"output_tokens": 5})
            ev("message_stop")
            self._chunk("")
            return
        self._json({"error": "no route"}, 404)


def main():
    print("== unit: extraction and grading")
    check(qe.extract_final("blah\n**Answer:** 42.") == "42", "extract Answer bold")
    check(qe.extract_final("풀이\n정답: 서울") == "서울", "extract ko answer")
    check(qe.extract_final("Answer: 1\nfinal answer: 2") == "2", "extract last line wins")
    check(qe.extract_final("no marker") is None, "extract none")
    check(qe.check_answer("num", 1500, "Answer: 1,500 m")[0], "num with comma+unit")
    check(not qe.check_answer("num", 15, "Answer: 16")[0], "num wrong")
    check(qe.check_answer("weekday", 0, "정답: 월요일")[0] and qe.check_answer("weekday", 0, "Answer: Monday")[0], "weekday ko/en")
    check(not qe.check_answer("weekday", 0, "Answer: Sunday")[0], "weekday wrong")
    check(qe.check_answer("date", "2024-03-05", "Answer: 2024-3-5")[0], "date normalise")
    check(qe.check_answer("alias", ["서울", "Seoul"], "정답: 서울입니다")[0], "alias ko suffix")
    check(qe.check_answer("alias", ["O"], "정답: O (산소)")[0] and not qe.check_answer("alias", ["O"], "정답: Os")[0], "alias symbol word boundary")
    check(qe.check_answer("names_exact", [["ABCDE", "FGHIJ"], ["KLMNP"]], "ABCDE, FGHIJ")[0], "vartrack exact")
    check(not qe.check_answer("names_exact", [["ABCDE", "FGHIJ"], ["KLMNP"]], "ABCDE, FGHIJ, KLMNP")[0], "vartrack distractor rejected")
    check(qe._safe_eval("(15+27)*3") == 126 and qe._safe_eval("1234 × 5678") == 7006652, "safe eval")
    check(qe.check_args({"city": "Paris"}, [("city", "alias", ["Paris"])]) == [] and qe.check_args({"city": "Lyon"}, [("city", "alias", ["Paris"])]), "tool args")
    print("== unit: McNemar")
    check(qc.mcnemar_exact(0, 0) == 1.0, "p(0,0)=1")
    check(abs(qc.mcnemar_exact(5, 0) - 0.0625) < 1e-12, "p(5,0)=0.0625")
    check(abs(qc.mcnemar_exact(3, 1) - 0.625) < 1e-12, "p(3,1)=0.625")
    check(abs(qc.mcnemar_exact(10, 0) - 2 / 1024) < 1e-12, "p(10,0)=2/1024")
    print("== unit: sandbox (" + qe.sandbox_mode() + ")")
    bad_ref = []
    for name, spec, ref, tests in qe.CODE_TASKS:
        ok, why = qe.run_code(ref, tests)
        if not ok:
            bad_ref.append((name, why))
    check(not bad_ref, f"all {len(qe.CODE_TASKS)} reference solutions pass their tests {bad_ref}")
    check(not qe.run_code("def fizzbuzz(n):\n    return []\n", qe.CODE_TASKS[1][3])[0], "wrong solution fails")
    t0 = time.time()
    ok, why = qe.run_code("def f():\n    pass\nwhile True:\n    pass\n", [], timeout=2)
    check(not ok and why == "timeout" and time.time() - t0 < 6, f"infinite loop times out ({why})")
    ok, why = qe.run_code("import socket\ndef f():\n    socket.create_connection(('127.0.0.1', 9))\n", ["f()"])
    check(not ok and ("network disabled" in why or "Network" in why or "OSError" in why), f"network blocked ({why})")
    check(qe.extract_code("x\n```python\na=1\n```\ny\n```python\nb=2\n```") == "b=2\n", "extract last code block")
    check(qe.extract_code("```python\ndef g(): return 1\n") == "def g(): return 1\n", "extract unterminated block")
    print("== determinism")
    f = qe.Filler(None)
    c = {"en": CPT, "ko": CPT}
    a1 = [qe.sha(json.dumps(i["messages"], ensure_ascii=False)) for i in qe.build_items(["ruler", "qa", "code", "tool"], SEED, f, c, LENGTHS)]
    a2 = [qe.sha(json.dumps(i["messages"], ensure_ascii=False)) for i in qe.build_items(["ruler", "qa", "code", "tool"], SEED, f, c, LENGTHS)]
    a3 = [qe.sha(json.dumps(i["messages"], ensure_ascii=False)) for i in qe.build_items(["ruler", "qa"], SEED + 1, f, c, LENGTHS)]
    check(a1 == a2, "same seed → same prompts")
    check(a1[:len(a3)] != a3, "different seed → different prompts")
    full = qe.build_items(["ruler", "qa", "code", "tool"], SEED, f, c, qe.DEFAULT_LENGTHS)
    cnt = {s: sum(1 for i in full if i["suite"] == s) for s in ("ruler", "qa", "code", "tool")}
    print("   counts (default lengths):", cnt, "· ko/en qa:", sum(1 for i in full if i["suite"] == "qa" and i["lang"] == "ko"), "/",
          sum(1 for i in full if i["suite"] == "qa" and i["lang"] == "en"))
    check(cnt == {"ruler": 35, "qa": 80, "code": 20, "tool": 22}, "item counts")
    ids = [i["id"] for i in full]
    check(len(ids) == len(set(ids)), "unique ids")
    r100 = [i for i in full if i["id"] == "ruler.100000.single_d90"][0]
    L = len(r100["messages"][0]["content"]) / CPT
    check(0.9 * 100000 < L < 1.02 * 100000, f"100K ruler prompt ≈ target ({L:.0f} tok at cpt {CPT})")
    for it in full:
        if it["check"] in ("contains", "contains_all") and it["suite"] == "ruler":
            vals = it["expected"] if isinstance(it["expected"], list) else [it["expected"]]
            if not all(v in it["messages"][0]["content"] for v in vals):
                check(False, f"needle present {it['id']}")
                break
    else:
        check(True, "every ruler needle is in its haystack")

    print("== fake server e2e")
    STATE["table"] = build_table()
    class Srv(ThreadingHTTPServer):
        def handle_error(self, request, client_address):  # when the client closes the connection after the response, the keep-alive read gets a reset — harmless
            pass
    srv = Srv(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    d = tempfile.mkdtemp(prefix="qe-test-")
    img = os.path.join(d, "corn.jpeg")
    with open(img, "wb") as fh:
        fh.write(b"\xff\xd8\xff\xe0fakejpeg")
    good, bad = os.path.join(d, "good.json"), os.path.join(d, "bad.json")
    STATE["mode"] = "good"
    rc = qe.main(["--base", base, "--out", good, "--seed", str(SEED), "--ruler-lengths", ",".join(map(str, LENGTHS)), "--image", img, "--label", "good"])
    check(rc == 0, "good run rc 0")
    G = json.load(open(good))
    check(G["meta"]["chars_per_token"] == {"en": 3.0, "ko": 3.0}, f"calibration measured cpt {G['meta']['chars_per_token']}")
    gi = {r["id"]: r for r in G["items"]}
    fails = sorted(i for i, r in gi.items() if r.get("pass") is False)
    check(fails == sorted(BROKEN_IN_GOOD | {"ops.stream.anthropic_default_thinking", "ops.stream.anthropic_default_thinking"}),
          f"good run fails only the planted ones: {fails}")
    dg = gi["ops.stream.anthropic_default_thinking"]["detail"]["diagnosis"]
    check(dg and "EMPTY TEXT" in dg and "thinking" in dg, f"anthropic empty-text diagnosis: {dg}")
    check(gi["ops.image.openai"]["pass"] and gi["ops.image.anthropic"]["pass"], "image items pass")
    check(gi["tool.followup_calc"]["pass"] and gi["tool.anthropic_weather"]["pass"] and gi["tool.nonstream_tsla"]["pass"], "tool followup / anthropic / nonstream")
    check(gi["ops.multiturn.recall_t6"]["pass"] and gi["ops.multiturn.all_turns"]["pass"], "multiturn")
    check(all(gi[f"ops.conc8.{i}"]["pass"] for i in range(8)), "conc8")
    check(gi["ops.short_after_long.overlap"]["pass"] and gi["ops.short_after_long.overlap"]["ttft_s"] is not None, "short after long (ttft recorded)")
    check(all(r.get("ttft_s") is not None for r in G["items"] if r["suite"] in ("ruler", "qa", "code")), "ttft recorded for streamed items")
    check(G["summary"]["code"]["rate"] == 1.0 and G["summary"]["ruler"]["by_length"] == {"1000": "7/7", "2000": "7/7"}, "summary fields")
    STATE["mode"] = "bad"
    rc = qe.main(["--base", base, "--out", bad, "--seed", str(SEED), "--ruler-lengths", ",".join(map(str, LENGTHS)), "--image", "/nonexistent.jpeg",
                  "--cpt", "3.0", "--label", "bad"])
    B = json.load(open(bad))
    bi = {r["id"]: r for r in B["items"]}
    check(bi["ops.image.openai"]["pass"] is None and "skipped" in bi["ops.image.openai"], "missing image → skipped")
    res = qc.compare(G, B)
    p2f = sorted(i for s in res["suites"].values() for i in s["pass_to_fail"])
    f2p = sorted(i for s in res["suites"].values() for i in s["fail_to_pass"])
    check(p2f == sorted(BROKEN_IN_BAD), f"pass→fail flips {p2f}")
    check(f2p == sorted(BROKEN_IN_GOOD), f"fail→pass flips {f2p}")
    q = res["suites"]["qa"]
    check(q["mcnemar_b"] == 2 and q["mcnemar_c"] == 1 and abs(q["p"] - 1.0) < 1e-12, f"qa McNemar b=2 c=1 p={q['p']}")
    t = res["suites"]["tool"]
    check(t["mcnemar_b"] == 2 and abs(t["p"] - 0.5) < 1e-12, f"tool McNemar b=2 c=0 p={t['p']}")
    check({"ops.image.openai", "ops.image.anthropic"} <= set(res["suites"]["ops"]["skipped"]) and not res["only_in_baseline"], "ops skipped items reported")
    check(not res["prompt_mismatch"], "prompts identical across runs (calibrated cpt = fixed cpt)")
    check(set(p2f) <= set(i for s in res["suites"].values() for i in s["output_differs"]), "flipped items listed as output-differing")
    rc = qc.main([good, bad, "--json", os.path.join(d, "cmp.json")])
    check(rc == 0 and os.path.exists(os.path.join(d, "cmp.json")), "compare CLI")
    srv.shutdown()
    print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    for m in FAILS:
        print("  -", m)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
