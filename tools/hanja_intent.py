#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Intended Han / kana use under the FN answer-span penalty (docs/qwen.md): 10 requests for a Japanese answer, 10 for a Chinese answer and
10 for Korean with Han characters in parentheses, sent under each mode (off · inf · <Δ> forced) and the sampling fields given (--sampling).
Pass rules: Japanese — ≥ 5 kana characters; Chinese — ≥ 10 ideographs and no Hangul syllable outside quotes of the prompt word; annotation —
at least two parenthesized groups holding an ideograph.

    python3 tools/hanja_intent.py --base http://127.0.0.1:8431 --modes off,inf,5,auto:5 --out intent.json
"""
import argparse, concurrent.futures as cf, json, re, urllib.request

KANA = re.compile("[぀-ゟ゠-ヿ]")
HAN = re.compile("[㐀-䶿一-鿿豈-﫿]")
HANGUL = re.compile("[가-힣]")
PAREN_HAN = re.compile(r"\([^()]*[㐀-䶿一-鿿豈-﫿][^()]*\)")

JA = ["다음 문장을 일본어로 번역해 줘: 오늘 날씨가 정말 좋네요. 산책하러 갈까요?", "일본어로 자기소개를 세 문장으로 써 줘.", "Please answer in Japanese: what is your favorite season and why?",
      "'감사합니다', '죄송합니다', '안녕하세요'를 일본어로 써 줘.", "일본어로 짧은 하이쿠를 하나 지어 줘.", "Translate to Japanese: The meeting starts at three o'clock tomorrow.",
      "일본어로 답해 줘: 도쿄에서 가 볼 만한 곳 세 군데는?", "이 메일을 일본어로 바꿔 줘: 보내주신 자료 잘 받았습니다. 검토 후 연락드리겠습니다.", "日本語で答えてください：好きな食べ物は何ですか？",
      "일본어 번역 부탁해요: 이 책은 정말 재미있어요."]
ZH = ["다음 문장을 중국어로 번역해 줘: 오늘 날씨가 정말 좋네요.", "중국어로 자기소개를 세 문장으로 써 줘.", "Please answer in Chinese: what is the capital of Korea?",
      "Translate to Chinese: I will arrive at the station at seven.", "중국어로 답해 줘: 만리장성은 언제 지어졌나요?", "用中文回答：你最喜欢的季节是什么？为什么？",
      "이 문장을 중국어 간체로 써 줘: 저는 한국에서 왔습니다.", "중국어로 짧은 시를 하나 써 줘.", "请把这句话翻译成中文：Thank you for your help.",
      "중국어로 바꿔 줘: 회의는 내일 오전 10시에 시작합니다."]
AN = ["'개항', '실학', '훈민정음', '정조'를 각각 괄호 안에 한자를 병기해서 설명해 줘.", "사자성어 다섯 개를 한자 병기(괄호)와 함께 뜻을 알려 줘.",
      "조선 시대 관청 이름 다섯 개를 괄호 안에 한자를 함께 써서 설명해 줘.", "'민주주의', '경제', '사회', '문화'의 한자어를 괄호로 병기해서 알려 줘.",
      "한국의 궁궐 네 곳의 이름을 괄호 안 한자 병기로 소개해 줘.", "고사성어 '와신상담'과 '토사구팽'의 한자와 유래를 설명해 줘. 한자는 괄호에 써 줘.",
      "학교 과목 이름 다섯 개(국어, 수학 등)를 괄호 안에 한자로 병기해 줘.", "'인(仁)', '의(義)'처럼 유교 덕목 다섯 개를 괄호 안 한자와 함께 설명해 줘.",
      "24절기 중 여섯 개를 괄호 안에 한자를 병기해서 알려 줘.", "한국 역사 속 왕조 이름 다섯 개를 괄호 안 한자와 함께 정리해 줘."]


SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def ask(base, prompt, mode, i):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 400, **SAMPLING, "seed": 77 + i,
            "chat_template_kwargs": {"enable_thinking": False}}
    if mode == "off": body["hanja_filter"] = False
    elif mode.startswith("auto"):
        if ":" in mode: body["hanja_penalty"] = mode.split(":")[1]
    else: body.update({"hanja_filter": True, "hanja_penalty": mode})
    r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-client-test": "hanja-intent"}), timeout=600))
    return r["choices"][0]["message"].get("content") or ""


def judge(kind, text):
    if kind == "ja": return len(KANA.findall(text)) >= 5
    if kind == "zh": return len(HAN.findall(text)) >= 10 and len(HANGUL.findall(text)) <= 2
    return len(PAREN_HAN.findall(text)) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8431")
    ap.add_argument("--modes", default="off,inf,5,auto:5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sampling", default="", help="JSON replacing the sampling fields (temperature 0.6 · top_p 0.95 · top_k 20)")
    a = ap.parse_args()
    if a.sampling:
        SAMPLING.clear(); SAMPLING.update(json.loads(a.sampling))
    items = [("ja", p) for p in JA] + [("zh", p) for p in ZH] + [("an", p) for p in AN]
    res = {}
    for mode in a.modes.split(","):
        with cf.ThreadPoolExecutor(8) as ex:
            outs = list(ex.map(lambda ip: ask(a.base, ip[1][1], mode, ip[0]), enumerate(items)))
        ok = {k: sum(judge(k, o) for (kk, _), o in zip(items, outs) if kk == k) for k in ("ja", "zh", "an")}
        res[mode] = {"pass": ok, "outputs": [{"kind": k, "prompt": p, "out": o, "ok": judge(k, o)} for (k, p), o in zip(items, outs)]}
        print(f"[intent] {mode}: Japanese {ok['ja']}/10 · Chinese {ok['zh']}/10 · annotation {ok['an']}/10", flush=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
