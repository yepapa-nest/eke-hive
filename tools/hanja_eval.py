#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Han / kana leakage of Korean answers (the FN answer-span filter, docs/qwen.md): a fixed set of Korean free-response prompts (109 topics ×
8 forms, seed-shuffled), sampled (default temperature 0.6 · top_p 0.95 · top_k 20, fixed seeds), with the
filter off and on (or several penalties Δ: --modes off,inf,3,5,8). Counts per run: Han characters and kana characters in the answers, answers with a Han word (a run of ≥ 2 ideographs not
inside parentheses — 개항(開港) is an annotation, not a leak), answers with any kana, and output tokens.

    python3 tools/hanja_eval.py --base http://127.0.0.1:8431 --out hanja.json [--n 400] [--conc 8] [--thinking-share 0.25]
"""
import argparse, concurrent.futures as cf, json, random, re, sys, time, urllib.request

HAN = "㐀-䶿一-鿿豈-﫿"
KANA = "぀-ゟ゠-ヿㇰ-ㇿｦ-ﾟ"
RE_HAN, RE_KANA = re.compile(f"[{HAN}]"), re.compile(f"[{KANA}]")
RE_HAN_WORD = re.compile(f"[{HAN}]{{2,}}")
RE_PAREN = re.compile(r"\([^()]*\)|（[^（）]*）")

TOPICS = ["조선 시대의 과거 제도", "한국의 김치 문화", "반도체 공정", "기후 변화와 해수면 상승", "주식과 채권의 차이", "고려청자", "한글 창제",
          "임진왜란", "개항과 근대화", "인공지능 윤리", "양자 컴퓨터", "광합성", "블록체인", "한국 전쟁", "세종대왕의 업적", "불교의 전래",
          "유교 사상", "실학", "동학 농민 운동", "삼국 통일", "훈민정음", "백제 문화", "신라 금관", "한옥 구조", "온돌", "판소리", "탈춤",
          "태권도의 역사", "한국 경제 성장", "새마을 운동", "IMF 외환 위기", "K-팝의 세계화", "드라마 수출", "전기차 배터리", "수소 에너지",
          "원자력 발전", "미세먼지", "저출산 문제", "고령화 사회", "부동산 정책", "금리와 물가", "환율 변동", "무역 수지", "스마트폰 산업",
          "조선업", "자동차 산업", "철강 산업", "농업 기술", "해양 생태계", "독도", "제주도 화산 지형", "백두산", "한강의 역사", "서울의 궁궐",
          "경주 역사 지구", "한국의 사찰", "한방 의학", "침술", "인삼", "발효 음식", "장 담그기", "한국어 높임법", "사자성어의 뜻", "한자어와 고유어",
          "맞춤법", "띄어쓰기", "외래어 표기법", "문학의 갈래", "시조", "가사 문학", "현대 소설", "근대 신문", "독립운동", "임시 정부",
          "헌법의 기본 원리", "삼권 분립", "선거 제도", "지방 자치", "형법과 민법", "계약서 작성", "근로기준법", "세금 신고", "건강 보험",
          "국민연금", "운동과 건강", "수면 위생", "스트레스 관리", "영양소", "백신의 원리", "항생제 내성", "유전자 가위", "뇌과학", "우주 탐사",
          "블랙홀", "상대성 이론", "열역학", "전자기학", "화학 결합", "산과 염기", "미적분", "확률과 통계", "알고리즘", "데이터베이스", "운영체제",
          "네트워크 프로토콜", "보안과 암호", "클라우드 컴퓨팅", "프로그래밍 언어", "소프트웨어 테스트", "프로젝트 관리", "회의 진행", "보고서 작성"]
FORMS = ["{t}에 대해 초등학생도 이해할 수 있게 설명해 줘.", "{t}의 핵심을 다섯 문장으로 요약해 줘.", "{t}의 장점과 한계를 각각 세 가지씩 알려 줘.",
         "{t}와 관련된 역사적 배경을 설명해 줘.", "{t}에 관한 짧은 수필을 써 줘.", "{t}를 처음 공부하는 사람에게 추천할 학습 순서를 알려 줘.",
         "{t}에 대해 자주 하는 오해 세 가지와 바로잡는 설명을 써 줘.", "{t}에 관해 친구에게 보내는 메시지처럼 편하게 설명해 줘."]


def prompts(n, seed=20261031):
    rnd = random.Random(seed)
    out = [f.format(t=t) for t in TOPICS for f in FORMS]
    rnd.shuffle(out)
    return out[:n]


def run_one(base, p, i, filt, thinking, a):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p,
            "top_k": a.top_k, "seed": a.seed_base + i, "chat_template_kwargs": {"enable_thinking": thinking},
            **({"min_p": a.min_p} if a.min_p > 0 else {}), **({"presence_penalty": a.presence_penalty} if a.presence_penalty else {}),
            **({"reasoning_effort": a.effort} if thinking and a.effort else {})}
    body.update(filt)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-client-test": "hanja-eval"})
    for attempt in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=900))
            m = r["choices"][0]["message"]
            return {"i": i, "thinking": thinking, "content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
                    "tokens": r["usage"]["completion_tokens"]}
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2)
    return {"i": i, "error": err}


def stats(rows):
    s = {"answers": 0, "han_chars": 0, "kana_chars": 0, "han_word_answers": 0, "kana_answers": 0, "tokens": 0, "errors": 0, "examples": []}
    for r in rows:
        if "error" in r:
            s["errors"] += 1
            continue
        c = r["content"]
        s["answers"] += 1
        s["tokens"] += r["tokens"]
        s["han_chars"] += len(RE_HAN.findall(c))
        s["kana_chars"] += len(RE_KANA.findall(c))
        w = RE_HAN_WORD.findall(RE_PAREN.sub("", c))
        if w:
            s["han_word_answers"] += 1
            if len(s["examples"]) < 12:
                s["examples"].append({"i": r["i"], "words": w[:4]})
        if RE_KANA.search(c):
            s["kana_answers"] += 1
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8431")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--thinking-share", type=float, default=0.25)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--effort", default="", help="thinking level for the thinking share (low / medium); empty = the template default")
    ap.add_argument("--modes", default="off,on", help="comma list: off · on · inf · <Δ> · auto · auto:<Δ>")
    a = ap.parse_args()
    ps = prompts(a.n)
    nth = int(len(ps) * a.thinking_share)
    res = {"meta": {"n": len(ps), "thinking": nth, "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "min_p": a.min_p,
                    "presence_penalty": a.presence_penalty, "seed_base": a.seed_base, "effort": a.effort, "max_tokens": a.max_tokens}}
    for mode in a.modes.split(","):
        # off = no penalty · on = the server's default Δ · inf / a number = that Δ (forced on) · auto[:Δ] = the server's default rule
        #   (HIVE_HANJA_FILTER=1 with auto-off), optionally with Δ
        if mode == "off": filt = {"hanja_filter": False}
        elif mode == "on": filt = {"hanja_filter": True}
        elif mode.startswith("auto"): filt = {"hanja_penalty": mode.split(":")[1]} if ":" in mode else {}
        else: filt = {"hanja_filter": True, "hanja_penalty": mode}
        t0 = time.time()
        with cf.ThreadPoolExecutor(a.conc) as ex:
            rows = list(ex.map(lambda ip: run_one(a.base, ip[1], ip[0], filt, ip[0] < nth, a), enumerate(ps)))
        st = stats(rows)
        st["wall_s"] = round(time.time() - t0, 1)
        res[mode] = st
        res[mode + "_rows"] = rows
        print(f"[hanja] filter {mode}: answers {st['answers']} · Han chars {st['han_chars']} · kana chars {st['kana_chars']} · "
              f"answers with a Han word {st['han_word_answers']} · with kana {st['kana_answers']} · tokens {st['tokens']} · errors {st['errors']} · "
              f"{st['wall_s']} s", flush=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
