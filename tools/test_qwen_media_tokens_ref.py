#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# MediaTokens for the Qwen3.8-Flash-Next family adapter against the family's own image path, with the real checkpoint tokenizer on the CPU.
#   python3 tools/test_qwen_media_tokens_ref.py --ckpt /path/to/Qwen3.8-Flash-Next-NVFP4 [--steps 200]
# For every request of randomly grown conversations with images (in user messages and inside tool results), with images added, replaced and
# removed and earlier messages edited, MEDIATOK.prepare must return exactly what tokenize_with_images (families/qwen.py prepare_inputs)
# returns — the ids, the image metadata and the patch bytes — and its token end offsets must keep every image span whole. A request with
# a video part, or with a literal pad token in its text, takes the family path (prepare raises). Each distinct image is preprocessed once.
# Two mutants of the expansion must be caught.
import argparse, base64, io, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261030)
    a = ap.parse_args()
    import hive_server as s
    from PIL import Image
    s.load_modules(a.ckpt)
    assert s.adapter_family() and s.FAMILY == "qwen4_exp", s.FAMILY
    os.environ["HIVE_TOKEN_CACHE_VERIFY"] = "0"  # the background re-check is this test's job
    assert s.MEDIATOK.usable(), "MediaTokens not usable for the Qwen adapter"
    rnd = random.Random(a.seed)
    fails = []

    def check(what, cond, detail=""):
        print(("ok   " if cond else "FAIL ") + what + ("" if cond else f" — {detail}"))
        if not cond:
            fails.append(what)

    def png(w, h):
        im = Image.new("RGB", (w, h), tuple(rnd.randrange(256) for _ in range(3)))
        for _ in range(6):
            x0, y0 = rnd.randrange(w), rnd.randrange(h)
            im.paste(tuple(rnd.randrange(256) for _ in range(3)), (x0, y0, min(w, x0 + rnd.randrange(1, w)), min(h, y0 + rnd.randrange(1, h))))
        b = io.BytesIO(); im.save(b, "PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    pool = [png(rnd.choice([40, 96, 300, 640, 1280]), rnd.choice([40, 80, 300, 512, 900])) for _ in range(8)]
    img = lambda: {"type": "image_url", "image_url": {"url": rnd.choice(pool)}}
    words = ("한국어 문장과 English words, code `x = f(y)` 그리고 숫자 12345 · 표 | a | b | 줄바꿈\n").split(" ")
    text = lambda n: " ".join(rnd.choice(words) for _ in range(n))
    tools = [{"type": "function", "function": {"name": "browser", "description": "browse",
                                               "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}}]

    def grow(m):
        k, cid = rnd.random(), f"c{len(m)}"
        if k < 0.35:
            m += [{"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function", "function": {"name": "browser", "arguments": json.dumps({"url": "x"})}}]},
                  {"role": "tool", "tool_call_id": cid, "content": [{"type": "text", "text": text(rnd.randrange(5, 300))}] + ([img()] if rnd.random() < 0.6 else [])}]
        elif k < 0.6:
            m += [{"role": "assistant", "content": text(rnd.randrange(3, 120))},
                  {"role": "user", "content": [{"type": "text", "text": text(rnd.randrange(3, 80))}] + [img() for _ in range(rnd.choice([0, 1, 1, 2]))]}]
        elif k < 0.75 and len(m) > 3:  # edit an earlier user message (the token cache must cut there)
            for msg in m[1:]:
                if msg["role"] == "user":
                    msg["content"] = [{"type": "text", "text": text(rnd.randrange(3, 40))}] + ([img()] if rnd.random() < 0.5 else [])
                    break
        elif k < 0.85:  # drop images from the last user message
            for msg in reversed(m):
                if msg["role"] == "user" and isinstance(msg["content"], list):
                    msg["content"] = [p for p in msg["content"] if p.get("type") == "text"] or [{"type": "text", "text": "x"}]
                    break
        else:
            m += [{"role": "assistant", "content": text(5)}, {"role": "user", "content": text(rnd.randrange(3, 60))}]

    n_req = n_img = 0
    for conv in range(6):
        key = f"conv{conv}"
        m = [{"role": "system", "content": "너는 비서다."}, {"role": "user", "content": [{"type": "text", "text": "시작"}, img()]}]
        for step in range(a.steps // 6):
            grow(m)
            msgs = [dict(x) for x in m]
            prompt, media = s.encode_prompt(msgs, tools, "thinking", "low")
            if not media:
                continue
            want = s.tokenize_with_images(prompt, media)
            got = s.MEDIATOK.prepare(key, prompt, media)
            ids, ends, metas, blob, spans, pos = got
            same = (list(want[0]), want[1], want[2]) == (ids, metas, blob)
            whole = all(len({ends[j] for j in range(st, st + n)}) == 1 for st, n in spans) and len(ends) == len(ids)
            n_req += 1; n_img += len(media)
            if not same or not whole:
                check(f"{key} step {step}: equal to prepare_inputs and spans whole", False,
                      f"ids {list(want[0]) == ids} metas {want[1] == metas} blob {want[2] == blob} whole {whole}")
                break
    check(f"{n_req} requests · {n_img} images: ids, metas and patch bytes equal to prepare_inputs; every span keeps one end offset", not fails)
    check(f"each distinct image preprocessed once (cache holds {len(s.MEDIATOK.images)} ≤ {len(pool)})", len(s.MEDIATOK.images) <= len(pool))

    # video parts and a literal pad token take the family path
    vid = [{"role": "user", "content": [{"type": "text", "text": "v"}, {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]}]
    try:
        p, md = s.encode_prompt(vid, None, "chat", None)
        s.MEDIATOK.prepare("v", p, md)
        check("video part raises (family path)", False, "no raise")
    except ValueError:
        check("video part raises (family path)", True)
    lit = [{"role": "user", "content": [{"type": "text", "text": "literal <|image_pad|> here"}, img()]}]
    p, md = s.encode_prompt(lit, None, "chat", None)
    try:
        s.MEDIATOK.prepare("lit", p, md)
        check("literal pad token in text raises (family path)", False, "no raise")
    except ValueError:
        check("literal pad token in text raises (family path)", True)

    # mutants: the expansion must be checked against prepare_inputs, not merely run
    m = [{"role": "user", "content": [{"type": "text", "text": "두 장"}, img(), img()]}]
    p, md = s.encode_prompt(m, None, "chat", None)
    want = s.tokenize_with_images(p, md)
    orig = s.ENC.image_part_patches
    s.MEDIATOK.images.clear()
    s.ENC.image_part_patches = lambda d: (lambda r: (r[0], r[2], r[1]))(orig(d))  # gh/gw swapped
    got = s.MEDIATOK.prepare("mut1", p, md)
    check("mutant gh/gw swap detected", (list(want[0]), want[1], want[2]) != (got[0], got[2], got[3]) or all(x["n_vit_h"] == x["n_vit_w"] for x in want[1]))
    s.ENC.image_part_patches = orig
    s.MEDIATOK.images.clear()
    s.ENC.image_part_patches = lambda d: (lambda r: (r[0][:-2] + b"\0\0", r[1], r[2]))(orig(d))  # last patch value changed
    got = s.MEDIATOK.prepare("mut2", p, md)
    check("mutant patch bytes detected", want[2] != got[3])
    s.ENC.image_part_patches = orig
    if fails:
        print(f"qwen media tokens: {len(fails)} failures"); sys.exit(1)
    print("qwen media tokens: ALL OK")


if __name__ == "__main__":
    main()
