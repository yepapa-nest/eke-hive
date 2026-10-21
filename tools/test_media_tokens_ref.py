#!/usr/bin/env python3
# MediaTokens against the reference image path, with the real checkpoint (tokenizer, encoding, image_processor) on the CPU.
#   python3 tools/test_media_tokens_ref.py --ckpt /path/to/DeepSeek-V4.1-Flash[-1017] [--steps 300] [--big 300000]
# For every request of randomly grown conversations with images (in user messages and inside tool results), with images added,
# replaced and removed and earlier messages edited or dropped, MEDIATOK.prepare must return exactly what tokenize_with_images
# returns — the ids, the image metadata and the patch bytes — and its token end offsets must keep every image span whole
# (no boundary hint inside a span). Two mutants of the expansion must be caught. Then one large conversation is timed
# (whole tokenization vs the incremental path on its next turn).
import argparse, base64, glob, io, json, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--big", type=int, default=300000, help="approximate prompt tokens of the timed conversation (0 = skip)")
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()
    import hive_server as s
    from PIL import Image
    s.CFG = json.load(open(os.path.join(a.ckpt, "config.json")))
    s.ARGS = s.VisionArgs(s.CFG)
    s.load_modules(a.ckpt)
    os.environ["HIVE_TOKEN_CACHE_VERIFY"] = "0"  # the background re-check is this test's job
    assert s.MEDIATOK.usable(), "MediaTokens not usable with this checkpoint"
    rnd = random.Random(a.seed)
    docs = "".join(open(p).read() for p in sorted(glob.glob(os.path.join(HERE, "..", "docs", "*.md"))))

    def png(w, h):
        im = Image.new("RGB", (w, h), tuple(rnd.randrange(256) for _ in range(3)))
        for _ in range(6):
            x0, y0 = rnd.randrange(w), rnd.randrange(h)
            im.paste(tuple(rnd.randrange(256) for _ in range(3)), (x0, y0, min(w, x0 + rnd.randrange(1, w)), min(h, y0 + rnd.randrange(1, h))))
        b = io.BytesIO(); im.save(b, "PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    pool = [png(rnd.choice([32, 96, 200, 640, 1280, 2400]), rnd.choice([32, 80, 300, 512, 900])) for _ in range(10)]
    img = lambda: {"type": "image_url", "image_url": {"url": rnd.choice(pool)}}
    text = lambda n: docs[(o := rnd.randrange(max(1, len(docs) - n))): o + n]
    tools = [{"type": "function", "function": {"name": "browser", "description": "browse", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}}]

    def grow(m):
        k = rnd.random()
        cid = f"c{len(m)}"
        if k < 0.35:
            m += [{"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function", "function": {"name": "browser", "arguments": json.dumps({"url": "x"})}}]},
                  {"role": "tool", "tool_call_id": cid, "content": [{"type": "text", "text": text(rnd.randrange(50, 4000))}] + ([img()] if rnd.random() < 0.6 else [])}]
        elif k < 0.6:
            m += [{"role": "assistant", "content": text(rnd.randrange(20, 800))},
                  {"role": "user", "content": [{"type": "text", "text": text(rnd.randrange(10, 600))}] + [img() for _ in range(rnd.randrange(0, 3))]}]
        elif k < 0.75 and len(m) > 3:  # replace an image somewhere
            for x in rnd.sample(m, len(m)):
                if isinstance(x.get("content"), list) and any(p.get("type") == "image_url" for p in x["content"]):
                    x["content"] = [img() if p.get("type") == "image_url" else p for p in x["content"]]
                    break
        elif k < 0.85 and len(m) > 4:  # edit an earlier text message
            i = rnd.randrange(1, len(m) - 1)
            if isinstance(m[i].get("content"), str) and m[i]["content"]:
                m[i] = dict(m[i], content=m[i]["content"][: len(m[i]["content"]) // 2] + " edited")
        elif k < 0.92 and len(m) > 6:  # drop the last exchange
            del m[-2:]
        else:
            m += [{"role": "assistant", "content": text(rnd.randrange(20, 400))}, {"role": "user", "content": text(rnd.randrange(10, 300))}]
        if m[-1]["role"] == "assistant":
            m.append({"role": "user", "content": "go on"})
        return m

    def check(key, messages, mode):
        prompt, media = s.encode_prompt(json.loads(json.dumps(messages)), tools, mode, "high" if mode == "thinking" else None)
        if not media:
            return None
        want = s.tokenize_with_images(prompt, media)
        got = s.MEDIATOK.prepare(key, prompt, media)
        ids, ends, metas, blob, spans, pos = got
        ok = list(want[0]) == list(ids) and want[1] == metas and want[2] == blob and len(ends) == len(ids)
        for st, n in spans:  # every span token ends where its placeholder ends; the token before a span ends no later
            ok = ok and len(set(ends[st:st + n])) == 1 and (st == 0 or ends[st - 1] <= ends[st])
            ok = ok and all(s._out_of_spans(t, spans) == st for t in range(st + 1, st + n))
        return ok

    n_req = n_img = bad = 0
    for conv in range(max(1, a.steps // 30)):
        key = f"conv{conv}"
        mode = rnd.choice(["thinking", "chat"])
        m = [{"role": "system", "content": text(2000)}, {"role": "user", "content": [{"type": "text", "text": text(800)}, img()]}]
        for _ in range(30):
            r = check(key, m, mode)
            if r is not None:
                n_req += 1; n_img += 1
                if not r:
                    bad += 1; print("MISMATCH", key, len(m), flush=True)
            m = grow(m)
    print(f"byte-identical: {n_req - bad}/{n_req} image requests ({len(pool)} distinct images)", flush=True)

    # mutants of the expansion must be caught
    caught = 0
    orig = s.MediaTokens._image
    for name, mut in (("span one token short", lambda e: [e[0], e[1], e[2], e[3][:-1], e[4]]),
                      ("other vit grid", lambda e: [e[0], e[1] + 1, e[2], e[3], e[4]])):
        def patched(self, record, args_key, _mut=mut):
            k, e = orig(self, record, args_key)
            return ("mut",) + k, _mut(e)
        s.MediaTokens._image = patched
        s.MEDIATOK = s.MediaTokens()
        m = [{"role": "system", "content": "sys"}, {"role": "user", "content": [{"type": "text", "text": "look"}, img()]}]
        caught += check("mut-" + name, m, "chat") is False
        s.MediaTokens._image = orig
    s.MEDIATOK = s.MediaTokens()
    print(f"mutants caught: {caught}/2", flush=True)

    if a.big:
        m = [{"role": "system", "content": "sys"}]
        while len(json.dumps(m)) < a.big * 3.2:
            m += [{"role": "user", "content": [{"type": "text", "text": text(6000)}] + ([img()] if rnd.random() < 0.15 else [])},
                  {"role": "assistant", "content": text(1500)}]
        m.append({"role": "user", "content": "q1"})
        prompt, media = s.encode_prompt(json.loads(json.dumps(m)), tools, "chat", None)
        t = time.perf_counter(); want = s.tokenize_with_images(prompt, media); t_whole = time.perf_counter() - t
        s.MEDIATOK.prepare("big", prompt, media)  # first request of the conversation: whole tokenization through the cache
        m += [{"role": "assistant", "content": "a1"}, {"role": "user", "content": [{"type": "text", "text": "q2"}, img()]}]
        prompt, media = s.encode_prompt(json.loads(json.dumps(m)), tools, "chat", None)
        t = time.perf_counter(); got = s.MEDIATOK.prepare("big", prompt, media); t_inc = time.perf_counter() - t
        want2 = s.tokenize_with_images(prompt, media)
        same = list(want2[0]) == list(got[0]) and want2[1] == got[2] and want2[2] == got[3]
        print(f"big: {len(want[0])} tokens · whole {t_whole * 1000:.0f} ms · next turn incremental {t_inc * 1000:.0f} ms · identical {same}", flush=True)
        bad += 0 if same else 1
    sys.exit(0 if bad == 0 and caught == 2 and n_req > 0 else 1)


if __name__ == "__main__":
    main()
