#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Build a draft vocabulary for the MTP draft head (HIVE_GLM_DRAFT_VOCAB): the token ids most frequent in assistant
answers of public chat corpora, written as little-endian int32 ids (most frequent first; special tokens always included).

The draft only proposes tokens; the verify step scores every draft against the full lm_head, so a smaller draft
vocabulary never changes the output — a token outside it can only be missed as a draft (one rejected step).

usage: build_draft_vocab.py --ckpt DIR --out FILE --size N [--include-script hangul] [--eval EVAL.json[l] ...] CORPUS.json[l][.gz] ...
  Corpora: ShareGPT-style JSON/JSONL ("conversations": [{"from": "gpt"|"assistant", "value": ...}]), flat message JSONL with
  a "role" (only assistant messages are used, e.g. OpenAssistant oasst2 *.messages.jsonl.gz), or JSONL with a
  "text"/"content"/"completion" field. 10 % of the answers are held out and the coverage of the list on them is printed;
  --eval corpora are only measured (coverage of the list on their answers), never counted.
  --balance: every corpus weighs the same (counts scaled to the largest corpus' token total) — e.g. short Korean
  reviews next to long English answers.
  --lang en: flat message corpora (with a "lang" field) keep only those languages.
  --include-script hangul: every vocabulary token whose text contains Hangul is in the list (before the frequency ranking),
  so Korean is covered without a Korean corpus.
"""
import argparse, collections, json, os, random, struct, sys


LANGS: set = set()


def answers(path):
    import gzip

    def from_obj(o):
        role = o.get("role")
        if LANGS and o.get("lang") is not None and o.get("lang") not in LANGS:
            return
        if isinstance(role, str) and "text" in o and not o.get("conversations") and not o.get("messages"):
            if role in ("assistant", "gpt", "model") and isinstance(o.get("text"), str) and o["text"].strip():
                yield o["text"]
            return
        conv = o.get("conversations") or o.get("messages")
        if isinstance(conv, list):
            for m in conv:
                if isinstance(m, dict) and (m.get("from") or m.get("role")) in ("gpt", "assistant", "chatgpt", "model"):
                    v = m.get("value") or m.get("content")
                    if isinstance(v, str) and v.strip():
                        yield v
            return
        for k in ("completion", "text", "content", "output", "answer"):
            v = o.get(k)
            if isinstance(v, str) and v.strip():
                yield v
                return
    with (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")) as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            for o in json.load(f):
                if isinstance(o, dict):
                    yield from from_obj(o)
        else:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        o = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(o, dict):
                        yield from from_obj(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=65536)
    ap.add_argument("--max-answers", type=int, default=200000, help="per corpus")
    ap.add_argument("--include-script", choices=["hangul"], action="append", default=[])
    ap.add_argument("--eval", nargs="+", default=[], help="corpora whose answers are only measured")
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--lang", action="append", default=[])
    ap.add_argument("--compare", help="an existing list (same format): its coverage on the --eval corpora is printed too")
    ap.add_argument("corpora", nargs="+")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    LANGS.update(a.lang)
    rnd = random.Random(20261006)
    per_train, per_held = [], []
    for path in a.corpora:
        train, held = collections.Counter(), collections.Counter()
        per_train.append(train); per_held.append(held)
        n = 0
        batch = []
        for text in answers(path):
            batch.append(text)
            n += 1
            if len(batch) == 256 or n >= a.max_answers:
                for t, ids in zip(batch, tok(batch, add_special_tokens=False)["input_ids"]):
                    (held if rnd.random() < 0.1 else train).update(ids)
                batch = []
            if n >= a.max_answers:
                break
        for t, ids in zip(batch, tok(batch, add_special_tokens=False)["input_ids"] if batch else []):
            (held if rnd.random() < 0.1 else train).update(ids)
        print(f"{os.path.basename(path)}: {n} answers · {sum(train.values())} tokens", file=sys.stderr)
    top = max(sum(c.values()) for c in per_train) or 1
    train, held = collections.Counter(), collections.Counter()
    for tc, hc in zip(per_train, per_held):
        w = top / max(1, sum(tc.values())) if a.balance else 1.0
        for i, c in tc.items():
            train[i] += c * w
        for i, c in hc.items():
            held[i] += c * w
    special = sorted(set(tok.all_special_ids) | {i for i, t in (getattr(tok, "added_tokens_decoder", {}) or {}).items()})
    script = []
    if "hangul" in a.include_script:
        import re
        hangul = re.compile("[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")
        script = [i for i in range(len(tok)) if i not in special and hangul.search(tok.decode([i]))]
    fixed = special + [i for i in script if i not in set(special)]
    have0 = set(fixed)
    ranked = [i for i, _ in train.most_common() if i not in have0]
    ids = fixed + ranked[: max(0, a.size - len(fixed))]
    have = set(ids)
    tot = sum(held.values())
    cov = sum(c for i, c in held.items() if i in have) / tot if tot else 0.0
    evals = {}
    for path in a.eval:
        ev = collections.Counter()
        texts = list(answers(path))[: a.max_answers]
        for k in range(0, len(texts), 256):
            for ids_ in tok(texts[k:k + 256], add_special_tokens=False)["input_ids"]:
                ev.update(ids_)
        n = sum(ev.values())
        evals[os.path.basename(path)] = {"tokens": n, "coverage": round(sum(c for i, c in ev.items() if i in have) / n, 5) if n else None}
        if a.compare:
            with open(a.compare, "rb") as f:
                raw = f.read()
            other = set(struct.unpack(f"<{len(raw) // 4}i", raw))
            evals[os.path.basename(path)]["compare_coverage"] = round(sum(c for i, c in ev.items() if i in other) / n, 5) if n else None
            evals[os.path.basename(path)]["overlap"] = round(len(have & other) / len(other), 4)
    with open(a.out, "wb") as f:
        f.write(struct.pack(f"<{len(ids)}i", *ids))
    print(json.dumps({"size": len(ids), "script_tokens": len(script), "train_tokens": sum(train.values()), "heldout_tokens": tot,
                      "heldout_coverage": round(cov, 5), "distinct_seen": len(train), "eval": evals}))


if __name__ == "__main__":
    main()
