#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Token list of the FN answer-span CJK filter (HIVE_QWEN_TOKEN_MASK, docs/qwen.md "Han / kana filter"): every vocabulary token whose text
(a) is two or more Han ideographs and nothing else (spaces aside) — Chinese words such as 您的 · 贵公司 — or (b) contains a kana character.
Single-ideograph tokens stay allowed (Korean writes Sino-Korean words with their Han characters in parentheses: 개항(開港)). Tokens whose bytes
do not decode to whole characters on their own (byte-level pieces of a multi-byte character) are left alone.

    python3 tools/build_cjk_mask.py /models/nvidia-Qwen3.8-Flash-Next-FP4-Local config/fn-cjk-mask.bin

Output: little-endian int32 token ids, ascending (the format of the draft-vocabulary files). Prints the counts per rule.
"""
import struct
import sys
import unicodedata

HAN = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)]
KANA = [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F)]


def _in(c, ranges):
    o = ord(c)
    return any(a <= o <= b for a, b in ranges)


def classify(text: str):
    """'han' (≥ 2 ideographs, nothing else but spaces), 'kana' (any kana), or None"""
    if not text or "�" in text:
        return None
    if any(_in(c, KANA) for c in text):
        return "kana"
    core = [c for c in text if not c.isspace()]
    if len(core) >= 2 and all(_in(c, HAN) for c in core):
        return "han"
    return None


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    n = len(tok)
    ids, cnt = [], {"han": 0, "kana": 0}
    for i in range(n):
        t = tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        k = classify(unicodedata.normalize("NFC", t))
        if k:
            ids.append(i)
            cnt[k] += 1
    with open(sys.argv[2], "wb") as f:
        f.write(struct.pack(f"<{len(ids)}i", *ids))
    print(f"vocabulary {n}: {len(ids)} masked — multi-ideograph {cnt['han']}, with kana {cnt['kana']} → {sys.argv[2]}")


if __name__ == "__main__":
    main()
