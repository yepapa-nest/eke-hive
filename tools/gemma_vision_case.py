# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""A Gemma image case through the server's own preprocessing (server/families/gemma.py): a chat request with one or more images →
OUT/prompt.json (token ids), OUT/images.json + OUT/img_K.bf16 (the format of oracle/gemma4_oracle.py --images and gemma_check
HIVE_CHECK_IMAGES), OUT/request.json (the chat request, for a live server).

  python3 tools/gemma_vision_case.py --ckpt DIR --image a.png [--image b.jpg] --text "What is in the picture?" --out OUT [--hf]

--hf: also run the checkpoint's Hugging Face processor (transformers with gemma4_unified) on the same request and compare the token ids,
the merged patches and their positions with this project's preprocessing (reported, not asserted: resampling may differ by a level).
"""
import argparse, base64, json, mimetypes, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", action="append", required=True)
    ap.add_argument("--text", default="Describe this image.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--hf", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer
    from families import gemma

    tok = AutoTokenizer.from_pretrained(a.ckpt)
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    gemma.init(tok, cfg, a.ckpt)
    parts = []
    for p in a.image:
        mt = mimetypes.guess_type(p)[0] or "image/png"
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mt};base64," + base64.b64encode(open(p, "rb").read()).decode()}})
    parts.append({"type": "text", "text": a.text})
    messages = [{"role": "user", "content": parts}]
    json.dump({"messages": messages}, open(os.path.join(a.out, "request.json"), "w"))
    mode = "thinking" if a.thinking else "chat"
    prompt, media = gemma.encode_messages(messages, mode, return_multi_modal_data=True)
    ids, metas, blob = gemma.prepare_inputs(prompt, media["images"], tok)
    json.dump(ids, open(os.path.join(a.out, "prompt.json"), "w"))
    out, off = [], 0
    for k, m in enumerate(metas):
        f = f"img_{k}.bf16"
        open(os.path.join(a.out, f), "wb").write(blob[off:off + m["nbytes"]])
        off += m["nbytes"]
        out.append({"start": m["start"], "gh": m["n_vit_h"], "gw": m["n_vit_w"], "file": f})
    json.dump(out, open(os.path.join(a.out, "images.json"), "w"))
    print(json.dumps({"tokens": len(ids), "images": out}))
    if not a.hf:
        return
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(a.ckpt)
    hf_text = prompt.replace(gemma.IMAGE_TOKEN, proc.image_token) if gemma.IMAGE_TOKEN != proc.image_token else prompt
    r = proc(text=[hf_text], images=[[Image.open(p).convert("RGB") for p in a.image]], return_tensors="pt", add_special_tokens=False)
    hf_ids = r["input_ids"][0].tolist()
    rep = {"ids_equal": hf_ids == ids, "hf_tokens": len(hf_ids)}
    pv, pp = r["pixel_values"], r["image_position_ids"]
    for k, m in enumerate(out):
        n = m["gh"] * m["gw"]
        u = np.fromfile(os.path.join(a.out, m["file"]), dtype=np.uint16).astype(np.uint32) << 16
        mine = torch.from_numpy(u.view(np.float32).reshape(n, -1).copy())
        theirs = pv[k][:n].float()
        valid = (pp[k] != -1).all(-1).sum().item()
        g = torch.arange(n)
        pos_ok = bool(torch.equal(pp[k][:n].long(), torch.stack([g % m["gw"], g // m["gw"]], -1)))
        d = (mine - theirs).abs()
        rep[f"image_{k}"] = {"soft_tokens": n, "hf_soft_tokens": valid, "positions_equal": pos_ok, "max_abs_diff": float(d.max()),
                             "mean_abs_diff": float(d.mean()), "max_diff_in_levels": round(float(d.max()) * 255, 2)}
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(a.out, "hf_compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
