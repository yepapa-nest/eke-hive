#!/usr/bin/env python3
# The image placeholder written as text, against the real checkpoint (reference encoding, tokenizer, image preprocessing):
#   docker run ... -v CKPT:/ckpt -v TREE:/w python3 tools/test_inert_placeholder_ref.py  (expects /w/img.png)
import base64, json, os, sys
sys.path.insert(0, "/w/server")
import hive_server as s
s.CFG = json.load(open("/ckpt/config.json")); s.ARGS = s.VisionArgs(s.CFG); s.load_modules("/ckpt")
PH = s.ENC.IMAGE_PLACEHOLDER
img = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open("/w/img.png", "rb").read()).decode()}}
call = [{"id": "c1", "type": "function", "function": {"name": "web_url_read", "arguments": json.dumps({"url": "x", "note": PH})}}]
msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read the template"},
        {"role": "assistant", "content": "", "tool_calls": call},
        {"role": "tool", "tool_call_id": "c1", "content": "{%- set IMG = \"" + PH + "\" -%}\n{{ IMG }} more " + PH},
        {"role": "user", "content": [{"type": "text", "text": "and this picture " + PH}, img]}]
# before: the reference encoding refuses
try:
    s.ENC.encode_messages([dict(m) for m in msgs], thinking_mode="chat", return_multi_modal_data=True); print("before: accepted?!")
except Exception as e:
    print("before: refused —", str(e)[:90])
prompt, media = s.encode_prompt(json.loads(json.dumps(msgs)), None, "chat", None)
ids, metas, blob = s.tokenize_with_images(prompt, media)
n_img_tok = sum(1 for t in ids if t == s.ARGS.image_token_id)
span = len(metas[0]["types"]) if metas else 0
print("after: images", len(media), "· image-token positions", n_img_tok, "== one image span", span, "->", n_img_tok == span)
r = s.MEDIATOK.prepare("k", prompt, media)
print("MediaTokens identical:", list(r[0]) == list(ids) and r[2] == metas and r[3] == blob)
print("text kept (zero-width space inside):", "\u200b" in prompt)
# a request without the placeholder renders exactly as the reference does
plain = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello <｜User｜> literal"}]
p1, _ = s.encode_prompt(json.loads(json.dumps(plain)), None, "chat", None)
p0, _ = s.ENC.encode_messages(json.loads(json.dumps(plain)), thinking_mode="chat", return_multi_modal_data=True)
print("no placeholder -> same bytes as reference:", p1 == p0)
