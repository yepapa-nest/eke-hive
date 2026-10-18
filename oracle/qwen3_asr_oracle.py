#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Qwen3-ASR oracle: an fp32 torch model of Qwen3-ASR 1.7B (config model_type "qwen3_asr") written for the hive engine to be scored
against. Independent implementation from the published model description and checkpoint layout; tools/test_qasr_ref_dump.py produces
the reference outputs it is validated with (greedy text on the clips, first-step logits).

Model (checkpoint prefix "thinker."):
  Front end  128-bin log-mel, 100 frames/s (server/speech/qasr_frontend.py).
  Audio encoder ("audio_tower.", width 1024, 24 layers, 16 heads of 64):
    · the mel frames are cut into chunks of 2*n_window = 100 frames; a shorter tail chunk is zero-padded to the longest chunk
      (i.e. to 100 frames whenever there is more than one chunk). Each chunk goes through three Conv2d(k=3, stride=2, pad=1) over
      (mel, time) with exact (erf) GELU after each: 1 -> 480 -> 480 -> 480 channels, 128 -> 64 -> 32 -> 16 mel rows, 100 -> 50 -> 25 -> 13
      frames. conv_out (no bias) maps the 480*16 values of a frame (channel-major: index c*16 + f) to 1024.
    · sinusoidal positions restart in every chunk: pos[t] = [sin(t*w_i), cos(t*w_i)], w_i = 10000^(-i/511), i = 0..511.
    · only the valid frames of every chunk are kept (13 per full chunk, ceil(ceil(ceil(r/2)/2)/2) for a tail of r frames), concatenated.
    · attention is block-diagonal: windows of 13 * (n_window_infer / 100) = 104 consecutive tokens (the last one shorter), bidirectional,
      softmax(q k^T / 8). Pre-LayerNorm blocks (eps 1e-5, with bias): x += out_proj(attn(LN(x))); x += fc2(gelu(fc1(LN(x)))). All
      linears have biases. Then ln_post -> proj1 -> gelu -> proj2 (1024 -> 2048).
  Decoder ("model.", Qwen3 1.7B: 28 layers, width 2048, 16 query / 8 key-value heads of 128, SwiGLU 6144, RMSNorm eps 1e-6):
    · input embeddings = embed_tokens(ids), with the audio encoder output written over the <|audio_pad|> positions (one pad token per
      audio token).
    · x += o_proj(attn(rmsnorm(x))); x += down(silu(gate(h)) * up(h)), h = rmsnorm(x). q and k get a per-head RMSNorm (weight [128])
      before the rotary embedding; no attention biases; causal softmax(q k^T / sqrt(128)) with each kv head shared by 2 query heads.
    · rotary embedding: "interleaved mRoPE" with sections [24, 20, 20] over 3 position axes — but every token (text and audio) gets the
      same position on all three axes (its index in the sequence, 0, 1, 2, ...), so each frequency pair sees the same position and it
      reduces to standard rotate-half RoPE: theta = 1e6, inv_freq_i = theta^(-2i/128), x[i] and x[i+64] rotated by pos*inv_freq_i.
    · final RMSNorm, lm_head (stored separately, identical to embed_tokens in this checkpoint).
  Generation: greedy (or sampled) until <|im_end|> (151645) or <|endoftext|> (151643).

Usage:
  qwen3_asr_oracle.py --ckpt <dir> --wavs <dir> [--transcript transcript.tsv] [--ref <ref dir>] [--dump <dir>] [--threads 16]
  --ref compares with tools/test_qasr_ref_dump.py output: prompt ids, generated ids/text per clip, cosine of first-step logits.
  --dump writes per clip: <clip>.audio.npy (encoder output [Ta, 2048]), <clip>.logits.npy (first-step logits), and oracle.json
  (prompt pieces, generated ids, text) for the engine comparison (tools/test_qasr_engine.py).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server" / "speech"))
import qasr_frontend as fe  # noqa: E402

AUDIO_PAD = 151676
EOS_IDS = (151645, 151643)
PROMPT = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
          "<|im_start|>assistant\n")


def load_weights(ckpt: Path, prefix: str = "thinker.") -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    idx = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for shard in sorted(set(idx.values())):
        with safe_open(str(ckpt / shard), "pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    out[k[len(prefix):]] = f.get_tensor(k).float()
    return out


class Qwen3ASROracle:
    def __init__(self, ckpt: str):
        self.ckpt = Path(ckpt)
        cfg = json.loads((self.ckpt / "config.json").read_text())["thinker_config"]
        self.ac, self.tc = cfg["audio_config"], cfg["text_config"]
        self.w = load_weights(self.ckpt)
        tc = self.tc
        self.L = tc["num_hidden_layers"]
        self.H, self.KV, self.D = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
        self.eps = tc["rms_norm_eps"]
        self.theta = float(tc["rope_theta"])
        d = self.ac["d_model"]
        half = d // 2
        inv = torch.exp(-math.log(10000.0) / (half - 1) * torch.arange(half, dtype=torch.float64))
        t = torch.arange(self.ac["max_source_positions"], dtype=torch.float64)[:, None] * inv[None, :]
        self.audio_pos = torch.cat([torch.sin(t), torch.cos(t)], dim=1).float()
        self.inv_freq = (1.0 / (self.theta ** (torch.arange(0, self.D, 2, dtype=torch.float64) / self.D))).float()

    # ---------------- audio encoder ----------------
    @staticmethod
    def conv_len(n: int) -> int:
        for _ in range(3):
            n = (n - 1) // 2 + 1
        return n

    def encode(self, mel: np.ndarray) -> torch.Tensor:
        w, ac = self.w, self.ac
        mel = torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32))
        n_frames = mel.shape[1]
        cl = 2 * ac["n_window"]
        lens = [cl] * (n_frames // cl) + ([n_frames % cl] if n_frames % cl else [])
        lp = max(lens)
        x = torch.zeros(len(lens), 1, mel.shape[0], lp)
        for i, n in enumerate(lens):
            x[i, 0, :, :n] = mel[:, i * cl:i * cl + n]
        for j in (1, 2, 3):
            x = F.gelu(F.conv2d(x, w[f"audio_tower.conv2d{j}.weight"], w[f"audio_tower.conv2d{j}.bias"], stride=2, padding=1))
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f) @ w["audio_tower.conv_out.weight"].T
        x = x + self.audio_pos[:t][None]
        h = torch.cat([x[i, :self.conv_len(n)] for i, n in enumerate(lens)], dim=0)
        win = self.conv_len(lp) * (ac["n_window_infer"] // cl)
        n_tok = h.shape[0]
        bounds = [(s, min(s + win, n_tok)) for s in range(0, n_tok, win)]
        nh = ac["encoder_attention_heads"]
        hd = ac["d_model"] // nh
        for li in range(ac["encoder_layers"]):
            p = f"audio_tower.layers.{li}."
            r = h
            y = F.layer_norm(h, (h.shape[1],), w[p + "self_attn_layer_norm.weight"], w[p + "self_attn_layer_norm.bias"], 1e-5)
            q = F.linear(y, w[p + "self_attn.q_proj.weight"], w[p + "self_attn.q_proj.bias"]).view(n_tok, nh, hd)
            k = F.linear(y, w[p + "self_attn.k_proj.weight"], w[p + "self_attn.k_proj.bias"]).view(n_tok, nh, hd)
            v = F.linear(y, w[p + "self_attn.v_proj.weight"], w[p + "self_attn.v_proj.bias"]).view(n_tok, nh, hd)
            o = torch.empty(n_tok, nh, hd)
            for s, e in bounds:
                sc = torch.einsum("qhd,khd->hqk", q[s:e], k[s:e]) / math.sqrt(hd)
                o[s:e] = torch.einsum("hqk,khd->qhd", torch.softmax(sc, dim=-1), v[s:e])
            h = r + F.linear(o.reshape(n_tok, -1), w[p + "self_attn.out_proj.weight"], w[p + "self_attn.out_proj.bias"])
            r = h
            y = F.layer_norm(h, (h.shape[1],), w[p + "final_layer_norm.weight"], w[p + "final_layer_norm.bias"], 1e-5)
            y = F.gelu(F.linear(y, w[p + "fc1.weight"], w[p + "fc1.bias"]))
            h = r + F.linear(y, w[p + "fc2.weight"], w[p + "fc2.bias"])
        h = F.layer_norm(h, (h.shape[1],), w["audio_tower.ln_post.weight"], w["audio_tower.ln_post.bias"], 1e-5)
        h = F.gelu(F.linear(h, w["audio_tower.proj1.weight"], w["audio_tower.proj1.bias"]))
        return F.linear(h, w["audio_tower.proj2.weight"], w["audio_tower.proj2.bias"])

    # ---------------- decoder ----------------
    def rms(self, x, wname):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w[wname]

    def rope(self, x, pos):  # x [T, heads, 128]
        ang = pos[:, None].float() * self.inv_freq[None, :]
        cos, sin = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
        a, b = x[..., : self.D // 2], x[..., self.D // 2:]
        return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)

    def forward(self, emb: torch.Tensor, cache: list, start: int) -> torch.Tensor:
        """emb [T, 2048] at positions start..start+T-1; cache = per-layer [K, V] lists (appended). Returns last-token logits."""
        w = self.w
        T = emb.shape[0]
        pos = torch.arange(start, start + T)
        x = emb
        rep = self.H // self.KV
        for li in range(self.L):
            p = f"model.layers.{li}."
            y = self.rms(x, p + "input_layernorm.weight")
            q = (y @ w[p + "self_attn.q_proj.weight"].T).view(T, self.H, self.D)
            k = (y @ w[p + "self_attn.k_proj.weight"].T).view(T, self.KV, self.D)
            v = (y @ w[p + "self_attn.v_proj.weight"].T).view(T, self.KV, self.D)
            q = self.rope(self.rms(q, p + "self_attn.q_norm.weight"), pos)
            k = self.rope(self.rms(k, p + "self_attn.k_norm.weight"), pos)
            if len(cache) <= li:
                cache.append([k, v])
            else:
                cache[li][0] = torch.cat([cache[li][0], k])
                cache[li][1] = torch.cat([cache[li][1], v])
            K, V = cache[li]
            S = K.shape[0]
            Kr, Vr = K.repeat_interleave(rep, dim=1), V.repeat_interleave(rep, dim=1)
            sc = torch.einsum("qhd,khd->hqk", q, Kr) / math.sqrt(self.D)
            qpos = pos[:, None]
            kpos = torch.arange(S)[None, :]
            sc = sc.masked_fill((kpos > qpos)[None], float("-inf"))
            o = torch.einsum("hqk,khd->qhd", torch.softmax(sc, dim=-1), Vr).reshape(T, -1)
            x = x + o @ w[p + "self_attn.o_proj.weight"].T
            y = self.rms(x, p + "post_attention_layernorm.weight")
            x = x + (F.silu(y @ w[p + "mlp.gate_proj.weight"].T) * (y @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T
        x = self.rms(x[-1:], "model.norm.weight")
        return (x @ w["lm_head.weight"].T)[0]

    def embed(self, ids):
        return self.w["model.embed_tokens.weight"][torch.as_tensor(ids, dtype=torch.long)]

    @torch.no_grad()
    def transcribe(self, mel: np.ndarray, pre: list[int], post: list[int], max_new: int = 256, temperature: float = 0.0,
                   seed: int = 0) -> dict:
        audio = self.encode(mel)
        emb = torch.cat([self.embed(pre), audio, self.embed(post)], dim=0)
        cache: list = []
        logits = self.forward(emb, cache, 0)
        first = logits.clone()
        n = emb.shape[0]
        out = []
        gen = torch.Generator().manual_seed(seed)
        for _ in range(max_new):
            if temperature > 0:
                tok = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen))
            else:
                tok = int(torch.argmax(logits))
            if tok in EOS_IDS:
                break
            out.append(tok)
            logits = self.forward(self.embed([tok]), cache, n)
            n += 1
        return {"audio": audio, "first_logits": first, "ids": out, "n_prompt": len(pre) + audio.shape[0] + len(post)}


def prompt_pieces(tok, prompt: str) -> tuple[list[int], list[int]]:
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    assert ids.count(AUDIO_PAD) == 1, "the prompt must contain exactly one <|audio_pad|>"
    i = ids.index(AUDIO_PAD)
    return ids[:i], ids[i + 1:]


def norm_text(s: str) -> str:
    """CER normalisation: drop whitespace and common punctuation (the rule the speech benchmarks this replaces used)."""
    import re
    return re.sub(r"[\s\.,\?\!~·'\"“”‘’…:;\-]", "", s)


def cer(ref: str, hyp: str) -> float:
    r, h = norm_text(ref), norm_text(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wavs", required=True)
    ap.add_argument("--transcript", default="")
    ap.add_argument("--language", default="Korean")
    ap.add_argument("--ref", default="")
    ap.add_argument("--dump", default="")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--only", default="")
    ap.add_argument("--layers", type=int, default=0, help="validation: keep only the first N decoder layers (engine: HIVE_QASR_DEBUG_LAYERS)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    t0 = time.time()
    m = Qwen3ASROracle(a.ckpt)
    if a.layers > 0:
        m.L = min(m.L, a.layers)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    pre, post = prompt_pieces(tok, PROMPT + (f"language {a.language}<asr_text>" if a.language else ""))
    ref = json.loads((Path(a.ref) / "ref.json").read_text()) if a.ref else {}
    gold = {}
    if a.transcript:
        for line in Path(a.transcript).read_text().splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                gold[k] = v
    dump = Path(a.dump) if a.dump else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
    rec = {"pre": pre, "post": post, "clips": {}}
    clips = sorted(Path(a.wavs).glob("*.wav"))
    if a.only:
        clips = [c for c in clips if c.name in set(a.only.split(","))]
    same = n_ref = 0
    cers = []
    for c in clips:
        t0 = time.time()
        pcm = fe.decode_wav(c.read_bytes())
        mel = fe.log_mel(pcm)
        r = m.transcribe(mel, pre, post, a.max_new)
        text = tok.decode(r["ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        line = f"{c.name}: {time.time() - t0:.1f}s n_prompt={r['n_prompt']} {len(r['ids'])} tok  {text}"
        if c.name in ref:
            rr = ref[c.name]
            want_prompt = pre + [AUDIO_PAD] * r["audio"].shape[0] + post
            n_ref += 1
            ref_ids = list(rr["ids"])
            while ref_ids and ref_ids[-1] in EOS_IDS:  # the reference keeps the stop token in its output
                ref_ids.pop()
            ok_ids = ref_ids == r["ids"]
            same += ok_ids
            line += f"  | prompt {'==' if rr['prompt_ids'] == want_prompt else '!='} ref, ids {'==' if ok_ids else '!='} ref"
            lp = Path(a.ref) / f"{c.name}.logits.npy"
            if lp.exists():
                lr = torch.from_numpy(np.load(lp))
                lo = r["first_logits"]
                cos = float(F.cosine_similarity(lr[None].double(), lo[None].double()))
                line += f", logits cos {cos:.7f} max|d| {float((lr - lo).abs().max()):.3e}"
            if not ok_ids:
                line += f"\n    ref: {rr['text']}"
        if c.name in gold:
            cers.append((cer(gold[c.name], text), len(norm_text(gold[c.name]))))
            line += f"  CER {cers[-1][0]:.3f}"
        print(line, flush=True)
        if dump:
            np.save(dump / f"{c.name}.audio.npy", r["audio"].numpy())
            np.save(dump / f"{c.name}.logits.npy", r["first_logits"].numpy())
            rec["clips"][c.name] = {"ids": r["ids"], "text": text, "n_prompt": r["n_prompt"]}
            (dump / "oracle.json").write_text(json.dumps(rec, ensure_ascii=False, indent=0))
    if n_ref:
        print(f"greedy ids identical to the reference: {same}/{n_ref}")
    if cers:
        print(f"CER vs transcript (character-weighted): {sum(c * n for c, n in cers) / max(1, sum(n for _, n in cers)):.4f} over {len(cers)} clips")
    return 0 if same == n_ref else 1


if __name__ == "__main__":
    sys.exit(main())
