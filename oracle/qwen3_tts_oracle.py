#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""qwen3_tts oracle: a golden fp32 CPU model of Qwen3-TTS 12Hz CustomVoice (talker + code predictor + 12.5 Hz codec decoder), the
scoring baseline for the qtts engine (engine/include/hive/qtts/qtts_engine.h holds the contract this file defines).

- This project's own code, written from the model's math. No third-party implementation is copied; the `reference` sub-command only RUNS
  the upstream `qwen_tts` package (when it is installed) to cross-check this file.
- Talker: Qwen3-style decoder (28 layers, hidden 2048, 16 q / 8 kv heads of 128, per-head q/k RMS norms, SwiGLU 6144, RMSNorm eps 1e-6,
  rotate-half RoPE theta 1e6; the checkpoint's 3-axis interleaved mRoPE reduces to plain RoPE because the three position axes are equal
  for text-only input). Inputs are embeddings: text rows = text_projection(text_embedding[id]) (Linear+bias, SiLU, Linear+bias) and codec
  rows = codec_embedding[id]; the logits come from codec_head over the 3072-entry codec vocabulary.
- Prompt (non-streaming text mode): [instruct rows] + role rows (first 3 template tokens) + (tts_pad x n, tts_bos) + codec prefix
  (think, think_bos, language, think_eos, speaker, pad) + (text rows, tts_eos) + codec_pad + (tts_pad + codec_bos). Every generated frame then
  feeds codec_embedding[c0] + sum_i predictor_embedding_i[c_{i+1}] + tts_pad.
- Code predictor: 5 decoder layers (hidden 1024, 16/8 heads of 128, SwiGLU 3072) on small_to_mtp(.) (Linear 2048->1024 + bias) of
  [talker hidden, codec_embedding[c0]] (positions 0 and 1), then one row per further codebook; lm_head[j] gives codebook j+1.
- Sampling: first codebook = repetition penalty 1.05 over the distinct first-codebook ids generated so far (logit > 0 ? /p : *p), ids
  2048..3071 except EOS 2150 suppressed, EOS also suppressed while fewer than 2 frames exist, temperature 0.9, top-k 50; the other 15 codebooks
  temperature 0.9, top-k 50. Greedy (argmax after penalty and suppression) for comparisons; --forced replays given codes (teacher forcing).
- Codec decoder: RVQ (1 semantic + 15 acoustic codebooks of 256 dims, codebook = embedding_sum / max(cluster_usage, 1e-5), 1x1 output
  projections to 512) -> causal conv k3 (512->1024) -> Linear 1024->512 -> 8 transformer layers (16 heads of 64, RoPE theta 1e4, causal sliding
  window of 72 frames, layer scales, RMSNorm eps 1e-5) -> norm -> Linear 512->1024 -> 2 x (causal transposed conv k2 s2, ConvNeXt block) ->
  causal conv k7 (1024->1536) -> 4 decoder blocks (SnakeBeta, causal transposed conv k=2s stride s for s in 8,5,4,3, residual units with
  dilations 1, 3, 9) -> SnakeBeta -> causal conv k7 (96->1) -> clamp(-1, 1). 1920 samples per frame at 24 kHz.
  `CodecStream` is the same decoder run chunk by chunk with carried state (conv histories, a 72-frame KV window); it reproduces the
  full-sequence decode (checked by the `decode` sub-command).
- Numerics: fp32 everywhere (the bf16 checkpoint values are exact in fp32).

Usage (the model directory holds config.json, model.safetensors, vocab.json, merges.txt and speech_tokenizer/):
  python3 oracle/qwen3_tts_oracle.py generate --model /models/Qwen3-TTS-12Hz-1.7B-CustomVoice --text "..." --out /tmp/o --greedy --frames 80
  python3 oracle/qwen3_tts_oracle.py generate ... --forced /tmp/o/codes.npy --out /tmp/f     (teacher forcing: logits.npy [F+1, 16, 3072])
  python3 oracle/qwen3_tts_oracle.py decode --model ... --codes /tmp/o/codes.npy --out /tmp/d  (full and streamed decode, compared)
  python3 oracle/qwen3_tts_oracle.py reference --model ... --text "..." --out /tmp/r --frames 80   (needs the qwen_tts package)
  python3 oracle/qwen3_tts_oracle.py compare --a /tmp/o --b /tmp/r
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920
CHUNK_FRAMES = 4


# ---------------------------------------------------------------------------------------------------------------------------------------
# weights
def load_safetensors(path: str, prefix: str = "") -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                out[k[len(prefix):]] = f.get_tensor(k).to(torch.float32)
    return out


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_tables(positions: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    # computed in float64 and rounded once (the engine uses the same host tables)
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    ang = positions.to(torch.float64)[:, None] * inv[None, :]
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos().to(torch.float32), ang.sin().to(torch.float32)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [T, H, D]; rotate-half form: (x1, x2) -> (x1 cos - x2 sin, x2 cos + x1 sin)
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


# ---------------------------------------------------------------------------------------------------------------------------------------
# talker / code predictor decoder stack (shared math)
class DecoderStack:
    """Pre-norm decoder layers with per-head q/k RMS norms, GQA attention, SwiGLU MLP and a final norm, with a growing KV cache."""

    def __init__(self, w: dict, prefix: str, n_layers: int, n_heads: int, n_kv: int, head_dim: int, eps: float, theta: float):
        self.L, self.H, self.KV, self.D, self.eps, self.theta = n_layers, n_heads, n_kv, head_dim, eps, theta
        self.layers = []
        for i in range(n_layers):
            p = f"{prefix}layers.{i}."
            self.layers.append({k: w[p + k] for k in (
                "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")})
        self.norm = w[prefix + "norm.weight"]
        self.reset()

    def reset(self) -> None:
        self.k = [None] * self.L
        self.v = [None] * self.L
        self.pos = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [T, hidden] (rows at positions pos..pos+T-1) -> normed hidden [T, hidden]; the KV cache grows by T."""
        T = x.shape[0]
        positions = torch.arange(self.pos, self.pos + T)
        cos, sin = rope_tables(positions, self.D, self.theta)
        h = x
        for i, ly in enumerate(self.layers):
            xn = rms_norm(h, ly["input_layernorm.weight"], self.eps)
            q = (xn @ ly["self_attn.q_proj.weight"].T).view(T, self.H, self.D)
            k = (xn @ ly["self_attn.k_proj.weight"].T).view(T, self.KV, self.D)
            v = (xn @ ly["self_attn.v_proj.weight"].T).view(T, self.KV, self.D)
            q = apply_rope(rms_norm(q, ly["self_attn.q_norm.weight"], self.eps), cos, sin)
            k = apply_rope(rms_norm(k, ly["self_attn.k_norm.weight"], self.eps), cos, sin)
            self.k[i] = k if self.k[i] is None else torch.cat([self.k[i], k], 0)
            self.v[i] = v if self.v[i] is None else torch.cat([self.v[i], v], 0)
            K, V = self.k[i], self.v[i]  # [S, KV, D]
            S = K.shape[0]
            g = self.H // self.KV
            Kx = K.repeat_interleave(g, dim=1)  # head h uses kv head h // g
            Vx = V.repeat_interleave(g, dim=1)
            sc = torch.einsum("thd,shd->hts", q, Kx) / math.sqrt(self.D)
            qpos = positions[:, None]
            kpos = torch.arange(S)[None, :]
            sc = sc.masked_fill((kpos > qpos)[None], float("-inf"))
            a = torch.einsum("hts,shd->thd", torch.softmax(sc, -1), Vx).reshape(T, self.H * self.D)
            h = h + a @ ly["self_attn.o_proj.weight"].T
            xn = rms_norm(h, ly["post_attention_layernorm.weight"], self.eps)
            h = h + (F.silu(xn @ ly["mlp.gate_proj.weight"].T) * (xn @ ly["mlp.up_proj.weight"].T)) @ ly["mlp.down_proj.weight"].T
        self.pos += T
        return rms_norm(h, self.norm, self.eps)


# ---------------------------------------------------------------------------------------------------------------------------------------
class Sampler:
    """Greedy, forced or seeded top-k sampling (numpy Gumbel-max over the kept set)."""

    def __init__(self, mode: str, seed: int = 0, temperature: float = 0.9, top_k: int = 50, forced: np.ndarray | None = None):
        self.mode, self.temperature, self.top_k, self.forced = mode, temperature, top_k, forced
        self.rng = np.random.default_rng(seed)

    def pick(self, logits: torch.Tensor, frame: int, cb: int) -> int:
        if self.mode == "forced":
            return int(self.forced[frame, cb])
        if self.mode == "greedy":
            return int(torch.argmax(logits).item())
        l = logits.double().numpy()
        kth = np.sort(l)[-self.top_k]
        keep = l >= kth
        z = np.where(keep, l / self.temperature + self.rng.gumbel(size=l.shape), -np.inf)
        return int(np.argmax(z))


class Talker:
    def __init__(self, model_dir: str):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.cfg = cfg
        tc = cfg["talker_config"]
        pc = tc["code_predictor_config"]
        self.tc, self.pc = tc, pc
        w = load_safetensors(os.path.join(model_dir, "model.safetensors"), "talker.")
        self.text_emb = w["model.text_embedding.weight"]
        self.codec_emb = w["model.codec_embedding.weight"]
        self.codec_head = w["codec_head.weight"]
        self.tp = (w["text_projection.linear_fc1.weight"], w["text_projection.linear_fc1.bias"],
                   w["text_projection.linear_fc2.weight"], w["text_projection.linear_fc2.bias"])
        self.stack = DecoderStack(w, "model.", tc["num_hidden_layers"], tc["num_attention_heads"], tc["num_key_value_heads"],
                                  tc["head_dim"], tc["rms_norm_eps"], tc["rope_theta"])
        self.pred = DecoderStack(w, "code_predictor.model.", pc["num_hidden_layers"], pc["num_attention_heads"],
                                 pc["num_key_value_heads"], pc["head_dim"], pc["rms_norm_eps"], pc["rope_theta"])
        self.mtp_w = w["code_predictor.small_to_mtp_projection.weight"]
        self.mtp_b = w["code_predictor.small_to_mtp_projection.bias"]
        self.pred_emb = [w[f"code_predictor.model.codec_embedding.{i}.weight"] for i in range(tc["num_code_groups"] - 1)]
        self.pred_head = [w[f"code_predictor.lm_head.{i}.weight"] for i in range(tc["num_code_groups"] - 1)]

    def text_rows(self, ids) -> torch.Tensor:
        e = self.text_emb[torch.as_tensor(list(ids), dtype=torch.long)]
        w1, b1, w2, b2 = self.tp
        return F.silu(e @ w1.T + b1) @ w2.T + b2

    def build_prompt(self, ids: list[int], instruct_ids: list[int] | None, speaker_id: int | None, language_id: int | None):
        """Talker input rows for the non-streaming text mode. ids = tokens of the assistant template; instruct_ids = tokens of the user
        instruct template (or None)."""
        tc, cfg = self.tc, self.cfg
        bos, eos, pad = self.text_rows([cfg["tts_bos_token_id"], cfg["tts_eos_token_id"], cfg["tts_pad_token_id"]])
        if language_id is None:
            prefix = [tc["codec_nothink_id"], tc["codec_think_bos_id"], tc["codec_think_eos_id"]]
        else:
            prefix = [tc["codec_think_id"], tc["codec_think_bos_id"], language_id, tc["codec_think_eos_id"]]
        codec = [self.codec_emb[c] for c in prefix]
        if speaker_id is not None:
            codec.append(self.codec_emb[speaker_id])
        codec += [self.codec_emb[tc["codec_pad_id"]], self.codec_emb[tc["codec_bos_id"]]]
        codec = torch.stack(codec)  # [n, 2048]; the last row (codec_bos) is used with tts_pad at the end
        n = codec.shape[0]
        rows = []
        if instruct_ids:
            rows.append(self.text_rows(instruct_ids))
        rows.append(self.text_rows(ids[:3]))                                       # <|im_start|>assistant\n
        rows.append(torch.cat([pad.expand(n - 2, -1), bos[None]], 0) + codec[:-1])  # (pad.., bos) + (prefix, speaker, pad)
        text = ids[3:-5]
        rows.append(torch.cat([self.text_rows(text), eos[None]], 0) + self.codec_emb[tc["codec_pad_id"]][None])
        rows.append((pad + self.codec_emb[tc["codec_bos_id"]])[None])
        return torch.cat(rows, 0), pad

    def first_logits(self, logits: torch.Tensor, history: set[int], n_frames: int) -> torch.Tensor:
        l = logits.clone()
        if history:
            idx = torch.tensor(sorted(history))
            v = l[idx]
            l[idx] = torch.where(v > 0, v / 1.05, v * 1.05)
        eos = self.tc["codec_eos_token_id"]
        V = self.tc["vocab_size"]
        keep = l[eos].clone()
        l[V - 1024:] = float("-inf")
        l[eos] = keep
        if n_frames < 2:
            l[eos] = float("-inf")
        return l

    @torch.no_grad()
    def generate(self, ids, instruct_ids, speaker_id, language_id, max_frames: int, sampler: Sampler, sub_sampler: Sampler,
                 dump_logits: bool = False):
        """Returns codes [F, 16] int, and (if dump_logits) the raw logits per (frame, codebook): first codebook = talker codec_head output
        before penalty/suppression, others = lm_head output; shape [F+1, 16, 3072] (row F holds the logits that produced EOS / the stop)."""
        self.stack.reset()
        prompt, pad = self.build_prompt(ids, instruct_ids, speaker_id, language_id)
        h = self.stack.forward(prompt)[-1]
        eos = self.tc["codec_eos_token_id"]
        codes, dumps, history = [], [], set()
        raw = h @ self.codec_head.T
        f = 0
        while True:
            dump = torch.full((16, 3072), float("nan"))
            dump[0] = raw
            tok0 = eos if f >= max_frames else sampler.pick(self.first_logits(raw, history, f), f, 0)
            if tok0 == eos:
                dumps.append(dump)
                break
            # code predictor
            self.pred.reset()
            x = torch.stack([h, self.codec_emb[tok0]]) @ self.mtp_w.T + self.mtp_b
            frame = [tok0]
            ph = self.pred.forward(x)[-1]
            for j in range(15):
                lg = ph @ self.pred_head[j].T
                dump[j + 1, :lg.shape[0]] = lg
                t = sub_sampler.pick(lg, f, j + 1)
                frame.append(t)
                if j < 14:
                    ph = self.pred.forward((self.pred_emb[j][t] @ self.mtp_w.T + self.mtp_b)[None])[-1]
            codes.append(frame)
            dumps.append(dump)
            history.add(tok0)
            f += 1
            x = self.codec_emb[tok0] + sum(self.pred_emb[i][frame[i + 1]] for i in range(15)) + pad
            h = self.stack.forward(x[None])[-1]
            raw = h @ self.codec_head.T
        codes = np.array(codes, dtype=np.int64).reshape(-1, 16)
        return codes, (torch.stack(dumps).numpy() if dump_logits else None), prompt.numpy()


# ---------------------------------------------------------------------------------------------------------------------------------------
# codec decoder
def snake(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SnakeBeta on [T, C]: x + sin^2(x e^a) / (e^b + 1e-9)."""
    return x + torch.sin(x * torch.exp(a)).pow(2) / (torch.exp(b) + 1e-9)


def causal_conv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, dilation: int = 1, groups: int = 1, hist=None):
    """Causal conv on time-major x [T, Cin]; w [Cout, Cin/groups, k]. hist = the previous (k-1)*dilation input rows (zeros at the start)."""
    k = w.shape[-1]
    H = (k - 1) * dilation
    if hist is None:
        hist = torch.zeros(H, x.shape[1])
    xx = torch.cat([hist, x], 0)
    y = F.conv1d(xx.T[None], w, b, dilation=dilation, groups=groups)[0].T
    return y, xx[xx.shape[0] - H:] if H else hist


def trans_conv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, stride: int, prev=None):
    """Causal transposed conv, kernel 2*stride or stride, trimmed on the right: out[n*s + j] = b + W[:,:,j]^T x[n] + W[:,:,j+s]^T x[n-1].
    w [Cin, Cout, k]; prev = the previous input row (zeros at the start)."""
    k = w.shape[-1]
    T = x.shape[0]
    y = torch.einsum("tc,cok->tko", x, w)  # [T, k, Cout]
    out = y[:, :stride, :].clone()
    if k > stride:
        assert k == 2 * stride
        p = torch.zeros(1, x.shape[1]) if prev is None else prev
        yp = torch.einsum("tc,cok->tko", torch.cat([p, x[:-1]], 0), w[:, :, stride:])
        out = out + yp
    return out.reshape(T * stride, -1) + b, x[-1:].clone()


class Codec:
    def __init__(self, model_dir: str):
        d = os.path.join(model_dir, "speech_tokenizer")
        self.cfg = json.load(open(os.path.join(d, "config.json")))["decoder_config"]
        w = load_safetensors(os.path.join(d, "model.safetensors"), "decoder.")
        self.w = w
        q = []
        for name, n in (("rvq_first", 1), ("rvq_rest", self.cfg["num_quantizers"] - 1)):
            for i in range(n):
                p = f"quantizer.{name}.vq.layers.{i}._codebook."
                q.append(w[p + "embedding_sum"] / w[p + "cluster_usage"].clamp(min=1e-5)[:, None])
        self.codebooks = q
        self.out_first = w["quantizer.rvq_first.output_proj.weight"][:, :, 0]
        self.out_rest = w["quantizer.rvq_rest.output_proj.weight"][:, :, 0]
        c = self.cfg
        self.nh, self.hd = c["num_attention_heads"], c["head_dim"]
        self.window = c["sliding_window"]
        self.rates = c["upsample_rates"]

    def rvq(self, codes: torch.Tensor) -> torch.Tensor:
        s0 = self.codebooks[0][codes[:, 0]]
        s1 = sum(self.codebooks[i][codes[:, i]] for i in range(1, codes.shape[1]))
        return s0 @ self.out_first.T + s1 @ self.out_rest.T  # [T, 512]

    def transformer(self, x: torch.Tensor, positions: torch.Tensor, kv: list | None):
        """x [T, 1024] -> [T, 1024]. kv: per layer (K [S, H, D], V, key positions [S]) of earlier frames (None = none); returns new kv
        trimmed to the window."""
        w, c = self.w, self.cfg
        p = "pre_transformer."
        h = x @ w[p + "input_proj.weight"].T + w[p + "input_proj.bias"]
        T = h.shape[0]
        cos, sin = rope_tables(positions, self.hd, c["rope_theta"])
        new_kv = []
        for i in range(c["num_hidden_layers"]):
            q_ = f"{p}layers.{i}."
            xn = rms_norm(h, w[q_ + "input_layernorm.weight"], c["rms_norm_eps"])
            q = apply_rope((xn @ w[q_ + "self_attn.q_proj.weight"].T).view(T, self.nh, self.hd), cos, sin)
            k = apply_rope((xn @ w[q_ + "self_attn.k_proj.weight"].T).view(T, self.nh, self.hd), cos, sin)
            v = (xn @ w[q_ + "self_attn.v_proj.weight"].T).view(T, self.nh, self.hd)
            if kv is not None and kv[i] is not None:
                K = torch.cat([kv[i][0], k], 0); V = torch.cat([kv[i][1], v], 0); P = torch.cat([kv[i][2], positions], 0)
            else:
                K, V, P = k, v, positions
            sc = torch.einsum("thd,shd->hts", q, K) / math.sqrt(self.hd)
            dpos = positions[:, None] - P[None, :]
            sc = sc.masked_fill(((dpos < 0) | (dpos >= self.window))[None], float("-inf"))
            a = torch.einsum("hts,shd->thd", torch.softmax(sc, -1), V).reshape(T, -1)
            h = h + w[q_ + "self_attn_layer_scale.scale"] * (a @ w[q_ + "self_attn.o_proj.weight"].T)
            xn = rms_norm(h, w[q_ + "post_attention_layernorm.weight"], c["rms_norm_eps"])
            m = (F.silu(xn @ w[q_ + "mlp.gate_proj.weight"].T) * (xn @ w[q_ + "mlp.up_proj.weight"].T)) @ w[q_ + "mlp.down_proj.weight"].T
            h = h + w[q_ + "mlp_layer_scale.scale"] * m
            keep = self.window - 1  # a later frame sees at most window-1 earlier frames
            new_kv.append((K[-keep:], V[-keep:], P[-keep:]))
        h = rms_norm(h, w[p + "norm.weight"], c["rms_norm_eps"])
        return h @ w[p + "output_proj.weight"].T + w[p + "output_proj.bias"], new_kv

    def convnext(self, x: torch.Tensor, i: int, hist):
        w = self.w
        p = f"upsample.{i}.1."
        y, hist = causal_conv(x, w[p + "dwconv.conv.weight"], w[p + "dwconv.conv.bias"], groups=x.shape[1], hist=hist)
        y = F.layer_norm(y, (y.shape[1],), w[p + "norm.weight"], w[p + "norm.bias"], eps=1e-6)
        y = F.gelu(y @ w[p + "pwconv1.weight"].T + w[p + "pwconv1.bias"]) @ w[p + "pwconv2.weight"].T + w[p + "pwconv2.bias"]
        return x + w[p + "gamma"] * y, hist

    @torch.no_grad()
    def run(self, codes: torch.Tensor, state: dict | None = None, frame0: int = 0):
        """Decode frames codes [T, 16] that start at absolute frame index frame0, carrying `state` (None = start of a stream with no
        history). Returns (wav [T*1920] float32 in [-1, 1], new state)."""
        w = self.w
        st = state or {}
        ns = {}
        x = self.rvq(codes)
        x, ns["pre"] = causal_conv(x, w["pre_conv.conv.weight"], w["pre_conv.conv.bias"], hist=st.get("pre"))
        x, ns["kv"] = self.transformer(x, torch.arange(frame0, frame0 + codes.shape[0]), st.get("kv"))
        for i in range(2):
            x, ns[f"up{i}"] = trans_conv(x, w[f"upsample.{i}.0.conv.weight"], w[f"upsample.{i}.0.conv.bias"], 2, st.get(f"up{i}"))
            x, ns[f"cnx{i}"] = self.convnext(x, i, st.get(f"cnx{i}"))
        x, ns["dec0"] = causal_conv(x, w["decoder.0.conv.weight"], w["decoder.0.conv.bias"], hist=st.get("dec0"))
        for b, s in enumerate(self.rates):
            p = f"decoder.{b + 1}.block."
            y = snake(x, w[p + "0.alpha"], w[p + "0.beta"])
            x, ns[f"b{b}t"] = trans_conv(y, w[p + "1.conv.weight"], w[p + "1.conv.bias"], s, st.get(f"b{b}t"))
            for r, dil in enumerate((1, 3, 9)):
                q = f"{p}{r + 2}."
                y = snake(x, w[q + "act1.alpha"], w[q + "act1.beta"])
                y, ns[f"b{b}r{r}"] = causal_conv(y, w[q + "conv1.conv.weight"], w[q + "conv1.conv.bias"], dilation=dil,
                                                 hist=st.get(f"b{b}r{r}"))
                y = snake(y, w[q + "act2.alpha"], w[q + "act2.beta"])
                y, _ = causal_conv(y, w[q + "conv2.conv.weight"], w[q + "conv2.conv.bias"])
                x = x + y
        y = snake(x, w["decoder.5.alpha"], w["decoder.5.beta"])
        y, ns["out"] = causal_conv(y, w["decoder.6.conv.weight"], w["decoder.6.conv.bias"], hist=st.get("out"))
        return y[:, 0].clamp(-1, 1), ns

    def decode_full(self, codes: np.ndarray) -> np.ndarray:
        """Whole sequence in one pass (the history-free definition)."""
        wav, _ = self.run(torch.as_tensor(codes, dtype=torch.long))
        return wav.numpy()

    def decode_stream(self, codes: np.ndarray, chunk: int = CHUNK_FRAMES) -> np.ndarray:
        """Chunk by chunk with carried state (what the engine does)."""
        st, out = None, []
        c = torch.as_tensor(codes, dtype=torch.long)
        for f0 in range(0, c.shape[0], chunk):
            wav, st = self.run(c[f0:f0 + chunk], st, f0)
            out.append(wav)
        return torch.cat(out).numpy()


# the cast the serving path applies: clip, scale, truncate toward zero
def to_pcm16(wav: np.ndarray) -> np.ndarray:
    return (np.clip(wav, -1, 1) * 32767).astype(np.int16)


def write_wav(path: str, wav: np.ndarray) -> None:
    import wave

    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SAMPLE_RATE)
        f.writeframes(to_pcm16(wav).tobytes())


# ---------------------------------------------------------------------------------------------------------------------------------------
# prompt tokens (the same templates the serving path uses)
def assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def instruct_text(instruct: str) -> str:
    return f"<|im_start|>user\n{instruct}<|im_end|>\n"


def tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


def resolve_ids(cfg: dict, speaker: str, language: str) -> tuple[int | None, int | None]:
    tc = cfg["talker_config"]
    spk = speaker.lower() if speaker else ""
    sid = tc["spk_id"][spk] if spk else None
    lang = language.lower()
    lid = None if lang == "auto" else tc["codec_language_id"][lang]
    if lang in ("chinese", "auto") and spk and tc["spk_is_dialect"].get(spk):
        lid = tc["codec_language_id"][tc["spk_is_dialect"][spk]]
    return sid, lid


# ---------------------------------------------------------------------------------------------------------------------------------------
def cmd_generate(a) -> None:
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    tok = tokenizer(a.model)
    ids = tok(assistant_text(a.text))["input_ids"]
    ins = tok(instruct_text(a.instruct))["input_ids"] if a.instruct else None
    t0 = time.time()
    tk = Talker(a.model)
    sid, lid = resolve_ids(tk.cfg, a.speaker, a.language)
    if a.forced:
        forced = np.load(a.forced)
        s = Sampler("forced", forced=forced)
        codes, dumps, prompt = tk.generate(ids, ins, sid, lid, forced.shape[0], s, s, dump_logits=True)
    else:
        mode = "greedy" if a.greedy else "sample"
        codes, dumps, prompt = tk.generate(ids, ins, sid, lid, a.frames, Sampler(mode, a.seed), Sampler(mode, a.seed + 1),
                                           dump_logits=a.dump_logits)
    print(f"[oracle] {codes.shape[0]} frames in {time.time() - t0:.1f}s", flush=True)
    np.save(os.path.join(a.out, "codes.npy"), codes)
    np.save(os.path.join(a.out, "prompt.npy"), prompt)
    json.dump({"ids": ids, "instruct_ids": ins, "speaker_id": sid, "language_id": lid},
              open(os.path.join(a.out, "ids.json"), "w"))
    if dumps is not None:
        np.save(os.path.join(a.out, "logits.npy"), dumps.astype(np.float32))
    if not a.no_audio and codes.shape[0]:
        wav = Codec(a.model).decode_full(codes)
        np.save(os.path.join(a.out, "wav.npy"), wav)
        write_wav(os.path.join(a.out, "out.wav"), wav)


def cmd_decode(a) -> None:
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    codes = np.load(a.codes)
    if a.frames:
        codes = codes[:a.frames]
    c = Codec(a.model)
    t0 = time.time()
    full = c.decode_full(codes)
    t1 = time.time()
    stream = c.decode_stream(codes)
    t2 = time.time()
    np.save(os.path.join(a.out, "wav_full.npy"), full)
    np.save(os.path.join(a.out, "wav_stream.npy"), stream)
    d = np.abs(full - stream)
    print(json.dumps({"frames": int(codes.shape[0]), "samples": int(full.shape[0]), "full_s": round(t1 - t0, 1),
                      "stream_s": round(t2 - t1, 1), "max_abs_diff": float(d.max()), "rms": float(np.sqrt((full ** 2).mean())),
                      "pcm16_mismatch": int((to_pcm16(full) != to_pcm16(stream)).sum())}))


def cmd_reference(a) -> None:
    """Runs the upstream qwen_tts package in fp32 on the CPU with greedy decoding (talker and code predictor) and dumps ids, codes and
    the waveform of its own speech-tokenizer decode."""
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    from qwen_tts import Qwen3TTSModel

    m = Qwen3TTSModel.from_pretrained(a.model, device_map="cpu", dtype=torch.float32, attn_implementation="eager")
    ids = m._tokenize_texts([m._build_assistant_text(a.text)])
    ins = [m._tokenize_texts([m._build_instruct_text(a.instruct)])[0]] if a.instruct else [None]
    t0 = time.time()
    with torch.no_grad():
        codes_list, _ = m.model.generate(input_ids=ids, instruct_ids=ins, languages=[a.language], speakers=[a.speaker],
                                         non_streaming_mode=True, max_new_tokens=a.frames, do_sample=False, subtalker_dosample=False)
        codes = codes_list[0]
        wavs, sr = m.model.speech_tokenizer.decode({"audio_codes": codes[None]})
    print(f"[reference] {codes.shape[0]} frames in {time.time() - t0:.1f}s", flush=True)
    np.save(os.path.join(a.out, "codes.npy"), codes.numpy().astype(np.int64))
    wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    np.save(os.path.join(a.out, "wav.npy"), wav)
    json.dump({"ids": ids[0][0].tolist(), "instruct_ids": None if ins[0] is None else ins[0][0].tolist()},
              open(os.path.join(a.out, "ids.json"), "w"))


def cmd_compare(a) -> None:
    A, B = a.a, a.b
    rep = {}
    ia, ib = json.load(open(os.path.join(A, "ids.json"))), json.load(open(os.path.join(B, "ids.json")))
    rep["ids_equal"] = ia["ids"] == ib["ids"]
    ca, cb = np.load(os.path.join(A, "codes.npy")), np.load(os.path.join(B, "codes.npy"))
    n = min(len(ca), len(cb))
    eq = (ca[:n] == cb[:n]).all(1)
    rep.update({"frames_a": int(len(ca)), "frames_b": int(len(cb)), "frames_identical": int(eq.sum()),
                "first_divergent_frame": int(np.argmin(eq)) if not eq.all() else None,
                "token_agreement": float((ca[:n] == cb[:n]).mean()) if n else None})
    wa, wb = os.path.join(A, "wav.npy"), os.path.join(B, "wav.npy")
    if os.path.exists(wa) and os.path.exists(wb):
        x, y = np.load(wa), np.load(wb)
        m = min(len(x), len(y))
        if eq.all() and len(ca) == len(cb):
            d = x[:m] - y[:m]
            rep.update({"wav_max_abs_diff": float(np.abs(d).max()),
                        "wav_snr_db": float(10 * np.log10((y[:m] ** 2).sum() / max((d ** 2).sum(), 1e-30)))})
    la, lb = os.path.join(A, "logits.npy"), os.path.join(B, "logits.npy")
    if os.path.exists(la) and os.path.exists(lb):
        rep.update(compare_logits(np.load(la), np.load(lb)))
    print(json.dumps(rep, indent=1))


def compare_logits(a: np.ndarray, b: np.ndarray) -> dict:
    """Per (frame, codebook) logit rows: top-1 agreement, max |diff| relative to the row's range, cosine."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rows = []
    for f in range(n):
        for cb in range(16):
            V = 3072 if cb == 0 else 2048
            x, y = a[f, cb, :V].astype(np.float64), b[f, cb, :V].astype(np.float64)
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                continue
            rows.append((int(np.argmax(x) == np.argmax(y)), float(np.abs(x - y).max() / (np.ptp(y) + 1e-9)),
                         float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)), cb))
    r = np.array(rows)
    out = {"logit_rows": int(len(r)), "top1_agree": float(r[:, 0].mean()), "max_rel_diff": float(r[:, 1].max()),
           "mean_rel_diff": float(r[:, 1].mean()), "min_cosine": float(r[:, 2].min())}
    t = r[r[:, 3] == 0]
    if len(t):
        out.update({"talker_top1_agree": float(t[:, 0].mean()), "talker_max_rel_diff": float(t[:, 1].max())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "reference"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--text", required=True)
        p.add_argument("--speaker", default="sohee")
        p.add_argument("--language", default="Korean")
        p.add_argument("--instruct", default="")
        p.add_argument("--frames", type=int, default=160)
        p.add_argument("--out", required=True)
        p.add_argument("--threads", type=int, default=16)
        if name == "generate":
            p.add_argument("--greedy", action="store_true")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--forced", default="")
            p.add_argument("--dump-logits", action="store_true")
            p.add_argument("--no-audio", action="store_true")
    p = sub.add_parser("decode")
    p.add_argument("--model", required=True)
    p.add_argument("--codes", required=True)
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=16)
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    a = ap.parse_args()
    {"generate": cmd_generate, "decode": cmd_decode, "reference": cmd_reference, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
