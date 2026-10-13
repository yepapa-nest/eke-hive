#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""q27 oracle: a golden CPU model of Qwen3.8-27B (config model_type "qwen3_5", compressed-tensors mixed FP8 / NVFP4 checkpoint), the
per-layer scoring baseline for hived_q27 / hived_q27_cpu (engine/include/hive/q27/q27_model.h holds the math contract this file defines).

- This project's own code, written from the model's math (no third-party implementation copied; transformers is only RUN, by
  tools/compare_q27.py --transformers, to cross-check this file). Batch 1. The prompt is one chunk (all rows at once, causal inside the
  chunk), then N greedy decode steps one row at a time — the incremental state (Gated DeltaNet conv / recurrent state, KV cache) carries
  over from chunk to chunk.
- Model: 64 decoder layers (every 4th full attention, the others Gated DeltaNet), pre-norm residual (h += mixer(norm(h)); h += mlp(norm(h)),
  (1 + w) RMS norms), dense SwiGLU MLP 17408, final (1 + w) norm, lm_head. MTP: one full-attention decoder layer fed by
  fc([norm(emb(x_{j+1})) | norm(h_j)]), its own (1 + w) final norm, the shared lm_head.
- Weights: FP8 e4m3 with one bf16 scale per output row (attention q/k/v/o, linear-attention in_proj_qkv / in_proj_z / out_proj, the MLP of
  layers 56-63, lm_head); NVFP4 (e2m1 pairs, low nibble = even column, an e4m3 scale per 16 columns, one fp32 global scale g per matrix:
  W = e2m1 · s16 / g) for the other MLPs; BF16 for everything else (in_proj_a / in_proj_b, norms, conv, MTP, vision). Activations are NOT
  quantized (W8A16 / W4A16 — the checkpoint's dynamic activation quantization is not part of this contract). The FP8 KV-cache scales
  (k_scale / v_scale) are not used: the KV cache is bf16.
- Numerics: fp32 arithmetic with a bf16 rounding wherever the model keeps bf16 (GEMM outputs, norm outputs, the residual stream, the
  activation product). A linear is y = bf16(Σ_k x_k · q[n, k] · scale[n]) computed as (x · qᵀ) · scale with q the exactly-representable
  pre-scale weight (FP8 value, or e2m1 · s16) — mathematically the dequantized product. `--dtype fp32` turns the roundings off.
- Output (--out): h_LNN.npy (residual stream after layer NN, prompt rows, [T, 5120] float32), logits.npy (prompt's last row — every row with
  --all-logits — then one row per decode step), decode.json, timing.json; with --mtp: mtp_top.json (per prompt position j < T−1: top-20 ids /
  logits of the MTP block fed (h_j, x_{j+1})) and the agreement of its top-1 with the next prompt token / the main model's next greedy token;
  with --vision-patches: vision.npy (encoder output [n/4, 5120]) and the image rows replace the embeddings of prompt rows [start, start + n/4).

Usage: python3 oracle/q27_oracle.py --ckpt /models/Qwen3.8-27B-NVFP4 --out /tmp/qo --prompt-ids 248045,872,198 --decode 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
ROUND = True


def bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32) if ROUND else x


_NP = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I64": np.int64, "I32": np.int32, "U8": np.uint8, "F8_E4M3": np.uint8}


class Ckpt:
    """Minimal safetensors reader (index json + every *.safetensors file in the directory; memmap, no conversion until read)."""

    def __init__(self, path: str):
        self.path = path
        self.where: dict = {}
        self.hdr: dict = {}
        self.mm: dict = {}
        for fn in sorted(os.listdir(path)):
            if not fn.endswith(".safetensors"):
                continue
            full = os.path.join(path, fn)
            with open(full, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                h = json.loads(f.read(n))
            self.hdr[full] = (h, 8 + n)
            for k in h:
                if k != "__metadata__":
                    self.where[k] = full

    def has(self, name: str) -> bool:
        return name in self.where

    def raw(self, name: str) -> np.ndarray:
        file = self.where[name]
        h, base = self.hdr[file]
        m = h[name]
        if file not in self.mm:
            self.mm[file] = np.memmap(file, dtype=np.uint8, mode="r")
        a, b = m["data_offsets"]
        return self.mm[file][base + a: base + b].view(_NP[m["dtype"]]).reshape(m["shape"])

    def dtype(self, name: str) -> str:
        return self.hdr[self.where[name]][0][name]["dtype"]

    def f32(self, name: str) -> torch.Tensor:
        dt, a = self.dtype(name), self.raw(name)
        if dt == "BF16":
            return torch.from_numpy((np.ascontiguousarray(a).astype(np.uint32) << 16).view(np.float32))
        if dt == "F8_E4M3":
            return torch.from_numpy(np.ascontiguousarray(a)).view(torch.float8_e4m3fn).to(torch.float32)
        if dt in ("F32", "F16"):
            return torch.from_numpy(np.ascontiguousarray(a).astype(np.float32))
        raise ValueError(f"{name}: dtype {dt} is not a float tensor")


class Lin:
    """One linear map: q bf16 [N, K] (exact pre-scale values) and the scale (None, fp32 [N] per row, or a scalar)."""

    def __init__(self, ck: Ckpt, base: str):
        if ck.has(base + ".weight_packed"):            # NVFP4 (compressed-tensors): e2m1 · s16 / global
            w = torch.from_numpy(np.ascontiguousarray(ck.raw(base + ".weight_packed")))
            s = ck.f32(base + ".weight_scale")
            g = float(ck.f32(base + ".weight_global_scale").reshape(-1)[0])
            N, K2 = w.shape
            v = torch.stack([E2M1[(w & 0xF).long()], E2M1[(w >> 4).long()]], -1).reshape(N, 2 * K2)
            self.q = (v * s.repeat_interleave(16, 1)).to(torch.bfloat16)   # exact: ≤ 6 significant bits
            self.scale = torch.tensor(np.float32(1.0) / np.float32(g))
            self.kind = "nvfp4"
        elif ck.dtype(base + ".weight") == "F8_E4M3":    # FP8, one scale per output row
            self.q = ck.f32(base + ".weight").to(torch.bfloat16)        # exact
            self.scale = ck.f32(base + ".weight_scale").reshape(-1)
            self.kind = "fp8"
        else:
            self.q = ck.f32(base + ".weight").to(torch.bfloat16)
            self.scale = None
            self.kind = "bf16"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.q.to(torch.float32).t()
        if self.scale is not None:
            y = y * self.scale
        return bf(y)


class Cfg:
    def __init__(self, ckpt: str):
        allc = json.load(open(os.path.join(ckpt, "config.json")))
        t = allc.get("text_config", allc)
        self.hidden, self.n_layers, self.vocab = t["hidden_size"], t["num_hidden_layers"], t["vocab_size"]
        self.inter = t["intermediate_size"]
        self.eps = t.get("rms_norm_eps", 1e-6)
        self.layer_types = t["layer_types"]
        self.kh, self.vh = t["linear_num_key_heads"], t["linear_num_value_heads"]
        self.kd, self.vd = t["linear_key_head_dim"], t["linear_value_head_dim"]
        self.conv_k = t.get("linear_conv_kernel_dim", 4)
        self.nh, self.nkv, self.hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        rp = t.get("rope_parameters", {})
        self.theta = float(rp.get("rope_theta", t.get("rope_theta", 1e7)))
        self.rot = int(self.hd * rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))
        self.section = list(rp.get("mrope_section", [11, 11, 10]))
        self.interleaved = bool(rp.get("mrope_interleaved", True))
        v = allc.get("vision_config", {})
        self.v_depth, self.v_dim, self.v_heads = v.get("depth", 27), v.get("hidden_size", 1152), v.get("num_heads", 16)
        self.v_inter, self.v_out, self.v_pos = v.get("intermediate_size", 4304), v.get("out_hidden_size", self.hidden), v.get("num_position_embeddings", 2304)


def rmsnorm1p(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """(1 + w) RMS norm over the last dim, fp32 math, one rounding."""
    return bf(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w))


class Rope:
    """Interleaved M-RoPE on dims 0..rot-1 (NeoX halves): pair i uses axis 1 if i%3==1 and i<3·sec[1], axis 2 if i%3==2 and i<3·sec[2]."""

    def __init__(self, cfg: Cfg):
        half = cfg.rot // 2
        self.half = half
        self.inv = cfg.theta ** (-torch.arange(0, half, dtype=torch.float64) * 2.0 / cfg.rot)
        axis = torch.zeros(half, dtype=torch.long)
        i = torch.arange(half)
        if cfg.interleaved:
            axis[(i % 3 == 1) & (i < 3 * cfg.section[1])] = 1
            axis[(i % 3 == 2) & (i < 3 * cfg.section[2])] = 2
        else:
            axis[cfg.section[0]: cfg.section[0] + cfg.section[1]] = 1
            axis[cfg.section[0] + cfg.section[1]:] = 2
        self.axis = axis

    def apply(self, x: torch.Tensor, pos3: torch.Tensor) -> torch.Tensor:
        """x [T, H, D] → rounded result (fp32 cos / sin)."""
        ang = pos3[:, self.axis].to(torch.float64) * self.inv[None, :]
        cos, sin = torch.cos(ang).float()[:, None, :], torch.sin(ang).float()[:, None, :]
        h = self.half
        x1, x2 = x[..., :h], x[..., h:2 * h]
        out = x.clone()
        out[..., :h] = x1 * cos - x2 * sin
        out[..., h:2 * h] = x2 * cos + x1 * sin
        return bf(out)


class Gdn:
    """Gated DeltaNet (per-token recurrence)."""

    def __init__(self, ck: Ckpt, cfg: Cfg, p: str):
        self.cfg = cfg
        self.qkv, self.z = Lin(ck, p + "in_proj_qkv"), Lin(ck, p + "in_proj_z")
        self.b, self.a = Lin(ck, p + "in_proj_b"), Lin(ck, p + "in_proj_a")
        self.out = Lin(ck, p + "out_proj")
        C = 2 * cfg.kh * cfg.kd + cfg.vh * cfg.vd
        self.cw = ck.f32(p + "conv1d.weight").reshape(C, cfg.conv_k)
        self.A_log, self.dt_bias, self.norm = ck.f32(p + "A_log"), ck.f32(p + "dt_bias"), ck.f32(p + "norm.weight")
        self.conv_hist = torch.zeros(cfg.conv_k - 1, C)
        self.S = torch.zeros(cfg.vh, cfg.kd, cfg.vd)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        T = x.shape[0]
        KW, VW, K1 = cfg.kh * cfg.kd, cfg.vh * cfg.vd, cfg.conv_k - 1
        qkv, z, b, a = self.qkv(x), self.z(x), self.b(x), self.a(x)
        seq = torch.cat([self.conv_hist, qkv], 0)
        y = torch.stack([(self.cw * seq[t:t + cfg.conv_k].t()).sum(-1) for t in range(T)])   # tap k−1 = current token
        self.conv_hist = seq[-K1:].clone()
        y = bf(F.silu(y))
        q = y[:, :KW].reshape(T, cfg.kh, cfg.kd)
        k = y[:, KW:2 * KW].reshape(T, cfg.kh, cfg.kd)
        v = y[:, 2 * KW:].reshape(T, cfg.vh, cfg.vd)
        q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * (cfg.kd ** -0.5)
        k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
        sp = a + self.dt_bias
        sp = torch.where(sp > 20.0, sp, torch.log1p(torch.exp(sp)))
        g = -torch.exp(self.A_log) * sp
        beta = bf(torch.sigmoid(b))
        rep = cfg.vh // cfg.kh
        qv, kv_ = q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1)
        o = torch.empty(T, cfg.vh, cfg.vd)
        S = self.S
        for t in range(T):
            S = S * torch.exp(g[t])[:, None, None]
            kvv = torch.einsum("hkv,hk->hv", S, kv_[t])
            delta = (v[t] - kvv) * beta[t][:, None]
            S = S + kv_[t][:, :, None] * delta[:, None, :]
            o[t] = torch.einsum("hkv,hk->hv", S, qv[t])
        self.S = S
        o = bf(o)
        # gated RMS norm: bf16(bf16(bf16(o · rsqrt(mean o² + eps)) · w) · silu(z))
        on = bf(o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + cfg.eps))
        on = bf(on * self.norm)
        on = bf(on * F.silu(z.reshape(T, cfg.vh, cfg.vd)))
        return self.out(on.reshape(T, VW))


class Attn:
    """Full attention: q / gate per head, (1 + w) q / k norm, partial M-RoPE, GQA, sigmoid output gate."""

    def __init__(self, ck: Ckpt, cfg: Cfg, rope: Rope, p: str):
        self.cfg, self.rope = cfg, rope
        self.qp, self.kp, self.vp, self.op = Lin(ck, p + "q_proj"), Lin(ck, p + "k_proj"), Lin(ck, p + "v_proj"), Lin(ck, p + "o_proj")
        self.qn, self.kn = ck.f32(p + "q_norm.weight"), ck.f32(p + "k_norm.weight")
        self.k: list = []
        self.v: list = []

    def __call__(self, x: torch.Tensor, pos0: int, pos3: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        T, H, G, D = x.shape[0], cfg.nh, cfg.nkv, cfg.hd
        qg = self.qp(x).reshape(T, H, 2 * D)
        q, gate = qg[..., :D], qg[..., D:]
        k = self.kp(x).reshape(T, G, D)
        v = self.vp(x).reshape(T, G, D)
        q = self.rope.apply(rmsnorm1p(q, self.qn, cfg.eps), pos3)
        k = self.rope.apply(rmsnorm1p(k, self.kn, cfg.eps), pos3)
        for t in range(T):
            self.k.append(k[t]); self.v.append(v[t])
        Kall, Vall = torch.stack(self.k), torch.stack(self.v)
        rep = H // G
        out = torch.empty(T, H, D)
        for t in range(T):
            n = pos0 + t + 1
            for g_ in range(G):
                Ks, Vs = Kall[:n, g_], Vall[:n, g_]
                s = (q[t, g_ * rep:(g_ + 1) * rep] @ Ks.t()) / math.sqrt(D)
                out[t, g_ * rep:(g_ + 1) * rep] = torch.softmax(s, -1) @ Vs
        o = bf(bf(out) * torch.sigmoid(gate))
        return self.op(o.reshape(T, H * D))


class Mlp:
    def __init__(self, ck: Ckpt, p: str):
        self.g, self.u, self.d = Lin(ck, p + "gate_proj"), Lin(ck, p + "up_proj"), Lin(ck, p + "down_proj")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.d(bf(bf(F.silu(self.g(x))) * self.u(x)))


class Vision:
    """Vision encoder (model.visual): patch linear + bilinear position grid, pre-LN blocks with axial 2D rope, merger 2×2."""

    def __init__(self, ck: Ckpt, cfg: Cfg):
        P = "model.visual."
        self.cfg = cfg
        self.patch_w = ck.f32(P + "patch_embed.proj.weight").reshape(cfg.v_dim, -1)
        self.patch_b = ck.f32(P + "patch_embed.proj.bias")
        self.pos = ck.f32(P + "pos_embed.weight")
        self.side = int(round(math.sqrt(cfg.v_pos)))
        self.blocks = []
        for l in range(cfg.v_depth):
            b = f"{P}blocks.{l}."
            self.blocks.append({n: ck.f32(b + n) for n in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "attn.qkv.weight", "attn.qkv.bias",
                                                         "attn.proj.weight", "attn.proj.bias", "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
                                                         "mlp.linear_fc2.weight", "mlp.linear_fc2.bias")})
        self.m = {n: ck.f32(P + "merger." + n) for n in ("norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias", "linear_fc2.weight", "linear_fc2.bias")}

    @staticmethod
    def ln(x, w, b, eps=1e-6):
        m = x.mean(-1, keepdim=True)
        v = ((x - m) ** 2).mean(-1, keepdim=True)
        return bf((x - m) * torch.rsqrt(v + eps) * w + b)

    def __call__(self, patches: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        """patches [gh·gw, 1536] (bf16 values, rows in 2×2 merge-block order) → [gh·gw/4, out]."""
        cfg, D, NH = self.cfg, self.cfg.v_dim, self.cfg.v_heads
        HD = D // NH
        N = gh * gw
        x = bf(bf(patches @ self.patch_w.t()) + self.patch_b)
        # merge-block order coordinates
        ys, xs = [], []
        for bh in range(gh // 2):
            for bw in range(gw // 2):
                for mh in range(2):
                    for mw in range(2):
                        ys.append(bh * 2 + mh); xs.append(bw * 2 + mw)
        ys, xs = torch.tensor(ys), torch.tensor(xs)
        side = self.side

        def samples(n):
            t = torch.zeros(n) if n == 1 else torch.arange(n, dtype=torch.float32) * float(side - 1) / float(n - 1)
            fl = torch.clamp(t.floor().long(), max=side - 1)
            return fl, torch.clamp(fl + 1, max=side - 1), t - fl.float()

        hf, hc, dh = samples(gh)
        wf, wc, dw = samples(gw)
        a, b_ = dh[ys][:, None], dw[xs][:, None]
        pe = ((1 - a) * (1 - b_) * self.pos[hf[ys] * side + wf[xs]] + (1 - a) * b_ * self.pos[hf[ys] * side + wc[xs]]
              + a * (1 - b_) * self.pos[hc[ys] * side + wf[xs]] + a * b_ * self.pos[hc[ys] * side + wc[xs]])
        x = bf(x + bf(pe))
        half = HD // 2           # 36: dims [0,18) rows, [18,36) columns, repeated for [36, 72)
        inv = 1.0 / (10000.0 ** (torch.arange(0, half // 2, dtype=torch.float32) * 2.0 / half))
        ang = torch.cat([ys[:, None].float() * inv[None, :], xs[:, None].float() * inv[None, :]], -1)   # [N, 36]
        cos, sin = torch.cos(ang), torch.sin(ang)

        def rot(t):  # t [N, NH, HD]
            t1, t2 = t[..., :half], t[..., half:]
            c, s = cos[:, None, :], sin[:, None, :]
            return bf(torch.cat([t1 * c - t2 * s, t2 * c + t1 * s], -1))

        for B in self.blocks:
            xn = self.ln(x, B["norm1.weight"], B["norm1.bias"])
            qkv = bf(bf(xn @ B["attn.qkv.weight"].t()) + B["attn.qkv.bias"]).reshape(N, 3, NH, HD)
            q, k, v = rot(qkv[:, 0]), rot(qkv[:, 1]), qkv[:, 2]
            att = torch.empty(N, NH, HD)
            for h in range(NH):
                s = (q[:, h] @ k[:, h].t()) / math.sqrt(HD)
                att[:, h] = torch.softmax(s, -1) @ v[:, h]
            att = bf(att).reshape(N, D)
            x = bf(bf(x + att @ B["attn.proj.weight"].t()) + B["attn.proj.bias"])
            xn = self.ln(x, B["norm2.weight"], B["norm2.bias"])
            f1 = bf(bf(xn @ B["mlp.linear_fc1.weight"].t()) + B["mlp.linear_fc1.bias"])
            f1 = bf(0.5 * f1 * (1.0 + torch.tanh(0.7978845608028654 * (f1 + 0.044715 * f1 ** 3))))
            x = bf(bf(x + f1 @ B["mlp.linear_fc2.weight"].t()) + B["mlp.linear_fc2.bias"])
        xn = self.ln(x, self.m["norm.weight"], self.m["norm.bias"]).reshape(N // 4, 4 * D)
        h = bf(bf(xn @ self.m["linear_fc1.weight"].t()) + self.m["linear_fc1.bias"])
        h = bf(0.5 * h * (1.0 + torch.erf(h * 0.7071067811865476)))
        return bf(bf(h @ self.m["linear_fc2.weight"].t()) + self.m["linear_fc2.bias"])


class Oracle:
    def __init__(self, ckpt: str, mtp: bool):
        self.ck = Ckpt(ckpt)
        self.cfg = cfg = Cfg(ckpt)
        self.rope = Rope(cfg)
        P = "model.language_model."
        self.blocks = []
        t0 = time.time()
        for l in range(cfg.n_layers):
            p = f"{P}layers.{l}."
            attn = cfg.layer_types[l] == "full_attention"
            core = Attn(self.ck, cfg, self.rope, p + "self_attn.") if attn else Gdn(self.ck, cfg, p + "linear_attn.")
            self.blocks.append((attn, core, Mlp(self.ck, p + "mlp."), self.ck.f32(p + "input_layernorm.weight"),
                                self.ck.f32(p + "post_attention_layernorm.weight")))
            if l % 16 == 15:
                print(f"[q27-oracle] layers 0..{l} loaded ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
        self.norm = self.ck.f32(P + "norm.weight")
        self.lm_head = Lin(self.ck, "lm_head")
        self.embed_raw = self.ck.raw(P + "embed_tokens.weight")
        self.mtp = None
        if mtp:
            m = "mtp."
            self.mtp = dict(fc=Lin(self.ck, m + "fc"), en=self.ck.f32(m + "pre_fc_norm_embedding.weight"), hn=self.ck.f32(m + "pre_fc_norm_hidden.weight"),
                            attn=Attn(self.ck, cfg, self.rope, m + "layers.0.self_attn."), mlp=Mlp(self.ck, m + "layers.0.mlp."),
                            ln1=self.ck.f32(m + "layers.0.input_layernorm.weight"), ln2=self.ck.f32(m + "layers.0.post_attention_layernorm.weight"),
                            norm=self.ck.f32(m + "norm.weight"))
        self.pos = 0
        self.rope_delta = 0
        self.dumps: list = []

    def embed(self, ids: list[int]) -> torch.Tensor:
        e = (np.stack([np.asarray(self.embed_raw[i]) for i in ids]).astype(np.uint32) << 16).view(np.float32)
        return torch.from_numpy(e)

    def coords(self, T: int, image=None) -> torch.Tensor:
        """Rope coordinates of rows pos .. pos+T−1; image = (row, n, gh, gw) of merged tokens in this chunk."""
        out = torch.zeros(T, 3, dtype=torch.long)
        for r in range(T):
            p = self.pos + r
            if image is not None and image[0] <= r < image[0] + image[1]:
                k = r - image[0]
                st = p - k + self.rope_delta
                out[r] = torch.tensor([st, st + k // image[3], st + k % image[3]])
                if k == image[1] - 1:
                    self.rope_delta += max(image[2], image[3]) - image[1]
            else:
                out[r] = p + self.rope_delta
        return out

    def forward(self, ids: list[int], keep_layers: bool, emb_override=None, image=None):
        """One chunk of rows → (final residual streams [T, 5120] (before the final norm), normed [T, 5120], logits [T, V])."""
        cfg = self.cfg
        T = len(ids)
        pos3 = self.coords(T, image)
        h = self.embed(ids)
        if emb_override is not None:
            r0, e = emb_override
            h[r0:r0 + len(e)] = e
        for l, (attn, core, mlp, ln1, ln2) in enumerate(self.blocks):
            x = rmsnorm1p(h, ln1, cfg.eps)
            y = core(x, self.pos, pos3) if attn else core(x)
            h = bf(h + y)
            h = bf(h + mlp(rmsnorm1p(h, ln2, cfg.eps)))
            if keep_layers:
                self.dumps.append((l, h.clone()))
            if l % 16 == 15:
                print(f"[q27-oracle] layer {l} done ({T} rows)", file=sys.stderr, flush=True)
        hn = rmsnorm1p(h, self.norm, cfg.eps)
        lg = self.lm_head_logits(hn)
        self.pos += T
        return h, hn, lg

    def lm_head_logits(self, xn: torch.Tensor) -> torch.Tensor:
        return (xn @ self.lm_head.q.to(torch.float32).t()) * self.lm_head.scale   # fp32 logits (no rounding)

    def mtp_block(self, h_rows: torch.Tensor, next_ids: list[int], pos0: int, emb=None) -> torch.Tensor:
        """MTP over entries j = pos0 .. pos0+n−1 (stream h_j, token x_{j+1}) at rope positions j → logits [n, V]."""
        cfg, m = self.cfg, self.mtp
        n = len(next_ids)
        e = self.embed(next_ids) if emb is None else emb
        x = torch.cat([rmsnorm1p(e, m["en"], cfg.eps), rmsnorm1p(h_rows, m["hn"], cfg.eps)], -1)
        h = m["fc"](x)
        pos3 = torch.arange(pos0, pos0 + n)[:, None].repeat(1, 3) + self.rope_delta
        h = bf(h + m["attn"](rmsnorm1p(h, m["ln1"], cfg.eps), pos0, pos3))
        h = bf(h + m["mlp"](rmsnorm1p(h, m["ln2"], cfg.eps)))
        return self.lm_head_logits(rmsnorm1p(h, m["norm"], cfg.eps))


def main():
    global ROUND
    ap = argparse.ArgumentParser(description="Qwen3.8-27B CPU golden model (per-layer baseline for hived_q27 / hived_q27_cpu).")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-ids", required=True, help="comma-separated ids, or a file (comma / whitespace separated, or a JSON list)")
    ap.add_argument("--decode", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--mtp", action="store_true", help="also run the MTP block over the prompt entries (mtp_top.json)")
    ap.add_argument("--mtp-hidden", choices=["pre", "post"], default="pre",
                    help="MTP hidden input: the residual stream before (pre) or after (post) the final norm")
    ap.add_argument("--all-logits", action="store_true")
    ap.add_argument("--vision-patches", help="raw bf16 file [gh·gw, 1536] (merge-block order) — encoded and placed at --vision-row")
    ap.add_argument("--vision-grid", help="gh,gw (patch grid, both even)")
    ap.add_argument("--vision-row", type=int, default=-1, help="prompt row of the first image token (−1 = only encode, vision.npy)")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    ROUND = a.dtype == "bf16"
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    src = a.prompt_ids
    if os.path.exists(src):
        txt = open(src).read().strip()
        ids = json.loads(txt) if txt.startswith("[") else [int(x) for x in txt.replace(",", " ").split()]
    else:
        ids = [int(x) for x in src.split(",") if x.strip()]
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    tm: dict = {"threads": torch.get_num_threads()}
    with torch.no_grad():
        o = Oracle(a.ckpt, a.mtp)
        tm["load_s"] = time.time() - t0
        emb_over, image = None, None
        if a.vision_patches:
            gh, gw = [int(x) for x in a.vision_grid.split(",")]
            raw = np.fromfile(a.vision_patches, dtype=np.uint16).reshape(gh * gw, -1)
            pt = torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32))
            tv = time.time()
            vis = Vision(o.ck, o.cfg)(pt, gh, gw)
            tm["vision_s"] = time.time() - tv
            np.save(os.path.join(a.out, "vision.npy"), vis.numpy().astype(np.float32))
            if a.vision_row >= 0:
                emb_over = (a.vision_row, vis)
                image = (a.vision_row, vis.shape[0], gh // 2, gw // 2)
        tp = time.time()
        h, hn, logits = o.forward(ids, True, emb_over, image)
        tm["prefill_s"] = time.time() - tp
        for l, hl in o.dumps:
            np.save(os.path.join(a.out, f"h_L{l:02d}.npy"), hl.numpy().astype(np.float32))
        np.save(os.path.join(a.out, "final_norm.npy"), hn.numpy().astype(np.float32))
        rows = [logits.numpy() if a.all_logits else logits[-1:].numpy()]
        if a.mtp and len(ids) > 1:
            tq = time.time()
            hin = h if a.mtp_hidden == "pre" else hn
            emb = o.embed(ids[1:])
            if emb_over is not None:
                r0, e = emb_over
                for r in range(len(ids) - 1):
                    if r0 <= r + 1 < r0 + len(e):
                        emb[r] = e[r + 1 - r0]
            ml = o.mtp_block(hin[:-1], ids[1:], 0, emb)
            tm["mtp_s"] = time.time() - tq
            top = torch.topk(ml, 20, dim=-1)
            main_next = logits.argmax(-1).tolist()          # main model's greedy token after row j+1
            d = top.indices[:, 0].tolist()
            hit_prompt = sum(1 for j in range(len(ids) - 2) if d[j] == ids[j + 2])
            hit_main = sum(1 for j in range(len(ids) - 1) if d[j] == main_next[j + 1]) if a.all_logits else -1
            json.dump({"positions": list(range(len(ids) - 1)), "ids": top.indices.tolist(), "logits": top.values.tolist(),
                       "mtp_hidden": a.mtp_hidden, "top1_vs_prompt": [hit_prompt, len(ids) - 2], "top1_vs_main": [hit_main, len(ids) - 1]},
                      open(os.path.join(a.out, "mtp_top.json"), "w"))
            print(f"[q27-oracle] MTP ({a.mtp_hidden}-norm hidden): top-1 = next prompt token {hit_prompt}/{len(ids) - 2}"
                  + (f" · = main greedy {hit_main}/{len(ids) - 1}" if a.all_logits else ""), file=sys.stderr)
        out_ids = []
        nxt = int(torch.argmax(logits[-1]))
        tm["decode_step_s"] = []
        for _ in range(a.decode):
            out_ids.append(nxt)
            td = time.time()
            _, _, lg = o.forward([nxt], False)
            tm["decode_step_s"].append(time.time() - td)
            rows.append(lg.numpy())
            nxt = int(torch.argmax(lg[-1]))
        np.save(os.path.join(a.out, "logits.npy"), np.concatenate(rows, 0).astype(np.float32))
    json.dump({"prompt": ids, "decoded": out_ids}, open(os.path.join(a.out, "decode.json"), "w"))
    tm["total_s"] = time.time() - t0
    json.dump(tm, open(os.path.join(a.out, "timing.json"), "w"))
    print(f"[q27-oracle] {len(ids)} prompt rows + {a.decode} decode steps in {tm['total_s']:.1f}s (load {tm['load_s']:.1f}s · prefill "
          f"{tm['prefill_s']:.1f}s) → {a.out} · decoded {out_ids}", file=sys.stderr)


if __name__ == "__main__":
    main()
