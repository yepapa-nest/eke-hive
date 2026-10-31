#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""qwen4 oracle: a golden CPU model of Qwen3.8-Flash-Next (config model_type "qwen4_exp", NVIDIA ModelOpt NVFP4 checkpoint), the
per-layer scoring baseline for hived_qwen (engine/include/hive/qwen/*.h hold the math contracts this file follows).

- This project's own code, written from the model's math (no third-party implementation copied). Batch 1, text only. The prompt is one
  chunk (all rows at once, causal inside the chunk), then N greedy decode steps one row at a time — the same incremental state
  (Gated DeltaNet conv/recurrent state, KV + indexer caches, PLE conv history) carries from chunk to chunk.
- Numerics: fp32 arithmetic with a bf16 rounding wherever the model keeps bf16 activations (GEMM outputs, norm outputs, the mixed hidden,
  the hidden stream itself). `--dtype fp32` turns the roundings off (to see how much of an engine difference is rounding). Routed experts
  are NVFP4 dequantized to fp32 weights (W4A16: activations are not quantized); only the experts a token routes to are dequantized.
- Output (--out): h_LNN.npy (hidden stream after layer NN, prompt rows, [T, 10240] float32), routing.json (per layer, per prompt row: top-10
  ids and weights), index.json (per full-attention layer, per prompt row: "dense" or the selected 4-token groups), logits.npy (prompt's
  last row, then one row per decode step), decode.json (decoded ids), ple.json (the PLE rows of every prompt token), and with --mtp:
  mtp_top.json (per prompt position j < T−1: draft top-20 ids / logits of the MTP block fed (h_j, x_{j+1})), mtp_logits.npy with --mtp-full.

Usage: python3 oracle/qwen4_oracle.py --ckpt /models/nvidia-Qwen3.8-Flash-Next-NVFP4 --out /tmp/qo --prompt-ids 248045,872,198 --decode 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
ROUND = True  # bf16 roundings on (--dtype bf16, default)


def bf(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 and back (the points where the model holds bf16)."""
    return x.to(torch.bfloat16).to(torch.float32) if ROUND else x


# ----------------------------------------------------------------------------------------------
# Checkpoint (own minimal safetensors reader: header json + memmap; no conversion of tensors that are never read)
# ----------------------------------------------------------------------------------------------
_NP = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I64": np.int64, "I32": np.int32, "U8": np.uint8, "F8_E4M3": np.uint8, "I8": np.int8,
       "U32": np.uint32}


class Ckpt:
    def __init__(self, path: str):
        self.path = path
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
        self.weight_map = dict(idx["weight_map"])
        # shards that the index does not list (e.g. the separate fp8 MTP/PLE file): add their tensors
        listed = set(self.weight_map.values())
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".safetensors") and fn not in listed:
                hdr, _ = self._header(os.path.join(path, fn))
                for k in hdr:
                    if k != "__metadata__":
                        self.weight_map.setdefault(k, fn)
        self._hdr: dict = {}
        self._mm: dict = {}

    @staticmethod
    def _header(file: str):
        with open(file, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            return json.loads(f.read(n)), 8 + n

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def meta(self, name: str):
        file = os.path.join(self.path, self.weight_map[name])
        if file not in self._hdr:
            self._hdr[file] = self._header(file)
        hdr, base = self._hdr[file]
        return file, hdr[name], base

    def raw(self, name: str) -> np.ndarray:
        """The tensor's bytes as a read-only memmap with its storage dtype and shape (BF16 / F8 as uint16 / uint8)."""
        file, m, base = self.meta(name)
        if file not in self._mm:
            self._mm[file] = np.memmap(file, dtype=np.uint8, mode="r")
        a, b = m["data_offsets"]
        arr = self._mm[file][base + a: base + b].view(_NP[m["dtype"]])
        return arr.reshape(m["shape"])

    def dtype(self, name: str) -> str:
        return self.meta(name)[1]["dtype"]

    def f32(self, name: str) -> torch.Tensor:
        """Any float tensor → fp32 torch tensor (copy). An MXFP8 weight (NVIDIA FP4-Local: F8_E4M3 + a U8 ue8m0 `<name>_scale` per 32
        columns) or an NVFP4 dense weight (U8 + F8_E4M3 `_scale` per 16 + F32 `_scale_2`) comes back dequantized."""
        dt, a = self.dtype(name), self.raw(name)
        if dt == "F8_E4M3" and self.has(name + "_scale") and self.dtype(name + "_scale") == "U8":
            s = torch.from_numpy(np.ascontiguousarray(self.raw(name + "_scale")).astype(np.int32) - 127).to(torch.float32)
            return fp8_rows_to_f32(a) * torch.exp2(s).repeat_interleave(32, dim=1)
        if dt == "U8" and name.endswith(".weight") and self.has(name + "_scale_2"):
            return nvfp4_dequant(self, name[: -len("weight")])
        if dt == "BF16":
            return torch.from_numpy((np.ascontiguousarray(a).astype(np.uint32) << 16).view(np.float32))
        if dt == "F8_E4M3":
            return torch.from_numpy(np.ascontiguousarray(a)).view(torch.float8_e4m3fn).to(torch.float32)
        if dt in ("F32", "F16"):
            return torch.from_numpy(np.ascontiguousarray(a).astype(np.float32))
        raise ValueError(f"{name}: dtype {dt} is not a float tensor")

    def i64(self, name: str) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(self.raw(name)).astype(np.int64))


def fp8_rows_to_f32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).view(torch.float8_e4m3fn).to(torch.float32)


def csf_scales(ck: Ckpt, base: str, R: int, C: int) -> np.ndarray:
    """Compressed NVFP4 block scales (NVIDIA FP4-Local, engine/include/hive/ckpt_quant.h): per 16 rows, 16 base code bytes then 16·C
    4-bit deltas row-major (even position = low nibble); code = base + delta; U32 exceptions = (code << 24) | flat index."""
    fx = np.ascontiguousarray(ck.raw(base + "weight_scale.nvfp4_csf_fixed"))
    ex = np.ascontiguousarray(ck.raw(base + "weight_scale.nvfp4_csf_exceptions"))
    G = R // 16
    body = fx[:, 16:]
    nib = np.empty((G, body.shape[1] * 2), np.int64)
    nib[:, 0::2] = body & 15
    nib[:, 1::2] = body >> 4
    codes = (fx[:, :16].astype(np.int64)[:, :, None] + nib.reshape(G, 16, C)).reshape(-1)
    codes[(ex & 0xFFFFFF).astype(np.int64)] = (ex >> 24).astype(np.int64)
    assert codes.max() <= 0x7E and codes.min() >= 0, base
    return codes.astype(np.uint8).reshape(R, C)


def nvfp4_dequant(ck: Ckpt, base: str) -> torch.Tensor:
    """ModelOpt NVFP4 matrix base+{weight, weight_scale, weight_scale_2} → fp32 [N, K]: low nibble = even column."""
    w = torch.from_numpy(np.ascontiguousarray(ck.raw(base + "weight")))            # uint8 [N, K/2]
    if ck.has(base + "weight_scale"):
        s = fp8_rows_to_f32(ck.raw(base + "weight_scale"))                          # e4m3 [N, K/16]
    else:
        s = fp8_rows_to_f32(csf_scales(ck, base, w.shape[0], w.shape[1] * 2 // 16))
    g = float(ck.f32(base + "weight_scale_2").reshape(-1)[0])
    N, K2 = w.shape
    lo = E2M1[(w & 0xF).long()]
    hi = E2M1[(w >> 4).long()]
    v = torch.stack([lo, hi], dim=-1).reshape(N, K2 * 2)
    return v * s.repeat_interleave(16, dim=1) * g


def fp8_block_dequant(ck: Ckpt, name: str) -> torch.Tensor:
    """FP8 e4m3 weight with a 128×128 block scale (name + "_scale_inv") → fp32 [N, K]."""
    w = fp8_rows_to_f32(ck.raw(name))
    s = ck.f32(name + "_scale_inv")
    N, K = w.shape
    s = s.repeat_interleave(128, dim=0)[:N].repeat_interleave(128, dim=1)[:, :K]
    return w * s


# ----------------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------------
class Cfg:
    def __init__(self, ckpt: str):
        allc = json.load(open(os.path.join(ckpt, "config.json")))
        t = allc.get("text_config", allc)
        self.hidden = t["hidden_size"]
        self.n_layers = t["num_hidden_layers"]
        self.vocab = t["vocab_size"]
        self.hc = t.get("hc_count", 4)
        self.rank = t.get("hc_lowrank", 320)
        self.eps = t.get("rms_norm_eps", 1e-6)
        self.layer_types = t["layer_types"]
        self.kh, self.vh = t["linear_num_key_heads"], t["linear_num_value_heads"]
        self.kd, self.vd = t["linear_key_head_dim"], t["linear_value_head_dim"]
        self.nh, self.nkv, self.hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        rp = t.get("rope_parameters", {})
        self.theta = float(rp.get("rope_theta", t.get("rope_theta", 1e7)))
        self.rot = int(self.hd * rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))
        self.section = list(rp.get("mrope_section", [11, 11, 10]))
        self.interleaved = bool(rp.get("mrope_interleaved", True))
        self.ih, self.idim = t["indexer_n_heads"], t["indexer_head_dim"]
        self.budget, self.ratio = t["indexer_budget"], t["indexer_compress_ratio"]
        self.E, self.K, self.I = t["num_experts"], t["num_experts_per_tok"], t["moe_intermediate_size"]
        self.norm_topk = bool(t.get("norm_topk_prob", True))
        self.ple_layers = [int(x) - 1 for x in t.get("ple_layer_ids", [])]
        self.ngram = t.get("ngram_size", 3)
        self.heads_per = t.get("heads_per_ngram", 8)
        self.eos = int(t["eos_token_id"])
        self.ple_dim = t.get("ple_embed_dim", self.hidden)
        self.ple_conv = t.get("ple_conv_kernel_size", 4)


def rmsnorm1p(x: torch.Tensor, w: torch.Tensor, group: int, eps: float) -> torch.Tensor:
    """(1 + w) RMS norm over groups of `group` elements of the last dim, fp32 math, one rounding."""
    shp = x.shape
    g = x.reshape(*shp[:-1], shp[-1] // group, group)
    y = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps)
    return bf(y.reshape(shp) * (1.0 + w))


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return bf(x @ w.t())


class Weights:
    """Dense (non-expert) weights, kept in RAM as fp32 after first use (layer by layer)."""

    def __init__(self, ck: Ckpt):
        self.ck = ck
        self.cache: dict = {}

    def __call__(self, name: str) -> torch.Tensor:
        t = self.cache.get(name)
        if t is None:
            t = self.ck.f32(name)
            self.cache[name] = t
        return t


class Rope:
    """Interleaved M-RoPE on dims 0..rot-1 (NeoX halves). Pair i uses axis 1 if i%3==1 and i<3·sec[1], axis 2 if i%3==2 and i<3·sec[2]."""

    def __init__(self, cfg: Cfg):
        half = cfg.rot // 2
        self.half = half
        self.inv = cfg.theta ** (-torch.arange(0, half, dtype=torch.float64) * 2.0 / cfg.rot)
        axis = torch.zeros(half, dtype=torch.long)
        if cfg.interleaved:
            i = torch.arange(half)
            axis[(i % 3 == 1) & (i < 3 * cfg.section[1])] = 1
            axis[(i % 3 == 2) & (i < 3 * cfg.section[2])] = 2
        else:
            axis[cfg.section[0]: cfg.section[0] + cfg.section[1]] = 1
            axis[cfg.section[0] + cfg.section[1]:] = 2
        self.axis = axis

    def apply(self, x: torch.Tensor, pos3: torch.Tensor) -> torch.Tensor:
        """x [T, H, D] (rotary on the first 2·half dims), pos3 [T, 3] int → rounded result."""
        p = pos3[:, self.axis].to(torch.float64)                 # [T, half]
        ang = p * self.inv[None, :]
        cos, sin = torch.cos(ang).float()[:, None, :], torch.sin(ang).float()[:, None, :]
        h = self.half
        x1, x2 = x[..., :h], x[..., h:2 * h]
        out = x.clone()
        out[..., :h] = x1 * cos - x2 * sin
        out[..., h:2 * h] = x2 * cos + x1 * sin
        return bf(out)


# ----------------------------------------------------------------------------------------------
# Hyper-connections (gated residual)
# ----------------------------------------------------------------------------------------------
def hc_mix(W: Weights, p: str, h: torch.Tensor, cfg: Cfg):
    """p = '...attn_hyper_connection.' etc. Returns (x [T, hidden], hn [T, hc·hidden])."""
    T, D, hc = h.shape[0], cfg.hidden, cfg.hc
    hn = rmsnorm1p(h, W(p + "hc_norm.weight"), D, cfg.eps)
    dl = linear(hn, W(p + "input_mix_weight_down.weight"))
    s = bf(F.silu(bf(dl / hc)))
    u = linear(s, W(p + "input_mix_weight_up.weight"))
    wgt = bf(torch.sigmoid(u))
    prod = bf(wgt * hn).reshape(T, hc, D)
    return bf(prod.mean(1)), hn


def hc_combine(W: Weights, p: str, h: torch.Tensor, y: torch.Tensor, hn: torch.Tensor, cfg: Cfg) -> torch.Tensor:
    T, D, hc = h.shape[0], cfg.hidden, cfg.hc
    inj = linear(hn, W(p + "block_inject_weight.weight"))      # [T, hc]
    g = bf(2.0 * torch.sigmoid(bf(inj / hc)))
    add = bf(y[:, None, :] * g[:, :, None])
    return bf(h.reshape(T, hc, D) + add).reshape(T, hc * D)


# ----------------------------------------------------------------------------------------------
# PLE
# ----------------------------------------------------------------------------------------------
class PleHash:
    def __init__(self, mult, size, offset, ngram: int, heads_per: int, eos: int):
        self.mult, self.size, self.offset = [int(x) for x in mult], [int(x) for x in size], [int(x) for x in offset]
        self.ngram, self.heads_per, self.eos = ngram, heads_per, eos

    def rows(self, x: int, p1: int, p2: int) -> list[int]:
        """Rows of token x whose predecessors are p1 = x_{t−1}, p2 = x_{t−2} (EOS before the start)."""
        b = p1
        c = self.eos if p1 == self.eos else p2
        mix2 = (x * self.mult[0]) ^ (b * self.mult[1])
        mix3 = mix2 ^ (c * self.mult[2])
        out = []
        for n, mix in ((2, mix2), (3, mix3)):
            for j in range((n - 2) * self.heads_per, (n - 1) * self.heads_per):
                out.append(mix % self.size[j] + self.offset[j])
        return out


class Ple:
    def __init__(self, ck: Ckpt, W: Weights, cfg: Cfg, layer: int):
        self.ck, self.W, self.cfg = ck, W, cfg
        self.p = f"model.language_model.layers.{layer}.ple."
        e = self.p + "ple_embedding."
        self.hash = PleHash(ck.i64(e + "layer_multipliers"), ck.i64(e + "ngram_heads_vocab_sizes"), ck.i64(e + "ngram_heads_offsets"),
                            cfg.ngram, cfg.heads_per, cfg.eos)
        self.nv = not ck.has(e + "ngram_embedding.weight_scale")   # FP4-Local: NVFP4 rows (per-16 e4m3 scales, fp32 scale_2)
        self.scale = float(ck.f32(e + ("ngram_embedding.weight_scale_2" if self.nv else "ngram_embedding.weight_scale")).reshape(-1)[0])
        self.shards, self.sshards = [], []
        i = 0
        while ck.has(f"{e}ngram_embedding.shard_{i}.weight"):
            self.shards.append(ck.raw(f"{e}ngram_embedding.shard_{i}.weight"))
            if self.nv:
                self.sshards.append(ck.raw(f"{e}ngram_embedding.shard_{i}.weight_scale"))
            i += 1
        assert self.shards, "PLE shards not found"
        self.per = self.shards[0].shape[0]
        self.hist = [cfg.eos, cfg.eos]          # x_{t−2}, x_{t−1}
        self.conv_hist = torch.zeros(9, cfg.hc * cfg.hidden)  # last 9 conv inputs (oldest first)
        self.dump: list = []

    def row(self, r: int) -> torch.Tensor:
        sh = r // self.per
        if self.nv:   # value = e2m1 · e4m3 (the scale_2 product is applied with self.scale, as the FP8 table's scale)
            w = torch.from_numpy(np.ascontiguousarray(self.shards[sh][r - sh * self.per]))
            v = torch.stack([E2M1[(w & 0xF).long()], E2M1[(w >> 4).long()]], dim=-1).reshape(-1)
            return v * fp8_rows_to_f32(self.sshards[sh][r - sh * self.per]).repeat_interleave(16)
        return fp8_rows_to_f32(self.shards[sh][r - sh * self.per])

    def __call__(self, ids: list[int], h: torch.Tensor) -> torch.Tensor:
        cfg, W, p = self.cfg, self.W, self.p
        T, D, hc = len(ids), cfg.hidden, cfg.hc
        emb = []
        for x in ids:
            rows = self.hash.rows(x, self.hist[1], self.hist[0])
            self.dump.append(rows)
            emb.append(torch.cat([self.row(r) for r in rows]))
            self.hist = [self.hist[1], x]
        e = bf(torch.stack(emb) * self.scale)                       # [T, 2560]
        key = linear(e, W(p + "key_proj.weight"))                    # [T, 10240]
        value = linear(e, W(p + "value_proj.weight"))                # [T, 2560]
        kn = rmsnorm1p(key, W(p + "norm_key.weight"), D, cfg.eps).reshape(T, hc, D)
        qn = rmsnorm1p(h, W(p + "norm_query.weight"), D, cfg.eps).reshape(T, hc, D)
        g = (kn * qn).sum(-1) / math.sqrt(D)
        g = torch.sign(g) * torch.sqrt(g.abs().clamp_min(1e-6))
        g = bf(torch.sigmoid(g))
        gv = bf(g[:, :, None] * value[:, None, :]).reshape(T, hc * D)
        gvn = rmsnorm1p(gv, W(p + "norm_conv.weight"), D, cfg.eps)
        w = W(p + "conv1d.weight").reshape(hc * D, cfg.ple_conv)    # taps x_{t−9}, x_{t−6}, x_{t−3}, x_t
        seq = torch.cat([self.conv_hist, gvn], 0)                    # [9 + T, C]
        out = torch.empty(T, hc * D)
        for t in range(T):
            j = 9 + t
            y = w[:, 0] * seq[j - 9] + w[:, 1] * seq[j - 6] + w[:, 2] * seq[j - 3] + w[:, 3] * seq[j]
            out[t] = bf(F.silu(bf(y)))
        self.conv_hist = seq[-9:].clone()
        return bf(h + bf(gv + out))


# ----------------------------------------------------------------------------------------------
# Gated DeltaNet
# ----------------------------------------------------------------------------------------------
class Gdn:
    def __init__(self, W: Weights, cfg: Cfg, layer: int):
        self.W, self.cfg = W, cfg
        self.p = f"model.language_model.layers.{layer}.linear_attn."
        C = 2 * cfg.kh * cfg.kd + cfg.vh * cfg.vd
        self.conv_hist = torch.zeros(3, C)
        self.S = torch.zeros(cfg.vh, cfg.kd, cfg.vd)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cfg, W, p = self.cfg, self.W, self.p
        T = x.shape[0]
        KW, VW = cfg.kh * cfg.kd, cfg.vh * cfg.vd
        qkv = linear(x, W(p + "in_proj_qkv.weight"))
        z = linear(x, W(p + "in_proj_z.weight"))
        b = linear(x, W(p + "in_proj_b.weight"))
        a = linear(x, W(p + "in_proj_a.weight"))
        cw = W(p + "conv1d.weight").reshape(qkv.shape[1], 4)
        seq = torch.cat([self.conv_hist, qkv], 0)
        y = torch.stack([(cw * seq[t:t + 4].t()).sum(-1) for t in range(T)])   # tap 3 = current
        self.conv_hist = seq[-3:].clone()
        y = bf(F.silu(y))
        q = y[:, :KW].reshape(T, cfg.kh, cfg.kd)
        k = y[:, KW:2 * KW].reshape(T, cfg.kh, cfg.kd)
        v = y[:, 2 * KW:].reshape(T, cfg.vh, cfg.vd)
        q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * (cfg.kd ** -0.5)
        k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
        sp = a + W(p + "dt_bias")
        sp = torch.where(sp > 20.0, sp, torch.log1p(torch.exp(sp)))
        g = -torch.exp(W(p + "A_log")) * sp                          # [T, vh]
        beta = bf(torch.sigmoid(b))
        rep = cfg.vh // cfg.kh
        qv, kv_ = q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1)   # value head h reads key head h // rep
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
        on = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + cfg.eps) * W(p + "norm.weight")
        on = bf(on * torch.sigmoid(z.reshape(T, cfg.vh, cfg.vd)))
        return linear(on.reshape(T, VW), W(p + "out_proj.weight"))


# ----------------------------------------------------------------------------------------------
# Full attention + QSA indexer
# ----------------------------------------------------------------------------------------------
class Attn:
    def __init__(self, W: Weights, cfg: Cfg, rope: Rope, prefix: str):
        self.W, self.cfg, self.rope, self.p = W, cfg, rope, prefix   # prefix = "...self_attn."
        self.k = []; self.v = []; self.ik = []; self.gk = {}; self.pos3 = []
        self.index_log: list = []

    def __call__(self, x: torch.Tensor, pos0: int, pos3: torch.Tensor) -> torch.Tensor:
        cfg, W, p = self.cfg, self.W, self.p
        T, H, G, D = x.shape[0], cfg.nh, cfg.nkv, cfg.hd
        qg = linear(x, W(p + "q_proj.weight")).reshape(T, H, 2 * D)
        q, gate = qg[..., :D], qg[..., D:]
        k = linear(x, W(p + "k_proj.weight")).reshape(T, G, D)
        v = linear(x, W(p + "v_proj.weight")).reshape(T, G, D)
        q = self.rope.apply(rmsnorm1p(q, W(p + "q_norm.weight"), D, cfg.eps), pos3)
        k = self.rope.apply(rmsnorm1p(k, W(p + "k_norm.weight"), D, cfg.eps), pos3)
        iqk = linear(x, W(p + "indexer.index_qk_proj.weight"))
        iq = iqk[:, :cfg.ih * cfg.idim].reshape(T, cfg.ih, cfg.idim)
        ik = iqk[:, cfg.ih * cfg.idim:]
        iq = self.rope.apply(rmsnorm1p(iq, W(p + "indexer.q_layernorm.weight"), cfg.idim, cfg.eps), pos3)
        for t in range(T):
            self.k.append(k[t]); self.v.append(v[t]); self.ik.append(ik[t]); self.pos3.append(pos3[t])
        r = cfg.ratio
        # complete the groups whose last token is in this chunk
        for t in range(T):
            P = pos0 + t
            if (P + 1) % r == 0:
                Gi = P // r
                mean = bf(torch.stack(self.ik[Gi * r:(Gi + 1) * r]).mean(0))
                kn = rmsnorm1p(mean[None, None, :], W(p + "indexer.k_layernorm.weight"), cfg.idim, cfg.eps)
                self.gk[Gi] = self.rope.apply(kn, self.pos3[Gi * r][None, :])[0, 0]
        Kall, Vall = torch.stack(self.k), torch.stack(self.v)        # [N, G, D]
        out = torch.empty(T, H, D)
        nblk = cfg.budget // r
        rep = H // G
        for t in range(T):
            P = pos0 + t
            vis = (P + 1) // r
            if vis <= nblk:
                sel = torch.arange(P + 1)
                self.index_log.append("dense")
            else:
                gk = torch.stack([self.gk[i] for i in range(vis)])     # [vis, idim]
                sc = torch.relu(iq[t] @ gk.t()).sum(0) / math.sqrt(cfg.idim)
                top = torch.topk(sc, nblk).indices.sort().values
                tail = torch.arange(r * vis, P + 1)
                sel = torch.cat([(top[:, None] * r + torch.arange(r)[None, :]).reshape(-1), tail])
                self.index_log.append(top.tolist())
            Ks, Vs = Kall[sel], Vall[sel]                           # [S, G, D]
            for h in range(H):
                g_ = h // rep
                s = (Ks[:, g_] @ q[t, h]) / math.sqrt(D)
                pr = torch.softmax(s, 0)
                out[t, h] = pr @ Vs[:, g_]
        o = bf(bf(out) * torch.sigmoid(gate))
        return linear(o.reshape(T, H * D), W(p + "o_proj.weight"))


# ----------------------------------------------------------------------------------------------
# MoE
# ----------------------------------------------------------------------------------------------
class Moe:
    def __init__(self, ck: Ckpt, W: Weights, cfg: Cfg, prefix: str, fp8_experts: bool, cache_n: int = 256):
        self.ck, self.W, self.cfg, self.p, self.fp8 = ck, W, cfg, prefix, fp8_experts   # prefix = "...mlp."
        self.cache: OrderedDict = OrderedDict()
        self.cache_n = cache_n
        self.routing: list = []

    def expert(self, e: int):
        t = self.cache.get(e)
        if t is not None:
            self.cache.move_to_end(e)
            return t
        b = f"{self.p}experts.{e}."
        if self.fp8:
            t = tuple(fp8_block_dequant(self.ck, b + n + ".weight") for n in ("gate_proj", "up_proj", "down_proj"))
        else:
            t = tuple(nvfp4_dequant(self.ck, b + n + ".") for n in ("gate_proj", "up_proj", "down_proj"))
        self.cache[e] = t
        if len(self.cache) > self.cache_n:
            self.cache.popitem(last=False)
        return t

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cfg, W, p = self.cfg, self.W, self.p
        T = x.shape[0]
        logits = linear(x, W(p + "gate.weight"))
        prob = torch.softmax(logits, -1)
        w, ids = torch.topk(prob, cfg.K, dim=-1)
        if cfg.norm_topk:
            w = w / w.sum(-1, keepdim=True)
        acc = torch.zeros(T, cfg.hidden)
        for t in range(T):
            self.routing.append({"ids": ids[t].tolist(), "w": [float(a) for a in w[t]]})
            for j in range(cfg.K):
                g, u, d = self.expert(int(ids[t, j]))
                gt = linear(x[t:t + 1], g); ut = linear(x[t:t + 1], u)
                y = bf(F.silu(gt) * ut)
                acc[t] += w[t, j] * linear(y, d)[0]
        sg = linear(x, W(p + "shared_expert.gate_proj.weight"))
        su = linear(x, W(p + "shared_expert.up_proj.weight"))
        sh = linear(bf(F.silu(sg) * su), W(p + "shared_expert.down_proj.weight"))
        gl = linear(x, W(p + "shared_expert_gate.weight"))             # [T, 1]
        return bf(bf(acc) + bf(torch.sigmoid(gl) * sh))


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
class Oracle:
    def __init__(self, ckpt: str, mtp: bool):
        self.ck = Ckpt(ckpt)
        self.cfg = Cfg(ckpt)
        self.W = Weights(self.ck)
        self.rope = Rope(self.cfg)
        cfg, P = self.cfg, "model.language_model."
        self.blocks = []
        for l in range(cfg.n_layers):
            attn = cfg.layer_types[l] == "full_attention"
            core = Attn(self.W, cfg, self.rope, f"{P}layers.{l}.self_attn.") if attn else Gdn(self.W, cfg, l)
            ple = Ple(self.ck, self.W, cfg, l) if l in cfg.ple_layers else None
            self.blocks.append((attn, core, ple, Moe(self.ck, self.W, cfg, f"{P}layers.{l}.mlp.", False)))
        self.mtp = None
        if mtp:
            fp8 = self.ck.dtype("mtp.layers.0.mlp.experts.0.gate_proj.weight") == "F8_E4M3"   # FP4-Local: NVFP4 MTP experts
            self.mtp = (Attn(self.W, cfg, self.rope, "mtp.layers.0.self_attn."), Moe(self.ck, self.W, cfg, "mtp.layers.0.mlp.", fp8, 64))
        self.pos = 0
        self.dumps: list = []

    def embed(self, ids: list[int]) -> torch.Tensor:
        a = self.ck.raw("model.language_model.embed_tokens.weight")
        e = (np.stack([np.asarray(a[i]) for i in ids]).astype(np.uint32) << 16).view(np.float32)
        return torch.from_numpy(e)

    def forward(self, ids: list[int], keep_layers: bool):
        """One chunk of rows. Returns (final hidden streams [T, 10240], logits of every row [T, V])."""
        cfg, W, P = self.cfg, self.W, "model.language_model."
        T = len(ids)
        pos3 = torch.arange(self.pos, self.pos + T)[:, None].repeat(1, 3)
        h = self.embed(ids).repeat(1, cfg.hc)
        for l, (attn, core, ple, moe) in enumerate(self.blocks):
            if ple is not None:
                h = ple(ids, h)
            pa = f"{P}layers.{l}.attn_hyper_connection."
            pm = f"{P}layers.{l}.mlp_hyper_connection."
            x, hn = hc_mix(W, pa, h, cfg)
            y = core(x, self.pos, pos3) if attn else core(x)
            h = hc_combine(W, pa, h, y, hn, cfg)
            x, hn = hc_mix(W, pm, h, cfg)
            h = hc_combine(W, pm, h, moe(x), hn, cfg)
            if keep_layers:
                self.dumps.append((l, h.clone()))
            if l % 8 == 7 or l == cfg.n_layers - 1:
                print(f"[oracle] layer {l} done ({T} rows)", file=sys.stderr, flush=True)
        xf, _ = hc_mix(W, f"{P}hyper_connection_mixer.", h, cfg)
        logits = xf @ W("lm_head.weight").t()
        self.pos += T
        return h, logits

    def mtp_draft(self, h_rows: torch.Tensor, next_ids: list[int], pos0: int) -> torch.Tensor:
        """MTP block over entries j = pos0 .. pos0+n−1 (h_j, x_{j+1}) at rope positions j → logits [n, V]."""
        cfg, W = self.cfg, self.W
        attn, moe = self.mtp
        n = len(next_ids)
        e = rmsnorm1p(self.embed(next_ids), W("mtp.pre_fc_norm_embedding.weight"), cfg.hidden, cfg.eps)
        e = linear(e, W("mtp.fc_embedding.weight"))
        hn = rmsnorm1p(h_rows, W("mtp.pre_fc_norm_hidden.weight"), cfg.hc * cfg.hidden, cfg.eps)
        fh = linear(hn.reshape(n, cfg.hc, cfg.hidden), W("mtp.fc_hidden.weight"))
        h = bf(e[:, None, :] + fh).reshape(n, cfg.hc * cfg.hidden)
        pos3 = torch.arange(pos0, pos0 + n)[:, None].repeat(1, 3)
        pa, pm = "mtp.layers.0.attn_hyper_connection.", "mtp.layers.0.mlp_hyper_connection."
        x, hnn = hc_mix(W, pa, h, cfg)
        h = hc_combine(W, pa, h, attn(x, pos0, pos3), hnn, cfg)
        x, hnn = hc_mix(W, pm, h, cfg)
        h = hc_combine(W, pm, h, moe(x), hnn, cfg)
        xf, _ = hc_mix(W, "mtp.hyper_connection_mixer.", h, cfg)
        return xf @ W("lm_head.weight").t()


def main():
    global ROUND
    ap = argparse.ArgumentParser(description="Qwen3.8-Flash-Next CPU golden model (per-layer baseline for hived_qwen).")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-ids", required=True, help="comma-separated ids, or a file with ids (comma / whitespace separated, or a JSON list)")
    ap.add_argument("--decode", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="bf16 = round where the model keeps bf16 (default)")
    ap.add_argument("--mtp", action="store_true", help="also run the MTP block over the prompt entries (mtp_top.json)")
    ap.add_argument("--mtp-full", action="store_true", help="with --mtp: also write mtp_logits.npy")
    ap.add_argument("--all-logits", action="store_true", help="logits.npy holds every prompt row (default: the last one)")
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
    tm = {"threads": torch.get_num_threads()}
    with torch.no_grad():
        o = Oracle(a.ckpt, a.mtp)
        tm["load_s"] = time.time() - t0
        tp = time.time()
        h, logits = o.forward(ids, True)
        tm["prefill_s"] = time.time() - tp
        for l, hl in o.dumps:
            np.save(os.path.join(a.out, f"h_L{l:02d}.npy"), hl.numpy().astype(np.float32))
        rows = [logits.numpy() if a.all_logits else logits[-1:].numpy()]
        mtp_out = None
        if a.mtp and len(ids) > 1:
            tq = time.time()
            ml = o.mtp_draft(h[:-1], ids[1:], 0)
            tm["mtp_s"] = time.time() - tq
            top = torch.topk(ml, 20, dim=-1)
            mtp_out = {"positions": list(range(len(ids) - 1)), "ids": top.indices.tolist(), "logits": top.values.tolist()}
            if a.mtp_full:
                np.save(os.path.join(a.out, "mtp_logits.npy"), ml.numpy().astype(np.float32))
        out_ids = []
        nxt = int(torch.argmax(logits[-1]))
        tm["decode_step_s"] = []
        for _ in range(a.decode):
            out_ids.append(nxt)
            td = time.time()
            _, lg = o.forward([nxt], False)
            tm["decode_step_s"].append(time.time() - td)
            rows.append(lg.numpy())
            nxt = int(torch.argmax(lg[-1]))
        np.save(os.path.join(a.out, "logits.npy"), np.concatenate(rows, 0).astype(np.float32))
    T = len(ids)
    json.dump({"routing": {str(l): b[3].routing[:T] for l, b in enumerate(o.blocks)}}, open(os.path.join(a.out, "routing.json"), "w"))
    json.dump({str(l): b[1].index_log[:T] for l, b in enumerate(o.blocks) if b[0]}, open(os.path.join(a.out, "index.json"), "w"))
    json.dump({"prompt": ids, "decoded": out_ids}, open(os.path.join(a.out, "decode.json"), "w"))
    ple = {str(l): b[2].dump[:T] for l, b in enumerate(o.blocks) if b[2] is not None}
    json.dump(ple, open(os.path.join(a.out, "ple.json"), "w"))
    if mtp_out is not None:
        json.dump(mtp_out, open(os.path.join(a.out, "mtp_top.json"), "w"))
    tm["total_s"] = time.time() - t0
    json.dump(tm, open(os.path.join(a.out, "timing.json"), "w"))
    print(f"[oracle] {T} prompt rows + {a.decode} decode steps in {tm['total_s']:.1f}s (load {tm['load_s']:.1f}s · prefill {tm['prefill_s']:.1f}s"
          f"{' · mtp %.1fs' % tm['mtp_s'] if 'mtp_s' in tm else ''} · decode {sum(tm['decode_step_s']):.1f}s) → {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
