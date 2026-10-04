#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Portions ported from the DeepSeek-V4.1-Flash reference implementation
# (inference/model.py, inference/kernel.py), Copyright (c) 2023 DeepSeek, MIT License.
"""dsv41 oracle: a golden model that ports the DeepSeek-V4.1-Flash reference implementation
(inference/model.py + kernel.py) to plain torch without tilelang. It is the per-layer scoring baseline for the hive engine.

- Batch 1, text only, one prefill chunk (start_pos=0) + N decode steps. Layer weights are streamed from safetensors one
  layer at a time and only the routed experts are dequantized, so it runs without a GPU (CPU). Target checkpoint =
  deepseek-ai/DeepSeek-V4.1-Flash (original MXFP4 experts + 32x32-block fp8 dense weights, ue8m0 scales).
- Numeric contracts follow the reference kernels' **definitions**: act_quant uses an e8m0 2^ceil(log2(amax/448)) scale
  + fp8 RNE, fp4 uses e2m1 RNE (ties to the even code), sparse_attn uses bf16 P·V + fp32 accumulation + sink, sinkhorn
  runs 20 iterations. The only place that may differ from the reference is the GEMM accumulation order (tensor-core
  block accumulation vs fp32 matmul). Scoring is therefore per-layer error and logit KL, not bit equality.
- Output (in the --out directory): per-layer residual stream (h_LNN.npy, [L,4,5120] fp32), routing (routing.json:
  top-6 ids/weights per layer and token), sparse index selection (index.json), logits (logits.npy), decoded tokens
  (decode.json).

Note: this file is not a "clean reimplementation" but a **mirror** of the reference code. Ported as-is even where the
reference looks odd (e.g. indexer scores computed with a bf16 einsum and topk taken in bf16; with many ties, topk tie
handling is implementation-defined).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)
# e2m1 RNE midpoints and tie direction (True = round to the upper code). In code order 0,.5,1,1.5,2,3,4,6 the even codes are 0,1,2,4.
FP4_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
FP4_TIE_UP = torch.tensor([False, True, False, True, False, True, False])
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

FP8_MAX = 448.0
FP8_MAX_INV = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
FP4_MAX = 6.0
FP4_MAX_INV = torch.tensor(1.0 / FP4_MAX, dtype=torch.float32)


# ----------------------------------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------------------------------
class Ckpt:
    """Maps tensor -> shard via model.safetensors.index.json and opens shards only when needed. Engram tables are looked up row by row via np.memmap."""

    def __init__(self, path: str):
        self.path = path
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
        self.weight_map = idx["weight_map"]
        self._handles = {}
        self._headers = {}

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def shard_of(self, name: str) -> str:
        return os.path.join(self.path, self.weight_map[name])

    def _open(self, shard: str):
        if shard not in self._handles:
            from safetensors import safe_open

            if not os.path.exists(shard):
                raise FileNotFoundError(f"shard not present yet: {shard}")
            self._handles[shard] = safe_open(shard, framework="pt", device="cpu")
        return self._handles[shard]

    def get(self, name: str) -> torch.Tensor:
        return self._open(self.shard_of(name)).get_tensor(name)

    def header(self, shard: str) -> tuple[dict, int]:
        """safetensors header (json) and the data start offset."""
        if shard not in self._headers:
            with open(shard, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(n))
            self._headers[shard] = (hdr, 8 + n)
        return self._headers[shard]

    def memmap_rows(self, name: str) -> tuple[np.memmap, tuple[int, ...], str]:
        """A [rows, cols] tensor as a uint8 memmap, for the engram tables (94GB). Returns (memmap[rows, cols_bytes], shape, dtype)"""
        shard = self.shard_of(name)
        hdr, base = self.header(shard)
        meta = hdr[name]
        start, end = meta["data_offsets"]
        shape = tuple(meta["shape"])
        rows = shape[0]
        row_bytes = (end - start) // rows
        mm = np.memmap(shard, dtype=np.uint8, mode="r", offset=base + start, shape=(rows, row_bytes))
        return mm, shape, meta["dtype"]


def e8m0_to_float(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.uint8:
        return torch.pow(2.0, t.to(torch.float32) - 127.0)
    return t.to(torch.float32)


def fp8_to_float(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.uint8:
        t = t.view(torch.float8_e4m3fn)
    return t.to(torch.float32)


# ----------------------------------------------------------------------------------------------
# Quantization primitives (definitions taken verbatim from kernel.py)
# ----------------------------------------------------------------------------------------------
def ceil_log2_pow2(x: torch.Tensor) -> torch.Tensor:
    """fast_round_scale: 2^ceil(log2(x)); unchanged if x is a power of two. x>0, fp32."""
    m, e = torch.frexp(x)  # x = m·2^e, m ∈ [0.5,1)
    e = e - (m == 0.5).to(e.dtype)
    return torch.ldexp(torch.ones_like(x), e)


def act_quant_fp8(x: torch.Tensor, block: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block fp8 (e4m3) quantization with an e8m0 scale (rounded up to a power of two). Returns (dequantized fp32
    values, fp32 scale). Reference: act_quant(x, 32, 'ue8m0', e8m0)."""
    xf = x.to(torch.float32)
    g = xf.unflatten(-1, (-1, block))
    amax = g.abs().amax(-1).clamp_min(1e-4)
    s = ceil_log2_pow2(amax * FP8_MAX_INV)
    y = torch.clamp(g / s.unsqueeze(-1), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    return (y * s.unsqueeze(-1)).flatten(-2), s


def act_quant_fp8_inplace(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    y, _ = act_quant_fp8(x, block)
    x.copy_(y.to(x.dtype))
    return x


def fp4_rne(v: torch.Tensor) -> torch.Tensor:
    """RNE of fp32 values with |v| <= 6 onto the e2m1 grid (ties to the even code)."""
    a = v.abs()
    up = (a.unsqueeze(-1) > FP4_MID).sum(-1)
    tie = ((a.unsqueeze(-1) == FP4_MID) & FP4_TIE_UP).any(-1)
    idx = up + tie.to(up.dtype)
    return FP4_GRID[idx] * torch.sign(v)


def fp4_act_quant_inplace(x: torch.Tensor, block: int, scale_e4m3: bool) -> torch.Tensor:
    """Reference fp4_act_quant(inplace=True). e8m0: amax>=6·2^-126, s=2^ceil(log2(amax/6)); e4m3 (compressed KV):
    amax>=6·2^-9, s=fp8(amax/6)."""
    xf = x.to(torch.float32)
    g = xf.unflatten(-1, (-1, block))
    amax = g.abs().amax(-1)
    if scale_e4m3:
        amax = amax.clamp_min(6 * 2.0**-9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        s = ceil_log2_pow2(amax * FP4_MAX_INV)
    y = fp4_rne(torch.clamp(g / s.unsqueeze(-1), -FP4_MAX, FP4_MAX)) * s.unsqueeze(-1)
    x.copy_(y.flatten(-2).to(x.dtype))
    return x


def deq_fp8_block(w: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """fp8 [N,K] + e8m0 [N/b, K/b] → fp32 [N,K]."""
    n, k = w.shape
    sf = e8m0_to_float(s)
    assert sf.shape == ((n + block - 1) // block, (k + block - 1) // block), (w.shape, s.shape)
    sf = sf.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return fp8_to_float(w) * sf


def deq_fp4(w: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """int8/uint8 [N,K/2] (two nibbles, low nibble first) + e8m0 [N,K/32] -> fp32 [N,K]."""
    n, k2 = w.shape
    b = w.view(torch.uint8)
    lo = (b & 0x0F).to(torch.long)
    hi = (b >> 4).to(torch.long)
    vals = torch.stack([FP4_TABLE[lo], FP4_TABLE[hi]], dim=-1).reshape(n, k2 * 2)
    sf = e8m0_to_float(s)
    assert sf.shape == (n, k2 * 2 // block), (w.shape, s.shape)
    return vals * sf.repeat_interleave(block, 1)


class Lin:
    """The three forms of the reference Linear: fp8 (32x32 blocks), fp4 (experts), bf16/fp32 as-is. forward follows the
    result dtype of the reference linear() (quantized GEMM output = default dtype bf16; F.linear keeps the input dtype)."""

    def __init__(self, kind: str, w: torch.Tensor):
        self.kind = kind
        self.w = w  # fp32 (dequantized) or the original bf16/fp32

    @classmethod
    def fp8(cls, ck: Ckpt, name: str):
        return cls("fp8", deq_fp8_block(ck.get(name + ".weight"), ck.get(name + ".scale")))

    @classmethod
    def fp4(cls, ck: Ckpt, name: str):
        return cls("fp4", deq_fp4(ck.get(name + ".weight"), ck.get(name + ".scale")))

    @classmethod
    def plain(cls, ck: Ckpt, name: str, dtype=None):
        w = ck.get(name + ".weight")
        return cls("plain", w.to(dtype) if dtype is not None else w)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind in ("fp8", "fp4"):
            # activations are fp8(32, e8m0)-quantized, then fp32-accumulated GEMM -> bf16 output
            a, _ = act_quant_fp8(x, 32)
            return (a @ self.w.T).to(torch.bfloat16)
        return F.linear(x, self.w)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (w * xf).to(dt)


# ----------------------------------------------------------------------------------------------
# RoPE
# ----------------------------------------------------------------------------------------------
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotates adjacent pairs of x[..., d] treated as complex numbers. x is [s,d] or [s,h,d] (no batch). Updated in place."""
    y = x
    xc = torch.view_as_complex(x.to(torch.float32).unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 2:
        freqs_cis = freqs_cis.view(xc.size(0), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(xc.size(0), 1, xc.size(-1))
    xr = torch.view_as_real(xc * freqs_cis).flatten(-2)
    y.copy_(xr)
    return y


# ----------------------------------------------------------------------------------------------
# Sparse attention, Sinkhorn (definitions from kernel.py)
# ----------------------------------------------------------------------------------------------
def sparse_attn(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, idxs: torch.Tensor, scale: float) -> torch.Tensor:
    """q [m,h,d] bf16, kv [n,d] bf16, idxs [m,topk] int (-1 = none). Scores = bf16·bf16 with fp32 accumulation; P is cast
    to bf16, multiplied with V (fp32 accumulation) and divided by the fp32 sum_exp (+sink). Rows that are all -1 give 0."""
    m, h, d = q.shape
    out = torch.empty_like(q)
    kvf = kv.to(torch.float32)
    sinkf = sink.to(torch.float32)
    for i in range(m):
        ix = idxs[i].to(torch.long)
        valid = ix >= 0
        g = kvf[ix.clamp_min(0)]  # [topk,d]
        s = (q[i].to(torch.float32) @ g.T) * scale  # [h,topk]
        s = s.masked_fill(~valid.unsqueeze(0), -float("inf"))
        mx = s.amax(-1).clamp_min(-1e30)  # the kernel's -1e30 initial value
        p = torch.exp(s - mx.unsqueeze(-1))
        denom = p.sum(-1) + torch.exp(sinkf - mx)
        acc = p.to(torch.bfloat16).to(torch.float32) @ g
        out[i] = (acc / denom.unsqueeze(-1)).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc: int, iters: int, eps: float):
    """mixes [n, (2+hc)·hc] fp32 → pre [n,hc], post [n,hc], comb [n,hc,hc]."""
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).view(-1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


# ----------------------------------------------------------------------------------------------
# Engram (definitions from engram.py)
# ----------------------------------------------------------------------------------------------
def build_compressed_token_map(tok) -> tuple[list[int], int]:
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    n = tok.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = [0] * n
    for tid in range(n):
        text = tok.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(tid)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[tid] = new_id
    return lookup, len(key_to_new)


def find_next_prime(start: int, seen: set[int]) -> int:
    from sympy import isprime

    c = start + 1
    while not isprime(c) or c in seen:
        c += 1
    return c


class EngramHash:
    DEAD = -1

    def __init__(self, cfg: dict, tok, max_seq_len: int):
        self.layer_ids = list(cfg["engram_layer_ids"])
        self.max_ngram = cfg["engram_max_ngram_size"]
        n_heads = cfg["engram_n_heads"]
        primes, seen = [], set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram - 1):
                sizes, cur = [], cfg["engram_vocab_size"] - 1
                for _ in range(n_heads):
                    cur = find_next_prime(cur, seen)
                    seen.add(cur)
                    sizes.append(cur)
                per_ngram.append(sizes)
            primes.append(per_ngram)
        token_map, vocab = build_compressed_token_map(tok)
        assert vocab == cfg["engram_compressed_vocab_size"], (vocab, cfg["engram_compressed_vocab_size"])
        self.pad_id = token_map[cfg["engram_pad_token_id"]]
        flat = [[p for per in layer for p in per] for layer in primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        max_long = np.iinfo(np.int64).max
        bound = max(1, (max_long // vocab) // 2)
        rows = []
        for lid in self.layer_ids:
            g = np.random.default_rng(10007 * lid)
            vals = g.integers(low=0, high=bound, size=(self.max_ngram,), dtype=np.int64)
            rows.append(torch.tensor(vals * 2 + 1))
        self.multipliers = torch.stack(rows)  # [n_layers, max_ngram]
        self.primes = torch.tensor(primes)  # [n_layers, max_ngram-1, n_heads]
        self.offsets = torch.tensor(np.array(offsets))  # [n_layers, (max_ngram-1)·n_heads]
        self.token_map = torch.tensor(token_map)
        self.cache = torch.empty(max_seq_len, dtype=torch.int64)

    def __call__(self, ids: torch.Tensor, start_pos: int, token_mask=None) -> torch.Tensor:
        """ids [L] -> [L, n_engram_layers, n_hash_cols] int64. token_mask False = image (dead)"""
        L = ids.numel()
        comp = self.token_map[ids]
        if token_mask is not None:
            comp = torch.where(token_mask, comp, torch.tensor(self.DEAD))
        self.cache[start_pos : start_pos + L] = comp
        pos = torch.arange(start_pos, start_pos + L)
        toks, blocked = [], torch.zeros(L, dtype=torch.bool)
        for shift in range(self.max_ngram):
            src = self.cache[(pos - shift).clamp_min(0)]
            blocked = blocked | (pos < shift) | (src == self.DEAD)
            toks.append(torch.where(blocked, self.pad_id, src))
        toks = torch.stack(toks, -1)  # [L, max_ngram]
        prod = toks.unsqueeze(1) * self.multipliers  # [L, n_layers, max_ngram]
        rolling, hashes = prod[..., 0], []
        for i in range(1, self.max_ngram):
            rolling = torch.bitwise_xor(rolling, prod[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, -1) + self.offsets


class EngramLayer:
    def __init__(self, ck: Ckpt, cfg: dict, layer_id: int, hash_index: int):
        p = f"layers.{layer_id}.engram"
        self.layer_id = layer_id
        self.hash_index = hash_index
        self.dim = cfg["hidden_size"]
        self.hc = cfg["hc_mult"]
        self.eps = cfg["rms_norm_eps"]
        self.wkv = Lin.fp8(ck, p + ".wkv")
        self.q_weight = ck.get(p + ".q_weight").to(torch.float32)
        self.k_weight = ck.get(p + ".k_weight").to(torch.float32)
        self.w_mm, self.w_shape, self.w_dtype = ck.memmap_rows(p + ".embed.weight")
        self.s_mm, self.s_shape, self.s_dtype = ck.memmap_rows(p + ".embed.scale")
        assert self.w_dtype == "F8_E4M3" and self.s_dtype == "F8_E8M0", (self.w_dtype, self.s_dtype)
        self.head_dim = self.w_shape[1]
        self.block = self.head_dim // self.s_shape[1]

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [L, cols] → bf16 [L, cols, head_dim]"""
        flat = ids.reshape(-1).numpy()
        order = np.argsort(flat, kind="stable")  # memmap reads are faster with sorted indices
        w = torch.from_numpy(np.ascontiguousarray(self.w_mm[flat[order]]))
        s = torch.from_numpy(np.ascontiguousarray(self.s_mm[flat[order]]))
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        w, s = w[inv], s[inv]
        vals = fp8_to_float(w).unflatten(-1, (-1, self.block)) * e8m0_to_float(s).unsqueeze(-1)
        return vals.flatten(-2).to(torch.bfloat16).view(*ids.shape, self.head_dim)

    def __call__(self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask=None) -> torch.Tensor:
        """x [L,hc,dim] bf16, hash_ids [L, cols]; token_mask False -> gate 0"""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))  # bf16 [L, dim·(hc+1)]
        key, value = kv.split([self.hc * self.dim, self.dim], dim=-1)
        key = key.to(torch.float32).view(-1, self.hc, self.dim)
        weight = self.q_weight * self.k_weight
        h = x.to(torch.float32)
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.to(torch.float32).unsqueeze(-2)).to(x.dtype)


# ----------------------------------------------------------------------------------------------
# Attention (Compressor, Indexer, Attention)
# ----------------------------------------------------------------------------------------------
@dataclass
class Shared:
    """Reference SharedAttentionRuntime: the source layer writes, later layers read."""

    compress_kv: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    topk_idxs: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


def window_topk_idxs(win: int, seqlen: int, start_pos: int) -> torch.Tensor:
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - win + 1).clamp(0) + torch.arange(min(seqlen, win))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % win + 1
        idxs = torch.cat([torch.arange(oldest, win), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs).unsqueeze(0)
    return idxs.int()


def select_candidate_blocks(logits: torch.Tensor, compress_lens, topk_blocks: int, block_size: int) -> torch.Tensor:
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-float("inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks) == last, float("inf"))
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -float("inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Attention:
    def __init__(self, ck: Ckpt, cfg: dict, layer_id: int, prefix: str, max_seq_len: int, shared: Shared):
        p = prefix
        self.layer_id = layer_id
        self.shared = shared
        self.dim = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.head_dim = cfg["head_dim"]
        self.rd = cfg["qk_rope_head_dim"]
        self.o_groups = cfg["o_groups"]
        self.o_lora_rank = cfg["o_lora_rank"]
        self.win = cfg["sliding_window"]
        self.eps = cfg["rms_norm_eps"]
        is_backbone = layer_id < cfg["num_hidden_layers"]
        self.ratio = cfg["compress_ratios"][layer_id] if is_backbone else 0  # draft stages are window-only (reference asserts compress_ratio == 0)
        self.max_seq_len = max_seq_len
        self.is_kv_source = is_backbone and layer_id in cfg["kv_source_layer_ids"]
        self.is_index_source = is_backbone and layer_id in cfg["index_source_layer_ids"]
        self.softmax_scale = self.head_dim**-0.5

        self.attn_sink = ck.get(p + ".attn_sink").to(torch.float32)
        self.wq_a = Lin.fp8(ck, p + ".wq_a")
        self.q_norm = ck.get(p + ".q_norm.weight")
        self.wq_b = Lin.fp8(ck, p + ".wq_b")
        self.wkv = Lin.fp8(ck, p + ".wkv")
        self.kv_norm = ck.get(p + ".kv_norm.weight")
        # wo_a: convert.py dequantizes it with the 32x32 block scales and stores it as bf16
        self.wo_a = deq_fp8_block(ck.get(p + ".wo_a.weight"), ck.get(p + ".wo_a.scale")).to(torch.bfloat16)
        self.wo_b = Lin.fp8(ck, p + ".wo_b")

        if self.ratio:
            orig, theta = cfg["rope_scaling"]["original_max_position_embeddings"], cfg["compress_rope_theta"]
        else:
            orig, theta = 0, cfg["rope_theta"]
        rs = cfg["rope_scaling"]
        self.freqs_cis = precompute_freqs_cis(self.rd, max_seq_len, orig, theta, rs["factor"], rs["beta_fast"], rs["beta_slow"])

        self.window_kv_cache = torch.zeros(self.win, self.head_dim, dtype=torch.bfloat16)
        if self.is_kv_source:
            cp = p + ".compressor"
            self.comp_norm = ck.get(cp + ".norm.weight")
            if self.ratio > 1:
                self.comp_wkv = Lin.plain(ck, cp + ".wkv", torch.float32)
                self.comp_wgate = Lin.plain(ck, cp + ".wgate", torch.float32)
                self.kv_state = torch.zeros(self.ratio, self.head_dim, dtype=torch.float32)
                self.score_state = torch.full((self.ratio, self.head_dim), -float("inf"), dtype=torch.float32)
            else:
                self.comp_wkv = Lin.plain(ck, cp + ".wkv")
            self.compress_kv_cache = torch.zeros(max_seq_len // self.ratio, self.head_dim, dtype=torch.bfloat16)
        if self.is_index_source:
            ip = p + ".indexer"
            self.idx_owns_k = layer_id in cfg["kv_source_layer_ids"]
            self.idx_is_cand_src = layer_id == cfg["candidate_source_layer_id"]
            self.idx_uses_cand = 0 <= cfg["candidate_source_layer_id"] < layer_id
            self.cand_topk_blocks = cfg["candidate_topk_blocks"]
            self.cand_block = cfg["candidate_block_size"]
            self.idx_n_heads = cfg["index_n_heads"]
            self.idx_head_dim = cfg["index_head_dim"]
            self.idx_topk = cfg["index_topk"]
            self.idx_scale = self.idx_head_dim**-0.5
            self.idx_wq_b = Lin.fp8(ck, ip + ".wq_b")
            self.idx_weights_proj = Lin.plain(ck, ip + ".weights_proj")
            if self.idx_owns_k:
                self.idx_wk = Lin.plain(ck, ip + ".wk")
                self.idx_k_norm = ck.get(ip + ".k_norm.weight")
                self.idx_k_cache = torch.zeros(max_seq_len // self.ratio, self.idx_head_dim, dtype=torch.bfloat16)

    # --- compressor ---------------------------------------------------------------------------
    def compressor(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        seqlen = x.size(0)
        ratio, dtype = self.ratio, x.dtype
        if ratio == 1:
            return rmsnorm(self.comp_wkv(x), self.comp_norm, self.eps)
        xf = x.to(torch.float32)
        kv, score = self.comp_wkv(xf), self.comp_wgate(xf)
        if start_pos == 0:
            should = seqlen >= ratio
            rem = seqlen % ratio
            cut = seqlen - rem
            if rem:
                kv, self.kv_state[:rem] = kv.split([cut, rem], dim=0)
                score, self.score_state[:rem] = score.split([cut, rem], dim=0)
            kv = kv.unflatten(0, (-1, ratio))
            score = score.unflatten(0, (-1, ratio))
            kv = (kv * score.softmax(dim=1)).sum(dim=1)
        else:
            should = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[slot] = kv.squeeze(0)
            self.score_state[slot] = score.squeeze(0)
            if should:
                kv = (self.kv_state * self.score_state.softmax(dim=0)).sum(dim=0, keepdim=True)
        if not should:
            return None
        return rmsnorm(kv.to(dtype), self.comp_norm, self.eps)

    # --- indexer ------------------------------------------------------------------------------
    def indexer(self, x: torch.Tensor, qr: torch.Tensor, latent: torch.Tensor | None, start_pos: int, offset: int, log: dict):
        seqlen = x.size(0)
        ratio, rd, end_pos = self.ratio, self.rd, start_pos + seqlen
        if self.idx_owns_k and latent is not None:
            freqs = self.freqs_cis[: seqlen - seqlen % ratio : ratio] if start_pos == 0 else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            k = rmsnorm(self.idx_wk(latent), self.idx_k_norm, self.eps)
            apply_rotary_emb(k[..., -rd:], freqs)
            fp4_act_quant_inplace(k, 32, False)
            self.idx_k_cache[start_pos // ratio : start_pos // ratio + k.size(0)] = k
            self.shared.index_k = self.idx_k_cache

        q = self.idx_wq_b(qr).unflatten(-1, (self.idx_n_heads, self.idx_head_dim))  # [s,h,d] bf16
        apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        fp4_act_quant_inplace(q, 32, False)

        index_k = self.shared.index_k[: end_pos // ratio]  # [t,d]
        weights = self.idx_weights_proj(x) * (self.idx_scale * self.idx_n_heads**-0.5)  # bf16 [s,h]
        score = torch.einsum("shd,td->sht", q, index_k)  # bf16
        score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=1)  # bf16 [s,t]

        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
            score.masked_fill_(torch.arange(seqlen // ratio) >= compress_lens, -float("inf"))
        else:
            compress_lens = end_pos // ratio

        if self.idx_is_cand_src:
            self.shared.candidates = select_candidate_blocks(score, compress_lens, self.cand_topk_blocks, self.cand_block)
        elif self.idx_uses_cand:
            score = score.masked_fill(~self.shared.candidates, -float("inf"))

        topk = min(self.idx_topk, end_pos // ratio)
        idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        out = torch.where(idxs < compress_lens, idxs + offset, -1).int()
        log.setdefault("index", {})[f"L{self.layer_id:02d}"] = (idxs.tolist(), out.tolist())
        return out

    # --- attention ---------------------------------------------------------------------------
    def _window_kv(self, x, freqs_cis, start_pos):
        seqlen = x.size(0)
        win = self.win
        kv = rmsnorm(self.wkv(x), self.kv_norm, self.eps)
        apply_rotary_emb(kv[..., -self.rd :], freqs_cis)
        act_quant_fp8_inplace(kv, 32)
        if start_pos == 0:
            if seqlen <= win:
                self.window_kv_cache[:seqlen] = kv
            else:
                cut = seqlen % win
                self.window_kv_cache[cut:win], self.window_kv_cache[:cut] = kv[-win:].split([win - cut, cut], dim=0)
            window_kv = kv
        else:
            self.window_kv_cache[start_pos % win] = kv.squeeze(0)
            window_kv = self.window_kv_cache
        return window_kv, window_topk_idxs(win, seqlen, start_pos)

    def _compress_kv(self, x, qr, start_pos, offset, log):
        seqlen = x.size(0)
        ratio = self.ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            self.shared.compress_kv = self.compress_kv_cache
        if self.is_index_source:
            if compress_len == 0:
                idxs = torch.empty(seqlen, 0, dtype=torch.int32)
            else:
                idxs = self.indexer(x, qr, latent, start_pos, offset, log)
            self.shared.topk_idxs = idxs
        else:
            idxs = self.shared.topk_idxs
        if latent is not None:
            freqs = self.freqs_cis[: seqlen - seqlen % ratio : ratio] if start_pos == 0 else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            apply_rotary_emb(latent[..., -self.rd :], freqs)
            fp4_act_quant_inplace(latent, 16, True)
            self.compress_kv_cache[start_pos // ratio : start_pos // ratio + latent.size(0)] = latent
        return self.shared.compress_kv[:compress_len], idxs

    def __call__(self, x: torch.Tensor, start_pos: int, log: dict) -> torch.Tensor:
        seqlen = x.size(0)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        rd = self.rd
        qr = rmsnorm(self.wq_a(x), self.q_norm, self.eps)
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
        if self.ratio:
            ckv, cidx = self._compress_kv(x, qr, start_pos, kv.size(0), log)
            kv = torch.cat([kv, ckv], dim=0)
            topk_idxs = torch.cat([topk_idxs, cidx], dim=-1)

        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)
        o = o.view(seqlen, self.o_groups, -1)
        wo_a = self.wo_a.view(self.o_groups, self.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a)
        return self.wo_b(o.flatten(1))


# ----------------------------------------------------------------------------------------------
# MoE
# ----------------------------------------------------------------------------------------------
class MoE:
    def __init__(self, ck: Ckpt, cfg: dict, layer_id: int, prefix: str, n_routed: int, n_act: int):
        self.ck = ck
        self.p = prefix + ".ffn"
        self.layer_id = layer_id
        self.dim = cfg["hidden_size"]
        self.n_routed, self.topk = n_routed, n_act
        self.route_scale = cfg["routed_scaling_factor"]
        self.limit = cfg["swiglu_limit"]
        assert cfg["scoring_func"] == "sqrtsoftplus" and cfg["norm_topk_prob"] and cfg["topk_method"] == "noaux_tc"
        self.gate_w = ck.get(self.p + ".gate.weight").to(torch.float32)
        self.gate_b = ck.get(self.p + ".gate.bias").to(torch.float32)
        self.gate_b_vl = ck.get(self.p + ".gate.bias_vl").to(torch.float32) if ck.has(self.p + ".gate.bias_vl") else None
        sp = self.p + ".shared_experts"
        self.sh = (Lin.fp8(ck, sp + ".w1"), Lin.fp8(ck, sp + ".w2"), Lin.fp8(ck, sp + ".w3"))
        self._expert_cache: dict[int, tuple] = {}

    def expert_weights(self, i: int):
        if i not in self._expert_cache:
            ep = f"{self.p}.experts.{i}"
            self._expert_cache[i] = (Lin.fp4(self.ck, ep + ".w1"), Lin.fp4(self.ck, ep + ".w2"), Lin.fp4(self.ck, ep + ".w3"))
            if len(self._expert_cache) > 64:  # memory cap (one fp32 expert = 141MB)
                self._expert_cache.pop(next(iter(self._expert_cache)))
        return self._expert_cache[i]

    @staticmethod
    def expert(w1, w2, w3, x: torch.Tensor, weights: torch.Tensor | None, limit: float) -> torch.Tensor:
        dtype = x.dtype
        gate = w1(x).to(torch.float32)
        up = w3(x).to(torch.float32)
        if limit > 0:
            up = torch.clamp(up, min=-limit, max=limit)
            gate = torch.clamp(gate, max=limit)
        y = F.silu(gate) * up
        if weights is not None:
            y = weights * y
        return w2(y.to(dtype))

    def gate(self, x: torch.Tensor, image_mask=None):
        scores = x.to(torch.float32) @ self.gate_w.T
        scores = F.softplus(scores).sqrt()
        bias = self.gate_b
        if image_mask is not None and self.gate_b_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.gate_b_vl, bias)
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices

    def __call__(self, x: torch.Tensor, log: dict, image_mask=None) -> torch.Tensor:
        weights, indices = self.gate(x, image_mask)
        log.setdefault("routing", {})[f"L{self.layer_id:02d}"] = {"ids": indices.tolist(), "w": weights.tolist()}
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed)
        for i in torch.nonzero(counts).flatten().tolist():
            idx, top = torch.where(indices == i)
            w1, w2, w3 = self.expert_weights(i)
            y[idx] += self.expert(w1, w2, w3, x[idx], weights[idx, top, None], self.limit)
        y += self.expert(*self.sh, x, None, self.limit)
        return y.to(x.dtype)


# ----------------------------------------------------------------------------------------------
# Block · Transformer
# ----------------------------------------------------------------------------------------------
class Block:
    def __init__(self, ck: Ckpt, cfg: dict, layer_id: int, max_seq_len: int, shared: Shared, prefix: str | None = None,
                 n_routed: int | None = None, n_act: int | None = None, attn_cls=None):
        p = prefix if prefix is not None else f"layers.{layer_id}"
        self.layer_id = layer_id
        self.eps = cfg["rms_norm_eps"]
        self.hc = cfg["hc_mult"]
        self.sinkhorn_iters = cfg["hc_sinkhorn_iters"]
        self.hc_eps = cfg["hc_eps"]
        self.attn = (attn_cls or Attention)(ck, cfg, layer_id, p + ".attn", max_seq_len, shared)
        self.ffn = MoE(ck, cfg, layer_id, p, n_routed if n_routed is not None else cfg["n_routed_experts"],
                       n_act if n_act is not None else cfg["num_experts_per_tok"])
        self.attn_norm = ck.get(p + ".attn_norm.weight")
        self.ffn_norm = ck.get(p + ".ffn_norm.weight")
        self.hc_attn = tuple(ck.get(f"{p}.hc_attn_{k}").to(torch.float32) for k in ("fn", "scale", "base"))
        self.hc_ffn = tuple(ck.get(f"{p}.hc_ffn_{k}").to(torch.float32) for k in ("fn", "scale", "base"))

    def hc_mixes(self, x: torch.Tensor, hc):
        fn, scale, base = hc
        xf = x.flatten(1).to(torch.float32)
        rsq = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(xf, fn) * rsq
        return hc_split_sinkhorn(mixes, scale, base, self.hc, self.sinkhorn_iters, self.hc_eps)

    @staticmethod
    def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return torch.sum(pre.unsqueeze(-1) * x.to(torch.float32), dim=1).to(x.dtype)

    @staticmethod
    def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1)
        return y.to(x.dtype)

    def __call__(self, x: torch.Tensor, start_pos: int, pre_mix: torch.Tensor, log: dict, image_mask=None):
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn)
        h = rmsnorm(self.hc_pre(x, pre_mix), self.attn_norm, self.eps)
        h = self.attn(h, start_pos, log)
        x = self.hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn)
        h = rmsnorm(self.hc_pre(x, attn_pre), self.ffn_norm, self.eps)
        h = self.ffn(h, log, image_mask)
        x = self.hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre


# ----------------------------------------------------------------------------------------------
# DSpark (MTP): reference DSparkAttention / DSparkBlock / Transformer.forward_spec
# ----------------------------------------------------------------------------------------------
class DSparkAttention(Attention):
    """The window KV comes from the main model's target-layer hidden state (main_x), not from its own tokens. Block rows see the whole ring + the whole block (non-causal)."""

    def _main_kv(self, main_x: torch.Tensor, pos0: int) -> torch.Tensor:
        n = main_x.size(0)
        kv = rmsnorm(self.wkv(main_x), self.kv_norm, self.eps)
        apply_rotary_emb(kv[..., -self.rd :], self.freqs_cis[pos0 : pos0 + n])
        act_quant_fp8_inplace(kv, 32)
        return kv

    def seed(self, main_x: torch.Tensor):
        """Prefill (start_pos 0): writes main_kv of the last win positions into the ring (reference DSparkAttention.forward start_pos==0 branch)"""
        kv = self._main_kv(main_x, 0)
        n, win = kv.size(0), self.win
        if n <= win:
            self.window_kv_cache[:n] = kv
        else:
            cut = n % win
            self.window_kv_cache[cut:win], self.window_kv_cache[:cut] = kv[-win:].split([win - cut, cut], dim=0)

    def write_main(self, main_x: torch.Tensor, pos0: int):
        """Decode: writes main_kv of positions pos0.. into slot pos % win (the reference does one position at a time; generalized to several rows)"""
        kv = self._main_kv(main_x, pos0)
        for i in range(kv.size(0)):
            self.window_kv_cache[(pos0 + i) % self.win] = kv[i]

    def __call__(self, x: torch.Tensor, start_pos: int, log: dict) -> torch.Tensor:
        """x [B, dim] = block rows (positions start_pos..start_pos+B-1). Does not write to the ring."""
        B = x.size(0)
        rd, win = self.rd, self.win
        freqs_cis = self.freqs_cis[start_pos : start_pos + B]
        qr = rmsnorm(self.wq_a(x), self.q_norm, self.eps)
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        kv = rmsnorm(self.wkv(x), self.kv_norm, self.eps)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        act_quant_fp8_inplace(kv, 32)
        filled = min(win, start_pos)  # reference get_dspark_topk_idxs(start_pos=hidden position): min(win, hidden_pos + 1)
        idxs = torch.cat([torch.arange(filled), win + torch.arange(B)]).int().unsqueeze(0).expand(B, -1).contiguous()
        kv_all = torch.cat([self.window_kv_cache, kv], dim=0)
        o = sparse_attn(q, kv_all, self.attn_sink, idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)
        o = o.view(B, self.o_groups, -1)
        wo_a = self.wo_a.view(self.o_groups, self.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a)
        log.setdefault("dspark_idxs", {})[f"L{self.layer_id}"] = idxs[0].tolist()
        return self.wo_b(o.flatten(1))


class DSparkStage(Block):
    """Checkpoint mtp.{s}: main-layer skeleton (128 experts, top-3) + stage-0 main_proj/main_norm + last-stage norm/markov/confidence."""

    def __init__(self, ck: Ckpt, cfg: dict, stage: int, max_seq_len: int, shared: Shared):
        p = f"mtp.{stage}"
        super().__init__(ck, cfg, cfg["num_hidden_layers"] + stage, max_seq_len, shared, prefix=p,
                         n_routed=cfg["dspark_n_routed_experts"], n_act=cfg["dspark_num_experts_per_tok"], attn_cls=DSparkAttention)
        self.stage = stage
        self.main_proj = Lin.fp8(ck, p + ".main_proj") if ck.has(p + ".main_proj.weight") else None
        self.main_norm = ck.get(p + ".main_norm.weight") if ck.has(p + ".main_norm.weight") else None
        self.norm = ck.get(p + ".norm.weight") if ck.has(p + ".norm.weight") else None
        if ck.has(p + ".markov_head.embed.weight"):
            self.markov_embed = ck.get(p + ".markov_head.embed.weight")  # bf16 [V, rank]
            self.markov_head = ck.get(p + ".markov_head.head.weight").to(torch.float32)  # reference ParallelHead: fp32 parameter
            self.conf_proj = ck.get(p + ".confidence_head.proj.weight").to(torch.float32)  # [1, dim+rank]
        else:
            self.markov_embed = self.markov_head = self.conf_proj = None


class VisionRef:
    """Loads the ViT + Aligner of the reference vision.py as-is (checkpoint keys vision.* / aligner.*)."""

    def __init__(self, ck: Ckpt, ckpt_path: str, cfg_full: dict):
        import importlib, types as _t
        sys.path.insert(0, os.path.join(ckpt_path, "inference"))
        vision = importlib.import_module("vision")
        v = cfg_full["vision_config"]
        args = _t.SimpleNamespace(
            vision_dim=v["hidden_size"], vision_n_heads=v["num_attention_heads"], vision_inter_dim=v["intermediate_size"],
            vision_patch_size=v["patch_size"], vision_rope_theta=v["rope_theta"], vision_n_layers=v["num_hidden_layers"],
            vision_downsample_ratio=v["downsample_ratio"], dim=cfg_full["text_config"]["hidden_size"])
        with torch.device("cpu"):
            torch.set_default_dtype(torch.bfloat16)
            self.vit = vision.ViT(args)
            self.aligner = vision.Aligner(args)
            torch.set_default_dtype(torch.float32)
        sd_v = {k[len("vision."):]: ck.get(k) for k in ck.weight_map if k.startswith("vision.")}
        sd_a = {k[len("aligner."):]: ck.get(k) for k in ck.weight_map if k.startswith("aligner.")}
        self.vit.load_state_dict(sd_v, strict=True)
        self.aligner.load_state_dict(sd_a, strict=True)
        self.vit.eval(); self.aligner.eval()
        self.image_start = ck.get("image_start"); self.image_end = ck.get("image_end"); self.image_newline = ck.get("image_newline")

    @torch.inference_mode()
    def encode(self, patches, n_h, n_w, dump: dict | None = None, tag: str = "", fp32: bool = False):
        """Unrolls the reference ViT.forward per block and dumps each stage. With fp32=True the module is promoted to fp32 to build a 'true' baseline."""
        import importlib
        vision = importlib.import_module("vision")
        vit, al = self.vit, self.aligner
        if fp32:
            import copy
            vit = copy.deepcopy(self.vit).float(); al = copy.deepcopy(self.aligner).float()
        x = vit.patch_embed(patches.to(torch.float32 if fp32 else torch.bfloat16))
        if dump is not None: dump[f"vit{tag}_patch"] = x.to(torch.float32).numpy()
        cos, sin = vision.get_vision_cos_sin(n_h, n_w, vit.rope_dim, vit.rope_theta)
        for i, blk in enumerate(vit.blocks):
            x = blk(x, cos, sin)
            if dump is not None and i in (0, 1, 7, 15, 31): dump[f"vit{tag}_blk{i}"] = x.to(torch.float32).numpy()
        x = vit.norm(x)
        if dump is not None: dump[f"vit{tag}_norm"] = x.to(torch.float32).numpy()
        out = al(x, n_h, n_w)
        if dump is not None: dump[f"vit{tag}_out"] = out.to(torch.float32).numpy()
        return out


class Oracle:
    def __init__(self, ckpt_path: str, max_seq_len: int, max_layer: int | None, tok, quiet=False):
        self.ck = Ckpt(ckpt_path)
        self.ckpt_path = ckpt_path
        self.cfg_full = json.load(open(os.path.join(ckpt_path, "config.json")))
        self.vision = None
        cfg = json.load(open(os.path.join(ckpt_path, "config.json")))
        self.cfg = cfg = cfg.get("text_config", cfg)
        self.n_layers = cfg["num_hidden_layers"] if max_layer is None else min(max_layer + 1, cfg["num_hidden_layers"])
        self.hc = cfg["hc_mult"]
        self.max_seq_len = max_seq_len
        self.shared = Shared()
        self.engram_hash = EngramHash(cfg, tok, max_seq_len) if cfg["engram_layer_ids"] else None
        self.embed = self.ck.get("embed.weight")
        # head/norm are read only if their shard exists, so layer validation works on a partial (still downloading) checkpoint
        try:
            self.norm = self.ck.get("norm.weight")
            self.head = self.ck.get("head.weight").to(torch.float32)
        except FileNotFoundError:
            self.norm = self.head = None
        self.layers: list[Block] = []
        self.engrams: dict[int, EngramLayer] = {}
        self.quiet = quiet
        # DSpark
        self.targets = list(cfg.get("dspark_target_layer_ids", []))
        self.block = cfg.get("dspark_block_size", 0)
        self.noise_token = cfg.get("dspark_noise_token_id", 0)
        self.stages: list[DSparkStage] = []
        self.last_main_hidden: torch.Tensor | None = None  # target-layer hidden state of the last forward [L, n_targets·dim] bf16

    def _stages(self) -> list[DSparkStage]:
        if not self.stages and self.block:
            for s in range(64):
                if not self.ck.has(f"mtp.{s}.attn_norm.weight"):
                    break
                t0 = time.time()
                self.stages.append(DSparkStage(self.ck, self.cfg, s, self.max_seq_len, self.shared))
                if not self.quiet:
                    print(f"  [load] dspark stage {s} {time.time() - t0:.1f}s", flush=True)
        return self.stages

    def _layer(self, i: int) -> Block:
        while len(self.layers) <= i:
            t0 = time.time()
            self.layers.append(Block(self.ck, self.cfg, len(self.layers), self.max_seq_len, self.shared))
            lid = len(self.layers) - 1
            if lid in self.cfg["engram_layer_ids"]:
                self.engrams[lid] = EngramLayer(self.ck, self.cfg, lid, self.cfg["engram_layer_ids"].index(lid))
            if not self.quiet:
                print(f"  [load] layer {lid} {time.time() - t0:.1f}s", flush=True)
        return self.layers[i]

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, start_pos: int, dump: dict | None = None, token_types=None, images=None):
        """ids [L] -> (logits [L,V] fp32 or None, log). token_types [L] (-1 = text); images: list of image_processor.ImageInput"""
        log: dict = {}
        image_mask = None if token_types is None else (token_types >= 0)
        engram_mask = None if image_mask is None else ~image_mask
        hashes = self.engram_hash(ids, start_pos, engram_mask) if self.engram_hash is not None else None
        h = self.embed[ids].clone()  # bf16 [L,dim]
        if images:
            if self.vision is None:
                self.vision = VisionRef(self.ck, self.ckpt_path, self.cfg_full)
            for img in images:
                types = img.types
                span = h[img.start : img.start + types.numel()]
                span[types == 0] = self.vision.image_start
                span[types == 3] = self.vision.image_end
                span[types == 2] = self.vision.image_newline
                emb = self.vision.encode(img.patches, img.n_vit_h, img.n_vit_w, dump, f"_{img.start}")
                span[types == 1] = emb.to(h.dtype)
                if dump is not None:
                    dump[f"image_emb_{img.start}"] = emb.to(torch.float32).numpy()
                    self.vision.encode(img.patches, img.n_vit_h, img.n_vit_w, dump, f"32_{img.start}", fp32=True)
        h = h.unsqueeze(1).repeat(1, self.hc, 1)
        pre_mix = torch.zeros(h.size(0), self.hc, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        main_hiddens = []
        for i in range(self.n_layers):
            t0 = time.time()
            layer = self._layer(i)
            if i in self.engrams:
                h = self.engrams[i](h, hashes[:, self.engrams[i].hash_index, :], engram_mask)
                if dump is not None:
                    dump[f"engram_L{i:02d}"] = h.to(torch.float32).numpy()
            if i in self.targets:  # the MTP head reads the target layers' attention input (hc mean), as in reference Transformer.forward
                main_hiddens.append(h.mean(dim=1))
            h, pre_mix = layer(h, start_pos, pre_mix, log, image_mask)
            if dump is not None:
                dump[f"h_L{i:02d}"] = h.to(torch.float32).numpy()
                dump[f"premix_L{i:02d}"] = pre_mix.numpy()
            if not self.quiet:
                print(f"  [fwd] layer {i} {time.time() - t0:.1f}s", flush=True)
        self.last_main_hidden = torch.cat(main_hiddens, dim=-1) if len(main_hiddens) == len(self.targets) and main_hiddens else None
        if self.head is None or self.n_layers < self.cfg["num_hidden_layers"]:
            return None, log
        x = Block.hc_pre(h, pre_mix)
        logits = rmsnorm(x, self.norm, self.cfg["rms_norm_eps"]).to(torch.float32) @ self.head.T
        return logits, log

    @torch.inference_mode()
    def forward_spec(self, tok: int, main_hidden: torch.Tensor, hidden_pos0: int, seed: bool, dump: dict | None = None):
        """Reference Transformer.forward_spec. main_hidden [n, n_targets·dim] = target-layer hidden states of positions hidden_pos0...
        seed=True (prefill): fill each stage's ring and return. Otherwise write n rows into the ring, then draft a block starting from tok (position hidden_pos0+n) -> (ids[B+1], logits[B,V], conf[B])."""
        stages = self._stages()
        s0 = stages[0]
        main_x = rmsnorm(s0.main_proj(main_hidden), s0.main_norm, self.cfg["rms_norm_eps"])
        if dump is not None:
            dump["spec_main_hidden"] = main_hidden.to(torch.float32).numpy()
            dump["spec_main_x"] = main_x.to(torch.float32).numpy()
        for st in stages:
            if seed:
                st.attn.seed(main_x)
            else:
                st.attn.write_main(main_x, hidden_pos0)
        if seed:
            return None
        log: dict = {}
        B = self.block
        pos0 = hidden_pos0 + main_hidden.size(0)
        ids = torch.full((B,), self.noise_token, dtype=torch.long)
        ids[0] = tok
        h = self.embed[ids].clone().unsqueeze(1).repeat(1, self.hc, 1)
        pre_mix = torch.zeros(B, self.hc, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        for st in stages:
            h, pre_mix = st(h, pos0, pre_mix, log, None)
            if dump is not None:
                dump[f"spec_h_M{st.stage}"] = h.to(torch.float32).numpy()
                dump[f"spec_premix_M{st.stage}"] = pre_mix.numpy()
        last = stages[-1]
        x = Block.hc_pre(h, pre_mix)  # [B, dim] bf16
        logits = rmsnorm(x, last.norm, self.cfg["rms_norm_eps"]).to(torch.float32) @ self.head.T  # [B, V] fp32
        out_ids = torch.empty(B + 1, dtype=torch.long)
        out_ids[0] = tok
        embeds = []
        for i in range(B):
            e = last.markov_embed[out_ids[i]]  # bf16 [rank]
            logits[i] += F.linear(e.float(), last.markov_head)
            embeds.append(e)
            out_ids[i + 1] = int(logits[i].argmax())  # greedy draft (same as the engine)
        emb = torch.stack(embeds, dim=0)
        conf = F.linear(torch.cat([x, emb], dim=-1).float(), last.conf_proj).squeeze(-1)  # [B] fp32
        if dump is not None:
            dump["spec_x"] = x.to(torch.float32).numpy()
            dump["spec_markov"] = emb.to(torch.float32).numpy()
            dump["spec_logits"] = logits.numpy()
            dump["spec_ids"] = out_ids.numpy()
            dump["spec_conf"] = conf.numpy()
        return out_ids, logits, conf, log


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("HIVE_CKPT"), help="checkpoint directory (default: $HIVE_CKPT)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--ids", default=None, help="comma-separated token ids (instead of --prompt)")
    ap.add_argument("--ids-file", help="Long fixture IDs, comma/whitespace separated; avoids exec argument limit")
    ap.add_argument("--bos", action="store_true", help="prepend bos (0)")
    ap.add_argument("--max-layer", type=int, default=None, help="run only up to this layer (for validating a partial checkpoint)")
    ap.add_argument("--decode", type=int, default=0, help="number of greedy decode steps")
    ap.add_argument("--spec", action="store_true", help="DSpark: prefill seeds the ring → 1 decode step → draft block golden at that position (spec.npz/spec.json)")
    ap.add_argument("--continue-ids", default=None, help="after the prefill, process these tokens one at a time (checks the decode path without the head)")
    ap.add_argument("--image", action="append", default=[], help="image file (prepended in reference form instead of an <image> placeholder in the prompt)")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if not a.ckpt:
        ap.error("--ckpt DIR (or HIVE_CKPT) is required")

    torch.set_num_threads(a.threads)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(a.ckpt, "tokenizer.json"))
    token_types = None
    images = None
    if a.image:
        # build the prompt with the reference encoding + image_processor: user message = images + text, chat mode
        sys.path.insert(0, os.path.join(a.ckpt, "encoding")); sys.path.insert(0, os.path.join(a.ckpt, "inference"))
        import importlib, types as _t
        enc = importlib.import_module("encoding"); ip = importlib.import_module("image_processor")
        from transformers import AutoTokenizer
        htok = AutoTokenizer.from_pretrained(a.ckpt)
        content = [{"type": "image_url", "image_url": {"url": pth}} for pth in a.image] + [{"type": "text", "text": a.prompt}]
        prompt, media = enc.encode_messages([{"role": "user", "content": content}], thinking_mode="chat", return_multi_modal_data=True)
        v = json.load(open(os.path.join(a.ckpt, "config.json")))["vision_config"]
        vargs = _t.SimpleNamespace(vision_patch_size=v["patch_size"], vision_downsample_ratio=v["downsample_ratio"],
                                   vision_max_n_token=v["max_image_tokens"], vision_min_pixels=v["min_pixels"], vision_max_wh_ratio=v.get("max_wh_ratio"),
                                   image_token_id=json.load(open(os.path.join(a.ckpt, "config.json")))["image_token_id"], vision_enabled=True)
        ids, types, images = ip.prepare_vl_inputs(prompt, media["images"], htok, vargs)
        token_types = torch.tensor(types)
        # save patches and metadata for engine tests
        os.makedirs(a.out, exist_ok=True)
        meta = []
        for i, img in enumerate(images or []):
            img.patches.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tofile(os.path.join(a.out, f"patches_{i:03d}.bf16"))
            meta.append({"start": img.start, "n_vit_h": img.n_vit_h, "n_vit_w": img.n_vit_w, "types": img.types.tolist(), "file": f"patches_{i:03d}.bf16"})
        json.dump({"ids": ids, "types": types, "images": meta}, open(os.path.join(a.out, "image_inputs.json"), "w"))
        print(f"[image] prompt tokens {len(ids)} images {len(meta)}", flush=True)
    elif a.ids_file:
        import re
        with open(a.ids_file) as f: ids = [int(t) for t in re.split(r'[\s,]+',f.read().strip()) if t]
    elif a.ids:
        ids = [int(t) for t in a.ids.split(",")]
    else:
        ids = tok.encode(a.prompt, add_special_tokens=False).ids
    if a.bos and not a.image:
        ids = [0] + ids
    os.makedirs(a.out, exist_ok=True)
    print(f"ids({len(ids)}) = {ids}", flush=True)

    t0 = time.time()
    orc = Oracle(a.ckpt, a.max_seq_len, a.max_layer, tok, quiet=a.quiet)
    print(f"[init] {time.time() - t0:.1f}s (engram hash + embed)", flush=True)

    dump: dict = {}
    logits, log = orc.forward(torch.tensor(ids), 0, dump, token_types, images)
    np.savez(os.path.join(a.out, "prefill.npz"), **dump)
    json.dump({"ids": ids, **log}, open(os.path.join(a.out, "prefill.json"), "w"))
    if a.continue_ids:
        cont = [int(t) for t in a.continue_ids.split(",")]
        for i, t in enumerate(cont):
            d: dict = {}
            _, clog = orc.forward(torch.tensor([t]), len(ids) + i, d)
            np.savez(os.path.join(a.out, f"decode_{i:03d}.npz"), **d)
            json.dump({"ids": [t], **clog}, open(os.path.join(a.out, f"decode_{i:03d}.json"), "w"))
            print(f"[continue {i}] pos={len(ids) + i} id={t}", flush=True)
    if logits is not None and a.spec:
        nxt = int(logits[-1].argmax())
        assert orc.last_main_hidden is not None, "no hidden state of the target layer (partial checkpoint?)"
        orc.forward_spec(nxt, orc.last_main_hidden, 0, seed=True)
        d: dict = {}
        lg, dlog = orc.forward(torch.tensor([nxt]), len(ids), d)
        np.savez(os.path.join(a.out, "decode_000.npz"), logits=lg.numpy(), **d)
        json.dump(dlog, open(os.path.join(a.out, "decode_000.json"), "w"))
        nxt2 = int(lg[-1].argmax())
        sd: dict = {}
        out_ids, slog, sconf, slog_ = orc.forward_spec(nxt2, orc.last_main_hidden, len(ids), seed=False, dump=sd)
        np.savez(os.path.join(a.out, "spec.npz"), **sd)
        json.dump({"tok": nxt2, "pos": len(ids) + 1, "ids": out_ids.tolist(), "conf": sconf.tolist(), **slog_}, open(os.path.join(a.out, "spec.json"), "w"))
        print(f"[spec] pos={len(ids) + 1} tok={nxt2} drafts={out_ids[1:].tolist()} conf={[round(c, 3) for c in sconf.tolist()]} "
              f"text={tok.decode(out_ids.tolist())!r}", flush=True)
        json.dump({"prompt_ids": ids, "generated": [nxt, nxt2], "text": tok.decode([nxt, nxt2])}, open(os.path.join(a.out, "decode.json"), "w"))
        print(f"[done] {time.time() - t0:.1f}s → {a.out}", flush=True)
        return
    if logits is not None:
        np.save(os.path.join(a.out, "logits.npy"), logits.numpy())
        nxt = int(logits[-1].argmax())
        print(f"[prefill] next = {nxt} {tok.decode([nxt])!r}", flush=True)
        out_ids = []
        for step in range(a.decode):
            pos = len(ids) + step
            out_ids.append(nxt)
            d: dict = {}
            lg, dlog = orc.forward(torch.tensor([nxt]), pos, d)
            np.savez(os.path.join(a.out, f"decode_{step:03d}.npz"), logits=lg.numpy(), **d)
            json.dump(dlog, open(os.path.join(a.out, f"decode_{step:03d}.json"), "w"))
            nxt = int(lg[-1].argmax())
            print(f"[decode {step}] pos={pos} next = {nxt} {tok.decode([nxt])!r}", flush=True)
        json.dump({"prompt_ids": ids, "generated": out_ids + [nxt], "text": tok.decode(out_ids + [nxt])}, open(os.path.join(a.out, "decode.json"), "w"))
    print(f"[done] {time.time() - t0:.1f}s → {a.out}", flush=True)


if __name__ == "__main__":
    main()
