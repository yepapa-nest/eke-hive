#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""CPU reference dumps for GLM-5.3-Flash (hive family "glm"), for engine/tests/glm_cpu_check.cpp and tools/compare_glm.py.

The reference numbers come from the transformers 5.18 GLM-5 Next modules (Glm5NextTextDecoderLayer, Glm5NextTextAttention,
Glm5NextTextMoE, Glm5NextVisionModel), run as a library in float32 on the CPU, one decoder layer at a time, with the checkpoint's
weights dequantized to float32: NVFP4 (e2m1 · e4m3 group scale · fp32 global scale), BF16, and — with --fp8 — the Z.ai FP8 128×128
tensors where the engine uses them (DSA q_a / q_b / kv_a / o, shared experts, the NextN layer's routed experts). This file only loads
weights into those modules, drives them and writes the outputs; it contains no model code of its own except the NextN (MTP) glue, which
follows the hive GPU contract (engine/src/glm/glm_mtp.cpp): entry j = eh_proj([enorm(embed(x_{j+1})) | hnorm(h_j)]) (embedding zeroed for
j = 0), one plain-residual layer (DSA attention of layer 45, MoE of layer 45), shared_head.norm, the main lm_head.

Outputs (float32 .npy) in --out:
  h_LNN.npy     [T, 4·4096] hyper-connection streams after layer NN (rows of the prompt)
  logits.npy    [1, vocab] logits of the last prompt row (only when the run reaches the last layer)
  fh.npy        [T, 4096] final normed hidden of every prompt row
  teacher.npy   [G + 1, vocab] with --teacher: logits of the last prompt row and of every appended token (teacher forcing)
  mtp_logits.npy[1, vocab] with --mtp-next TOK: NextN draft logits of entry T−1 = (h_{T−1}, TOK)
  vision.npy    [N/4, 4096] with --vision: encoder output of the patches in --vision-patches (also used for the prompt rows given by
                --image-at)
Runs layer-wise; memory stays around one layer's float32 weights (the routed experts of a layer are dequantized only when used).

  python3 oracle/glm5_oracle.py --ckpt /models/GLM-5.3-Flash-NVFP4 --fp8 /models/GLM-5.3-Flash-zai-fp8 --ids 1,2,3 --out /tmp/o
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

P = "model.language_model."
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


class Ckpt:
    """name → tensor through model.safetensors.index.json (shards opened on demand)."""

    def __init__(self, d: str):
        self.dir = d
        with open(os.path.join(d, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self.open = {}

    def has(self, n: str) -> bool:
        return n in self.map

    def get(self, n: str) -> torch.Tensor:
        fn = self.map[n]
        if fn not in self.open:
            self.open[fn] = safe_open(os.path.join(self.dir, fn), framework="pt")
        return self.open[fn].get_tensor(n)


def nvfp4(ck: Ckpt, base: str) -> torch.Tensor:
    w = ck.get(base + ".weight")            # uint8 [N, K/2], low nibble = even column
    s = ck.get(base + ".weight_scale")      # e4m3 [N, K/16]
    g = ck.get(base + ".weight_scale_2").float().reshape(())
    N, K2 = w.shape
    lo, hi = (w & 0x0F).long(), (w >> 4).long()
    v = torch.stack([E2M1[lo], E2M1[hi]], dim=-1).reshape(N, K2 * 2)
    return v * s.float().repeat_interleave(16, dim=1) * g


def fp8b(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    N, K = t.shape
    sf = s.float().repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
    return t.float() * sf


class Weights:
    def __init__(self, ckpt: str, fp8: str | None):
        self.ck = Ckpt(ckpt)
        self.f8 = Ckpt(fp8) if fp8 else None

    def dense(self, n: str) -> torch.Tensor:
        """BF16 / F32 tensor as float32, or the Z.ai FP8 copy when --fp8 holds one (the engine's precision rule)."""
        if self.f8 is not None and self.f8.has(n) and self.f8.has(n + "_scale_inv"):
            return fp8b(self.f8.get(n), self.f8.get(n + "_scale_inv"))
        return self.ck.get(n).float()

    def get(self, n: str) -> torch.Tensor:
        return self.ck.get(n).float()


def build_layer(tc, W: Weights, l: int, kind: str):
    """A transformers decoder layer (or the NextN pieces) with float32 weights from the checkpoint."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer
    with torch.device("meta"):
        layer = Glm5NextTextDecoderLayer(tc, l if l < tc.num_hidden_layers else 43)
    layer.to_empty(device="cpu")
    p = P + f"layers.{l}."
    sd = {}
    for which, mod in (("attn", "attn_hc"), ("ffn", "ffn_hc")):
        if W.ck.has(p + f"hc_{which}_fn"):
            sd[f"{mod}.fn"] = W.get(p + f"hc_{which}_fn")
            sd[f"{mod}.base"] = W.get(p + f"hc_{which}_base")
            sd[f"{mod}.scale"] = W.get(p + f"hc_{which}_scale")
    sd["input_layernorm.weight"] = W.get(p + "input_layernorm.weight")
    sd["post_attention_layernorm.weight"] = W.get(p + "post_attention_layernorm.weight")
    a = p + "self_attn."
    if layer.block_type == "linear_attention" and l < tc.num_hidden_layers:
        for n in ("q_proj", "k_proj", "v_proj", "b_proj", "g_a_proj", "g_b_proj", "o_proj"):
            sd[f"self_attn.{n}.weight"] = W.get(a + n + ".weight")
        sd["self_attn.forget_gate.f_a_proj.weight"] = W.get(a + "f_a_proj.weight")
        sd["self_attn.forget_gate.f_b_proj.weight"] = W.get(a + "f_b_proj.weight")
        sd["self_attn.forget_gate.dt_bias"] = W.get(a + "dt_bias")
        sd["self_attn.forget_gate.A_log"] = W.get(a + "A_log")
        sd["self_attn.o_norm.weight"] = W.get(a + "o_norm.weight")
        sd["self_attn.conv1d.weight"] = torch.cat([W.get(a + f"{x}_conv1d.weight") for x in ("q", "k", "v")], 0)
    else:
        for n in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
            sd[f"self_attn.{n}.weight"] = W.dense(a + n + ".weight")
        sd["self_attn.kv_b_proj.weight"] = W.get(a + "kv_b_proj.weight")
        sd["self_attn.q_a_layernorm.weight"] = W.get(a + "q_a_layernorm.weight")
        sd["self_attn.kv_a_layernorm.weight"] = W.get(a + "kv_a_layernorm.weight")
        i = a + "indexer."
        for n in ("wq_b.weight", "wk.weight", "weights_proj.weight", "k_norm.weight", "k_norm.bias", "index_kpool_compress_ape",
                  "index_kpool_compress_gate"):
            sd[f"self_attn.indexer.{n}"] = W.get(i + n)
    m = p + "mlp."
    lazy = None
    if l < tc.num_hidden_layers and not tc.mlp_layer_types[l] == "sparse":
        for n in ("gate_proj", "up_proj", "down_proj"):
            sd[f"mlp.{n}.weight"] = nvfp4(W.ck, m + n)
    else:
        sd["mlp.gate.weight"] = W.get(m + "gate.weight")
        sd["mlp.gate.e_score_correction_bias"] = W.get(m + "gate.e_score_correction_bias")
        for n in ("gate_proj", "up_proj", "down_proj"):
            sd[f"mlp.shared_experts.{n}.weight"] = W.dense(m + f"shared_experts.{n}.weight")
        lazy = m + "experts."
    missing = []
    with torch.no_grad():
        names = dict(layer.named_parameters())
        names.update(dict(layer.named_buffers()))
        for k, t in names.items():
            if k in sd:
                t.copy_(sd[k].reshape(t.shape))
            elif not k.startswith("mlp.experts.") and not (l >= tc.num_hidden_layers and k.split(".")[0] in ("attn_hc", "ffn_hc")):
                missing.append(k)
    if missing:
        raise RuntimeError(f"layer {l}: no checkpoint tensor for {missing}")
    if lazy is not None:
        attach_lazy_experts(layer.mlp.experts, W, lazy, l >= tc.num_hidden_layers)
    return layer


def attach_lazy_experts(ex, W: Weights, prefix: str, mtp: bool):
    """Dequantize a routed expert into the module's stacked parameters the first time the router selects it."""
    done = set()
    orig = ex.forward

    def load(e: int):
        b = prefix + f"{e}."
        if mtp:  # the NextN layer's experts: Z.ai FP8 copy (the engine has no other)
            g = fp8b(W.f8.get(b + "gate_proj.weight"), W.f8.get(b + "gate_proj.weight_scale_inv"))
            u = fp8b(W.f8.get(b + "up_proj.weight"), W.f8.get(b + "up_proj.weight_scale_inv"))
            d = fp8b(W.f8.get(b + "down_proj.weight"), W.f8.get(b + "down_proj.weight_scale_inv"))
        else:
            g, u, d = nvfp4(W.ck, b + "gate_proj"), nvfp4(W.ck, b + "up_proj"), nvfp4(W.ck, b + "down_proj")
        with torch.no_grad():
            ex.gate_up_proj[e].copy_(torch.cat([g, u], 0))
            ex.down_proj[e].copy_(d)

    def fwd(hidden_states, top_k_index, top_k_weights):
        for e in sorted(set(top_k_index.flatten().tolist())):
            if e not in done:
                load(e)
                done.add(e)
        return orig(hidden_states, top_k_index, top_k_weights)

    ex.forward = fwd


def save(out: str, name: str, a) -> None:
    np.save(os.path.join(out, name), np.ascontiguousarray(np.asarray(a, dtype=np.float32)))


def parse_ids(s: str) -> list[int]:
    if os.path.exists(s):
        s = open(s).read()
    return [int(x) for x in s.replace("[", " ").replace("]", " ").replace(",", " ").split()]


def vision_encode(cfg, W: Weights, patches: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel
    vc = cfg.vision_config
    vc._attn_implementation = "eager"
    with torch.device("meta"):
        vm = Glm5NextVisionModel(vc)
    vm.to_empty(device="cpu")
    vm.rotary_pos_emb.inv_freq = vm.rotary_pos_emb.compute_axial_rope_parameters(vc)[0]
    with torch.no_grad():
        for k, t in vm.named_parameters():
            t.copy_(W.get("model.visual." + k).reshape(t.shape))
    vm.eval()
    dt = torch.bfloat16 if os.environ.get("GLM_ORACLE_VISION_BF16") == "1" else torch.float32  # precision probe: the module run in bf16
    vm = vm.to(dt)
    with torch.no_grad():
        o = vm(patches.to(dt), grid_thw=torch.tensor([[1, gh, gw]]))
    return o.pooler_output.float()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fp8", default=None, help="Z.ai FP8 tensor directory (scripts/glm-fetch-fp8.py); as HIVE_GLM_FP8_DIR")
    ap.add_argument("--ids", required=True, help="prompt token ids: comma list or file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=None, help="a:b — run layers a..b only (input of a from <out>/h_L{a-1}.npy or the embeddings)")
    ap.add_argument("--teacher", default=None, help="generated token ids to append (teacher forcing): logits of every appended row")
    ap.add_argument("--mtp-next", type=int, default=None, help="token after the prompt: NextN draft logits of entry T-1")
    ap.add_argument("--vision-patches", default=None, help=".npy float32 [N, 1176] patches (merge-block order)")
    ap.add_argument("--grid", default=None, help="gh,gw of the patches")
    ap.add_argument("--image-at", type=int, default=None, help="prompt row of the first image token (rows take the encoder output)")
    ap.add_argument("--vision-only", action="store_true", help="only the vision encoder (vision.npy)")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    cfg = Glm5NextConfig.from_pretrained(a.ckpt)
    tc = cfg.text_config
    tc._attn_implementation = "eager"
    try:
        tc._experts_implementation = "eager"
    except Exception:
        pass
    W = Weights(a.ckpt, a.fp8)
    ids = parse_ids(a.ids)
    T = len(ids)
    extra = []
    if a.teacher:
        extra = parse_ids(a.teacher)
    if a.mtp_next is not None:
        extra = extra + [a.mtp_next] if not a.teacher else extra
    full = ids + extra
    TT = len(full)
    L = tc.num_hidden_layers
    l0, l1 = 0, L - 1
    if a.layers:
        l0, l1 = (int(x) for x in a.layers.split(":"))
    t0 = time.time()
    emb_w = W.get(P + "embed_tokens.weight") if l0 == 0 or a.mtp_next is not None else None
    vis = None
    if a.vision_patches:
        gh, gw = (int(x) for x in a.grid.split(","))
        pt = torch.from_numpy(np.load(a.vision_patches)).float()
        vis = vision_encode(cfg, W, pt, gh, gw)
        save(a.out, "vision.npy", vis.numpy())
        print(f"vision: {pt.shape[0]} patches ({gh}x{gw}) → {tuple(vis.shape)} · {time.time() - t0:.1f} s", flush=True)
        if a.vision_only:
            return 0
    if l0 == 0:
        x = emb_w[torch.tensor(full)]
        if vis is not None and a.image_at is not None:
            x[a.image_at:a.image_at + vis.shape[0]] = vis
        h = x[None, :, None, :].expand(1, TT, tc.hc_mult, tc.hidden_size).contiguous()
    else:
        prev = np.load(os.path.join(a.out, f"h_L{l0 - 1:02d}.npy"))
        if prev.shape[0] != TT:
            raise SystemExit(f"h_L{l0 - 1:02d}.npy has {prev.shape[0]} rows, need {TT}")
        h = torch.from_numpy(prev).reshape(1, TT, tc.hc_mult, tc.hidden_size)
    mask = torch.ones(1, TT, dtype=torch.bool)
    pos = torch.arange(TT)[None]
    for l in range(l0, l1 + 1):
        tl = time.time()
        layer = build_layer(tc, W, l, tc.layer_types[l])
        hook = None
        if hasattr(layer.mlp, "gate"):  # router choices of this layer (rows × top-k ids, selection order) → routes_LNN.npy
            def keep(_m, _i, o, l=l):
                np.save(os.path.join(a.out, f"routes_L{l:02d}.npy"), o[2].numpy().astype(np.int32))
            hook = layer.mlp.gate.register_forward_hook(keep)
        h, _ = layer(h, attention_mask=mask, position_ids=pos, past_key_values=None, prev_topk_indices=None)
        save(a.out, f"h_L{l:02d}.npy", h[0].reshape(TT, -1).numpy())
        if hook is not None:
            hook.remove()
        del layer
        gc.collect()  # the lazy-expert closure and the module form a cycle: free the layer's fp32 experts now
        print(f"layer {l:2d} ({tc.layer_types[l]}, {tc.mlp_layer_types[l]}): {time.time() - tl:.1f} s", flush=True)
    if l1 < L - 1:
        return 0
    fn = W.get(P + "norm.weight")
    hm = h[0].mean(dim=1)  # hyper head: unweighted mean of the streams
    fh = hm * torch.rsqrt(hm.pow(2).mean(-1, keepdim=True) + tc.rms_norm_eps) * fn
    save(a.out, "fh.npy", fh[:T].numpy())
    lm = W.get("lm_head.weight")
    save(a.out, "logits.npy", (fh[T - 1:T] @ lm.T).numpy())
    if a.teacher:
        tl = fh[T - 1:] @ lm.T
        save(a.out, "teacher.npy", tl.numpy())
        am = tl.argmax(-1).tolist()
        print("teacher argmax:", am)
        print("teacher given :", extra)
    if a.mtp_next is not None:
        if W.f8 is None:
            raise SystemExit("--mtp-next needs --fp8 (the NextN experts are the Z.ai FP8 copy)")
        p = P + f"layers.{L}."
        n = T  # entries 0 .. T-1: (h_j, x_{j+1})
        e = emb_w[torch.tensor(full[1:n + 1])].clone()
        e[0] = 0.0  # entry 0: embedding zeroed (engine contract)

        def rms(x, w):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + tc.rms_norm_eps) * w

        en = rms(e, W.get(p + "enorm.weight"))
        hn = rms(fh[:n], W.get(p + "hnorm.weight"))
        x0 = torch.cat([en, hn], -1) @ W.get(p + "eh_proj.weight").T
        layer = build_layer(tc, W, L, "deepseek_sparse_attention")
        m1 = torch.ones(1, n, dtype=torch.bool)
        xa, _, _ = layer.self_attn(hidden_states=layer.input_layernorm(x0[None]), attention_mask=m1, position_ids=torch.arange(n)[None],
                                   past_key_values=None, prev_topk_indices=None)
        h1 = x0[None] + xa
        y = layer.mlp(layer.post_attention_layernorm(h1)) + h1
        ho = rms(y[0], W.get(p + "shared_head.norm.weight"))
        save(a.out, "mtp_logits.npy", (ho[n - 1:n] @ lm.T).numpy())
        save(a.out, "mtp_h.npy", ho[n - 1:n].numpy())
        print("mtp draft argmax:", int((ho[n - 1] @ lm.T).argmax()))
    print(f"oracle done in {time.time() - t0:.1f} s · prompt {T} · rows {TT} · last-row argmax {int((fh[T - 1] @ lm.T).argmax())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
