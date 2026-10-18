# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Gemma 4 12B Unified (text and images) + its MTP assistant: an independent fp32 reference for the hive `gemma` family.

Math contract (what the engine must reproduce):
  embed      h = E[tok] * bf16(sqrt(3840))                         E = model.language_model.embed_tokens (bf16)
  layer      x += post_attention_layernorm(attn(input_layernorm(x)))
             x += post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))
             x *= layer_scalar                                       (one value per layer, from the checkpoint)
  RMSNorm    y = x * (mean(x^2) + 1e-6)^-0.5 * w                     (w, not 1 + w; fp32)
  mlp        down(gelu_tanh(gate(x)) * up(x))
  attention  q = rope(q_norm(q_proj x)); k = rope(k_norm(k_proj x)); v = rms(v_proj x) (no scale), scaling 1.0
             sliding layers (layer_types): head_dim 256, 16 q / 8 kv heads, window 1024 (key j visible to query i iff i-1024 < j <= i),
               rope "default" theta 1e4 over all 256 dims (rotate-half layout)
             full layers: head_dim 512, 16 q / 1 kv head, attention_k_eq_v: v = rms(k_proj x) (before k_norm / rope), causal,
               rope "proportional" theta 1e6: inv_freq[i] = theta^(-2i/512) for i < 64, 0 for 64 <= i < 256 (rotate-half layout)
  head       logits = 30 * tanh((norm(x) @ lm_head^T) / 30)          lm_head = checkpoint `lm_head.weight` (bf16)
  weights    compressed-tensors pack-quantized int4, symmetric, group 32 along the input: int32 holds 8 values, low nibble
             first, value = nibble - 8; W[o, i] = value * weight_scale[o, i // 32] (bf16 scale). Norms, embeddings, lm_head bf16.
  kv cache   the engine stores K and V as fp8 e4m3 (scale 1.0); --fp8-kv here rounds them the same way.
  suppress   generation_config suppress_tokens (end-of-image / end-of-audio markers) get logit -inf.
  images     (config use_bidirectional_attention "vision") an image's gh·gw soft tokens replace the placeholder rows' embeddings (no
             embedding scale) with the encoder-free vision embedder of its merged patches P [n, 48·48·3] (pixel / 255, rows (y, x, c), raster
             order — patch m at grid (m % gw, m / gw)):
               v = LN1(P) @ patch_dense^T + b;  v = LN2(v);  v = LN3(v + pos[x, 0] + pos[y, 1]);  e = rms(v) @ embedding_projection^T
             LN = LayerNorm with weight and bias, eps 1e-5; rms without scale, eps vision_config.rms_norm_eps. In attention every soft
             token of an image also sees the image's later soft tokens (both ways inside the span; causal everywhere else, window kept).
  assistant  (MTP drafter, bf16) pairs the target's hidden at the last processed position t with the embedding of the token that
             hidden predicted (sampled, not yet in the KV): z = pre_projection([E[tok_{t+1}]*sqrt(3840), h_t]) at rope position t (the last
             written slot — not advanced)
             -> 4 layers (types from its config; all KV-shared: only q_proj / q_norm / o_proj and the MLP, K/V = the target's last
             sliding layer and last full layer over the processed keys 0..t) -> norm -> logits = lm_head(z) (its own tied head, no
             softcap) and h = post_projection(z). The chain repeats with (draft, h) at the same position t. h_t is the target's final-normed hidden (--mtp-hidden raw = before the final norm, for comparison).

Usage (CPU, fp32; ~50 GB RAM for the dequantized target):
  python3 oracle/gemma4_oracle.py --ckpt DIR --tok DIR --prompt-ids ids.json --out OUTDIR [--decode 16] [--fp8-kv] [--images images.json]
      [--assistant DIR --mtp-k 3 --mtp-hidden normed|raw] [--dump-layers 0,5,23,47]
Writes OUTDIR/meta.json, h_LNN.npy (hidden after layer NN for the last prompt row), logits.npy (last prompt row),
decode.json (greedy continuation, and for --assistant the drafts proposed at each step and how many the target accepts).
--images: [{"start": row of the first soft token in the prompt ids, "gh": .., "gw": .., "file": raw bf16 patches [gh·gw, 6912]
(relative to the JSON's directory)}] — what tools/gemma_vision_case.py writes from a chat request through the server's preprocessing.
"""
import argparse, json, math, os, struct, sys, time
import numpy as np
import torch

torch.set_grad_enabled(False)


class Safetensors:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.base = 8 + n

    def has(self, name):
        return name in self.h

    def raw(self, name):
        m = self.h[name]
        a, b = m["data_offsets"]
        self.f.seek(self.base + a)
        return m, self.f.read(b - a)

    def tensor(self, name):
        m, raw = self.raw(name)
        dt = m["dtype"]
        if dt == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            return torch.from_numpy(u.view(np.float32).reshape(m["shape"]).copy())
        npdt = {"F32": np.float32, "I32": np.int32, "I64": np.int64, "F16": np.float16}[dt]
        return torch.from_numpy(np.frombuffer(raw, dtype=npdt).reshape(m["shape"]).copy())


def open_ckpt(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    shards = [Safetensors(os.path.join(d, f)) for f in files]

    class Multi:
        def find(self, name):
            for s in shards:
                if s.has(name):
                    return s
            raise KeyError(name)

        def has(self, name):
            return any(s.has(name) for s in shards)

        def tensor(self, name):
            return self.find(name).tensor(name).float()

    return Multi()


def linear_weight(ck, prefix):
    """[out, in] fp32 weight: dense (`.weight`) or int4 group-32 pack-quantized (`.weight_packed` + `.weight_scale`)."""
    if ck.has(prefix + ".weight"):
        return ck.tensor(prefix + ".weight")
    packed = ck.find(prefix + ".weight_packed").tensor(prefix + ".weight_packed").numpy().view(np.uint32)
    scale = ck.tensor(prefix + ".weight_scale")
    out, words = packed.shape
    nib = np.stack([(packed >> (4 * i)) & 0xF for i in range(8)], axis=-1).reshape(out, words * 8).astype(np.float32) - 8.0
    group = nib.shape[1] // scale.shape[1]
    return torch.from_numpy(nib) * scale.repeat_interleave(group, dim=1)


def rms(x, w=None, eps=1e-6):
    y = x * torch.pow((x * x).mean(-1, keepdim=True) + eps, -0.5)
    return y * w if w is not None else y


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def bf16(x):
    return x.to(torch.bfloat16).float()


def fp8(x):
    return x.to(torch.float8_e4m3fn).float()


def rope_tables(inv_freq, positions):
    f = positions[:, None].float() * inv_freq[None, :]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):  # x [T, H, D]
    d = x.shape[-1] // 2
    rot = torch.cat([-x[..., d:], x[..., :d]], dim=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


class Gemma4Text:
    def __init__(self, ckpt_dir, prefix="model.language_model.", fp8_kv=False):
        cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
        self.cfg = cfg.get("text_config", cfg)
        c = self.cfg
        self.ck = open_ckpt(ckpt_dir)
        self.pre = prefix
        self.fp8_kv = fp8_kv
        self.L = c["num_hidden_layers"]
        self.types = c["layer_types"]
        self.H = c["num_attention_heads"]
        self.win = c["sliding_window"]
        self.hidden = c["hidden_size"]
        rp = c["rope_parameters"]
        D = c["head_dim"]
        Dg = c["global_head_dim"]
        self.inv = {"sliding_attention": 1.0 / rp["sliding_attention"]["rope_theta"] ** (torch.arange(0, D, 2).float() / D)}
        n_rot = int(rp["full_attention"].get("partial_rotary_factor", 1.0) * Dg // 2)
        inv_g = torch.zeros(Dg // 2)
        inv_g[:n_rot] = 1.0 / rp["full_attention"]["rope_theta"] ** (torch.arange(0, 2 * n_rot, 2).float() / Dg)
        self.inv["full_attention"] = inv_g
        self.embed = self.ck.tensor(prefix + "embed_tokens.weight")
        self.embed_scale = float(torch.tensor(self.hidden ** 0.5, dtype=torch.bfloat16))
        self.norm_w = self.ck.tensor(prefix + "norm.weight")
        head_name = "lm_head.weight" if self.ck.has("lm_head.weight") else prefix + "embed_tokens.weight"
        self.head = self.ck.tensor(head_name)
        self.softcap = c.get("final_logit_softcapping")
        self.layers = [None] * self.L
        try:
            self.suppress = json.load(open(os.path.join(ckpt_dir, "generation_config.json"))).get("suppress_tokens") or []
        except OSError:
            self.suppress = []
        self.vcfg = cfg.get("vision_config")
        self.vis = None

    def vision(self):
        if self.vis is None:
            g = lambda n: self.ck.tensor(n)
            V = "model.vision_embedder."
            self.vis = {n: g(V + n) for n in ("patch_ln1.weight", "patch_ln1.bias", "patch_dense.weight", "patch_dense.bias", "patch_ln2.weight",
                                               "patch_ln2.bias", "pos_embedding", "pos_norm.weight", "pos_norm.bias")}
            self.vis["proj"] = g("model.embed_vision.embedding_projection.weight")
        return self.vis

    def vision_embed(self, patches, gh, gw):
        """merged patches [gh·gw, 6912] (fp32 of the bf16 input) → soft-token embeddings [gh·gw, hidden] (fp32)."""
        w = self.vision()
        ln = lambda x, n: torch.nn.functional.layer_norm(x, (x.shape[-1],), w[n + ".weight"], w[n + ".bias"], 1e-5)
        v = ln(patches, "patch_ln1") @ w["patch_dense.weight"].T + w["patch_dense.bias"]
        v = ln(v, "patch_ln2")
        m = torch.arange(gh * gw)
        v = v + w["pos_embedding"][m % gw, 0] + w["pos_embedding"][m // gw, 1]
        v = ln(v, "pos_norm")
        return rms(v, eps=self.vcfg.get("rms_norm_eps", 1e-6)) @ w["proj"].T

    def layer(self, i):
        if self.layers[i] is None:
            p = f"{self.pre}layers.{i}."
            g = lambda n: self.ck.tensor(p + n)
            w = {n: g(n + ".weight") for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                                               "post_feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm")}
            w["scalar"] = g("layer_scalar")
            for n in ("q_proj", "k_proj", "o_proj"):
                w[n] = linear_weight(self.ck, p + "self_attn." + n)
            w["v_proj"] = linear_weight(self.ck, p + "self_attn.v_proj") if (self.ck.has(p + "self_attn.v_proj.weight") or
                                                                          self.ck.has(p + "self_attn.v_proj.weight_packed")) else None
            for n in ("gate_proj", "up_proj", "down_proj"):
                w[n] = linear_weight(self.ck, p + "mlp." + n)
            self.layers[i] = w
        return self.layers[i]

    def new_cache(self):
        return {"k": [None] * self.L, "v": [None] * self.L, "pos": 0}

    def forward(self, ids, cache, dump=None, images=None):
        """ids: list[int] appended after cache['pos']; images: [(row in ids, gh, gw, patches fp32 [gh·gw, 6912])]. Returns (hidden after
        the final norm for every new row, raw last-layer hidden for every new row, logits of the last row)."""
        T = len(ids)
        p0 = cache["pos"]
        pos = torch.arange(p0, p0 + T)
        x = self.embed[torch.tensor(ids)] * self.embed_scale
        lim = pos.clone()   # last key each row sees: its own position, or the end of its image span
        for row, gh, gw, patches in images or []:
            n = gh * gw
            x[row:row + n] = self.vision_embed(patches, gh, gw)
            lim[row:row + n] = p0 + row + n - 1
        x = bf16(x)
        for i in range(self.L):
            w = self.layer(i)
            t = self.types[i]
            h = rms(x, w["input_layernorm"])
            q = (h @ w["q_proj"].T).view(T, self.H, -1)
            D = q.shape[-1]
            k_raw = (h @ w["k_proj"].T).view(T, -1, D)
            v_raw = (h @ w["v_proj"].T).view(T, -1, D) if w["v_proj"] is not None else k_raw
            cos, sin = rope_tables(self.inv[t], pos)
            q = apply_rope(rms(q, w["self_attn.q_norm"]), cos, sin)
            k = apply_rope(rms(k_raw, w["self_attn.k_norm"]), cos, sin)
            v = rms(v_raw)
            if self.fp8_kv:
                k, v = fp8(k), fp8(v)
            K = k if cache["k"][i] is None else torch.cat([cache["k"][i], k])
            V = v if cache["v"][i] is None else torch.cat([cache["v"][i], v])
            cache["k"][i], cache["v"][i] = K, V
            Hkv = K.shape[1]
            rep = self.H // Hkv
            Kr = K.repeat_interleave(rep, dim=1)
            Vr = V.repeat_interleave(rep, dim=1)
            s = torch.einsum("thd,shd->hts", q, Kr)  # scaling 1.0
            kpos = torch.arange(K.shape[0])
            mask = kpos[None, :] <= lim[:, None]
            if t == "sliding_attention":
                mask &= kpos[None, :] > pos[:, None] - self.win
            s = s.masked_fill(~mask[None], float("-inf"))
            a = torch.softmax(s, dim=-1)
            o = torch.einsum("hts,shd->thd", a, Vr).reshape(T, -1)
            x = x + rms(o @ w["o_proj"].T, w["post_attention_layernorm"])
            h = rms(x, w["pre_feedforward_layernorm"])
            m = (gelu_tanh(h @ w["gate_proj"].T) * (h @ w["up_proj"].T)) @ w["down_proj"].T
            x = x + rms(m, w["post_feedforward_layernorm"])
            x = x * w["scalar"]
            if dump is not None and i in dump["layers"]:
                np.save(os.path.join(dump["dir"], f"h_L{i:02d}.npy"), x[-1].numpy())
        cache["pos"] = p0 + T
        raw = x
        normed = rms(x, self.norm_w)
        logits = normed[-1] @ self.head.T
        if self.softcap:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        if self.suppress:
            logits[torch.tensor(self.suppress)] = float("-inf")
        return normed, raw, logits

    def last_kv(self, cache, ltype, upto):
        """K, V of the last layer of `ltype` (what the assistant reads), keys < upto."""
        i = max(j for j in range(self.L) if self.types[j] == ltype)
        return cache["k"][i][:upto], cache["v"][i][:upto]


class Gemma4Assistant:
    def __init__(self, d, target):
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.cfg = cfg
        self.tc = cfg.get("text_config", cfg)
        self.ck = open_ckpt(d)
        self.t = target
        names = [n for s in [self.ck.find("pre_projection.weight")] for n in s.h if n != "__metadata__"]
        mp = "model.language_model." if any(n.startswith("model.language_model.") for n in names) else "model."
        self.mp = mp
        g = self.ck.tensor
        self.pre_proj = g("pre_projection.weight")
        self.post_proj = g("post_projection.weight")
        self.norm_w = g(mp + "norm.weight")
        self.head = g("lm_head.weight") if self.ck.has("lm_head.weight") else g(mp + "embed_tokens.weight")
        self.softcap = self.tc.get("final_logit_softcapping")
        self.types = self.tc["layer_types"]
        self.H = self.tc["num_attention_heads"]
        self.win = self.tc["sliding_window"]
        self.layers = []
        for i in range(self.tc["num_hidden_layers"]):
            p = f"{mp}layers.{i}."
            w = {n: g(p + n + ".weight") for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                                                   "post_feedforward_layernorm", "self_attn.q_norm")}
            w["scalar"] = g(p + "layer_scalar") if self.ck.has(p + "layer_scalar") else torch.ones(1)
            for n in ("q_proj", "o_proj"):
                w[n] = linear_weight(self.ck, p + "self_attn." + n)
            for n in ("gate_proj", "up_proj", "down_proj"):
                w[n] = linear_weight(self.ck, p + "mlp." + n)
            self.layers.append(w)

    def step(self, tok, h, p, cache):
        """One draft step: query rope position p (= the last processed position, fixed for the chain), keys = everything the
        target has processed (0..p)."""
        t = self.t
        e = bf16(t.embed[tok] * t.embed_scale)
        z = torch.cat([e, h])[None] @ self.pre_proj.T
        pos = torch.tensor([p])
        for i, w in enumerate(self.layers):
            lt = self.types[i]
            K, V = t.last_kv(cache, lt, cache["pos"])
            hh = rms(z, w["input_layernorm"])
            q = (hh @ w["q_proj"].T).view(1, self.H, -1)
            cos, sin = rope_tables(t.inv[lt], pos)
            q = apply_rope(rms(q, w["self_attn.q_norm"]), cos, sin)
            rep = self.H // K.shape[1]
            Kr, Vr = K.repeat_interleave(rep, dim=1), V.repeat_interleave(rep, dim=1)
            kpos = torch.arange(K.shape[0])
            mask = kpos <= p
            if lt == "sliding_attention":
                mask &= kpos > p - self.win
            s = torch.einsum("thd,shd->hts", q, Kr).masked_fill(~mask[None, None], float("-inf"))
            o = torch.einsum("hts,shd->thd", torch.softmax(s, -1), Vr).reshape(1, -1)
            z = z + rms(o @ w["o_proj"].T, w["post_attention_layernorm"])
            hh = rms(z, w["pre_feedforward_layernorm"])
            m = (gelu_tanh(hh @ w["gate_proj"].T) * (hh @ w["up_proj"].T)) @ w["down_proj"].T
            z = z + rms(m, w["post_feedforward_layernorm"])
            z = z * w["scalar"]
        zn = rms(z, self.norm_w)
        logits = (zn @ self.head.T)[0]
        if self.softcap:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return int(torch.argmax(logits)), (zn @ self.post_proj.T)[0], logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt-ids", required=True, help="JSON file with a list of token ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode", type=int, default=0)
    ap.add_argument("--fp8-kv", action="store_true")
    ap.add_argument("--dump-layers", default="0,5,23,46,47")
    ap.add_argument("--assistant")
    ap.add_argument("--mtp-k", type=int, default=3)
    ap.add_argument("--mtp-hidden", choices=["normed", "raw"], default="normed")
    ap.add_argument("--eos", default="1,106,50")
    ap.add_argument("--images", help="JSON list of images (see the module docstring)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ids = json.load(open(a.prompt_ids))
    t0 = time.time()
    m = Gemma4Text(a.ckpt, fp8_kv=a.fp8_kv)
    asst = Gemma4Assistant(a.assistant, m) if a.assistant else None
    cache = m.new_cache()
    dump = {"layers": {int(x) for x in a.dump_layers.split(",") if x}, "dir": a.out}
    images = []
    if a.images:
        base = os.path.dirname(os.path.abspath(a.images))
        for im in json.load(open(a.images)):
            u = np.fromfile(os.path.join(base, im["file"]), dtype=np.uint16).astype(np.uint32) << 16
            images.append((im["start"], im["gh"], im["gw"], torch.from_numpy(u.view(np.float32).reshape(im["gh"] * im["gw"], -1).copy())))
    normed, raw, logits = m.forward(ids, cache, dump, images)
    np.save(os.path.join(a.out, "logits.npy"), logits.numpy())
    np.save(os.path.join(a.out, "h_final.npy"), normed[-1].numpy())
    eos = {int(x) for x in a.eos.split(",")}
    out, drafts = [], []
    tok = int(torch.argmax(logits))
    for step in range(a.decode):
        out.append(tok)
        if tok in eos:
            break
        nrm, rw, lg = m.forward([tok], cache)
        nxt = int(torch.argmax(lg))
        if asst is not None:
            # `tok` is processed (position t = pos - 1); `nxt` is the target's own next token (out[step + 1], not processed yet).
            # The chain starts from (nxt, h_t) at rope position t; draft j predicts out[step + 2 + j].
            prev, h, d = nxt, (nrm if a.mtp_hidden == "normed" else rw)[-1], []
            for _ in range(a.mtp_k):
                prev, h, _ = asst.step(prev, h, cache["pos"] - 1, cache)
                d.append(prev)
            drafts.append(d)
        tok = nxt
    meta = {"prompt_tokens": len(ids), "images": len(images), "fp8_kv": a.fp8_kv, "seconds": round(time.time() - t0, 1), "mtp_hidden": a.mtp_hidden}
    if asst is not None:
        # acceptance: drafts made after step s predict out[s+2 .. s+1+k] (out[s+1] is the target's own token)
        acc = []
        for s, d in enumerate(drafts):
            if s + 2 >= len(out):
                break
            n = 0
            for j, x in enumerate(d):
                if s + 2 + j < len(out) and out[s + 2 + j] == x:
                    n += 1
                else:
                    break
            acc.append(n)
        meta["mtp_accept_mean"] = float(np.mean(acc)) if acc else None
        meta["mtp_accept"] = acc
        # speculative walk as a server runs it: verify at s, accept n drafts, the next verify is at s + n + 1
        # (accept length = accepted drafts + the target's own token, the number SGLang reports as spec_accept_length)
        s_, steps, hist = 0, 0, [0] * (a.mtp_k + 1)
        while s_ < len(acc):
            n = acc[s_]
            hist[n] += 1
            steps += 1
            s_ += n + 1
        meta["spec_walk"] = {"verify_steps": steps, "accept_length": round(sum((n + 1) * c for n, c in enumerate(hist)) / max(1, steps), 3),
                             "accepted_hist": hist}
    json.dump({"tokens": out, "drafts": drafts}, open(os.path.join(a.out, "decode.json"), "w"))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
