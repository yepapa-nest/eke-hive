#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""DeepSeek-V4.1-Flash CPU engine (hived_ds_cpu, engine/src/ds/cpu) validation against the CPU oracle (oracle/dsv41_oracle.py).

One directory format on both sides ("raw golden"): meta.json + per-step raw arrays, step k = s{k}_ (s0 = the prompt, s1.. = one decode step each):
  meta.json                {"ids": prompt ids, "steps": [{"start": first position, "rows": rows, "ids": [...]}, ...], "generated": [...]}
  s{k}_h_L{l}.f32          hidden stream after layer l, float32 [rows, 4, 5120] (values are bf16)
  s{k}_premix_L{l}.f32     pre-mix after layer l, float32 [rows, 4]
  s{k}_engram_L{l}.f32     hidden stream after the engram of layer l, float32 [rows, 4, 5120]
  s{k}_logits.f32          float32 [n, vocab] (the oracle: the last row of the step; the engine: every row it produced logits for)
  s{k}_route.json          {"L03": {"rows": [first row, ...], "ids": [[6 ids], ...], "w": [[6 weights], ...]}, ...} (last rows only)
  s{k}_index.json          {"L02": {"rows": [...], "sel": [[selected compressed positions], ...]}, ...}
Rows of an engine file may be fewer than the step's rows (decoder tail mode keeps only the last rows above layer 20): the comparison uses the
  last rows of the golden then.

Subcommands
  oracle   run the oracle on a prompt (token ids or text files) streaming every layer to disk (a 4K prompt is 13 GB of hidden streams — the oracle's
           npz path would hold them all in memory) and without its expert cache (all 384 experts of every layer are used by a long prompt; the cache
           would keep 64 dequantized fp32 experts = 9 GB per layer). Same math: only the cache policy and the dump sink differ.
  export   oracle npz golden directory (tools/validate_layers.sh format) → raw golden directory (above)
  compare  engine dump directory vs raw golden directory → per layer / per step numbers, exit 1 below the thresholds
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

DIM, HC = 5120, 4


# ---------------------------------------------------------------------------------------------------------------------------------------
# oracle (streaming)
# ---------------------------------------------------------------------------------------------------------------------------------------
class Sink(dict):
    """Dump sink for Oracle.forward: writes each array as s{step}_{name}.f32 at once and keeps nothing (keys() is empty, so np.savez of it is empty)."""

    def __init__(self, out: str, step: int):
        super().__init__()
        self.out, self.step = out, step

    def __setitem__(self, key, value):
        m = re.match(r"^(h|premix|engram)_L(\d+)$", key)
        name = f"{m.group(1)}_L{int(m.group(2))}" if m else key
        np.ascontiguousarray(value, dtype=np.float32).tofile(os.path.join(self.out, f"s{self.step}_{name}.f32"))


def trim_log(log: dict, rows: int, keep: int) -> tuple[dict, dict]:
    """routing / index of the last `keep` rows"""
    r0 = max(0, rows - keep)
    route, index = {}, {}
    for k, v in log.get("routing", {}).items():
        route[k] = {"rows": list(range(r0, rows)), "ids": v["ids"][r0:], "w": v["w"][r0:]}
    for k, (idxs, out) in log.get("index", {}).items():
        sel = []
        for m in range(r0, rows):
            sel.append(sorted(int(i) for i, o in zip(idxs[m], out[m]) if o >= 0))
        index[k] = {"rows": list(range(r0, rows)), "sel": sel}
    return route, index


def cmd_oracle(a) -> int:
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle"))
    import dsv41_oracle as O

    # no expert cache (see the module docstring) — the same Lin.fp4 dequantization per call
    def expert_weights(self, i):
        ep = f"{self.p}.experts.{i}"
        return (O.Lin.fp4(self.ck, ep + ".w1"), O.Lin.fp4(self.ck, ep + ".w2"), O.Lin.fp4(self.ck, ep + ".w3"))

    O.MoE.expert_weights = expert_weights
    torch.set_num_threads(a.threads)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(a.ckpt, "tokenizer.json"))
    if a.ids_file:
        ids = [int(t) for t in re.split(r"[\s,]+", open(a.ids_file).read().strip()) if t]
    elif a.ids:
        ids = [int(t) for t in a.ids.split(",")]
    else:
        text = "".join(open(p, encoding="utf-8", errors="replace").read() + "\n" for p in a.text)
        ids = tok.encode(text, add_special_tokens=False).ids
    if a.tokens:
        ids = ids[: a.tokens - (1 if a.bos else 0)]
    if a.bos:
        ids = [0] + ids
    os.makedirs(a.out, exist_ok=True)
    print(f"[oracle] {len(ids)} prompt tokens → {a.out}", flush=True)
    t0 = time.time()
    orc = O.Oracle(a.ckpt, a.max_seq_len, a.max_layer, tok, quiet=a.quiet)
    meta = {"ids": ids, "steps": [], "generated": []}
    logits, log = orc.forward(torch.tensor(ids), 0, Sink(a.out, 0))
    route, index = trim_log(log, len(ids), a.log_rows)
    json.dump(route, open(os.path.join(a.out, "s0_route.json"), "w"))
    json.dump(index, open(os.path.join(a.out, "s0_index.json"), "w"))
    meta["steps"].append({"start": 0, "rows": len(ids), "ids": ids})
    print(f"[oracle] prefill {time.time() - t0:.0f}s", flush=True)
    if logits is not None:
        logits[-1:].numpy().astype(np.float32).tofile(os.path.join(a.out, "s0_logits.f32"))
        nxt = int(logits[-1].argmax())
        del logits
        for k in range(a.decode):
            pos = len(ids) + k
            meta["generated"].append(nxt)
            lg, dlog = orc.forward(torch.tensor([nxt]), pos, Sink(a.out, k + 1))
            route, index = trim_log(dlog, 1, 1)
            json.dump(route, open(os.path.join(a.out, f"s{k + 1}_route.json"), "w"))
            json.dump(index, open(os.path.join(a.out, f"s{k + 1}_index.json"), "w"))
            lg[-1:].numpy().astype(np.float32).tofile(os.path.join(a.out, f"s{k + 1}_logits.f32"))
            meta["steps"].append({"start": pos, "rows": 1, "ids": [nxt]})
            nxt = int(lg[-1].argmax())
            print(f"[oracle] decode {k} pos {pos} → {nxt} {tok.decode([nxt])!r} ({time.time() - t0:.0f}s)", flush=True)
        meta["generated"].append(nxt)
    elif a.continue_ids:
        for k, t in enumerate(int(x) for x in a.continue_ids.split(",")):
            pos = len(ids) + k
            _, dlog = orc.forward(torch.tensor([t]), pos, Sink(a.out, k + 1))
            route, index = trim_log(dlog, 1, 1)
            json.dump(route, open(os.path.join(a.out, f"s{k + 1}_route.json"), "w"))
            json.dump(index, open(os.path.join(a.out, f"s{k + 1}_index.json"), "w"))
            meta["steps"].append({"start": pos, "rows": 1, "ids": [t]})
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))
    print(f"[oracle] done {time.time() - t0:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------------------------------------
# export (npz golden → raw golden)
# ---------------------------------------------------------------------------------------------------------------------------------------
def cmd_export(a) -> int:
    os.makedirs(a.out, exist_ok=True)
    gj = json.load(open(os.path.join(a.golden, "prefill.json")))
    ids = gj["ids"]
    steps = [("prefill", 0, len(ids), ids)]
    for p in sorted(glob.glob(os.path.join(a.golden, "decode_[0-9][0-9][0-9].json"))):
        k = int(p[-8:-5])
        dj = json.load(open(p))
        steps.append((f"decode_{k:03d}", len(ids) + k, 1, dj.get("ids", [])))
    meta = {"ids": ids, "steps": [], "generated": []}
    if os.path.exists(os.path.join(a.golden, "decode.json")):
        meta["generated"] = json.load(open(os.path.join(a.golden, "decode.json"))).get("generated", [])
    for s, (name, start, rows, sids) in enumerate(steps):
        g = np.load(os.path.join(a.golden, name + ".npz"))
        for k in g.files:
            m = re.match(r"^(h|premix|engram)_L(\d+)$", k)
            if m:
                g[k].astype(np.float32).tofile(os.path.join(a.out, f"s{s}_{m.group(1)}_L{int(m.group(2))}.f32"))
            elif k.startswith("image_emb_") or k.startswith("vit_"):
                g[k].astype(np.float32).tofile(os.path.join(a.out, f"s{s}_{k}.f32"))
        lj = json.load(open(os.path.join(a.golden, name + ".json")))
        route, index = trim_log(lj, rows, rows)
        json.dump(route, open(os.path.join(a.out, f"s{s}_route.json"), "w"))
        json.dump(index, open(os.path.join(a.out, f"s{s}_index.json"), "w"))
        if s == 0 and os.path.exists(os.path.join(a.golden, "logits.npy")):
            np.load(os.path.join(a.golden, "logits.npy"))[-1:].astype(np.float32).tofile(os.path.join(a.out, "s0_logits.f32"))
        if "logits" in g.files:
            g["logits"][-1:].astype(np.float32).tofile(os.path.join(a.out, f"s{s}_logits.f32"))
        if not sids and s > 0 and meta["generated"]:
            sids = [meta["generated"][s - 1]]
        meta["steps"].append({"start": start, "rows": rows, "ids": sids})
    # DSpark golden (oracle --spec): spec.npz / spec.json → spec_*.f32 + spec.json
    if os.path.exists(os.path.join(a.golden, "spec.npz")):
        g = np.load(os.path.join(a.golden, "spec.npz"))
        for k in g.files:
            g[k].astype(np.float32).tofile(os.path.join(a.out, f"{k}.f32"))
        sj = json.load(open(os.path.join(a.golden, "spec.json")))
        json.dump({k: sj[k] for k in ("tok", "pos", "ids", "conf")}, open(os.path.join(a.out, "spec.json"), "w"))
    if os.path.exists(os.path.join(a.golden, "image_inputs.json")):
        ii = json.load(open(os.path.join(a.golden, "image_inputs.json")))
        for im in ii["images"]:
            im["file"] = os.path.join(os.path.abspath(a.golden), im["file"])
        json.dump(ii, open(os.path.join(a.out, "image_inputs.json"), "w"))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))
    print(f"exported {len(steps)} steps → {a.out}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------------------------------------------------------------------
def load_rows(path: str, cols: int) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    v = np.fromfile(path, dtype=np.float32)
    return v.reshape(-1, cols)


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return (a64 * b64).sum(-1) / np.maximum(np.linalg.norm(a64, axis=-1) * np.linalg.norm(b64, axis=-1), 1e-30)


def cmd_compare(a) -> int:
    meta = json.load(open(os.path.join(a.golden, "meta.json")))
    fails = []
    worst_cos, worst_at = 2.0, ""
    route_tot = route_same = 0
    idx_tot = idx_same = 0.0
    nsteps = len(meta["steps"])
    for s, st in enumerate(meta["steps"]):
        layers = sorted(int(re.search(r"_h_L(\d+)\.f32$", p).group(1)) for p in glob.glob(os.path.join(a.golden, f"s{s}_h_L*.f32")))
        if a.layers:
            lo, hi = (int(x) for x in a.layers.split("-"))
            layers = [l for l in layers if lo <= l <= hi]
        if not layers:
            continue
        gr = json.load(open(os.path.join(a.golden, f"s{s}_route.json"))) if os.path.exists(os.path.join(a.golden, f"s{s}_route.json")) else {}
        gi = json.load(open(os.path.join(a.golden, f"s{s}_index.json"))) if os.path.exists(os.path.join(a.golden, f"s{s}_index.json")) else {}
        er = json.load(open(os.path.join(a.dump, f"s{s}_route.json"))) if os.path.exists(os.path.join(a.dump, f"s{s}_route.json")) else {}
        ei = json.load(open(os.path.join(a.dump, f"s{s}_index.json"))) if os.path.exists(os.path.join(a.dump, f"s{s}_index.json")) else {}
        print(f"== step {s}: positions {st['start']}..{st['start'] + st['rows'] - 1} ({st['rows']} rows)")
        for l in layers:
            line = f"L{l:02d}"
            for kind in (["engram", "h"] if os.path.exists(os.path.join(a.golden, f"s{s}_engram_L{l}.f32")) else ["h"]):
                g = load_rows(os.path.join(a.golden, f"s{s}_{kind}_L{l}.f32"), HC * DIM)
                e = load_rows(os.path.join(a.dump, f"s{s}_{kind}_L{l}.f32"), HC * DIM)
                if e is None:
                    fails.append(f"s{s} L{l} {kind}: missing in the engine dump")
                    line += f" {kind}: missing"
                    continue
                pp = os.path.join(a.dump, f"s{s}_{kind}_L{l}.pos")
                if os.path.exists(pp):  # engine rows carry their positions (chunked prefill, decoder tail rows)
                    pos = np.fromfile(pp, dtype=np.int64) - st["start"]
                    keep = (pos >= 0) & (pos < len(g))
                    e, g = e[keep], g[pos[keep]]
                else:
                    n = min(len(g), len(e))
                    g, e = g[-n:], e[-n:]
                if a.rows:
                    g, e = g[-a.rows:], e[-a.rows:]
                c = cos_rows(e, g)
                rel = np.abs(e - g).max() / max(np.abs(g).max(), 1e-30)
                rrms = np.sqrt(((e - g) ** 2).mean()) / max(np.sqrt((g ** 2).mean()), 1e-30)
                line += f" {kind}: cos min {c.min():.6f} mean {c.mean():.6f} rel-rms {rrms:.2e} max-rel {rel:.2e} ({len(c)} rows)"
                if c.min() < worst_cos:
                    worst_cos, worst_at = float(c.min()), f"s{s} L{l} {kind} row {int(np.argmin(c))}"
                if c.min() < a.min_cos:
                    fails.append(f"s{s} L{l} {kind}: cos {c.min():.6f}")
            key = f"L{l:02d}"
            if key in gr and key in er:
                gm = {r: set(i) for r, i in zip(gr[key]["rows"], gr[key]["ids"])}
                em = {r: set(i) for r, i in zip(er[key]["rows"], er[key]["ids"])}
                common = [r for r in gm if r in em]
                if common:
                    same = sum(len(gm[r] & em[r]) for r in common)
                    tot = sum(len(gm[r]) for r in common)
                    route_tot += tot
                    route_same += same
                    line += f" · route {100.0 * same / max(tot, 1):.1f}%"
            if key in gi and key in ei:
                gm = {r: set(i) for r, i in zip(gi[key]["rows"], gi[key]["sel"])}
                em = {r: set(i) for r, i in zip(ei[key]["rows"], ei[key]["sel"])}
                common = [r for r in gm if r in em]
                if common:
                    agree = [len(gm[r] & em[r]) / len(gm[r]) if gm[r] else float(not em[r]) for r in common]
                    idx_tot += len(agree)
                    idx_same += sum(agree)
                    line += f" · index {100.0 * np.mean(agree):.2f}%"
            print(line)
        gp = os.path.join(a.golden, f"s{s}_logits.f32")
        if os.path.exists(gp):
            gl = np.fromfile(gp, dtype=np.float32)
            el_path = os.path.join(a.dump, f"s{s}_logits.f32")
            if not os.path.exists(el_path):
                fails.append(f"s{s}: logits missing")
                continue
            V = gl.size
            el = np.fromfile(el_path, dtype=np.float32).reshape(-1, V)[-1]
            p = np.exp(gl - gl.max()); p /= p.sum()
            q = np.exp(el - el.max()); q /= q.sum()
            kl = float((p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum())
            top_g, top_e = int(gl.argmax()), int(el.argmax())
            t20 = len(set(np.argsort(-gl)[:20]) & set(np.argsort(-el)[:20])) / 20
            print(f"   logits: top1 oracle {top_g} engine {top_e} {'same' if top_g == top_e else 'DIFFERENT'} · top20 overlap {t20:.2f} · KL {kl:.3e} · "
                  f"max|Δ| {np.abs(gl - el).max():.3f} · cos {float(cos_rows(gl[None], el[None])[0]):.6f}")
            if top_g != top_e:
                fails.append(f"s{s}: argmax {top_e} != oracle {top_g}")
    if route_tot:
        print(f"routing set agreement {100.0 * route_same / route_tot:.2f}% over {route_tot // 6} rows·layers")
    if idx_tot:
        print(f"index set agreement {100.0 * idx_same / idx_tot:.2f}% over {int(idx_tot)} rows·layers")
    print(f"lowest cosine {worst_cos:.6f} at {worst_at}")
    # DSpark
    if os.path.exists(os.path.join(a.golden, "spec.json")) and os.path.exists(os.path.join(a.dump, "spec.json")):
        gs, es = json.load(open(os.path.join(a.golden, "spec.json"))), json.load(open(os.path.join(a.dump, "spec.json")))
        print(f"== DSpark at pos {gs['pos']}: oracle drafts {gs['ids'][1:]} conf {[round(c, 3) for c in gs['conf']]}")
        print(f"                 engine drafts {es['ids'][1:]} conf {[round(c, 3) for c in es['conf']]}")
        for k in ("spec_main_x", "spec_x", "spec_logits"):
            gp, ep = os.path.join(a.golden, k + ".f32"), os.path.join(a.dump, k + ".f32")
            if os.path.exists(gp) and os.path.exists(ep):
                g, e = np.fromfile(gp, dtype=np.float32), np.fromfile(ep, dtype=np.float32)
                if g.size == e.size:
                    rows = 5 if k != "spec_main_x" else 1
                    c = cos_rows(e.reshape(rows, -1), g.reshape(rows, -1))
                    print(f"   {k}: cos min {c.min():.6f}")
        for s in range(3):
            gp, ep = os.path.join(a.golden, f"spec_h_M{s}.f32"), os.path.join(a.dump, f"spec_h_M{s}.f32")
            if os.path.exists(gp) and os.path.exists(ep):
                c = cos_rows(np.fromfile(ep, dtype=np.float32).reshape(5, -1), np.fromfile(gp, dtype=np.float32).reshape(5, -1))
                print(f"   stage {s} h: cos min {c.min():.6f}")
        if gs["ids"] != es["ids"]:
            fails.append(f"DSpark drafts differ: {es['ids']} vs {gs['ids']}")
    # vision
    for gp in sorted(glob.glob(os.path.join(a.golden, "s0_image_emb_*.f32")) + glob.glob(os.path.join(a.golden, "s0_vit_*.f32"))):
        ep = os.path.join(a.dump, os.path.basename(gp))
        if os.path.exists(ep):
            g, e = np.fromfile(gp, dtype=np.float32), np.fromfile(ep, dtype=np.float32)
            if g.size == e.size:
                cols = DIM if "image_emb" in gp or gp.endswith("_out.f32") else 1024
                c = cos_rows(e.reshape(-1, cols), g.reshape(-1, cols))
                print(f"{os.path.basename(gp)}: cos min {c.min():.6f} mean {c.mean():.6f} max|Δ| {np.abs(e - g).max():.3e}")
    if fails:
        print("FAIL:\n  " + "\n  ".join(fails[:40]) + (f"\n  ... {len(fails) - 40} more" if len(fails) > 40 else ""))
        return 1
    print("OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--ckpt", default=os.environ.get("HIVE_CKPT"))
    o.add_argument("--out", required=True)
    o.add_argument("--text", nargs="*", default=[], help="text files (concatenated, tokenized without special tokens)")
    o.add_argument("--ids", default=None)
    o.add_argument("--ids-file", default=None)
    o.add_argument("--tokens", type=int, default=0, help="cut the prompt to this many tokens (bos included)")
    o.add_argument("--bos", action="store_true")
    o.add_argument("--decode", type=int, default=0)
    o.add_argument("--continue-ids", default=None)
    o.add_argument("--max-layer", type=int, default=None)
    o.add_argument("--max-seq-len", type=int, default=8192)
    o.add_argument("--log-rows", type=int, default=64, help="routing / index of this many last prompt rows go to the json")
    o.add_argument("--threads", type=int, default=24)
    o.add_argument("--quiet", action="store_true")
    e = sub.add_parser("export")
    e.add_argument("golden")
    e.add_argument("out")
    c = sub.add_parser("compare")
    c.add_argument("golden")
    c.add_argument("dump")
    c.add_argument("--layers", default=None, help="lo-hi")
    c.add_argument("--rows", type=int, default=0, help="compare only the last N rows of every step")
    c.add_argument("--min-cos", type=float, default=0.995)
    a = ap.parse_args()
    if a.cmd == "oracle":
        a.out = os.path.abspath(a.out)
        return cmd_oracle(a)
    if a.cmd == "export":
        return cmd_export(a)
    return cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
