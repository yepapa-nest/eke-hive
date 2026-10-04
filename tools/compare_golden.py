#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Per-layer comparison of the oracle golden (prefill.npz/json) against an engine dump (--dump directory).
Usage: compare_golden.py --golden /out/golden/l0 --dump /out/dump/l0 [--layers 0-39]"""
import argparse
import json
import os

import numpy as np


def bf16_file(path, shape):
    raw = np.fromfile(path, dtype=np.uint16)
    f = (raw.astype(np.uint32) << 16).view(np.float32)
    return f.reshape(shape)


def stats(a, b):
    d = np.abs(a - b)
    ref = np.abs(b).max() + 1e-30
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return f"max|Δ|={d.max():.3e} mean|Δ|={d.mean():.3e} rel={d.max() / ref:.3e} cos={cos:.6f} (ref max {np.abs(b).max():.3e})"


def check_array(a, label, actual, expected):
    # A missing layer dump or a shape mismatch is an error (must not end with exit 0).
    if actual is None or actual.shape != expected.shape:
        a.errors.append(label + ': missing/shape mismatch')
        return
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        a.errors.append(label + ': nonfinite')
        return
    rel = float(np.max(np.abs(actual - expected)) / max(float(np.max(np.abs(expected))), 1e-30))
    den = float(np.linalg.norm(actual) * np.linalg.norm(expected))
    cos = float(np.sum(actual * expected) / den) if den else float(np.array_equal(actual, expected))
    if rel > a.max_rel or cos < a.min_cos:
        a.errors.append(f'{label}: rel={rel:.6g}, cos={cos:.6g}')


def load_engine(a, pre, name, shape, chunks):
    """Read an engine dump. With chunks=[(prefix, rows)], several chunks (split prefill) are concatenated by rows. With --rows/--row, one row of a batched dump."""
    if not chunks:
        path = os.path.join(a.dump, f"{pre}{name}")
        if not os.path.exists(path):
            return None
        full = bf16_file(path, (-1,) + tuple(shape[1:]))
        if full.shape[0] == shape[0]:
            return full
        if getattr(a, "rows", 0) and full.shape[0] == a.rows:
            return full[a.row : a.row + 1]
        return None
    parts = []
    for cpre, rows in chunks:
        path = os.path.join(a.dump, f"{cpre}{name}")
        if not os.path.exists(path):
            return None
        parts.append(bf16_file(path, (rows,) + tuple(shape[1:])))
    return np.concatenate(parts, 0)


def compare_step(a, g, gj, pre, title, chunks=None, tail=0):
    ids = gj["ids"]
    M = len(ids)
    layers = sorted(int(k[3:]) for k in g.files if k.startswith("h_L"))
    if not layers:
        a.errors.append(title + ': no golden layers')
        return
    print(f"== {title}: tokens {M} · golden layers {layers[0]}..{layers[-1]}" + (f" · last {tail} rows only" if tail else ""))
    for l in layers:
        ref = g[f"h_L{l:02d}"]  # [M,4,dim] fp32
        if tail and l >= a.tail_layer:
            ref = ref[-tail:]
            path = os.path.join(a.dump, f"{pre}h_L{l}.bf16")
            eng = None
            if os.path.exists(path):
                eng = bf16_file(path, (-1, ref.shape[1], ref.shape[2]))[-tail:]
        else:
            eng = load_engine(a, pre, f"h_L{l}.bf16", ref.shape, chunks)
        if eng is None:
            print(f"L{l:02d}: no engine dump")
            a.errors.append(f'{title} L{l}: missing dump')
            continue
        check_array(a, f'{title} L{l}', eng, ref)
        line = f"L{l:02d} h: {stats(eng, ref)}"
        # Routing comparison (skipped for chunked prefill)
        rp = os.path.join(a.dump, f"{pre}route_ids_L{l}.i32")
        if not chunks and f'L{l:02d}' in gj.get('routing', {}) and not os.path.exists(rp):
            a.errors.append(f'{title} L{l}: missing routing')
        if not chunks and os.path.exists(rp) and "routing" in gj and f"L{l:02d}" in gj["routing"]:
            gi = np.array(gj["routing"][f"L{l:02d}"]["ids"])
            ei = np.fromfile(rp, dtype=np.int32).reshape(-1, gi.shape[1])
            if getattr(a, "rows", 0) and ei.shape[0] == a.rows and gi.shape[0] == 1:
                ei = ei[a.row : a.row + 1]
            same = np.mean([len(set(gi[m]) & set(ei[m])) / gi.shape[1] for m in range(M)])
            if same < a.min_routing: a.errors.append(f'{title} L{l}: routing={same:.6g}')
            gw = np.array(gj["routing"][f"L{l:02d}"]["w"])
            ew = np.fromfile(os.path.join(a.dump, f"{pre}route_w_L{l}.f32"), dtype=np.float32).reshape(-1, gw.shape[1])
            if getattr(a, "rows", 0) and ew.shape[0] == a.rows and gw.shape[0] == 1:
                ew = ew[a.row : a.row + 1]
            line += f" · routing set-agree {same * 100:.1f}% w max|Δ| {np.abs(np.sort(gw, 1) - np.sort(ew, 1)).max():.2e}"
            if same == 1:
                check_array(a, f'{title} L{l} route weights', np.take_along_axis(ew, ei.argsort(1), axis=1), np.take_along_axis(gw, gi.argsort(1), axis=1))
        # Index comparison
        if not chunks and "index" in gj and f"L{l:02d}" in gj["index"]:
            # Compared by group id. The golden's global indices (entry 1) are offset by prefill M / decode win, the engine's by win+M (that forward's row count),
            # so they are different coordinate systems (comparing them directly gave 0 % / 85.7 % on L2 although the actual sets were identical).
            gidx = np.array(gj["index"][f"L{l:02d}"][0])
            ip = os.path.join(a.dump, f"{pre}attn_idx_L{l}.i32")
            if os.path.exists(ip):
                eidx = np.fromfile(ip, dtype=np.int32).reshape(-1, 128 + 512)
                eoff = 128 + eidx.shape[0]
                if getattr(a, "rows", 0) and eidx.shape[0] == a.rows and M == 1:
                    eidx = eidx[a.row : a.row + 1]
                eidx = eidx[:, 128 : 128 + gidx.shape[1]]
                eidx = np.where(eidx >= 0, eidx - eoff, -1)
                agree = np.mean([len(set(gidx[m][gidx[m] >= 0]) & set(eidx[m][eidx[m] >= 0])) / max(1, (gidx[m] >= 0).sum()) for m in range(M)])
                line += f" · index agree {agree * 100:.1f}%"
                if agree < a.min_index: a.errors.append(f'{title} L{l}: index={agree:.6g}')
            else:
                a.errors.append(f'{title} L{l}: missing index')
        # engram
        if f"engram_L{l:02d}" in g.files:
            ee = load_engine(a, pre, f"engram_L{l}.bf16", ref.shape, chunks)
            check_array(a, f'{title} L{l} engram', ee, g[f'engram_L{l:02d}'])
            if ee is not None:
                line += f"\n      engram: {stats(ee, g[f'engram_L{l:02d}'])}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument('--max-rel', type=float, default=0.05)
    ap.add_argument('--min-cos', type=float, default=0.999)
    ap.add_argument('--min-routing', type=float, default=1.0)
    ap.add_argument('--min-index', type=float, default=1.0)
    ap.add_argument('--require-argmax', action='store_true')
    ap.add_argument('--report-only', action='store_true', help='Diagnostic only; never use for release validation')
    ap.add_argument("--chunk", type=int, default=0, help="engine prefill chunk size (set for a chunked prefill)")
    ap.add_argument("--tail", type=int, default=0, help="decoder tail mode: from layer --tail-layer on, compare only the last N rows (rows of the engine dump)")
    ap.add_argument("--tail-layer", type=int, default=20)
    ap.add_argument("--rows", type=int, default=0, help="row count of a batched dump (when a decode-step dump has M rows)")
    ap.add_argument("--row", type=int, default=0, help="engine row to compare")
    ap.add_argument("--decode-offset", type=int, default=0, help="golden decode_000 corresponds to engine step s{offset+1} (corrects for the number of prefill steps)")
    ap.add_argument("--prefill-step", type=int, default=0, help="the prefill of this sequence is engine step s{k}_ (second sequence of a batch test)")
    a = ap.parse_args()
    a.errors = []
    g = np.load(os.path.join(a.golden, "prefill.npz"))
    gj = json.load(open(os.path.join(a.golden, "prefill.json")))
    for key in [k for k in g.files if k.startswith("image_emb_")]:
        ref = g[key]
        path = os.path.join(a.dump, f"s0_{key}.bf16")
        if os.path.exists(path):
            actual = bf16_file(path, ref.shape)
            check_array(a, key, actual, ref)
            print(f"{key}: {stats(actual, ref)}")
        else: a.errors.append(key + ': missing image embedding')
    M = len(gj["ids"])
    chunks = None
    nsteps = 1
    if a.chunk and a.chunk < M:
        chunks = []
        i = 0
        while i < M:
            rows = min(a.chunk, M - i)
            chunks.append((f"s{len(chunks)}_", rows))
            i += rows
        nsteps = len(chunks)
    pre_step = f"s{a.prefill_step}_"
    a.rows_saved = a.rows
    if a.prefill_step:  # the prefill step is a single-sequence dump
        a.rows = 0
    compare_step(a, g, gj, pre_step, f"prefill(chunk {a.chunk or M})", chunks, a.tail)
    a.rows = a.rows_saved
    step = 0
    while os.path.exists(os.path.join(a.golden, f"decode_{step:03d}.npz")):
        gd = np.load(os.path.join(a.golden, f"decode_{step:03d}.npz"))
        gdj = json.load(open(os.path.join(a.golden, f"decode_{step:03d}.json")))
        compare_step(a, gd, gdj, f"s{step + nsteps + a.decode_offset}_", f"decode step {step}")
        if 'logits' in gd.files:
            lp = os.path.join(a.dump, f's{step+nsteps+a.decode_offset}_logits.f32')
            ref = gd['logits'][-1]
            if os.path.exists(lp):
                rows=np.fromfile(lp,dtype=np.float32).reshape(-1,ref.size)
                actual=rows[a.row] if a.rows else rows[-1]
            else: actual=None
            check_array(a,f'decode {step} logits',actual,ref)
            if a.require_argmax and actual is not None and actual.argmax()!=ref.argmax(): a.errors.append(f'decode {step}: argmax differs')
        step += 1
    lp = os.path.join(a.dump, f"s{a.prefill_step + nsteps - 1}_logits.f32")
    if os.path.exists(os.path.join(a.golden, 'logits.npy')) and not os.path.exists(lp):
        a.errors.append('missing logits')
    if os.path.exists(lp) and os.path.exists(os.path.join(a.golden, "logits.npy")):
        gl = np.load(os.path.join(a.golden, "logits.npy"))[-1]
        el = np.fromfile(lp, dtype=np.float32)
        check_array(a, 'logits', el, gl)
        if a.require_argmax and el.argmax() != gl.argmax(): a.errors.append('logits argmax differs')
        p = np.exp(gl - gl.max()); p /= p.sum()
        q = np.exp(el - el.max()); q /= q.sum()
        kl = float((p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum())
        print(f"logits(last): {stats(el, gl)} · argmax golden {gl.argmax()} engine {el.argmax()} · KL(golden‖engine) {kl:.3e}")
    print(json.dumps({'ok': not a.errors, 'errors': a.errors, 'report_only': a.report_only}))
    return 0 if not a.errors or a.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
