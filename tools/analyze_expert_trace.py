#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Expert usage trace (HIVE_EXPERT_TRACE) analysis — which patterns experts follow, which are used rarely and which heavily.

Record format = engine/include/hive/runtime.h (xtrace_*):
  'S' u8 kind · u16 M · u16 n · f64 unix_ms · n×(u32 uid, u32 pos)     kind 0 batched decode · 1 chunk/single · 2 verify · 3 draft
      if M >= 65535 (--prefill-tile >= 4), M=0xFFFF and the last tag is {0xFFFFFFFF, real M} (counted in n — older analyzers still read the length correctly)
  'L' u8 layer · u16 k · u16 M · M·k × u16                               decode policy path (M < prefill_threshold) — experts of row m = [m·k, (m+1)·k)
  'H' u8 layer · u16 E · u32 M · E × u16                                 per-layer use counts of a streaming prefill chunk

Usage:
  analyze_expert_trace.py <trace.bin> [--axes axes.tsv] [--slots 4183] [--lru-max 20000000] [--out DIR]
    axes.tsv = lines of "start_unix_s<TAB>end_unix_s<TAB>name" (benchmark axis intervals) — if given, reports per-axis (domain) distributions and differences.
Outputs (DIR, default = <name>.analysis/ next to the trace): report.md · counts_decode.npy · counts_prefill.npy [L,E] · by_axis.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

KIND = {0: "decode", 1: "chunk", 2: "verify", 3: "draft"}


def parse(path: Path, n_layers_hint: int = 43, E_hint: int = 384):
    """Scans the records and builds (raw decode access sequence, cumulative counts). The raw sequence is collected as numpy array pieces to save memory."""
    buf = memoryview(path.read_bytes())
    n = len(buf)
    off = 0
    cnt_dec = collections.defaultdict(lambda: np.zeros(512, np.int64))   # layer -> per expert (decode policy path, including draft/verify)
    cnt_pre = collections.defaultdict(lambda: np.zeros(512, np.int64))   # layer -> per expert (streaming prefill)
    cnt_kind = collections.Counter()
    # Per-row access sequence of batched decode (kind 0): sequence uid -> [(pos, layer, k experts)]
    class ChronologicalSteps(collections.defaultdict):
        pass
    seq_steps = ChronologicalSteps(list)
    seq_steps.events = []
    step_times: list[float] = []            # kind-0 step times (ms)
    step_layer_rows: list = []              # per kind-0 step (time, layer, ids[M,k]) — for axis splitting (the raw data is large, so only per-layer sums are kept)
    cur_kind, cur_M, cur_tags, cur_t = None, 0, [], 0.0
    axis_counts_raw: list = []              # (time, layer, expert array) — decode and prefill of all paths, assigned to axes by time
    n_rec = 0
    def need(k):  # stopping the daemon can cut the 1 MB buffer mid-way and truncate the last record — stop there
        return off + k <= n
    while off < n:
        tag = buf[off]
        if tag == 0x53:  # 'S'
            if not need(14) or not need(14 + 8 * struct.unpack_from("<H", buf, off + 4)[0]):
                print(f"(last record truncated @ {off} — using only what precedes it)", file=sys.stderr); break
            kind = buf[off + 1]
            M, nt = struct.unpack_from("<HH", buf, off + 2)
            (t,) = struct.unpack_from("<d", buf, off + 6)
            tags = np.frombuffer(buf, dtype="<u4", count=2 * nt, offset=off + 14).reshape(nt, 2) if nt else np.zeros((0, 2), np.uint32)
            if M == 0xFFFF and nt >= 1 and int(tags[-1, 0]) == 0xFFFFFFFF:  # B9: M >= 65535 has the end tag {0xFFFFFFFF, real M}
                M = int(tags[-1, 1]); tags = tags[:-1]
            off += 14 + 8 * nt
            cur_kind, cur_M, cur_tags, cur_t = kind, M, tags, t
            cnt_kind[KIND.get(kind, str(kind))] += 1
            if kind == 0:
                step_times.append(t)
        elif tag == 0x4C:  # 'L'
            if not need(6):
                break
            l = buf[off + 1]
            k, M = struct.unpack_from("<HH", buf, off + 2)
            if not need(6 + 2 * M * k):
                print(f"(last record truncated @ {off} — using only what precedes it)", file=sys.stderr); break
            ids = np.frombuffer(buf, dtype="<u2", count=M * k, offset=off + 6).astype(np.int32)
            off += 6 + 2 * M * k
            np.add.at(cnt_dec[l], ids, 1)
            axis_counts_raw.append((cur_t, l, np.bincount(ids, minlength=512)))
            if cur_kind == 0 and len(cur_tags) == M:
                rows = ids.reshape(M, k)
                for m in range(M):
                    seq_steps[int(cur_tags[m, 0])].append((int(cur_tags[m, 1]), l, rows[m]))
                    seq_steps.events.append((int(cur_tags[m, 1]), int(cur_tags[m, 0]), l, rows[m]))
        elif tag == 0x48:  # 'H'
            if not need(8):
                break
            l = buf[off + 1]
            (E,) = struct.unpack_from("<H", buf, off + 2)
            if not need(8 + 2 * E):
                print(f"(last record truncated @ {off} — using only what precedes it)", file=sys.stderr); break
            (M,) = struct.unpack_from("<I", buf, off + 4)
            c = np.frombuffer(buf, dtype="<u2", count=E, offset=off + 8).astype(np.int64)
            off += 8 + 2 * E
            cnt_pre[l][:E] += c
            axis_counts_raw.append((cur_t, l, np.pad(c, (0, 512 - E))))
        else:
            print(f"⚠️unknown record 0x{tag:02x} @ {off} — stopping here (the end of the file may be truncated)", file=sys.stderr)
            break
        n_rec += 1
    return cnt_dec, cnt_pre, cnt_kind, seq_steps, step_times, axis_counts_raw, n_rec


def to_mat(d, L, E):
    m = np.zeros((L, E), np.int64)
    for l, v in d.items():
        if l < L:
            m[l] = v[:E]
    return m


def coverage(row: np.ndarray, fracs=(0.5, 0.8, 0.95)):
    s = np.sort(row)[::-1]
    tot = s.sum()
    if tot == 0:
        return [0] * len(fracs)
    cs = np.cumsum(s) / tot
    return [int(np.searchsorted(cs, f) + 1) for f in fracs]


def gini(row: np.ndarray) -> float:
    x = np.sort(row.astype(float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    m = (p + q) / 2
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def reuse_stats(seq_steps, n_main_layers: int, max_gap: int = 16):
    """Reuse within a sequence: per layer, the probability that an expert of token t is used again at t+g, and the number of distinct experts touched by n consecutive tokens."""
    reuse = np.zeros(max_gap + 1)
    reuse_n = np.zeros(max_gap + 1)
    distinct = collections.defaultdict(list)  # n -> [distinct expert count per layer and window]
    for uid, steps in seq_steps.items():
        by_layer = collections.defaultdict(list)
        for pos, l, e in steps:
            if l < n_main_layers:
                by_layer[l].append((pos, e))
        for l, arr in by_layer.items():
            # Preserve recording order: sorting first hides reused uid epochs.
            # if pos goes backwards (session reuse = new request), split there
            runs, cur = [], []
            last = -1
            for pos, e in arr:
                if last >= 0 and pos != last + 1:
                    if cur:
                        runs.append(cur)
                    cur = []
                cur.append(set(int(x) for x in e))
                last = pos
            if cur:
                runs.append(cur)
            for run in runs:
                T = len(run)
                for g in range(1, max_gap + 1):
                    for t in range(0, T - g, max(1, (T - g) // 64)):
                        reuse[g] += len(run[t] & run[t + g]) / max(1, len(run[t]))
                        reuse_n[g] += 1
                for w in (1, 2, 3, 4, 5, 8, 16):
                    for t in range(0, T - w + 1, max(1, (T - w + 1) // 32)):
                        u = set()
                        for s in run[t:t + w]:
                            u |= s
                        distinct[w].append(len(u))
    reuse_p = {g: (reuse[g] / reuse_n[g] if reuse_n[g] else None) for g in range(1, max_gap + 1)}
    distinct_m = {w: (float(np.mean(v)) if v else None) for w, v in sorted(distinct.items())}
    return reuse_p, distinct_m


def lru_hit(seq_steps, slots: int, n_main_layers: int, E: int, limit: int):
    """Global LRU cache (layer/expert keys) simulation vs static top N — decode access sequence only (in time order)."""
    events = getattr(seq_steps, 'events', None)
    if events is None:
        raise ValueError('LRU needs chronological events, not uid/position sorted data')
    od = collections.OrderedDict()
    hit = tot = 0
    freq = collections.Counter()
    for pos, uid, l, e in events:
        if l >= n_main_layers: continue
        for x in e:
            key = int(l) * E + int(x)
            freq[key] += 1
            tot += 1
            if key in od:
                hit += 1
                od.move_to_end(key)
            else:
                od[key] = True
                if len(od) > slots:
                    od.popitem(last=False)
            if tot >= limit:
                break
        if tot >= limit:
            break
    top = sum(c for _, c in freq.most_common(slots))
    return (hit / tot if tot else None), (top / tot if tot else None), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--axes", help="axes file: lines of start_unix_s\\tend_unix_s\\tname")
    ap.add_argument("--slots", type=int, default=4183, help="VRAM expert slot count (for the LRU vs static top-N comparison)")
    ap.add_argument("--layers", type=int, default=40, help="main-model layer count (draft layers 40-42 are counted separately)")
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--lru-max", type=int, default=20_000_000)
    ap.add_argument("--out")
    a = ap.parse_args()
    tp = Path(a.trace)
    out = Path(a.out) if a.out else tp.with_suffix(".analysis")
    out.mkdir(parents=True, exist_ok=True)
    L, E = a.layers, a.experts
    cnt_dec, cnt_pre, cnt_kind, seq_steps, step_times, axis_raw, n_rec = parse(tp)
    D, P = to_mat(cnt_dec, L, E), to_mat(cnt_pre, L, E)
    np.save(out / "counts_decode.npy", D)
    np.save(out / "counts_prefill.npy", P)
    lines = [f"# Expert usage trace analysis — {tp.name}", "",
             f"- records {n_rec:,} · step kinds {dict(cnt_kind)} · decode batch steps {len(step_times):,}"
             + (f" · span {(step_times[-1] - step_times[0]) / 3.6e6:.2f} h" if len(step_times) > 1 else ""),
             f"- decode policy-path routings {D.sum():,} · streaming prefill routings {P.sum():,} (layers 0-{L - 1}, experts {E})", ""]

    for name, M in (("Decode (cache hit + CPU miss path)", D), ("Prefill (streaming)", P)):
        if M.sum() == 0:
            continue
        lines += [f"## {name}", "", "| layer | routings | experts covering 50%·80%·95% | unused experts | max/mean | Gini | top 5 (id:share%) |",
                  "| --: | --: | --- | --: | --: | --: | --- |"]
        for l in range(L):
            row = M[l]
            tot = row.sum()
            if tot == 0:
                continue
            c = coverage(row)
            top = np.argsort(row)[::-1][:5]
            lines.append(f"| {l} | {tot:,} | {c[0]} · {c[1]} · {c[2]} | {(row == 0).sum()} | {row.max() / row.mean():.1f} | {gini(row):.3f} | "
                         + " ".join(f"{int(e)}:{row[e] / tot * 100:.1f}" for e in top) + " |")
        flat = M.flatten()
        c = coverage(flat)
        lines += ["", f"- overall (layer×expert {L * E:,} cells): cells covering 50/80/95% = {c[0]:,} / {c[1]:,} / {c[2]:,} · unused cells {(flat == 0).sum():,} · Gini {gini(flat):.3f}", ""]

    if D.sum() and P.sum():
        per_layer = [jsd(D[l].astype(float), P[l].astype(float)) for l in range(L) if D[l].sum() and P[l].sum()]
        lines += [f"- decode ↔ prefill distribution difference (JSD, layer mean, 0 = identical · 1 = disjoint): {np.mean(per_layer):.3f}", ""]

    reuse_p, distinct_m = reuse_stats(seq_steps, L)
    if any(v is not None for v in reuse_p.values()):
        lines += ["## Reuse within a sequence (batch decode · per-layer mean)", "",
                  "| gap g | share of token t's experts used again at t+g |", "| --: | --: |"]
        for g, v in reuse_p.items():
            if v is not None and (g <= 8 or g == 16):
                lines.append(f"| {g} | {v * 100:.1f}% |")
        lines += ["", "| consecutive tokens n | distinct experts per layer (mean) | per token |", "| --: | --: | --: |"]
        for w, v in distinct_m.items():
            if v is not None:
                lines.append(f"| {w} | {v:.1f} | {v / w:.2f} |")
        lines += ["", "(reference: random routing gives 384·(1−(1−6/384)^n) per layer for n tokens — n=2 11.9 · 5 29.1 · 16 89.9)", ""]
        hit, top, tot = lru_hit(seq_steps, a.slots, L, E, a.lru_max)
        if tot:
            lines += [f"## Cache simulation (slots {a.slots:,} · decode accesses {tot:,} · approximate order, sequences concatenated)", "",
                      f"- global LRU hit {hit * 100:.1f}% · static top {a.slots:,} cells (hindsight-optimal fixed set) hit {top * 100:.1f}%", ""]

    if a.axes:
        axes = []
        for ln in Path(a.axes).read_text().splitlines():
            if ln.strip() and not ln.startswith("#"):
                s0, s1, name = ln.split("\t")[:3]
                axes.append((float(s0) * 1000, float(s1) * 1000, name))
        by = {name: np.zeros((L, 512), np.int64) for _, _, name in axes}
        for t, l, c in axis_raw:
            if l >= L:
                continue
            for s0, s1, name in axes:
                if s0 <= t <= s1:
                    by[name][l] += c
                    break
        names = [n for n in by if by[n].sum()]
        json.dump({n: by[n][:, :E].tolist() for n in names}, open(out / "by_axis.json", "w"))
        if names:
            lines += ["## By axis (domain)", "", "| axis | routings | cells covering 95% | JSD vs overall distribution |", "| --- | --: | --: | --: |"]
            allm = sum(by[n] for n in names)
            for n in names:
                f = by[n][:, :E].flatten()
                lines.append(f"| {n} | {f.sum():,} | {coverage(f)[2]:,} | {jsd(f.astype(float), allm[:, :E].flatten().astype(float)):.3f} |")
            lines += ["", "JSD between axes (layer-summed distributions):", ""]
            for i, x in enumerate(names):
                for y in names[i + 1:]:
                    lines.append(f"- {x} ↔ {y}: {jsd(by[x][:, :E].flatten().astype(float), by[y][:, :E].flatten().astype(float)):.3f}")
            lines.append("")
    report = "\n".join(lines)
    (out / "report.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n→ {out}/report.md")


if __name__ == "__main__":
    main()
