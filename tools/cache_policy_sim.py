#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Offline cache policy simulation — replays an expert trace (expert-trace-*.bin) in **file order (= real time order)** and
compares the decode hit rate of VRAM slot policies at the same slot count.

Replay unit = a step (one 'S' record + per-layer 'L' (raw decode path) / 'H' (streaming prefill histogram)).
  · hit = fraction of 'L' accesses in decode-path steps (kind 0 batched decode · 2 verify · 3 draft) that are resident — same
    definition as hit/routed in the engine's [cache] log
  · prefill ('H') does not count hits; it only adds to scores and EMAs (the engine also observes during prefill)
Policies:
  lru   : global LRU, insert on every miss (recency reference value — not engine behaviour)
  hive  : old synchronous approximation (does not reproduce the current engine) — observe (score += 1) · after each decode-path
          step decay 0.97 → promote at most 4 (score ≥ 2, victim = lowest-scoring resident, only if the candidate is higher) ·
          after a prefill step warm_cache (if residency < 90 %, fill up to 2048 from the top entries with score ≥ 1)
  ema   : the residency policy of Dettmers "Runtime Dynamic Compression of MoE" (ICLR 2026) with miss = CPU (no masking) — two
          routing EMAs (half-lives 100 · 1000 steps, averaged) · every 256 steps ≤ 4 swaps per layer (incoming score > outgoing
          score × 1.05) · equal slots per layer (paper setting)
  ema_g : same, but a global pool (score order across layers, ≤ 4 × layers swaps per checkpoint)
  recN  : up to N experts missed in this decode step (by decayed frequency) become resident from the next step (assumes
          asynchronous promotion) · victim = least recently used.
          PCIe budget: expert 18.8 MB → N=16 is 300 MB per step ≈ 11 ms (28 GB/s) — the question is whether it fits in a side
          stream within the token time (~33 ms)
A replay of the exact engine policy is tools/cache_replay.py (builds expert_store.cpp unmodified · current policy vs
  HIVE_CACHE_POLICY · upper bound). 'hive' here is an old approximation (promote 4 · synchronous) and must not be used as the
  engine hit rate — this file is kept only for quick Python sketches of new policy ideas.
Usage: cache_policy_sim.py trace.bin [--slots 3792] [--policies lru,hive,ema,ema_g] [--max-steps N]
"""
import argparse, collections, struct
import numpy as np


def parse_steps(path, max_steps=0):
    buf = memoryview(open(path, "rb").read())
    n, off = len(buf), 0
    steps = []  # (kind, [(l, ids int64[M*k])], [(l, counts float64[E])])
    cur = None
    while off < n:
        tag = buf[off]
        if tag == 0x53:  # 'S'
            if off + 14 > n: break
            kind = buf[off + 1]
            M, nt = struct.unpack_from("<HH", buf, off + 2)
            if off + 14 + 8 * nt > n: break
            off += 14 + 8 * nt
            if cur is not None:
                steps.append(cur)
                if max_steps and len(steps) >= max_steps: return steps
            cur = (kind, [], [])
        elif tag == 0x4C:  # 'L'
            if off + 6 > n: break
            l = buf[off + 1]
            k, M = struct.unpack_from("<HH", buf, off + 2)
            if off + 6 + 2 * M * k > n: break
            ids = np.frombuffer(buf, dtype="<u2", count=M * k, offset=off + 6).astype(np.int64)
            off += 6 + 2 * M * k
            if cur is not None: cur[1].append((l, ids))
        elif tag == 0x48:  # 'H'
            if off + 8 > n: break
            l = buf[off + 1]
            (E,) = struct.unpack_from("<H", buf, off + 2)
            if off + 8 + 2 * E > n: break
            c = np.frombuffer(buf, dtype="<u2", count=E, offset=off + 8).astype(np.float64)
            off += 8 + 2 * E
            if cur is not None: cur[2].append((l, c))
        else:
            break
    if cur is not None: steps.append(cur)
    return steps


def next_use_tables(steps, L, E):
    """Per decode-path step, the list of keys used → per key, the array of steps that use it (future information for Belady)."""
    occ = collections.defaultdict(list)
    t = 0
    for kind, Ls, Hs in steps:
        if kind == 1: continue
        t += 1
        for l, ids in Ls:
            if l < L:
                for kk in np.unique(l * E + ids).tolist(): occ[kk].append(t)
    return {k: np.array(v) for k, v in occ.items()}


def sim(steps, policy, slots, L, E, eval_from=0.0, occ=None):
    K = L * E
    n_dec = sum(1 for s in steps if s[0] != 1)
    eval_t0 = int(n_dec * eval_from)  # count hits only from this decode step on (earlier steps are warm-up / training)
    dec_t = 0
    resident = np.zeros(K, bool)
    n_res = 0
    hit = tot = 0
    score = np.zeros(K)
    od = collections.OrderedDict()
    e1 = np.zeros(K); e2 = np.zeros(K)
    d1, d2 = 0.5 ** (1 / 100), 0.5 ** (1 / 1000)
    per_layer = slots // L
    t = 0
    last_use = np.zeros(K); score_rec = np.zeros(K)
    INF = 1 << 40
    nxt = np.full(K, INF, dtype=np.int64)  # Belady: next step that uses each key (after the current step)
    ptr = {}
    if policy.startswith("bel") and occ is not None:
        for kk, arr in occ.items(): nxt[kk] = arr[0]; ptr[kk] = 0
    Tr = np.zeros((L, E, E), np.float32); Cnt = np.zeros((L, E), np.float32); prev_sets = [None] * L  # per-layer step-to-step transitions (predictor)
    for kind, Ls, Hs in steps:
        decode_path = kind != 1
        if decode_path: dec_t += 1
        counting = decode_path and dec_t > eval_t0
        for l, ids in Ls:
            if l >= L: continue
            keys = l * E + ids
            if decode_path:
                if counting: tot += keys.size
                if policy == "lru":
                    for kk in keys.tolist():
                        if kk in od:
                            if counting: hit += 1
                            od.move_to_end(kk)
                        else:
                            od[kk] = True
                            if len(od) > slots: od.popitem(last=False)
                    continue
                if counting: hit += int(resident[keys].sum())
            if policy == "hive": np.add.at(score, keys, 1.0)
            elif policy != "lru": np.add.at(e1, keys, 1.0); np.add.at(e2, keys, 1.0)
        for l, c in Hs:
            if l >= L: continue
            sl = slice(l * E, (l + 1) * E)
            if policy == "hive": score[sl] += c[:E]
            elif policy != "lru": e1[sl] += c[:E]; e2[sl] += c[:E]
        if policy == "lru": continue
        # ---- additional policies: belN = recN with a clairvoyant victim (the one reused latest — theoretical upper bound) · predN = predict
        #      next-step use from per-layer step transition frequencies and pre-promote the top N non-residents (even without a miss),
        #      victim = lowest prediction · autoN = FreeToken-style bandwidth ratio (misses × 0.25, capped at N)
        if decode_path and (policy.startswith("bel") or policy.startswith("pred") or policy.startswith("auto")):
            t += 1
            for l, ids in Ls:
                if l < L: last_use[l * E + ids] = t
            miss = [l * E + ids[~resident[l * E + ids]] for l, ids in Ls if l < L]
            miss = np.unique(np.concatenate(miss)) if miss else np.zeros(0, np.int64)
            if policy.startswith("bel"):
                parts = [l * E + ids for l, ids in Ls if l < L]  # draft (MTP) steps may have no main-model layers
                used = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
                for kk in used.tolist():  # advance the next-use pointers (past the current t)
                    arr = occ.get(kk)
                    if arr is None: continue
                    p = ptr[kk]
                    while p < arr.size and arr[p] <= dec_t: p += 1
                    ptr[kk] = p; nxt[kk] = arr[p] if p < arr.size else INF
                P = int(policy[3:])
                cand = miss[np.argsort(nxt[miss])][:P]  # misses that will be reused soonest first
                for kk in cand:
                    if n_res < slots: resident[kk] = True; n_res += 1; continue
                    res = np.flatnonzero(resident)
                    v = res[np.argmax(nxt[res])]
                    if nxt[v] <= nxt[kk]: continue
                    resident[v] = False; resident[kk] = True
            elif policy.startswith("auto"):
                P = int(policy[4:])
                n_up = min(P, int(np.ceil(miss.size * 0.25)))  # same ratio as the engine (runtime promote_miss_ratio 0.25 ≈ PCIe 27.9 ÷ host ~110 GB/s)
                if n_up and miss.size:
                    cand = miss[np.argsort(-score_rec[miss])][:n_up]
                    for kk in cand:
                        if n_res < slots: resident[kk] = True; n_res += 1
                        else:
                            res = np.flatnonzero(resident)
                            v = res[np.argmin(last_use[res])]
                            resident[v] = False; resident[kk] = True
                score_rec *= 0.97
                for l, ids in Ls:
                    if l < L: np.add.at(score_rec, l * E + ids, 1.0)
            else:  # pred
                P = int(policy[4:])
                pred = np.zeros(K, np.float32)
                for l, ids in Ls:
                    if l >= L: continue
                    cur = np.unique(ids)
                    if prev_sets[l] is not None:  # learn transitions (online, causal): previous step's set → this step's set
                        pv = prev_sets[l]
                        Tr[l][np.ix_(pv, cur)] += 1.0; Cnt[l][pv] += 1.0
                    prev_sets[l] = cur
                    w_ = Tr[l][cur] / np.maximum(Cnt[l][cur], 1.0)[:, None]
                    pred[l * E:(l + 1) * E] = w_.sum(axis=0)
                pred += 1e-3 * score_rec  # tie-break: decayed frequency
                nr = np.flatnonzero(~resident)
                cand = nr[np.argsort(-pred[nr])][:P]
                res = np.flatnonzero(resident)
                order_out = res[np.lexsort((last_use[res], pred[res]))] if res.size else res
                oi = 0
                for kk in cand:
                    if n_res < slots: resident[kk] = True; n_res += 1; continue
                    if oi >= order_out.size: break
                    v = order_out[oi]
                    if pred[v] >= pred[kk]: break
                    resident[v] = False; resident[kk] = True; oi += 1
                score_rec *= 0.97
                for l, ids in Ls:
                    if l < L: np.add.at(score_rec, l * E + ids, 1.0)
            continue
        if policy.startswith("rec"):  # recN: asynchronously promote up to N of this decode step's misses (recently used = LRU victim). The miss itself runs on the CPU (as in the engine)
            P = int(policy[3:])
            if decode_path:
                t += 1
                for l, ids in Ls:
                    if l < L: last_use[l * E + ids] = t
                miss = [l * E + ids[~resident[l * E + ids]] for l, ids in Ls if l < L]
                miss = np.unique(np.concatenate(miss)) if miss else np.zeros(0, np.int64)
                if miss.size:
                    miss = miss[np.argsort(-score_rec[miss])][:P] if P < miss.size else miss
                    for kk in miss:
                        if n_res < slots: resident[kk] = True; n_res += 1
                        else:
                            res = np.flatnonzero(resident)
                            v = res[np.argmin(last_use[res])]
                            resident[v] = False; resident[kk] = True
                score_rec *= 0.97
                for l, ids in Ls:
                    if l < L: np.add.at(score_rec, l * E + ids, 1.0)
            continue
        if policy == "hive":
            if decode_path:
                score *= 0.97
                cand = np.flatnonzero(~resident & (score >= 2.0))
                if cand.size:
                    cand = cand[np.argsort(-score[cand])[:4]]
                    for kk in cand:
                        if n_res < slots: resident[kk] = True; n_res += 1; continue
                        res = np.flatnonzero(resident)
                        v = res[np.argmin(score[res])]
                        if score[v] >= score[kk]: break
                        resident[v] = False; resident[kk] = True
            elif n_res < 0.9 * slots:  # warm_cache
                cand = np.flatnonzero(~resident & (score >= 1.0))
                cand = cand[np.argsort(-score[cand])[:min(2048, slots - n_res)]]
                resident[cand] = True; n_res += cand.size
        elif policy in ("ema", "ema_g"):
            e1 *= d1; e2 *= d2
            t += 1
            if t % 256 == 1:  # checkpoint (from the first step — empty slots are filled without limit)
                sc = 0.5 * e1 + 0.5 * e2
                groups = [np.arange(l * E, (l + 1) * E) for l in range(L)] if policy == "ema" else [np.arange(K)]
                cap = per_layer if policy == "ema" else slots
                lim = 4 if policy == "ema" else 4 * L
                for g in groups:
                    r = g[resident[g]]; nr = g[~resident[g]]
                    order_in = nr[np.argsort(-sc[nr])]
                    free = cap - r.size
                    if free > 0:
                        resident[order_in[:free]] = True; order_in = order_in[free:]
                        r = g[resident[g]]
                    order_out = r[np.argsort(sc[r])]
                    nsw = 0
                    for ki, ko in zip(order_in, order_out):
                        if nsw >= lim or sc[ki] <= sc[ko] * 1.05: break
                        resident[ko] = False; resident[ki] = True; nsw += 1
    return (hit / tot if tot else float("nan")), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("--slots", type=int, default=3792)
    ap.add_argument("--policies", default="lru,hive,ema,ema_g"); ap.add_argument("--layers", type=int, default=40); ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-from", type=float, default=0.0, help="count hits only from this fraction of the decode steps on (0.5 = second half — compares without the predictor's learning share)")
    a = ap.parse_args()
    print('SIMULATION ONLY: cold-empty, synchronous historical model; no pending events, DMA contention, MTP cache integration or live-policy parity. Not measured engine hit rate/tok/s.', flush=True)
    steps = parse_steps(a.trace, a.max_steps)
    kinds = collections.Counter(s[0] for s in steps)
    print(f"steps {len(steps):,} {dict(kinds)} · slots {a.slots:,} · eval from {a.eval_from:.2f}", flush=True)
    occ = next_use_tables(steps, a.layers, a.experts) if any(p.startswith("bel") for p in a.policies.split(",")) else None
    for pol in a.policies.split(","):
        h, n = sim(steps, pol, a.slots, a.layers, a.experts, a.eval_from, occ)
        print(f"  {pol:6s} decode hit {h * 100:5.1f}%  (accesses {n:,})", flush=True)


if __name__ == "__main__":
    main()
