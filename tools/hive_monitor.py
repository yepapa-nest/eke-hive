#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""hive_monitor.py — incremental hived.log parser → per-request JSONL + per-minute engine metrics JSONL.

Purpose: keep and summarize everything later performance work needs once traffic is routed to the local engine (hive).
This file **only reads the log** (engine and server unmodified). Daily summaries: hive_daily_report.py.

Input: hived.log (the container supervisor runs `hived ... >> /out/logs/hived.log 2>&1` — one fd opened with O_APPEND, no timestamps).
Output (default = next to the log):
  requests/YYYY-MM-DD.jsonl   one line per request (local date of its end time)
  metrics/YYYY-MM-DD.jsonl    one line per minute (engine samples: [cache]·[profile]·[decode-host]·[decode-sync]·[decode-miss]·[early-route]·[mtp]·
                              elastic reclaim·snapshot·warm·batch window·errors·restarts·sleep/wake)
  monitor/state.json          inode + logical offset + parser state (open requests, in-progress minute bucket) — survives restarts and rotation

Timestamps: hived.log contains **no timestamps at all** (supervisor lines included). A line's time is therefore the "read time" linearly
  interpolated by byte position (previous read time t0 at offset o0 ↔ current read time t1 at offset o1). The error is <= t1−t0, so running
  resident with --follow --interval 2 gives ±2 s. A request's start and first-token times are the final line's interpolated time minus the
  durations in the line (prefill ms, decode ms) (the durations themselves come from the engine clock and are exact). If there is no previous
  read or it is older than --max-interp seconds (default 900), ts_quality="backfill" (time = current read time, only the date is meaningful).
  → When the engine appends wall-clock fields to the request line (" · wall t0=… t1=… t2=… queue_ms=… rid=…"), those are used instead.

Rotation: an external log rotator cuts the front of hived.log with `fallocate --collapse-range`, moves it to an archive and
  records {inode, mode, logical_start, cut, archive} in <log>.rotations.jsonl. This parser advances by a per-inode **logical offset** (total
  bytes cut + position in the current file), so even if it falls behind a rotation it continues reading the cut range from the archive (.gz
  included). Rotation and reading are serialized with a flock on <log>.lock (so the parser never sees the moment right after a cut before the
  journal entry exists).

usage:
  hive_monitor.py [--log PATH] [--out DIR] [--state PATH] [--once | --follow --interval 2] [--max-bytes N]
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import math
import os
import re
import sys
import time
from datetime import datetime

# $HIVE_STATE_DIR/logs/hived.log, otherwise <repo>/run/logs/hived.log (the default state dir of scripts/hive-start.sh, see scripts/lib.sh)
DEFAULT_LOG_DIR = os.path.join(os.environ.get("HIVE_STATE_DIR") or os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir)), "run"), "logs")
DEFAULT_LOG = os.path.join(DEFAULT_LOG_DIR, "hived.log")
SCHEMA = 1

# ---------------------------------------------------------------------------------------------------------------------
# Line formats (engine printf sites — from reading hived.cpp/runtime.cpp, cross-checked against the shapes of 200K real log lines)
# ---------------------------------------------------------------------------------------------------------------------
N = r"(-?[\d.]+)"
LY_TAIL = re.compile(r" · layer yields (\d+) last ([\d.]+) ms")  # T11 tail of chunk/round lines (if present, the gap = last — earlier segments were already counted by the yield head line)
RE = {
    # hived.cpp finish_active: request finished (the mtp clause is absent in older builds)
    "final": re.compile(r"^\[hived\] (\S+): prefill (\d+) tok ([\d.]+) ms · decode (\d+) tok ([\d.]+) ms \(([\d.]+) tok/s, batch (\d+)\) · "
                        r"hit (\d+) cpu (\d+) · resident (\d+)/(\d+)(?: · mtp (\d+)/(\d+) in (\d+) steps)? · (\S+)(?: · (.*?))?\s*$"),
    "reuse": re.compile(r"^\[hived\] (\S+): reuse (\d+)/(\d+) via (\S+) \((.*)\)\s*$"),
    "chunk": re.compile(r"^\[hived\] (\S+): prefill chunk (\d+) M (\d+) · (\d+)→(\d+)/(\d+) · upper (\S+) · tail rows (\d+) · "
                        r"hit (\d+) streamed (\d+) cpu (\d+) dma_rows (\d+) · cpu wait ([\d.]+) ms span ([\d.]+) ms · ([\d.]+) ms"),
    "ptotal": re.compile(r"^\[hived\] (\S+): prefill total (\d+) rows in (\d+) chunks · hit (\d+) streamed (\d+) cpu (\d+) dma_rows (\d+) · "
                         r"cpu wait ([\d.]+) ms span ([\d.]+) ms · ([\d.]+) ms"),
    "bround": re.compile(r"^\[hived\] batch prefill round (\d+): (\d+) sequences \(([^)]*)\) · (\d+) rows · slots (\d+)/(\d+) · "
                         r"hit (\d+) streamed (\d+) cpu (\d+) dma_rows (\d+) · ([\d.]+) ms"),
    "bfail": re.compile(r"^\[hived\] batch prefill failed: (.*) — (\d+) requests retried"),
    "pending": re.compile(r"^\[hived\] (\S+): prefill_pending=(\d+)"),
    "pfxsnap": re.compile(r"^\[hived\] (\S+): prefix snapshot at (\d+) \(upper (\S+) · (\d+) entries\)"),
    "ckptdrop": re.compile(r"^\[hived\] (\S+): deferred prompt checkpoint dropped(.*)$"),
    "reqfail": re.compile(r"^\[hived\] request failed: (.*)$"),
    "reqfail_sid": re.compile(r"^\[hived\] (\S+): request failed: (.*)$"),  # engine request-line format with sid — attribution is certain, not guessed
    "decfail": re.compile(r"^\[hived\] decode failed: (.*)$"),
    "window": re.compile(r"^\[hived\] batch window: \+(\d+) requests in ([\d.]+) ms \((\S+) · (\d+) jobs · slots left (-?\d+)\)"),
    "yield": re.compile(r"^\[hived\] prefill yield: (\d+) short requests? admitted in ([\d.]+) ms"),
    # HIVE_LAYER_YIELD (T11): layer yielding inside a long prefill forward — head line (the preceding prefill segment = gap), done line, tail of chunk/round lines (last resume → end of forward)
    "lyield": re.compile(r"^\[hived\] layer yield: (\S+) · prefill ([\d.]+) ms since resume"),
    "lydone": re.compile(r"^\[hived\] layer yield done: admitted (\d+) · decode steps (\d+) · ([\d.]+) ms"),
    "warm": re.compile(r"^\[hived\] warm cache: (\d+) experts in ([\d.]+) ms"),
    "warmdefer": re.compile(r"^\[runtime\] deferred warm: (\d+) experts over (\d+) steps"),  # HIVE_WARM_DEFER: counted as a warm run with no synchronous time
    "warmstart": re.compile(r"^\[hived\] warm start: (\d+)/(\d+) experts from \S+ in ([\d.]+) ms"),
    "ready": re.compile(r"^\[hived\] ready in ([\d.]+)s"),
    "supervisor": re.compile(r"^\[supervisor\] (?:★ )?build dir=(\S+) binary=\S+ sha256=(\S+) elf-build-id=(\S+)"),
    "child_exit": re.compile(r"^\[supervisor\] child exited status=(\d+)"),
    "sleep": re.compile(r"^\[hived\] sleep: ([\d.]+) ms \(drain ([\d.]+) ms\) · freed ([\d.]+) MiB"),
    "wake": re.compile(r"^\[hived\] wake: ([\d.]+) ms \(alloc ([\d.]+) · warm ([\d.]+)\) · slots (\d+)"),
    "lifecycle_err": re.compile(r"^\[hived\] (sleep failed|wake failed|wake: warm stopped|graceful stop: drain deadline)(.*)$"),
    "graceful": re.compile(r"^\[hived\] graceful stop(?: complete)?(.*)$"),
    # runtime.cpp
    "cache": re.compile(r"^\[cache\] step (\d+) M=(\d+) routed (\d+) hit (\d+) cpu (\d+) streamed (\d+) · resident (\d+) pending (\d+) / (\d+)(.*)$"),
    "profile": re.compile(r"^\[profile M=(\d+)\] total ([\d.]+) ms:(.*)$"),
    "dhost": re.compile(r"^\[decode-host M=(\d+)( verify)?\] (.*)$"),
    "dsync": re.compile(r"^\[decode-sync M=(\d+)\] layer waits (\d+) · host wait ([\d.]+) ms"),
    "dmiss": re.compile(r"^\[decode-miss M=(\d+)\] (.*)$"),
    "eroute": re.compile(r"^\[early-route M=(\d+)\] layers (\d+) · host ahead of front end (\d+) · (?:table on own stream (\d+) · )?"
                         r"absorbed (\d+) \(resync (\d+) · missing (\d+) · untrusted (\d+)\)"),
    "elastic": re.compile(r"^\[runtime\] elastic reclaim #(\d+): evicted (\d+) · dropped promotions (\d+) \(total\) · ([\d.]+) ms · slots (\d+) → (\d+)"),
    "snapshot": re.compile(r"^\[snapshot\] pos (\d+) d2h_bytes (\d+) host_bytes (\d+) async (\d+) delta (\d+)"),
    "stepgraph": re.compile(r"^\[step-graph M=(\d+)\] .*?wait ([\d.]+) · replay ([\d.]+) ms.*?gpu span ([\d.]+) ms"),
    "pprof": re.compile(r"^\[prefill-prof M=(\d+) pos=(\d+) tail=(\d+)\] wall ([\d.]+) ms"),
    # HIVE_PROFILE host-time lines (hived.cpp [step-host] every N steps of a kind · GLM runtime [call-host] every N calls · GLM engine [fwd-host] on sample steps)
    "step_host": re.compile(r"^\[step-host ([\w-]+)\] steps (\d+) · gap ([\d.]+) \(n (\d+)\) · pre (-?[\d.]+) · draft ([\d.]+) · fwd ([\d.]+) · post ([\d.]+) ms/step"),
    "call_host": re.compile(r"^\[call-host ([\w-]+)\] calls (\d+) · rows ([\d.]+) · eng ([\d.]+) · logits ([\d.]+) · cands ([\d.]+) · rest (-?[\d.]+) ms/call"),
    "glm_prefill": re.compile(r"^\[glm-prefill T=(\d+) pos=(\d+)\] wall ([\d.]+) · yield ([\d.]+) · moe prep ([\d.]+) gpu ([\d.]+) cpu-join ([\d.]+) · "
                             r"attn kda ([\d.]+) dsa ([\d.]+) · other (-?[\d.]+) ms · rows hit (\d+) cpu (\d+) · streamed (\d+) \(([\d.]+) GB"),
    "glm_ckpt": re.compile(r"^\[glm-ckpt\] pos (\d+) base (-?\d+) · copied ([\d.]+) of ([\d.]+) MB · ([\d.]+) ms"),
    "glm_ckpt_verify": re.compile(r"^\[glm-ckpt\] verify pos (\d+) base (\d+) · shared ([\d.]+) MB · (.*)$"),
    "fwd_host": re.compile(r"^\[fwd-host M=(\d+)( verify)?\] host ([\d.]+) · entry ([\d.]+) · window ([\d.]+) · head ([\d.]+) ms"),
    # hived.cpp MTP(HIVE_TRACE_MTP)
    "mtp_pos": re.compile(r"^\[mtp\] pos (\d+) draft ([\d.]+) ms · verify (\d+) rows ([\d.]+) ms · accepted (\d+)/(\d+)"),
    "mtp_nodraft": re.compile(r"^\[mtp\] pos (\d+) draft ([\d.]+) ms · (?:gate2 )?no draft"),
    "mtp_gate2": re.compile(r"^\[mtp\] gate2 k (\d+) · expect ([\d.]+) tok / ([\d.]+) ms \(T1 ([\d.]+)\) · got (\d+) tok / ([\d.]+) ms · net (-?[\d.]+) ms"),
    "mtp_g2T1": re.compile(r"T1 ([\d.]+)"),
    "mtp_batch": re.compile(r"^\[mtp\] batch S (\d+) rows (\d+) · accepted \[([^\]]*)\] · expect ([\d.]+) tok / ([\d.]+) ms \(T\(S\) ([\d.]+)\) · "
                            r"got (\d+) tok / ([\d.]+) ms · draft ([\d.]+) ms · net (-?[\d.]+) ms"),
    "mtp_batch_nodraft": re.compile(r"^\[mtp\] batch S (\d+) draft ([\d.]+) ms · no draft"),
    "oom": re.compile(r"^cudaMalloc\(([\d.]+) MiB\) failed: out of memory"),
    # HIVE_DECODE_PREGATE (runtime.cpp finish(l), every sampled step): values are **cumulative since startup** (pg::Acc, store pf_stats — never reset)
    #   → the parser adds the difference from the previous line to the minute bucket (the baseline is cleared on a restart banner). recall/precision are recomputed from the sums (no mean of means).
    "pregate": re.compile(r"^\[pregate M=(\d+)\] layers (\d+) · posted (\d+) late (\d+) untrusted (\d+) · prefetch experts (\d+) · cpu experts (\d+) · "
                          r"covered (\d+) \(recall [\d.]+ · precision [\d.]+\) · pool req (\d+) experts (\d+) read ([\d.]+) MiB · finished (\d+) aborted (\d+)"),
    "pregate_on": re.compile(r"^\[runtime\] decode pregate on \(top-(\d+) before attention"),
}
PG_FIELDS = ("layers", "posted", "late", "untrusted", "pred", "actual", "covered", "pool_req", "pool_experts", "read_mib", "finished", "aborted")
RE_DMA = re.compile(r"dma\.span ([\d.]+) dma\.wait ([\d.]+|-) ms \((\d+) layers\)")  # wait "-" = no copy was waited on in that step (an observed variant)
RE_PROMO = re.compile(r"promo ([\d.]+) MiB")
RE_VFREE = re.compile(r"vram free (\d+) MiB")
RE_EXTRA = re.compile(r"(\w+)=(\S+)")  # the "· wall t0=… t1=… t2=… queue_ms=… rid=…" fields the engine appends to the request line
RE_STEP_WALL = re.compile(r"· wall ([\d.]+) ms")  # per-step wall-clock ms on the [cache] line
RE_KV = re.compile(r"([A-Za-z][\w.\-]*) (-?[\d.]+)")
RE_DH = {k: re.compile(p) for k, p in {
    "sync": r"sync ([\d.]+)", "prep": r"· prep ([\d.]+)", "launch": r"launch ([\d.]+)", "cpu": r"cpu ([\d.]+) vs gpu ([\d.]+)",
    "cpu_bound": r"cpu-bound layers (\d+)/(\d+)", "tail": r"tail ([\d.]+) ms", "front": r"front ([\d.]+)",
    "gpu_idle": r"gpu-idle ([\d.]+) of ([\d.]+) ms", "cpu_layers": r"\(cpu layers (\d+)\)",
    "promo_wait": r"promo wait ([\d.]+) ms/step",  # DeepSeek: step-head commit wait for paced promotions, mean per step since the previous line
    "defer_wait": r"defer wait ([\d.]+)",
    "next": r"· next ([\d.]+)"}.items()}     # GLM: host time blocked on deferred CPU experts in this sample step
RE_SHAPE_NUM = re.compile(r"[0-9a-f]{8,}|\d+(?:\.\d+)?")

DIAG_PREFIXES = ("[decode-host", "[decode-sync", "[decode-miss", "[decode-split", "[early-route", "[step-graph", "[pregate")
DECODE_MARKERS = ("cache", "mtp_pos", "mtp_nodraft", "mtp_batch", "mtp_batch_nodraft")
LONG_PREFILL = 16384  # stall attribution threshold (decode gaps of other requests while a >= 16K prefill runs)


# ---------------------------------------------------------------------------------------------------------------------
# Mergeable log histograms (minute buckets → summed by the daily report, 5 % resolution)
# ---------------------------------------------------------------------------------------------------------------------
HB = math.log(1.05)


def hist_add(h: dict, x: float, n: int = 1) -> None:
    k = "z" if x <= 0 else str(int(math.floor(math.log(x) / HB)))
    h[k] = h.get(k, 0) + n


def hist_merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = a.get(k, 0) + v
    return a


def hist_quantile(h: dict, q: float):
    tot = sum(h.values())
    if not tot:
        return None
    keys = sorted(h, key=lambda k: -1e9 if k == "z" else int(k))
    acc, want = 0, q * tot
    for k in keys:
        acc += h[k]
        if acc >= want:
            return 0.0 if k == "z" else round(math.exp((int(k) + 0.5) * HB), 3)
    return None


def local_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="milliseconds")


def local_day(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def local_minute(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M")


def fnum(s):
    return float(s) if "." in s else int(s)


# ---------------------------------------------------------------------------------------------------------------------
# Minute buckets
# ---------------------------------------------------------------------------------------------------------------------
def new_bucket(minute: str) -> dict:
    return {
        "minute": minute, "lines": 0, "ts_quality": "interp",
        # [cache] step (one decode forward_batch)
        "decode_steps": 0, "routed": 0, "hit": 0, "cpu": 0, "streamed": 0, "batch_hist": {}, "conc_hist": {},
        "resident_last": None, "slots_last": None, "pending_max": 0, "vram_free_min": None,
        "dma_span_ms": 0.0, "dma_wait_ms": 0.0, "dma_samples": 0, "dma_wait_samples": 0, "promo_mib": 0.0,
        # [profile M=k] total — decode (followed by [cache]) / MTP draft (mtp.*) / other (prefill, verify)
        "step_ms_hist": {}, "step_ms_by_m": {}, "prof_other_n": 0, "prof_mtp_n": 0, "prof_sections_ms": {}, "prof_decode_n": 0,
        "prof_verify_n": 0, "verify_ms_hist": {}, "step_wall_ms_hist": {}, "step_wall_by_m": {},
        # [decode-host]
        "dh_n": 0, "dh_verify_n": 0, "dh_gpu_idle_ms": 0.0, "dh_of_ms": 0.0, "dh_cpu_ms": 0.0, "dh_gpu_ms": 0.0,
        "dh_cpu_bound_layers": 0, "dh_layers": 0, "dh_tail_ms": 0.0, "dh_front_ms": 0.0, "gpu_idle_pct_hist": {},
        "dh_promo_wait_n": 0, "dh_promo_wait_ms": 0.0, "dh_defer_wait_ms": 0.0, "dh_sync_ms": 0.0, "dh_next_ms": 0.0,
        # HIVE_PROFILE host time — sums (a line's means × its count): [step-host kind] · [call-host kind] · [fwd-host]
        "step_host": {}, "call_host": {}, "fwd_host": {"n": 0, "host_ms": 0.0, "entry_ms": 0.0, "window_ms": 0.0, "head_ms": 0.0},
        # [glm-prefill] per context bucket (pos + T at the end of the call): sums
        "glm_prefill": {},
        # [glm-ckpt] prompt / boundary / archive images (GLM save_image): count, with a shared prefix, host ms, copied and image MB · verify lines
        "ckpt": {"n": 0, "delta_n": 0, "ms": 0.0, "ms_max": 0.0, "copied_mb": 0.0, "total_mb": 0.0, "verify_n": 0, "verify_bad": 0},
        "dsync_n": 0, "dsync_wait_ms": 0.0,
        "dmiss_n": 0, "dmiss": {},
        "eroute": {"layers": 0, "ahead": 0, "own_stream": 0, "absorbed": 0, "resync": 0, "missing": 0, "untrusted": 0},
        "pregate": {"n": 0, **{k: 0 for k in PG_FIELDS}},  # P1 pre-gate (sum of differences from the previous line; only read_mib is a float)
        # prefill
        "prefill_chunks": 0, "prefill_chunk_ms": 0.0, "prefill_rows": 0, "batch_rounds": 0, "batch_round_seqs": 0,
        "batch_windows": 0, "batch_window_ms": 0.0, "yield_admits": 0,
        "layer_yields": 0, "layer_yield_steps": 0, "layer_yield_ms": 0.0,  # HIVE_LAYER_YIELD (T11)
        # decode gaps during long prefills (lower bound of prefill compute ms between another request's decode steps)
        "stall_n": 0, "stall_ms_max": 0.0, "stall_ms_sum": 0.0, "stall_hist": {},
        # MTP
        "mtp_steps": 0, "mtp_nodraft": 0, "mtp_drafted": 0, "mtp_accepted": 0, "mtp_tokens": 0, "mtp_draft_ms": 0.0, "mtp_verify_ms": 0.0,
        "mtp_saved_ms": 0.0, "mtp_gate2_n": 0, "mtp_gate2_net_sum": 0.0, "mtp_batch_n": 0, "mtp_batch_nodraft": 0,
        # misc
        "reclaims": 0, "reclaim_evicted": 0, "reclaim_ms": 0.0, "snapshots": 0, "snapshot_d2h_bytes": 0,
        "warm_n": 0, "warm_experts": 0, "warm_ms": 0.0, "stepgraph_n": 0, "stepgraph_wait_ms": 0.0,
        "requests_done": 0, "requests_started": 0, "errors": [], "events": [], "warnings": 0, "unparsed": {},
    }


# ---------------------------------------------------------------------------------------------------------------------
# Parser (all state is JSON-serializable — stored as is in state.json)
# ---------------------------------------------------------------------------------------------------------------------
class Parser:
    def __init__(self, state: dict | None = None):
        s = state or {}
        self.open: dict = s.get("open", {})            # sid → request record (in progress)
        self.seq: dict = s.get("seq", {})              # sid → number of requests opened so far
        self.bucket: dict | None = s.get("bucket")
        self.gap_ms: float = s.get("gap_ms", 0.0)      # prefill compute ms accumulated since the last decode marker
        self.gap_sids: list = s.get("gap_sids", [])
        self.round_sids: list = s.get("round_sids", [])  # sids of the previous batch round (prevents double-counting the ms repeated by the following chunk lines)
        self.pending_prof: dict | None = s.get("pending_prof")
        self.t1: float | None = s.get("t1")            # latest single-step cost (T1) of MTP gate2
        self.build: dict = s.get("build", {})
        self.pg_last: list | None = s.get("pg_last")  # cumulative values of the previous [pregate] line (for differences; None after a restart)
        self.counts: dict = s.get("counts", {"lines": 0, "requests": 0, "unparsed": 0})
        self.out_requests: list = []
        self.out_metrics: list = []

    def state(self) -> dict:
        return {"open": self.open, "seq": self.seq, "bucket": self.bucket, "gap_ms": self.gap_ms, "gap_sids": self.gap_sids,
                "round_sids": self.round_sids, "pending_prof": self.pending_prof, "t1": self.t1, "build": self.build, "pg_last": self.pg_last, "counts": self.counts}

    # -- buckets ---------------------------------------------------------------------------------------------------------
    def _b(self, ts: float, quality: str) -> dict:
        m = local_minute(ts)
        if self.bucket is None or self.bucket["minute"] != m:
            if self.bucket is not None and self.bucket["minute"] > m:
                return self.bucket  # interpolated times never go backwards (monotone), but defensively: put it in the previous bucket
            self.flush_bucket()
            self.bucket = new_bucket(m)
        if quality != "interp":
            self.bucket["ts_quality"] = quality
        return self.bucket

    def flush_bucket(self) -> None:
        if self.bucket is not None and self.bucket["lines"]:
            self.out_metrics.append(self.bucket)
        self.bucket = None

    # -- requests ---------------------------------------------------------------------------------------------------------
    def _open(self, sid: str, ts: float, loff: int) -> dict:
        r = self.open.get(sid)
        if r is None:
            n = self.seq.get(sid, 0) + 1
            self.seq[sid] = n
            r = {"id": f"{sid}#{n}", "sid": sid, "seq": n, "phase": "prefill", "first_seen_ts": ts, "first_loff": loff,
                 "prompt_tokens": None, "cached_prefix_tokens": 0, "reuse_via": None, "reuse_detail": None,
                 "prefill_chunks": 0, "prefill_rows": 0, "prefill_chunk_ms": 0.0, "prefill_batch_rounds": 0,
                 "dec_steps": 0, "dec_m_sum": 0, "dec_m_max": 0, "conc_start": len(self.open) + 1, "conc_max": len(self.open) + 1,
                 "stall_caused_n": 0, "stall_caused_ms": 0.0, "stall_caused_max_ms": 0.0, "stalled_max_ms": 0.0, "stalled_sum_ms": 0.0,
                 "errors": [], "prefix_snapshots": 0, "build": self.build.get("sha256")}
            self.open[sid] = r
            for o in self.open.values():
                o["conc_max"] = max(o["conc_max"], len(self.open))
            self._b(ts, "interp")["requests_started"] += 1
        return r

    def _close(self, r: dict, ts: float, quality: str, err_s: float, finish: str, final: dict | None, loff: int) -> None:
        self.open.pop(r["sid"], None)
        rec = {"schema": SCHEMA, "id": r["id"], "sid": r["sid"], "seq": r["seq"], "finish": finish, "ts_quality": quality,
               "ts_err_s": round(err_s, 2), "end_ts": round(ts, 3), "end": local_iso(ts), "loff_end": loff, "loff_start": r["first_loff"],
               "build": r.get("build")}
        pt = r["prompt_tokens"]
        rec["cached_prefix_tokens"] = r["cached_prefix_tokens"]
        rec["reuse_via"] = r["reuse_via"]
        if final:
            rec.update(final)
            if pt is None:
                pt = r["cached_prefix_tokens"] + final["prefill_tokens"]
            dur = (final["prefill_ms"] + final["decode_ms"]) / 1000.0
            rec["start_ts"] = round(ts - dur, 3)
            rec["start"] = local_iso(ts - dur)
            rec["first_token_ts"] = round(ts - final["decode_ms"] / 1000.0, 3)
            rec["ttft_ms"] = final["prefill_ms"]  # engine t0 (prefill start after admission) → t1 (first token) — queueing is not in the log
            if final.get("queue_ms") is not None:  # with the engine's wall fields: daemon receipt → first token
                rec["ttft_with_queue_ms"] = round(final["queue_ms"] + final["prefill_ms"], 1)
            rec["prefill_tok_s"] = round(final["prefill_tokens"] * 1000.0 / final["prefill_ms"], 1) if final["prefill_ms"] > 0 else None
            tot = final["decode_hit"] + final["decode_cpu"]
            rec["decode_cpu_share"] = round(final["decode_cpu"] / tot, 4) if tot else None
            if final.get("mtp_drafted"):
                rec["mtp_accept_rate"] = round(final["mtp_accepted"] / final["mtp_drafted"], 4)
        else:
            rec["start_ts"] = round(r["first_seen_ts"], 3)
            rec["start"] = local_iso(r["first_seen_ts"])
        rec["prompt_tokens"] = pt
        for k in ("prefill_chunks", "prefill_rows", "prefill_batch_rounds", "prefix_snapshots", "conc_start", "conc_max",
                  "stall_caused_n", "stalled_max_ms", "stalled_sum_ms", "stall_caused_max_ms"):
            rec[k] = r[k]
        rec["stall_caused_ms"] = round(r["stall_caused_ms"], 1)
        rec["prefill_chunk_ms"] = round(r["prefill_chunk_ms"], 1)
        for k in ("prefill_hit", "prefill_streamed", "prefill_cpu", "prefill_dma_rows", "prefill_cpu_wait_ms", "prefill_cpu_span_ms",
                  "prefill_total_ms", "prefill_shared_round"):
            if k in r:
                rec[k] = r[k]
        rec["decode_steps_seen"] = r["dec_steps"]
        rec["decode_batch_avg"] = round(r["dec_m_sum"] / r["dec_steps"], 2) if r["dec_steps"] else None
        rec["decode_batch_max"] = r["dec_m_max"] or None
        rec["errors"] = r["errors"]
        self.out_requests.append(rec)
        self.counts["requests"] += 1
        self._b(ts, quality)["requests_done"] += 1

    def _abandon_all(self, ts, quality, err_s, why, loff, only_phase=None):
        for r in list(self.open.values()):
            if only_phase and r["phase"] != only_phase:
                continue
            self._close(r, ts, quality, err_s, why, None, loff)

    # -- decode progress markers (stall measurement) -----------------------------------------------------------------------------------
    def _decode_mark(self, b: dict, m: int) -> None:
        decoding = [r for r in self.open.values() if r["phase"] == "decode"]
        if self.gap_ms > 0 and decoding:
            owners = [self.open[s] for s in self.gap_sids if s in self.open]
            longp = [o for o in owners if (o["prompt_tokens"] or 0) >= LONG_PREFILL]
            victims = [r for r in decoding if r["sid"] not in self.gap_sids]
            if victims:
                g = round(self.gap_ms, 1)
                if longp:  # only gaps during a >= 16K prefill count as stalls (short-prefill gaps go to the bucket hist only)
                    b["stall_n"] += 1
                    b["stall_ms_sum"] += g
                    b["stall_ms_max"] = max(b["stall_ms_max"], g)
                    for o in longp:
                        o["stall_caused_n"] += 1
                        o["stall_caused_ms"] += g
                        o["stall_caused_max_ms"] = max(o["stall_caused_max_ms"], g)
                    for v in victims:
                        v["stalled_max_ms"] = max(v["stalled_max_ms"], g)
                        v["stalled_sum_ms"] = round(v["stalled_sum_ms"] + g, 1)
                hist_add(b["stall_hist"], g)
        self.gap_ms = 0.0
        self.gap_sids = []
        self.round_sids = []
        b["conc_hist"][str(len(self.open))] = b["conc_hist"].get(str(len(self.open)), 0) + 1
        for r in decoding:
            r["dec_steps"] += 1
            r["dec_m_sum"] += m
            r["dec_m_max"] = max(r["dec_m_max"], m)

    def _gap(self, sid_list, ms):
        self.gap_ms += ms
        for s in sid_list:
            if s not in self.gap_sids:
                self.gap_sids.append(s)

    # -- one line ----------------------------------------------------------------------------------------------------------
    def feed(self, line: str, ts: float, loff: int, quality: str = "interp", err_s: float = 0.0) -> None:
        self.counts["lines"] += 1
        b = self._b(ts, quality)
        b["lines"] += 1
        # Classification of the previous [profile] line (observed order: [decode-sync] → [profile] → [early-route]/[decode-miss]/[decode-host] → [cache] step).
        #   Diagnostic lines are skipped and the next "real" line decides: [cache] step = decode step (M), [mtp] = MTP verify step (M = verify rows), mtp.* sections = draft, anything else = prefill etc.
        pp = self.pending_prof
        if pp is not None and not line.startswith(DIAG_PREFIXES):
            self.pending_prof = None
            if pp["mtp"]:
                b["prof_mtp_n"] += 1
            elif line.startswith("[cache] step"):
                b["prof_decode_n"] += 1
                hist_add(b["step_ms_hist"], pp["ms"])
                hist_add(b["step_ms_by_m"].setdefault(str(pp["m"]), {}), pp["ms"])
            elif line.startswith("[mtp] "):
                b["prof_verify_n"] += 1
                hist_add(b["verify_ms_hist"], pp["ms"])
            else:
                b["prof_other_n"] += 1
        if not line or line[0] not in "[c":
            self._unparsed(b, line)
            return
        head = line[:16]
        if head.startswith("[cache] step"):
            m = RE["cache"].match(line)
            if m:
                M = int(m[2])
                b["decode_steps"] += 1
                b["routed"] += int(m[3]); b["hit"] += int(m[4]); b["cpu"] += int(m[5]); b["streamed"] += int(m[6])
                b["resident_last"] = int(m[7]); b["slots_last"] = int(m[9]); b["pending_max"] = max(b["pending_max"], int(m[8]))
                rest = m[10]
                if rest:
                    d = RE_DMA.search(rest)
                    if d:
                        b["dma_span_ms"] += float(d[1]); b["dma_samples"] += 1
                        if d[2] != "-":  # "-" = no wait measured in that step (runtime.cpp cw_collect n_wait 0)
                            b["dma_wait_ms"] += float(d[2]); b["dma_wait_samples"] += 1
                    p = RE_PROMO.search(rest)
                    if p:
                        b["promo_mib"] += float(p[1])
                    w = RE_STEP_WALL.search(rest)
                    if w:  # wall-clock ms of every decode step (all steps, not just the sampled [profile] ones)
                        hist_add(b["step_wall_ms_hist"], float(w[1]))
                        hist_add(b["step_wall_by_m"].setdefault(str(M), {}), float(w[1]))
                    v = RE_VFREE.search(rest)
                    if v:
                        vf = int(v[1])
                        b["vram_free_min"] = vf if b["vram_free_min"] is None else min(b["vram_free_min"], vf)
                b["batch_hist"][str(M)] = b["batch_hist"].get(str(M), 0) + 1
                self._decode_mark(b, M)
                return
        elif head.startswith("[profile M="):
            m = RE["profile"].match(line)
            if m:
                body = m[3]
                secs = body.split(" · ")[0]
                is_mtp = secs.lstrip().startswith("mtp.")  # draft forward (first section mtp.embed) — distinguished from the mtp.sync at the end of a normal step
                for k, v in RE_KV.findall(secs):
                    b["prof_sections_ms"][k] = round(b["prof_sections_ms"].get(k, 0.0) + float(v), 3)
                self.pending_prof = {"m": int(m[1]), "ms": float(m[2]), "mtp": is_mtp}
                return  # the dma clause is also attached to the same step's [cache] line, so it is counted only there (no double counting)
        elif head.startswith("[mtp] "):
            m = RE["mtp_gate2"].match(line)
            if m:
                self.t1 = float(m[4])
                b["mtp_gate2_n"] += 1
                b["mtp_gate2_net_sum"] += float(m[7])
                return
            m = RE["mtp_pos"].match(line)
            if m:
                dr, vr, acc, drafted = float(m[2]), float(m[4]), int(m[5]), int(m[6])
                b["mtp_steps"] += 1; b["mtp_draft_ms"] += dr; b["mtp_verify_ms"] += vr
                b["mtp_accepted"] += acc; b["mtp_drafted"] += drafted; b["mtp_tokens"] += acc + 1
                if self.t1:
                    b["mtp_saved_ms"] += (acc + 1) * self.t1 - dr - vr
                self._decode_mark(b, 1)
                return
            m = RE["mtp_nodraft"].match(line)
            if m:
                dr = float(m[2])
                b["mtp_nodraft"] += 1; b["mtp_draft_ms"] += dr; b["mtp_saved_ms"] -= dr
                t1 = RE["mtp_g2T1"].search(line)
                if t1:
                    self.t1 = float(t1[1])
                # no draft → the normal decode step that follows ([cache] line) leaves the progress marker
                return
            m = RE["mtp_batch"].match(line)
            if m:
                S, got, gms, dr, ts_cost = int(m[1]), int(m[7]), float(m[8]), float(m[9]), float(m[6])
                acc = drafted = 0
                for pair in m[3].split(","):
                    if "/" in pair:
                        a, k = pair.split("/")
                        acc += int(a); drafted += int(k)
                b["mtp_batch_n"] += 1; b["mtp_accepted"] += acc; b["mtp_drafted"] += drafted; b["mtp_tokens"] += got
                b["mtp_draft_ms"] += dr; b["mtp_verify_ms"] += gms
                b["mtp_saved_ms"] += got * ts_cost / max(S, 1) - gms - dr
                self._decode_mark(b, S)
                return
            m = RE["mtp_batch_nodraft"].match(line)
            if m:
                b["mtp_batch_nodraft"] += 1; dr = float(m[2]); b["mtp_draft_ms"] += dr; b["mtp_saved_ms"] -= dr
                return
        elif head.startswith("[decode-host M="):
            m = RE["dhost"].match(line)
            if m:
                body = m[3]
                if m[2]:
                    b["dh_verify_n"] += 1
                b["dh_n"] += 1
                g = RE_DH["gpu_idle"].search(body)
                if g:
                    gi, of = float(g[1]), float(g[2])
                    b["dh_gpu_idle_ms"] += gi; b["dh_of_ms"] += of
                    if of > 0:
                        hist_add(b["gpu_idle_pct_hist"], 100.0 * gi / of)
                c = RE_DH["cpu"].search(body)
                if c:
                    b["dh_cpu_ms"] += float(c[1]); b["dh_gpu_ms"] += float(c[2])
                cb = RE_DH["cpu_bound"].search(body)
                if cb:
                    b["dh_cpu_bound_layers"] += int(cb[1]); b["dh_layers"] += int(cb[2])
                t = RE_DH["tail"].search(body)
                if t:
                    b["dh_tail_ms"] += float(t[1])
                f = RE_DH["front"].search(body)
                if f:
                    b["dh_front_ms"] += float(f[1])
                pw = RE_DH["promo_wait"].search(body)
                if pw:
                    b["dh_promo_wait_n"] += 1; b["dh_promo_wait_ms"] += float(pw[1])
                dw = RE_DH["defer_wait"].search(body)
                if dw:
                    b["dh_defer_wait_ms"] += float(dw[1])
                sy = RE_DH["sync"].match(body)
                if sy:
                    b["dh_sync_ms"] += float(sy[1])
                nx = RE_DH["next"].search(body)
                if nx:
                    b["dh_next_ms"] += float(nx[1])
                return
        elif head.startswith(("[step-host ", "[call-host ", "[fwd-host M=", "[glm-prefill T=", "[glm-ckpt] ")):
            self._host_time(line, b)
            return
        elif head.startswith("[decode-sync M="):
            m = RE["dsync"].match(line)
            if m:
                b["dsync_n"] += 1; b["dsync_wait_ms"] += float(m[3])
                return
        elif head.startswith("[decode-miss M="):
            m = RE["dmiss"].match(line)
            if m:
                b["dmiss_n"] += 1
                for k, v in RE_KV.findall(m[2].replace("(", " ").replace(")", " ")):
                    if k in ("precision", "recall"):
                        continue
                    b["dmiss"][k] = round(b["dmiss"].get(k, 0) + fnum(v), 3)
                return
        elif head.startswith("[early-route M="):
            m = RE["eroute"].match(line)
            if m:
                e = b["eroute"]  # each line is "totals since the previous line" (runtime.cpp dh_report prints and resets them) — except resync/missing/untrusted, which are gate totals
                e["layers"] += int(m[2]); e["ahead"] += int(m[3]); e["own_stream"] += int(m[4] or 0); e["absorbed"] += int(m[5])
                e["resync"] = max(e["resync"], int(m[6])); e["missing"] = max(e["missing"], int(m[7])); e["untrusted"] = max(e["untrusted"], int(m[8]))
                return
            if "⚠️" in line:
                b["warnings"] += 1
                return
        elif head.startswith("[pregate M="):
            m = RE["pregate"].match(line)
            if m:
                cur = [int(m[i]) for i in range(2, 11)] + [float(m[11]), int(m[12]), int(m[13])]
                prev = self.pg_last
                # if the total went down (first line after a restart, banner missed), that line's value itself is this period's share
                delta = [c - p for c, p in zip(cur, prev)] if prev and all(c >= p for c, p in zip(cur, prev)) else cur
                pg = b["pregate"]
                pg["n"] += 1
                for k, d in zip(PG_FIELDS, delta):
                    pg[k] = round(pg[k] + d, 3) if k == "read_mib" else pg[k] + d
                self.pg_last = cur
                return
        elif head.startswith("[snapshot]"):
            m = RE["snapshot"].match(line)
            if m:
                b["snapshots"] += 1; b["snapshot_d2h_bytes"] += int(m[2])
                return
        elif head.startswith("[runtime]"):
            m = RE["elastic"].match(line)
            if m:
                b["reclaims"] += 1; b["reclaim_evicted"] += int(m[2]); b["reclaim_ms"] += float(m[4])
                return
            m = RE["pregate_on"].match(line)
            if m:  # P1 startup banner — records the enabled K (absent when off = "pre-gate off" in the report)
                b["events"].append({"kind": "pregate_on", "k": int(m[1])})
                self.pg_last = None
                return
            if "FATAL" in line:
                b["errors"].append({"kind": "fatal", "msg": line[:300]})
                return
            if "⚠️" in line:
                b["warnings"] += 1
            return  # other [runtime] lines are startup banners
        elif head.startswith("[step-graph M="):
            m = RE["stepgraph"].match(line)
            if m:
                b["stepgraph_n"] += 1; b["stepgraph_wait_ms"] += float(m[2])
            elif "⚠️" in line:
                b["warnings"] += 1
            return
        elif head.startswith(("[decode-split", "[prefill-prof", "[store]", "[model]", "[hive]",
                              "[pregate-stat", "[engram-digest]", "[devmem]")):  # diagnostic lines with no report field (were counted as unparsed)
            if "⚠️" in line:
                b["warnings"] += 1
            return  # startup banners and sampled diagnostics (not aggregated into minute buckets)
        elif head.startswith("[supervisor]"):
            m = RE["supervisor"].match(line)
            if m:
                self.build = {"dir": m[1], "sha256": m[2], "elf": m[3]}
                b["events"].append({"kind": "container_start", "build": self.build})
                self._abandon_all(ts, quality, err_s, "daemon_restart", loff)
                self.gap_ms = 0.0; self.gap_sids = []; self.round_sids = []
                self.pg_last = None  # P1 totals live for the process lifetime — a new container starts from 0
                return
            m = RE["child_exit"].match(line)
            if m:
                b["events"].append({"kind": "child_exit", "status": int(m[1])})
                return
        elif head.startswith("[hived] "):
            self._hived(line, ts, loff, quality, err_s, b)
            return
        elif line.startswith("cudaMalloc"):
            b["errors"].append({"kind": "oom", "msg": line[:300]})
            return
        self._unparsed(b, line)

    def _host_time(self, line, b):
        """HIVE_PROFILE host-time lines → sums in the minute bucket (means × count, so minutes and days add up)."""
        m = RE["step_host"].match(line)
        if m:
            n, gn = int(m[2]), int(m[4])
            d = b["step_host"].setdefault(m[1], {"steps": 0, "gap_n": 0, "gap_ms": 0.0, "pre_ms": 0.0, "draft_ms": 0.0, "fwd_ms": 0.0, "post_ms": 0.0})
            d["steps"] += n; d["gap_n"] += gn; d["gap_ms"] += float(m[3]) * gn
            for i, k in ((5, "pre_ms"), (6, "draft_ms"), (7, "fwd_ms"), (8, "post_ms")):
                d[k] += float(m[i]) * n
            return
        m = RE["call_host"].match(line)
        if m:
            n = int(m[2])
            d = b["call_host"].setdefault(m[1], {"calls": 0, "rows": 0.0, "eng_ms": 0.0, "logits_ms": 0.0, "cands_ms": 0.0, "rest_ms": 0.0})
            d["calls"] += n
            for i, k in ((3, "rows"), (4, "eng_ms"), (5, "logits_ms"), (6, "cands_ms"), (7, "rest_ms")):
                d[k] += float(m[i]) * n
            return
        m = RE["glm_prefill"].match(line)
        if m:
            T, end = int(m[1]), int(m[1]) + int(m[2])
            bk = "<16K" if end < 16384 else "16-64K" if end < 65536 else "64K+"
            d = b["glm_prefill"].setdefault(bk, {"n": 0, "rows": 0, "wall_ms": 0.0, "yield_ms": 0.0, "moe_prep_ms": 0.0, "moe_gpu_ms": 0.0,
                                                 "cpu_join_ms": 0.0, "attn_kda_ms": 0.0, "attn_dsa_ms": 0.0, "other_ms": 0.0,
                                                 "hit_rows": 0, "cpu_rows": 0, "streamed": 0, "streamed_gb": 0.0})
            d["n"] += 1; d["rows"] += T
            for i, k in ((3, "wall_ms"), (4, "yield_ms"), (5, "moe_prep_ms"), (6, "moe_gpu_ms"), (7, "cpu_join_ms"), (8, "attn_kda_ms"),
                         (9, "attn_dsa_ms"), (10, "other_ms"), (14, "streamed_gb")):
                d[k] += float(m[i])
            for i, k in ((11, "hit_rows"), (12, "cpu_rows"), (13, "streamed")):
                d[k] += int(m[i])
            return
        m = RE["glm_ckpt"].match(line)
        if m:
            k = b["ckpt"]
            k["n"] += 1; k["delta_n"] += int(m[2]) >= 0; k["ms"] += float(m[5]); k["ms_max"] = max(k["ms_max"], float(m[5]))
            k["copied_mb"] += float(m[3]); k["total_mb"] += float(m[4])
            return
        m = RE["glm_ckpt_verify"].match(line)
        if m:
            k = b["ckpt"]
            k["verify_n"] += 1
            if not m[4].startswith("ok"):
                k["verify_bad"] += 1; b["warnings"] += 1
                b["errors"].append({"kind": "ckpt_verify", "msg": line[:300]})
            return
        m = RE["fwd_host"].match(line)
        if m:
            f = b["fwd_host"]
            f["n"] += 1
            for i, k in ((3, "host_ms"), (4, "entry_ms"), (5, "window_ms"), (6, "head_ms")):
                f[k] += float(m[i])
            return
        self._unparsed(b, line)

    def _hived(self, line, ts, loff, quality, err_s, b):
        m = RE["final"].match(line)
        if m:
            sid = m[1]
            r = self.open.get(sid) or self._open(sid, ts, loff)
            final = {"prefill_tokens": int(m[2]), "prefill_ms": float(m[3]), "decode_tokens": int(m[4]), "decode_ms": float(m[5]),
                     "decode_tok_s": float(m[6]), "batch_at_end": int(m[7]), "decode_hit": int(m[8]), "decode_cpu": int(m[9]),
                     "resident": int(m[10]), "slots": int(m[11])}
            if m[12] is not None:
                final.update({"mtp_accepted": int(m[12]), "mtp_drafted": int(m[13]), "mtp_steps": int(m[14])})
            if m[16]:  # engine wall clock — used instead of the interpolated time when present
                ex = dict(RE_EXTRA.findall(m[16]))
                if "rid" in ex:
                    final["rid"] = ex["rid"]
                if "queue_ms" in ex:
                    final["queue_ms"] = float(ex["queue_ms"])
                if "t2" in ex:
                    try:
                        ts, quality, err_s = float(ex["t2"]) / 1000.0, "engine", 0.0
                    except ValueError:
                        pass
            self._close(r, ts, quality, err_s, m[15], final, loff)
            return
        m = RE["chunk"].match(line)
        if m:
            sid, ci, M, a, z, tot, ms = m[1], int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6]), float(m[15])
            r = self.open.get(sid)
            if r is not None and r["phase"] == "decode":  # a new request with the same sid (the previous request had no final line = failed/lost)
                self._close(r, ts, quality, err_s, "lost", None, loff); r = None
            r = r or self._open(sid, ts, loff)
            r["prompt_tokens"] = tot
            r["prefill_chunks"] += 1; r["prefill_rows"] += M; r["prefill_chunk_ms"] += ms
            b["prefill_chunks"] += 1; b["prefill_rows"] += M
            if sid in self.round_sids:  # the batch round line already added this ms to the gap (chunk lines repeat the round ms)
                self.round_sids.remove(sid)
                r["prefill_shared_round"] = True
            else:
                b["prefill_chunk_ms"] += ms
                ly = LY_TAIL.search(line)  # T11: if the forward yielded, the gap is only the segment after the last resume (ms is the prefill share excluding the yield time)
                self._gap([sid], float(ly[2]) if ly else ms)
            return
        m = RE["reuse"].match(line)
        if m:
            sid = m[1]
            r = self.open.get(sid)
            # hived may print two reuse lines for one request — "via reset/live/…" followed by "via shared-prefix (was …)" (hived.cpp shared-prefix reuse).
            #   The second line is therefore not a new request; treating it as one would close the first record as lost although the request ended with stop.
            same_req = m[4] == "shared-prefix" and r is not None and r["phase"] != "decode" and not r["prefill_chunks"]
            if r is not None and not same_req and (r["phase"] == "decode" or r["prefill_chunks"] or r["reuse_via"]):
                self._close(r, ts, quality, err_s, "lost", None, loff); r = None
            if same_req:
                r["reuse_was"] = r["reuse_via"]
            r = r or self._open(sid, ts, loff)
            r["cached_prefix_tokens"] = int(m[2]); r["prompt_tokens"] = int(m[3]); r["reuse_via"] = m[4]; r["reuse_detail"] = m[5]
            if int(m[2]) >= int(m[3]):  # everything reused (checkpoint logits) — the prefill loop does not run
                r["phase"] = "decode"
            return
        m = RE["ptotal"].match(line)
        if m:
            sid = m[1]
            r = self.open.get(sid) or self._open(sid, ts, loff)
            r.update({"prefill_rows_total": int(m[2]), "prefill_hit": int(m[4]), "prefill_streamed": int(m[5]), "prefill_cpu": int(m[6]),
                      "prefill_dma_rows": int(m[7]), "prefill_cpu_wait_ms": float(m[8]), "prefill_cpu_span_ms": float(m[9]),
                      "prefill_total_ms": float(m[10])})
            r["phase"] = "decode"
            return
        m = RE["bround"].match(line)
        if m:
            ms = float(m[11])
            sids = []
            for part in m[3].split(","):
                sid = part.rsplit(":", 1)[0]
                if sid:
                    sids.append(sid)
                    rr = self.open.get(sid) or self._open(sid, ts, loff)
                    rr["prefill_batch_rounds"] += 1
            b["batch_rounds"] += 1; b["batch_round_seqs"] += int(m[2]); b["prefill_chunk_ms"] += ms
            ly = LY_TAIL.search(line)  # T11 (same as for chunk lines)
            self._gap(sids, float(ly[2]) if ly else ms)
            self.round_sids = list(sids)
            return
        m = RE["pending"].match(line) or RE["pfxsnap"].match(line)
        if m:
            r = self.open.get(m[1])
            if r is not None and line.find("prefix snapshot") > 0:
                r["prefix_snapshots"] += 1
            return
        m = RE["window"].match(line)
        if m:
            b["batch_windows"] += 1; b["batch_window_ms"] += float(m[2])
            return
        m = RE["yield"].match(line)
        if m:
            b["yield_admits"] += int(m[1])
            return
        m = RE["lyield"].match(line)
        if m:  # T11: the prefill segment up to the yield is a gap (the decode markers inside the yield end it). Item = "sid:prompt id count" (comes before the chunk line)
            sids = []
            for part in m[1].split(","):
                sid, _, rows = part.rpartition(":")
                if not sid or sid == "-":
                    continue
                r = self.open.get(sid) or self._open(sid, ts, loff)
                if rows.isdigit() and not r.get("prompt_tokens"):
                    r["prompt_tokens"] = int(rows)
                sids.append(sid)
            self._gap(sids, float(m[2]))
            b["layer_yields"] = b.get("layer_yields", 0) + 1
            return
        m = RE["lydone"].match(line)
        if m:
            b["layer_yield_steps"] = b.get("layer_yield_steps", 0) + int(m[2])
            b["layer_yield_ms"] = round(b.get("layer_yield_ms", 0.0) + float(m[3]), 1)
            return
        m = RE["warm"].match(line)
        if m:
            b["warm_n"] += 1; b["warm_experts"] += int(m[1]); b["warm_ms"] += float(m[2])
            return
        m = RE["warmdefer"].match(line)
        if m:
            b["warm_n"] += 1; b["warm_experts"] += int(m[1])
            return
        m = RE["reqfail"].match(line)
        if m:
            # admission/prefill failure — there is no final line. The sid is not printed, so it is attributed to the most recently opened request in the prefill phase (a guess — attributed="guess")
            cands = [r for r in self.open.values() if r["phase"] == "prefill"]
            err = {"kind": "request_failed", "msg": m[1][:300]}
            b["errors"].append(dict(err))
            if cands:
                r = max(cands, key=lambda r: r["first_loff"])
                r["errors"].append({**err, "attributed": "guess"})
                self._close(r, ts, quality, err_s, "error", None, loff)
            return
        m = RE["reqfail_sid"].match(line)
        if m:
            err = {"kind": "request_failed", "msg": m[2][:300]}
            b["errors"].append(dict(err))
            r = self.open.get(m[1]) or self._open(m[1], ts, loff)
            r["errors"].append(err)
            self._close(r, ts, quality, err_s, "error", None, loff)
            return
        m = RE["decfail"].match(line)
        if m:
            err = {"kind": "decode_failed", "msg": m[1][:300]}
            b["errors"].append(dict(err))
            for r in [r for r in self.open.values() if r["phase"] == "decode"]:  # the engine ends all active requests with an error (no final lines)
                r["errors"].append(err)
                self._close(r, ts, quality, err_s, "error", None, loff)
            return
        m = RE["bfail"].match(line)
        if m:
            b["errors"].append({"kind": "batch_prefill_failed", "msg": m[1][:300], "retried": int(m[2])})
            return
        m = RE["ckptdrop"].match(line)
        if m:
            b["warnings"] += 1
            return
        m = RE["ready"].match(line)
        if m:
            b["events"].append({"kind": "ready", "load_s": float(m[1])})
            return
        m = RE["warmstart"].match(line)
        if m:
            b["events"].append({"kind": "warm_start", "experts": int(m[1]), "ms": float(m[3])})
            return
        m = RE["sleep"].match(line)
        if m:
            b["events"].append({"kind": "sleep", "ms": float(m[1]), "drain_ms": float(m[2]), "freed_mib": float(m[3])})
            return
        m = RE["wake"].match(line)
        if m:
            b["events"].append({"kind": "wake", "ms": float(m[1]), "warm_ms": float(m[3]), "slots": int(m[4])})
            return
        m = RE["lifecycle_err"].match(line)
        if m:
            b["errors"].append({"kind": m[1].replace(" ", "_").replace(":", ""), "msg": line[:300]})
            return
        m = RE["graceful"].match(line)
        if m:
            b["events"].append({"kind": "graceful_stop", "msg": line[:200]})
            if "complete" in line:
                self._abandon_all(ts, quality, err_s, "daemon_restart", loff)
            return
        # remaining [hived] lines = startup banners (listening, session pool, cache fit, prefix share …) and sleep request notices
        if "⚠️" in line:
            b["warnings"] += 1

    def _unparsed(self, b, line):
        self.counts["unparsed"] += 1
        shape = RE_SHAPE_NUM.sub("N", line[:60]).strip()
        if not shape:
            return
        u = b["unparsed"]
        if shape in u or len(u) < 20:
            u[shape] = u.get(shape, 0) + 1


# ---------------------------------------------------------------------------------------------------------------------
# Log source: inode + logical offset + rotation journal
# ---------------------------------------------------------------------------------------------------------------------
def read_journal(log_path: str) -> list:
    p = log_path + ".rotations.jsonl"
    out = []
    try:
        with open(p) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        out.append(json.loads(ln))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return out


def read_archive(entry: dict, log_dir: str, start: int, limit: int) -> bytes | None:
    """Read [start, start+limit) (offsets within the archive) from a rotation archive. None if absent."""
    name = entry.get("archive")
    if not name:
        return None
    base = name if os.path.isabs(name) else os.path.join(log_dir, name)
    for cand in (base, base + ".gz"):
        if os.path.exists(cand):
            op = gzip.open if cand.endswith(".gz") else open
            with op(cand, "rb") as f:
                f.seek(start)
                return f.read(limit)
    return None


class Source:
    def __init__(self, log_path: str, st: dict):
        self.path = log_path
        self.dir = os.path.dirname(os.path.abspath(log_path))
        self.inode = st.get("inode")
        self.loff = st.get("loff", 0)
        self.gaps: list = []

    def read(self, max_bytes: int) -> tuple[bytes, int, int]:
        """New bytes as (data, start logical offset, end logical offset). data covers complete lines only (including the final \\n)."""
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return b"", self.loff, self.loff
        if st.st_ino != self.inode:
            if self.inode is not None:
                self.gaps.append({"kind": "inode_changed", "old": self.inode, "new": st.st_ino, "at_loff": self.loff})
            self.inode, self.loff = st.st_ino, 0
        entries = sorted((e for e in read_journal(self.path) if e.get("inode") == self.inode), key=lambda e: e["logical_start"])
        collapsed = 0
        for e in entries:
            collapsed = max(collapsed, e["logical_start"] + e["cut"] + e.get("lost_bytes", 0))
        start = self.loff
        buf = bytearray()
        pos = self.loff
        # 1) the cut range (archive) — only when the parser fell behind the rotation
        for e in entries:
            a0, a1 = e["logical_start"], e["logical_start"] + e["cut"]
            if pos >= a1 + e.get("lost_bytes", 0) or len(buf) >= max_bytes:
                continue
            if pos < a0:  # a hole between journal entries (should be impossible, but defensively)
                self.gaps.append({"kind": "journal_hole", "from": pos, "to": a0}); pos = a0
            if pos < a1:
                data = read_archive(e, self.dir, pos - a0, min(a1 - pos, max_bytes - len(buf)))
                if data is None:
                    self.gaps.append({"kind": "archive_missing", "archive": e.get("archive"), "from": pos, "to": a1})
                    pos = a1
                else:
                    buf += data; pos += len(data)
                    if pos < a1:
                        break
            if pos >= a1 and e.get("lost_bytes"):
                self.gaps.append({"kind": "truncate_window_lost", "bytes_max": e["lost_bytes"]}); pos = a1 + e["lost_bytes"]
        # 2) the current file
        if len(buf) < max_bytes and pos >= collapsed:
            phys = pos - collapsed
            if phys > st.st_size:  # truncated without a journal entry (external truncate) — start over
                self.gaps.append({"kind": "truncated_without_journal", "phys": phys, "size": st.st_size})
                phys = 0; pos = collapsed
                if not buf:
                    start = pos
            with open(self.path, "rb") as f:
                f.seek(phys)
                buf += f.read(max_bytes - len(buf))
        cut = buf.rfind(b"\n")
        if cut < 0:
            return b"", start, start
        data = bytes(buf[:cut + 1])
        return data, start, start + len(data)


# ---------------------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------------------
def append_jsonl(path: str, recs: list) -> None:
    if not recs:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_outputs(out_dir: str, parser: Parser) -> None:
    by_day: dict = {}
    for r in parser.out_requests:
        by_day.setdefault(("requests", local_day(r["end_ts"])), []).append(r)
    for mb in parser.out_metrics:
        by_day.setdefault(("metrics", mb["minute"][:10]), []).append(mb)
    for (kind, day), recs in sorted(by_day.items()):
        append_jsonl(os.path.join(out_dir, kind, f"{day}.jsonl"), recs)
    parser.out_requests.clear()
    parser.out_metrics.clear()


def save_state(path: str, st: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_state(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def run_once(log_path: str, out_dir: str, state_path: str, max_bytes: int, max_interp: float, now: float | None = None,
             lock: bool = True, flush: bool = False, from_end: bool = False) -> dict:
    """Read once (repeatedly, max_bytes at a time, until the end) and write outputs and state. Returns a summary."""
    st = load_state(state_path)
    if not st and from_end and os.path.exists(log_path):  # first install: skip the past (backfill without timestamps) and start from now
        jst = os.stat(log_path)
        coll = max([e["logical_start"] + e["cut"] for e in read_journal(log_path) if e.get("inode") == jst.st_ino], default=0)
        with open(log_path, "rb") as f:  # start at a line boundary
            f.seek(max(0, jst.st_size - 65536))
            tail = f.read()
        nl = tail.rfind(b"\n")
        st = {"inode": jst.st_ino, "loff": coll + (jst.st_size - len(tail) + nl + 1 if nl >= 0 else 0), "t_read": now or time.time()}
    parser = Parser(st.get("parser"))
    src = Source(log_path, st)
    lockf = None
    if lock:
        lockf = open(log_path + ".lock", "a")
        fcntl.flock(lockf, fcntl.LOCK_SH)
    try:
        t_now = time.time() if now is None else now
        t_prev, o_prev = st.get("t_read"), st.get("loff", 0)
        # Interpolating over the whole read [o_prev, o_end] needs the end offset first; since chunks are read one at a time the interpolation range cannot be fixed in advance,
        # so each chunk uses end = t_now (only backfill ever exceeds max_bytes in one go, and that is backfill quality anyway).
        quality = "interp"
        if t_prev is None or t_now - t_prev > max_interp or src.inode != st.get("inode"):
            quality = "backfill"
        total_lines = 0
        while True:
            data, a, z = src.read(max_bytes)
            if not data:
                break
            span = max(z - o_prev, 1)
            if quality == "interp" and z - a >= max_bytes:
                quality = "backfill"  # fell too far behind — interpolation is meaningless
            err = 0.0 if quality == "backfill" else (t_now - t_prev)
            pos = a
            for raw in data.split(b"\n")[:-1]:
                line = raw.decode("utf-8", errors="replace").rstrip("\r")
                if quality == "interp":
                    ts = t_prev + (t_now - t_prev) * ((pos + len(raw) + 1 - o_prev) / span)
                else:
                    ts = t_now
                parser.feed(line, ts, pos, quality, err)
                pos += len(raw) + 1
                total_lines += 1
            src.loff = z
            write_outputs(out_dir, parser)
            save_state(state_path, {"schema": SCHEMA, "inode": src.inode, "loff": src.loff, "t_read": t_now,
                                    "parser": parser.state(), "log": log_path})
        if total_lines and (flush or quality == "backfill"):  # backfill timestamps carry no meaning, so the in-progress bucket is emitted immediately
            parser.flush_bucket()
            write_outputs(out_dir, parser)
            save_state(state_path, {"schema": SCHEMA, "inode": src.inode, "loff": src.loff, "t_read": t_now, "parser": parser.state(), "log": log_path})
        if total_lines == 0:
            st2 = {"schema": SCHEMA, "inode": src.inode, "loff": src.loff, "t_read": t_now, "parser": parser.state(), "log": log_path}
            # after a long quiet period, emit the in-progress minute bucket (do not wait for the next line)
            if parser.bucket and parser.bucket["minute"] < local_minute(t_now):
                parser.flush_bucket()
                write_outputs(out_dir, parser)
                st2["parser"] = parser.state()
            save_state(state_path, st2)
        if src.gaps:
            append_jsonl(os.path.join(out_dir, "monitor", "gaps.jsonl"), [{"ts": local_iso(t_now), **g} for g in src.gaps])
        return {"lines": total_lines, "loff": src.loff, "open_requests": len(parser.open), "quality": quality, "gaps": src.gaps}
    finally:
        if lockf:
            fcntl.flock(lockf, fcntl.LOCK_UN)
            lockf.close()


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--log", default=DEFAULT_LOG, help="hived.log to read (default: $HIVE_STATE_DIR/logs/hived.log, else <repo>/run/logs/hived.log)")
    cli.add_argument("--out", default=None, help="output root (default = the log directory)")
    cli.add_argument("--state", default=None, help="default <out>/monitor/state.json")
    cli.add_argument("--follow", action="store_true", help="stay running: read every --interval seconds (timestamp error = interval)")
    cli.add_argument("--interval", type=float, default=2.0)
    cli.add_argument("--max-bytes", type=int, default=64 << 20)
    cli.add_argument("--max-interp", type=float, default=900.0, help="if the previous read is older than this many seconds, lines get backfill quality")
    cli.add_argument("--no-lock", action="store_true")
    cli.add_argument("--flush", action="store_true", help="also emit the in-progress minute bucket at the end (for one-off runs)")
    cli.add_argument("--from-end", action="store_true", help="without a state file, start at the current end of the log (skip the past backfill — for a first install)")
    cli.add_argument("--quiet", action="store_true")
    a = cli.parse_args(argv)
    out = a.out or os.path.dirname(os.path.abspath(a.log))
    state = a.state or os.path.join(out, "monitor", "state.json")
    while True:
        r = run_once(a.log, out, state, a.max_bytes, a.max_interp, lock=not a.no_lock, flush=a.flush, from_end=a.from_end)
        if not a.quiet and (r["lines"] or not a.follow):
            print(json.dumps({"ts": local_iso(time.time()), **r}, ensure_ascii=False), flush=True)
        if not a.follow:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
