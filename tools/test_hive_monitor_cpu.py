#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""hive_monitor.py · hive_daily_report.py · (if present) the log rotation script (HIVE_ROTATE_SCRIPT) — CPU only (no GPU, service or SSH).

Fixtures:
  fixtures/hived_real_excerpt.log  7 excerpts cut from a real hived.log (tail of 200K lines) — batch prefill rounds, old early-route format,
                                   MTP single/batch, cancel, OOM, restart banners. Prompt text is not in the log (sid = hash).
  fixtures/hived_synthetic.log     printf variants not yet seen in the real log (exactly the hived.cpp formats): request/decode/batch failures,
                                   sleep/wake, prefill yield, step-graph, gate2 no draft, graceful stop, unknown lines.
"""
import copy
import gzip
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import hive_monitor as hm  # noqa: E402

spec = importlib.util.spec_from_file_location("hive_daily_report", HERE / "hive_daily_report.py")
hr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr)

REAL = HERE / "fixtures" / "hived_real_excerpt.log"
SYN = HERE / "fixtures" / "hived_synthetic.log"
ROTATE = Path(os.environ["HIVE_ROTATE_SCRIPT"]) if os.environ.get("HIVE_ROTATE_SCRIPT") else None  # no rotation script ships with the repo — set it to test one
TS_KEYS = {"end_ts", "end", "start_ts", "start", "first_token_ts", "ts_quality", "ts_err_s"}


def records(out):
    rs = []
    for p in sorted((Path(out) / "requests").glob("*.jsonl")):
        rs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rs


def strip_ts(rs):
    return [{k: r[k] for k in r if k not in TS_KEYS} for r in rs]


def run(log, out, now=None, max_bytes=64 << 20, flush=False):
    return hm.run_once(str(log), str(out), str(Path(out) / "monitor" / "state.json"), max_bytes, 900.0, now=now, flush=flush)


def single_pass(src_bytes, tmp):
    d = Path(tmp) / "single"
    d.mkdir()
    (d / "hived.log").write_bytes(src_bytes)
    run(d / "hived.log", d / "out", now=1000.0)
    return strip_ts(records(d / "out"))


class Parse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hivemon-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_real_excerpt(self):
        log = Path(self.tmp) / "hived.log"
        shutil.copy(REAL, log)
        run(log, Path(self.tmp) / "out", flush=True)
        rs = records(Path(self.tmp) / "out")
        finals = [l for l in REAL.read_text().splitlines() if hm.RE["final"].match(l)]
        done = [r for r in rs if r.get("prefill_ms") is not None]
        self.assertEqual(len(done), len(finals))
        self.assertEqual(len({r["id"] for r in rs}), len(rs))
        # request spanning 2 batch prefill rounds (real log lines 189740 and 189898): prompt 33853, reused 759, 2 chunks, 2 rounds
        r = next(r for r in rs if r["sid"] == "2bc199e02ba35c74")
        self.assertEqual((r["prompt_tokens"], r["cached_prefix_tokens"], r["prefill_chunks"], r["prefill_batch_rounds"]), (33853, 759, 2, 2))
        self.assertTrue(r.get("prefill_shared_round"))
        # final line values as is
        r = next(r for r in rs if r["sid"] == "0b5be76b5512753b" and r.get("decode_tokens") == 520)
        self.assertEqual((r["prefill_tokens"], r["ttft_ms"], r["decode_tok_s"], r["batch_at_end"], r["mtp_accepted"], r["mtp_drafted"], r["finish"]),
                         (1347, 4482.0, 34.3, 2, 11, 18, "stop"))
        self.assertEqual(r["prompt_tokens"], 1347)
        # every line format was parsed (a new printf shows up as unparsed)
        mets = [json.loads(l) for p in (Path(self.tmp) / "out" / "metrics").glob("*.jsonl") for l in p.read_text().splitlines()]
        self.assertEqual(sum(sum(m["unparsed"].values()) for m in mets), 0, [m["unparsed"] for m in mets])
        m = mets[0]
        self.assertGreater(m["decode_steps"], 0)
        self.assertGreater(m["prof_decode_n"], 0)
        self.assertGreater(m["eroute"]["layers"], 0)
        self.assertEqual(m["mtp_batch_n"], 1)
        self.assertTrue(any(e["kind"] == "oom" for e in m["errors"]))
        # during a >= 16K batch prefill round (7321 ms) a request was decoding → at least one stall, attributed to the causing request
        self.assertGreaterEqual(m["stall_n"], 1)
        self.assertTrue(any(r["stall_caused_n"] for r in rs))

    def test_incremental_equals_single_pass(self):
        data = REAL.read_bytes()
        want = single_pass(data, self.tmp)
        log = Path(self.tmp) / "hived.log"
        out = Path(self.tmp) / "inc"
        rnd = random.Random(7)
        pos, now = 0, 1000.0
        log.write_bytes(b"")
        while pos < len(data):
            n = rnd.randint(1, 30000)  # also cut in the middle of a line
            with open(log, "ab") as f:
                f.write(data[pos:pos + n])
            pos += n
            now += 1.0
            run(log, out, now=now, max_bytes=rnd.choice([4096, 1 << 20]))
        self.assertEqual(strip_ts(records(out)), want)

    def test_interp_timestamps(self):
        log = Path(self.tmp) / "hived.log"
        out = Path(self.tmp) / "out"
        data = REAL.read_bytes()
        half = data.index(b"\n", len(data) // 2) + 1
        log.write_bytes(data[:half])
        run(log, out, now=1000.0)
        with open(log, "ab") as f:
            f.write(data[half:])
        run(log, out, now=1010.0)
        rs = [r for r in records(out) if r["ts_quality"] == "interp"]
        self.assertTrue(rs)
        ends = [r["end_ts"] for r in rs]
        self.assertEqual(ends, sorted(ends))
        self.assertTrue(all(1000.0 <= t <= 1010.0 for t in ends))
        self.assertTrue(all(r["ts_err_s"] == 10.0 for r in rs))
        for r in rs:
            if r.get("prefill_ms") is not None:
                self.assertAlmostEqual(r["end_ts"] - r["start_ts"], (r["prefill_ms"] + r["decode_ms"]) / 1000, places=2)

    def _rotate_py(self, log, cut, archive_name, gz=False):
        """Emulates in Python what the log rotation script does (cut the first bytes into an archive, keep the same inode, journal entry)."""
        st = os.stat(log)
        data = Path(log).read_bytes()
        arch = Path(log).parent / "archive"
        arch.mkdir(exist_ok=True)
        if gz:
            with gzip.open(arch / (archive_name + ".gz"), "wb") as f:
                f.write(data[:cut])
        else:
            (arch / archive_name).write_bytes(data[:cut])
        prev = hm.read_journal(str(log))
        lstart = max([e["logical_start"] + e["cut"] for e in prev if e["inode"] == st.st_ino], default=0)
        with open(log, "r+b") as f:  # same inode
            f.seek(0); f.write(data[cut:]); f.truncate()
        self.assertEqual(os.stat(log).st_ino, st.st_ino)
        with open(str(log) + ".rotations.jsonl", "a") as f:
            f.write(json.dumps({"inode": st.st_ino, "mode": "collapse", "logical_start": lstart, "cut": cut, "archive": f"archive/{archive_name}"}) + "\n")

    def test_rotation_caught_up_and_lagging(self):
        data = REAL.read_bytes()
        want = single_pass(data, self.tmp)
        for lag, gz in ((False, False), (True, False), (True, True)):
            with self.subTest(lag=lag, gz=gz):
                d = Path(self.tmp) / f"rot-{lag}-{gz}"
                d.mkdir()
                log, out = d / "hived.log", d / "out"
                a, b = len(data) // 3, 2 * len(data) // 3
                log.write_bytes(data[:a])
                run(log, out, now=1000.0)
                with open(log, "ab") as f:
                    f.write(data[a:b])
                if not lag:
                    run(log, out, now=1001.0)
                cut = (b - 1) // 4096 * 4096 - 4096  # block multiple — lands in the middle of a line
                self.assertNotEqual(data[cut - 1:cut], b"\n")
                self._rotate_py(log, cut, "hived.log-1", gz=gz)
                with open(log, "ab") as f:
                    f.write(data[b:])
                run(log, out, now=1002.0)
                # second rotation (cumulative logical_start)
                self._rotate_py(log, 4096, "hived.log-2")
                run(log, out, now=1003.0)
                self.assertEqual(strip_ts(records(out)), want)
                gaps = (out / "monitor" / "gaps.jsonl")
                self.assertFalse(gaps.exists(), gaps.read_text() if gaps.exists() else "")

    def test_truncate_without_journal_and_inode_change(self):
        d = Path(self.tmp)
        log, out = d / "hived.log", d / "out"
        log.write_bytes(REAL.read_bytes())
        run(log, out, now=1000.0)
        with open(log, "r+b") as f:
            f.truncate(0)
        syn = SYN.read_bytes()
        with open(log, "ab") as f:
            f.write(syn)
        r = run(log, out, now=1001.0)
        self.assertEqual(r["gaps"][0]["kind"], "truncated_without_journal")
        self.assertTrue(any(x["sid"] == "cccc000000000003" for x in records(out)))
        new = d / "hived.log.new"
        new.write_bytes(syn)
        os.replace(new, log)  # new inode (created first so the inode number cannot be reused even if the old one is no longer open)
        r = run(log, out, now=1002.0)
        self.assertTrue(any(g["kind"] == "inode_changed" for g in r["gaps"]))
        self.assertEqual(sum(1 for x in records(out) if x["sid"] == "cccc000000000003"), 2)

    def test_from_end(self):
        d = Path(self.tmp)
        log, out = d / "hived.log", d / "out"
        shutil.copy(REAL, log)
        hm.run_once(str(log), str(out), str(out / "monitor" / "state.json"), 1 << 20, 900.0, now=1000.0, from_end=True)
        self.assertEqual(records(out), [])
        with open(log, "ab") as f:
            f.write(SYN.read_bytes())
        run(log, out, now=1005.0)
        self.assertEqual({r["ts_quality"] for r in records(out) if r["sid"] == "cccc000000000003"}, {"interp"})

    def test_partial_line_waits(self):
        d = Path(self.tmp)
        log, out = d / "hived.log", d / "out"
        line = b"[hived] eeee000000000005: prefill 10 tok 5 ms \xc2\xb7 decode 3 tok 4 ms (500.0 tok/s, batch 1) \xc2\xb7 hit 1 cpu 0 \xc2\xb7 resident 1/2 \xc2\xb7 stop\n"
        log.write_bytes(line[:40])
        run(log, out, now=1000.0)
        self.assertEqual(records(out), [])
        with open(log, "ab") as f:
            f.write(line[40:])
        run(log, out, now=1001.0)
        self.assertEqual([r["sid"] for r in records(out)], ["eeee000000000005"])

    def test_synthetic_variants(self):
        d = Path(self.tmp)
        log, out = d / "hived.log", d / "out"
        shutil.copy(SYN, log)
        run(log, out, flush=True)
        rs = {r["sid"]: r for r in records(out)}
        self.assertEqual(rs["aaaa000000000001"]["finish"], "error")  # request failed → the request in prefill (guessed attribution)
        self.assertEqual(rs["aaaa000000000001"]["errors"][0]["attributed"], "guess")
        self.assertEqual(rs["bbbb000000000002"]["finish"], "error")  # decode failed → every request in decode
        self.assertEqual(rs["bbbb000000000002"]["cached_prefix_tokens"], 50)
        self.assertEqual(rs["cccc000000000003"]["finish"], "stop")
        self.assertEqual(rs["cccc000000000003"]["prompt_tokens"], 400)
        self.assertEqual(rs["dddd000000000004"]["finish"], "daemon_restart")  # graceful stop complete
        m = [json.loads(l) for p in (out / "metrics").glob("*.jsonl") for l in p.read_text().splitlines()][0]
        kinds = {e["kind"] for e in m["errors"]}
        self.assertTrue({"request_failed", "decode_failed", "batch_prefill_failed", "wake_failed"} <= kinds, kinds)
        ev = {e["kind"] for e in m["events"]}
        self.assertTrue({"container_start", "ready", "sleep", "wake", "graceful_stop"} <= ev, ev)
        self.assertEqual(m["yield_admits"], 2)
        self.assertEqual(m["stepgraph_n"], 1)
        self.assertEqual(m["mtp_nodraft"], 1)
        self.assertEqual(m["mtp_batch_nodraft"], 1)
        self.assertEqual(m["vram_free_min"], 1234)
        self.assertEqual(m["dma_wait_samples"], 0)
        self.assertEqual(sum(m["unparsed"].values()), 1)
        # engine wall-clock fields (engine wall clock, queue wait, rid) — when present, engine times replace interpolation
        f = rs["ffff000000000006"]
        self.assertEqual((f["ts_quality"], f["end_ts"], f["rid"], f["ttft_with_queue_ms"], f["finish"]), ("engine", 1790800000.32, "r-abc", 155.0, "stop"))
        self.assertEqual(sum(m["step_wall_ms_hist"].values()), 1)
        self.assertEqual((rs["9999000000000007"]["finish"], rs["9999000000000007"]["errors"][0].get("attributed")), ("error", None))


    def test_layer_yield_stall(self):
        """T11: layer yielding inside a long prefill forward — the gap is the segment between yields (head line ms, chunk-line tail `last`), not the whole forward.
        Control: with the yield lines removed from the same log (old format), the gap = the whole chunk ms."""
        d = Path(self.tmp)
        cache = "[cache] step {n} M=1 routed 240 hit 207 cpu 33 streamed 0 · resident 4438 pending 8 / 4446 · dma.span 0.00 dma.wait - ms (0 layers)\n"
        head = ("[hived] D: prefill chunk 1 M 100 · 0→100/100 · upper on · tail rows 0 · hit 0 streamed 0 cpu 0 dma_rows 0 · cpu wait 0 ms span 0 ms · 20 ms\n"
                "[hived] D: prefill total 100 rows in 1 chunks · hit 0 streamed 0 cpu 0 dma_rows 0 · cpu wait 0 ms span 0 ms · 20 ms\n" + cache.format(n=1))
        ly = ("[hived] layer yield: L:30000 · prefill 600 ms since resume · rows 128 · active 1\n" + cache.format(n=2) +
              "[hived] layer yield done: admitted 0 · decode steps 1 · 31 ms\n"
              "[hived] layer yield: L:30000 · prefill 650 ms since resume · rows 128 · active 1\n" + cache.format(n=3) +
              "[hived] layer yield done: admitted 0 · decode steps 1 · 29 ms\n")
        chunk = ("[hived] L: prefill chunk 1 M 30000 · 0→30000/30000 · upper on · tail rows 0 · hit 0 streamed 0 cpu 0 dma_rows 0 · cpu wait 0 ms span 0 ms · "
                 "13000 ms{tail}\n")
        for name, text, want_max, want_n in (
                ("on", head + ly + chunk.format(tail=" · layer yields 2 last 700 ms") + cache.format(n=4), 700.0, 3),
                ("off", head + chunk.format(tail="") + cache.format(n=4), 13000.0, 1)):
            sub = d / name
            sub.mkdir()
            (sub / "hived.log").write_text(text)
            run(sub / "hived.log", sub / "out", flush=True)
            m = [json.loads(l) for p in (sub / "out" / "metrics").glob("*.jsonl") for l in p.read_text().splitlines()][0]
            self.assertEqual((m["stall_ms_max"], m["stall_n"]), (want_max, want_n), name)
            if name == "on":
                self.assertEqual((m["layer_yields"], m["layer_yield_steps"], m["layer_yield_ms"]), (2, 2, 60.0))

    def test_glm_sample_lines_and_promo_wait(self):
        """GLM sample-step lines (glm_engine.cpp sample_report · glm_runtime.cpp cache_line — the DeepSeek formats) and the DeepSeek
        [decode-host] promo wait clause: parsed into the same metric fields (exact printf formats of 2026-10-05)."""
        d = Path(self.tmp)
        glm = ("[cache] step 0 M=1 routed 1824 hit 1610 cpu 214 streamed 0 · resident 4420 pending 3 / 4446 · vram free 2400 MiB · wall 14.21 ms\n"
               "[profile M=4] total 41.20 ms: embed 0.02 hc 1.10 kda 6.40 dsa 3.20 dense 0.40 router 0.90 shared 4.10 predict 0.00 experts 23.98 head 1.10\n"
               "[decode-host M=4 verify] sync 2.10 · cpu 18.40 vs gpu 9.20 (cpu-bound layers 30/76) · tail 6.30 ms · next 1.40 · defer wait 0.80 · "
               "gpu-idle 9.80 of 40.10 ms (cpu layers 70)\n"
               "[early-route M=4] layers 76 · host ahead of front end 51 · absorbed 1 (resync 0 · missing 1 · untrusted 0)\n"
               "[cache] step 1 M=4 routed 7296 hit 6500 cpu 796 streamed 0 · resident 4420 pending 0 / 4446 · wall 42.00 ms\n")
        ds = ("[decode-host M=1 verify] sync 0.10 · prep 0.20 · launch 0.30 · cpu 1.00 vs gpu 2.00 (cpu-bound layers 3/80) · tail 0.40 ms · next 0.50 · "
              "front 0.60 · post 0.10 · hprep 0.20 · gpu-idle 1.20 of 18.00 ms (cpu layers 20) · promo wait 0.437 ms/step\n")
        (d / "hived.log").write_text(glm + ds)
        run(d / "hived.log", d / "out", flush=True)
        m = [json.loads(l) for p in (d / "out" / "metrics").glob("*.jsonl") for l in p.read_text().splitlines()][0]
        self.assertEqual(sum(m["unparsed"].values()), 0, m["unparsed"])
        self.assertEqual((m["routed"], m["hit"], m["cpu"], m["decode_steps"]), (1824 + 7296, 1610 + 6500, 214 + 796, 2))
        self.assertEqual((m["dh_n"], m["dh_verify_n"], m["dh_layers"], m["dh_cpu_bound_layers"]), (2, 2, 156, 33))
        self.assertAlmostEqual(m["dh_gpu_idle_ms"], 11.0)
        self.assertAlmostEqual(m["dh_tail_ms"], 6.7)
        self.assertAlmostEqual(m["dh_defer_wait_ms"], 0.8)
        self.assertEqual(m["dh_promo_wait_n"], 1)
        self.assertAlmostEqual(m["dh_promo_wait_ms"], 0.437)
        self.assertEqual((m["eroute"]["layers"], m["eroute"]["ahead"], m["eroute"]["absorbed"], m["eroute"]["missing"]), (76, 51, 1, 1))
        self.assertAlmostEqual(m["prof_sections_ms"]["experts"], 23.98)
        self.assertEqual(m["vram_free_min"], 2400)


class Report(unittest.TestCase):
    def test_report_and_compare(self):
        tmp = tempfile.mkdtemp(prefix="hiverep-")
        try:
            log, out = Path(tmp) / "hived.log", Path(tmp) / "out"
            shutil.copy(REAL, log)
            run(log, out, now=time.mktime((2026, 10, 1, 12, 0, 0, 0, 0, -1)), flush=True)
            # previous day = the same data shifted one day back (tests the comparison path)
            for kind in ("requests", "metrics"):
                src = out / kind / "2026-10-01.jsonl"
                (out / kind / "2026-09-30.jsonl").write_text(src.read_text().replace("2026-10-01T", "2026-09-30T"))
            rc = hr.main(["--root", str(out), "--day", "2026-10-01"])
            self.assertEqual(rc, 0)
            J = json.loads((out / "reports" / "2026-10-01.json").read_text())
            S = J["summary"]
            n_final = sum(1 for l in REAL.read_text().splitlines() if hm.RE["final"].match(l))
            self.assertEqual(S["requests"]["completed"], n_final)
            self.assertTrue(set(S["by_prompt_bucket"]) <= {"<1K", "1-4K", "4-16K", "16-64K", "64K+"})
            self.assertIn("16-64K", S["by_prompt_bucket"])
            self.assertIsNotNone(S["cache"]["decode_hit_pct"])
            self.assertIsNotNone(S["mtp"]["accept_rate"])
            self.assertEqual(J["compare"]["requests.completed"]["delta"], 0)
            md = (out / "reports" / "2026-10-01.md").read_text()
            for h in ("by prompt length", "Decode speed by batch", "Concurrency", "long prefills", "Expert cache", "MTP", "Errors", "Slowest requests", "previous day"):
                self.assertIn(h, md)
            # duplicate output (at-least-once) is filtered by the report
            p = out / "requests" / "2026-10-01.jsonl"
            p.write_text(p.read_text() + p.read_text())
            hr.main(["--root", str(out), "--day", "2026-10-01"])
            J2 = json.loads((out / "reports" / "2026-10-01.json").read_text())
            self.assertEqual(J2["summary"]["requests"]["total"], S["requests"]["total"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ReportTrafficFilter(unittest.TestCase):
    """--clients keeps only engine requests whose server record (joined by daemon request id) came from the given user-agent prefixes and is not test-tagged."""
    def test_clients_filter(self):
        tmp = tempfile.mkdtemp(prefix="hivetf-")
        try:
            log, out = Path(tmp) / "hived.log", Path(tmp) / "out"
            shutil.copy(REAL, log)
            run(log, out, now=time.mktime((2026, 10, 1, 12, 0, 0, 0, 0, -1)), flush=True)
            rp = out / "requests" / "2026-10-01.jsonl"
            reqs = [json.loads(l) for l in rp.read_text().splitlines()]
            for i, r in enumerate(reqs):  # the fixture log predates daemon request ids on request lines — give each engine request one
                r["rid"] = r.get("rid") or f"d{i}"
            rp.write_text("".join(json.dumps(r) + "\n" for r in reqs))
            rids = [r["rid"] for r in reqs]
            self.assertGreaterEqual(len(rids), 3)
            t = time.mktime((2026, 10, 1, 11, 0, 0, 0, 0, -1)) * 1000
            recs = []
            for i, rid in enumerate(rids):
                ua = "AsyncOpenAI/Python 2.36.0" if i % 2 == 0 else "Python/3.12 aiohttp/3.13.2"
                c = {"user-agent": ua}
                if i == 0:
                    c["x-client-test"] = "1"  # test-tagged traffic is excluded even from a kept client
                recs.append({"rid": f"r{i}", "daemon_rid": rid, "t_recv": t + i, "t_done": t + i + 5, "client": c, "thinking_mode": "chat"})
            (out / "requests-server.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
            want = sum(1 for i in range(len(rids)) if i % 2 == 0 and i != 0)
            self.assertEqual(hr.main(["--root", str(out), "--day", "2026-10-01", "--clients", "AsyncOpenAI"]), 0)
            S = json.loads((out / "reports" / "2026-10-01.json").read_text())["summary"]
            self.assertEqual(S["requests"]["total"], want)
            self.assertEqual((S["traffic"]["kept"], S["traffic"]["total"]), (want, len(reqs)))
            self.assertIn("test-tagged", S["traffic"]["excluded"])
            self.assertIn("traffic filter", (out / "reports" / "2026-10-01.md").read_text())
            # no filter = every engine request, no traffic block (unchanged default)
            os.environ.pop("HIVE_REPORT_CLIENTS", None)
            self.assertEqual(hr.main(["--root", str(out), "--day", "2026-10-01"]), 0)
            S2 = json.loads((out / "reports" / "2026-10-01.json").read_text())["summary"]
            self.assertEqual(S2["requests"]["total"], len(reqs))
            self.assertNotIn("traffic", S2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ThinkingPregate(unittest.TestCase):
    """[pregate] cumulative totals → summed differences per minute bucket (on restart, the line's own value), plus the thinking-metrics section from server records (requests-server.jsonl)."""
    def test_pregate_deltas_and_thinking_section(self):
        tmp = tempfile.mkdtemp(prefix="hivepg-")
        try:
            d = Path(tmp)
            log, out = d / "hived.log", d / "out"
            banner = "[supervisor] ★ build dir=/hive/engine/build binary=/x sha256=abc elf-build-id=def"  # old banner form (existing logs)
            banner_new = "[supervisor] build dir=/hive/engine/build binary=/x sha256=abc elf-build-id=def"  # current scripts/hive-entrypoint.sh form
            on = "[runtime] decode pregate on (top-8 before attention · CPU pool prefetch of predicted misses · owned items on)"
            lines = [banner, on,
                     "[pregate M=1] layers 100 · posted 90 late 8 untrusted 2 · prefetch experts 800 · cpu experts 300 · covered 240 (recall 0.80 · precision 0.30) · "
                     "pool req 90 experts 700 read 12.5 MiB · finished 85 aborted 5",
                     "[pregate M=1] layers 300 · posted 280 late 15 untrusted 5 · prefetch experts 2400 · cpu experts 1000 · covered 840 (recall 0.84 · precision 0.35) · "
                     "pool req 280 experts 2200 read 40.0 MiB · finished 270 aborted 10",
                     banner_new, on,  # restart — totals start again from 0 (smaller than the second line → that line's value is this period's share)
                     "[pregate M=2] layers 40 · posted 40 late 0 untrusted 0 · prefetch experts 320 · cpu experts 100 · covered 60 (recall 0.60 · precision 0.19) · "
                     "pool req 40 experts 300 read 5.0 MiB · finished 40 aborted 0"]
            log.write_text("\n".join(lines) + "\n")
            now = time.mktime((2026, 10, 1, 12, 0, 0, 0, 0, -1))
            run(log, out, now=now, flush=True)
            ms = [json.loads(l) for p in (out / "metrics").glob("*.jsonl") for l in p.read_text().splitlines()]
            pg = {k: sum((m["pregate"].get(k) or 0) for m in ms) for k in hm.PG_FIELDS + ("n",)}
            self.assertEqual(pg["n"], 3)
            self.assertEqual((pg["layers"], pg["posted"], pg["late"], pg["pred"], pg["actual"], pg["covered"]), (340, 320, 15, 2720, 1100, 900))
            self.assertAlmostEqual(pg["read_mib"], 45.0)
            self.assertEqual(sum(1 for m in ms for e in m["events"] if e["kind"] == "pregate_on"), 2)
            self.assertEqual(sum(sum(m["unparsed"].values()) for m in ms), 0)
            # server records (shape of hive_server.py log_request) — 5 for that day + 1 for the previous day (filtered) + a broken line
            t0 = now * 1000
            recs = []

            def rec(i, mode, effort, n, cap=None, forced=None, tt=None):
                r = {"t_recv": str(t0 + i * 1000), "t_first": str(t0 + i * 1000 + 500), "t_done": str(t0 + i * 1000 + 5000), "rid": f"r{i}", "session": f"s{i % 2}",
                     "status": 200, "outcome": "done", "prompt_tokens": 100 + i, "thinking_mode": mode, "effort": effort, "think_cap": cap,
                     "done": {"n": n, "finish": "stop"}}
                if forced is not None:
                    r["think_forced"], r["think_tokens"] = forced, tt
                recs.append(r)
            rec(0, "chat", None, 50)
            rec(1, "thinking", 75, 3000, 3072, True, 3072)
            rec(2, "thinking", 75, 900, 3072, False, 400)
            rec(3, "thinking", 50, 700, 1024, True, 1024)
            rec(4, "thinking", 100, 20000)
            recs.append({"t_recv": str(t0 - 86400000), "t_done": str(t0 - 86400000 + 100), "rid": "old", "status": 200, "thinking_mode": "chat",
                         "done": {"n": 5, "finish": "stop"}})
            (out / "requests-server.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\nnot json\n")
            self.assertEqual(hr.main(["--root", str(out), "--day", "2026-10-01"]), 0)
            S = json.loads((out / "reports" / "2026-10-01.json").read_text())["summary"]
            th = S["thinking"]
            self.assertEqual((th["completed"], th["with_cap"], th["forced"], th["capped_share"]), (5, 3, 2, round(2 / 3, 4)))
            med = th["by_effort"]["thinking/75(medium)"]
            self.assertEqual((med["n"], med["with_cap"], med["forced"], med["caps"], med["think_tokens_p50"], med["completion_sum"]), (2, 2, 1, [3072], 1736.0, 3900))
            self.assertEqual(th["by_effort"]["thinking/100(xhigh)"]["with_cap"], 0)
            self.assertEqual(th["by_effort"]["chat"]["n"], 1)
            self.assertEqual(th["longest"][0]["effort"], "thinking/100(xhigh)")
            P = S["pregate"]
            self.assertEqual((P["on"], P["k"], P["recall"], P["precision"], P["read_mib"]), (True, 8, round(900 / 1100, 4), round(900 / 2720, 4), 45.0))
            md = (out / "reports" / "2026-10-01.md").read_text()
            for h in ("Decode pre-gate", "recall", "Thinking (reasoning)", "thinking/75(medium)", "top output tokens"):
                self.assertIn(h, md)
            # off + no server records: both sections show "off"/"none" and the report is otherwise normal
            (out / "requests-server.jsonl").unlink()
            for p in (out / "metrics").glob("*.jsonl"):
                p.write_text("")
            self.assertEqual(hr.main(["--root", str(out), "--day", "2026-10-01"]), 0)
            S2 = json.loads((out / "reports" / "2026-10-01.json").read_text())["summary"]
            self.assertEqual((S2["pregate"]["on"], S2["thinking"]["completed"]), (False, 0))
            md2 = (out / "reports" / "2026-10-01.md").read_text()
            self.assertIn("off (no startup banner", md2)
            self.assertIn("no server records", md2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def collapse_supported(d):
    p = Path(d) / "probe"
    p.write_bytes(b"x" * 8192)
    r = subprocess.run(["fallocate", "--collapse-range", "--offset", "0", "--length", "4096", str(p)], capture_output=True)
    ok = r.returncode == 0 and p.stat().st_size == 4096
    p.unlink()
    return ok


@unittest.skipUnless(ROTATE is not None and ROTATE.exists(), f"rotation script not found (HIVE_ROTATE_SCRIPT={ROTATE})")
class RotateScript(unittest.TestCase):
    """Runs the real log rotation script together with a concurrent O_APPEND writer and the monitor — zero lost or duplicated lines."""

    def test_concurrent_writer_no_loss(self):
        tmp = tempfile.mkdtemp(prefix="hiverot-")
        try:
            if not collapse_supported(tmp):
                self.skipTest("this filesystem does not support collapse-range")
            log = Path(tmp) / "hived.log"
            log.touch()
            writer = subprocess.Popen([sys.executable, "-c", r"""
import os, sys, time
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_APPEND)   # opened like hived (>> = O_APPEND)
i = 0
t_end = time.time() + float(sys.argv[2])
while time.time() < t_end:
    os.write(fd, ("[hived] %016x: prefill 10 tok 5 ms · decode 3 tok 4 ms (500.0 tok/s, batch 1) · hit 1 cpu 0 · resident 1/2 · mtp 0/0 in 0 steps · stop\n" % i).encode())
    i += 1
    if i % 20 == 0:
        time.sleep(0.01)   # ~2000 lines/s — just enough for rotation and the monitor to interleave several times
print(i)
""", str(log), "4"], stdout=subprocess.PIPE, text=True)
            env = dict(os.environ, HIVE_LOG_DIR=tmp, HIVE_LOG_FILES="hived.log", HIVE_LOG_ROTATE_MIN_MB="0")
            out = Path(tmp) / "out"
            rot = 0
            t0 = time.time()
            while writer.poll() is None:
                time.sleep(0.2)
                run(log, out, now=time.time())
                if log.stat().st_size > 8192:
                    r = subprocess.run(["bash", str(ROTATE), "--force"], env=env, capture_output=True, text=True)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    self.assertIn("collapse", r.stdout, r.stdout + r.stderr)
                    rot += 1
            n = int(writer.stdout.read().strip())
            run(log, out, now=time.time())
            self.assertGreaterEqual(rot, 3)
            # 1) reconstruct the original: archives (journal order) + live = contiguous 0..n-1
            J = [json.loads(l) for l in (Path(str(log) + ".rotations.jsonl")).read_text().splitlines()]
            blob = b"".join((Path(tmp) / e["archive"]).read_bytes() for e in sorted(J, key=lambda e: e["logical_start"])) + log.read_bytes()
            lines = blob.decode().splitlines()
            self.assertEqual(len(lines), n)
            self.assertEqual([int(l.split()[1][:-1], 16) for l in lines], list(range(n)))
            # 2) the monitor also has exactly n records (no duplicates, no losses)
            ids = [r["sid"] for r in records(out)]
            self.assertEqual(len(ids), n)
            self.assertEqual(len(set(ids)), n)
            print(f"\n  rotate: {rot} rotations · {n} lines · {time.time() - t0:.1f}s · 0 lost", file=sys.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
