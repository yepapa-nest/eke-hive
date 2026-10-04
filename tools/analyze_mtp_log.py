#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Aggregate the DSpark speculative-decoding log in hived.log (the `[mtp] …` lines of HIVE_TRACE_MTP=1) — for calibrating the threshold (--mtp-conf).
Usage: analyze_mtp_log.py hived.log [there is no --since 'HH:MM' — trim the file with tail and pass that]
Output: step count, draft/accept totals, tokens per step, step time (draft, verify), effective tok/s, acceptance rate per draft position i,
acceptance rate per confidence bucket (position 1 = first draft)."""
import re
import sys
from collections import defaultdict

LINE = re.compile(r"\[mtp\] pos (\d+) draft ([\d.]+) ms · verify (\d+) rows ([\d.]+) ms · accepted (\d+)/(\d+) · conf \[([^\]]*)\] · hit (\d+) cpu (\d+)")
NODRAFT = re.compile(r"\[mtp\] pos (\d+) draft ([\d.]+) ms · no draft \(conf0 ([-\d.]+)(?:, single ([\d.]+) ms)?\)")


def main():
    path = sys.argv[1]
    steps = 0; drafted = 0; accepted = 0; t_draft = 0.0; t_verify = 0.0; tokens = 0; nodraft = 0
    pos_tot = defaultdict(int); pos_acc = defaultdict(int)  # submitted / accepted at position i (1..) (a position is only submitted if all earlier ones were accepted, so this is a conditional acceptance rate)
    bucket_tot = defaultdict(int); bucket_acc = defaultdict(int)
    hit = 0; cpu = 0
    t_single = 0.0; missing_single = 0
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if m:
            steps += 1
            k = int(m.group(6)); a = int(m.group(5))
            drafted += k; accepted += a; tokens += a + 1
            t_draft += float(m.group(2)); t_verify += float(m.group(4))
            hit += int(m.group(8)); cpu += int(m.group(9))
            confs = [float(x) for x in m.group(7).split(",") if x.strip()]
            # Only accepted positions and the first rejection were tested against
            # the true prefix. Later speculative rows are censored, not failures.
            for i in range(min(k, a + 1)):
                pos_tot[i + 1] += 1
                b = int(confs[i] // 1.0) if i < len(confs) else 0
                bucket_tot[b] += 1
                if i < a:
                    pos_acc[i + 1] += 1
                    bucket_acc[b] += 1
            continue
        m = NODRAFT.search(line)
        if m:
            nodraft += 1
            t_draft += float(m.group(2))
            tokens += 1
            if m.group(4) is None: missing_single += 1
            else: t_single += float(m.group(4))
    if steps + nodraft == 0:
        print("[mtp] no lines"); return
    # The "single N ms" of a no-draft line is not a measurement but the daemon's EMA estimate (hived.cpp ema_single_ms) — kept out of the observed time and shown separately as an estimate.
    t_total = t_draft + t_verify
    print(f"steps {steps} (+{nodraft} no-draft) · drafted {drafted} accepted {accepted} ({accepted / max(1, drafted):.1%}) · tokens/step {tokens / (steps+nodraft):.2f}")
    print(f"time/step draft {t_draft / (steps + nodraft):.1f} ms · verify {t_verify / max(1,steps):.1f} ms · no-draft single EMA estimate total {t_single:.1f} ms (estimate, not observed; missing {missing_single})")
    print(f"observed-log tokens/s {tokens / max(1e-9,t_total)*1000:.1f} (draft+verify logged time only; no-draft single decode time is unlogged — upper bound); partial, excludes unlogged skip steps/scheduling/transport; NOT service throughput")
    if t_single: print(f"estimated tokens/s incl. EMA single estimate {tokens / max(1e-9, t_total + t_single) * 1000:.1f} (ESTIMATE)")
    print(f"verify hit/(hit+CPU) {hit / max(1, hit + cpu):.1%} (hit {hit} cpu {cpu}; DMA rows absent, not total hit rate)")
    print("position  submitted  accepted  rate(conditional)")
    for i in sorted(pos_tot):
        print(f"  {i:<8}{pos_tot[i]:<11}{pos_acc[i]:<10}{pos_acc[i] / pos_tot[i]:.1%}")
    print("conf bucket(floor)  submitted  accepted  rate")
    for b in sorted(bucket_tot):
        print(f"  [{b:+d},{b + 1:+d})        {bucket_tot[b]:<11}{bucket_acc[b]:<10}{bucket_acc[b] / bucket_tot[b]:.1%}")


if __name__ == "__main__":
    main()
