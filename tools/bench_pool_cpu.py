#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Real ExpertStore CPU pool on fake CUDA/NUMA at the real expert shape (dim 5120, inter 2304):
one decode layer's CPU misses (start_jobs -> wait_jobs) with HIVE_CPU_GEMV2 unset vs =1 (and, on top of
GEMV2=1: HIVE_CPU_SPLIT13 / HIVE_CPU_P2_PREFETCH / both; same-expert split jobs vs one merged multi-row job), plus a bit check
against expert_forward. No GPU. Numbers are for the machine running the benchmark, not necessarily the serving workstation (Zen2 3975WX).

usage: tools/bench_pool_cpu.py [threads_per_node=8] [experts=16] [iters=200]
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().with_name('cpu_fake')))
import harness  # noqa: E402

ROOT = harness.ROOT


def main():
    args = sys.argv[1:] or ['8', '16', '200']
    os.environ.setdefault('HIVE_TEST_SANITIZER', 'none')  # benchmark without instrumentation (-O3); the suite's default UBSan distorts timings
    with tempfile.TemporaryDirectory(prefix='hive-bench-pool-') as td:
        exe = harness.build(td, 'bench_pool', '', [ROOT / 'engine/tests/bench_pool_cpu.cpp', ROOT / 'engine/src/expert_store.cpp',
                                                   ROOT / 'engine/src/cpu/expert_cpu.cpp', harness.FAKE / 'fake_checkpoint.cpp'], extra=['-O3'])
        rc = 0
        # GEMV2 off/on + (on top of GEMV2 on) HIVE_CPU_SPLIT13, HIVE_CPU_P2_PREFETCH, both. BENCH_POOL_VARIANTS=base runs only the first two variants.
        variants = [{'HIVE_CPU_GEMV2': ''}, {'HIVE_CPU_GEMV2': '1'}]
        if os.environ.get('BENCH_POOL_VARIANTS', 'all') != 'base':
            variants += [{'HIVE_CPU_GEMV2': '1', 'HIVE_CPU_SPLIT13': '1'}, {'HIVE_CPU_GEMV2': '1', 'HIVE_CPU_P2_PREFETCH': '1'},
                         {'HIVE_CPU_GEMV2': '1', 'HIVE_CPU_SPLIT13': '1', 'HIVE_CPU_P2_PREFETCH': '1'}]
        # HIVE_CPU_MULTIROW2 (multi-row K-tile kernel), HIVE_CPU_FINE (finer items for multi-row jobs) — on top of GEMV2 on.
        #   BENCH_POOL_VARIANTS=b3 runs only the GEMV2-on baseline + these variants (quick A/B).
        b3 = [{'HIVE_CPU_GEMV2': '1', 'HIVE_CPU_MULTIROW2': '1'}, {'HIVE_CPU_GEMV2': '1', 'HIVE_CPU_FINE': '1'},
              {'HIVE_CPU_GEMV2': '1', 'HIVE_CPU_MULTIROW2': '1', 'HIVE_CPU_FINE': '1'},
              {'HIVE_CPU_GEMV2': '1', 'HIVE_CPU_MULTIROW2': '1', 'HIVE_CPU_FINE': '1', 'HIVE_CPU_SPLIT13': '1'},
              {'HIVE_CPU_GEMV2': '1', 'HIVE_CPU_MULTIROW2': '2', 'HIVE_CPU_FINE': '1'}, {'HIVE_CPU_GEMV2': '1', 'HIVE_CPU_MULTIROW2': '3', 'HIVE_CPU_FINE': '1'}]
        #   (MULTIROW2 value = minimum row count that uses _v3 — "1" = default 5, "2"/"3" = threshold comparison: Zen2's slow L2 may favor a lower threshold)
        if os.environ.get('BENCH_POOL_VARIANTS') == 'b3':
            variants = [{'HIVE_CPU_GEMV2': '1'}, {'HIVE_CPU_GEMV2': '1', 'HIVE_CPU_SPLIT13': '1'}] + b3
        elif os.environ.get('BENCH_POOL_VARIANTS', 'all') != 'base':
            variants += b3
        for var in variants:
            env = {**os.environ, 'HIVE_CPU_SPLIT13': '', 'HIVE_CPU_P2_PREFETCH': '', 'HIVE_CPU_MULTIROW2': '', 'HIVE_CPU_FINE': '', **var}
            p = subprocess.run([exe, *args], env=env, capture_output=True, text=True, timeout=1800)
            print(p.stdout + p.stderr, end='')
            rc |= p.returncode
        return rc


if __name__ == '__main__':
    sys.exit(main())
