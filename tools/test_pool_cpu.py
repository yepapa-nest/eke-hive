#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""REAL ExpertStore CPU worker pool (engine/src/expert_store.cpp + cpu/expert_cpu.cpp) on fake
CUDA/NUMA: exact outputs vs expert_forward over many random batches with stealing, allocation
failure at every allocation inside start_jobs, spin mode, pool shutdown, jobs_done_ms published
by wait_jobs (D7, strict), cache-side checks (staging-reuse D2D stream ordering, cache
event trace format + tools/check_cache_events.py replay, "0"/"" switch parsing D8, touch_at B6). No GPU.

HIVE_TEST_SANITIZER=thread (under `setarch x86_64 -R`) additionally:
  * every run is race-checked with NO suppressions (the D2 items rebuild race and the D6
    destructor UAF are fixed; they are plain passes, not XFAILs);
  * NEG  injected race (test writes an input the pool reads) MUST be reported;
  * NEG  mutant expert_store.cpp with the pre-fix protocol (no start_jobs quiesce, loop bound
         read from items.size()) MUST be reported, proving the stress still reaches the D2 window.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake')]
import harness  # noqa: E402

ROOT = harness.ROOT
STORE = ROOT / 'engine/src/expert_store.cpp'
OTHER = [ROOT / 'engine/src/cpu/expert_cpu.cpp', harness.FAKE / 'fake_checkpoint.cpp']
TEST = ROOT / 'engine/tests/test_expert_pool_cpu.cpp'
QUIESCE = '  while (pool_busy_.load() != 0) std::this_thread::yield();\n'
NITEMS = '      const int n = q->n_items.load(std::memory_order_acquire);\n'
LINE = re.compile(r'^[BUEPCDI],-?\d+,-?\d+,\d+\.\d{3}$')


def run(exe, args, env_extra=None, timeout=300, cwd=None):
    env = {k: v for k, v in harness.tsan_env().items() if not k.startswith(('HIVE_CACHE_', 'FAKE_CUDA_'))}
    env.update(env_extra or {})
    p = subprocess.run([exe, *map(str, args)], env=env, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    return p.returncode, p.stdout + p.stderr


def cache_checks(exe, td, failures):
    trace = Path(td) / 'events.csv'
    for reuse, want in (('1', 1), ('true', 1), ('0', 0), ('', 0)):
        if trace.exists():
            trace.unlink()
        env = {'HIVE_CACHE_REUSE_STAGE': reuse, 'CACHE_EXPECT_D2D': str(want), 'FAKE_CUDA_ASYNC': '1',
               'FAKE_CUDA_DELAY_US': '300', 'HIVE_CACHE_EVENTS': str(trace)}
        rc, out = run(exe, (0, 1, 'cache'), env)
        print('\n'.join(l for l in out.splitlines() if l.startswith(('pool CPU', 'FAIL', 'WARNING: ThreadSanitizer'))))
        if rc != 0:
            failures.append(f'cache checks REUSE_STAGE={reuse!r}: exit {rc}\n' + out[-4000:])
            continue
        lines = trace.read_text().splitlines()
        bad = [l for l in lines if not LINE.match(l)]
        if not lines or bad:
            failures.append(f'cache event trace format: {len(lines)} lines, bad {bad[:3]}')
        chk = subprocess.run([sys.executable, str(ROOT / 'tools/check_cache_events.py'), str(trace)], capture_output=True, text=True)
        if chk.returncode != 0 or '"ok": true' not in chk.stdout:
            failures.append('check_cache_events.py rejected the trace: ' + chk.stdout + chk.stderr)
    print(f'pool CPU: cache event trace format ok, check_cache_events.py replay ok ({len(lines)} lines)')
    # HIVE_STAGE_COPY2D: records produced by 8 calls including 2D copies (staging · promotion · REUSE_STAGE D2D source) are byte-identical to the 1D path (including cache_main's independent reference check)
    for reuse, want in (('1', 1), ('', 0)):
        env = {'HIVE_STAGE_COPY2D': '1', 'HIVE_CACHE_REUSE_STAGE': reuse, 'CACHE_EXPECT_D2D': str(want), 'FAKE_CUDA_ASYNC': '1', 'FAKE_CUDA_DELAY_US': '300'}
        rc, out = run(exe, (0, 1, 'cache'), env)
        print('\n'.join(l for l in out.splitlines() if l.startswith(('pool CPU: copy_rec', 'FAIL', 'WARNING: ThreadSanitizer'))))
        if rc != 0:
            failures.append(f'cache checks HIVE_STAGE_COPY2D=1 REUSE_STAGE={reuse!r}: exit {rc}\n' + out[-4000:])
    # D8: HIVE_CACHE_EVENTS="0" / "" = off — no file named "0" and no trace
    for val in ('0', ''):
        cwd = Path(td) / ('cwd-' + (val or 'empty'))
        cwd.mkdir()
        rc, out = run(exe, (0, 1, 'cache'), {'HIVE_CACHE_EVENTS': val, 'CACHE_EXPECT_D2D': '0'}, cwd=cwd)
        if rc != 0 or any(cwd.iterdir()):
            failures.append(f'HIVE_CACHE_EVENTS={val!r}: exit {rc}, files {[p.name for p in cwd.iterdir()]}\n' + out[-2000:])
    print('pool CPU: HIVE_CACHE_EVENTS="0"/"" leaves tracing off (no file created)')


def main():
    tsan = harness.sanitizer() == 'thread'
    with tempfile.TemporaryDirectory(prefix='hive-pool-test-') as td:
        exe = harness.build(td, 'pool', '', [TEST, STORE, *OTHER])
        failures = []
        strict = {'HIVE_POOL_STRICT_TDONE': '1'}  # D7 regression: any stale jobs_done_ms after wait_jobs fails
        # HIVE_CPU_SPLIT13 (separate w1/w3 items · the second of the pair quantizes) · HIVE_CPU_P2_PREFETCH (w2 prefetch while waiting at the barrier) — same bit comparison · TSAN
        b2 = {'HIVE_CPU_SPLIT13': '1', 'HIVE_CPU_P2_PREFETCH': '1'}
        runs = [((400, 4), {}), ((200, 3), {'HIVE_CPU_SPIN_US': '50'}), ((150, 2), {'FAKE_NUMA_NODES': '1'}), ((1000, 4), {}),
                ((400, 4), {'HIVE_CPU_SPLIT13': '1'}), ((200, 3), {'HIVE_CPU_P2_PREFETCH': '1'}), ((400, 4), b2),
                ((150, 2), {**b2, 'FAKE_NUMA_NODES': '1'}), ((200, 3), {**b2, 'HIVE_CPU_SPIN_US': '50', 'HIVE_CPU_P1': '64'})]
        # HIVE_CPU_MULTIROW2 (=2: every R ≥ 2 uses _v3 — widest coverage) · HIVE_CPU_FINE (multi-row jobs in 4 pieces · P2/2) — same bit comparison · TSAN.
        #   odd P2 (17 → FINE 8) · P1 64 (half 32) · mixed with SPLIT13 (R = 1 jobs use the SPLIT13 pair · R ≥ 2 the 4 FINE pieces in one group) · GEMV2 on/off
        b3 = {'HIVE_CPU_MULTIROW2': '2', 'HIVE_CPU_FINE': '1'}
        runs += [((400, 4), {'HIVE_CPU_MULTIROW2': '2'}), ((400, 4), {'HIVE_CPU_FINE': '1'}), ((400, 4), {**b3, 'HIVE_CPU_GEMV2': '1'}),
                 ((200, 3), {**b3, 'HIVE_CPU_SPLIT13': '1', 'HIVE_CPU_P2': '17'}), ((200, 3), {**b3, **b2, 'HIVE_CPU_P1': '64', 'HIVE_CPU_SPIN_US': '50'}),
                 ((150, 2), {**b3, 'FAKE_NUMA_NODES': '1'})]
        # HIVE_DECODE_PREGATE: owned mode (fixed worker per item · no stealing · core pinning attempted) + prefetch requests ahead of each group — same bit comparison · TSAN.
        #   spin 0 (sleep) · default · short · one NUMA node (half record = pool index) · piece groups (SPLIT13 · FINE → absorbed into regular claiming · prefetch unchanged) · 1 and 3 workers (remainder shares)
        pg = {'HIVE_DECODE_PREGATE': '8'}
        runs += [((400, 4), pg), ((200, 3), {**pg, 'HIVE_DECODE_PREGATE_SPIN_US': '1'}), ((200, 4), {**pg, 'HIVE_DECODE_PREGATE_SPIN_US': '50', 'HIVE_CPU_SPIN_US': '50'}),
                 ((150, 2), {**pg, 'FAKE_NUMA_NODES': '1'}), ((200, 3), {**pg, **b2}), ((200, 3), {**pg, **b3}), ((150, 1), pg),
                 ((200, 3), {**pg, 'HIVE_CPU_P1': '64', 'HIVE_CPU_P2': '17'})]
        if tsan:
            runs = [((300, 4), {}), ((150, 3), {'HIVE_CPU_SPIN_US': '50'}), ((300, 4), {}), ((300, 4), b2), ((150, 3), {**b2, 'HIVE_CPU_SPIN_US': '50'}),
                    ((300, 4), b3), ((150, 3), {**b3, 'HIVE_CPU_SPLIT13': '1', 'HIVE_CPU_SPIN_US': '50', 'HIVE_CPU_P2': '17'}),
                    ((300, 4), pg), ((150, 3), {**pg, 'HIVE_DECODE_PREGATE_SPIN_US': '1'}), ((150, 3), {**pg, **b2, 'HIVE_CPU_SPIN_US': '50'})]
        for args, extra in runs:
            rc, out = run(exe, args, {**strict, **extra})
            lines = [l for l in out.splitlines() if l.startswith(('pool CPU', 'FAIL', 'WARNING: ThreadSanitizer', 'SUMMARY'))]
            print('\n'.join(lines))
            if rc != 0:
                failures.append(f'pool run {args} {extra}: exit {rc}\n' + out[-4000:])
        cache_checks(exe, td, failures)
        # SPLIT13 negative control: if the **first** of the pair quantizes (the partner matrix is not done yet), the output must be wrong (bit comparison) or TSAN must report the race
        src = STORE.read_text()
        q = '* nblk + (it.q0 - base) / p1_rows].fetch_add(1, std::memory_order_acq_rel) == it.parts - 1;'
        assert src.count(q) == 1, 'SPLIT13 quant line moved: update this test'
        mut13 = Path(td) / 'expert_store_split13_mutant.cpp'
        mut13.write_text(src.replace(q, q.replace('== it.parts - 1;', '== 0;')))
        m13 = harness.build(td, 'pool-split13-mutant', '', [TEST, mut13, *OTHER], extra=['-I' + str(ROOT / 'engine/src')])
        rc, out = run(m13, (200, 4), {'HIVE_CPU_SPLIT13': '1'})
        print(f'SPLIT13 negative control (first finisher quantizes): {"DETECTED" if rc != 0 else "NOT DETECTED"} (exit {rc})')
        if rc == 0:
            failures.append('SPLIT13 mutant (first finisher quantizes) passed\n' + out[-3000:])
        # FINE: must be caught if the **second** of four pieces quantizes (the remaining pieces are not done) · and if the quantize range shrinks to the piece [r0, r1) (half the block only)
        for name, a, b in (('second of four quantizes', q, q.replace('== it.parts - 1;', '== (it.parts == 4 ? 1 : it.parts - 1);')),
                           ('quantizes only its own half', 'const int q0 = it.parts == 1 ? it.r0 : it.q0, q1 = it.parts == 1 ? it.r1 : it.q1;',
                            'const int q0 = it.r0, q1 = it.r1;')):
            assert src.count(a) == 1, 'FINE quant lines moved: update this test'
            mf = Path(td) / 'expert_store_fine_mutant.cpp'
            mf.write_text(src.replace(a, b))
            me = harness.build(td, 'pool-fine-mutant-' + str(len(name)), '', [TEST, mf, *OTHER], extra=['-I' + str(ROOT / 'engine/src')])
            rc, out = run(me, (200, 4), {'HIVE_CPU_FINE': '1'})
            print(f'FINE negative control ({name}): {"DETECTED" if rc != 0 else "NOT DETECTED"} (exit {rc})')
            if rc == 0:
                failures.append(f'FINE mutant ({name}) passed\n' + out[-3000:])
        if tsan:
            rc, out = run(exe, (200, 4), {'HIVE_POOL_INJECT_RACE': '1'})
            neg = rc == 66 and 'test_expert_pool_cpu.cpp' in out and 'data race' in out
            print(f'TSAN negative control (injected input write during pool compute): {"DETECTED" if neg else "NOT DETECTED"}')
            if not neg:
                failures.append('TSAN negative control not detected\n' + out[-3000:])
            src = STORE.read_text()
            assert src.count(QUIESCE) == 1 and src.count(NITEMS) == 1, 'D2 fix lines moved: update QUIESCE/NITEMS in this test'
            # mutant = the pre-fix protocol (no quiesce, bound read from the vector being rebuilt): the stress must still reach
            # the straggler-vs-rebuild window the quiesce closes, or the unsuppressed PASS above proves nothing.
            mut = Path(td) / 'expert_store_mutant.cpp'
            mut.write_text(src.replace(QUIESCE, '  // mutant: quiesce removed\n')
                           .replace(NITEMS, '      const int n = (int)q->items.size();  // mutant\n'))
            mexe = harness.build(td, 'pool-mutant', '', [TEST, mut, *OTHER], extra=['-I' + str(ROOT / 'engine/src')])
            got = False
            for attempt in range(5):
                rc, out = run(mexe, (300, 4))
                if rc == 66 and 'expert_store_mutant.cpp' in out:
                    got = True
                    break
            print(f'TSAN negative control (pre-fix D2 protocol mutant): {"DETECTED" if got else "NOT DETECTED"} after {attempt + 1} run(s)')
            # PREGATE negative control: removing the ownership check (every worker takes every item) makes several workers write the same output — TSAN must catch it for the owned-group PASS to mean anything
            own = '      if (it.own != tid) continue;\n'
            assert src.count(own) == 1, 'P1 owned check moved: update this test'
            mo = Path(td) / 'expert_store_owned_mutant.cpp'
            mo.write_text(src.replace(own, '      // mutant: ownership check removed\n'))
            moexe = harness.build(td, 'pool-owned-mutant', '', [TEST, mo, *OTHER], extra=['-I' + str(ROOT / 'engine/src')])
            rc, out = run(moexe, (150, 3), {'HIVE_DECODE_PREGATE': '8'})
            det = rc != 0
            print(f'TSAN negative control (owned batch without the ownership check): {"DETECTED" if det else "NOT DETECTED"} (exit {rc})')
            if not det:
                failures.append('P1 owned mutant (no ownership check) not detected\n' + out[-3000:])
            if not got:
                failures.append('D2 mutant (no quiesce) not detected by TSAN in 5 runs\n' + out[-3000:])
        if failures:
            print('\n'.join(failures))
            sys.exit(1)
        print('pool CPU suite OK (sanitizer %s)' % harness.sanitizer())


if __name__ == '__main__':
    main()
