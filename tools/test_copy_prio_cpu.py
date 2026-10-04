#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HIVE_DECODE_COPY_PRIO — CPU checks of paced promotion H2D (no GPU).

engine/tests/test_copy_prio_cpu.cpp on fake CUDA/NUMA with the REAL ExpertStore (engine/src/expert_store.cpp) and the REAL
hs::Dispatcher (decode_dispatch.h, after_dma hook): paced == immediate (slot table, stats, H2D bytes, VRAM bytes), residency
at the first head after the last piece (t+2), FIFO, flush before the next decision, dispatcher-thread pacing with exact bytes.
Stream-op ORDER (gate after the layer's demand DMA, per-layer budget, event after the last piece) is checked on the production
promo_pump text in tools/test_cache_cpu.py.
Runs twice: synchronous fake CUDA and FAKE_CUDA_ASYNC=1 (per-stream worker threads — a commit before the copies land would read
0xCD bytes). Negative control: expert_store.cpp without the "unissued batch" stop in commit_pending MUST fail.
HIVE_TEST_SANITIZER=thread (under `setarch x86_64 -R`): every run race-checked.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake')]
import harness  # noqa: E402

ROOT = harness.ROOT
TEST = ROOT / 'engine/tests/test_copy_prio_cpu.cpp'
STORE = ROOT / 'engine/src/expert_store.cpp'
OTHER = [ROOT / 'engine/src/cpu/expert_cpu.cpp', harness.FAKE / 'fake_checkpoint.cpp']
STOP = '    if (!b.issued) return;\n'


def run(exe, args, env_extra=None):
    env = harness.tsan_env()
    env.update(env_extra or {})
    p = subprocess.run([exe, *map(str, args)], env=env, capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def main():
    tsan = harness.sanitizer() == 'thread'
    failures = []
    with tempfile.TemporaryDirectory(prefix='hive-copyprio-test-') as td:
        exe = harness.build(td, 'copyprio', '', [TEST, STORE, *OTHER], extra=['-I' + str(ROOT / 'engine/tests')])
        args = (120, 60, 3) if tsan else (400, 300, 3)
        for mode in ('0', '1'):
            rc, out = run(exe, args, {'FAKE_CUDA_ASYNC': mode})
            print(f'[FAKE_CUDA_ASYNC={mode}] ' + ' | '.join(l for l in out.splitlines() if l.startswith(('copy-prio CPU', 'FAIL', 'WARNING: ThreadSanitizer', 'SUMMARY'))))
            if rc != 0:
                failures.append(f'run async={mode} {args}: exit {rc}\n' + out[-4000:])
        src = STORE.read_text()
        assert src.count(STOP) == 1, 'commit_pending unissued stop moved: update STOP'
        mut = Path(td) / 'expert_store_mut.cpp'
        mut.write_text(src.replace(STOP, ''))
        mexe = harness.build(td, 'copyprio-mutant', '', [TEST, mut, *OTHER], extra=['-I' + str(ROOT / 'engine/tests')])
        rc, out = run(mexe, (20, 10, 2))
        caught = rc != 0 and 'FAIL' in out
        print(f'negative control (commit_pending without the unissued stop): {"DETECTED" if caught else "NOT DETECTED"}')
        if not caught:
            failures.append('mutant without the unissued stop passed\n' + out[-3000:])
    if failures:
        print('\n'.join(failures))
        sys.exit(1)
    print('copy-prio CPU suite OK (sanitizer %s)' % harness.sanitizer())


if __name__ == '__main__':
    main()
