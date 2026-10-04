#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""PRODUCTION save_image/load_image/reset_seq + HostImagePool + SeqImage/Seq fences on fake CUDA
(async worker-thread streams), option matrix HIVE_CKPT_ASYNC x HIVE_CKPT_PINNED_POOL_MB x
HIVE_CKPT_DELTA, plus injected pinned cudaHostAlloc failures (FAKE_HOST_ALLOC_FAIL) under
HIVE_CKPT_ASYNC=1 proving the pageable fallback (no abort, image == device, restore exact). No GPU. HIVE_TEST_SANITIZER=thread (under `setarch x86_64 -R`) race-checks the
fences; the fence negative control lives in tools/test_daemon_cpu.py (HIVE_DAEMON_NEGATIVE=fence).
Round 2: also builds engine/tests/test_runtime_host_cpu.cpp (M7 HIVE_PREFILL_PAUSE_PROMOTE on the production
promote_after_step + real ExpertStore, switch values 1/true/0/""/unset, plus REUSE_STAGE victim fence) and
engine/tests/test_prefill_host_cpu.cpp (production host helpers of HIVE_EARLY_STREAM / HIVE_ENGRAM_PAR / HIVE_IDX_MSUB_ACTUAL
and VitCache of HIVE_VIT_CACHE_MB)."""
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake')]
import harness  # noqa: E402


def main():
    tsan = harness.sanitizer() == 'thread'
    steps, pool_iters = ('400', '800') if tsan else ('900', '3000')
    # 4th field = FAKE_HOST_ALLOC_FAIL (every N-th pinned cudaHostAlloc fails; 1 = all): the async pinned-failure
    # fallback (pageable + synchronous copy, no abort) must keep every forward/restore exact.
    matrix = [(*m, '0') for m in itertools.product(('0', '1'), ('0', '1', '64'), ('0', '1'))]
    inject = [('1', '0', '0', '1'), ('1', '0', '1', '3'), ('1', '64', '1', '1'), ('1', '64', '0', '3'), ('0', '0', '1', '1')]
    if os.environ.get('HIVE_SNAPSHOT_QUICK'):
        matrix = [('0', '0', '0', '0'), ('1', '0', '1', '0'), ('1', '1', '1', '0'), ('1', '64', '0', '0')]
        inject = [('1', '0', '1', '3'), ('1', '64', '1', '1')]
    matrix += inject
    with tempfile.TemporaryDirectory(prefix='hive-snapshot-test-') as td:
        gen = Path(td) / 'fake_runtime_gen.cpp'
        gen.write_text(harness.runtime_source())
        objs = harness.objects(td, [gen, harness.ROOT / 'engine/src/expert_store.cpp', harness.ROOT / 'engine/src/cpu/expert_cpu.cpp',
                                    harness.FAKE / 'fake_checkpoint.cpp'])
        exe = harness.build(td, 'snapshot', '', [harness.ROOT / 'engine/tests/test_snapshot_cpu.cpp', *objs])
        bad = []
        # round 2: M7 promote_after_step (production text, real ExpertStore) + host helpers of L1/L2/L3/M3.
        rexe = harness.build(td, 'runtime_host', '', [harness.ROOT / 'engine/tests/test_runtime_host_cpu.cpp', *objs])
        for val, mode, extra in (('1', 'on', {}), ('true', 'on', {}), ('0', 'off', {}), ('', 'off', {}), (None, 'off', {}),
                                 ('1', 'on', {'HIVE_CACHE_REUSE_STAGE': '1', 'FAKE_CUDA_ASYNC': '1'})):
            env = harness.tsan_env()
            env.pop('HIVE_PREFILL_PAUSE_PROMOTE', None)
            if val is not None:
                env['HIVE_PREFILL_PAUSE_PROMOTE'] = val
            env.update(extra)
            p = subprocess.run([rexe, mode], env=env, capture_output=True, text=True, timeout=600)
            out = p.stdout + p.stderr
            print(f'[HIVE_PREFILL_PAUSE_PROMOTE={val!r} {extra}] ' + ' / '.join(l for l in out.splitlines() if l.startswith(('runtime host CPU', 'FAIL', 'WARNING: ThreadSanitizer'))))
            if p.returncode:
                bad.append(('m7', val, mode, str(extra), p.returncode, out[-4000:]))
        (Path(td) / 'prefill_helpers_gen.h').write_text(harness.prefill_helpers_source())
        hexe = harness.build(td, 'prefill_host', '', [harness.ROOT / 'engine/tests/test_prefill_host_cpu.cpp'], extra=['-I' + td])
        p = subprocess.run([hexe], env=harness.tsan_env(), capture_output=True, text=True, timeout=600)
        out = p.stdout + p.stderr
        print('\n'.join(l for l in out.splitlines() if l.startswith(('prefill host CPU', 'FAIL', 'WARNING: ThreadSanitizer'))))
        if p.returncode:
            bad.append(('prefill-host', '', '', '', p.returncode, out[-4000:]))
        for asyn, pool, delta, fail in matrix:
            env = harness.tsan_supp_env(harness.tsan_env(), tmp_dir=td)
            env.update({'HIVE_CKPT_ASYNC': asyn, 'HIVE_CKPT_PINNED_POOL_MB': pool, 'HIVE_CKPT_DELTA': delta,
                        'FAKE_HOST_ALLOC_FAIL': fail, 'FAKE_CUDA_ASYNC': '1', 'FAKE_CUDA_DELAY_US': '0' if tsan else '30'})
            p = subprocess.run([exe, steps, pool_iters], env=env, capture_output=True, text=True, timeout=600)
            out = p.stdout + p.stderr
            lines = [l for l in out.splitlines() if l.startswith(('snapshot CPU', 'FAIL', 'WARNING: ThreadSanitizer', 'SUMMARY'))]
            print(f'[async={asyn} pool_mb={pool} delta={delta} host_alloc_fail={fail}] ' + ' / '.join(lines))
            rc = p.returncode
            warned = out.count('snapshot pinned allocation failed')
            if asyn != '0' and fail != '0' and warned != 1:  # absorbed, logged exactly once per process
                rc = rc or 101
                out += f'\nexpected exactly one pinned-fallback warning, got {warned}'
            if (asyn == '0' or fail == '0') and warned:
                rc = rc or 102
                out += '\npinned-fallback warning without an injected async failure'
            if rc:
                bad.append((asyn, pool, delta, fail, rc, out[-4000:]))
        if tsan:  # negative control: fake forward skips the production snapshot-fence wait -> must be reported
            env = harness.tsan_supp_env(harness.tsan_env(), tmp_dir=td)
            env.update({'HIVE_CKPT_ASYNC': '1', 'HIVE_CKPT_PINNED_POOL_MB': '64', 'HIVE_CKPT_DELTA': '1', 'FAKE_CUDA_ASYNC': '1',
                        'FAKE_SKIP_SNAPSHOT_FENCE': '1'})
            p = subprocess.run([exe, steps, '10'], env=env, capture_output=True, text=True, timeout=600)
            out = p.stdout + p.stderr
            neg = p.returncode == 66 and 'data race' in out and 'fakecuda' in out
            print(f'TSAN negative control (forward without snapshot-fence wait): {"DETECTED" if neg else "NOT DETECTED"}')
            if not neg:
                bad.append(('neg', '', '', '', p.returncode, out[-3000:]))
        for b in bad:
            print('FAIL', b[:5], '\n', b[5])
        if bad:
            sys.exit(1)
        print(f'snapshot CPU suite OK ({len(matrix)} option combinations, sanitizer {harness.sanitizer()})')


if __name__ == '__main__':
    main()
