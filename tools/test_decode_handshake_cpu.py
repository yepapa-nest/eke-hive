#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HIVE_DECODE_STEP_GRAPH — CPU checks of the decode step handshake (no GPU).

engine/tests/test_decode_handshake_cpu.cpp on fake CUDA/NUMA with the REAL ExpertStore CPU pool:
  * hs::sort_desc_by_count (device plan + host replay) == std::sort with the legacy comparator;
  * hs::plan_host / build_tables (the function hs_plan's thread 0 runs) == a transcription of the legacy
    Runtime::moe_decode_experts classification and tables — this script checks that every transcribed legacy
    line is still verbatim in runtime.cpp (inside moe_decode_experts) and that forward_batch_step builds its DMA
    share table with the same expression as the legacy gpu_share_left line;
  * hs::Dispatcher handshake stress (fake GPU thread speaking the mailbox protocol): CPU rows bit-exact vs
    cpu::expert_forward, DMA copy order, empty steps, GPU wait timeout + disarm, callback exception -> cpu_err,
    sleep-poll mode.
HIVE_TEST_SANITIZER=thread (under `setarch x86_64 -R`): every run race-checked with no suppressions, plus a
negative control: a mutant dispatcher that publishes done_seq with a relaxed store MUST be reported.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake')]
import harness  # noqa: E402

ROOT = harness.ROOT
TEST = ROOT / 'engine/tests/test_decode_handshake_cpu.cpp'
STORE = ROOT / 'engine/src/expert_store.cpp'
OTHER = [ROOT / 'engine/src/cpu/expert_cpu.cpp', harness.FAKE / 'fake_checkpoint.cpp']
DISPATCH = ROOT / 'engine/include/hive/decode_dispatch.h'

LEGACY_LINES = [  # transcribed into legacy_plan() in the C++ test — must stay verbatim in Runtime::moe_decode_experts
    '  for (int i = 0; i < R; ++i) rows_by_e[fill[w.route_ids_h[i]]++] = i;',
    '  const int dma_cap = kind == kDmaShort ? short_dma_cap_ : 8;',
    '  else if (n_miss >= 3) gpu_share_left = std::min(std::min(dma_cap, store_.staging_slots()), '
    'std::max(opt_.decode_gpu_share, (int)(frac * n_miss + 0.5f)));',
    '  std::sort(miss_e, miss_e + n_miss_e, [&](int a, int b) { return n_of(a) > n_of(b); });',
    '    if (gpu_share_left > 0) { --gpu_share_left; str_e[n_str_e++] = miss_e[i]; }',
    '    for (int i0 = cnt[e]; i0 < cnt[e + 1]; i0 += 8) {',
    '      d.row0 = goff; d.n = std::min(8, cnt[e + 1] - i0);',
    '    const int si = stage_next_++ % store_.staging_slots();',
    '      for (int i0 = cnt[e]; i0 < cnt[e + 1]; i0 += ExpertStore::kMaxRows) {',
]
STEP_SHARE = ('    P.share[n] = n_miss >= 3 ? std::min(std::min(dma_cap, store_.staging_slots()), '
              'std::max(opt_.decode_gpu_share, (int)(frac * n_miss + 0.5f))) : 0;')
TEST_SHARE = ('    share[n] = n_miss >= 3 ? std::min(std::min(dma_cap, staging_slots), '
              'std::max(decode_gpu_share, (int)(frac * n_miss + 0.5f))) : 0;')
DONE_STORE = '    st_rel(&io_.ctrl->done_seq, seq);'


def source_guards():
    rt = (ROOT / 'engine/src/runtime.cpp').read_text()
    i = rt.index('void Runtime::moe_decode_experts(')
    body = rt[i:rt.index('\n}\n', i)]
    missing = [l for l in LEGACY_LINES if l not in body]
    assert not missing, 'legacy moe_decode_experts lines changed — update legacy_plan() in the C++ test:\n' + '\n'.join(missing)
    j = rt.index('bool Runtime::forward_batch_step(')
    step = rt[j:rt.index('\n}\n', j)]
    assert STEP_SHARE in step, 'forward_batch_step DMA share expression drifted from the legacy gpu_share_left line'
    assert TEST_SHARE in TEST.read_text(), 'test make_share() drifted'
    # Regression measured on the GPU (only the first step of a run returned 0x2: lazy module loading of the dispatcher's first hs_signal launch
    #   was blocked behind the spin-wait kernel): hs_preload must be called before the dispatcher is created (= first arm). The deadlock itself
    #   is GPU driver behaviour and cannot be reproduced on the CPU, so the call order is pinned as source text.
    for name, text in (('runtime.cpp forward_batch_step', step), ('test_decode_step_graph.cu', (ROOT / 'engine/tests/test_decode_step_graph.cu').read_text())):
        a, b = text.find('k::hs_preload('), text.find('std::make_unique<hs::Dispatcher>')
        assert 0 <= a < b, f'{name}: k::hs_preload must run before the dispatcher is created (first-step lazy-loading deadlock)'
    assert 'moe_decode_fused_dev' in (ROOT / 'engine/src/kernels/decode_handshake.cu').read_text() and 'a.fused = fused;' in step, \
        'step graph must follow HIVE_DECODE_FUSED (D1) like moe_decode_experts'
    print(f'handshake CPU: {len(LEGACY_LINES)} transcribed legacy lines verbatim in moe_decode_experts · step share expression == legacy · '
          'hs_preload before dispatcher · D1 fused wired')


def run(exe, args, env_extra=None, timeout=900):
    env = harness.tsan_env()
    env.update(env_extra or {})
    p = subprocess.run([exe, *map(str, args)], env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def main():
    source_guards()
    tsan = harness.sanitizer() == 'thread'
    failures = []
    with tempfile.TemporaryDirectory(prefix='hive-hs-test-') as td:
        exe = harness.build(td, 'handshake', '', [TEST, STORE, *OTHER])
        args = (4000, 60, 3) if tsan else (20000, 300, 3)
        rc, out = run(exe, args)
        print('\n'.join(l for l in out.splitlines() if l.startswith(('handshake CPU', 'FAIL', 'WARNING: ThreadSanitizer', 'SUMMARY'))))
        if rc != 0:
            failures.append(f'handshake run {args}: exit {rc}\n' + out[-4000:])
        if tsan:
            # negative control: done_seq published with a relaxed store — the fake GPU's reads of cpu_out race with the pool's writes
            src = DISPATCH.read_text()
            assert src.count(DONE_STORE) == 1, 'done_seq store line moved: update DONE_STORE'
            (Path(td) / 'decode_dispatch_mut.h').write_text(src.replace(
                DONE_STORE, '    std::atomic_ref<uint32_t>(io_.ctrl->done_seq).store(seq, std::memory_order_relaxed);  // mutant'))
            mtest = Path(td) / 'test_decode_handshake_mut.cpp'  # quote include resolves next to the including file first
            mtest.write_text(TEST.read_text().replace('#include "hive/decode_dispatch.h"', '#include "decode_dispatch_mut.h"'))
            mexe = harness.build(td, 'handshake-mutant', '', [mtest, STORE, *OTHER])
            got = False
            for attempt in range(3):
                rc, out = run(mexe, (200, 30, 3))
                if rc == 66 and 'data race' in out:
                    got = True
                    break
            print(f'TSAN negative control (relaxed done_seq publication): {"DETECTED" if got else "NOT DETECTED"} after {attempt + 1} run(s)')
            if not got:
                failures.append('relaxed done_seq mutant not detected by TSAN\n' + out[-3000:])
    if failures:
        print('\n'.join(failures))
        sys.exit(1)
    print('handshake CPU suite OK (sanitizer %s)' % harness.sanitizer())


if __name__ == '__main__':
    main()
