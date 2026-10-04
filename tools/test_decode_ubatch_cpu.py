#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""O1 HIVE_DECODE_HOST_FAST / HIVE_DECODE_UBATCH — CPU checks (no GPU).

engine/tests/test_decode_ubatch_cpu.cpp on fake CUDA (tools/cpu_fake):
  * dov::group_routes_sparse (HOST_FAST grouping) == the legacy dense counting sort, cell by cell;
  * dov::run_ubatch (the PRODUCTION header engine/include/hive/decode_overlap.h) driving a fake GPU stream
    (FAKE_CUDA_ASYNC=1 = real worker thread) + a real one-batch CPU pool thread: final state == per-half sequential,
    host tables read at GPU execution time carry the right (half, layer) tag, the pool never gets a second batch,
    every op runs exactly once per (half, layer).
Runs synchronous and async fake CUDA. Negative controls (async): five mutated orderings of run_ubatch MUST fail.
HIVE_TEST_SANITIZER=thread (under `setarch x86_64 -R`): every run race-checked.
Also: production runtime.cpp text contracts for the O1 switches (env_on parsing, default path call order).
"""
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake')]
import harness  # noqa: E402

ROOT = harness.ROOT
TEST = ROOT / 'engine/tests/test_decode_ubatch_cpu.cpp'
HDR = ROOT / 'engine/include/hive/decode_overlap.h'
RT = ROOT / 'engine/src/runtime.cpp'

# (name, old, new) — each mutant reorders one step of run_ubatch
MUTANTS = [
    ('prep A before front A done',
     '    o.wait_front(0, l); o.launch(0, l, true);\n',
     '    if (more) o.prep(0, l + 1);\n    o.wait_front(0, l); o.launch(0, l, true);\n'),
    ('B CPU started while A busy',
     'o.launch(1, l, o.cpu_idle(0, l));',
     'o.launch(1, l, true);'),
    ('accum A after next front A',
     '    o.wait_cpu(0, l); o.start_cpu(1, l); o.accum(0, l);\n    if (more) o.front(0, l + 1);\n',
     '    o.wait_cpu(0, l); o.start_cpu(1, l);\n    if (more) o.front(0, l + 1);\n    o.accum(0, l);\n'),
    ('launch B before routing arrived',
     'o.wait_front(1, l); o.launch(1, l, o.cpu_idle(0, l));',
     'o.launch(1, l, o.cpu_idle(0, l)); o.wait_front(1, l);'),
    ('B CPU never started',
     'o.wait_cpu(0, l); o.start_cpu(1, l); o.accum(0, l);',
     'o.wait_cpu(0, l); o.accum(0, l);'),
]


def run(exe, args, async_mode):
    env = harness.tsan_env()
    env['FAKE_CUDA_ASYNC'] = async_mode
    p = subprocess.run([exe, *map(str, args)], env=env, capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def contracts():
    rt = RT.read_text()
    # switch parsing: env_on (unset/""/"0" = off) in the constructor option block
    assert rt.count('host_fast_ = env_on("HIVE_DECODE_HOST_FAST");') == 1
    assert rt.count('ubatch_ = env_on("HIVE_DECODE_UBATCH");') == 1
    # default decode layer = launch + finish (same calls/order as the former moe_decode_experts)
    i = rt.index('void Runtime::decode_layer(')
    body = rt[i:rt.index('\n}\n', i)]
    a = body.index('moe_decode_experts(L, l, M, stats, p, true, true, true);')
    b = body.index('moe_decode_finish(p, stats);')
    assert a < b and body.index('if (host_fast_ && l + 1 < model_.n_loaded_layers())') > a
    # forward_batch: ubatch only when M >= 2 and no dump/inject/expert trace; else the layer loop
    j = rt.index('void Runtime::forward_batch(')
    fb = rt[j:rt.index('\n}\n', j)]
    assert 'const bool ub = ubatch_ && M >= 2 && opt_.dump_dir.empty() && opt_.inject_dir.empty() && !xtrace_;' in fb
    assert 'if (!stepped && ub) decode_layers_ubatch(seqs, M, stats);' in fb
    # launch: HOST_FAST branches only under `fast`; observe text single (cache_replay contract)
    k = rt.index('void Runtime::moe_decode_experts(const LayerWeights& L, int l, int M, ForwardStats* stats, MoePend& p,')
    ln = rt[k:rt.index('\n}\n', k)]
    assert ln.count('store_.observe(') == 1 and ln.count('observe_rows();') == 2
    assert 'if (!fast && pump) promo_pump(l, model_.n_loaded_layers());' in ln
    print('ubatch CPU: runtime.cpp O1 contracts OK')


def main():
    contracts()
    tsan = harness.sanitizer() == 'thread'
    failures = []
    with tempfile.TemporaryDirectory(prefix='hive-ubatch-test-') as td:
        exe = harness.build(td, 'ubatch', '', [TEST])
        args = (1500, 20, 40) if tsan else (4000, 60, 60)
        for mode in ('0', '1'):
            rc, out = run(exe, args, mode)
            print(f'[FAKE_CUDA_ASYNC={mode}] ' + ' | '.join(l for l in out.splitlines() if l.startswith(('ubatch CPU', 'FAIL', 'WARNING: ThreadSanitizer', 'SUMMARY'))))
            if rc != 0:
                failures.append(f'run async={mode}: exit {rc}\n' + out[-4000:])
        src = HDR.read_text()
        for n, (name, old, new) in enumerate(MUTANTS):
            assert src.count(old) == 1, f'mutant "{name}": anchor moved in decode_overlap.h — update MUTANTS'
            d = Path(td) / f'mut{n}'
            (d / 'hive').mkdir(parents=True)
            (d / 'hive' / 'decode_overlap.h').write_text(src.replace(old, new))
            (d / 'test.cpp').write_text(TEST.read_text())  # "hive/decode_overlap.h" is searched first in the including file's directory → the mutated header
            mexe = harness.build(td, f'ubatch-mut{n}', '', [d / 'test.cpp'])
            rc, out = run(mexe, (10, 8 if tsan else 20, 80), '1')
            caught = rc != 0
            print(f'negative control ({name}): {"DETECTED" if caught else "NOT DETECTED"}')
            if not caught:
                failures.append(f'mutant "{name}" passed\n' + out[-2000:])
    if failures:
        print('\n'.join(failures))
        sys.exit(1)
    print('ubatch CPU suite OK (sanitizer %s)' % harness.sanitizer())


if __name__ == '__main__':
    main()
