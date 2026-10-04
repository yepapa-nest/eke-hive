#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HIVE_CACHE_ELASTIC — CPU checks (no GPU).

1. engine/tests/test_cache_elastic_cpu.cpp on fake CUDA/NUMA with the REAL ExpertStore, synchronous and FAKE_CUDA_ASYNC=1 streams:
   reserve/alloc_cache placement, lent store == plain store, reclaim/lend cycles (invariants + VRAM bytes after the elastic range is
   overwritten), E4 paced promotions. Built with -D_GLIBCXX_ASSERTIONS.
2. The cache-event traces of the cycle runs replay cleanly through tools/check_cache_events.py (reclaim 'E'/'D' events match the state machine).
3. Negative controls: expert_store.cpp whose elastic_reclaim forgets to unmap evicted keys MUST fail; under TSAN, one that does not
   wait for in-flight promotions into elastic slots MUST race (the test overwrites the reclaimed range like the prefill work buffers do).
4. Text contracts: the switch is parsed with env_on in runtime.cpp; the default path (switch off) returns before touching any buffer; the
   R5 alloc_cache sizing lines are untouched.
HIVE_TEST_SANITIZER=thread (under `setarch x86_64 -R`) race-checks every run.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake')]
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import harness  # noqa: E402
import check_cache_events  # noqa: E402

ROOT = harness.ROOT
TEST = ROOT / 'engine/tests/test_cache_elastic_cpu.cpp'
STORE = ROOT / 'engine/src/expert_store.cpp'
RUNTIME = ROOT / 'engine/src/runtime.cpp'
OTHER = [ROOT / 'engine/src/cpu/expert_cpu.cpp', harness.FAKE / 'fake_checkpoint.cpp']
EVICT = '    if (key >= 0 && slot_of_[key] == s) { slot_of_[key] = -1; ++evictions_; ++ev; trace_event(\'E\', key, s); }\n'


def build_unit(td, name, store):
    return harness.build(td, name, '', [TEST, store, *OTHER], extra=['-I' + str(ROOT / 'engine/tests'), '-D_GLIBCXX_ASSERTIONS'])


def run(exe, args, asyn):
    env = harness.tsan_env()
    env['FAKE_CUDA_ASYNC'] = '1' if asyn else '0'
    p = subprocess.run([exe, *map(str, args)], env=env, capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def contracts(failures):
    rt, st = RUNTIME.read_text(), STORE.read_text()
    need = [
        (rt, '    if (!env_on("HIVE_CACHE_ELASTIC")) return;\n'),
        (rt, '      if (end != v && end && *end == 0 && x >= 2) S = (int)std::min<long>(x, 1 << 20);\n'),
        (rt, '  ElasticScope::setup(*this);'),
        (rt, '  ElasticScope elastic_scope(*this, M, false);'),
        (rt, '  ElasticScope elastic_scope(*this, 0, true);'),
        (st, '  pending_.assign(n_slots_, -1);\n  slot_last_use_.assign(n_slots_, 0);\n  trace_event(\'B\', n_slots_, E_);\n'),
        (st, EVICT),
    ]
    for text, needle in need:
        if text.count(needle) != 1:
            failures.append(f'text contract missing or duplicated: {needle[:100]!r}')
    # every ElasticScope entry is a no-op when the switch is off: the scope constructor, lend_if_idle and teardown all test E.on first
    i = rt.index('struct Runtime::ElasticScope {')
    body = rt[i:rt.index('\n};\n', i)]
    for head in ('    if (!E.on) return;\n    if (!force', '    if (!E.on || E.depth > 0', '    if (!E.on) return;\n    view(r, false);'):
        if head not in body:
            failures.append(f'ElasticScope default-off guard missing: {head!r}')
    print('elastic: switch/text contracts checked')


def main():
    failures = []
    contracts(failures)
    tsan = harness.sanitizer() == 'thread'
    rounds = 60 if tsan else 300
    with tempfile.TemporaryDirectory(prefix='hive-elastic-test-') as td:
        exe = build_unit(td, 'elastic', STORE)
        for asyn in (False, True):
            for f in Path(td).glob('elastic-events*.csv'):
                f.unlink()
            rc, out = run(exe, (rounds, td), asyn)
            print(f'[async={int(asyn)}] ' + ' | '.join(l for l in out.splitlines() if l.startswith(('elastic CPU', 'FAIL', 'WARNING: ThreadSanitizer', 'SUMMARY'))))
            if rc != 0:
                failures.append(f'unit (async={int(asyn)}): exit {rc}\n' + out[-4000:])
            for tr in sorted(Path(td).glob('elastic-events*.csv')):
                try:
                    with open(tr) as f:
                        res = check_cache_events.replay(f)
                    print(f'  trace {tr.name}: replay ok · events {sum(res["counts"].values())} · E {res["counts"].get("E", 0)} · D {res["counts"].get("D", 0)}')
                except ValueError as e:
                    failures.append(f'trace {tr.name} (async={int(asyn)}): {e}')
        src = STORE.read_text()
        if src.count(EVICT) == 1:
            mut = Path(td) / 'expert_store_mut.cpp'
            mut.write_text(src.replace(EVICT, EVICT.replace('slot_of_[key] = -1; ', '')))
            mexe = build_unit(td, 'elastic-mutant', mut)
            rc, out = run(mexe, (40, td), False)
            print(f'negative control (reclaim keeps evicted keys mapped): {"DETECTED" if rc != 0 else "NOT DETECTED"}')
            if rc == 0:
                failures.append('mutant elastic_reclaim without unmapping passed\n' + out[-3000:])
        wait = '    CUDA_CHECK(cudaEventSynchronize(b.evt));\n'
        if tsan and src.count(wait) == 1:  # the wait for in-flight promotions into elastic slots: a race only a sanitizer sees (async streams)
            mut = Path(td) / 'expert_store_nowait.cpp'
            mut.write_text(src.replace(wait, '    (void)b.evt;\n'))
            mexe = build_unit(td, 'elastic-nowait', mut)
            rc, out = run(mexe, (200, td), True)
            print(f'negative control (reclaim without waiting for elastic-slot promotions, TSAN): {"DETECTED" if rc != 0 else "NOT DETECTED"}')
            if rc == 0:
                failures.append('mutant elastic_reclaim without the promotion wait passed under TSAN\n' + out[-3000:])
    if failures:
        print('\n'.join('FAIL ' + f for f in failures))
        sys.exit(1)
    print('elastic CPU suite OK (sanitizer %s)' % harness.sanitizer())


if __name__ == '__main__':
    main()
