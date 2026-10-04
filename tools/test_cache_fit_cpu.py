#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HIVE_CACHE_FIT / HIVE_ENGRAM_ALIAS — CPU checks (no GPU).

1. engine/tests/test_cache_fit_cpu.cpp on fake CUDA/NUMA with the REAL ExpertStore: fit_slots arithmetic, deferred slot allocation
   (alloc_cache) == eager allocation in lockstep (slot table, stats, resident order, VRAM bytes), one 'B' trace line with the final
   count, alloc_cache(0). Built with -D_GLIBCXX_ASSERTIONS so an unsized per-slot table is an out-of-range abort.
   Negative control: expert_store.cpp whose alloc_cache forgets to size pending_ MUST fail.
2. The REAL hived.cpp (tools/test_daemon_cpu.py harness, fake runtime) with HIVE_CACHE_FIT=1: the cache is sized after the session
   pool from the (fake) free VRAM minus HIVE_CACHE_RESERVE_MB, stats report exactly that slot count, and the session scenario's
   tokens equal the oracle (and the default run). Default run (switch unset) keeps --vram-cache-mb.
3. Text contracts: switch parsing expressions (env_on), the warm start moves behind the pool only when the switch is on, and the
   runtime.cpp engram alias keeps the legacy allocation lines verbatim in the off branch.
HIVE_TEST_SANITIZER=thread (under `setarch x86_64 -R`) race-checks every run.
"""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake')]
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import harness  # noqa: E402

ROOT = harness.ROOT
TEST = ROOT / 'engine/tests/test_cache_fit_cpu.cpp'
STORE = ROOT / 'engine/src/expert_store.cpp'
HIVED = ROOT / 'engine/src/hived.cpp'
RUNTIME = ROOT / 'engine/src/runtime.cpp'
RESERVE = ROOT / 'engine/include/hive/cache_reserve.h'  # HIVE_CACHE_RESERVE_MB parsing, shared by hived and the GLM KV growth
OTHER = [ROOT / 'engine/src/cpu/expert_cpu.cpp', harness.FAKE / 'fake_checkpoint.cpp']
SIZE_LINE = '  pending_.assign(n_slots_, -1);\n  slot_last_use_.assign(n_slots_, 0);\n  trace_event(\'B\', n_slots_, E_);\n'
REC = 3 * (512 * 512 // 2 + 512 * 512 // 32)  # fake runtime cfg (dim 512 · inter 512) record = 417,792 B (4096-aligned)
FAKE_FREE_MB = 64 * 1024                        # tools/cpu_fake/include/fake_cuda.h cudaMemGetInfo


def build_unit(td, name, store):
    return harness.build(td, name, '', [TEST, store, *OTHER], extra=['-I' + str(ROOT / 'engine/tests'), '-D_GLIBCXX_ASSERTIONS'])


def run(exe, args):
    p = subprocess.run([exe, *map(str, args)], env=harness.tsan_env(), capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def contracts(failures):
    st, hv, rt, rs = STORE.read_text(), HIVED.read_text(), RUNTIME.read_text(), RESERVE.read_text()
    need = [
        (hv, 'static const bool cache_fit = env_on("HIVE_CACHE_FIT");'),
        (rs, 'const double mb = v && *v && !(end != v && end && *end == 0) ? 2400.0 : x;'),
        (hv, 'const size_t cache_reserve = cache_reserve_bytes();'),
        (hv, 'ExpertStore store(c, nl, (size_t)(cache_mb * 1048576.0), cpu_threads, n_mtp, cache_fit || start_asleep);'),  # start asleep (default off = the original arguments)
        (hv, '  if (!cache_fit) warm_start();\n  double last_state_save = now_ms();'),
        (st, "  n_slots_ = defer_cache_ ? 0 : (int)(vram_cache_bytes / lay_.total);\n  if (n_slots_ > 0) slots_.alloc((size_t)n_slots_ * lay_.total);\n"),
        (rt, 'w->eg_alias = env_on("HIVE_ENGRAM_ALIAS") && b_kv <= w->q.n && b_vals + b_q + b_sc <= w->o.n;'),
        # off branch = the legacy allocation lines, verbatim
        (rt, '    w->eg_v.alloc((size_t)M * cols * hd); w->eg_s.alloc((size_t)M * cols * hd / 32);\n'
             '    w->eg_vals.alloc((size_t)M * cols * hd * 2);\n'
             '    w->eg_q.alloc((size_t)M * cols * hd); w->eg_sc.alloc((size_t)M * cols * hd / 32);\n'
             '    w->eg_kv.alloc((size_t)M * dim * (hc + 1) * 2);\n    }\n'),
    ]
    for text, needle in need:
        if text.count(needle) != 1:
            failures.append(f'text contract missing or duplicated: {needle[:100]!r}')
    # the warm-start body is only reachable through warm_start() — exactly two call sites (off: old place · on: after the pool) + one in
    # the start-asleep block (HIVE_START_ASLEEP · default off — reads the list only; the call sits under `if (start_asleep) {`)
    if len(re.findall(r'\bwarm_start\(\);', hv)) != 3 or hv.count('    if (cache_fit) warm_start();  // reads the list only') != 1:
        failures.append('warm_start() must have exactly two call sites (+ the start-asleep one)')
    # aliasing premise: the four engram work buffers are touched only inside engram_dev (every other w.eg_* use would need a lifetime review)
    i = rt.index('void Runtime::engram_dev(')
    body = rt[i:rt.index('\n}\n', i)]
    for b in ('eg_vals', 'eg_q', 'eg_sc', 'eg_kv'):
        pat = re.compile(r'\b(w|w_|W)(\.|->)' + b + r'\b')
        outside = [l for l in rt.replace(body, '').splitlines() if pat.search(l.split('//')[0]) and '.alloc(' not in l and '.p = ' not in l and 'eg_alias' not in l]
        if outside or not pat.search(body):
            failures.append(f'{b}: used outside engram_dev ({outside[:2]}) — re-check the HIVE_ENGRAM_ALIAS lifetime')
    for b in ('eg_v', 'eg_s'):  # dead device buffers (kernels read the mapped eg_v_d/eg_s_d)
        if [l for l in rt.splitlines() if re.search(r'\bw(\.|->)' + b + r'\b(?!_)', l.split('//')[0]) and '.alloc(' not in l]:
            failures.append(f'{b} is now used on the device: HIVE_ENGRAM_ALIAS must allocate it again')
    print('cache-fit: switch/text contracts checked')


def daemon(td, failures):
    import test_daemon_cpu as tdc
    exe = tdc.build_daemon(td, 'hived-fit')
    # fit_slots subtracts a 64 MiB allocation margin: pick the reserve so that exactly 7 fake records fit
    reserve_mb = FAKE_FREE_MB - 64 - (7 * REC + (1 << 20) - 1) // (1 << 20)
    want = (FAKE_FREE_MB * (1 << 20) - reserve_mb * (1 << 20) - (64 << 20)) // REC
    results = {}
    for name, env in (('default', {}), ('fit', {'HIVE_CACHE_FIT': '1', 'HIVE_CACHE_RESERVE_MB': str(reserve_mb)}),
                      ('fit-zero', {'HIVE_CACHE_FIT': '1', 'HIVE_CACHE_RESERVE_MB': str(FAKE_FREE_MB)}),
                      ('fit0', {'HIVE_CACHE_FIT': '0', 'HIVE_CACHE_RESERVE_MB': '5'})):
        h = tdc.Hived(exe, td, env, args=('--vram-cache-mb', '2'), name='fit-' + name)
        ck = tdc.Checker('cache-fit ' + name)
        try:
            slots = h.stats().get('slots')
            tdc.sc_session(h, ck, {})
            log = h.tail(400)
        finally:
            h.stop()
        results[name] = [t[1] for t in ck.transcript]
        failures += [f'[{name}] {f}' for f in ck.failures]
        if name in ('default', 'fit0'):
            exp = (2 << 20) // REC
            if slots != exp or 'cache fit' in log:
                failures.append(f'[{name}] slots {slots} want {exp} (--vram-cache-mb 2) and no fit line')
        elif name == 'fit':
            if slots != want or f'[hived] cache fit: {want} slots' not in log:
                failures.append(f'[fit] slots {slots} want {want}\n' + log[-2000:])
            if log.find('session pool') > log.find('[hived] cache fit'):
                failures.append('[fit] cache sized before the session pool')
        else:
            if slots != 0:
                failures.append(f'[fit-zero] slots {slots} want 0 (reserve = all free)')
        print(f'cache-fit daemon [{name}]: slots {slots} · {ck.checks} checks · {len(ck.failures)} failures')
    for name in ('fit', 'fit-zero', 'fit0'):
        if results[name] != results['default']:
            failures.append(f'[{name}] token transcript differs from the default run')


def main():
    failures = []
    contracts(failures)
    tsan = harness.sanitizer() == 'thread'
    with tempfile.TemporaryDirectory(prefix='hive-cachefit-test-') as td:
        exe = build_unit(td, 'cachefit', STORE)
        rc, out = run(exe, (60 if tsan else 400, td))
        print(' | '.join(l for l in out.splitlines() if l.startswith(('cache-fit CPU', 'FAIL', 'WARNING: ThreadSanitizer', 'SUMMARY'))))
        if rc != 0:
            failures.append(f'unit: exit {rc}\n' + out[-4000:])
        src = STORE.read_text()
        if src.count(SIZE_LINE) != 1:
            failures.append('alloc_cache sizing lines moved: update SIZE_LINE')
        else:
            mut = Path(td) / 'expert_store_mut.cpp'
            mut.write_text(src.replace(SIZE_LINE, SIZE_LINE.replace("  pending_.assign(n_slots_, -1);\n", '')))
            mexe = build_unit(td, 'cachefit-mutant', mut)
            rc, out = run(mexe, (20, td))
            caught = rc != 0
            print(f'negative control (alloc_cache without sizing pending_): {"DETECTED" if caught else "NOT DETECTED"}')
            if not caught:
                failures.append('mutant alloc_cache without pending_ sizing passed\n' + out[-3000:])
        daemon(td, failures)
    if failures:
        print('\n'.join('FAIL ' + f for f in failures))
        sys.exit(1)
    print('cache-fit CPU suite OK (sanitizer %s)' % harness.sanitizer())


if __name__ == '__main__':
    main()
