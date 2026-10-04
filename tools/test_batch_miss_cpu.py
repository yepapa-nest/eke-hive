#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Batch-decode miss handling — CPU checks (no GPU).

* engine/tests/test_batch_miss_cpu.cpp on the PRODUCTION header engine/include/hive/batch_miss.h:
  staging-ring tracker safety/completeness against a simulated ring with untracked writers, assign() == the legacy
  classification loop whenever nothing is in the ring, share/priority rules when it is, cross-step prediction, a
  synthetic temporally-local batch world (reuse happens, never stale).
* Negative controls: two mutated headers (tracker without the per-layer margin · pending ring copies not preferred)
  MUST fail the test.
* Production text contracts: runtime.cpp parses HIVE_DECODE_CPU_FIRST / HIVE_DECODE_STAGE_HIT with env_on and
  HIVE_DECODE_PREFETCH with bmiss::parse_prefetch, builds bmiss_ only when one is on, gates B on the batch-decode
  layer, keeps the legacy classification loop verbatim for B == nullptr; expert_store.cpp parses HIVE_CPU_SPLIT13 /
  HIVE_CPU_P2_PREFETCH with store_env_on.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / 'engine/tests/test_batch_miss_cpu.cpp'
HDR = ROOT / 'engine/include/hive/batch_miss.h'
RT = ROOT / 'engine/src/runtime.cpp'
STORE = ROOT / 'engine/src/expert_store.cpp'

MUTANTS = [
    ('tracker ignores the per-layer margin',
     '(uint64_t)age + (uint64_t)std::max(0, margin) > (uint64_t)S', '(uint64_t)age > (uint64_t)S'),
    ('pending ring copies do not take the DMA share first',
     'if (m[i].stage == 0) { where[i] = kDmaReuse; --left; }', 'if (false) {}'),
]


def build(td, name, inc_first=None):
    exe = Path(td) / name
    cmd = ['g++', '-std=c++20', '-O1', '-Wall', '-Wextra']
    if inc_first:
        cmd += ['-I' + str(inc_first)]
    else:
        cmd += ['-Werror']
    cmd += ['-I' + str(ROOT / 'engine/include'), str(TEST), '-o', str(exe)]
    subprocess.run(cmd, check=True, capture_output=bool(inc_first))  # discard mutant build warnings (unused arguments etc.)
    return exe


def contracts():
    rt = RT.read_text()
    for needle in ('const bool cpu_first = env_on("HIVE_DECODE_CPU_FIRST"), stage_hit = env_on("HIVE_DECODE_STAGE_HIT");',
                   'const int prefetch = env_on("HIVE_DECODE_PREFETCH") ? bmiss::parse_prefetch(getenv("HIVE_DECODE_PREFETCH")) : 0;',
                   'if (cpu_first || stage_hit || prefetch > 0) {',
                   'BatchMiss* B = bmiss_ && batch_ && M >= 2 && kind == kDmaDecode && !ub_active_ && !dsplit_ && opt_.cpu_for_misses ? bmiss_.get() : nullptr;',
                   'const bool sh = B && B->stage_hit;',
                   'const bool defer_copy = B && B->cpu_first;',
                   '  if (!sh) {\n  for (int i = 0; i < n_miss_e; ++i) {\n    if (gpu_share_left > 0) { --gpu_share_left; str_e[n_str_e++] = miss_e[i]; }\n'
                   '    else cpu_e[n_cpu_e++] = miss_e[i];\n  }\n  n_str_dma = n_str_e;\n',
                   'const int pre = sh ? str_si[ei] : -1;',
                   'if (!fast && pump && !defer_copy) promo_pump(l, model_.n_loaded_layers());',
                   '  bm_report(M);'):
        assert rt.count(needle) == 1, f'runtime.cpp contract missing or duplicated: {needle[:90]!r}'
    assert rt.count('bmiss_ = std::make_shared<BatchMiss>();') == 1
    st = STORE.read_text()
    for needle in ('split13_ = store_env_on("HIVE_CPU_SPLIT13");', 'p2_prefetch_ = store_env_on("HIVE_CPU_P2_PREFETCH");',
                   # the single-item case (parts 1) is this one line — parts_of is 1 when off (SPLIT13 and FINE both off); FINE applies only to jobs with R >= 2
                   'if (parts == 1) { p->items.push_back({(int)ji, 1, n, b0, b1}); continue; }',
                   'auto parts_of = [&](const Job& jb) { return fine_ && jb.R >= 2 ? 4 : split13_ ? 2 : 1; };',
                   'fine_ = store_env_on("HIVE_CPU_FINE");', 'copy2d_ = store_env_on("HIVE_STAGE_COPY2D");',
                   'const int p2 = fine_ && jobs[ji].R >= 2 ? std::max(1, P2 / 2) : P2;'):
        assert st.count(needle) == 1, f'expert_store.cpp contract missing or duplicated: {needle[:90]!r}'
    print('batch_miss: runtime.cpp / expert_store.cpp switch and default-path text contracts ok')


def main():
    with tempfile.TemporaryDirectory(prefix='hive-batch-miss-') as td:
        exe = build(td, 'batch_miss')
        p = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
        print(p.stdout + p.stderr, end='')
        if p.returncode != 0:
            sys.exit(1)
        src = HDR.read_text()
        for i, (name, old, new) in enumerate(MUTANTS):
            assert src.count(old) == 1, f'mutant anchor moved: {name}'
            inc = Path(td) / f'mutinc{i}' / 'hive'
            inc.mkdir(parents=True)
            (inc / 'batch_miss.h').write_text(src.replace(old, new))
            mexe = build(td, f'mutexe{i}', inc.parent)
            m = subprocess.run([str(mexe)], capture_output=True, text=True, timeout=600)
            ok = m.returncode != 0
            print(f'batch_miss negative control ({name}): {"DETECTED" if ok else "NOT DETECTED"}')
            if not ok:
                sys.exit(1)
    contracts()
    print('test_batch_miss_cpu OK')


if __name__ == '__main__':
    main()
