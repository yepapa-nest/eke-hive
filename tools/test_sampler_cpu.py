#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""hive/sampler.h CPU distribution tests + negative controls (header mutants must fail). No GPU, no service.

engine/tests/test_sampler_cpu.cpp: sample/sample_cands == definition (rank-bucket TV against the noise floor), lossless speculative
acceptance rule (joint distribution), top_p truncation mechanism, request seed.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HDR = ROOT / 'engine/include/hive/sampler.h'
MUTANTS = [
    ('sample_cands ignores temperature in candidate probabilities', '(float)(expf((c.val[i] - c.mx) * it) / (double)c.sum)', '(float)(expf((c.val[i] - c.mx)) / (double)c.sum)'),
    ('sample cuts top_p late', 'while (k2 < keep) { c += p[k2].first; ++k2; if (c >= top_p) break; }', 'while (k2 < keep) { c += p[k2].first; ++k2; if (c >= top_p + 0.04) break; }'),
    ('sample_cands skips the full-sort fallback', '  if (top_k <= 0 && mass < need) return sample(full_logits, temperature, top_p, top_k, min_p, rng);  // mass outside the candidates is needed — full path\n', ''),
    ('request_seed uses the clock only', '  return splitmix64(base ^ splitmix64(counter.fetch_add(1, std::memory_order_relaxed) + 1) ^ ns);', '  (void)base; return ns / 1000000;'),
]


def build(td, name, inc):
    exe = str(Path(td) / name)
    subprocess.run(['g++', '-std=c++20', '-O2', '-iquote', str(inc), '-I' + str(ROOT / 'engine/include'), str(ROOT / 'engine/tests/test_sampler_cpu.cpp'), '-o', exe], check=True)
    return exe


def main():
    fails = []
    with tempfile.TemporaryDirectory() as td:
        exe = build(td, 'sampler', ROOT / 'engine/include')
        p = subprocess.run([exe], capture_output=True, text=True, timeout=1800)
        print(p.stdout, end='')
        if p.returncode != 0:
            print(p.stderr)
            fails.append('sampler unit')
        text = HDR.read_text()
        for i, (name, old, new) in enumerate(MUTANTS):
            if text.count(old) != 1:
                fails.append(f'mutant anchor [{name}]')
                continue
            inc = Path(td) / f'inc{i}'
            (inc / 'hive').mkdir(parents=True)
            (inc / 'hive/sampler.h').write_text(text.replace(old, new))
            mexe = build(td, f'sampler_mut{i}', inc)
            q = subprocess.run([mexe, '--quick'], capture_output=True, text=True, timeout=1800)
            print(f'  sampler mutant [{name}]: {"DETECTED" if q.returncode != 0 else "NOT DETECTED"}')
            if q.returncode == 0:
                fails.append(f'mutant [{name}] not detected')
    if fails:
        print('FAILED:', fails)
        sys.exit(1)
    print('sampler CPU checks passed')


if __name__ == '__main__':
    main()
