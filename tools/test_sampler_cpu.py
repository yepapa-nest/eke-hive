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
    ('order-free path ignores min_p', '      e[i] = (min_p > 0.f && x < min_p) ? 0.f : x;', '      e[i] = x;'),
    ('order-free path ignores temperature', '      const float x = expf((logits[i] - mx) / temperature);', '      const float x = expf(logits[i] - mx);'),
    ('order-free path draws over part of the mass', '    std::uniform_real_distribution<double> U(0.0, sum);', '    std::uniform_real_distribution<double> U(0.0, sum * 0.9);'),
    ('two-stage path never leaves the candidates', '    if (u < ec || all_in || V == 0) {', '    if (true) {'),
    ('two-stage rest pass ignores temperature', '      e[i] = cand ? 0.f : expf((full_logits[i] - c.mx) * it);', '      e[i] = cand ? 0.f : expf(full_logits[i] - c.mx);'),
    ('two-stage rest pass counts the candidates again', '      e[i] = cand ? 0.f : expf((full_logits[i] - c.mx) * it);', '      e[i] = expf((full_logits[i] - c.mx) * it);'),
    ('request_seed uses the clock only', '  return splitmix64(base ^ splitmix64(counter.fetch_add(1, std::memory_order_relaxed) + 1) ^ ns);', '  (void)base; return ns / 1000000;'),
]
# hive/spec_sample.h (HIVE_MTP_SAMPLE_DRAFTS) mutants — run with --spec (its own tests only)
SPEC_HDR = ROOT / 'engine/include/hive/spec_sample.h'
SPEC_MUTANTS = [
    ('accept ignores the ratio p/q', '  return std::uniform_real_distribution<double>(0.0, 1.0)(rng) < px / qx;', '  return true;'),
    ('sparse residual ignores q', '    for (size_t i = 0; i < p.kept.size(); ++i) { r[i] = std::max(0.0, p.kept[i].second / p.total - q.prob(p.kept[i].first)); sum += r[i]; }',
     '    for (size_t i = 0; i < p.kept.size(); ++i) { r[i] = p.kept[i].second / p.total; sum += r[i]; }'),
    ('dense residual ignores q', '    r[t] = (float)std::max(0.0, pt - qt);', '    r[t] = (float)pt;'),
    ('nucleus cut late', '  if (top_p > 0.f && top_p < 1.f) { double c = 0; int k2 = 0; while (k2 < keep) { c += p[k2].first; ++k2; if (c >= top_p) break; } keep = std::max(1, k2); }\n  return keep;',
     '  if (top_p > 0.f && top_p < 1.f) { double c = 0; int k2 = 0; while (k2 < keep) { c += p[k2].first; ++k2; if (c >= top_p + 0.04) break; } keep = std::max(1, k2); }\n  return keep;'),
    ('dist_cands skips the full fallback', '  if (top_k <= 0 && mass < need) return dist_full(logits, V, temperature, top_p, top_k, min_p);  // mass outside the candidates is needed — full path',
     '  (void)need;'),
    ('dense draw never leaves the candidates', '    if (u < ec || nvalid >= V) {', '    if (true) {'),
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
        stext = SPEC_HDR.read_text()
        for i, (name, old, new) in enumerate(SPEC_MUTANTS):
            if stext.count(old) != 1:
                fails.append(f'spec mutant anchor [{name}]')
                continue
            inc = Path(td) / f'sinc{i}'
            (inc / 'hive').mkdir(parents=True)
            (inc / 'hive/spec_sample.h').write_text(stext.replace(old, new))
            mexe = build(td, f'sampler_smut{i}', inc)
            q = subprocess.run([mexe, '--spec'], capture_output=True, text=True, timeout=3600)
            print(f'  spec_sample mutant [{name}]: {"DETECTED" if q.returncode != 0 else "NOT DETECTED"}')
            if q.returncode == 0:
                fails.append(f'spec mutant [{name}] not detected')
    if fails:
        print('FAILED:', fails)
        sys.exit(1)
    print('sampler CPU checks passed')


if __name__ == '__main__':
    main()
