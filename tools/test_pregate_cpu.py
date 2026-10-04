#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HIVE_DECODE_PREGATE — without a GPU.

1. engine/tests/test_pregate_cpu.cpp: switch parsing, owned spans (pg::owned_spans) match the start_jobs item layout and cover each row exactly once, prediction cleanup (plan), totals.
   Negative control: with a wrong owner formula the layout comparison must fail.
2. Source text contracts on production code (runtime.cpp, expert_store.cpp):
   - both places parse the switch with pg::parse_k(getenv("HIVE_DECODE_PREGATE")) — the store (owned mode) and the runtime (prediction) see the same value
   - start_jobs owner index = phase-1 (r0 - r1_0)/P1, phase-2 nblk + (r0 - r2_0)/P2 (same as the owned_spans enumeration); prefetch P1/P2 use the same env vars and defaults as start_jobs
   - with pg_arm the prediction launch comes right after hc_attn_pre; pg_read is waited on before hc_post (which overwrites h); pg_done is waited on at the end of the front (capture join)
   - owned batches only in decode layer calls (moe_decode_*) — start_jobs in prefill (moe_experts_multi) is unchanged
   - when off, pf_ does not exist (worker wake-up conditions and prefetch are unchanged)
3. Bit comparison and TSAN of the pool in owned mode are done by tools/test_pool_cpu.py (HIVE_DECODE_PREGATE run set, mutant with the owner check removed).
"""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RT = (ROOT / 'engine/src/runtime.cpp').read_text()
ES = (ROOT / 'engine/src/expert_store.cpp').read_text()


def once(text, needle, what):
    n = text.count(needle)
    assert n == 1, f'{what}: expected once, found {n}: {needle[:100]!r}'


def main():
    with tempfile.TemporaryDirectory(prefix='hive-pregate-') as td:
        exe = Path(td) / 'pg'
        subprocess.run(['g++', '-std=c++20', '-O1', '-I' + str(ROOT / 'engine/include'), str(ROOT / 'engine/tests/test_pregate_cpu.cpp'), '-o', str(exe)], check=True)
        p = subprocess.run([str(exe)], capture_output=True, text=True)
        print(p.stdout.strip())
        assert p.returncode == 0, p.stdout + p.stderr
        n = subprocess.run([str(exe), 'neg'], capture_output=True, text=True)
        print(n.stdout.strip())
        assert n.returncode == 1, 'negative control (wrong owner rule) was not detected'
    # 2. text contracts
    once(RT, 'pregate_k_ = pg::parse_k(getenv("HIVE_DECODE_PREGATE"));', 'runtime switch')
    once(ES, 'owned_ = pg::parse_k(getenv("HIVE_DECODE_PREGATE")) > 0;', 'store switch')
    once(ES, 'const int idx = it.phase == 1 ? (it.r0 - r1_0) / P1 : nblk + (it.r0 - r2_0) / P2;', 'start_jobs owner index')
    once(ES, 'it.own = pg::owner(jobs[(size_t)it.job].e, idx, T);', 'start_jobs owner rule')
    once(ES, 'pg::owned_spans(es[i], tid, T, I2, D2, P1, P2,', 'prefetch spans')
    for env in ('HIVE_CPU_P1', 'HIVE_CPU_P2'):
        decl = f'static const int P{env[-1]} = getenv("{env}") ? atoi(getenv("{env}")) : 32;'
        assert ES.count(decl) == 2, f'{env}: start_jobs and prefetch must read the same variable with the same default ({ES.count(decl)} copies)'
    once(RT, '  hc_attn_pre(L, M);\n  pmark("hc_attn");\n  if (dov_ && dov_->pg_arm) pregate_dev(L, M);', 'pregate launch right after hc_attn_pre')
    once(RT, '  if (dov_ && dov_->pg_arm) CUDA_CHECK(cudaStreamWaitEvent(st_, dov_->pg_read, 0));  // P1: hc_post below overwrites h — wait until the prediction has read h\n'
             '  k::hc_post(w.attn_out.as<bf16>(), w.post_a.as<float>(), w.comb_a.as<float>(), M, hc, dim, w.h.as<bf16>(), st_);', 'pg_read before hc_post')
    once(RT, '  if (dov_ && dov_->pg_arm) CUDA_CHECK(cudaStreamWaitEvent(st_, dov_->pg_done, 0));', 'pg_done join at front end')
    assert RT.count('store_.start_jobs(') == 4, 'start_jobs call sites changed: re-check which are decode layers'
    # HIVE_DECODE_DEFER: the deferred batch of a decode layer runs while the next layer starts — dynamic claiming (not owned): no
    #   prefetch was planned for it, and it must not wait for a particular worker
    once(RT, 'r.store_.start_jobs(r.djobs_, false);', 'deferred decode batch stays on dynamic claiming')
    assert RT.count('/*owned=*/pregate_k_ > 0') == 2, 'owned batches must be requested by the two decode-layer call sites only'
    once(RT, '    store_.start_jobs(jobs);\n', 'prefill start_jobs stays on dynamic claiming')
    once(ES, '  if (owned_) {\n    pf_ = new PfReq;', 'pf_ only when the switch is on')
    once(ES, '    if (!store->pf_) return false;', 'wake condition unchanged when off')
    print('pregate CPU: runtime.cpp / expert_store.cpp contracts OK')


if __name__ == '__main__':
    main()
