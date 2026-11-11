#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Qwen3.8-Flash-Next engine — text contracts of host-side rules the GPU sources must keep (CPU, no build: the engine needs CUDA).

Each contract is a defect found 2026-11-11 by reading the code; the numerical checks run on the GPU (engine/tests/test_qwen_qsa.cu,
qwen_check, tools/validate_gpu.sh).

* Layer yield: every member forward() decides per forward (assigned before its layer loop) is saved by yield_enter and restored by
  yield_exit, and forward() re-checks the grouped-prefill route against the expert cache after a boundary that yielded. (A decode
  run inside the yield left pf_dev_ = false for the parked prefill.)
* qsa_select launches the scoring whenever it runs under a context cap (CUDA graph capture). (An MTP draft chain captured across the
  2,048-token boundary held steps without scoring launches.)
* The single-sequence MTP draft graph is not keyed by a sequence pointer and its chain reads no sequence buffer directly.
* The layer-major prefill synchronizes the stream after each block of the PLE layer (the n-gram rows live in one pinned buffer).
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
ENG = (ROOT / 'engine/src/qwen/qwen_engine.cpp').read_text()
HDR = (ROOT / 'engine/include/hive/qwen/qwen_engine.h').read_text()
MTP = (ROOT / 'engine/src/qwen/qwen_mtp.cpp').read_text()
PRE = (ROOT / 'engine/src/qwen/qwen_prefill.cpp').read_text()
QSA = (ROOT / 'engine/src/qwen/qsa.cu').read_text()


def fn(text, start):
    i = text.index(start)
    return text[i:text.index('\n}\n', i) + 3]


def yield_state():
    fwd = fn(ENG, 'void QwenEngine::forward(')
    prologue = fwd[:fwd.index('if (!prefill && graph_forward(')]
    # members assigned at statement level before the layer loop (chained assignments included): the per-forward decisions
    decided = set()
    for stmt in re.split(r'[;{}]', prologue):
        for m in re.finditer(r'(?<![\w.>\[])([a-z][a-z0-9_]*_)\s*=(?!=)', stmt):
            decided.add(m.group(1))
    decided -= {'gdn_ptr_pin_'}   # (element writes of the pinned pointer table — not matched, listed for clarity)
    assert {'dr_now_', 'hyb_now_', 'map_now_', 'pf_dev_', 'hyb_rows_'} <= decided, 'forward() prologue changed: %s' % sorted(decided)
    # + what the forwards run inside a yield set elsewhere: refresh_rec_table, combine_mlp, verify_batch, arm_mask, the graph phases, the MTP steps
    must = decided | {'nonres_now_', 'pre_normed_', 'verify_', 'vdyn_', 'mask_armed_', 'gphase_', 'graph_ctx_', 'mtp_slot_', 'route_rows_'}
    enter, leave = fn(ENG, 'bool QwenEngine::yield_enter('), fn(ENG, 'void QwenEngine::yield_exit(')
    i = HDR.index('  struct YieldSave {')
    save = HDR[i:HDR.index('\n  };\n', i)]
    for m in sorted(must):
        f = m.rstrip('_')
        assert re.search(r'sv\.%s = %s;' % (f, m), enter), 'yield_enter does not save %s' % m
        assert re.search(r'%s = sv\.%s;' % (m, f), leave), 'yield_exit does not restore %s' % m
        assert re.search(r'\b%s\b' % f, save), 'YieldSave has no field for %s' % m
    assert '++yields_;' in leave and 'CUDA_CHECK' not in leave and 'HIVE_CHECK' not in leave, 'yield_exit must count the yield and must not throw'
    assert ('      const uint64_t y0 = yields_;\n      layer_boundary(l);\n' in fwd and
            'if (yields_ != y0 && pf_dev_) pf_dev_ = device_route_ready() && refresh_rec_table();' in fwd), \
        'forward() must re-check pf_dev_ against the expert cache after a layer boundary that yielded'
    return len(must)


def qsa_cap():
    sel = fn(QSA, 'void qsa_select(')
    assert 'const bool scoring = ctx_cap > 0 || (max_pos + 1) / kQsaRatio > kQsaTopBlocks;' in sel, \
        'qsa_select: under a context cap (graph capture) the scoring launches must not depend on max_pos'
    # the dense rows of such a call stay dense: the kernels decide per row
    exp = fn(QSA, '__global__ void qsa_expand_kernel(')
    assert exp.index('if (V <= kQsaTopBlocks) {') < exp.index('const int32_t* tp = top'), 'qsa_expand_kernel: dense rows must not read top'
    assert 'T_rows[i] = V > kQsaTopBlocks ? V : 0;' in fn(QSA, '__global__ void qsa_visible_kernel('), 'qsa_visible_kernel: dense rows take T = 0'
    # the only caller that passes a cap does so in graph phases only
    assert ENG.count('qsa_select(') == 1 and 'stream_, gphase_ ? graph_ctx_ : 0, one_seq);' in ENG, 'attn_layer: the context cap belongs to the graph phases'


def mtp_key():
    d = fn(MTP, 'int QwenEngine::mtp_draft(')
    key = d[d.index('const std::string key ='):]
    key = key[:key.index(';')]
    assert 'h_last' not in key and 'uintptr_t' not in key, 'mtp_draft: the graph key must not hold a sequence pointer'
    chain = d[d.index('auto chain = [&] {'):d.index('// CUDA graph of the draft chain')]
    assert 'const bf16* hin = hin0;' in chain and 's.h_last' not in chain, 'mtp_draft: the chain must read the hidden state from the engine buffer'
    assert 'CUDA_CHECK(cudaMemcpyAsync(hin0, s.h_last.p, (size_t)HC * 2, cudaMemcpyDeviceToDevice, stream_));' in d[:d.index('auto chain = [&] {')]
    assert 'al(mtp_hout_, 3 * HC * 2);' in ENG, 'mtp_hout_ must hold the chain input row besides the two step outputs'


def ple_sync():
    lm = fn(PRE, 'void QwenEngine::prefill_layer_major(')
    loop = lm[lm.index('layer_front(l, brows[b], ids + r0, true, hcur_);'):lm.index('// MoE over all T rows')]
    assert 'if (L.ple) CUDA_CHECK(cudaStreamSynchronize(stream_));' in loop, 'layer-major prefill: sync after each block of the PLE layer'


def main():
    n = yield_state()
    qsa_cap()
    mtp_key()
    ple_sync()
    print('Qwen engine text contracts passed (%d per-forward members saved across a layer yield)' % n)


if __name__ == '__main__':
    sys.exit(main())
