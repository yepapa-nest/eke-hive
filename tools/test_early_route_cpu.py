#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HIVE_DECODE_EARLY_ROUTE — CPU checks (no GPU).

engine/tests/test_early_route_cpu.cpp on the PRODUCTION header engine/include/hive/early_route.h:
  * er::Gate (post-number wait): normal posts, a counter that skipped ahead (kResync), a launch without a post
    (kMissing), an unclosed previous launch whose stale post arrives first (untrusted) — kOk only ever with this
    launch's routing;
  * er::run_layer (the decode_layer host order) against a fake in-order GPU stream thread + a one-batch CPU pool
    thread: final state == sequential reference, every table/routing/activation read carries the right layer tag.
Negative controls: mutated orderings / gate rules MUST fail (TSAN-only ones are marked).
HIVE_TEST_SANITIZER=thread (under `setarch x86_64 -R`): every run race-checked.
Also production runtime.cpp text contracts: switch parsing (env_on), the deferred CPU unpack is the same
expression as moe_decode_experts' unpack_row (verbatim lines), same CPU-row marks as HOST_FAST, the routing
half is posted right after the router only while decode_layer arms it, and the default path is untouched.

T2 (batch regression, service A/B: c4 93.4 -> 60.9, c8 110.7 -> 68.5, c1 unchanged):
copy_engine_model() replays one decode step (40 layers) through a small event model of the production launch
order — st_ (front graph A, table H2D, expert groups), side_ (demand DMA), one non-preemptive H2D copy engine,
the CPU pool, and the production dma_frac adaptation (constants/rule parsed from runtime.cpp/runtime.h).
Variants: base (legacy decode_layer + B2 CPU_FIRST), early route with the table H2D on st_ (the first version) and
early route with the table on its own stream (T2 fix, what runtime.cpp does now — detected from the text).
It is a MODEL (copy-engine FIFO behaviour and timings are assumptions, not measurements); it shows the
mechanism: the table copy waits behind demand DMA -> resident groups and dma_e0_ move behind the DMA ->
the DMA sample loses the copy time -> dma_frac climbs to its ceiling -> more DMA per layer. M=1 (misses < 3,
no DMA) is unaffected, as in the service A/B. The GPU confirmation is test_early_route [.. cpufirst].
"""
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake')]
import harness  # noqa: E402

ROOT = harness.ROOT
TEST = ROOT / 'engine/tests/test_early_route_cpu.cpp'
HDR = ROOT / 'engine/include/hive/early_route.h'
RT = ROOT / 'engine/src/runtime.cpp'

# (name, old, new, tsan_only)
MUTANTS = [
    ('experts launched before the routing post', '  o.wait_route(l);\n  o.launch(l);\n', '  o.launch(l);\n  o.wait_route(l);\n', False),
    ('CPU unpack before the front graph ended', '  o.wait_front(l);\n  o.start_cpu(l);\n', '  o.start_cpu(l);\n  o.wait_front(l);\n', False),
    ('next-layer tables before the front graph ended', '  o.wait_front(l);\n  o.start_cpu(l);\n  o.prep_next(l);\n',
     '  o.prep_next(l);\n  o.wait_front(l);\n  o.start_cpu(l);\n', False),
    ('finish before the CPU start', '  o.start_cpu(l);\n  o.prep_next(l);\n  o.finish(l);\n', '  o.finish(l);\n  o.start_cpu(l);\n  o.prep_next(l);\n', False),
    ('unclosed launch trusted', '    const bool ok = !open;\n', '    const bool ok = true;\n', False),
    ('post number read without acquire', 'load(std::memory_order_acquire); }', 'load(std::memory_order_relaxed); }', True),
]

UNPACK_LINES = [  # moe_decode_experts' unpack_row body — the deferred unpack in decode_layer must be the same expression
    '      if (unpack2) { cpu::unpack_act_row(w.xq_h + (size_t)m * dim, w.xs_h + (size_t)m * (dim / 32), dim, w.a_f_h + (size_t)m * dim, '
    'w.a_s_h + (size_t)m * (dim / 32)); return; }',
    '      for (int d = 0; d < dim; ++d) w.a_f_h[(size_t)m * dim + d] = lut[w.xq_h[(size_t)m * dim + d]];',
    '      for (int b = 0; b < dim / 32; ++b) w.a_s_h[(size_t)m * (dim / 32) + b] = e8m0_to_f32(w.xs_h[(size_t)m * (dim / 32) + b]);',
]
NEED_LINE = 'for (int ei = 0; ei < n_cpu_e; ++ei) for (int i = cnt[cpu_e[ei]]; i < cnt[cpu_e[ei] + 1]; ++i) need[rows_by_e[i] / k] = 1;'


def body_of(rt, head):
    i = rt.index(head)
    return rt[i:rt.index('\n}\n', i)]


def contracts():
    rt = RT.read_text()
    assert rt.count('dov_->er_on = env_on("HIVE_DECODE_EARLY_ROUTE");') == 1, 'switch parsing must be env_on (unset/""/"0" = off)'
    dl = body_of(rt, 'void Runtime::decode_layer(')
    mx = body_of(rt, 'void Runtime::moe_decode_experts(const LayerWeights& L, int l, int M, ForwardStats* stats, MoePend& p,')
    rs = body_of(rt, 'void Runtime::moe_router_shared(')
    for line in UNPACK_LINES:
        assert line in mx, 'moe_decode_experts unpack_row changed — update UNPACK_LINES and the deferred unpack in decode_layer:\n' + line
        assert line in dl, 'decode_layer deferred unpack drifted from moe_decode_experts unpack_row:\n' + line
    # deferred unpack marks exactly the rows the immediate paths unpack (all rows / HOST_FAST CPU rows)
    d0 = mx.index('if (p.defer_unpack) {')
    d1 = mx.index('    } else\n    if (!fast) { for (int m = 0; m < M; ++m) unpack_row(m); }', d0)
    blk = mx[d0:d1]
    assert 'need.assign((size_t)M, fast ? 0 : 1);' in blk and NEED_LINE in blk and mx.count(NEED_LINE) == 2, 'deferred CPU-row marks drifted from HOST_FAST'
    # early branch: gated by the switch + fused router path + no dump/inject; returns before the legacy path; legacy calls unchanged
    e0 = dl.index('if (D.er_on && fuse_ && M <= 8 && opt_.dump_dir.empty() && opt_.inject_dir.empty()) {')
    e1 = dl.index('    er::run_layer(l, /*prepped=*/true, ops);\n    return;\n  }', e0)
    early = dl[e0:e1]
    assert 'r.moe_decode_experts(L, l, M, s, q, /*start_cpu=*/false, true, true);' in early and 'q.defer_unpack = true;' in early
    legacy = dl[e1:]
    a = legacy.index('moe_decode_experts(L, l, M, stats, p, true, true, true);')
    assert a < legacy.index('moe_decode_finish(p, stats);') and legacy.index('decode_stream_wait(st_, prof_step);') < a
    # router post only while armed, only on the fused router path; the full route_to_host is the unarmed (default) path
    assert 'const bool er_post = dov_ && dov_->er_arm && fuse_ && M <= 8;' in rs
    r0 = rs.index('pmark("moe.router");')
    assert rs.index('if (er_post) k::er_route_post(', r0) < rs.index('attn2_shared_experts', r0)
    assert ('  if (er_post) { if (copy_acts) k::route_to_host(w.ids.as<int32_t>(), w.rw.as<float>(), w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), 0, M * dim,'
            in rs) and '  else\n  k::route_to_host(w.ids.as<int32_t>(), w.rw.as<float>(), w.xq.as<uint8_t>(), w.xs.as<uint8_t>(), M * k, copy_acts ? M * dim : 0,' in rs
    # arming is scoped to the decode_layer front body (reset by RAII)
    assert early.count('D.er_arm = true;') == 1 and 'struct Arm { DecodeOverlap& D; ~Arm() { D.er_arm = false; } } arm{D};' in early
    assert rt.count('er_arm = true') == 1, 'er_arm must only be set by decode_layer'
    print('early-route CPU: runtime.cpp S1 contracts OK')


def copy_engine_model():
    """T2 — see the module docstring. Times in µs. Returns nothing; asserts."""
    import re
    rt, rh = RT.read_text(), (ROOT / 'engine/include/hive/runtime.h').read_text()
    fdef = float(re.search(r'kDmaFracDefault\[4\] = \{([0-9.]+)f', rh).group(1))
    flo = float(re.search(r'kDmaFracLo\[4\] = \{([0-9.]+)f', rt).group(1))
    fhi = float(re.search(r'kDmaFracHi\[4\] = \{([0-9.]+)f', rt).group(1))
    fstep = float(re.search(r'kDmaFracStep = ([0-9.]+)f;', rt).group(1))
    # the model below is the decode kind: its band stays 1.2 (HIVE_DMA_BAND_SHORT narrows only the two short-prefill kinds)
    assert 'const float band = kd == kDmaShort || kd == kDmaStreamShort ? band_short : 1.2f;' in rt
    assert 'if (dma_evt_cpu_ms_ > gpu_ms * band) f = std::min(kDmaFracHi[kd], f + kDmaFracStep);' in rt
    assert 'else if (gpu_ms > dma_evt_cpu_ms_ * band) f = std::max(kDmaFracLo[kd], f - kDmaFracStep);' in rt
    assert 'else if (n_miss >= 3) gpu_share_left = std::min(std::min(dma_cap, store_.staging_slots()), std::max(opt_.decode_gpu_share, (int)(frac * n_miss + 0.5f)));' in rt
    mx = body_of(rt, 'void Runtime::moe_decode_experts(const LayerWeights& L, int l, int M, ForwardStats* stats, MoePend& p,')
    # production launch order facts the model encodes
    t_st = mx.index('CUDA_CHECK(cudaMemcpyAsync(w.tbl_dev.p, w.tbl_h, total, cudaMemcpyHostToDevice, st_));')
    t_dc = mx.index('for (int j = 0; j < n_pc; ++j) issue_copy(pc_e[j], pc_si[j], pc_pos[j]);')
    assert t_st < t_dc, 'CPU_FIRST: the table H2D must be issued before the deferred demand DMA'
    assert mx.index('if (start_cpu) { p.t_cpu0 = now_ms(); store_.start_jobs(jobs, /*owned=*/pregate_k_ > 0); p.started = true; }') < t_st, 'CPU_FIRST: pool starts before the table'
    assert mx.index('if (time_dma) { CUDA_CHECK(cudaEventRecord(dma_e0_, st_));') > t_st, 'dma_e0_ is recorded on st_ after the table H2D'
    fixed = ('if (p.er_tbl && D.er_tbl_st) {\n      CUDA_CHECK(cudaMemcpyAsync(w.tbl_dev.p, w.tbl_h, total, cudaMemcpyHostToDevice, D.er_tbl_st));\n'
             '      CUDA_CHECK(cudaEventRecord(D.er_tbl, D.er_tbl_st));\n      CUDA_CHECK(cudaStreamWaitEvent(st_, D.er_tbl, 0));') in mx
    dl = body_of(rt, 'void Runtime::decode_layer(')
    fixed = fixed and 'q.defer_unpack = true; q.er_tbl = true;' in dl and rt.count('er_tbl = true') == 1
    prod = 'own' if fixed else 'st'

    F, SH, WAKE, HC, UNP, HL, TBL, CP, RES, DG, ACC = 260, 60, 5, 20, 15, 40, 5, 670, 100, 30, 10
    cpu_ms = lambda n: 0 if n == 0 else 50 + 150 * n  # pool span by job count (model)

    def step(variant, n_miss, NL=40, steps=30):
        frac, eng_free, t, st_hist = fdef, 0.0, 0.0, []
        blocked = 0
        for s_ in range(steps):
            t0 = t
            for _ in range(NL):
                a_end = t + F
                post = a_end - SH
                n_dma = min(8, max(1, int(frac * n_miss + 0.5))) if n_miss >= 3 else 0
                n_cpu = n_miss - n_dma
                if variant == 'base':  # sync at A end -> classify -> unpack -> pool -> table (st_ idle) -> resident -> deferred DMA
                    h1 = a_end + WAKE + HC
                    cpu0 = h1 + UNP
                    tbl_sub, tbl_ready = cpu0 + 5, cpu0 + 5
                    cp_sub = tbl_sub + 10
                else:  # early route: woken by the post -> classify -> table + resident + deferred DMA issued; pool after A end
                    h1 = post + WAKE + HC
                    tbl_sub = h1 + 5
                    tbl_ready = max(a_end, tbl_sub) if variant == 'st' else tbl_sub  # 'st': behind graph A in stream order
                    cp_sub = tbl_sub + 10
                    cpu0 = max(a_end, h1 + HL) + UNP
                # one non-preemptive H2D engine: when idle, the earliest-submitted READY copy goes next
                pend = [(tbl_ready, tbl_sub, TBL, 'tbl')] + [(cp_sub, cp_sub + 0.1 * i, CP, 'dma') for i in range(n_dma)]
                done, clock = {}, eng_free
                while pend:
                    ready = [x for x in pend if x[0] <= clock]
                    if not ready:
                        clock = min(x[0] for x in pend)
                        continue
                    x = min(ready, key=lambda y: y[1])
                    pend.remove(x)
                    if x[3] == 'tbl' and any(d.startswith('dma') for d in done):
                        blocked += 1
                    clock += x[2]
                    done[x[3] + str(len(done))] = clock
                eng_free = clock
                tbl_done = next(v for k_, v in done.items() if k_.startswith('tbl'))
                dma_done = max([v for k_, v in done.items() if k_.startswith('dma')], default=0.0)
                e0 = max(a_end, tbl_done)
                res_end = e0 + RES
                g_end = max(res_end, dma_done) + DG * n_dma if n_dma else res_end
                c_end = cpu0 + cpu_ms(n_cpu)
                if n_dma and n_cpu:  # production DMA sample: e0 -> e1 on st_ vs pool span
                    g_ms, c_ms = g_end - e0, cpu_ms(n_cpu)
                    if c_ms > g_ms * 1.2: frac = min(fhi, frac + fstep)
                    elif g_ms > c_ms * 1.2: frac = max(flo, frac - fstep)
                elif n_miss >= 3:
                    frac = max(fdef, frac - fstep) if frac > fdef else min(fdef, frac + fstep)
                t = max(g_end, c_end + ACC if n_cpu else 0)
            st_hist.append(t - t0)
        return sum(st_hist[-10:]) / 10 / 1000, frac, blocked

    out = {}
    for n_miss, label in ((1, 'M=1'), (4, 'M=4')):
        for v in ('base', 'st', 'own'):
            out[(label, v)] = step(v, n_miss)
        b, st_, own = (out[(label, v)] for v in ('base', 'st', 'own'))
        print(f'early-route model {label}: step ms base {b[0]:.2f} (frac {b[1]:.2f}) · early+table on st_ {st_[0]:.2f} (frac {st_[1]:.2f}, table behind DMA '
              f'{st_[2]}x) · early+own stream {own[0]:.2f} (frac {own[1]:.2f}, table behind DMA {own[2]}x) · production = {prod}')
    b1, s1, o1 = (out[('M=1', v)] for v in ('base', 'st', 'own'))
    b4, s4, o4 = (out[('M=4', v)] for v in ('base', 'st', 'own'))
    assert s1[2] == 0 and abs(s1[0] - o1[0]) < 1e-9 and s1[0] <= b1[0], 'M=1 has no DMA: table placement must not matter'
    assert s4[2] > 0 and s4[1] > b4[1] + fstep and s4[0] > 1.2 * b4[0], 'model must reproduce the 09-30 batch collapse for the table-on-st_ variant'
    assert o4[2] == 0 and o4[0] <= b4[0] * 1.02, 'own-stream table must not regress vs base'
    assert prod == 'own', 'runtime.cpp early route still issues the expert table H2D on st_ behind the front graph (T2 regression)'
    print('early-route model: T2 table-stream contract OK')


def run(exe, args):
    p = subprocess.run([exe, *map(str, args)], env=harness.tsan_env(), capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def main():
    contracts()
    copy_engine_model()
    tsan = harness.sanitizer() == 'thread'
    failures = []
    with tempfile.TemporaryDirectory(prefix='hive-early-route-test-') as td:
        exe = harness.build(td, 'early_route', '', [TEST])
        rc, out = run(exe, (1500, 20, 40) if tsan else (4000, 60, 40))
        print(' | '.join(l for l in out.splitlines() if l.startswith(('early-route CPU', 'FAIL', 'WARNING: ThreadSanitizer', 'SUMMARY'))))
        if rc != 0:
            failures.append(f'run: exit {rc}\n' + out[-4000:])
        src = HDR.read_text()
        for n, (name, old, new, tsan_only) in enumerate(MUTANTS):
            assert src.count(old) == 1, f'mutant "{name}": anchor moved in early_route.h — update MUTANTS'
            if tsan_only and not tsan:
                print(f'negative control ({name}): skipped (TSAN only)')
                continue
            d = Path(td) / f'mut{n}'
            (d / 'hive').mkdir(parents=True)
            (d / 'hive' / 'early_route.h').write_text(src.replace(old, new))
            (d / 'test.cpp').write_text(TEST.read_text())  # "hive/early_route.h" is looked up relative to the including file's directory first → the mutated header
            mexe = harness.build(td, f'early-route-mut{n}', '', [d / 'test.cpp'])
            rc, out = run(mexe, (600, 6, 40))
            caught = rc != 0
            print(f'negative control ({name}): {"DETECTED" if caught else "NOT DETECTED"}')
            if not caught:
                failures.append(f'mutant "{name}" passed\n' + out[-2000:])
    if failures:
        print('\n'.join(failures))
        sys.exit(1)
    print('early-route CPU suite OK (sanitizer %s)' % harness.sanitizer())


if __name__ == '__main__':
    main()
