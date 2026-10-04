#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""A/B option plumbing (no GPU): every HIVE_* option of docs/configuration.md is parsed by
the production code so that unset / "" / "0" mean off (= the default path) and other values are
used as given, and the launchers forward HIVE_* by value (not by presence).

Three layers, from strongest to weakest:
  1. behaviour of the real ExpertStore / Runtime-ctor text / save_image (engine/tests/test_options_cpu.cpp)
  2. exact source expressions evaluated with the same env (hived main statics, runtime prefetch/tile lines)
  3. the launch scripts (scripts/hive-start.sh, scripts/hive-run.sh) run against stub docker; a static audit for
     presence-only getenv switches
KNOWN deviations are asserted as XFAIL with the reason (see KNOWN below) and reported, not hidden.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake')]
import harness  # noqa: E402

ROOT = harness.ROOT
UNSET = None
VALUES = [UNSET, '', '0', '1', '2']
SWITCH = {UNSET: False, '': False, '0': False, '1': True, '2': True}
SWITCH_NUM = {UNSET: 0, '': 0, '0': 0, '1': 1, '2': 1, 'true': 1}

# KNOWN contract deviations (XFAIL): (option, value) -> reason
KNOWN = {}  # HIVE_CACHE_EVENTS=0 is fixed in expert_store.cpp (store_env_on) — no known deviations


def behaviour(exe, td):
    """Layer 1: one process per (option, value); other doc options unset."""
    cases = {
        'HIVE_CACHE_REUSE_STAGE': SWITCH, 'HIVE_MTP_CACHE': SWITCH, 'HIVE_PHASE_SCORE': SWITCH, 'HIVE_PROMOTE_SCORE': SWITCH,
        'HIVE_CKPT_ASYNC': SWITCH, 'HIVE_CKPT_DELTA': SWITCH,
        'HIVE_PROMOTE_BYTES': {UNSET: 0, '': 0, '0': 0, '1': 1, '2': 2, '123456': 123456, '-5': 0},
        'HIVE_CACHE_EVENTS': {UNSET: False, '': False, '0': False, 'PATH': True},
    }
    base_env = {k: v for k, v in harness.tsan_supp_env(harness.tsan_env(), tmp_dir=td).items() if not k.startswith('HIVE_') or k == 'HIVE_TEST_SANITIZER'}
    rows, bad, xfail = [], [], []

    def run(env_over, cwd):
        env = dict(base_env); env.update(env_over)
        p = subprocess.run([exe], env=env, cwd=cwd, capture_output=True, text=True, timeout=120)
        if p.returncode:
            raise RuntimeError(p.stdout + p.stderr)
        return json.loads(p.stdout.strip().splitlines()[-1])

    default = run({}, td)
    for opt, table in cases.items():
        for val, want in table.items():
            cwd = tempfile.mkdtemp(dir=td)
            env = {} if val is UNSET else {opt: (str(Path(cwd) / 'events.csv') if val == 'PATH' else val)}
            got = run(env, cwd)[opt]
            ok = got == want
            if opt == 'HIVE_CACHE_EVENTS' and val == '0':
                got = got or (Path(cwd) / '0').exists()
                ok = got == want
            rows.append((opt, val, want, got))
            if not ok:
                (xfail if (opt, val) in KNOWN else bad).append((opt, val, want, got))
            # an "off" value must give exactly the default observation for every option
            if want in (False, 0) and val is not UNSET:
                full = run(env, cwd)
                if (opt, val) not in KNOWN and full != default:
                    bad.append((opt, val, 'default path', {k: v for k, v in full.items() if default[k] != v}))
    # pinned pool: exists only with async on AND a positive size
    for a in (UNSET, '', '0', '1'):
        for mb in (UNSET, '', '0', '8'):
            env = {k: v for k, v in (('HIVE_CKPT_ASYNC', a), ('HIVE_CKPT_PINNED_POOL_MB', mb)) if v is not None}
            got = run(env, td)['HIVE_CKPT_PINNED_POOL_MB']
            want = a == '1' and mb == '8'
            rows.append(('HIVE_CKPT_PINNED_POOL_MB', f'async={a!r} mb={mb!r}', want, got))
            if got != want:
                bad.append(('HIVE_CKPT_PINNED_POOL_MB', (a, mb), want, got))
    return rows, bad, xfail


def expression(path, pattern):
    text = (ROOT / path).read_text()
    m = re.search(pattern, text)
    assert m, f'anchor drifted: {path}: {pattern}'
    line = text[:m.start()].count('\n') + 1
    return m.group(1), f'{path}:{line}'


def source_exprs(td):
    """Layer 2: exact source expressions compiled into a probe, evaluated per environment."""
    exprs = {
        'HIVE_IMAGE_CKPT': expression('engine/src/hived.cpp', r'static const bool image_ckpt = (getenv\("HIVE_IMAGE_CKPT"\)[^;]*);'),
        'HIVE_CKPT_HISTORY': expression('engine/src/hived.cpp', r'static const int history_limit = (getenv\("HIVE_CKPT_HISTORY"\)[^;]*);'),
        'HIVE_PREFILL_BUDGET_MS': expression('engine/src/hived.cpp', r'const double prefill_budget_ms = (getenv\("HIVE_PREFILL_BUDGET_MS"\)[^;]*);'),
        'HIVE_TILE_GROUP_GEMM': expression('engine/src/runtime.cpp', r'static const bool tile_group_gemm = (env_on\("HIVE_TILE_GROUP_GEMM"\));'),
        'HIVE_TAIL_SPLIT': expression('engine/src/hived.cpp', r'static const bool tail_split = (getenv\("HIVE_TAIL_SPLIT"\)[^;]*);'),
        # round 2 (hived's env_on is the same formula as runtime's env_on — compared below)
        'HIVE_PREFILL_PAUSE_PROMOTE': expression('engine/src/runtime.cpp', r'prefill_pause_promote_ = (env_on\("HIVE_PREFILL_PAUSE_PROMOTE"\));'),
        'HIVE_EARLY_STREAM': expression('engine/src/runtime.cpp', r'early_stream_ = (env_on\("HIVE_EARLY_STREAM"\));'),
        'HIVE_ENGRAM_PAR': expression('engine/src/runtime.cpp', r'engram_par_ = (env_on\("HIVE_ENGRAM_PAR"\));'),
        'HIVE_IDX_MSUB_ACTUAL': expression('engine/src/runtime.cpp', r'idx_msub_actual_ = (env_on\("HIVE_IDX_MSUB_ACTUAL"\));'),
        'HIVE_PREFIX_SHARE': expression('engine/src/hived.cpp', r'static const bool prefix_share = (env_on\("HIVE_PREFIX_SHARE"\));'),
        'HIVE_DEFER_CKPT': expression('engine/src/hived.cpp', r'static const bool defer_ckpt = (env_on\("HIVE_DEFER_CKPT"\));'),
        'HIVE_GRACEFUL_STOP_S': expression('engine/src/hived.cpp', r'const double graceful_stop_s = (getenv\("HIVE_GRACEFUL_STOP_S"\)[^;]*);'),
    }
    exprs['HIVE_GROUPED_PREFILL'] = expression('engine/src/runtime.cpp', r'static const bool grouped_prefill = (env_on\("HIVE_GROUPED_PREFILL"\));')
    exprs['HIVE_DECODE_FUSED'] = expression('engine/src/runtime.cpp', r'static const bool decode_fused = (env_on\("HIVE_DECODE_FUSED"\));')
    exprs['HIVE_DECODE_BLOCKING_SYNC'] = expression('engine/src/runtime.cpp', r'static const bool blocking = (env_on\("HIVE_DECODE_BLOCKING_SYNC"\));')
    exprs['HIVE_DECODE_SYNC_SPIN_US'] = expression('engine/src/runtime.cpp', r'static const int spin_us = (getenv\("HIVE_DECODE_SYNC_SPIN_US"\)[^;]*);')
    exprs['HIVE_DECODE_ATTN_FUSED'] = expression('engine/src/runtime.cpp', r'static const bool dec_attn_fused = (env_on\("HIVE_DECODE_ATTN_FUSED"\));')
    for name in ('HIVE_DECODE_GEMV2', 'HIVE_DECODE_PDL'):
        e, loc = expression('engine/src/kernels/gemv_decode.cu', r'static const bool on = (gd_env_on\("' + name + r'"\));')
        exprs[name] = (e.replace('gd_env_on', 'env_on'), loc)
    e, loc = expression('engine/src/expert_store.cpp', r'policy_ = (store_env_on\("HIVE_CACHE_POLICY"\)) \?')
    exprs['HIVE_CACHE_POLICY'] = (e.replace('store_env_on', 'env_on'), loc)
    exprs['HIVE_DECODE_STEP_GRAPH'] = expression('engine/src/runtime.cpp', r'step_graph_ = (env_on\("HIVE_DECODE_STEP_GRAPH"\));')
    e, loc = expression('engine/src/runtime.cpp', r'static const bool split_balance = (env_on\("HIVE_PREFILL_SPLIT"\)) &&')
    exprs['HIVE_PREFILL_SPLIT'] = (e, loc)  # only the "on" precondition (the 'balance' value is decided by prefill_split.h parse_mode — checked by test_prefill_split_cpu)
    e, loc = expression('engine/src/runtime.cpp', r'static const long tmo = \[\] \{ (const char\* v = getenv\("HIVE_DECODE_STEP_TIMEOUT_MS"\);[^}]*)\}\(\);')
    exprs['HIVE_DECODE_STEP_TIMEOUT_MS'] = ('[] { const long kStepTimeoutMsDefault = 10000; ' + e + ' }()', loc)
    exprs['HIVE_DECODE_STEP_POLL_US'] = ('[] { int poll = 0; if (const char* v = getenv("HIVE_DECODE_STEP_POLL_US"); v && *v) poll = std::max(0, atoi(v)); return poll; }()',
                                         expression('engine/src/runtime.cpp', r'(getenv\("HIVE_DECODE_STEP_POLL_US"\))')[1])
    exprs['HIVE_DECODE_STEP_SPIN_US'] = ('[] { const char* v = getenv("HIVE_DECODE_STEP_SPIN_US"); return v && *v ? std::max(0, atoi(v)) : 0; }()',
                                         expression('engine/src/runtime.cpp', r'(getenv\("HIVE_DECODE_STEP_SPIN_US"\))')[1])
    e, loc = expression('engine/src/kernels/moe_decode.cu', r'static const bool fused2 = (md_env_on\("HIVE_DECODE_FUSED2"\));')
    exprs['HIVE_DECODE_FUSED2'] = (e.replace('md_env_on', 'env_on'), loc)
    exprs['HIVE_DECODE_FUSED2_STAGES'] = ('[] { const char* t = getenv("HIVE_DECODE_FUSED2_STAGES"); const int n = t && *t ? atoi(t) : 0; return n <= 0 ? 4 : n <= 3 ? 3 : n <= 5 ? 4 : 6; }()',
                                          expression('engine/src/kernels/moe_decode.cu', r'(getenv\("HIVE_DECODE_FUSED2_STAGES"\))')[1])
    for name in ('HIVE_HC_DECODE_FUSED', 'HIVE_HC_SINKHORN_PAR'):
        e, loc = expression('engine/src/kernels/hc_decode_fused.cu', r'static const bool on = (hcdf_env_on\("' + name + r'"\));')
        exprs[name] = (e.replace('hcdf_env_on', 'env_on'), loc)
    exprs['HIVE_DECODE_COPY_PRIO'] = expression('engine/src/runtime.cpp', r'copy_prio_ = (env_on\("HIVE_DECODE_COPY_PRIO"\));')
    e, loc = expression('engine/src/runtime.cpp', r'static const bool decode_split = (env_on\("HIVE_DECODE_SPLIT"\)) &&')
    exprs['HIVE_DECODE_SPLIT'] = (e, loc)  # only the "on" precondition (the 'balance' value is decided by decode_split.h parse_mode — checked by test_decode_split_cpu)
    exprs['HIVE_CPU_GEMV2'] = ('[] { const char* v = getenv("HIVE_CPU_GEMV2"); return v && *v && strcmp(v, "0") != 0; }()',
                               expression('engine/src/cpu/expert_cpu.cpp', r'(getenv\("HIVE_CPU_GEMV2"\))')[1])
    exprs['HIVE_DECODE_HOST_FAST'] = expression('engine/src/runtime.cpp', r'host_fast_ = (env_on\("HIVE_DECODE_HOST_FAST"\));')
    exprs['HIVE_DECODE_UBATCH'] = expression('engine/src/runtime.cpp', r'ubatch_ = (env_on\("HIVE_DECODE_UBATCH"\));')
    e, loc = expression('engine/src/kernels/decode_attn2.cu', r'static const bool on = (a2_env_on\("HIVE_DECODE_ATTN2"\));')
    exprs['HIVE_DECODE_ATTN2'] = (e.replace('a2_env_on', 'env_on'), loc)
    for name in ('HIVE_DECODE_ROUTER3', 'HIVE_DECODE_SPARSE3', 'HIVE_DECODE_QKV3', 'HIVE_DECODE_HCMIX3'):
        e, loc = expression('engine/src/kernels/decode_attn3.cu', r'static const bool on = (a3_env_on\("' + name + r'"\));')
        exprs[name] = (e.replace('a3_env_on', 'env_on'), loc)
    exprs['HIVE_DECODE_SPARSE3_G'] = ('[] { const char* v = getenv("HIVE_DECODE_SPARSE3_G"); const int x = v && *v ? atoi(v) : 0; return (x == 1 || x == 2 || x == 4 || x == 8) ? x : 0; }()',
                                      expression('engine/src/kernels/decode_attn3.cu', r'(getenv\("HIVE_DECODE_SPARSE3_G"\))')[1])
    exprs['HIVE_DECODE_CPU_FIRST'] = ('env_on("HIVE_DECODE_CPU_FIRST")', expression('engine/src/runtime.cpp', r'(env_on\("HIVE_DECODE_CPU_FIRST"\))')[1])
    exprs['HIVE_DECODE_STAGE_HIT'] = ('env_on("HIVE_DECODE_STAGE_HIT")', expression('engine/src/runtime.cpp', r'(env_on\("HIVE_DECODE_STAGE_HIT"\))')[1])
    exprs['HIVE_DECODE_PREFETCH'] = ('[] { const char* v = getenv("HIVE_DECODE_PREFETCH"); if (!(v && *v && strcmp(v, "0") != 0)) return 0; char* end = nullptr; const long n = strtol(v, &end, 10); if (end != v && end && *end == 0 && n >= 1) return (int)std::min<long>(n, 4); return 1; }()',
                                     expression('engine/src/runtime.cpp', r'(env_on\("HIVE_DECODE_PREFETCH"\))')[1])
    for name in ('HIVE_CPU_SPLIT13', 'HIVE_CPU_P2_PREFETCH'):
        e, loc = expression('engine/src/expert_store.cpp', r'(store_env_on\("' + name + r'"\))')
        exprs[name] = (e.replace('store_env_on', 'env_on'), loc)
    for name, var in (('HIVE_MTP_GATE2', 'mtp_gate2'), ('HIVE_BATCH_WINDOW', 'batch_window'), ('HIVE_BATCH_SJF', 'batch_sjf'), ('HIVE_PREFILL_FAIR', 'prefill_fair')):
        exprs[name] = expression('engine/src/hived.cpp', r'static const bool ' + var + r' = (env_on\("' + name + r'"\));')
    exprs['HIVE_BATCH_WINDOW_MS'] = ('[] { const char* v = getenv("HIVE_BATCH_WINDOW_MS"); const double x = v && *v ? atof(v) : 50.0; return std::isfinite(x) && x > 0 ? std::min(x, 10000.0) : 0.0; }()',
                                     expression('engine/src/hived.cpp', r'(getenv\("HIVE_BATCH_WINDOW_MS"\))')[1])
    exprs['HIVE_DECODE_EARLY_ROUTE'] = expression('engine/src/runtime.cpp', r'er_on = (env_on\("HIVE_DECODE_EARLY_ROUTE"\));')
    exprs['HIVE_CPU_MULTIROW2'] = ('[] { const char* v = getenv("HIVE_CPU_MULTIROW2"); if (!(v && *v && strcmp(v, "0") != 0)) return 0; char* end = nullptr; const long x = strtol(v, &end, 10); return end != v && end && *end == 0 && x >= 2 && x <= 8 ? (int)x : 5; }()',
                                   expression('engine/src/cpu/expert_cpu.cpp', r'(getenv\("HIVE_CPU_MULTIROW2"\))')[1])
    exprs['HIVE_CPU_UNPACK2'] = ('[] { const char* v = getenv("HIVE_CPU_UNPACK2"); return v && *v && strcmp(v, "0") != 0; }()',
                                 expression('engine/src/cpu/expert_cpu.cpp', r'(getenv\("HIVE_CPU_UNPACK2"\))')[1])
    for name in ('HIVE_CPU_FINE', 'HIVE_STAGE_COPY2D'):
        e, loc = expression('engine/src/expert_store.cpp', r'(store_env_on\("' + name + r'"\))')
        exprs[name] = (e.replace('store_env_on', 'env_on'), loc)
    # round 11 (R1–R5)
    exprs['HIVE_MTP_VERIFY2'] = expression('engine/src/runtime.cpp', r'verify2_ = (env_on\("HIVE_MTP_VERIFY2"\));')
    exprs['HIVE_MTP_BATCH'] = expression('engine/src/runtime.cpp', r'mtp_batch_ = (env_on\("HIVE_MTP_BATCH"\));')
    exprs['HIVE_MTP_GATE3'] = expression('engine/src/hived.cpp', r'static const bool mtp_gate3 = (env_on\("HIVE_MTP_GATE3"\));')
    for name, var in (('HIVE_PREFILL_TAIL_SHORT', 'tail_short'), ('HIVE_PREFETCH_SCALE', 'prefetch_scale'), ('HIVE_PREFILL_PROF', 'prof')):
        exprs[name] = expression('engine/src/runtime.cpp', r'spo_\.' + var + r' = (env_on\("' + name + r'"\));')
    sph = (ROOT / 'engine/include/hive/short_prefill.h').read_text()
    assert 'return end != v && end && *end == 0 && n >= 2 && n <= (1L << 30) ? (int)n : 4096;' in sph, 'parse_short_adapt drifted'
    exprs['HIVE_PREFILL_SHORT_ADAPT'] = ('[] { const char* v = getenv("HIVE_PREFILL_SHORT_ADAPT"); if (!(v && *v && strcmp(v, "0") != 0)) return 0; char* end = nullptr; const long n = strtol(v, &end, 10); return end != v && end && *end == 0 && n >= 2 && n <= (1L << 30) ? (int)n : 4096; }()',
                                         expression('engine/src/runtime.cpp', r'(env_on\("HIVE_PREFILL_SHORT_ADAPT"\))')[1])
    for name in ('HIVE_DECODE_TOPK2', 'HIVE_DECODE_IDXSCORE2'):
        e, loc = expression('engine/src/kernels/decode_longctx.cu', r'static const bool on = (lctx_env_on\("' + name + r'"\));')
        exprs[name] = (e.replace('lctx_env_on', 'env_on'), loc)
    exprs['HIVE_DECODE_IDXSCORE2_KPT'] = ('[] { const char* v = getenv("HIVE_DECODE_IDXSCORE2_KPT"); const int x = v && *v ? atoi(v) : 2; return (x == 1 || x == 2 || x == 4) ? x : 2; }()',
                                          expression('engine/src/kernels/decode_longctx.cu', r'(getenv\("HIVE_DECODE_IDXSCORE2_KPT"\))')[1])
    e, loc = expression('engine/src/kernels/moe_decode.cu', r'static const bool fused3 = (md_env_on\("HIVE_DECODE_FUSED3"\));')
    exprs['HIVE_DECODE_FUSED3'] = (e.replace('md_env_on', 'env_on'), loc)
    exprs['HIVE_DECODE_FUSED3_STAGES'] = ('[] { const char* t = getenv("HIVE_DECODE_FUSED3_STAGES"); const int n = t && *t ? atoi(t) : 0; return n <= 0 ? 4 : n <= 3 ? 3 : n <= 4 ? 4 : 5; }()',
                                          expression('engine/src/kernels/moe_decode.cu', r'(getenv\("HIVE_DECODE_FUSED3_STAGES"\))')[1])
    exprs['HIVE_PREFILL_MULTI_TAIL_SHORT'] = expression('engine/src/runtime.cpp', r'spo_\.multi_tail_short = (env_on\("HIVE_PREFILL_MULTI_TAIL_SHORT"\));')
    assert 'const long want = end != v && end && *end == 0 && n >= 2 && n <= (1L << 30) ? n : floor;' in sph and 'const long floor = std::max<long>(9, (long)max_batch + 1);' in sph, 'parse_small drifted'
    exprs['HIVE_PREFILL_SMALL'] = ('[] { const char* v = getenv("HIVE_PREFILL_SMALL"); if (!(v && *v && strcmp(v, "0") != 0)) return 0L; char* end = nullptr; const long n = strtol(v, &end, 10); const long floor = std::max<long>(9, 8L + 1); const long want = end != v && end && *end == 0 && n >= 2 && n <= (1L << 30) ? n : floor; return std::max(want, floor); }()',
                                   expression('engine/src/runtime.cpp', r'(env_on\("HIVE_PREFILL_SMALL"\))')[1])
    exprs['HIVE_MTP_VERIFY2_FUSED'] = expression('engine/src/runtime.cpp', r'static const bool v2_fused = (env_on\("HIVE_MTP_VERIFY2_FUSED"\));')
    rt_el = (ROOT / 'engine/src/runtime.cpp').read_text()
    assert 'if (end != v && end && *end == 0 && x >= 2) S = (int)std::min<long>(x, 1 << 20);' in rt_el, 'HIVE_CACHE_ELASTIC parse drifted'
    exprs['HIVE_CACHE_ELASTIC'] = ('[] { if (!env_on("HIVE_CACHE_ELASTIC")) return 0; const int Sdec = 128; int S = Sdec; const char* v = getenv("HIVE_CACHE_ELASTIC"); char* end = nullptr; const long x = strtol(v, &end, 10); if (end != v && end && *end == 0 && x >= 2) S = (int)std::min<long>(x, 1 << 20); return std::max(S, Sdec); }()',
                                   expression('engine/src/runtime.cpp', r'(env_on\("HIVE_CACHE_ELASTIC"\))')[1])
    exprs['HIVE_MTP_VERIFY_ROWIND'] = expression('engine/src/runtime.cpp', r'verify_rowind_ = (env_on\("HIVE_MTP_VERIFY_ROWIND"\));')
    exprs['HIVE_PREFILL_YIELD'] = expression('engine/src/hived.cpp', r'static const bool prefill_yield = (env_on\("HIVE_PREFILL_YIELD"\));')
    hv_y = (ROOT / 'engine/src/hived.cpp').read_text()
    assert 'return v && *v && end && *end == 0 && x > 0 ? (size_t)std::min<long long>(x, 1LL << 30) : (size_t)0;' in hv_y, 'HIVE_PREFILL_YIELD_MAX parse drifted'
    exprs['HIVE_PREFILL_YIELD_MAX'] = ('[] { const char* v = getenv("HIVE_PREFILL_YIELD_MAX"); char* end = nullptr; const long long x = v && *v ? strtoll(v, &end, 10) : 0; return v && *v && end && *end == 0 && x > 0 ? (double)std::min<long long>(x, 1LL << 30) : 0.0; }()',
                                       expression('engine/src/hived.cpp', r'(getenv\("HIVE_PREFILL_YIELD_MAX"\))')[1])
    exprs['HIVE_CACHE_FIT'] = expression('engine/src/hived.cpp', r'static const bool cache_fit = (env_on\("HIVE_CACHE_FIT"\));')
    exprs['HIVE_CACHE_RESERVE_MB'] = ('[] { const char* v = getenv("HIVE_CACHE_RESERVE_MB"); char* end = nullptr; const double x = v && *v ? strtod(v, &end) : 2400.0; const double mb = v && *v && !(end != v && end && *end == 0) ? 2400.0 : x; return std::isfinite(mb) && mb >= 0 ? std::min(mb, 1e7) : 2400.0; }()',
                                      expression('engine/include/hive/cache_reserve.h', r'(getenv\("HIVE_CACHE_RESERVE_MB"\))')[1])
    # the parse lives in hive/cache_reserve.h (shared with the GLM KV growth); hived calls it once
    rssrc = (ROOT / 'engine/include/hive/cache_reserve.h').read_text()
    assert 'const double mb = v && *v && !(end != v && end && *end == 0) ? 2400.0 : x;' in rssrc, 'HIVE_CACHE_RESERVE_MB parse drifted'
    assert (ROOT / 'engine/src/hived.cpp').read_text().count('cache_reserve_bytes()') == 1, 'hived must size the cache with cache_reserve_bytes()'
    rtsrc = (ROOT / 'engine/src/runtime.cpp').read_text()
    assert 'w->eg_alias = env_on("HIVE_ENGRAM_ALIAS") && ' in rtsrc, 'HIVE_ENGRAM_ALIAS parse drifted'
    exprs['HIVE_ENGRAM_ALIAS'] = ('env_on("HIVE_ENGRAM_ALIAS")', expression('engine/src/runtime.cpp', r'(env_on\("HIVE_ENGRAM_ALIAS"\))')[1])
    exprs['HIVE_BATCH_PREFILL'] = expression('engine/src/runtime.cpp', r'batch_prefill_ = (env_on\("HIVE_BATCH_PREFILL"\));')
    pe, pe_loc = expression('engine/src/hived.cpp', r'static const int prefix_extra = ([^;]*);')
    exprs['HIVE_PREFIX_EXTRA_CHUNKS'] = (pe.replace('prefix_share', 'env_on("HIVE_PREFIX_SHARE")'), pe_loc)
    # cold load (parsers = inline functions of hive/bulk_load.h — the probe includes that header as is and calls them)
    for name, call in (('HIVE_LOAD_PAR', 'hive::load_par_threads()'), ('HIVE_LOAD_PREFAULT', 'hive::load_prefault_threads()'),
                       ('HIVE_LOAD_CHUNK_MB', '(hive::load_chunk_bytes() >> 20)'), ('HIVE_LOAD_BUFFERED', '(!hive::load_opts_from_env(8).direct)'),
                       ('HIVE_LOAD_CHECKSUM', 'hive::load_checksum_on()')):
        exprs[name] = (call, expression('engine/include/hive/bulk_load.h', r'(getenv\("' + name + r'"\))')[1])
    # HIVE_ENGRAM_SSD* (parsers = inline functions of hive/engram_ssd.h — the probe includes that header as is and calls them)
    for name, call in (('HIVE_ENGRAM_SSD', 'hive::engram_ssd_mode()'), ('HIVE_ENGRAM_SSD_CACHE_MB', '(hive::engram_ssd_cache_bytes() >> 20)'),
                       ('HIVE_ENGRAM_SSD_THREADS', 'hive::engram_ssd_threads()'), ('HIVE_ENGRAM_SSD_NO_PREFETCH', 'hive::engram_ssd_no_prefetch()'),
                       ('HIVE_ENGRAM_DIGEST', 'hive::engram_digest_on()')):
        exprs[name] = (call, expression('engine/include/hive/engram_ssd.h', r'(getenv\("' + name + r'"\))')[1])
    assert 'essd_mode_ = engram_ssd_mode();' in (ROOT / 'engine/src/expert_store.cpp').read_text(), 'HIVE_ENGRAM_SSD store switch drifted'
    assert 'engram_digest_ = engram_digest_on();' in (ROOT / 'engine/src/runtime.cpp').read_text(), 'HIVE_ENGRAM_DIGEST runtime switch drifted'
    hvl = (ROOT / 'engine/src/hived.cpp').read_text()
    for needle in ('const int load_par = load_par_threads();', 'const int load_prefault = load_prefault_threads();', 'if (load_checksum_on()) {',
                   'store.load_bulk(model.ckpt(), nl + n_mtp, engram_tables, load_opts_from_env(load_par), true);'):
        assert needle in hvl, f'hived load switch drifted: {needle}'
    hv = (ROOT / 'engine/src/hived.cpp').read_text()
    assert 'bool env_on(const char* name) { const char* v = getenv(name); return v && *v && strcmp(v, "0") != 0; }' in hv, 'hived env_on drifted from runtime env_on'
    rt = (ROOT / 'engine/src/runtime.cpp').read_text()
    i = rt.index('  static const int depth_env = getenv("HIVE_PREFETCH")')
    block = rt[i:rt.index('  if (depth <= 0 ||', i)]
    loc = f'engine/src/runtime.cpp:{rt[:i].count(chr(10)) + 1}'
    env_on = rt[rt.index('bool env_on(const char* name)'):].split('\n', 1)[0]
    j = rt.index('  if (const char* v = getenv("HIVE_SHORT_DMA_CAP")')
    short_cap = rt[j:rt.index('\n', j)]
    exprs['HIVE_SHORT_DMA_CAP'] = ('short_dma_cap()', f'engine/src/runtime.cpp:{rt[:j].count(chr(10)) + 1}')
    k = rt.index('  if (const char* v = getenv("HIVE_VIT_CACHE_MB")')
    vit = rt[k:rt.index('\n  }\n', k) + 4]
    vit_body = re.sub(r'vit_cache_ = std::make_unique<VitCache>\((.*)\);', r'bytes = \1;', vit)
    assert 'bytes =' in vit_body, 'HIVE_VIT_CACHE_MB block drifted'
    exprs['HIVE_VIT_CACHE_MB'] = ('vit_cache_bytes()', f'engine/src/runtime.cpp:{rt[:k].count(chr(10)) + 1}')
    m1 = rt.index('inline int host_tile_count(')
    htc = rt[m1:rt.index('\n}\n', m1) + 3]
    exprs['HIVE_PREFILL_HOST_TILES'] = ('host_tile_count(env_on("HIVE_PREFILL_HOST_TILES"), getenv("HIVE_PREFILL_HOST_MB"), 671088640, true, 3, 16384, 262144)',
                                        f'engine/src/runtime.cpp:{rt[:m1].count(chr(10)) + 1}')
    m2 = rt.index('    static const float dma_frac_prefill = [] {')
    dfp = rt[m2:rt.index('}();', m2) + 4].replace('static const float dma_frac_prefill', 'const float v_')
    exprs['HIVE_DMA_FRAC_PREFILL'] = ('dma_frac_prefill()', f'engine/src/runtime.cpp:{rt[:m2].count(chr(10)) + 1}')
    # T11 HIVE_LAYER_YIELD*: exactly the expressions in hived.cpp (parsing in hive/layer_yield.h — the table below measures each value)
    for name, var in (('HIVE_LAYER_YIELD', 'ly_period'), ('HIVE_LAYER_YIELD_SHARE', 'ly_share'), ('HIVE_LAYER_YIELD_STEPS', 'ly_steps')):
        exprs[name] = expression('engine/src/hived.cpp', r'static const (?:double|int) ' + var + r' = (ly::parse_\w+\(getenv\("' + name + r'"\)\));')
    # P1 HIVE_DECODE_PREGATE (parsing = hive/pregate.h pg::parse_k — runtime and store use the same formula) · spin limit after prefetch (exactly the expression in expert_store.cpp)
    exprs['HIVE_DECODE_PREGATE'] = expression('engine/src/runtime.cpp', r'pregate_k_ = (pg::parse_k\(getenv\("HIVE_DECODE_PREGATE"\)\));')
    exprs['HIVE_DECODE_PREGATE_SPIN_US'] = expression('engine/src/expert_store.cpp', r'pf_spin_us_ = (\[\] \{ const char\* v = getenv\("HIVE_DECODE_PREGATE_SPIN_US"\);.*?\}\(\));')
    m3 = (ROOT / 'engine/src/hived.cpp').read_text()
    assert 'return v && *v && end && *end == 0 && x > 0 ? (size_t)std::min<long long>(x, 1LL << 30) : (size_t)0;\n  }();\n  const bool layer_yield' in m3, 'HIVE_LAYER_YIELD_MAX parse drifted'
    exprs['HIVE_LAYER_YIELD_MAX'] = ('[] { const char* v = getenv("HIVE_LAYER_YIELD_MAX"); char* end = nullptr; const long long x = v && *v ? strtoll(v, &end, 10) : 0; return v && *v && end && *end == 0 && x > 0 ? (double)std::min<long long>(x, 1LL << 30) : 0.0; }()',
                                     expression('engine/src/hived.cpp', r'(getenv\("HIVE_LAYER_YIELD_MAX"\))')[1])
    code = ('#include "hive/bulk_load.h"\n#include "hive/engram_ssd.h"\n#include "hive/pregate.h"\nnamespace pg = hive::pg;\n#include <algorithm>\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <cmath>\n#include "hive/layer_yield.h"\nnamespace ly = hive::ly;\n' + env_on + '\n'
            + htc + 'float dma_frac_prefill() {\n' + dfp + '\n  return v_;\n}\n'
            'double vit_cache_bytes() {\n  double bytes = 0;\n' + vit_body + '\n  return bytes;\n}\n'
            'int short_dma_cap() {\n  int short_dma_cap_ = 8;  // runtime.h default\n' + short_cap + '\n  return short_dma_cap_;\n}\n'
            'struct L { size_t total = 100; }; struct S { int staging_slots() const { return 8; } L layout() const { return {}; } } store_;\n'
            'int prefetch_depth() {\n' + block + '  return depth;\n}\nint main() {\n')
    for name, (e, _) in exprs.items():
        code += f'  printf("%s=%.17g\\n", "{name}", (double)({e}));\n'
    code += '  printf("HIVE_PREFETCH_DEPTH=%d\\n", prefetch_depth());\n}\n'
    exe = harness.build(td, 'exprs', code, [], ['-fno-sanitize=all'])
    tables = {
        'HIVE_IMAGE_CKPT': {UNSET: 0, '': 0, '0': 0, '1': 1, '2': 1},
        'HIVE_CKPT_HISTORY': {UNSET: 1, '': 1, '0': 1, '1': 1, '3': 3},
        'HIVE_PREFILL_BUDGET_MS': {UNSET: 0, '': 0, '0': 0, '5': 5, '-3': 0, 'nan': 0},
        'HIVE_TILE_GROUP_GEMM': {UNSET: 0, '': 0, '0': 0, '1': 1, '2': 1},
        'HIVE_TAIL_SPLIT': {UNSET: 0, '': 0, '0': 0, '1': 1, '2': 1},
        'HIVE_PREFILL_PAUSE_PROMOTE': SWITCH_NUM, 'HIVE_EARLY_STREAM': SWITCH_NUM, 'HIVE_ENGRAM_PAR': SWITCH_NUM,
        'HIVE_IDX_MSUB_ACTUAL': SWITCH_NUM, 'HIVE_PREFIX_SHARE': SWITCH_NUM, 'HIVE_DEFER_CKPT': SWITCH_NUM,
        'HIVE_GRACEFUL_STOP_S': {UNSET: 0, '': 0, '0': 0, '-1': 0, '2.5': 2.5, '30': 30},
        'HIVE_PREFIX_EXTRA_CHUNKS': {UNSET: 0, '': 0, '0': 0, '3': 0},  # not read without HIVE_PREFIX_SHARE (0) — the enabled case is below
        'HIVE_MTP_GATE2': SWITCH_NUM, 'HIVE_BATCH_WINDOW': SWITCH_NUM, 'HIVE_BATCH_SJF': SWITCH_NUM, 'HIVE_PREFILL_FAIR': SWITCH_NUM,
        'HIVE_BATCH_WINDOW_MS': {UNSET: 50, '': 50, '0': 0, '-1': 0, '120': 120, '99999': 10000},
        'HIVE_MTP_VERIFY2': SWITCH_NUM, 'HIVE_MTP_BATCH': SWITCH_NUM, 'HIVE_MTP_GATE3': SWITCH_NUM,
        'HIVE_PREFILL_TAIL_SHORT': SWITCH_NUM, 'HIVE_PREFETCH_SCALE': SWITCH_NUM, 'HIVE_PREFILL_PROF': SWITCH_NUM,
        'HIVE_PREFILL_SHORT_ADAPT': {UNSET: 0, '': 0, '0': 0, '1': 4096, 'x': 4096, '-5': 4096, '256': 256},
        'HIVE_DECODE_TOPK2': SWITCH_NUM, 'HIVE_DECODE_IDXSCORE2': SWITCH_NUM,
        'HIVE_DECODE_IDXSCORE2_KPT': {UNSET: 2, '': 2, '0': 2, '1': 1, '3': 2, '4': 4},
        'HIVE_DECODE_FUSED3': SWITCH_NUM, 'HIVE_DECODE_FUSED3_STAGES': {UNSET: 4, '': 4, '0': 4, '2': 3, '4': 4, '5': 5, '9': 5},
        'HIVE_PREFILL_MULTI_TAIL_SHORT': SWITCH_NUM, 'HIVE_PREFILL_SMALL': {UNSET: 0, '': 0, '0': 0, '1': 9, 'x': 9, '4': 9, '64': 64},  # assuming max_batch 8
        'HIVE_MTP_VERIFY2_FUSED': SWITCH_NUM, 'HIVE_MTP_VERIFY_ROWIND': SWITCH_NUM,
        'HIVE_CACHE_ELASTIC': {UNSET: 0, '': 0, '0': 0, '1': 128, 'x': 128, '64': 128, '1023': 1023},  # small-panel rows S (limit below = service window 128)
        'HIVE_PREFILL_YIELD': SWITCH_NUM, 'HIVE_PREFILL_YIELD_MAX': {UNSET: 0, '': 0, '0': 0, '-3': 0, 'x': 0, '4096': 4096},  # 0 = default (prefill_threshold − 1) · only read when YIELD is on (only the expression here)
        'HIVE_CACHE_FIT': SWITCH_NUM, 'HIVE_ENGRAM_ALIAS': SWITCH_NUM,
        'HIVE_LOAD_PAR': {UNSET: 0, '': 0, '0': 0, '1': 8, '2': 2, 'x': 8, '12': 12, '999': 64},
        'HIVE_LOAD_PREFAULT': {UNSET: 0, '': 0, '0': 0, '1': 16, 'x': 16, '4': 4, '999': 64},
        'HIVE_LOAD_CHUNK_MB': {UNSET: 32, '': 32, '0': 32, 'x': 32, '8': 8, '2000': 32},
        'HIVE_LOAD_BUFFERED': SWITCH_NUM, 'HIVE_LOAD_CHECKSUM': SWITCH_NUM,
        'HIVE_ENGRAM_SSD': {UNSET: 0, '': 0, '0': 0, '1': 1, 'on': 1, '2': 2, 'all': 2},
        'HIVE_ENGRAM_SSD_CACHE_MB': {UNSET: 2048, '': 2048, 'x': 2048, '-3': 2048, '0': 0, '512': 512},
        'HIVE_ENGRAM_SSD_THREADS': {UNSET: 64, '': 64, '0': 64, 'x': 64, '128': 128, '999': 256},
        'HIVE_ENGRAM_SSD_NO_PREFETCH': SWITCH_NUM, 'HIVE_ENGRAM_DIGEST': SWITCH_NUM,
        # T11: period ms (0 = off) · share · steps · admit limit (0 = default threshold − 1)
        'HIVE_LAYER_YIELD': {UNSET: 0, '': 0, '0': 0, '1': 500, 'on': 500, '49': 500, '75': 75, '1e9': 60000, 'x': 500},
        'HIVE_LAYER_YIELD_SHARE': {UNSET: 0.05, '': 0.05, '0': 0.05, 'x': 0.05, '0.95': 0.05, '0.1': 0.1},
        'HIVE_LAYER_YIELD_STEPS': {UNSET: 4, '': 4, '0': 4, 'x': 4, '2': 2, '999': 64},
        'HIVE_LAYER_YIELD_MAX': {UNSET: 0, '': 0, '0': 0, '-3': 0, 'x': 0, '300': 300},
        'HIVE_CACHE_RESERVE_MB': {UNSET: 2400, '': 2400, 'x': 2400, '12abc': 2400, '-1': 2400, '0': 0, '3000': 3000},
        'HIVE_DECODE_EARLY_ROUTE': SWITCH_NUM, 'HIVE_CPU_UNPACK2': SWITCH_NUM, 'HIVE_CPU_FINE': SWITCH_NUM, 'HIVE_STAGE_COPY2D': SWITCH_NUM,
        'HIVE_CPU_MULTIROW2': {UNSET: 0, '': 0, '0': 0, '1': 5, '3': 3, '9': 5, 'x': 5},
        'HIVE_DECODE_ROUTER3': SWITCH_NUM, 'HIVE_DECODE_SPARSE3': SWITCH_NUM, 'HIVE_DECODE_QKV3': SWITCH_NUM, 'HIVE_DECODE_HCMIX3': SWITCH_NUM,
        'HIVE_DECODE_SPARSE3_G': {UNSET: 0, '': 0, '0': 0, '3': 0, '4': 4, '8': 8},
        'HIVE_DECODE_PREGATE': {UNSET: 0, '': 0, '0': 0, '1': 8, 'x': 8, '6': 6, '40': 16},  # P1 top K (1 or non-numeric = default 8)
        'HIVE_DECODE_PREGATE_SPIN_US': {UNSET: 1000, '': 1000, '0': 1000, 'x': 1000, '-5': 1000, '1': 1, '300': 300, '999999': 100000},
        'HIVE_DECODE_CPU_FIRST': SWITCH_NUM, 'HIVE_DECODE_STAGE_HIT': SWITCH_NUM, 'HIVE_CPU_SPLIT13': SWITCH_NUM, 'HIVE_CPU_P2_PREFETCH': SWITCH_NUM,
        'HIVE_DECODE_PREFETCH': {UNSET: 0, '': 0, '0': 0, '1': 1, '3': 3, '9': 4, 'x': 1},
        'HIVE_DECODE_ATTN2': SWITCH_NUM, 'HIVE_DECODE_HOST_FAST': SWITCH_NUM, 'HIVE_DECODE_UBATCH': SWITCH_NUM,
        'HIVE_DECODE_SPLIT': {UNSET: 0, '': 0, '0': 0, 'balance': 1}, 'HIVE_CPU_GEMV2': SWITCH_NUM,
        'HIVE_GROUPED_PREFILL': SWITCH_NUM, 'HIVE_BATCH_PREFILL': SWITCH_NUM,
        'HIVE_DECODE_FUSED2': SWITCH_NUM, 'HIVE_DECODE_FUSED2_STAGES': {UNSET: 4, '': 4, '0': 4, '2': 3, '4': 4, '5': 4, '9': 6},
        'HIVE_HC_DECODE_FUSED': SWITCH_NUM, 'HIVE_HC_SINKHORN_PAR': SWITCH_NUM, 'HIVE_DECODE_COPY_PRIO': SWITCH_NUM,
        'HIVE_DECODE_STEP_GRAPH': SWITCH_NUM, 'HIVE_PREFILL_SPLIT': {UNSET: 0, '': 0, '0': 0, 'balance': 1},
        'HIVE_DECODE_STEP_TIMEOUT_MS': {UNSET: 10000, '': 10000, '0': 10000, '-5': 10000, '2500': 2500},
        'HIVE_DECODE_STEP_POLL_US': {UNSET: 0, '': 0, '0': 0, '-1': 0, '50': 50},
        'HIVE_DECODE_STEP_SPIN_US': {UNSET: 0, '': 0, '0': 0, '20': 20},
        'HIVE_DECODE_FUSED': SWITCH_NUM, 'HIVE_DECODE_BLOCKING_SYNC': SWITCH_NUM, 'HIVE_DECODE_ATTN_FUSED': SWITCH_NUM,
        'HIVE_DECODE_GEMV2': SWITCH_NUM, 'HIVE_DECODE_PDL': SWITCH_NUM,
        'HIVE_DECODE_SYNC_SPIN_US': {UNSET: 0, '': 0, '0': 0, '-3': 0, '50': 50},
        'HIVE_CACHE_POLICY': {UNSET: 0, '': 0, '0': 0, 'seq': 1},  # only the on/off decision (name parsing is parse_cache_policy — unknown names are absorbed into the default policy)
        # host tiles (640 MiB h, 3 resident, 16K × 16 = 262144 limit → at most 13): on alone = 2 · MB value decides the count
        'HIVE_PREFILL_HOST_TILES': {UNSET: 0, '': 0, '0': 0, '1': 2, 'true': 2},
        'HIVE_DMA_FRAC_PREFILL': {UNSET: -1, '': -1, '0': -1, '1.5': -1, 'x': -1, '0.9': 0.8999999761581421, '1': 1},  # float32 0.9
        'HIVE_VIT_CACHE_MB': {UNSET: 0, '': 0, '0': 0, '-4': 0, 'x': 0, '1': 1048576, '0.5': 524288},
        # unset/""/"0"/garbage → 8 (the pre-option cap); value otherwise
        'HIVE_SHORT_DMA_CAP': {UNSET: 8, '': 8, '0': 8, '-2': 8, 'x': 8, '4': 4, '32': 32},
        # depth = min(staging 8, HIVE_PREFETCH) then capped by HIVE_PREFETCH_BYTES / record(100 B)
        'HIVE_PREFETCH': {UNSET: 0, '': 0, '0': 0, '4': 4, '300': 8},
    }
    rows, bad = [], []
    for name, table in tables.items():
        key = 'HIVE_PREFETCH_DEPTH' if name == 'HIVE_PREFETCH' else name
        for val, want in table.items():
            env = {k: v for k, v in os.environ.items() if not k.startswith('HIVE_')}
            if val is not None:
                env[name] = val
            out = dict(l.split('=', 1) for l in subprocess.run([exe], env=env, capture_output=True, text=True, check=True).stdout.split())
            got = float(out[key])
            rows.append((name, val, want, got))
            if got != want:
                bad.append((name, val, want, got))
    for val, want in ((UNSET, 1), ('', 1), ('0', 0), ('3', 3), ('-2', 0)):
        env = {k: v for k, v in os.environ.items() if not k.startswith('HIVE_')}
        env['HIVE_PREFIX_SHARE'] = '1'
        if val is not None:
            env['HIVE_PREFIX_EXTRA_CHUNKS'] = val
        got = float(dict(l.split('=', 1) for l in subprocess.run([exe], env=env, capture_output=True, text=True, check=True).stdout.split())['HIVE_PREFIX_EXTRA_CHUNKS'])
        rows.append(('HIVE_PREFIX_EXTRA_CHUNKS+SHARE', val, want, got))
        if got != want:
            bad.append(('HIVE_PREFIX_EXTRA_CHUNKS+SHARE', val, want, got))
    for val, want in ((UNSET, 2), ('', 2), ('0', 0), ('x', 0), ('-5', 0), ('1280', 2), ('6400', 10), ('1e9', 13)):
        env = {k: v for k, v in os.environ.items() if not k.startswith('HIVE_')}
        env['HIVE_PREFILL_HOST_TILES'] = '1'
        if val is not None:
            env['HIVE_PREFILL_HOST_MB'] = val
        got = float(dict(l.split('=', 1) for l in subprocess.run([exe], env=env, capture_output=True, text=True, check=True).stdout.split())['HIVE_PREFILL_HOST_TILES'])
        rows.append(('HIVE_PREFILL_HOST_MB', val, want, got))
        if got != want:
            bad.append(('HIVE_PREFILL_HOST_MB', val, want, got))
    for bytes_val, want in ((UNSET, 4), ('', 0), ('0', 0), ('250', 2), ('100000', 4)):
        env = {k: v for k, v in os.environ.items() if not k.startswith('HIVE_')}
        env['HIVE_PREFETCH'] = '4'
        if bytes_val is not None:
            env['HIVE_PREFETCH_BYTES'] = bytes_val
        got = int(dict(l.split('=', 1) for l in subprocess.run([exe], env=env, capture_output=True, text=True, check=True).stdout.split())['HIVE_PREFETCH_DEPTH'])
        rows.append(('HIVE_PREFETCH_BYTES', bytes_val, want, got))
        if got != want:
            bad.append(('HIVE_PREFETCH_BYTES', bytes_val, want, got))
    locs = {n: l for n, (_, l) in exprs.items()}
    locs['HIVE_PREFETCH'] = locs['HIVE_PREFETCH_BYTES'] = loc
    return rows, bad, locs


def run_launcher(td, script, args=(), env_extra=None):
    """Run scripts/<script> for real in a scratch copy of the repository with stub docker/nvidia-smi/pgrep on PATH.
    Returns the argv of the recorded `docker run` call (list of str)."""
    root = Path(tempfile.mkdtemp(prefix='launcher-', dir=td))
    (root / 'scripts').mkdir()
    for name in ('lib.sh', script):
        (root / 'scripts' / name).write_text((ROOT / 'scripts' / name).read_text())
    (root / 'config').mkdir()
    (root / 'config/hive.env').write_text((ROOT / 'config/hive.env').read_text())
    for build in ('build', 'build-dev'):
        (root / 'engine' / build).mkdir(parents=True)
        hived = root / 'engine' / build / 'hived'
        hived.write_text('#!/bin/sh\n')
        hived.chmod(0o755)
    (root / 'ckpt').mkdir()
    (root / 'ckpt/config.json').write_text('{}')
    (root / 'state/engram').mkdir(parents=True)
    stub = root / 'bin'
    stub.mkdir()
    log = root / 'docker.log'
    stubs = {'docker': '#!/bin/bash\n[ "$1" = run ] && printf "%s\\0" "$@" > "$STUB_DOCKER_LOG"\nexit 0\n',
             'nvidia-smi': '#!/bin/sh\necho 999999\n', 'pgrep': '#!/bin/sh\nexit 1\n'}
    for name, body in stubs.items():
        (stub / name).write_text(body)
        (stub / name).chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith('HIVE_')}
    env.update({'PATH': f'{stub}:/usr/bin:/bin', 'STUB_DOCKER_LOG': str(log), 'HIVE_CKPT': str(root / 'ckpt'),
                'HIVE_STATE_DIR': str(root / 'state'), 'HIVE_RAM_WAIT_GB': '0'})
    env.update(env_extra or {})
    r = subprocess.run(['bash', str(root / 'scripts' / script), *args], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f'{script} failed: {r.stderr}'
    return [x for x in log.read_text().split('\0') if x], env


def forwarded(argv):
    """-e NAME=VALUE pairs of a docker run argv -> {NAME: VALUE} (HIVE_* only)."""
    out = {}
    for k, v in zip(argv, argv[1:]):
        if k == '-e' and v.startswith('HIVE_'):
            name, val = v.split('=', 1)
            out[name] = val
    return out


def launchers(td):
    """Layer 3a: scripts/hive-start.sh and scripts/hive-run.sh, run for real against stub docker, forward every HIVE_*
    variable by value (not by presence); hive-start selects the build only by --build, hive-run honours HIVE_BUILD."""
    bad = []
    probe = {'HIVE_A_EMPTY': '', 'HIVE_B_ZERO': '0', 'HIVE_C_SPACE': 'a b', 'HIVE_D_STAR': '*'}
    for script in ('hive-start.sh', 'hive-run.sh'):
        args = ('true',) if script == 'hive-run.sh' else ()
        argv, env = run_launcher(td, script, args, probe)
        got = forwarded(argv)
        for name, val in env.items():
            if name.startswith('HIVE_') and got.get(name) != val:
                bad.append((script, name, val, got.get(name)))
    for script, args, shell_build, want in (('hive-start.sh', (), 'build-dev', 'build'),
                                            ('hive-start.sh', ('--build', 'build-dev'), None, 'build-dev'),
                                            ('hive-start.sh', ('--build', 'build-dev'), 'build', 'build-dev'),
                                            ('hive-run.sh', ('true',), 'build-dev', 'build-dev'),
                                            ('hive-run.sh', ('true',), None, 'build')):
        argv, _ = run_launcher(td, script, args, {} if shell_build is None else {'HIVE_BUILD': shell_build})
        got = forwarded(argv).get('HIVE_BUILD')
        if got != want:
            bad.append((script, args, 'HIVE_BUILD', shell_build, 'want', want, 'got', got))
    local_env = Path(td) / 'local.env'  # a HIVE_BUILD from a configuration file does not select the service build either
    local_env.write_text('HIVE_BUILD=build-dev\n')
    for script, args, want in (('hive-start.sh', (), 'build'), ('hive-run.sh', ('true',), 'build-dev')):
        got = forwarded(run_launcher(td, script, args, {'HIVE_LOCAL_ENV': str(local_env)})[0]).get('HIVE_BUILD')
        if got != want:
            bad.append((script, 'HIVE_LOCAL_ENV HIVE_BUILD=build-dev', 'want', want, 'got', got))
    ep = (ROOT / 'scripts/hive-entrypoint.sh').read_text()
    if not re.search(r'^BIN_DIR=/hive/engine/\$\{HIVE_BUILD:-build\}$', ep, re.M):
        bad.append(('scripts/hive-entrypoint.sh', 'HIVE_BUILD default changed'))
    for val, want in ((None, 'build'), ('', 'build'), ('build-dev', 'build-dev')):
        e = {k: v for k, v in os.environ.items() if not k.startswith('HIVE_')}
        if val is not None:
            e['HIVE_BUILD'] = val
        got = subprocess.run(['bash', '-c', 'echo "${HIVE_BUILD:-build}"'], env=e, capture_output=True, text=True).stdout.strip()
        if got != want:
            bad.append(('HIVE_BUILD', val, got))
    return bad


def presence_audit():
    """Layer 3b: no HIVE_* switch may be decided by getenv(...) != nullptr alone."""
    bad, paths = [], []
    conv = r'(atoi|atof|atoll|strtol|strtoll|strcmp|env_on|ly::parse_\w+|pg::parse_\w+)\s*\('  # T11 ly::parse_* = value parsing of hive/layer_yield.h (measured by the table)
    for path in sorted((ROOT / 'engine/src').rglob('*.c*')):
        lines = path.read_text().splitlines()
        for n, line in enumerate(lines, 1):
            for m in re.finditer(r'getenv\("(HIVE_[A-Z0-9_]+)"\)', line):
                name, where = m.group(1), f'{path.relative_to(ROOT)}:{n}'
                if re.search(r'getenv\("' + name + r'"\)\s*\?\s*getenv\("' + name + r'"\)', line):  # `getenv(X) ? getenv(X) : ""` = the value itself (a path)
                    if (name, where) not in paths:
                        paths.append((name, where))
                    continue
                var = re.search(r'const char\*\s*(\w+)\s*=\s*getenv\("' + name, line)
                if var:  # value bound to a variable: converted within the next lines, or used as a path
                    near = '\n'.join(lines[n - 1:n + 3])
                    if re.search(r'(atoi|atof|atoll|strtol|strtoll)\s*\(\s*' + var.group(1) + r'\s*\)', near):
                        continue
                    paths.append((name, where))
                elif not re.search(conv, line):
                    bad.append((name, where, line.strip()[:140]))
    return bad, paths


# Optional engine switches whose parsing and behaviour this test must observe (every one listed in docs/configuration.md).
DOC_OPTIONS = (
    'HIVE_BATCH_PREFILL', 'HIVE_BATCH_SJF', 'HIVE_BATCH_WINDOW', 'HIVE_BATCH_WINDOW_MS', 'HIVE_CACHE_ELASTIC',
    'HIVE_CACHE_EVENTS', 'HIVE_CACHE_FIT', 'HIVE_CACHE_POLICY', 'HIVE_CACHE_RESERVE_MB', 'HIVE_CACHE_REUSE_STAGE',
    'HIVE_CKPT_ASYNC', 'HIVE_CKPT_DELTA', 'HIVE_CKPT_HISTORY', 'HIVE_CKPT_PINNED_POOL_MB', 'HIVE_CPU_FINE',
    'HIVE_CPU_GEMV2', 'HIVE_CPU_MULTIROW2', 'HIVE_CPU_P2_PREFETCH', 'HIVE_CPU_SPLIT13', 'HIVE_CPU_UNPACK2',
    'HIVE_DECODE_ATTN2', 'HIVE_DECODE_ATTN_FUSED', 'HIVE_DECODE_BLOCKING_SYNC', 'HIVE_DECODE_COPY_PRIO',
    'HIVE_DECODE_CPU_FIRST', 'HIVE_DECODE_EARLY_ROUTE', 'HIVE_DECODE_FUSED', 'HIVE_DECODE_FUSED2',
    'HIVE_DECODE_FUSED2_STAGES', 'HIVE_DECODE_FUSED3', 'HIVE_DECODE_GEMV2', 'HIVE_DECODE_HCMIX3',
    'HIVE_DECODE_HOST_FAST', 'HIVE_DECODE_IDXSCORE2', 'HIVE_DECODE_PDL', 'HIVE_DECODE_PREFETCH',
    'HIVE_DECODE_PREGATE', 'HIVE_DECODE_PREGATE_SPIN_US', 'HIVE_DECODE_QKV3', 'HIVE_DECODE_ROUTER3',
    'HIVE_DECODE_SPARSE3', 'HIVE_DECODE_SPARSE3_G', 'HIVE_DECODE_SPLIT', 'HIVE_DECODE_STAGE_HIT',
    'HIVE_DECODE_STEP_GRAPH', 'HIVE_DECODE_STEP_POLL_US', 'HIVE_DECODE_STEP_SPIN_US', 'HIVE_DECODE_STEP_TIMEOUT_MS',
    'HIVE_DECODE_SYNC_SPIN_US', 'HIVE_DECODE_TOPK2', 'HIVE_DECODE_UBATCH', 'HIVE_DEFER_CKPT',
    'HIVE_DMA_FRAC_PREFILL', 'HIVE_EARLY_STREAM', 'HIVE_ENGRAM_ALIAS', 'HIVE_ENGRAM_DIGEST', 'HIVE_ENGRAM_PAR',
    'HIVE_ENGRAM_SSD', 'HIVE_ENGRAM_SSD_CACHE_MB', 'HIVE_ENGRAM_SSD_NO_PREFETCH', 'HIVE_ENGRAM_SSD_THREADS',
    'HIVE_GRACEFUL_STOP_S', 'HIVE_GROUPED_PREFILL', 'HIVE_HC_DECODE_FUSED', 'HIVE_HC_SINKHORN_PAR',
    'HIVE_IDX_MSUB_ACTUAL', 'HIVE_IMAGE_CKPT', 'HIVE_LAYER_YIELD', 'HIVE_LOAD_BUFFERED', 'HIVE_LOAD_CHECKSUM',
    'HIVE_LOAD_CHUNK_MB', 'HIVE_LOAD_PAR', 'HIVE_LOAD_PREFAULT', 'HIVE_MTP_BATCH', 'HIVE_MTP_CACHE',
    'HIVE_MTP_GATE2', 'HIVE_MTP_GATE3', 'HIVE_MTP_VERIFY2', 'HIVE_MTP_VERIFY2_FUSED', 'HIVE_MTP_VERIFY_ROWIND',
    'HIVE_PHASE_SCORE', 'HIVE_PREFETCH', 'HIVE_PREFETCH_BYTES', 'HIVE_PREFETCH_SCALE', 'HIVE_PREFILL_BUDGET_MS',
    'HIVE_PREFILL_FAIR', 'HIVE_PREFILL_HOST_MB', 'HIVE_PREFILL_HOST_TILES', 'HIVE_PREFILL_MULTI_TAIL_SHORT',
    'HIVE_PREFILL_PAUSE_PROMOTE', 'HIVE_PREFILL_PROF', 'HIVE_PREFILL_SHORT_ADAPT', 'HIVE_PREFILL_SMALL',
    'HIVE_PREFILL_SPLIT', 'HIVE_PREFILL_TAIL_SHORT', 'HIVE_PREFILL_YIELD', 'HIVE_PREFILL_YIELD_MAX',
    'HIVE_PREFIX_EXTRA_CHUNKS', 'HIVE_PREFIX_SHARE', 'HIVE_PROMOTE_BYTES', 'HIVE_PROMOTE_SCORE',
    'HIVE_SHORT_DMA_CAP', 'HIVE_STAGE_COPY2D', 'HIVE_TAIL_SPLIT', 'HIVE_TILE_GROUP_GEMM', 'HIVE_VIT_CACHE_MB',
)


def main():
    doc_opts = sorted(DOC_OPTIONS)
    conf = (ROOT / 'docs/configuration.md').read_text()
    undocumented = [o for o in doc_opts if '`' + o + '`' not in conf]
    assert not undocumented, f'options missing from docs/configuration.md: {undocumented}'
    with tempfile.TemporaryDirectory(prefix='hive-options-test-') as td:
        gen = Path(td) / 'fake_runtime_gen.cpp'
        gen.write_text(harness.runtime_source())
        objs = harness.objects(td, [gen, ROOT / 'engine/src/expert_store.cpp', ROOT / 'engine/src/cpu/expert_cpu.cpp', harness.FAKE / 'fake_checkpoint.cpp'])
        exe = harness.build(td, 'options', '', [ROOT / 'engine/tests/test_options_cpu.cpp', *objs])
        rows1, bad1, xfail = behaviour(exe, td)
        rows2, bad2, locs = source_exprs(td)
        bad3 = launchers(td)
    bad4, paths = presence_audit()
    covered = {r[0] for r in rows1 + rows2}
    missing = [o for o in doc_opts if o not in covered]
    print(f'options: {len(doc_opts)} doc options, {len(rows1)} behaviour + {len(rows2)} source-expression observations')
    for opt in doc_opts:
        vals = [(v, g) for o, v, _, g in rows1 + rows2 if o == opt]
        print(f'  {opt:28s} {locs.get(opt, "behaviour")}: ' + ', '.join(f'{v!r}->{g}' for v, g in vals))
    print('  path-valued options (any set value is a path): ' + ', '.join(f'{n} {l}' for n, l in paths))
    for opt, val, want, got in xfail:
        print(f'XFAIL {opt}={val!r}: want {want} got {got} — {KNOWN[(opt, val)]}')
    fails = [('behaviour', b) for b in bad1] + [('source', b) for b in bad2] + [('launcher', b) for b in bad3] + \
            [('presence-only', b) for b in bad4] + [('doc option without a check', m) for m in missing]
    for f in fails:
        print('FAIL', f)
    if fails:
        sys.exit(1)
    print('options CPU: unset/""/0 = default path, values forwarded as given, launchers forward by value, no presence-only switches')


if __name__ == '__main__':
    main()
