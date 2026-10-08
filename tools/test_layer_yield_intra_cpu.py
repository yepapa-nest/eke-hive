#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""T11b HIVE_LAYER_YIELD_INTRA — CPU checks (no GPU, no CUDA build).

1. engine/tests/test_layer_yield_intra_cpu.cpp (hive/layer_yield.h + the REAL ExpertStore on fake CUDA): HIVE_LAYER_YIELD_CAP parsing,
   the look-ahead decision (intra_due) and segment EMA, yield_point_intra (decode-only rows, point kind, restore after exceptions, cost bound,
   park failure), staging hold (copy_to_staging while held aborts). Negative control: expert_store.cpp without the abort → a held write
   overwrites a pre-copied record (and the unit test fails).
2. Source contracts on engine/src/runtime.cpp (each with negative controls — every mutant must be detected):
   * staging writers: every function that writes the staging ring is known; the decode writers (moe_decode_experts, bm_prefetch,
     forward_batch_step) are gated by the hold, the prefill writers (prefetch_layer_experts, moe_experts_multi) are reachable only from
     prefill forwards; ExpertStore::copy_to_staging aborts while held.
   * Work members: every member of struct Runtime::Work is in exactly one of — the elastic list (ElasticScope::setup), the layer-boundary park
     list, the intra park list, the LY-INTRA-WORK table (dead at every intra point, with the reason; "= x" = same memory as x).
   * park/unpark of intra points: pre-copies restored (not cleared), hold set and restored, a unit visit closed and reopened, deferred decode
     batch flushed on unpark; off path = the intra points return before anything else; quiesce discards a deferred batch.
3. The REAL hived.cpp on the fake runtime (FAKE_INTRA points inside each fake layer): intra yields run decode only (a prefill forward inside
   one is a fake-state error; admissions follow boundary yields only), a running decoder's longest gap shrinks versus the switch off, the
   staging hold is respected by decode (FAKE_PF + FAKE_DECODE_STAGE; FAKE_IGNORE_HOLD = mutant → the real store aborts), a decode failure
   inside an intra yield leaves the paused prompt correct and no hold behind; switch off = no intra line. Mutant: hived admitting at intra
   points (want/cap without the intra rule) → detected.
Run: nice -n 19 python3 tools/test_layer_yield_intra_cpu.py
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake'), str(Path(__file__).resolve().parent)]
import harness  # noqa: E402
import test_daemon_cpu as t  # noqa: E402

ROOT = harness.ROOT
RT = ROOT / 'engine/src/runtime.cpp'
STORE = ROOT / 'engine/src/expert_store.cpp'
HIVED = ROOT / 'engine/src/hived.cpp'
TEST = ROOT / 'engine/tests/test_layer_yield_intra_cpu.cpp'
OTHER = [ROOT / 'engine/src/cpu/expert_cpu.cpp', harness.FAKE / 'fake_checkpoint.cpp']
SLOW = t.SLOW


# ---- 1. C++ unit test --------------------------------------------------------------------------------------------------------------------------------
STORE_GUARD = ('  if (staging_held()) {\n'
               '    fprintf(stderr, "[store] FATAL: staging slot %d written while held by an intra-layer yield (layer %d expert %d) — a decode staging writer ignored the hold\\n", slot, l, e);\n'
               '    abort();\n  }\n')


def unit(td, ck):
    exe = harness.build(td, 'lyi_unit', '', [TEST, STORE, *OTHER], extra=['-I' + str(ROOT / 'engine/tests'), '-D_GLIBCXX_ASSERTIONS'])
    p = subprocess.run([exe], env=harness.tsan_env(), capture_output=True, text=True, timeout=600)
    ck.true('unit test passes', p.returncode == 0 and 'all passed' in p.stdout, (p.stdout + p.stderr)[-1500:])
    p = subprocess.run([exe, '--staging-overwrite'], capture_output=True, text=True, timeout=600)
    ck.true('real store: a held write aborts before touching the record', p.returncode != 0 and 'OVERWRITTEN' not in p.stdout and 'FATAL: staging slot' in p.stderr,
            f'rc {p.returncode} {p.stdout[-200:]}')
    src = STORE.read_text()
    assert src.count(STORE_GUARD) == 1, 'expert_store.cpp hold guard drifted'
    mut = Path(td) / 'expert_store_nohold.cpp'
    mut.write_text(src.replace(STORE_GUARD, ''))
    exe2 = harness.build(td, 'lyi_unit_mut', '', [TEST, mut, *OTHER], extra=['-I' + str(ROOT / 'engine/include'), '-I' + str(ROOT / 'engine/tests'), '-D_GLIBCXX_ASSERTIONS'])
    p = subprocess.run([exe2, '--staging-overwrite'], capture_output=True, text=True, timeout=600)
    ck.true('mutant store (no hold check): the held write overwrites the pre-copy', p.returncode == 0 and 'OVERWRITTEN' in p.stdout, p.stdout[-200:] + p.stderr[-300:])
    p = subprocess.run([exe2], capture_output=True, text=True, timeout=600)
    ck.true('mutant store: unit test detects it', p.returncode != 0, p.stdout[-300:])


# ---- 2. source contracts ---------------------------------------------------------------------------------------------------------------------------
FUNC_RE = re.compile(r'^[A-Za-z][^;]*?\bRuntime::(\w+)\(')
WRITER_RE = re.compile(r'copy_to_staging\(|stage_next_\s*(?:\+\+|\+=)|plan_(?:batches|early_stream)\(')
DECODE_WRITERS = {
    'moe_decode_experts': ['const bool held = store_.staging_held();', 'if (held) gpu_share_left = 0;', 'BatchMiss* B = !held && ',
                           '&& !vrowind && !held) {  // R1 ROWIND layers stay outside the model'],
    'bm_prefetch': ['  if (store_.staging_held()) return;'],
    'forward_batch_step': ['M * k > hs::kMaxR || store_.staging_held())'],
}
PREFILL_WRITERS = {'prefetch_layer_experts', 'moe_experts_multi'}  # called only by prefill forwards (never inside a decode-only yield)


def functions(src):
    """[(name, body)] of Runtime member function definitions (body = text up to the next definition)."""
    out, cur, buf = [], None, []
    for line in src.splitlines(keepends=True):
        m = FUNC_RE.match(line)
        if m and not line.rstrip().endswith(';'):
            if cur:
                out.append((cur, ''.join(buf)))
            cur, buf = m[1], []
        if cur:
            buf.append(line)
    if cur:
        out.append((cur, ''.join(buf)))
    return out


def staging_contract(rt, store):
    """Returns a list of failures (empty = ok)."""
    bad = []
    fns = functions(rt)
    writers = {}
    for name, body in fns:
        lines = [l for l in body.splitlines() if not l.lstrip().startswith('//')]
        if any(WRITER_RE.search(l.split('//')[0]) for l in lines):
            writers.setdefault(name, body)
    unknown = set(writers) - set(DECODE_WRITERS) - PREFILL_WRITERS
    if unknown:
        bad.append(f'unclassified staging writers {sorted(unknown)} (decode writers need the hold gate, prefill writers must be prefill-only)')
    for name, gates in DECODE_WRITERS.items():
        body = writers.get(name)
        if body is None:
            bad.append(f'{name}: no longer writes staging (update the contract)')
            continue
        for g in gates:
            if g not in body:
                bad.append(f'{name}: hold gate missing: {g!r}')
        first = WRITER_RE.search(body)
        held = body.find('staging_held()')
        if held < 0 or (first and held > first.start()):
            bad.append(f'{name}: hold read after the first staging write')
    calls = {n: [] for n in PREFILL_WRITERS}
    for name, body in fns:
        for n in PREFILL_WRITERS:
            if re.search(rf'(?<![\w:]){n}\(', body) and name != n:
                calls[n].append(name)
    if set(calls['prefetch_layer_experts']) - {'forward', 'forward_multi', 'layer_forward'}:
        bad.append(f'prefetch_layer_experts called from {sorted(calls["prefetch_layer_experts"])}')
    if set(calls['moe_experts_multi']) - {'forward', 'forward_multi', 'moe_experts'}:
        bad.append(f'moe_experts_multi called from {sorted(calls["moe_experts_multi"])}')
    if '  if ((M >= opt_.prefill_threshold || in_prefill_) && !verify_ && !short_moe) prefetch_layer_experts(l, M);' not in rt:
        bad.append('layer_forward: pre-copy no longer gated to prefill (not verify)')
    if '  if (prefill) moe_experts(seq, L, l, M, stats);' not in rt or '    in_prefill_ = false; score_prefill_ = false; small_fwd_ = false;' not in rt:
        bad.append('moe(): streaming experts no longer gated to prefill, or the park no longer clears in_prefill_')
    cs = store[store.index('void ExpertStore::copy_to_staging(int slot, int l, int e, cudaStream_t st) {'):]
    cs = cs[:cs.index('\n}\n')]
    if STORE_GUARD not in cs or cs.index(STORE_GUARD) > cs.index('copy_rec_async('):
        bad.append('ExpertStore::copy_to_staging: hold abort missing or after the copy')
    return bad


def parse_work_members(rt):
    w = rt[rt.index('struct Runtime::Work {'):]
    w = w[w.index('{') + 1:w.index('\n};\n')]
    w = re.sub(r'\n  struct Elastic \{.*?\n  \} el;', '\n  Elastic el;', w, flags=re.S)
    w = re.sub(r'//[^\n]*', '', w)
    w = re.sub(r'~Work\(\)\s*\{.*?\}\s*\}', '', w, flags=re.S)
    while re.search(r'<[^<>]*>', w):
        w = re.sub(r'<[^<>]*>', '', w)
    names = []
    for stmt in w.split(';'):
        stmt = re.sub(r'\s+', ' ', stmt).strip()
        if not stmt:
            continue
        stmt = re.sub(r'=\s*[^,]+', '', stmt)
        for i, part in enumerate(stmt.split(',')):
            m = re.search(r'(\w+)\s*$', part.strip())
            assert m, f'cannot parse Work declaration {stmt!r}'
            names.append(m[1])
    return names


def work_lists(rt):
    setup = rt[rt.index('  static void setup(Runtime& r) {'):]
    setup = setup[:setup.index('    size_t off = 0, small = 0;')]
    elastic = set(re.findall(r'&w\.(\w+)', setup))
    p = rt[rt.index('void Runtime::layer_yield_point(int kind, int layer, bool last) {'):]
    p = p[:p.index('\n}\n')]
    park = p[:p.index('auto unpark')]
    boundary = set(re.findall(r'\b(?:dv|hb)\(w\.(\w+)', park)) | set(re.findall(r'P\.dev\.push_back\(\{w\.(\w+)\.p', park))
    intra = set(re.findall(r'\biv\(w\.(\w+)', park))
    tab = rt[rt.index('// LY-INTRA-WORK BEGIN'):rt.index('// LY-INTRA-WORK END')]
    table = {}
    for line in tab.splitlines()[1:]:
        m = re.match(r'^//   (\w+): (.*)$', line)
        assert m, f'bad LY-INTRA-WORK line {line!r}'
        assert m[1] not in table, f'duplicate table entry {m[1]}'
        table[m[1]] = m[2].strip()
    return elastic, boundary, intra, table


def work_contract(rt):
    bad = []
    members = parse_work_members(rt)
    mset = set(members)
    if len(mset) != len(members):
        bad.append('duplicate Work member names')
    elastic, boundary, intra, table = work_lists(rt)
    for nm, s in (('elastic', elastic), ('boundary', boundary), ('intra', intra), ('table', set(table))):
        if s - mset:
            bad.append(f'{nm} list names non-members {sorted(s - mset)}')
    cat = {}
    for m in members:
        hits = [c for c, s in (('elastic', elastic), ('boundary', boundary - elastic), ('intra', intra), ('table', set(table))) if m in s]
        if len(hits) != 1:
            bad.append(f'Work member {m}: in {hits or "no category"} (must be exactly one)')
        if hits:
            cat[m] = hits[0]
    if intra & (elastic | boundary):
        bad.append(f'intra park list repeats {sorted(intra & (elastic | boundary))}')
    for m, why in table.items():
        if not why:
            bad.append(f'{m}: no reason')
        a = re.match(r'^= (\w+)', why)
        if a and (a[1] not in mset or a[1] == m):
            bad.append(f'{m}: alias of unknown {a[1]}')
    # the boundary list's elastic members are not copied at intra points (decode uses the small layout) — the park must say so
    if 'if (intra && elastic_member(b)) return;' not in rt:
        bad.append('intra park copies elastic members (or the skip moved)')
    return bad, cat


PARK_NEED = [
    ('off path first', '  if (!ly_.on() || ly_.depth >= ly_.h.max_depth || capturing_) return;\n  const bool intra = kind != ly::kLayer;\n'
                       '  if (intra && !(ly_live_ && ly_.intra && w_->el.big)) return;'),
    ('unit closed', '    if (P.unit >= 0) unit_select(-1);'),
    ('pre-copies kept + hold', '      P.pf = pf_; P.pf_l = pf_l_; P.hold = store_.staging_held(); P.live = ly_live_;\n      if (!pf_.empty()) store_.set_staging_hold(true);\n      ly_live_ = false;'),
    ('deferred flush on unpark', '    drain();\n    deferred_flush();'),
    ('pre-copies restored', '    else { pf_ = std::move(P.pf); pf_l_ = P.pf_l; P.pf.clear(); store_.set_staging_hold(P.hold); ly_live_ = P.live; }'),
    ('unit reopened', '    if (P.unit >= 0) unit_select(P.unit);  // T11b (3)'),
    ('forward ly_live_', '  ly_live_ = ly_ok && ly_intra_;\n  const int tail_layer'),
    ('forward_multi ly_live_', '  ly_live_ = ly_ok && ly_intra_;\n  auto host_k'),
    ('quiesce discards deferred', '  def_pending_ = false; djobs_.clear();'),
    ('intra needs elastic', 'bool Runtime::ly_intra_ok() const { const Work::Elastic& E = w_->el; return ly_intra_ && E.on && E.complete && opt_.cpu_for_misses; }'),
    ('incomplete elastic (q/o) disables intra', '    else E.complete = false;  // T11b'),
]
HIVED_NEED = [
    ('want: decode only at intra points', '      if (rt.layer_yield_kind() != ly::kLayer) return active.empty() ? 0 : 1;'),
    ('run: no admission at intra points', '      const size_t cap = nested || kind != ly::kLayer ? 0 : std::min(std::max(ly_max, ly_mid), (size_t)std::max(0, R));'),
]


def park_contract(rt, hv):
    bad = [f'park/unpark: {what}' for what, s in PARK_NEED if s not in rt]
    bad += [f'hived: {what}' for what, s in HIVED_NEED if s not in hv]
    return bad


def contracts(ck):
    rt, st, hv = RT.read_text(), STORE.read_text(), HIVED.read_text()
    b = staging_contract(rt, st)
    ck.true('contract: staging writers known and gated', not b, '; '.join(b))
    b, cat = work_contract(rt)
    ck.true('contract: every Work member classified exactly once', not b, '; '.join(b[:8]))
    counts = {}
    for c in cat.values():
        counts[c] = counts.get(c, 0) + 1
    print(f'  Work members: {len(cat)} · {counts}', flush=True)
    b = park_contract(rt, hv)
    ck.true('contract: intra park/unpark rules', not b, '; '.join(b))
    # negative controls (each mutant must be detected)
    muts = []
    for name, gates in DECODE_WRITERS.items():
        for g in gates:
            muts.append((f'staging: {name} without {g.strip()[:40]!r}', 'stage', rt.replace(g, g.replace('held', 'h0ld').replace('staging_h0ld', 'staging_slots'), 1)))
    muts.append(('staging: a new writer function', 'stage', rt + '\nvoid Runtime::zz_new_writer() {\n  store_.copy_to_staging(0, 0, 0, side_);\n}\n'))
    muts.append(('staging: prefill writer called from decode', 'stage', rt.replace('void Runtime::bm_prefetch(int l2) {\n', 'void Runtime::bm_prefetch(int l2) {\n  prefetch_layer_experts(l2);\n', 1)))
    muts.append(('work: a new Work member', 'work', rt.replace('struct Runtime::Work {\n  int M = 0;\n', 'struct Runtime::Work {\n  int M = 0;\n  DevBuf scratch_new;\n', 1)))
    muts.append(('work: table entry removed (iscore)', 'work', re.sub(r'\n//   iscore: [^\n]*', '', rt, count=1)))
    muts.append(('work: iqp not parked at intra points', 'work', rt.replace(' iv(w.iqp, (size_t)c.index_n_heads * kvp::IDX_ROW);', '', 1)))
    muts.append(('work: a parked member also in the table', 'work', rt.replace('//   hflat: ', '//   pre_a: duplicate\n//   hflat: ', 1)))
    for what, s in PARK_NEED:
        muts.append((f'park: {what}', 'park', rt.replace(s, '', 1)))
    for desc, kind, m in muts:
        if m == rt:
            ck.failures.append(f'mutant "{desc}": did not change the source (contract text drifted)')
            continue
        if kind == 'stage':
            det = bool(staging_contract(m, st))
        elif kind == 'work':
            try:
                det = bool(work_contract(m)[0])
            except AssertionError:
                det = True
        else:
            det = bool(park_contract(m, hv))
        ck.checks += 1
        if not det:
            ck.failures.append(f'mutant "{desc}" was NOT detected')
    st_m = st.replace(STORE_GUARD, '')
    ck.true('mutant: store without the hold abort detected', bool(staging_contract(rt, st_m)))
    for what, s in HIVED_NEED:
        ck.true(f'mutant: hived {what} removed → detected', bool(park_contract(rt, hv.replace(s, '//'))))
    print(f'  contracts: {len(muts) + 1 + len(HIVED_NEED)} mutants', flush=True)


# ---- 3. real hived.cpp on the fake runtime ---------------------------------------------------------------------------------------------------------
LAYERS, FWD_MS, PERIOD, NI = 4, 1200, 100, 3   # 300 ms layers, 4 parts of 75 ms
BASE = {'FAKE_LAYERS': str(LAYERS), 'FAKE_PREFILL_MS': str(FWD_MS * SLOW), 'FAKE_PREFILL_SHORT_MS': str(20 * SLOW), 'FAKE_INTRA': str(NI),
        'HIVE_LAYER_YIELD': str(PERIOD), 'FAKE_STEP_MS': '5'}
ON = {**BASE, 'HIVE_LAYER_YIELD_INTRA': '1'}
ONE_LAYER = {**ON, 'FAKE_LAYERS': '1', 'FAKE_INTRA': '5'}  # a forward with intra points only (no layer boundary to admit at)
LY_RE = re.compile(r'^\[hived\] layer yield: (\S+) · prefill ([\d.]+) ms since resume · rows (\d+) · active (\d+)(.*)$')
ADMIT_RE = re.compile(r'^\[hived\] prefill yield: (\d+) short requests? admitted')


def ylines(h):
    return h.tail(1000000).splitlines()


def counts(h):
    intra = boundary = 0
    for l in ylines(h):
        m = LY_RE.match(l)
        if m:
            if ' · intra ' in m[5]:
                intra += 1
            else:
                boundary += 1
    return intra, boundary


def n_admits(h):
    return sum(int(m[1]) for m in (ADMIT_RE.match(l) for l in ylines(h)) if m)


def admissions_after_intra(h):
    last = None
    bad = 0
    for l in ylines(h):
        m = LY_RE.match(l)
        if m:
            last = ' · intra ' in m[5]
        elif ADMIT_RE.match(l) and last:
            bad += 1
    return bad


def sc_decoder(h, ck, tag, on, short=False, fail_expected=False, short_inside=None):
    """A running decoder + a one-forward long prompt (+ optionally a short request 0.25 s in). Returns the decoder's longest gap."""
    D = t.tokens(100, 1300)
    times, res = [], {}
    th = t.launch(h, 'D', D, 3000, res, 'D', on_token=lambda k: times.append(time.monotonic()))
    deadline = time.monotonic() + 10 * SLOW
    while len(times) < 5 and time.monotonic() < deadline:
        time.sleep(.01)
    L = t.tokens(700, 1301)
    t0 = time.monotonic()
    ths = [t.launch(h, 'L', L, 3, res, 'L')]
    S = t.tokens(60, 1302)
    if short:
        time.sleep(.25 * SLOW)
        ths.append(t.launch(h, 'S', S, 3, res, 'S'))
    for x in ths:
        x.join(60 * SLOW)
    t1 = time.monotonic()
    h.cancel('D')
    th.join(60 * SLOW)
    t.done_ok(ck, f'{tag}: long prompt', res['L'], L, 3, 0)
    if short:
        t.done_ok(ck, f'{tag}: short prompt', res['S'], S, 3, 0)
        if short_inside is not None:  # a one-layer forward has no boundary: the short request must wait for it (never admitted at an intra point)
            ck.true(f'{tag}: short request admitted inside the long forward == {short_inside}', (n_admits(h) > 0) == short_inside, f'{n_admits(h)} admissions')
    if fail_expected:
        ck.true(f'{tag}: decoder failed inside the intra yield', res['D']['error'] is not None and 'decode failed' in str(res['D']['error']), repr(res['D']['error']))
    else:
        ck.true(f'{tag}: decoder ok', res['D']['error'] is None, repr(res['D']['error']))
        ck.eq(f'{tag}: decoder tokens follow its oracle', res['D']['tokens'], t.oracle(D, len(res['D']['tokens'])))
    win = [x for x in times if t0 <= x <= t1]
    gaps = [b - a for a, b in zip([t0] + win, win + [t1])]
    return max(gaps) if gaps else t1 - t0


def daemon(exe, td, ck):
    gap = {}
    for name, env, fn in (
        ('intra-off', BASE, lambda h: gap.__setitem__('off', sc_decoder(h, ck, 'off', False, short=True))),
        ('intra-on', ON, lambda h: gap.__setitem__('on', sc_decoder(h, ck, 'on', True, short=True))),
        ('hold', {**ON, 'FAKE_PF': '1', 'FAKE_DECODE_STAGE': '1'}, lambda h: sc_decoder(h, ck, 'hold', True)),
        ('fail', {**ON, 'FAKE_PF': '1', 'FAKE_FAIL_DECODE_INTRA': '2'}, lambda h: sc_decoder(h, ck, 'fail', True, fail_expected=True)),
        ('batch', {**ON, **t.BATCH}, lambda h: sc_decoder(h, ck, 'batch', True, short=True)),
        ('one-layer', ONE_LAYER, lambda h: sc_decoder(h, ck, 'one-layer', True, short=True, short_inside=False)),
    ):
        h = t.Hived(exe, td, env, args=t.TILE3, name='lyi-' + name)
        try:
            fn(h)
            ni, nb = counts(h)
            on = 'HIVE_LAYER_YIELD_INTRA' in env
            if on:
                ck.true(f'{name}: intra yields happened', ni >= (1 if name == 'fail' else 2), f'{ni} intra · {nb} boundary')
            else:
                ck.eq(f'{name}: no intra yield with the switch off', ni, 0)
            ck.eq(f'{name}: admissions only at layer boundaries', admissions_after_intra(h), 0)
            if name == 'fail':  # the hold must be released: a fresh request after the failure runs (decode + prefill heads check the hold)
                X = t.tokens(150, 1310)
                t.done_ok(ck, 'fail: fresh request after the failure', h.generate('X', X, 3), X, 3, 0)
            ck.true(f'{name}: no fake-state error', 'fake state' not in h.tail(1000000), [l for l in ylines(h) if 'fake state' in l][:2])
            print(f'  {name}: intra {ni} · boundary {nb}', flush=True)
        finally:
            t.stop_clean(h, ck)
    if 'on' in gap and 'off' in gap:
        ck.true('decoder: longest gap shorter with intra points', gap['on'] < gap['off'] * .8, f"on {gap['on']:.3f}s off {gap['off']:.3f}s")
        print(f"  decoder longest gap: off {gap['off']:.3f}s · on {gap['on']:.3f}s", flush=True)


def daemon_mutants(exe, td, ck):
    # decode staging writer that ignores the hold → the real copy_to_staging aborts the daemon
    h = t.Hived(exe, td, {**ON, 'FAKE_PF': '1', 'FAKE_DECODE_STAGE': '1', 'FAKE_IGNORE_HOLD': '1'}, args=t.TILE3, name='lyi-mut-hold')
    c2 = t.Checker('mut')
    try:
        try:
            sc_decoder(h, c2, 'mut-hold', True)
        except Exception:  # noqa: BLE001
            pass
        log = h.tail(1000000)
    finally:
        try:
            h.stop()
        except Exception:  # noqa: BLE001
            pass
    ck.true('mutant: decode writer ignoring the hold → store abort', 'FATAL: staging slot' in log, log[-500:])
    # hived admitting at intra points (want scans the queue, run keeps its cap) → a prefill forward inside an intra yield
    src = HIVED.read_text()
    want_line, cap_line = HIVED_NEED[0][1], HIVED_NEED[1][1]
    assert src.count(want_line) == 1 and src.count(cap_line) == 1
    m = src.replace(want_line, '').replace(cap_line, '      const size_t cap = nested ? 0 : std::min(std::max(ly_max, ly_mid), (size_t)std::max(0, R));')
    exe2 = t.build_daemon(td, name='hived_mut_intra_admit', hived_source=m)
    h = t.Hived(exe2, td, ONE_LAYER, args=t.TILE3, name='lyi-mut-admit')
    c3 = t.Checker('mut')
    try:
        try:
            sc_decoder(h, c3, 'mut-admit', True, short=True, short_inside=False)
        except Exception:  # noqa: BLE001
            pass
        log = h.tail(1000000)
    finally:
        try:
            h.stop()
        except Exception:  # noqa: BLE001
            pass
    ck.true('mutant: hived admitting at intra points → detected', 'prefill forward inside an intra-layer yield' in log or bool(c3.failures), log[-500:])


def main():
    ck = t.Checker('layer-yield-intra')
    contracts(ck)
    with tempfile.TemporaryDirectory(prefix='hive-lyi-') as td:
        t0 = time.monotonic()
        unit(td, ck)
        print(f'  unit test + store negative control: {time.monotonic() - t0:.1f}s', flush=True)
        exe = t.build_daemon(td)
        daemon(exe, td, ck)
        if os.environ.get('LYI_MUTANT', '1') != '0':
            daemon_mutants(exe, td, ck)
    for f in ck.failures:
        print('FAIL', f)
    print(f't11b layer yield intra: {ck.checks} checks, {len(ck.failures)} failures')
    sys.exit(1 if ck.failures else 0)


if __name__ == '__main__':
    main()
