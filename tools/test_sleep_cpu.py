#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Z1 hive sleep/wake — CPU checks (no GPU, no service).

1. store   : engine/tests/test_sleep_store_cpu.cpp — the REAL ExpertStore sleep_release / wake_alloc / restore_keys on fake CUDA
             (FAKE_CUDA_ASYNC 0 and 1): plain / elastic (lent and reclaimed) / E4 paced stores, VRAM-short wakes, reserve fit; resident set,
             scores and record bytes after wake. The cache-event traces replay cleanly through tools/check_cache_events.py.
2. daemon  : the REAL engine/src/hived.cpp on the fake runtime (tools/cpu_fake — every fake forward aborts while asleep):
             sleep/wake idempotence · sleep with an in-flight request (it finishes first, new requests stay queued, served after wake) ·
             timeout_s → "not idle" and the sleep is withdrawn · wake withdraws a draining sleep · cancel of a queued request while asleep ·
             wake failure (VRAM short) stays asleep, the next wake works · concurrent wake + request · graceful stop while asleep ·
             session continuation after sleep/wake == the same requests on a daemon that never slept (tokens, cached_prefix, prefill_tokens) ·
             resident experts restored (count) · stats state/vram/sleep/wake fields.
3. server  : server/hive_server.py endpoints (/release_memory_occupation · /resume_memory_occupation · /admin/sleep · /admin/wake · /health
             state) — status mapping with a fake daemon, and the real Daemon.control against the real hived socket.
HIVE_TEST_SANITIZER=thread (under `setarch x86_64 -R`) race-checks the C++ parts. CPU limit: run with nice/taskset (see common rules).
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake')]
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import harness  # noqa: E402
import check_cache_events  # noqa: E402
import test_daemon_cpu as D  # noqa: E402

ROOT = harness.ROOT
SLOW = D.SLOW
FAILS = []


def check(what, cond, detail=''):
    if not cond:
        FAILS.append(f'{what} {detail}')
        print(f'FAIL {what} {detail}')


# ---------------------------------------------------------------------------------------------------------- 1. store
STORE = ROOT / 'engine/src/expert_store.cpp'
# negative controls: each mutant of the production sleep/wake text MUST fail the unit (synchronous fake CUDA)
MUTANTS = {
    'sleep keeps the slot table': ('  std::fill(slot_of_.begin(), slot_of_.end(), -1);\n', ''),
    'wake forgets the lent/reclaimed state': ('n_slots_ = sleep_lent_ ? n : n_base_;', 'n_slots_ = n;'),
    'sleep skips commit_all': ('  commit_all();  // E4: issue all pieces', '  // commit_all();  // E4: issue all pieces'),
    'restore_keys reseeds scores': ('    const int n = promote_keys(chunk, batch, side, batch);',
                                    '    for (int k : chunk) score_[(size_t)k] += 1.f;\n    const int n = promote_keys(chunk, batch, side, batch);'),
}


def build_store_unit(td, name, store, extra=()):
    return harness.build(td, name, '', [ROOT / 'engine/tests/test_sleep_store_cpu.cpp', store, ROOT / 'engine/src/cpu/expert_cpu.cpp',
                                        harness.FAKE / 'fake_checkpoint.cpp'], extra=['-I' + str(ROOT / 'engine/tests'), '-D_GLIBCXX_ASSERTIONS', *extra])


def store_mutants(td):
    text = STORE.read_text()
    for i, (name, (old, new)) in enumerate(MUTANTS.items()):
        if text.count(old) != 1:
            check(f'mutant anchor [{name}]', False, 'anchor drifted')
            continue
        src = Path(td) / f'store_mut{i}.cpp'
        src.write_text(text.replace(old, new))
        asyn = 'commit_all' in name  # a pre-commit copy writing into a released slot is only visible with async streams + ASAN (heap-use-after-free)
        exe = build_store_unit(td, f'sleep_store_mut{i}', src, extra=['-fsanitize=address'] if asyn else [])
        env = harness.tsan_env()
        env['FAKE_CUDA_ASYNC'] = '1' if asyn else '0'
        if asyn:
            env['FAKE_CUDA_DELAY_US'] = '200'
        p = subprocess.run([exe, '40', td], env=env, capture_output=True, text=True, timeout=900)
        print(f'  mutant [{name}]: {"DETECTED" if p.returncode != 0 else "NOT DETECTED"}')
        check(f'mutant [{name}] detected', p.returncode != 0)


def devmem_tests(td):
    exe = harness.build(td, 'devmem', '', [ROOT / 'engine/tests/test_devmem_cpu.cpp'], extra=['-D_GLIBCXX_ASSERTIONS'])
    for mode, env_add in (('on', {'HIVE_SLEEP_VMM': '1'}), ('prepin', {'HIVE_SLEEP_VMM': '1'}), ('off', {})):
        env = harness.tsan_env()
        env.pop('HIVE_SLEEP_VMM', None)
        env.update(env_add)
        p = subprocess.run([exe, mode], env=env, capture_output=True, text=True, timeout=600)
        out = p.stdout + p.stderr
        print(' | '.join(l for l in out.splitlines() if l.startswith(('devmem CPU', 'FAIL', 'WARNING: ThreadSanitizer', 'SUMMARY'))))
        check(f'devmem unit ({mode})', p.returncode == 0, out[-3000:])
    # negative controls: devmem.h mutants MUST fail the unit (a copy of engine/include with one header changed, searched first)
    import shutil
    hdr = (ROOT / 'engine/include/hive/devmem.h').read_text()
    # (name, mode, [(original, replacement)...]) — prepin mutants: allocates the shadow copy in advance but never attaches it; ignores keep; swaps in a populated copy while asleep
    for i, (name, mode, edits) in enumerate((
            ('restore skips the H2D copy', 'on', [('if (ext && cudaMemcpy((void*)va, r.shadow, ext, cudaMemcpyHostToDevice) != cudaSuccess)', 'if (false)')]),
            ('release skips the host copy', 'on', [('if (ext && cudaMemcpy(r.shadow, (void*)va, ext, cudaMemcpyDeviceToHost) != cudaSuccess)', 'if (false)')]),
            ('prepare never attaches', 'prepin', [('          used = true;\n', '')]),
            ('prepare ignores keep', 'prepin', [('if (!r.keep || !r.mapped || r.shadow) continue;', 'if (!r.mapped || r.shadow) continue;')]),
            ('prepare replaces shadows while asleep', 'prepin', [('    if (s.released) return rep;\n', ''),
                                                                 ('if (!r.keep || !r.mapped || r.shadow) continue;', 'if (!r.keep) continue;'),
                                                                 ('if (it != s.regions.end() && !s.released) {', 'if (it != s.regions.end()) {'),
                                                                 ('if (r.keep && r.mapped && !r.shadow && want >= std::min(r.used, r.size)) {',
                                                                  'if (r.keep) { if (r.shadow) { if (r.shadow_pinned) cudaFreeHost(r.shadow); else ::free(r.shadow); }')]))):
        if any(hdr.count(old) != 1 for old, _ in edits):
            check(f'devmem mutant anchor [{name}]', False, 'anchor drifted')
            continue
        text = hdr
        for old, new in edits:
            text = text.replace(old, new)
        inc = Path(td) / f'inc_mut{i}'
        shutil.copytree(ROOT / 'engine/include', inc, dirs_exist_ok=True)
        (inc / 'hive/devmem.h').write_text(text)
        mexe = harness.build(td, f'devmem_mut{i}', '', [ROOT / 'engine/tests/test_devmem_cpu.cpp'], extra=['-iquote', str(inc)])
        env = harness.tsan_env()
        env['HIVE_SLEEP_VMM'] = '1'
        p = subprocess.run([mexe, mode], env=env, capture_output=True, text=True, timeout=600)
        print(f'  devmem mutant [{name}]: {"DETECTED" if p.returncode != 0 else "NOT DETECTED"}')
        check(f'devmem mutant [{name}] detected', p.returncode != 0)


def store_tests(td):
    exe = build_store_unit(td, 'sleep_store', STORE)
    rounds = 60 if harness.sanitizer() == 'thread' else 150
    for asyn in ('0', '1'):
        for f in Path(td).glob('sleep-events*.csv'):
            f.unlink()
        env = harness.tsan_env()
        env['FAKE_CUDA_ASYNC'] = asyn
        if asyn == '1':
            env['FAKE_CUDA_DELAY_US'] = '30'  # so promotion copies are still in flight during sleep_release
        p = subprocess.run([exe, str(rounds), td], env=env, capture_output=True, text=True, timeout=900)
        out = p.stdout + p.stderr
        print(f'[async={asyn}] ' + ' | '.join(l for l in out.splitlines() if l.startswith(('sleep CPU', 'FAIL', 'WARNING: ThreadSanitizer', 'SUMMARY'))))
        check(f'store unit async={asyn}', p.returncode == 0, out[-3000:])
        for name in ('sleep-events.csv', 'sleep-events-paced.csv'):
            f = Path(td) / name
            if not f.exists():
                check(f'trace {name}', False, 'missing')
                continue
            try:
                res = check_cache_events.replay(f.read_text().splitlines(True))
                check(f'trace {name} has sleep/wake B events', res['counts'].get('B', 0) > 10, repr(res['counts']))
            except ValueError as e:
                check(f'trace {name} replays', False, str(e))


# ---------------------------------------------------------------------------------------------------------- 2. daemon
class Bg(threading.Thread):
    def __init__(self, fn, *a, **kw):
        super().__init__(daemon=True)
        self.fn, self.a, self.kw, self.res, self.t_done, self.exc = fn, a, kw, None, None, None
        self.start()

    def run(self):
        try:
            self.res = self.fn(*self.a, **self.kw)
        except Exception as e:  # noqa: BLE001
            self.exc = e
        self.t_done = time.monotonic()


_LIVE = []


class Hived(D.Hived):  # collects the daemon processes so none are left behind even if a test fails midway (mutants)
    def __init__(self, *args, **kwargs):
        super(Hived, self).__init__(*args, **kwargs)
        _LIVE.append(self)


def kill_live():
    for h in _LIVE:
        if h.proc.poll() is None:
            h.proc.kill()
            h.proc.wait()
    _LIVE.clear()


def wait_state(h, want, timeout=10):
    deadline = time.monotonic() + timeout * SLOW
    seen = []
    while time.monotonic() < deadline:
        st = h.stats().get('state')
        if not seen or seen[-1] != st:
            seen.append(st)
        if st == want:
            return seen
        time.sleep(.005)
    return seen


def ok_tokens(what, r, ids, n):
    check(f'{what}: no error', r is not None and r['error'] is None and r['done'] is not None, repr(r and r['error']))
    if r and r['done']:
        check(f'{what}: tokens (state oracle)', r['tokens'] == D.oracle(ids, n), f"{r['tokens'][:5]}…")


def daemon_tests(td, exe=None):
    exe = exe or D.build_daemon(td)
    args = ('--vram-cache-mb', '8')
    env = {'FAKE_PLACE': '1'}
    A = D.tokens(300, 1)
    B = D.tokens(40, 2)

    # (a) session continuation: a daemon that slept and woke == a daemon that never slept
    def convo(h, sleep_between, level=1, upgrade=False):
        r1 = h.generate('s1', A, 12)
        st0 = h.stats()
        info = {}
        if sleep_between:
            info['sleep'] = h.op({'op': 'sleep', 'level': 1 if upgrade else level})
            info['stats_asleep'] = h.stats()
            info['sleep2'] = h.op({'op': 'sleep', 'level': level})
            info['wake'] = h.op({'op': 'wake'})
            info['wake2'] = h.op({'op': 'wake'})
            info['stats_awake'] = h.stats()
        ids2 = A + r1['tokens'] + B
        r2 = h.generate('s1', ids2, 12)
        return r1, r2, st0, info, ids2

    h0 = Hived(exe, td, env=env, args=args, name='nosleep')
    c0 = convo(h0, False)
    h0.stop()
    h = Hived(exe, td, env=env, args=args, name='sleep')
    r1, r2, st0, info, ids2 = convo(h, True)
    ok_tokens('(a) turn 1', r1, A, 12)
    ok_tokens('(a) turn 2 after sleep/wake', r2, ids2, 12)
    for k in ('cached_prefix', 'prefill_tokens', 'n', 'finish'):
        check(f'(a) turn 2 {k} == no-sleep daemon', r2['done'] and c0[1]['done'] and r2['done'][k] == c0[1]['done'][k],
              f"{r2['done'] and r2['done'][k]} vs {c0[1]['done'] and c0[1]['done'][k]}")
    check('(a) turn 2 tokens == no-sleep daemon', r2['tokens'] == c0[1]['tokens'])
    check('(a) prefix reused after sleep', (r2['done'] or {}).get('cached_prefix', 0) >= len(A), repr(r2['done']))
    s = info['sleep']
    check('(a) sleep reply', s.get('ok') is True and s.get('state') == 'sleeping', repr(s))
    res0 = st0.get('unique_resident', 0)
    check('(a) residents before sleep (FAKE_PLACE)', res0 > 0, repr(st0.get('unique_resident')))
    check('(a) sleep saved the resident keys', s.get('sleep', {}).get('resident_keys') == res0, repr(s.get('sleep')))
    sa = info['stats_asleep']
    check('(a) stats while asleep', sa.get('state') == 'sleeping' and sa.get('slots') == 0 and sa.get('resident') == 0 and 'vram' in sa and 'sleep' in sa, repr(
        {k: sa.get(k) for k in ('state', 'slots', 'resident', 'vram')}))
    kept = s.get('sleep', {}).get('vram_kept', {})
    check('(a) vram_kept table', set(kept) == {'model_weights_and_cuda_context_mib', 'session_pool_mib', 'runtime_decode_buffers_and_rest_mib'}, repr(kept))
    check('(a) sleep idempotent', info['sleep2'].get('ok') is True and info['sleep2'].get('state') == 'sleeping', repr(info['sleep2']))
    w = info['wake']
    check('(a) wake reply', w.get('ok') is True and w.get('state') == 'ready' and w.get('wake', {}).get('warmed') == res0, repr(w))
    check('(a) wake idempotent', info['wake2'].get('ok') is True and info['wake2'].get('state') == 'ready', repr(info['wake2']))
    aw = info['stats_awake']
    check('(a) residents restored', aw.get('unique_resident') == res0 and aw.get('slots') == st0.get('slots') and aw.get('state') == 'ready',
          f"{aw.get('unique_resident')}/{aw.get('slots')} vs {res0}/{st0.get('slots')}")

    # (a2) level 2 (session KV to RAM as well) — straight to deep sleep, and escalating from light to deep sleep: continuation must equal the never-slept daemon
    for name, up in (('deep', False), ('deep-upgrade', True)):
        h2 = Hived(exe, td, env=env, args=args, name=name)
        q1, q2, _, inf2, ids_q2 = convo(h2, True, level=2, upgrade=up)
        sl2 = inf2['sleep2']['sleep']
        check(f'({name}) level 2 report', sl2.get('level') == 2 and sl2.get('sessions_offloaded') == 1 and sl2.get('vram_kept', {}).get('session_pool_mib') == 0
              and 'VmRSS_mib' in sl2.get('host_memory', {}) and 'experts_pinned_mib' in sl2.get('host_memory', {}) and 'engram_mib' in sl2.get('host_memory', {}), repr(sl2))
        if up:
            check(f'({name}) first sleep was level 1', inf2['sleep'].get('sleep', {}).get('level') == 1, repr(inf2['sleep']))
        ok_tokens(f'({name}) turn 2 after deep sleep/wake', q2, ids_q2, 12)
        for k in ('cached_prefix', 'prefill_tokens'):
            check(f'({name}) turn 2 {k} == no-sleep daemon', q2['done'] and c0[1]['done'] and q2['done'][k] == c0[1]['done'][k],
                  f"{q2['done'] and q2['done'][k]} vs {c0[1]['done'] and c0[1]['done'][k]}")
        check(f'({name}) turn 2 tokens == no-sleep daemon', q2['tokens'] == c0[1]['tokens'])
        check(f'({name}) wake after deep sleep', inf2['wake'].get('ok') is True and inf2['stats_awake'].get('state') == 'ready', repr(inf2['wake']))
        ids_n = D.tokens(80, 21)
        ok_tokens(f'({name}) new session after deep wake', h2.generate('s-new', ids_n, 6), ids_n, 6)
        h2.alive()
        h2.stop()

    # (a3) level 3 (VMM — started with HIVE_SLEEP_VMM=1): directly and escalating 1->3 — continuation == never-slept daemon, report (vmm released/restored); level 3 on a daemon without VMM is absorbed as 2
    for name, up in (('vmm', False), ('vmm-upgrade', True)):
        h3 = Hived(exe, td, env={**env, 'HIVE_SLEEP_VMM': '1'}, args=args, name=name)
        q1, q2, _, inf3, ids_q3 = convo(h3, True, level=3, upgrade=up)
        sl3 = inf3['sleep2']['sleep']
        check(f'({name}) level 3 report', sl3.get('level') == 3 and sl3.get('vmm', {}).get('released_mib', 0) > 0 and 'error' not in sl3.get('vmm', {})
              and sl3.get('host_memory', {}).get('dense_host_copy_mib', 0) > 0 and 'note' not in sl3, repr(sl3))
        check(f'({name}) wake restored VMM', inf3['wake'].get('wake', {}).get('vmm', {}).get('restored_mib', 0) > 0 and inf3['wake'].get('wake', {}).get('level') == 3,
              repr(inf3['wake']))
        ok_tokens(f'({name}) turn 2 after level-3 sleep/wake', q2, ids_q3, 12)
        check(f'({name}) turn 2 == no-sleep daemon', q2['tokens'] == c0[1]['tokens'] and q2['done'] and q2['done']['cached_prefix'] == c0[1]['done']['cached_prefix'])
        ok_tokens(f'({name}) turn 1 with VMM device memory (no sleep yet)', q1, A, 12)
        h3.alive()
        h3.stop()
    r3 = h.op({'op': 'sleep', 'level': 3})
    check('(a3) level 3 without VMM → asleep at level 2 with a note', r3.get('ok') is True and r3.get('sleep', {}).get('level') == 2
          and 'HIVE_SLEEP_VMM' in r3.get('sleep', {}).get('note', ''), repr(r3))
    check('(a3) wake from absorbed level 2', h.op({'op': 'wake'}).get('state') == 'ready')

    # (sa) start asleep (HIVE_START_ASLEEP): ready = sleeping (level 3, 0 slots); requests are queued; the first wake allocates the slots for the first time and warms them from the cache-state list
    import struct
    keys = list(range(10))
    for name, extra_env, want_slots in (('start-asleep', {}, None), ('start-asleep-fit', {'HIVE_CACHE_FIT': '1', 'HIVE_CACHE_RESERVE_MB': str(65536 - 64 - 8)}, None)):
        cs = Path(td) / f'cache-state-{name}.bin'  # the daemon overwrites this file when it goes to sleep — one per variant
        cs.write_bytes(struct.pack('<4i', 0x48564353, 2, 8, len(keys)) + struct.pack(f'<{len(keys)}i', *keys))
        hs = Hived(exe, td, env={**env, 'HIVE_START_ASLEEP': '1', **extra_env}, args=(*args, '--cache-state', str(cs)), name=name)
        st = hs.stats()
        si = st.get('sleep', {})
        check(f'({name}) ready asleep', st.get('state') == 'sleeping' and st.get('slots') == 0 and si.get('started_asleep') is True and si.get('level') == 3
              and si.get('resident_keys') == len(keys), repr({k: st.get(k) for k in ('state', 'slots', 'sleep')}))
        ids_s = D.tokens(60, 40)
        q = Bg(hs.generate, 'ss', ids_s, 6)
        time.sleep(.3 * SLOW)
        check(f'({name}) request waits while started asleep', q.is_alive())
        wk = hs.op({'op': 'wake'})
        q.join(30 * SLOW)
        aw = hs.stats()
        check(f'({name}) first wake', wk.get('ok') is True and aw.get('state') == 'ready' and aw.get('slots', 0) > 0 and wk.get('wake', {}).get('warmed') == len(keys)
              and aw.get('unique_resident', 0) >= len(keys) and wk.get('wake', {}).get('vmm', {}).get('restored_mib', 0) > 0, repr((wk, {k: aw.get(k) for k in ('slots', 'unique_resident')})))
        ok_tokens(f'({name}) request served after the first wake', q.res, ids_s, 6)
        r2 = hs.op({'op': 'sleep', 'level': 3})
        check(f'({name}) normal sleep after a started-asleep wake', r2.get('sleep', {}).get('level') == 3 and r2.get('sleep', {}).get('slots_released') == aw.get('slots'),
              repr(r2))
        w2 = hs.op({'op': 'wake'})
        check(f'({name}) wake again', w2.get('state') == 'ready' and w2.get('wake', {}).get('warmed') == r2.get('sleep', {}).get('resident_keys', -1) > 0, repr(w2))
        ids_t = D.tokens(40, 41)
        ok_tokens(f'({name}) serving after the second wake', hs.generate('st', ids_t, 5), ids_t, 5)
        hs.alive()
        hs.stop()

    # (b) sleep with an in-flight request: that request runs to completion, the sleep reply comes after it, requests arriving meanwhile are queued -> served after wake
    ids_b = D.tokens(200, 3)
    first = threading.Event()
    g = Bg(h.generate, 'sb', ids_b, 120, on_token=lambda n: first.set())
    first.wait(20 * SLOW)
    sl = Bg(h.op, {'op': 'sleep'})
    seen = wait_state(h, 'draining')
    check('(b) draining observed', 'draining' in seen, repr(seen))
    ids_q = D.tokens(50, 4)
    q = Bg(h.generate, 'sq', ids_q, 6)
    qc = Bg(h.generate, 'sqc', D.tokens(40, 12), 4)  # cancel from the queue during sleep -> immediately done cancel (does not wait for the wake)
    time.sleep(.05 * SLOW)
    h.cancel('sqc')
    qc.join(5 * SLOW)
    check('(b) queued request cancelled while draining/asleep → done cancel', qc.res and qc.res['done'] and qc.res['done']['finish'] == 'cancel'
          and qc.res['tokens'] == [], repr(qc.res))
    sl.join(60 * SLOW)
    g.join(60 * SLOW)
    ok_tokens('(b) in-flight request completed', g.res, ids_b, 120)
    check('(b) in-flight finish length (not cut)', g.res and g.res['done'] and g.res['done']['finish'] == 'length' and len(g.res['tokens']) == 120)
    check('(b) sleep reply after the in-flight request', sl.res and sl.res.get('ok') and sl.t_done >= g.t_done - 0.05, repr(sl.res))
    time.sleep(.3 * SLOW)
    check('(b) queued request not served while asleep', q.is_alive() and h.stats().get('state') == 'sleeping')
    check('(b) pending request counted', h.stats().get('pending_requests') == 1, repr(h.stats().get('pending_requests')))
    wk = h.op({'op': 'wake'})
    q.join(30 * SLOW)
    check('(b) wake', wk.get('ok') is True, repr(wk))
    ok_tokens('(b) queued request served after wake', q.res, ids_q, 6)
    check('(b) queued request finished after the wake reply', q.t_done is not None)

    # (c) timeout_s -> not idle, sleep withdrawn, in-flight request unaffected
    first.clear()
    ids_c = D.tokens(150, 5)
    g = Bg(h.generate, 'sc', ids_c, 150, on_token=lambda n: first.set())
    first.wait(20 * SLOW)
    rc = h.op({'op': 'sleep', 'timeout_s': 0.05})
    check('(c) not idle', rc.get('error') == 'not idle' and rc.get('state') == 'ready' and rc.get('retryable') is True and rc.get('running', 0) >= 1, repr(rc))
    rbad = Bg(h.op, {'op': 'sleep', 'timeout_s': 'soon'})  # non-number = no deadline (normalized) — sleeps after the in-flight request finishes
    g.join(60 * SLOW)
    rbad.join(60 * SLOW)
    ok_tokens('(c) request unaffected', g.res, ids_c, 150)
    check('(c) non-numeric timeout normalised (waits, then sleeps)', rbad.res and rbad.res.get('ok') is True and rbad.res.get('state') == 'sleeping', repr(rbad.res))
    check('(c) wake', h.op({'op': 'wake'}).get('state') == 'ready')

    # (d) a wake withdraws an in-progress sleep
    first.clear()
    ids_d = D.tokens(150, 6)
    g = Bg(h.generate, 'sd', ids_d, 150, on_token=lambda n: first.set())
    first.wait(20 * SLOW)
    sl = Bg(h.op, {'op': 'sleep'})
    wait_state(h, 'draining')
    wk = h.op({'op': 'wake'})
    sl.join(20 * SLOW)
    check('(d) wake while draining', wk.get('ok') is True and wk.get('state') == 'ready', repr(wk))
    check('(d) sleep withdrawn', sl.res and 'sleep cancelled' in str(sl.res.get('error')), repr(sl.res))
    g.join(60 * SLOW)
    ok_tokens('(d) request unaffected', g.res, ids_d, 150)
    check('(d) state ready', h.stats().get('state') == 'ready')

    # (e) cancel of a queued request while asleep
    check('(e) sleep', h.op({'op': 'sleep'}).get('state') == 'sleeping')
    q = Bg(h.generate, 'se', D.tokens(30, 7), 5)
    time.sleep(.2 * SLOW)
    h.cancel('se')
    q.join(5 * SLOW)
    check('(e) cancelled while asleep → done cancel', q.res and q.res['done'] and q.res['done']['finish'] == 'cancel' and q.res['tokens'] == [], repr(q.res))
    check('(e) wake', h.op({'op': 'wake'}).get('state') == 'ready')
    ids_e = D.tokens(30, 8)
    ok_tokens('(e) new request after wake', h.generate('se', ids_e, 5), ids_e, 5)
    h.alive()
    h.stop()

    # (f) wake failure (VRAM short) -> stays asleep, the next wake succeeds; (g) wake concurrent with a request
    h = Hived(exe, td, env={**env, 'FAKE_WAKE_FAIL': '1', 'FAKE_WAKE_MS': '60'}, args=args, name='wakefail')
    ids_f = D.tokens(100, 9)
    ok_tokens('(f) warm-up request', h.generate('sf', ids_f, 6), ids_f, 6)
    check('(f) sleep', h.op({'op': 'sleep'}).get('state') == 'sleeping')
    wf = h.op({'op': 'wake'})
    check('(f) wake failure stays asleep', 'wake failed' in str(wf.get('error')) and wf.get('state') == 'sleeping' and wf.get('retryable') is True, repr(wf))
    check('(f) stats after failure', h.stats().get('state') == 'sleeping' and h.stats().get('wake', {}).get('phase') == 'failed')
    ids_g = D.tokens(60, 10)
    q = Bg(h.generate, 'sg', ids_g, 6)
    time.sleep(.05)
    wk = Bg(h.op, {'op': 'wake'})
    seen = wait_state(h, 'ready')
    wk.join(20 * SLOW)
    q.join(30 * SLOW)
    check('(g) waking observed', 'waking' in seen, repr(seen))
    check('(g) wake ok', wk.res and wk.res.get('ok') is True, repr(wk.res))
    ok_tokens('(g) request sent before wake', q.res, ids_g, 6)
    check('(g) request finished after the wake', q.t_done is not None and wk.t_done is not None and q.t_done >= wk.t_done - 0.05)
    h.alive()
    h.stop()

    # (h) graceful stop (HIVE_GRACEFUL_STOP_S) of a sleeping daemon: queued requests = shutdown error, exit 0
    h = Hived(exe, td, env={**env, 'HIVE_GRACEFUL_STOP_S': '5'}, args=args, name='stopasleep')
    check('(h) sleep', h.op({'op': 'sleep'}).get('state') == 'sleeping')
    q = Bg(h.generate, 'sh', D.tokens(20, 11), 4)
    time.sleep(.2 * SLOW)
    rc = h.stop()
    q.join(10 * SLOW)
    check('(h) graceful stop while asleep exits 0', rc == 0, repr(rc))
    check('(h) queued request gets the shutdown error', q.res and 'shutting down' in str(q.res['error']), repr(q.res))

    # (i) the server's Daemon.control against a real hived socket
    h = Hived(exe, td, env=env, args=args, name='server')
    with h.connect() as so:  # context overflow: reason string unchanged + the numbers the server uses to build the 400 (max_ctx, prompt_tokens); max_ctx in stats
        so.sendall((json.dumps({'op': 'generate', 'session': 'ov', 'ids': D.tokens(4096, 30), 'max_tokens': 4}) + '\n').encode())
        ov = json.loads(so.makefile().readline())
    check('(i) daemon overflow reason + numbers', ov == {'error': 'context overflow', 'max_ctx': 4096, 'prompt_tokens': 4096}, repr(ov))
    check('(i) stats max_ctx', h.stats().get('max_ctx') == 4096)
    s = load_server()
    dm = s.Daemon(h.sock)
    out = asyncio.run(_ctl_roundtrip(dm))
    check('(i) Daemon.control sleep', out[0].get('state') == 'sleeping', repr(out[0]))
    check('(i) Daemon.control wake', out[1].get('state') == 'ready', repr(out[1]))
    h.alive()
    h.stop()
    print('daemon: sleep/wake state machine checks done')


HIVED = ROOT / 'engine/src/hived.cpp'
SLEEP_WAIT = ('      cv.wait_for(lk, std::chrono::seconds(1), [&] { return !ctl_q.empty() || stop_state.load() != 0; });'
              '  // sweep cancelled queued requests every second\n')
# negative controls (HIVE_SLEEP_NEGATIVE=1 — each rebuilds hived): every mutant of the production state machine MUST fail daemon_tests
DAEMON_MUTANTS = {
    'sleep without draining': ('        if (active.empty()) { do_sleep(); continue; }\n', '        { do_sleep(); continue; }\n'),
    # both layers (the continue in the asleep branch and the sleep_asked admission gate) must be removed for it to admit — a mutant removing only one layer behaves identically (measured: not detected = equivalent)
    'admit while asleep': [(SLEEP_WAIT + '      continue;\n', SLEEP_WAIT),
                           ('sleep_asked.store(power.load() != kPowerReady || pending);', 'sleep_asked.store(pending);')],
    'wake without warm': ('      warmed = rt.wake_warm(sleep_keys,', '      if (0) warmed = rt.wake_warm(sleep_keys,'),
    'no cancel sweep while asleep': ('      sweep_cancelled_queue();\n', ''),
}


def daemon_mutants(td):
    text = HIVED.read_text()
    for i, (name, pairs) in enumerate(DAEMON_MUTANTS.items()):
        pairs = pairs if isinstance(pairs, list) else [pairs]
        if any(text.count(old) != 1 for old, _ in pairs):
            check(f'daemon mutant anchor [{name}]', False, 'anchor drifted')
            continue
        src = text
        for old, new in pairs:
            src = src.replace(old, new)
        exe = D.build_daemon(td, name=f'hived_mut{i}', hived_source=src)
        n0 = len(FAILS)
        try:
            daemon_tests(td, exe)
        except Exception as e:  # noqa: BLE001 — a dead daemon (fake forward abort) also counts as detection
            FAILS.append(f'exception {e!r}'[:200])
        kill_live()
        detected = len(FAILS) > n0
        del FAILS[n0:]
        print(f'  daemon mutant [{name}]: {"DETECTED" if detected else "NOT DETECTED"}')
        check(f'daemon mutant [{name}] detected', detected)


async def _ctl_roundtrip(dm):
    a = await dm.control('sleep')
    b = await dm.control('wake')
    return a, b


# ---------------------------------------------------------------------------------------------------------- 3. server
def load_server():
    spec = importlib.util.spec_from_file_location('server_sleep_test', ROOT / 'server/hive_server.py')
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    return s


class FakeReq:
    def __init__(self, body=None, raw=False): self.body, self.raw = body, raw

    async def json(self):
        if self.raw:
            raise ValueError('not json')
        return self.body


class FakeDaemon:
    def __init__(self, replies, stats=None, fail=None): self.replies, self.calls, self._stats, self.fail = list(replies), [], stats, fail

    async def control(self, op, **kw):
        self.calls.append((op, kw))
        if self.fail:
            raise self.fail
        return self.replies.pop(0)

    async def stats(self):
        if self._stats is None:
            raise FileNotFoundError('no socket')
        return self._stats


def server_tests():
    s = load_server()

    async def run():
        out = {}
        s.DAEMON = FakeDaemon([{'ok': True, 'state': 'sleeping'}])
        out['sleep_ok'] = (await s.release_memory_occupation(FakeReq({})), s.DAEMON.calls)
        s.DAEMON = FakeDaemon([{'error': 'not idle', 'state': 'ready', 'running': 2, 'retryable': True}])
        out['not_idle'] = (await s.release_memory_occupation(FakeReq({'timeout_s': 3})), s.DAEMON.calls)
        s.DAEMON = FakeDaemon([{'ok': True, 'state': 'sleeping'}])
        await s.release_memory_occupation(FakeReq({'level': 2}))
        out['level2'] = s.DAEMON.calls
        s.DAEMON = FakeDaemon([{'ok': True, 'state': 'sleeping'}])
        await s.release_memory_occupation(FakeReq({'level': 3.0}))
        out['level3'] = s.DAEMON.calls
        for name, body in (('bool', {'timeout_s': True, 'level': True}), ('neg', {'timeout_s': -1, 'level': 0}), ('str', {'timeout_s': '5', 'level': '2'}),
                           ('raw', None)):
            s.DAEMON = FakeDaemon([{'ok': True, 'state': 'sleeping'}])
            await s.release_memory_occupation(FakeReq(body, raw=body is None))
            out['norm_' + name] = s.DAEMON.calls
        s.DAEMON = FakeDaemon([{'error': 'wake failed: not enough free VRAM', 'state': 'sleeping', 'retryable': True}])
        out['wake_fail'] = await s.resume_memory_occupation(FakeReq({}))
        s.DAEMON = FakeDaemon([{'error': 'sleep cancelled by a wake request', 'state': 'ready'}])
        out['cancelled'] = await s.release_memory_occupation(FakeReq({}))
        s.DAEMON = FakeDaemon([], fail=FileNotFoundError('missing'))
        out['loading'] = await s.resume_memory_occupation(FakeReq({}))
        s.DAEMON = FakeDaemon([], stats={'state': 'sleeping', 'vram': {'used_mib': 1.0}})
        out['health'] = await s.health()
        s.DAEMON = FakeDaemon([], stats=None)
        out['health_loading'] = await s.health()
        return out

    o = asyncio.run(run())
    r, calls = o['sleep_ok']
    check('server: sleep ok → 200 body', r == {'ok': True, 'state': 'sleeping'} and calls == [('sleep', {})], repr((r, calls)))
    r, calls = o['not_idle']
    check('server: not idle → 400', getattr(r, 'status_code', None) == 400 and calls == [('sleep', {'timeout_s': 3.0})], repr(calls))
    check('server: level 2 passed through', o['level2'] == [('sleep', {'level': 2})], repr(o['level2']))
    check('server: level 3 passed through', o['level3'] == [('sleep', {'level': 3})], repr(o['level3']))
    for name in ('bool', 'neg', 'str', 'raw'):
        check(f'server: timeout_s {name} normalised away', o['norm_' + name] == [('sleep', {})], repr(o['norm_' + name]))
    check('server: wake failed → 503', getattr(o['wake_fail'], 'status_code', None) == 503)
    check('server: sleep cancelled → 409', getattr(o['cancelled'], 'status_code', None) == 409)
    lo = o['loading']
    check('server: daemon missing → 503 loading', getattr(lo, 'status_code', None) == 503 and json.loads(lo.body).get('state') == 'loading')
    hh = o['health']
    check('server: /health state', hh.get('state') == 'sleeping' and hh.get('vram') == {'used_mib': 1.0} and hh.get('ok') is True, repr(hh))
    hl = o['health_loading']
    check('server: /health loading', getattr(hl, 'status_code', None) == 503 and json.loads(hl.body).get('state') == 'loading')
    # Request logging: one JSONL line per request — receive / first token / end times, endpoint, status, stream, max_tokens, session, rid, tags, done
    with tempfile.TemporaryDirectory() as ld:
        logp = Path(ld) / 'req.jsonl'
        os.environ['HIVE_REQUEST_LOG'] = str(logp)
        try:
            class Tok:
                eos_token_id, eos_token = 99, '<EOS>'
                def decode(self, ids, **kw): return ''.join({1: 'a', 2: 'b', 99: '<EOS>'}[i] for i in ids)
            class Enc:
                def parse_message_from_completion_text(self, text, **kw): return {'content': text.replace('<EOS>', '')}
            class GenDaemon:
                async def generate(self, session, *a, **kw):
                    for i in (1, 2):
                        yield {'id': i}
                    yield {'done': True, 'n': 2, 'finish': 'length', 'prefill_ms': 1.0}
                def cancel(self, session, rid=''): pass
            class Hdr(FakeReq):
                headers = {'X-Hive-Feature': 'chat', 'X-Gw-Route': 'r1', 'User-Agent': 'client/1', 'Authorization': 'secret'}
            s.TOK, s.ENC, s.DAEMON = Tok(), Enc(), GenDaemon()
            s.encode_prompt = lambda *a: ('prompt', [])
            s.tokenize_with_images = lambda *a: ([10, 11], [], b'')
            asyncio.run(s.chat(Hdr({'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 2, 'user': 'u1', 'metadata': {'alias': 'chat'}})))
            recs = [json.loads(l) for l in logp.read_text().splitlines()]
            r = recs[0] if recs else {}
            check('server log: one JSONL record', len(recs) == 1, repr(recs))
            check('server log: fields', r.get('endpoint') == '/v1/chat/completions' and r.get('status') == 200 and r.get('stream') is False and r.get('max_tokens') == 2
                  and r.get('t_recv') and r.get('t_first') and r.get('t_done') and r['t_recv'] <= r['t_first'] <= r['t_done'] and r.get('session') and r.get('rid')
                  and (r.get('done') or {}).get('finish') == 'length' and r.get('outcome') == 'done', repr(r))
            c = r.get('client', {})
            check('server log: client tags (no secrets)', c.get('x-hive-feature') == 'chat' and c.get('user') == 'u1' and c.get('metadata') == {'alias': 'chat'}
                  and c.get('user-agent') == 'client/1' and 'authorization' not in c and 'x-gw-route' not in c, repr(c))
            # HIVE_CLIENT_TAG_PREFIXES: comma-separated, case-insensitive, replaces the default list (unset/empty = x-hive,x-client)
            os.environ['HIVE_CLIENT_TAG_PREFIXES'] = ' X-Gw , x-client'
            try:
                c2 = s.client_tags(Hdr({}), {})
                check('server log: HIVE_CLIENT_TAG_PREFIXES', c2.get('x-gw-route') == 'r1' and 'x-hive-feature' not in c2 and c2.get('user-agent') == 'client/1'
                      and 'authorization' not in c2, repr(c2))
                os.environ['HIVE_CLIENT_TAG_PREFIXES'] = ''
                c3 = s.client_tags(Hdr({}), {})
                check('server log: empty HIVE_CLIENT_TAG_PREFIXES = default', c3.get('x-hive-feature') == 'chat' and 'x-gw-route' not in c3, repr(c3))
            finally:
                os.environ.pop('HIVE_CLIENT_TAG_PREFIXES', None)
        finally:
            os.environ.pop('HIVE_REQUEST_LOG', None)
    # context overflow = 400 + OpenAI context_length_exceeded (wording that matches the overflow phrase and input-token regex clients look for)
    import re
    class OvDaemon:
        def __init__(self, stats, err=None): self._s, self.err, self.gen = stats, err, 0
        async def stats(self): return self._s
        async def generate(self, session, *a, **kw):
            self.gen += 1
            yield self.err or {'done': True, 'n': 0, 'finish': 'length'}
        def cancel(self, session, rid=''): pass
    s.tokenize_with_images = lambda *a: (list(range(10)), [], b'')
    for stream in (False, True):
        s._MAX_CTX = None
        s.DAEMON = OvDaemon({'max_ctx': 10})
        r = asyncio.run(s.chat(FakeReq({'messages': [{'role': 'user', 'content': 'x'}], 'stream': stream})))
        body = json.loads(r.body) if hasattr(r, 'body') else {}
        msg = body.get('error', {}).get('message', '')
        check(f'overflow pre-check (stream={stream}): 400 before the stream', getattr(r, 'status_code', None) == 400 and s.DAEMON.gen == 0
              and body['error'].get('code') == 'context_length_exceeded' and body['error'].get('type') == 'invalid_request_error', repr(body))
        check('overflow text matches client overflow markers', 'maximum context length' in msg.lower() and
              re.search(r"prompt contains at least (\d+) input tokens", msg).group(1) == '10' and 'is 10 tokens' in msg, msg)
    s._MAX_CTX = None
    s.DAEMON = OvDaemon({}, {'error': 'context overflow', 'max_ctx': 8, 'prompt_tokens': 10})
    r = asyncio.run(s.chat(FakeReq({'messages': [{'role': 'user', 'content': 'x'}]})))
    body = json.loads(r.body) if hasattr(r, 'body') else {}
    check('overflow from the daemon reason → 400', getattr(r, 'status_code', None) == 400 and body.get('error', {}).get('code') == 'context_length_exceeded'
          and 'is 8 tokens' in body['error']['message'], repr(body))
    s._MAX_CTX = None
    s.DAEMON = OvDaemon({}, {'error': 'model incomplete (partial checkpoint)'})
    r = asyncio.run(s.chat(FakeReq({'messages': [{'role': 'user', 'content': 'x'}]})))
    check('other daemon errors stay 502', getattr(r, 'status_code', None) == 502)
    s._MAX_CTX = None
    paths = {(r.path, tuple(sorted(r.methods))) for r in s.app.routes if hasattr(r, 'methods')}
    for path in ('/release_memory_occupation', '/admin/sleep', '/resume_memory_occupation', '/admin/wake'):
        check(f'server: route POST {path}', (path, ('POST',)) in paths)
    print('server: endpoint checks done')


def main():
    only = [x for x in os.environ.get('HIVE_SLEEP_TESTS', 'devmem,store,daemon,server').split(',') if x]
    with tempfile.TemporaryDirectory(prefix='hive-sleep-test-') as td:
        if 'devmem' in only:
            devmem_tests(td)
        if 'store' in only:
            store_tests(td)
            if harness.sanitizer() != 'thread':
                store_mutants(td)
        if 'daemon' in only:
            daemon_tests(td)
            if os.environ.get('HIVE_SLEEP_NEGATIVE'):
                daemon_mutants(td)
    if 'server' in only:
        server_tests()
    if FAILS:
        print(f'sleep CPU: {len(FAILS)} failure(s)')
        sys.exit(1)
    print('sleep CPU: all OK (' + ','.join(only) + ')')


if __name__ == '__main__':
    main()
