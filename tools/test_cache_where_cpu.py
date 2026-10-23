#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""{"op":"where"} and {"op":"load"} on the REAL engine/src/hived.cpp (fake CUDA/runtime, tools/test_daemon_cpu.py harness), with
HIVE_SESSION_STORE on.

Checked, per case of the contract (docs/architecture.md, "Session cache location"):
  * none       — a session the daemon never saw: where none, tokens 0; load answers loaded false ("no stored checkpoint");
  * vram       — a resident session: where vram, tokens = its live token history (prompt + fed-back output); its checkpoint is listed under
                 tiers.ram and, once written, tiers.ssd; load leaves it alone (loaded false, still vram);
  * ram        — a session evicted from the pool (a pool of 2): where ram, tokens = its furthest image; load leaves it alone;
  * ssd        — after SIGKILL + restart: where ssd, tokens = the stored checkpoint position, bytes = the stored ranges;
  * load → ram — loaded true, where ram with the same tokens; the next turn resumes from it (cached_prefix = the checkpoint) without a
                 disk read on the request path (no "restored from the session store" line, store loads unchanged); tokens == oracle;
  * two loads of one session at once read the disk once; load of a session already in VRAM after a load: loaded false;
  * an image larger than the host budget is not read; flush → where none;
  * asleep (level 2): a resident session offloaded to RAM reads ram; a stored one loads into RAM while asleep (the fake runtime aborts on
    any forward while asleep), and after wake the next turn resumes from it;
  * a long prefill in progress: where answers at once (receiver thread), and a load queued meanwhile is answered after it, correctly.
No GPU. Run: nice -n 19 python3 tools/test_cache_where_cpu.py
"""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import test_daemon_cpu as t  # noqa: E402

SLOW = t.SLOW


def where(h, *sids):
    r = h.op({'op': 'where', 'sessions': list(sids)})
    assert r.get('ok') is True, r
    return r['sessions']


def store_stats(h):
    return h.stats().get('session_store') or {}


def wait_writes(h, n, timeout=20):
    deadline = time.monotonic() + timeout * SLOW
    while time.monotonic() < deadline:
        st = store_stats(h)
        if st.get('writes', 0) >= n and st.get('queue', 1) == 0:
            return st
        time.sleep(.05)
    return store_stats(h)


def wait_where(h, sid, want, timeout=5):
    deadline = time.monotonic() + timeout * SLOW
    while True:
        w = where(h, sid)[sid]
        if w['where'] == want or time.monotonic() > deadline:
            return w
        time.sleep(.05)


def kill(h, ck):
    h.proc.kill()
    h.proc.wait(10)
    h.log.close()
    log = h.log_path.read_text(errors='replace')
    ck.true(f'{h.log_path.name}: no sanitizer report', 'ThreadSanitizer' not in log and 'runtime error' not in log, log[-2000:])


def run(exe, td):
    ck = t.Checker('cache-where')
    store = str(Path(td) / 'store')
    env = {'HIVE_SESSION_STORE': store, 'HIVE_CKPT_DELTA': '1'}
    A, B, C = t.tokens(700, 2001), t.tokens(600, 2002), t.tokens(500, 2003)

    # ---- none / vram / ram on one daemon ----
    h = t.Hived(exe, td, env, args=('--max-batch', '2'), name='w1')  # pool = max(max_sessions, max_batch) = 2
    w = where(h, 'nobody')['nobody']
    ck.eq('none: where', (w['where'], w['tokens'], w['bytes'], w['tiers']), ('none', 0, 0, {}))
    r = h.op({'op': 'load', 'session': 'nobody'})
    ck.eq('none: load answer', (r.get('ok'), r.get('loaded'), r.get('where'), r.get('reason')), (True, False, 'none', 'no stored checkpoint'))
    ck.true('bad load request refused', 'error' in h.op({'op': 'load'}), '')
    r = h.generate('A', A, 4)
    t.done_ok(ck, 'A turn 1', r, A, 4, 0)
    wait_writes(h, 1)
    w = where(h, 'A')['A']
    ck.eq('vram: where', w['where'], 'vram')
    ck.true('vram: tokens = live history', len(A) <= w['tokens'] <= len(A) + 4, json.dumps(w))
    ck.eq('vram: device bytes not attributed', w['bytes'], 0)
    ck.true('vram: the prompt checkpoint is listed under ram and ssd',
            w['tiers'].get('ram', {}).get('tokens') == len(A) and w['tiers'].get('ram', {}).get('bytes', 0) > 0 and
            w['tiers'].get('ssd', {}).get('tokens') == len(A) and w['tiers'].get('ssd', {}).get('bytes', 0) > 0, json.dumps(w))
    r = h.op({'op': 'load', 'session': 'A'})
    ck.eq('vram: load leaves it', (r.get('loaded'), r.get('where'), r.get('reason')), (False, 'vram', 'already in memory'))
    for sid, ids in (('B', B), ('C', C)):  # two more sessions: max_sessions 2 evicts A (least recently used) to an archived image
        r = h.generate(sid, ids, 3)
        t.done_ok(ck, f'{sid} turn', r, ids, 3, 0)
    w = where(h, 'A')['A']
    ck.eq('ram: evicted session', w['where'], 'ram')
    ck.true('ram: tokens = the furthest image (last state at eviction)', len(A) <= w['tokens'] <= len(A) + 4 and w['bytes'] > 0, json.dumps(w))
    ck.true('ram: no vram tier', 'vram' not in w['tiers'], json.dumps(w))
    r = h.op({'op': 'load', 'session': 'A'})
    ck.eq('ram: load leaves it', (r.get('loaded'), r.get('where')), (False, 'ram'))
    ws = where(h, 'A', 'B', 'C', 'nobody')
    ck.eq('several sessions in one call', sorted((k, v['where']) for k, v in ws.items()),
          [('A', 'ram'), ('B', 'vram'), ('C', 'vram'), ('nobody', 'none')])
    wait_writes(h, 3)
    kill(h, ck)

    # ---- ssd only after a restart, then load ----
    h = t.Hived(exe, td, env, name='w2')
    w = where(h, 'A')['A']
    ck.eq('ssd: where after a restart', (w['where'], w['tokens']), ('ssd', len(A)))
    ck.true('ssd: bytes = stored ranges', w['bytes'] > 0 and list(w['tiers']) == ['ssd'], json.dumps(w))
    loads0 = store_stats(h).get('loads', 0)
    res = [None, None]

    def ld(i):
        res[i] = h.op({'op': 'load', 'session': 'A'})
    ths = [threading.Thread(target=ld, args=(i,)) for i in range(2)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    ck.true('load: both answered, one of them loaded', sorted(bool(x.get('loaded')) for x in res) in ([False, True], [True, True]) and
            all(x.get('where') == 'ram' for x in res), json.dumps(res))
    first = next(x for x in res if x.get('loaded'))
    ck.eq('load: ram tokens = checkpoint position', first.get('tokens'), len(A))
    ck.true('load: read time and bytes reported', first.get('read_bytes', 0) > 0 and first.get('read_ms', -1) >= 0, json.dumps(first))
    ck.eq('load: the disk was read once', store_stats(h).get('loads', 0) - loads0, 1)
    ck.true('load: log line', 'stored checkpoint (pos 700' in h.tail(100000) and 'attached in RAM' in h.tail(100000), h.tail(20))
    w = where(h, 'A')['A']
    ck.true('load: where ram, ssd still listed', w['where'] == 'ram' and w['tiers'].get('ssd', {}).get('tokens') == len(A), json.dumps(w))
    A2 = A + t.tokens(40, 2004)
    r = h.generate('A', A2, 3)
    t.done_ok(ck, 'turn after load', r, A2, 3, len(A))
    ck.true('turn after load: no disk read on the request path', 'A: checkpoint restored from the session store' not in h.tail(100000) and
            store_stats(h).get('loads', 0) - loads0 == 1, json.dumps(store_stats(h)))
    r = h.op({'op': 'load', 'session': 'A'})
    ck.eq('load of a session now in VRAM', (r.get('loaded'), r.get('where')), (False, 'vram'))
    # flush drops everything: where none
    h.op({'op': 'flush'})
    w = where(h, 'A')['A']
    ck.eq('flush: where none', w['where'], 'none')
    kill(h, ck)

    # ---- host budget: an image larger than host_session_mb is not read ----
    h = t.Hived(exe, td, env, args=('--host-session-mb', '0.001'), name='w3')
    h.generate('D', t.tokens(600, 2005), 2)
    wait_writes(h, 1)
    kill(h, ck)
    h = t.Hived(exe, td, env, args=('--host-session-mb', '0.001'), name='w4')
    l0 = store_stats(h).get('loads', 0)
    r = h.op({'op': 'load', 'session': 'D'})
    ck.eq('budget: not read', (r.get('loaded'), r.get('where'), r.get('reason'), store_stats(h).get('loads', 0) - l0),
          (False, 'ssd', 'larger than the host session budget', 0))
    kill(h, ck)

    # ---- asleep: offloaded sessions read ram; a stored session loads while asleep (no forward), resumes after wake ----
    h = t.Hived(exe, td, env, name='w5')
    E = t.tokens(650, 2006)
    h.generate('E', E, 3)
    F = t.tokens(550, 2007)
    h.generate('F', F, 3)
    wait_writes(h, 2)
    kill(h, ck)
    h = t.Hived(exe, td, env, name='w6')
    G = t.tokens(500, 2008)
    h.generate('G', G, 3)
    r = h.op({'op': 'sleep', 'level': 2})
    ck.true('asleep: sleep ok', r.get('ok') is True and r.get('state') == 'sleeping', json.dumps(r))
    w = where(h, 'G', 'E')
    ck.eq('asleep level 2: resident session offloaded to RAM', w['G']['where'], 'ram')
    ck.eq('asleep: stored session on disk', w['E']['where'], 'ssd')
    r = h.op({'op': 'load', 'session': 'E'})
    ck.eq('asleep: load attaches in RAM', (r.get('loaded'), r.get('where'), r.get('tokens')), (True, 'ram', len(E)))
    ck.eq('asleep: still asleep', h.stats().get('state'), 'sleeping')
    r = h.op({'op': 'wake'})
    ck.true('wake ok', r.get('ok') is True, json.dumps(r))
    E2 = E + t.tokens(30, 2009)
    r = h.generate('E', E2, 3)
    t.done_ok(ck, 'turn after a load while asleep', r, E2, 3, len(E))
    G2 = G + t.tokens(30, 2010)
    r = h.generate('G', G2, 3)
    t.done_ok(ck, 'offloaded session after wake', r, G2, 3, None)
    t.stop_clean(h, ck)

    # ---- a long prefill in progress: where answers at once, a load is answered after the forward ----
    slow = {**env, 'FAKE_PREFILL_MS': str(1500 * SLOW), 'HIVE_STORE_CANARY': '0'}  # every fake forward is slow: skip the canary's 2 x 2K-token prefill
    h = t.Hived(exe, td, slow, name='w7')
    out = {}
    long_ids = t.tokens(900, 2011)
    th = threading.Thread(target=lambda: out.update(r=h.generate('L', long_ids, 2)))
    th.start()
    time.sleep(.4 * SLOW)
    t0 = time.monotonic()
    w = where(h, 'F', 'L')
    dt = time.monotonic() - t0
    ck.true('during a prefill: where answers at once', dt < 0.5 * SLOW and w['F']['where'] == 'ssd', f'{dt:.3f}s {json.dumps(w)}')
    r = h.op({'op': 'load', 'session': 'F'})
    th.join()
    ck.eq('during a prefill: the load is attached', (r.get('loaded'), r.get('where'), r.get('tokens')), (True, 'ram', len(F)))
    t.done_ok(ck, 'the long prefill', out['r'], long_ids, 2, 0)
    F2 = F + t.tokens(25, 2012)
    r = h.generate('F', F2, 2)
    t.done_ok(ck, 'turn after a load during a prefill', r, F2, 2, len(F))
    t.stop_clean(h, ck)
    return ck


def main():
    with tempfile.TemporaryDirectory(prefix='hive-where-') as td:
        t0 = time.monotonic()
        exe = t.build_daemon(td)
        print(f'built real hived.cpp on fake CUDA/runtime in {time.monotonic() - t0:.1f}s (sanitizer {t.harness.sanitizer()})', flush=True)
        ck = run(exe, td)
    for f in ck.failures:
        print('FAIL', f)
    print(f'cache where/load: {ck.checks} checks, {len(ck.failures)} failures')
    sys.exit(1 if ck.failures else 0)


if __name__ == '__main__':
    main()
