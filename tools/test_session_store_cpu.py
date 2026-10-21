#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HIVE_SESSION_STORE (hive/session_store.h) on the REAL engine/src/hived.cpp (fake CUDA/runtime, tools/test_daemon_cpu.py harness).

Before it, an engine restart or a model switch dropped every conversation (RAM only): the first request of a long room read the whole
prompt again (measured 2026-10-15: 379,914 GLM tokens, 138 s). Checked:
  * a conversation → SIGKILL (the service stops with `docker rm -f`) → restart → the next turn resumes from the prompt checkpoint on
    disk (cached_prefix = the previous prompt) and its tokens equal the state oracle,
  * a boundary snapshot (HIVE_PREFIX_SHARE) written by one conversation serves another conversation after the restart,
  * a follow-up turn writes only its increment (the previous image's segments are on disk already),
  * a damaged pack is detected (length/hash) — the request prefills from 0 and is still correct,
  * another daemon binary (layout key) never reads the store, the disk budget drops the oldest entries, flush empties the store,
  * negative control: without HIVE_SESSION_STORE nothing is reused after the restart.
No GPU. Run: nice -n 19 python3 tools/test_session_store_cpu.py
"""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import test_daemon_cpu as t  # noqa: E402

SLOW = t.SLOW


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


def kill(h):
    h.proc.kill()
    h.proc.wait(10)
    h.log.close()


def run(exe, td):
    ck = t.Checker('session-store')
    store = str(Path(td) / 'store')
    env = {'HIVE_SESSION_STORE': store, 'HIVE_PREFIX_SHARE': '1', 'HIVE_CKPT_DELTA': '1'}  # delta images as in service (HIVE_CKPT_DELTA=1 in its configs)
    A = t.tokens(700, 1600)
    # 1. conversation, then SIGKILL
    h = t.Hived(exe, td, env, name='st1')
    r = h.generate('S', A, 3)
    t.done_ok(ck, 'first turn', r, A, 3, 0)
    st = wait_writes(h, 1)
    ck.true('the prompt checkpoint reached the disk', st.get('writes', 0) >= 1 and st.get('ckpts', 0) >= 1, json.dumps(st))
    full = st.get('written', 0)
    ck.true('startup line', '[hived] session store on:' in h.tail(100000), '')
    kill(h)
    # 2. restart: the next turn (previous answer dropped, a new message) resumes from the checkpoint on disk
    h = t.Hived(exe, td, env, name='st2')
    B = A + t.tokens(40, 1601)
    r = h.generate('S', B, 3)
    t.done_ok(ck, 'turn after the restart', r, B, 3, len(A))
    ck.true('restore line', 'checkpoint restored from the session store' in h.tail(100000), '')
    st = wait_writes(h, 1)
    # the rows restored from disk are known there: the follow-up writes its new rows, tokens and logits, not the whole image again
    ck.true('the follow-up checkpoint writes its increment only', 0 < st.get('written', 0) < full, f"{st.get('written')} vs first {full}")
    # 3. a boundary snapshot of one conversation serves another after a restart
    C = t.tokens(500, 1602)
    r = h.generate('X', C + t.tokens(30, 1603), 3, boundaries=[500])
    ck.true('X done', r['done'] is not None, repr(r['error']))
    wait_writes(h, 3)
    kill(h)
    h = t.Hived(exe, td, env, name='st3')
    D = C + t.tokens(60, 1604)
    r = h.generate('Y', D, 3)
    t.done_ok(ck, 'other conversation after the restart', r, D, 3, 500)
    ck.true('snapshot restore line', 'restored from the session store' in h.tail(100000), '')
    kill(h)
    # 3b. a deeper snapshot on disk than the session's own checkpoint: only the snapshot is read, and the request's new checkpoint is an
    #     increment over it (the restored snapshot is its delta base)
    h = t.Hived(exe, td, env, name='st3b')
    F = t.tokens(500, 1610)
    r = h.generate('X2', F, 3); wait_writes(h, 1)                              # X2's checkpoint at 500
    G = F + t.tokens(300, 1611)
    r = h.generate('Z2', G + t.tokens(20, 1612), 3, boundaries=[800])          # another conversation leaves a snapshot at 800
    st = wait_writes(h, 3)
    kill(h)
    h = t.Hived(exe, td, env, name='st3c')
    H = G + t.tokens(50, 1613)
    w0 = store_stats(h).get('written', 0)
    r = h.generate('X2', H, 3)
    t.done_ok(ck, 'deeper snapshot after the restart', r, H, 3, 800)
    log = h.tail(1000000)
    ck.true('the shallower checkpoint was not read', 'X2: checkpoint restored' not in log and 'X2: boundary snapshot at 800 restored' in log, '')
    st = wait_writes(h, 1)
    ck.true('the checkpoint after a snapshot restore is an increment', 0 < st.get('written', 0) - w0 < full, f"{st.get('written', 0) - w0} vs full {full}")
    kill(h)
    # 3d. a checkpoint whose last token the next prompt changes (a Qwen prompt ends "<think>\n", merged into another token next turn) is not read
    h = t.Hived(exe, td, env, name='st3d')
    P = t.tokens(400, 1620)
    h.generate('Q2', P, 2); wait_writes(h, 1); kill(h)
    h = t.Hived(exe, td, env, name='st3e')
    P2 = P[:-1] + [P[-1] + 1] + t.tokens(30, 1621)
    r = h.generate('Q2', P2, 2)
    t.done_ok(ck, 'changed last token', r, P2, 2, 0)
    ck.true('a checkpoint whose last token differs is not read', 'Q2: checkpoint restored' not in h.tail(1000000) and store_stats(h).get('loads', 0) == 0,
            json.dumps(store_stats(h)))
    kill(h)
    # 3f. M4 (2026-10-21): known_ entries of segments no image holds any more were erased only with their pack — the index grew with every
    #     segment ever written. Twenty follow-up turns of one conversation (each checkpoint replaces the previous one and shares its prefix rows):
    #     after the last write the index holds only segments some image still holds (the host ledger's segments). Allocations whose
    #     address is reused overwrite their entry, so the leak showed only for segments whose address was not reused (here 46 vs 42).
    h = t.Hived(exe, td, env, name='st3f')
    K = t.tokens(300, 1630)
    for turn in range(20):
        r = h.generate('K', K, 2)
        t.done_ok(ck, f'M4 turn {turn}', r, K, 2, None)
        wait_writes(h, turn + 1)
        K = K + t.tokens(40, 1631 + turn)
    st, hs = store_stats(h), h.stats()
    # the pruning is amortized (each write scans twice the entries it adds — session_store.h prune_known): the index stays within twice the
    #   live segments; convergence over long random runs and "live entries never dropped" are engine/tests/test_session_store_index_cpu.cpp
    ck.true('M4: the store index stays within twice the live segments', 0 < st.get('known', 0) <= 2 * hs.get('host_segments', 0),
            f"known {st.get('known')} · host segments {hs.get('host_segments')} · written {st.get('written')}")
    kill(h)
    # 4. a damaged pack: detected, the request prefills from 0 and stays correct
    packs = sorted(Path(store).glob('*/packs/*.pack'), key=lambda p: p.stat().st_size)
    for p in packs:
        with open(p, 'r+b') as f:
            f.seek(max(0, p.stat().st_size // 2)); b = f.read(1); f.seek(max(0, p.stat().st_size // 2)); f.write(bytes([b[0] ^ 0xFF]) if b else b'\0')
    h = t.Hived(exe, td, env, name='st4')
    E = B + t.tokens(40, 1605)                                                 # continues S's checkpoint (so the damaged pack is read)
    r = h.generate('S', E, 3)
    t.done_ok(ck, 'damaged store', r, E, 3, None)
    ck.true('damaged pack refused (load_fail counted)', store_stats(h).get('load_fail', 0) >= 1, json.dumps(store_stats(h)))
    # 5. flush empties the store
    h.op({'op': 'flush'})
    ck.eq('flush empties the store', (store_stats(h).get('ckpts'), store_stats(h).get('snaps')), (0, 0))
    kill(h)
    # 6. another daemon binary = another layout key: never reads this store
    h = t.Hived(exe, td, env, name='st5')
    r = h.generate('K', A, 3); wait_writes(h, 1); kill(h)
    other = str(Path(td) / 'hived_other')
    shutil.copy(exe, other)
    h = t.Hived(other, td, env, name='st6')
    r = h.generate('K', A + t.tokens(10, 1606), 3)
    t.done_ok(ck, 'other binary', r, A + t.tokens(10, 1606), 3, 0)
    kill(h)
    # 6b. the same daemon's directory of another build is dropped at start-up (it can never be read); other daemons' directories stay
    stale = Path(store) / ('hived-' + '0' * 16); (stale / 'packs').mkdir(parents=True); (stale / 'packs' / 'x.pack').write_bytes(b'x')
    foreign = Path(store) / ('hived_other-' + '1' * 16); (foreign / 'metas').mkdir(parents=True)
    h = t.Hived(exe, td, env, name='st6b')
    ck.true('a stale build directory of this daemon is dropped', not stale.exists(), str(list(Path(store).iterdir())))
    ck.true("another daemon's directory stays", foreign.exists(), '')
    kill(h)
    # 6c. a rebuild (another build id, the same daemon) resumes the store once its canary agrees — 2026-10-18/19: five rebuilds in a day
    #     each dropped every stored conversation
    h = t.Hived(exe, td, env, name='st6c')
    R = t.tokens(600, 1630)
    h.generate('R', R, 3); wait_writes(h, 1)
    ck.true('the canary is on disk', store_stats(h).get('canary') is True, json.dumps(store_stats(h)))
    kill(h)
    td2 = Path(td) / 'rebuild'; td2.mkdir()
    src = (t.harness.ROOT / 'engine/src/hived.cpp').read_text()
    rebuilt = t.build_daemon(str(td2), hived_source='#undef HIVE_BUILD_ID\n#define HIVE_BUILD_ID "rebuilt-for-the-test"\n' + src)
    h = t.Hived(rebuilt, td, env, name='st6d', build_id='rebuilt-for-the-test')
    R2 = R + t.tokens(30, 1631)
    r = h.generate('R', R2, 3)
    t.done_ok(ck, 'a rebuild resumes the stored conversation', r, R2, 3, len(R))
    ck.true('canary kept by the rebuild', 'session store canary:' in h.tail(100000) and '· kept ·' in h.tail(100000), h.tail(3000))
    kill(h)
    # 6d. a store whose canary is gone cannot be checked: its entries are dropped, a new canary is written
    for m in Path(store).glob('hived-*/metas/canary-*.json'):
        m.unlink()
    h = t.Hived(exe, td, env, name='st6e')
    ck.true('no canary: the entries are dropped', 'unverifiable entries dropped' in h.tail(100000), h.tail(3000))
    r = h.generate('R', R2 + t.tokens(20, 1632), 3)
    t.done_ok(ck, 'after the drop the turn prefills from the start', r, R2 + t.tokens(20, 1632), 3, 0)
    kill(h)
    # 6e. a canary this build reads differently (its image damaged: unreadable) drops the store too
    h = t.Hived(exe, td, env, name='st6f')
    h.generate('R3', R, 3); wait_writes(h, 1); kill(h)
    for m in Path(store).glob('hived-*/metas/canary-*.json'):
        meta = json.loads(m.read_text()); meta['fields'][0] = meta['fields'][0] + 1 if isinstance(meta['fields'][0], int) else meta['fields'][0]
        m.write_text(json.dumps(meta))
    h = t.Hived(exe, td, env, name='st6g')
    log = h.tail(100000)
    ck.true('a canary that does not match drops the store', 'entries dropped' in log and '· kept ·' not in log, log[-3000:])
    r = h.generate('R3', R + t.tokens(10, 1633), 3)
    t.done_ok(ck, 'after a canary mismatch the turn prefills from the start', r, R + t.tokens(10, 1633), 3, 0)
    kill(h)
    # 7. budget: a tiny budget keeps only the newest entry
    small = {**env, 'HIVE_SESSION_STORE': str(Path(td) / 'store-small'), 'HIVE_SESSION_STORE_GB': '0.000004'}  # ~4 KB
    h = t.Hived(exe, td, small, name='st7')
    h.generate('P', t.tokens(300, 1607), 2); wait_writes(h, 1)
    h.generate('Q', t.tokens(300, 1608), 2); st = wait_writes(h, 2)
    ck.true('the budget dropped the oldest entry', st.get('evicted', 0) >= 1 and st.get('ckpts', 0) + st.get('snaps', 0) <= 1, json.dumps(st))
    kill(h)
    # 8. negative control: no store, nothing reused after a restart
    h = t.Hived(exe, td, {'HIVE_PREFIX_SHARE': '1'}, name='st8')
    h.generate('N', A, 3); kill(h)
    h = t.Hived(exe, td, {'HIVE_PREFIX_SHARE': '1'}, name='st9')
    r = h.generate('N', B, 3)
    t.done_ok(ck, 'control without the store', r, B, 3, 0)
    ck.true('control: no store stats', 'session_store' not in h.stats(), '')
    t.stop_clean(h, ck)
    return ck


def main():
    with tempfile.TemporaryDirectory(prefix='hive-store-') as td:
        t0 = time.monotonic()
        exe = t.build_daemon(td)
        print(f'built real hived.cpp on fake CUDA/runtime in {time.monotonic() - t0:.1f}s (sanitizer {t.harness.sanitizer()})', flush=True)
        ck = run(exe, td)
    for f in ck.failures:
        print('FAIL', f)
    print(f'session store: {ck.checks} checks, {len(ck.failures)} failures')
    sys.exit(1 if ck.failures else 0)


if __name__ == '__main__':
    main()
