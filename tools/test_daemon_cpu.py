#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""The REAL engine/src/hived.cpp on CPU: production ingress, scheduler, session/prefix/checkpoint
selection, SocketWriter, publish_stats, ExpertStore + CPU worker pool, and the production
save_image/load_image/reset_seq text from runtime.cpp — over a real Unix socket. The model is
tools/cpu_fake/fake_runtime.inc: a deterministic state oracle, so every emitted token is
predicted here and a wrong prefix reuse, restore or fence ordering fails the test.

No GPU, no model weights, no service. Options:
  HIVE_TEST_SANITIZER=thread (run under `setarch x86_64 -R`), HIVE_DAEMON_CONFIGS=a,b
Certifies host logic only (see docs); CUDA kernels, real stream semantics and model numerics
are NOT exercised.
"""
import json
import os
from pathlib import Path
import random
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake')]
import harness  # noqa: E402

MASK = (1 << 64) - 1
BASIS, PRIME = 1469598103934665603, 1099511628211
PATCH_BYTES = 16
TSAN = harness.sanitizer() == 'thread'
SLOW = 4 if TSAN else 1


def fnv(data, h=BASIS):
    for b in data:
        h = ((h ^ b) * PRIME) & MASK
    return h


def contents(ids, images=()):
    out = [t & 0xffffffff for t in ids]
    for im, blob in images:
        h = fnv(blob) & 0xffffffff
        for p in range(im['start'], im['start'] + len(im['types'])):
            out[p] |= h << 32
    return out


def oracle(ids, n, images=()):
    h = BASIS
    for c in contents(ids, images):
        h = fnv(c.to_bytes(8, 'little'), h)
    out = []
    for _ in range(n):
        t = 10 + h % 4000
        out.append(t)
        h = fnv(t.to_bytes(8, 'little'), h)
    return out


def tokens(n, seed, avoid=(7,)):
    r = random.Random(seed)
    out = []
    while len(out) < n:
        t = r.randrange(20, 4000)
        if t not in avoid:
            out.append(t)
    return out


def image(start, h, w, seed):
    blob = random.Random(seed).randbytes(h * w * PATCH_BYTES)
    meta = {'start': start, 'n_vit_h': h, 'n_vit_w': w, 'types': [0] + [1] * (h * w - 2) + [3], 'nbytes': len(blob)}
    return meta, blob


# The fake Model has no kv_source layer, so Runtime::tail_mode_for (production text) is always false and C1 /
# tail-split decisions are unobservable. FAKE_KV_SOURCE=1 gives it one (only tail_mode_for reads kv_source_layers in
# the generated source — checked below); unset = the unchanged fake. Applied here to the generated text until
# tools/cpu_fake/fake_runtime.inc carries the same line.
KV_ANCHOR = 'cfg_.swiglu_limit = 10.f;'
KV_PATCH = KV_ANCHOR + '\n  if (fake_env("FAKE_KV_SOURCE", 0)) cfg_.kv_source_layers = {0};  // tail_mode_for needs a kv source layer'


def build_daemon(td, name='hived', hived_source=None):
    gen_text = harness.runtime_source()
    if 'FAKE_KV_SOURCE' not in gen_text:
        assert gen_text.count(KV_ANCHOR) == 1, 'fake Model anchor drifted'
        gen_text = gen_text.replace(KV_ANCHOR, KV_PATCH)
    assert [l for l in gen_text.splitlines() if 'kv_source_layers' in l and 'FAKE_KV_SOURCE' not in l and 'tail_layer' not in l] == [], \
        'kv_source_layers now read outside tail_mode_for: FAKE_KV_SOURCE would change more than C1'
    gen = Path(td) / 'fake_runtime_gen_kv.cpp'
    gen.write_text(gen_text)
    src = harness.ROOT / 'engine/src/hived.cpp'
    if hived_source is not None:
        src = Path(td) / (name + '_hived.cpp')
        src.write_text(hived_source)
    objs = harness.objects(td, [src, gen, harness.ROOT / 'engine/src/expert_store.cpp', harness.ROOT / 'engine/src/cpu/expert_cpu.cpp',
                                harness.FAKE / 'fake_checkpoint.cpp'])
    return harness.build(td, name, '', objs)


class Hived:
    count = 0

    def __init__(self, exe, td, env=None, args=(), name='hived'):
        Hived.count += 1
        self.sock = str(Path(td) / f'{name}{Hived.count}.sock')
        self.log_path = Path(td) / f'{name}{Hived.count}.log'
        self.log = open(self.log_path, 'w')
        e = harness.tsan_supp_env(harness.tsan_env())
        e.update({'FAKE_STEP_MS': '3', 'FAKE_CHECK_CHUNK': '1'})  # FAKE_CHECK_CHUNK: fake forward aborts on a chunk above max_chunk x prefill_slots
        e.update(env or {})
        self.env = e
        self.proc = subprocess.Popen([exe, '--ckpt', 'fake', '--sock', self.sock, '--max-ctx', '4096', '--max-chunk', '256',
                                      '--max-sessions', '2', '--max-batch', '4', '--cache-state', '', '--cpu-threads', '2', '--vram-cache-mb', '0', *args],
                                     env=e, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 20 * SLOW
        while True:
            self.alive()
            try:
                if self.stats().get('build_id') == 'cpu-fake':
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('hived did not start\n' + self.tail())
            time.sleep(.02)

    def tail(self, n=60):
        self.log.flush()
        return '\n'.join(self.log_path.read_text(errors='replace').splitlines()[-n:])

    def alive(self):
        if self.proc.poll() is not None:
            raise AssertionError(f'hived exited {self.proc.returncode}\n' + self.tail())

    eagain = 0  # connect() refused with EAGAIN: listen(16) backlog full (acceptor takes one fd per poll round)

    def connect(self):
        deadline = time.monotonic() + 10 * SLOW
        while True:
            s = socket.socket(socket.AF_UNIX)
            s.settimeout(30 * SLOW)
            try:
                s.connect(self.sock)
                return s
            except BlockingIOError:
                s.close()
                Hived.eagain += 1
                if time.monotonic() > deadline:
                    raise
                time.sleep(.005)
            except OSError:
                s.close()
                raise

    def op(self, obj):
        with self.connect() as s:
            s.sendall((json.dumps(obj) + '\n').encode())
            return json.loads(s.makefile().readline())

    def stats(self):
        return self.op({'op': 'stats'})

    def cancel(self, session, rid=None):
        return self.op({'op': 'cancel', 'session': session, **({} if rid is None else {'rid': rid})})

    def generate(self, session, ids, max_tokens, images=(), on_token=None, close_after=None, rid=None, boundaries=None, prefix_extra=None):
        req = {'op': 'generate', 'session': session, 'ids': ids, 'temperature': 0}
        if prefix_extra is not None:
            req['prefix_extra'] = prefix_extra
        if max_tokens is not None:
            req['max_tokens'] = max_tokens
        if rid is not None:
            req['rid'] = rid
        if boundaries is not None:
            req['boundaries'] = boundaries
        blob = b''
        if images:
            req['images'] = [m for m, _ in images]
            blob = b''.join(b for _, b in images)
            req['bin'] = len(blob)
        s = self.connect()
        try:
            s.sendall((json.dumps(req) + '\n').encode() + blob)
            out = {'tokens': [], 'done': None, 'error': None, 'progress': 0}
            for line in s.makefile():
                m = json.loads(line)
                if 'id' in m:
                    out['tokens'].append(m['id'])
                    if on_token:
                        on_token(len(out['tokens']))
                    if close_after and len(out['tokens']) >= close_after:
                        return out
                elif 'progress' in m:
                    out['progress'] += 1
                elif m.get('done'):
                    out['done'] = m
                    break
                elif 'error' in m:
                    out['error'] = m['error']
                    break
            return out
        finally:
            s.close()

    def fds(self):
        return len(os.listdir(f'/proc/{self.proc.pid}/fd'))

    def settled_fds(self, quiet=.3):
        # Take the baseline fd count only once it is stable: SocketWriter closes the stats connection used for the readiness check asynchronously (after the D10 inline send),
        #   so one extra fd can stay open for a while after start-up — counting it in the baseline caused intermittent "got 5 want 6" at the end (seen twice under TSAN).
        deadline = time.monotonic() + 5 * SLOW
        last, since = self.fds(), time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(.05)
            n = self.fds()
            if n != last:
                last, since = n, time.monotonic()
            elif time.monotonic() - since >= quiet * SLOW:
                break
        return last

    def stop(self):
        self.alive()
        self.proc.send_signal(signal.SIGTERM)
        rc = self.proc.wait(timeout=10 * SLOW)
        self.log.close()
        return rc


class Checker:
    def __init__(self, name):
        self.name, self.failures, self.checks, self.transcript = name, [], 0, []

    def eq(self, what, got, want):
        self.checks += 1
        if got != want:
            self.failures.append(f'{what}: got {got!r} want {want!r}')

    def true(self, what, cond, detail=''):
        self.checks += 1
        if not cond:
            self.failures.append(f'{what} {detail}')


def done_ok(ck, what, r, ids, n, cached, images=()):
    ck.true(f'{what}: no error', r['error'] is None and r['done'] is not None, repr(r['error']))
    ck.transcript.append((what, r['tokens'], r['error'], r['done'] and r['done']['cached_prefix'], r['done'] and r['done']['finish']))
    if r['done'] is None:
        return r
    ck.eq(f'{what}: tokens (state oracle)', r['tokens'], oracle(ids, n, images))
    if cached is not None:
        ck.eq(f'{what}: cached_prefix', r['done']['cached_prefix'], cached)
    ck.eq(f'{what}: prefill_tokens', r['done']['prefill_tokens'], len(ids) - r['done']['cached_prefix'])
    return r


# ---------------------------------------------------------------------------------------------------------
def sc_session(h, ck, cfg):
    history = int(cfg.get('HIVE_CKPT_HISTORY') or 1) >= 2
    A = tokens(300, 1)
    r1 = done_ok(ck, 'turn1', h.generate('s1', A, 6), A, 6, 0)
    ids2 = A + r1['tokens'] + tokens(40, 2)
    r2 = done_ok(ck, 'turn2 continues live state', h.generate('s1', ids2, 6), ids2, 6, len(A) + 5)
    done_ok(ck, 'regenerate exact prompt (checkpoint logits)', h.generate('s1', ids2, 6), ids2, 6, len(ids2))
    ids3 = list(ids2); ids3[100] = 4001  # edit inside the first turn (4001 is outside tokens())
    done_ok(ck, 'edit earlier turn: no mismatched prefix reuse', h.generate('s1', ids3, 6), ids3, 6, 0)
    ids4 = ids3[:-5] + tokens(9, 3)
    done_ok(ck, 'edit last message after checkpoint: reset', h.generate('s1', ids4, 5), ids4, 5, 0)
    # history frames: T3 edits the last user message; only a stored earlier prompt-end can be reused
    A2 = tokens(280, 4)
    t1 = done_ok(ck, 'hist T1', h.generate('s2', A2, 5), A2, 5, 0)
    T2 = A2 + t1['tokens'] + tokens(30, 5)
    done_ok(ck, 'hist T2', h.generate('s2', T2, 5), T2, 5, len(A2) + 4)
    T3 = A2 + t1['tokens'] + tokens(30, 6)
    done_ok(ck, 'hist T3 edited last message', h.generate('s2', T3, 5), T3, 5, len(A2) if history else 0)
    done_ok(ck, 'hist T4 exact regen', h.generate('s2', T3, 5), T3, 5, len(T3))


def sc_images(h, ck, cfg):
    image_ckpt = cfg.get('HIVE_IMAGE_CKPT') not in (None, '', '0')
    P = tokens(300, 10)
    im1 = image(50, 3, 3, 1)
    im2 = (im1[0], random.Random(99).randbytes(len(im1[1])))
    r1 = done_ok(ck, 'image turn1', h.generate('i1', P, 5, [im1]), P, 5, 0, [im1])
    ids2 = P + r1['tokens'] + tokens(20, 11)
    r2 = done_ok(ck, 'image turn2 same image: live reuse', h.generate('i1', ids2, 5, [im1]), ids2, 5, len(P) + 4, [im1])
    done_ok(ck, 'image bytes changed, same tokens: no reuse', h.generate('i1', ids2, 5, [im2]), ids2, 5, 0, [im2])
    done_ok(ck, 'exact regen with changed image', h.generate('i1', ids2, 5, [im2]), ids2, 5, len(ids2) if image_ckpt else 0, [im2])
    done_ok(ck, 'text-only request after image session', h.generate('i1', ids2, 5), ids2, 5, 0)
    # continuation of a live image session WITHOUT the image (placeholders now plain text): must not reuse
    q1 = done_ok(ck, 'image session i2 turn1', h.generate('i2', P, 5, [im1]), P, 5, 0, [im1])
    ids5 = P + q1['tokens'] + tokens(12, 12)
    done_ok(ck, 'text-only continuation of an image state: no reuse', h.generate('i2', ids5, 5), ids5, 5, 0)
    moved = ({**im1[0], 'start': 60}, im1[1])
    done_ok(ck, 'image moved', h.generate('i1', ids2, 5, [moved]), ids2, 5, 0, [moved])
    del r2


def sc_flush(h, ck):
    """{"op":"flush"}: clears every reusable prompt state when idle (the next request prefills from scratch); refused while decoding."""
    P = tokens(300, 150)
    done_ok(ck, 'flush: first request', h.generate('fl', P, 3), P, 3, 0)
    Q = P + tokens(20, 151)
    done_ok(ck, 'flush control: without a flush the prompt checkpoint is reused', h.generate('fl', Q, 3), Q, 3, len(P))
    r = h.op({'op': 'flush'})
    ck.true('flush: ok when idle', r.get('ok') is True and r.get('sessions', 0) >= 1, repr(r))
    R = Q + tokens(10, 152)
    done_ok(ck, 'flush: the next request starts from a reset', h.generate('fl', R, 3), R, 3, 0)
    ck.true('flush: logged', any('flush: cleared' in l for l in log_lines(h, 'flush:')))
    got, res = threading.Event(), {}
    th = threading.Thread(target=lambda: res.update(h.generate('fm', tokens(280, 153), 400, on_token=lambda n: n >= 2 and got.set())))
    th.start()
    ck.true('flush: decoding started', got.wait(20 * SLOW))
    busy = h.op({'op': 'flush'})
    ck.true('flush: refused while decoding', busy.get('error') == 'not idle' and busy.get('retryable') is True, repr(busy))
    th.join(60 * SLOW)
    ck.true('flush: the running request was not disturbed', res.get('done') and res['done']['finish'] == 'length', repr(res.get('done')))
    ck.true('flush: idle again -> ok', h.op({'op': 'flush'}).get('ok') is True)


def sc_cancel(h, ck, cfg):
    P = tokens(270, 20)
    got = threading.Event()
    res = {}
    th = threading.Thread(target=lambda: res.update(h.generate('c1', P, 400, on_token=lambda n: n >= 3 and got.set())))
    th.start()
    ck.true('cancel: first tokens streamed', got.wait(20 * SLOW))
    t0 = time.monotonic()
    ck.eq('cancel op', h.cancel('c1'), {'ok': True})
    th.join(30 * SLOW)
    ck.true('cancel: request finished', not th.is_alive())
    ck.true('cancel: finish=cancel', res.get('done') and res['done']['finish'] == 'cancel', repr(res.get('done')))
    ck.true('cancel: prompt stopped early', len(res.get('tokens', [])) < 400)
    ck.true('cancel latency < 2s', time.monotonic() - t0 < 2 * SLOW)
    n = len(res['tokens'])
    ck.eq('cancel: emitted tokens follow oracle', res['tokens'], oracle(P, n))
    ids = P + res['tokens'] + tokens(10, 21)
    done_ok(ck, 'after cancel: session reusable, live state = every emitted token', h.generate('c1', ids, 4), ids, 4, len(P) + n)
    # re-generate immediately after cancel: admission must finish the cancelled request, not answer "session busy"
    got2, res2 = threading.Event(), {}
    Q = tokens(260, 22)
    th = threading.Thread(target=lambda: res2.update(h.generate('c2', Q, 400, on_token=lambda n: n >= 2 and got2.set())))
    th.start()
    got2.wait(20 * SLOW)
    h.cancel('c2')
    R = tokens(265, 23)
    done_ok(ck, 'regenerate right after cancel', h.generate('c2', R, 4), R, 4, None)
    th.join(30 * SLOW)
    ck.true('cancelled request terminal', res2.get('done') is not None and res2['done']['finish'] == 'cancel', repr(res2))
    # concurrent same-session request without cancel is refused, the running one is unaffected
    got3, res3 = threading.Event(), {}
    S = tokens(262, 24)
    th = threading.Thread(target=lambda: res3.update(h.generate('c3', S, 300, on_token=lambda n: n >= 2 and got3.set())))
    th.start()
    got3.wait(20 * SLOW)
    busy = h.generate('c3', tokens(263, 25), 3)
    ck.eq('same session while active', busy['error'], 'session busy')
    h.cancel('c3')
    th.join(30 * SLOW)
    ck.true('busy: running request continues then cancels', res3.get('done') is not None, repr(res3))
    if res3.get('done'):
        ck.eq('busy: running request tokens', res3['tokens'], oracle(S, len(res3['tokens'])))


def sc_rid_cancel(h, ck):
    # D5: a cancel carrying a rid applies only to that request — a late cancel of finished request A must not hit the next request B of the same session
    P = tokens(270, 40)
    done_ok(ck, 'rid: A completes', h.generate('r1', P, 3, rid='A'), P, 3, None)
    got, res = threading.Event(), {}
    Q = tokens(275, 41)
    th = threading.Thread(target=lambda: res.update(h.generate('r1', Q, 400, rid='B', on_token=lambda n: n >= 3 and got.set())))
    th.start()
    ck.true('rid: B streams', got.wait(20 * SLOW))
    ck.eq('rid: late cancel of finished A', h.cancel('r1', rid='A'), {'ok': True})
    ck.eq('rid: cancel of unknown rid', h.cancel('r1', rid='nope'), {'ok': True})
    ck.eq('rid: non-string rid is ignored (normalised away)', h.op({'op': 'cancel', 'session': 'zz', 'rid': 5}), {'ok': True})
    time.sleep(.3 * SLOW)
    ck.true('rid: B still running after A/unknown cancels', th.is_alive())
    ck.eq('rid: cancel B', h.cancel('r1', rid='B'), {'ok': True})
    th.join(30 * SLOW)
    ck.true('rid: B cancelled by its own rid', res.get('done') is not None and res['done']['finish'] == 'cancel', repr(res.get('done')))
    ck.eq('rid: B tokens follow oracle', res.get('tokens'), oracle(Q, len(res.get('tokens', []))))


def sc_max_tokens_default(h, ck):
    # max_tokens absent · 0 · beyond the remaining context = the whole remaining context (max_ctx 4096 − prompt), rather than a default of 256 / a "context overflow" error.
    P = tokens(4000, 50)
    for what, mt in (('absent', None), ('zero', 0), ('oversize', 10 ** 7), ('string', 'x')):
        r = h.generate(f'mt-{what}', P, mt)
        ck.true(f'max_tokens {what}: no error', r.get('error') is None, repr(r.get('error')))
        ck.eq(f'max_tokens {what}: generates up to the remaining context', len(r['tokens']), 4096 - len(P))
        ck.eq(f'max_tokens {what}: tokens follow oracle', r['tokens'], oracle(P, len(r['tokens'])))
    r = h.generate('mt-explicit', P, 7)
    ck.eq('max_tokens explicit small is honoured', len(r['tokens']), 7)
    r = h.generate('mt-full', tokens(4096, 51), None)
    ck.eq('prompt filling the whole context still overflows', r.get('error'), 'context overflow')


def sc_concurrent(h, ck, cfg):
    results, errs = {}, []

    def user(k):
        try:
            A = tokens(260 + k, 30 + k)
            r1 = h.generate(f'k{k}', A, 6)
            ids = A + r1['tokens'] + tokens(15, 40 + k)
            r2 = h.generate(f'k{k}', ids, 6)
            results[k] = (A, r1, ids, r2)
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))

    ths = [threading.Thread(target=user, args=(k,)) for k in range(7)]
    for t in ths: t.start()
    for t in ths: t.join(120 * SLOW)
    ck.eq('concurrent: client errors', errs, [])
    for k, (A, r1, ids, r2) in sorted(results.items()):
        done_ok(ck, f'concurrent k{k} turn1', r1, A, 6, 0)
        done_ok(ck, f'concurrent k{k} turn2 (live/archived state)', r2, ids, 6, len(A) + 5)
    # stats are a <=1/s engine snapshot (and not refreshed while idle), so use the engine log for the path taken
    log = h.tail(100000)
    ck.true('concurrent: evicted sessions restored from archived live state', ' via archived ' in log)


def sc_disconnect(h, ck, cfg):
    P = tokens(261, 50)
    r = h.generate('d1', P, 300, close_after=2)
    ck.eq('disconnect: got first tokens', len(r['tokens']), 2)
    Q = tokens(264, 51)
    t0 = time.monotonic()
    while True:  # the engine notices a vanished client at its next write; until then the session is legitimately busy
        r = h.generate('d1', Q, 3)
        if r['error'] != 'session busy' or time.monotonic() - t0 > 5 * SLOW:
            break
        time.sleep(.01)
    ck.true('disconnect: session released within 1s', time.monotonic() - t0 < 1 * SLOW, f'{time.monotonic() - t0:.2f}s')
    done_ok(ck, 'after client disconnect: same session usable', r, Q, 3, None)
    for i in range(20):  # abandoned partial headers/images
        s = h.connect(); s.sendall(b'{"op":"generate","bin":100000}\n' if i % 2 else b'{"op":'); s.close()


def stats_hammer(h, stop, errs, counter):
    while not stop.is_set():
        try:
            s = h.stats()
            if s.get('build_id') != 'cpu-fake' or 'pending_requests' not in s:
                errs.append('bad stats ' + repr(s)[:200])
            counter[0] += 1
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))
        time.sleep(.002)


def wait_fds(h, base, ck):
    deadline = time.monotonic() + 10 * SLOW
    while h.fds() > base and time.monotonic() < deadline:
        time.sleep(.05)
    ck.eq('fds released (baseline after start)', h.fds(), base)


def run_config(exe, td, name, cfg):
    ck = Checker(name)
    h = Hived(exe, td, cfg, args=CONFIG_ARGS.get(name, ()), name=name)
    base = h.settled_fds()
    stop, errs, counter = threading.Event(), [], [0]
    ham = threading.Thread(target=stats_hammer, args=(h, stop, errs, counter))
    ham.start()
    try:
        for sc in (sc_session, sc_images, sc_cancel, sc_concurrent, sc_disconnect, lambda h, ck, cfg: sc_flush(h, ck)):
            sc(h, ck, cfg)
            h.alive()
    finally:
        stop.set(); ham.join()
    ck.eq('concurrent stats readers: errors', errs[:3], [])
    # HIVE_WARM_BUSY_CAP: startup banner iff the switch is on (the fake store has no slots, so the warm path itself never runs here —
    #   the capped warm is checked on the GPU: "[hived] warm cache: busy cap N -> n experts" lines in service logs)
    ck.eq('warm busy cap banner iff switch on', '[hived] warm cache busy cap:' in h.tail(400000), cfg.get('HIVE_WARM_BUSY_CAP', '0') not in ('', '0'))
    # HIVE_WARM_DEFER: same — the banner iff on; the paced warm itself (Runtime::warm_defer → promote_after_step) runs only with slots (service logs: "[runtime] deferred warm:")
    ck.eq('warm defer banner iff switch on', '[hived] warm cache deferred: <=' in h.tail(400000), cfg.get('HIVE_WARM_DEFER', '0') not in ('', '0'))
    if cfg.get('HIVE_BATCH_PREFILL') == '1':  # the switch must actually batch the standard scenarios (sc_concurrent's users)
        ck.true('H3: standard scenarios ran multi-sequence rounds', any(n >= 2 for n, _ in batch_rounds(h)), repr(batch_rounds(h)[:5]))
    ck.true('stats readers ran', counter[0] > 20, str(counter[0]))
    wait_fds(h, base, ck)
    st = h.stats()
    ck.eq('no pending requests at end', st['pending_requests'], 0)
    rc = h.stop()
    ck.eq('SIGTERM exit (no graceful shutdown path in hived)', rc, -signal.SIGTERM)
    log = h.log_path.read_text(errors='replace')
    ck.true('no sanitizer report', 'ThreadSanitizer' not in log and 'runtime error' not in log, log[-3000:])
    return ck, counter[0]


def run_failures(exe, td):
    ck = Checker('failures')
    h = Hived(exe, td, {'FAKE_FAIL_TOKEN': '7', 'FAKE_FAIL_DECODE_POS': '335'}, name='fail')
    base = h.settled_fds()
    A = tokens(300, 60)
    r1 = done_ok(ck, 'fail T1', h.generate('f1', A, 4), A, 4, 0)
    bad = A + r1['tokens'] + [7] + tokens(20, 61)
    r = h.generate('f1', bad, 4)
    ck.eq('host exception in prefill -> request error', r['error'], 'request failed: fake injected forward failure')
    ids = A + r1['tokens'] + tokens(20, 62)
    done_ok(ck, 'after prefill exception: broken live state not reused (checkpoint)', h.generate('f1', ids, 4), ids, 4, len(A))
    G = tokens(333, 63)
    r = h.generate('g1', G, 6)
    ck.eq('decode exception -> error to active request', r['error'], 'decode failed: fake injected decode failure')
    ck.eq('decode exception: tokens before failure follow oracle', r['tokens'], oracle(G, len(r['tokens'])))
    ids = G + r['tokens'] + tokens(5, 64)
    done_ok(ck, 'after decode exception: session recovers from checkpoint', h.generate('g1', ids, 3), ids, 3, len(G))
    wait_fds(h, base, ck)
    ck.eq('failures: SIGTERM exit', h.stop(), -signal.SIGTERM)
    return ck


def run_sigterm_inflight(exe, td):
    """hived has no graceful shutdown: SIGTERM with requests in flight must not hang and clients get EOF."""
    ck = Checker('sigterm-inflight')
    h = Hived(exe, td, {'FAKE_STEP_MS': '20'}, name='term')
    res, got = [{}, {}], threading.Event()
    ths = [threading.Thread(target=lambda i=i: res[i].update(h.generate(f't{i}', tokens(270, 70 + i), 1000, on_token=lambda n: n >= 2 and got.set())))
           for i in range(2)]
    for t in ths: t.start()
    ck.true('streams started', got.wait(20 * SLOW))
    t0 = time.monotonic()
    h.proc.send_signal(signal.SIGTERM)
    rc = h.proc.wait(timeout=10 * SLOW)
    ck.true('exit promptly after SIGTERM', time.monotonic() - t0 < 3 * SLOW)
    ck.eq('exit status = killed by SIGTERM (default action)', rc, -signal.SIGTERM)
    for t in ths: t.join(10 * SLOW)
    ck.true('clients unblocked with EOF, no terminal message', all(not t.is_alive() for t in ths) and all(r.get('done') is None and r.get('error') is None for r in res), repr([{k: v for k, v in r.items() if k != 'tokens'} for r in res]))
    ck.true('socket path left behind', Path(h.sock).exists())
    h2 = Hived(exe, td, name='term')  # restart on a new path; stale path handling below
    h2.stop()
    # a restart on the SAME stale path must work (hived unlinks before bind)
    p = subprocess.Popen([exe, '--ckpt', 'fake', '--sock', h.sock, '--max-ctx', '4096', '--cache-state', '', '--cpu-threads', '1', '--vram-cache-mb', '0'],
                         env=harness.tsan_supp_env(harness.tsan_env()), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = False
    for _ in range(int(200 * SLOW)):
        try:
            with socket.socket(socket.AF_UNIX) as s:
                s.settimeout(2); s.connect(h.sock); s.sendall(b'{"op":"stats"}\n')
                ok = json.loads(s.makefile().readline()).get('build_id') == 'cpu-fake'
                break
        except OSError:
            time.sleep(.05)
    p.send_signal(signal.SIGTERM); p.wait(10 * SLOW)
    ck.true('restart on the stale socket path serves stats', ok)
    return ck


# ---------------------------------------------------------------------------------------------------------
# Prefill chunking checks: D3 (C1 next-chunk bound), M1 tail-aware split, B2 per-chunk log, D9 reclamation order,
# D10 disconnect before the next chunk, stats freshness, accept burst.
CHUNK_RE = re.compile(r'^\[hived\] (\S+): prefill chunk (\d+) M (\d+) · (\d+)→(\d+)/(\d+) · upper (on|skip) · tail rows (-?\d+) · '
                      r'hit (-?\d+) streamed (-?\d+) cpu (-?\d+) dma_rows (-?\d+) · cpu wait [-\d.]+ ms span [-\d.]+ ms · [\d.]+ ms$')
TOTAL_RE = re.compile(r'^\[hived\] (\S+): prefill total (\d+) rows in (\d+) chunks ·')
TAIL_ARGS = ('--max-chunk', '1536', '--decoder-tail', '1100')  # tail mode = M >= 1024 (threshold) and M > 1100


def chunks(h, sid):
    return [(int(m[3]), m[7]) for m in (CHUNK_RE.match(l) for l in h.tail(100000).splitlines()) if m and m[1] == sid]


def sc_d3(h, ck, want):
    """HIVE_PREFILL_BUDGET_MS>0 and no active decoder: the next chunk is chunk_cap, so C1 stays on (D3)."""
    P = tokens(3500, 80)
    done_ok(ck, 'D3 solo prefill with budget', h.generate('d3', P, 3), P, 3, 0)
    ck.eq('D3: chunk sizes / upper_needed (budget on, no decoder)', chunks(h, 'd3'), want)
    lines = [l for l in h.tail(100000).splitlines() if '[hived] d3: prefill chunk' in l]
    ck.true('B2: every chunk line parses (M, i/total, upper, tail rows, hit/streamed/cpu/dma deltas, cpu wait/span, ms)',
            lines and all(CHUNK_RE.match(l) for l in lines), repr(lines[:2]))
    tot = [TOTAL_RE.match(l) for l in h.tail(100000).splitlines()]
    tot = [m for m in tot if m and m[1] == 'd3']
    ck.true('B2: per-request prefill total line', len(tot) == 1 and int(tot[0][2]) == 3500 and int(tot[0][3]) == len(want), repr(tot))


def sc_tail_split(h, ck, want):
    P = tokens(2400, 81)
    done_ok(ck, 'tail split: tokens unchanged (lossless chunking)', h.generate('ts', P, 3), P, 3, 0)
    ck.eq('tail split: chunk sizes / upper_needed', chunks(h, 'ts'), want)


def sc_disconnect_prefill(h, ck):
    """D10: a client gone during chunk k is seen at chunk k's progress write, before chunk k+1 runs."""
    P = tokens(1100, 82)  # 5 chunks of <=256 at FAKE_PREFILL_MS each
    s = h.connect()
    s.sendall((json.dumps({'op': 'generate', 'session': 'dp', 'ids': P, 'max_tokens': 3, 'temperature': 0}) + '\n').encode())
    f = s.makefile()
    first = json.loads(f.readline())
    ck.true('D10: first progress line', 'progress' in first, repr(first))
    f.close(); s.close()
    deadline, tot = time.monotonic() + 20 * SLOW, None
    while tot is None and time.monotonic() < deadline:
        tot = next((m for m in (TOTAL_RE.match(l) for l in h.tail(100000).splitlines()) if m and m[1] == 'dp'), None)
        time.sleep(.05)
    ck.true('D10: prefill stopped', tot is not None)
    ck.eq('D10: chunks run after the disconnect (1 = only the one in flight)', tot and int(tot[3]), 2)
    Q = tokens(300, 83)
    done_ok(ck, 'D10: session usable after the disconnect', h.generate('dp', Q, 3), Q, 3, None)


def sc_stats_fresh(h, ck):
    """publish_stats runs at request finish (and per prefill chunk), not only at the 1/s loop top."""
    s0 = h.stats()
    ck.eq('stats: fresh daemon has no sessions', s0['sessions'], 0)
    P = tokens(300, 84)
    done_ok(ck, 'stats fresh request', h.generate('sf', P, 4), P, 4, 0)
    s1 = h.stats()  # immediately: the engine is idle in its 30 s cv wait now
    ck.eq('stats: session visible right after done', s1['sessions'], 1)
    ck.true('stats: snapshot newer than before the request', s1['snapshot_ms'] > s0['snapshot_ms'])
    # the prompt checkpoint (>= 300 packed comp rows of 288 B) is taken after the last prefill chunk: only the finish publish sees it
    ck.true('stats: host bytes include the prompt checkpoint', s1['host_session_bytes'] >= 300 * 288, repr(s1['host_session_bytes']))


def sc_accept_burst(h, ck, n=300):
    """listen(SOMAXCONN) + accept until EAGAIN: a connect burst gets no EAGAIN and every cancel op is answered."""
    try:
        out = subprocess.run(['ss', '-xlnH'], capture_output=True, text=True, timeout=10).stdout
        row = next((l.split() for l in out.splitlines() if h.sock in l), None)
        if row:  # listening socket: Send-Q = backlog
            ck.true('accept: listen backlog > 16', int(row[3]) > 16, repr(row))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    socks, refused = [], 0
    for _ in range(n):  # no retry: one connect attempt each, as fast as possible
        s = socket.socket(socket.AF_UNIX)
        s.settimeout(30 * SLOW)
        try:
            s.connect(h.sock)
            socks.append(s)
        except BlockingIOError:
            refused += 1
            s.close()
    ck.eq('accept burst: connect EAGAIN (backlog full)', refused, 0)
    for s in socks:
        s.sendall(b'{"op":"cancel","session":"nobody"}\n')
    ok = 0
    for s in socks:
        try:
            ok += json.loads(s.makefile().readline()) == {'ok': True}
        except (OSError, ValueError):
            pass
        s.close()
    ck.eq('accept burst: cancel ops answered', ok, n)


def sc_d9(h, ck, calib):
    """D9: the revived session is taken out before reclamation; the oldest other state goes first."""
    A, B, C = tokens(300, 85), tokens(300, 86), tokens(300, 87)
    ra = done_ok(ck, 'D9 A1', h.generate('ra', A, 5), A, 5, 0)
    sizes = [h.stats()['host_session_bytes']]
    rb = done_ok(ck, 'D9 B1', h.generate('rb', B, 5), B, 5, 0)
    done_ok(ck, 'D9 C1 (evicts A)', h.generate('rc', C, 5), C, 5, 0)
    sizes.append(h.stats()['host_session_bytes'])
    if calib:
        return sizes
    # revive A: evicts B (its live state is saved) -> over budget -> reclaim; old code trimmed A (oldest, still in archived)
    A2 = A + ra['tokens'] + tokens(20, 88)
    done_ok(ck, 'D9 A2 revived from archived live state (not trimmed)', h.generate('ra', A2, 5), A2, 5, len(A) + 4)
    # C (resident, newer than archived B) kept its prompt checkpoint: an edited last message reuses exactly the prompt
    C2 = C + tokens(20, 90)
    done_ok(ck, 'D9 C2 prompt checkpoint kept (newer than the reclaimed B)', h.generate('rc', C2, 5), C2, 5, len(C))
    B2 = B + rb['tokens'] + tokens(20, 89)
    done_ok(ck, 'D9 B2 oldest state was the one reclaimed', h.generate('rb', B2, 5), B2, 5, 0)
    return sizes


def d9_budget_mb(exe, td):
    """Budget between 'after C1' (3 ckpt + A live) and 'at A2 revive' (+ B live), measured on this build."""
    ck = Checker('d9-calibration')
    h = Hived(exe, td, args=('--max-batch', '1', '--max-sessions', '2', '--host-session-mb', '4096'), name='d9cal')
    try:
        one, after_c1 = sc_d9(h, ck, True)
    finally:
        h.stop()
    live = after_c1 - 3 * one
    assert live > one // 4 and not ck.failures, (one, after_c1, ck.failures)
    return (after_c1 + live / 2) / 1048576.0


def stop_clean(h, ck):
    h.stop()
    log = h.log_path.read_text(errors='replace')
    ck.true(f'{h.log_path.name}: no sanitizer report', 'ThreadSanitizer' not in log and 'runtime error' not in log, log[-2000:])


def run_extras(exe, td):
    ck = Checker('review-fixes')
    kv = {'FAKE_KV_SOURCE': '1'}
    h = Hived(exe, td, name='fresh')
    try:
        sc_stats_fresh(h, ck)
        sc_accept_burst(h, ck)
        sc_rid_cancel(h, ck)
        sc_max_tokens_default(h, ck)
    finally:
        stop_clean(h, ck)
    h = Hived(exe, td, {**kv, 'HIVE_PREFILL_BUDGET_MS': '50'}, args=TAIL_ARGS, name='d3')
    try:
        sc_d3(h, ck, [(1536, 'skip'), (1536, 'on'), (428, 'on')])
    finally:
        stop_clean(h, ck)
    for val, want in ((None, [(1536, 'on'), (864, 'on')]), ('0', [(1536, 'on'), (864, 'on')]), ('', [(1536, 'on'), (864, 'on')]),
                      ('1', [(1299, 'skip'), (1101, 'on')])):
        h = Hived(exe, td, {**kv, **({} if val is None else {'HIVE_TAIL_SPLIT': val})}, args=TAIL_ARGS, name='ts')
        try:
            sc_tail_split(h, ck, want)
            if val == '1':  # image requests keep the single-chunk rule; default fake (no kv source) never splits
                ck.true('tail split: startup line names the tail-mode minimum', 'final prefill chunk >= 1101 rows' in h.tail(100000))
        finally:
            stop_clean(h, ck)
    h = Hived(exe, td, {'FAKE_PREFILL_MS': str(300 * SLOW)}, name='dp')
    try:
        sc_disconnect_prefill(h, ck)
    finally:
        stop_clean(h, ck)
    mb = d9_budget_mb(exe, td)
    h = Hived(exe, td, args=('--max-batch', '1', '--max-sessions', '2', '--host-session-mb', f'{mb:.6f}'), name='d9')
    try:
        sc_d9(h, ck, False)
    finally:
        stop_clean(h, ck)
    return ck


# ---------------------------------------------------------------------------------------------------------
# Reuse paths (all opt-in): H5 boundary snapshots + cross-session prefix sharing (HIVE_PREFIX_SHARE), M3 image-turn
# reuse (HIVE_IMAGE_CKPT), M8 deferred prompt checkpoint (HIVE_DEFER_CKPT), M7 prefill_pending, graceful stop
# (HIVE_GRACEFUL_STOP_S). The fake runtime ignores upper_needed, so C1 decisions are checked through the chunk/reuse log.
SHARE = {'HIVE_PREFIX_SHARE': '1'}
PEND_RE = re.compile(r'^\[hived\] (\S+): prefill_pending=(\d)( \(multi-chunk\))?$')


def log_lines(h, needle):
    return [l for l in h.tail(100000).splitlines() if needle in l]


def pend(h, sid):
    return [m[2] + (m[3] or '') for m in (PEND_RE.match(l) for l in h.tail(100000).splitlines()) if m and m[1] == sid]


def new_chunks(h, sid, before):
    return chunks(h, sid)[before:]


def sc_share_cross(h, ck):
    SYS = tokens(300, 100)
    ids1 = SYS + tokens(50, 101)
    done_ok(ck, 'H5 A: first session (cut at the system boundary)', h.generate('sa', ids1, 4, boundaries=[300]), ids1, 4, 0)
    ck.eq('H5 A: chunks (256 natural; hint 300 inside the last chunk -> one extra cut)', chunks(h, 'sa'), [(256, 'on'), (44, 'on'), (50, 'on')])
    ck.true('H5 A: snapshot at the boundary', any('sa: prefix snapshot at 300 (upper on' in l for l in log_lines(h, 'prefix snapshot')))
    ids2 = SYS + tokens(60, 102)
    done_ok(ck, 'H5 B: other session, same system prefix -> shared reuse', h.generate('sb', ids2, 4, boundaries=[300]), ids2, 4, 300)
    ck.true('H5 B: reuse via shared-prefix', any('sb: reuse 300/360 via shared-prefix' in l for l in log_lines(h, 'via shared-prefix')))
    ids3 = tokens(300, 103) + ids1[300:]  # different system prompt of the same length
    done_ok(ck, 'H5 C: different system prompt -> no reuse', h.generate('sc', ids3, 4, boundaries=[300]), ids3, 4, 0)
    ids4 = SYS[:299] + [4002] + ids1[300:]  # differs only in the last token of the prefix
    done_ok(ck, 'H5 D: prefix differs in its last token -> no reuse', h.generate('sd', ids4, 4), ids4, 4, 0)
    st = h.stats()
    ck.true('H5 stats', st.get('prefix_share', {}).get('hits', 0) >= 1 and st['prefix_share']['entries'] >= 1, repr(st.get('prefix_share')))
    ids5 = SYS + tokens(20, 104)  # malformed hints are absorbed (ignored), never an error; the stored snapshot still applies
    done_ok(ck, 'H5 malformed boundaries ignored', h.generate('se', ids5, 3, boundaries=['x', -1, 1e9, None, 10 ** 12, 300.5, [300]]), ids5, 3, 300)
    done_ok(ck, 'H5 boundaries not a list: ignored', h.generate('sf', ids5, 3, boundaries={'a': 300}), ids5, 3, 300)


def sc_share_edit(h, ck):
    SYS = tokens(300, 110)
    T1 = SYS + tokens(40, 111)
    r1 = done_ok(ck, 'H5 edit T1', h.generate('se1', T1, 6, boundaries=[300]), T1, 6, 0)
    end1 = len(T1) + 6  # end of the completed assistant turn
    T2 = T1 + r1['tokens'] + tokens(30, 112)
    n0 = len(chunks(h, 'se1'))
    done_ok(ck, 'H5 edit T2 continues live', h.generate('se1', T2, 6, boundaries=[300, end1]), T2, 6, len(T1) + 5)
    ck.eq('H5 edit T2 chunks (cut at the end of the assistant turn)', new_chunks(h, 'se1', n0), [(1, 'on'), (30, 'on')])
    T3 = T1 + r1['tokens'] + tokens(30, 113)  # edit the latest user message
    done_ok(ck, 'H5 edit latest message -> longest boundary (assistant end)', h.generate('se1', T3, 6, boundaries=[300, end1]), T3, 6, end1)
    T4 = SYS + tokens(40, 114) + r1['tokens'] + tokens(30, 112)  # edit the first user message
    done_ok(ck, 'H5 edit first user message -> system boundary', h.generate('se1', T4, 6, boundaries=[300, end1]), T4, 6, 300)


def sc_share_image(h, ck):
    P = tokens(320, 140)
    im1 = image(50, 3, 3, 141)
    im2 = (im1[0], random.Random(142).randbytes(len(im1[1])))
    A = P + tokens(40, 143)
    done_ok(ck, 'H5 image A (boundary after the image)', h.generate('ia', A, 3, [im1], boundaries=[320]), A, 3, 0, [im1])
    B = P + tokens(30, 144)
    done_ok(ck, 'H5 image B: same image bytes -> shared reuse', h.generate('ib', B, 3, [im1], boundaries=[320]), B, 3, 320, [im1])
    done_ok(ck, 'H5 image C: same tokens, other image bytes -> no reuse', h.generate('ic', B, 3, [im2]), B, 3, 0, [im2])
    done_ok(ck, 'H5 image D: text-only request over an image prefix -> no reuse', h.generate('id', B, 3), B, 3, 0)


def sc_share_c1(h, ck):
    X = tokens(1536, 120)
    A = X + tokens(1300, 121)
    done_ok(ck, 'C1 A: snapshot after an upper-skipped chunk', h.generate('ca', A, 3, boundaries=[1536]), A, 3, 0)
    ck.eq('C1 A chunks', chunks(h, 'ca'), [(1536, 'skip'), (1300, 'on')])
    ck.true('C1 A: snapshot flagged upper skip', any('ca: prefix snapshot at 1536 (upper skip' in l for l in log_lines(h, 'prefix snapshot')))
    B = X + tokens(200, 122)
    done_ok(ck, 'C1 B: continuation not tail mode -> flagged snapshot NOT reused', h.generate('cb', B, 3), B, 3, 0)
    C = X + tokens(1300, 123)
    done_ok(ck, 'C1 C: continuation in tail mode -> flagged snapshot reused', h.generate('cc', C, 3), C, 3, 1536)
    ck.eq('C1 C chunks', chunks(h, 'cc'), [(1300, 'on')])
    F = X + tokens(1300, 128)  # restored from the skipped snapshot: a hint inside the first chunk must not make it non-tail
    done_ok(ck, 'C1 F: restored skipped snapshot, hint in the first chunk not cut', h.generate('cf', F, 3, boundaries=[1600]), F, 3, 1536)
    ck.eq('C1 F chunks', chunks(h, 'cf'), [(1300, 'on')])
    D =tokens(1536, 124) + tokens(1300, 125)  # a cut after an upper-skipped chunk must keep that chunk in tail mode
    done_ok(ck, 'C1 D: cut that would leave a non-tail chunk after a skip is not taken', h.generate('cd', D, 3, boundaries=[1600]), D, 3, 0)
    ck.eq('C1 D chunks', chunks(h, 'cd'), [(1536, 'skip'), (1300, 'on')])
    E = tokens(1536, 126) + tokens(1300, 127)
    done_ok(ck, 'C1 E: tail-mode cut after a skip is taken', h.generate('ce', E, 3, boundaries=[2700]), E, 3, 0)
    ck.eq('C1 E chunks', chunks(h, 'ce'), [(1536, 'skip'), (1164, 'on'), (136, 'on')])


def sc_share_free(h, ck):
    P = tokens(520, 130)
    done_ok(ck, 'H5 free cut: 520 with hint 300', h.generate('fa', P, 3, boundaries=[300]), P, 3, 0)
    ck.eq('H5 free cut chunks (still 3 chunks)', chunks(h, 'fa'), [(256, 'on'), (44, 'on'), (220, 'on')])
    Q = tokens(350, 131)
    done_ok(ck, 'H5 extra chunks 0: hint inside the last chunk', h.generate('fb', Q, 3, boundaries=[300]), Q, 3, 0)
    ck.eq('H5 extra 0: not cut', chunks(h, 'fb'), [(256, 'on'), (94, 'on')])
    ck.true('H5 extra 0: no snapshot', not any('fb: prefix snapshot' in l for l in log_lines(h, 'prefix snapshot')))
    R = P[:300] + tokens(30, 132)
    done_ok(ck, 'H5 free-cut snapshot reused by another session', h.generate('fc', R, 3), R, 3, 300)
    # per-request budget: "prefix_extra" overrides the global 0 for that request only; malformed values are absorbed (global budget)
    X = tokens(350, 133)
    done_ok(ck, 'per-request extra 1: hint inside the last chunk', h.generate('xa', X, 3, boundaries=[300], prefix_extra=1), X, 3, 0)
    ck.eq('per-request extra 1: cut at the hint', chunks(h, 'xa'), [(256, 'on'), (44, 'on'), (50, 'on')])
    ck.true('per-request extra 1: snapshot at the hint', any('xa: prefix snapshot at 300' in l for l in log_lines(h, 'prefix snapshot')))
    Y = X[:300] + tokens(40, 134)
    done_ok(ck, 'per-request extra 1: the next request resumes from that snapshot', h.generate('xb', Y, 3), Y, 3, 300)
    for k, bad in enumerate(('1', -1, 1.5, None)):
        Z = tokens(350, 140 + k)
        sid = f'xz{k}'
        req_extra = bad if bad is not None else 'absent'
        done_ok(ck, f'per-request extra {req_extra!r}: absorbed', h.generate(sid, Z, 3, boundaries=[300], prefix_extra=bad), Z, 3, 0)
        ck.eq(f'per-request extra {req_extra!r}: global budget (no cut)', chunks(h, sid), [(256, 'on'), (94, 'on')])


def sc_share_free_cost(h, ck):
    """'free' = the remaining plan keeps its chunk count, streaming-chunk count and C1 skips (extra budget 0 here;
    tail mode = M >= 1024 and M > 1100, streaming = M >= 1024)."""
    P = tokens(3572, 133)  # natural: 1536 (skip) · 1536 · 500
    done_ok(ck, 'H5 cost: cut would lose the C1 skip of this chunk', h.generate('ka', P, 3, boundaries=[1050]), P, 3, 0)
    ck.eq('H5 cost: not cut (C1 skip kept)', chunks(h, 'ka'), [(1536, 'skip'), (1536, 'on'), (500, 'on')])
    Q = tokens(3572, 134)
    done_ok(ck, 'H5 cost: cut keeps count, streaming chunks and C1', h.generate('kb', Q, 3, boundaries=[1200]), Q, 3, 0)
    ck.eq('H5 cost: free cut taken', chunks(h, 'kb'), [(1200, 'skip'), (1536, 'on'), (836, 'on')])
    R = tokens(3972, 135)  # natural: 1536 (skip) · 1536 · 900 (short)
    done_ok(ck, 'H5 cost: cut would turn the short tail into a streaming chunk', h.generate('kc', R, 3, boundaries=[1200]), R, 3, 0)
    ck.eq('H5 cost: not cut (streaming chunks kept)', chunks(h, 'kc'), [(1536, 'skip'), (1536, 'on'), (900, 'on')])
    S = tokens(3572, 136)  # after a skipped chunk the cut must stay tail mode (C1 correctness), and it is free here
    done_ok(ck, 'H5 cost: free tail-mode cut after a skip', h.generate('kd', S, 3, boundaries=[2936]), S, 3, 0)
    ck.eq('H5 cost: cut after skip', chunks(h, 'kd'), [(1536, 'skip'), (1400, 'on'), (636, 'on')])


def sc_image_turns(h, ck, image_ckpt):
    P = tokens(300, 150)
    im1 = image(50, 3, 3, 151)
    im2 = (im1[0], random.Random(152).randbytes(len(im1[1])))
    done_ok(ck, 'M3 turn1', h.generate('m', P, 5, [im1]), P, 5, 0, [im1])
    ids2 = P + tokens(600, 153)  # the reply is not in the next prompt (reference drop_thinking): the live state cannot continue
    n0 = len(chunks(h, 'm'))
    done_ok(ck, 'M3 turn2 same image', h.generate('m', ids2, 5, [im1]), ids2, 5, 300 if image_ckpt else 0, [im1])
    ck.eq('M3 turn2 chunks (image only inside the reused prefix -> text rules: tile 2 x 256)', new_chunks(h, 'm', n0),
          [(512, 'on'), (88, 'on')] if image_ckpt else [(256, 'on')] * 3 + [(132, 'on')])
    done_ok(ck, 'M3 turn3 image bytes changed -> no reuse', h.generate('m', ids2, 5, [im2]), ids2, 5, 0, [im2])
    im3 = image(len(ids2) + 20, 3, 3, 154)
    ids4 = ids2 + tokens(20, 155) + tokens(9, 156) + tokens(10, 157)
    done_ok(ck, 'M3 turn4 new image in the suffix', h.generate('m', ids4, 5, [im2, im3]), ids4, 5, len(ids2) if image_ckpt else 0, [im2, im3])
    done_ok(ck, 'M3 turn5 exact regen', h.generate('m', ids4, 5, [im2, im3]), ids4, 5, len(ids4) if image_ckpt else 0, [im2, im3])


def sc_pending(h, ck):
    P = tokens(700, 160)
    done_ok(ck, 'M7 multi-chunk prefill', h.generate('p1', P, 3), P, 3, 0)
    Q = tokens(200, 161)
    done_ok(ck, 'M7 single-chunk prefill', h.generate('p2', Q, 3), Q, 3, 0)
    ck.eq('M7 p1: set before the first chunk, cleared after the last', pend(h, 'p1'), ['1 (multi-chunk)', '0'])
    ck.eq('M7 p2: single chunk never sets it', pend(h, 'p2'), [])
    bad = tokens(400, 162) + [7] + tokens(100, 163)  # FAKE_FAIL_TOKEN=7: exception in the 2nd chunk
    r = h.generate('p3', bad, 3)
    ck.eq('M7 p3 exception', r['error'], 'request failed: fake injected forward failure')
    ck.eq('M7 p3: cleared on exception', pend(h, 'p3'), ['1 (multi-chunk)', '0'])
    im = image(300, 17, 17, 164)  # 289 positions > max_chunk 256: fails at the 3rd chunk (early return)
    Pi = tokens(700, 165)
    r = h.generate('p4', Pi, 3, [im])
    ck.eq('M7 p4 image span error', r['error'], 'image span longer than chunk')
    ck.eq('M7 p4: cleared on early return', pend(h, 'p4'), ['1 (multi-chunk)', '0'])


def sc_pending_cancel(h, ck):
    """M7 on the cancel path: a slow multi-chunk prefill cancelled after its first chunk."""
    P = tokens(1100, 166)
    s = h.connect()
    s.sendall((json.dumps({'op': 'generate', 'session': 'pc', 'ids': P, 'max_tokens': 3, 'temperature': 0}) + '\n').encode())
    f = s.makefile()
    first = json.loads(f.readline())
    ck.true('M7 cancel: first progress line', 'progress' in first, repr(first))
    ck.eq('M7 cancel op', h.cancel('pc'), {'ok': True})
    last = None
    for line in f:
        last = json.loads(line)
        if last.get('done') or last.get('error'):
            break
    f.close(); s.close()
    ck.true('M7 cancel: finish=cancel', last and last.get('finish') == 'cancel', repr(last))
    ck.eq('M7 pc: cleared on cancel', pend(h, 'pc'), ['1 (multi-chunk)', '0'])


def snapshot_trace(exe, td, env, name):
    ck = Checker(name)
    h = Hived(exe, td, {'HIVE_TRACE_CACHE': '1', 'HIVE_CKPT_DELTA': '1', 'HIVE_CKPT_HISTORY': '3', **env}, name=name)
    try:
        sc_session(h, ck, {'HIVE_CKPT_HISTORY': '3'})
    finally:
        stop_clean(h, ck)
    lines = [l for l in h.log_path.read_text(errors='replace').splitlines() if l.startswith('[snapshot] pos ')]
    return ck, lines


def sc_hints_ignored(h, ck):
    """HIVE_PREFIX_SHARE off: the optional "boundaries" field is ignored (no cut, no snapshot, same tokens)."""
    P = tokens(350, 137)
    done_ok(ck, 'hints ignored when off', h.generate('ho', P, 3, boundaries=[300]), P, 3, 0)
    ck.eq('hints ignored when off: natural chunks', chunks(h, 'ho'), [(256, 'on'), (94, 'on')])
    Q = P[:300] + tokens(20, 138)
    done_ok(ck, 'hints ignored when off: no shared reuse', h.generate('hp', Q, 3, boundaries=[300]), Q, 3, 0)
    ck.true('hints ignored when off: no snapshot/stats', not log_lines(h, 'prefix snapshot') and 'prefix_share' not in h.stats())


def run_round2(exe, td):
    ck = Checker('round2')
    h = Hived(exe, td, name='hintsoff')
    try:
        sc_hints_ignored(h, ck)
    finally:
        stop_clean(h, ck)
    h = Hived(exe, td, {**SHARE, 'HIVE_IMAGE_CKPT': '1'}, name='share')
    try:
        sc_share_cross(h, ck)
        sc_share_edit(h, ck)
        sc_share_image(h, ck)
    finally:
        stop_clean(h, ck)
    h = Hived(exe, td, {**SHARE, 'HIVE_PREFIX_EXTRA_CHUNKS': '0'}, name='sharefree')
    try:
        sc_share_free(h, ck)
    finally:
        stop_clean(h, ck)
    h = Hived(exe, td, {**SHARE, 'HIVE_PREFIX_EXTRA_CHUNKS': '0', 'FAKE_KV_SOURCE': '1'}, args=TAIL_ARGS, name='sharecost')
    try:
        sc_share_free_cost(h, ck)
    finally:
        stop_clean(h, ck)
    h = Hived(exe, td, {**SHARE, 'FAKE_KV_SOURCE': '1'}, args=TAIL_ARGS, name='sharec1')
    try:
        sc_share_c1(h, ck)
    finally:
        stop_clean(h, ck)
    for on in (False, True):
        h = Hived(exe, td, {'HIVE_IMAGE_CKPT': '1' if on else '0'}, args=('--prefill-tile', '2'), name='m3')
        try:
            sc_image_turns(h, ck, on)
        finally:
            stop_clean(h, ck)
    h = Hived(exe, td, {'FAKE_FAIL_TOKEN': '7'}, name='pend')
    try:
        sc_pending(h, ck)
    finally:
        stop_clean(h, ck)
    h = Hived(exe, td, {'FAKE_PREFILL_MS': str(200 * SLOW)}, name='pendc')
    try:
        sc_pending_cancel(h, ck)
    finally:
        stop_clean(h, ck)
    # M8: the deferred prompt checkpoint is the same snapshot as the synchronous one (same positions, bytes, delta bases)
    c1, base = snapshot_trace(exe, td, {}, 'trace-sync')
    c2, deferred = snapshot_trace(exe, td, {'HIVE_DEFER_CKPT': '1'}, 'trace-defer')
    for c in (c1, c2):
        ck.failures += [c.name + ': ' + f for f in c.failures]; ck.checks += c.checks
    ck.true('M8: snapshot trace present', len(base) >= 5, repr(base[:3]))
    ck.eq('M8: deferred snapshots == synchronous snapshots (pos, d2h bytes, host bytes, delta)', deferred, base)
    ck.eq('M8: responses identical', c2.transcript, c1.transcript)
    return ck


def run_graceful(exe, td):
    ck = Checker('graceful-stop')
    drain = 1.5 * SLOW
    h = Hived(exe, td, {'FAKE_STEP_MS': '20', 'HIVE_GRACEFUL_STOP_S': str(drain)}, name='grace')
    res, got = [{}, {}], [threading.Event(), threading.Event()]
    long_ids, short_ids = tokens(270, 170), tokens(265, 171)
    ths = [threading.Thread(target=lambda: res[0].update(h.generate('g0', long_ids, 2000, on_token=lambda n: n >= 2 and got[0].set()))),
           threading.Thread(target=lambda: res[1].update(h.generate('g1', short_ids, 30, on_token=lambda n: n >= 2 and got[1].set())))]
    for t in ths: t.start()
    ck.true('graceful: streams started', got[0].wait(20 * SLOW) and got[1].wait(20 * SLOW))
    t0 = time.monotonic()
    h.proc.send_signal(signal.SIGTERM)
    time.sleep(.2 * SLOW)
    late = h.generate('g2', tokens(262, 172), 3)
    ck.eq('graceful: new request refused while draining', late['error'], 'daemon shutting down')
    ck.eq('graceful: control ops still served while draining', h.stats().get('build_id'), 'cpu-fake')
    rc = h.proc.wait(timeout=30 * SLOW)
    el = time.monotonic() - t0
    h.log.close()
    for t in ths: t.join(10 * SLOW)
    ck.eq('graceful: exit status 0', rc, 0)
    ck.true('graceful: exit after the drain deadline', drain - .1 <= el < drain + 8 * SLOW, f'{el:.2f}s')
    ck.true('graceful: short request finished within the drain', res[1].get('done') is not None and res[1]['done']['finish'] == 'length', repr(res[1].get('error')))
    ck.eq('graceful: short request tokens', res[1].get('tokens'), oracle(short_ids, 30))
    ck.eq('graceful: long request gets the shutdown terminal message', res[0].get('error'), 'daemon shutting down: generation cancelled')
    ck.eq('graceful: long request tokens follow the oracle', res[0].get('tokens'), oracle(long_ids, len(res[0].get('tokens', []))))
    ck.true('graceful: socket file removed', not Path(h.sock).exists())
    log = h.log_path.read_text(errors='replace')
    ck.true('graceful: completion logged', 'graceful stop complete (all replies flushed) — exit 0' in log, log[-1500:])
    ck.true('graceful: no sanitizer report', 'ThreadSanitizer' not in log and 'runtime error' not in log, log[-2000:])
    # idle daemon, SIGINT
    h = Hived(exe, td, {'HIVE_GRACEFUL_STOP_S': '5'}, name='grace')
    t0 = time.monotonic()
    h.proc.send_signal(signal.SIGINT)
    rc = h.proc.wait(timeout=10 * SLOW)
    h.log.close()
    ck.eq('graceful idle SIGINT: exit 0', rc, 0)
    ck.true('graceful idle: prompt exit', time.monotonic() - t0 < 3 * SLOW)
    ck.true('graceful idle: socket file removed', not Path(h.sock).exists())
    # second signal while draining = immediate default action
    h = Hived(exe, td, {'FAKE_STEP_MS': '20', 'HIVE_GRACEFUL_STOP_S': '60'}, name='grace')
    r2, g2 = {}, threading.Event()
    th = threading.Thread(target=lambda: r2.update(h.generate('g3', tokens(270, 173), 2000, on_token=lambda n: n >= 2 and g2.set())))
    th.start()
    ck.true('graceful 2nd signal: stream started', g2.wait(20 * SLOW))
    h.proc.send_signal(signal.SIGTERM)
    time.sleep(.3 * SLOW)
    h.proc.send_signal(signal.SIGTERM)
    rc = h.proc.wait(timeout=10 * SLOW)
    h.log.close()
    th.join(10 * SLOW)
    ck.eq('graceful: second SIGTERM = default action', rc, -signal.SIGTERM)
    # "0" / "" = off: the default immediate SIGTERM
    for v in ('0', ''):
        h = Hived(exe, td, {'HIVE_GRACEFUL_STOP_S': v}, name='grace')
        ck.eq(f'HIVE_GRACEFUL_STOP_S={v!r}: SIGTERM default action', h.stop(), -signal.SIGTERM)
        ck.true(f'HIVE_GRACEFUL_STOP_S={v!r}: socket left behind (default)', Path(h.sock).exists())
    return ck


# ---------------------------------------------------------------------------------------------------------
# H3 batch prefill (HIVE_BATCH_PREFILL) and H2 host tiles (HIVE_PREFILL_HOST_TILES / HIVE_PREFILL_HOST_MB).
# The fake forward_multi enforces the production contract (one part per sequence, sum slots_for <= prefill_slots, >1 part
# only with batch on) and logs "[fake] forward_multi parts N"; every token is still predicted by the state oracle.
BATCH = {'HIVE_BATCH_PREFILL': '1'}
NO_REPLY = {'tokens': [], 'done': None, 'error': 'no reply (daemon gone?)'}
BATCH_ROUND_RE = re.compile(r'^\[hived\] batch prefill round \d+: (\d+) sequences \(([^)]*)\) · (\d+) rows · slots (\d+)/(\d+) ·')


def batch_rounds(h):
    return [(int(m[1]), m[2]) for m in (BATCH_ROUND_RE.match(l) for l in h.tail(100000).splitlines()) if m]


def launch(h, sid, ids, n, res, key, **kw):
    t = threading.Thread(target=lambda: res.__setitem__(key, h.generate(sid, ids, n, **kw)))
    t.start()
    return t


def behind_blocker(h, reqs, blocker_len=700, blocker_sid='bz'):
    """Queue `reqs` [(sid, ids, n, kwargs)] while a slow one-round prefill runs, so the next admission sees them all waiting."""
    res, ths = {}, []
    B = tokens(blocker_len, 300 + len(blocker_sid))
    ths.append(launch(h, blocker_sid, B, 3, res, blocker_sid))
    time.sleep(.12 * SLOW)  # blocker is inside its FAKE_PREFILL_MS forward
    for sid, ids, n, kw in reqs:
        ths.append(launch(h, sid, ids, n, res, sid, **kw))
        time.sleep(.005)
    for t in ths: t.join(60 * SLOW)
    return res, B


def sc_batch_basic(h, ck, want_parts):
    ids = {f'b{k}': tokens(200 + 7 * k, 310 + k) for k in range(want_parts)}
    res, B = behind_blocker(h, [(sid, v, 5, {}) for sid, v in ids.items()])
    done_ok(ck, 'H3 blocker', res['bz'], B, 3, 0)
    for sid, v in ids.items():
        done_ok(ck, f'H3 {sid} batched: tokens = its own oracle', res[sid], v, 5, 0)
    rounds = batch_rounds(h)
    ck.true(f'H3: a round prefilled {want_parts} sequences together', any(n == want_parts for n, _ in rounds), repr(rounds))
    ck.true('H3: fake forward_multi saw the parts', any(f'[fake] forward_multi parts {want_parts} ' in l for l in log_lines(h, '[fake] forward_multi')))
    # continuation turns of the batched sessions reuse their own live state
    for sid, v in ids.items():
        nxt = v + res[sid]['tokens'] + tokens(20, 320 + len(sid))
        done_ok(ck, f'H3 {sid} turn 2 continues its live state', h.generate(sid, nxt, 4), nxt, 4, len(v) + 4)


def sc_batch_cancel(h, ck):
    """A member cancelled mid-batch ends as 'cancel'; the others are unaffected."""
    ids = {'c0': tokens(300, 330), 'c1': tokens(600, 331), 'c2': tokens(300, 332)}
    res, got = {}, threading.Event()
    B = tokens(700, 333)
    ths = [launch(h, 'cz', B, 3, res, 'cz')]
    time.sleep(.12 * SLOW)
    ths.append(launch(h, 'c0', ids['c0'], 5, res, 'c0'))
    time.sleep(.005)
    # c1: raw socket so its cancel is sent right after its first progress line (= after its first batch round)
    s = h.connect()
    s.sendall((json.dumps({'op': 'generate', 'session': 'c1', 'ids': ids['c1'], 'max_tokens': 50, 'temperature': 0}) + '\n').encode())
    time.sleep(.005)
    ths.append(launch(h, 'c2', ids['c2'], 5, res, 'c2'))
    f = s.makefile()
    line = f.readline()
    first = json.loads(line) if line.strip() else {}
    ck.true('H3 cancel: c1 first progress', 'progress' in first, repr(first))
    ck.eq('H3 cancel op', h.cancel('c1'), {'ok': True})
    last, toks = None, []
    for line in f:
        last = json.loads(line)
        if 'id' in last:
            toks.append(last['id'])
        if last.get('done') or last.get('error'):
            break
    f.close(); s.close()
    for t in ths: t.join(60 * SLOW)
    ck.true('H3 cancel: c1 finish=cancel', last and last.get('finish') == 'cancel', repr(last))
    ck.eq('H3 cancel: c1 tokens before the cancel follow its oracle', toks, oracle(ids['c1'], len(toks)))
    for sid in ('c0', 'c2'):
        done_ok(ck, f'H3 cancel: {sid} unaffected', res.get(sid) or NO_REPLY, ids[sid], 5, 0)
    ck.true('H3 cancel: c1 was prefilled in a multi-sequence round', any('c1:' in who and n >= 2 for n, who in batch_rounds(h)), repr(batch_rounds(h)))


def sc_batch_error(h, ck):
    """A member whose prefill throws (FAKE_FAIL_TOKEN=7) fails alone: the batch is retried one by one."""
    ids = {'e0': tokens(200, 340), 'e1': tokens(100, 341) + [7] + tokens(99, 342), 'e2': tokens(210, 343)}
    res, B = behind_blocker(h, [(sid, v, 4, {}) for sid, v in ids.items()], blocker_sid='ez')
    done_ok(ck, 'H3 error: blocker', res['ez'], B, 3, 0)
    ck.eq('H3 error: failing member gets the single-path error', res['e1']['error'], 'request failed: fake injected forward failure')
    for sid in ('e0', 'e2'):
        done_ok(ck, f'H3 error: {sid} still correct after the solo retry', res[sid], ids[sid], 4, None)
    ck.true('H3 error: batch failure logged and retried', log_lines(h, 'batch prefill failed: fake injected forward failure'))
    again = ids['e0'] + res['e0']['tokens'] + tokens(15, 344)
    done_ok(ck, 'H3 error: e0 session usable afterwards', h.generate('e0', again, 3), again, 3, None)


def sc_batch_share(h, ck):
    """H5 prefix share + batch: members restore the shared boundary snapshot, then prefill their suffixes together."""
    SYS = tokens(300, 350)
    first = SYS + tokens(60, 351)
    done_ok(ck, 'H3+H5: snapshot at the system boundary', h.generate('sh', first, 3, boundaries=[300]), first, 3, 0)
    ids = {f's{k}': SYS + tokens(200, 352 + k) for k in range(3)}
    res, B = behind_blocker(h, [(sid, v, 4, {'boundaries': [300]}) for sid, v in ids.items()], blocker_sid='sz')
    for sid, v in ids.items():
        done_ok(ck, f'H3+H5 {sid}: shared prefix reused inside the batch', res[sid], v, 4, 300)
    ck.true('H3+H5: suffixes prefilled together', any(n == 3 for n, _ in batch_rounds(h)), repr(batch_rounds(h)))


def sc_batch_first_token(h, ck):
    """Each member's first token goes out as soon as its own prefill ends (between rounds), not after the whole batch."""
    S, Lg = tokens(200, 360), tokens(1400, 361)
    t_first = {}
    res, ths = {}, []
    B = tokens(700, 362)
    ths.append(launch(h, 'fz', B, 3, res, 'fz'))
    time.sleep(.12 * SLOW)
    for sid, v in (('fs', S), ('fl', Lg)):
        ths.append(launch(h, sid, v, 3, res, sid, on_token=lambda n, sid=sid: t_first.setdefault(sid, time.monotonic())))
        time.sleep(.005)
    for t in ths: t.join(60 * SLOW)
    done_ok(ck, 'H3 first token: short member', res['fs'], S, 3, 0)
    done_ok(ck, 'H3 first token: long member', res['fl'], Lg, 3, 0)
    ck.true('H3 first token: short and long in one round', any('fs:' in who and 'fl:' in who for n, who in batch_rounds(h)), repr(batch_rounds(h)))
    ck.true('H3 first token: short member streams before the long prefill ends',
            'fs' in t_first and 'fl' in t_first and t_first['fl'] - t_first['fs'] > .15 * SLOW, repr(t_first))


def sc_host_tiles(h, ck, want):
    P = tokens(1000, 370)
    done_ok(ck, 'H2: tokens unchanged (lossless chunking)', h.generate('ht', P, 4), P, 4, 0)
    ck.eq('H2: chunk sizes / upper_needed', chunks(h, 'ht'), want)


def run_batch(exe, td):
    ck = Checker('round3-batch')
    slow = {**BATCH, 'FAKE_PREFILL_MS': str(250 * SLOW)}
    for name, fn in (('basic', lambda h: sc_batch_basic(h, ck, 3)), ('cancel', lambda h: sc_batch_cancel(h, ck)),
                     ('share', lambda h: sc_batch_share(h, ck)), ('first', lambda h: sc_batch_first_token(h, ck))):
        env = {**slow, **(SHARE if name == 'share' else {})}
        h = Hived(exe, td, env, args=TILE3, name='b3' + name)
        try:
            fn(h)
        finally:
            stop_clean(h, ck)
    h = Hived(exe, td, {**slow, 'FAKE_FAIL_TOKEN': '7'}, args=TILE3, name='b3error')
    try:
        sc_batch_error(h, ck)
    finally:
        stop_clean(h, ck)
    # H3 + H2: 2 resident + 2 host slots -> 4 sequences in one forward
    h = Hived(exe, td, {**slow, 'HIVE_PREFILL_HOST_TILES': '1', 'FAKE_KV_SOURCE': '1'}, args=TILE2_TAIL, name='b3host')
    try:
        sc_batch_basic(h, ck, 3)  # max_batch 4 = blocker + 3
        ck.true('H2+H3 startup: prefill slots 4 (tile 2 + host 2)', 'prefill slots 4 (tile 2 + host 2) · text chunk cap 1024 · batch prefill on' in h.tail(100000))
    finally:
        stop_clean(h, ck)
    # H3 without a second slot: batch switch on but inactive -> the single path
    h = Hived(exe, td, slow, name='b3inactive')
    try:
        P = tokens(300, 380)
        done_ok(ck, 'H3 inactive (1 slot): request served by the single path', h.generate('bi', P, 3), P, 3, 0)
        ck.true('H3 inactive: startup says so', 'batch prefill on but inactive (needs >= 2 slots)' in h.tail(100000))
        ck.eq('H3 inactive: no batch rounds', batch_rounds(h), [])
    finally:
        stop_clean(h, ck)
    # H2 alone: FAKE_KV_SOURCE + tail args make tail_mode_for(max_chunk) true; 2 resident + 2 host slots = one 1000-row chunk
    for env, want in (({'HIVE_PREFILL_HOST_TILES': '1'}, [(1000, 'on')]),
                      ({'HIVE_PREFILL_HOST_TILES': '1', 'HIVE_PREFILL_HOST_MB': '0'}, [(512, 'skip'), (488, 'on')]),   # budget 0 = current behaviour
                      ({'HIVE_PREFILL_HOST_TILES': '1', 'HIVE_PREFILL_HOST_MB': 'x'}, [(512, 'skip'), (488, 'on')]),   # malformed = 0 (absorbed)
                      ({'HIVE_PREFILL_HOST_TILES': '1', 'HIVE_PREFILL_HOST_MB': '1.1'}, [(768, 'skip'), (232, 'on')]),  # 1.1 MiB = 1 fake tile (1 MiB)
                      ({'HIVE_PREFILL_HOST_TILES': '0'}, [(512, 'skip'), (488, 'on')]), ({}, [(512, 'skip'), (488, 'on')])):
        h = Hived(exe, td, {'FAKE_KV_SOURCE': '1', **env}, args=TILE2_TAIL, name='h2')
        try:
            sc_host_tiles(h, ck, want)
            on = env.get('HIVE_PREFILL_HOST_TILES') == '1' and env.get('HIVE_PREFILL_HOST_MB') not in ('0', 'x')
            ck.eq(f'H2 {env}: startup slots line', any('[hived] prefill slots' in l for l in h.tail(100000).splitlines()), on)
        finally:
            stop_clean(h, ck)
    return ck


# ---------------------------------------------------------------------------------------------------------
# P1: prefill scheduling on top of H3 — HIVE_BATCH_WINDOW(_MS) admission window, HIVE_BATCH_SJF slot
# sharing (a request joining a long batched prefill gets slots at the next round boundary), HIVE_PREFILL_FAIR decode share
# between back-to-back prefills. Scheduling only: every token is still the state oracle of its own prompt.
P1_ALL = {'HIVE_BATCH_WINDOW': '1', 'HIVE_BATCH_SJF': '1', 'HIVE_PREFILL_FAIR': '1'}
WINDOW_RE = re.compile(r'^\[hived\] batch window: \+(\d+) requests in (\d+) ms \((\w+) ·')


def windows(h):
    return [(int(m[1]), m[3]) for m in (WINDOW_RE.match(l) for l in h.tail(100000).splitlines()) if m]


def raw_generate(h, sid, ids, max_tokens):
    s = h.connect()
    s.sendall((json.dumps({'op': 'generate', 'session': sid, 'ids': ids, 'max_tokens': max_tokens, 'temperature': 0}) + '\n').encode())
    return s, s.makefile()


def sc_window(h, ck, on):
    """Staggered arrivals (0/40/80 ms) of three 1-slot prompts: with the window they share one forward, without it the first runs alone."""
    ids = {f'w{k}': tokens(200 + 3 * k, 400 + k) for k in range(3)}
    res, ths = {}, []
    for sid, v in ids.items():
        ths.append(launch(h, sid, v, 4, res, sid))
        time.sleep(.04 * SLOW)
    for t in ths: t.join(60 * SLOW)
    for sid, v in ids.items():
        done_ok(ck, f'P1a window={on} {sid}: tokens = its own oracle', res[sid], v, 4, 0)
    rounds = batch_rounds(h)
    if on:
        ck.true('P1a window: all three prefilled in one forward', any(n == 3 for n, _ in rounds), repr(rounds))
        ck.true('P1a window: log says +2 requests joined', any(n == 2 for n, _ in windows(h)), repr(windows(h)))
    else:
        ck.true('P1a no window: first request ran alone (control)', not any(n == 3 for n, _ in rounds) and windows(h) == [], repr(rounds))


def sc_window_skips(h, ck):
    """No wait when the head fills every slot (768 = 3 slots) or is shorter than the streaming floor (prefill_threshold 128)."""
    n0 = len(windows(h))
    for sid, v in (('wf', tokens(768, 410)), ('ws', tokens(60, 411))):
        done_ok(ck, f'P1a window skip {sid}', h.generate(sid, v, 3), v, 3, 0)
    ck.eq('P1a window: no wait for a slot-filling or short head', windows(h)[n0:], [])


def sc_window_decoder(h, ck):
    """An active decoder keeps stepping while the window waits (window 400 ms, no joiner arrives)."""
    D = tokens(100, 412)
    t_tok = []
    res = {}
    th = launch(h, 'wd', D, 600 * SLOW, res, 'wd', on_token=lambda n: t_tok.append(time.monotonic()))
    deadline = time.monotonic() + 10 * SLOW
    while len(t_tok) < 5 and time.monotonic() < deadline:
        time.sleep(.01)
    H = tokens(300, 413)
    t0 = time.monotonic()
    r = h.generate('wh', H, 3)
    t1 = time.monotonic()
    th.join(60 * SLOW)
    done_ok(ck, 'P1a window with a decoder: head', r, H, 3, 0)
    done_ok(ck, 'P1a window with a decoder: decoder', res['wd'], D, 600 * SLOW, 0)
    during = sum(1 for t in t_tok if t0 < t < t1)
    ck.true('P1a window: waited the full window (nobody joined)', any(w == (0, 'window') for w in windows(h)), repr(windows(h)))
    ck.true('P1a window: decoder kept stepping during the window', during >= 20, f'{during} tokens')


def sc_sjf_join(h, ck, on):
    """A 200-row request (and a 60-row short one) arriving during round 1 of a 2000-row prefill (768/round): with slot sharing
    both are served at the next round boundary; without it they wait for the long prefill's slots."""
    H, J, S = tokens(2000, 420), tokens(200, 421), tokens(60, 422)
    t_first, res, ths = {}, {}, []
    ths.append(launch(h, 'hd', H, 3, res, 'hd', on_token=lambda n: t_first.setdefault('hd', time.monotonic())))
    time.sleep(.12 * SLOW)  # hd is inside its first 768-row forward
    for sid, v in (('jn', J), ('st', S)):
        ths.append(launch(h, sid, v, 3, res, sid, on_token=lambda n, sid=sid: t_first.setdefault(sid, time.monotonic())))
        time.sleep(.005)
    for t in ths: t.join(60 * SLOW)
    for sid, v in (('hd', H), ('jn', J), ('st', S)):
        done_ok(ck, f'P1b sjf={on} {sid}: tokens = its own oracle', res[sid], v, 3, 0)
    rounds = batch_rounds(h)
    if on:
        ck.eq('P1b: head chunks 768 / 512 (1 slot given to the joiner) / 720', [m for m, _ in chunks(h, 'hd')], [768, 512, 720])
        ck.true('P1b: joiner shares round 2 with the head', any('hd:512' in who and 'jn:200' in who for n, who in rounds), repr(rounds))
        ck.true('P1b: joiner and short request stream before the long prefill ends',
                all(k in t_first for k in ('hd', 'jn', 'st')) and t_first['hd'] - max(t_first['jn'], t_first['st']) > .15 * SLOW, repr(t_first))
    else:
        ck.eq('P1b control: head keeps every slot (768 / 768 / 464)', [m for m, _ in chunks(h, 'hd')], [768, 768, 464])
        ck.true('P1b control: joiner waits for the head\'s last round', any('hd:464' in who and 'jn:200' in who for n, who in rounds), repr(rounds))
    # continuation of the joiner reuses its live state
    nxt = J + res['jn']['tokens'] + tokens(10, 423)
    done_ok(ck, f'P1b sjf={on} joiner turn 2', h.generate('jn', nxt, 3), nxt, 3, len(J) + 2)


def sc_sjf_reserve(h, ck):
    """Joiners needing 2 and 1 slots arrive during the head's round 1: the oldest request (the head) keeps one slot per round."""
    H, J1, J2 = tokens(2000, 425), tokens(400, 426), tokens(200, 427)
    res, ths = {}, []
    ths.append(launch(h, 'hr', H, 3, res, 'hr'))
    time.sleep(.12 * SLOW)
    for sid, v in (('j1', J1), ('j2', J2)):
        ths.append(launch(h, sid, v, 3, res, sid))
        time.sleep(.005)
    for t in ths: t.join(60 * SLOW)
    for sid, v in (('hr', H), ('j1', J1), ('j2', J2)):
        done_ok(ck, f'P1b reserve {sid}: tokens = its own oracle', res[sid], v, 3, 0)
    ck.eq('P1b reserve: head gets one slot in round 2 (768 / 256 / 512 / 464)', [m for m, _ in chunks(h, 'hr')], [768, 256, 512, 464])
    ck.true('P1b reserve: round 2 = head 256 + 2-slot joiner', any('hr:256' in who and 'j1:400' in who for n, who in batch_rounds(h)), repr(batch_rounds(h)))


def sc_sjf_cancel(h, ck):
    """The long head is cancelled after its first round while a joiner shares round 2: the joiner is unaffected, the head ends 'cancel'."""
    H, J = tokens(2000, 430), tokens(200, 431)
    s, f = raw_generate(h, 'hc', H, 50)
    time.sleep(.12 * SLOW)  # hc is inside its first round: the joiner is queued for round 2
    res = {}
    th = launch(h, 'jc', J, 4, res, 'jc')
    first = json.loads(f.readline() or '{}')  # progress after round 1 -> cancel lands during round 2
    ck.true('P1b cancel: head first progress', 'progress' in first, repr(first))
    ck.eq('P1b cancel op', h.cancel('hc'), {'ok': True})
    last = None
    for line in f:
        last = json.loads(line)
        if last.get('done') or last.get('error'):
            break
    f.close(); s.close()
    th.join(60 * SLOW)
    ck.true('P1b cancel: head finish=cancel', last and last.get('finish') == 'cancel', repr(last))
    done_ok(ck, 'P1b cancel: joiner unaffected', res.get('jc') or NO_REPLY, J, 4, 0)
    ck.true('P1b cancel: joiner shared a round with the cancelled head', any('hc:' in who and 'jc:' in who for n, who in batch_rounds(h)), repr(batch_rounds(h)))
    again = H[:900] + tokens(50, 432)
    done_ok(ck, 'P1b cancel: cancelled session usable afterwards', h.generate('hc', again, 3), again, 3, None)


def sc_sjf_error(h, ck):
    """A joiner whose prefill throws (token 7) fails alone at the round it joined; the long head and a third request stay correct."""
    H, E, K = tokens(2000, 440), tokens(100, 441) + [7] + tokens(99, 442), tokens(210, 443)
    res, ths = {}, []
    ths.append(launch(h, 'he', H, 3, res, 'he'))
    time.sleep(.12 * SLOW)
    for sid, v in (('ee', E), ('ke', K)):
        ths.append(launch(h, sid, v, 3, res, sid))
        time.sleep(.005)
    for t in ths: t.join(60 * SLOW)
    ck.eq('P1b error: failing joiner gets the single-path error', (res.get('ee') or NO_REPLY)['error'], 'request failed: fake injected forward failure')
    done_ok(ck, 'P1b error: long head correct after the solo retry', res.get('he') or NO_REPLY, H, 3, None)
    done_ok(ck, 'P1b error: third request correct', res.get('ke') or NO_REPLY, K, 3, None)
    ck.true('P1b error: failure was inside a joined round', log_lines(h, 'batch prefill failed: fake injected forward failure'))


def sc_fair(h, ck, on, gap=.005):
    """A decoder is running; three single-round prefills arrive back to back. With the fair gap the decoder gets decode_share of
    wall time between them (not one step per prefill)."""
    D = tokens(100, 450)
    t_tok, res = [], {}
    th = launch(h, 'fd', D, 3000, res, 'fd', on_token=lambda n: t_tok.append(time.monotonic()))
    deadline = time.monotonic() + 10 * SLOW
    while len(t_tok) < 5 and time.monotonic() < deadline:
        time.sleep(.01)
    ps = {f'f{k}': tokens(200, 451 + k) for k in range(3)}
    t_done, ths = {}, []
    for sid, v in ps.items():
        ths.append(launch(h, sid, v, 1, res, sid, on_token=lambda n, sid=sid: t_done.setdefault(sid, time.monotonic())))
        time.sleep(gap)  # batch on: .15 s = each arrives during the previous (final-round) forward, so it is a separate admission
    for t in ths: t.join(60 * SLOW)
    h.cancel('fd')
    th.join(60 * SLOW)
    for sid, v in ps.items():
        done_ok(ck, f'P1c fair={on} {sid}', res[sid], v, 1, 0)
    ck.true(f'P1c fair={on}: decoder ended by cancel', (res['fd']['done'] or {}).get('finish') == 'cancel', repr(res['fd']['done']))
    ck.eq(f'P1c fair={on}: decoder tokens follow its oracle', res['fd']['tokens'], oracle(D, len(res['fd']['tokens'])))
    a, b = min(t_done.values()), max(t_done.values())
    during = sum(1 for t in t_tok if a < t < b)
    if on:
        ck.true('P1c fair: decoder stepped between the back-to-back prefills', during >= 20, f'{during} tokens between first and last prefill')
    else:
        ck.true('P1c control: about one decode step per prefill', during <= 8, f'{during} tokens')
    return during


def run_p1(exe, td):
    ck = Checker('round10-p1')
    slow = {**BATCH, 'FAKE_PREFILL_MS': str(250 * SLOW)}
    win = {'HIVE_BATCH_WINDOW': '1', 'HIVE_BATCH_WINDOW_MS': str(400 * SLOW)}
    for name, env, fn in (
            ('win', {**slow, **win}, lambda h: (sc_window(h, ck, True), sc_window_skips(h, ck), sc_window_decoder(h, ck))),
            ('nowin', slow, lambda h: sc_window(h, ck, False)),
            ('winms0', {**slow, 'HIVE_BATCH_WINDOW': '1', 'HIVE_BATCH_WINDOW_MS': '0'}, lambda h: sc_window(h, ck, False)),
            ('sjf', {**slow, 'HIVE_BATCH_SJF': '1'}, lambda h: (sc_sjf_join(h, ck, True), sc_sjf_reserve(h, ck), sc_sjf_cancel(h, ck))),
            ('nosjf', slow, lambda h: sc_sjf_join(h, ck, False)),
            ('sjferr', {**slow, 'HIVE_BATCH_SJF': '1', 'FAKE_FAIL_TOKEN': '7'}, lambda h: sc_sjf_error(h, ck)),
            ('all', {**slow, **P1_ALL, 'HIVE_BATCH_WINDOW_MS': str(100 * SLOW)}, lambda h: (sc_sjf_join(h, ck, True), sc_batch_cancel(h, ck))),
            ('fair', {'FAKE_PREFILL_MS': str(250 * SLOW), 'HIVE_PREFILL_FAIR': '1'}, lambda h: sc_fair(h, ck, True)),
            ('nofair', {'FAKE_PREFILL_MS': str(250 * SLOW)}, lambda h: sc_fair(h, ck, False)),
            ('fairbatch', {**slow, 'HIVE_PREFILL_FAIR': '1', 'HIVE_BATCH_SJF': '1'}, lambda h: sc_fair(h, ck, True, gap=.15 * SLOW))):
        h = Hived(exe, td, env, args=TILE3, name='p1' + name)
        try:
            fn(h)
            if name in ('win', 'sjf', 'all', 'fair'):
                ck.true(f'P1 {name}: startup line', '[hived] prefill scheduling:' in h.tail(100000))
            if name in ('nowin', 'nosjf', 'nofair'):
                ck.true(f'P1 {name}: switches off = no startup line', '[hived] prefill scheduling:' not in h.tail(100000))
        finally:
            stop_clean(h, ck)
    return ck


# H2 host-side machinery: PRODUCTION tile_plan/host_tile_count/plan_units/HostStager on fake CUDA streams (real threads with
# FAKE_CUDA_ASYNC=1). Each stager mutant removes one ordering wait and MUST produce a wrong value.
STAGER_MUTANTS = [
    ("H2D does not wait for the tile's previous D2H", '      CUDA_CHECK(cudaStreamWaitEvent(up_, stored_[(size_t)k], 0));\n'),
    ('compute does not wait for the H2D', '    CUDA_CHECK(cudaStreamWaitEvent(st_, loaded_[(size_t)b], 0));\n'),
    ('D2H does not wait for the compute', '      CUDA_CHECK(cudaStreamWaitEvent(down_, done_[(size_t)b], 0));\n'),
    ("H2D does not wait for the buffer's previous D2H", '    if (free_rec_[(size_t)b]) CUDA_CHECK(cudaStreamWaitEvent(up_, free_[(size_t)b], 0));\n'),
]


def build_host_tiles(td, name, text):
    d = Path(td) / name
    d.mkdir(exist_ok=True)
    (d / 'host_tiles_gen.h').write_text(text)
    return harness.build(str(d), 'host_tiles', '', [harness.FAKE / 'test_host_tiles.cpp'], extra=['-I' + str(d)])


def run_host_tiles(td):
    ck = Checker('round3-host-tiles')
    base = harness.host_tiles_source()
    exe = build_host_tiles(td, 'ht', base)
    for mode in ({'FAKE_CUDA_ASYNC': '0'}, {'FAKE_CUDA_ASYNC': '1', 'FAKE_CUDA_DELAY_US': '40'}):
        r = subprocess.run([exe], env={**harness.tsan_env(), **mode}, capture_output=True, text=True, timeout=300 * SLOW)
        ck.true(f'host tiles CPU test {mode}', r.returncode == 0, (r.stdout + r.stderr)[-2000:])
    if harness.sanitizer() != 'thread':  # mutants: value checks (TSAN run above already covers the races of the real text)
        for i, (name, line) in enumerate(STAGER_MUTANTS):
            assert base.count(line) == 1, f'stager mutant anchor drifted: {name}'
            mexe = build_host_tiles(td, f'htm{i}', base.replace(line, ''))
            r = subprocess.run([mexe, 'stress'], env={**os.environ, 'FAKE_CUDA_ASYNC': '1', 'FAKE_CUDA_DELAY_US': '40'}, capture_output=True, text=True, timeout=300)
            ck.true(f'stager mutant [{name}] detected', r.returncode != 0, (r.stdout + r.stderr)[-500:])
    return ck


OFF = {k: '0' for k in ('HIVE_MTP_CACHE', 'HIVE_PHASE_SCORE', 'HIVE_PROMOTE_SCORE', 'HIVE_PROMOTE_BYTES', 'HIVE_CACHE_REUSE_STAGE',
                        'HIVE_CKPT_DELTA', 'HIVE_CKPT_ASYNC', 'HIVE_CKPT_PINNED_POOL_MB', 'HIVE_IMAGE_CKPT', 'HIVE_PREFILL_BUDGET_MS',
                        'HIVE_TILE_GROUP_GEMM', 'HIVE_PREFETCH', 'HIVE_PREFIX_SHARE', 'HIVE_DEFER_CKPT', 'HIVE_GRACEFUL_STOP_S',
                        'HIVE_BATCH_PREFILL', 'HIVE_PREFILL_HOST_TILES', 'HIVE_PREFILL_HOST_MB',
                        'HIVE_BATCH_WINDOW', 'HIVE_BATCH_WINDOW_MS', 'HIVE_BATCH_SJF', 'HIVE_PREFILL_FAIR')}
OFF['HIVE_CKPT_HISTORY'] = ''
OFF_EMPTY = {k: '' for k in OFF}
OFF_EMPTY['HIVE_CKPT_HISTORY'] = '0'
CONFIGS = {
    'default': {},
    'warm-busy-cap': {'HIVE_WARM_BUSY_CAP': '1'},
    'warm-defer': {'HIVE_WARM_DEFER': '32'},  # post-prefill warm handed to the decode steps (banner checked in run_config)  # synchronous warm after a prefill capped while others decode (log line checked in run_config)
    'all-off-explicit': OFF,
    'all-off-empty': OFF_EMPTY,
    'ckpt-async-pool-delta-history-image': {'HIVE_CKPT_DELTA': '1', 'HIVE_CKPT_ASYNC': '1', 'HIVE_CKPT_PINNED_POOL_MB': '64',
                                            'HIVE_CKPT_HISTORY': '3', 'HIVE_IMAGE_CKPT': '1', 'FAKE_CUDA_ASYNC': '1', 'FAKE_CUDA_DELAY_US': '100'},
    'ckpt-async-no-pool': {'HIVE_CKPT_ASYNC': '1', 'HIVE_CKPT_HISTORY': '2', 'FAKE_CUDA_ASYNC': '1', 'FAKE_CUDA_DELAY_US': '100'},
    'ckpt-async-tiny-pool': {'HIVE_CKPT_ASYNC': '1', 'HIVE_CKPT_PINNED_POOL_MB': '1', 'HIVE_CKPT_DELTA': '1', 'FAKE_CUDA_ASYNC': '1'},
    # deferred checkpoint / prefix share: must answer exactly like their reference config (compared below)
    'defer-ckpt': {'HIVE_DEFER_CKPT': '1'},
    'defer-ckpt-async-pool-delta-history-image': {'HIVE_DEFER_CKPT': '1', 'HIVE_CKPT_DELTA': '1', 'HIVE_CKPT_ASYNC': '1', 'HIVE_CKPT_PINNED_POOL_MB': '64',
                                                  'HIVE_CKPT_HISTORY': '3', 'HIVE_IMAGE_CKPT': '1', 'FAKE_CUDA_ASYNC': '1', 'FAKE_CUDA_DELAY_US': '100'},
    'prefix-share-no-hints': {'HIVE_PREFIX_SHARE': '1'},
    # H3 batch prefill / H2 host tiles on the standard scenarios — lossless chunking/batching, so the
    # responses must equal the same-args reference with the switches off (sc_concurrent's 7 users are batched here)
    'tile3': {},
    'batch-prefill': {'HIVE_BATCH_PREFILL': '1'},
    'tile2-tail': {'FAKE_KV_SOURCE': '1'},
    'batch-host-tiles': {'HIVE_BATCH_PREFILL': '1', 'HIVE_PREFILL_HOST_TILES': '1', 'FAKE_KV_SOURCE': '1'},
    # P1: scheduling-only switches on the standard scenarios — same responses as the tile3 / tile2-tail reference
    'batch-p1': {'HIVE_BATCH_PREFILL': '1', 'HIVE_BATCH_WINDOW': '1', 'HIVE_BATCH_WINDOW_MS': '30', 'HIVE_BATCH_SJF': '1', 'HIVE_PREFILL_FAIR': '1'},
    'batch-p1-host-tiles': {'HIVE_BATCH_PREFILL': '1', 'HIVE_PREFILL_HOST_TILES': '1', 'FAKE_KV_SOURCE': '1', 'HIVE_BATCH_WINDOW': '1',
                            'HIVE_BATCH_WINDOW_MS': '30', 'HIVE_BATCH_SJF': '1', 'HIVE_PREFILL_FAIR': '1'},
}
TILE3 = ('--prefill-tile', '3', '--prefill-threshold', '128')
TILE2_TAIL = ('--prefill-tile', '2', '--prefill-threshold', '128', '--decoder-tail', '100')
CONFIG_ARGS = {'tile3': TILE3, 'batch-prefill': TILE3, 'tile2-tail': TILE2_TAIL, 'batch-host-tiles': TILE2_TAIL,
               'batch-p1': TILE3, 'batch-p1-host-tiles': TILE2_TAIL}
SAME_AS = {'all-off-explicit': 'default', 'all-off-empty': 'default', 'defer-ckpt': 'default', 'prefix-share-no-hints': 'default',
           'defer-ckpt-async-pool-delta-history-image': 'ckpt-async-pool-delta-history-image',
           'batch-prefill': 'tile3', 'batch-host-tiles': 'tile2-tail', 'batch-p1': 'tile3', 'batch-p1-host-tiles': 'tile2-tail'}


def run_negative(exe, td, kind):
    """Negative controls: must FAIL. Returns (detected, detail)."""
    if kind == 'fence':  # fake forward skips the production snapshot-fence wait: device writes race the async D2H
        cfg = {**CONFIGS['ckpt-async-pool-delta-history-image'], 'FAKE_SKIP_SNAPSHOT_FENCE': '1', 'FAKE_CUDA_DELAY_US': '2000'}
    else:
        raise ValueError(kind)
    try:
        ck, _ = run_config(exe, td, 'negative-' + kind, cfg)
    except (AssertionError, OSError) as e:  # daemon died (TSAN exit 66) or stopped serving
        log = sorted(Path(td).glob(f'negative-{kind}*.log'))[-1].read_text(errors='replace')
        i = log.find('WARNING: ThreadSanitizer')
        return i >= 0, (log[i:i + 3000] if i >= 0 else repr(e) + log[-2000:])
    return bool(ck.failures), '\n'.join(ck.failures[:5])


MUTANTS = [  # (name, production text, mutated text, runner) — built in a temp dir only; each MUST fail
    ('image signature ignored', 'std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());return a==b;', 'return true;', 'images'),
    ('broken flag ignored', 'const bool live_ok = S->seq && !S->seq->broken && S->seq->pos', 'const bool live_ok = S->seq && S->seq->pos', 'failures'),
    ('live prefix not required to cover the session', 'if (live_ok && live_lcp == S->tokens.size() && ids.size() > live_lcp',
     'if (live_ok && live_lcp > 0 && ids.size() > live_lcp', 'session'),
    ('checkpoint position not required', 'if (k == (size_t)S->ckpt->pos && (ids.size() > k || exact))', 'if (k > 0 && (ids.size() > k || exact))', 'session'),
    # prefill chunking: each old behaviour must be caught by run_extras' scenarios
    ('D3: next chunk sized by budget/busy without a decoder', 'const size_t next_min = rem_after > 0 ? plan_chunk(i + (size_t)M, next_cap) : 0;',
     'const size_t next_min = std::min<size_t>(rem_after, prefill_budget_ms>0 ? std::min<size_t>(chunk_cap,1024) : busy_chunk > 0 ? '
     'std::min<size_t>(chunk_cap, busy_chunk) : chunk_cap);', 'd3'),
    ('D9: revived session not protected from reclamation', 'if (evicted) archive_trim(A->sid);', 'if (evicted) archive_trim("");', 'd9'),
    ('stats not published at request finish', 'publish_stats(true);  // done', '(void)0;  // done', 'stats'),
    ('D5 rid cancel falls back to session-latest', 'if (!rid.empty()) {  // rid', 'if (false) {  // rid', 'rid'),
    # reuse paths: every reuse condition of the new paths
    ('H5 shared snapshot token prefix not compared',
     '|| !std::equal(e.image->tokens.begin(), e.image->tokens.end(), ids.begin())) continue;', ') continue;', 'share'),
    ('H5 shared snapshot image signature ignored', 'if (!images_match(e.sig, k)) continue;', '', 'share'),
    ('H5 C1 flag ignored at restore', 'if (e.upper_skipped && !(all_images.empty()', 'if (false && !(all_images.empty()', 'share_c1'),
    ('H5 C1 flag not recorded at save', 'snapshot_boundary(i, !upper_needed);', 'snapshot_boundary(i, false);', 'share_c1'),
    ('H5 boundary cut ignores the previous skipped chunk', 'if (last_upper_skipped && !rt.tail_mode_for((int)(h - i))) continue;', '', 'share_c1'),
    ('H5 restored skip flag not carried into the first chunk', 'last_upper_skipped = best->upper_skipped;', '(void)0;', 'share_c1'),
    ('M3 suffix images dropped with the prefix ones', 'if ((size_t)x.start + x.types.size() > common) suffix.push_back(x);', '', 'm3'),
    ('M8 deferred checkpoint not taken before the first step',
     'if (defer_ckpt) for (auto& A : active) if (A->defer_ckpt || A->defer_warm) run_deferred(*A);', '', 'defer'),
    ('graceful: deadline cancel reported as a normal done', 'if (stop_state.load() >= 2 && A.finish == "cancel")', 'if (false)', 'graceful'),
    # H3: the batch planner must respect the round's slot budget (the fake forward_multi aborts on overflow)
    ('H3 round slot budget not charged', 'slots -= rt.slots_for(J.M);', '(void)0;', 'batch'),
    ('H3 member chunk not bounded by the round slots', 'const size_t chunk_now = std::min(cap_now(), lim);', 'const size_t chunk_now = cap_now();', 'batch'),
    # P1: a joiner must get slots at the next round; the window must keep decoders stepping
    ('P1b slot share disabled (FIFO only)', '(rem < stream_floor ? shorts : rem <= cap_now_j(J) ? fins : rest).push_back(j);', 'rest.push_back(j);', 'p1sjf'),
    ('P1b oldest request not reserved a slot', 'const int reserve = !rest.empty() && rest.front() == 0 ? 1 : 0;', 'const int reserve = 0;', 'p1sjf'),
    ('P1a window stalls decoders', 'if (!active.empty()) decode_step();  // when there are decoders', 'if (false) decode_step();  // when there are decoders', 'p1win'),
    ('P1c fair gap skipped', 'if (more) decode_between(last_prefill_tc);', '(void)more;', 'p1fair'),
    ('H2 text chunk cap ignores the host tiles', '(host tiles included — equals prefill_tile when off)\n    chunk_cap = all_images.empty() ? (size_t)max_chunk * (size_t)rt.prefill_slots()',
     '(host tiles included — equals prefill_tile when off)\n    chunk_cap = all_images.empty() ? (size_t)max_chunk * (size_t)rt.prefill_tile()', 'h2'),
]


def run_mutants(td):
    src = (harness.ROOT / 'engine/src/hived.cpp').read_text()
    out = []
    only = os.environ.get('HIVE_DAEMON_MUTANTS', '')  # optional name-prefix filter (e.g. "P1") — unset = all
    for i, (name, old, new, runner) in enumerate(MUTANTS):
        assert src.count(old) == 1, f'mutant anchor drifted: {name}'
        if only and not name.startswith(only):
            continue
        exe = build_daemon(td, f'mutant{i}', src.replace(old, new))
        try:
            if runner == 'failures':
                ck = run_failures(exe, td)
            elif runner == 'graceful':
                ck = run_graceful(exe, td)
            elif runner in ('share', 'share_c1', 'm3', 'defer'):
                ck = Checker(name)
                if runner == 'share':
                    h = Hived(exe, td, {**SHARE, 'HIVE_IMAGE_CKPT': '1'}, name=f'mutant{i}')
                    sc_share_cross(h, ck); sc_share_image(h, ck)
                elif runner == 'share_c1':
                    h = Hived(exe, td, {**SHARE, 'FAKE_KV_SOURCE': '1'}, args=TAIL_ARGS, name=f'mutant{i}')
                    sc_share_c1(h, ck)
                elif runner == 'm3':
                    h = Hived(exe, td, {'HIVE_IMAGE_CKPT': '1'}, args=('--prefill-tile', '2'), name=f'mutant{i}')
                    sc_image_turns(h, ck, True)
                else:
                    h = Hived(exe, td, {'HIVE_DEFER_CKPT': '1'}, name=f'mutant{i}')
                    sc_session(h, ck, {})
                h.stop()
            elif runner in ('batch', 'h2'):
                ck = Checker(name)
                if runner == 'batch':
                    h = Hived(exe, td, {**BATCH, 'FAKE_PREFILL_MS': str(250 * SLOW)}, args=TILE3, name=f'mutant{i}')
                    sc_batch_cancel(h, ck)
                else:
                    h = Hived(exe, td, {'FAKE_KV_SOURCE': '1', 'HIVE_PREFILL_HOST_TILES': '1'}, args=TILE2_TAIL, name=f'mutant{i}')
                    sc_host_tiles(h, ck, [(1000, 'on')])
                h.stop()
            elif runner in ('p1sjf', 'p1win', 'p1fair'):
                ck = Checker(name)
                slow = {**BATCH, 'FAKE_PREFILL_MS': str(250 * SLOW)}
                if runner == 'p1sjf':
                    h = Hived(exe, td, {**slow, 'HIVE_BATCH_SJF': '1'}, args=TILE3, name=f'mutant{i}')
                    sc_sjf_join(h, ck, True); sc_sjf_reserve(h, ck)
                elif runner == 'p1win':
                    h = Hived(exe, td, {**slow, 'HIVE_BATCH_WINDOW': '1', 'HIVE_BATCH_WINDOW_MS': str(400 * SLOW)}, args=TILE3, name=f'mutant{i}')
                    sc_window_decoder(h, ck)
                else:
                    h = Hived(exe, td, {'FAKE_PREFILL_MS': str(250 * SLOW), 'HIVE_PREFILL_FAIR': '1'}, args=TILE3, name=f'mutant{i}')
                    sc_fair(h, ck, True)
                h.stop()
            elif runner in ('d3', 'd9', 'stats', 'rid'):
                ck = Checker(name)
                if runner == 'd3':
                    h = Hived(exe, td, {'FAKE_KV_SOURCE': '1', 'HIVE_PREFILL_BUDGET_MS': '50'}, args=TAIL_ARGS, name=f'mutant{i}')
                    sc_d3(h, ck, [(1536, 'skip'), (1536, 'on'), (428, 'on')])
                elif runner == 'd9':
                    mb = d9_budget_mb(exe, td)
                    h = Hived(exe, td, args=('--max-batch', '1', '--max-sessions', '2', '--host-session-mb', f'{mb:.6f}'), name=f'mutant{i}')
                    sc_d9(h, ck, False)
                elif runner == 'rid':
                    h = Hived(exe, td, name=f'mutant{i}')
                    sc_rid_cancel(h, ck)
                else:
                    h = Hived(exe, td, name=f'mutant{i}')
                    sc_stats_fresh(h, ck)
                h.stop()
            else:
                ck = Checker(name)
                h = Hived(exe, td, name=f'mutant{i}')
                (sc_images if runner == 'images' else sc_session)(h, ck, {})
                h.stop()
            detected, detail = bool(ck.failures), (ck.failures[:1] or [''])[0]
        except (AssertionError, OSError) as e:
            detected, detail = True, 'daemon died: ' + str(e)[:200]
        out.append((name, detected, detail))
    return out


def run_step_host(exe, td):
    """HIVE_PROFILE=N [step-host <kind>] lines (hived.cpp decode_step wrapper): one line per N steps, the monitor's format, outputs unchanged,
    and a gap that spans another request's prefill is not counted (prefill_epoch) — with a 300 ms fake prefill admitted while a request
    decodes, a counted gap that large means the exclusion is broken. No lines without HIVE_PROFILE."""
    import hive_monitor as hm
    ck = Checker('step-host')
    pre_ms = 300 * SLOW
    h = Hived(exe, td, {'HIVE_PROFILE': '4', 'FAKE_PREFILL_MS': str(pre_ms)}, name='stephost')
    A, B = tokens(40, 11), tokens(60, 12)
    out = {}
    t = threading.Thread(target=lambda: out.__setitem__('a', h.generate('sa', A, 160)))
    t.start()
    time.sleep(pre_ms / 1000 + .15 * SLOW)  # a is decoding when b's prefill runs
    done_ok(ck, 'b (admitted while a decodes)', h.generate('sb', B, 8), B, 8, 0)
    t.join()
    done_ok(ck, 'a (decoding across b\'s prefill)', out['a'], A, 160, 0)
    h.stop()
    lines = [l for l in h.log_path.read_text(errors='replace').splitlines() if l.startswith('[step-host')]
    ms = [hm.RE['step_host'].match(l) for l in lines]
    ck.checks += 1
    if not lines or not all(ms):
        ck.failures.append(f'[step-host] lines missing or not in the monitor format: {lines[:3]}')
    else:
        ck.checks += 3
        if any(int(m[2]) != 4 for m in ms):
            ck.failures.append('a [step-host] line does not cover HIVE_PROFILE=4 steps')
        if any(int(m[4]) > int(m[2]) for m in ms):
            ck.failures.append('more counted gaps than steps')
        worst = max(float(m[3]) * int(m[4]) for m in ms)  # the line's counted gap total (a mean over 4 steps would hide one 300 ms gap)
        if worst >= pre_ms / 2:
            ck.failures.append(f'a gap spanning a prefill was counted (gaps of one line total {worst:.1f} ms ≥ {pre_ms / 2:.0f} ms)')
    h2 = Hived(exe, td, name='stephost-off')
    done_ok(ck, 'no HIVE_PROFILE', h2.generate('s', A, 12), A, 12, 0)
    h2.stop()
    ck.checks += 1
    if any(l.startswith('[step-host') for l in h2.log_path.read_text(errors='replace').splitlines()):
        ck.failures.append('[step-host] printed without HIVE_PROFILE')
    return ck


def main():
    if os.environ.get('HIVE_DAEMON_NEGATIVE') == 'mutants':
        with tempfile.TemporaryDirectory(prefix='hive-daemon-mut-') as td:
            res = run_mutants(td)
        for name, detected, detail in res:
            print(f'  mutant [{name}]: {"DETECTED" if detected else "NOT DETECTED"} — {detail[:160]}')
        sys.exit(0 if all(d for _, d, _ in res) else 1)
    if os.environ.get('HIVE_DAEMON_NEGATIVE'):
        kind = os.environ['HIVE_DAEMON_NEGATIVE']
        with tempfile.TemporaryDirectory(prefix='hive-daemon-neg-') as td:
            exe = build_daemon(td)
            detected, detail = run_negative(exe, td, kind)
            print(detail)
            print(f'negative control {kind}: {"DETECTED (expected)" if detected else "NOT DETECTED"}')
            sys.exit(0 if detected else 1)
    names = [n for n in os.environ.get('HIVE_DAEMON_CONFIGS', ','.join(CONFIGS)).split(',') if n]
    extra = os.environ.get('HIVE_DAEMON_EXTRA', 'failures,sigterm,review,round2,graceful,batch,p1,hosttiles,stephost').split(',')
    with tempfile.TemporaryDirectory(prefix='hive-daemon-test-') as td:
        t0 = time.monotonic()
        exe = build_daemon(td)
        print(f'built real hived.cpp on fake CUDA/runtime in {time.monotonic() - t0:.1f}s (sanitizer {harness.sanitizer()})')
        checkers = []
        for n in names:
            ck, nstats = run_config(exe, td, n, CONFIGS[n])
            checkers.append(ck)
            print(f'  {n}: {ck.checks} checks, {len(ck.failures)} failures, {nstats} concurrent stats replies')
        by = {ck.name: ck for ck in checkers}
        # every option off = byte-identical responses to the default; round-2 options without their trigger (no hints) or
        # with a pure reordering (deferred checkpoint) = identical to their reference config
        for off, ref in SAME_AS.items():
            if off in by and ref in by:
                same = by[off].transcript == by[ref].transcript
                by[off].checks += 1
                if not same:
                    by[off].failures.append(f'transcript differs from {ref}: ' + repr(
                        [(a, b) for a, b in zip(by[off].transcript, by[ref].transcript) if a != b][:2]))
                print(f'  {off}: responses identical to {ref}: {same} ({len(by[off].transcript)} requests)')
        if 'failures' in extra:
            checkers.append(run_failures(exe, td))
        if 'sigterm' in extra:
            checkers.append(run_sigterm_inflight(exe, td))
        if 'review' in extra:
            checkers.append(run_extras(exe, td))
        if 'round2' in extra:
            checkers.append(run_round2(exe, td))
        if 'graceful' in extra:
            checkers.append(run_graceful(exe, td))
        if 'batch' in extra:
            checkers.append(run_batch(exe, td))
        if 'p1' in extra:
            checkers.append(run_p1(exe, td))
        if 'hosttiles' in extra:
            checkers.append(run_host_tiles(td))
        if 'stephost' in extra:
            checkers.append(run_step_host(exe, td))
        for ck in checkers[len(names):]:
            print(f'  {ck.name}: {ck.checks} checks, {len(ck.failures)} failures')
        bad = [(ck.name, f) for ck in checkers for f in ck.failures]
        for n, f in bad:
            print(f'FAIL [{n}] {f}')
        if bad:
            sys.exit(1)
        print(f'  connect EAGAIN retries (backlog full): {Hived.eagain}')
        print('daemon CPU: session reuse/edit/regen/history, image invalidation, cancel/regenerate/busy, concurrent sessions + '
              'archive restore, disconnect/fd release, host+decode exception recovery, concurrent stats, SIGTERM in flight, '
              'round 2 (prefix share/C1 flag/boundary cuts, image-turn reuse, deferred ckpt, prefill_pending, graceful stop) OK')


if __name__ == '__main__':
    main()
