#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Cold load HIVE_LOAD_PAR — CPU checks (no GPU, no service, no model weights).

engine/tests/test_load_par_cpu.cpp on fake CUDA/NUMA with the REAL safetensors.cpp (real files, mmap), the REAL ExpertStore
(engine/src/expert_store.cpp) and hive/bulk_load.h:
  * bulk_load == pread reference for random segments (O_DIRECT and buffered, 1/3/8 threads, 4 KiB..32 MiB chunks, past-EOF / missing file → false)
  * ExpertStore::load_bulk == the per-layer path (load_layer_experts + pin_all + load_engram): every half record (incl. alignment tail),
    every engram table and its metadata — 5 option variants, plus a failed bulk load that falls back to the per-layer path
  * DensePrefetch filter (no experts / engram tables / vision) and byte count
  * switch parsing (HIVE_LOAD_PAR · HIVE_LOAD_CHUNK_MB · HIVE_LOAD_BUFFERED)
Daemon: the REAL engine/src/hived.cpp on the fake runtime with HIVE_LOAD_PAR=1 (the fake checkpoint has no shard files, so the bulk load
fails and the store reloads with the per-layer path — the absorb path) serves the same session transcript as the default daemon,
logs the fallback and the `load phases` line.
Negative control: a mutant load_bulk that swaps the two halves of w2 MUST fail the unit.
The work dir must support O_DIRECT for the O_DIRECT assertions (tmpfs does not) — /tmp is tried first, then a dir next to the repo.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake')]
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import harness  # noqa: E402
import test_daemon_cpu as D  # noqa: E402

ROOT = harness.ROOT
STORE = ROOT / 'engine/src/expert_store.cpp'


def direct_ok(d):
    p = Path(d) / 'probe'
    p.write_bytes(b'\0' * 8192)
    try:
        fd = os.open(p, os.O_RDONLY | os.O_DIRECT)
        os.close(fd)
        return True
    except OSError:
        return False
    finally:
        p.unlink()


def build(td, name, store):
    return harness.build(td, name, '', [ROOT / 'engine/tests/test_load_par_cpu.cpp', store, ROOT / 'engine/src/cpu/expert_cpu.cpp',
                                        ROOT / 'engine/src/safetensors.cpp'], extra=['-D_GLIBCXX_ASSERTIONS'])


def main():
    fails = []
    with tempfile.TemporaryDirectory() as td:
        exe = build(td, 'test_load_par', STORE)
        text = STORE.read_text()
        old = "  const size_t at[6] = {hlay_.w1, hlay_.s1, hlay_.w3, hlay_.s3, hlay_.w2, hlay_.s2};"
        assert text.count(old) == 1, 'mutant anchor drifted'
        mut = Path(td) / 'store_mut.cpp'
        mut.write_text(text.replace(old, old + "\n  const bool swap_w2 = true;").replace(
            "segs.push_back({f, (uint64_t)(t.offset + n * half[k])", "segs.push_back({f, (uint64_t)(t.offset + ((k == 4 && swap_w2) ? 1 - n : n) * half[k])"))
        mexe = build(td, 'test_load_par_mut', mut)
        work = None
        for cand in [tempfile.gettempdir(), str(ROOT.parent)]:
            if direct_ok(cand):
                work = cand
                break
        with tempfile.TemporaryDirectory(dir=work, prefix='hive-loadpar-') as wd:
            env = harness.tsan_env()
            env['HIVE_TEST_EXPECT_DIRECT'] = '1' if work else '0'
            for k in ('HIVE_LOAD_PAR', 'HIVE_LOAD_CHUNK_MB', 'HIVE_LOAD_BUFFERED'):
                env.pop(k, None)
            p = subprocess.run([exe, wd], env=env, capture_output=True, text=True, timeout=900)
            print(p.stdout.strip())
            if p.returncode != 0:
                print(p.stderr[-4000:])
                fails.append('test_load_par_cpu')
            m = subprocess.run([mexe, wd], env=env, capture_output=True, text=True, timeout=900)
            detected = m.returncode != 0
            print(f'  mutant [w2 halves swapped in load_bulk]: {"DETECTED" if detected else "NOT DETECTED"}')
            if not detected:
                fails.append('mutant not detected')
        dexe = D.build_daemon(td)
        transcripts, logs = {}, {}
        for name, env in (('default', {}), ('loadpar', {'HIVE_LOAD_PAR': '1'})):
            ck = D.Checker(name)
            h = D.Hived(dexe, td, env, name=name)
            D.sc_session(h, ck, {})
            h.alive()
            h.stop()
            logs[name] = h.log_path.read_text(errors='replace')
            transcripts[name] = ck.transcript
            if ck.failures:
                fails.append(f'daemon {name}: {ck.failures[:3]}')
        same = transcripts['default'] == transcripts['loadpar']
        print(f'  daemon HIVE_LOAD_PAR=1 (fallback path) transcript == default: {same} ({len(transcripts["default"])} requests)')
        if not same:
            fails.append('daemon transcript differs')
        for name, needle, want in (('loadpar', 'bulk load failed', True), ('loadpar', '(load par 8 · prefault 0)', True), ('default', '(load par 0 · prefault 0)', True),
                                   ('default', 'bulk load', False), ('default', 'load par:', False)):
            if (needle in logs[name]) != want:
                fails.append(f'daemon {name} log: {needle!r} present={not want}')
        if not work:
            print('  note: no O_DIRECT-capable temp dir — O_DIRECT path ran as buffered fallback')
    if fails:
        print('FAIL', fails)
        sys.exit(1)
    print('test_load_par_cpu.py passed')


if __name__ == '__main__':
    main()
