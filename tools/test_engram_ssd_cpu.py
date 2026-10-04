#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""HIVE_ENGRAM_SSD — CPU checks (no GPU, no service, no model weights).

engine/tests/test_engram_ssd_cpu.cpp on fake CUDA/NUMA with the REAL hive/engram_ssd.h, engram_hash.cpp, expert_store.cpp, safetensors.cpp and the
REAL runtime.cpp engram_gather (sliced by tools/cpu_fake/harness.py) as the RAM-mode reference:
  * EngramHash::peek == compute (random tokens, image marks, short histories)
  * EngramSsd::gather == engram_gather bytes: mode 1/2 × cache rows 0..all × read threads 1/3/64 × prefetch on/off × gather threads 1/8,
    decode-like and prefill-like steps; prefetch reads exactly the needed rows (big cache, drained queue → every lookup cached)
  * 4 concurrent sequences on a 300-row cache
  * faults: transient EIO retried · EINVAL → 4 KiB alignment · persistent EIO fails only that gather and is not cached · range check message
  * real ExpertStore load_engram / load_bulk with HIVE_ENGRAM_SSD=1/2 == RAM store · values 0 B in RAM · HostPrefault::predict skips SSD tables
Negative controls (each MUST fail the unit): row address off by one in the reader · peek history window off by one ·
RAM-path scales read from the value-table shard (regression check — the fixture keeps one table's scales in another shard).
The work dir must support O_DIRECT for the O_DIRECT assertions (tmpfs does not) — /tmp is tried first, then a dir next to the repo.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parent / 'cpu_fake')]
import harness  # noqa: E402

ROOT = harness.ROOT
HDR = ROOT / 'engine/include/hive/engram_ssd.h'
HASH = ROOT / 'engine/src/engram_hash.cpp'


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


STORE = ROOT / 'engine/src/expert_store.cpp'


def build(td, name, inc=None, hash_src=HASH, store_src=STORE):
    exe = str(Path(td) / name)
    cmd = harness.flags()
    if inc:
        cmd.insert(1, '-I' + str(inc))  # the mutated header comes before engine/include
    cmd += ['-I' + td, '-D_GLIBCXX_ASSERTIONS']
    srcs = [ROOT / 'engine/tests/test_engram_ssd_cpu.cpp', hash_src, store_src, ROOT / 'engine/src/cpu/expert_cpu.cpp',
            ROOT / 'engine/src/safetensors.cpp']
    subprocess.run(cmd + [str(s) for s in srcs] + ['-o', exe], check=True)
    return exe


def main():
    fails = []
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / 'prefill_helpers_gen.h').write_text(harness.prefill_helpers_source())
        exe = build(td, 'test_engram_ssd')
        # Mutant 1: the row address read is off by one row
        text = HDR.read_text()
        old = 'const uint64_t off = T.off[kind] + (uint64_t)row * T.rb[kind];'
        assert text.count(old) == 1, 'mutant anchor drifted (reader)'
        mi = Path(td) / 'mut1' / 'hive'
        mi.mkdir(parents=True)
        (mi / 'engram_ssd.h').write_text(text.replace(old, 'const uint64_t off = T.off[kind] + (uint64_t)(row + 1) * T.rb[kind];'))
        m1 = build(td, 'test_engram_ssd_mut1', inc=mi.parent)
        # Mutant 2: the peek history window is off by one position
        htext = HASH.read_text()
        old2 = '  const int64_t base = start - tail_n;'
        assert htext.count(old2) == 1, 'mutant anchor drifted (peek)'
        mh = Path(td) / 'engram_hash_mut.cpp'
        mh.write_text(htext.replace(old2, '  const int64_t base = start - tail_n + 1;'))
        m2 = build(td, 'test_engram_ssd_mut2', hash_src=mh)
        # Mutant 3 (negative control for the scale-shard fix): the RAM path reads scales from the value-table shard (the old defect — with the split-shard fixture the RAM store is wrong)
        stext = STORE.read_text()
        old3 = '  par_read(ck.shard_for(p + "scale").fd(), T.scales.base, s.offset, s.nbytes);\n  double sec'
        assert stext.count(old3) == 1, 'mutant anchor drifted (RAM scale shard)'
        ms = Path(td) / 'expert_store_mut.cpp'
        ms.write_text(stext.replace(old3, '  par_read(ck.shard_for(p + "weight").fd(), T.scales.base, s.offset, s.nbytes);\n  double sec'))
        m3 = build(td, 'test_engram_ssd_mut3', store_src=ms)
        work = None
        for cand in [tempfile.gettempdir(), str(ROOT.parent)]:
            if direct_ok(cand):
                work = cand
                break
        env = harness.tsan_env()
        env['HIVE_TEST_EXPECT_DIRECT'] = '1' if work else '0'
        for k in list(env):
            if k.startswith('HIVE_ENGRAM') or k.startswith('HIVE_LOAD'):
                env.pop(k)
        for name, x, want_ok in (('unit', exe, True), ('mutant [row address +1 in the reader]', m1, False), ('mutant [peek history window +1]', m2, False),
                                    ('mutant [RAM scales read from the value shard]', m3, False)):
            with tempfile.TemporaryDirectory(dir=work, prefix='hive-engramssd-') as wd:
                p = subprocess.run([x, wd], env=env, capture_output=True, text=True, timeout=2400)  # the full matrix takes minutes; HIVE_TEST_QUICK=1 (CI) runs a reduced one
            if want_ok:
                print(p.stdout.strip())
                if p.returncode != 0:
                    print(p.stderr[-4000:])
                    fails.append(name)
            else:
                detected = p.returncode != 0
                first = next((l for l in p.stderr.splitlines() if l.startswith('FAIL')), '')
                print(f'  {name}: {"DETECTED" if detected else "NOT DETECTED"} {first[:160]}')
                if not detected:
                    fails.append(name + ' not detected')
        if not work:
            print('  note: no O_DIRECT-capable temp dir — reads ran buffered')
    if fails:
        print('FAIL', fails)
        sys.exit(1)
    print('test_engram_ssd_cpu.py passed')


if __name__ == '__main__':
    main()
