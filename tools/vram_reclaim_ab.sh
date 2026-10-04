#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# R5 GPU comparison (run on a GPU machine): checks that HIVE_ENGRAM_ALIAS (lending the engram work buffer inside q/o; reclaims 1,190 MiB at 16K chunks)
#   does not change values — prefills the same prompt with the switch at 0/1 and compares the end-of-prefill logits (--logits-out, all fp32) and the greedy token sequence **byte for byte**.
#   Covers every path through the engram layers (the model's engram_layer_ids): (A) a single chunk <= 16K, (B) tiled prefill (--prefill-tile 3, prompt longer than max_chunk 16K),
#   (C) short-prompt decode (M <= 8 decode engram_dev, graph capture/replay) — each pair both with --no-cpu (all experts on GPU: removes the timing-dependent accumulation order
#   of the CPU/DMA split) and under serving conditions (CPU misses on). Verdict: --no-cpu pairs must be bit-identical in logits and tokens to PASS. CPU-on pairs are only reported, alongside a same-switch rerun (noise floor).
#   HIVE_CACHE_FIT only changes the slot count (value-neutral) — check only that M >= HIVE_CACHE_RESERVE_MB in the startup log "[hived] cache fit: N slots ... VRAM free M MiB" (not started here).
#   Usage (inside the engine container, with the GPU idle): scripts/hive-run.sh "bash /hive/tools/vram_reclaim_ab.sh [ntok=40000] [build=build]"
set -euo pipefail
[ -n "${HIVE_CKPT_MOUNT:-}" ] && export HIVE_CKPT=$HIVE_CKPT_MOUNT  # inside scripts/hive-run.sh the checkpoint is mounted there
N=${1:-40000}; B=${2:-${HIVE_BUILD:-build}}
mkdir -p /out/vram-reclaim
OUT=$(mktemp -d /out/vram-reclaim/run.XXXXXX)
echo "artifacts: $OUT"
python3 - "$N" > $OUT/long.txt <<'PY'
import sys, random
n = int(sys.argv[1]); r = random.Random(20260930)
print(",".join(str(r.randrange(1000, 120000)) for _ in range(n)))
PY
python3 - 9000 > $OUT/mid.txt <<'PY'
import sys, random
n = int(sys.argv[1]); r = random.Random(930)
print(",".join(str(r.randrange(1000, 120000)) for _ in range(n)))
PY
echo "1000,2000,3000,4000,5000,6000,7000,8000" > $OUT/short.txt
DECODE=32
COMMON="--decode $DECODE --vram-cache-mb ${CACHE_MB:-60000} --cpu-threads 16 --max-ctx 65536 --max-chunk 16384 --engram /out/engram --no-mtp"
cd /hive/engine || exit 1
run() {  # name, HIVE_ENGRAM_ALIAS value, extra args...
  local name=$1 alias=$2; shift 2
  echo "### $name (HIVE_ENGRAM_ALIAS=$alias): $*"
  HIVE_ENGRAM_ALIAS=$alias ./$B/hive $COMMON "$@" > $OUT/$name.out 2> $OUT/$name.err
  grep -E "engram alias|prefill [0-9]+ tokens|VRAM free|FATAL|error|failed|Aborted" $OUT/$name.err | cut -c1-240 || true
}
for c in mid long short; do
  extra=""; [ $c = long ] && extra="--prefill-tile 3"
  lo=""; [ $c != short ] && lo="--logits-out"
  run ${c}_nocpu_0 0 --ids-file $OUT/$c.txt $extra --no-cpu ${lo:+$lo $OUT/${c}_nocpu_0.logits}
  run ${c}_nocpu_1 1 --ids-file $OUT/$c.txt $extra --no-cpu ${lo:+$lo $OUT/${c}_nocpu_1.logits}
  run ${c}_cpu_0a 0 --ids-file $OUT/$c.txt $extra ${lo:+$lo $OUT/${c}_cpu_0a.logits}
  run ${c}_cpu_0b 0 --ids-file $OUT/$c.txt $extra ${lo:+$lo $OUT/${c}_cpu_0b.logits}
  run ${c}_cpu_1 1 --ids-file $OUT/$c.txt $extra ${lo:+$lo $OUT/${c}_cpu_1.logits}
done
python3 - "$OUT" <<'PY'
import sys, os, numpy as np
o = sys.argv[1]
def toks(n):
    for l in open(os.path.join(o, n + '.out')):
        if l.startswith('generated:'): return l.split()[1:]
    return None
def lg(n):
    p = os.path.join(o, n + '.logits')
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None
bad = 0
for c in ('mid', 'long', 'short'):
    a, b = f'{c}_nocpu_0', f'{c}_nocpu_1'
    ta, tb = toks(a), toks(b)
    la, lb = lg(a), lg(b)
    same_t = ta is not None and ta == tb
    same_l = (la is None and lb is None) or (la is not None and lb is not None and la.tobytes() == lb.tobytes())
    ok = same_t and same_l
    bad += not ok
    print(f'{c:5s} --no-cpu alias 0 vs 1: tokens {"same" if same_t else "DIFF"} · logits {"bit-identical" if same_l else "DIFF"} → {"PASS" if ok else "FAIL"}')
    for x, y in ((f'{c}_cpu_0a', f'{c}_cpu_0b'), (f'{c}_cpu_0a', f'{c}_cpu_1')):
        lx, ly = lg(x), lg(y)
        d = float(np.max(np.abs(lx - ly))) if lx is not None and ly is not None else 0.0
        print(f'      cpu on {x} vs {y}: tokens {"same" if toks(x) == toks(y) else "diff"} · max|Δlogit| {d:.3g} (report only — CPU/DMA split noise floor)')
print('vram_reclaim_ab:', 'PASS' if bad == 0 else f'FAIL ({bad})')
sys.exit(1 if bad else 0)
PY
