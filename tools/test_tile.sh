#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Tiled-prefill validation: the same long prompt (~40K pseudo-random tokens) is prefilled with
#   (A) sequential chunks (--prefill-tile 1, three 16K chunks) · (B) tiled (--prefill-tile 3, one forward) · (C) tiled with the CPU share off (--no-cpu, checks the activation-copy path)
#   and the end-of-prefill logits (--logits-out) and 32 greedy tokens are compared. (D) checks graph address safety: after a 4-step decode of a short prompt (graph capture),
#   a different sequence is tile-prefilled and decoded → its tokens must equal the same run with HIVE_NO_GRAPH=1 (E) (they diverge if the Work buffers are not restored).
#   Pitfalls avoided: a single --ids argument of 244KB exceeds the execve limit (128KB) and fails silently · dump paths/names must match or the comparison always says "missing" · per-layer dumps reach tens of GB.
#   Usage (inside the build container, with the GPU free): scripts/hive-run.sh "bash /hive/tools/test_tile.sh [ntok=40000] [tile=3] [build=build]"
export HIVE_CACHE_PRIOR=0 HIVE_DECODE_DEFER=0  # exactness check: the decode routing changes of step 21 (on in config/hive.env) deliberately differ from the reference
set -euo pipefail
[ -n "${HIVE_CKPT_MOUNT:-}" ] && export HIVE_CKPT=$HIVE_CKPT_MOUNT  # inside scripts/hive-run.sh the checkpoint is mounted there
N=${1:-40000}; TILE=${2:-3}; B=${3:-${HIVE_BUILD:-build}}
mkdir -p /out/tile-test
OUT=$(mktemp -d /out/tile-test/run.XXXXXX)
echo "artifacts: $OUT"
python3 - "$N" > $OUT/ids.txt <<'PY'
import sys, random
n = int(sys.argv[1]); r = random.Random(20260929)
print(",".join(str(r.randrange(1000, 120000)) for _ in range(n)))
PY
echo "1000,2000,3000,4000,5000,6000,7000,8000" > $OUT/short.txt
CACHE=${CACHE_MB:-60000}
DECODE=32  # hive --decode N prints N+1 tokens (hive_main.cpp: one push per step + a final push) — the expected length is derived from this
COMMON="--decode $DECODE --vram-cache-mb $CACHE --cpu-threads 16 --max-ctx 65536 --max-chunk 16384 --engram /out/engram --no-mtp"
cd /hive/engine || exit 1
run() {  # name · extra arguments…
  local name=$1; shift
  echo "### $name: $*"
  HIVE_PROFILE=1 ./$B/hive $COMMON "$@" > $OUT/$name.out 2> $OUT/$name.err
  echo "  engine exit 0"
  grep -E "prefill [0-9]+ tokens|prefill tile|interleaved|FATAL|error|failed|Aborted" $OUT/$name.err | cut -c1-300 || true
  grep -o "moe.gpu_experts [0-9.]*" $OUT/$name.err | awk '{s+=$2} END {if (NR) printf "  moe.gpu_experts total %.0f ms (%d samples)\n", s, NR}' || true
  grep "^generated:" $OUT/$name.out | cut -c1-200
}
run A --ids-file $OUT/ids.txt --prefill-tile 1 --logits-out $OUT/A.logits
run B --ids-file $OUT/ids.txt --prefill-tile $TILE --logits-out $OUT/B.logits
run C --ids-file $OUT/ids.txt --prefill-tile $TILE --no-cpu --logits-out $OUT/C.logits
run D --ids-file $OUT/short.txt --prefill-tile $TILE --interleave-tiled 40000
HIVE_NO_GRAPH=1 run E --ids-file $OUT/short.txt --prefill-tile $TILE --interleave-tiled 40000
python3 - "$OUT" "$((DECODE + 1))" <<'PY'
import sys, os, numpy as np
o = sys.argv[1]; want = int(sys.argv[2])  # = DECODE + 1 (D4)
def lg(n):
    p = os.path.join(o, n + ".logits")
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) and os.path.getsize(p) else None
def gen(n):
    p = os.path.join(o, n + ".out")
    if not os.path.exists(p): return None
    for line in open(p):
        if line.startswith("generated:"): return line.split()[1:]
    return None
ok = True
for a, b in (("A", "B"), ("B", "C")):
    x, y = lg(a), lg(b)
    if x is None or y is None: print(f"{a}↔{b}: logits missing"); ok = False; continue
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        print(f'{a}↔{b}: invalid logits'); ok = False; continue
    d = float(np.abs(x - y).max()); cos = float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
    good = x.argmax() == y.argmax() and cos > 0.999
    print(f"{a}↔{b} logits max|Δ| {d:.4f} (scale {np.abs(x).max():.2f}) cos {cos:.6f} argmax {x.argmax()}/{y.argmax()}  {'OK' if good else 'FAIL'}")
    ok &= good
for a, b in (("A", "B"), ("B", "C"), ("D", "E")):
    x, y = gen(a), gen(b)
    if x is None or y is None: print(f"{a}↔{b}: tokens missing"); ok = False; continue
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[n]: n += 1
    good = len(x) == len(y) == want and x == y
    print(f"{a}↔{b} greedy tokens: first {n}/{len(x)}/{len(y)} match  {'OK' if good else 'FAIL'}")
    ok &= good
print("ALL OK" if ok else "SOME FAIL")
raise SystemExit(0 if ok else 1)
PY
