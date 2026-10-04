#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Per-layer isolated check: replaces the input of layer l with the golden output of layer l-1 (--inject) and measures only layer l's own error.
# Separates accumulated error (bf16 noise growing across layers) from a bug in that layer. Usage (inside the engine container): validate_inject.sh N [cache_mb]
set -euo pipefail
[ -n "${HIVE_CKPT_MOUNT:-}" ] && export HIVE_CKPT=$HIVE_CKPT_MOUNT  # inside scripts/hive-run.sh the checkpoint is mounted there
export HIVE_CACHE_PRIOR=0 HIVE_DECODE_DEFER=0  # exactness check: the decode routing changes of step 21 (on in config/hive.env) deliberately differ from the reference
N=${1:?max layer}; CACHE=${2:-4096}
G=/out/golden/l$N; R=/out/golden/l$N-raw
mkdir -p /out/dump
D=$(mktemp -d /out/dump/inject.XXXXXX)
echo "artifacts: $D"
[ -f $G/prefill.npz ] || { echo "no golden: $G"; exit 1; }
[ -f $R/s0_h_L0.f32 ] || python3 /hive/tools/export_golden_raw.py $G $R
IDS=$(python3 /hive/tools/ids_of.py $G)
/hive/engine/${HIVE_BUILD:-build}/hive --max-layer $N --ids $IDS --continue-ids 13,1052,5 --dump $D --inject $R --vram-cache-mb $CACHE --cpu-threads 16 \
  --max-ctx 4096 --max-chunk 64 --vision-max-patches 64 --engram /out/engram > "$D/engine.log" 2>&1
grep -E "error|failed" "$D/engine.log" || true
python3 /hive/tools/compare_golden.py --golden $G --dump $D
