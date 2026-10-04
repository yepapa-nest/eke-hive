#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Compare layers 0..N with the oracle golden (14-token text prefill + 3 decode steps). Usage: validate_layers.sh N [cache_mb]
# Run inside the engine container (/hive and /out mounted). If engram layers (1, 14) are included, /out/engram must exist.
set -euo pipefail
[ -n "${HIVE_CKPT_MOUNT:-}" ] && export HIVE_CKPT=$HIVE_CKPT_MOUNT  # inside scripts/hive-run.sh the checkpoint is mounted there
export HIVE_CACHE_PRIOR=0 HIVE_DECODE_DEFER=0  # exactness check: the decode routing changes of step 21 (on in config/hive.env) deliberately differ from the reference
N=${1:?max layer}; CACHE=${2:-512}
G=/out/golden/l$N
mkdir -p "$G" /out/dump
D=$(mktemp -d /out/dump/layer.XXXXXX)
echo "artifacts: $D"
if [ ! -f $G/prefill.npz ]; then
  python3 /hive/oracle/dsv41_oracle.py --prompt "The capital of France is Paris, and the capital of Germany is" --bos --max-layer $N \
    --continue-ids 13,1052,5 --out $G --threads 24 --quiet 2>&1 | grep -v transformers | tail -2
fi
IDS=$(python3 /hive/tools/ids_of.py $G)
/hive/engine/${HIVE_BUILD:-build}/hive --max-layer $N --ids $IDS --continue-ids 13,1052,5 --dump $D --vram-cache-mb $CACHE --cpu-threads 16 \
  --max-ctx 4096 --max-chunk 64 --vision-max-patches 64 --engram /out/engram 2>&1 | grep -E "prefill|continue|error|failed|store\] engram"
python3 /hive/tools/compare_golden.py --golden $G --dump $D
