#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Decode fusion kernel (HIVE_FUSE) check: runs the same prompt and 6 decode steps with fusion on/off, compares routing (bitwise), final state and logits, and prints profiles.
# Usage (inside the engine container): test_fuse.sh [max_layer=0] [cache_mb=2048]. Covers both graph mode (production path) and eager execution.
# pipefail — otherwise a failing exit code of compare_runs.py is swallowed by the trailing | grep and the script always exits 0.
#   grep exits 1 when it has nothing to print, so it is wrapped as { grep ... || true; }; comparison failures are collected in FAIL and returned at the end.
set -uo pipefail
[ -n "${HIVE_CKPT_MOUNT:-}" ] && export HIVE_CKPT=$HIVE_CKPT_MOUNT  # inside scripts/hive-run.sh the checkpoint is mounted there
FAIL=0
N=${1:-0}; CACHE=${2:-2048}
G=/out/golden/l$N
IDS=$(python3 /hive/tools/ids_of.py $G)
COMMON="--max-layer $N --ids $IDS --continue-ids 13,1052,5,13,1052,5 --vram-cache-mb $CACHE --cpu-threads 16 --max-ctx 4096 --max-chunk 64 --engram /out/engram --vision-max-patches 64"
for MODE in graph eager; do
  A=/out/dump/fuse1_${MODE}_l$N; B=/out/dump/fuse0_${MODE}_l$N
  rm -rf $A $B; mkdir -p $A $B
  ENV=""; [ $MODE = eager ] && ENV="HIVE_NO_GRAPH=1" || ENV="HIVE_GRAPH_DUMP=1"
  echo "### $MODE: fused"; env $ENV HIVE_FUSE=1 /hive/engine/${HIVE_BUILD:-build}/hive $COMMON --dump $A 2>&1 | grep -E "continue|error|failed" | tail -2
  echo "### $MODE: chain"; env $ENV HIVE_FUSE=0 /hive/engine/${HIVE_BUILD:-build}/hive $COMMON --dump $B 2>&1 | grep -E "continue|error|failed" | tail -2
  python3 /hive/tools/compare_runs.py $A $B | { grep -vE "route_ids.*ok$" || true; } || FAIL=1
done
echo "### profile M=1 (graph) fused vs chain"
HIVE_FUSE=1 HIVE_PROFILE=1 /hive/engine/${HIVE_BUILD:-build}/hive $COMMON --promote 2 --gpu-share 1 2>&1 | grep -E "profile M=1" | tail -1
HIVE_FUSE=0 HIVE_PROFILE=1 /hive/engine/${HIVE_BUILD:-build}/hive $COMMON --promote 2 --gpu-share 1 2>&1 | grep -E "profile M=1" | tail -1
exit $FAIL
