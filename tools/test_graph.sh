#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Graph path check: run the same prompt and 6 decode steps (A) with CUDA graphs and (B) eagerly, compare routing, final state and logits,
# and print single and batched (2-sequence) profiles. Usage (inside the engine container, /hive and /out mounted): test_graph.sh [max_layer=0] [cache_mb=2048]
# pipefail — otherwise a failing exit code of compare_runs.py is swallowed by the following | grep and the script always exits 0.
#   grep exits 1 when it filters out every line, so it is wrapped in { grep ... || true; }; comparison failures are collected in FAIL and returned at the end.
set -uo pipefail
[ -n "${HIVE_CKPT_MOUNT:-}" ] && export HIVE_CKPT=$HIVE_CKPT_MOUNT  # inside scripts/hive-run.sh the checkpoint is mounted there
FAIL=0
N=${1:-0}; CACHE=${2:-2048}
G=/out/golden/l$N
IDS=$(python3 /hive/tools/ids_of.py $G)
COMMON="--max-layer $N --ids $IDS --continue-ids 13,1052,5,13,1052,5 --vram-cache-mb $CACHE --cpu-threads 16 --max-ctx 4096 --max-chunk 64 --engram /out/engram --vision-max-patches 64"
A=/out/dump/graphA_l$N; B=/out/dump/graphB_l$N
rm -rf $A $B; mkdir -p $A $B
echo "### A: graph"
HIVE_GRAPH_DUMP=1 /hive/engine/${HIVE_BUILD:-build}/hive $COMMON --dump $A 2>&1 | grep -E "continue|error|failed|graph" | tail -3
echo "### B: eager"
HIVE_NO_GRAPH=1 /hive/engine/${HIVE_BUILD:-build}/hive $COMMON --dump $B 2>&1 | grep -E "continue|error|failed" | tail -2
python3 /hive/tools/compare_runs.py $A $B | { grep -vE "route_ids.*ok$" || true; } || FAIL=1
echo "### profile M=1 (graph)"
HIVE_PROFILE=1 /hive/engine/${HIVE_BUILD:-build}/hive $COMMON --promote 2 --gpu-share 1 2>&1 | grep -E "profile M=1" | tail -2
echo "### profile M=1 (eager)"
HIVE_NO_GRAPH=1 HIVE_PROFILE=1 /hive/engine/${HIVE_BUILD:-build}/hive $COMMON --promote 2 --gpu-share 1 2>&1 | grep -E "profile M=1" | tail -1
exit $FAIL
