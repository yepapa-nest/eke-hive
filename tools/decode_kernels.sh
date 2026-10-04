#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Measures decode (M=1) GPU time per kernel at CUDA graph node granularity — to find which kernels make up segA.comp (~0.34ms per layer).
#   Background: launch fusion (HIVE_FUSE) gave −12% in eager mode but no gain in the graph section (0.35→0.36ms) — kernel time itself dominates, not the launch count.
#   ms/step includes the prefill share of the short golden prompt (kernels whose calls are a multiple of layers×steps are the decode share).
#   So instead of guessing at "more fusion", targets are chosen by per-kernel time ÷ the bandwidth floor of the bytes that kernel reads.
# Usage (inside the build container, only while no other process is using the GPU): scripts/hive-run.sh "bash /hive/tools/decode_kernels.sh [max_layer=39] [decode=24] [build=build]"
export HIVE_CACHE_PRIOR=0 HIVE_DECODE_DEFER=0  # exactness check: the decode routing changes of step 21 (on in config/hive.env) deliberately differ from the reference
set -u
[ -n "${HIVE_CKPT_MOUNT:-}" ] && export HIVE_CKPT=$HIVE_CKPT_MOUNT  # inside scripts/hive-run.sh the checkpoint is mounted there
N=${1:-39}; STEPS=${2:-24}; B=${3:-${HIVE_BUILD:-build}}
G=/out/golden/l0
IDS=$(python3 /hive/tools/ids_of.py $G)
OUT=/out/nsys/decode_l${N}_$(date +%m%d%H%M)
mkdir -p $(dirname $OUT)
command -v nsys >/dev/null || { echo "nsys not found (the image needs nsight-systems)"; exit 1; }
HIVE_PROFILE=1 nsys profile --cuda-graph-trace=node --trace=cuda -o $OUT -f true \
  /hive/engine/$B/hive --max-layer $N --ids $IDS --decode $STEPS --vram-cache-mb ${CACHE_MB:-68000} --cpu-threads 16 --max-ctx 8192 \
  --max-chunk 2048 --engram /out/engram --no-mtp 2>&1 | grep -E "profile M=1|error|failed" | tail -3
nsys stats --report cuda_gpu_kern_sum --format csv --output $OUT $OUT.nsys-rep >/dev/null 2>&1
python3 - "$OUT"_cuda_gpu_kern_sum.csv "$STEPS" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
steps = int(sys.argv[2])
print(f"{'kernel':60s} {'calls':>7s} {'avg µs':>8s} {'ms/step':>8s}")
for r in sorted(rows, key=lambda r: -float(r['Total Time (ns)']))[:30]:
    name = r['Name'][:60]
    print(f"{name:60s} {int(r['Instances']):7d} {float(r['Avg (ns)'])/1e3:8.1f} {float(r['Total Time (ns)'])/1e6/steps:8.2f}")
PY
echo "artifacts: $OUT.nsys-rep (per-layer/per-kernel times: nsys-ui timeline or --report cuda_gpu_trace)"
