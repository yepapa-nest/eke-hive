#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Layer 0: two-sequence batch decode check + single-sequence profile + full-size image recheck (run inside the engine container)
set -u
[ -n "${HIVE_CKPT_MOUNT:-}" ] && export HIVE_CKPT=$HIVE_CKPT_MOUNT  # inside scripts/hive-run.sh the checkpoint is mounted there
IDS=$(python3 /hive/tools/ids_of.py /out/golden/l0b)
rm -f /out/dump/l0b/*
/hive/engine/build/hive --max-layer 0 --ids $IDS --continue-ids 13,1052,5 --batch-golden /out/golden/l0c --dump /out/dump/l0b \
  --vram-cache-mb 2048 --cpu-threads 16 --max-ctx 4096 --max-chunk 64 2>&1 | grep -E "batch|error|failed"
echo "### seq0 rows of batch dumps"
python3 /hive/tools/compare_golden.py --golden /out/golden/l0b --dump /out/dump/l0b --rows 2 --row 0 --decode-offset 1 | grep -E "^L00"
echo "### seq1"
python3 /hive/tools/compare_golden.py --golden /out/golden/l0c --dump /out/dump/l0b --rows 2 --row 1 --decode-offset 1 --prefill-step 1 | grep -E "^L00"
echo "### profile single-seq"
HIVE_PROFILE=1 /hive/engine/build/hive --max-layer 0 --ids $IDS --continue-ids 13,1052,5,13,1052,5 --vram-cache-mb 2048 --cpu-threads 16 \
  --max-ctx 4096 --max-chunk 64 --promote 2 --gpu-share 1 2>&1 | grep -E "profile M=1" | tail -2
rm -f /out/dump/img/*
/hive/engine/build/hive --max-layer 0 --images-from /out/golden/img --dump /out/dump/img --vram-cache-mb 2048 --cpu-threads 16 --max-ctx 4096 \
  --max-chunk 256 2>&1 | grep -E "prefill|error|failed"
python3 /hive/tools/compare_golden.py --golden /out/golden/img --dump /out/dump/img | grep -E "^L00"
