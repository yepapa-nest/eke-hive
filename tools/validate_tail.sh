#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Decoder tail mode check: build an all-layer golden with a 3,000-token prompt (oracle, takes a long time), then compare the engine with tail on/off.
# Run inside the engine container once all shards for layers 20+ are present: validate_tail.sh [tail=2688]
set -u
[ -n "${HIVE_CKPT_MOUNT:-}" ] && export HIVE_CKPT=$HIVE_CKPT_MOUNT  # inside scripts/hive-run.sh the checkpoint is mounted there
export HIVE_CACHE_PRIOR=0 HIVE_DECODE_DEFER=0  # exactness check: the decode routing changes of step 21 (on in config/hive.env) deliberately differ from the reference
TAIL=${1:-2688}
G=/out/golden/tail3k; D=/out/dump/tail3k
mkdir -p $G $D
if [ ! -f $G/prefill.npz ]; then
  # Repeat text to get roughly 3,000 tokens (tokenizer count). Append 3 continuation steps.
  python3 - <<'EOF' > /tmp/tail_prompt.txt
words = "The quick brown fox jumps over the lazy dog while the river flows past the old stone bridge and the wind carries the scent of pine . "
print((words * 130)[:20000])
EOF
  python3 /hive/oracle/dsv41_oracle.py --prompt "$(cat /tmp/tail_prompt.txt)" --bos --continue-ids 13,1052,5 --out $G --threads 32 --quiet 2>&1 | grep -v transformers | tail -2
fi
IDS=$(python3 /hive/tools/ids_of.py $G)
N=$(echo $IDS | tr ',' '\n' | wc -l)
echo "prompt tokens $N · tail $TAIL"
for MODE in 0 $TAIL; do
  rm -f $D/*
  /hive/engine/${HIVE_BUILD:-build}/hive --ids $IDS --continue-ids 13,1052,5 --dump $D --vram-cache-mb 40000 --cpu-threads 16 --max-ctx 8192 \
    --max-chunk 4096 --engram /out/engram --decoder-tail $MODE 2>&1 | grep -E "prefill|continue|error|failed"
  echo "### decoder-tail=$MODE"
  if [ "$MODE" = 0 ]; then python3 /hive/tools/compare_golden.py --golden $G --dump $D | grep -E "^L(19|20|30|39)|decode|logits";
  else python3 /hive/tools/compare_golden.py --golden $G --dump $D --tail 128 | grep -E "^L(19|20|30|39)|decode|logits"; fi
done
