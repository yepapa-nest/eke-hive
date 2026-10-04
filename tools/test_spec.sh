#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# DSpark (MTP) validation — inside the engine container. ① oracle --spec golden (all 40 layers + 3 draft stages, CPU) ② engine --decode 1 --spec 1 --dump ③ compare ④ acceptance rate and time over N speculative steps.
# Usage: test_spec.sh [golden_tag=spec1] [steps=8] [cache_mb=8192]   (env SKIP_ORACLE=1 skips ① — when the golden already exists)
# The prompt uses the same ids as golden l39 (ids_of.py) — pass IDS=... for a different prompt.
set -euo pipefail
[ -n "${HIVE_CKPT_MOUNT:-}" ] && export HIVE_CKPT=$HIVE_CKPT_MOUNT  # inside scripts/hive-run.sh the checkpoint is mounted there
TAG=${1:-spec1}; STEPS=${2:-8}; CACHE=${3:-8192}
G=/out/golden/$TAG
mkdir -p /out/dump
D=$(mktemp -d /out/dump/spec.XXXXXX)
echo "artifacts: $D"
IDS=${IDS:-$(python3 /hive/tools/ids_of.py /out/golden/l39 2>/dev/null || python3 /hive/tools/ids_of.py /out/golden/l0)}
COMMON="--ids $IDS --vram-cache-mb $CACHE --cpu-threads 16 --max-ctx 4096 --max-chunk 64 --engram /out/engram --vision-max-patches 64"
if [ "${SKIP_ORACLE:-0}" != 1 ]; then
  mkdir -p /out/golden
  # The golden is built in a temporary directory (so a failure midway does not leave a half-written golden) and moved to /out/golden/$TAG
  #   when done, so a SKIP_ORACLE=1 rerun finds it under golden_tag. An existing tag is moved aside to .old.<timestamp>.
  TMPG=$(mktemp -d /out/golden/.spec-$TAG.XXXXXX)
  echo "### oracle --spec → $TMPG → $G (CPU, tens of minutes)"
  nice -n 19 python3 /hive/oracle/dsv41_oracle.py --ids $IDS --spec --out $TMPG --threads ${ORACLE_THREADS:-8} --quiet 2>&1 | tail -4
  if [ -e "$G" ]; then mv "$G" "$G.old.$(date +%Y%m%d-%H%M%S)"; fi
  mv "$TMPG" "$G"
fi
echo "### engine: prefill → decode 1 → spec 1 (dump) → compare"
HIVE_NO_GRAPH=1 /hive/engine/${HIVE_BUILD:-build}/hive $COMMON --decode 1 --spec 1 --dump $D 2>&1 | grep -E "spec|decode|dspark|error|failed" | tail -5
python3 /hive/tools/compare_spec.py --golden $G --dump $D
echo "### engine: spec ${STEPS} steps (graph on, promote 8)"
HIVE_PROFILE=1 /hive/engine/${HIVE_BUILD:-build}/hive $COMMON --decode 1 --spec $STEPS --promote 8 --gpu-share 0 > "$D/spec-run.log" 2>&1
grep -E "^\[spec|profile M=|generated" "$D/spec-run.log" | tail -$((STEPS + 3)) || true
COUNT=$(python3 - "$D/spec-run.log" <<'PY'
import re,sys
matches=re.findall(r'^generated:(.*)$',open(sys.argv[1]).read(),re.M)
assert matches and len(matches[-1].split())>1, 'missing/empty speculative output'
print(len(matches[-1].split()))
PY
)
echo "### greedy no-MTP reference: $COUNT tokens"
/hive/engine/${HIVE_BUILD:-build}/hive $COMMON --no-mtp --decode "$((COUNT - 1))" --promote 8 --gpu-share 0 > "$D/no-mtp.log" 2>&1
python3 - "$D/spec-run.log" "$D/no-mtp.log" <<'PY'
import re,sys
def tokens(path):
    matches=re.findall(r'^generated:(.*)$',open(path).read(),re.M)
    assert matches, 'missing generated output'
    return [int(x) for x in matches[-1].split()]
a,b=map(tokens,sys.argv[1:]);ok=bool(a) and a==b
print('greedy MTP parity:', 'OK' if ok else 'FAIL', 'tokens',len(a),len(b))
raise SystemExit(0 if ok else 1)
PY
echo '### future-token independence and all rollback positions'
/hive/engine/${HIVE_BUILD:-build}/hive $COMMON --verify-test > "$D/causality.log" 2>&1
grep -F '[verify-test]' "$D/causality.log"
