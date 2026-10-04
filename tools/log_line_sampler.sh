#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# log_line_sampler.sh — hived.log and server.log have no timestamps (profile and cache lines). To map "axis (time span) → log line
#   span" during analysis, record epoch and the line counts of both logs to a CSV every N seconds (used during full benchmark runs). usage: log_line_sampler.sh <out.csv> [interval=10]
#   Logs: $HIVE_STATE_DIR/logs (default run/logs in the repository, as scripts/hive-start.sh writes them).
OUT=${1:?out.csv}; IV=${2:-10}
LOGS=${HIVE_STATE_DIR:-$(cd "$(dirname "$0")/.." && pwd)/run}/logs; HL=$LOGS/hived.log; SL=$LOGS/server.log
[ -s "$OUT" ] || echo "ts,hived_lines,server_lines" > "$OUT"
while true; do echo "$(date +%s),$(wc -l < "$HL"),$(wc -l < "$SL")" >> "$OUT"; sleep "$IV"; done
