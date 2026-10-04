#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Thinking-budget quality comparison (low/medium/xhigh, each with and without a budget) — direct to hive on :8430, via tools/quality_eval.py --effort/--think-budget.
# Default scope = qa, code, tool, 122 questions (long-context ruler and ops excluded — irrelevant to judging thinking quality and twice the time). Override with SUITE=all.
# Usage: bash tools/quality_budget_sweep.sh [LEVEL:BUDGET ...]   (default = low:1024 medium:3072 xhigh:16384 — the default per-effort thinking budgets)
#   For each level runs (1) with the budget and (2) without it (no cap — only max_tokens +32768), and records pass rate, flips and McNemar p via quality_compare.
#   Preconditions: no other traffic reaches this hive instance (mixed-in real requests distort latency and ordering); hive is running with its final configuration.
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
OUT=${OUT:-${HIVE_STATE_DIR:-run}/quality-sweep}; mkdir -p "$OUT"  # results directory (override with OUT=...)
SUITE=${SUITE:-qa,code,tool}
PAIRS=("$@"); [ ${#PAIRS[@]} -gt 0 ] || PAIRS=(low:1024 medium:3072 xhigh:16384)
run() {  # name, level, budget (empty = none)
  local name=$1 lvl=$2 budget=$3 t0=$(date +%s)
  python3 tools/quality_eval.py --out "$OUT/$name.json" --suite "$SUITE" --label "$name" --effort "$lvl" ${budget:+--think-budget "$budget"} > "$OUT/$name.log" 2>&1
  echo "=== $name ($(( $(date +%s) - t0 ))s)"; sed -n '/== summary/,$p' "$OUT/$name.log" | head -8
}
for pair in "${PAIRS[@]}"; do
  lvl=${pair%%:*}; budget=${pair#*:}
  run "$lvl-budget$budget" "$lvl" "$budget"
  run "$lvl-nobudget" "$lvl" ""
  echo "--- compare $lvl: nobudget → budget$budget"; python3 tools/quality_compare.py "$OUT/$lvl-nobudget.json" "$OUT/$lvl-budget$budget.json" --json "$OUT/cmp-$lvl.json" | tail -12
done
echo "sweep done $(date +%H:%M:%S)"
