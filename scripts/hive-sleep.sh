#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Put a running daemon to sleep (give VRAM back while weights and conversations stay in RAM/VRAM) and wake it.
#   scripts/hive-sleep.sh sleep|wake|prime|state
#   sleep  POST /release_memory_occupation. Requests in flight finish first (nothing is cut off or cancelled);
#          new requests queue inside hived until it wakes.
#   wake   POST /resume_memory_occupation: re-allocate the expert cache and reload the experts that were resident.
#          If VRAM is still taken by something else, hived stays asleep (HTTP 503) and this exits 1 — wake again later.
#   prime  no-op (kept so callers can treat this like other engines' sleep scripts).
#   state  awake | asleep | absent, as reported by the daemon (/health).
# Environment:
#   HIVE_SLEEP_LEVEL     1 (default) keep session KV in VRAM · 2 also move session KV to RAM ·
#                        3 also move dense weights, work buffers and cuBLAS state to pinned host copies (needs a
#                        daemon started with HIVE_SLEEP_VMM=1 or HIVE_START_ASLEEP=1; otherwise hived uses 2)
#   HIVE_SLEEP_TIMEOUT_S optional: give up sleeping if requests in flight take longer than this (default: wait)
#   HIVE_PORT            API port (from the loaded configuration)
#   HIVE_SLEEP_LOG       optional log file
#   Hooks (optional shell commands, run with HIVE_PORT set): HIVE_HOOK_BEFORE_SLEEP (e.g. stop routing new
#   traffic here and drain idle keep-alive connections), HIVE_HOOK_SLEEP_FAILED, HIVE_HOOK_AFTER_SLEEP,
#   HIVE_HOOK_AFTER_WAKE (e.g. route traffic back).
set -uo pipefail
. "$(dirname "$0")/lib.sh"; hive_load_config
API=http://127.0.0.1:$HIVE_PORT
LOG=${HIVE_SLEEP_LOG:-/dev/null}
log() { printf '[hive-sleep] %s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG" >&2; }
ms() { echo $(( ($(date +%s%N) - $1) / 1000000 )); }
hook() { local cmd=${!1:-}; [ -z "$cmd" ] || HIVE_PORT=$HIVE_PORT bash -c "$cmd"; }

daemon_state() {  # ready|draining|sleeping|waking|loading|absent
  local h; h=$(curl -s -m 3 "$API/health" 2>/dev/null) || { echo absent; return; }
  [ -n "$h" ] || { echo absent; return; }
  printf '%s' "$h" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("state") or "absent")
except Exception: print("absent")'
}
real_state() { case "$(daemon_state)" in sleeping) echo asleep ;; absent|loading) echo absent ;; *) echo awake ;; esac; }
vram() { nvidia-smi -i "$HIVE_GPU" --query-gpu=memory.used --format=csv,noheader 2>/dev/null; }
post() {  # post <path> <json> -> body, then the HTTP status on the last line
  curl -s -m 86400 -X POST "$API/$1" -H 'Content-Type: application/json' -d "$2" -w $'\n%{http_code}'
}

do_sleep() {
  local s; s=$(daemon_state)
  case "$s" in
    absent|loading) log "no daemon, or still loading ($s) — nothing to put to sleep"; return 2 ;;
    sleeping) log "already asleep"; hook HIVE_HOOK_AFTER_SLEEP; return 0 ;;
  esac
  hook HIVE_HOOK_BEFORE_SLEEP
  local t body code req
  req="{\"level\": ${HIVE_SLEEP_LEVEL:-1}}"
  [ -n "${HIVE_SLEEP_TIMEOUT_S:-}" ] && req="{\"timeout_s\": ${HIVE_SLEEP_TIMEOUT_S}, \"level\": ${HIVE_SLEEP_LEVEL:-1}}"
  t=$(date +%s%N)
  body=$(post release_memory_occupation "$req"); code=${body##*$'\n'}; body=${body%$'\n'*}
  if [ "$code" != 200 ]; then
    log "release returned HTTP $code: $body"
    hook HIVE_HOOK_SLEEP_FAILED
    return 1
  fi
  hook HIVE_HOOK_AFTER_SLEEP
  log "asleep after $(ms "$t") ms · VRAM $(vram) · $(printf '%s' "$body" | python3 -c 'import json,sys
try:
  s=json.load(sys.stdin).get("sleep",{}); k=s.get("vram_kept",{})
  print("hived level %s · %.0f ms (drain %.0f ms) · freed %.0f MiB · kept %.0f MiB = model+ctx %.0f + sessions %.0f + rest %.0f · RSS %.0f MiB"%(s.get("level"),s.get("ms",0),s.get("drain_ms",0),s.get("freed_mib",0),s.get("vram_used_mib",0),k.get("model_weights_and_cuda_context_mib",0),k.get("session_pool_mib",0),k.get("runtime_decode_buffers_and_rest_mib",0),s.get("host_memory",{}).get("VmRSS_mib",0)))
except Exception as e: print("?",e)')"
}

do_wake() {
  local s; s=$(daemon_state)
  case "$s" in absent|loading) log "no daemon, or still loading ($s)"; return 2 ;; esac
  local t body code
  t=$(date +%s%N)
  body=$(post resume_memory_occupation '{}'); code=${body##*$'\n'}; body=${body%$'\n'*}
  if [ "$code" != 200 ]; then
    log "resume returned HTTP $code — still asleep (wake again once VRAM is free): $body"
    return 1
  fi
  hook HIVE_HOOK_AFTER_WAKE
  log "awake after $(ms "$t") ms · VRAM $(vram) · $(printf '%s' "$body" | python3 -c 'import json,sys
try:
  w=json.load(sys.stdin).get("wake",{})
  print("hived %.0f ms (alloc %.0f · warm %.0f) · slots %s/%s · warmed %s/%s"%(w.get("ms",0),w.get("alloc_ms",0),w.get("warm_ms",0),w.get("slots"),w.get("slots_before_sleep"),w.get("warmed"),w.get("resident_keys")))
except Exception as e: print("?",e)')"
}

case "${1:-}" in
  sleep) do_sleep ;;
  wake) do_wake ;;
  prime) exit 0 ;;
  state) real_state ;;
  *) echo "usage: $0 sleep|wake|prime|state" >&2; exit 2 ;;
esac
