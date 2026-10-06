#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Start the daemon (hived for DeepSeek-V4.1-Flash, hived_glm for GLM-5.3-Flash — chosen from the checkpoint) and the OpenAI/Anthropic-compatible API server in one container.
#   scripts/hive-start.sh [--build DIR] [--vram-cache-mb N] [--max-ctx N] [--max-chunk N] [--port P]
#                         [--cpu-threads N] [--max-batch N] [extra hived arguments...]
# Settings come from HIVE_CONFIG (default config/hive.env, or config/glm.env for a GLM checkpoint; see docs/configuration.md);
# command-line flags override them.
# The binary is chosen only by --build (default engine/build). HIVE_BUILD from the shell is ignored on purpose:
# a value left over from a test session would otherwise start an unreviewed build without any visible trace.
set -u
. "$(dirname "$0")/lib.sh"
shell_build=${HIVE_BUILD:-}; unset HIVE_BUILD
hive_load_config; hive_need_ckpt; hive_detect_family
BUILD=build
read -r -a EXTRA <<< "${HIVE_DAEMON_ARGS:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --vram-cache-mb) HIVE_CACHE_MB=$2; shift 2 ;;
    --max-ctx) HIVE_MAX_CTX=$2; shift 2 ;;
    --max-chunk) HIVE_MAX_CHUNK=$2; shift 2 ;;
    --port) HIVE_PORT=$2; shift 2 ;;
    --cpu-threads) HIVE_CPU_THREADS=$2; shift 2 ;;
    --max-batch) HIVE_MAX_BATCH=$2; shift 2 ;;
    --build) BUILD=$2; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
export HIVE_BUILD=$BUILD HIVE_PORT
if [ -n "$shell_build" ] && [ "$shell_build" != "$BUILD" ]; then
  echo "note: HIVE_BUILD=$shell_build in the shell is ignored; starting engine/$BUILD (use --build $shell_build to change)" >&2
fi
[ -x "$HIVE_ROOT/engine/$BUILD/$HIVE_DAEMON_BIN" ] || { echo "engine/$BUILD/$HIVE_DAEMON_BIN not found — run scripts/build.sh first" >&2; exit 1; }
# GLM: no engram tables; the expert cache is sized after the session pool (HIVE_CACHE_FIT) unless set explicitly; the optional
#   Z.ai FP8 tensors (HIVE_GLM_FP8_DIR, docs/glm.md) are mounted read-only at the same path.
FAMILY_MOUNTS=()
if [ "$HIVE_MODEL_FAMILY" = glm ]; then
  : "${HIVE_CACHE_FIT:=1}"; export HIVE_CACHE_FIT
  if [ -n "${HIVE_GLM_FP8_DIR:-}" ] && [ "$HIVE_GLM_FP8_DIR" != 0 ]; then
    [ -f "$HIVE_GLM_FP8_DIR/model.safetensors.index.json" ] || { echo "HIVE_GLM_FP8_DIR=$HIVE_GLM_FP8_DIR has no model.safetensors.index.json" >&2; exit 1; }
    FAMILY_MOUNTS=(-v "$HIVE_GLM_FP8_DIR":"$HIVE_GLM_FP8_DIR":ro)
  fi
fi

# Starting asleep (HIVE_START_ASLEEP=1 or --start-asleep) only needs VRAM for loading (dense weights + work buffers,
# then moved to host copies); the expert cache is allocated by the first wake.
asleep_start=0
case "${HIVE_START_ASLEEP:-}" in ""|0) ;; *) asleep_start=1 ;; esac
for a in "${EXTRA[@]}"; do [ "$a" = --start-asleep ] && asleep_start=1; done
# VRAM pre-check: with HIVE_CACHE_FIT the cache is sized to the free VRAM at start, so only the fixed part (dense weights, sessions,
#   reserve) is required; without it the configured cache plus the fixed part must fit.
case "${HIVE_CACHE_FIT:-}" in ""|0) need_mb=$((HIVE_CACHE_MB + 12000)) ;; *) need_mb=$((12000 + ${HIVE_CACHE_RESERVE_MB:-2400})) ;; esac
[ "$asleep_start" = 1 ] && need_mb=14000
free_mb=$(nvidia-smi -i "$HIVE_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${free_mb:-0}" -lt "$need_mb" ]; then
  echo "free VRAM ${free_mb:-?} MiB < needed $need_mb MiB — stop other GPU users (or lower HIVE_CACHE_MB when HIVE_CACHE_FIT is off; nvidia-smi must be on the host PATH)" >&2; exit 1
fi

mkdir -p "$HIVE_STATE_DIR/logs"
[ "$HIVE_MODEL_FAMILY" = glm ] || [ -d "$HIVE_STATE_DIR/engram" ] || { echo "$HIVE_STATE_DIR/engram missing — run scripts/prepare.sh first" >&2; exit 1; }
docker rm -f "$HIVE_CONTAINER" >/dev/null 2>&1
# Wait until a previous daemon has exited and released its pinned RAM: starting a new one while the old
# pinned memory (~480 GB for DeepSeek, ~385 GB for GLM) is still being freed gets the new process OOM-killed. MemAvailable alone is not
# enough (it counts the page cache holding the checkpoint), so the old process must also be gone.
# HIVE_RAM_WAIT_GB above the installed RAM would wait the full ten minutes on every start, so it is clamped to MemTotal − 8 GiB.
total_gb=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 / 1024 ))
wait_gb=$HIVE_RAM_WAIT_GB
[ "$wait_gb" -gt $((total_gb - 8)) ] && { wait_gb=$((total_gb - 8)); echo "HIVE_RAM_WAIT_GB=$HIVE_RAM_WAIT_GB exceeds the installed RAM ($total_gb GiB): waiting for $wait_gb GiB instead"; }
need_kb=$(( wait_gb * 1024 * 1024 ))
for i in $(seq 1 120); do
  avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if ! pgrep -x hived >/dev/null && ! pgrep -x hived_glm >/dev/null && [ "$avail_kb" -ge "$need_kb" ]; then break; fi
  [ "$i" -eq 1 ] && echo "waiting for the previous hived to exit and free RAM (MemAvailable $((avail_kb / 1024 / 1024)) GiB)..."
  sleep 5
done

hive_env_args
docker run -d --name "$HIVE_CONTAINER" --gpus "device=$HIVE_GPU" -u "$(id -u):$(id -g)" -e HOME=/tmp "${HIVE_ENV_ARGS[@]}" \
  --ulimit memlock=-1:-1 --cap-add SYS_NICE --security-opt seccomp=unconfined --network host \
  -v "$HIVE_CKPT":"$HIVE_CKPT_MOUNT":ro "${FAMILY_MOUNTS[@]}" -v "$HIVE_ROOT":/hive -v "$HIVE_STATE_DIR":/out -w /hive/engine \
  --entrypoint bash "$HIVE_IMAGE" /hive/scripts/hive-entrypoint.sh \
    --ckpt "$HIVE_CKPT_MOUNT" --engram /out/engram --sock /out/hive.sock --vram-cache-mb "$HIVE_CACHE_MB" \
    --cpu-threads "$HIVE_CPU_THREADS" --max-ctx "$HIVE_MAX_CTX" --max-chunk "$HIVE_MAX_CHUNK" --max-batch "$HIVE_MAX_BATCH" \
    ${HIVE_HOST_SESSION_MB:+--host-session-mb "$HIVE_HOST_SESSION_MB"} "${EXTRA[@]}" >/dev/null
echo "$HIVE_DAEMON_BIN ($HIVE_MODEL_FAMILY) starting — log $HIVE_STATE_DIR/logs/hived.log · API :$HIVE_PORT (ready when the log prints '[hived] ready in')"
