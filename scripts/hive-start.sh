#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Start the daemon (hived for DeepSeek-V4.1-Flash, hived_glm for GLM-5.3-Flash, hived_qwen for Qwen3.8-Flash-Next — chosen from the checkpoint)
#   and the OpenAI/Anthropic-compatible API server in one container.
#   scripts/hive-start.sh [--build DIR] [--vram-cache-mb N] [--max-ctx N] [--max-chunk N] [--port P]
#                         [--cpu-threads N] [--max-batch N] [extra hived arguments...]
# Settings come from HIVE_CONFIG (default config/hive.env, config/glm.env for a GLM checkpoint, config/qwen.env for a Qwen3.8-Flash-Next
#   checkpoint; see docs/configuration.md), or from a model + VRAM profile: HIVE_MODEL=ds|glm|fn|27b HIVE_VRAM_GB=96|72|48|32|24|16|12|8|0
#   (config/profiles/, scripts/lib.sh hive_apply_profile — explicit settings still win);
#   HIVE_DRY_RUN=1 prints the resolved settings and the daemon command line and stops (no GPU, no docker);
# command-line flags override them.
# The binary is chosen only by --build (default engine/build). HIVE_BUILD from the shell is ignored on purpose:
# a value left over from a test session would otherwise start an unreviewed build without any visible trace.
set -u
. "$(dirname "$0")/lib.sh"
shell_build=${HIVE_BUILD:-}; unset HIVE_BUILD
hive_load_config; hive_need_ckpt; hive_detect_family
# Qwen3-ASR (qwen3_asr) is not a hived family: its own container runs server/speech/asr_server.py on libhive_qasr.so (docs/qasr.md)
[ "$HIVE_MODEL_FAMILY" = qasr ] && exec "$(dirname "$0")/qasr-start.sh" "$@"
# Qwen3-TTS (qwen3_tts) likewise: server/speech/tts_server.py on libhive_qtts.so in its own container (docs/qtts.md)
[ "$HIVE_MODEL_FAMILY" = qtts ] && exec "$(dirname "$0")/qtts-start.sh" "$@"
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
hive_profile_preflight
GEMMA_MOUNTS=()
# Gemma: the MTP assistant checkpoint (HIVE_GEMMA_ASSISTANT; unset = a known assistant directory next to HIVE_CKPT, 0 = no drafter),
#   mounted read-only at the same path
if [ "${HIVE_MODEL_FAMILY:-}" = gemma ]; then
  if [ -z "${HIVE_GEMMA_ASSISTANT:-}" ]; then
    for n in google-gemma-4-12B-it-qat-q4_0-unquantized-assistant gemma-4-12B-it-qat-q4_0-unquantized-assistant; do
      [ -f "$(dirname "$HIVE_CKPT")/$n/config.json" ] && { HIVE_GEMMA_ASSISTANT=$(dirname "$HIVE_CKPT")/$n; break; }
    done
  fi
  if [ -n "${HIVE_GEMMA_ASSISTANT:-}" ] && [ "$HIVE_GEMMA_ASSISTANT" != 0 ]; then
    [ -f "$HIVE_GEMMA_ASSISTANT/config.json" ] || { echo "HIVE_GEMMA_ASSISTANT=$HIVE_GEMMA_ASSISTANT has no config.json" >&2; exit 1; }
    export HIVE_GEMMA_ASSISTANT
    GEMMA_MOUNTS=(-v "$HIVE_GEMMA_ASSISTANT":"$HIVE_GEMMA_ASSISTANT":ro)
  else
    unset HIVE_GEMMA_ASSISTANT
  fi
fi

dry_run=0
case "${HIVE_DRY_RUN:-}" in ""|0) ;; *) dry_run=1 ;; esac
if [ "$dry_run" = 1 ]; then  # resolved settings and the command line only (HIVE_DRY_RUN — tools/test_profiles_cpu.sh)
  bin=$HIVE_DAEMON_BIN
  [ "${HIVE_GPU:-0}" = none ] && bin=$(hive_cpu_daemon "$HIVE_DAEMON_BIN")
  [ "$HIVE_MODEL_FAMILY" = glm ] && { : "${HIVE_CACHE_FIT:=1}"; export HIVE_CACHE_FIT; }
  echo "[dry-run] resolved environment:"
  compgen -e | grep '^HIVE_' | sort | while read -r v; do echo "  $v=${!v}"; done
  echo "[dry-run] daemon: engine/$BUILD/${bin:-?}$([ -x "$HIVE_ROOT/engine/$BUILD/$bin" ] || echo " (not built)") · gpu ${HIVE_GPU:-0}"
  echo "[dry-run] args: --ckpt $HIVE_CKPT_MOUNT --engram /out/engram --sock /out/hive.sock --vram-cache-mb $HIVE_CACHE_MB --cpu-threads $HIVE_CPU_THREADS" \
       "--max-ctx $HIVE_MAX_CTX --max-chunk $HIVE_MAX_CHUNK --max-batch $HIVE_MAX_BATCH ${HIVE_HOST_SESSION_MB:+--host-session-mb $HIVE_HOST_SESSION_MB }${EXTRA[*]}"
  exit 0
fi
[ "${HIVE_GPU:-0}" = none ] || [ -x "$HIVE_ROOT/engine/$BUILD/$HIVE_DAEMON_BIN" ] || { echo "engine/$BUILD/$HIVE_DAEMON_BIN not found — run scripts/build.sh first" >&2; exit 1; }
# GLM: no engram tables; the expert cache is sized after the session pool (HIVE_CACHE_FIT) unless set explicitly; the optional
#   Z.ai FP8 tensors (HIVE_GLM_FP8_DIR, docs/glm.md) are mounted read-only at the same path.
FAMILY_MOUNTS=("${GEMMA_MOUNTS[@]}")
if [ "$HIVE_MODEL_FAMILY" = glm ]; then
  : "${HIVE_CACHE_FIT:=1}"; export HIVE_CACHE_FIT
  if [ -n "${HIVE_GLM_FP8_DIR:-}" ] && [ "$HIVE_GLM_FP8_DIR" != 0 ]; then
    [ -f "$HIVE_GLM_FP8_DIR/model.safetensors.index.json" ] || { echo "HIVE_GLM_FP8_DIR=$HIVE_GLM_FP8_DIR has no model.safetensors.index.json" >&2; exit 1; }
    FAMILY_MOUNTS=(-v "$HIVE_GLM_FP8_DIR":"$HIVE_GLM_FP8_DIR":ro)
  fi
fi
if [ "${HIVE_MODEL_FAMILY:-}" = q27 ] && [ -n "${HIVE_Q27_DFLASH:-}" ] && [ "$HIVE_Q27_DFLASH" != 0 ]; then   # DFlash2 drafter, read-only at the same path
  [ -f "$HIVE_Q27_DFLASH/model.safetensors" ] || { echo "HIVE_Q27_DFLASH=$HIVE_Q27_DFLASH has no model.safetensors" >&2; exit 1; }
  FAMILY_MOUNTS+=(-v "$HIVE_Q27_DFLASH":"$HIVE_Q27_DFLASH":ro)
fi

# HIVE_SESSION_STORE: the persistent session store (docs/configuration.md) — a host directory, mounted read-write at the same path
#   (the daemon keeps one subdirectory per model and build in it; several families may share the directory)
if [ -n "${HIVE_SESSION_STORE:-}" ] && [ "$HIVE_SESSION_STORE" != 0 ]; then
  mkdir -p "$HIVE_SESSION_STORE" || { echo "HIVE_SESSION_STORE=$HIVE_SESSION_STORE cannot be created" >&2; exit 1; }
  FAMILY_MOUNTS+=(-v "$HIVE_SESSION_STORE":"$HIVE_SESSION_STORE")
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
# HIVE_VRAM_LIMIT_MB (docs/configuration.md): the daemon behaves as if the card had this much memory and holds the rest as a ballast, so
#   exactly that much must be free — the fixed 12,000 MiB guess above is for the whole card and would refuse a small limit's start.
vram_limit=${HIVE_VRAM_LIMIT_MB:-}
case "$vram_limit" in ""|0) vram_limit= ;; *[!0-9]*) echo "HIVE_VRAM_LIMIT_MB=$vram_limit is not a whole number of MiB — ignored" >&2; vram_limit= ;; esac
if [ -n "$vram_limit" ]; then need_mb=$vram_limit; export HIVE_VRAM_LIMIT_MB; else unset HIVE_VRAM_LIMIT_MB; fi
# HIVE_GPU=none (CPU only) needs no VRAM.
free_mb=$need_mb
[ "${HIVE_GPU:-0}" = none ] || free_mb=$(nvidia-smi -i "$HIVE_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${free_mb:-0}" -lt "$need_mb" ]; then
  echo "free VRAM ${free_mb:-?} MiB < needed $need_mb MiB${vram_limit:+ (HIVE_VRAM_LIMIT_MB)} — stop other GPU users (or lower HIVE_CACHE_MB when HIVE_CACHE_FIT is off; nvidia-smi must be on the host PATH)" >&2; exit 1
fi

mkdir -p "$HIVE_STATE_DIR/logs"
[ "$HIVE_MODEL_FAMILY" != deepseek ] || [ -d "$HIVE_STATE_DIR/engram" ] || { echo "$HIVE_STATE_DIR/engram missing — run scripts/prepare.sh first" >&2; exit 1; }
docker rm -f "$HIVE_CONTAINER" >/dev/null 2>&1
# HIVE_START_WAIT_OTHERS=0: other families' daemons may stay alive beside this one (e.g. one asleep, or two models sharing the card
#   under VRAM caps) — then wait only for the RAM, not for every other hived process to exit (default 1 = unchanged).
# hived_gemma is not waited for: the live chat model runs beside a large model under VRAM caps (its pinned RAM is small).
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
  # CPU only: another family's GPU daemon may keep running beside this one — wait only for a previous CPU daemon and for the RAM.
  if [ "${HIVE_GPU:-0}" = none ]; then
    if ! pgrep -x hived_qwen_cpu >/dev/null && ! pgrep -x hived_q27_cpu >/dev/null && ! pgrep -x hived_glm_cpu >/dev/null && ! pgrep -x hived_ds_cpu >/dev/null && [ "$avail_kb" -ge "$need_kb" ]; then break; fi
  elif { [ "${HIVE_START_WAIT_OTHERS:-1}" = 0 ] || { ! pgrep -x hived >/dev/null && ! pgrep -x hived_glm >/dev/null && ! pgrep -x hived_qwen >/dev/null && ! pgrep -x hived_qwen_cpu >/dev/null && ! pgrep -x hived_q27 >/dev/null && ! pgrep -x hived_q27_cpu >/dev/null; }; } && [ "$avail_kb" -ge "$need_kb" ]; then break; fi
  [ "$i" -eq 1 ] && echo "waiting for the previous hived to exit and free RAM (MemAvailable $((avail_kb / 1024 / 1024)) GiB)..."
  sleep 5
done

hive_env_args
# HIVE_GPU=none: CPU only — no GPU is attached and the family's CPU daemon runs (hived_qwen_cpu · hived_glm_cpu · hived_ds_cpu · hived_q27_cpu)
GPU_ARGS=(--gpus "device=$HIVE_GPU")
if [ "${HIVE_GPU:-0}" = none ]; then
  cpu_bin=$(hive_cpu_daemon "$HIVE_DAEMON_BIN")
  [ -n "$cpu_bin" ] || { echo "HIVE_GPU=none: no CPU-only daemon for the $HIVE_MODEL_FAMILY family yet" >&2; exit 1; }
  HIVE_DAEMON_BIN=$cpu_bin
  [ -x "$HIVE_ROOT/engine/$BUILD/$HIVE_DAEMON_BIN" ] || { echo "engine/$BUILD/$HIVE_DAEMON_BIN not found — run scripts/build.sh first" >&2; exit 1; }
  # SMT helper threads join only GEMM dispatches (prefill), never decode: +2–4 % prefill measured on FN / 27B, +16–39 % MoE prefill
  #   in isolation, output unchanged (cpuopt 2026-10-09). Set HIVE_CPU_POOL_SMT=0 to turn off.
  : "${HIVE_CPU_POOL_SMT:=1}"; export HIVE_CPU_POOL_SMT
  export HIVE_DAEMON_BIN
  hive_env_args
  GPU_ARGS=()
fi
docker run -d --name "$HIVE_CONTAINER" "${GPU_ARGS[@]}" -u "$(id -u):$(id -g)" -e HOME=/tmp "${HIVE_ENV_ARGS[@]}" \
  --ulimit memlock=-1:-1 --cap-add SYS_NICE --security-opt seccomp=unconfined --network host \
  -v "$HIVE_CKPT":"$HIVE_CKPT_MOUNT":ro "${FAMILY_MOUNTS[@]}" -v "$HIVE_ROOT":/hive -v "$HIVE_STATE_DIR":/out -w /hive/engine \
  --entrypoint bash "$HIVE_IMAGE" /hive/scripts/hive-entrypoint.sh \
    --ckpt "$HIVE_CKPT_MOUNT" --engram /out/engram --sock /out/hive.sock --vram-cache-mb "$HIVE_CACHE_MB" \
    --cpu-threads "$HIVE_CPU_THREADS" --max-ctx "$HIVE_MAX_CTX" --max-chunk "$HIVE_MAX_CHUNK" --max-batch "$HIVE_MAX_BATCH" \
    ${HIVE_HOST_SESSION_MB:+--host-session-mb "$HIVE_HOST_SESSION_MB"} "${EXTRA[@]}" >/dev/null
echo "$HIVE_DAEMON_BIN ($HIVE_MODEL_FAMILY) starting — log $HIVE_STATE_DIR/logs/hived.log · API :$HIVE_PORT (ready when the log prints '[hived] ready in')"
