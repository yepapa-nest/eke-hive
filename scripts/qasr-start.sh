#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Start the Qwen3-ASR speech recognition server (server/speech/asr_server.py on libhive_qasr.so) in its own container — docs/qasr.md.
#   scripts/qasr-start.sh [--build DIR] [--port P] [--max-ctx N] [--max-seqs N] [--wait]
#   Normally reached through scripts/hive-start.sh, which hands a qwen3_asr checkpoint (or HIVE_MODEL_FAMILY=qasr, HIVE_MODEL=asr) to this
#   script with its arguments; it can also be run directly. Settings: config/qasr.env (HIVE_CONTAINER hive_qasr, HIVE_PORT 8444,
#   HIVE_QASR_MAX_SEQS 2, HIVE_MAX_CTX 2048), overridden by the environment and the flags above.
#   HIVE_VRAM_LIMIT_MB caps the engine's start-up plan (the start fails, with the reason in the log, if the plan does not fit; nothing
#   is allocated per request). HIVE_DRY_RUN=1 prints the resolved settings and the command line and stops. Stop: scripts/hive-stop.sh
#   with the same settings, or `docker rm -f hive_qasr`.
set -u
. "$(dirname "$0")/lib.sh"
: "${HIVE_MODEL_FAMILY:=qasr}"; export HIVE_MODEL_FAMILY
hive_load_config; hive_need_ckpt
BUILD=build
wait_ready=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) HIVE_PORT=$2; shift 2 ;;
    --max-ctx) HIVE_MAX_CTX=$2; shift 2 ;;
    --max-seqs) HIVE_QASR_MAX_SEQS=$2; shift 2 ;;
    --build) BUILD=$2; shift 2 ;;
    --wait) wait_ready=1; shift ;;
    *) echo "qasr-start.sh: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
: "${HIVE_PORT:=8444}" "${HIVE_MAX_CTX:=2048}" "${HIVE_QASR_MAX_SEQS:=2}" "${HIVE_QASR_SERVED_NAME:=asr}" "${HIVE_BIND:=127.0.0.1}"
: "${HIVE_CONTAINER:=hive_qasr}" "${HIVE_IMAGE:=eke-hive:latest}" "${HIVE_GPU:=0}"
LIB=engine/$BUILD/libhive_qasr.so
vram_limit=${HIVE_VRAM_LIMIT_MB:-}
case "$vram_limit" in ""|0) vram_limit= ;; *[!0-9]*) echo "HIVE_VRAM_LIMIT_MB=$vram_limit is not a whole number of MiB — ignored" >&2; vram_limit= ;; esac
ARGS=(--ckpt "$HIVE_CKPT_MOUNT" --host "$HIVE_BIND" --port "$HIVE_PORT" --served-model-name "$HIVE_QASR_SERVED_NAME"
      --max-seqs "$HIVE_QASR_MAX_SEQS" --max-ctx "$HIVE_MAX_CTX" --lib "/hive/$LIB" ${vram_limit:+--vram-limit-mb "$vram_limit"})
case "${HIVE_DRY_RUN:-}" in ""|0) ;; *)
  echo "[dry-run] container $HIVE_CONTAINER · image $HIVE_IMAGE · gpu $HIVE_GPU · checkpoint $HIVE_CKPT -> $HIVE_CKPT_MOUNT"
  echo "[dry-run] library: $LIB$([ -f "$HIVE_ROOT/$LIB" ] || echo " (not built)")"
  echo "[dry-run] server: python3 server/speech/asr_server.py ${ARGS[*]}"
  exit 0 ;;
esac
[ -f "$HIVE_ROOT/$LIB" ] || { echo "$LIB not found — build the hive_qasr target (docs/qasr.md)" >&2; exit 1; }
# The engine allocates everything at start: ~5.1 GB with the shipped settings (or exactly the limit's worth when one is set).
need_mb=${vram_limit:-5600}
free_mb=$(nvidia-smi -i "$HIVE_GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$free_mb" ] && [ "$free_mb" -lt "$need_mb" ]; then
  echo "free VRAM $free_mb MiB < needed $need_mb MiB — the ASR engine cannot start beside the current GPU users" >&2; exit 1
fi
mkdir -p "$HIVE_STATE_DIR/logs"
docker rm -f "$HIVE_CONTAINER" >/dev/null 2>&1
# OPENBLAS/OMP threads 1: the log-mel matmul is tiny, and a BLAS thread per host core would burn a CPU-limited container's quota and
#   stall the engine's worker thread (asr_server.py sets the same defaults).
docker run -d --name "$HIVE_CONTAINER" --gpus "device=$HIVE_GPU" -u "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 ${vram_limit:+-e HIVE_VRAM_LIMIT_MB="$vram_limit"} --network host \
  -v "$HIVE_CKPT":"$HIVE_CKPT_MOUNT":ro -v "$HIVE_ROOT":/hive -v "$HIVE_STATE_DIR":/out -w /hive \
  --entrypoint bash "$HIVE_IMAGE" -c 'exec python3 /hive/server/speech/asr_server.py "$@" >> /out/logs/asr_server.log 2>&1' qasr "${ARGS[@]}" >/dev/null ||
  { echo "docker run failed" >&2; exit 1; }
echo "$HIVE_CONTAINER starting — log $HIVE_STATE_DIR/logs/asr_server.log · API $HIVE_BIND:$HIVE_PORT (/generate, /health)"
if [ "$wait_ready" = 1 ]; then
  for _ in $(seq 1 120); do
    curl -sf "http://$HIVE_BIND:$HIVE_PORT/health" >/dev/null 2>&1 && { echo "$HIVE_CONTAINER ready"; exit 0; }
    [ "$(docker container inspect -f '{{.State.Running}}' "$HIVE_CONTAINER" 2>/dev/null)" = true ] ||
      { echo "$HIVE_CONTAINER exited — tail of the log:" >&2; tail -5 "$HIVE_STATE_DIR/logs/asr_server.log" >&2; exit 1; }
    sleep 1
  done
  echo "$HIVE_CONTAINER not ready after 120 s" >&2; exit 1
fi
