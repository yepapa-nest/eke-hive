#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Run one command in a GPU container with the repository at /hive, the checkpoint at $HIVE_CKPT_MOUNT and the
# state directory at /out (for kernel tests and validation scripts). Stop the server first: it uses the whole GPU.
#   scripts/hive-run.sh "<bash command run in /hive/engine>"
set -u
. "$(dirname "$0")/lib.sh"; hive_load_config
hive_env_args
FAMILY_MOUNTS=()  # GLM: the optional Z.ai FP8 tensors, mounted read-only at the same path as in scripts/hive-start.sh
[ -n "${HIVE_GLM_FP8_DIR:-}" ] && [ "$HIVE_GLM_FP8_DIR" != 0 ] && [ -d "$HIVE_GLM_FP8_DIR" ] && FAMILY_MOUNTS=(-v "$HIVE_GLM_FP8_DIR":"$HIVE_GLM_FP8_DIR":ro)
exec docker run --rm --gpus "device=$HIVE_GPU" -u "$(id -u):$(id -g)" -e HOME=/tmp "${HIVE_ENV_ARGS[@]}" \
  --ulimit memlock=-1:-1 --cap-add SYS_NICE --security-opt seccomp=unconfined \
  ${HIVE_CKPT:+-v "$HIVE_CKPT":"$HIVE_CKPT_MOUNT":ro} -v "$HIVE_ROOT":/hive -v "$HIVE_STATE_DIR":/out "${FAMILY_MOUNTS[@]}" -w /hive/engine \
  --entrypoint bash "$HIVE_IMAGE" -c "$*"
