#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# One-time preparation for DeepSeek-V4.1-Flash: export the Engram hash constants from the checkpoint into $HIVE_STATE_DIR/engram.
# GLM-5.3-Flash has no Engram tables and skips this step.
set -euo pipefail
. "$(dirname "$0")/lib.sh"; hive_load_config; hive_need_ckpt
mkdir -p "$HIVE_STATE_DIR/logs"
exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$HIVE_CKPT":"$HIVE_CKPT_MOUNT":ro -v "$HIVE_ROOT":/hive -v "$HIVE_STATE_DIR":/out "$HIVE_IMAGE" \
  python3 /hive/oracle/export_engram_hash.py --ckpt "$HIVE_CKPT_MOUNT" --out /out/engram
