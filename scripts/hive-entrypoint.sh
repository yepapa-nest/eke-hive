#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Container supervisor started by hive-start.sh: runs the daemon (hived or hived_glm) and the API server; when either exits, the other is
# stopped and the container exits with that status.
set -uo pipefail
daemon_pid= server_pid=
cleanup() {
  trap - TERM INT EXIT
  [ -z "$daemon_pid" ] || kill -TERM "$daemon_pid" 2>/dev/null || true
  [ -z "$server_pid" ] || kill -TERM "$server_pid" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Record exactly which binary is being started.
BIN_DIR=/hive/engine/${HIVE_BUILD:-build}
BIN=${HIVE_DAEMON_BIN:-hived}  # hived (DeepSeek) or hived_glm (GLM) — chosen by hive-start.sh from the checkpoint
BIN_ID=$( { sha256sum "$BIN_DIR/$BIN" 2>/dev/null | cut -c1-16; } || true)
BIN_ELF=$( { readelf -n "$BIN_DIR/$BIN" 2>/dev/null | awk '/Build ID/{print $3}'; } || true)
BIN_TS=$( { stat -c %y "$BIN_DIR/$BIN" 2>/dev/null | cut -c1-19; } || true)
msg="[supervisor] build dir=$BIN_DIR binary=$BIN_DIR/$BIN sha256=${BIN_ID:-?} elf-build-id=${BIN_ELF:-?} mtime=${BIN_TS:-?}${HIVE_BUILD:+ (HIVE_BUILD=$HIVE_BUILD)}"
[ "${HIVE_BUILD:-build}" = build ] || msg="$msg (not the default engine/build)"
echo "$msg" >&2; echo "$msg" >> /out/logs/hived.log
"$BIN_DIR/$BIN" "$@" >> /out/logs/hived.log 2>&1 &
daemon_pid=$!
python3 /hive/server/hive_server.py --ckpt "${HIVE_CKPT_MOUNT:-/model}" --sock /out/hive.sock --host "${HIVE_BIND:-127.0.0.1}" --port "${HIVE_PORT:-8430}" >> /out/logs/server.log 2>&1 &
server_pid=$!
wait -n "$daemon_pid" "$server_pid"
status=$?
echo "[supervisor] child exited status=$status; stopping sibling" >&2
exit "$status"
