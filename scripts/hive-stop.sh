#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
. "$(dirname "$0")/lib.sh"; hive_load_config
docker container inspect "$HIVE_CONTAINER" >/dev/null 2>&1 || { echo "$HIVE_CONTAINER not running"; exit 0; }
# `docker rm -f` can fail with "did not receive an exit event" while the daemon is still exiting (releasing pinned host memory takes
#   seconds) — the container is then still there. Wait for it to go instead of reporting it stopped (a start right after found the GPU
#   still held: 2026-10-06).
docker rm -f "$HIVE_CONTAINER" >/dev/null 2>&1
for _ in $(seq 1 180); do
  docker container inspect "$HIVE_CONTAINER" >/dev/null 2>&1 || { echo "$HIVE_CONTAINER stopped"; exit 0; }
  [ "$(docker container inspect -f '{{.State.Running}}' "$HIVE_CONTAINER" 2>/dev/null)" = false ] && docker rm "$HIVE_CONTAINER" >/dev/null 2>&1
  sleep 1
done
echo "$HIVE_CONTAINER did not stop within 180 s" >&2
exit 1
