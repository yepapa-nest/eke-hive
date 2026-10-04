#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Build the engine inside the project image (no GPU needed).
#   scripts/build.sh [BUILD_DIR] [JOBS]     default: engine/build, all cores
set -euo pipefail
. "$(dirname "$0")/lib.sh"; hive_load_config
DIR=${1:-$HIVE_BUILD}; JOBS=${2:-$(nproc)}
docker image inspect "$HIVE_IMAGE" >/dev/null 2>&1 || docker build -t "$HIVE_IMAGE" -f "$HIVE_ROOT/docker/Dockerfile" "$HIVE_ROOT"
exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$HIVE_ROOT":/hive -w /hive/engine "$HIVE_IMAGE" \
  bash -c "cmake -S . -B '$DIR' -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build '$DIR' -j $JOBS"
