#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
. "$(dirname "$0")/lib.sh"; hive_load_config
docker rm -f "$HIVE_CONTAINER" >/dev/null 2>&1 && echo "$HIVE_CONTAINER stopped" || echo "$HIVE_CONTAINER not running"
# `docker rm -f` occasionally reports "did not receive exit event" and leaves an Exited container behind.
docker container inspect "$HIVE_CONTAINER" >/dev/null 2>&1 && docker rm "$HIVE_CONTAINER" >/dev/null 2>&1
exit 0
