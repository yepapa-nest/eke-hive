#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# ThreadSanitizer variants of the CPU suite + negative controls. No GPU, docker, SSH, or service.
# TSAN on this host aborts with "unexpected memory mapping" under ASLR, so only the test
# processes run under `setarch x86_64 -R` (see docs/validation.md).
# The known pool races (D2, D6) are fixed; tools/cpu_fake/tsan-known-pool-race.supp is empty and
# test_pool_cpu.py proves the D2 window with a must-detect mutant.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export HIVE_TEST_SANITIZER=thread TSAN_OPTIONS=${TSAN_OPTIONS:-halt_on_error=1:exitcode=66}
HIVE_TEST_BUILD_DIR=${HIVE_TEST_BUILD_DIR:-$(mktemp -d -t hive-tsan-objs-XXXXXX)}; export HIVE_TEST_BUILD_DIR
trap 'rm -rf "$HIVE_TEST_BUILD_DIR"' EXIT
R="setarch x86_64 -R"
echo '== TSAN: existing host fixtures'; $R python3 tools/test_host_cpu.py
echo '== TSAN: expert CPU pool (unsuppressed stress, cache checks, injected-race + D2-mutant negative controls)'; $R python3 tools/test_pool_cpu.py
echo '== TSAN: decode step handshake (dispatcher + real pool + fake GPU; relaxed-publication mutant must be reported)'; $R python3 tools/test_decode_handshake_cpu.py
echo '== TSAN: decode early route (S1 HIVE_DECODE_EARLY_ROUTE — Gate + run_layer on a fake GPU thread; relaxed post-number mutant must be reported)'; $R python3 tools/test_early_route_cpu.py
echo '== TSAN: paced promotion H2D (E4 HIVE_DECODE_COPY_PRIO — dispatcher after_dma + real store/pool)'; $R python3 tools/test_copy_prio_cpu.py
echo '== TSAN: snapshot fences + pinned pool matrix (+ skipped-fence negative control)'; $R python3 tools/test_snapshot_cpu.py
echo '== TSAN: real hived.cpp daemon matrix'; $R python3 tools/test_daemon_cpu.py
echo '== TSAN: server <-> daemon sessions'; $R python3 tools/test_server_session_cpu.py
echo '== TSAN: sleep/wake (Z1 — real store release/restore on async fake streams + real hived state machine)'; HIVE_SLEEP_TESTS=store,daemon $R python3 tools/test_sleep_cpu.py
echo '== TSAN negative control: daemon without snapshot-fence wait must be reported'
HIVE_DAEMON_NEGATIVE=fence $R python3 tools/test_daemon_cpu.py | tail -1
echo '== oracle negative controls (UBSan build): 20 hived.cpp mutants must each fail'
HIVE_TEST_SANITIZER=undefined HIVE_DAEMON_NEGATIVE=mutants python3 tools/test_daemon_cpu.py
echo 'TSAN suite passed (no suppressed races); GPU NOT executed.'
