#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# No GPU, model loading, service launch/restart, or SSH.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Shared, content-hashed object cache for the fake-CUDA builds below (removed on exit).
HIVE_TEST_BUILD_DIR=${HIVE_TEST_BUILD_DIR:-$(mktemp -d -t hive-cpu-objs-XXXXXX)}; export HIVE_TEST_BUILD_DIR
trap 'rm -rf "$HIVE_TEST_BUILD_DIR"' EXIT
python3 tools/test_cache_cpu.py
python3 tools/test_cache_replay_cpu.py
python3 tools/test_host_cpu.py
T=$(mktemp -d)/grouped_plan; g++ -std=c++20 -O1 -Iengine/include engine/tests/test_grouped_plan_cpu.cpp -o "$T" && "$T"  # grouped prefill plan (slot/release invariants)
T=$(mktemp -d)/prefill_split; g++ -std=c++20 -O1 -Iengine/include engine/tests/test_prefill_split_cpu.cpp -o "$T" && "$T"  # HIVE_PREFILL_SPLIT=balance cost model (cold start, limits, convergence)
T=$(mktemp -d)/decode_split; g++ -std=c++20 -O1 -Iengine/include engine/tests/test_decode_split_cpu.cpp -o "$T" && "$T"  # HIVE_DECODE_SPLIT=balance cost model (table, cold start, verify back-off, exhaustive minimum, synthetic world)
T=$(mktemp -d)/bench_expert_cpu; g++ -std=c++20 -O3 -mavx2 -mfma -mf16c -pthread -Itools/cpu_fake/include -Iengine/include engine/tests/bench_expert_cpu.cpp engine/src/cpu/expert_cpu.cpp -o "$T" && "$T" --quick  # HIVE_CPU_GEMV2 _v2 == _ref bitwise (timing: bench_expert_cpu without arguments; pool: tools/bench_pool_cpu.py)
T=$(mktemp -d)/mtp_gate; g++ -std=c++20 -O1 -Iengine/include engine/tests/test_mtp_gate_cpu.cpp -o "$T" && "$T"  # HIVE_MTP_GATE2 gate (table, interpolation, freshness, exhaustive k, trap reproduction, learning, back-off, synthetic world); batch re-measurement and first-use samples (trap + negative control of the old rule)
T=$(mktemp -d)/verify_rows; g++ -std=c++20 -O1 -Iengine/include engine/tests/test_verify_rows_cpu.cpp -o "$T" && "$T"  # HIVE_MTP_VERIFY2/BATCH verify-row design (table formulas = verify_rows.h, sequential comparison, future independence, continuation after rollback, multi-step batch schedule == sequence alone, 7 negative controls)
T=$(mktemp -d)/short_prefill; g++ -std=c++20 -O1 -Iengine/include engine/tests/test_short_prefill_cpu.cpp -o "$T" && "$T"  # short-prefill decision formulas (TAIL_SHORT, PREFETCH_SCALE, SHORT_ADAPT — off = baseline candidates/kinds) + runtime.cpp text contracts
python3 tools/test_validation_cpu.py
python3 tools/test_server_cpu.py
python3 tools/test_glm_family_cpu.py  # GLM family: effort mapping, unset = high, HIVE_GLM_NOTHINK
python3 tools/test_hive_monitor_cpu.py  # hived.log monitor, daily report, lossless concurrent writes with an external log-rotate script (if present)
# Real CPU expert pool, snapshot/pinned pool/fences, option plumbing, real hived.cpp
# on fake CUDA/runtime, server<->daemon sessions. TSAN variants + negative controls: tools/test_tsan_cpu.sh
python3 tools/test_pool_cpu.py
python3 tools/test_decode_handshake_cpu.py  # HIVE_DECODE_STEP_GRAPH: sort and plan == baseline, dispatcher handshake (real pool)
python3 tools/test_copy_prio_cpu.py  # HIVE_DECODE_COPY_PRIO: paced promotion == immediate promotion (decisions, bytes), resident at t+2, dispatcher pacing
python3 tools/test_cache_fit_cpu.py  # HIVE_CACHE_FIT: deferred slot allocation == baseline (real ExpertStore comparison), real hived startup order (after the session pool), HIVE_ENGRAM_ALIAS text contract
python3 tools/test_decode_ubatch_cpu.py  # HIVE_DECODE_HOST_FAST grouping == baseline, HIVE_DECODE_UBATCH half-pipeline order (fake GPU thread + pool, 5 mutants)
python3 tools/test_pregate_cpu.py  # HIVE_DECODE_PREGATE: owned spans == start_jobs items (exact cover, negative control), prediction cleanup, text contracts (switch, launch, waits, ownership only on decode layers)
python3 tools/test_batch_miss_cpu.py  # batched decode misses: ring tracking safe and complete, classification == baseline (empty ring), prediction, 2 mutants, switch string contract
python3 tools/test_early_route_cpu.py  # HIVE_DECODE_EARLY_ROUTE: post-number wait (Gate), per-layer host order (run_layer) — fake GPU thread + pool, 6 mutants, runtime.cpp text contracts
python3 tools/test_snapshot_cpu.py
python3 tools/test_options_cpu.py
python3 tools/test_daemon_cpu.py
python3 tools/test_mtp_cpu.py  # real hived.cpp DSpark schedule: oracle tokens (MTP on/off, gate 1/2, wrong drafts, batch), gate 2 active, old gate trap reproduction, switch parsing, first-use samples (negative control = runtime hiding first uses); mutants with HIVE_MTP_NEGATIVE=1
python3 tools/test_server_session_cpu.py
python3 tools/bench_mtp_batch.py --self-test  # GPU-window measurement tool (comparison and log aggregation functions)
python3 -B -c 'import ast,pathlib; files=[p for base in ("server","tools","oracle") for p in pathlib.Path(base).rglob("*.py")]; [ast.parse(p.read_text(),filename=str(p)) for p in files]; print("Python AST:",len(files),"passed")'
for file in tools/*.sh scripts/*.sh; do bash -n "$file"; done
if git rev-parse --git-dir >/dev/null 2>&1; then git diff --check; fi  # a release archive is not a checkout
python3 tools/test_verify_fused_cpu.py
python3 tools/test_cache_elastic_cpu.py
python3 tools/test_prefill_yield_cpu.py
python3 tools/test_layer_yield_cpu.py  # HIVE_LAYER_YIELD: yield at layer boundaries inside a long prefill forward (real hived.cpp, fake layer model, lent-state comparison, 4 mutants, lent-list text contract)
python3 tools/test_quality_eval_cpu.py
python3 tools/test_gap_eval_cpu.py
python3 tools/test_sampler_cpu.py  # hive/sampler.h: sample/sample_cands == definition, lossless speculative acceptance rule (joint distribution), top_p truncation mechanism, request seeds, 4 header mutants
python3 tools/test_think_cap_cpu.py  # thinking cap (think_cap): default off unchanged, </think> after exactly N tokens (plain, MTP, wrong drafts, batched MTP), exit phrase (think_exit_ids), absorption, server fields/records, 6 mutants
python3 tools/test_sleep_cpu.py  # sleep/wake: real ExpertStore release, re-allocation, restore (bytes, scores, state, 4 mutants), real hived state machine, server endpoints
python3 tools/test_load_par_cpu.py  # HIVE_LOAD_PAR cold load: bulk_load == pread, real ExpertStore load_bulk == per-layer load (5 variants + failure absorption), real hived absorption path transcript identical, 1 mutant
python3 tools/test_engram_ssd_cpu.py  # HIVE_ENGRAM_SSD: peek == compute, SSD gather == RAM engram_gather bytes (modes, cache, threads, read-ahead, concurrency), failure absorption, real ExpertStore (split shards included), 3 negative controls
echo 'CPU suite passed; GPU numerical/performance verification NOT executed.'
