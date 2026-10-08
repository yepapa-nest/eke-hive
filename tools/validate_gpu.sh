#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# GPU validation bundle — runs, in order and inside the container, the GPU checks for code written without a GPU (A, B, C1, D-1, D-2,
#   P1, P2, P3, P5, P6), **before** hive is started. Run only when no other process is using the GPU; do not run even the kernel-level tests
#   next to another GPU workload.
# Usage: scripts/hive-run.sh "bash /hive/tools/validate_gpu.sh [build]"
export HIVE_CACHE_PRIOR=0 HIVE_DECODE_DEFER=0  # exactness check: the decode routing changes of step 21 (on in config/hive.env) deliberately differ from the reference
set -uo pipefail
B=${1:-${HIVE_BUILD:-build}}
export HIVE_BUILD="$B"
cd /hive/engine || exit 1
fail=0
run() { echo "=== $1"; if ./$B/$1; then echo "--- $1 OK"; else echo "--- $1 FAIL"; fail=1; fi; echo; }
# 1) Kernel level: correctness (bit-identical or within tolerance) + speed samples
run test_idx_cand      # P1 candidate-pool indexer — must be bit-identical to the original path
run test_attn_flash    # P2 flash attention — within tolerance of v1/v2 + speed of one M=2048 layer (factor vs v2)
run test_woa           # P3 wo_a dequant+cuBLAS — compared with the original kernel and a host reference + speed
run test_hc_mix        # P5 hc mix — compared with the cuBLAS path + speed
run test_mx3           # D-1 prefill fp8 x fp4 tensor-core GEMM (option HIVE_MX_PREFILL=1) — must pass before it can be enabled
run test_idx_tc        # D-2 indexer fp4 x fp4 tensor core (option HIVE_IDX_TC=1)
run test_kernels       # existing kernel regressions (fusion, topk, gemm)
run test_idx_cand_tc   # tensor-core candidate-pool indexer (option HIVE_IDX_TC=1)
run test_hc_fuse       # decode hc: two kernels merged (HIVE_HC_FUSE2) — bit-identical
run test_idx_f32       # indexer fp32 scores (HIVE_IDX_F32) — host reference and top-512 overlap with the bf16 version (informational)
run test_mxg           # block-scale tile GEMM (option HIVE_MX_GEMM=1) — compared with tc and a reference + speed at real shapes (factor vs tc)
echo "Kernel-level result: $([ $fail -eq 0 ] && echo all passed || echo failures present)"
# 1b) HIVE_LAYER_YIELD_INTRA (opt-in: loads the model three times, ~minutes each): set HIVE_LY_INTRA_CKPT=<checkpoint dir in the container>
#   (e.g. /model) — a 65,536-row prompt prefilled while 2 decoders take 2 greedy steps per yield (period 50 ms), switch off / on / off again,
#   with fixed DMA shares, no promotion and pre-copies on (HIVE_PREFETCH=8 → the staging hold engages). Pass = the paused prompt's logits
#   (prefill + 32 decode steps) and SeqImage bytes identical off vs on (tools/compare_layer_yield_intra.py; off vs off2 = noise floor).
if [ -n "${HIVE_LY_INTRA_CKPT:-}" ]; then
  echo "=== layer_yield_intra"
  LYI=(./$B/test_layer_yield_intra --ckpt "$HIVE_LY_INTRA_CKPT" --engram /out/engram --len 65536 --dec 2 --dec-steps 2 --period 50 --steps 32 --promote 0)
  LYE=(env HIVE_DMA_FRAC=0.5 HIVE_DMA_FRAC_PREFILL=0.7 HIVE_PREFETCH=8 HIVE_PREFILL_SPLIT=0 HIVE_DECODE_SPLIT=0 HIVE_WARM_DEFER=0)
  "${LYE[@]}" HIVE_LAYER_YIELD_INTRA=0 "${LYI[@]}" --out /out/ly-intra-off &&
  "${LYE[@]}" HIVE_LAYER_YIELD_INTRA=1 "${LYI[@]}" --out /out/ly-intra-on &&
  "${LYE[@]}" HIVE_LAYER_YIELD_INTRA=0 "${LYI[@]}" --out /out/ly-intra-off2 &&
  python3 /hive/tools/compare_layer_yield_intra.py /out/ly-intra-off /out/ly-intra-on /out/ly-intra-off2
  rc=$?; if [ $rc -eq 0 ]; then echo "--- layer_yield_intra OK"; else echo "--- layer_yield_intra FAIL (rc $rc)"; fail=1; fi; echo
fi
echo "Validated build=$HIVE_BUILD; pass HIVE_BUILD=$HIVE_BUILD explicitly to the follow-up commands below as well."
cat <<'EOF'

Follow-up checks (run separately):
  2) Per-layer golden comparison (no daemon):
       bash /hive/tools/validate_layers.sh 0     # also layers 1,2,8,14,20,24,28
       bash /hive/tools/validate_inject.sh 20    # per-layer injection (layer argument required; isolates accumulated error)
     Repeat with HIVE_ATTN_V2=1 and check that the difference stays at v1/v2 noise level.
  3) Speculative decoding: bash /hive/tools/test_spec.sh
     (acceptance on random tokens must stay low (<=40%); greedy output must match the no-MTP output)
  4) Daemon end-to-end, after scripts/hive-start.sh: python3 /hive/tools/e2e_smoke.py
     (a second turn in the same session must log "[hived] ... reuse N/M via prompt-ckpt|live")
  5) Tiled prefill: bash /hive/tools/test_tile.sh 40000 3
     (token sequences and logits of sequential chunks vs one tile must match)
  6) Performance: HIVE_PROFILE=1 for one 16K chunked prefill, and bash /hive/tools/decode_kernels.sh
  7) Intra-layer yields (HIVE_LAYER_YIELD_INTRA): HIVE_LY_INTRA_CKPT=/model bash /hive/tools/validate_gpu.sh  (step 1b above), then live:
       python3 /hive/tools/layer_yield_check.py --long 85000 --label off|on  (decoder max gap during the long prompt; tokens vs an off2 run)
EOF
# Switches (value "0" or empty = off throughout): HIVE_ATTN_V1/V2/FLASH32/TC, HIVE_WOA_TC, HIVE_HC_CUBLAS/HC_MIX_ROWS4/HC_FUSE2, HIVE_IDX_TC/IDX_F32 (exclusive),
#   HIVE_MX_GEMM (> MX_PREFILL), HIVE_STAGING/PREFETCH/PREFETCH_MIN_ROWS, HIVE_NO_GRAPH, HIVE_PROFILE (N>0) — all default to the original behaviour
exit $fail
