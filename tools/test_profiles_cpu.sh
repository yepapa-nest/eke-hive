#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# CPU-only test of the model + VRAM profiles (config/profiles/, scripts/lib.sh hive_apply_profile): dry-runs scripts/hive-start.sh
#   (HIVE_DRY_RUN=1 — no docker, no GPU) for every HIVE_MODEL × HIVE_VRAM_GB, against fake checkpoints (config.json only) and a temporary
#   state directory, and checks the derived settings, the precedence of explicit settings, the RAM-vs-disk switch and the pre-flight stop.
#   tools/test_profiles_cpu.sh            (exit 0 = all passed)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/models" "$T/state/engram"
mk() { mkdir -p "$T/models/$1"; echo "{\"model_type\": \"$2\"}" > "$T/models/$1/config.json"; }
mk DeepSeek-V4.1-Flash deepseek_v41
mk GLM-5.3-Flash-NVFP4 glm5_next
mk Qwen3.8-Flash-Next-NVFP4 qwen4_exp
mk Qwen3.8-27B-NVFP4 qwen3_5
fails=0 n=0
# run a dry-run with a clean HIVE_* environment plus the given assignments; output in $OUT, status in $RC
run() {
  OUT=$(env -i PATH="$PATH" HOME="$T" HIVE_MODELS_DIR="$T/models" HIVE_STATE_DIR="$T/state" HIVE_DRY_RUN=1 \
        HIVE_TEST_MEMTOTAL_GB=${MEMT:-1024} HIVE_TEST_MEMAVAIL_GB=${MEMA:-1000} HIVE_TEST_CKPT_GB=${CKGB:-9999} "$@" \
        bash "$ROOT/scripts/hive-start.sh" 2>&1); RC=$?
}
val() { sed -n "s/^  $1=//p" <<< "$OUT" | head -1; }
check() {  # name, got, want
  n=$((n + 1))
  if [ "$2" != "$3" ]; then fails=$((fails + 1)); echo "FAIL $1: got '$2', want '$3'"; fi
}
has() {  # name, pattern (grep -E) in OUT
  n=$((n + 1)); grep -qE -- "$2" <<< "$OUT" || { fails=$((fails + 1)); echo "FAIL $1: no /$2/ in output"; }
}

declare -A FAM=([ds]=deepseek [glm]=glm [fn]=qwen [27b]=q27)
declare -A BIN=([ds]=hived [glm]=hived_glm [fn]=hived_qwen [27b]=hived_q27)
declare -A CK=([ds]=DeepSeek-V4.1-Flash [glm]=GLM-5.3-Flash-NVFP4 [fn]=Qwen3.8-Flash-Next-NVFP4 [27b]=Qwen3.8-27B-NVFP4)
# 1. every combination resolves, with the derived values
for m in ds glm fn 27b; do
  for gb in 96 72 48 32 24 16 12 8 0; do
    c=$m-$gb
    run HIVE_MODEL=$m HIVE_VRAM_GB=$gb
    check "$c rc" "$RC" 0
    [ "$RC" = 0 ] || { echo "$OUT" | tail -5; continue; }
    check "$c profile" "$(val HIVE_PROFILE)" "$c"
    check "$c family" "$(val HIVE_MODEL_FAMILY)" "${FAM[$m]}"
    check "$c ckpt" "$(val HIVE_CKPT)" "$T/models/${CK[$m]}"
    if [ "$gb" = 0 ]; then
      check "$c gpu" "$(val HIVE_GPU)" none
      check "$c limit" "$(val HIVE_VRAM_LIMIT_MB)" ""
      has "$c cpu daemon" "daemon: engine/build/$(c=${BIN[$m]}_cpu; echo ${c/hived_cpu/hived_ds_cpu})"
    else
      check "$c gpu" "$(val HIVE_GPU)" 0
      if [ "$gb" = 96 ]; then check "$c limit" "$(val HIVE_VRAM_LIMIT_MB)" ""
      else check "$c limit" "$(val HIVE_VRAM_LIMIT_MB)" $((gb * 1024)); fi
      has "$c daemon" "daemon: engine/build/${BIN[$m]}( |$)"
    fi
    check "$c ram wait" "$(val HIVE_RAM_WAIT_GB)" "$(val HIVE_PROFILE_NEED_GB)"
    has "$c preflight line" "^\[profile\] $c: family ${FAM[$m]}"
    # every key of the profile files reaches the environment (or is a launcher-only key)
    for f in "$ROOT/config/profiles/$c.env" "$ROOT/config/profiles/$m.env"; do
      while IFS='=' read -r k v; do
        case "$k" in ''|\#*) continue ;; esac
        if [ "$k" = HIVE_DAEMON_ARGS ]; then has "$c $k" "args: .* ${v}$"; continue; fi
        [ "$f" = "$ROOT/config/profiles/$m.env" ] && grep -q "^$k=" "$ROOT/config/profiles/$c.env" && continue  # the size file wins
        check "$c $k" "$(val "$k")" "$v"
      done < "$f"
    done
  done
done
# 2. shipped knobs at 96 GB equal the family config (no behaviour change for the big card)
run HIVE_MODEL=ds HIVE_VRAM_GB=96
check "ds-96 max chunk" "$(val HIVE_MAX_CHUNK)" 16384
check "ds-96 daemon args" "$(val HIVE_DAEMON_ARGS)" "--gpu-share 0 --promote 8 --prefill-tile 3 --decoder-replay"
run HIVE_MODEL=glm HIVE_VRAM_GB=96
check "glm-96 reserve" "$(val HIVE_CACHE_RESERVE_MB)" 3000
# 3. precedence: the environment wins; an explicit HIVE_CONFIG wins over the profile; HIVE_CKPT is kept
run HIVE_MODEL=ds HIVE_VRAM_GB=12 HIVE_MAX_CHUNK=2048 HIVE_VRAM_LIMIT_MB=10000 HIVE_GPU=1
check "env wins chunk" "$(val HIVE_MAX_CHUNK)" 2048
check "env wins limit" "$(val HIVE_VRAM_LIMIT_MB)" 10000
check "env wins gpu" "$(val HIVE_GPU)" 1
printf 'HIVE_MAX_CHUNK=8192\nHIVE_CACHE_RESERVE_MB=777\n' > "$T/explicit.env"
run HIVE_MODEL=fn HIVE_VRAM_GB=24 HIVE_CONFIG="$T/explicit.env"
check "explicit config chunk" "$(val HIVE_MAX_CHUNK)" 8192
check "explicit config reserve" "$(val HIVE_CACHE_RESERVE_MB)" 777
check "explicit config, profile fills the rest" "$(val HIVE_QWEN_PREFILL_TILES)" 1
printf 'HIVE_MAX_CTX=4096\n' > "$T/local.env"
run HIVE_MODEL=27b HIVE_VRAM_GB=24 HIVE_LOCAL_ENV="$T/local.env"
check "local env wins" "$(val HIVE_MAX_CTX)" 4096
mkdir -p "$T/other/ds"; cp "$T/models/DeepSeek-V4.1-Flash/config.json" "$T/other/ds/"
run HIVE_MODEL=ds HIVE_VRAM_GB=48 HIVE_CKPT="$T/other/ds"
check "HIVE_CKPT kept" "$(val HIVE_CKPT)" "$T/other/ds"
run HIVE_MODELS_DIR="$T/none:$T/models" HIVE_MODEL=glm HIVE_VRAM_GB=48
check "models dir list" "$(val HIVE_CKPT)" "$T/models/GLM-5.3-Flash-NVFP4"
# 4. without HIVE_MODEL nothing changes (the family config from the checkpoint, no profile keys)
run HIVE_CKPT="$T/models/Qwen3.8-Flash-Next-NVFP4"
check "no profile rc" "$RC" 0
check "no profile family" "$(val HIVE_MODEL_FAMILY)" qwen
check "no profile chunk" "$(val HIVE_MAX_CHUNK)" 16384
check "no profile key" "$(val HIVE_PROFILE)" ""
n=$((n + 1)); grep -q '^\[profile\]' <<< "$OUT" && { fails=$((fails + 1)); echo "FAIL no profile: pre-flight printed"; }
# 5. the big table: RAM when the installed RAM holds the RAM-mode need, disk otherwise; an explicit switch is kept
MEMT=512 run HIVE_MODEL=ds HIVE_VRAM_GB=96
check "ds 512 GB engram" "$(val HIVE_ENGRAM_SSD)" 0
MEMT=384 MEMA=380 run HIVE_MODEL=ds HIVE_VRAM_GB=96
check "ds 384 GB engram" "$(val HIVE_ENGRAM_SSD)" 1
check "ds 384 GB need" "$(val HIVE_PROFILE_NEED_GB)" 300
check "ds 384 GB rc" "$RC" 0
MEMT=512 run HIVE_MODEL=ds HIVE_VRAM_GB=0
check "ds cpu 512 GB engram" "$(val HIVE_DS_CPU_ENGRAM)" mmap
check "ds cpu: GPU switch untouched" "$(val HIVE_ENGRAM_SSD)" ""
MEMT=768 run HIVE_MODEL=ds HIVE_VRAM_GB=0
check "ds cpu 768 GB engram" "$(val HIVE_DS_CPU_ENGRAM)" ram
MEMT=200 run HIVE_MODEL=fn HIVE_VRAM_GB=24
check "fn 200 GB ple" "$(val HIVE_QWEN_PLE)" ssd
MEMT=512 run HIVE_MODEL=fn HIVE_VRAM_GB=24
check "fn 512 GB ple" "$(val HIVE_QWEN_PLE)" ram
MEMT=200 run HIVE_MODEL=fn HIVE_VRAM_GB=24 HIVE_QWEN_PLE=ram
has "explicit ple kept" "big table: ram"
check "explicit ple too big rc" "$RC" 1
has "explicit ple too big hint" "set HIVE_QWEN_PLE=ssd"
# 6. pre-flight: warn on MemAvailable and an incomplete checkpoint, stop only when the installed RAM cannot hold the model
MEMT=512 MEMA=100 run HIVE_MODEL=ds HIVE_VRAM_GB=96
check "low available rc" "$RC" 0
has "low available warning" "warning: 100 GB available now"
CKGB=200 run HIVE_MODEL=ds HIVE_VRAM_GB=96
check "short ckpt rc" "$RC" 0
has "short ckpt warning" "incomplete download"
MEMT=256 run HIVE_MODEL=ds HIVE_VRAM_GB=96
check "impossible rc" "$RC" 1
has "impossible message" "cannot start here"
MEMT=32 run HIVE_MODEL=27b HIVE_VRAM_GB=16
check "27b 32 GB rc" "$RC" 1
# 7. bad input
run HIVE_MODEL=llama HIVE_VRAM_GB=24
check "bad model rc" "$RC" 1
run HIVE_MODEL=ds HIVE_VRAM_GB=40
check "bad size rc" "$RC" 1
run HIVE_MODEL=fn
check "default size" "$(val HIVE_PROFILE)" fn-96

echo "test_profiles_cpu: $((n - fails))/$n passed"
[ "$fails" = 0 ]
