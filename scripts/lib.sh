# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Shared by the scripts in this directory: repository root and configuration loading.
#   precedence: environment > file named by HIVE_LOCAL_ENV > HIVE_CONFIG (KEY=value lines, values taken literally).
#   HIVE_CONFIG defaults to config/glm.env for a GLM-5.3-Flash checkpoint (HIVE_CKPT, or HIVE_MODEL_FAMILY=glm), to config/qwen.env
#   for a Qwen3.8-Flash-Next checkpoint (or HIVE_MODEL_FAMILY=qwen) and to config/hive.env otherwise; a relative HIVE_CONFIG that does not exist from the current directory is taken from the repository root.
HIVE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

hive_load_env_file() {  # set KEY=value from a file for every KEY not already set
  local file=$1 line key val
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}
    case "$line" in ''|'#'*) continue ;; esac
    key=${line%%=*}; val=${line#*=}
    [[ $key =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [ -n "${!key+x}" ] && continue
    printf -v "$key" '%s' "$val"; export "$key"
  done < "$file"
}

# ---- model + VRAM profile (HIVE_MODEL, HIVE_VRAM_GB — docs/configuration.md "Model and VRAM profile") ----------------------------
#   HIVE_MODEL=ds|glm|fn|27b and HIVE_VRAM_GB=96|72|48|32|24|16|12|8|0 (0 = CPU only; default 96) select config/profiles/<model>-<size>.env (the
#   knobs measured good for that card) and config/profiles/<model>.env (checkpoint names, measured RAM / disk needs, the switch that moves the big
#   table to disk). Everything they set is a default: the environment, HIVE_LOCAL_ENV and an explicit HIVE_CONFIG win; the family's shipped
#   config fills what the profile leaves open. Without HIVE_MODEL nothing here runs (the launcher behaves as before).
#   Derived: HIVE_MODEL_FAMILY · HIVE_CKPT (first <dir>/<name>/config.json over HIVE_MODELS_DIR, a colon list, default /models) · HIVE_GPU=none
#   for 0 · HIVE_VRAM_LIMIT_MB = GB × 1024 below 96 (96 = the whole card) · the big table in RAM or on disk by installed RAM (MemTotal − 8 GiB,
#   the start script's own clamp, against the profile's RAM need) · HIVE_RAM_WAIT_GB = that need.
hive_mem_gb() {  # $1 = MemTotal | MemAvailable, in GiB (HIVE_TEST_MEMTOTAL_GB / HIVE_TEST_MEMAVAIL_GB replace the reading — CPU tests)
  case "$1" in
    MemTotal) [ -n "${HIVE_TEST_MEMTOTAL_GB:-}" ] && { echo "$HIVE_TEST_MEMTOTAL_GB"; return; } ;;
    MemAvailable) [ -n "${HIVE_TEST_MEMAVAIL_GB:-}" ] && { echo "$HIVE_TEST_MEMAVAIL_GB"; return; } ;;
  esac
  echo $(( $(awk -v k="$1:" '$1==k{print $2}' /proc/meminfo) / 1024 / 1024 ))
}

hive_apply_profile() {
  [ -n "${HIVE_MODEL:-}" ] || return 0
  case "$HIVE_MODEL" in ds|glm|fn|27b) ;; *) echo "HIVE_MODEL must be ds, glm, fn or 27b (got '$HIVE_MODEL')" >&2; exit 1 ;; esac
  : "${HIVE_VRAM_GB:=96}"
  case "$HIVE_VRAM_GB" in 96|72|48|32|24|16|12|8|0) ;; *) echo "HIVE_VRAM_GB must be one of 96 72 48 32 24 16 12 8 0 (got '$HIVE_VRAM_GB')" >&2; exit 1 ;; esac
  export HIVE_MODEL HIVE_VRAM_GB
  HIVE_PROFILE=$HIVE_MODEL-$HIVE_VRAM_GB; export HIVE_PROFILE
  local dir=$HIVE_ROOT/config/profiles
  [ -f "$dir/$HIVE_PROFILE.env" ] || { echo "profile not found: $dir/$HIVE_PROFILE.env" >&2; exit 1; }
  hive_load_env_file "$dir/$HIVE_PROFILE.env"   # the size's knobs (may also override the model's needs)
  hive_load_env_file "$dir/$HIVE_MODEL.env"      # the model's data
  : "${HIVE_MODEL_FAMILY:=$HIVE_PROFILE_FAMILY}"; export HIVE_MODEL_FAMILY
  if [ -z "${HIVE_CKPT:-}" ]; then
    local d n IFS_save=$IFS
    IFS=:
    for d in ${HIVE_MODELS_DIR:-/models}; do
      IFS=$IFS_save
      for n in $HIVE_PROFILE_CKPT_NAMES; do [ -f "$d/$n/config.json" ] && { HIVE_CKPT=$d/$n; break 2; }; done
      IFS=:
    done
    IFS=$IFS_save
    if [ -n "${HIVE_CKPT:-}" ]; then export HIVE_CKPT
    else echo "profile $HIVE_PROFILE: no checkpoint ($HIVE_PROFILE_CKPT_NAMES) under HIVE_MODELS_DIR=${HIVE_MODELS_DIR:-/models} — set HIVE_CKPT or HIVE_MODELS_DIR" >&2; fi
  fi
  if [ "$HIVE_VRAM_GB" = 0 ]; then : "${HIVE_GPU:=none}"; export HIVE_GPU
  elif [ "$HIVE_VRAM_GB" != 96 ]; then : "${HIVE_VRAM_LIMIT_MB:=$((HIVE_VRAM_GB * 1024))}"; export HIVE_VRAM_LIMIT_MB
  fi
  # big table (DS Engram 203 GB · FN PLE 47.7 GB) in RAM or on disk: RAM when the installed RAM holds the RAM-mode need
  local avail_gb need_gb=${HIVE_PROFILE_RAM_GB:-0}
  avail_gb=$(( $(hive_mem_gb MemTotal) - 8 ))
  HIVE_PROFILE_TABLE=ram
  if [ -n "${HIVE_PROFILE_SSD_KEY:-}" ]; then
    local key=$HIVE_PROFILE_SSD_KEY
    if [ -z "${!key+x}" ]; then
      if [ "$need_gb" -gt "$avail_gb" ]; then printf -v "$key" '%s' "$HIVE_PROFILE_SSD_ON"; else printf -v "$key" '%s' "$HIVE_PROFILE_SSD_OFF"; fi
      export "$key"
    fi
    [ "${!key}" = "$HIVE_PROFILE_SSD_ON" ] && { HIVE_PROFILE_TABLE=disk; need_gb=${HIVE_PROFILE_RAM_GB_SSD:-$need_gb}; }
  fi
  export HIVE_PROFILE_TABLE
  HIVE_PROFILE_NEED_GB=$need_gb; export HIVE_PROFILE_NEED_GB
  : "${HIVE_RAM_WAIT_GB:=$need_gb}"; export HIVE_RAM_WAIT_GB
}

# Pre-flight of a profile (hive-start.sh): installed / available RAM against the profile's measured need, the checkpoint's size on disk, the
#   state directory, and a spinning disk under a disk-resident table. Warns; stops only when the start cannot work (the installed RAM below the
#   need of the table placement in use — the daemon would be OOM-killed while pinning its weights).
hive_profile_preflight() {
  [ -n "${HIVE_MODEL:-}" ] || return 0
  local total avail need=${HIVE_PROFILE_NEED_GB:-0} ok=1
  total=$(hive_mem_gb MemTotal); avail=$(hive_mem_gb MemAvailable)
  echo "[profile] $HIVE_PROFILE: family $HIVE_MODEL_FAMILY · checkpoint ${HIVE_CKPT:-?} · ${HIVE_VRAM_LIMIT_MB:+VRAM limit $HIVE_VRAM_LIMIT_MB MiB · }${HIVE_GPU:+GPU $HIVE_GPU · }big table: $HIVE_PROFILE_TABLE · RAM need $need GB (installed $total, available $avail)${HIVE_HOST_SESSION_MB:+ · session images ≤ $HIVE_HOST_SESSION_MB MiB}"
  if [ "$need" -gt $((total - 8)) ]; then
    if [ -n "${HIVE_PROFILE_SSD_KEY:-}" ] && [ "$HIVE_PROFILE_TABLE" = ram ] && [ "${HIVE_PROFILE_RAM_GB_SSD:-0}" -le $((total - 8)) ]; then
      echo "[profile] the RAM-mode need ($need GB) is above the installed RAM − 8 ($((total - 8)) GB): set $HIVE_PROFILE_SSD_KEY=$HIVE_PROFILE_SSD_ON (needs ${HIVE_PROFILE_RAM_GB_SSD} GB) or unset it to let the profile choose" >&2
    else
      echo "[profile] this model needs $need GB of host RAM even with its table on disk; installed RAM − 8 is $((total - 8)) GB — it cannot start here" >&2
    fi
    ok=0
  elif [ "$need" -gt "$avail" ]; then
    echo "[profile] warning: $avail GB available now, $need GB needed — other processes hold memory (the start waits up to 10 min for it)" >&2
  fi
  if [ -n "${HIVE_CKPT:-}" ] && [ -d "$HIVE_CKPT" ] && [ -n "${HIVE_PROFILE_DISK_GB:-}" ]; then
    local ck_gb
    ck_gb=$(( $(du -s -k "$HIVE_CKPT" 2>/dev/null | awk '{print $1}') / 1024 / 1024 ))
    [ -n "${HIVE_TEST_CKPT_GB:-}" ] && ck_gb=$HIVE_TEST_CKPT_GB
    [ $((ck_gb * 100)) -lt $((HIVE_PROFILE_DISK_GB * 95)) ] &&
      echo "[profile] warning: the checkpoint holds $ck_gb GB on disk, the full one $HIVE_PROFILE_DISK_GB GB — incomplete download?" >&2
    if [ "$HIVE_PROFILE_TABLE" = disk ]; then
      local dev rot
      dev=$(df --output=source "$HIVE_CKPT" 2>/dev/null | tail -1)
      rot=$(lsblk -ndo ROTA "$dev" 2>/dev/null | head -1 | tr -d ' ')
      [ "$rot" = 1 ] && echo "[profile] warning: $HIVE_PROFILE_SSD_KEY reads table rows from $dev, a rotating disk — expect a large slowdown (an NVMe is assumed)" >&2
    fi
  fi
  local st_free
  st_free=$(df -k --output=avail "${HIVE_STATE_DIR:-.}" 2>/dev/null | tail -1)
  [ -n "$st_free" ] && [ "$st_free" -lt $((2 * 1024 * 1024)) ] && echo "[profile] warning: less than 2 GB free under HIVE_STATE_DIR=$HIVE_STATE_DIR (logs, snapshots)" >&2
  [ "$ok" = 1 ] || exit 1
}

# CPU-only daemon of a family (HIVE_GPU=none)
hive_cpu_daemon() {
  case "$1" in
    hived_qwen) echo hived_qwen_cpu ;; hived_glm) echo hived_glm_cpu ;; hived) echo hived_ds_cpu ;; hived_q27) echo hived_q27_cpu ;; *) echo "" ;;
  esac
}

hive_load_config() {
  [ -n "${HIVE_LOCAL_ENV:-}" ] && hive_load_env_file "$HIVE_LOCAL_ENV"
  # an explicitly named HIVE_CONFIG wins over the profile; the default (family) config only fills what the profile leaves open
  if [ -n "${HIVE_MODEL:-}" ] && [ -n "${HIVE_CONFIG:-}" ]; then
    local ecfg=$HIVE_CONFIG
    case "$ecfg" in /*) ;; *) [ -f "$ecfg" ] || ecfg=$HIVE_ROOT/$ecfg ;; esac
    hive_load_env_file "$ecfg"
  fi
  hive_apply_profile
  local cfg=${HIVE_CONFIG:-}
  if [ -z "$cfg" ]; then
    cfg=$HIVE_ROOT/config/hive.env
    if [ "${HIVE_MODEL_FAMILY:-}" = glm ] || { [ -z "${HIVE_MODEL_FAMILY:-}" ] && [ -n "${HIVE_CKPT:-}" ] &&
        grep -q '"model_type"[[:space:]]*:[[:space:]]*"glm5_next"' "$HIVE_CKPT/config.json" 2>/dev/null; }; then
      cfg=$HIVE_ROOT/config/glm.env
    elif [ "${HIVE_MODEL_FAMILY:-}" = qwen ] || { [ -z "${HIVE_MODEL_FAMILY:-}" ] && [ -n "${HIVE_CKPT:-}" ] &&
        grep -q '"model_type"[[:space:]]*:[[:space:]]*"qwen4_exp"' "$HIVE_CKPT/config.json" 2>/dev/null; }; then
      cfg=$HIVE_ROOT/config/qwen.env
    elif [ "${HIVE_MODEL_FAMILY:-}" = q27 ] || { [ -z "${HIVE_MODEL_FAMILY:-}" ] && [ -n "${HIVE_CKPT:-}" ] &&
        grep -q '"model_type"[[:space:]]*:[[:space:]]*"qwen3_5"' "$HIVE_CKPT/config.json" 2>/dev/null; }; then
      cfg=$HIVE_ROOT/config/q27.env
    fi
  fi
  case "$cfg" in /*) ;; *) [ -f "$cfg" ] || cfg=$HIVE_ROOT/$cfg ;; esac
  [ -f "$cfg" ] || { echo "configuration file not found: $cfg (HIVE_CONFIG)" >&2; exit 1; }
  hive_load_env_file "$cfg"
  # session images kept in host RAM (hived --host-session-mb): at most 25 % of what the installed RAM leaves after the model's need
  #   (HIVE_RAM_WAIT_GB) and 8 GiB, floor 4096 — the configs' 98304 fit a 1 TB host, a smaller one gets a proportional budget
  if [ -n "${HIVE_HOST_SESSION_MB:-}" ]; then
    local hs_cap
    hs_cap=$(( ( $(hive_mem_gb MemTotal) - ${HIVE_RAM_WAIT_GB:-0} - 8 ) * 1024 / 4 ))
    [ "$hs_cap" -lt 4096 ] && hs_cap=4096
    [ "$HIVE_HOST_SESSION_MB" -gt "$hs_cap" ] && HIVE_HOST_SESSION_MB=$hs_cap
    export HIVE_HOST_SESSION_MB
  fi
  : "${HIVE_STATE_DIR:=$HIVE_ROOT/run}"
  : "${HIVE_CKPT_MOUNT:=/model}"
  : "${HIVE_BUILD:=build}"
  export HIVE_STATE_DIR HIVE_CKPT_MOUNT HIVE_BUILD
  mkdir -p "$HIVE_STATE_DIR"  # before any bind mount: docker would otherwise create it owned by root
}

hive_need_ckpt() {
  [ -n "${HIVE_CKPT:-}" ] && [ -f "$HIVE_CKPT/config.json" ] && return 0
  echo "HIVE_CKPT must point at a supported checkpoint directory (DeepSeek-V4.1-Flash, GLM-5.3-Flash NVFP4 or Qwen3.8-Flash-Next NVFP4; got '${HIVE_CKPT:-}')" >&2
  exit 1
}

# Model family from the checkpoint's config.json model_type (HIVE_MODEL_FAMILY overrides): deepseek (default), glm or qwen.
#   Sets HIVE_MODEL_FAMILY and HIVE_DAEMON_BIN (hived / hived_glm / hived_qwen — the same daemon source built per family).
hive_detect_family() {
  if [ -z "${HIVE_MODEL_FAMILY:-}" ]; then
    if grep -q '"model_type"[[:space:]]*:[[:space:]]*"glm5_next"' "$HIVE_CKPT/config.json" 2>/dev/null; then HIVE_MODEL_FAMILY=glm
    elif grep -q '"model_type"[[:space:]]*:[[:space:]]*"qwen4_exp"' "$HIVE_CKPT/config.json" 2>/dev/null; then HIVE_MODEL_FAMILY=qwen
    elif grep -q '"model_type"[[:space:]]*:[[:space:]]*"qwen3_5"' "$HIVE_CKPT/config.json" 2>/dev/null; then HIVE_MODEL_FAMILY=q27
    else HIVE_MODEL_FAMILY=deepseek; fi
  fi
  case "$HIVE_MODEL_FAMILY" in
    glm) HIVE_DAEMON_BIN=hived_glm ;;
    qwen) HIVE_DAEMON_BIN=hived_qwen ;;
    q27) HIVE_DAEMON_BIN=hived_q27 ;;
    deepseek) HIVE_DAEMON_BIN=hived ;;
    *) echo "HIVE_MODEL_FAMILY must be deepseek, glm, qwen or q27 (got '$HIVE_MODEL_FAMILY')" >&2; exit 1 ;;
  esac
  export HIVE_MODEL_FAMILY HIVE_DAEMON_BIN
}

# docker -e arguments for every HIVE_* variable (values passed through unchanged)
hive_env_args() {
  local v
  HIVE_ENV_ARGS=()
  for v in $(compgen -e | grep '^HIVE_'); do HIVE_ENV_ARGS+=(-e "$v=${!v}"); done
}
