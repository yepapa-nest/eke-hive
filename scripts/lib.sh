# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
# Shared by the scripts in this directory: repository root and configuration loading.
#   precedence: environment > file named by HIVE_LOCAL_ENV > HIVE_CONFIG (KEY=value lines, values taken literally).
#   HIVE_CONFIG defaults to config/glm.env for a GLM-5.3-Flash checkpoint (HIVE_CKPT, or HIVE_MODEL_FAMILY=glm) and to
#   config/hive.env otherwise; a relative HIVE_CONFIG that does not exist from the current directory is taken from the repository root.
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

hive_load_config() {
  [ -n "${HIVE_LOCAL_ENV:-}" ] && hive_load_env_file "$HIVE_LOCAL_ENV"
  local cfg=${HIVE_CONFIG:-}
  if [ -z "$cfg" ]; then
    cfg=$HIVE_ROOT/config/hive.env
    if [ "${HIVE_MODEL_FAMILY:-}" = glm ] || { [ -z "${HIVE_MODEL_FAMILY:-}" ] && [ -n "${HIVE_CKPT:-}" ] &&
        grep -q '"model_type"[[:space:]]*:[[:space:]]*"glm5_next"' "$HIVE_CKPT/config.json" 2>/dev/null; }; then
      cfg=$HIVE_ROOT/config/glm.env
    fi
  fi
  case "$cfg" in /*) ;; *) [ -f "$cfg" ] || cfg=$HIVE_ROOT/$cfg ;; esac
  [ -f "$cfg" ] || { echo "configuration file not found: $cfg (HIVE_CONFIG)" >&2; exit 1; }
  hive_load_env_file "$cfg"
  : "${HIVE_STATE_DIR:=$HIVE_ROOT/run}"
  : "${HIVE_CKPT_MOUNT:=/model}"
  : "${HIVE_BUILD:=build}"
  export HIVE_STATE_DIR HIVE_CKPT_MOUNT HIVE_BUILD
  mkdir -p "$HIVE_STATE_DIR"  # before any bind mount: docker would otherwise create it owned by root
}

hive_need_ckpt() {
  [ -n "${HIVE_CKPT:-}" ] && [ -f "$HIVE_CKPT/config.json" ] && return 0
  echo "HIVE_CKPT must point at a supported checkpoint directory (DeepSeek-V4.1-Flash or GLM-5.3-Flash NVFP4; got '${HIVE_CKPT:-}')" >&2
  exit 1
}

# Model family from the checkpoint's config.json model_type (HIVE_MODEL_FAMILY overrides): deepseek (default) or glm.
#   Sets HIVE_MODEL_FAMILY and HIVE_DAEMON_BIN (hived / hived_glm — the same daemon source built per family).
hive_detect_family() {
  if [ -z "${HIVE_MODEL_FAMILY:-}" ]; then
    if grep -q '"model_type"[[:space:]]*:[[:space:]]*"glm5_next"' "$HIVE_CKPT/config.json" 2>/dev/null; then HIVE_MODEL_FAMILY=glm
    else HIVE_MODEL_FAMILY=deepseek; fi
  fi
  case "$HIVE_MODEL_FAMILY" in
    glm) HIVE_DAEMON_BIN=hived_glm ;;
    deepseek) HIVE_DAEMON_BIN=hived ;;
    *) echo "HIVE_MODEL_FAMILY must be deepseek or glm (got '$HIVE_MODEL_FAMILY')" >&2; exit 1 ;;
  esac
  export HIVE_MODEL_FAMILY HIVE_DAEMON_BIN
}

# docker -e arguments for every HIVE_* variable (values passed through unchanged)
hive_env_args() {
  local v
  HIVE_ENV_ARGS=()
  for v in $(compgen -e | grep '^HIVE_'); do HIVE_ENV_ARGS+=(-e "$v=${!v}"); done
}
