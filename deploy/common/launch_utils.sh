#!/usr/bin/env bash
if [[ "${BASH_VERSINFO[0]}" -lt 4 || ( "${BASH_VERSINFO[0]}" -eq 4 && "${BASH_VERSINFO[1]}" -lt 3 ) ]]; then
    echo "launch_utils.sh requires bash 4.3+ (for wait -n), found ${BASH_VERSION}" >&2
    exit 1
fi

DEPLOY_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DEPLOY_COMMON_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

export KERNEL_CACHE_BASE="${KERNEL_CACHE_BASE:-$REPO_ROOT/.cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$KERNEL_CACHE_BASE/kernels/flashinfer-sm120}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$KERNEL_CACHE_BASE/kernels/triton-sm120}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR"

export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
export DYN_FILE_KV="${DYN_FILE_KV:-$KERNEL_CACHE_BASE/dynamo_store_kv}"

# Every project entrypoint runs through the uv-managed environment (uv sync
# into .venv at the repo root). Resolve the venv interpreter once; --no-sync
# keeps launch-time paths from re-resolving or mutating the pinned manifest.
resolve_venv_python() {
    if ! command -v uv >/dev/null 2>&1; then
        log "FATAL: uv not found on PATH; install it first (curl -LsSf https://astral.sh/uv/install.sh | sh)"
        exit 1
    fi
    PYTHON_BIN="$(cd "$REPO_ROOT" && uv run --no-sync python3 -c 'import sys; print(sys.executable)')" \
        || { log "FATAL: could not resolve the uv-managed interpreter; run 'uv sync' first (make setup)"; exit 1; }
    if [[ ! -x "$PYTHON_BIN" ]]; then
        log "FATAL: resolved interpreter is not executable: $PYTHON_BIN"
        exit 1
    fi
    log "python: $PYTHON_BIN"
}

load_args_file() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        log "FATAL: argument file not found: $file"
        exit 1
    fi
    grep -vE '^[[:space:]]*(#|$)' "$file" | tr '[:space:]' '\n' | grep -v '^$' || true
}

args_file_value() {
    awk -v flag="$2" '$1 == flag { print $2; exit }' "$1"
}

require_sps_table() {
    local sps="$1"
    if [[ ! -f "$sps" ]]; then
        log "FATAL: SPS table missing at $sps"
        log "Generate it first: make calibrate A='sps' (scripts/calibrate_sps_sts.sh)"
        exit 1
    fi
}

filter_flag_and_value() {
    local -n _arr="$1"
    local flag="$2"
    local -a out=()
    local skip=0 word
    for word in "${_arr[@]}"; do
        if [[ "$skip" -eq 1 ]]; then
            skip=0
            continue
        fi
        if [[ "$word" == "$flag" ]]; then
            skip=1
            continue
        fi
        out+=("$word")
    done
    _arr=("${out[@]}")
}

write_pid() {
    local name="$1" pid="$2"
    mkdir -p "$REPO_ROOT/run"
    echo "$pid" > "$REPO_ROOT/run/$name.pid"
    log "recorded $name pid $pid (run/$name.pid)"
}

wait_any_exit() {
    local status=0
    wait -n || status=$?
    if [[ $status -ne 0 && $status -ne 143 ]]; then
        log "a child process exited with status $status; shutting down the rest"
    fi
    exit "$status"
}
