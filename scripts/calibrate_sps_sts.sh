#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../deploy/common/launch_utils.sh"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] calibrate: $*"
}

resolve_venv_python
cd "$REPO_ROOT"

CALIB_PORT="${CALIB_PORT:-30000}"
CALIB_BASE_URL="${CALIB_BASE_URL:-http://localhost:$CALIB_PORT}"
STS_PORT="${STS_PORT:-30001}"
STS_COLLECT_DIR="${STS_COLLECT_DIR:-$REPO_ROOT/.cache/sts}"
SPS_OUT="configs/dspark_sps_table.json"
STS_OUT="configs/dspark_sts_table.json"

base_server_args() {
    grep -vE '^[[:space:]]*(#|$)' configs/decode.args \
        | grep -v -- '--disaggregation-mode' \
        | grep -v -- '--disaggregation-bootstrap-port' \
        | grep -v -- '--disaggregation-transfer-backend' \
        | grep -v -- '--speculative-dspark-sps-table-path' \
        | grep -v -- '--speculative-dspark-confidence-sts-path' \
        | tr '[:space:]' '\n' | grep -v '^$'
}

wait_for_server() {
    local url="$1" timeout="${2:-3600}" attempt
    for attempt in $(seq 1 "$timeout"); do
        if curl -fsS --max-time 2 "$url/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

case "${1:-}" in
sps)
    shift || true
    log "starting SPS calibration server on port $CALIB_PORT"
    export SGLANG_RAGGED_VERIFY_MODE=static
    export SGLANG_DSPARK_ENABLE_SPS_RECORD=1
    export SGLANG_SIMULATE_ACC_LEN="${SGLANG_SIMULATE_ACC_LEN:-1.0}"

    "$PYTHON_BIN" -m sglang.launch_server $(base_server_args) --mem-fraction-static 0.88 --cuda-graph-max-bs-decode 32 --disable-flashinfer-autotune --port "$CALIB_PORT" "$@" &
    SERVER_PID=$!
    trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT

    log "waiting for server to be ready..."
    if ! wait_for_server "$CALIB_BASE_URL"; then
        log "FATAL: server did not start within timeout"
        exit 1
    fi
    log "server ready, running SPS profiler -> $SPS_OUT"

    "$PYTHON_BIN" -m sglang.benchmark.dspark_sps_profiler all \
        --base-url "$CALIB_BASE_URL" \
        --max-batch-size 32 \
        --out "$SPS_OUT"

    log "SPS table written to $SPS_OUT"
    kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
    trap - EXIT
    ;;
sts)
    shift || true
    mkdir -p "$STS_COLLECT_DIR"
    export SGLANG_RAGGED_VERIFY_MODE=compact
    export SGLANG_DSPARK_STS_COLLECT_PATH="$STS_COLLECT_DIR/sts"

    log "starting STS collection server on port $STS_PORT"
    log "drive traffic against http://localhost:$STS_PORT, then press Ctrl+C to fit the table"

    "$PYTHON_BIN" -m sglang.launch_server $(base_server_args) --port "$STS_PORT" "$@" &
    SERVER_PID=$!

   sts_fits() {
        kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
        DATA_GLOB="${STS_DATA_GLOB:-$STS_COLLECT_DIR/sts.*.pt}"
        if ! compgen -G "$DATA_GLOB" >/dev/null; then
            log "FATAL: no STS shards match $DATA_GLOB"
            exit 1
        fi
        shard_count=$(ls $DATA_GLOB | wc -l)
        log "fitting STS calibration from $shard_count shard(s) -> $STS_OUT"
        "$PYTHON_BIN" -m sglang.benchmark.dspark_sts_fit \
            --data-glob "$DATA_GLOB" \
            --out "$STS_OUT"
        log "STS calibration written to $STS_OUT"
    }
    trap sts_fits EXIT

    wait $SERVER_PID
    ;;
*)
    echo "usage: $0 {sps|sts} [extra args pass through to the underlying CLI]" >&2
    echo "" >&2
    echo "  sps   Generate the SPS step-cost table (auto server lifecycle)" >&2
    echo "  sts   Collect STS data, then fit calibration on Ctrl+C" >&2
    exit 1
    ;;
esac
