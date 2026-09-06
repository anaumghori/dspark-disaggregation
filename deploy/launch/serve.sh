#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../common/launch_utils.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../common/gpu_utils.sh"

cd "$REPO_ROOT"
log "working directory: $REPO_ROOT"
log "kernel caches: flashinfer=$FLASHINFER_WORKSPACE_BASE triton=$TRITON_CACHE_DIR"
log "discovery backend: $DYN_DISCOVERY_BACKEND (registry at $DYN_FILE_KV)"

export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-compact}"
log "SGLANG_RAGGED_VERIFY_MODE=$SGLANG_RAGGED_VERIFY_MODE"

SPS_TABLE="$(args_file_value "$REPO_ROOT/configs/decode.args" --speculative-dspark-sps-table-path)"
mapfile -t PREFILL_ARGS < <(load_args_file "$REPO_ROOT/configs/prefill.args")
mapfile -t DECODE_ARGS < <(load_args_file "$REPO_ROOT/configs/decode.args")
DSPARK_MODE="${DSPARK_MODE:-on}"
if [[ "$DSPARK_MODE" == "off" ]]; then
    log "DSPARK_MODE=off: running the decode pool without DSpark speculative decoding (ablation baseline)"
    for flag in --speculative-algorithm --speculative-dspark-block-size \
                --speculative-dspark-sps-table-path --speculative-dspark-confidence-sts-path \
                --speculative-moe-runner-backend; do
        filter_flag_and_value DECODE_ARGS "$flag"
    done
    unset SGLANG_RAGGED_VERIFY_MODE
elif [[ "$DSPARK_MODE" != "on" ]]; then
    log "FATAL: DSPARK_MODE must be 'on' or 'off' (got '$DSPARK_MODE')"
    exit 1
else
    require_sps_table "$REPO_ROOT/$SPS_TABLE"
fi
STS_TABLE="$(args_file_value "$REPO_ROOT/configs/decode.args" --speculative-dspark-confidence-sts-path)"
if [[ -n "$STS_TABLE" && ! -f "$REPO_ROOT/$STS_TABLE" ]]; then
    log "STS calibration table not found at $REPO_ROOT/$STS_TABLE; dropping --speculative-dspark-confidence-sts-path (engine runs without STS tuning)"
    filter_flag_and_value DECODE_ARGS "--speculative-dspark-confidence-sts-path"
elif [[ -n "$STS_TABLE" ]]; then
    log "STS calibration table present: $REPO_ROOT/$STS_TABLE"
fi

PREFILL_GPUS="${PREFILL_GPUS:-0,1}"
DECODE_GPUS="${DECODE_GPUS:-2,3}"

IFS=',' read -r -a prefill_ids <<< "$PREFILL_GPUS"
IFS=',' read -r -a decode_ids <<< "$DECODE_GPUS"
if [[ ${#prefill_ids[@]} -ne 2 || ${#decode_ids[@]} -ne 2 ]]; then
    log "FATAL: each pool must list exactly 2 GPUs (TP=2); got prefill=[$PREFILL_GPUS] decode=[$DECODE_GPUS]"
    exit 1
fi
for pid_ in "${prefill_ids[@]}"; do
    for did in "${decode_ids[@]}"; do
        if [[ "$pid_" == "$did" ]]; then
            log "FATAL: GPU $pid_ is assigned to both pools (prefill=[$PREFILL_GPUS] decode=[$DECODE_GPUS]); pools must not overlap"
            exit 1
        fi
    done
done
log "prefill pool: GPUs $PREFILL_GPUS  decode pool: GPUs $DECODE_GPUS"
print_gpu_inventory

GPU_MEM_ARGS="$(build_sglang_gpu_mem_args)"
export DYN_HTTP_PORT="${DYN_HTTP_PORT:-8000}"
export DYN_SYSTEM_PORT1="${DYN_SYSTEM_PORT1:-8081}"
export DYN_SYSTEM_PORT2="${DYN_SYSTEM_PORT2:-8082}"

resolve_venv_python

# KV events from the prefill worker feed the frontend's cache-aware routing
# (dynamo.frontend --router-mode kv); decode-side events are not required for
# PD routing. Each ZMQ publisher binds its own port, so the prefill worker is
# the only worker that publishes here.
KV_EVENTS_PORT="${DYN_KV_EVENTS_PORT:-5557}"
KV_EVENTS_CONFIG="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVENTS_PORT}\"}"

# The disaggregation bootstrap port must match across the prefill/decode pair.
# When DYN_DISAGG_BOOTSTRAP_PORT is set it is passed to both workers verbatim;
# when unset, dynamo.sglang auto-reserves a free port on each worker and the
# prefill worker advertises its resolved bootstrap address through the runtime
# registry for the decode side to discover.
DISAGG_BOOTSTRAP_ARGS=()
if [[ -n "${DYN_DISAGG_BOOTSTRAP_PORT:-}" ]]; then
    DISAGG_BOOTSTRAP_ARGS=(--disaggregation-bootstrap-port "$DYN_DISAGG_BOOTSTRAP_PORT")
    log "bootstrap port: $DYN_DISAGG_BOOTSTRAP_PORT (pinned via DYN_DISAGG_BOOTSTRAP_PORT)"
else
    log "bootstrap port: auto-reserved by dynamo.sglang (set DYN_DISAGG_BOOTSTRAP_PORT to pin)"
fi

PREFILL_EXTRA=()
DECODE_EXTRA=()
serve_usage() {
    echo "usage: serve.sh [--prefill-args \"<sglang args>\"] [--decode-args \"<sglang args>\"]" >&2
    echo "  --prefill-args  extra SGLang ServerArgs appended to the prefill worker only" >&2
    echo "  --decode-args   extra SGLang ServerArgs appended to the decode worker only" >&2
}
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefill-args)
            [[ $# -ge 2 ]] || { log "FATAL: --prefill-args requires a value"; exit 1; }
            read -ra _prefill_extra <<< "$2"
            PREFILL_EXTRA+=("${_prefill_extra[@]}")
            shift 2
            ;;
        --decode-args)
            [[ $# -ge 2 ]] || { log "FATAL: --decode-args requires a value"; exit 1; }
            read -ra _decode_extra <<< "$2"
            DECODE_EXTRA+=("${_decode_extra[@]}")
            shift 2
            ;;
        -h|--help)
            serve_usage
            exit 0
            ;;
        *)
            log "FATAL: unknown serve.sh argument '$1' (worker-specific flags must be scoped with --prefill-args/--decode-args so they are not applied to both pools)"
            exit 1
            ;;
    esac
done

cleanup() {
    log "serve shutting down; killing remaining children"
    local pid_file pid
    shopt -s nullglob
    for pid_file in "$REPO_ROOT"/run/*.pid; do
        pid="$(cat "$pid_file" 2>/dev/null)" || continue
        [[ -n "$pid" ]] || continue
        kill "$pid" 2>/dev/null || true
    done
    shopt -u nullglob
    wait 2>/dev/null || true
}
trap cleanup EXIT

log "launching dynamo.frontend on HTTP port $DYN_HTTP_PORT (router-mode: kv)"
OTEL_SERVICE_NAME=dynamo-frontend \
"$PYTHON_BIN" -m dynamo.frontend --router-mode kv &
write_pid frontend "$!"

log "launching prefill worker (system status port $DYN_SYSTEM_PORT1, kv events on $KV_EVENTS_PORT)"
log "prefill args: ${PREFILL_ARGS[*]} ${GPU_MEM_ARGS} ${PREFILL_EXTRA[*]:-}"
CUDA_VISIBLE_DEVICES="$PREFILL_GPUS" \
OTEL_SERVICE_NAME=dynamo-worker-prefill DYN_SYSTEM_PORT="$DYN_SYSTEM_PORT1" \
"$PYTHON_BIN" -m dynamo.sglang \
  "${PREFILL_ARGS[@]}" \
  --kv-events-config "$KV_EVENTS_CONFIG" \
  ${DISAGG_BOOTSTRAP_ARGS[@]+"${DISAGG_BOOTSTRAP_ARGS[@]}"} \
  $GPU_MEM_ARGS \
  ${PREFILL_EXTRA[@]+"${PREFILL_EXTRA[@]}"} &
write_pid worker-prefill "$!"

log "launching decode worker (DSpark mode: $DSPARK_MODE, system status port $DYN_SYSTEM_PORT2)"
log "decode args: ${DECODE_ARGS[*]} ${GPU_MEM_ARGS} ${DECODE_EXTRA[*]:-}"
CUDA_VISIBLE_DEVICES="$DECODE_GPUS" \
OTEL_SERVICE_NAME=dynamo-worker-decode DYN_SYSTEM_PORT="$DYN_SYSTEM_PORT2" \
"$PYTHON_BIN" -m dynamo.sglang \
  "${DECODE_ARGS[@]}" \
  ${DISAGG_BOOTSTRAP_ARGS[@]+"${DISAGG_BOOTSTRAP_ARGS[@]}"} \
  $GPU_MEM_ARGS \
  ${DECODE_EXTRA[@]+"${DECODE_EXTRA[@]}"} &
write_pid worker-decode "$!"

log "frontend: http://0.0.0.0:$DYN_HTTP_PORT/v1/models"
log "prefill metrics: http://0.0.0.0:$DYN_SYSTEM_PORT1/metrics  decode metrics: http://0.0.0.0:$DYN_SYSTEM_PORT2/metrics"

# DSPARK_SERVE_FOREGROUND: when set to a non-empty value, skip the blocking
# wait_any_exit so an external orchestrator (e.g. Modal) can manage the
# process lifecycle.  The trap still fires on normal script exit to clean up
# any remaining children.
if [[ -z "${DSPARK_SERVE_FOREGROUND:-}" ]]; then
    wait_any_exit
else
    log "DSPARK_SERVE_FOREGROUND=$DSPARK_SERVE_FOREGROUND: skipping wait_any_exit (external orchestrator owns lifecycle)"
    log "serve.sh is ready; PID $$ keeps the trap alive for cleanup"
    # Block indefinitely so the trap remains registered.  The caller kills
    # this script (SIGTERM) when it wants the deployment torn down, which
    # triggers cleanup() above.
    wait $$ 2>/dev/null || true
fi
