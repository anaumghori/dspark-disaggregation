#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../deploy/common/launch_utils.sh"

cd "$REPO_ROOT"
mkdir -p run

SWEEP="${SWEEP:-1 4 16 32 64 128}"
DURATION="${DURATION:-120}"
ENDPOINT="${ENDPOINT:-http://localhost:8000}"
DATASET_FILE="${DATASET_FILE:-$REPO_ROOT/benchmarks/datasets/agentic_code22.jsonl}"
DECODE_METRICS_PORT="${DYN_SYSTEM_PORT2:-8082}"
PREFILL_METRICS_PORT="${DYN_SYSTEM_PORT1:-8081}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
ABLATED_RUNS=()

if [[ ! -f "$DATASET_FILE" ]]; then
    log "FATAL: prompt dataset not found at $DATASET_FILE"
    log "Build it first (no GPU needed): make dataset"
    exit 1
fi
DATASET_SAMPLES=$(wc -l < "$DATASET_FILE" | tr -d ' ')

POLLER_PID=""

cleanup() {
    if [[ -n "$POLLER_PID" ]]; then
        kill "$POLLER_PID" 2>/dev/null || true
    fi
    if [[ ${#ABLATED_RUNS[@]} -gt 0 ]]; then
        ./deploy/launch/stop.sh >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

wait_url() {
    local url="$1" deadline=$(( $(date +%s) + READY_TIMEOUT ))
    while (( $(date +%s) < deadline )); do
        if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    return 1
}

decode_spec_active() {
    curl -fsS --max-time 5 "http://localhost:$DECODE_METRICS_PORT/metrics" 2>/dev/null \
        | grep -q '^sglang:spec_accept_length'
}

poll_spec_metrics() {
    local out="$1"
    : > "$out"
    while :; do
        local ts
        ts=$(date +%s)
        curl -fsS --max-time 5 "http://localhost:$DECODE_METRICS_PORT/metrics" 2>/dev/null \
            | awk -v ts="$ts" '/^sglang:spec_(accept_length|accept_rate|block_accept_length)/ \
                { print ts, $1, $NF }' >> "$out"
        sleep "$POLL_INTERVAL"
    done
}

run_phase() {
    local phase="$1"
    local run_id="ablation-dspark-$phase"
    local metrics_tsv="benchmarks/$run_id/spec-metrics.tsv"

    log "phase $phase: launching deployment"
    mkdir -p "benchmarks/$run_id"
    DSPARK_MODE="$phase" ./deploy/launch/serve.sh > "run/serve-$phase.log" 2>&1 &
    ABLATED_RUNS+=("$phase")

    log "phase $phase: waiting for frontend readiness (timeout ${READY_TIMEOUT}s)"
    wait_url "$ENDPOINT/v1/models" || { log "FATAL: frontend not ready; see run/serve-$phase.log"; exit 1; }
    wait_url "http://localhost:$DECODE_METRICS_PORT/metrics" || { log "FATAL: decode worker not ready"; exit 1; }
    wait_url "http://localhost:$PREFILL_METRICS_PORT/metrics" || { log "FATAL: prefill worker not ready"; exit 1; }

    local active=off
    # DSpark gauge appears only after workers finish loading (~60s) and initialize DSpark.
    # Poll with timeout instead of instant check to avoid false FATAL on slow startup.
    local deadline=$(( $(date +%s) + 300 ))
    while (( $(date +%s) < deadline )); do
        if decode_spec_active; then
            active=on
            break
        fi
        sleep 5
    done
    if [[ "$phase" == "on" && "$active" != "on" ]]; then
        log "FATAL: DSpark metrics absent with DSPARK_MODE=on; speculative decoding is not active"
        exit 1
    fi
    if [[ "$phase" == "off" && "$active" == "on" ]]; then
        log "FATAL: DSpark metrics present with DSPARK_MODE=off; the ablation baseline is contaminated"
        exit 1
    fi
    log "phase $phase: deployment ready (spec metrics active: $active)"

    poll_spec_metrics "$metrics_tsv" &
    POLLER_PID=$!

    local -a bench_args=(--mode real --sweep "$SWEEP" --duration "$DURATION"
        --warmup "${WARMUP:-16}" --endpoint "$ENDPOINT" --run-id "$run_id"
        --dataset-file "$DATASET_FILE" --no-fail-fast)
    [[ -n "${OSL:-}" ]] && bench_args+=(--osl "$OSL")

    log "phase $phase: starting benchmark sweep"
    if ! ./scripts/benchmark.sh "${bench_args[@]}"; then
        log "WARN: phase $phase benchmark reported failures; continuing so both phases are comparable"
    fi

    kill "$POLLER_PID" 2>/dev/null || true
    POLLER_PID=""
    log "phase $phase: stopping deployment"
    ./deploy/launch/stop.sh >/dev/null 2>&1 || true
    ABLATED_RUNS=()
    sleep 10
}

echo ""
echo "  DSpark ablation study"
echo "  sweep:     $SWEEP"
echo "  duration:  ${DURATION}s per point"
echo "  workload:  real agentic-coding prompts"
echo "  dataset:   $DATASET_FILE ($DATASET_SAMPLES samples)"
echo ""

run_phase on
run_phase off

log "rebuilding experiment registry and report"
uv run python3 scripts/analyze.py registry || log "WARN: registry rebuild failed"
uv run python3 scripts/analyze.py report || log "WARN: report generation failed"

log "pruning redundant raw artifacts (analysis inputs are retained)"
uv run python3 - <<'PYPRUNE' || log "WARN: artifact pruning failed"
import json
from pathlib import Path

root = Path("benchmarks")
removed = 0
for run_id in ("ablation-dspark-on", "ablation-dspark-off"):
    run = root / run_id
    if not run.exists():
        continue
    passed = set()
    summary = run / "sweep-summary.json"
    if summary.exists():
        try:
            for entry in json.loads(summary.read_text()).get("runs", []):
                if entry.get("status") == "PASS":
                    passed.add(f"c{entry['concurrency']}")
        except json.JSONDecodeError:
            pass
    for conc in sorted(run.glob("c*")):
        if not conc.is_dir() or conc.name not in passed:
            continue
        for name in ("profile_export_aiperf.csv",
                     "profile_export_raw.jsonl", "aiperf-stdout.log"):
            f = conc / name
            if f.exists():
                f.unlink()
                removed += 1
print(f"pruned {removed} redundant artifacts")
PYPRUNE

echo ""
echo "  Ablation complete. Report: benchmarks/report/REPORT.md"
