#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../deploy/common/launch_utils.sh"

resolve_venv_python

benchmark_usage() {
    echo "usage: benchmark.sh [options]"
    echo "  --endpoint URL         frontend endpoint (default: http://localhost:8000)"
    echo "  --model NAME           served model name (default: deepseek-v4-flash-dspark)"
    echo "  --mode MODE            fixed|real (default: fixed)"
    echo "  --isl N --osl N        fixed-mode input/output token lengths"
    echo "  --sweep \"1 4 16 ...\"  concurrency sweep"
    echo "  --duration SECONDS     per concurrency point"
    echo "  --warmup N             warmup requests per point"
    echo "  --prefix-reuse PCT     shared-prefix workload percentage"
    echo "  --prefix-groups N      number of prefix groups"
    echo "  --run-id ID            artifact directory name (default: UTC timestamp)"
    echo "  --random-seed N        dataset/traffic seed (default: 42)"
    echo "  --tokenizer NAME       HuggingFace tokenizer override"
    echo "  --dataset-file PATH    real-mode prompt dataset (jsonl; default benchmarks/datasets/agentic_code22.jsonl)"
    echo "  --no-fail-fast         continue the sweep after a failed concurrency point"
}

ENDPOINT="${ENDPOINT:-http://localhost:8000}"
MODEL="${MODEL:-deepseek-v4-flash-dspark}"

PREFILL_WORKER_PORT="${DYN_SYSTEM_PORT1:-8081}"
DECODE_WORKER_PORT="${DYN_SYSTEM_PORT2:-8082}"

MODE="${MODE:-fixed}"

ISL="${ISL:-4096}"
OSL="${OSL:-512}"

SEQUENCE_DISTRIBUTION="${SEQUENCE_DISTRIBUTION:-1024,256:20;4096,512:10;8192,1024:20;16384,1024:20;32768,1024:15;65536,1024:5;98304,1024:5;126976,1024:5}"

SWEEP="${SWEEP:-1 4 16 32 64 128}"

DURATION="${DURATION:-180}"
WARMUP="${WARMUP:-16}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
GRACE_PERIOD="${GRACE_PERIOD:-300}"

PREFIX_REUSE="${PREFIX_REUSE:-0}"
PREFIX_GROUPS="${PREFIX_GROUPS:-8}"

DATASET_FILE="${DATASET_FILE:-$REPO_ROOT/benchmarks/datasets/agentic_code22.jsonl}"

BENCHMARK_ROOT="${BENCHMARK_ROOT:-$REPO_ROOT/benchmarks}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_DIR="$BENCHMARK_ROOT/$RUN_ID"

RANDOM_SEED="${RANDOM_SEED:-42}"

TOKENIZER="${TOKENIZER:-}"

NUM_DATASET_ENTRIES="${NUM_DATASET_ENTRIES:-4096}"

AIPERF_HTTP_CONNECTION_LIMIT="${AIPERF_HTTP_CONNECTION_LIMIT:-512}"

FAIL_FAST="${FAIL_FAST:-1}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --endpoint)      ENDPOINT="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --mode)          MODE="$2"; shift 2 ;;
        --isl)           ISL="$2"; shift 2 ;;
        --osl)           OSL="$2"; shift 2 ;;
        --sweep)         SWEEP="$2"; shift 2 ;;
        --duration)      DURATION="$2"; shift 2 ;;
        --warmup)        WARMUP="$2"; shift 2 ;;
        --prefix-reuse)  PREFIX_REUSE="$2"; shift 2 ;;
        --prefix-groups) PREFIX_GROUPS="$2"; shift 2 ;;
        --run-id)        RUN_ID="$2"; RUN_DIR="$BENCHMARK_ROOT/$RUN_ID"; shift 2 ;;
        --random-seed)   RANDOM_SEED="$2"; shift 2 ;;
        --tokenizer)     TOKENIZER="$2"; shift 2 ;;
        --dataset-file)  DATASET_FILE="$2"; shift 2 ;;
        --no-fail-fast)  FAIL_FAST=0; shift ;;
        -h|--help)
            benchmark_usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1 (use --help for usage)" >&2
            exit 1
            ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
TOTAL_CONCURRENCIES=0
COMPLETED_CONCURRENCIES=0

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    echo -e "  ${GREEN}✓${NC} $1"
}

fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo -e "  ${RED}✗${NC} $1"
    [[ -n "${2:-}" ]] && echo -e "    ${2}"
}

warn() {
    WARN_COUNT=$((WARN_COUNT + 1))
    echo -e "  ${YELLOW}⚠${NC} $1"
    [[ -n "${2:-}" ]] && echo -e "    ${2}"
}

info() {
    echo -e "  ${BLUE}ℹ${NC} $1"
}

section() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

BENCHMARK_PID=$$
cleanup() {
    echo ""
    warn "Benchmark interrupted by signal. Partial results saved to: $RUN_DIR"
    echo ""
    exit 130
}
trap cleanup INT TERM

section "DSpark Production Benchmark — Pre-flight"

if ! command -v aiperf &>/dev/null; then
    fail "AIPerf not installed"
    echo "         Install: uv tool install aiperf==0.12.0 (or re-run scripts/setup_host.sh)"
    exit 1
fi
AIPERF_VERSION=$(aiperf --version 2>/dev/null || echo "unknown")
pass "AIPerf installed: $AIPERF_VERSION"

HEALTH_RESPONSE=$(curl -fsS --max-time 5 "$ENDPOINT/v1/models" 2>/dev/null || echo "CURL_FAILED")
if [[ "$HEALTH_RESPONSE" == "CURL_FAILED" ]]; then
    fail "Cannot reach frontend at $ENDPOINT"
    echo "         Is the deployment running? Check: make serve"
    exit 1
fi
pass "Frontend reachable at $ENDPOINT"

MODEL_FOUND=$(echo "$HEALTH_RESPONSE" | grep -o "\"id\":\"$MODEL\"" || echo "")
if [[ -z "$MODEL_FOUND" ]]; then
    fail "Model '$MODEL' not found in /v1/models"
    echo "         Available models:"
    echo "$HEALTH_RESPONSE" | grep -o '"id":"[^"]*"' | sed 's/"id":"/           - /' | sed 's/"//'
    exit 1
fi
pass "Model '$MODEL' is registered"

PREFILL_UP=false
DECODE_UP=false
if curl -fsS --max-time 3 "http://localhost:$PREFILL_WORKER_PORT/metrics" &>/dev/null; then
    PREFILL_UP=true
    pass "Prefill worker metrics reachable on :$PREFILL_WORKER_PORT"
else
    warn "Prefill worker metrics unreachable on :$PREFILL_WORKER_PORT (metrics may not be enabled)"
fi
if curl -fsS --max-time 3 "http://localhost:$DECODE_WORKER_PORT/metrics" &>/dev/null; then
    DECODE_UP=true
    pass "Decode worker metrics reachable on :$DECODE_WORKER_PORT"
else
    warn "Decode worker metrics unreachable on :$DECODE_WORKER_PORT (metrics may not be enabled)"
fi

if [[ -z "$TOKENIZER" ]]; then
    TOKENIZER=$(echo "$HEALTH_RESPONSE" | "$PYTHON_BIN" -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for m in data.get('data', []):
        if m.get('id') == '$MODEL':
            print(m.get('owned_by', ''))
            break
except: pass
" 2>/dev/null || echo "")
    if [[ -z "$TOKENIZER" ]]; then
        TOKENIZER="$MODEL"
        info "Tokenizer not explicitly set; AIPerf will resolve from model: $TOKENIZER"
    else
        pass "Tokenizer auto-detected: $TOKENIZER"
    fi
fi

if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    pass "GPU inventory: $GPU_COUNT device(s) visible"
else
    warn "nvidia-smi not found; GPU telemetry will be skipped"
fi

if [[ "$MODE" == "mixed" ]]; then
    TOTAL_WEIGHT=$(echo "$SEQUENCE_DISTRIBUTION" | "$PYTHON_BIN" -c "
import sys
total = 0
for entry in sys.stdin.read().strip().split(';'):
    pair, weight = entry.split(':')
    total += int(weight)
print(total)
" 2>/dev/null || echo "0")
    if [[ "$TOTAL_WEIGHT" != "100" ]]; then
        fail "Mixed mode distribution weights must sum to 100, got $TOTAL_WEIGHT"
        exit 1
    fi
    pass "Mixed mode distribution validated (total weight: $TOTAL_WEIGHT)"
fi

if [[ "$MODE" == "real" ]]; then
    if [[ ! -f "$DATASET_FILE" ]]; then
        fail "MODE=real requires the prompt dataset at $DATASET_FILE"
        echo "         Build it first: make dataset"
        exit 1
    fi
    SAMPLE_COUNT=$(wc -l < "$DATASET_FILE" | tr -d ' ')
    pass "Real-prompt dataset: $SAMPLE_COUNT samples ($DATASET_FILE)"
    if [[ "$SAMPLE_COUNT" -lt 32 ]]; then
        warn "Dataset has fewer than 32 samples; high-concurrency points may exhaust the prompt pool"
    fi
fi

if [[ "$MODE" == "fixed" ]]; then
    RUN_SHAPE="isl-${ISL}_osl-${OSL}"
elif [[ "$MODE" == "real" ]]; then
    DATASET_HASH=$(printf '%s' "$DATASET_FILE" | sha256sum | cut -c1-8)
    RUN_SHAPE="real-$(basename "$DATASET_FILE" .jsonl)-${DATASET_HASH}"
else
    DIST_HASH=$(printf '%s' "$SEQUENCE_DISTRIBUTION" | sha256sum | cut -c1-8)
    RUN_SHAPE="mixed-${DIST_HASH}"
fi

section "DSpark Production Benchmark — Setup"

mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/run-config.json" <<CONF
{
  "run_id": "$RUN_ID",
  "run_shape": "$RUN_SHAPE",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "endpoint": "$ENDPOINT",
  "model": "$MODEL",
  "tokenizer": "$TOKENIZER",
  "mode": "$MODE",
  "isl": $ISL,
  "osl": $OSL,
  "sequence_distribution": "$SEQUENCE_DISTRIBUTION",
  "concurrencies": "$SWEEP",
  "benchmark_duration_seconds": $DURATION,
  "warmup_requests": $WARMUP,
  "request_timeout_seconds": $REQUEST_TIMEOUT,
  "grace_period_seconds": $GRACE_PERIOD,
  "random_seed": $RANDOM_SEED,
  "prefix_reuse_percent": $PREFIX_REUSE,
  "prefix_groups": $PREFIX_GROUPS,
  "num_dataset_entries": $NUM_DATASET_ENTRIES,
  "dataset_file": "$DATASET_FILE",
  "aiperf_version": "$AIPERF_VERSION",
  "fail_fast": $FAIL_FAST
}
CONF
pass "Run configuration written to $RUN_DIR/run-config.json"
info "Run ID:    $RUN_ID"
info "Run shape: $RUN_SHAPE"
info "Mode:      $MODE"
info "Sweep:     $SWEEP"

section "DSpark Production Benchmark — Pre-benchmark Snapshots"

if command -v nvidia-smi &>/dev/null; then
    GPU_QUERY="index,name,compute_cap,memory.total,memory.used,memory.free,temperature.gpu,power.draw,utilization.gpu"
    {
        echo "phase,$(nvidia-smi --query-gpu="$GPU_QUERY" --format=csv 2>/dev/null | head -1)"
        nvidia-smi --query-gpu="$GPU_QUERY" --format=csv,noheader,nounits 2>/dev/null | sed 's/^/before,/'
    } > "$RUN_DIR/gpu-snapshots.csv"
    pass "GPU snapshot saved (gpu-snapshots.csv)"
fi

section "DSpark Production Benchmark — Concurrency Sweep"

echo '{"runs":[' > "$RUN_DIR/sweep-summary.json"
FIRST_RUN=true
TOTAL_CONCURRENCIES=$(echo "$SWEEP" | wc -w | tr -d ' ')

for CONCURRENCY in $SWEEP; do
    echo ""
    echo -e "${CYAN}━━━ Concurrency $CONCURRENCY / $TOTAL_CONCURRENCIES ━━━${NC}"
    echo ""

    CONC_DIR="$RUN_DIR/c${CONCURRENCY}"
    mkdir -p "$CONC_DIR"

    if [[ -f "$CONC_DIR/profile_export_aiperf.json" ]]; then
        warn "Artifacts already exist for concurrency $CONCURRENCY — skipping"
        echo "    To re-run: delete $CONC_DIR first"
        COMPLETED_CONCURRENCIES=$((COMPLETED_CONCURRENCIES + 1))
        continue
    fi

    AIPERF_ARGS=(
        aiperf profile
        --model "$MODEL"
        --url "$ENDPOINT"
        --endpoint-type chat
        --streaming
        --concurrency "$CONCURRENCY"
        --benchmark-duration "$DURATION"
        --warmup-request-count "$WARMUP"
        --request-timeout-seconds "$REQUEST_TIMEOUT"
        --benchmark-grace-period "$GRACE_PERIOD"
        --extra-inputs "ignore_eos:true"
        --extra-inputs "temperature:0.0"
        --num-dataset_entries "$NUM_DATASET_ENTRIES"
        --random-seed "$RANDOM_SEED"
        --ui simple
        --artifact-dir "$CONC_DIR"
        --server-metrics-formats json csv jsonl
    )

    if [[ -n "$TOKENIZER" ]]; then
        AIPERF_ARGS+=(--tokenizer "$TOKENIZER")
    fi

    if [[ "$MODE" == "fixed" ]]; then
        AIPERF_ARGS+=(
            --isl "$ISL"
            --isl-stddev 0
            --osl "$OSL"
            --osl-stddev 0
            --extra-inputs "max_tokens:$OSL"
            --extra-inputs "min_tokens:$OSL"
        )
    elif [[ "$MODE" == "mixed" ]]; then
        AIPERF_ARGS+=(--sequence-distribution "$SEQUENCE_DISTRIBUTION")
    elif [[ "$MODE" == "real" ]]; then
        AIPERF_ARGS+=(
            --custom-dataset-type multi-turn
            --custom-dataset-path "$DATASET_FILE"
            --extra-inputs "max_tokens:$OSL"
        )
    fi

    if [[ "$PREFIX_REUSE" -gt 0 && "$MODE" != "real" ]]; then
        PREFIX_TOKENS=$(( ISL * PREFIX_REUSE / 100 ))
        UNIQUE_TOKENS=$(( ISL - PREFIX_TOKENS ))
        if [[ "$PREFIX_TOKENS" -lt 1 || "$UNIQUE_TOKENS" -lt 1 ]]; then
            fail "PREFIX_REUSE=$PREFIX_REUSE leaves insufficient tokens (ISL=$ISL)"
            continue
        fi
        AIPERF_ARGS+=(
            --prefix-prompt-pool-size "$PREFIX_GROUPS"
            --prefix-prompt-length "$PREFIX_TOKENS"
        )
    fi

    log "Starting AIPerf: concurrency=$CONCURRENCY mode=$MODE shape=$RUN_SHAPE"
    info "Command: ${AIPERF_ARGS[*]}"

    BENCH_START=$(date +%s)
    AIPERF_EXIT=0
    "${AIPERF_ARGS[@]}" > "$CONC_DIR/aiperf-stdout.log" 2> "$CONC_DIR/aiperf-stderr.log" || AIPERF_EXIT=$?
    BENCH_END=$(date +%s)
    BENCH_ELAPSED=$(( BENCH_END - BENCH_START ))

    if [[ $AIPERF_EXIT -eq 0 ]]; then
        STATUS="PASS"
        pass "Concurrency $CONCURRENCY completed in ${BENCH_ELAPSED}s"
        COMPLETED_CONCURRENCIES=$((COMPLETED_CONCURRENCIES + 1))
    else
        STATUS="FAIL"
        fail "Concurrency $CONCURRENCY failed (exit code $AIPERF_EXIT, ${BENCH_ELAPSED}s)"
        if [[ -f "$CONC_DIR/aiperf-stderr.log" ]]; then
            tail -20 "$CONC_DIR/aiperf-stderr.log" | sed 's/^/    /'
        fi
        if [[ "$FAIL_FAST" -eq 1 ]]; then
            fail "FAIL_FAST=1 — aborting sweep"
            break
        fi
    fi

    METRICS_JSON=""
    if [[ -f "$CONC_DIR/profile_export_aiperf.json" ]]; then
        METRICS_JSON=$(cat "$CONC_DIR/profile_export_aiperf.json")
    fi

    if [[ "$FIRST_RUN" == "true" ]]; then
        FIRST_RUN=false
    else
        echo "," >> "$RUN_DIR/sweep-summary.json"
    fi

    TTFT_MEAN=$(echo "$METRICS_JSON" | "$PYTHON_BIN" -c "
import sys, json
try:
    d = json.load(sys.stdin)
    metrics = d.get('benchmark_results', {}).get('llm_metrics', [])
    for m in metrics:
        if 'Time to First Token' in m.get('name', '') and m.get('percentile') == 'mean':
            print(f\"{m['value']:.1f}\")
            break
    else:
        print('N/A')
except: print('N/A')
" 2>/dev/null || echo "N/A")

    ITL_MEAN=$(echo "$METRICS_JSON" | "$PYTHON_BIN" -c "
import sys, json
try:
    d = json.load(sys.stdin)
    metrics = d.get('benchmark_results', {}).get('llm_metrics', [])
    for m in metrics:
        if 'Inter Token Latency' in m.get('name', '') and m.get('percentile') == 'mean':
            print(f\"{m['value']:.1f}\")
            break
    else:
        print('N/A')
except: print('N/A')
" 2>/dev/null || echo "N/A")

    THROUGHPUT=$(echo "$METRICS_JSON" | "$PYTHON_BIN" -c "
import sys, json
try:
    d = json.load(sys.stdin)
    metrics = d.get('benchmark_results', {}).get('llm_metrics', [])
    for m in metrics:
        if 'Output Token Throughput' in m.get('name', '') and 'Per User' not in m.get('name', ''):
            print(f\"{m['value']:.2f}\")
            break
    else:
        print('N/A')
except: print('N/A')
" 2>/dev/null || echo "N/A")

    REQ_THROUGHPUT=$(echo "$METRICS_JSON" | "$PYTHON_BIN" -c "
import sys, json
try:
    d = json.load(sys.stdin)
    metrics = d.get('benchmark_results', {}).get('llm_metrics', [])
    for m in metrics:
        if 'Request Throughput' in m.get('name', ''):
            print(f\"{m['value']:.2f}\")
            break
    else:
        print('N/A')
except: print('N/A')
" 2>/dev/null || echo "N/A")

    echo ""
    echo -e "  ${BOLD}Concurrency $CONCURRENCY Results:${NC}"
    echo -e "    TTFT (mean):           ${TTFT_MEAN} ms"
    echo -e "    ITL (mean):            ${ITL_MEAN} ms"
    echo -e "    Token Throughput:      ${THROUGHPUT} tokens/s"
    echo -e "    Request Throughput:    ${REQ_THROUGHPUT} req/s"
    echo -e "    Duration:              ${BENCH_ELAPSED}s"
    echo -e "    Status:                $STATUS"
    echo ""

    cat >> "$RUN_DIR/sweep-summary.json" <<ENTRY
{
  "concurrency": $CONCURRENCY,
  "status": "$STATUS",
  "elapsed_seconds": $BENCH_ELAPSED,
  "ttft_mean_ms": "$TTFT_MEAN",
  "itl_mean_ms": "$ITL_MEAN",
  "token_throughput": "$THROUGHPUT",
  "request_throughput": "$REQ_THROUGHPUT",
  "artifact_dir": "c$CONCURRENCY"
}
ENTRY

done

echo ']}' >> "$RUN_DIR/sweep-summary.json"

section "DSpark Production Benchmark — Post-benchmark Snapshots"

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu="$GPU_QUERY" --format=csv,noheader,nounits 2>/dev/null | sed 's/^/after,/' >> "$RUN_DIR/gpu-snapshots.csv"
    pass "Post-benchmark GPU snapshot appended"
fi

if [[ "$PREFILL_UP" == "true" ]]; then
    curl -fsS --max-time 5 "http://localhost:$PREFILL_WORKER_PORT/metrics" > "$RUN_DIR/worker-metrics-after.prefill.prom" 2>/dev/null || true
fi
if [[ "$DECODE_UP" == "true" ]]; then
    curl -fsS --max-time 5 "http://localhost:$DECODE_WORKER_PORT/metrics" > "$RUN_DIR/worker-metrics-after.decode.prom" 2>/dev/null || true
fi
pass "Post-benchmark worker /metrics snapshots saved"

section "DSpark Production Benchmark — Summary Report"

cat <<REPORT
================================================================================
  DSpark Production Benchmark Report
  Run ID:    $RUN_ID
  Run Shape: $RUN_SHAPE
  Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
================================================================================

Configuration:
  Endpoint:    $ENDPOINT
  Model:       $MODEL
  Mode:        $MODE
  ISL:         $ISL (ignored in real mode)
  OSL:         $OSL
  Sweep:       $SWEEP
  Duration:    ${DURATION}s per concurrency
  Warmup:      $WARMUP requests
  Prefix:      ${PREFIX_REUSE}% reuse, ${PREFIX_GROUPS} groups
  Seed:        $RANDOM_SEED

Results:
REPORT

"$PYTHON_BIN" - "$RUN_DIR/sweep-summary.json" <<'PYREPORT'
import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)

header = f"{'Concurrency':>12}  {'TTFT (ms)':>10}  {'ITL (ms)':>10}  {'Tokens/s':>12}  {'Req/s':>10}  {'Status':>6}  {'Time':>6}"
print(header)
print("-" * len(header))

for run in data['runs']:
    conc = run['concurrency']
    ttft = run.get('ttft_mean_ms', 'N/A')
    itl = run.get('itl_mean_ms', 'N/A')
    tput = run.get('token_throughput', 'N/A')
    rps = run.get('request_throughput', 'N/A')
    status = run['status']
    elapsed = f"{run['elapsed_seconds']}s"
    print(f"{conc:>12}  {ttft:>10}  {itl:>10}  {tput:>12}  {rps:>10}  {status:>6}  {elapsed:>6}")
PYREPORT

cat <<REPORT

Artifacts:
  Config:     $RUN_DIR/run-config.json
  Summary:    $RUN_DIR/sweep-summary.json
  GPU:        $RUN_DIR/gpu-snapshots.csv

Verdict:
  Total concurrency points:  $TOTAL_CONCURRENCIES
  Completed:                 $COMPLETED_CONCURRENCIES
  Passed:                    $PASS_COUNT
  Failed:                    $FAIL_COUNT
  Warnings:                  $WARN_COUNT

$(if [[ $FAIL_COUNT -eq 0 ]]; then echo "  ✓ ALL CONCURRENCY POINTS PASSED"; else echo "  ✗ SOME CONCURRENCY POINTS FAILED — review logs in $RUN_DIR"; fi)

================================================================================
REPORT

section "DSpark Production Benchmark — Verdict"

echo -e "  ${BOLD}Run:${NC}     $RUN_ID"
echo -e "  ${BOLD}Shape:${NC}   $RUN_SHAPE"
echo -e "  ${BOLD}Artifacts:${NC} $RUN_DIR"
echo ""

echo -e "  ${GREEN}Passed:${NC}  $PASS_COUNT"
echo -e "  ${RED}Failed:${NC}  $FAIL_COUNT"
echo -e "  ${YELLOW}Warned:${NC}  $WARN_COUNT"
echo ""

if [[ $FAIL_COUNT -eq 0 ]]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✓ BENCHMARK COMPLETE — ALL CONCURRENCY POINTS PASSED${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  Results saved to: $RUN_DIR"
    echo "  View raw data:    cat $RUN_DIR/sweep-summary.json"
    echo ""
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  ✗ BENCHMARK COMPLETE — $FAIL_COUNT CONCURRENCY POINT(S) FAILED${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  Review failures in: $RUN_DIR"
    echo "  Check logs:         $RUN_DIR/c*/aiperf-stderr.log"
    echo ""
    exit 1
fi
