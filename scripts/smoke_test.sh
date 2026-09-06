#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../deploy/common/launch_utils.sh"
resolve_venv_python

ENDPOINT="http://localhost:8000"
MODEL="deepseek-v4-flash-dspark"
PREFILL_WORKER_PORT="${DYN_SYSTEM_PORT1:-8081}"
DECODE_WORKER_PORT="${DYN_SYSTEM_PORT2:-8082}"
TIMEOUT=60

while [[ $# -gt 0 ]]; do
    case $1 in
        --endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        -h|--help)
            echo "usage: smoke_test.sh [--endpoint URL]"
            echo "  --endpoint URL   frontend endpoint to test (default: http://localhost:8000)"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1 (use --help for usage)"
            exit 1
            ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0

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
    echo -e "  ${YELLOW}⚠${NC} $1"
    [[ -n "${2:-}" ]] && echo -e "    ${2}"
}

info() {
    echo -e "  ${BLUE}ℹ${NC} $1"
}

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  DSpark Smoke Test${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Endpoint:   $ENDPOINT"
echo "  Model:      $MODEL"
echo "  Workers:    prefill localhost:$PREFILL_WORKER_PORT  decode localhost:$DECODE_WORKER_PORT"
echo "  Timeout:    ${TIMEOUT}s"
echo ""

echo -e "${BLUE}[1/5] Frontend Reachability${NC}"

HEALTH_RESPONSE=$(curl -fsS --max-time 5 "$ENDPOINT/v1/models" 2>/dev/null || echo "CURL_FAILED")

if [[ "$HEALTH_RESPONSE" == "CURL_FAILED" ]]; then
    fail "Cannot reach frontend at $ENDPOINT"
    echo "         Is the frontend running? Check: make serve"
    echo ""
    echo -e "${RED}Smoke test aborted: frontend unreachable${NC}"
    exit 1
fi

pass "Frontend is reachable at $ENDPOINT"
info "Response: $HEALTH_RESPONSE"

echo ""
echo -e "${BLUE}[2/5] Model Registration${NC}"

MODEL_FOUND=$("$PYTHON_BIN" -c '
import json, sys
try:
    d = json.load(sys.stdin)
    ids = [m.get("id", "") for m in d.get("data", [])]
except Exception:
    sys.exit(1)
sys.exit(0 if "'"$MODEL"'" in ids else 1)
' <<<"$HEALTH_RESPONSE" && echo "found" || echo "")

if [[ -z "$MODEL_FOUND" ]]; then
    fail "Model '$MODEL' not found in /v1/models"
    echo "         Available models:"
    "$PYTHON_BIN" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for m in d.get("data", []):
    print("           - " + str(m.get("id", "")))
' <<<"$HEALTH_RESPONSE" || true
    echo ""
    echo -e "${RED}Smoke test aborted: model not registered${NC}"
    exit 1
fi

pass "Model '$MODEL' is registered"

echo ""
echo -e "${BLUE}[3/5] Request Completion${NC}"

REQUEST_BODY=$(cat <<EOF
{
    "model": "$MODEL",
    "messages": [
        {"role": "user", "content": "Reply with exactly one word: operational"}
    ],
    "temperature": 0,
    "max_tokens": 8
}
EOF
)

info "Sending deterministic request..."
START_TIME=$(date +%s%N)

RESPONSE=$(curl -fsS --max-time "$TIMEOUT" \
    -H "Content-Type: application/json" \
    -d "$REQUEST_BODY" \
    "$ENDPOINT/v1/chat/completions" 2>/dev/null || echo "CURL_FAILED")

END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

if [[ "$RESPONSE" == "CURL_FAILED" ]]; then
    fail "Request failed or timed out after ${TIMEOUT}s"
    echo "         Check worker logs for errors"
    echo ""
    echo -e "${RED}Smoke test aborted: request failed${NC}"
    exit 1
fi

PARSED=$(python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    content = d["choices"][0]["message"]["content"]
    model = d.get("model", "")
except Exception:
    sys.exit(1)
print(json.dumps({"content": content, "model": model}))
' <<<"$RESPONSE" 2>/dev/null) || PARSED=""

if [[ -z "$PARSED" ]]; then
    fail "Could not parse the chat completion response"
    echo "    Raw response head: $(echo "$RESPONSE" | head -c 300)"
    echo ""
    echo -e "${RED}Smoke test aborted: unparseable response${NC}"
    exit 1
fi

RESPONSE_CONTENT=$(python3 -c 'import json, sys; print(json.loads(sys.argv[1])["content"])' "$PARSED")
RESPONSE_MODEL=$(python3 -c 'import json, sys; print(json.loads(sys.argv[1])["model"])' "$PARSED")

info "Response content: '$RESPONSE_CONTENT'"
info "Response model: $RESPONSE_MODEL"
info "Elapsed: ${ELAPSED_MS}ms"

if [[ -z "$RESPONSE_CONTENT" ]]; then
    fail "Response content is empty"
elif [[ "$RESPONSE_CONTENT" != *"operational"* ]]; then
    fail "Response content does not contain expected text"
    echo "    Expected: 'operational' | Got: '$RESPONSE_CONTENT'"
else
    pass "Response content is correct"
fi

if [[ "$RESPONSE_MODEL" == "$MODEL" ]]; then
    pass "Response model matches: $RESPONSE_MODEL"
else
    fail "Response model mismatch"
    echo "    Expected: $MODEL | Got: $RESPONSE_MODEL"
fi

if [[ $ELAPSED_MS -lt 5000 ]]; then
    pass "Response time: ${ELAPSED_MS}ms (healthy)"
elif [[ $ELAPSED_MS -lt 30000 ]]; then
    warn "Response time slow: ${ELAPSED_MS}ms"
    echo "    First request may be slower due to model warmup"
else
    fail "Response time too slow: ${ELAPSED_MS}ms"
    echo "    Expected < 5s for warm model"
fi

echo ""
echo -e "${BLUE}[4/5] Worker Health${NC}"

PREFILL_METRICS=$(curl -fsS --max-time 5 "http://localhost:$PREFILL_WORKER_PORT/metrics" 2>/dev/null || echo "CURL_FAILED")
DECODE_METRICS=$(curl -fsS --max-time 5 "http://localhost:$DECODE_WORKER_PORT/metrics" 2>/dev/null || echo "CURL_FAILED")
METRICS_RESPONSE="$DECODE_METRICS"

if [[ "$PREFILL_METRICS" == "CURL_FAILED" ]]; then
    warn "Cannot reach prefill worker metrics at port $PREFILL_WORKER_PORT"
    echo "    Metrics may not be enabled (--enable-metrics)"
else
    pass "Prefill worker metrics endpoint is healthy"
fi

if [[ "$DECODE_METRICS" == "CURL_FAILED" ]]; then
    warn "Cannot reach decode worker metrics at port $DECODE_WORKER_PORT"
    echo "    Metrics may not be enabled (--enable-metrics)"
else
    if echo "$DECODE_METRICS" | grep -q "sglang:num_running_reqs"; then
        pass "Decode worker metrics endpoint is healthy"
        RUNNING=$(echo "$DECODE_METRICS" | grep "sglang:num_running_reqs" | grep -v "#" | awk '{print $2}')
        info "Running requests: ${RUNNING:-0}"
    else
        warn "Metrics endpoint responding but missing expected metrics"
    fi

    if echo "$DECODE_METRICS" | grep -q "dynamo"; then
        pass "Dynamo component metrics present"
    fi
fi

echo ""
echo -e "${BLUE}[5/5] DSpark Speculative Decoding${NC}"

if [[ "$METRICS_RESPONSE" == "CURL_FAILED" ]]; then
    warn "Cannot verify DSpark: metrics endpoint unreachable"
else
    if echo "$METRICS_RESPONSE" | grep -q "sglang:spec_accept_rate"; then
        pass "DSpark metrics present"
        
        ACCEPT_RATE=$(echo "$METRICS_RESPONSE" | grep "sglang:spec_accept_rate" | grep -v "#" | awk '{print $2}')
        if [[ -n "$ACCEPT_RATE" ]]; then
            ACCEPT_PCT=$(awk -v rate="$ACCEPT_RATE" 'BEGIN { printf "%.1f", rate * 100 }')
            info "Acceptance rate: ${ACCEPT_PCT}%"
            
            if [[ "$ACCEPT_RATE" != "0" && "$ACCEPT_RATE" != "0.0" ]]; then
                pass "Acceptance rate is non-zero (speculation active)"
            else
                warn "Acceptance rate is zero"
                echo "    No speculative traffic yet or DSpark disabled"
            fi
        fi
    else
        fail "DSpark metrics not found"
        echo "    Check --speculative-algorithm DSPARK is set"
    fi
    
    if echo "$METRICS_RESPONSE" | grep -q "sglang:spec_accept_length"; then
        pass "DSpark verify length metrics present"
    fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC}  $PASS_COUNT"
echo -e "  ${RED}Failed:${NC}  $FAIL_COUNT"
echo ""

if [[ $FAIL_COUNT -eq 0 ]]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✓ SMOKE TEST PASSED${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  The DSpark serving system is operational."
    echo "  Endpoint: $ENDPOINT"
    echo "  Model:    $MODEL"
    echo ""
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  ✗ SMOKE TEST FAILED${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  The DSpark serving system has issues."
    echo ""
    echo "  Troubleshooting:"
    echo "    1. Is the frontend running?        → make serve"
    echo "    2. Is the model loaded?             → check worker logs"
    echo "    3. Are metrics enabled?             → add --enable-metrics"
    echo "    4. Is DSpark configured?            → add --speculative-algorithm DSPARK"
    echo ""
    exit 1
fi
