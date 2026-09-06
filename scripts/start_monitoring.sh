#!/usr/bin/env bash
set -euo pipefail

PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
START_GRAFANA=true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

while [[ $# -gt 0 ]]; do
    case $1 in
        --prometheus-port)
            PROMETHEUS_PORT="$2"
            shift 2
            ;;
        --grafana-port)
            GRAFANA_PORT="$2"
            shift 2
            ;;
        --no-grafana)
            START_GRAFANA=false
            shift
            ;;
        -h|--help)
            head -17 "$0" | tail -14
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info() {
    echo -e "  ${BLUE}ℹ${NC} $1"
}

success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

error() {
    echo -e "  ${RED}✗${NC} $1"
}

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Starting DSpark Monitoring${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if ! command -v prometheus &> /dev/null; then
    error "Prometheus not found"
    echo "         Install: https://prometheus.io/download/"
    echo "         Or: brew install prometheus (macOS)"
    echo "         Or: apt install prometheus (Ubuntu)"
    exit 1
fi

if [[ "$START_GRAFANA" == "true" ]]; then
    if command -v grafana-server &> /dev/null; then
        GRAFANA_BIN="$(command -v grafana-server)"
        GRAFANA_INVOCATION=(grafana-server)
    elif command -v grafana &> /dev/null; then
        GRAFANA_BIN="$(command -v grafana)"
        GRAFANA_INVOCATION=(grafana server)
    else
        error "Grafana not found"
        echo "         Install: https://grafana.com/grafana/download"
        echo "         Or: brew install grafana (macOS)"
        echo "         Or: apt install grafana (Ubuntu)"
        exit 1
    fi

    GRAFANA_HOMEPATH="${GRAFANA_HOMEPATH:-}"
    if [[ -z "$GRAFANA_HOMEPATH" ]]; then
        GRAFANA_REAL_BIN="$(readlink -f "$GRAFANA_BIN" 2>/dev/null || echo "$GRAFANA_BIN")"
        for candidate in "$(dirname "$GRAFANA_REAL_BIN")" /usr/share/grafana /usr/local/share/grafana /opt/homebrew/share/grafana; do
            if [[ -f "$candidate/conf/defaults.ini" ]]; then
                GRAFANA_HOMEPATH="$candidate"
                break
            fi
        done
    fi
    if [[ -z "$GRAFANA_HOMEPATH" || ! -f "$GRAFANA_HOMEPATH/conf/defaults.ini" ]]; then
        error "Cannot locate Grafana's homepath (conf/defaults.ini not found)"
        echo "         Searched: $(dirname "$GRAFANA_REAL_BIN" 2>/dev/null), /usr/share/grafana, /usr/local/share/grafana, /opt/homebrew/share/grafana"
        echo "         Set GRAFANA_HOMEPATH=/path/to/grafana and re-run."
        exit 1
    fi
    info "Grafana homepath: $GRAFANA_HOMEPATH"
fi

if lsof -Pi ":$PROMETHEUS_PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    error "Port $PROMETHEUS_PORT already in use"
    echo "         Stop the existing process or use --prometheus-port"
    exit 1
fi

if [[ "$START_GRAFANA" == "true" ]] && lsof -Pi ":$GRAFANA_PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    error "Port $GRAFANA_PORT already in use"
    echo "         Stop the existing process or use --grafana-port"
    exit 1
fi

info "Starting Prometheus on port $PROMETHEUS_PORT..."

PROMETHEUS_CONFIG="$REPO_ROOT/observability/prometheus.yml"
PROMETHEUS_DATA_DIR="$REPO_ROOT/.prometheus-data"

mkdir -p "$PROMETHEUS_DATA_DIR" "$REPO_ROOT/run"

prometheus \
    --config.file="$PROMETHEUS_CONFIG" \
    --storage.tsdb.path="$PROMETHEUS_DATA_DIR" \
    --web.listen-address=":$PROMETHEUS_PORT" \
    --web.enable-lifecycle \
    > "$REPO_ROOT/run/prometheus.log" 2>&1 &

PROMETHEUS_PID=$!
echo "$PROMETHEUS_PID" > "$REPO_ROOT/.prometheus.pid"

wait_for_http() {
    local url="$1" pid_file="$2" attempt
    for attempt in $(seq 1 20); do
        if curl -fsS --max-time 2 "$url" > /dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            return 1
        fi
        sleep 1
    done
    return 1
}

if wait_for_http "http://localhost:$PROMETHEUS_PORT/-/healthy" "$REPO_ROOT/.prometheus.pid"; then
    success "Prometheus started (PID: $PROMETHEUS_PID)"
    info "Prometheus: http://localhost:$PROMETHEUS_PORT"
else
    error "Prometheus failed to start or never became healthy"
    echo "         Log: $REPO_ROOT/run/prometheus.log"
    echo "         Check: prometheus --config.file=$PROMETHEUS_CONFIG"
    exit 1
fi

if [[ "$START_GRAFANA" == "true" ]]; then
    info "Starting Grafana on port $GRAFANA_PORT..."
    
    GRAFANA_DATA_DIR="$REPO_ROOT/.grafana-data"
    GRAFANA_PROVISIONING="$REPO_ROOT/observability/grafana/provisioning"
    
    mkdir -p "$GRAFANA_DATA_DIR"
    mkdir -p "$GRAFANA_PROVISIONING/dashboards"
    mkdir -p "$GRAFANA_PROVISIONING/datasources"
    
    cat > "$GRAFANA_PROVISIONING/datasources/prometheus.yml" <<EOF
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:$PROMETHEUS_PORT
    isDefault: true
EOF
    
    cat > "$GRAFANA_PROVISIONING/dashboards/dashboards.yml" <<EOF
apiVersion: 1
providers:
  - name: DSpark
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: $REPO_ROOT/observability
      foldersFromFilesStructure: false
EOF
    
    "$GRAFANA_INVOCATION[@]" \
        --homepath="$GRAFANA_HOMEPATH" \
        cfg:paths.provisioning="$GRAFANA_PROVISIONING" \
        cfg:paths.data="$GRAFANA_DATA_DIR" \
        cfg:server.http_port="$GRAFANA_PORT" \
        > "$REPO_ROOT/run/grafana.log" 2>&1 &

    GRAFANA_PID=$!
    echo "$GRAFANA_PID" > "$REPO_ROOT/.grafana.pid"

    if wait_for_http "http://localhost:$GRAFANA_PORT/api/health" "$REPO_ROOT/.grafana.pid"; then
        success "Grafana started (PID: $GRAFANA_PID)"
    else
        error "Grafana failed to start or never became healthy"
        echo "         Log: $REPO_ROOT/run/grafana.log"
        echo "         Check: $GRAFANA_INVOCATION --homepath=$GRAFANA_HOMEPATH"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Monitoring Started${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Prometheus:  http://localhost:$PROMETHEUS_PORT"
echo "  Grafana:     http://localhost:$GRAFANA_PORT"
echo "               Username: admin"
echo "               Password: admin"
echo ""
echo "  DSpark Dashboard:"
echo "    http://localhost:$GRAFANA_PORT/d/dspark-serving/dspark-serving-deepseek-v4-flash-sm-120"
echo ""
echo "  To stop: make stop-monitoring"
echo ""
