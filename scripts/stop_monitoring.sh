#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

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

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Stopping DSpark Monitoring${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [[ -f "$REPO_ROOT/.prometheus.pid" ]]; then
    PROMETHEUS_PID=$(cat "$REPO_ROOT/.prometheus.pid")
    if kill -0 "$PROMETHEUS_PID" 2>/dev/null; then
        info "Stopping Prometheus (PID: $PROMETHEUS_PID)..."
        kill "$PROMETHEUS_PID"
        success "Prometheus stopped"
    else
        info "Prometheus not running"
    fi
    rm -f "$REPO_ROOT/.prometheus.pid"
else
    info "No Prometheus PID file found"
fi

if [[ -f "$REPO_ROOT/.grafana.pid" ]]; then
    GRAFANA_PID=$(cat "$REPO_ROOT/.grafana.pid")
    if kill -0 "$GRAFANA_PID" 2>/dev/null; then
        info "Stopping Grafana (PID: $GRAFANA_PID)..."
        kill "$GRAFANA_PID"
        success "Grafana stopped"
    else
        info "Grafana not running"
    fi
    rm -f "$REPO_ROOT/.grafana.pid"
else
    info "No Grafana PID file found"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Monitoring Stopped${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
