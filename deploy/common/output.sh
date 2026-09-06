#!/usr/bin/env bash

LOG_PREFIX="${LOG_PREFIX:-}"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${LOG_PREFIX:+$LOG_PREFIX: }$*"
}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    echo -e "  ${GREEN}✓${NC} $1"
}

fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo -e "  ${RED}✗${NC} $1"
    if [[ -n "${2:-}" ]]; then echo -e "    ${2}"; fi
}

warn() {
    WARN_COUNT=$((WARN_COUNT + 1))
    echo -e "  ${YELLOW}⚠${NC} $1"
    if [[ -n "${2:-}" ]]; then echo -e "    ${2}"; fi
}

info() {
    echo -e "  ${BLUE}ℹ${NC} $1"
}