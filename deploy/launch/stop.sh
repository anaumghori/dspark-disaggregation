#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../common/launch_utils.sh"

stopped=0
for pid_file in "$REPO_ROOT"/run/*.pid; do
    [[ -f "$pid_file" ]] || continue
    name="$(basename "$pid_file" .pid)"
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
        log "stopping $name (pid $pid)"
        kill "$pid" 2>/dev/null || true
    else
        log "$name (pid $pid) is not running"
    fi
    rm -f "$pid_file"
    stopped=$((stopped + 1))
done

for pattern in "python3 -m dynamo.sglang" "python3 -m dynamo.frontend"; do
    if pkill -f "$pattern" 2>/dev/null; then
        log "pkill matched leftover process: $pattern"
        stopped=$((stopped + 1))
    fi
done

log "stop complete ($stopped process groups handled)"
