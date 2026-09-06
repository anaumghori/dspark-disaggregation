#!/usr/bin/env bash
set -euo pipefail

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] apply_sm120_patches: $*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/../patches"

if [[ ! -d "$PATCH_DIR" ]] || ! ls "$PATCH_DIR"/*.patch >/dev/null 2>&1; then
    log "no patches present under $PATCH_DIR; nothing to do"
    exit 0
fi

SGLANG_PARENT="$(python3 -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))')"
SGLANG_VERSION="$(python3 -c 'import importlib.metadata; print(importlib.metadata.version("sglang"))')"
log "installed sglang $SGLANG_VERSION under $SGLANG_PARENT"

applied=0
skipped=0
for patch_file in "$PATCH_DIR"/*.patch; do
    name="$(basename "$patch_file")"
    if patch -p1 -d "$SGLANG_PARENT" --dry-run --silent -R < "$patch_file" >/dev/null 2>&1; then
        log "$name: already applied, skipping"
        skipped=$((skipped + 1))
    elif patch -p1 -d "$SGLANG_PARENT" --dry-run --silent < "$patch_file" >/dev/null 2>&1; then
        log "$name: applying"
        patch -p1 -d "$SGLANG_PARENT" < "$patch_file"
        applied=$((applied + 1))
    else
        log "FATAL: $name does not apply cleanly to the installed sglang $SGLANG_VERSION"
        log "The installed snapshot has likely moved past the patch's context; rebase the patch (see its header for the upstream status) or pin the manifest back to the validated snapshot."
        exit 1
    fi
done

log "overlay complete: $applied applied, $skipped already present"
