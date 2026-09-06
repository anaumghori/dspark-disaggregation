#!/usr/bin/env bash

build_sglang_gpu_mem_args() {
    if [[ -n "${_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS:-}" ]]; then
        echo "--max-total-tokens ${_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS} --mem-fraction-static 0.9"
        return 0
    fi
    echo ""
}

print_gpu_inventory() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "FATAL: nvidia-smi not found; no NVIDIA driver/toolkit on PATH"
        exit 1
    fi
    log "GPU inventory visible to this process:"
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader | while IFS=, read -r idx name cc mem; do
        log "  gpu ${idx} name=${name} compute_cap=${cc//[[:space:]]/} memory=${mem}"
    done
    local n_gpus
    n_gpus=$(nvidia-smi --list-gpus | wc -l)
    if [[ "$n_gpus" -lt 1 ]]; then
        log "FATAL: nvidia-smi lists no GPUs"
        exit 1
    fi
}
