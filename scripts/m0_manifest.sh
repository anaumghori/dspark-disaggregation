#!/usr/bin/env bash
set -euo pipefail

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] m0_manifest: $*"
}

fail() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] m0_manifest: FAILED: $*" >&2
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

command -v nvcc >/dev/null 2>&1 || fail "nvcc not on PATH; FlashInfer JIT requires a CUDA devel environment (CUDA_HOME set, \$CUDA_HOME/bin on PATH)"
log "nvcc: $(nvcc --version | tail -1)"
[[ -n "${CUDA_HOME:-}" ]] && log "CUDA_HOME=$CUDA_HOME" || log "CUDA_HOME not set (nvcc on PATH is sufficient if CUDA_HOME defaults resolve)"

# ldconfig cache can be stale in Modal snapshots (e.g. after adding libibverbs);
# refresh and fall back to direct filesystem check so manifest doesn't fail
# spuriously with `libnuma.so.1 not found` when the file is present.
ldconfig 2>/dev/null || true
if ! ldconfig -p 2>/dev/null | grep -q "libnuma.so.1"; then
  if [ -f "/usr/lib/x86_64-linux-gnu/libnuma.so.1" ] || [ -f "/lib/x86_64-linux-gnu/libnuma.so.1" ] || [ -f "/usr/lib/libnuma.so.1" ]; then
    log "libnuma.so.1 found via filesystem (ldconfig cache was stale)"
  else
    fail "libnuma.so.1 not found; sglang-kernel's common_ops.abi3.so links against it (apt install libnuma-dev)"
  fi
fi

export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$REPO_ROOT/.cache/kernels/flashinfer-sm120}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/.cache/kernels/triton-sm120}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR"
log "kernel caches: flashinfer=$FLASHINFER_WORKSPACE_BASE triton=$TRITON_CACHE_DIR"

DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
if [[ "$DISCOVERY_BACKEND" != "file" ]]; then
    fail "DYN_DISCOVERY_BACKEND=$DISCOVERY_BACKEND but this deployment runs single-node with the file registry (unset it or set it to 'file')"
fi
log "discovery backend: file (registry at ${DYN_FILE_KV:-<launch default>})"

log "running assertion suite on $EXPECTED_GPUS expected GPU(s), HF_HOME=$HF_HOME"
python3 - "$EXPECTED_GPUS" "$HF_HOME" <<'PY'
import enum
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

expected_gpus = int(sys.argv[1])
hf_home = Path(sys.argv[2])
failures = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"[m0] {status}: {name}" + (f" ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


versions = {}
for dist in ("sglang", "flashinfer-python", "ai-dynamo", "ai-dynamo-runtime", "mooncake-transfer-engine", "blake3", "torch", "sglang-kernel"):
    try:
        versions[dist] = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        versions[dist] = None
# mooncake installs as cuda13 variant (mooncake-transfer-engine-cuda13) on CUDA 13;
# fall back to that name so the check passes in both cases (same as smoke test)
if versions.get("mooncake-transfer-engine") is None:
    try:
        versions["mooncake-transfer-engine"] = importlib.metadata.version("mooncake-transfer-engine-cuda13")
    except importlib.metadata.PackageNotFoundError:
        pass
for dist, ver in versions.items():
    print(f"[m0] version {dist}={ver}")
check("flashinfer pinned to 0.6.18", versions["flashinfer-python"] == "0.6.18", versions["flashinfer-python"])
check("mooncake-transfer-engine installed", versions["mooncake-transfer-engine"] is not None, versions["mooncake-transfer-engine"])
check("ai-dynamo installed", versions["ai-dynamo"] is not None)
check("ai-dynamo-runtime installed", versions["ai-dynamo-runtime"] is not None)
check("sglang installed", versions["sglang"] is not None)

check("sglang top-level API (Engine, Runtime, __version__)",
      all(hasattr(importlib.import_module("sglang"), attr) for attr in ("Engine", "Runtime", "__version__")))

from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
check("SpeculativeAlgorithm.DSPARK exists", hasattr(SpeculativeAlgorithm, "DSPARK"))

from sglang.srt.speculative.ragged_verify import RaggedVerifyMode, read_ragged_verify_mode
mode_values = {m.value for m in RaggedVerifyMode}
check("RaggedVerifyMode values static/cap-accept/compact", mode_values == {"static", "cap-accept", "compact"}, str(sorted(mode_values)))
check("ragged-verify mode default is static", read_ragged_verify_mode() == RaggedVerifyMode.STATIC)

from sglang.srt.speculative.dspark_disaggregation import build_dspark_disagg_draft_input
check("dspark_disaggregation.build_dspark_disagg_draft_input importable", callable(build_dspark_disagg_draft_input))

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.utils import is_sm120_supported, maybe_reindex_device_id
check("utils surface (is_sm120_supported, maybe_reindex_device_id)", callable(is_sm120_supported) and callable(maybe_reindex_device_id))

from sglang.srt.disaggregation.mooncake import MooncakeKVBootstrapServer, MooncakeKVManager, MooncakeKVReceiver, MooncakeKVSender
from sglang.srt.disaggregation.utils import TransferBackend
check("Mooncake transfer backend classes importable", all(cls is not None for cls in (MooncakeKVBootstrapServer, MooncakeKVManager, MooncakeKVReceiver, MooncakeKVSender)))
check(
    "TransferBackend is a plain Enum with MOONCAKE='mooncake'",
    issubclass(TransferBackend, enum.Enum) and not issubclass(TransferBackend, str)
    and TransferBackend.MOONCAKE.value == "mooncake",
)

import sglang.benchmark.dspark_sps_profiler  # noqa: F401
import sglang.benchmark.dspark_sts_fit  # noqa: F401
check("calibration CLIs importable (sglang.benchmark.dspark_sps_profiler, dspark_sts_fit)", True)

from dynamo.common.backend.engine import LLMEngine
check("dynamo LLMEngine contract importable", LLMEngine is not None)

import torch

check("torch CUDA available", torch.cuda.is_available(), f"torch {torch.__version__}")
device_count = torch.cuda.device_count()
check(f"CUDA device count == {expected_gpus}", device_count == expected_gpus, f"found {device_count}")
for idx in range(device_count):
    props = torch.cuda.get_device_properties(idx)
    print(f"[m0] gpu {idx}: {props.name} cc={props.major}.{props.minor} memory={props.total_memory / (1 << 30):.1f} GiB")
    check(f"gpu {idx} is sm_120", (props.major, props.minor) == (12, 0), f"cc={props.major}.{props.minor}")

if device_count >= 2:
    peer_failures = []
    for i in range(device_count):
        for j in range(device_count):
            if i != j and not torch.cuda.can_device_access_peer(i, j):
                peer_failures.append(f"{i}->{j}")
    if peer_failures:
        print(f"[m0] note: no peer access for pairs {peer_failures} (relevant to the disaggregated KV-transfer path; see the topology notes)")
    check("at least one GPU peer-access path exists", len(peer_failures) < device_count * (device_count - 1))

import sglang

attn_metadata = Path(sglang.__file__).parent / "kernels" / "ops" / "speculative" / "dspark" / "dspark_attn_metadata.py"
check("dspark_attn_metadata.py present in installed sglang", attn_metadata.exists(), str(attn_metadata))

from flashinfer.jit.env import FLASHINFER_CSRC_DIR

expected_sources = [
    "group_gemm_mxfp4_groupwise_sm120.cu",
    "group_gemm_nvfp4_groupwise_sm120.cu",
    "group_gemm_fp8_groupwise_sm120.cu",
    "sparse_mla_sm120_decode_dsv4.cu",
]
csrc_dir = Path(FLASHINFER_CSRC_DIR)
for name in expected_sources:
    check(f"flashinfer ships sm_120 kernel source {name}", (csrc_dir / name).exists(), str(csrc_dir / name))

import re

dsv4_decode_src = csrc_dir / "sparse_mla_sm120_decode_dsv4.cu"
dsv4_topk_192 = dsv4_decode_src.exists() and bool(
    re.search(r"DSV4_DISPATCH\(\s*\d+\s*,\s*192\s*\)", dsv4_decode_src.read_text())
)
check(
    "flashinfer sm120 DSV4 decode kernel instantiates topk 192 (DSpark draft SWA window)",
    dsv4_topk_192,
    str(dsv4_decode_src),
)

import sglang.srt.layers.deep_gemm_wrapper as dgw

deepgemm_enabled = getattr(dgw, "ENABLE_JIT_DEEPGEMM", None)
if deepgemm_enabled is None:
    print("[m0] note: deep_gemm wrapper does not expose ENABLE_JIT_DEEPGEMM at this snapshot")
else:
    check("DeepGEMM disabled on sm_120 (no tcgen05/TMEM)", deepgemm_enabled is False, str(deepgemm_enabled))

model_repo = "deepseek-ai/DeepSeek-V4-Flash-0731"
snapshot_glob = sorted((hf_home / "hub" / f"models--{model_repo.replace('/', '--')}" / "snapshots").glob("*/config.json"))
if not snapshot_glob:
    print(f"[m0] note: checkpoint {model_repo} not found under {hf_home}/hub; checkpoint assertions skipped (download it or set HF_HOME)")
else:
    config = json.loads(snapshot_glob[-1].read_text())
    snap = snapshot_glob[-1].parent
    print(f"[m0] checkpoint snapshot: {snap}")
    q = config.get("quantization_config", {})
    check("checkpoint architectures", config.get("architectures") == ["DeepseekV4ForCausalLM"], str(config.get("architectures")))
    check("quantization_config is fp8 e4m3", q.get("quant_method") == "fp8" and q.get("fmt") == "e4m3")
    check("quantization scale_fmt ue8m0", q.get("scale_fmt") == "ue8m0")
    check("weight_block_size [128,128]", q.get("weight_block_size") == [128, 128], str(q.get("weight_block_size")))
    check("top-level expert_dtype is fp4", config.get("expert_dtype") == "fp4", str(config.get("expert_dtype")))
    check("dspark_block_size == 5", config.get("dspark_block_size") == 5)
    check("dspark_markov_rank == 256", config.get("dspark_markov_rank") == 256)
    check("dspark_target_layer_ids == [40,41,42]", config.get("dspark_target_layer_ids") == [40, 41, 42])
    check("dspark_noise_token_id == 128799", config.get("dspark_noise_token_id") == 128799)
    check("index_topk == 512", config.get("index_topk") == 512)
    check("max_position_embeddings == 1048576", config.get("max_position_embeddings") == 1048576)

    index_path = snap / "model.safetensors.index.json"
    if index_path.exists():
        keys = list(json.loads(index_path.read_text())["weight_map"])
        check("mtp.* drafter weights present", any(k.startswith("mtp.") for k in keys))
        check("trained confidence head present (mtp.*.confidence_head.*)", any("confidence_head" in k for k in keys))
        check("markov head present (mtp.*.markov_head.*)", any("markov_head" in k for k in keys))
    else:
        print("[m0] note: model.safetensors.index.json not in snapshot yet; weight-group assertions skipped")

if failures:
    print(f"[m0] FAILED: {len(failures)} assertion(s): {failures}")
    sys.exit(1)
print("[m0] all assertions passed")
PY

log "manifest complete"
