#!/usr/bin/env bash
set -euo pipefail

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] setup_host: $*"
}

if [[ ${EUID} -ne 0 && -z "${SUDO:-}" ]]; then
    log "not running as root; apt commands will be prefixed with sudo"
    SUDO=sudo
fi
SUDO=${SUDO:-}
log "apt build dependencies: libnuma (sglang-kernel native lib), g++/pkg-config/clang/libclang/cmake (Dynamo runtime + nixl-sys bindgen)"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq libnuma-dev curl unzip g++ pkg-config clang libclang-dev cmake

if ! command -v nvcc >/dev/null 2>&1; then
    log "FATAL: nvcc not found on PATH. This deployment requires a CUDA devel image (FlashInfer JIT compiles its SM120 kernels at first use); set CUDA_HOME and put \$CUDA_HOME/bin on PATH."
    exit 1
fi
log "nvcc found: $(nvcc --version | tail -1)"

if ! command -v uv >/dev/null 2>&1; then
    log "installing uv (Python environment manager used by all entrypoints)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    export UV_BINDIR="$HOME/.local/bin"
fi
log "uv found: $(uv --version)"

# AIPerf load generator (scripts/benchmark.sh). Installed as an isolated uv
# tool, NOT a project dependency: every stable aiperf release pins
# aiohttp~=3.13.3, which is mutually exclusive with ai-dynamo 1.4.2's
# aiohttp>=3.14.3,<4.0. An isolated tool environment sidesteps the conflict.
if ! command -v aiperf >/dev/null 2>&1; then
    log "installing aiperf 0.12.0 as an isolated uv tool (benchmark load generator)"
    uv tool install "aiperf==0.12.0"
fi
log "aiperf found: $(aiperf --version 2>/dev/null || echo unknown)"

if ! command -v cargo >/dev/null 2>&1; then
    log "installing Rust toolchain via rustup (setuptools-rust builds the SGLang PyO3 extensions; edition 2024 requires >= 1.85)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
    export PATH="$HOME/.cargo/bin:$PATH"
fi
log "cargo found: $(cargo --version) (the Dynamo repo pins 1.96.1 via rust-toolchain.toml; rustup installs it on demand during the runtime build)"

PROTOC_VERSION=25.3
if ! command -v protoc >/dev/null 2>&1 || [[ "$(protoc --version | awk '{print $2}')" != "$PROTOC_VERSION" ]]; then
    log "installing protoc $PROTOC_VERSION (ai-dynamo-runtime compiles gRPC protos via tonic-build; apt's 3.21 is too old)"
    case $(uname -m) in
        x86_64)  PROTOC_ZIP="protoc-${PROTOC_VERSION}-linux-x86_64.zip" ;;
        aarch64) PROTOC_ZIP="protoc-${PROTOC_VERSION}-linux-aarch_64.zip" ;;
        *) log "FATAL: unsupported architecture $(uname -m) for the protoc release zip"; exit 1 ;;
    esac
    curl -fsSL -o /tmp/protoc.zip "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/${PROTOC_ZIP}"
    $SUDO unzip -o /tmp/protoc.zip -d /usr/local bin/protoc include/*
    $SUDO ln -sf /usr/local/bin/protoc /usr/bin/protoc
fi
export PROTOC="${PROTOC:-/usr/local/bin/protoc}"
log "protoc found: $(protoc --version) at PROTOC=$PROTOC"

# Prometheus (metrics backend for make monitoring)
if ! command -v prometheus >/dev/null 2>&1; then
    log "installing Prometheus 3.1.0"
    PROM_VERSION="3.1.0"
    case $(uname -m) in
        x86_64)  PROM_ARCH="amd64" ;;
        aarch64) PROM_ARCH="arm64" ;;
        *) log "FATAL: unsupported architecture $(uname -m) for Prometheus"; exit 1 ;;
    esac
    curl -fsSL -o /tmp/prom.tgz "https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-${PROM_ARCH}.tar.gz"
    tar -xzf /tmp/prom.tgz -C /tmp
    $SUDO cp "/tmp/prometheus-${PROM_VERSION}.linux-${PROM_ARCH}/prometheus" /usr/local/bin/
    $SUDO cp "/tmp/prometheus-${PROM_VERSION}.linux-${PROM_ARCH}/promtool" /usr/local/bin/
    rm -rf /tmp/prom.tgz "/tmp/prometheus-${PROM_VERSION}.linux-${PROM_ARCH}"
fi
log "prometheus found: $(prometheus --version 2>/dev/null | head -1 || echo unknown)"

# Grafana (dashboard frontend for make monitoring)
if ! command -v grafana-server >/dev/null 2>&1; then
    log "installing Grafana 11.4.0"
    case $(uname -m) in
        x86_64)  GRAFANA_ARCH="amd64" ;;
        aarch64) GRAFANA_ARCH="arm64" ;;
        *) log "FATAL: unsupported architecture $(uname -m) for Grafana"; exit 1 ;;
    esac
    curl -fsSL -o /tmp/grafana.deb "https://dl.grafana.com/oss/release/grafana_11.4.0_${GRAFANA_ARCH}.deb"
    $SUDO apt-get install -y -qq /tmp/grafana.deb
    rm -f /tmp/grafana.deb
fi
log "grafana found: $(grafana-server -v 2>/dev/null || echo unknown)"

log "host setup complete. Next: make calibrate A='sps' (downloads the model and generates the SPS table)"
