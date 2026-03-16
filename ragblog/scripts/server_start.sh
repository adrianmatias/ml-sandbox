#!/usr/bin/env bash
# =============================================================================
# scripts/server_start.sh
#
# Starts llama-server (llama.cpp HTTP server) for LLM and ensures
# Ollama is running for embeddings. Reads configuration from const.py.
#
# Why llama-server instead of Ollama for LLM
# -------------------------------------------
# llama-server gives precise control over GPU layer allocation via
# --n-gpu-layers, making it trivial to fit large models in limited VRAM.
# It also exposes exactly the same OpenAI-compatible /v1 API.
#
# Hybrid setup
# -------------
# - LLM (port 8080): llama.cpp server with GPU acceleration
# - Embeddings (port 11434): Ollama server (CPU or GPU)
#
# Prerequisites
# -------------
#   1. Build llama.cpp with CUDA support:
#        bash scripts/build_llama_server.sh
#
#   2. Ensure Ollama is installed and running:
#        systemctl status ollama
#
#   3. Pull required models in Ollama:
#        ollama pull qwen3-embedding:8b
#
# Usage
# -----
#   # Start LLM server (background):
#   bash scripts/server_start.sh
#
#   # Stop LLM server:
#   bash scripts/server_start.sh stop
#
#   # Check status:
#   bash scripts/server_start.sh status
# =============================================================================

set -euo pipefail

# Get configuration from const.py
CONFIG=$(cd /home/mat/PycharmProjects/ml-sandbox/ragblog && uv run python scripts/get_model_config.py)
AUG_HF_REPO=$(echo "${CONFIG}" | python3 -c "import sys, json; print(json.load(sys.stdin)['aug']['hf_repo'])")
AUG_QUANT=$(echo "${CONFIG}" | python3 -c "import sys, json; print(json.load(sys.stdin)['aug']['quant'])")
AUG_PORT=$(echo "${CONFIG}" | python3 -c "import sys, json; print(json.load(sys.stdin)['aug']['port'])")

LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-${HOME}/llama.cpp/build}"
LLAMA_SERVER="${LLAMA_BUILD_DIR}/bin/llama-server"

LLM_PID_FILE="/tmp/llama_server_llm.pid"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info() { echo "[INFO]  $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

stop_server() {
    local pid_file="$1"
    local label="$2"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            info "Stopping ${label} (PID ${pid})..."
            kill "${pid}"
        fi
        rm -f "${pid_file}"
    else
        info "${label} not running (no pid file)"
    fi
}

check_ollama() {
    if curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
        info "Ollama server is running on port 11434"
        return 0
    else
        die "Ollama server not running on port 11434. Please start it first: systemctl start ollama"
    fi
}

# ---------------------------------------------------------------------------
# Status subcommand
# ---------------------------------------------------------------------------
if [[ "${1:-start}" == "status" ]]; then
    info "Checking server status..."
    
    # Check LLM server
    if [[ -f "${LLM_PID_FILE}" ]]; then
        pid=$(cat "${LLM_PID_FILE}")
        if kill -0 "${pid}" 2>/dev/null; then
            info "LLM server: RUNNING (PID ${pid}, port ${AUG_PORT})"
        else
            info "LLM server: STOPPED (stale pid file)"
        fi
    else
        info "LLM server: STOPPED"
    fi
    
    # Check Ollama
    if curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
        info "Ollama server: RUNNING (port 11434)"
    else
        info "Ollama server: STOPPED"
    fi
    
    exit 0
fi

# ---------------------------------------------------------------------------
# Stop subcommand
# ---------------------------------------------------------------------------
if [[ "${1:-start}" == "stop" ]]; then
    stop_server "${LLM_PID_FILE}" "LLM server"
    info "Note: Ollama server (if running) must be stopped separately: systemctl stop ollama"
    info "Done."
    exit 0
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
[[ -x "${LLAMA_SERVER}" ]] || die "llama-server not found at ${LLAMA_SERVER}.
Run: bash scripts/build_llama_server.sh"

check_ollama

info "Starting servers with configuration from const.py:"
info "  LLM: ${AUG_HF_REPO}:${AUG_QUANT} (port ${AUG_PORT})"
info "  Emb: Ollama on port 11434"

# ---------------------------------------------------------------------------
# Start LLM server (llama.cpp)
# ---------------------------------------------------------------------------
info "Starting LLM server on port ${AUG_PORT}..."
"${LLAMA_SERVER}" \
    -hf "${AUG_HF_REPO}:${AUG_QUANT}" \
    --port "${AUG_PORT}" \
    --host 127.0.0.1 \
    --ctx-size 32768 \
    --reasoning off \
    -ngl 999 \
    > /tmp/llama_server_llm.log 2>&1 &
echo $! > "${LLM_PID_FILE}"
info "  LLM server PID: $(cat ${LLM_PID_FILE})"

# ---------------------------------------------------------------------------
# Wait for server to be ready
# ---------------------------------------------------------------------------
info "Waiting for LLM server to become ready..."
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${AUG_PORT}/health" >/dev/null 2>&1; then
        info "  LLM server: ready"
        break
    fi
    sleep 2
    if [[ "${i}" -eq 60 ]]; then
        die "LLM server did not become ready in 120s. Check /tmp/llama_server_llm.log"
    fi
done

info ""
info "============================================================"
info "  Servers running."
info ""
info "  LLM (llama.cpp): http://127.0.0.1:${AUG_PORT}/v1"
info "  Emb (Ollama):    http://127.0.0.1:11434/v1"
info ""
info "  To stop: bash scripts/server_start.sh stop"
info "  To check status: bash scripts/server_start.sh status"
info "============================================================"
