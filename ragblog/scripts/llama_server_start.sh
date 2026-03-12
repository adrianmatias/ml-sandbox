#!/usr/bin/env bash
# =============================================================================
# scripts/llama_server_start.sh
#
# Starts llama-server (llama.cpp HTTP server) as an OpenAI-compatible backend
# for both the augmentation LLM and the embedding model.
#
# Why llama-server instead of Ollama
# ------------------------------------
# llama-server gives precise control over GPU layer allocation via
# --n-gpu-layers, making it trivial to fit large models in limited VRAM.
# It also exposes exactly the same OpenAI-compatible /v1 API that Ollama does,
# so the only change in the project is const.py:Api.base_url.
#
# Two-server setup
# -----------------
# The LLM (port 8080) and the embedding model (port 8081) run as separate
# processes so each can be loaded/unloaded independently.  After retrieval
# completes, you can stop the embedding server to free VRAM before the LLM
# generates its response — equivalent to what keep_alive=0 did for Ollama,
# but fully explicit.
#
# Prerequisites
# -------------
#   1. Build llama.cpp with CUDA support (or use the CPU-only build):
#        bash scripts/build_llama_server.sh
#
#   2. Run scripts/quantize_27b_q3.sh once to produce the Q3_K_M GGUF.
#
#   3. Set LLAMA_BUILD_DIR or accept the default (~/llama.cpp/build).
#
# Usage
# -----
#   # Start both servers (background):
#   bash scripts/llama_server_start.sh
#
#   # Stop both servers:
#   bash scripts/llama_server_start.sh stop
#
# GPU VRAM budget (RTX 5060 Ti, 16 GB)
# --------------------------------------
#   LLM  Q3_K_M 27B: weights ~13.4 GB + KV cache ~0.3 GB + graph ~0.8 GB
#   Emb  Q4_K_M  8B: weights  ~4.4 GB  (loaded only during retrieval)
#
#   -> Never both loaded at the same time with this two-server approach.
# =============================================================================

set -euo pipefail

LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-${HOME}/llama.cpp/build}"
LLAMA_SERVER="${LLAMA_BUILD_DIR}/bin/llama-server"

# GGUFs — defaults to the Ollama blob paths; override via env vars if needed
LLM_GGUF="${LLM_GGUF:-${HOME}/.ollama/models/blobs/qwen3.5-27b-q3_K_M.gguf}"
EMB_GGUF="${EMB_GGUF:-${HOME}/.ollama/models/blobs/sha256-3fcd3febec8b3fd64435204db75bf0dd73b91e8d0661e0331acfe7e7c3120b85}"

LLM_PORT=8080
EMB_PORT=8081

LLM_PID_FILE="/tmp/llama_server_llm.pid"
EMB_PID_FILE="/tmp/llama_server_emb.pid"

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

# ---------------------------------------------------------------------------
# Stop subcommand
# ---------------------------------------------------------------------------
if [[ "${1:-start}" == "stop" ]]; then
    stop_server "${LLM_PID_FILE}" "LLM server"
    stop_server "${EMB_PID_FILE}" "Embedding server"
    info "Done."
    exit 0
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
[[ -x "${LLAMA_SERVER}" ]] || die "llama-server not found at ${LLAMA_SERVER}.
Run: bash scripts/build_llama_server.sh"

[[ -f "${LLM_GGUF}" ]] || die "LLM GGUF not found: ${LLM_GGUF}.
Run: bash scripts/quantize_27b_q3.sh"

[[ -f "${EMB_GGUF}" ]] || die "Embedding GGUF not found: ${EMB_GGUF}.
Check that ollama pull qwen3-embedding:8b has been run."

# ---------------------------------------------------------------------------
# Start embedding server (port 8081)
# Load first, use during retrieval, then stop before starting the LLM.
# Run with --embedding flag to enable the /v1/embeddings endpoint.
# Only a few layers need GPU for the 8B model; use --n-gpu-layers 999 to
# offload all of them.
# ---------------------------------------------------------------------------
info "Starting embedding server on port ${EMB_PORT}..."
"${LLAMA_SERVER}" \
    --model "${EMB_GGUF}" \
    --port "${EMB_PORT}" \
    --host 127.0.0.1 \
    --n-gpu-layers 999 \
    --ctx-size 512 \
    --embedding \
    --log-disable \
    > /tmp/llama_server_emb.log 2>&1 &
echo $! > "${EMB_PID_FILE}"
info "  Embedding server PID: $(cat ${EMB_PID_FILE})  log: /tmp/llama_server_emb.log"

# ---------------------------------------------------------------------------
# Start LLM server (port 8080)
# --n-gpu-layers 999  → offload all layers (Q3_K_M fits in 16 GB after emb
#                       is stopped).
# --ctx-size 2048     → matches num_ctx in rag.py.
# --no-mmap           → avoid page-cache competition with emb server.
# ---------------------------------------------------------------------------
info "Starting LLM server on port ${LLM_PORT}..."
"${LLAMA_SERVER}" \
    --model "${LLM_GGUF}" \
    --port "${LLM_PORT}" \
    --host 127.0.0.1 \
    --n-gpu-layers 999 \
    --ctx-size 2048 \
    --no-mmap \
    --log-disable \
    > /tmp/llama_server_llm.log 2>&1 &
echo $! > "${LLM_PID_FILE}"
info "  LLM server PID: $(cat ${LLM_PID_FILE})  log: /tmp/llama_server_llm.log"

# ---------------------------------------------------------------------------
# Wait for both servers to be ready
# ---------------------------------------------------------------------------
info "Waiting for servers to become ready..."
for port in "${EMB_PORT}" "${LLM_PORT}"; do
    for i in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            info "  Port ${port}: ready"
            break
        fi
        sleep 1
        if [[ "${i}" -eq 30 ]]; then
            die "Server on port ${port} did not become ready in 30s. Check logs."
        fi
    done
done

info ""
info "============================================================"
info "  Both servers running."
info ""
info "  LLM server  : http://127.0.0.1:${LLM_PORT}/v1"
info "  Emb server  : http://127.0.0.1:${EMB_PORT}/v1"
info ""
info "  NOTE: const.py Api.base_url currently points to a single"
info "  endpoint.  For the two-server setup you have two options:"
info ""
info "  Option A — single server, sequential loading (simplest):"
info "    Use port 8080 for both LLM and embeddings."
info "    llama-server handles one model at a time."
info ""
info "  Option B — two servers, dedicated ports:"
info "    Set Api.base_url  = 'http://127.0.0.1:8080/v1'  (LLM)"
info "    Add Api.emb_url   = 'http://127.0.0.1:8081/v1'  (embeddings)"
info "    Update VectorDB._make_embeddings() to use CONST.api.emb_url."
info ""
info "  To stop:  bash scripts/llama_server_start.sh stop"
info "============================================================"
