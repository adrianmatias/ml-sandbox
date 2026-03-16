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

# Models — LLM uses your cached 35B; EMB uses small nomic embed (auto-downloads via -hf)
#LLM_GGUF="${LLM_GGUF:-${HOME}/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-Q3_K_S.gguf}"
LLM_GGUF="${LLM_GGUF:-${HOME}/.cache/llama.cpp/unsloth_Qwen3.5-27B-GGUF_Qwen3.5-27B-Q3_K_S.gguf}"
EMB_MODEL="${EMB_MODEL:-nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M}"

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
Use LLM_GGUF=... or -hf manually."

info "Starting two-server setup: LLM (27B) on 8080 + nomic-embed on 8081"

# ---------------------------------------------------------------------------
# Start LLM server (port 8080) - 27B model, no emb flag
# ---------------------------------------------------------------------------
info "Starting LLM server on port ${LLM_PORT} (27B-Q3_K_S)..."
"${LLAMA_SERVER}" \
    --model "${LLM_GGUF}" \
    --port "${LLM_PORT}" \
    --host 127.0.0.1 \
    --n-gpu-layers 999 \
    --ctx-size 2048 \
    --log-disable \
    > /tmp/llama_server_llm.log 2>&1 &
echo $! > "${LLM_PID_FILE}"
info "  LLM server PID: $(cat ${LLM_PID_FILE})"

# ---------------------------------------------------------------------------
# Start embedding server (port 8081) - nomic-embed with default pooling
# ---------------------------------------------------------------------------
info "Starting embedding server on port ${EMB_PORT} (nomic-embed, CPU-only)..."
"${LLAMA_SERVER}" \
    -hf "${EMB_MODEL}" \
    --port "${EMB_PORT}" \
    --host 127.0.0.1 \
    --n-gpu-layers 0 \
    --ctx-size 512 \
    --log-disable \
    > /tmp/llama_server_emb.log 2>&1 &
echo $! > "${EMB_PID_FILE}"
info "  Embedding server PID: $(cat ${EMB_PID_FILE})"

# ---------------------------------------------------------------------------
# Wait for servers to be ready
# ---------------------------------------------------------------------------
info "Waiting for servers to become ready..."
for port in "${LLM_PORT}" "${EMB_PORT}"; do
    for i in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            info "  Port ${port}: ready"
            break
        fi
        sleep 1
        if [[ "${i}" -eq 30 ]]; then
            die "Server on port ${port} did not become ready in 30s."
        fi
    done
done

info ""
info "============================================================"
info "  Both servers running."
info ""
info "  LLM server (27B)  : http://127.0.0.1:${LLM_PORT}/v1"
info "  Emb server (CPU)  : http://127.0.0.1:${EMB_PORT}/v1"
info ""
info "  To stop: bash scripts/llama_server_start.sh stop"
info "============================================================"
info "  To stop: bash scripts/llama_server_start.sh stop"
info "============================================================"
info "  Server(s) running."
info ""
info "  LLM (+embeddings): http://127.0.0.1:${LLM_PORT}/v1"
if [[ -f "${EMB_PID_FILE}" ]]; then
    info "  Dedicated emb   : http://127.0.0.1:${EMB_PORT}/v1"
fi
info ""
info "  Using 35B-A3B model. Run with: bash scripts/llama_server_start.sh"
info "  To stop: bash scripts/llama_server_start.sh stop"
info "============================================================"
