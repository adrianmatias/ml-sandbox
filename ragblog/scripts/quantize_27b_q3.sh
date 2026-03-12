#!/usr/bin/env bash
# =============================================================================
# scripts/quantize_27b_q3.sh
#
# Builds llama-quantize from llama.cpp source, requantizes qwen3.5:27b from
# Q4_K_M (~17 GB) down to Q3_K_M (~13.4 GB), and registers the result as a
# new Ollama model "qwen3.5:27b-q3_K_M".
#
# Why this exists
# ---------------
# The RTX 5060 Ti has 16 GB VRAM.  qwen3.5:27b Q4_K_M is 17 GB of weights
# alone; Ollama also needs ~3 GB for the KV cache and compute graph, forcing
# 19/65 layers onto CPU and making generation very slow.
# Q3_K_M brings weights to ~13.4 GB, leaving ~2.6 GB for KV + compute graph
# and allowing all 65 layers to live on GPU.
#
# Usage
# -----
#   bash scripts/quantize_27b_q3.sh
#
# Requirements
# ------------
#   - cmake       (sudo apt install cmake)
#   - build-essential (sudo apt install build-essential)
#   - ollama      (must be installed and the qwen3.5:27b model must be pulled)
#   - ~20 GB free disk space (build dir + output GGUF)
#
# After this script completes, use the model in Python:
#   from src.const import LLM
#   rag = Rag(aug=LLM.QWEN_3_5_27B_Q3)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLAMA_CPP_DIR="${HOME}/llama.cpp"
LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_DIR}/build"
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"

# Source GGUF blob: the Q4_K_M file that ollama pull puts in the blobs dir.
# This digest comes from ollama show qwen3.5:27b --modelfile.
SOURCE_GGUF="${HOME}/.ollama/models/blobs/sha256-d4b8b4f4c350f5d322dc8235175eeae02d32c6f3fd70bdb9ea481e3abb7d7fc4"

OUTPUT_GGUF="${HOME}/.ollama/models/blobs/qwen3.5-27b-q3_K_M.gguf"
MODELFILE_PATH="/tmp/Modelfile.qwen3.5-27b-q3"
OLLAMA_MODEL_NAME="qwen3.5:27b-q3_K_M"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "'$1' not found. Install with: $2"
}

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
info "Checking prerequisites..."
require_cmd ollama "https://ollama.com/download"
require_cmd gcc    "sudo apt install build-essential"
require_cmd make   "sudo apt install build-essential"

# cmake check with install hint
if ! command -v cmake >/dev/null 2>&1; then
    info "cmake not found — installing via apt..."
    sudo apt-get install -y cmake
fi
require_cmd cmake "sudo apt install cmake"

if [[ ! -f "${SOURCE_GGUF}" ]]; then
    die "Source GGUF not found at ${SOURCE_GGUF}.
Run 'ollama pull qwen3.5:27b' first, then re-run this script."
fi

SOURCE_SIZE_GB=$(du -BG "${SOURCE_GGUF}" | awk '{print $1}')
info "Source GGUF: ${SOURCE_GGUF} (${SOURCE_SIZE_GB})"

# ---------------------------------------------------------------------------
# Step 1 — Clone or update llama.cpp
# ---------------------------------------------------------------------------
info "Step 1/4 — Setting up llama.cpp source..."
if [[ -d "${LLAMA_CPP_DIR}/.git" ]]; then
    info "  llama.cpp already cloned at ${LLAMA_CPP_DIR}, pulling latest..."
    git -C "${LLAMA_CPP_DIR}" pull --ff-only
else
    info "  Cloning llama.cpp into ${LLAMA_CPP_DIR}..."
    git clone --depth=1 "${LLAMA_CPP_REPO}" "${LLAMA_CPP_DIR}"
fi

# ---------------------------------------------------------------------------
# Step 2 — Build llama-quantize (CPU-only; no CUDA needed for quantization)
# ---------------------------------------------------------------------------
info "Step 2/4 — Building llama-quantize (CPU only, ~3-5 min)..."
mkdir -p "${LLAMA_CPP_BUILD_DIR}"
cmake -S "${LLAMA_CPP_DIR}" \
      -B "${LLAMA_CPP_BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_QUANTIZE=ON \
      2>&1 | tail -5

# Build only the quantize target to keep it fast
cmake --build "${LLAMA_CPP_BUILD_DIR}" \
      --target llama-quantize \
      --config Release \
      -j "$(nproc)" \
      2>&1 | tail -10

QUANTIZE_BIN="${LLAMA_CPP_BUILD_DIR}/bin/llama-quantize"
[[ -x "${QUANTIZE_BIN}" ]] || die "Build succeeded but ${QUANTIZE_BIN} not found."
info "  Built: ${QUANTIZE_BIN}"

# ---------------------------------------------------------------------------
# Step 3 — Requantize Q4_K_M -> Q3_K_M
# ---------------------------------------------------------------------------
info "Step 3/4 — Requantizing to Q3_K_M (~5-10 min, CPU-bound)..."
info "  Input : ${SOURCE_GGUF}"
info "  Output: ${OUTPUT_GGUF}"

# Q3_K_M type integer = 15  (see llama.cpp/src/llama.cpp ggml_type enum)
"${QUANTIZE_BIN}" \
    "${SOURCE_GGUF}" \
    "${OUTPUT_GGUF}" \
    Q3_K_M

OUTPUT_SIZE_GB=$(du -BG "${OUTPUT_GGUF}" | awk '{print $1}')
info "  Output size: ${OUTPUT_SIZE_GB} (expected ~13-14 GB)"

# ---------------------------------------------------------------------------
# Step 4 — Create Ollama model from Modelfile
# ---------------------------------------------------------------------------
info "Step 4/4 — Registering '${OLLAMA_MODEL_NAME}' with Ollama..."

cat > "${MODELFILE_PATH}" <<'MODELFILE'
# Modelfile for qwen3.5:27b-q3_K_M
# Requantized from Q4_K_M to Q3_K_M using llama-quantize to fit in 16 GB VRAM.
# Original model: qwen3.5:27b (Alibaba, Apache 2.0)
FROM __OUTPUT_GGUF__
TEMPLATE {{ .Prompt }}
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER presence_penalty 1.5
PARAMETER temperature 1
PARAMETER top_k 20
PARAMETER top_p 0.95
MODELFILE

# Substitute the actual output path into the Modelfile
sed -i "s|__OUTPUT_GGUF__|${OUTPUT_GGUF}|g" "${MODELFILE_PATH}"

info "  Modelfile written to ${MODELFILE_PATH}"
ollama create "${OLLAMA_MODEL_NAME}" -f "${MODELFILE_PATH}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
info ""
info "============================================================"
info "  Model registered: ${OLLAMA_MODEL_NAME}"
info "  VRAM breakdown (expected with 16 GB GPU):"
info "    Weights on GPU : ~12.5 GiB  (all 65 layers)"
info "    KV cache       :  ~0.3 GiB  (q4_0, num_ctx=2048)"
info "    Compute graph  :  ~0.8 GiB"
info "    Total          : ~13.6 GiB  -> fits in 15 GiB available"
info ""
info "  Use in Python:"
info "    from src.const import LLM"
info "    rag = Rag(aug=LLM.QWEN_3_5_27B_Q3)"
info ""
info "  Or set as default in src/const.py:"
info "    aug: LLM = LLM.QWEN_3_5_27B_Q3"
info "============================================================"
