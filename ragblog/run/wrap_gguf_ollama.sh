#!/usr/bin/env bash

# wrap_gguf_ollama.sh
# Usage: ./wrap_gguf_ollama.sh [model-name] [gguf-path]

MODEL_NAME=${1}
GGUF_PATH=${2}

MODEL_DIR=models/$MODEL_NAME
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

ln -sf "$(realpath "$GGUF_PATH")" model.gguf

cat > Modelfile << 'EOF'
FROM model.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
EOF

ollama create "$MODEL_NAME" -f Modelfile
ollama run "$MODEL_NAME" "hey, man!"