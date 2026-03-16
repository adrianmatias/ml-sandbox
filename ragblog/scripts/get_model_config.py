#!/usr/bin/env python3
"""Helper script to extract model configuration from const.py for server scripts."""

import json
import sys

sys.path.insert(0, "/home/mat/PycharmProjects/ml-sandbox/ragblog")

from src.const import CONST, LLM


def get_model_config():
    """Get model configuration for server startup."""
    aug_model = CONST.model.aug
    emb_model = CONST.model.emb

    # Map LLM enum to actual model identifiers
    model_map = {
        LLM.QWEN_3_5_9B_Q4: {
            "hf_repo": "unsloth/Qwen3.5-9B-GGUF",
            "quant": "Q4_K_M",
            "gguf_name": "unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-Q4_K_M.gguf",
        },
        LLM.QWEN_3_5_27B_Q3: {
            "hf_repo": "unsloth/Qwen3.5-27B-GGUF",
            "quant": "Q3_K_S",
            "gguf_name": "unsloth_Qwen3.5-27B-GGUF_Qwen3.5-27B-Q3_K_S.gguf",
        },
    }

    aug_config = model_map.get(
        aug_model, {"hf_repo": str(aug_model), "quant": "", "gguf_name": ""}
    )

    return {
        "aug": {
            "model": str(aug_model),
            "hf_repo": aug_config["hf_repo"],
            "quant": aug_config["quant"],
            "gguf_name": aug_config["gguf_name"],
            "port": 8080,
        },
        "emb": {
            "model": str(emb_model),
            "port": 11434,  # Ollama default
        },
        "api": {
            "base_url": CONST.api.base_url,
            "emb_url": CONST.api.emb_url,
        },
    }


if __name__ == "__main__":
    config = get_model_config()
    print(json.dumps(config, indent=2))
