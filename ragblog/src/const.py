import os.path
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Loc:
    root: Path = Path(os.path.dirname(__file__)).parent
    data: Path = root / "data"
    vect_db: Path = data / "vect_db"
    eval_data: Path = data / "eval"
    eval_set: Path = eval_data / "eval_set.jsonl"
    results: Path = eval_data / "results"


class LLM(StrEnum):
    QWEN_2_5_14B = "qwen2.5:14b"
    GPT_OSS_20B = "unsloth/gpt-oss-20b-GGUF:Q8_0"
    QWEN_3_5_9B = "qwen3.5:9b"
    QWEN_3_5_27B = "qwen3.5:27b"
    QWEN_3_5_27B_Q3 = "unsloth/Qwen3.5-27B-GGUF:Q3_K_S"
    QWEN_3_5_9B_Q8 = "unsloth/Qwen3.5-9B-GGUF:Q8_0"
    QWEN_3_emb_8B = "qwen3-embedding:8b"


@dataclass(frozen=True)
class Api:
    """Backend inference server configuration.

    Hybrid setup: LLM via llama.cpp on 8080 (GPU),
    embeddings via Ollama on 11434 (CPU).
    """

    base_url: str = "http://127.0.0.1:8080/v1"
    emb_url: str = "http://127.0.0.1:11434/v1"
    api_key: str = "none"


@dataclass(frozen=True)
class Model:
    aug: LLM = LLM.GPT_OSS_20B
    emb: LLM = LLM.QWEN_3_emb_8B
    eval_set: LLM = LLM.QWEN_2_5_14B
    eval_aug: LLM = LLM.QWEN_2_5_14B


@dataclass(frozen=True)
class Eval:
    metric_list: List[str] = field(
        default_factory=lambda: [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        ]
    )
    testset_size: int = 16


@dataclass(frozen=True)
class Const:
    api = Api()
    loc = Loc()
    eval = Eval()
    model = Model()


CONST = Const()
