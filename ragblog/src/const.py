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
    GPT_OSS_20B = "gpt-oss:20b"
    QWEN_3_5_9B = "qwen3.5:9b"
    QWEN_3_5_27B = "qwen3.5:27b"
    QWEN_3_5_27B_Q3 = (
        "qwen3.5:27b-q3_K_M"  # requantized; see scripts/quantize_27b_q3.sh
    )
    QWEN_3_emb_8B = "qwen3-embedding:8b"


@dataclass(frozen=True)
class Api:
    """Backend inference server configuration.

    Both Ollama and llama-server expose an OpenAI-compatible /v1 endpoint,
    so only the base_url differs between providers.

    Ollama  (default): base_url="http://localhost:11434/v1", api_key="ollama"
    llama-server:      base_url="http://localhost:8080/v1",  api_key="none"

    To switch providers change the two fields below; no other file needs editing.
    See scripts/llama_server_start.sh for the llama-server launch command.
    """

    base_url: str = "http://localhost:8080/v1"
    api_key: str = "none"


@dataclass(frozen=True)
class Model:
    aug: LLM = (
        LLM.QWEN_3_5_27B_Q3
    )  # Q3_K_M fits fully in 16 GB VRAM; see scripts/quantize_27b_q3.sh
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
