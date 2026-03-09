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
    testset: Path = eval_data / "testset.jsonl"
    results: Path = eval_data / "results"


class LLM(StrEnum):
    QWEN_2_5_14B = "qwen2.5:14b"
    GPT_OSS_20B = "gpt-oss:20b"
    QWEN_3_5_9B = "qwen3.5:9b"
    QWEN_3_emb_8B = "qwen3-embedding:8b"


@dataclass(frozen=True)
class Model:
    rag: LLM = LLM.GPT_OSS_20B
    test_dataset: LLM = LLM.QWEN_2_5_14B
    embedding: LLM = LLM.QWEN_3_emb_8B


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
    testset_size: int = 8


@dataclass(frozen=True)
class Const:
    loc = Loc()
    eval = Eval()
    model = Model()


CONST = Const()
