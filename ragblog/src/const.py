import os.path
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class Eval:
    metrics: List[str] = field(
        default_factory=lambda: [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        ]
    )
    default_testset_size: int = 8
    default_llm_model: str = "qwen2.5:14b"
    default_embedding_model: str = "qwen3-embedding:8b"


@dataclass(frozen=True)
class Const:
    loc = Loc()
    eval = Eval()


CONST = Const()
