"""Model comparison harness for RAG evaluation across multiple LLMs.

Orchestrates sequential evaluation of different augmentation models,
managing server lifecycle to maintain single-model VRAM residency.

Key insight: Separate response generation (llama-server) from evaluation
(ollama) to avoid loading both in VRAM simultaneously.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.const import CONST, LLM
from src.evaluation.eval_set import EvalSet
from src.evaluation.rag_eval import PrecomputedResponse, RagEval
from src.llama_server_manager import LlamaServerManager
from src.logger_custom import LOGGER, log_init
from src.rag import Rag


@dataclass
class ComparisonResult:
    """Aggregated results for a single model evaluation."""

    model: LLM
    result_list: List[Dict]
    score_map: Dict[str, float] = field(default_factory=dict)


@log_init
class ModelComparator:
    """Compares RAG performance across multiple augmentation models.

    Two-phase approach to minimize VRAM usage:
    1. Generate responses (llama-server only, ~11GB)
    2. Evaluate responses (ollama only, ~4GB)
    """

    def __init__(
        self,
        eval_set: EvalSet,
        server_manager: LlamaServerManager,
        result_dir: Optional[Path] = None,
    ):
        self.eval_set = eval_set
        self.server_manager = server_manager
        self.result_dir = result_dir or Path("data/eval/comparisons")
        self.comparison_map: Dict[LLM, ComparisonResult] = {}
        self.response_map: Dict[LLM, List[PrecomputedResponse]] = {}

    async def compare_all(self, model_list: List[LLM]) -> Dict[LLM, ComparisonResult]:
        """Two-phase evaluation: generate responses, then evaluate.

        Phase 1: Generate responses for all models (llama-server only)
        Phase 2: Evaluate all responses (ollama only)

        Args:
            model_list: List of LLM enums to evaluate.

        Returns:
            Map of model to comparison results.
        """
        self.comparison_map = {}
        self.response_map = {}

        # Phase 1: Generate responses for all models
        LOGGER.info("=== Phase 1: Generating responses ===")
        for model in model_list:
            LOGGER.info(f"Generating responses for: {model}")
            responses = await self.generate_responses(model)
            self.response_map[model] = responses

        # Stop llama-server before evaluation
        self.server_manager.stop()
        LOGGER.info("llama-server stopped, VRAM freed for evaluation")

        # Phase 2: Evaluate all responses
        LOGGER.info("=== Phase 2: Evaluating responses ===")
        for model, responses in self.response_map.items():
            LOGGER.info(f"Evaluating responses for: {model}")
            result = await self.evaluate_responses(model, responses)
            self.comparison_map[model] = result

        return self.comparison_map

    async def generate_responses(self, model: LLM) -> List[PrecomputedResponse]:
        """Generate RAG responses using llama-server.

        Args:
            model: LLM enum for augmentation.

        Returns:
            List of PrecomputedResponse objects.
        """
        try:
            self.server_manager.start(model)
            rag = Rag(aug=model)
            item_list = self.eval_set.to_item_list()

            responses = []
            for item in item_list:
                response = rag.query(item.question)
                contexts = rag.get_contexts(item.question)
                responses.append(
                    PrecomputedResponse(
                        user_input=item.question,
                        response=response,
                        retrieved_contexts=contexts,
                        reference=item.ground_truth,
                    )
                )
                LOGGER.info(f"Generated response {len(responses)}/{len(item_list)}")

            return responses
        except Exception as e:
            LOGGER.error(f"Response generation failed for {model}: {e}")
            return []
        finally:
            # Don't stop here - let compare_all manage lifecycle
            pass

    async def evaluate_responses(
        self, model: LLM, responses: List[PrecomputedResponse]
    ) -> ComparisonResult:
        """Evaluate pre-generated responses using ollama.

        Args:
            model: LLM enum for identification.
            responses: Pre-generated PrecomputedResponse objects.

        Returns:
            ComparisonResult with metric scores.
        """
        if not responses:
            return ComparisonResult(model=model, result_list=[], score_map={})

        try:
            rag = Rag(aug=model)
            result_list = await RagEval(rag).evaluate_from_responses(responses)
            score_map = self.aggregate_scores(result_list)

            return ComparisonResult(
                model=model,
                result_list=result_list,
                score_map=score_map,
            )
        except Exception as e:
            LOGGER.error(f"Evaluation failed for {model}: {e}")
            return ComparisonResult(model=model, result_list=[], score_map={})

    def aggregate_scores(self, result_list: List[Dict]) -> Dict[str, float]:
        """Compute mean scores across evaluation metrics.

        Args:
            result_list: List of per-row evaluation results.

        Returns:
            Map of metric name to mean score.
        """
        score_map = {}
        for metric in CONST.eval.metric_list:
            scores = [
                r[metric]
                for r in result_list
                if isinstance(r.get(metric), (int, float))
            ]
            score_map[metric] = sum(scores) / len(scores) if scores else 0.0
        return score_map

    def save_results(self) -> None:
        """Persist comparison results to disk.

        Writes individual JSON files per model and summary CSV.
        """
        self.result_dir.mkdir(parents=True, exist_ok=True)

        for model, result in self.comparison_map.items():
            model_name = str(model).replace("/", "_").replace(":", "_")
            json_path = self.result_dir / f"{model_name}.json"
            with open(json_path, "w") as f:
                json.dump(result.result_list, f, indent=2, ensure_ascii=False)
            LOGGER.info(f"Saved results: {json_path}")

        self.save_comparison_table()

    def save_comparison_table(self) -> None:
        """Generate and save comparison DataFrame as CSV."""
        df = self.to_dataframe()
        csv_path = self.result_dir / "comparison_table.csv"
        df.to_csv(csv_path)
        LOGGER.info(f"Saved comparison table: {csv_path}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert comparison results to pandas DataFrame.

        Returns:
            DataFrame with models as rows, metrics as columns.
        """
        row_list = []
        for model, result in self.comparison_map.items():
            row = {"model": str(model)}
            row.update(result.score_map)
            row_list.append(row)

        return pd.DataFrame(row_list).set_index("model")

    def print_summary(self) -> None:
        """Print formatted comparison summary to console."""
        df = self.to_dataframe()
        print("\n📊 Model Comparison Summary")
        print("=" * 60)
        print(df.to_string())
        print("=" * 60)
