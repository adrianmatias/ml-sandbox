#!/usr/bin/env python3
"""Evaluate RAG system - thin orchestration script."""

import argparse
import json

from src.const import CONST
from src.evaluation.eval_set import EvalSet
from src.evaluation.evaluator import RagEval
from src.rag import Rag


def save_results(name: str, results: list[dict]) -> None:
    """Save evaluation results."""
    CONST.loc.results.mkdir(parents=True, exist_ok=True)

    json_path = CONST.loc.results / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    md_path = CONST.loc.results / f"{name}.md"
    with open(md_path, "w") as f:
        f.write("# RAG Evaluation Results\n\n")
        f.write(f"Configuration: {name}\n\n")

        metrics = CONST.eval.metric_list
        agg_scores = {m: [] for m in metrics}

        for result in results:
            for m in metrics:
                score = result.get(m, 0)
                if isinstance(score, (int, float)):
                    agg_scores[m].append(score)

        f.write("## Aggregate Scores\n\n")
        for m in metrics:
            scores = agg_scores[m]
            avg = sum(scores) / len(scores) if scores else 0
            f.write(f"- {m}: {avg:.3f}\n")

        f.write("\n✅ Evaluation completed successfully.")


def main():
    """High-level evaluation orchestration."""
    parser = argparse.ArgumentParser(description="Evaluate RAG configuration")
    parser.add_argument("--name", required=True, help="Name of the configuration")
    args = parser.parse_args()

    print(f"🚀 Starting evaluation: {args.name}")

    rag = Rag(is_ready_vector_db=True)
    testset = EvalSet()
    testset.load()

    evaluator = RagEval(rag)
    results = evaluator.evaluate(testset)

    save_results(args.name, results)
    print("✅ Evaluation pipeline completed!")


if __name__ == "__main__":
    main()
