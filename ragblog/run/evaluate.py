#!/usr/bin/env python3
"""Evaluate a RAG configuration using Ragas."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.const import CONST
from src.evaluation.evaluator import RAGEvaluator, load_testset
from src.rag import Rag


def save_results(name: str, results: List[Dict[str, Any]]) -> None:
    """Save evaluation results."""
    CONST.loc.results.mkdir(parents=True, exist_ok=True)
    
    json_path = CONST.loc.results / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    md_path = CONST.loc.results / f"{name}.md"
    with open(md_path, "w") as f:
        f.write("# RAG Evaluation Results\n\n")
        f.write(f"Configuration: {name}\n\n")
        
        metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
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
        
        f.write("\nResults saved successfully.")

    print(f"Results saved to {json_path} and {md_path}")


def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate RAG configuration")
    parser.add_argument("--name", required=True, help="Name of the configuration (e.g., baseline)")
    args = parser.parse_args()

    print(f"Starting evaluation for: {args.name}")

    rag = Rag(is_ready_vector_db=True)
    evaluator = RAGEvaluator(rag)
    
    testset = load_testset()
    print(f"Evaluating {len(testset)} test cases...")
    
    results = evaluator.evaluate(testset)
    save_results(args.name, results)
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
