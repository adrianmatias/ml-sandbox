import json
import os
from statistics import mean, stdev
from typing import Any, Dict, List

from ragblog.conf import CONF
from ragblog.logger_custom import LoggerCustom
from ragblog.rag_pipeline import (
    JSONLoaderConf,
    RAGChainConf,
    RagEvaluatorConf,
    RagPipeline,
    RagPipelineConf,
    TextSplitterConf,
    VectorStoreConf,
)


def load_test_queries() -> List[str]:
    """Load a small set of test queries."""
    return [
        "What is the author's experience with travel?",
        "Describe the relation between Helena and Alejandra.",
        "What are the author's thoughts on technology?",
        "How does the author balance work and personal life?",
        "What are some key life lessons from the blog?",
    ]


def evaluate_retrieval(evaluator, question: str, docs: List[Any]) -> Dict[str, Any]:
    """Evaluate the quality of retrieved documents."""
    # For each doc, use LLM to judge relevance to question
    relevance_scores = []
    for doc in docs:
        prompt = f"""On a scale of 1-5, how relevant is the following document
to the question: '{question}'?

Document: {doc.page_content[:500]}...

Provide a score (1-5) and a brief explanation."""
        response = evaluator.llm.invoke(prompt)
        # Parse score (simplified)
        score = 3.0  # default
        for line in response.split("\n"):
            if "score" in line.lower():
                import re

                match = re.search(r"(\d+(\.\d+)?)", line)
                if match:
                    score = float(match.group(1))
                    break
        relevance_scores.append(score)

    avg_relevance = mean(relevance_scores) if relevance_scores else 0
    precision_at_k = (
        sum(1 for s in relevance_scores if s >= 3) / len(relevance_scores)
        if relevance_scores
        else 0
    )

    return {
        "avg_relevance": avg_relevance,
        "precision_at_k": precision_at_k,
        "individual_scores": relevance_scores,
    }


def run_batch_evaluation(num_runs: int = 3) -> Dict[str, Any]:
    """Run batch evaluation on test queries, averaging over multiple runs."""
    logger = LoggerCustom().get_logger()

    rp_conf = RagPipelineConf(
        loader=JSONLoaderConf(
            file_path=os.path.join(CONF.path.data, "blog.jsonl"),
            jq_schema=".text",
            text_content=False,
            json_lines=True,
        ),
        splitter=TextSplitterConf(chunk_size=1000, chunk_overlap=10),
        vectorstore=VectorStoreConf(
            embedding_model="qwen3-embedding:8b",
            persist_directory=CONF.path.chroma,
        ),
        ragchain=RAGChainConf(
            prompt_model="rlm/rag-prompt-llama", llm_model="gpt-oss:20b"
        ),
        evaluator=RagEvaluatorConf(),
        is_db_ready=False,
        is_debug=False,
    )

    pipeline = RagPipeline(conf=rp_conf, logger=logger)
    queries = load_test_queries()

    results = {}
    for query in queries:
        query_results = {
            "retrieval_scores": [],
            "answer_scores": [],
            "overall_scores": [],
        }

        for run in range(num_runs):
            logger.info(f"Evaluating query: {query} (run {run + 1}/{num_runs})")

            # Get retriever and retrieve docs
            retriever = pipeline.get_vectorstore().as_retriever(k=10)
            docs = retriever.invoke(query)

            # Evaluate retrieval
            retrieval_eval = evaluate_retrieval(pipeline.evaluator, query, docs)
            query_results["retrieval_scores"].append(retrieval_eval)

            # Generate answer
            context = pipeline.format_docs(docs)
            # Note: Assuming query_with_context method exists
            response = pipeline.query_with_context(query, context)

            # Evaluate answer
            answer_eval = pipeline.evaluate_answer(query, response, context)
            query_results["answer_scores"].append(answer_eval)
            query_results["overall_scores"].append(answer_eval["overall_score"])

        # Aggregate results for this query
        results[query] = {
            "retrieval_avg_relevance": mean(
                [r["avg_relevance"] for r in query_results["retrieval_scores"]]
            ),
            "retrieval_precision_at_k": mean(
                [r["precision_at_k"] for r in query_results["retrieval_scores"]]
            ),
            "answer_overall_avg": mean(query_results["overall_scores"]),
            "answer_overall_std": stdev(query_results["overall_scores"])
            if len(query_results["overall_scores"]) > 1
            else 0,
            "runs": num_runs,
        }

    # Overall averages
    overall = {
        "avg_retrieval_relevance": mean(
            [r["retrieval_avg_relevance"] for r in results.values()]
        ),
        "avg_retrieval_precision": mean(
            [r["retrieval_precision_at_k"] for r in results.values()]
        ),
        "avg_answer_score": mean([r["answer_overall_avg"] for r in results.values()]),
        "std_answer_score": mean([r["answer_overall_std"] for r in results.values()]),
        "total_queries": len(queries),
        "runs_per_query": num_runs,
    }

    results["overall"] = overall
    return results


if __name__ == "__main__":
    results = run_batch_evaluation(num_runs=3)

    output_path = os.path.join(CONF.path.data, "batch_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Batch evaluation completed. Results saved to:", output_path)
    print("Overall averages:")
    print(f"  Retrieval Relevance: {results['overall']['avg_retrieval_relevance']:.2f}")
    print(
        f"  Retrieval Precision@10: {results['overall']['avg_retrieval_precision']:.2f}"
    )
    print(
        f"  Answer Quality: {results['overall']['avg_answer_score']:.2f} ± "
        f"{results['overall']['std_answer_score']:.2f}"
    )
