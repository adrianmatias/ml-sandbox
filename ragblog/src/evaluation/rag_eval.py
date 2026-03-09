import asyncio
from typing import Any, Dict, List

from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas.backends import InMemoryBackend
from ragas.dataset import Dataset
from ragas.embeddings.base import embedding_factory
from ragas.experiment import experiment
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from src.const import CONST
from src.evaluation.eval_set import EvalSet
from src.rag import Rag


class EvalResult(BaseModel):
    """Evaluation result for a single test case."""

    user_input: str
    response: str
    retrieved_contexts: List[str]
    reference: str
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0


class RagEval:
    """Evaluates a Rag instance against a EvalSet.

    Manifests clear separation between the evaluated RAG system (Rag)
    and the evaluation harness (RagEval), each using dedicated models
    from CONST.model.rag and CONST.model.eval respectively.
    """

    def __init__(self, rag: Rag):
        self.rag = rag

        client = AsyncOpenAI(
            base_url=CONST.api.ollama_base_url,
            api_key=CONST.api.ollama_api_key,
        )
        llm = llm_factory(CONST.model.eval, client=client)
        emb = embedding_factory("openai", CONST.model.embedding, client=client)

        self.metrics = {
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
        }

    def evaluate(self, eval_set: EvalSet) -> List[Dict[str, Any]]:
        """Evaluate the RAG pipeline against the eval_set.

        Args:
            eval_set: EvalSet with evaluation questions and ground truths.

        Returns:
            List of per-row metric scores as dicts.
        """
        return asyncio.run(self._aevaluate(eval_set))

    async def _aevaluate(self, eval_set: EvalSet) -> List[Dict[str, Any]]:
        """Async evaluation using the @experiment decorator per-row pattern."""
        rag = self.rag
        metrics = self.metrics

        @experiment(EvalResult)
        async def run_row(row: dict) -> EvalResult:
            user_input = row["user_input"]
            reference = row["reference"]
            response = rag.query(user_input)
            contexts = rag.get_contexts(user_input)

            cp = await metrics["context_precision"].ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=contexts,
            )
            cr = await metrics["context_recall"].ascore(
                user_input=user_input,
                retrieved_contexts=contexts,
                reference=reference,
            )
            faith = await metrics["faithfulness"].ascore(
                user_input=user_input,
                response=response,
                retrieved_contexts=contexts,
            )
            ar = await metrics["answer_relevancy"].ascore(
                user_input=user_input,
                response=response,
            )

            return EvalResult(
                user_input=user_input,
                response=response,
                retrieved_contexts=contexts,
                reference=reference,
                context_precision=cp.value,
                context_recall=cr.value,
                faithfulness=faith.value,
                answer_relevancy=ar.value,
            )

        item_list = eval_set.to_item_list()
        data = [
            {"user_input": item.question, "reference": item.ground_truth}
            for item in item_list
        ]

        backend = InMemoryBackend()
        dataset = Dataset("ragblog-eval_set", backend=backend, data=data)

        print(f"Evaluating {len(eval_set)} test cases...")
        result_exp = await run_row.arun(
            dataset,
            name="ragblog-eval",
            backend=InMemoryBackend(),
        )

        return [row.model_dump() for row in result_exp]
