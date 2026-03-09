from typing import Any, Dict, List

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from src.const import CONST
from src.evaluation.testset import TestSet
from src.rag import Rag


class RAGEvaluator:
    """RAG evaluator using TestSet abstraction - clean OO design."""

    def __init__(self, rag: Rag):
        self.rag = rag
        self.llm = ChatOllama(
            model=CONST.eval.default_llm_model,
            temperature=0.0,
        )
        self.llm_wrapper = LangchainLLMWrapper(self.llm)
        self.embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=CONST.eval.default_embedding_model)
        )
        self.metrics = [
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(llm=self.llm_wrapper),
            AnswerRelevancy(llm=self.llm_wrapper),
        ]

    def evaluate(self, testset: TestSet) -> List[Dict[str, Any]]:
        """Evaluate using TestSet object."""
        dataset_dict = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        retriever = self.rag.vector_db.as_retriever(k=10)
        test_items = testset.to_testset_items()

        print(f"Evaluating {len(testset)} test cases using TestSet abstraction...")

        for i, item in enumerate(test_items):
            print(f"  [{i + 1}/{len(testset)}] {item.question[:65]}...")

            answer = self.rag.query(item.question)
            docs = retriever.invoke(item.question)
            contexts = [doc.page_content for doc in docs]

            dataset_dict["question"].append(item.question)
            dataset_dict["answer"].append(answer)
            dataset_dict["contexts"].append(contexts)
            dataset_dict["ground_truth"].append(item.ground_truth)

        dataset = Dataset.from_dict(dataset_dict)

        print("Running Ragas evaluation...")
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm_wrapper,
            embeddings=self.embeddings,
        )

        return result.to_pandas().to_dict("records")
