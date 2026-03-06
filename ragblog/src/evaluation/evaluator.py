import json
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from src.const import CONST
from src.rag import Rag


@dataclass
class TestsetItem:
    """Represents a test case from native Ragas TestsetGenerator output."""

    question: str
    ground_truth: str
    reference_contexts: List[str]
    persona_name: str = ""
    query_style: str = ""


class RAGEvaluator:
    """Clean OO evaluator following project patterns from src/rag.py etc."""

    def __init__(self, rag: Rag):
        """Initialize with RAG instance."""
        self.rag = rag

        self.llm = ChatOllama(
            model="qwen2.5:14b",
            temperature=0.0,
        )
        self.llm_wrapper = LangchainLLMWrapper(self.llm)

        self.embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model="qwen3-embedding:8b")
        )

        self.metrics = [
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(llm=self.llm_wrapper),
            AnswerRelevancy(llm=self.llm_wrapper),
        ]

    def _to_test_item(self, raw_item: Dict[str, Any]) -> TestsetItem:
        """Convert native Ragas format to structured TestsetItem."""
        return TestsetItem(
            question=raw_item.get("user_input") or raw_item.get("question", ""),
            ground_truth=raw_item.get("reference") or raw_item.get("ground_truth", ""),
            reference_contexts=raw_item.get("reference_contexts", []),
            persona_name=raw_item.get("persona_name", ""),
            query_style=raw_item.get("query_style", ""),
        )

    def evaluate(self, raw_testset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate using native Ragas testset format."""
        dataset_dict = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        retriever = self.rag.vector_db.as_retriever(k=10)
        print(f"Evaluating {len(raw_testset)} test cases...")

        for i, raw_item in enumerate(raw_testset):
            item = self._to_test_item(raw_item)
            print(f"  [{i+1}/{len(raw_testset)}] {item.question[:60]}...")

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


def load_testset() -> List[Dict[str, Any]]:
    """Load native Ragas testset format."""
    testset_path = CONST.loc.testset
    if not testset_path.exists():
        raise FileNotFoundError(f"Testset not found at {testset_path}. Run generate_testset.py first.")

    testset = []
    with open(testset_path, "r") as f:
        for line in f:
            if line.strip():
                testset.append(json.loads(line.strip()))

    print(f"Loaded {len(testset)} test cases from native Ragas format")
    return testset
