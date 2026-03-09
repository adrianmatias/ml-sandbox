import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.document_loaders import JSONLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

from src.const import CONST


@dataclass
class TestSetItem:
    """Represents a single test case from Ragas TestsetGenerator."""

    question: str
    ground_truth: str
    reference_contexts: List[str]
    persona_name: str = ""
    query_style: str = ""


class TestSet:
    """Encapsulates testset creation, loading, and management."""

    def __init__(self):
        self.data: List[Dict[str, Any]] = []
        self.testset_path: Path = CONST.loc.testset

    def generate(self, testset_size: int = None) -> None:
        """Generate synthetic testset using Ragas."""
        size = testset_size or CONST.eval.testset_size
        print("📚 Loading blog documents for testset generation...")

        loader = JSONLoader(
            file_path=CONST.loc.data / "blog.jsonl",
            jq_schema=".text",
            json_lines=True,
        )
        documents = loader.load()
        print(f"   Loaded {len(documents)} documents.")

        print("🤖 Setting up generation models...")
        generator_llm = LangchainLLMWrapper(
            ChatOllama(
                model=CONST.model.test_dataset,
                temperature=0.0,
            )
        )

        generator_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=CONST.model.embedding)
        )

        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
        )

        print(f"🔄 Generating testset with {size} samples...")
        ragas_testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=size,
        )

        df = ragas_testset.to_pandas()
        self.data = []

        for _, row in df.iterrows():
            self.data.append(
                {
                    "user_input": row.get("user_input", row.get("question", "")),
                    "reference": row.get("reference", row.get("ground_truth", "")),
                    "reference_contexts": row.get("reference_contexts", []),
                    "persona_name": row.get("persona_name", ""),
                    "query_style": row.get("query_style", ""),
                }
            )

        self.save()
        print(f"✅ Generated and saved {len(self.data)} test cases")

    def load(self) -> List[Dict[str, Any]]:
        if not self.testset_path.exists():
            raise FileNotFoundError(
                f"Testset not found at {self.testset_path}. Run generate() first."
            )

        self.data = []
        with open(self.testset_path, "r") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))

        print(f"Loaded {len(self.data)} test cases from {self.testset_path}")
        return self.data

    def save(self) -> None:
        CONST.loc.eval_data.mkdir(parents=True, exist_ok=True)
        with open(self.testset_path, "w") as f:
            for item in self.data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved testset to {self.testset_path}")

    def to_testset_items(self) -> List[TestSetItem]:
        return [
            TestSetItem(
                question=item.get("user_input") or item.get("question", ""),
                ground_truth=item.get("reference") or item.get("ground_truth", ""),
                reference_contexts=item.get("reference_contexts", []),
                persona_name=item.get("persona_name", ""),
                query_style=item.get("query_style", ""),
            )
            for item in self.data
        ]

    def __len__(self) -> int:
        return len(self.data)
