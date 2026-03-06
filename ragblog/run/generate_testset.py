#!/usr/bin/env python3
"""Generate a testset for RAG evaluation using Ragas – fixed for local Ollama."""

import json
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import JSONLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

from src.const import CONST


def main():
    """Generate synthetic testset and save it."""
    print("📚 Loading blog documents...")
    loader = JSONLoader(
        file_path=CONST.loc.data / "blog.jsonl",
        jq_schema=".text",
        json_lines=True,
    )
    documents = loader.load()
    print(f"   Loaded {len(documents)} documents.")

    # === Use modern wrappers + ChatOllama (this fixes the parser issue) ===
    print("🤖 Setting up generation models...")
    # Change this line to a stronger structured-output model if you want:
    # Recommended: qwen2.5:14b or qwen2.5:32b (much more reliable than gpt-oss:20b)
    generator_llm = LangchainLLMWrapper(
        ChatOllama(
            model="qwen2.5:14b",
            temperature=0.0,
        )
    )

    generator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="qwen3-embedding:8b")
    )

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )

    print("🔄 Generating testset (5–15 min depending on model)...")
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=3,
    )

    # Save
    CONST.loc.eval_data.mkdir(parents=True, exist_ok=True)
    output_path = CONST.loc.testset

    # Clean JSONL save (Ragas-native method)
    testset.to_pandas().to_json(output_path, orient="records", lines=True, force_ascii=False)

    print(f"✅ Testset generated and saved to:\n   {output_path}")
    print(f"   → {len(testset)} high-quality QA pairs ready for evaluation")


if __name__ == "__main__":
    main()