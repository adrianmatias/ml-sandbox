"""Ollama embeddings implementation for LangChain."""

from typing import List

import requests
from langchain_core.embeddings import Embeddings

from src.const import CONST


class OllamaEmbeddings(Embeddings):
    """Ollama embeddings wrapper for LangChain.

    Uses Ollama's native /api/embeddings endpoint instead of
    OpenAI-compatible /v1/embeddings.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or CONST.model.emb
        self.base_url = base_url or "http://127.0.0.1:11434"
        # Remove /v1 suffix if present
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]

    def _embed(self, text: str) -> List[float]:
        """Embed a single text using Ollama API."""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        return self._embed(text)
