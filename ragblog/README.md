# ragblog

Retrieval Augmented Generation (RAG) system for semantic search and Q&A over blog content.

## Objective

Enable intelligent question answering over blog archives by combining semantic search with local LLM inference. Crawls blog content, indexes it with vector embeddings, and answers natural language queries using retrieved context.

## Main Functionality

- **Crawler**: Scrapes blog posts from Blogger sites, extracts titles and content
- **Document Processing**: Loads JSONL, splits into chunks, generates embeddings
- **Vector Store**: Persists documents in ChromaDB with Ollama embeddings (`nomic-embed-text`)
- **RAG Pipeline**: Retrieves relevant chunks, augments LLM context, generates answers via local Ollama (`llama3.1`)

## Approach

1. Crawl → Extract URLs → Parse HTML → Save as JSONL
2. Load → Split (1000 chars, 10 overlap) → Embed → Store in Chroma
3. Query → Retrieve top-k chunks → Prompt LLM with context → Return answer

Uses dataclass-based configuration, modular pipeline components, and functional patterns.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## CI

```bash
sh ci/lint.sh   # ruff check && ruff format
sh ci/test.sh   # pytest -vv
```

## Run

```bash
cd ragblog && uv run python run.py
```
