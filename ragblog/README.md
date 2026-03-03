# ragblog

Retrieval Augmented Generation (RAG) system for semantic search and Q&A over blog content.

## Objective

Enable intelligent question answering over blog archives by combining semantic search with local LLM inference. Crawls blog content, indexes it with vector embeddings, and answers natural language queries using retrieved context.

## Main Functionality

- **Crawler**: Scrapes blog posts from Blogger sites, extracts text and metadata.
- **Vector Store**: Embeds documents using local models (e.g., Ollama), stores in ChromaDB for fast retrieval.
- **RAG Pipeline**: Combines retrieval with LLM generation for contextually grounded answers.
- **Local LLM**: Uses Ollama for privacy-preserving inference with models like Llama3, Mistral.

## Evaluation

The system includes comprehensive evaluation tools for RAG quality assessment:

### Answer Evaluation
- **Relevance**: How well answers address the question (1-5 scale)
- **Faithfulness**: Accuracy of context reflection
- **Coherence**: Logical flow and clarity
- Configurable metric weights for overall scoring

### Retrieval Evaluation  
- Document relevance assessment before answer generation
- Precision@k metrics for retrieval quality
- Handles embedding effectiveness evaluation

### Batch Evaluation
- Multi-query evaluation with statistical aggregation
- Multiple runs per query to account for LLM non-determinism
- Results saved as `data/batch_evaluation_{llm-model}.json`

Run batch evaluation:
```bash
uv run python ragblog/batch_evaluator.py
```

## Usage

### Setup

1. Install dependencies: `uv sync`
2. Download Ollama models: `ollama pull llama3`
3. Crawl blog data: `uv run python ragblog/run.py` (uncomment crawler lines)
4. Build vector store: Run pipeline with `is_db_ready=False`

### Query

Run `uv run python ragblog/run.py` to query the system with a sample question.

## Architecture

- **ragblog/conf.py**: Configuration dataclass
- **ragblog/crawler.py**: Web scraping logic
- **ragblog/rag_pipeline.py**: Core RAG components
- **ragblog/logger_custom.py**: Logging setup
- **ragblog/rag_evaluator.py**: Evaluation logic
- **ragblog/batch_evaluator.py**: Batch evaluation runner

## Requirements

- Python 3.10+
- Ollama for local LLM inference
- ChromaDB for vector storage
