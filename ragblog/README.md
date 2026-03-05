# ragblog

Retrieval Augmented Generation (RAG) system for semantic search and Q&A over blog content.

## Objective

Enable intelligent question answering over blog archives by combining semantic search with local LLM inference. Crawls blog content, indexes it with vector embeddings, and answers natural language queries using retrieved context.

## Main Functionality

- **Crawler**: Scrapes blog posts from Blogger sites, extracts titles and content
- **Document Processing**: Loads JSONL, splits into chunks, generates embeddings
- **Vector Store**: Persists documents in ChromaDB with Ollama embeddings (`qwen3-embedding:8b`)
- **RAG Pipeline**: Retrieves relevant chunks, augments LLM context, generates answers via local Ollama (`gpt-oss:20b`)

## Approach

1. Crawl → Extract URLs → Parse HTML → Save as JSONL
2. Load → Split (1000 chars, 10 overlap) → Embed → Store in Chroma
3. Query → Retrieve top-k chunks → Prompt LLM with context → Return answer

Uses dataclass-based configuration, modular pipeline components, and functional patterns.

## Setup

### module

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-embedding:8b
ollama pull gpt-oss:20b
```

## CI

```bash
sh ci/lint.sh   # ruff check && ruff format
sh ci/test.sh   # pytest -vv
```

## Run

```bash
uv run python run/query.py
```

## Code Style

- clean code
  - object orientation
  - functional programming
  - classes, dataclasses and pydantic when justified
  - use conf where needed
  - hierarchical conf is clearer than long flat param lists
  - avoid underscore as function prefix for private. instead handle complexity as class method
- pytest TDD
- naming
  - self tabulated
  - promote 3 char naming
  - lexicography sorteable names if conceptually related
    - conf_data, conf_model instead of data_conf, model_conf
  - avoid plural naming: element_list instead of elements
  - self documenting naming
  - any code needing comments for clarification should be refactored into a named entity
  - each class implemented in its own file, snake_cased
    - caseclass and pydantic could be bundled into a single file
- **Configs**: `@dataclass(frozen=True)` hierarchical
- **Logging**: `LOGGER = logging.getLogger(__name__)` INFO/ERROR
- **Errors**: Specific `except`, log + sentinel return (`[]`)
- **Docs**: Google-style (`Args:`, `Returns:`)
- **No comments**: Refactor to named funcs/classes
- **Tests**: TDD `TestPascalName`, fixtures, `tmp_path`, strict asserts
