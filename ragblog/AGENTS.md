# Agent Instructions for ragblog

This is a Python project for Retrieval Augmented Generation (RAG) on crawled blog content.

## Build/Lint/Test Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest -vv

# Run a single test
uv run pytest -vv tests/test_post.py::test_basic

# Run tests for a specific file
uv run pytest -vv tests/test_crawler.py

# Lint/format code (from ci/ directory)
sh ci/lint.sh

# Run the application
uv run python ragblog/run.py
```

## Project Structure

- `ragblog/` - Main source code
- `tests/` - Test files (pytest)
- `chroma/` - Vector store persistence
- `data/` - Crawled blog data (JSONL format)
- `ci/` - CI scripts (lint.sh, test.sh)

## Code Style Guidelines

### Formatting
- **Ruff** formatter (88 char line length, double quotes, spaces)
- **Ruff** linter (E, F, I rules enabled - errors, pyflakes, import sorting)
- Target Python: 3.10+
- Run `sh ci/lint.sh` before committing

### Import Order
Order: stdlib → third-party → local (enforced by ruff I001)

```python
from __future__ import annotations

# Standard library
import json
from dataclasses import dataclass
from typing import List, Any

# Third-party
import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma

# Local
from ragblog.conf import CONF
from ragblog.logger_custom import LoggerCustom
```

### Type Hints
- Use type hints for function parameters and return types
- Use `typing` module for complex types (List, Any, Optional)
- Use dataclasses for configuration objects

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    chunk_size: int

class MyClass:
    def process(self, items: List[str]) -> None:
        pass
```

### Naming Conventions
- **Classes**: PascalCase (`RagPipeline`, `ConfCrawler`)
- **Functions/variables**: snake_case (`get_url_list`, `post_count_min`)
- **Constants**: UPPER_CASE (`CONF`, `LOGGER`)
- **Private**: prefix with underscore `_internal_method`
- **Dataclasses**: Use `Conf` suffix for config classes (`RagPipelineConf`)

### Error Handling
- Check HTTP response status codes explicitly
- Use logging rather than print statements
- Return empty/null objects instead of None (see `PostEmpty`)

```python
if response.status_code != 200:
    LOGGER.info(f"Failed to load {url}")
    return PostEmpty()
```

### Logging
- Use the custom `LoggerCustom` class for all logging
- Log at appropriate levels (INFO for operations, DEBUG for details)

```python
from ragblog.logger_custom import LoggerCustom

LOGGER = LoggerCustom().get_logger()
LOGGER.info(f"Processing {len(items)} items")
```

### Docstrings
- Use Google-style docstrings
- Include Args and Returns sections for public functions

```python
def fetch_data(url: str) -> Data:
    """Fetch data from URL.

    Args:
        url: The URL to fetch from.

    Returns:
        The fetched data object.
    """
    ...
```

### Testing
- Use pytest with fixtures
- Mock external dependencies (HTTP requests, vector stores)
- Test files prefixed with `test_`
- Test functions prefixed with `test_`
- Use descriptive test names

### Configuration
- Store config in `ragblog/conf.py`
- Use dataclasses for type-safe configuration
- Import config as: `from ragblog.conf import CONF`

## Dependencies

Key libraries:
- **LangChain** (langchain, langchain-community, langchain-chroma) - RAG pipeline
- **ChromaDB** - Vector store for embeddings
- **BeautifulSoup4** + **requests** - Web crawling
- **Pydantic** - Data validationc
- **pytest** - Testing

## Running the Application

The main entry point is `ragblog/run.py`. It:
1. Crawls blog posts using `Crawler`
2. Processes and stores them in ChromaDB
3. Runs RAG queries against the vector store

Ensure Ollama is running locally for embeddings and LLM inference.
