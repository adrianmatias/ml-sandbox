# AGENTS.md - Coding Guidelines for AI Agents

## Build/Lint/Test Commands

### Primary Commands (ragblog project)
```bash
# Run all tests
cd /home/mat/PycharmProjects/ml-sandbox/ragblog && uv run pytest -vv

# Run single test
cd /home/mat/PycharmProjects/ml-sandbox/ragblog && uv run pytest -vv tests/test_crawler.py::test_get_url_list

# Lint and format
cd /home/mat/PycharmProjects/ml-sandbox/ragblog && ruff check . && ruff format .
```

### Other Projects
Most subprojects use Poetry or Conda:
```bash
# Poetry-based (tao-nlp, financial, anything)
cd <project> && poetry run pytest

# Conda-based (makemore, pytorchtext, nlpscrap, mlops, bag4message)
cd <project> && pytest  # or python -m pytest
```

## Code Style Guidelines

### Formatting
- **Line length**: 88 characters (Black-compatible)
- **Quotes**: Double quotes for strings
- **Indent**: 4 spaces (no tabs)
- **Target Python**: 3.10+

### Import Order (isort with black profile)
1. Standard library imports
2. Third-party imports
3. Local application imports

```python
import os
from dataclasses import dataclass
from typing import List

import requests
from bs4 import BeautifulSoup

from ragblog.conf import CONF
from ragblog.post import Post
```

### Naming Conventions
- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE` (module-level)
- **Private**: `_leading_underscore`

### Type Hints
- Use type hints for function parameters and returns
- Use `from __future__ import annotations` for forward references
- Use `List[Type]`, `Optional[Type]`, `Dict[str, Type]` from typing

```python
def process_data(items: List[str]) -> Dict[str, int]:
    ...
```

### Error Handling
- Use specific exceptions, avoid bare `except:`
- Log errors with the LOGGER (module-level constant)
- Return sentinel objects (e.g., `PostEmpty`) for recoverable failures

```python
try:
    response = requests.get(url)
    response.raise_for_status()
except requests.RequestException as e:
    LOGGER.error(f"Request failed: {e}")
    return PostEmpty()
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

### Configuration Pattern
Use dataclasses for configuration:

```python
@dataclass
class ConfCrawler:
    url: str = "https://example.com"
    post_count_min: int = 2
```

### Logging
- Use module-level LOGGER constant (uppercase)
- Include relevant context in log messages

```python
LOGGER = LoggerCustom().get_logger()
LOGGER.info(f"Processing {len(items)} items")
```

## Pre-commit Hooks
The repo uses pre-commit with:
- `black` for formatting
- `isort` for import sorting

Run manually: `pre-commit run --all-files`

## Ruff Configuration
```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.ruff.lint.pydocstyle]
convention = "google"
```
