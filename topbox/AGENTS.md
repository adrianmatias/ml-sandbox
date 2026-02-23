# AGENTS.md - Coding Guidelines for AI Agents in topbox

## Build/Lint/Test Commands

### Primary Commands (topbox project)
```bash
# Sync deps (recommended before work)
uv sync

# Run all tests
uv run pytest -vv

# Run single test file
uv run pytest -vv tests/test_conf.py

# Run single test class
uv run pytest -vv tests/test_crawl_min.py::TestExtractMatches

# Run single test
uv run pytest -vv tests/test_crawl_min.py::test_extract_matches_success

# Lint and format
uv run ruff check . && uv run ruff format .

# Typecheck (mypy)
uv run mypy .

# Coverage
uv run coverage run -m pytest && uv run coverage report --fail-under=90
```

## Code Style Guidelines

### Formatting
- **Line length**: 88 characters (ruff/black-compatible)
- **Quotes**: Double quotes for strings (`"str"`)
- **Indent**: 4 spaces (no tabs)
- **Target Python**: 3.10+ (pyproject.toml)

### Import Order (isort with black profile)
1. `__future__` imports first
2. Standard library imports
3. Third-party imports
4. Local application imports (src/topbox/*)

```python
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup

from topbox.conf import ConfCrawler
from topbox.crawler_live_box import Match
```

### Naming Conventions
- **Functions/variables**: `snake_case` (short, self-documenting, promote 3-5 chars where clear e.g. `opp`, `res`)
- **Classes**: `PascalCase` (e.g. `ConfCrawlerMin`, `TestExtractMatches`)
- **Constants**: `UPPER_CASE` (module-level)
- **Private**: `_leading_underscore`
- **Dataclasses**: `PascalCase` for conf (e.g. `ConfCrawlerMin`)
- **Files**: `snake_case` for code, `snake_case.json` for data (e.g., `fighters.json`, `name_mapping.json`)

### Type Hints
- Always use type hints for params/returns
- Use built-ins post-3.9: `list[str]`, `dict[str, int]`, `str | None`
- Forward refs: `__future__ annotations`
- No `typing.Optional/List/Dict` unless <3.10

```python
def extract_matches(name: str, url: str) -> list[Match]:
    ...
```

### Error Handling
- Specific exceptions (no bare `except`)
- Log with `LOGGER` (module-level)
- Return sentinels (e.g. `[]`, `pd.DataFrame()` for failures)
- `raise_for_status()` for requests

```python
try:
    resp = requests.get(url)
    resp.raise_for_status()
except requests.RequestException as e:
    LOGGER.error(f"Request failed: {e}")
    return []
```

### Docstrings
- Google-style for public functions/classes
- `Args:` and `Returns:` sections
- Concise, no examples unless complex

```python
def get_matches(conf: ConfCrawler) -> list[Match]:
    \"\"\"Get matches from BoxRec profiles.

    Args:
        conf: Crawler config.

    Returns:
        List of Match objects.
    \"\"\"
    ...
```

### Configuration Pattern
- `@dataclass(frozen=True)` for configs
- Hierarchical (e.g. `ConfCrawler`, `ConfDataset`)
- Defaults sensible, override via init

```python
@dataclass(frozen=True)
class ConfCrawler:
    base_url: str = "https://boxrec.com"
    max_pages: int = 10
```

### Logging
- Module-level `LOGGER = logging.getLogger(__name__)`
- Levels: INFO for progress, WARNING no table, ERROR failures
- Context: f-strings with len, names

```python
LOGGER.info(f"Got {len(fights)} fights for {name}")
```

### Testing (pytest TDD)
- Class `TestPascalName`
- Fixtures for sample data (`sample_html`, `sample_matches`)
- Test defaults, overrides, edge (no table, filter)
- tmp_path for files
- Strict: assert len, columns, types

## Pre-commit Hooks
Repo uses pre-commit with ruff, isort, black.
Manual: `uv run pre-commit run --all-files` (install via `pre-commit install`)

## Ruff Configuration (pyproject.toml)
```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

## Additional Patterns
- **Graph**: networkx DiGraph, winner -> loser edge
- **Dataset**: pd.DataFrame(vars(m)), to_parquet, date filter
- **No comments**: Refactor to named funcs/classes
- **Security**: No secrets, timeouts on requests
- **Pipeline**: crawl -> dataset -> pagerank -> top N print

Total lines: ~170. Follow for all changes!
