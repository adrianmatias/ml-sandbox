# TopBox: Boxer Ranking Pipeline

TopBox computes PageRank for top boxers from match graphs scraped from BoxRec.

**Graph**: Directed edge `winner → loser`.
**Dataset**: `boxer_a`, `boxer_b`, `is_a_win`, `date` (parquet).
**Output**: Top N boxers by PageRank score.

## 🚀 Quick Start

```bash
git clone <repo>
cd topbox
uv sync
uv run python run_topbox.py
```

Prints top boxers (demo crawl; BoxRec blocks bots—use proxies/VPN).

## 📋 Pipeline

```python
from topbox.conf import ConfCrawler, ConfDataset, ConfPagerank
from topbox.crawler import get_matches
from topbox.dataset import create_dataset
from topbox.pagerank import compute_ranks

conf_c = ConfCrawler()
matches = get_matches(conf_c)  # BoxRec profiles → list[Match]

conf_d = ConfDataset()
df = create_dataset(matches, conf_d)  # → data/matches.parquet

conf_p = ConfPagerank(top_n=10)
ranks = compute_ranks(df, conf_p)  # → [('Usyk', 0.25), ...]
```

## 🛠️ Installation

```bash
uv sync  # deps + editable src/topbox
```

**Dev**: `uv sync --extra dev` (ruff, pytest, mypy, coverage).

## 🧪 CI

```bash
sh ci/ruff.sh
sh ci/test.sh
```

## 📐 Code Style


- clean code
  - object orientation
  - functional programming
  - classes, dataclasses and pydantic when justified
  - use conf where needed
  - hierarchical conf is clearer than long flat param lists
- pytest TDD
- naming
  - self tabulated
  - promote 3 char naming
  - avoid plural naming: element_list instead of elements
  - self documenting naming
  - any code needing comments for clarification should be refactored into a named entity


- **Layout**: src-layout (`src/topbox/`)
- **Line length**: 88 (ruff)
- **Quotes**: Double (`"str"`)
- **Types**: `list[str] | None`, `__future__ annotations`
- **Imports**: `__future__` → std → 3rd → local (`isort black`)
- **Naming**: `snake_case` funcs/vars (short/self-doc, e.g. `opp`), `PascalCase` classes
- **Configs**: `@dataclass(frozen=True)` hierarchical
- **Logging**: `LOGGER = logging.getLogger(__name__)` INFO/ERROR
- **Errors**: Specific `except`, log + sentinel return (`[]`)
- **Docs**: Google-style (`Args:`, `Returns:`)
- **No comments**: Refactor to named funcs/classes
- **Tests**: TDD `TestPascalName`, fixtures, `tmp_path`, strict asserts

See [AGENTS.md](AGENTS.md) for full guidelines (~150 lines).

## 📚 Data Source

- **BoxRec**: Profiles (`/en/proboxer/ID`), table `.table1` (date/opp/res).
- **Note**: Anti-bot (403); demo hardcoded. Prod: headers/rotating proxies/Selenium.
- **Graph**: ~100s fights → ranks converge fast.

## 🛡️ Security/Notes

- Timeouts on requests
- No secrets committed
- Parquet efficient, date filter

## Roadmap

- Real crawl (Selenium/events list)
- Divisions filter
- ML models (topbox/models.py)
- CLI `uv run topbox --division heavy`

Total lines: ~250. Clean, TDD, 90%+ coverage.
