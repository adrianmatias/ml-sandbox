# TopBox: Boxer Ranking Pipeline

TopBox computes centrality scores for top boxers from match graphs scraped from box.live.

**Graph Representation**: Matches mapped to directed/undirected graphs with edges based on fight outcomes.
- **Modes**: `loser_to_winner` (loser → winner, ranks dominant boxers), `winner_to_loser` (winner → loser, ranks "tough" boxers), `undirected` (mutual edges, association-based).
- **Alternatives to Explore**: Weighted edges (by recency/significance), eigenvector centrality, bipartite (boxers/fights), time-decaying PageRank.
**Dataset**: `boxer_a`, `boxer_b`, `is_a_win`, `date` (parquet).
**Output**: Top N boxers by centrality score, saved to `topbox.csv`.

## Quick Start

```bash
git clone <repo>
cd topbox
uv sync
uv run python run_topbox.py
```

Writes top boxers to `topbox.csv` (demo crawl; box.live blocks bots—use proxies/VPN).

Sample output (`head -15 topbox.csv`):

```
Rank,Boxer,Score
1,Dmitry Bivol,0.0660
2,Artur Beterbiev,0.0627
3,Daniel Dubois,0.0292
4,Canelo Alvarez,0.0291
5,Oleksandr Usyk,0.0289
6,Terence Crawford,0.0211
7,Joe Joyce,0.0200
8,Filip Hrgovic,0.0193
9,Chris Eubank Jr,0.0163
10,Kazuto Ioka,0.0162
11,Floyd Mayweather Jr,0.0150
12,Shakur Stevenson,0.0138
13,Gennady Golovkin,0.0126
14,Anthony Joshua,0.0126
```

## Pipeline

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
ranks = compute_ranks(df, conf_p, mode="loser_to_winner")  # → [('Usyk', 0.25), ...]
```

## Installation

```bash
uv sync  # deps + editable src/topbox
```

**Dev**: `uv sync --extra dev` (ruff, pytest, mypy, coverage).

## CI

```bash
sh ci/ruff.sh
sh ci/test.sh
```

## Code Style


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


- **Configs**: `@dataclass(frozen=True)` hierarchical
- **Logging**: `LOGGER = logging.getLogger(__name__)` INFO/ERROR
- **Errors**: Specific `except`, log + sentinel return (`[]`)
- **Docs**: Google-style (`Args:`, `Returns:`)
- **No comments**: Refactor to named funcs/classes
- **Tests**: TDD `TestPascalName`, fixtures, `tmp_path`, strict asserts

See [AGENTS.md](AGENTS.md) for full guidelines (~150 lines).

## Data Source

- **box.live**: Profiles, "Recent Contests" table (date/opp/result).
- **Note**: Anti-bot measures (403); demo hardcoded. Prod: headers/rotating proxies/Selenium. Data is incomplete (partial boxer set, recent matches only).
- **Graph**: ~100s fights → ranks converge fast.



## Roadmap

- Real crawl (Selenium/events list)
- Divisions filter
- ML models (topbox/models.py)
- CLI `uv run topbox --division heavy`

Total lines: ~250. Clean, TDD, 90%+ coverage.
