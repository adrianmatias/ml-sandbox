# TopBox: Boxer Ranking Pipeline

TopBox computes centrality scores for top boxers from match graphs scraped from box.live. Boxer list sourced from Wikipedia champion pages for broad coverage.

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
1,Dmitry Bivol,0.0351
2,Artur Beterbiev,0.0330
3,Canelo Alvarez,0.0135
4,Terence Crawford,0.0095
5,Oleksandr Usyk,0.0094
6,Kenshiro Teraji,0.0092
7,Shakur Stevenson,0.0082
8,Naoya Inoue,0.0079
9,Teofimo Lopez,0.0060
10,Ryan Garcia,0.0059
11,Robson Conceicao,0.0058
12,O'Shaquie Foster,0.0058
13,Masamichi Yabuki,0.0052
14,Jesse ‘Bam’ Rodriguez,0.0052
```

## Pipeline

```python
from topbox.conf import ConfCrawler, ConfDataset, ConfPagerank, ConfWikipedia
from topbox.crawler import get_matches
from topbox.dataset import create_dataset
from topbox.pagerank import compute_ranks

conf_w = ConfWikipedia()  # Wikipedia config for boxer list
conf_c = ConfCrawler()
matches = get_matches(conf_c)  # Crawl box.live → list[Match]

conf_d = ConfDataset()
df = create_dataset(matches, conf_d)  # → data/matches.parquet (raw rows)

conf_p = ConfPagerank(top_n=10)
ranks = compute_ranks(df, conf_p, mode="loser_to_winner")  # Dedup + rank → [('Usyk', 0.25), ...]
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

See [AGENTS.md](AGENTS.md) for full guidelines (~150 lines).

## Data Source

- **Wikipedia**: Boxer list from champion pages (WBA/WBC/IBF/WBO/current) — ~1,767 names.
- **box.live**: Profiles, "Recent Contests" table (date/opp/result).
- **Note**: Anti-bot measures (403); demo crawl. Prod: headers/rotating proxies/Selenium. Data incomplete (partial matches) but broader boxer coverage.
- **Graph**: ~100s fights → ranks converge fast.

