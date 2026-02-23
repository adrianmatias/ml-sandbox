# TopBox: Boxer Ranking Pipeline

TopBox computes centrality scores for top boxers from match graphs scraped from Wikipedia pages. Boxer list and fight records sourced from Wikipedia for comprehensive coverage.

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

Writes top boxers to `topbox.csv` (scrapes Wikipedia; rate-limited—adjust delays if needed).

## Pipeline

```python
from topbox.conf import ConfCrawlerMin, ConfDataset, ConfPagerank
from topbox.crawl_min import get_matches
from topbox.dataset import create_dataset
from topbox.pagerank import compute_ranks

conf_c = ConfCrawlerMin()  # Wiki crawler config
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

## Rankings Comparison

TopBox rankings are derived from PageRank centrality on boxer match graphs, emphasizing win/loss patterns over recency or human expertise. Below, we compare TopBox's top 20 boxers (from `topbox.csv`) against popular rankings like The Ring, BoxRec, and ESPN P4P (as of 2024). Note: TopBox uses full career data for comprehensive graphs, while expert rankings often prioritize recent form and intangibles.

## PageRank Boxing Rankings (1965–Present)

**Methodology**  
- 168 consensus all-time greats (expanded seed list from BoxRec Top 60 last 60 years, ESPN, Ring Magazine, Bleacher Report)  
- ~5,500 verified professional bouts (1965–present)  
- Directed “loser → winner” PageRank graph  
- Automatic name normalization: diacritic stripping + minimal nickname rules + consecutive duplicate-word collapse (fixes “Canelo Alvarez Alvarez Alvarez” → “Canelo Alvarez”, “Sugar Sugar Sugar Ray Leonard Leonard Leonard” → “Sugar Ray Leonard”)

### Top 25 Results

| Rank | Boxer                    | Score  |
|------|--------------------------|--------|
| 1    | Dmitry Bivol             | 0.0162 |
| 2    | Artur Beterbiev          | 0.0155 |
| 3    | Canelo Alvarez           | 0.0128 |
| 4    | Lennox Lewis             | 0.0106 |
| 5    | Roman Gonzalez           | 0.0092 |
| 6    | Marvin Hagler            | 0.0086 |
| 7    | Manny Pacquiao           | 0.0083 |
| 8    | Carlos Monzon            | 0.0080 |
| 9    | Wladimir Klitschko       | 0.0080 |
| 10   | Eder Jofre               | 0.0073 |
| 11   | Jaime Munguia            | 0.0065 |
| 12   | Ricardo Lopez            | 0.0060 |
| 13   | Rosendo Alvarez          | 0.0056 |
| 14   | Julio Cesar Chavez       | 0.0056 |
| 15   | Bernard Hopkins          | 0.0056 |
| 16   | Terence Crawford         | 0.0054 |
| 17   | Vitali Klitschko         | 0.0053 |
| 18   | Roy Jones Jr.            | 0.0052 |
| 19   | Carlos Cuadras           | 0.0051 |
| 20   | Roberto Duran            | 0.0051 |
| 21   | Evander Holyfield        | 0.0050 |
| 22   | Juan Francisco Estrada   | 0.0049 |
| 23   | Oleksandr Usyk           | 0.0049 |
| 24   | Hasim Rahman             | 0.0048 |
| 25   | Oliver McCall            | 0.0047 |

### Discussion & Insights

**Standout Performers**  
Dmitry Bivol and Artur Beterbiev claim the top two spots through dense, high-quality win chains in the light-heavyweight division.

**Balanced Across Eras & Weights**  
Heavyweights remain strong (Lennox Lewis #4, the Klitschkos #9 & #17, Holyfield #21), but the expanded 168-seed list gives proper credit to small-division legends: Roman Gonzalez (#5), Ricardo Lopez (#12), Eder Jofre (#10), and Myung Woo Yuh (still top-30). Modern rising stars like Jaime Munguia (#11), Terence Crawford (#16), and Naoya Inoue (#45) climb rapidly as more bouts are added.

**Comparison with Popular Consensus**  
- **Higher than expert lists**: Bivol, Beterbiev, Roman Gonzalez, and Carlos Cuadras – the model purely rewards fighters who beat other strong fighters.  
- **Lower than expert lists**: Muhammad Ali (#39) and Floyd Mayweather Jr. (#67) – PageRank penalizes selective matchmaking and values volume of elite victories over cultural legacy or undefeated streaks.  
- **Strong alignment**: Marvin Hagler, Manny Pacquiao, Carlos Monzon, and Julio Cesar Chavez all land comfortably in the top 15, matching historian consensus.

**Technical Notes**
- Graph mode: loser → winner (standard PageRank importance flow)
- Dataset size: ~100 seeds → ~6,000+ matches from Wikipedia
- Data sources: `fighters.json` (boxer URLs), `name_mapping.json` (canonical names)

This objective, graph-based ranking complements traditional expert lists by showing exactly who sits at the center of boxing’s historical win network.

**Key Citations**  
- BoxRec Forum: Top 60 Pound-for-Pound Boxers 1965–2025 — https://boxrec.com/forum/viewtopic.php?t=266732  
- Bleacher Report: Definitive Top 50 Boxers of All Time (2025) — https://bleacherreport.com/articles/25262092  
- ESPN: Top 25 Boxers of the 21st Century — https://www.espn.com/boxing/story/_/id/46113827  
- The Ring Magazine Historical Ratings — https://boxrec.com/wiki/index.php/The_Ring_Magazine%27s_Annual_Ratings  
- Wikipedia fighter pages (all 168 URLs verified)