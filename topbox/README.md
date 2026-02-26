# TopBox: Boxer Ranking Pipeline

TopBox computes centrality scores for boxers from directed match graphs scraped from Wikipedia. The default uses loser→winner edges with recency weighting.

**Graph Representation**
Matches are mapped to a directed graph. Default mode `loser_to_winner` flows importance from loser to winner.
- Edges receive exponential decay weight = max(0.1, exp(−years_ago / 8)) where years_ago is calculated from fight date to 2026.
- This makes recent fights contribute more to final scores while older results retain influence.
- Other modes: `winner_to_loser`, `undirected`.

**Dataset**: `boxer_a`, `boxer_b`, `is_a_win`, `date` (parquet).  
**Output**: Top N boxers by PageRank score, saved to `topbox.csv`.

## Quick Start

```bash
git clone <repo>
cd topbox
uv sync
uv run python -m topbox.run
```

Writes top boxers to `topbox.csv` (scrapes Wikipedia; rate-limited—adjust delays if needed).

## Pipeline

```python
from topbox.crawler_wiki import get_matches
from topbox.dataset import Dataset
from topbox.page_rank_box import PageRankBox

ds = Dataset(save_path="data/match.parquet", min_date="1950-01-01")
matches = get_matches()  # Crawl Wikipedia → list[Match]
ds.create_from_matches(matches)  # → data/match.parquet (raw rows)

ranks = PageRankBox(top_n=10, mode="loser_to_winner").compute(ds.df)  # Dedup + rank → DataFrame
```

## Installation

```bash
uv sync  # deps + editable src/topbox
```

**Dev**: `uv sync --group dev` (ruff, pytest, mypy, coverage).

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


## Data Source

- **Wikipedia**: Boxer list from champion pages (WBA/WBC/IBF/WBO/current) — ~1,767 names.
- **box.live**: Profiles, "Recent Contests" table (date/opp/result).
- **Note**: Anti-bot measures (403); demo crawl. Prod: headers/rotating proxies/Selenium. Data incomplete (partial matches) but broader boxer coverage.
- **Graph**: ~100s fights → ranks converge fast.

## PageRank Boxing Rankings (1965–Present)

**Methodology**
- 168 consensus all-time greats (expanded seed list from BoxRec Top 60 last 60 years, ESPN, Ring Magazine, Bleacher Report)
- ~5,500 verified professional bouts (1965–present)
- Directed "loser → winner" PageRank graph
- Automatic name normalization: diacritic stripping + minimal nickname rules + consecutive duplicate-word collapse (fixes "Canelo Alvarez Alvarez Alvarez" → "Canelo Alvarez", "Sugar Sugar Sugar Ray Leonard Leonard Leonard" → "Sugar Ray Leonard")
- Recency weighting: exponential decay (τ = 8 years) on every edge — older wins are down-weighted but never removed
- Weight classes deliberately omitted: they are attributes of individual fights, not fixed properties of boxers. Fighters like Canelo Alvarez and Terence Crawford regularly cross divisions; the global graph captures cross-division wins as they occurred, as a botom-up info component.

### Top 25

| Rank | Boxer | Score |
|------|-------|-------|
| 1 | Dmitry Bivol | 0.0191 |
| 2 | Artur Beterbiev | 0.0183 |
| 3 | Canelo Alvarez | 0.0116 |
| 4 | Roman Gonzalez | 0.0093 |
| 5 | Marvin Hagler | 0.0089 |
| 6 | Manny Pacquiao | 0.0086 |
| 7 | Lennox Lewis | 0.0086 |
| 8 | Terence Crawford | 0.0085 |
| 9 | Carlos Monzon | 0.0082 |
| 10 | Oleksandr Usyk | 0.0068 |
| 11 | Wladimir Klitschko | 0.0064 |
| 12 | Eder Jofre | 0.0062 |
| 13 | Ricardo Lopez | 0.0061 |
| 14 | Anthony Joshua | 0.0060 |
| 15 | Rosendo Alvarez | 0.0057 |
| 16 | Juan Francisco Estrada | 0.0056 |
| 17 | Gervonta Davis | 0.0056 |
| 18 | Julio Cesar Chavez | 0.0054 |
| 19 | Roberto Duran | 0.0053 |
| 20 | Ryan Garcia | 0.0050 |
| 21 | Lamont Roach Jr | 0.0049 |
| 22 | Jaime Munguia | 0.0049 |
| 23 | Bernard Hopkins | 0.0047 |
| 24 | Myung Woo Yuh | 0.0047 |
| 25 | Luis Manuel Rodriguez | 0.0047 |

### Discussion

**Standout performers**: Dmitry Bivol and Artur Beterbiev claim the top two spots through dense, high-quality win chains in the light-heavyweight division.

**Recent activity** lifts Crawford (#8), Usyk (#10), Joshua (#14), Gervonta Davis (#17), and Garcia (#20). **Historical volume** still matters: Hagler, Pacquiao, Monzon, and Chavez remain high despite the recency weighting.

**Higher than expert lists**: Bivol, Beterbiev, Roman Gonzalez — the model purely rewards fighters who beat other strong fighters.

**Lower than expert lists**: Muhammad Ali (#38) and Floyd Mayweather Jr. (outside top 50) — PageRank penalizes selective matchmaking and values volume of elite victories over cultural legacy or undefeated streaks.

**Strong alignment**: Marvin Hagler, Manny Pacquiao, Carlos Monzon, and Julio Cesar Chavez all land comfortably in the top 20, matching historian consensus.

**Technical notes**
- Graph mode: loser → winner (standard PageRank importance flow)
- Dataset: ~168 seeds → ~5,500+ matches from Wikipedia
- Data sources: `fighters.json` (boxer URLs)

This objective, graph-based ranking complements traditional expert lists by showing exactly who sits at the center of boxing's historical win network.

**Key Citations**
- BoxRec Forum: Top 60 Pound-for-Pound Boxers 1965–2025 — https://boxrec.com/forum/viewtopic.php?t=266732
- Bleacher Report: Definitive Top 50 Boxers of All Time (2025) — https://bleacherreport.com/articles/25262092
- ESPN: Top 25 Boxers of the 21st Century — https://www.espn.com/boxing/story/_/id/46113827
- The Ring Magazine Historical Ratings — https://boxrec.com/wiki/index.php/The_Ring_Magazine%27s_Annual_Ratings
- Wikipedia fighter pages (all 168 URLs verified)
