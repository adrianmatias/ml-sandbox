# TopBox: Boxer Ranking Pipeline

TopBox computes centrality scores for boxers from directed match graphs scraped from Wikipedia. The default uses loser→winner edges with recency weighting.

**Graph Representation**
Matches are mapped to a directed `loser→winner` DiGraph (PageRank flows importance from loser to winner).
- **Draws**: Mutual half-weight edges (each fighter gets 0.5 × fight weight).
- Edges: exponential decay `max(0.1, exp(−years_ago / 8))` (τ=8 years; recent fights dominate).

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
- 168 consensus all-time greats (expanded seed from BoxRec/ESPN/Ring/Bleacher Report).
- ~5,500 pro bouts (1965–present) from Wikipedia.
- Directed "loser → winner" PageRank DiGraph.
- **Draws**: Mutual 0.5-weight edges (total fight weight = 1.0).
- Name normalization: NFKD + nicknames + dedup words.
- Recency: `max(0.1, exp(−years/8))`.
- Weight classes deliberately omitted: they are attributes of individual fights, not fixed properties of boxers. Fighters like Canelo Alvarez and Terence Crawford regularly cross divisions; the global graph captures cross-division wins as they occurred, as a botTom-up info component.

### Top 25

| Rank | Boxer | Score |
|------|-------|-------|
| 1 | Dmitry Bivol | 0.0206 |
| 2 | Artur Beterbiev | 0.0191 |
| 3 | Gervonta Davis | 0.0159 |
| 4 | Lamont Roach Jr | 0.0132 |
| 5 | Canelo Alvarez | 0.0109 |
| 6 | Roman Gonzalez | 0.0099 |
| 7 | Manny Pacquiao | 0.0098 |
| 8 | Terence Crawford | 0.0093 |
| 9 | Carlos Monzon | 0.0088 |
| 10 | Marvin Hagler | 0.0078 |
| 11 | Oleksandr Usyk | 0.0067 |
| 12 | Lennox Lewis | 0.0066 |
| 13 | Ricardo Lopez | 0.0064 |
| 14 | Juan Francisco Estrada | 0.006 |
| 15 | Anthony Joshua | 0.006 |
| 16 | Wladimir Klitschko | 0.0059 |
| 17 | Evander Holyfield | 0.0054 |
| 18 | Julio Cesar Chavez | 0.0053 |
| 19 | Roberto Duran | 0.0052 |
| 20 | Ryan Garcia | 0.0052 |
| 21 | Hector Camacho | 0.005 |
| 22 | Rosendo Alvarez | 0.005 |
| 23 | Myung Woo Yuh | 0.0049 |
| 24 | Roy Jones Jr | 0.0046 |
| 25 | Bernard Hopkins | 0.0046 |

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
