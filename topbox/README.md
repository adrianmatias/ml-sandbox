# TopBox: Boxer Ranking Pipeline

TopBox computes centrality scores for boxers from directed match graphs scraped from Wikipedia. The default produces a consolidated view that gives deeper voice to historical win chains.

**Graph Representation**  
Matches are mapped to a directed `loser→winner` DiGraph (PageRank flows importance from loser to winner).  
- **Draws**: Mutual half-weight edges (each fighter receives 0.5 × fight weight; total per fight remains 1.0).  
- Edges carry exponential time weighting. Two lenses are available:  
  - **Consolidated** (default): older fights receive higher weight (`exp(+years_ago / 8)`), allowing historical depth to anchor the network.  
  - **Recent**: newer fights receive higher weight (`exp(-years_ago / 8)`), amplifying current activity.  

**Dataset**: `boxer_a`, `boxer_b`, `is_a_win`, `date` (parquet).  
**Output**: Two files — `topbox_consolidated.csv` (default) and `topbox_recent.csv`.

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
matches = get_matches()
ds.create_from_matches(matches)

# Consolidated lens (default, historical depth)
ranks_cons = PageRankBox(top_n=25, is_consolidated=True).compute(ds.df)
# Recent lens
ranks_rec = PageRankBox(top_n=25, is_consolidated=False).compute(ds.df)
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
- Time weighting: consolidated (historical depth) or recent (current activity)
- Weight classes deliberately omitted: they are attributes of individual fights, not fixed properties of boxers. Fighters like Canelo Alvarez and Terence Crawford regularly cross divisions; the global graph captures cross-division wins as they occurred, as a bottom-up info component.

### Consolidated Lens — Historical Depth (topbox_consolidated.csv)

| rank | boxer | score |
| --- | --- | --- |
| 1 | Canelo Alvarez | 0.0124 |
| 2 | Marvin Hagler | 0.0114 |
| 3 | Gervonta Davis | 0.0114 |
| 4 | Dmitry Bivol | 0.0111 |
| 5 | Artur Beterbiev | 0.0108 |
| 6 | Lamont Roach Jr | 0.0098 |
| 7 | Roman Gonzalez | 0.0089 |
| 8 | Roberto Duran | 0.0089 |
| 9 | Carlos Monzon | 0.0083 |
| 10 | Lennox Lewis | 0.0079 |
| 11 | Manny Pacquiao | 0.0074 |
| 12 | Evander Holyfield | 0.0067 |
| 13 | Wladimir Klitschko | 0.0064 |
| 14 | Carlos Cuadras | 0.0064 |
| 15 | Roy Jones Jr | 0.0062 |
| 16 | Muhammad Ali | 0.0062 |
| 17 | Bernard Hopkins | 0.0058 |
| 18 | Ricardo Lopez | 0.0057 |
| 19 | Julio Cesar Chavez | 0.0056 |
| 20 | Luis Manuel Rodriguez | 0.0055 |
### Recent Lens — Current Activity (topbox_recent.csv)
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


### Limitations of the Observed Network
- The graph reveals a necessary asymmetry: active fighters have not yet encountered the defeats that almost every career eventually records towards the end. Their centrality therefore reflects only the observed ascending side of the trajectory. Retired boxers carry every loss, every late decline.
- This is not a distortion to be corrected with projections or prime-year filters. It is the raw signal of the data as it stands today. Stripping away any attempt to estimate unseen outcomes preserves the first principle of the model: show exactly what the win network has produced up to this moment. The ranking accepts the paradox without adjustment.

### Discussion

**Standout performers**: in the Consolidated lens: Canelo Alvarez, Marvin Hagler and Roberto Duran rise through deep historical win chains. In the Recent lens: Dmitry Bivol and Artur Beterbiev dominate via current light-heavyweight density.

**Recent activity** lifts Crawford (#8), Usyk (#10), Joshua (#14), Gervonta Davis (#17), and Garcia (#20). **Historical volume** still matters: Hagler, Pacquiao, Monzon, and Chavez remain high despite the recency weighting.

**Higher than expert lists**: Bivol, Beterbiev, Roman Gonzalez — the model purely rewards fighters who beat other strong fighters.

**Lower than expert lists**: Muhammad Ali (16/30) and Floyd Mayweather Jr. (37) — PageRank penalizes selective matchmaking and values volume of elite victories over cultural legacy or undefeated streaks.

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
