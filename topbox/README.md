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

## timeline lens

`uv run python -m topbox.run_timeline` computes PageRank on every year-end window
(fights fought no later than Y) and dissolves the consolidated/recent duality into
rank trajectories.

Outputs:
- `data/topbox_timeline.csv` — long format `snapshot_year, rank, boxer, score`, one PageRank per year.
- `data/topbox_trajectory.csv` — per-boxer functionals: `first_ranked_year`, `last_ranked_year`, `presence`, `peak_year`, `peak_rank`, `peak_score`, `peak_offset`, `last_rank`, `integral` (sum of snapshot scores).

Comparing fighters at equal career stage (peak centrality, trajectory shape)
replaces comparing truncated active careers against completed ones.

## youtube

0013 page rank for boxers

not the tao with adrian matias

https://youtu.be/mGPn4-F_N-k?si=gvjE7JGJa4LuOm_d


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

## PageRank Boxing Rankings (1951–Present)

**Methodology**
- 480 seeded boxers (consensus all-time greats expanded from BoxRec/ESPN/Ring/Bleacher Report).
- 8,149 pro bouts (1951–present) from Wikipedia; 7,844 after dedup — committed as `data/match.parquet`.
- Directed "loser → winner" PageRank DiGraph.
- **Draws**: Mutual 0.5-weight edges (total fight weight = 1.0).
- Name normalization: NFKD + nicknames + dedup words.
- Time weighting: consolidated (historical depth) or recent (current activity)
- Weight classes deliberately omitted: they are attributes of individual fights, not fixed properties of boxers. Fighters like Canelo Alvarez and Terence Crawford regularly cross divisions; the global graph captures cross-division wins as they occurred, as a bottom-up info component.

### Consolidated Lens — Historical Depth (topbox_consolidated.csv)

| rank | boxer | score |
| --- | --- | --- |
| 1 | Canelo Alvarez | 0.0126 |
| 2 | Dmitry Bivol | 0.0116 |
| 3 | Marvin Hagler | 0.0114 |
| 4 | Gervonta Davis | 0.0112 |
| 5 | Artur Beterbiev | 0.0112 |
| 6 | Lamont Roach Jr | 0.0097 |
| 7 | Roman Gonzalez | 0.0089 |
| 8 | Roberto Duran | 0.0089 |
| 9 | Carlos Monzon | 0.0083 |
| 10 | Lennox Lewis | 0.0079 |

### Recent Lens — Current Activity (topbox_recent.csv)

| rank | boxer | score |
| --- | --- | --- |
| 1 | Dmitry Bivol | 0.0219 |
| 2 | Artur Beterbiev | 0.0201 |
| 3 | Gervonta Davis | 0.0151 |
| 4 | Lamont Roach Jr | 0.0125 |
| 5 | Canelo Alvarez | 0.0111 |
| 6 | Roman Gonzalez | 0.0098 |
| 7 | Terence Crawford | 0.009 |
| 8 | Carlos Monzon | 0.0088 |
| 9 | Marvin Hagler | 0.0079 |
| 10 | Manny Pacquiao | 0.0078 |

## Full Picture — Insights from Year-End Snapshots

The static lenses answer "where does flow sit today"; the timeline answers
"how did each career move through the network". Four findings survive contact
with the data:

**Peak parity dissolves the duality.** Years from first-ranked to peak cluster
tightly across six decades: Monzon 9, Hagler 10, Bivol 11, Duran / Beterbiev /
Gervonta / Chocolatito 12, Canelo 13, Ali 14. Active fighters are not on an
anomalous ascending trajectory — at equal career stage they have already
peaked where their predecessors peaked. Compare peaks, not current states.

**Decline is measurable, not just accepted.** Among top-30 peakers, median
final-rank ÷ peak-rank: retirees ≤1989 fall 36.5× (9% finish top-50);
1990–2005 → 10.6×; 2006–2018 → 8.9×; active 2019+ → 2.9× (64% still top-50).
The active-vs-retired asymmetry is real, mechanical and now bounded.

**Draws-as-flow distort the head of static rankings.** Lamont Roach Jr enters
the dataset through two opponent-row fights (a 2016 loss, the 2025 Davis
draw) yet ranks #4–#6 in both lenses. Drop draw flow (`draw_share=0`) and he
falls to #671. One half-edge from a hub ≈ 667 places.

**Network inertia is real.** Cumulative windows never forget: Monzon
re-takes #1 in 1988, 1991–92, 1996–98, 2002–03 — after retirement. The #1
lineage runs Ortiz (59–66) → Monzon (72–79) → Duran (80–82) → Hagler (83–87)
→ Holyfield/Chavez → Lewis → Pacquiao/Gonzalez/Canelo → Bivol (25–26), with
snapshot kingship reading as accumulated history, not a contemporary poll.

Example trajectories (rank per snapshot year):

| boxer | 1975 | 1985 | 1995 | 2005 | 2015 | 2020 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Carlos Monzon | 1 | 2 | 3 | 2 | 2 | 7 | 8 |
| Marvin Hagler | 15 | 1 | 2 | 3 | 4 | 10 | 9 |
| Muhammad Ali | 4 | 4 | 10 | 18 | 29 | 32 | 33 |
| Canelo Alvarez | – | – | – | 440 | 19 | 1 | 5 |
| Dmitry Bivol | – | – | – | – | 515 | 87 | 1 |

### Limitations of the Observed Network
- The graph reveals a necessary asymmetry: active fighters have not yet encountered the defeats that almost every career eventually records towards the end. Their centrality therefore reflects only the observed ascending side of the trajectory. Retired boxers carry every loss, every late decline.
- This is not a distortion to be corrected with projections or prime-year filters. It is the raw signal of the data as it stands today. Stripping away any attempt to estimate unseen outcomes preserves the first principle of the model: show exactly what the win network has produced up to this moment. The ranking accepts the paradox without adjustment — and the timeline quantifies it per cohort.
- Seed rot: dozens of seed URLs 404 or resolve to tableless pages; completeness stays era-and-notability biased. Snowball crawling through opponents is the planned fix.
- Entity resolution is rule-based (`Lamont Roach Jr.` vs `Lamont Roach Jr`); fine rank differences below ~50 should not be over-read until an identity pass exists.

### Discussion

**Standout performers**: in the Consolidated lens: Canelo Alvarez, Marvin Hagler and Roberto Duran rise through deep historical win chains. In the Recent lens: Dmitry Bivol and Artur Beterbiev dominate via current light-heavyweight density.

**Recent activity** lifts Crawford (#8), Usyk (#10), Joshua (#14), Gervonta Davis (#17), and Garcia (#20). **Historical volume** still matters: Hagler, Pacquiao, Monzon, and Chavez remain high despite the recency weighting.

**Higher than expert lists**: Bivol, Beterbiev, Roman Gonzalez — the model purely rewards fighters who beat other strong fighters.

**Lower than expert lists**: Muhammad Ali (16/30) and Floyd Mayweather Jr. (37) — PageRank penalizes selective matchmaking and values volume of elite victories over cultural legacy or undefeated streaks.

**Strong alignment**: Marvin Hagler, Manny Pacquiao, Carlos Monzon, and Julio Cesar Chavez all land comfortably in the top 20, matching historian consensus.

**Technical notes**
- Graph mode: loser → winner (standard PageRank importance flow)
- Dataset: 480 seeds → 8,149 matches (7,844 deduped) from Wikipedia, committed as `data/match.parquet`
- Data sources: `data/fighter_seed.json` (boxer URLs)
- Timeline: `uv run python -m topbox.run_timeline` → `topbox_timeline.csv` + `topbox_trajectory.csv`

This objective, graph-based ranking complements traditional expert lists by showing exactly who sits at the center of boxing's historical win network.

**Key Citations**
- BoxRec Forum: Top 60 Pound-for-Pound Boxers 1965–2025 — https://boxrec.com/forum/viewtopic.php?t=266732
- Bleacher Report: Definitive Top 50 Boxers of All Time (2025) — https://bleacherreport.com/articles/25262092
- ESPN: Top 25 Boxers of the 21st Century — https://www.espn.com/boxing/story/_/id/46113827
- The Ring Magazine Historical Ratings — https://boxrec.com/wiki/index.php/The_Ring_Magazine%27s_Annual_Ratings
- Wikipedia fighter pages (all 168 URLs verified)
