# agt — agent side of the dialog

Append-only log. Counterpart file: `mat.md` (maintained by mat).
One `##` entry per turn, newest at the bottom.

---

## 2026-08-22 — response to "inconclusive conclusions, contradictory duality"

Context, from `mat.md` (mat): the conclusions feel inconclusive, pending on a
contradictory duality; probably the best truth is that there is no ground truth,
but results seem too sensitive to assumptions, parameters, dataset completion.
Open to something deeper or different: other algorithms, alternative graph
representations, better/wider crawling, or just a different interpretation.

I read the full module (`crawler_wiki`, `dataset`, `page_rank_box`, `run`,
`crawler_live_box`, tests) and ran a quick quantitative pass over the two
shipped outputs (`data/topbox_consolidated.csv`, `data/topbox_recent.csv`,
5,000 rows each). Findings and proposal below.

### 1. Diagnosis — "too sensitive" is really three separate problems

**(a) Identifiability.** PageRank on the loser→winner graph estimates
*centrality in the recorded win network*, not boxing ability. These coincide
only under strong assumptions the data does not satisfy. The README already
concedes this ("the ranking accepts the paradox"). So the duality is not a bug:
consolidated and recent are two projections of one temporal object, and neither
is "who was best".

**(b) Structural bias from the crawl design.** The graph is a star-neighborhood
around the seed list: every seed contributes its full table, but non-seed
opponents appear only inside seed tables. Consequences:

- A seed's wins over non-seeds are recorded; a seed's losses *to* non-seeds are
  missing unless that non-seed is also a seed. Seeds are systematically
  over-rewarded — flow enters them that can never leave through the missing
  edges. Sinks hoard PageRank.
- **The evaluation leaks into the input.** The 168/480 seeds come from
  BoxRec/ESPN/Ring consensus lists, so "PageRank agrees with expert lists" is
  partly circular: the experts chose the node set. The pipeline cannot
  falsify itself against the very consensus that built it.

**(c) Concrete modeling artifacts** — each one a measurable sensitivity:

1. **Consolidated kernel is unbounded exponential growth**: `exp(+years_ago/8)`.
   A 74-year-old fight weighs ~10,400× a current one. The lens is therefore a
   direct function of `min_date`: move the 1950 cutoff and the entire ranking
   re-anchors on whatever sits at the left edge of the window — the sparsest,
   least complete part of the crawl.
2. **Recent kernel hits its floor at ~18 years** (`weight_min=0.1`):
   everything before ~2008 enters at identical flat weight. "Recent" is really
   "18-year decay + flat all-history". That is why Monzon/Hagler survive the
   recent top-10. Also the docstring calls tau a half-life; `exp(-t/8)` has
   half-life `8·ln2 ≈ 5.5y`.
3. **Draw half-edges are the cheapest centrality in the graph.** A draw pays
   0.5·w inbound from a strong node at zero loss-risk. Lamont Roach Jr at #6
   consolidated / #4 recent — above Duran, Monzon, Pacquiao — is the expected
   signature of drawing a high-centrality hub (the Davis draw) while having a
   short, thinly-recorded record. (Hypothesis from output shape; verify against
   `match.parquet` once it exists — see §2.5.)
4. **No evidence-count correction.** PageRank conflates "strong" with "thinly
   recorded": a node with 2 recorded fights adjacent to a hub outranks a node
   with 60 fights. Low out-degree + hub neighbor = inflated rank.
5. **Score tail is rounding noise.** Top-25 hold ~24–26% of score mass; only
   ~200 fighters have score > 0.001. The remaining ~4,800 CSV rows are
   0.0000-after-rounding — an ordering over noise.

**Measured sensitivity, and a surprise.** Spearman between the two lenses over
all 5,000 fighters: **0.998**. Top-10 overlap 8/10, top-25 18/25, top-50 40/50.
So the duality barely moves the bulk of the ranking — topology dominates; the
lens matters only at the very top (e.g. Jake Paul 948→94, Mario Barrios
931→102). The instability mat feels is concentrated exactly where score mass is
thin and artifacts (1)–(4) bite. That is good news: it means the fix is
targeted, not a full rethink.

### 2. What I would change first — keep the graph, fix the estimator

1. **Bound the time kernel.** Replace `exp(±t/8)` with a bounded kernel, e.g.
   `w(t) = w_min + (1-w_min)·exp(-((t-t0)/tau)²)` centered on a reference era,
   or a logistic in `t`. Then `min_date` stops being a hidden super-parameter.
2. **Shrink by evidence.** Multiply score by `n/(n+k)` (recorded-fight count),
   or filter `deg ≥ k` before ranking, and ship `score` with `n` so the reader
   sees confidence. Kills the thin-record riders.
3. **Make draws explicit, not half-wins.** Either drop draws from the edge set
   (count them only as evidence for shrinkage), or add a small global
   draw-constant so a draw can never outvalue a win. Sensitivity of the top-25
   to `draw_share ∈ {0, 0.25, 0.5}` should be reported, not assumed.
4. **Snowball the crawl.** Every opponent parsed from any table becomes a new
   seed (bounded by depth/budget). This directly repairs bias (b): losses to
   non-seeds stop vanishing. It is the single highest-leverage data change.
5. **Commit `data/match.parquet`.** The repo ships outputs without the input;
   results are currently not reproducible from the repo. Store it (it is small)
   or a regeneration script + date pin.

### 3. The deeper move — stop publishing point estimates

"No ground truth" is exactly right, and the honest response to an
unidentifiable quantity is a **distribution, not a number**:

- **Ensemble/bootstrap.** Resample fights (bootstrap), and sweep
  `tau ∈ [4..16]`, `alpha ∈ {0.7..0.9}`, `draw_share ∈ {0..0.5}`, `min_date`,
  both lenses → a rank distribution per boxer. Output: `P(rank ≤ 10)`,
  median rank, and stable top-k *sets*. "Canelo is #1 in 61% of settings;
  {Canelo, Hagler, Bivol, Beterbiev, Gervonta} is the stable top-5" is a
  falsifiable, robust claim — the current single CSV is neither.
- **External benchmark as objective, not input.** The Ring annual ratings
  (already cited, mirrored on BoxRec wiki) give per-year expert rankings.
  Use them *only* for evaluation: Kendall tau(model, Ring) per era. Then
  hyperparameters are tuned against something outside the pipeline, and
  "sensitive to parameters" becomes "parameters have a measurable optimum".
  Build the evaluation universe independently of the seeds to break the
  circularity from §1b.
- **Second opinion from a different model family.** Bradley–Terry / Elo-style
  paired-comparison (or Colley/Massey) models win *probability* per pair,
  handle incomplete schedules natively, are upset-aware, and come with
  standard errors for free. Where PageRank and BT agree → consensus core;
  where they disagree (expect: volume-vs-quality profiles) is the interesting
  output. Method disagreement is signal, not failure.

### 4. The reframe that dissolves the duality — temporal snapshots

Instead of two static lenses, compute PageRank on **year-end snapshot graphs**
(only fights ≤ year Y). This yields a *rank trajectory per boxer*: Hagler's
complete career curve vs Canelo's curve-to-date. Then compare well-defined
functionals: **peak centrality**, **career-integral centrality**, **age of
peak**. The active-vs-retired paradox stops being accepted-by-fiat (README
§Limitations) and becomes a comparison of trajectories at equal career stage.
The consolidated/recent duality collapses into two readouts of one object —
and Jake Paul spiking in the recent snapshot while his trajectory decays after
each retired opponent stops fighting is visible *as a shape*, not hidden in a
scalar.

Optional extensions on the same data: margin-of-victory edge weights (KO/UD/SD
is already in the scraped tables and currently discarded), a weight-class
multiplex layer for a p4p view on top of the global graph, and Katz centrality
with attenuation as a chain-robustness check on the "deep win chains" story.

### 5. Minimal concrete path (in order)

1. Commit `match.parquet`; add min-degree + shrinkage; bound the kernel. (est)
2. Snapshot trajectories + peak/integral functionals. (reframe)
3. Bootstrap ensemble → rank distributions; report stable sets. (uncertainty)
4. Ring-ratings benchmark → Kendall tau; tune kernel params against it. (truth)
5. Snowball crawl to depth 1–2; rebuild; re-run benchmark. (data)
6. Bradley–Terry as second opinion; publish agreement/disagreement sets.

My read: the graph perspective is right — the inconclusiveness lives in the
*estimator and the presentation*, not in the representation. Steps 1–3 alone
convert "contradictory duality" into "quantified stability with two
projections", which is a much stronger place to stand.

— agent (ox-alpha), topbox session 2026-08-22

---

## 2026-08-22 — sliding-window snapshots: implemented, run, results

Context, from `mat.md` (mat): agrees the sliding window snapshots through years
dissolve the temporal dissonance — go implement, run and discuss as proposed.

### What was built

- `topbox/conf_timeline.py` — `ConfTimeline` frozen dataclass (year_first,
  year_last, step, min_fights, top_n).
- `topbox/page_rank_timeline.py` — `PageRankTimeline`: one PageRank per
  year-end window (fights ≤ Y), long-format output.
- `topbox/trajectory_digest.py` — `TrajectoryDigest`: per-boxer functionals
  (first/last ranked year, presence, peak_year, peak_rank, peak_score,
  peak_offset, last_rank, integral = sum of snapshot scores).
- `topbox/run_timeline.py` — `python -m topbox.run_timeline`.
- `topbox/page_rank_box.compute` gained optional `round_digits=None` so
  trajectory integrals use full-precision scores (default unchanged).
- `html5lib` added as dependency: pandas falls back to it on markup lxml
  rejects; without it those fighters silently parsed to zero matches.
- Tests `test_page_rank_timeline.py`, `test_trajectory_digest.py`; suite now
  55 passing, ruff clean.
- Data regenerated and persisted: `data/match.parquet` (8,149 fights crawled
  from 480 seeds → 7,844 after dedup). The repo previously shipped outputs
  without inputs. Both lens CSVs regenerated from this crawl reproduce the
  shipped shape.

### Run

74 snapshots (1953–2026), 178,741 timeline rows, 5,751 boxer trajectories.
Outputs: `data/topbox_timeline.csv`, `data/topbox_trajectory.csv`.

### Results

**1. Lineage of #1.** Ortiz (59–66) → Monzon (72–79) → Duran (80–82) → Hagler
(83–87) → Holyfield/Chavez interludes → Lewis → Pacquiao/Gonzalez/Canelo →
Bivol (25–26). But cumulative windows never forget: Monzon re-takes #1 in
1988, 1991–92, 1996–98 and 2002–03 — a decade after retirement, years after
his death. "Who leads year Y" under a cumulative window is a statement about
accumulated history (network inertia), not a contemporary poll. The honest
readouts are the trajectory functionals, which is exactly what the digest
provides.

**2. Peak parity dissolves the duality.** Years from first-ranked to peak:
Monzon 9, Hagler 10, Bivol 11, Duran / Beterbiev / Gervonta / Chocolatito 12,
Canelo 13, Ali 14. The band is tight across six decades. The modern actives
are not on some anomalous ascending trajectory — they have already reached
peak at the career stage where their predecessors peaked. Compare peaks at
equal stage and Bivol's #1 sits legitimately beside Hagler's #1.

**3. The accepted paradox becomes a measurement.** Among fighters peaking
top-30, median last_rank ÷ peak_rank by true career end (from fight dates):
retired ≤1989 → 36.5× (9% finish top-50); 1990–2005 → 10.6× (24%);
2006–2018 → 8.9× (28%); active 2019+ → 2.9× (64%). Decline is mechanical,
era-dependent and now bounded — no longer accepted-by-fiat.

**4. Draw-farming verified.** Lamont Roach Jr enters the dataset through
exactly two opponent-row fights (a 2016 loss, the 2025 Davis draw). His peak
is #4 recent / #6 consolidated. Counterfactual with `draw_share=0`: he drops
#4 → #671. One half-edge from a hub ≈ 667 rank places — the strongest single
confirmation that draws-as-flow plus thin records distort the head of the
static rankings. The earlier recommendation stands: draws as evidence for
confidence weighting, not as flow.

**5. Trajectories are the deliverable.** Decadal ranks: Hagler 15 → 1 (1985)
→ 9 today; Ali 4 → 33 monotone decay; Canelo 440 (2005) → 1 (2020) → 5;
Bivol unranked (2010) → 515 (2015) → 87 (2020) → 1 (2025–26); Jake Paul
peaks at #91. A single final-year column was hiding all of these shapes.

**6. Caveats surfaced while running.**
- Seed rot: dozens of seed URLs 404 or resolve to tableless pages (e.g.
  `Danny_Garcia` lands on a different person). Completeness remains
  era-and-notability biased, as suspected — snowball crawling stays the fix.
- Identity: raw rows keep crawler strings (`Lamont Roach Jr.`), the graph
  normalizes (`Lamont Roach Jr`). Entity resolution is still rule-based;
  fine rank differences below ~50 should not be over-read until an id pass
  exists.

**Next** (unchanged priorities): ensemble/bootstrap uncertainty bands around
the trajectories; Ring-ratings benchmark as tuning objective; Bradley–Terry
second opinion; snowball crawl depth 1–2.

— agent (ox-alpha), topbox session 2026-08-22

---

## 2026-08-22 — readme full picture + commit

Context, from `mat.md` (mat): comfortable with the take — update the readme
with concise insights of the full picture, then commit.

- README updated: methodology numbers refreshed (480 seeds, 8,149 bouts,
  parquet committed), both lens tables refreshed from the regenerated crawl,
  new section "Full Picture — Insights from Year-End Snapshots" with the four
  findings (peak parity, decline cohorts, draw-farming counterfactual,
  network inertia) plus a decadal trajectory table; limitations now include
  seed rot and rule-based identity caveats.
- Committed as `c33aa56` on master: code, tests, data (parquet + trajectory
  digest + regenerated lens CSVs), dialog logs. The 8 MB regenerable
  `topbox_timeline.csv` is gitignored on purpose (`python -m topbox.run_timeline`
  rebuilds it in seconds); repo-local git identity set to mat's existing
  author email.

— agent (ox-alpha), topbox session 2026-08-22
