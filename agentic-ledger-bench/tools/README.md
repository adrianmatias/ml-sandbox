# tools — the gates, runnable

Two scripts, no dependencies beyond a Python 3.12+ interpreter (and `ruff`/`vulture` for the drill,
picked up from `PATH` or installed on the fly with `uvx`).

## `complexity.py` — a budget, not a report

```bash
python3 tools/complexity.py reference/python-tooling               # gate: exit 1 when over budget
python3 tools/complexity.py --budget 12 --test-budget 15 <paths>   # explicit budgets
python3 tools/complexity.py --json reference/python-tooling        # machine-readable, for CI tables
```

McCabe per function, with the decision set `radon` uses (if/elif, for, while, `and`/`or` as separate
decisions, ternary, `except`, comprehension `if`s, `match` cases beyond the first), so `--json`
output is comparable with `radon cc` on the same tree. Nested defs are their own unit. Tests are
gated too, at a looser budget, because a 40-branch test helper is a maintenance problem even when it
is not production code.

When a function is over budget, **split it**. Do not raise the budget for one file — one-off
exceptions rot into permanent rot.

## `gate-drill.sh` — a gate you have not seen fail is not a gate

```bash
bash tools/gate-drill.sh                       # defaults to reference/python-tooling
bash tools/gate-drill.sh <arm-dir>
```

For each gate the drill grades two rows: **green** on the clean tree, **red** on a scratch copy
carrying the defect that gate exists to catch.

| gate | planted defect |
|---|---|
| `ruff` | an unused import |
| `complexity.py` | a 13-branch function against a budget of 12 |
| `vulture` | a documented function that is never called — the recorded `is_feasible_pair` class |

Plus the control that makes the point: the same dead function at **`--min-confidence 80` is
invisible**. Vulture reports unused functions at 60% confidence, so an 80 threshold reports nothing
at all, for any codebase, and exits 0. That is how a dead-code gate becomes a no-op while reading as
clean.

Nothing in the repository is modified: plants live in a scratch copy under `mktemp -d`, removed by a
trap on any exit.

## Known trap, handled by both scripts

A tool that cannot parse the tree must never be graded on it. `vulture` under an interpreter older
than 3.12 prints one `invalid syntax` line for a PEP 695 `type X = ...` alias, skips the file, and
its output otherwise reads like a clean scan — the same silent-skip class as `radon`. The drill
resolves tools against the running interpreter (`uvx --python <current>`) and reports **SKIP** rather
than grading a tool that never parsed the code.
