# Python with the tool stack enforced

The fair arm. Same specification as every other implementation, but `ty`, `ruff` and
`semgrep` are **mandatory gates** that must be green on the final code. This is the
arm that answers "would Python have been fine if the tools had been on from the
start?" — the original comparison ran without them, which was not a fair test.

Standard library only: **zero runtime dependencies**.

## Running

```bash
python3 ledger.py baseline ../../benchmark/data/statement.csv
python3 ledger.py accounts ../../benchmark/data/accounts.json
PYTHONPATH=. python3 -m unittest discover -s tests -t . -p "test_*.py"
```

## The gates

```bash
ty check --output-format concise .     # strict: warnings are errors (./ty.toml)
ruff check --no-cache .                # curated real-bug set (./ruff.toml)
semgrep --metrics=off --config=p/python .
```

`check.sh` runs all three. Both `ruff.toml` and `ty.toml` are **in this directory** and are
byte-identical to the ones the agent was gated with — they were missing from the published tree until
§29.3, which meant the lint gate could not be reproduced from a clone. Re-run with them restored,
this arm is **clean**: `ruff check .` reports nothing, and the same rule set over
`../python-unchecked` reports **138 findings**. There are **zero suppressions in production code**:
no `# type: ignore`, no `# noqa` outside two justified ones in test helpers.

`check.sh` runs all three gates from this directory and works on any machine: tools come from `PATH`
or are installed on the fly with `uvx`, on the running interpreter — `uvx ty` under Python 3.11
reports `Cannot use 'type' alias statement on Python 3.11` for a file this arm is entitled to write,
which is an interpreter artefact and not a defect in the arm. `semgrep` writes settings derived from
`$HOME/.semgrep` on startup, so `check.sh` redirects every path it touches under `.semgrep-home/`
(ignored by git) rather than failing on a read-only `$HOME`.

## What the gates caught, and what they missed

- `ty` found **3 real defects** on first run, including a name used before definition
  that would have raised `NameError` on a path **no test covered**.
- `ruff` found 43; **one** was a real smell. 39 were `TRY003` false positives, which
  the agent satisfied structurally (error classes own their messages) rather than
  suppressing.
- `semgrep` found **nothing**, first run and last.
- The defect that actually mattered — a currency predicate defined, documented and
  **never called**, so EUR could pair with USD — was caught by a **test**, not by any
  static tool. Details in `docs/FINDINGS.md` §11.

The per-tool ledger with rule codes and a real-defect / false-positive / style
classification is in `notes/NOTES.md`.
