# Reference implementations

Three implementations of `benchmark/ledger-spec.md`, written independently by three
agents that never saw each other's code. All three produce identical output on the
benchmark data (0 mismatches across 923 transactions × 8 fields).

| directory | language / setup | why it exists |
|---|---|---|
| `python-unchecked/` | Python, no static analysis | the "before" arm: what Python looks like with no type checker, no linter |
| `python-tooling/` | Python + `ty` + `ruff` + `semgrep` | the **fair** arm: Python with its tool stack enforced from the start |
| `scala/` | Scala 3.9.0 LTS | the typed-language arm |

`python-unchecked` is run as a module because it has no top-level entry script:

```bash
cd python-unchecked && PYTHONPATH=. python3 -m ledger baseline ../../benchmark/data/statement.csv
```

Each directory has its own README with the exact commands. `docs/FINDINGS.md` has the
metrics and the verdict.
