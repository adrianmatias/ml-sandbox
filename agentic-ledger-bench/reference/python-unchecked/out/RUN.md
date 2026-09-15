<!-- Historical record of the original run. The absolute paths below are from
     the author's machine at the time; the equivalent commands for this
     repository are in ../README.md. -->

# Python ledger — how to run it (2026-09-15)

## Environment

| | |
|---|---|
| Python | **3.14.7** (`/usr/bin/python3`), from the system interpreter |
| Third-party dependencies | **none** — standard library only (spec §9.1) |
| Test runner | `unittest` (stdlib). `pytest` is not installed and was not added |
| Virtualenv / packaging | none; the module is run from source, no install step |
| Toolchain setup time | **0 s.** No interpreter, package or toolchain was installed |
| Type checker / linter | **none available.** `mypy`, `pyflakes` and `ruff` are all absent from this interpreter |

All commands below are run from the module root:

```
/home/mat/Documents/obsidian/mat/lab/python-wt/python
```

## The CLI (spec §6)

```
python3 -m ledger baseline <statement.csv>     # §3 envelope on stdout
python3 -m ledger accounts <accounts.json>     # §5 envelope on stdout
python3 -m ledger --help
```

Run from the module root, so `ledger` is importable as a package. (`python3
ledger/cli.py` does not work — the package uses relative imports by design.)

Exit codes: `0` well-formed input (even when checks report violations), `1`
input that cannot be parsed at all, `2` usage error.

## Exact commands that produced `out/`

```sh
cd /home/mat/Documents/obsidian/mat/lab/python-wt/python

# out/baseline.json — the real reference statement (spec §3, §8)
python3 -m ledger baseline ../raw/kutxa_movimientos_2026-05-25.csv > out/baseline.json

# out/accounts.json — the shared cross-language iteration-A fixture
# (CLARIFICATIONS.md §2; the fixtures live beside the frozen spec, outside this
# git worktree, so this path is absolute)
python3 -m ledger accounts /home/mat/Documents/obsidian/mat/lab/spec/shared-fixtures/accounts.json > out/accounts.json
```

Each exits `0`. Both files are written by the CLI and were not hand-edited; the
`reconciled` change from CLARIFICATIONS.md §1 was applied by re-running the
first command, not by editing the JSON.

## Tests

```sh
cd /home/mat/Documents/obsidian/mat/lab/python-wt/python
python3 -m unittest discover -s tests        # 133 tests
python3 -m unittest tests.test_baseline      # 74 tests (spec §3, §4, §6, §8)
python3 -m unittest tests.test_accounts      # 59 tests (spec §5)
python3 -m unittest discover -s tests -v     # per-test names
```

`python3 -m unittest discover -s tests -t .` fails with `Start directory is not
importable`; run discovery without `-t` from the module root (recorded as
tooling friction in NOTES.md).

## Reproducing the verification by hand

```sh
# the reference-file numbers of spec §8, straight out of the artifact
python3 - <<'EOF'
import json
c = json.load(open('out/baseline.json'))['data']['checks']
print(c['rowCount'], c['skippedEmptyRows'], c['reconciled'],
      len(c['valueDateInversions']), len(c['continuityErrors']),
      len(c['cumulativeErrors']), c['finalBalance'])
EOF
# -> 923 [2] True 29 0 0 37713.30

# the shared-fixture numbers of CLARIFICATIONS.md §2
python3 - <<'EOF'
import json
d = json.load(open('out/accounts.json'))['data']
print(len(d['accounts']), d['transferCheck']['pairs'],
      d['netWorth']['finalNetWorthFromAmounts'],
      d['netWorth']['finalNetWorthFromBalances'],
      [a['closingBalance'] for a in d['accounts']])
EOF
# -> 3 2 1139.25 1139.25 ['455.25', '429.50', '254.50']
```

## Layout

```
python/
  ledger/               # production module, 11 files, 1357 lines
    strictdecimal.py    #   exact decimal amounts, canonical rendering (§4.1)
    dates.py            #   strict DD/MM/YYYY (§4.2)
    csvio.py            #   line reader + minimal RFC-4180 tokeniser
    model.py            #   domain values + statement parsing (§2, §3, §4)
    checks.py           #   the three baseline checks (§3)
    accounts.py         #   iteration A: accounts + transfers (§5)
    envelope.py         #   frozen JSON envelope, exit codes (§6)
    errors.py           #   frozen error kinds (§6)
    cli.py              #   argument handling (§6)
    __main__.py         #   python3 -m ledger
    __init__.py
  tests/                # 2 files, 1735 lines, 133 tests
    fixtures/           # 12 §4 edge-case fixtures
    fixtures/accounts/  # 6 files: 3 statements + 3 registries for iteration A
  out/                  # baseline.json, accounts.json, RUN.md
  notes/NOTES.md
```

The default registry statement path is `<id>.csv`, resolved relative to the
registry file's own directory, when an account entry has no `statement` field.
