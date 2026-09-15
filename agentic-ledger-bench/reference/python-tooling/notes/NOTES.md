# Python ledger — run notes (2026-09-15)

Arm: **Python 3.14.7 with `ty` + `ruff` + `semgrep` as mandatory gates** (EXPERIMENT.md).
Workspace: `lab/fair-py-wt/python/`. Toolchain install time: **0 s** — all three binaries were
pre-installed at `lab/.tools/bin`; nothing was downloaded, and no dependency was added to the module.

## Toolchain

- `ty 0.0.81` — `ty check --output-format concise .` — strict, warnings are errors (`./ty.toml`).
- `ruff 0.16.7` — `ruff check --no-cache .` — the curated real-bug set in `./ruff.toml`.
- `semgrep` — `--metrics=off --config=p/python python/`, all cache/config paths redirected into
  `lab/.tools/semgrep-*` because `/home/mat` outside the workspace is read-only.
- All three are wrapped by **`python/check.sh`** (exit non-zero if any gate fails). Run:
  `bash python/check.sh` → `ALL GATES GREEN (ty, ruff, semgrep)`.
- `ruff.toml` and `ty.toml` are **byte-identical to how they shipped**
  (`ruff.toml` sha256 `bd45379…`, `ty.toml` sha256 `77a474a…`). `verify.py`, the spec, the raw CSV and
  the shared fixtures are also untouched (hashes in `out/RUN.md`).
- **Two `# noqa`s exist in the whole package, both in test helpers, neither in production code:**
  `E402` (a deliberate import after a `sys.path` insert) and `S603` (`subprocess.run` with a fixed
  argv of `sys.executable` + our own CLI). Every other finding was satisfied by changing code.
- Python run command: `python3` (3.14.7, `/usr/bin/python3`). No `pytest`, no `uv` needed.

## Baseline

- wall-clock, first command → all tests green: **≈ 9 min** (12:33:52 → 12:42:5x, 2026-09-15).
  First CLI output matching the real file's expected numbers: **12:38** (≈ 5 min in).
- compile/run attempts: **0 compile steps** (interpreted). CLI invocations before the numbers were
  right: **1** (the first run already produced 923 / `[2]` / `37713.30` / 29 / 0 / 0).
- test execution attempts: **6** in total, of which **4** were failure → fix cycles
  (runner/import wiring, then real assertion mismatches, then a genuine production bug, then the
  property-test expectation) and **2** were the final green runs.
- tests passed on FIRST execution (yes/no): **no**. The first two runs did not even import (path
  wiring), the third ran and failed on 7 assertions plus the property test.
- lines of production code: **2187** across `python/ledger.py` + `python/ledger/*.py`
  (`accounts.py` 616, `transfers.py` 279, `errors.py` 272, `checks.py` 240, `statement.py` 169,
  `amounts.py` 130, `domain.py` 112, `jsonshape.py` 111, `ledger.py` 107, `baseline.py` 70,
  `dates.py` 42, `textio.py` 29, `__init__.py` 10).
- source files in final module: **13** production `.py` + **8** test `.py` + **8** generated fixtures.
- external dependencies: **none**. Standard library only (`csv`, `json`, `hashlib`, `decimal`,
  `datetime`, `pathlib`, `argparse`-free hand-rolled CLI, `unittest`) — spec §9.1 satisfied without
  needing an exception.
- exact command that runs the CLI:
  `python3 python/ledger.py baseline raw/kutxa_movimientos_2026-05-25.csv`
- tests: `PYTHONPATH=python python3 -m unittest discover -s python -t python -p "test_*.py"` → **108 tests, OK**.

## Iteration A (multi-account + transfers)

- wall-clock: **≈ 3 min** of implementation (12:39 → 12:42), then the fixture/expectation fixes above.
- files added: `python/ledger/transfers.py`, `python/tests/test_accounts.py`,
  `python/tests/fixtures/{A,B,C}.csv`, `registry.json`, `registry_mismatch.json`.
  (`python/ledger/accounts.py` existed as a 60-line module during the baseline because the CLI
  imports both commands; it grew from that stub to 616 lines here. Stated plainly so the
  "files added vs modified" number is not flattering by accident.)
- files modified: `python/tests/make_fixtures.py`, `python/tests/test_checks.py`,
  `python/ledger/accounts.py`, `python/ledger/__init__.py` (unchanged in the end).
- did the compiler/type checker find any change site for you? (yes/no + concrete evidence):
  **No — and I want to be precise about that.** I wrote `accounts.py` and `transfers.py` before
  running any checker, so nothing *found* a change site for me. `ty` was run **once** during
  iteration A, immediately after both files were complete, and it reported exactly **one** error:

  ```
  python/ledger/transfers.py:103:24: error[invalid-argument-type]
      Argument is incorrect: Expected `Decimal`, found `Literal[0]`
  ```

  — `Amount(0)` instead of `Decimal(0)`, a real latent bug caught by the `NewType`. That is a
  change site, but it is a one-token fix found *after* the design, not a site the checker led me to.
  The actual `-250`/`+250` currency bug (below) was found by a test I wrote, not by any tool.
- free functions added that re-derive an amount/date (the §5 pressure): **one** — `value_date_gap_days`,
  and it is deliberate: it is the single place transfer date arithmetic happens. Nothing re-derives an
  amount; magnitude equality is `candidate.amount == -outflow.amount` and conservation sums the
  endpoints' own `Amount` values. No sixth helper appeared.
- tests that failed and why (the four cycles, honestly):
  1. `ModuleNotFoundError: No module named 'ledger'` ×2 — runner/`sys.path` wiring, not a product bug.
  2. **A real production bug.** `is_feasible_pair` (currency equality) was written and documented but
     **never called** by `find_transfers`; the candidate filter compared amounts and accounts only.
     `test_different_currencies_never_pair` failed with an EUR/USD pair. Fixed by actually consulting
     the currency map in the search. **No gate caught this** — see the tooling report below.
  3. Fixture/test-expectation defects of my own: the collision fixture's balances did not reconcile,
     `clean.csv` had no inversion although its docstring promised one, and physical CSV line numbers
     were off by one in three assertions. Fixed in the fixture generator and the assertions.
  4. The property test's original expectation was wrong in an interesting way — I asserted that a
     corrupted balance would poison every later row of the cumulative check. It does not: only the
     anchored row is off when the corruption is not at row 0 (where perturbing the anchor shifts
     `openingBalance` instead). The check was right; my mental model was wrong. The test now encodes
     the real behaviour, including the opposite `delta` sign for the anchor case.

## The three tools — what they actually did

Every finding, with line numbers as they were at the time, classified honestly. "First run" = the
first invocation of that tool against this package.

### `ty` (ty 0.0.81, strict: warnings are errors)

| # | First-run finding | Classification | What I did |
|---|---|---|---|
| 1 | `statement.py:48:15 unresolved-reference` — `IOError_` used when not defined | **real defect** | Would have been a `NameError` on the first unreadable file. The class was later renamed `InputError` (see the `N801`/`N818` note). |
| 2 | `statement.py:50:15 unresolved-reference` — same name | **real defect** | Same fix. |
| 3 | `transfers.py:103:24 invalid-argument-type` — `Expected Decimal, found Literal[0]` | **real defect** | `Amount(0)` → `Amount(Decimal(0))`. `NewType` over `Decimal` made a sloppy zero visible. |

Findings at first run: **3**. Findings remaining at the end: **0**.
Later, once `python/tests/` joined the checked tree, `ty` reported **12 more** diagnostics, all in
tests: **12 real defects** in the test code (subscripting `object`, `len(object)`, `int(object)`,
passing `dict[str, str|float|None|…]` where `dict[str, object]` was expected). All twelve were fixed
by replacing bare `assert isinstance(...)` narrowing (which `ty` deliberately does not treat as a
narrowing contract for the surrounding block) with typed helper functions that return the narrowed
type. Total `ty` findings across the whole run: **15**, all real, none suppressed.

### `ruff` (ruff 0.16.7, `ruff.toml` as shipped)

Findings at first run over `python/`: **43** (`ruff check .` at the repo root: 63, the difference
being the 20 pre-existing findings in the orchestrator's `verify.py`). Findings remaining at the
end: **0 in `python/`**; the 20 in `verify.py` are untouched and out of this arm's scope — the file
is read-only, and modifying it would be tampering with the arbiter.

| Rule | Count | Where | Classification | Resolution |
|---|---|---|---|---|
| `TRY003` | 39 | `amounts.py`, `dates.py`, `statement.py`, `accounts.py` | **39/39 false positives** | See below — this is the one rule I consider misapplied. |
| `I001` | 2 | `ledger/__init__.py`, `statement.py` | style/consistency only | Reordered (import sorting is purely cosmetic). |
| `RUF007` | 2 | `checks.py:149,198` | style/consistency only | `zip(xs, xs[1:])` → `itertools.pairwise(xs)`; genuinely clearer. |
| `N801` + `N818` | 2 | `errors.py:57` | style/consistency only | `IOError_` → `InputError` (it was an ugly name; the linter was right for the wrong reason). |
| `E501` | 1 | `transfers.py:165` | style/consistency only | Split the sort key into a named function. |
| `B007` | 1 | `transfers.py:219` | **real defect smell** | An unused loop variable revealed that the loop should iterate entries as a unit; restructured, and it reads better. |
| Later (`python/tests/`) | | | | |
| `S101` | 25 | tests | style/consistency only | Replaced the 34 bare `assert isinstance` statements with typed narrowing helpers (`as_dict`/`as_list`/`as_str`/`as_int`) — better error messages, and it fixed the 12 `ty` errors above. |
| `TRY003`/`TRY004` | 7 | `tests/helpers.py` | **false positives** | Same rule, fixed the same way as production: shape errors became classes that own their messages. |
| `S311` | 3 | `tests/test_checks.py` | **false positive in context, but the rule was right about the code** | The property tests wanted a *reproducible* stream, so they now use a self-contained LCG (`SequenceSource`) instead of `random`; determinism is now explicit rather than promised. |
| `T201` | 1 | `tests/make_fixtures.py` | **real defect (rule intent)** | `print` in a module → `sys.stdout.write` in the `__main__` block. |
| `UP031`, `E501` | 3 | `tests/test_checks.py` | style/consistency only | Percent formats → f-strings; wrapped long lines. |
| `F401` | 2 | tests | style/consistency only | Removed unused imports (one of them was `OpeningBalance`, unused since a refactor). |
| `S603` | 1 | `tests/helpers.py:203` | **false positive** | Fixed argv, `sys.executable` + our own CLI; suppressed narrowly with a reason (`# noqa: S603`). |
| `E402` | 1 | `tests/helpers.py:17` | **false positive** | Import deliberately after a `sys.path` insert; suppressed narrowly with a reason (`# noqa: E402`). |

**On `TRY003` (39 findings, all false positives).** The rule asks that long messages not be passed
to an exception *from outside* the class. I agree with the intent and I satisfied it — but the check
fires on `raise Cls("any message")` even when `Cls` is defined in the same file and takes a *required*
message parameter. There is no way to satisfy it while keeping a message at the raise site, so I
restructured the error hierarchy: `AmountError`, `DateError`, `RegistryError` and their variants now
own their message text, and every raise is a one-line construction whose shape is designed by the
class (`raise AmbiguousAmountError(text, thousands_group=True, line=line)`). This is a genuine
improvement in the production code — message formats now live in exactly one place — but it is worth
recording that `TRY003` cost the most effort of any rule and found no bug.

### `semgrep` (`--config=p/python`, 151 rules)

Findings at first run: **0**. Findings remaining at the end: **0**. 13 targets scanned, ~100 % parsed,
0 findings throughout. It never flagged anything, not once, in either phase — including during
iteration A, when the currency predicate bug was live in the tree.

One operational note, recorded because it is a real trap for the next arm: `semgrep` scans **files
tracked by git** by default, and this worktree does not track `python/`. It still scanned the
directory because `.gitignore` does not match it, but a file that ever lands under an ignored path
becomes invisible to the gate. The `--config=p/python` run also crashed once inside `semgrep-core`
(a cache-lock race while another `semgrep` ran against the same `SEMGREP_CACHE_DIR`) and succeeded on
retry; a wrapper that treats that non-zero exit as a hard failure is correct behaviour, but the
operator should expect to re-run it.

## Did any tool catch something a test would not have caught?

**Honest answer: no — with one qualified exception, and the exception is a linter, not a type
checker.**

- **`semgrep`: nothing, ever.** 0 findings on 151 rules across both phases. On this codebase
  (`csv` + `decimal` + `json`, no network, no subprocess, no SQL, no crypto) it contributed exactly
  zero signal. It is a security/pattern scanner being pointed at code with no security surface.
- **`ruff`: yes, but nothing a careful review would have missed.** The one finding that pointed at
  real behaviour rather than style was `B007` (`transfers.py:219`), and it pointed at a *loop shape*,
  not at a bug. Every `ruff` finding in production code was either a false positive (`TRY003`) or
  cosmetic. It found nothing semantically wrong that my tests did not also cover.
- **`ty`: yes, three times in production, and all three were real.** `IOError_` used-before-defined
  (a guaranteed `NameError` on the error path — no test covered that path when it was found),
  `Amount(0)` where a `Decimal` was required, and later twelve test-code type errors. The
  `NameError` is the strongest example: the tests all passed while it sat in the tree, because no
  test had exercised the *unreadable file* branch. A type checker finds that class of defect
  statically and a test suite finds it only if someone thinks to write the test.
- **The bug that actually mattered was found by neither.** `is_feasible_pair` — the currency half of
  the §5.2 pairing predicate — was written, documented, and never called; transfers paired across
  EUR and USD. `ty` was happy (the function was perfectly well typed, just unused), `ruff` was happy
  (a defined-but-unused *public* function is invisible to `F401`, which only tracks imports),
  `semgrep` was happy. **A test caught it.** `ruff` does have `ARG` and `F841` for unused
  *arguments* and *locals*; a linter rule for a module-level function that is defined and never
  referenced anywhere (including by its own module) would have caught this, and its absence is the
  single most useful thing I learned from the arm.

## Did full annotation change how I designed the code?

**Yes, twice, and both times it changed the shape of the module rather than adding decoration.**

1. **`NewType` made boundary errors unrepresentable, and then found one.**
   `Amount`, `Balance`, `LineNumber` are `NewType`s over `Decimal`/`Decimal`/`int`, and the
   transaction model stores `amount: Amount` and `balance_after: Balance` separately. That is why
   `ty` could flag `Amount(0)`: a raw `int` zero is not a `Decimal`. Without the annotation the same
   expression is silently legal and produces `Decimal(0)` anyway, so the bug would have lain dormant
   — harmless here, but the same class of mistake with `str` money would not be.
2. **The parser became total, not exception-driven, because the annotations made the split obvious.**
   `parse_decimal`/`parse_date` are annotated to *raise* `AmountError`/`DateError`, and
   `ParsedStatement` carries `row_errors: tuple[RowFailure, ...]` next to
   `transactions: tuple[Transaction, ...]`. Writing the return type down forced the question "what
   happens to a bad row?" and the answer — collect it and keep going, because §3.4 forbids
   short-circuiting — fell out of the signature. My first draft returned `Decimal` and let callers
   wrap failures; the annotation made that look wrong before I wrote any caller.
3. **Conversely, one place where annotation actively fought me** (recorded for fairness): `ty`
   does not narrow through a bare `assert isinstance(x, dict)` for the code *after* the assert when
   the value comes from `json.loads`. The idiomatic test-suite pattern
   (`data = json.load(...); assert isinstance(data, dict); data["checks"]`) is an error under strict
   checking. The fix — typed narrowing helpers that *return* the narrowed type — is better code, but
   the fight was real and it cost the most rewriting of anything in the run.

## Friction

- **Spec friction (ambiguities I had to decide, and how).**
  1. *Malformed rows vs exit code 1.* §3.4 says a check must never stop at the first violation, §6
     says exit 1 when "the input cannot be parsed at all", and the envelope has room for exactly one
     `error`. I read "cannot be parsed at all" as structural (missing header, malformed quoting) and
     made **row-level** failures total: a bad row is skipped, recorded in a new
     `checks.parseErrors` array (kind `ROW`/`AMOUNT`/`DATE`, with its line), every good row still
     appears in `transactions`, and the process exits **1**. That last part is the one place where my
     output could surprise a reader: exit code 1 with `ok: true`, because the envelope carries data.
     The alternative — abort with a single `error` — discards 900 good rows to report one bad one and
     contradicts §3.4's spirit. The arbiter only reads `checks.*`, and the reference file parses
     cleanly, so this is invisible on the file that matters.
  2. *What "observation point" means for §5.5.* The spec says `netWorth(i)` and
     `netWorthFromBalances(i)` must agree "at every observation point" without defining one. I define
     a point as `(bookingDate, line)` of any row of any account, sort globally, and let an account
     contribute only from its first row onward (before that it has a declared opening balance but no
     observed balance, and inventing one would be a guess). Both totals are recomputed from scratch
     at each point, so neither derives from the other.
  3. *Which end of a pair is "nearest" when the tie-break is ambiguous.* §5.2 fixes the choice of
     *pair* (nearest date, then lowest line number) but not the order in which outflows are offered.
     I visit outflows by `(valueDate, account id, line)`, which is deterministic and independent of
     registry order, and the shared fixture's answer (2 pairs, `A:3→B:4` and `A:4→C:2`) is unaffected.
  4. *`1.234` in a broader locale.* §4.1 says reject it. `1,234,567.89` and `1.234.567,89` are
     accepted as grouped amounts, and `1234,56.78` is rejected as ambiguous, since the two
     conventions contradict each other inside one value.
  5. `reconciled` gets a small extension: per `CLARIFICATIONS.md` §1 it is `true` iff
     `continuityErrors` and `cumulativeErrors` are empty. I additionally require `parseErrors` to be
     empty — a statement with unreadable rows is not "reconciled" in any useful sense — and say so
     here rather than hiding it.
- **Language/tooling friction, with the concrete command or error.**
  - `semgrep` cannot start at all without redirections: `OSError: [Errno 30] Read-only file system:
    '/home/mat/.config/.semgrep'` — every path must be redirected into the workspace (EXPERIMENT.md's
    invocation is correct and was used verbatim).
  - `semgrep-core` crashed once with an OCaml backtrace while a second `semgrep` held the same
    `SEMGREP_CACHE_DIR`; it passed on the immediate retry with no change to the code.
  - `unittest` grabs `TestCase.run`, so a class attribute named `run` holding the parsed model made
    the whole discovery fail with `TypeError: 'BaselineRun' object is not callable`. Renamed to
    `baseline`. Pure Python trap, zero diagnostic value.
  - `python3 -m unittest discover` needs `-t python` and a package-style layout or the test modules
    cannot import the module under test; the first two runs were dead on arrival for this reason.
  - `ruff`'s `[lint.per-file-ignores] "tests/*"` does **not** match `python/tests/*` (patterns are
    relative to the config file), so the tests' `assert`s were linted as production code. I did not
    touch `ruff.toml`; I removed the bare asserts instead (see above), which is strictly better.
  - `ruff` at the repository root also lints the orchestrator's `verify.py` and fails on it
    (20 findings, e.g. `TRY003`, `PTH123`, `T201`). That file is read-only and is the arbiter, so
    `check.sh` scopes the gate: those findings are printed, and any finding under `python/` fails.
- **What was easier than expected.** `decimal.Decimal` never once surprised me: exact arithmetic,
  no float path, and `format_money` as the single serialisation point meant the canonical
  `37713.30` form never had to be debugged. The whole §3 baseline — parser, three checks, canonical
  JSON — produced `923 / [2] / 37713.30 / 29 / 0 / 0` on its **first** invocation, and matched the
  arbiter on the first `verify.py --check`.

## Assessment

- **Would you extend this codebase? Why.** Yes, with one caveat. The module is small, standard-library
  only, deterministic and total: adding a fourth account is a registry entry, and a new check is one
  function returning a tuple plus one line in the JSON writer. The caveat is the JSON layer
  (`jsonshape.py` and the `*_object` helpers): it is the only place where a field name is written as a
  string, so a rename there is not checked by anything, and the two commands each build their own
  transaction objects. If this grew, I would give the envelope a typed `TypedDict` shape and a single
  transaction serialiser.
- **What the second developer to touch this would trip on.**
  1. The exit-code rule: a statement with unparseable rows exits **1** while `ok` is still `true` and
     `data` is present. That is deliberate (see friction 1) and documented in the CLI docstring, but it
     is the first thing that looks like a bug.
  2. The error-class hierarchy in `errors.py` is deliberately chatty (one class per failure shape)
     because `TRY003` forbids message text at the raise site. Without that context it looks like
     over-engineering.
  3. `transfers.py` orders outflows by `(valueDate, account id, line)`; changing that order can change
     *which* pair is found when two outflows compete for one inflow. The pair-selection rule itself is
     fixed by the spec; the visitation order is my choice and it is load-bearing for determinism.
  4. `python/tests/fixtures/` is **generated** by `python/tests/make_fixtures.py` (it needs byte-exact
     CRLF and BOM control). Editing a fixture by hand works until the next generator run, and the
     reference fixture `raw/kutxa_movimientos_2026-05-25.csv` must never be regenerated or edited.
