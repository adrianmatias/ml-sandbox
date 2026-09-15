# Python ledger — run notes (2026-09-15)

Timeline (UTC, all measured, `date -u` at each checkpoint):

| checkpoint | time |
|---|---|
| first command (`pwd`, `date`, read the spec) | `09:52:14Z` |
| baseline: all tests green | `09:56:45Z` |
| iteration A: all tests green | `09:59:51Z` |

Both windows include thinking/reading time, not only keystrokes.

---

## Toolchain

- **Python 3.14.7** at `/usr/bin/python3`. Nothing was installed, downloaded or
  compiled: no pip, no venv, no `uv` invocation.
- **Time spent installing/starting a toolchain: 0 s** (counted separately, per
  §9.5). `uv` 0.12.5 is present but unused — the stdlib was sufficient.
- Commands used:
  - `python3 -m py_compile ledger/*.py` — syntax check
  - `python3 -m ledger baseline <csv>` / `python3 -m ledger accounts <json>`
  - `python3 -m unittest discover -s tests`
  - `python3 - <<'EOF' … EOF` heredocs for one-off probes of the raw file
- **No type checker and no linter is available**: `python3 -m mypy`,
  `python3 -m pyflakes` and `python3 -m ruff` all report `No module named …`.
  So there is no static-analysis signal in this language at all; the test suite
  is the only mechanical check. This is the single biggest asymmetry with the
  Scala side and it shows up directly in the metrics below.

---

## Baseline

- **wall-clock, first command → all tests green: 4 min 31 s** (09:52:14Z →
  09:56:45Z).
- **compile/run attempts: 11**, broken down honestly:
  - 1 `py_compile` — passed first time, no syntax errors.
  - 1 CLI run — failed, but only because `ledger/cli.py` imports
    `ledger.accounts`, which did not exist yet as anything but a stub. Fixing it
    took a 12-line stub, not a parser change.
  - 8 `unittest discover` runs against the baseline suite.
  - 1 CLI run that failed on a wrong relative path to the shared fixtures.
- **test-failure → fix cycles: 6.** At baseline green the suite was **74 tests**
  and it passed on run 8. The first execution was `Ran 68 tests … FAILED
  (failures=11, errors=5)`.
- **tests passed on FIRST execution: no.**
- **lines of production code: 916** at baseline green (the 10 files of `ledger/`
  excluding the 12-line `accounts.py` stub); **1357** across **11 files** after
  iteration A. Both are `wc -l` over `ledger/*.py`, so docstrings and blank
  lines are included.
- **source files in final module: 11** (`ledger/`), of which 1 was added by
  iteration A (`accounts.py`; the stub it replaced is counted as modified, not
  added).
- **external dependencies: none.** Standard library only (`decimal`,
  `datetime`, `hashlib`, `json`, `unittest`, `subprocess`, `tempfile`,
  `random`). Nothing to record under §9.1's "smallest standard option".
- **exact command that runs the CLI:**
  `python3 -m ledger baseline ../raw/kutxa_movimientos_2026-05-25.csv`

### What the 6 fix cycles were

**Four were bugs in my code or fixtures:**

1. The ambiguity rule was checked *after* the two-decimal rule, so `1.234` was
   rejected as "too precise" instead of as **ambiguous** — the exact diagnostic
   §4.1 cares about. Also `canonical()` called `Decimal.quantize` under the
   default 28-digit context, so a 32-digit value raised `InvalidOperation`.
   Python's `Decimal` is a *context-governed decimal float*, not an
   arbitrary-precision type; both had to be fixed at `decimal.MAX_PREC`.
2. `1.234,56` was rejected as `"amount is not numeric"` because a structural
   pre-check ran before the dual-separator test. Reordered so the ambiguity
   diagnostic is reachable.
3. My `comma_decimal.csv` fixture was itself invalid: `1234,56` unquoted in a
   comma-delimited file is six fields, not four. The fixture now quotes it, and
   a separate test asserts the unquoted form is a `ROW` error — the honest
   behaviour, since the delimiter makes it genuinely unparseable.
4. My collision fixture could not satisfy both requirements at once. Identical
   `(bookingDate, amount, balanceAfter)` pairs *and* §3.1 continuity are only
   simultaneously possible if the colliding pair is separated by a compensating
   pair returning to the same balance — which is exactly the shape the real file
   uses at 243/247, 288/290 and 446/448. Two earlier versions of this fixture
   were wrong and were thrown away.

**Two were wrong expectations in my tests** (the implementation was right):

5. I asserted that the two §3 checks would agree on which rows are wrong. They
   never do: one bad `saldo` produces `[3, 4]` from §3.1 (local, re-anchors) and
   `[3, 4, 5]` from §3.2 (never re-anchors). The test now *encodes* that
   divergence as the evidence of §3.5 independence, which is a better test than
   the one I meant to write.
6. I asserted colliding rows would have different `checksum` values. They must
   not: the checksum is a content fingerprint over
   `amount|balanceAfter|bookingDate`, so genuinely distinct rows with identical
   content share it. That is the sharper finding — neither the natural key *nor*
   the checksum can be a dedup key, only the line number can. Assertion
   inverted.

One more I mis-diagnosed: I wrote a helper that read `.strip()` to detect an
empty row, but `",,,".strip()` is `",,,"`, not `""`. My probe was wrong, not the
parser; the parser was already skipping line 2 correctly.

**Also worth recording as a dead end:** the first draft of `ledger/accounts.py`
was written and then deleted unread. It contained two half-finished functions
(a `_feasible_pairs` whose sort key was built with a nonsense `and`-chained
expression, and a `_net_worth_from_balances` called twice, once from inside a
generator that recomputed the same thing). Rather than patch it, the file was
rewritten from scratch. That whole draft cost roughly 3 minutes and produced
nothing.

---

## Iteration A (multi-account + transfers)

- **wall-clock: 3 min 6 s** (09:56:45Z → 09:59:51Z) for the first green
  iteration-A run. A later, separate round of **~1 min** was needed to absorb
  CLARIFICATIONS.md §2 (the shared fixture). Baseline and iteration A are
  therefore cleanly separable: 4 m 31 s vs. 3 m 06 s.
- **files added: 7** (verified against the tree, not estimated)
  - `tests/test_accounts.py` (842 lines, 59 tests)
  - `tests/fixtures/accounts/` — `current.csv`, `savings.csv`, `joint.csv`,
    `registry.json`, `registry_two.json`, `registry_currency_mismatch.json`
    (6 files: 3 statements + 3 registries)
  - plus `out/accounts.json` as an artifact, and temporary multi-account
    fixtures that the property tests generate in temp dirs at run time (not
    checked in)
- **files modified: 4**
  - `ledger/accounts.py` — the 12-line stub replaced by the real module
    (423 lines)
  - `ledger/checks.py` — `build_checks` gained an optional `opening_balance`
    override, and `reconciled` was redefined per CLARIFICATIONS.md §1
  - `tests/test_baseline.py` — three tests updated for the clarified
    `reconciled` definition
  - `out/baseline.json`, `out/accounts.json` regenerated (not hand-edited)

  So: 7 files added, and 3 pre-existing source/test files touched
  (`accounts.py`, `checks.py`, `test_baseline.py`). Note that two of iteration
  A's three source edits landed in files the baseline had already written — the
  extension was not containment-clean.
- **did the compiler/type checker find any change site for you? — no.**
  Concretely: there is no type checker installed (`python3 -m mypy` →
  `No module named mypy`), and iteration A required **zero changes** to
  `model.py`, `csvio.py`, `strictdecimal.py` or `dates.py` — so even a compiler
  would have had nothing to report, because the extension reused the existing
  signatures unchanged. The change sites were found by grepping and reading:
  `grep -n "build_checks" ledger/*.py` to find the one caller,
  `grep -rn "reconciled"` to find the boolean's dependants, and reading
  `model.py`/`checks.py` to confirm `ParsedStatement` already exposed
  everything §5 needed. Note the counter-case: the `opening_balance` bug (below)
  is *exactly* the kind of thing a type checker would not have caught either,
  since both the registry value and the derived value are `Decimal`.
- **free functions added that re-derive an amount/date (the §5 pressure):**
  one site. `Movement.magnitude` re-derives `|amount|` by sign-flipping the
  `Decimal` rather than giving an outflow a typed absolute value. It is reported
  in the artifact as `data.internals.amountOrDateRederivationSites = 1`;
  the run counter `amountOrDateRederivations = 68` is deliberately reported
  alongside it but is a much weaker number (it is inflated by the pairing loop's
  O(outflows × inflows) comparisons). No free function re-derives a *date*: the
  one date computation, `days_between`, is a helper in `dates.py` used by the
  baseline too. I did not reach the "sixth free function" stage — the pairing
  predicate stayed inside one function — but the pressure the spec predicts is
  real and visible in `magnitude`.
- **tests that failed and why: 9 failing assertions across 3 `unittest` runs
  (133 tests at green).** Two of them found **real bugs in my iteration-A code**,
  which is the useful part:
  1. `build_checks` ignored the registry's `openingBalance` and re-derived the
     anchor from row 1, contradicting the design I had documented in the same
     file. Caught by
     `test_registry_opening_balance_anchors_the_cumulative_check`. Fixed by
     adding the explicit `opening_balance` override parameter — a *shared* file
     (`checks.py`) had to change, which is exactly the kind of ripple the
     exercise is measuring.
  2. `net_worth_observations` computed **both** sides from the same
     amount-accumulated list, so §5.5's check was vacuous — it could never
     report anything. Caught by
     `test_net_worth_mismatch_is_reported_at_every_point_after_the_defect`, and
     only caught because that test asserts on a deliberately corrupted balance.
     Fixed by reading the second side from each account's recorded
     `balanceAfter` column.
  The other 7 were wrong expectations in the new tests (line numbers off by one,
  a helper that silently discarded per-account opening balances, and a fixture
  whose value dates went *forward* when the test needed them to go backwards).

---

## Friction

### Spec friction (ambiguities, decisions I had to make)

1. **`reconciled` and value-date inversions — the call I had to make, later
   clarified.** §3 defines `reconciled` as "true iff every array below is empty",
   and `valueDateInversions` is one of the arrays below. I read that literally
   and made `reconciled` **false** for the reference file, even though §3.1/§3.2
   are clean. It was an uncomfortable reading — it makes every real statement
   unreconciled forever, since a value-date inversion is a fact about the data
   that no amount of checking can remove — but it was the literal text, so I
   implemented it and noted the discomfort here rather than silently choosing
   the nicer reading. **CLARIFICATIONS.md §1 then resolved it the other way**:
   `reconciled` is now true exactly when `continuityErrors` and
   `cumulativeErrors` are empty, so `out/baseline.json` reports `reconciled:
   true` alongside 29 inversions. Recorded because it is precisely the kind of
   ambiguity the spec asked to be logged, and because it was unguessable from
   the spec text alone.
2. **The reference file has no BOM**, contrary to §2's "UTF-8 **with BOM** on
   the first line". `od -An -tx1` gives `66 65 63 68` (`fech`), and the first
   line is plain `fecha,fecha valor,importe,saldo`. Handled by reading with
   `utf-8-sig`, which strips a BOM when present and is a no-op when absent, so
   both readings work. I did not treat the mismatch as an error.
3. **Header field spacing.** "exactly four columns" is silent on whitespace, so
   header and data fields are `.strip()`ed before validation. A padded header
   therefore parses; a padded amount `" 400 "` parses as `400.00`.
4. **`openingBalance` overrides the derived anchor in accounts mode.** §3.1
   defines `openingBalance = balanceAfter[0] - amount[0]`, while §5 supplies an
   `openingBalance` per account in the registry. I made the registry value
   authoritative (§5's input is explicit data; §3's formula is a fallback), which
   means an account whose registry opening disagrees with its first row now
   produces cumulative errors — a behaviour change relative to a pure §3
   reading. It is identical on the reference file (both are `0.00`) and on the
   shared fixture (all openings `0.00`).
5. **`unpairedTransferCandidates` semantics.** §5.3 says "leave at `0`" and also
   names it a "reserved counter". I emit `0` unconditionally and documented it as
   reserved rather than computing a real count, reading "leave at 0" as the
   binding instruction.
6. **Blank physical lines.** §2 requires one *fully empty row* (`,,,`) to be
   skipped. A completely blank line is not mentioned. I treat it as an empty
   record and skip it too — a strict broadening that cannot change the reference
   file's result (it has no blank lines) and matches the intent.
7. **Comma-delimited file vs. comma decimal.** §4.1 requires accepting `1234,56`
   as an amount, but §2's file is comma-delimited, where an *unquoted* `1234,56`
   is genuinely two fields and no parser can recover the intent. Resolved by
   supporting comma decimals in `parse_amount_token` (used for the registry and
   for quoted fields) and reporting `ROW` for the unquoted form in a statement.
   A parser that guessed here would be reinterpreting the delimiter.
8. **`1.2345`.** §4.1 makes exactly-three-trailing-digits ambiguous. Four
   trailing digits are therefore unambiguous but not representable at two decimal
   places; rejected as `"amount has more than two decimal places"`, a different
   `message` under the same frozen `AMOUNT` kind.
9. **Two fields beyond §3's table.** `checks.openingBalance` and
   `checks.finalBalance` are emitted in addition to the six specified fields.
   §8 makes the final balance an explicitly asserted number, so exposing it in
   the artifact is useful; both are additive and neither reorders the frozen
   keys. Flagged in case the Scala side matches field-for-field.
10. **Exit code for unparseable rows.** §6 says `1` "when the input cannot be
    parsed at all" and §3.4 forbids short-circuiting — but §3.4 governs the three
    *checks*, not the parser. I read "at all" per-message: the first unreadable
    row terminates parsing and emits the error envelope with its line number.
    The alternative (report every bad row) has no representation in the frozen
    envelope, which carries a single `error` object.

### Language/tooling friction, with the concrete command or error

1. **`python3 -m unittest discover -s tests -t .` → `ImportError: Start
   directory is not importable: '…/python/tests'`.** Discovery needs `tests/` to
   be a package when an explicit top-level directory is given. Resolved by
   dropping `-t`: `python3 -m unittest discover -s tests` works from the module
   root and finds both test modules. (There is no `tests/__init__.py`; the test
   modules `sys.path.insert` the module root themselves.)
2. **`decimal.Decimal` is not arbitrary precision by default.** The spec demands
   arbitrary-precision money, and `Decimal` looks like it delivers that, but
   every operation is governed by a context whose default is **28 significant
   digits**. Two concrete symptoms, both hit:
   - `canonical(Decimal("123456789012345678901234567890.12"))` →
     `decimal.InvalidOperation` from `quantize`.
   - `Decimal("1.005").quantize(...)` silently rounds unless the context is
     raised.
   A binary `float` cannot enter the model (that part is enforced at one entry
   point, `strictdecimal.as_money`), but the *second* trap — silent rounding at
   28 digits — is invisible in the type name. Every quantise and every sum now
   runs under `decimal.localcontext()` with `ctx.prec = decimal.MAX_PREC`, and
   `exact_sum` asserts the result's exponent.
3. **No static analysis available at all.** `No module named mypy` / `pyflakes`
   / `ruff`. In practice every error in this build was found by a test or by
   reading, never by tooling. The `opening_balance` override bug is the clearest
   example: a type checker would have seen `Decimal` on both sides.
4. **`csv` unavailable by choice.** The stdlib `csv` module exists but its
   error behaviour on unbalanced quotes (`_csv.Error` at an arbitrary point, or
   silent continuation-line merging) maps badly onto the frozen `ROW` kind, so
   `csvio.py` implements a 40-line explicit tokeniser. This is a *choice*, not a
   stdlib gap — recorded as such.

### What was easier than expected

- **Exact decimal arithmetic and canonical rendering were trivial.** `Decimal`
  plus `format(quantize(Decimal("0.01")), "f")` gives `37713.30` and never an
  exponent, with no library and no configuration. Making a float *structurally*
  unable to enter the domain model took one guard in one function, and the
  property test that respells 200 values through five spellings is five lines.
- **The domain model needed no change for iteration A.** `ParsedStatement`
  already carried `transactions`, `skipped_empty_rows` and `opening_balance`, so
  §5 could be layered on top by reading the existing types. Zero lines changed
  in `model.py`, `csvio.py`, `strictdecimal.py`, `dates.py`.
- **`datetime.date` gives real calendar validation for free.** There is no
  roll-over to suppress: `date(2026, 2, 31)` raises, so §4.2's "do not use a
  lenient parser" only required rejecting the wrong *shapes*.
- **Deterministic JSON is free** if no dict is ever built from a `set` — plain
  dict insertion order plus `json.dumps(..., sort_keys=False)`. Two runs are
  byte-identical, which is asserted rather than assumed.

---

## Assessment

### Would you extend this codebase? Why / why not

**Yes, with one reservation.** The layering is honest: parsing produces values
that cannot be invalid, the checks are pure functions over those values, and the
envelope is a separate concern, so iteration A genuinely was an addition rather
than a rewrite — three files touched, two of them new, and the parser untouched.
The reservation is that the type system carries almost none of this. The
guarantees that make the code pleasant — "a `Transaction` cannot hold an
unparsed amount", "an amount is never a float", "`balance_after` came from the
file", "`opening_balance` came from the registry" — all live in docstrings, in
one runtime guard, and in tests. Nothing mechanical stops the next person from
writing `Transaction(amount=Decimal(str(some_float)))` and quietly reintroducing
binary floats, and §5's `opening_balance` bug (a `Decimal` silently replaced by a
different `Decimal`) is proof that even the types that *are* there do no work.

### What the second developer to touch this would trip on

1. **`Decimal` context precision.** Most likely single mistake. Someone will
   write a fresh `sum(...)` or `.quantize()` outside `localcontext()` and get
   either silent rounding at 28 digits or a bare `InvalidOperation`, with a
   traceback that names `decimal` and not their code. It is guarded in
   `strictdecimal.py` but only for the helpers there.
2. **The `openingBalance` override in `build_checks`.** `build_checks(statement)`
   and `build_checks(statement, opening_balance)` differ in *semantics*, not just
   in an added parameter: the first anchors on row 1, the second on the registry.
   Passing the wrong one produces plausible, wrong cumulative errors. Whether
   `opening_balance` should be a field of `ParsedStatement` (set by its
   constructor from either source) is a real design question I left open.
3. **Silent byte-identical absence.** §2 says the file has a BOM; it does not.
   Anything that reads statements with a mode other than `utf-8-sig` will
   prepend `\ufeff` to the first header cell and fail the header check on a file
   that looks perfect in an editor.
4. **The `rawAmount`/`rawBalance` contract.** They are the file text minus CSV
   quoting and surrounding whitespace, not the raw bytes. For a quoted
   `"1234,56"` the recorded `rawAmount` is `1234,56` (no quotes). This is
   documented but easy to misread as "byte-exact".
5. **`unpairedTransferCandidates` is a lie by design.** It is always `0` because
   §5.3 said to leave it there. A reader will assume it counts something.
6. **The collision-shaped data.** Three real pairs of rows in the reference file
   are indistinguishable except by line number, and their checksums are equal by
   construction. Anyone who adds deduplication, or keys a dict on the checksum
   (a natural-looking choice for an idempotent import), will **silently drop
   three real transactions**. This is the sharpest trap in the codebase and it is
   asserted in two tests so that it cannot regress quietly.
7. **The pairing rule has two readings.** I implemented the global one
   (enumerate all feasible pairs, order by gap then line, greedy accept), which
   is registry-order independent. A per-outflow-row greedy scan gives a
   different answer when ties exist. Both satisfy the spec's words; only one is
   implemented, and the choice is only visible in the tie-breaking tests.
