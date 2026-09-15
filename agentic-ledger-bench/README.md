# agentic-ledger-bench

Does a **typed, statically-checked language** make code written by AI agents easier
to build and extend than Python with its full tool stack enforced? This is a small,
runnable experiment that tries to answer it with measurements instead of argument.

Three independent implementations of the same frozen specification, written by three
separate agents that never saw each other's code, all producing **identical output**:

| implementation | language | production LOC | tests | deps |
|---|---|---|---|---|
| `reference/python-unchecked` | Python, no static analysis at all | 1357 | 133 | none |
| `reference/python-tooling` | Python + `ty` + `ruff` + `semgrep` enforced | 2187 | 108 | none |
| `reference/scala` | Scala 3.9.0 LTS | 1291 | 92 | none |

All three parse a bank statement CSV, reconcile it, and emit canonical JSON. Run
`bash benchmark/run.sh` to reproduce all of it.

---

> **The one-line verdict:** for agent-written code, spend the constraint budget on the *gates and the
> boundary* — enforced static checking, a derived codec, a scoped dead-code gate, an exhaustiveness
> convention, and tests for wiring — and choose the language for ecosystem and domain, not for
> safety. Where an invariant must be *structural* rather than *conventional*, that is a per-property
> decision to be earned explicitly, not a per-language default.
>
> The whole experiment in one list: [§28 of `docs/FINDINGS.md`](docs/FINDINGS.md#28-the-consolidated-conclusions--the-whole-experiment-in-one-list).

---

## The headline result

**The language was not the load-bearing variable. The boundary was.**

1. **With its tool stack enforced, Python matched Scala on every gate** — and beat it
   on the one defect that mattered most. Adding a field to a Scala `case class`
   compiled clean with zero warnings while the **hand-written JSON codec silently
   dropped it from the output**. The same defect in typed Python raises
   `ValidationError` at construction and is flagged by `ty` as `missing-argument`.
   The asymmetry is not the language; it is **who owns the codec**. `pydantic`
   derives serialisation from the model definition, so a new field cannot diverge
   from its writer. A hand-written codec can.

2. **The bug that actually mattered was invisible to every static tool in both
   languages.** In the Python+tooling arm, a currency-equality predicate
   (`is_feasible_pair`) was written, documented, and **never called**, so `EUR`
   would pair with `USD`. `ty`, `ruff` and `semgrep` all passed. A **test** caught it.
   Static analysis buys type correctness; it does not buy **wiring** correctness.

3. **`ty` is the whole of the tooling value, and it is worth real money.** It found
   three real defects on first run, including a name used before definition that
   would have raised `NameError` on a path **no test covered**. `ruff` produced 43
   findings of which one was a real smell (39 were `TRY003` false positives).
   **`semgrep` returned zero findings on every codebase tested** — which `docs/FINDINGS.md`
   §20 explains: its `p/python` ruleset is 138 security and 13 audit rules with **zero**
   correctness or style rules, so a clean 2,000-line file is unmatchable by construction.
   The null result says nothing about the code.

   For context on the type checkers, the live `python/typing` conformance dashboard
   (fetched 2026-09-15) reads: zuban 99.7% · pyrefly 96.9% · pycroscope 95.2% ·
   **ty 93.8%** · pyright 93.4% · mypy 74.8%. Note the dashboard's own disclaimer —
   conformance "is not representative of many of the things users typically care about",
   and it should not be the primary basis for choosing a checker.

4. **The honest price of enforcement** was ~60% more production code (2187 vs 1357
   LOC) and 1.28s vs 0.94s per test suite. Cheap for a caught `NameError`.

5. **Compile-cycle cost is a non-issue when measured properly.** Scala's cycle is
   **1.43s** against Python's **0.94s** — about 1.5×, on a ~1.4-second loop. (An
   earlier measurement claiming ~10× was an artifact of invoking `scala-cli` against
   a directory instead of `test .` from the project root, paying JVM/Bloop startup
   each time. The correction is recorded in `docs/FINDINGS.md` §3.)

6. **Dead code: the scorecard inverts the intuition, but the asymmetry is narrow.**
   `-Wunused:all -Werror` catches an unused **private** member as a compile error and
   is **structurally blind to unused public methods** — and it has **no reachability
   closure**, so a dead function called only by another dead function stays hidden.
   `ruff` finds neither. `vulture` finds both, plus unused public functions
   (its default `min_confidence` is 0). Measured scope, and what is *not* covered:
   **no Python type checker except pyright/basedpyright has any dead-code diagnostic**
   (and pyright's `reportUnusedFunction` is off by default, and never fires on a
   class-scoped method).

   So Scala's real edge is not capability — vulture covers a strictly larger class — it
   is **delivery**: an in-language compile error versus a third-party tool with a
   confidence score and a whitelist. The trap to know: **vulture fails silently under
   Python 3.11**, which cannot parse PEP 695 `type X = ...` — it prints a syntax error
   and reports nothing for that file, reading as green. Put the dead-code gate on a
   current interpreter, and verify it is actually looking at your code.

---

## Reproduce it

```bash
bash benchmark/run.sh
```

This computes ground truth with the arbiter, runs every available implementation,
and compares them field by field. Expected output ends with:

```
== cross-implementation agreement ==
  unchecked vs tooling: 923 transactions x 8 fields, 0 mismatches
checks passed: 2   failed: 0
```

Only CPython is required for the Python arms. The Scala arm is skipped unless
`scala-cli` is on `PATH` — see `reference/scala/README.md`.

### The data

`benchmark/data/statement.csv` is **synthetic**, generated deterministically by
`benchmark/make_statement.py`. The original experiment ran on a real bank export,
which cannot be published; this generator reproduces exactly the invariants the
spec's assertions depend on:

| property | value | why it matters |
|---|---|---|
| transactions | 923 | the assertion count |
| final balance | 37713.30 | must be reconstructed exactly, no float drift |
| `fecha != fecha valor` | 29 rows | booking date and value date are **not interchangeable** |
| value-date inversions | 29 | a single collapsed date field computes the wrong answer |
| natural-key collisions | 3 dates | rows identical on `(fecha, importe, saldo)` — naive dedup yields 920 |
| continuity / cumulative errors | 0 | proves the reconciliation actually holds |
| skipped empty row | line 2 | the `,,,` row must be skipped, not treated as an error |

The generator self-verifies before writing and refuses to emit a file that does not
satisfy all six. `python3 benchmark/make_statement.py` regenerates it byte-identically.

### The arbiter

`benchmark/verify.py` was written **independently of all three implementations**,
from the specification and the data alone, and was validated with positive and
negative controls before use. It is the arbiter: an implementation passing its own
tests proves nothing, and in the original run an agent's own test suite missed two
real defects that this arbiter would have caught.

---

## Layout

```
benchmark/
  ledger-spec.md        frozen specification both languages implemented (§3 baseline, §5 iteration A)
  CLARIFICATIONS.md     two binding clarifications issued mid-run (see "What went wrong" below)
  make_statement.py     deterministic generator for the synthetic data, self-verifying
  data/statement.csv    the benchmark input
  data/A.csv B.csv C.csv accounts.json   shared multi-account fixture for iteration A
  verify.py             the arbiter (independent ground truth)
  run.sh                runs everything and compares
reference/
  python-unchecked/     Python with no static analysis (the "before" arm)
  python-tooling/       Python with ty + ruff + semgrep enforced (the fair arm)
  scala/                Scala 3.9.0 LTS
docs/
  FINDINGS.md           the full write-up: method, results, verdict, and limitations,
                        re-validated number-by-number in §26–§28 (corrections recorded there)
  tooling-review.md     the contributed review of Python tooling against this experiment
```

---

## What went wrong (kept, because it is the point)

Publishing only the clean result would misrepresent how this kind of measurement
actually goes. The original run had to be corrected four times:

- **The first comparison was not fair.** It measured typed Scala against Python with
  **no type checker installed at all**. The `python-tooling` arm exists to fix that,
  and it changed the conclusion.
- **A timing claim was wrong by ~10×**, because of a `scala-cli` invocation error.
  See §3 of `docs/FINDINGS.md` for the retraction.
- **The specification had two real errors**, both caught by the agents rather than by
  the author: it claimed the input had a UTF-8 BOM (it does not), and its
  comma-decimal rule is unreachable for an unquoted field in a comma-delimited file.
- **The `reconciled` definition was ambiguous**, and the two implementations resolved
  it in opposite directions — which is exactly why `CLARIFICATIONS.md` exists.

## Limitations

- **n=1 per arm, one agent each.** Agent variance is uncontrolled and cannot be
  resolved at this sample size. Treat the numbers as a signal, not a distribution.
- **The arms are not LOC-matched.** The Python+tooling arm also built a fixture
  generator and a stricter schema, so its 2187-vs-1357 LOC gap mixes added rigor with
  added scope.
- **Iteration A's wall-clock is not cleanly separable** for the Scala arm: that agent
  wrote iteration A in the same pass as the baseline, so only its test cycles are
  separated. It flagged this itself.
- **Iteration-A JSON schemas diverged** between implementations (semantics identical,
  field names and nesting different), so that phase is compared semantically, not
  field-for-field. Only the baseline envelope was frozen.

## License

Same as the parent repository.
