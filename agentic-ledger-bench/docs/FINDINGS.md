# Verdict: the ledger experiment — what Scala actually bought, and what it did not

**Date:** 2026-09-15 · **Status:** complete · Both halves green and independently verified.
**Evidence:** `evidence/v1/` · **Protocol and pre-committed criteria:** `PROTOCOL.md` · **Arbiter:** `verify.py`

---

## 1. Headline

Both implementations are **correct and semantically identical**. The hypothesis being tested —
*"agents brute-force a pile of functions in Python because Python tolerates it; a typed Scala domain
compounds instead"* — was **not confirmed and not refuted**. It was **mis-tested**, and the reason is
the most valuable thing this experiment produced:

> The Scala agent's own counterfactual showed that **the compiler is blind exactly where the extension
> happens.** Adding a field to the `Checks` type compiled with **zero errors and zero warnings**, and the
> hand-written JSON codec **silently dropped it from the output**. I reproduced this independently.

So the guarantee you buy with Scala covers the *type algebra*, and the defects live in the *hand-written
glue at the boundary* — which is precisely the glue every project rewrites, and precisely where the
§5 "sixth free function" pressure points. `agt4s`'s real job therefore is not "typed utilities"; it is
**generated, checked boundary code**, in whichever language you choose.

---

## 2. What is solidly established (independently verified, not self-reported)

| check | result |
|---|---|
| Python baseline vs arbiter | **exact** — 923 transactions, 8 fields each, **0 mismatches**, checksums identical |
| Scala baseline vs arbiter | **exact** — same |
| Python vs Scala, field by field | **0 mismatches** across 923 × 8 |
| Key order within envelope and transaction | **identical** |
| Real-file facts (both) | 923 txns · `skippedEmptyRows=[2]` · final `37713.30` · **29** inversions · 0 continuity · 0 cumulative errors · 92 rows where booking ≠ value date |
| Collision rows preserved | lines 243/247, 288/290, 446/448 — a naive dedup would yield 920, both kept 923 |
| Shared fixture (both) | exactly **2** pairs, all four decoys correctly left unpaired, net worth `1139.25` by both methods, residual `0.00` |
| Tests | Python **133 OK**; Scala **92 OK** |
| Runtime dependencies | **zero** in both (munit test-only on the Scala side) |

The checksum agreement deserves emphasis: both implementations independently computed a content
fingerprint and got `ad6223ae9f8b7c1f` for line 3, matching an independent `sha256sum`.

---

## 3. Corrected measurement — I was wrong, and here is the retraction

My first timing runs measured **9–13s** per Scala cycle and I nearly reported a ~10× penalty. That was
my error, for two reasons: I invoked `scala-cli` against a **directory path** (`scala-wt/scala`) rather
than running `scala-cli test .` from the **project root**, and `./.tools/scala-cli` is a ~147 MB native
launcher that pays JVM/server startup when the Bloop daemon is not already serving the project.

Re-measured properly, from the project root, warm Bloop, five runs each, a **real one-line production
edit** before each iteration:

| | Python | Scala 3.9.0 |
|---|---|---|
| full test suite, warm | **0.94s** (median, 133 tests) | **1.43s** (median, 92 tests) |
| range over 5 runs | 0.91–0.99s | 1.38–3.54s |
| cold first compile | n/a | 8.8s |
| realistic agent session | server-less, always fast | fast **only with Bloop warm** |

**The honest ratio is ~1.5×, not 10×** — and it is a 1.5× on a 1.4-second cycle, i.e. noise next to
model latency. The Scala agent's self-reported 1.8s median was accurate; my earlier number was an
artifact of my own invocation.

---

## 4. Side-by-side, with evidence class labelled

| metric | Python | Scala | class |
|---|---|---|---|
| production LOC | 1357 (11 files) | 1291 (11 files) | objective |
| code-only LOC | — | 973 (of which **197 = hand-written JSON**; Scala stdlib has none) | objective |
| test LOC | 1735 | 672 | objective |
| test:prod ratio | 1.28 | 0.52 | objective |
| runtime deps | 0 | 0 | objective |
| baseline wall-clock | 4m31s | 8m42s (incl. 1m44s toolchain) | self-reported |
| iteration A wall-clock | 3m06s | **not cleanly separable** — see §6 | self-reported |
| compile/run attempts | 11 | 3 | self-reported |
| test-fix cycles | 6 | 3 | self-reported |
| baseline green on first execution | no (68 tests, 11F+5E) | no (62/65) | self-reported |
| implementation defects found by tests | **2 real bugs** | **0** | mixed |
| iteration A: files added / modified | 7 / 3 (2 of them shared) | 2 / 0 | mixed |
| compiler found change sites | no (no checker installed at all) | **no** | verified |
| §5 re-derivation sites | 1 (`Movement.magnitude`) | 1 (`Money.abs` computed twice) | self-reported |

**The two bugs Python's tests caught are the most interesting number in the table:**
`build_checks` silently ignored the registry's `openingBalance` and re-derived the anchor from row 1;
and `net_worth_observations` computed *both* sides of the §5.5 check from the same accumulator, making
it vacuous. The agent noted the sharp point itself: **a type checker would not have caught either**,
because in both cases the wrong value and the right value are the same type (`Decimal`). Scala's
equivalent suite found **zero implementation defects** — its four failures were all wrong expectations
in its own tests.

---

## 5. The finding that matters most: the compiler's blind spot is the boundary

Reproduced independently on a scratch copy:

1. **Added `silentDropProbe: Int = 42` to `Checks`.** `scala-cli compile` → *"Compiled project"*, **no
   errors, no warnings**, even with `-Wunused:all`. Result: `checks` JSON keys unchanged, field
   **absent** — the hand-written codec in `Wire.scala` simply does not know about it. An agent that
   changes the model and forgets the codec ships a silently wrong artifact, and every gate passes.
2. **Added `ErrorKind.SILENT_DROP_PROBE`.** Compiles clean, even with `-Xfatal-warnings`, because
   `Cli.scala` renders it as `error.kind.toString`. The "frozen kind list" of spec §6 is therefore
   enforced by **convention only** — the compiler cannot tell a legal kind from an invented one.

Scala's `case class` + `enum` make the *interior* total, and that is real, verified value: the type
algebra did not leak. But **hand-written serialisation is a hole in the type system in any language**.
The countermeasure is not a different language — it is **deriving the codec from the type** (or
generating it), so the boundary is checked too. That is a concrete, buildable `agt4s` component, and
it is language-agnostic in principle.

---

## 6. Where the experiment is weak — stated plainly

1. **Iteration A is not cleanly measured on the Scala side.** That agent wrote `Transfers.scala` and
   `Accounts.scala` in the same pass as the baseline (its code existed before the module's first
   compile), so it separated only the *test* cycles. Its own report flags this as weakening "the very
   comparison this experiment is for". Its "0 files modified" is correspondingly softer than it looks:
   `Cli.scala` and `Wire.scala` were authored after iteration A was already designed.
   The Python side *is* cleanly separable (4m31s then 3m06s), and its extension was **not**
   containment-clean — it changed the shared `checks.py`.
2. **n=1 per language, one agent each.** Differences may be agent variance, not language. This is the
   single largest threat and it is not resolvable at this sample size.
3. **The Python agent had no type checker at all** (`mypy`, `pyflakes`, `ruff` all absent). So this
   experiment compared **typed Scala against untyped-and-unchecked Python** — it did *not* test the
   hypothesis as Mat framed it ("I type Python with astral-`ty`"). **The next iteration that matters is
   re-running the Scala half's counterpart with `ty` in strict mode.** Until then the comparison is
   against the wrong baseline.
4. **The iteration-A schemas diverged** (I froze only the baseline envelope): Python emitted
   `transferCheck`/`internals` and nested `from`/`to`; Scala emitted `netWorthConserved`/
   `netWorthTotal`/`netWorthMismatches` with flat `outflow`/`inflow`. Semantics matched exactly;
   field-for-field comparison was impossible. My gap, not theirs.
5. **Spec errors I shipped**, both caught by the agents: §2 claimed the reference file has a UTF-8 BOM
   — **it has none** (verified: first bytes `66 65 63 68`), and §4.1's comma-decimal rule is
   unreachable for an unquoted field in a comma-delimited file (both agents correctly reported `ROW`
   instead of guessing). Two of the best findings in the run came from agents checking my work.
6. **My own invocation error**, retracted in §3, briefly produced a wrong 10× claim.

---

## 7. Verdict against the pre-committed criteria

`PROTOCOL.md` fixed three outcomes. The result is **inconclusive on the hypothesis, with two firm
findings that outrank it**:

- Not "supported": the Scala side did **not** show materially cheaper extension (its measurement is
  not cleanly separable, it modified 0 files partly by construction, and the compiler found no change
  site for either agent).
- Not cleanly "not supported" either, because the comparison ran against **unchecked** Python rather
  than `ty`-strict Python, and because the Scala type algebra verifiably did not leak.
- **What is established:** (a) output parity is achievable to the byte — both languages pass every
  gate; (b) the cycle-time penalty I feared is ~1.5× on a 1.4s cycle, i.e. **not** a reason to choose
  a language; (c) **the compiler cannot see the boundary**, and that is where the bugs were; (d) tests
  caught both real defects, and they were **type-identical** errors a checker could not have flagged.

## 8. What to do next (ordered by information per hour)

1. **Re-run the Python half with `ty` in strict mode** and re-measure the same two cycles. This is the
   missing control and it is cheap.
2. **Freeze an iteration-B schema** (including the accounts envelope) before implementing, so the
   comparison is field-for-field like the baseline was.
3. **Make the boundary generated, not hand-written**, in both languages, and re-run the `Checks`-field
   counterfactual. If a derived codec turns the silent drop into a compile error, that is the real
   `agt4s` product — and it is worth more than a ledger.
4. Only then decide the language question, on the axis that survived: **where the compiler can still
   see the code.**

---

# Round 2 — the fair arm, the original module, and the conjecture

*Added 2026-09-15 after the user's follow-up: use `~/pro/notpro`'s original module, and run Python
**with ty + ruff + semgrep enforced** — the comparison that should have been run first.*

## 9. The original module, and a demonstrated silent-failure defect

`~/pro/notpro` contains two candidates: `src/ledger.py` (47 lines, pandas + matplotlib exploratory
script — not a module) and **`capital/src/kutxa.py` (65 lines, the real one)**. The original is better
than the previous round's spec assumed: its `parse_amount` correctly distinguishes Spanish from
English formats, and on the real file it returns the right final balance (`37713.3`).

It also has a defect, demonstrated rather than asserted. `parse_amount` ends in `except ValueError:
return 0.0`, so **one malformed `saldo` silently rewrites the answer**:

| input | reported balance |
|---|---|
| 3-row ledger, clean | `40300.0` (correct) |
| same ledger, one `saldo` = `"n/a"` | **`0.0`** — no exception, no warning |

Tool response to that module:

| tool | result |
|---|---|
| `ty` | **All checks passed** — `parse_amount` is annotated `-> float`, and `0.0` *is* a float |
| `ruff` (curated) | 6 findings; the relevant one is **BLE001 blind-except** — it flagged the cause, not the effect |
| `semgrep` (`p/python`, 151 rules) | **0 findings** |
| the repo's own tests | did not catch it |

## 10. The fair arm: Python with its tool stack enforced

Same frozen spec, same data, same arbiter; `ty.toml` and `ruff.toml` supplied and **verifiably
unmodified** (hashes in its `RUN.md`), `check.sh` gates on all three. Final state: **108 tests OK, `out/`
byte-deterministic across runs, arbiter PASS, all three gates green, zero dependencies (stdlib only),
zero `# type: ignore`, zero `# noqa` in production** (two in test helpers, both justified).

| arm | prod LOC | files | tests | suite time | deps |
|---|---|---|---|---|---|
| Python, unchecked | 1357 | 11 | 133 | 0.94s | 0 |
| Scala 3.9.0 | 1291 | 11 | 92 | 1.43s | 0 |
| **Python + enforced tooling** | 2187 | 13 | 108 | **1.28s** | 0 |

**What the tooling actually bought:**

- **`ty` earned its place — it was the only tool that found real defects.** First run: 3 errors, all
  real, including `IOError_` **used before definition**, which would have raised `NameError` on the
  unreadable-file path that **no test covered**. That is precisely the class of defect the previous
  round's unchecked Python shipped unnoticed (its `ty` run found a wrong declaration at `accounts.py:287`).
- **`ruff` produced one real smell among 43 findings**; 39 were `TRY003` false positives (it rejects
  `raise Cls("msg")` even for a same-file class with a required message parameter). The agent satisfied
  it structurally instead of suppressing it. Notably, `ruff` **did** catch the original module's
  `BLE001` — the cause of §9's silent-zero bug.
- **`semgrep` found nothing. Zero. On every codebase tested** (original module, unchecked arm, fair arm).
  As a gate in this stack it contributed no signal at all.

## 11. The bug no tool caught — in both languages

The fair arm's own most important defect: **`is_feasible_pair` (currency equality) was written,
documented, and never called by `find_transfers`**, so `EUR` would pair with `USD`. `ty` saw a
well-typed unused private function. `ruff`'s `F401` only tracks imports. `semgrep` saw nothing.
**A test caught it.**

So I checked the obvious fairness gap — whether Scala's equivalent gate had been enabled. It had not.
The Scala arm's `project.scala` carried only `-deprecation -feature`. A direct probe on Scala 3.9.0:

```
scala-cli compile WarnProbe.scala -Wunused:all -Xfatal-warnings
[error] unused private member
```

**The Scala compiler catches that defect class as a compile error — but only with a flag the arm never
enabled.** And there is no Python equivalent: `ruff` cannot see an unused *private function*.

## 12. And where Python's tooling beat the Scala codec

The Scala arm's signature failure was: add a field to `Checks` → compiles clean → **hand-written JSON
codec silently drops it**. The same defect, injected into Python:

| | result |
|---|---|
| Scala, hand-written codec | compiles clean; field **silently absent from output** |
| Python + `pydantic` (required field), construction site omits it | raises `ValidationError` at construction |
| Python + `ty`, same file | `error[missing-argument]` |

The asymmetry is not the language, it is **who owns the codec**. Pydantic derives serialisation from
the model definition, so a new field cannot diverge from its writer. Scala's codec was hand-written in
a separate file, so adding a field was invisible to the compiler. This confirms §5 from the other
direction: generate the boundary, and Python is *ahead* on this defect class.

## 13. Verdict — the conjecture, answered

> *Conjecture: agents brute-force functions in Python because it tolerates it; the fix is a typed Scala
> domain, and the fair test is Python plus its available tooling.*

**The fair comparison does not support the conjecture, and reframes it.**

1. **Python + enforced tooling matches, and on the boundary defect exceeds, the Scala arm.** Both pass
   every gate with identical semantics (923 transactions, `37713.30`, 29 inversions, 2 transfer pairs,
   `1139.25` by both methods) — and on "new field silently dropped", typed Python caught it while the
   Scala codec did not. **The language is not the load-bearing variable.**
2. **Compile-time/check-time enforcement is worth real money, and `ty` is the proof.** It caught a
   `NameError` on a path no test covered — a defect that would have shipped. The fair arm cost **~60%
   more production code** (2187 vs 1357 LOC) than unchecked Python and **1.28s vs 0.94s** per suite:
   that is the honest price of the guarantee, and it is cheap.
3. **But the defect class that actually mattered was invisible to every static tool in both languages** —
   a rule defined and never wired. Tests found it in Python; only an *unconfigured* compiler flag would
   have found it in Scala. Static analysis buys type-correctness; it does not buy **wiring correctness**.
4. **`semgrep` returned zero findings on every codebase. `ruff`'s value was one real finding in 40+.**
   A tool stack is not a strategy; `ty` was the whole of the value here.
5. **What this means for the decision.** Choose the language for the ecosystem, the team and the domain —
   not for safety, because with enforced tooling both are safe to the same practical depth. Then spend
   the effort where this experiment says the defects actually live: **(a) generate the boundary**
   (pydantic-style, never hand-write the codec), **(b) enable the unused/dead-code gate** in whichever
   language (`-Wunused:all -Xfatal-warnings` in Scala; there is no Python equivalent, which is a real
   and specific point *for* Scala), and **(c) keep tests for wiring and integration**, which is where
   both toolchains were blind.

### Residual uncertainty, stated plainly

- n=1 per arm, different agents; agent variance is uncontrolled and cannot be resolved at this size.
- The arms are not LOC-matched (the fair arm built a fixture generator and a stricter schema), so the
  2187-vs-1357 gap mixes added rigor with added scope.
- The fair arm extended `reconciled` to require `parseErrors` empty — a defensible reading, recorded as
  spec friction, invisible on the real file.
- Iteration-A wall-clock figures remain self-reported and, for Scala, not cleanly separable (§6).

---

# Round 3 — complexity, dead code, and the alternative checkers (measured)

*Added 2026-09-15. Every number here was produced by running the tool, not read from a blog.*

## 14. Complexity: enforcement redistributes it, it does not reduce the total

`radon` over the two Python arms (mean cyclomatic complexity hides block-count
differences, so totals are reported too):

| arm | blocks | total CC | mean CC | worst block |
|---|---|---|---|---|
| python-unchecked | 67 | 250 | **3.73** | **28** |
| python + enforced tooling | 154 | 337 | **2.19** | **14** |

**Total complexity is comparable (337 vs 250); the distribution is not.** Enforcement
roughly doubled the number of units and halved the worst case. That is the real
mechanism behind "taming complexity" — not less logic, but logic broken into pieces
small enough to reason about and to test. It is also the honest cost: ~60% more
production lines for the same behaviour.

## 15. Dead code: the one place a compiler has a structural advantage — and its limit

The defect that no static tool caught was an **unused public function**
(`is_feasible_pair`, silently never called). Measured behaviour per tool:

| tool | unused **private** function | unused **public** function |
|---|---|---|
| Scala `-Wunused:all -Xfatal-warnings` | **compile error** | **silent** — compiles clean |
| `ruff` (curated set) | silent | silent (`F401` tracks imports only) |
| `vulture --min-confidence 60` | flags | **flags** (`unused function ... (60% confidence)`) |
| `semgrep` (`p/python`, 151 rules) | silent | silent |

So the scorecard inverts the intuition: **for library-style code, Python's `vulture`
is stronger than the Scala compiler**, because `-Wunused` deliberately does not report
public members — a public API is assumed to be used by someone outside the compilation
unit. Scala wins on private members by turning it into a hard error.

**And `vulture` has a silent-failure trap of its own.** Installed under Python 3.11 it
cannot parse PEP 695 syntax (`type X = ...`), prints
`invalid syntax at "type CandidateEntry = ..."`, and reports **nothing** for that file —
including the dead function it was supposed to find. Under Python 3.14 it finds it. A
dead-code gate that silently no-ops on modern syntax is worse than no gate, because it
reads as green.

**Precision measured:** `vulture` on the fair arm flagged 6 functions; 1 was genuinely
dead (`_sum_amounts`, one occurrence in the whole tree), 4 were false positives
(serialisation helpers that `ledger.py` imports and calls), and 1 was the real defect.
That is roughly 1-in-3 precision — usable as a signal, not as a hard gate without a
whitelist.

## 16. Type checkers: `mypy` cannot parse modern Python

Same code, same directory, each tool with its own invocation:

| tool | result on the fair arm | time |
|---|---|---|
| `ty` 0.0.81 | **All checks passed** | 0.04s |
| `ty --error-on-warning` | All checks passed | 0.04s |
| `mypy` 2.3.1 | **`Invalid syntax` — cannot parse `type X = ...`** | 0.07s |
| `pyright` | 1 error (recursive `JSONObject` alias) | 1.24s |
| `basedpyright` | 49 errors, 64 warnings (its stricter defaults) | 1.30s |
| `pyrefly` | 0 errors | 0.10s |

The `mypy` failure is not an install artifact: the `uv`-installed binary runs on its own
**Python 3.11.16** interpreter, and `--python-version 3.14` produces the same error
("you likely need to run mypy using Python 3.14 or newer"). Replacing the PEP 695 alias
with the classic form makes `mypy` run and report 31 errors, so the tool works — it just
**cannot see modern syntax**, which for an agent writing current-idiom Python is close to
disqualifying as the single type gate.

`ty` at 0.04s with zero findings is 30× faster than `pyright` and is the only checker
whose verdict the agent could act on continuously.

## 17. What this round changes in the conclusion

§13 stands, with two sharpenings:

- **The language still is not the load-bearing variable** — but the *configuration* is.
  The Scala arm underperformed its own compiler because `-Wunused:all` was never enabled;
  the Python arm only matched Scala because `ty` was. Neither result was a property of the
  language; both were properties of what was switched on.
- **For the specific defect that mattered — a rule defined and never wired — the best
  available gate in either ecosystem is `vulture` on a current interpreter**, plus tests.
  That is an uncomfortable conclusion for a "use Scala for safety" argument, and it is
  what the measurements say.
