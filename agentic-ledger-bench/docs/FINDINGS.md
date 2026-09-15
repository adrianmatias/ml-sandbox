# Verdict: the ledger experiment — what Scala actually bought, and what it did not

**Date:** 2026-09-15 · **Status:** complete, and re-validated end-to-end on the same day (§26–§28).
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
spec §5 "sixth free function" pressure points. `agt4s`'s real job therefore is not "typed utilities"; it is
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

*Basis (stated after the validation pass, §26): production and test LOC are the agents' worktree
files, raw lines, before this repository's black/isort formatting pass — the published Python
copies therefore recount slightly higher (python-unchecked: 1413 raw prod / 1976 raw test;
python-tooling: 2230 / 1879), while the Scala tree was never reformatted and matches exactly (2223
raw across all files). "Code-only" strips blank and comment lines; the Scala figure 973 is
block-comment-stripped, and `Json.scala` is 197 raw / 172 code.*

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

*Basis (stated after the validation pass, §26): both rows scan the `ledger/` **package only**. The
tooling arm's top-level CLI shim (`ledger.py`, 107 lines) is excluded — with it the row reads 158
blocks / 351 total CC / mean 2.22 (worst still 14). The unchecked arm has no separate CLI file, so
its row is the full arm. Re-verified 2026-09-15: the package-only figures reproduce exactly.*

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
reads as green. `radon` has the same trap (reproduced in the validation pass, §27):
under an interpreter older than 3.12 it prints one `ERROR: invalid syntax` line for the
file, skips every block in it, and **still exits 0** — a complexity gate that passes
while having looked at less code than its output implies. The rule is the same for
every tool: verify it is actually looking at your code, on the interpreter you write for.

**Precision measured — corrected by the validation pass (§27).** `vulture` 2.16 on the
fair arm's *package* scan flags 6 functions; that count reproduces exactly. The original
read was "1 genuinely dead, 4 false positives, 1 real defect, roughly 1-in-3 precision".
On re-checking each flag, **3 of the 6 are real**: `_sum_amounts` is dead, and so is
`domain.balance` — imported nowhere (only the `Balance` *type* is), called nowhere. The
other three (`envelope_data`, `envelope_error`, `baseline_data`) are helpers the
top-level CLI shim calls. And the measurement turns out to be **scan-scope dependent**:

| scan scope | function flags | what it means |
|---|---|---|
| package only | 6 — 3 real, 3 FP | the basis of the original precision figure |
| package + CLI shim | **3 — 3 real** | adding the entry point removes the shim-called FPs |
| package + shim + tests | **1** (`_sum_amounts`) | **test references mask `is_feasible_pair`** — the natural whole-tree CI scan hides the wiring defect behind the tests that exercise it |

So a dead-code gate needs a stated scan scope, not only a whitelist: the widest, most
"natural" scope is precisely the one that misses the defect class it was added for.

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

---

# Round 4 — corrections from a deeper tooling review

*Added 2026-09-15. A follow-up review (`dead-code-tooling-brief.md`) checked these claims against
tool sources and live compiles. Several numbers published above needed correcting. Everything in this
section was re-verified by me directly against the live sources before being written down.*

## 18. Corrections to what this document previously said

| what was published | corrected | how verified |
|---|---|---|
| `ty` conformance **86.5%** | **93.8%** (ty 0.0.81, live dashboard) | parsed `python/typing` `conformance/results/results.html` today |
| "`ruff` only catches unused imports" | **wrong and unfair to ruff.** Ruff also covers unused locals (`F841`), annotations (`F842`), unpacked variables (`RUF059`), and `ARG001`–`ARG005` are **stable**, just not default-on. It *does* ship unused-private detection (`PYI018/046/047/049`) — **but only for `.pyi` stub files** | rule docs + `--select` runs |
| Scala `-Wunused:all` described as the dead-code gate | `all` does **not** include `nowarn` (read via `isChoiceSet`, not `allOr`), and **`synthetics` does not exist in Scala 3** — it is Scala 2.13-only | compiled with `scala-cli` on 3.9.0 |
| `vulture --min-confidence 60` implied as the setting | vulture's **default `min_confidence` is 0, not 60** — that default, not the tool, is the usual source of CI noise. With no flags it flags an unused private function, an unused *public* function, and an unused method | ran `vulture` with no flags |
| `-Xfatal-warnings` used in the Scala commands | it is a **deprecated alias** of `-Werror` | scalac 3.9.0 output |

The live conformance dashboard also carries its own disclaimer, which is worth quoting because it
undercuts citing conformance as a selection criterion: *"While specification conformance is important
for the ecosystem, we don't recommend using it as the primary basis for choosing a type checker."*

## 19. The dead-code scorecard, corrected and completed

Measured scope of unused-code detection per tool (the column that decides the question):

| tool | module `_fn` | method `_m` | method `__m` | unused **public** fn | enabled |
|---|---|---|---|---|---|
| `vulture` 2.16 | yes | yes | yes | **yes** | on |
| `dead` 2.1.0 | yes | yes | yes | yes | on |
| `pyright` `reportUnusedFunction` | yes | **no** | yes | yes | **off by default** |
| `basedpyright` | yes | **no** | yes | yes | on |
| `pylint` `W0238` | no | no | yes | no | on |
| `ruff` | no | no | no | no | — |
| `ty` / `pyrefly` / `mypy` | **no dead-code diagnostic at all** | | | | — |

Verified directly: `pyright` with `reportUnusedFunction` enabled flags the module-level `_unused_private`
but is silent by default. **No Python type checker except pyright/basedpyright has any dead-code
diagnostic** — that is the precise form of the gap this document was reaching for, and it is narrower
than "no tool exists".

**Two further corrections to the Scala side, both verified by compiling:**

- Scala's `-Wunused:all -Werror` on a dead `rootDead()` called only by another dead
  `onlyCalledByDead()` reports **only the intermediate function** — there is **no reachability
  closure**. Transitive dead code stays hidden.
- Combined with §15: Scala catches the *direct private* case and nothing else. Vulture catches a
  strictly larger class. The asymmetry is not capability, it is **delivery**: Scala's is an
  in-language compile error, Python's is a third-party tool with a confidence score and a whitelist.

## 20. Why `semgrep` returned zero findings — the null result is explained, not mysterious

Its `p/python` ruleset is **151 rules: 138 security, 13 audit, and zero correctness or style rules**,
restricted to single-function analysis. A clean 2,000-line ledger is **unmatchable by construction**.
So the zero-findings result across three codebases carried no information about the code — and it is
a caution against reading "clean" from a tool that was never looking for that class of problem.

## 21. The evidence that most affects an agentic build cycle

- A **Meta/Pyrefly agent-loop experiment (PyCon US 2026)** found type checking helped **only on
  well-typed code (80% → 84% success)**; on low-coverage code it *"distracted the agent."* That is
  the strongest published caution against adding type checking as a blanket gate, and it is directly
  relevant to a controlled agentic cycle: the benefit is conditional on the code already being typed.
- **50.8% of 7,357 real-world Python suppressions are "practically useless"** — suppression debt is
  the norm, which is why the experiment forbade `# type: ignore` / `# noqa` to reach green.
- **24.17% of agent-modified Python files introduce new Pylint issues, and 73.5% of those PRs merged
  anyway** — the review gate does not catch what the linter does.

## 22. What remains unmeasured (do not fill with opinion)

No false-positive *rate* measurement exists for vulture or dead; no per-rule FP rates for any Python
linter; no marginal-gain-per-additional-tool study; no mutation-testing wall-clock multiplier for
Python or for agent loops; no dead-code prevalence figure for Python; and **no controlled study
showing static analysis reduces downstream defect density in LLM-generated Python.** §17's ordering
is a judgement built on the measurements above, not a finding from a study.

---

# Round 5 — the question correctness and static analysis left out

*Added 2026-09-15, in response to the fair objection: "I get no conclusion about enforcing
typed/OO/func programming into Python, we just got the side of correctness and static analysis."*

That objection is correct. Every earlier arm measured **correctness**. None measured whether
**enforcing a style** — or **adding lean dependencies** — changes anything. Two further arms were
built to close that gap, and they are the two most useful results in this document.

## 23. Exhaustiveness: the one place structural enforcement decisively beats convention

This is the clearest measured difference between the two languages in the whole experiment, and it
comes from a five-probe experiment on real code plus a direct reproduction. The question: if an agent
adds a new variant to a closed sum type, does anything catch it?

| setup | what happens when a variant is added | caught by |
|---|---|---|
| **Scala 3** `enum` + `match`, no flags | `warn: match may not be exhaustive` | compiler |
| **Scala 3**, `-Werror` | **`error: It would fail on pattern case: AMOUNT`** — build fails | compiler |
| **Python**, `enum` + `match`, **no wildcard**, non-optional return | `error[invalid-return-type]` (the gap shows up as a return-type error) | `ty` |
| **Python**, `enum` + `match`, non-optional return, **`case _ as x: assert_never(x)`** | `error[type-assertion-failure]` — **exhaustiveness, properly** | `ty` |
| **Python**, the no-wildcard probe but the function may return `None` (no `assert_never`) | **`All checks passed!`** — missing case is silent (the fall-through satisfies the Optional return) | nothing |
| **Python**, `enum` + `match` with an ordinary **wildcard** | **`ty` exit 0, `ruff` exit 0, tests pass — the new variant is silently routed to the wildcard's branch** | nothing |

(The Optional-return row re-measured in the validation pass, §26: `assert_never` still catches
the missing case even with an Optional return — it is the *absence* of `assert_never` combined
with an Optional return that goes silent. The row above states that precise combination.)

Reproduced directly: with `case _: return 0` present, adding `UNSUPPORTED` to the enum produces no
diagnostic of any kind, and `exit_code(Failure(UNSUPPORTED, ...))` returns the wildcard's `0`.
`assert_never` also only fails at **runtime**, on the path that reaches it.

**The conclusion, which answers the outstanding question.** Typed functional Python *is* enforceable —
**for `enum` + `assert_never`, and nothing else.** Use a dataclass union, or a wildcard arm, or (if
you drop `assert_never`) a return type that admits `None`, and the guarantee silently disappears. Worse: **nothing in the
toolchain tells you which of those two worlds you are in.** Scala's enforcement is structural (the
language refuses the match); Python's is conventional (you must remember `assert_never`, and keep
remembering it). For code written by agents — which is exactly code whose invariants are not held in
anyone's head — that difference is real, and it is the strongest argument for a typed language this
experiment produced.

## 24. Lean dependencies: what they bought, measured, with numbers

A second arm built the same module with the libraries a pragmatic engineer would reach for. **Two of
ten survived; eight were rejected with measurements** (`val-py/experiments/out/`):

| dependency | verdict | the number that decided it |
|---|---|---|
| **pydantic 2.13.5** | **kept** | 28 ms import, 5.5 µs/row; validated untrusted registry JSON and derived the envelope. **But it did not replace the money rule** — strict mode rejects the spec's own decimal *strings*, lax mode accepts JSON floats |
| **beartype 0.22.9** | **kept** | the only layer that stops a **binary float entering the Decimal domain when it arrives as `Any`** (e.g. from `json.loads`), where `ty` is silent by construction |
| typeguard | rejected | same catch, **11.47 ms vs beartype's 0.52 ms** for 20 calls over a 1000-element tuple |
| icontract / deal | rejected | module invariants are properties of a 923-row *result set*, not function contracts |
| toolz / funcy | rejected | saved **one** source line over the stdlib loop |
| returns | rejected on principle | a `Result` chain short-circuits on first failure — **exactly what §3.4 forbids** |
| pandas 3.0.5 | **rejected** | see below |
| polars 1.44.2 | rejected | exact, but rounds `1.234` → `1.23` where the spec demands refusal |

### The pandas/polars result — "just use pandas" is wrong here, with numbers

Reproduced independently. Default `read_csv` gives `float64` for `importe`/`saldo`, and:

- **`37713.30` can never be printed** — the value is `37713.3`, so the required canonical form and
  every `sha256` checksum input are unrecoverable.
- **260–381 of 923 amount texts cannot be reproduced** from the float (count depends on the
  normalisation used, which is itself the point).
- The empty `,,,` row is read as a data row, **shifting every `line` number in the spec's identity field**.
- pandas on a ragged row **silently remaps columns**: `{'fecha': [1234], 'fecha valor': [56], ...}`.
- pandas has **no working exact-decimal `read_csv`**: `dtype=decimal128` raises `ArrowInvalid`, and
  `converters={'importe': Decimal}` raises `InvalidOperation` on the empty row. Only
  "read as text, then cast with pyarrow" works — pandas reading text and pyarrow casting it.

**Honest correction to the arm's own headline:** it reported the float cumulative check failing
**912/923** rows. That is true only at zero tolerance. Measured drift: **0 mismatches at 1e-9**,
worst absolute drift over 923 rows **8.0e-11 EUR**. So the float check is *tolerance-dependent* — and
the correct, tolerance-independent statement is stronger anyway: **pandas cannot emit the canonical
decimal string or the raw text the spec requires, at any tolerance.** The library is disqualified by
representation, not by arithmetic drift.

### The one defect class a runtime dependency owns

Injection, not anecdote: for a float arriving as `Any` into a `Decimal` field, **`ty` is silent,
beartype catches it, and a plain dataclass accepts it** (`0.1+0.1+0.1 = 0.30000000000000004`).
That is a genuine gap nothing else covers — and the arm is honest that **no runtime dependency caught
a defect its tests would have missed during the build.** It is insurance, not a discovered bug.

### Dependencies also obstructed correctness, four times

1. `beartype` + the idiomatic `if TYPE_CHECKING:` import → **runtime crash**
   (`ForwardRef ... unimportable`), and static checkers are silent because it is a type-only import.
2. pydantic's `populate_by_name` + alias **defeated `extra="forbid"`**, so a misspelled registry key
   validated instead of erroring.
3. pydantic strict mode cannot express the shared fixture's own format.
4. pandas' silent column remap (above).

## 25. Complexity and tech-debt compounding — with all five arms

| arm | prod LOC | files | tests | suite | deps | defects caught by tests |
|---|---|---|---|---|---|---|
| stdlib Python | 1357 | 11 | 133 | 0.78s | 0 | 2 |
| Python + tooling | 2187 | 13 | 108 | 1.14s | 0 | 1 |
| Scala 3.9.0 | 1291 | 11 | 92 | 1.43s | 0 | 0 |
| **functional (enforced style)** | **1932** | 10 | 57 | 0.24s | 0 | 2 |
| **pydantic + beartype** | **1384** | 15 | 108 | 0.33s | 2 | 1 |

*Basis corrections applied by the validation pass (§26): the tooling arm was previously reported
here at 2080 LOC / 12 files — that was the `ledger/` package only, excluding the top-level CLI shim
(`ledger.py`, 107 lines) that every other row includes, so the row now carries the full-arm figure
used in §10, §13 and the README. Its "defects caught by tests" is likewise corrected from 2 to **1**
(`is_feasible_pair`): the arm's other real defects — two used-before-definition and one `Amount(0)`
sloppy-zero — were caught by `ty`, not by tests. Suite times re-measured 2026-09-15 in one warm pass
on one machine (§26); earlier sessions recorded 0.94 / 1.28 / 1.43 / 0.30 / 0.47 s for these rows.*

Read the LOC column carefully: the enforced-functional arm cost **+42%** over the stdlib arm
(1932 vs 1357) for **the same guarantees** — and it produced **no defect the tests would not have
caught**. The two real defects in that arm were a `Decimal.quantize` precision ceiling and two
plausible-but-wrong net-worth models; a test and the clarifications table caught them. The style bought
**iteration speed**, not correctness. Combined with §14's finding that enforcement *redistributes*
complexity rather than reducing it (mean CC 3.73 → 2.19, worst 28 → 14, total 250 → 337), the pattern
across all five arms is consistent: **every form of enforcement costs lines and buys a guarantee — and
the guarantee is only worth it where it is structural rather than conventional.**

### Caveat that limits this table

The suite times are **not mutually comparable**: the arms ran on different Python versions (3.12.14 vs
3.14.7) under different venvs. The functional arm's 0.24s is largely a smaller suite on a fast
interpreter, not a style win. The LOC and defect columns are comparable; the timing column is not, and
is reported only for completeness.

---

# Round 6 — validation pass: every number re-run, and the conclusions consolidated

*Added 2026-09-15 (a fresh session, run at `glm-5.3-flash:max`), on the request to review the
findings, clean and validate the conclusions, and push. Nothing here was taken on faith from the
earlier rounds: each claim was re-executed against the frozen worktrees (`lab/*-wt`, the agents'
original code) and the published tree, with the tools re-run today. What did not reproduce was
corrected in place above and is listed here; nothing was quietly rewritten.*

## 26. What was re-verified, claim by claim

| claim | published | re-run today | verdict |
|---|---|---|---|
| arbiter + cross-implementation agreement | 2 checks pass, 0 mismatches | `bash benchmark/run.sh` → PASS / PASS / SKIP (scala-cli not on PATH, as documented); 923 × 8 fields, 0 mismatches | holds |
| test suites | 133 / 108 / 57 / 108 / 92 | 133 OK · 108 OK · 57 OK · 108 OK (own venv) · 92 OK (worktree, warm Bloop) | holds — counts exact |
| Scala warm cycle | 1.43s (§3 five-run median) | 1.37–1.39s warm | holds |
| production LOC | 1357 / 2187 / 1291 / 1932 / 1384 | exact against the worktrees | holds — basis now stated (§4) |
| radon, unchecked arm | 67 / 250 / 3.73 / 28 | identical | holds |
| radon, tooling arm | 154 / 337 / 2.19 / 14 | identical on the package-only basis; **158 / 351 / 2.22 / 14** including the CLI shim | holds — basis now stated (§14) |
| `ty` 0.0.81 on the tooling arm | clean, 0.04s | `All checks passed!`, 0.05s | holds |
| `mypy` 2.3.1 | Invalid syntax on PEP 695 | py3.11 → `transfers.py:165: error: Invalid syntax`; py3.14 → parses and reports type errors | holds |
| vulture on the tooling arm | 6 flags: 1 dead, 4 FP, 1 defect | 6 flags reproduce on the package-only scan; **3 of the 6 are real** (`domain.balance` is dead too) | corrected (§15) |
| exhaustiveness probes (§23) | all six rows | all reproduce; the Optional-return row re-stated precisely | holds |

Two published-tree facts a reader hits immediately, now fixed and documented: the published Scala
arm was missing its own test fixtures entirely — `fixtures/` (31 synthetic files) and
`spec/shared-fixtures/accounts.json` are now committed, taking the suite from 43 to 68 of 92 green
from a fresh clone; the remaining 24 need the private bank export and fail loudly (note added to
`reference/scala/README.md`). And the raw line counts of the published Python copies differ
slightly from the tables because they were black/isort-formatted — note added to §4.

## 27. What this pass found that the rounds had not

1. **`vulture`'s verdict is scan-scope dependent, and the natural CI scope hides the defect.**
   Package-only scan: 6 flags, 3 real. Adding the CLI shim: 3 flags, 3 real. Adding the tests:
   **1 flag — `is_feasible_pair` is masked by its own test references**, and only `_sum_amounts`
   survives. The whole-tree scan that reads as green is precisely the one that misses the wiring
   defect. A dead-code gate needs a stated scan scope, not just a whitelist (§15).
2. **`domain.balance` is dead code the original precision count miscategorised as a false
   positive.** Imported nowhere (only the `Balance` *type* is), called nowhere. The fair arm ships
   **two** dead functions, not one — which strengthens, not weakens, the case for a dead-code gate.
3. **`radon` joins the silent-failure trap class.** Under an interpreter older than 3.12 it prints
   one `ERROR: invalid syntax` line for a PEP 695 file, skips every block in it, and **exits 0** —
   a complexity gate that passes while not looking at the file. Same class as vulture (§15) and
   mypy (§16): on current-idiom Python, verify the gate is actually looking at your code.

## 28. The consolidated conclusions — the whole experiment in one list

Each tagged: **[re-verified]** re-executed in this pass · **[verified]** executed in an earlier
round and checked against its recorded evidence · **[self-reported]** the agents' own numbers.

1. **The language was not the load-bearing variable; the enforced configuration was.** Python with
   its tool stack enforced matched Scala on every gate and beat its hand-written codec on the
   boundary defect. Both results were properties of what was switched on — the Scala arm never
   enabled `-Wunused:all`; the Python arm won only because `ty` was on. **[verified]**
2. **The boundary is where the defects live; generate it, never hand-write it.** Adding a field to
   a Scala `case class` compiled clean while the hand-written codec silently dropped it; the same
   defect is a construction-time error under pydantic and a static error under `ty`. The
   countermeasure — a codec derived from the type — is language-independent and is the one concrete
   product this experiment points at. **[verified]**
3. **Wiring correctness needs tests plus a correctly-scoped dead-code gate.** The one defect no
   static tool caught was a rule defined and never wired (`is_feasible_pair`). Scala's compiler
   catches unused privates as compile errors (`-Wunused:all -Werror`) but has no reachability
   closure and is blind to unused publics; `vulture` covers a strictly larger class but only with
   the right scan scope, and the widest scope is the one that misses it (§27.1). **[re-verified]**
4. **`ty` was the whole of the static-analysis value; a tool stack is not a strategy.** It caught a
   `NameError` on a path no test covered; `ruff` produced one real smell in 43 findings; `semgrep`
   returned zero findings on every codebase because its ruleset is security-only by construction.
   **[verified]**
5. **Enforcement redistributes complexity rather than reducing it.** Blocks 67 → 154, total CC
   250 → 337, mean 3.73 → 2.19, worst 28 → 14: the same logic broken small enough to reason about
   and test, at ~60% more lines. That is the real mechanism behind "taming complexity".
   **[re-verified]**
6. **Structural beats conventional — measured, not argued.** Exhaustiveness is structural in Scala
   (warn by default, build failure under `-Werror`) and merely conventional in Python: it exists
   only for `enum` + `assert_never`, and dissolves silently under a wildcard arm, a dataclass
   union, or a drop of the `assert_never` habit with an Optional return. Nothing in the Python
   toolchain tells you which of the two worlds you are in. **[re-verified]**
7. **Every tool in the stack failed silently somewhere.** vulture: a PEP 695 file skipped, reads as
   green (py≤3.11). mypy 2.3.1: `Invalid syntax` on current idiom unless run on py≥3.14. radon:
   skips a file and still exits 0. semgrep: a clean ledger is unmatchable *by construction* — its
   null result carried no information. The gate that matters is verifying the gate. **[re-verified]**
8. **The compile-cycle tax is noise beside model latency.** ~1.5× on a ~1.4-second warm cycle; the
   earlier ~10× claim was an invocation artifact and is retracted in §3. **[re-verified]**
9. **Enforced style and lean dependencies bought speed and insurance, not correctness.** The
   functional arm cost +42% lines for zero prevented defects; 2 of 10 candidate dependencies
   survived (pydantic for a derived boundary, beartype for the `Any`-float gap); "just use pandas"
   is disqualified by representation — it cannot emit the canonical decimal string at any
   tolerance. **[verified]**
10. **What still limits every number here: n=1 per arm, one agent each, and self-reported
    wall-clocks for the extension phase.** §22's unmeasured list stands. This conclusion set is
    what five arms on one real module can honestly support — no more. **[stated]**

**The one-line verdict:** for agent-written code, spend the constraint budget on the *gates and the
boundary* — enforced static checking, a derived codec, a scoped dead-code gate, an exhaustiveness
convention, and tests for wiring — and choose the language for ecosystem and domain, not for
safety. Where an invariant must be *structural* rather than *conventional*, that is a per-property
decision to be earned explicitly, not a per-language default.

**Adopting it:** the list above is the portable part. Turned into rules a project can paste into its
agent guide, it becomes five of them — the type checker as an enforced gate (and *verify the gate is
actually looking at your code*), the boundary derived from the type rather than hand-written, wiring
covered by tests plus a **scoped** dead-code gate, exhaustiveness structural or written down, and the
line cost of enforcement stated out loud. None of that adds a claim: every rule traces to a numbered
conclusion above.
