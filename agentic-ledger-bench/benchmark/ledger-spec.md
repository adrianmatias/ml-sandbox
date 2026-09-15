# Ledger module — frozen specification (v1, 2026-09-15)

**This document is the shared contract for a Python-vs-Scala comparison.** Both implementations are
built from this spec, independently, by agents that never see each other's work. The spec is frozen:
if you believe something in it is wrong or ambiguous, implement the most reasonable reading, then
record the deviation in `NOTES.md` under "spec friction". Do not change this file.

The point of the exercise is to measure the real cost of building and then **extending** the same
module in two languages. Correctness against this contract is what makes the two results comparable.

---

## 1. Deliverables

1. A module implementing §3 (baseline) and §5 (iteration A).
2. A CLI with the exact interface in §6.
3. `notes/NOTES.md` with the required metrics of §7 (a template is provided).
4. The tests of §8.

Reference data (read-only): `data/statement.csv` — synthetic, see `make_statement.py`
Your synthetic fixtures live under your own directory (§4).

---

## 2. Input format — Kutxa statement CSV

Header line, exactly four columns, in this order:

```
fecha,fecha valor,importe,saldo
```

- `fecha` — booking date, `DD/MM/YYYY`
- `fecha valor` — value date, `DD/MM/YYYY`. **Semantically distinct from `fecha`.** In the reference
  file 92 of 923 rows differ and value-date order contains **29 inversions** while booking order
  contains none. Never collapse these two fields into one.
- `importe` — signed amount, positive = credit, negative = debit
- `saldo` — account balance *after* the row

Format facts, all verified against the reference file:

- UTF-8 **with BOM** on the first line.
- Comma-delimited. Amounts contain **no thousands separators** and use `.` as decimal separator in
  this file — but see the parser requirements below, which must be broader than this file.
- Up to two decimal places.
- One fully empty row (`,,,`) must be skipped, not treated as an error.
- Rows are already sorted by `fecha` with no inversions.

---

## 3. Baseline behaviour

Export exactly two artifacts, both deterministic JSON (see §6 for the envelope):

**`transactions`** — one entry per data row, in file order, each with:

| field | type | rule |
|---|---|---|
| `line` | integer | 1-based physical line number in the CSV, header = line 1 |
| `bookingDate` | string | ISO `YYYY-MM-DD` |
| `valueDate` | string | ISO `YYYY-MM-DD` |
| `amount` | string | canonical decimal string |
| `balanceAfter` | string | canonical decimal string |
| `rawAmount` | string | the amount exactly as it appeared in the file |
| `rawBalance` | string | the balance exactly as it appeared in the file |
| `checksum` | string | hex of the **first 8 bytes** of `sha256(amount || "|" || balanceAfter || "|" || bookingDate)` |

**`checks`** — an object:

| field | type | meaning |
|---|---|---|
| `rowCount` | integer | rows parsed into transactions |
| `skippedEmptyRows` | array of integers | 1-based lines of skipped empty rows |
| `reconciled` | boolean | true iff every array below is empty |
| `continuityErrors` | array of `{line, balanceExpected, balanceActual, delta}` | see 3.1 |
| `cumulativeErrors` | array of `{line, balanceFromSum, balanceActual, delta}` | see 3.2 |
| `valueDateInversions` | array of `{line, previousLine}` | see 3.3 |

### 3.1 Continuity check

Over transactions **in file order**: `balanceAfter[i] == balanceAfter[i-1] + amount[i]`, with the
first row anchored at `openingBalance = balanceAfter[0] - amount[0]`. Because the reference file is
sorted and complete this check is **nearly tautological — it passes on the real file.** It is still
required: it is the cheap structural check.

### 3.2 Cumulative check

Independently of 3.1: running sum `S(i) = openingBalance + Σ amount[1..i]`, and `S(i) == balanceAfter[i]`
for every row. On the reference file this must reconstruct a final `37713.30` exactly.

### 3.3 Value-date inversion check

Count every adjacent pair where `valueDate` goes **backwards** relative to the previous row's
`valueDate` (booking order is the frame; value dates are what may move backwards). On the reference
file this must find **exactly 29**.

### 3.4 Accumulation requirement

Every check must traverse the **entire** file and report **all** violations. Never `raise`/throw, never
`return`/short-circuit on the first violation. A check that reports one error and stops is wrong.

### 3.5 Strictness

The two checks of 3.1 and 3.2 must compute their expectations **independently** — you may not derive
one from the other's running state.

---

## 4. Synthetic fixtures (you must create these)

Real files are well-behaved; the bugs live in the edge cases. Build your own fixture CSVs covering at
least: a fully empty row; a quoted field; a comma-decimal amount (`1234,56`); a malformed amount
(`abc`, `1.2.3`, `--5`); an ambiguous `1.234` (must be **rejected** as ambiguous, not guessed); a
missing `saldo`; a date that is not `DD/MM/YYYY`; CRLF line endings; a row where a **collision on
`(bookingDate, amount, balanceAfter)`** occurs (the reference file contains three such pairs at lines
243/247, 288/290 and 446/448 — any dedup keyed on natural fields silently merges real transactions,
which is a defect).

### 4.1 Amount parsing rules

Accept `.` or `,` as decimal separator. **Reject as ambiguous** any bare amount containing both, or any
value with a separator followed by exactly three digits and no other separator (`1.234` — could be
1234 or 1.234 depending on locale; guessing here is the real-world bug). Reject empty, non-numeric,
and multi-sign values. Money must be **exact decimal** — arbitrary precision. A binary float must not
be able to enter the domain model.

### 4.2 Date parsing rules

Strict `DD/MM/YYYY` with real calendar validation (reject `31/02/2026`). Do not use a lenient parser
that silently rolls dates over.

---

## 5. Iteration A — multi-account reconciliation and transfer detection

This is the *extension* exercise: implement it in the same module, after the baseline works. The real
CSV has no concept/beneficiary column, so this iteration is exercised on **synthetic fixtures only** —
that is expected and intended. Produce `accounts.json` per §6's interface.

**Inputs:** an accounts registry file listing `{id, currency, openingBalance}` per account, and one
statement CSV per account.

**Required behaviour:**

1. Per-account checks from §3, computed independently for each account.
2. **Transfer detection.** A transfer is a pairing of one **outflow** row from one account with one
   **inflow** row of another account where `|outflow.amount| == inflow.amount`, both in the same
   currency, satisfying `outflow.valueDate <= inflow.valueDate <= outflow.valueDate + toleranceDays`
   (default tolerance `3`, configurable). Constraints: each row pairs at most once; at most one
   outgoing and one incoming pair per row; prefer the **nearest** feasible date, breaking ties by the
   **lowest line number** so results are deterministic.
3. Rows left unpaired are simply not transfers — report nothing per row. Report a reserved counter
   `unpairedTransferCandidates` (leave at `0`; reserved for a later iteration).
4. **NEW CHECK:** `netWorthConserved` — if *every* row belonging to a pairing is marked as a transfer,
   then the change in summed net worth attributable to those rows alone must be exactly zero.
5. `netWorth(i) = Σ over accounts (openingBalance + Σ amount)`, and a `netWorthFromBalances(i) = Σ
   balanceAfter`. The two must agree at every observation point; report all mismatches.
6. Adding a **third account** to a fixture must require no change to the parsing code.

**Deliberate design pressure (this is the point of the iteration):** the pairing predicate and the
conservation invariant are exactly the kind of rule that Python represents as a scattered pile of
helper functions and that a closed, typed domain can make total and compiler-checked. If you find
yourself adding a sixth free function that re-derives an amount or a date, note it in `NOTES.md`.

---

## 6. CLI interface (frozen — both implementations must match exactly)

```
<prog> baseline <statement.csv>              -> prints the §3 JSON envelope to stdout
<prog> accounts <accounts.json>              -> prints the §5 JSON envelope to stdout
<prog> --help
```

**Envelope shape**

```json
{"ok": true,  "module": "ledger", "version": 1, "data": { ... }}
{"ok": false, "module": "ledger", "version": 1, "error": {"kind": "...", "message": "...", "line": 42}}
```

- Key order must be stable and identical between the two implementations: `ok`, `module`, `version`,
  then `data` (or `error`).
- **Determinism:** no timestamps, no absolute paths, no locale-dependent formatting, no hash-order
  iteration. Two runs on the same input must be byte-identical. Array order follows file order
  (transactions) or ascending line numbers (error arrays).
- **Exit codes:** `0` when the file is well-formed (even if checks report violations); `1` when the
  input cannot be parsed at all; `2` on usage error.
- The error `kind` values are frozen: `IO`, `HEADER`, `ROW`, `AMOUNT`, `DATE`, `REGISTRY`.
- Amounts serialise as canonical decimal strings — `37713.30`, not `37713.3`, and never `3.77133E4`.

---

## 7. Required metrics — record these honestly in `notes/NOTES.md`

Fill in the template that ships next to this spec. Record what actually happened, including failed
attempts and dead ends; a clean-looking log that hides the friction is worthless for this comparison.

Required fields: wall-clock from first command to all tests passing · number of compile/run attempts ·
number of test-failure → fix cycles · did the baseline pass tests **on the first execution** (yes/no) ·
total lines of production code · number of source files in the final module · external dependencies ·
which toolchain commands you used · the exact command that runs the CLI · for iteration A: files added,
files modified, and **did the compiler/type checker find any of the change sites for you** (be concrete,
and say "no" if the answer is no) · spec friction (ambiguities, things you had to decide) · anything the
language or tooling made harder or easier, with the concrete command or error as evidence.

Do not report a metric you did not measure. `unknown` is a valid answer and is better than a guess.

---

## 8. Testing

- Tests must cover §4's fixtures and both §3 and §5.
- A test suite that passes while the real reference file fails the §3.3 count of 29 is worthless —
  **assert the real numbers**: 923 transactions, `skippedEmptyRows = [2]`, `finalBalance = 37713.30`,
  `valueDateInversions.length = 29`, `continuityErrors` empty, `cumulativeErrors` empty.
- Include at least one property-style test, not only example-based tests.

---

## 9. Fairness rules

1. **Standard library first.** If the language's standard library can do it, do not add a dependency.
   If you genuinely cannot, choose the smallest, most standard option and record it in `NOTES.md`.
2. Do not consult, copy, or read the other language's implementation. It will not exist in your
   workspace.
3. Do not weaken, skip, or delete a test to get to green.
4. Do not modify this spec. Record friction instead.
5. Time spent installing a toolchain **counts** in the wall-clock metric — note it separately.

---

## 10. Workspace layout (create your side only)

```
lab/
  spec/ledger-spec.md          <- this file, read-only, do not modify
  benchmark/data/              <- reference CSV (read-only)
  python/                      <- PYTHON AGENT: everything you write goes here
    notes/NOTES.md             <- required metrics
  scala/                       <- SCALA AGENT: everything you write goes here
    notes/NOTES.md             <- required metrics
```

Paths above are relative to this benchmark directory.

Write **only** inside your own directory. Do not create, edit or read files in the other language's
directory.

### 10.1 Notes template — copy this shape into your `notes/NOTES.md`

```
# <language> ledger — run notes (2026-09-15)

## Toolchain
- version(s) used, and exact commands
- time spent installing/starting a toolchain (counted separately from build time)

## Baseline
- wall-clock, first command -> all tests green: 
- compile/run attempts:
- test-failure -> fix cycles:
- tests passed on FIRST execution (yes/no):
- lines of production code:
- source files in final module:
- external dependencies:
- exact command that runs the CLI:

## Iteration A (multi-account + transfers)
- wall-clock:
- files added:
- files modified:
- did the compiler/type checker find any change site for you? (yes/no + concrete evidence):
- free functions added that re-derive an amount/date (the §5 pressure):
- tests that failed and why:

## Friction
- spec friction (ambiguities, decisions you had to make):
- language/tooling friction, with the concrete command or error:
- what was easier than expected:

## Assessment
- would you extend this codebase? why / why not
- what the second developer to touch this would trip on
```

### 10.2 Environment notes for the Scala side

`scala-cli` 1.17.0 is pre-downloaded at `lab/.tools/scala-cli` (workspace-local; the home cache is
read-only in this sandbox). Invoke it as:

```
COURSIER_CACHE=$PWD/.tools/coursier \
  XDG_CACHE_HOME=$PWD/.tools/xdg-cache \
  $PWD/.tools/scala-cli <args>
```

Its default Scala version is 3.9.0. A JDK is on `PATH` (`java -version` = 26.0.2); a JDK 17 also
exists under `~/.local/share/mise/installs/java/17.0` if the newer one causes trouble — if you switch,
say so in the notes and count the time. `mise` shims may be needed for some tools; `uv` is available
for Python. If a package manager fails because `/home/mat` is read-only outside the workspace, work
around it with workspace-local paths and record it as friction.
