# Shared iteration-A fixture + two clarifications (issued 2026-09-15, after agents started)

These are the **only** two changes to the frozen spec. Both are clarifications of text that was
ambiguous, not new requirements. Both must be implemented.

---

## Clarification 1 — what `reconciled` means (spec §3, "checks" table)

The spec's line *"true iff every array below is empty"* is ambiguous, and the two implementations
disagreed on it. The **binding** definition is:

> `reconciled` is `true` exactly when these two arrays are empty: **`continuityErrors`** and
> **`cumulativeErrors`**.
>
> `valueDateInversions` and `skippedEmptyRows` are **factual observations**, not errors, and must
> **not** affect `reconciled`. A value-date inversion is inherent to the data and cannot be
> "reconciled away" — on the reference file there are exactly 29 of them.

Consequence for the reference file: **`reconciled` must be `true`** even though
`valueDateInversions.length == 29`. Keep all three arrays populated exactly as specified; only the
boolean's definition changes.

---

## Clarification 2 — use these shared fixtures for iteration A

Per-language fixtures cannot be compared across languages. Every implementation must additionally
produce output for **these** inputs, which live in `spec/shared-fixtures/`:

- `accounts.json` — registry, ids `A`, `B`, `C`, all `EUR`, all opening balances `0.00`, `toleranceDays` 3
- `A.csv`, `B.csv`, `C.csv` — one statement per account

Expected results, computed independently by the orchestrator and **not** to be taken from any
implementation:

| property | expected value |
|---|---|
| accounts | 3 |
| netWorth (Σ opening + Σ amounts) | `1139.25` |
| netWorth (Σ final balances) | `1139.25` — must agree |
| per-account final balance | A `455.25`, B `429.50`, C `254.50` |
| continuity errors, all accounts | `0` |
| cumulative errors, all accounts | `0` |
| **transfer pairs** | **exactly 2** |

The two pairs, under spec §5's rule (same currency, equal absolute amount, other account,
`outflow.valueDate <= inflow.valueDate <= outflow.valueDate + toleranceDays`, prefer nearest
feasible date, tie broken by lowest line number):

1. `A` line 3, `-250`, value date `2026-01-05` → `B` line 4, `+250`, value date `2026-01-06` (gap 1 day)
2. `A` line 4, `-300`, value date `2026-01-10` → `C` line 2, `+300`, value date `2026-01-10` (gap 0 days)

Net effect of both pairs must be exactly `0.00`, so `netWorthConserved` holds.

**Deliberate traps in this fixture** — do not "fix" them, they are the test:

- `B` line 3 (`-250`) and `C` line 4 (`+75`) are **not** transfers under the rule: the amounts differ,
  and `C`'s `+75` value date (15/01) precedes the candidate outflow's value date (20/01).
- `A` line 5 (`+120.50`, value date 15/01) is a decoy candidate for `B` line 6 (`-120.50`) and
  `C` line 4 (`-120.50`): its date is *after* them, so the ordering constraint fails.
- `B` line 6 (`-120.50`) and `C` line 4 (`-120.50`) are equal in magnitude and both EUR but must
  **not** pair with each other — both are outflows.
- `A` line 2 (`+1000`) and `B` line 2 (`+500`) are opening credits with no counterparty — not transfers.

Write this fixture's output to `out/accounts.json` (in addition to whatever output your own
per-language fixtures produce). Do not hand-edit it; it must come from your CLI.

---

## Note on scope

Because the real reference file has only four columns (`fecha, fecha valor, importe, saldo`) and no
counterparty or concept field, transfer detection cannot be exercised on it. That is expected:
iteration A runs on fixtures. The real file remains the arbiter for §3.
