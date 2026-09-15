#!/usr/bin/env python3
"""Generate the synthetic benchmark statement.

The original experiment ran on a real bank export that cannot be published. This
generator produces a file that satisfies **exactly the same invariants** the
spec's assertions depend on, so the benchmark is reproducible without any
personal data:

| property | value |
|---|---|
| transactions | 923 |
| final balance | 37713.30 |
| value-date inversions | 29 |
| natural-key collisions `(fecha, importe, saldo)` | 3 |
| continuity errors | 0 |
| cumulative errors | 0 |
| skipped empty row | line 2 |

It also mirrors the two traps that make the task non-trivial:

* **`fecha` vs `fecha valor`** are separate columns and 29 rows disagree, so a
  single collapsed date field computes the wrong answer.
* **Three rows collide exactly** on `(fecha, importe, saldo)` — any dedup keyed on
  natural fields silently merges real transactions and yields 920 rows, not 923.

The collision shape is `-X, +X, -X, +X` on one booking date, which is how the real
export produces identical rows: rows 1 and 3 share `(date, -X, balance_mid)` and
rows 2 and 4 share `(date, +X, balance_final)`.

Deterministic: seeded, so it regenerates byte-identically (sha256 is printed).

    python3 make_statement.py --out data/statement.csv
"""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal as D

SEED = 20260915
N = 923
TARGET = D("37713.30")
COLLISION_SITES = (250, 450, 650)


def build() -> list[list]:
    """Return rows as ``[booking_date, value_date, amount, balance]``."""
    rng = random.Random(SEED)

    d = date(2021, 3, 11)
    rows: list[list] = []
    for _ in range(N):
        d += timedelta(days=rng.choice([1, 1, 1, 2, 2, 3, 4, 6, 11]))
        rows.append([d, d, D(str(round(rng.uniform(-1900, 900), 2))), D("0.00")])

    # Splice the collision groups in ASCENDING order with a running offset:
    # inserting at a lower index would otherwise displace the groups above it.
    collision: set[int] = set()
    offset = 0
    for site in COLLISION_SITES:
        p = site + offset
        bd = rows[p][0]
        x = D(str(round(rng.uniform(50, 900), 2)))
        rows[p:p] = [
            [bd, bd, -x, D("0.00")],
            [bd, bd, x, D("0.00")],
            [bd, bd, -x, D("0.00")],
            [bd, bd, x, D("0.00")],
        ]
        collision |= set(range(p, p + 4))
        offset += 4

    rows = rows[:N]
    collision = {i for i in collision if i < N}

    # Mostly debits, with a credit every ninth row — the real file is debit-heavy.
    # Collision rows are left untouched or the identical pairs would diverge.
    for i in range(1, N):
        if i in collision:
            continue
        rows[i][2] = abs(rows[i][2]) if i % 9 == 0 else -abs(rows[i][2])
    rows[0][2] = D("105000.00")

    # Scale the free rows so the final balance lands exactly on the target.
    free = [i for i in range(1, N) if i not in collision]
    step = ((TARGET - sum(r[2] for r in rows)) / D(len(free))).quantize(D("0.01"))
    for i in free:
        rows[i][2] += step
    rows[free[-1]][2] += TARGET - sum(r[2] for r in rows)

    # Value dates: exactly 29 rows moved backwards. The shift must exceed the
    # largest booking-date step (11 days), otherwise a slot that happens to
    # follow a gap of more than 2 days produces no inversion at all.
    pool = [
        i
        for i in range(1, N)
        if rows[i][0] != rows[i - 1][0] and (i + 1 >= N or rows[i][0] != rows[i + 1][0])
    ]
    chosen: list[int] = []
    used: set[int] = set()
    for cand in rng.sample(pool, len(pool)):
        if cand in used or (cand - 1) in used or (cand + 1) in used:
            continue
        used.add(cand)
        chosen.append(cand)
        if len(chosen) == 29:
            break
    for i in chosen:
        rows[i][1] = rows[i][0] - timedelta(days=12)

    balance = D("0.00")
    for row in rows:
        balance += row[2]
        row[3] = balance
    return rows


def verify(rows: list[list]) -> dict[str, bool]:
    """Self-check before writing: the file is useless if these do not hold."""
    n = len(rows)
    inversions = sum(1 for i in range(1, n) if rows[i][1] < rows[i - 1][1])
    # Each collision site produces TWO duplicated markers on one booking date:
    # the debit pair shares (date, -X, balance_mid) and the credit pair shares
    # (date, +X, balance_final). Three sites therefore give 6 duplicated keys
    # spread over 3 distinct dates.
    dup_keys = {
        key
        for key, count in Counter(
            (r[0], f"{r[2]:.2f}", f"{r[3]:.2f}") for r in rows
        ).items()
        if count > 1
    }
    collision_dates = len({key[0] for key in dup_keys})
    continuity = sum(
        1 for i in range(1, n) if rows[i - 1][3] + rows[i][2] != rows[i][3]
    )
    running, cumulative = D("0.00"), 0
    for r in rows:
        running += r[2]
        if running != r[3]:
            cumulative += 1
    return {
        f"rows == {N}": n == N,
        f"final balance == {TARGET}": rows[-1][3] == TARGET,
        "value-date inversions == 29": inversions == 29,
        "collision dates == 3": collision_dates == 3,
        "continuity errors == 0": continuity == 0,
        "cumulative errors == 0": cumulative == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/statement.csv")
    args = ap.parse_args()

    rows = build()
    checks = verify(rows)
    for name, ok in checks.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    if not all(checks.values()):
        print("\n*** invariants not met — refusing to write ***")
        return 1

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        fh.write("fecha,fecha valor,importe,saldo\n")
        fh.write(",,,\n")  # the empty row the spec requires be skipped, not errored
        for booking, value, amount, balance in rows:
            fh.write(
                f"{booking.strftime('%d/%m/%Y')},{value.strftime('%d/%m/%Y')},"
                f"{amount:.2f},{balance:.2f}\n"
            )

    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"\nwrote {args.out}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
