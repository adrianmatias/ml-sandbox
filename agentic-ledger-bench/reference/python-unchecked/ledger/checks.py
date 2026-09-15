"""The three baseline checks (spec §3.1–§3.4).

Two rules shape this module:

* **Accumulation (§3.4).** Every check walks the entire statement and returns
  *all* violations.  No ``raise``, no early ``return``, no `break` on the first
  bad row.
* **Independence (§3.5).** :func:`continuity_errors` and
  :func:`cumulative_errors` derive their expectations by different routes: the
  first from the *neighbouring row's* recorded balance, the second from a
  running total anchored at the opening balance.  Neither reads the other's
  state; nothing here reuses a shared accumulator.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable, Sequence

from .model import ParsedStatement, Transaction
from .strictdecimal import canonical, exact_sum

LINE = "line"
PREVIOUS_LINE = "previousLine"


def continuity_errors(
    transactions: Sequence[Transaction], opening_balance: Decimal
) -> list[dict]:
    """§3.1 — ``balanceAfter[i] == balanceAfter[i-1] + amount[i]``.

    Reported as ``{line, balanceExpected, balanceActual, delta}`` where
    ``delta = balanceActual - balanceExpected``.  ``opening_balance`` is passed
    in because the envelope reports it, but the expectation for row *i* is built
    purely from row *i-1*: that is what makes this the *local* check.
    """
    errors: list[dict] = []
    previous: Transaction | None = None
    for transaction in transactions:
        if previous is not None:
            expected = previous.balance_after + transaction.amount
            actual = transaction.balance_after
            if actual != expected:
                errors.append(
                    {
                        LINE: transaction.line,
                        "balanceExpected": canonical(expected),
                        "balanceActual": canonical(actual),
                        "delta": canonical(actual - expected),
                    }
                )
        previous = transaction
    return errors


def cumulative_errors(
    transactions: Sequence[Transaction], opening_balance: Decimal
) -> list[dict]:
    """§3.2 — ``S(i) = openingBalance + Σ amount[1..i]`` must equal ``balanceAfter[i]``.

    Reported as ``{line, balanceFromSum, balanceActual, delta}`` where
    ``delta = balanceActual - balanceFromSum``.  The running total is carried
    independently of any recorded balance, so a single wrong ``saldo`` does not
    propagate into silently matching neighbours the way §3.1's anchor does.
    """
    errors: list[dict] = []
    running: Decimal = opening_balance
    for transaction in transactions:
        running = running + transaction.amount
        actual = transaction.balance_after
        if running != actual:
            errors.append(
                {
                    LINE: transaction.line,
                    "balanceFromSum": canonical(running),
                    "balanceActual": canonical(actual),
                    "delta": canonical(actual - running),
                }
            )
    return errors


def value_date_inversions(transactions: Sequence[Transaction]) -> list[dict]:
    """§3.3 — adjacent pairs whose ``valueDate`` moves backwards.

    Booking order is the frame; value dates are what may move backwards.  Equal
    value dates are not inversions.  Reported as ``{line, previousLine}``.
    """
    inversions: list[dict] = []
    previous: Transaction | None = None
    for transaction in transactions:
        if previous is not None and transaction.value_date < previous.value_date:
            inversions.append({LINE: transaction.line, PREVIOUS_LINE: previous.line})
        previous = transaction
    return inversions


def build_checks(
    statement: ParsedStatement, opening_balance: Decimal | None = None
) -> dict:
    """Assemble the §3 ``checks`` object, fully accumulated and canonically ordered.

    ``opening_balance`` overrides the statement's own anchor (§5's registry
    ``openingBalance`` is authoritative for an account, and must not be
    re-derived from the first row).  ``None`` keeps the baseline behaviour:
    ``balanceAfter[0] - amount[0]``.

    ``reconciled`` follows CLARIFICATIONS.md §1 (binding over the spec's looser
    "true iff every array below is empty"): it is true exactly when
    ``continuityErrors`` and ``cumulativeErrors`` are both empty.
    ``valueDateInversions`` and ``skippedEmptyRows`` are factual observations
    about the data, not defects, and deliberately do not affect the boolean —
    a value-date inversion cannot be reconciled away.
    """
    anchor = statement.opening_balance if opening_balance is None else opening_balance
    continuity = continuity_errors(statement.transactions, anchor)
    cumulative = cumulative_errors(statement.transactions, anchor)
    inversions = value_date_inversions(statement.transactions)
    return {
        "rowCount": len(statement.transactions),
        "skippedEmptyRows": list(statement.skipped_empty_rows),
        "reconciled": not (continuity or cumulative),
        "continuityErrors": continuity,
        "cumulativeErrors": cumulative,
        "valueDateInversions": inversions,
        "openingBalance": canonical(anchor),
        "finalBalance": canonical(statement.final_balance),
    }


def build_baseline(statement: ParsedStatement) -> dict:
    """The §3 ``data`` payload: transactions in file order, then checks."""
    return {
        "transactions": [
            transaction.as_json() for transaction in statement.transactions
        ],
        "checks": build_checks(statement),
    }


def total_of(transactions: Iterable[Transaction]) -> Decimal:
    """Exact sum of row amounts (kept here so callers never re-derive with floats)."""
    return exact_sum(transaction.amount for transaction in transactions)
