"""The three §3 checks: continuity, cumulative reconstruction, value-date inversions.

Strictness rules that shape this module:

* §3.4 — every check traverses the **whole** file and reports **all** violations;
  nothing short-circuits and nothing raises on a violation.
* §3.5 — continuity and cumulative must compute their expectations
  **independently**. Continuity compares each row against the *previous row's
  balance*; cumulative compares against ``opening_balance + Σ amount``. Neither
  reads the other's running state, and each is a separate function here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise
from typing import Final

from ledger.domain import Balance, LineNumber, RowFailure, Transaction

OPENING_SOURCE_FIRST_ROW: Final[str] = "firstRow"
"""``openingBalance`` was derived as ``balanceAfter[0] - amount[0]`` (spec §3.1)."""

OPENING_SOURCE_DECLARED: Final[str] = "declared"
"""``openingBalance`` came from the accounts registry (spec §5)."""

OPENING_SOURCE_ZERO: Final[str] = "zero"
"""The statement is empty, so the opening balance is the declared one (or zero)."""


@dataclass(frozen=True, slots=True)
class ContinuityError:
    """``balanceAfter[i] != balanceAfter[i-1] + amount[i]`` (spec §3.1)."""

    line: LineNumber
    balance_expected: Balance
    balance_actual: Balance
    delta: Decimal


@dataclass(frozen=True, slots=True)
class CumulativeError:
    """``balanceAfter[i] != openingBalance + Σ amount[1..i]`` (spec §3.2)."""

    line: LineNumber
    balance_from_sum: Balance
    balance_actual: Balance
    delta: Decimal


@dataclass(frozen=True, slots=True)
class ValueDateInversion:
    """An adjacent pair whose value date moves backwards (spec §3.3).

    This is a factual observation about the data, not an error: per
    ``CLARIFICATIONS.md`` §1 it never affects ``reconciled``.
    """

    line: LineNumber
    previous_line: LineNumber


@dataclass(frozen=True, slots=True)
class OpeningBalance:
    """The anchor every cumulative expectation is measured from.

    Attributes:
        amount: ``balanceAfter[0] - amount[0]`` when the statement is non-empty.
        source: Why this value is trusted — see the ``OPENING_SOURCE_*`` constants.
        declared_mismatch: ``(declared, derived)`` when a registry-declared
            opening balance disagrees with the statement's own first row.
    """

    amount: Balance
    source: str
    declared_mismatch: tuple[Balance, Balance] | None = None


@dataclass(frozen=True, slots=True)
class CheckReport:
    """Every §3 observation for one statement, all of them complete.

    Attributes:
        row_count: Rows parsed into transactions.
        skipped_empty_rows: Lines of fully empty rows, ascending.
        continuity_errors: All §3.1 violations, in file order.
        cumulative_errors: All §3.2 violations, in file order.
        value_date_inversions: All §3.3 violations, in file order.
        parse_errors: Rows that could not be parsed at all, in ascending line order.
    """

    row_count: int
    skipped_empty_rows: tuple[LineNumber, ...]
    continuity_errors: tuple[ContinuityError, ...]
    cumulative_errors: tuple[CumulativeError, ...]
    value_date_inversions: tuple[ValueDateInversion, ...]
    parse_errors: tuple[RowFailure, ...]

    @property
    def reconciled(self) -> bool:
        """True iff no continuity error and no cumulative error was found.

        Binding definition from ``CLARIFICATIONS.md`` §1: value-date inversions
        and skipped empty rows are observations, not errors, and do not
        influence this flag.
        """
        return not self.continuity_errors and not self.cumulative_errors


def derive_opening_balance(
    transactions: Sequence[Transaction],
    declared: Balance | None = None,
) -> OpeningBalance:
    """Anchor the cumulative check at ``balanceAfter[0] - amount[0]`` (spec §3.1).

    Args:
        transactions: Parsed rows in file order; may be empty.
        declared: An opening balance from the accounts registry, if any. When
            given, the derived value is cross-checked against it and any
            disagreement is reported rather than silently preferred.

    Returns:
        The anchor plus the provenance of the value.
    """
    if not transactions:
        return OpeningBalance(
            amount=declared if declared is not None else Balance(Decimal(0)),
            source=(
                OPENING_SOURCE_DECLARED if declared is not None else OPENING_SOURCE_ZERO
            ),
        )
    first = transactions[0]
    derived = Balance(first.balance_after - first.amount)
    if declared is None:
        return OpeningBalance(amount=derived, source=OPENING_SOURCE_FIRST_ROW)
    if declared == derived:
        return OpeningBalance(amount=derived, source=OPENING_SOURCE_FIRST_ROW)
    return OpeningBalance(
        amount=derived,
        source=OPENING_SOURCE_FIRST_ROW,
        declared_mismatch=(declared, derived),
    )


def check_continuity(
    transactions: Sequence[Transaction],
) -> tuple[ContinuityError, ...]:
    """§3.1 — each row's balance equals the previous row's balance plus its amount.

    The first row has no predecessor and therefore cannot violate this check;
    it is the anchor (see :func:`derive_opening_balance`).
    """
    errors: list[ContinuityError] = []
    for previous, current in pairwise(transactions):
        expected = Balance(previous.balance_after + current.amount)
        if expected != current.balance_after:
            errors.append(
                ContinuityError(
                    line=current.line,
                    balance_expected=expected,
                    balance_actual=current.balance_after,
                    delta=Decimal(current.balance_after - expected),
                )
            )
    return tuple(errors)


def check_cumulative(
    transactions: Sequence[Transaction],
    opening: OpeningBalance,
) -> tuple[CumulativeError, ...]:
    """§3.2 — running ``opening + Σ amount`` must reconstruct every balance.

    Computed from ``opening`` and the amounts alone. It deliberately does not
    read any state produced by :func:`check_continuity` (spec §3.5).
    """
    errors: list[CumulativeError] = []
    running = Decimal(opening.amount)
    for transaction in transactions:
        running += transaction.amount
        expected = Balance(running)
        if expected != transaction.balance_after:
            errors.append(
                CumulativeError(
                    line=transaction.line,
                    balance_from_sum=expected,
                    balance_actual=transaction.balance_after,
                    delta=Decimal(transaction.balance_after - expected),
                )
            )
    return tuple(errors)


def check_value_date_inversions(
    transactions: Sequence[Transaction],
) -> tuple[ValueDateInversion, ...]:
    """§3.3 — every adjacent pair whose value date moves backwards.

    Booking order is the frame; the two dates are never collapsed, so rows
    whose booking and value dates differ are compared on their value dates.
    """
    inversions: list[ValueDateInversion] = []
    for previous, current in pairwise(transactions):
        if current.value_date < previous.value_date:
            inversions.append(
                ValueDateInversion(line=current.line, previous_line=previous.line)
            )
    return tuple(inversions)


def run_checks(
    transactions: Sequence[Transaction],
    *,
    skipped_empty_rows: Sequence[LineNumber] = (),
    parse_errors: Sequence[RowFailure] = (),
    declared_opening: Balance | None = None,
) -> CheckReport:
    """Run all three §3 checks over one statement and collect every observation.

    Args:
        transactions: Parsed rows in file order.
        skipped_empty_rows: Lines of fully empty rows, as recorded by the parser.
        parse_errors: Per-row failures recorded by the parser.
        declared_opening: Opening balance from a registry, when one is known.

    Returns:
        A complete :class:`CheckReport`; no check ever stops early.
    """
    opening = derive_opening_balance(transactions, declared_opening)
    return CheckReport(
        row_count=len(transactions),
        skipped_empty_rows=tuple(skipped_empty_rows),
        continuity_errors=check_continuity(transactions),
        cumulative_errors=check_cumulative(transactions, opening),
        value_date_inversions=check_value_date_inversions(transactions),
        parse_errors=tuple(parse_errors),
    )


def final_balance(transactions: Sequence[Transaction], fallback: Balance) -> Balance:
    """Balance after the last row, or ``fallback`` when the statement is empty."""
    if not transactions:
        return fallback
    return transactions[-1].balance_after
