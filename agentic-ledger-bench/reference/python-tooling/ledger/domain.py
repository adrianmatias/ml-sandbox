"""The domain model: exact-decimal money, calendar dates and immutable statements.

Design rule taken from spec §4.1: *a binary float must not be able to enter the
domain model.* Every monetary value here is a :class:`decimal.Decimal` created
from a **string**; ``float`` never appears in the parse path or in any signature
of this package. ``format_money`` is the single serialisation point, so the
canonical ``37713.30`` form cannot drift between the baseline and iteration A.
"""

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Final, NewType

Amount = NewType("Amount", Decimal)
"""A signed money amount: positive credit, negative debit (spec §2)."""

Balance = NewType("Balance", Decimal)
"""An account balance *after* a row (spec §2)."""

LineNumber = NewType("LineNumber", int)
"""1-based physical line number in a CSV file, header counted as line 1."""

TWO_PLACES: Final[Decimal] = Decimal("0.01")
"""Quantum used for canonical decimal serialisation."""


def amount(value: str) -> Amount:
    """Wrap an exact decimal string as an :class:`Amount`.

    This is the only sanctioned way to create money from a literal: going
    through ``Decimal(str)`` keeps the value exact and never touches a float.
    """
    return Amount(Decimal(value))


def balance(value: str) -> Balance:
    """Wrap an exact decimal string as a :class:`Balance`.

    See :func:`amount` for why the input is a string and not a float.
    """
    return Balance(Decimal(value))


def format_money(value: Decimal) -> str:
    """Serialise money canonically: ``37713.30``, never ``37713.3`` or ``3.77133E4``.

    Both implementations must agree byte for byte (spec §6), so this is the one
    and only money serialiser in the package.
    """
    return str(value.quantize(TWO_PLACES, rounding=ROUND_HALF_EVEN))


@dataclass(frozen=True, slots=True)
class Transaction:
    """One parsed statement row, in file order.

    Attributes:
        line: 1-based physical line in the source CSV.
        booking_date: ``fecha`` — booking date.
        value_date: ``fecha valor`` — value date, semantically distinct from
            :attr:`booking_date` (spec §2) and never collapsed into it.
        amount: Exact signed amount.
        balance_after: Exact balance after this row.
        raw_amount: The amount exactly as it appeared in the file.
        raw_balance: The balance exactly as it appeared in the file.
        checksum: Hex of the first 8 bytes of
            ``sha256(amount|balanceAfter|bookingDate)`` (spec §3).
    """

    line: LineNumber
    booking_date: date
    value_date: date
    amount: Amount
    balance_after: Balance
    raw_amount: str
    raw_balance: str
    checksum: str


@dataclass(frozen=True, slots=True)
class RowFailure:
    """A row that could not be turned into a :class:`Transaction`.

    Attributes:
        line: 1-based physical line number.
        kind: Frozen error kind (``ROW``, ``AMOUNT`` or ``DATE``).
        message: Explanation, carrying no locale-dependent formatting.
    """

    line: LineNumber
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class ParsedStatement:
    """Everything a successful parse produces, including the empty rows it skipped.

    Attributes:
        transactions: Parsed rows, in file order.
        skipped_empty_rows: 1-based lines of fully empty rows (``,,,``), which
            spec §2 requires to be skipped rather than reported as errors.
        row_errors: Non-fatal per-row failures, in ascending line order. A row
            that cannot be parsed is excluded from ``transactions``, and its
            failure is reported instead of aborting the whole file — see the
            "spec friction" entry in ``notes/NOTES.md``.
    """

    transactions: tuple[Transaction, ...]
    skipped_empty_rows: tuple[LineNumber, ...]
    row_errors: tuple[RowFailure, ...]
