"""Domain model and parsing for a Kutxa statement (spec §2, §3, §4).

The model is deliberately opaque-by-construction: a ``Transaction`` can only be
built by :func:`parse_statement`, and every field on it has been validated and
converted to an exact ``Decimal`` or a real ``date``.  There is no "raw string
that is probably an amount" state anywhere past this module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Iterator

from . import csvio, strictdecimal
from .dates import DateError, iso, parse_date_token
from .errors import AMOUNT, DATE, HEADER, ROW, LedgerError
from .strictdecimal import AmountError, canonical

#: Frozen header, exactly four columns in this order (spec §2).
EXPECTED_HEADER = ("fecha", "fecha valor", "importe", "saldo")


@dataclass(frozen=True, slots=True)
class Transaction:
    """One data row of a statement, fully validated."""

    line: int
    booking_date: date
    value_date: date
    amount: Decimal
    balance_after: Decimal
    raw_amount: str
    raw_balance: str

    @property
    def checksum(self) -> str:
        """Hex of the first 8 bytes of sha256(amount|balanceAfter|bookingDate).

        Built from the *canonical* amount and balance, so the digest depends on
        the value and not on how the file happened to spell it (``400`` and
        ``400.00`` hash identically).
        """
        preimage = "|".join(
            (
                canonical(self.amount),
                canonical(self.balance_after),
                iso(self.booking_date),
            )
        )
        return hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]

    def as_json(self) -> dict:
        """Frozen key order: line, bookingDate, valueDate, amount, balanceAfter,
        rawAmount, rawBalance, checksum (spec §3)."""
        return {
            "line": self.line,
            "bookingDate": iso(self.booking_date),
            "valueDate": iso(self.value_date),
            "amount": canonical(self.amount),
            "balanceAfter": canonical(self.balance_after),
            "rawAmount": self.raw_amount,
            "rawBalance": self.raw_balance,
            "checksum": self.checksum,
        }


@dataclass(frozen=True, slots=True)
class ParsedStatement:
    """A parsed statement: the rows in file order plus what was skipped."""

    transactions: tuple[Transaction, ...]
    skipped_empty_rows: tuple[int, ...]
    opening_balance: Decimal
    """``balanceAfter[0] - amount[0]`` — the anchor of both §3 checks.

    ``Decimal(0)`` for an empty statement; the two checks report nothing for an
    empty statement, because there is nothing to be continuous with.
    """

    @property
    def final_balance(self) -> Decimal:
        return self.transactions[-1].balance_after if self.transactions else Decimal(0)

    def __iter__(self) -> Iterator[Transaction]:
        return iter(self.transactions)

    def __len__(self) -> int:
        return len(self.transactions)


def parse_statement(path: str) -> ParsedStatement:
    """Parse a statement CSV into exact domain values (spec §2, §4).

    Raises ``LedgerError`` with a frozen ``kind`` on the first unreadable row:
    parse errors are fatal and reported individually (spec §6), unlike *check*
    violations, which accumulate (spec §3.4).
    """
    try:
        lines = csvio.read_csv_lines(path)
    except FileNotFoundError as exc:
        raise LedgerError("IO", f"statement file not found: {path}") from exc
    except IsADirectoryError as exc:
        raise LedgerError("IO", f"statement path is a directory: {path}") from exc
    except PermissionError as exc:
        raise LedgerError("IO", f"statement file is not readable: {path}") from exc
    except UnicodeDecodeError as exc:
        raise LedgerError("IO", f"statement file is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise LedgerError(
            "IO", f"cannot read statement file: {path}: {exc.strerror}"
        ) from exc

    if not lines:
        raise LedgerError(HEADER, "statement file is empty: no header line", 1)

    _check_header(lines[0], path)

    transactions: list[Transaction] = []
    skipped: list[int] = []

    for line_number, raw_line in enumerate(lines[1:], start=2):
        try:
            fields = csvio.tokenize_line(raw_line)
        except LedgerError as exc:
            raise LedgerError(ROW, exc.message, line_number) from exc

        if csvio.is_empty_record(fields):
            skipped.append(line_number)
            continue

        if len(fields) != len(EXPECTED_HEADER):
            raise LedgerError(
                ROW,
                f"expected {len(EXPECTED_HEADER)} fields, found {len(fields)}",
                line_number,
            )

        booking_raw, value_raw, amount_raw, balance_raw = fields
        booking_date = _date(booking_raw, "fecha", line_number)
        value_date = _date(value_raw, "fecha valor", line_number)
        amount = _amount(amount_raw, "importe", line_number)
        balance_after = _amount(balance_raw, "saldo", line_number)

        transactions.append(
            Transaction(
                line=line_number,
                booking_date=booking_date,
                value_date=value_date,
                amount=amount,
                balance_after=balance_after,
                raw_amount=amount_raw.strip(),
                raw_balance=balance_raw.strip(),
            )
        )

    opening = (
        transactions[0].balance_after - transactions[0].amount
        if transactions
        else Decimal(0)
    )
    return ParsedStatement(tuple(transactions), tuple(skipped), opening)


def _check_header(header_line: str, path: str) -> None:
    """The header must be exactly the four frozen columns, in order."""
    try:
        fields = csvio.tokenize_line(header_line)
    except LedgerError as exc:
        raise LedgerError(HEADER, exc.message, 1) from exc
    normalized = tuple(field.strip() for field in fields)
    if normalized == EXPECTED_HEADER:
        return
    if len(normalized) != len(EXPECTED_HEADER):
        raise LedgerError(
            HEADER,
            f"expected {len(EXPECTED_HEADER)} header columns, found {len(normalized)}",
            1,
        )
    for position, (found, expected) in enumerate(
        zip(normalized, EXPECTED_HEADER), start=1
    ):
        if found != expected:
            raise LedgerError(
                HEADER,
                f"header column {position} is {found!r}, expected {expected!r}",
                1,
            )
    raise LedgerError(HEADER, "header does not match the expected columns", 1)


def _date(token: str, field_name: str, line_number: int) -> date:
    try:
        return parse_date_token(token)
    except DateError as exc:
        raise LedgerError(DATE, f"{field_name}: {exc.message}", line_number) from exc


def _amount(token: str, field_name: str, line_number: int) -> Decimal:
    try:
        return strictdecimal.parse_amount_token(token)
    except AmountError as exc:
        raise LedgerError(AMOUNT, f"{field_name}: {exc.message}", line_number) from exc
