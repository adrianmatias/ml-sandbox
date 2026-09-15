"""Kutxa statement CSV parser — spec §2 plus the §4.1/§4.2 strictness rules.

The parser is deliberately total: a row it cannot understand becomes a
:class:`~ledger.domain.RowFailure` and the scan continues to the end of the
file, because spec §3.4 forbids short-circuiting and §8 asks for *all*
violations. Structure problems (missing header, malformed quoting) are fatal
and raise.
"""

import csv
import hashlib
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from typing import Final

from ledger.amounts import parse_amount, parse_balance
from ledger.dates import parse_date
from ledger.domain import (
    LineNumber,
    ParsedStatement,
    RowFailure,
    Transaction,
    format_money,
)
from ledger.errors import (
    AmountError,
    DateError,
    EmptyStatementError,
    HeaderMismatchError,
    LedgerError,
    RowError,
    ShortRowError,
    UnreadableCsvError,
)
from ledger.textio import read_utf8_text

EXPECTED_HEADER: Final[tuple[str, ...]] = ("fecha", "fecha valor", "importe", "saldo")
"""The frozen four-column header of spec §2."""

REQUIRED_COLUMNS: Final[int] = len(EXPECTED_HEADER)
"""Rows with fewer physical columns than this are structural errors."""

_HEADER_LINE: Final[int] = 1
_FIRST_DATA_LINE: Final[int] = 2
_CHECKSUM_BYTES: Final[int] = 8


def compute_checksum(amount_text: str, balance_text: str, booking_iso: str) -> str:
    """Hex of the first 8 bytes of ``sha256(amount|balanceAfter|bookingDate)``.

    Args:
        amount_text: Canonical amount string, e.g. ``-250.00``.
        balance_text: Canonical balance string, e.g. ``750.00``.
        booking_iso: ISO booking date, e.g. ``2026-01-05``.

    Returns:
        The first 16 hex characters of the digest — 8 bytes, per spec §3.
    """
    payload = f"{amount_text}|{balance_text}|{booking_iso}".encode()
    return hashlib.sha256(payload).hexdigest()[: _CHECKSUM_BYTES * 2]


def parse_rows(text: str) -> list[list[str]]:
    """Split CSV text into rows, honouring quotes and CRLF line endings.

    Raises:
        UnreadableCsvError: If the CSV quoting itself is malformed.
    """
    try:
        return list(csv.reader(StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise UnreadableCsvError(str(exc)) from exc


def validate_header(rows: Sequence[Sequence[str]]) -> None:
    """Check the first physical line against the frozen §2 header.

    Raises:
        HeaderError: If the file is empty.
        HeaderMismatchError: If the header line differs from §2.
    """
    if not rows:
        raise EmptyStatementError
    actual = tuple(cell.strip() for cell in rows[0])
    if actual != EXPECTED_HEADER:
        raise HeaderMismatchError(actual, _HEADER_LINE)


def _row_failure(line: int, error: LedgerError) -> RowFailure:
    """Turn a row-scoped :class:`LedgerError` into a serialisable failure."""
    return RowFailure(LineNumber(line), error.kind, error.message)


def _build_transaction(line: int, cells: Sequence[str]) -> Transaction:
    """Parse one non-empty data row into a :class:`Transaction`.

    Raises:
        ShortRowError: If the row has fewer than four columns.
        DateError: If either date field is not a real ``DD/MM/YYYY`` date.
        AmountError: If either money field is empty, ambiguous or malformed.
    """
    if len(cells) < REQUIRED_COLUMNS:
        raise ShortRowError(len(cells), REQUIRED_COLUMNS, line)
    raw_booking, raw_value, raw_amount, raw_balance = (
        cell.strip() for cell in cells[:4]
    )
    booking_date = parse_date(raw_booking, line)
    value_date = parse_date(raw_value, line)
    amount = parse_amount(raw_amount, line)
    balance_after = parse_balance(raw_balance, line)
    amount_text = format_money(amount)
    balance_text = format_money(balance_after)
    booking_iso = booking_date.isoformat()
    return Transaction(
        line=LineNumber(line),
        booking_date=booking_date,
        value_date=value_date,
        amount=amount,
        balance_after=balance_after,
        raw_amount=raw_amount,
        raw_balance=raw_balance,
        checksum=compute_checksum(amount_text, balance_text, booking_iso),
    )


def parse_statement(text: str) -> ParsedStatement:
    """Parse a complete statement CSV.

    Args:
        text: The whole file, decoded as UTF-8 with any BOM already removed.

    Returns:
        The parsed transactions in file order, plus skipped empty rows and the
        per-row failures that were tolerated.

    Raises:
        HeaderError: If the header is missing or does not match §2.
        UnreadableCsvError: If the CSV quoting is malformed.
    """
    rows = parse_rows(text)
    validate_header(rows)
    transactions: list[Transaction] = []
    skipped: list[LineNumber] = []
    failures: list[RowFailure] = []
    for index in range(_FIRST_DATA_LINE - 1, len(rows)):
        line = index + 1
        cells = rows[index]
        if not any(cell.strip() for cell in cells):
            skipped.append(LineNumber(line))
            continue
        try:
            transactions.append(_build_transaction(line, cells))
        except (RowError, DateError, AmountError) as exc:
            failures.append(_row_failure(line, exc))
    return ParsedStatement(
        transactions=tuple(transactions),
        skipped_empty_rows=tuple(skipped),
        row_errors=tuple(failures),
    )


def parse_statement_file(path: Path) -> ParsedStatement:
    """Read and parse a statement CSV from disk.

    Raises:
        InputError: If the file cannot be read.
        HeaderError: If the header is missing or does not match §2.
        UnreadableCsvError: If the CSV quoting is malformed.
    """
    return parse_statement(read_utf8_text(path))
