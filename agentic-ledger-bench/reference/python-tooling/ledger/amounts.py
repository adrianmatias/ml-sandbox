"""Amount and balance parsing — spec §4.1.

The rules, restated because they are the whole point of this module:

* ``.`` or ``,`` may be the decimal separator;
* an amount containing **both** separators is rejected as ambiguous;
* a separator followed by exactly three digits, with no other separator, is
  rejected as ambiguous (``1.234`` may be ``1234`` or ``1.234``);
* empty, non-numeric and multi-sign values are rejected;
* the result is an exact :class:`~decimal.Decimal` built from a string, so no
  binary float can enter the domain model.

Rejections raise :class:`~ledger.errors.AmbiguousAmountError` or
:class:`~ledger.errors.MalformedAmountError` directly rather than an internal
error that has to be re-wrapped by every caller; one rejection, one class.
"""

from decimal import Decimal, InvalidOperation

from ledger.domain import Amount, Balance
from ledger.errors import AmbiguousAmountError, MalformedAmountError

_SIGNS = frozenset("+-")


def _reject_multiple_signs(text: str, line: int | None) -> None:
    """Reject ``--5``, ``+-5`` and every other multi-sign value."""
    body = text[1:] if text[0] in _SIGNS else text
    if any(ch in _SIGNS for ch in body):
        raise MalformedAmountError(text, "multi-sign amount", line)


def _reject_non_numeric(text: str, line: int | None) -> None:
    """Reject anything that is not digits, at most one sign and at most two separators."""
    body = text[1:] if text[0] in _SIGNS else text
    if body == "" or any(
        not (ch.isascii() and (ch.isdigit() or ch in ".,")) for ch in body
    ):
        raise MalformedAmountError(text, "non-numeric amount", line)


def _split_grouped(text: str, separator: str, other: str, line: int | None) -> str:
    """Validate a locale-grouped amount such as ``1.234.567,89``.

    In a grouped amount every ``separator`` must introduce a full thousands
    group of exactly three digits, and the *other* separator — when present —
    must appear at most once as the decimal separator.
    """
    if text.count(other) > 1:
        raise AmbiguousAmountError(text, line=line)
    integer_part = text
    fraction = ""
    if other in text:
        integer_part, _, fraction = text.partition(other)
    groups = integer_part.split(separator)
    if len(groups) < 2 or not groups[0] or groups[0] == "-":
        raise MalformedAmountError(text, "malformed amount", line)
    if any(len(group) != 3 or not group.isdigit() for group in groups[1:]):
        raise MalformedAmountError(text, "malformed thousands groups", line)
    if fraction and not fraction.isdigit():
        raise MalformedAmountError(text, "malformed amount", line)
    return f"{''.join(groups)}.{fraction}" if fraction else "".join(groups)


def _validate_single_separator(text: str, separator: str, line: int | None) -> None:
    """Reject the ambiguous bare ``1.234`` shape for a single-separator amount.

    A separator followed by exactly three digits and no other separator could be
    a thousands separator or a decimal separator; spec §4.1 says reject rather
    than guess.
    """
    if text.count(separator) != 1:
        raise AmbiguousAmountError(text, line=line)
    integer_part, _, fraction = text.partition(separator)
    if fraction == "" or not fraction.isdigit():
        raise MalformedAmountError(text, "malformed amount", line)
    if len(fraction) == 3:
        raise AmbiguousAmountError(text, thousands_group=True, line=line)
    if integer_part in ("", "-", "+") or not integer_part.lstrip("+-").isdigit():
        raise MalformedAmountError(text, "malformed amount", line)


def _normalise(text: str, line: int | None) -> str:
    """Turn an accepted textual amount into a plain ``-1234.56`` decimal string."""
    if "," in text and "." in text:
        # Both separators are present, so the meaning is unambiguous: the
        # rightmost one is the decimal separator and the other one groups digits.
        decimal_separator = "," if text.rfind(",") > text.rfind(".") else "."
        grouping_separator = "." if decimal_separator == "," else ","
        return _split_grouped(text, grouping_separator, decimal_separator, line)
    if "," in text:
        _validate_single_separator(text, ",", line)
        return text.replace(",", ".")
    if "." in text:
        _validate_single_separator(text, ".", line)
    return text


def parse_decimal(text: str, line: int | None = None) -> Decimal:
    """Parse one amount/balance field into an exact decimal.

    Args:
        text: The field as it appeared in the file; surrounding whitespace is
            insignificant, the returned value never is.
        line: 1-based physical line, so a rejection can be attributed to a row.

    Returns:
        The exact decimal value.

    Raises:
        AmountError: If the field is empty, ambiguous or malformed under §4.1.
    """
    stripped = text.strip()
    if not stripped:
        raise MalformedAmountError(text, "empty amount", line)
    _reject_multiple_signs(stripped, line)
    _reject_non_numeric(stripped, line)
    normalised = _normalise(stripped, line)
    try:
        return Decimal(normalised)
    except InvalidOperation as exc:
        raise MalformedAmountError(text, "unparsable decimal", line) from exc


def parse_amount(text: str, line: int | None = None) -> Amount:
    """Parse an ``importe`` field (signed: positive credit, negative debit)."""
    return Amount(parse_decimal(text, line))


def parse_balance(text: str, line: int | None = None) -> Balance:
    """Parse a ``saldo`` field (balance after the row)."""
    return Balance(parse_decimal(text, line))
