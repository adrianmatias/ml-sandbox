"""Date parsing — spec §4.2.

Strict ``DD/MM/YYYY`` with real calendar validation. ``datetime.date`` already
refuses ``31/02/2026``; the work done here is refusing everything *lenient*
around it (``1/2/2026``, ``31/02/26``, ``2026-02-31``, ``13/13/2026``).
"""

from datetime import date

from ledger.errors import MalformedDateError

_DAY_MONTH_DIGITS = 2
_YEAR_DIGITS = 4


def parse_date(text: str, line: int | None = None) -> date:
    """Parse one ``fecha``/``fecha valor`` field into a real calendar date.

    Args:
        text: The field as it appeared in the file; surrounding whitespace is
            insignificant.
        line: 1-based physical line, so a rejection can be attributed to a row.

    Returns:
        The calendar date.

    Raises:
        DateError: If the field is not a real ``DD/MM/YYYY`` date.
    """
    stripped = text.strip()
    parts = stripped.split("/")
    if len(parts) != 3 or any(
        not part.isascii() or not part.isdigit() for part in parts
    ):
        raise MalformedDateError(text, "not a DD/MM/YYYY date", line)
    day_text, month_text, year_text = parts
    if len(day_text) != _DAY_MONTH_DIGITS or len(month_text) != _DAY_MONTH_DIGITS:
        raise MalformedDateError(text, "day and month must be two digits", line)
    if len(year_text) != _YEAR_DIGITS:
        raise MalformedDateError(text, "year must be four digits", line)
    try:
        return date(int(year_text), int(month_text), int(day_text))
    except ValueError as exc:
        raise MalformedDateError(text, "not a real calendar date", line) from exc
