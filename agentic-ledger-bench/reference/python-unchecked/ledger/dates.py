"""Strict ``DD/MM/YYYY`` dates (spec §2, §4.2).

``datetime.date`` already refuses ``31/02/2026`` — there is no roll-over to
suppress — so the only work here is to refuse *shapes* the constructor would
not have complained about anyway (``1/2/2026``, ``2026-02-01``, ``01/02/26``)
and non-ASCII digits that ``int()`` would happily accept.
"""

from __future__ import annotations

from datetime import date, timedelta

from .strictdecimal import digits_of

_DATE_FORMAT = "DD/MM/YYYY"
_MIN_YEAR = 1900
_MAX_YEAR = 2999


class DateError(ValueError):
    """Raised when a raw date token is not a strict, real ``DD/MM/YYYY`` date."""

    __slots__ = ("message", "token")

    def __init__(self, message: str, token: str) -> None:
        super().__init__(message)
        self.message = message
        self.token = token


def parse_date_token(token: str) -> date:
    """Parse exactly ``DD/MM/YYYY``; raise ``DateError`` otherwise."""
    raw = token.strip() if token is not None else ""
    if raw == "":
        raise DateError("date is empty", raw)
    if len(raw) != 10 or raw[2] != "/" or raw[5] != "/":
        raise DateError(f"date is not {_DATE_FORMAT}", raw)
    day_text, month_text, year_text = raw[0:2], raw[3:5], raw[6:10]
    if not (digits_of(day_text) and digits_of(month_text) and digits_of(year_text)):
        raise DateError(f"date is not {_DATE_FORMAT}", raw)
    day, month, year = int(day_text), int(month_text), int(year_text)
    if not (_MIN_YEAR <= year <= _MAX_YEAR):
        raise DateError("date year out of supported range", raw)
    try:
        # Real calendar validation: 31/02 and 30/02 raise rather than roll over.
        return date(year, month, day)
    except ValueError as exc:
        raise DateError("date is not a real calendar date", raw) from exc


def iso(value: date) -> str:
    """ISO ``YYYY-MM-DD``, via an explicit format rather than ``isoformat``."""
    return f"{value.year:04d}-{value.month:02d}-{value.day:02d}"


def days_between(earlier: date, later: date) -> int:
    """Whole days from ``earlier`` to ``later`` (negative when reversed)."""
    return (later - earlier).days


def timedelta_days(days: int) -> timedelta:
    """A day count as a ``timedelta``; date arithmetic lives here, not in callers."""
    return timedelta(days=days)


def add_days(value: date, days: int) -> date:
    """``value`` shifted by ``days`` (negative shifts backwards)."""
    return value + timedelta(days=days)
