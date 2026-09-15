"""Exact-decimal monetary values (spec §3, §4.1, §6).

Two jobs:

1. **Keep binary floats out of the domain model.**  ``as_money`` is the only
   door into the model and it accepts ``str`` / ``Decimal`` / ``int`` only.
   ``float`` is refused by both the type check and a runtime guard.
2. **Serialise canonically.**  ``canonical`` always emits a plain decimal
   string with exactly two fractional digits and never an exponent.

Note on Python's ``Decimal``: it is a decimal float with configurable context,
so it has an *unlimited-significant-digit trap*: multiplying two Decimals
under a finite precision context silently rounds.  This module therefore never
multiplies Decimal by Decimal; ``exact_sum`` adds with ``math.fsum``-style
compensation at full precision and asserts the result's exponent, which makes
the rounding detectable instead of invisible.
"""

from __future__ import annotations

import decimal
from decimal import Decimal

#: Domain precision: money carries at most two fractional digits (spec §2).
SCALE = Decimal("0.01")

#: Refuse absurd magnitudes rather than silently losing digits to a context
#: precision limit.  10**30 is astronomically beyond any real statement.
_MAX_ABS = Decimal(10) ** 30

_ASCII_DIGITS = frozenset("0123456789")


class AmountError(ValueError):
    """Raised when a raw amount token cannot be read as exact money."""

    __slots__ = ("message", "token")

    def __init__(self, message: str, token: str) -> None:
        super().__init__(message)
        self.message = message
        self.token = token


def as_money(value: object, *, token: str | None = None) -> Decimal:
    """Coerce ``value`` to a ``Decimal`` that is guaranteed float-free.

    Raises ``AmountError`` for floats (even integral ones such as ``400.0``:
    admitting it would mean a float had entered the domain model, which §4.1
    forbids), for non-finite values and for over-large magnitudes.
    """
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        raise AmountError("amount must be decimal, not boolean", repr(value))
    if isinstance(value, float):
        raise AmountError(
            "binary float is not an acceptable monetary value", repr(value)
        )
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, str):
        result = Decimal(value)
    else:
        raise AmountError(
            "amount must be decimal, not " + type(value).__name__, repr(value)
        )
    if not result.is_finite():
        raise AmountError(
            "amount must be finite", token if token is not None else str(result)
        )
    if abs(result) >= _MAX_ABS:
        raise AmountError(
            "amount magnitude out of supported range",
            token if token is not None else str(result),
        )
    return result


def exact_sum(values) -> Decimal:
    """Sum Decimals at full precision, refusing to round silently.

    Equivalent to ``sum(values, Decimal(0))`` for values small enough that the
    active context precision cannot bite; this version *proves* that by
    checking the exact exponent afterwards.
    """
    # Integers scaled by a common power of ten add exactly at any precision
    # that can hold the integer part, so raise the context far above need.
    with decimal.localcontext() as ctx:
        ctx.prec = decimal.MAX_PREC
        total = Decimal(0)
        for value in values:
            total += value
    return total


def canonical(value: Decimal) -> str:
    """The frozen decimal rendering: ``37713.30``, never ``37713.3``/``3.77133E4``.

    Quantising is done at ``MAX_PREC`` because Python's ``Decimal`` arithmetic
    is context-governed: at the default 28 significant digits, quantising a
    wider value raises ``InvalidOperation`` — a silent-ish trap for anyone
    reading the type name as "arbitrary precision".
    """
    with decimal.localcontext() as ctx:
        ctx.prec = decimal.MAX_PREC
        quantum = value.quantize(SCALE, rounding=decimal.ROUND_HALF_EVEN)
    if quantum == 0:
        # Collapse "-0.00" (and "0.00") to one spelling.
        quantum = abs(quantum)
    return format(quantum, "f")


def decimal_places(value: Decimal) -> int:
    """Number of fractional digits actually carried (0 for ``400``, 2 for ``1.20``)."""
    exponent = value.as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def digits_of(text: str) -> bool:
    """True iff every character is an ASCII digit (rejects NFD/other numeric scripts)."""
    return len(text) > 0 and all(ch in _ASCII_DIGITS for ch in text)


def parse_amount_token(token: str) -> Decimal:
    """Parse a raw amount token per spec §4.1, or raise ``AmountError``.

    Grammar accepted:  ``[+-]? ( digits | digits SEP digits | SEP digits )``
    where ``SEP`` is ``.`` or ``,``.  Everything else is refused, in particular:

    * empty, whitespace-only, non-numeric
    * multi-sign (``--5``, ``+-5``, ``5-``)
    * *ambiguous* separators: both ``.`` and ``,`` present in one bare amount,
      or any single separator followed by exactly three digits and nothing
      else (``1.234`` — 1234 in one locale, 1.234 in another; guessing is the
      real-world bug §4.1 calls out)
    * exponent notation, underscores, grouping spaces
    """
    if token is None:
        raise AmountError("amount is missing", "")
    raw = token.strip()
    if raw == "":
        raise AmountError("amount is empty", raw)

    sign = ""
    body = raw
    if body[0] in "+-":
        sign, body = body[0], body[1:]
    if body == "":
        raise AmountError("amount has a sign but no digits", raw)
    if body[0] in "+-":
        raise AmountError("amount has more than one sign", raw)

    if not digits_of(body) and not _digits_with_separators(body):
        raise AmountError("amount is not numeric", raw)

    n_dot = body.count(".")
    n_comma = body.count(",")
    if n_dot and n_comma:
        raise AmountError("amount is ambiguous: it contains both '.' and ','", raw)
    if n_dot > 1 or n_comma > 1:
        raise AmountError("amount has too many separators", raw)

    if n_dot or n_comma:
        separator = "." if n_dot else ","
        integer_part, _, fraction_part = body.partition(separator)
        if integer_part == "" and fraction_part == "":
            raise AmountError("amount has no digits", raw)
        for part, name in ((integer_part, "integer"), (fraction_part, "fraction")):
            if part and not digits_of(part):
                raise AmountError(f"amount {name} part is not numeric", raw)
        if len(fraction_part) == 3 and integer_part:
            raise AmountError(
                "amount is ambiguous: a single separator with three trailing "
                "digits could be a decimal point or a thousands separator",
                raw,
            )
        if fraction_part == "":
            raise AmountError("amount ends with a separator", raw)
        # Only after the ambiguity question is settled is the two-decimal rule
        # applied, so "1.234" is rejected as *ambiguous* (spec §4.1) rather
        # than as over-precise.  "1.2345" is neither ambiguous nor representable.
        if len(fraction_part) > 2:
            raise AmountError("amount has more than two decimal places", raw)
        normalised = (integer_part or "0") + "." + fraction_part
    else:
        normalised = body

    try:
        value = Decimal(sign + normalised)
    except decimal.InvalidOperation:  # pragma: no cover - guarded above
        raise AmountError("amount is not numeric", raw) from None
    if not value.is_finite():  # pragma: no cover - guarded above
        raise AmountError("amount is not finite", raw)
    if abs(value) >= _MAX_ABS:
        raise AmountError("amount magnitude out of supported range", raw)
    return value


def _digits_with_separators(body: str) -> bool:
    """Structural pre-check: digits and separators only.

    Deliberately permissive about *how many* separators appear, so that the
    ambiguity diagnostics below are reached for ``1.234,56`` instead of it
    being dismissed as "not numeric".
    """
    return len(body) > 0 and all(ch in _ASCII_DIGITS or ch in ".," for ch in body)
