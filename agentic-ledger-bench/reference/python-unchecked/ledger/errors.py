"""Frozen error envelope vocabulary (spec §6).

The `kind` values are part of the contract and must not be extended or renamed.
"""

from __future__ import annotations

#: The only legal error kinds (spec §6).
IO = "IO"
HEADER = "HEADER"
ROW = "ROW"
AMOUNT = "AMOUNT"
DATE = "DATE"
REGISTRY = "REGISTRY"

ALL_KINDS = (IO, HEADER, ROW, AMOUNT, DATE, REGISTRY)


class LedgerError(Exception):
    """A fatal input error, reported as the frozen error envelope on stdout.

    ``line`` is the 1-based physical CSV line the error is attributable to, or
    ``None`` when the error is not row-scoped (missing file, bad header,
    damaged registry).  Only the key that carries information is emitted.
    """

    __slots__ = ("kind", "message", "line")

    def __init__(self, kind: str, message: str, line: int | None = None) -> None:
        if kind not in ALL_KINDS:
            raise ValueError(f"unknown error kind: {kind!r}")
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.line = line

    def envelope_error(self) -> dict:
        """The ``error`` object, with stable key order: kind, message, [line]."""
        error = {"kind": self.kind, "message": self.message}
        if self.line is not None:
            error["line"] = self.line
        return error

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LedgerError({self.kind!r}, {self.message!r}, line={self.line!r})"
