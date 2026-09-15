"""Structured errors carrying the frozen §6 error ``kind`` values.

The CLI never prints a traceback: every recoverable failure is a
:class:`LedgerError` subclass whose ``kind`` is one of the frozen envelope
values. Failures that carry data — an offending amount, a mismatched header —
are their own class whose ``__init__`` owns the message text, so the ``raise``
sites stay one line and the message format lives in exactly one place.
"""

from typing import Final, Literal, TypeAlias

ErrorKind = Literal["IO", "HEADER", "ROW", "AMOUNT", "DATE", "REGISTRY"]
"""The frozen ``error.kind`` vocabulary of spec §6."""

Scalar: TypeAlias = str | int | float | bool | None
"""A value that is already lossless JSON; used by the envelope writer."""

KIND_IO: Final[ErrorKind] = "IO"
KIND_HEADER: Final[ErrorKind] = "HEADER"
KIND_ROW: Final[ErrorKind] = "ROW"
KIND_AMOUNT: Final[ErrorKind] = "AMOUNT"
KIND_DATE: Final[ErrorKind] = "DATE"
KIND_REGISTRY: Final[ErrorKind] = "REGISTRY"


class LedgerError(Exception):
    """A failure that maps onto the §6 error envelope.

    Attributes:
        kind: One of :data:`ErrorKind`; serialised verbatim as ``error.kind``.
        message: Human-readable, locale-independent explanation.
        line: 1-based physical line the failure is attributed to, if any.
    """

    kind: ErrorKind

    def __init__(self, message: str, line: int | None = None) -> None:
        """Build the error.

        Args:
            message: Explanation; must not contain locale-dependent text.
            line: Optional 1-based physical line number.
        """
        super().__init__(message)
        self.message = message
        self.line = line

    def __str__(self) -> str:
        """Explain the failure, mentioning the line when one is known."""
        if self.line is None:
            return self.message
        return f"line {self.line}: {self.message}"

    def as_error_object(self) -> dict[str, Scalar]:
        """Render the ``error`` member of the §6 envelope."""
        result: dict[str, Scalar] = {"kind": self.kind, "message": self.message}
        if self.line is not None:
            result["line"] = self.line
        return result


class InputError(LedgerError):
    """The input file could not be read at all."""

    kind = KIND_IO

    def __init__(self, name: str, reason: str, line: int | None = None) -> None:
        """Build an ``IO`` error naming the file and the underlying reason."""
        super().__init__(f"cannot read {name}: {reason}", line)


class HeaderError(LedgerError):
    """The header line is missing or does not match §2."""

    kind = KIND_HEADER


class EmptyStatementError(HeaderError):
    """The file has no lines at all, so it has no header either."""

    def __init__(self) -> None:
        """Build a ``HEADER`` error for a completely empty file."""
        super().__init__("empty file: no header line", 1)


class HeaderMismatchError(HeaderError):
    """The header line exists but is not the frozen four-column header."""

    def __init__(self, found: tuple[str, ...], line: int | None = None) -> None:
        """Build a ``HEADER`` error describing what was found instead."""
        super().__init__(f"header mismatch: found {','.join(found)}", line)


class RowError(LedgerError):
    """A data row is structurally wrong (wrong arity, unreadable field)."""

    kind = KIND_ROW

    def __init__(self, message: str, line: int | None = None) -> None:
        """Build a ``ROW`` error."""
        super().__init__(message, line)


class ShortRowError(RowError):
    """A data row has fewer physical columns than §2 requires."""

    def __init__(self, found: int, required: int, line: int | None = None) -> None:
        """Build a ``ROW`` error describing the column-count mismatch."""
        super().__init__(f"expected {required} columns, found {found}", line)


class UnreadableCsvError(RowError):
    """The CSV quoting itself is malformed."""

    def __init__(self, detail: str, line: int | None = None) -> None:
        """Build a ``ROW`` error for a CSV-level failure."""
        super().__init__(f"malformed CSV: {detail}", line)


class AmountError(LedgerError):
    """An amount or balance field cannot be interpreted unambiguously (§4.1)."""

    kind = KIND_AMOUNT

    def __init__(self, message: str, line: int | None = None) -> None:
        """Build an ``AMOUNT`` error."""
        super().__init__(message, line)


class AmbiguousAmountError(AmountError):
    """A single value could be read two ways; §4.1 says reject, never guess."""

    def __init__(
        self,
        text: str,
        *,
        both_separators: bool = False,
        thousands_group: bool = False,
        line: int | None = None,
    ) -> None:
        """Build an ``AMOUNT`` error explaining which ambiguity was hit."""
        if both_separators:
            reason = "both separators present"
        elif thousands_group:
            reason = "separator followed by three digits"
        else:
            reason = "ambiguous separator"
        super().__init__(f"ambiguous amount ({reason}): {text!r}", line)


class MalformedAmountError(AmountError):
    """A value that cannot be money at all under §4.1."""

    def __init__(self, text: str, reason: str, line: int | None = None) -> None:
        """Build an ``AMOUNT`` error pairing the offending text with the reason."""
        super().__init__(f"{reason}: {text!r}", line)


class DateError(LedgerError):
    """A date field is not a real ``DD/MM/YYYY`` calendar date (§4.2)."""

    kind = KIND_DATE

    def __init__(self, message: str, line: int | None = None) -> None:
        """Build a ``DATE`` error."""
        super().__init__(message, line)


class MalformedDateError(DateError):
    """A date field that is not a real ``DD/MM/YYYY`` calendar date."""

    def __init__(self, text: str, reason: str, line: int | None = None) -> None:
        """Build a ``DATE`` error pairing the offending text with the reason."""
        super().__init__(f"{reason}: {text!r}", line)


class RegistryError(LedgerError):
    """The accounts registry is missing, malformed or self-inconsistent (§5)."""

    kind = KIND_REGISTRY

    def __init__(self, message: str, line: int | None = None) -> None:
        """Build a ``REGISTRY`` error."""
        super().__init__(message, line)


class RegistryNotObjectError(RegistryError):
    """The registry document, or one of its members, is not a JSON object."""

    def __init__(self, what: str) -> None:
        """Build a ``REGISTRY`` error naming the offending member."""
        super().__init__(f"{what} must be a JSON object")


class RegistryNonStringKeyError(RegistryError):
    """A JSON object in the registry carries a key that is not a string."""

    def __init__(self, what: str) -> None:
        """Build a ``REGISTRY`` error naming the offending member."""
        super().__init__(f"{what} has a non-string key")


class RegistryNotArrayError(RegistryError):
    """A member that must be a JSON array is not one."""

    def __init__(self, what: str) -> None:
        """Build a ``REGISTRY`` error naming the offending member."""
        super().__init__(f"{what} must be a JSON array")


class RegistryMissingMemberError(RegistryError):
    """A required string member is absent, empty or not a string."""

    def __init__(self, what: str, key: str) -> None:
        """Build a ``REGISTRY`` error naming the member and its owner."""
        super().__init__(f"{what}.{key} must be a non-empty string")


class RegistryUnknownKeysError(RegistryError):
    """The registry or an account entry carries keys this version does not define."""

    def __init__(self, what: str, keys: tuple[str, ...]) -> None:
        """Build a ``REGISTRY`` error listing the unexpected keys."""
        super().__init__(f"{what} has unknown keys: {', '.join(keys)}")


class RegistryToleranceError(RegistryError):
    """``toleranceDays`` is present but is not a non-negative integer."""

    def __init__(self, value: object) -> None:
        """Build a ``REGISTRY`` error quoting the offending value."""
        super().__init__(
            f"toleranceDays must be a non-negative integer, found {value!r}"
        )


class RegistryVersionError(RegistryError):
    """The registry declares a schema version this build does not understand."""

    def __init__(self, version: object) -> None:
        """Build a ``REGISTRY`` error quoting the offending version."""
        super().__init__(f"unsupported registry version: {version!r}")


class UnsupportedRegistryJsonError(RegistryError):
    """The registry file is not valid JSON at all."""

    def __init__(self, detail: str) -> None:
        """Build a ``REGISTRY`` error carrying the decoder's complaint."""
        super().__init__(f"registry is not valid JSON: {detail}")


class RegistryOpeningBalanceError(RegistryError):
    """An account's ``openingBalance`` is missing, not a string, or not a decimal."""

    def __init__(self, account_id: str, value: object) -> None:
        """Build a ``REGISTRY`` error naming the account and quoting the value."""
        super().__init__(
            f"account {account_id}.openingBalance is not a decimal: {value!r}"
        )


class RegistryDuplicateAccountError(RegistryError):
    """Two registry entries share the same id, which would alias two statements."""

    def __init__(self, account_id: str) -> None:
        """Build a ``REGISTRY`` error naming the duplicated id."""
        super().__init__(f"duplicate account id: {account_id}")


class RegistryEmptyError(RegistryError):
    """The registry lists no accounts, so there is nothing to reconcile."""

    def __init__(self) -> None:
        """Build a ``REGISTRY`` error for an empty account list."""
        super().__init__("registry lists no accounts")
