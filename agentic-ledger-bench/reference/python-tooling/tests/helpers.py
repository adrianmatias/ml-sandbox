"""Shared test helpers: paths, subprocess CLI invocation and temporary files."""

import json
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeVar

PYTHON_DIR: Final[Path] = Path(__file__).resolve().parent.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from . import make_fixtures  # noqa: E402 - deliberate: after the sys.path setup above

V = TypeVar("V")


class NonPositiveBoundError(ValueError):
    """``SequenceSource.below`` was asked for a range that cannot exist."""

    def __init__(self, bound: int) -> None:
        """Name the offending bound."""
        super().__init__(f"bound must be positive, found {bound}")


class SequenceSource:
    """A tiny deterministic integer source for the property tests.

    The property tests need a *reproducible* stream of numbers — a failing round
    must be replayable from its seed — so they use this self-contained generator
    instead of the ``random`` module, whose stateful global API is the thing the
    ``S311`` lint rule exists to keep out of a codebase.

    Attributes:
        state: The current generator state; advancing it is the whole algorithm.
    """

    MULTIPLIER: Final[int] = 6364136223846793005
    INCREMENT: Final[int] = 1442695040888963407
    MODULUS: Final[int] = 1 << 64

    def __init__(self, seed: int) -> None:
        """Seed the generator."""
        self.state: int = seed % self.MODULUS

    def next_int(self) -> int:
        """Return the next value in the stream, in ``[0, 2**63)``."""
        self.state = (self.MULTIPLIER * self.state + self.INCREMENT) % self.MODULUS
        return self.state >> 1

    def below(self, bound: int) -> int:
        """Return a value in ``[0, bound)``."""
        if bound <= 0:
            raise NonPositiveBoundError(bound)
        return self.next_int() % bound

    def between(self, low: int, high: int) -> int:
        """Return a value in ``[low, high)``."""
        return low + self.below(high - low)

    def sign(self) -> int:
        """Return ``-1`` or ``+1``."""
        return -1 if self.next_int() % 2 else 1


# PYTHON_DIR is this arm's directory; the benchmark root is two levels up
# (``reference/python-tooling`` -> ``reference`` -> repo root).
REPO_ROOT: Final[Path] = PYTHON_DIR.parent.parent
CLI: Final[Path] = PYTHON_DIR / "ledger.py"
FIXTURES: Final[Path] = PYTHON_DIR / "tests" / "fixtures"
RAW_CSV: Final[Path] = REPO_ROOT / "benchmark" / "data" / "statement.csv"
SHARED_FIXTURES: Final[Path] = REPO_ROOT / "benchmark" / "data"

# ---------------------------------------------------------------------------
# The assertions below pin facts about the ORIGINAL private bank export the
# experiment first ran on: exact collision line numbers, the exact count of rows
# whose booking and value dates differ, and the literal last-row spelling.
# The published benchmark data is synthetic (see benchmark/make_statement.py)
# and reproduces the *structural* invariants — 923 rows, 37713.30, 29
# inversions, 3 collision dates — but not those incidental values.
#
# Rewriting them would falsify the evidence, so they are gated instead. Set
# ORIGINAL_EXPORT = True when running against the real export.
# ---------------------------------------------------------------------------
ORIGINAL_EXPORT = False


EXPECTED_ROW_COUNT: Final[int] = 923
EXPECTED_SKIPPED_EMPTY_ROWS: Final[list[int]] = [2]
EXPECTED_FINAL_BALANCE: Final[str] = "37713.30"
EXPECTED_VALUE_DATE_INVERSIONS: Final[int] = 29
EXPECTED_SHARED_NET_WORTH: Final[str] = "1139.25"
EXPECTED_SHARED_PAIRS: Final[int] = 2


@dataclass(frozen=True, slots=True)
class CliResult:
    """One CLI invocation's outcome.

    Attributes:
        argv: The arguments that were run.
        returncode: The process exit code.
        stdout: Raw stdout text.
        stderr: Raw stderr text.
    """

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def json(self) -> dict[str, object]:
        """Parse stdout as the §6 envelope."""
        return as_dict(json.loads(self.stdout), "stdout")


class JsonShapeError(TypeError, AssertionError):
    """A JSON member did not have the shape the test requires.

    Deriving from both ``TypeError`` (the classic type-guard failure) and
    ``AssertionError`` keeps it an honest test failure and green under the
    ``TRY004`` lint rule, which asks for ``TypeError`` on a bad type.
    """


class NotAnObjectError(JsonShapeError):
    """A value that had to be a JSON object was not one."""

    def __init__(self, what: str) -> None:
        """Name the offending member."""
        super().__init__(f"{what} is not a JSON object")


class NotAnArrayError(JsonShapeError):
    """A value that had to be a JSON array was not one."""

    def __init__(self, what: str) -> None:
        """Name the offending member."""
        super().__init__(f"{what} is not a JSON array")


class NotAStringError(JsonShapeError):
    """A value that had to be a string was not one."""

    def __init__(self, what: str) -> None:
        """Name the offending member."""
        super().__init__(f"{what} is not a string")


class NotAnIntError(JsonShapeError):
    """A value that had to be an integer was not one."""

    def __init__(self, what: str) -> None:
        """Name the offending member."""
        super().__init__(f"{what} is not an integer")


class NotASuccessEnvelopeError(JsonShapeError):
    """The CLI returned an envelope whose ``ok`` member is not ``True``."""

    def __init__(self, ok: object) -> None:
        """Quote the offending ``ok`` value."""
        super().__init__(f"envelope is not a success, ok={ok!r}")

    def __str__(self) -> str:
        """Render the message without the exception-class prefix."""
        return self.args[0] if self.args else ""


class MissingMemberError(JsonShapeError):
    """A required JSON member was absent."""

    def __init__(self, key: str) -> None:
        """Name the missing member."""
        super().__init__(f"missing member: {key}")


def as_dict(value: object, what: str) -> dict[str, object]:
    """Narrow a JSON value to an object, failing the test with context if it is not."""
    if not isinstance(value, dict):
        raise NotAnObjectError(what)
    return {str(key): item for key, item in value.items()}


def as_list(value: object, what: str) -> list[object]:
    """Narrow a JSON value to an array."""
    if not isinstance(value, list):
        raise NotAnArrayError(what)
    return list(value)


def as_str(value: object, what: str) -> str:
    """Narrow a JSON value to a string."""
    if not isinstance(value, str):
        raise NotAStringError(what)
    return value


def as_int(value: object, what: str) -> int:
    """Narrow a JSON value to an integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise NotAnIntError(what)
    return value


def member(mapping: Mapping[str, V], key: str) -> V:
    """Read a required member of a JSON object."""
    try:
        return mapping[key]
    except KeyError as exc:
        raise MissingMemberError(key) from exc


def run_cli(*args: str) -> CliResult:
    """Run the CLI in a subprocess, exactly as the spec's users would."""
    completed = subprocess.run(  # noqa: S603 - fixed argv, sys.executable + our own CLI
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(PYTHON_DIR),
    )
    return CliResult(
        argv=tuple(args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def fixture(name: str) -> Path:
    """Path of a generated §4 fixture, regenerating the set if it is missing."""
    path = FIXTURES / name
    if not path.exists():
        make_fixtures.write_fixtures(FIXTURES)
    return path


@contextmanager
def scratch_dir(name: str) -> Iterator[Path]:
    """A throwaway directory under ``python/tests/.tmp`` for tests that write files."""
    root = PYTHON_DIR / "tests" / ".tmp"
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    try:
        yield target
    finally:
        shutil.rmtree(target, ignore_errors=True)


def data_of(envelope: dict[str, object]) -> dict[str, object]:
    """Return the ``data`` member of a success envelope, checking its shape."""
    if envelope.get("ok") is not True:
        raise NotASuccessEnvelopeError(member(envelope, "ok"))
    return as_dict(member(envelope, "data"), "data")


def checks_of(envelope: dict[str, object]) -> dict[str, object]:
    """Return the baseline ``checks`` object, checking its shape."""
    return as_dict(member(data_of(envelope), "checks"), "checks")


def count_of(value: object, what: str) -> int:
    """Length of a JSON array member, checking that it really is an array."""
    return len(as_list(value, what))
