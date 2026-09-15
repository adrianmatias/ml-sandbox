"""Command line interface — the frozen interface of spec §6.

    <prog> baseline <statement.csv>    -> §3 envelope on stdout
    <prog> accounts <accounts.json>    -> §5 envelope on stdout
    <prog> --help

Exit codes: ``0`` well-formed input (even when checks report violations),
``1`` input that cannot be parsed at all, ``2`` usage error.  Diagnostics (usage
text, unexpected exceptions) go to stderr so stdout carries only the envelope.
"""

from __future__ import annotations

import sys

from . import accounts as accounts_module
from . import envelope
from .checks import build_baseline
from .errors import IO, LedgerError
from .model import parse_statement

USAGE = """\
ledger - Kutxa statement checks (spec v1)

usage:
  ledger baseline <statement.csv>    parse a statement and emit the checks JSON
  ledger accounts <accounts.json>    reconcile a multi-account registry
  ledger --help                      show this message

exit codes:
  0  input well-formed (check violations are reported inside the JSON)
  1  input could not be parsed
  2  usage error
"""


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("--help", "-h", "help"):
        stream = sys.stdout if args else sys.stderr
        stream.write(USAGE)
        return envelope.EXIT_OK if args else envelope.EXIT_USAGE

    command, rest = args[0], args[1:]

    if command == "baseline":
        if len(rest) != 1:
            return _usage_error("baseline takes exactly one statement path")
        return _run(lambda: build_baseline(parse_statement(rest[0])))

    if command == "accounts":
        if len(rest) != 1:
            return _usage_error("accounts takes exactly one registry path")
        return _run(lambda: accounts_module.build_accounts(rest[0]))

    return _usage_error(f"unknown command: {command}")


def _run(produce) -> int:
    """Run a producer, print exactly one envelope, map errors to exit codes."""
    try:
        data = produce()
    except LedgerError as exc:
        sys.stdout.write(envelope.render(envelope.fail(exc)))
        return envelope.EXIT_UNPARSEABLE
    except OSError as exc:
        error = LedgerError(IO, f"i/o error: {exc.strerror or exc}")
        sys.stdout.write(envelope.render(envelope.fail(error)))
        return envelope.EXIT_UNPARSEABLE
    sys.stdout.write(envelope.render(envelope.ok(data)))
    return envelope.EXIT_OK


def _usage_error(message: str) -> int:
    sys.stderr.write(f"{message}\n\n{USAGE}")
    return envelope.EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    sys.exit(main())
