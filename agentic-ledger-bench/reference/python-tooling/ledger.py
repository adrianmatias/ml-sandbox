#!/usr/bin/env python3
"""CLI entry point — the frozen §6 interface.

```
python/ledger.py baseline <statement.csv>   -> the §3 JSON envelope on stdout
python/ledger.py accounts <accounts.json>   -> the §5 JSON envelope on stdout
python/ledger.py --help
```

Exit codes (spec §6): ``0`` when the input is well formed even if checks report
violations, ``1`` when the input cannot be parsed, ``2`` on a usage error.
"""

import json
import sys
from pathlib import Path
from typing import Final, TextIO

from ledger.accounts import run_accounts
from ledger.baseline import run_baseline
from ledger.errors import InputError, LedgerError
from ledger.jsonshape import JSONObject, baseline_data, envelope_data, envelope_error

EXIT_OK: Final[int] = 0
EXIT_UNPARSEABLE: Final[int] = 1
EXIT_USAGE: Final[int] = 2

USAGE: Final[str] = (
    "usage: ledger.py baseline <statement.csv>\n"
    "       ledger.py accounts <accounts.json>\n"
    "       ledger.py --help\n"
)

_HELP: Final[str] = (
    USAGE
    + "\n"
    + "baseline  parse a Kutxa statement CSV and print the spec §3 JSON envelope\n"
    + "accounts  reconcile a multi-account registry (spec §5) and print its envelope\n"
    + "\n"
    + "Exit codes: 0 = well formed, 1 = input could not be parsed, 2 = usage error.\n"
)


def emit(payload: JSONObject, stream: TextIO) -> None:
    """Write one envelope as deterministic JSON.

    Key order is preserved (``sort_keys=False``) because spec §6 freezes it, and
    the text is UTF-8 with no locale-dependent formatting anywhere.

    Args:
        payload: An envelope built by :mod:`ledger.jsonshape`.
        stream: The text stream to write to.
    """
    stream.write(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    )


def _run_baseline_command(target: str) -> int:
    """Execute ``baseline <statement.csv>`` and return the process exit code."""
    run = run_baseline(Path(target))
    emit(envelope_data(baseline_data(run)), sys.stdout)
    return EXIT_OK if run.valid else EXIT_UNPARSEABLE


def _run_accounts_command(target: str) -> int:
    """Execute ``accounts <accounts.json>`` and return the process exit code."""
    run = run_accounts(Path(target))
    emit(envelope_data(run.payload()), sys.stdout)
    return EXIT_OK if run.valid else EXIT_UNPARSEABLE


def main(argv: list[str]) -> int:
    """Run the CLI.

    Args:
        argv: Arguments **without** the program name.

    Returns:
        The process exit code.
    """
    if not argv:
        sys.stderr.write(USAGE)
        return EXIT_USAGE
    if argv[0] in {"-h", "--help", "help"}:
        sys.stdout.write(_HELP)
        return EXIT_OK
    if len(argv) != 2:
        sys.stderr.write(USAGE)
        return EXIT_USAGE
    command, target = argv[0], argv[1]
    try:
        if command == "baseline":
            return _run_baseline_command(target)
        if command == "accounts":
            return _run_accounts_command(target)
    except LedgerError as error:
        emit(envelope_error(error), sys.stdout)
        return EXIT_UNPARSEABLE
    except OSError as error:
        name = Path(target).name
        emit(envelope_error(InputError(name, error.strerror or str(error))), sys.stdout)
        return EXIT_UNPARSEABLE
    sys.stderr.write(f"unknown command: {command}\n{USAGE}")
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
