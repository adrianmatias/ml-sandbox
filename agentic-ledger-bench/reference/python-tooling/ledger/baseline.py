"""Baseline entry point — spec §3 via the §6 CLI.

:func:`run_baseline` is the whole of ``ledger baseline <statement.csv>``: read,
parse, run the three checks, hand back the result. Nothing here formats money by
hand; every amount goes through :func:`ledger.domain.format_money` so both
commands serialise identically.
"""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from ledger.checks import CheckReport, run_checks
from ledger.domain import Balance, ParsedStatement, Transaction
from ledger.statement import parse_statement_file


@dataclass(frozen=True, slots=True)
class BaselineRun:
    """The complete §3 result for one statement file.

    Attributes:
        source: Basename of the input, never an absolute path (spec §6 determinism).
        statement: The parsed rows, skipped lines and tolerated row failures.
        checks: The three checks plus the parse-error list.
    """

    source: str
    statement: ParsedStatement
    checks: CheckReport

    @property
    def transactions(self) -> tuple[Transaction, ...]:
        """Parsed rows, in file order."""
        return self.statement.transactions

    @property
    def final_balance(self) -> Balance:
        """Balance after the last row; zero when the statement has no rows."""
        if not self.statement.transactions:
            return Balance(Decimal(0))
        return self.statement.transactions[-1].balance_after

    @property
    def valid(self) -> bool:
        """False when any row failed to parse, which makes the CLI exit code 1."""
        return not self.statement.row_errors


def run_baseline(path: Path) -> BaselineRun:
    """Parse a statement file and run the §3 checks over it.

    Args:
        path: The statement CSV to read.

    Returns:
        The complete run, including any tolerated per-row failures.

    Raises:
        InputError: If the file cannot be read.
        HeaderError: If the header is missing or does not match §2.
        RowError: If the CSV quoting is malformed.
    """
    statement = parse_statement_file(path)
    checks = run_checks(
        statement.transactions,
        skipped_empty_rows=statement.skipped_empty_rows,
        parse_errors=statement.row_errors,
    )
    return BaselineRun(source=path.name, statement=statement, checks=checks)
