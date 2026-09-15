"""Canonical JSON rendering of the two result shapes (spec §3 and §5).

Every writer here builds dicts with **literal key order**, because spec §6 freezes
the key order and forbids hash-order iteration. No timestamps, no absolute paths,
no locale-dependent formatting: two runs on the same input are byte-identical.
"""

from typing import TypeAlias

from ledger.baseline import BaselineRun
from ledger.checks import CheckReport, OpeningBalance
from ledger.domain import Transaction, format_money
from ledger.errors import LedgerError, Scalar

JSONScalar: TypeAlias = Scalar
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]
"""The only shapes allowed to cross into :mod:`json`."""

MODULE_NAME = "ledger"
MODULE_VERSION = 1


def envelope_data(data: JSONObject) -> JSONObject:
    """Wrap a payload in the frozen §6 success envelope."""
    return {"ok": True, "module": MODULE_NAME, "version": MODULE_VERSION, "data": data}


def envelope_error(error: LedgerError) -> JSONObject:
    """Wrap a failure in the frozen §6 error envelope."""
    return {
        "ok": False,
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "error": error.as_error_object(),
    }


def transaction_object(transaction: Transaction) -> JSONObject:
    """Serialise one §3 transaction entry."""
    return {
        "line": transaction.line,
        "bookingDate": transaction.booking_date.isoformat(),
        "valueDate": transaction.value_date.isoformat(),
        "amount": format_money(transaction.amount),
        "balanceAfter": format_money(transaction.balance_after),
        "rawAmount": transaction.raw_amount,
        "rawBalance": transaction.raw_balance,
        "checksum": transaction.checksum,
    }


def check_object(checks: CheckReport) -> JSONObject:
    """Serialise the §3 ``checks`` object, including the parser's own findings."""
    return {
        "rowCount": checks.row_count,
        "skippedEmptyRows": [int(line) for line in checks.skipped_empty_rows],
        "reconciled": checks.reconciled,
        "continuityErrors": [
            {
                "line": error.line,
                "balanceExpected": format_money(error.balance_expected),
                "balanceActual": format_money(error.balance_actual),
                "delta": format_money(error.delta),
            }
            for error in checks.continuity_errors
        ],
        "cumulativeErrors": [
            {
                "line": error.line,
                "balanceFromSum": format_money(error.balance_from_sum),
                "balanceActual": format_money(error.balance_actual),
                "delta": format_money(error.delta),
            }
            for error in checks.cumulative_errors
        ],
        "valueDateInversions": [
            {"line": inversion.line, "previousLine": inversion.previous_line}
            for inversion in checks.value_date_inversions
        ],
        "parseErrors": [
            {"line": failure.line, "kind": failure.kind, "message": failure.message}
            for failure in checks.parse_errors
        ],
    }


def opening_object(opening: OpeningBalance) -> JSONObject:
    """Serialise the provenance of the cumulative check's anchor."""
    result: JSONObject = {
        "amount": format_money(opening.amount),
        "source": opening.source,
    }
    if opening.declared_mismatch is not None:
        declared, derived = opening.declared_mismatch
        result["declaredMismatch"] = {
            "declared": format_money(declared),
            "derived": format_money(derived),
        }
    return result


def baseline_data(run: BaselineRun) -> JSONObject:
    """Serialise a complete §3 run."""
    checks = check_object(run.checks)
    checks["finalBalance"] = format_money(run.final_balance)
    return {
        "source": run.source,
        "transactions": [transaction_object(t) for t in run.transactions],
        "checks": checks,
    }
