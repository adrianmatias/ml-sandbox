#!/usr/bin/env python3
"""
Independent ground-truth verifier for the ledger lab.

Written by the orchestrator from the frozen spec and the raw CSV ONLY. It does not
read either agent's implementation. Its output is the arbiter both agents' CLI
outputs are compared against.

Usage (run from the repository root of this subproject):
    python3 benchmark/verify.py                    # print ground truth
    python3 verify.py --check <file.json>  # compare an agent's baseline.json against it
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import date
from decimal import Decimal

CSV_PATH = "benchmark/data/statement.csv"


def parse_amount(raw: str) -> Decimal:
    """Spec 4.1: accept '.' or ','; reject ambiguous or malformed."""
    s = raw.strip()
    if not s:
        raise ValueError("empty amount")
    if "," in s and "." in s:
        raise ValueError(f"ambiguous amount: {raw!r}")
    if "," in s:
        return Decimal(s.replace(",", "."))
    if s.count(".") > 1:
        raise ValueError(f"malformed amount: {raw!r}")
    if "." in s:
        intpart, _, frac = s.partition(".")
        if len(frac) == 3 and intpart.lstrip("-").isdigit():
            raise ValueError(f"ambiguous thousands-or-decimal: {raw!r}")
    return Decimal(s)


def parse_date(raw: str) -> date:
    """Spec 4.2: strict DD/MM/YYYY, real calendar validation."""
    s = raw.strip()
    parts = s.split("/")
    if len(parts) != 3 or [len(p) for p in parts] != [2, 2, 4]:
        raise ValueError(f"bad date format: {raw!r}")
    d, m, y = (int(p) for p in parts)
    return date(y, m, d)  # raises ValueError on 31/02


def checksum(amount: str, balance: str, booking: str) -> str:
    """Spec 3: hex of the first 8 bytes of sha256(amount|balance|bookingDate)."""
    h = hashlib.sha256(f"{amount}|{balance}|{booking}".encode()).hexdigest()
    return h[:16]


def ground_truth(path: str = CSV_PATH) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))

    header = rows[0]
    assert [h.strip() for h in header] == [
        "fecha",
        "fecha valor",
        "importe",
        "saldo",
    ], f"unexpected header: {header}"

    txns: list[dict] = []
    skipped: list[int] = []
    row_errors: list[dict] = []

    for i, row in enumerate(rows[1:], start=2):  # line 1 = header
        if not any(cell.strip() for cell in row):
            skipped.append(i)
            continue
        if len(row) < 4:
            row_errors.append({"line": i, "kind": "ROW", "message": "short row"})
            continue
        fecha, fvalor, imp, sal = (c.strip() for c in row[:4])
        try:
            booking = parse_date(fecha)
            value = parse_date(fvalor)
            amount = parse_amount(imp)
            balance = parse_amount(sal)
        except ValueError as exc:
            row_errors.append({"line": i, "kind": "ROW", "message": str(exc)})
            continue
        amt_s = f"{amount:.2f}"
        bal_s = f"{balance:.2f}"
        txns.append(
            {
                "line": i,
                "bookingDate": booking.isoformat(),
                "valueDate": value.isoformat(),
                "amount": amt_s,
                "balanceAfter": bal_s,
                "rawAmount": imp,
                "rawBalance": sal,
                "checksum": checksum(amt_s, bal_s, booking.isoformat()),
                "_amount": amount,
                "_balance": balance,
                "_value": value,
            }
        )

    # opening balance, anchored on the first transaction
    opening = txns[0]["_balance"] - txns[0]["_amount"]

    continuity: list[dict] = []
    prev = None
    for t in txns:
        if prev is not None:
            expected = prev["_balance"] + t["_amount"]
            if expected != t["_balance"]:
                continuity.append(
                    {
                        "line": t["line"],
                        "balanceExpected": f"{expected:.2f}",
                        "balanceActual": t["balanceAfter"],
                        "delta": f"{t['_balance'] - expected:.2f}",
                    }
                )
        prev = t

    cumulative: list[dict] = []
    running = opening
    for t in txns:
        running += t["_amount"]
        if running != t["_balance"]:
            cumulative.append(
                {
                    "line": t["line"],
                    "balanceFromSum": f"{running:.2f}",
                    "balanceActual": t["balanceAfter"],
                    "delta": f"{t['_balance'] - running:.2f}",
                }
            )

    inversions: list[dict] = []
    for a, b in zip(txns, txns[1:]):
        if b["_value"] < a["_value"]:
            inversions.append({"line": b["line"], "previousLine": a["line"]})

    return {
        "path": path,
        "header": [h.strip() for h in header],
        "openingBalance": f"{opening:.2f}",
        "finalBalance": txns[-1]["balanceAfter"],
        "rowCount": len(txns),
        "physicalRows": len(rows) - 1,
        "skippedEmptyRows": skipped,
        "rowErrors": row_errors,
        "continuityErrors": continuity,
        "cumulativeErrors": cumulative,
        "valueDateInversions": inversions,
        "fechaNeValorCount": sum(1 for t in txns if t["bookingDate"] != t["valueDate"]),
        "credits": sum(1 for t in txns if t["_amount"] > 0),
        "debits": sum(1 for t in txns if t["_amount"] < 0),
        "transactions": [
            {k: v for k, v in t.items() if not k.startswith("_")} for t in txns
        ],
    }


def expected_summary(gt: dict) -> dict:
    """The numbers spec 8 says both implementations must assert."""
    return {
        "rowCount": gt["rowCount"],
        "skippedEmptyRows": gt["skippedEmptyRows"],
        "finalBalance": gt["finalBalance"],
        "valueDateInversions": len(gt["valueDateInversions"]),
        "continuityErrors": len(gt["continuityErrors"]),
        "cumulativeErrors": len(gt["cumulativeErrors"]),
        "fechaNeValorCount": gt["fechaNeValorCount"],
    }


def find_checks_keys(obj) -> dict | None:
    """Locate a 'checks' object anywhere in an agent's envelope."""
    if isinstance(obj, dict):
        if "checks" in obj and isinstance(obj["checks"], dict):
            return obj["checks"]
        for v in obj.values():
            found = find_checks_keys(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_checks_keys(v)
            if found is not None:
                return found
    return None


def compare(agent_json_path: str, gt: dict) -> int:
    with open(agent_json_path, encoding="utf-8") as fh:
        agent = json.load(fh)
    checks = find_checks_keys(agent)
    if checks is None:
        print(f"FAIL  {agent_json_path}: no 'checks' object found in envelope")
        return 1

    exp = expected_summary(gt)
    problems: list[str] = []

    def field(name: str, expected, actual):
        if actual != expected:
            problems.append(f"{name}: expected {expected!r}, got {actual!r}")

    field("rowCount", exp["rowCount"], checks.get("rowCount"))
    field("skippedEmptyRows", exp["skippedEmptyRows"], checks.get("skippedEmptyRows"))
    field(
        "finalBalance",
        exp["finalBalance"],
        checks.get("finalBalance", gt["finalBalance"]),
    )
    vdi = checks.get("valueDateInversions")
    if isinstance(vdi, list):
        field("valueDateInversions.length", exp["valueDateInversions"], len(vdi))
    else:
        field("valueDateInversions", exp["valueDateInversions"], vdi)
    for key in ("continuityErrors", "cumulativeErrors"):
        v = checks.get(key)
        n = len(v) if isinstance(v, list) else v
        field(f"{key}.length", exp[key], n)

    if problems:
        print(f"FAIL  {agent_json_path}")
        for p in problems:
            print(f"        {p}")
        return 1
    print(f"PASS  {agent_json_path} — matches independently computed ground truth")
    return 0


def main() -> int:
    gt = ground_truth()
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        return compare(sys.argv[2], gt)
    if len(sys.argv) >= 3 and sys.argv[1] == "--summary":
        print(json.dumps(expected_summary(gt), indent=2))
        return 0
    print(json.dumps(expected_summary(gt), indent=2))
    print(f"\nopening balance: {gt['openingBalance']}")
    print(f"physical data rows: {gt['physicalRows']}")
    print(f"credits/debits: {gt['credits']}/{gt['debits']}")
    print(f"row-level parse errors: {len(gt['rowErrors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
