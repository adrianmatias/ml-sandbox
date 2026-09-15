"""Iteration A — multi-account reconciliation and transfer detection (spec §5).

Extension points relative to the baseline (see NOTES.md for the added/modified
split): nothing in :mod:`ledger.model`, :mod:`ledger.checks`,
:mod:`ledger.strictdecimal` or :mod:`ledger.dates` changes.  A registry is
parsed here, one statement per account goes through the baseline parser, the
baseline checks are applied per account, and a transfer-pairing layer plus the
two new §5 invariants are added on top.

Decisions taken where §5 leaves room (also recorded in NOTES.md, "spec
friction"):

* ``openingBalance`` comes from the registry, so per-account §3.1/§3.2 anchor on
  it instead of re-deriving it from the first row.  On the reference file the
  two are the same number (0), so the baseline reading is unchanged.
* Pairing compares ``valueDate``, as §5 spells out, even though the booking date
  is the frame used by §3.3.
* "Prefer the nearest feasible date, break ties by the lowest line number" is
  implemented as: enumerate every feasible pair, order by (gap, outflow row,
  inflow row), then accept greedily.  This is the rule's global reading and it
  makes the result independent of registry order.  (A per-outflow-row greedy
  scan would give a different, order-sensitive answer on ties; that reading is
  *not* implemented and is noted as friction.)
* §5.5's "observation point" is taken to be each account row, visited in
  registry order then file order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal

from .checks import build_checks
from .dates import days_between
from .errors import IO, REGISTRY, LedgerError
from .model import ParsedStatement, Transaction, parse_statement
from .strictdecimal import AmountError, canonical, exact_sum, parse_amount_token

#: Default date tolerance for a transfer pair (spec §5.2).
DEFAULT_TOLERANCE_DAYS = 3
_MAX_TOLERANCE_DAYS = 3650

#: Spec §5's "design pressure" metric.  Two numbers are reported in the JSON
#: under ``data.internals`` so the claims in NOTES.md are reproducible from the
#: artifact rather than asserted by hand:
#:
#: ``amountOrDateRederivationSites``
#:     Distinct places in this module where an amount or a date is *re-derived*
#:     from a raw field instead of carried as a typed value.  This is the number
#:     the spec's "if you find yourself adding a sixth free function that
#:     re-derives an amount or a date" is really asking about, and it is
#:     maintained by hand next to the code it describes.
#: ``amountOrDateRederivations``
#:     How many times those sites actually executed in the run.  A much weaker
#:     signal — it is inflated by the pairing loop's O(outflows × inflows)
#:     comparisons — and is reported only so the difference is visible.
_REDERIVATION_SITES = 1
_DERIVATIONS = 0


def _derive() -> None:
    global _DERIVATIONS
    _DERIVATIONS += 1


def reset_derivation_count() -> None:
    global _DERIVATIONS
    _DERIVATIONS = 0


def derivation_count() -> int:
    """How many times a re-derivation site executed in the last run."""
    return _DERIVATIONS


def rederivation_site_count() -> int:
    """How many distinct re-derivation sites this module contains."""
    return _REDERIVATION_SITES


@dataclass(frozen=True, slots=True)
class Account:
    """One registry entry plus its parsed statement."""

    index: int
    account_id: str
    currency: str
    opening_balance: Decimal
    statement_label: str
    statement: ParsedStatement

    @property
    def rows(self) -> tuple[Transaction, ...]:
        return self.statement.transactions


@dataclass(frozen=True, slots=True)
class Movement:
    """A row considered as one side of a possible transfer."""

    account: Account
    transaction: Transaction

    @property
    def magnitude(self) -> Decimal:
        """Absolute amount.  One of the counted re-derivations (§5's pressure)."""
        _derive()
        return (
            -self.transaction.amount
            if self.transaction.amount < 0
            else self.transaction.amount
        )


@dataclass(frozen=True, slots=True)
class Pairing:
    """One accepted transfer: an outflow row and the inflow row that settles it."""

    outflow: Movement
    inflow: Movement
    gap_days: int

    def as_json(self) -> dict:
        out_tx, in_tx = self.outflow.transaction, self.inflow.transaction
        return {
            "amount": canonical(self.outflow.magnitude),
            "currency": self.outflow.account.currency,
            "from": {
                "account": self.outflow.account.account_id,
                "line": out_tx.line,
                "valueDate": out_tx.value_date.isoformat(),
            },
            "to": {
                "account": self.inflow.account.account_id,
                "line": in_tx.line,
                "valueDate": in_tx.value_date.isoformat(),
            },
            "gapDays": self.gap_days,
        }


# --------------------------------------------------------------------------- #
# Registry (spec §5 "Inputs")
# --------------------------------------------------------------------------- #


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            text = handle.read()
    except FileNotFoundError as exc:
        raise LedgerError(IO, f"registry file not found: {path}") from exc
    except IsADirectoryError as exc:
        raise LedgerError(IO, f"registry path is a directory: {path}") from exc
    except UnicodeDecodeError as exc:
        raise LedgerError(REGISTRY, f"registry is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise LedgerError(
            IO, f"cannot read registry file: {path}: {exc.strerror}"
        ) from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LedgerError(
            REGISTRY, f"registry is not valid JSON: {exc.msg} at offset {exc.pos}"
        ) from exc
    if not isinstance(document, dict):
        raise LedgerError(REGISTRY, "registry must be a JSON object")
    return document


def _tolerance_days(document: dict) -> int:
    if "toleranceDays" not in document:
        return DEFAULT_TOLERANCE_DAYS
    value = document["toleranceDays"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise LedgerError(REGISTRY, "toleranceDays must be an integer")
    if not 0 <= value <= _MAX_TOLERANCE_DAYS:
        raise LedgerError(
            REGISTRY, f"toleranceDays must be between 0 and {_MAX_TOLERANCE_DAYS}"
        )
    return value


def _money_field(raw: object, field: str, account_id: str) -> Decimal:
    if not isinstance(raw, str):
        raise LedgerError(
            REGISTRY, f"account {account_id!r}: {field} must be a decimal string"
        )
    try:
        # Same rules, same exactness as a statement amount (spec §4.1).
        return parse_amount_token(raw)
    except AmountError as exc:
        raise LedgerError(
            REGISTRY, f"account {account_id!r}: {field}: {exc.message}"
        ) from exc


def _load_registry(path: str) -> tuple[list[dict], int]:
    document = _read_json(path)
    tolerance = _tolerance_days(document)

    entries = document.get("accounts")
    if not isinstance(entries, list):
        raise LedgerError(REGISTRY, "registry must contain an 'accounts' array")
    if not entries:
        raise LedgerError(REGISTRY, "registry 'accounts' array is empty")

    specs: list[dict] = []
    seen: set[str] = set()
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise LedgerError(REGISTRY, f"account #{position} must be a JSON object")
        account_id = entry.get("id")
        if not isinstance(account_id, str) or not account_id.strip():
            raise LedgerError(REGISTRY, f"account #{position} has no usable 'id'")
        if account_id in seen:
            raise LedgerError(REGISTRY, f"duplicate account id: {account_id!r}")
        seen.add(account_id)

        currency = entry.get("currency")
        if not isinstance(currency, str) or not currency.strip():
            raise LedgerError(
                REGISTRY, f"account {account_id!r} has no usable 'currency'"
            )

        opening = _money_field(
            entry.get("openingBalance"), "openingBalance", account_id
        )

        statement = entry.get("statement")
        if statement is None or (
            isinstance(statement, str) and statement.strip() == ""
        ):
            statement = f"{account_id}.csv"
        if not isinstance(statement, str):
            raise LedgerError(
                REGISTRY, f"account {account_id!r}: 'statement' must be a string"
            )

        specs.append(
            {
                "id": account_id,
                "currency": currency.strip().upper(),
                "opening_balance": opening,
                "statement": statement,
            }
        )
    return specs, tolerance


def _load_accounts(specs: list[dict], registry_path: str) -> list[Account]:
    base = registry_path.rsplit("/", 1)[0] if "/" in registry_path else ""
    accounts: list[Account] = []
    for index, spec in enumerate(specs):
        relative = spec["statement"]
        resolved = (
            relative if relative.startswith("/") or not base else f"{base}/{relative}"
        )
        accounts.append(
            Account(
                index=index,
                account_id=spec["id"],
                currency=spec["currency"],
                opening_balance=spec["opening_balance"],
                statement_label=relative.rsplit("/", 1)[-1],
                statement=parse_statement(resolved),
            )
        )
    return accounts


# --------------------------------------------------------------------------- #
# Transfer detection (spec §5.2)
# --------------------------------------------------------------------------- #


def _sort_key_of(movement: Movement) -> tuple[int, int]:
    """Global row order: registry order for the account, then file order."""
    return (movement.account.index, movement.transaction.line)


def detect_transfers(accounts: list[Account], tolerance: int) -> list[Pairing]:
    """Greedily accept feasible pairs; every row may pair at most once."""
    outflows = [
        Movement(account, tx)
        for account in accounts
        for tx in account.rows
        if tx.amount < 0
    ]
    inflows = [
        Movement(account, tx)
        for account in accounts
        for tx in account.rows
        if tx.amount > 0
    ]

    candidates: list[tuple[tuple[int, int, int], Pairing]] = []
    for outflow in outflows:
        for inflow in inflows:
            if outflow.account.index == inflow.account.index:
                continue
            if outflow.account.currency != inflow.account.currency:
                continue
            if outflow.magnitude != inflow.magnitude:
                continue
            gap = days_between(
                outflow.transaction.value_date, inflow.transaction.value_date
            )
            if gap < 0 or gap > tolerance:
                continue
            # Ordered by nearest date first, then by line number (§5.2).
            key = (gap, *_sort_key_of(outflow), *_sort_key_of(inflow))
            candidates.append((key, Pairing(outflow, inflow, gap)))

    candidates.sort(key=lambda item: item[0])

    used: set[tuple[int, int]] = set()
    accepted: list[Pairing] = []
    for _, pairing in candidates:
        out_key = _sort_key_of(pairing.outflow)
        in_key = _sort_key_of(pairing.inflow)
        if out_key in used or in_key in used:
            continue
        used.add(out_key)
        used.add(in_key)
        accepted.append(pairing)

    accepted.sort(key=lambda p: (*_sort_key_of(p.outflow), *_sort_key_of(p.inflow)))
    return accepted


def net_worth_conserved(pairings: list[Pairing]) -> tuple[bool, Decimal]:
    """§5.4 — the rows that make up pairings must change net worth by exactly zero.

    Vacuously true when there are no pairings.  ``delta`` is the signed change,
    so a violation is reportable and not merely detectable.
    """
    movements: list[Decimal] = []
    for pairing in pairings:
        movements.append(pairing.outflow.transaction.amount)
        movements.append(pairing.inflow.transaction.amount)
    delta = exact_sum(movements)
    return delta == 0, delta


def net_worth_observations(accounts: list[Account]) -> tuple[dict, list[dict]]:
    """§5.5 — both net-worth computations at every observation point.

    An observation point is one account row, visited in registry order and then
    file order.  At each one:

    * ``netWorthFromAmounts`` = ``Σ (openingBalance + Σ amount)`` over accounts,
      carried incrementally as accounts advance;
    * ``netWorthFromBalances`` = ``Σ (latest recorded balanceAfter)`` over
      accounts, using the opening balance for an account that has no row yet.

    Every row advances both by the same amount on well-formed input, but they
    are read from *different sources* on purpose: the first from summed
    ``openingBalance + amount``, the second from each account's recorded
    ``balanceAfter`` column. In this module ``balance_after`` is exactly what
    the file said, so a row whose ``saldo`` disagrees with its amount makes the
    two diverge — which is the defect §5.5 asks to be reported. Returns the
    totals (both methods, at the final observation point) plus every mismatch.
    """
    amounts = [account.opening_balance for account in accounts]
    records = [account.opening_balance for account in accounts]
    positions = [0] * len(accounts)

    observations = 0
    mismatches: list[dict] = []
    while True:
        account = next(
            (a for a in accounts if positions[a.index] < len(a.rows)),
            None,
        )
        if account is None:
            break
        transaction = account.rows[positions[account.index]]
        positions[account.index] += 1

        amounts[account.index] = amounts[account.index] + transaction.amount
        records[account.index] = transaction.balance_after
        from_amounts = exact_sum(amounts)
        from_balances = exact_sum(records)

        observations += 1
        if from_amounts != from_balances:
            mismatches.append(
                {
                    "account": account.account_id,
                    "line": transaction.line,
                    "netWorthFromAmounts": canonical(from_amounts),
                    "netWorthFromBalances": canonical(from_balances),
                    "delta": canonical(from_balances - from_amounts),
                }
            )

    totals = {
        "observations": observations,
        "finalNetWorthFromAmounts": canonical(exact_sum(amounts)),
        "finalNetWorthFromBalances": canonical(exact_sum(records)),
    }
    return totals, mismatches


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _account_report(account: Account) -> dict:
    # The registry's openingBalance is the authoritative anchor and overrides
    # the value the parser would otherwise derive from the first row.
    checks = build_checks(account.statement, account.opening_balance)
    return {
        "id": account.account_id,
        "currency": account.currency,
        "statement": account.statement_label,
        "openingBalance": canonical(account.opening_balance),
        "closingBalance": canonical(account.statement.final_balance),
        "checks": {
            "rowCount": checks["rowCount"],
            "skippedEmptyRows": checks["skippedEmptyRows"],
            "reconciled": checks["reconciled"],
            "continuityErrors": checks["continuityErrors"],
            "cumulativeErrors": checks["cumulativeErrors"],
            "valueDateInversions": checks["valueDateInversions"],
        },
    }


def build_accounts(registry_path: str) -> dict:
    """Parse a registry, reconcile every account, detect transfers (spec §5)."""
    reset_derivation_count()
    specs, tolerance = _load_registry(registry_path)
    accounts = _load_accounts(specs, registry_path)

    pairings = detect_transfers(accounts, tolerance)
    conserved, delta = net_worth_conserved(pairings)
    totals, mismatches = net_worth_observations(accounts)

    return {
        "toleranceDays": tolerance,
        "accounts": [_account_report(account) for account in accounts],
        "transfers": [pairing.as_json() for pairing in pairings],
        "transferCheck": {
            "pairs": len(pairings),
            "pairedRows": len(pairings) * 2,
            # §5.3: unpaired rows are simply not transfers; this counter is
            # reserved for a later iteration and stays 0 by design.
            "unpairedTransferCandidates": 0,
            "netWorthConserved": conserved,
            "netWorthDelta": canonical(delta),
        },
        "netWorth": {
            "openingNetWorth": canonical(
                exact_sum(a.opening_balance for a in accounts)
            ),
            "observations": totals["observations"],
            "finalNetWorthFromAmounts": totals["finalNetWorthFromAmounts"],
            "finalNetWorthFromBalances": totals["finalNetWorthFromBalances"],
            "reconciled": not mismatches,
            "mismatches": mismatches,
        },
        "internals": {
            "amountOrDateRederivationSites": rederivation_site_count(),
            "amountOrDateRederivations": derivation_count(),
        },
    }
