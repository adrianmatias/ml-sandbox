"""Iteration A — multi-account reconciliation and transfer detection (spec §5).

One registry JSON plus one statement CSV per account go in; a complete
reconciliation comes out:

1. every §3 check, computed independently per account;
2. transfer detection (:mod:`ledger.transfers`);
3. ``netWorthConserved`` — the paired rows must cancel exactly;
4. ``netWorth`` (openings + amounts) and ``netWorthFromBalances`` (balances),
   compared at **every** observation point, with all mismatches reported;
5. an account set of any size — adding a third account needs no parsing change,
   because nothing here knows how many accounts the registry lists.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final

from ledger.checks import (
    CheckReport,
    OpeningBalance,
    derive_opening_balance,
    run_checks,
)
from ledger.domain import (
    Balance,
    LineNumber,
    ParsedStatement,
    Transaction,
    format_money,
)
from ledger.errors import (
    InputError,
    RegistryDuplicateAccountError,
    RegistryEmptyError,
    RegistryMissingMemberError,
    RegistryNonStringKeyError,
    RegistryNotArrayError,
    RegistryNotObjectError,
    RegistryOpeningBalanceError,
    RegistryToleranceError,
    RegistryUnknownKeysError,
    RegistryVersionError,
    UnsupportedRegistryJsonError,
)
from ledger.jsonshape import JSONObject, check_object, opening_object
from ledger.statement import parse_statement_file
from ledger.transfers import (
    DEFAULT_TOLERANCE_DAYS,
    UNPAIRED_TRANSFER_CANDIDATES,
    TransferAccount,
    TransferEndpoint,
    TransferMatch,
    TransferPairing,
    find_transfers,
    paired_endpoints,
)

REGISTRY_VERSION: Final[int] = 1
"""Registry schema version this module understands."""

REGISTRY_KEYS: Final[frozenset[str]] = frozenset(
    {"version", "toleranceDays", "accounts"}
)
"""Keys the registry file may carry; anything else is a registry error."""

ACCOUNT_KEYS: Final[frozenset[str]] = frozenset({"id", "currency", "openingBalance"})
"""Keys each entry of ``accounts`` may carry."""


@dataclass(frozen=True, slots=True)
class AccountEntry:
    """One account from the registry.

    Attributes:
        account_id: Stable id, e.g. ``A``; also the statement file stem.
        currency: ISO currency code.
        opening_balance: Declared opening balance, exact decimal.
    """

    account_id: str
    currency: str
    opening_balance: Balance


@dataclass(frozen=True, slots=True)
class AccountRegistry:
    """The parsed registry file.

    Attributes:
        tolerance_days: Transfer window, spec §5.2 (default 3).
        entries: Accounts in registry order.
    """

    tolerance_days: int
    entries: tuple[AccountEntry, ...]


@dataclass(frozen=True, slots=True)
class AccountResult:
    """The reconciliation of one account.

    Attributes:
        entry: The registry entry this result belongs to.
        statement: Parsed rows, skipped empty lines and tolerated row failures.
        checks: Every §3 check for this account.
        opening: Provenance of the cumulative anchor, including a declared
            opening balance that disagrees with the statement's first row.
    """

    entry: AccountEntry
    statement: ParsedStatement
    checks: CheckReport
    opening: OpeningBalance

    @property
    def account_id(self) -> str:
        """Registry id of this account."""
        return self.entry.account_id

    @property
    def currency(self) -> str:
        """ISO currency code of this account."""
        return self.entry.currency

    @property
    def transactions(self) -> tuple[Transaction, ...]:
        """Parsed rows, in file order."""
        return self.statement.transactions

    @property
    def final_balance(self) -> Balance:
        """Balance after the last row, or the declared opening balance if empty."""
        if not self.statement.transactions:
            return self.entry.opening_balance
        return self.statement.transactions[-1].balance_after

    @property
    def valid(self) -> bool:
        """False when any row failed to parse, which makes the CLI exit code 1."""
        return not self.statement.row_errors


@dataclass(frozen=True, slots=True)
class NetWorthObservation:
    """Net worth at one observation point, computed two independent ways.

    Attributes:
        account_id: The account whose row defines the point.
        line: 1-based line of that row.
        observed_on: The row's booking date.
        net_worth: Σ over accounts of ``declared opening + Σ amount``.
        net_worth_from_balances: Σ over accounts of the balance after their most
            recent row at or before this point.
    """

    account_id: str
    line: LineNumber
    observed_on: str
    net_worth: Balance
    net_worth_from_balances: Balance


@dataclass(frozen=True, slots=True)
class NetWorthMismatch:
    """A point where the two independent net-worth computations disagree."""

    account_id: str
    line: LineNumber
    net_worth: Balance
    net_worth_from_balances: Balance


@dataclass(frozen=True, slots=True)
class NetWorthSeries:
    """The whole net-worth timeline plus every mismatch found on it."""

    observations: tuple[NetWorthObservation, ...]
    mismatches: tuple[NetWorthMismatch, ...]

    @property
    def agrees(self) -> bool:
        """True iff no observation point disagreed."""
        return not self.mismatches


@dataclass(frozen=True, slots=True)
class AccountsRun:
    """The complete §5 result.

    Attributes:
        registry: The parsed registry.
        accounts: One reconciliation per registry entry, in registry order.
        pairing: All detected transfers.
        series: The net-worth timeline and its mismatches.
    """

    registry: AccountRegistry
    accounts: tuple[AccountResult, ...]
    pairing: TransferPairing
    series: NetWorthSeries

    @property
    def valid(self) -> bool:
        """False when any account had an unparsable row."""
        return all(account.valid for account in self.accounts)

    @property
    def net_worth_final(self) -> Balance:
        """Net worth at the last observation point; zero for an empty registry."""
        if not self.series.observations:
            return Balance(Decimal(0))
        return self.series.observations[-1].net_worth

    @property
    def parity_holds(self) -> bool:
        """Spec §5.5 — both net-worth computations agree everywhere."""
        return self.series.agrees

    @property
    def all_reconciled(self) -> bool:
        """True iff no account reported a continuity or cumulative error."""
        return all(account.checks.reconciled for account in self.accounts)

    @property
    def transfer_accounts(self) -> tuple[TransferAccount, ...]:
        """The accounts projected onto what transfer detection needs."""
        return tuple(
            TransferAccount(
                account_id=account.account_id,
                currency=account.currency,
                transactions=account.transactions,
            )
            for account in self.accounts
        )

    @property
    def paired_rows(self) -> tuple[TransferEndpoint, ...]:
        """Every row consumed by a match, in ascending ``(account, line)`` order."""
        return paired_endpoints(self.transfer_accounts, self.pairing)

    def payload(self) -> JSONObject:
        """Render the §5 data member of the envelope."""
        return accounts_data(self)


# --------------------------------------------------------------------------- #
# Registry loading
# --------------------------------------------------------------------------- #


def _require_object(raw: object, what: str) -> dict[str, object]:
    """Narrow ``raw`` to a JSON object, or raise a registry error."""
    if not isinstance(raw, dict):
        raise RegistryNotObjectError(what)
    result: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise RegistryNonStringKeyError(what)
        result[key] = value
    return result


def _require_str(mapping: dict[str, object], key: str, what: str) -> str:
    """Read a required string member."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise RegistryMissingMemberError(what, key)
    return value


def _require_list(raw: object, what: str) -> list[object]:
    """Narrow ``raw`` to a JSON array, or raise a registry error."""
    if not isinstance(raw, list):
        raise RegistryNotArrayError(what)
    return list(raw)


def _reject_unknown_keys(
    mapping: dict[str, object], allowed: frozenset[str], what: str
) -> None:
    """Fail on unexpected members rather than silently ignoring a typo."""
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise RegistryUnknownKeysError(what, tuple(unknown))


def _parse_tolerance_days(mapping: dict[str, object]) -> int:
    """Read ``toleranceDays``, defaulting to spec §5.2's 3."""
    if "toleranceDays" not in mapping:
        return DEFAULT_TOLERANCE_DAYS
    value = mapping["toleranceDays"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RegistryToleranceError(value)
    return value


def _parse_opening_balance(mapping: dict[str, object], account_id: str) -> Balance:
    """Read an exact-decimal ``openingBalance`` from the registry."""
    value = mapping.get("openingBalance")
    if not isinstance(value, str):
        raise RegistryOpeningBalanceError(account_id, value)
    try:
        return Balance(Decimal(value))
    except InvalidOperation as exc:
        raise RegistryOpeningBalanceError(account_id, value) from exc


def parse_registry(document: str) -> AccountRegistry:
    """Parse the accounts registry JSON (spec §5).

    Args:
        document: The whole registry file as text.

    Returns:
        The registry, in file order.

    Raises:
        RegistryError: If the JSON is malformed or any member is missing, the
            wrong type, or unknown.
    """
    try:
        decoded = json.loads(document)
    except json.JSONDecodeError as exc:
        raise UnsupportedRegistryJsonError(exc.msg) from exc
    mapping = _require_object(decoded, "registry")
    _reject_unknown_keys(mapping, REGISTRY_KEYS, "registry")
    version = mapping.get("version")
    if version != REGISTRY_VERSION:
        raise RegistryVersionError(version)
    tolerance_days = _parse_tolerance_days(mapping)
    entries: list[AccountEntry] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(
        _require_list(mapping.get("accounts"), "accounts")
    ):
        entry = _require_object(raw_entry, f"accounts[{index}]")
        _reject_unknown_keys(entry, ACCOUNT_KEYS, f"accounts[{index}]")
        account_id = _require_str(entry, "id", f"accounts[{index}]")
        if account_id in seen:
            raise RegistryDuplicateAccountError(account_id)
        seen.add(account_id)
        entries.append(
            AccountEntry(
                account_id=account_id,
                currency=_require_str(entry, "currency", f"accounts[{index}]"),
                opening_balance=_parse_opening_balance(entry, account_id),
            )
        )
    if not entries:
        raise RegistryEmptyError
    return AccountRegistry(tolerance_days=tolerance_days, entries=tuple(entries))


def load_registry(path: Path) -> AccountRegistry:
    """Read and parse the registry file.

    Raises:
        InputError: If the file cannot be read.
        RegistryError: If the registry is malformed.
    """
    try:
        document = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise InputError(path.name, exc.strerror or str(exc)) from exc
    except UnicodeDecodeError as exc:
        raise InputError(path.name, f"not valid UTF-8 ({exc.reason})") from exc
    return parse_registry(document)


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #


def _account_result(entry: AccountEntry, path: Path) -> AccountResult:
    """Parse one account's statement and run every §3 check on it."""
    statement = parse_statement_file(path)
    opening = derive_opening_balance(statement.transactions, entry.opening_balance)
    checks = run_checks(
        statement.transactions,
        skipped_empty_rows=statement.skipped_empty_rows,
        parse_errors=statement.row_errors,
        declared_opening=entry.opening_balance,
    )
    return AccountResult(
        entry=entry, statement=statement, checks=checks, opening=opening
    )


def _sum_amounts(transactions: Sequence[Transaction]) -> Decimal:
    """Sum the signed amounts of a row sequence."""
    total = Decimal(0)
    for transaction in transactions:
        total += transaction.amount
    return total


def _account_observation_points(account: AccountResult) -> list[tuple[str, int, str]]:
    """Every point at which this account's net worth is observed.

    Returns:
        ``(bookingDate ISO, line, accountId)`` for each of the account's rows.
    """
    return [
        (
            transaction.booking_date.isoformat(),
            int(transaction.line),
            account.account_id,
        )
        for transaction in account.transactions
    ]


def _net_worth_at(
    accounts: Sequence[AccountResult],
    point_date: str,
    point_line: int,
) -> Balance:
    """Σ over accounts that have started: declared opening + Σ amounts so far."""
    total = Decimal(0)
    for account in accounts:
        total += account.entry.opening_balance
        for transaction in account.transactions:
            key = (transaction.booking_date.isoformat(), int(transaction.line))
            if key <= (point_date, point_line):
                total += transaction.amount
    return Balance(total)


def _balances_at(
    accounts: Sequence[AccountResult],
    point_date: str,
    point_line: int,
) -> Balance:
    """Σ over accounts that have started: balance after their latest row so far."""
    total = Decimal(0)
    for account in accounts:
        latest: Balance | None = None
        for transaction in account.transactions:
            key = (transaction.booking_date.isoformat(), int(transaction.line))
            if key <= (point_date, point_line):
                latest = transaction.balance_after
        if latest is not None:
            total += latest
    return Balance(total)


def build_net_worth_series(accounts: Sequence[AccountResult]) -> NetWorthSeries:
    """Compare §5.5's two independent net-worth computations at every observation point.

    An observation point is one row of one account, keyed by
    ``(bookingDate, line)``. Both totals are rebuilt from scratch at each point
    — ``netWorth`` from declared openings and amounts, ``netWorthFromBalances``
    from the balances themselves — so neither can be derived from the other.
    Accounts that have not started yet contribute nothing rather than a guess.
    """
    points: list[tuple[str, int, str]] = []
    for account in accounts:
        points.extend(_account_observation_points(account))
    points.sort()

    observations: list[NetWorthObservation] = []
    mismatches: list[NetWorthMismatch] = []
    for point_date, point_line, account_id in points:
        net_worth = _net_worth_at(accounts, point_date, point_line)
        from_balances = _balances_at(accounts, point_date, point_line)
        observations.append(
            NetWorthObservation(
                account_id=account_id,
                line=LineNumber(point_line),
                observed_on=point_date,
                net_worth=net_worth,
                net_worth_from_balances=from_balances,
            )
        )
        if net_worth != from_balances:
            mismatches.append(
                NetWorthMismatch(
                    account_id=account_id,
                    line=LineNumber(point_line),
                    net_worth=net_worth,
                    net_worth_from_balances=from_balances,
                )
            )
    return NetWorthSeries(
        observations=tuple(observations), mismatches=tuple(mismatches)
    )


def reconcile(
    registry: AccountRegistry,
    statements: Sequence[AccountResult],
) -> AccountsRun:
    """Combine per-account checks, transfer detection and the net-worth timeline."""
    run = AccountsRun(
        registry=registry,
        accounts=tuple(statements),
        pairing=TransferPairing((), frozenset(), UNPAIRED_TRANSFER_CANDIDATES),
        series=NetWorthSeries((), ()),
    )
    pairing = find_transfers(run.transfer_accounts, registry.tolerance_days)
    return AccountsRun(
        registry=registry,
        accounts=run.accounts,
        pairing=pairing,
        series=build_net_worth_series(statements),
    )


def run_accounts(registry_path: Path) -> AccountsRun:
    """Reconcile every account named by a registry file (spec §5).

    Args:
        registry_path: Path to ``accounts.json``. Each account's statement is
            read from ``<id>.csv`` next to it.

    Returns:
        The complete §5 run.

    Raises:
        InputError: If the registry or any statement cannot be read.
        HeaderError: If any statement's header does not match §2.
        RowError: If any statement's CSV quoting is malformed.
        RegistryError: If the registry is malformed.
    """
    registry = load_registry(registry_path)
    results = tuple(
        _account_result(entry, registry_path.with_name(f"{entry.account_id}.csv"))
        for entry in registry.entries
    )
    return reconcile(registry, results)


# --------------------------------------------------------------------------- #
# JSON rendering
# --------------------------------------------------------------------------- #


def endpoint_object(endpoint: TransferEndpoint) -> JSONObject:
    """Serialise one side of a transfer."""
    return {
        "account": endpoint.account_id,
        "line": endpoint.line,
        "valueDate": endpoint.value_date.isoformat(),
        "amount": format_money(endpoint.amount),
    }


def match_object(match: TransferMatch) -> JSONObject:
    """Serialise one detected transfer."""
    return {
        "outflow": endpoint_object(match.outflow),
        "inflow": endpoint_object(match.inflow),
        "gapDays": match.gap_days,
        "netEffect": format_money(match.net_effect),
    }


def account_object(account: AccountResult) -> JSONObject:
    """Serialise one account's reconciliation."""
    return {
        "id": account.account_id,
        "currency": account.currency,
        "openingBalance": format_money(account.entry.opening_balance),
        "openingBalanceApplied": opening_object(account.opening),
        "finalBalance": format_money(account.final_balance),
        "transactions": [
            {
                "line": transaction.line,
                "bookingDate": transaction.booking_date.isoformat(),
                "valueDate": transaction.value_date.isoformat(),
                "amount": format_money(transaction.amount),
                "balanceAfter": format_money(transaction.balance_after),
                "rawAmount": transaction.raw_amount,
                "rawBalance": transaction.raw_balance,
                "checksum": transaction.checksum,
            }
            for transaction in account.transactions
        ],
        "checks": check_object(account.checks),
    }


def observation_object(observation: NetWorthObservation) -> JSONObject:
    """Serialise one net-worth observation point."""
    return {
        "account": observation.account_id,
        "line": observation.line,
        "observedOn": observation.observed_on,
        "netWorth": format_money(observation.net_worth),
        "netWorthFromBalances": format_money(observation.net_worth_from_balances),
    }


def mismatch_object(mismatch: NetWorthMismatch) -> JSONObject:
    """Serialise one net-worth disagreement."""
    return {
        "account": mismatch.account_id,
        "line": mismatch.line,
        "netWorth": format_money(mismatch.net_worth),
        "netWorthFromBalances": format_money(mismatch.net_worth_from_balances),
    }


def accounts_data(run: AccountsRun) -> JSONObject:
    """Serialise a complete §5 run."""
    registry = run.registry
    return {
        "registry": {
            "version": REGISTRY_VERSION,
            "toleranceDays": registry.tolerance_days,
            "accounts": [
                {
                    "id": entry.account_id,
                    "currency": entry.currency,
                    "openingBalance": format_money(entry.opening_balance),
                }
                for entry in registry.entries
            ],
        },
        "accounts": [account_object(account) for account in run.accounts],
        "netWorthSeries": [observation_object(o) for o in run.series.observations],
        "netWorthMismatches": [mismatch_object(m) for m in run.series.mismatches],
        "netWorthFinal": format_money(run.net_worth_final),
        "netWorthAgrees": run.parity_holds,
        "transfers": {
            "pairs": [match_object(match) for match in run.pairing.matches],
            "pairCount": len(run.pairing.matches),
            "pairedRows": [endpoint_object(endpoint) for endpoint in run.paired_rows],
            "unpairedTransferCandidates": run.pairing.unpaired_transfer_candidates,
        },
        "netWorthConserved": run.pairing.net_worth_conserved,
        "netWorthConservedDetail": {
            "pairedNetEffect": format_money(run.pairing.net_effect),
        },
        "reconciled": run.all_reconciled,
    }
