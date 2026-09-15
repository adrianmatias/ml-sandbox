"""Transfer detection — spec §5.2 and §5.4.

A transfer pairs one **outflow** row of one account with one **inflow** row of a
*different* account when:

* ``|outflow.amount| == inflow.amount`` (exact decimal equality of magnitudes),
* both accounts use the same currency,
* ``outflow.valueDate <= inflow.valueDate <= outflow.valueDate + toleranceDays``.

Constraints: each row pairs at most once; at most one outgoing and one incoming
pair per row; prefer the **nearest** feasible date, ties broken by the **lowest
line number**, so the result is deterministic. Rows left unpaired are simply not
transfers and are reported nowhere.

The whole predicate lives in :func:`is_feasible_pair` and the date arithmetic in
:func:`value_date_gap_days`; nothing else in the package re-derives an amount or
a date.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Final

from ledger.domain import Amount, LineNumber, Transaction

DEFAULT_TOLERANCE_DAYS: Final[int] = 3
"""Spec §5.2's default tolerance between an outflow and its matching inflow."""

UNPAIRED_TRANSFER_CANDIDATES: Final[int] = 0
"""Reserved counter (spec §5.3): always 0 in this iteration, by construction."""


@dataclass(frozen=True, slots=True)
class TransferAccount:
    """One account as transfer detection sees it: identity, currency and rows.

    Attributes:
        account_id: Registry id, e.g. ``A``.
        currency: ISO currency code; only equal codes may pair.
        transactions: Parsed rows in file order.
    """

    account_id: str
    currency: str
    transactions: tuple[Transaction, ...]


@dataclass(frozen=True, slots=True)
class TransferEndpoint:
    """One side of a detected transfer, reduced to the fields pairing cares about.

    Attributes:
        account_id: Account the row belongs to.
        line: 1-based physical line within that account's statement.
        value_date: The row's value date — the date the pairing rule uses.
        amount: The signed amount, so conservation can sum endpoints directly.
    """

    account_id: str
    line: LineNumber
    value_date: date
    amount: Amount


@dataclass(frozen=True, slots=True)
class TransferMatch:
    """A detected outflow → inflow pair.

    Attributes:
        outflow: The debited side.
        inflow: The credited side, on another account.
        gap_days: ``inflow.valueDate - outflow.valueDate`` in days, never negative.
    """

    outflow: TransferEndpoint
    inflow: TransferEndpoint
    gap_days: int

    @property
    def net_effect(self) -> Amount:
        """Sum of both endpoints: exactly zero for a well-formed pair."""
        return Amount(self.outflow.amount + self.inflow.amount)


@dataclass(frozen=True, slots=True)
class TransferPairing:
    """The complete pairing result for one run.

    Attributes:
        matches: Detected pairs, in the deterministic order of the algorithm.
        paired_lines: ``(account_id, line)`` of every row consumed by a match.
        unpaired_transfer_candidates: Reserved counter, always 0 (spec §5.3).
    """

    matches: tuple[TransferMatch, ...]
    paired_lines: frozenset[tuple[str, int]]
    unpaired_transfer_candidates: int

    @property
    def net_effect(self) -> Amount:
        """Sum over all matched endpoints; exactly zero iff net worth is conserved."""
        total = Amount(Decimal(0))
        for match in self.matches:
            total = Amount(total + match.net_effect)
        return total

    @property
    def net_worth_conserved(self) -> bool:
        """Spec §5.4 — true iff every paired row's net effect cancels exactly."""
        return self.net_effect == 0


def value_date_gap_days(outflow: Transaction, inflow: Transaction) -> int:
    """Days from an outflow's value date to an inflow's value date.

    The single place in the package where transfer date arithmetic happens.
    """
    return (inflow.value_date - outflow.value_date).days


def is_feasible_pair(
    outflow: TransferAccount,
    inflow: TransferAccount,
    tolerance_days: int = DEFAULT_TOLERANCE_DAYS,
) -> bool:
    """The §5.2 pairing predicate, as one readable total function.

    Args:
        outflow: Account holding the candidate debit row.
        inflow: Account holding the candidate credit row.
        tolerance_days: Maximum allowed value-date gap.

    Returns:
        True when the rows are on different accounts, share a currency, and the
        inflow's value date falls inside
        ``[outflow.valueDate, outflow.valueDate + toleranceDays]``.
    """

    def _window_holds(
        out_rows: Sequence[Transaction], in_rows: Sequence[Transaction]
    ) -> bool:
        return any(
            0 <= value_date_gap_days(deb, cred) <= tolerance_days
            for deb in out_rows
            for cred in in_rows
        )

    return (
        outflow.account_id != inflow.account_id
        and outflow.currency == inflow.currency
        and _window_holds(outflow.transactions, inflow.transactions)
    )


def _endpoint(account_id: str, transaction: Transaction) -> TransferEndpoint:
    """Project a transaction onto the only fields transfer logic may use."""
    return TransferEndpoint(
        account_id=account_id,
        line=transaction.line,
        value_date=transaction.value_date,
        amount=transaction.amount,
    )


type CandidateEntry = tuple[str, int, Transaction]
"""An unpaired inflow as ``(account_id, line, transaction)``."""


def _candidate_key(entry: CandidateEntry, outflow: Transaction) -> tuple[int, int, str]:
    """Sort key of a candidate inflow: nearest gap, then lowest line number."""
    account_id, line, transaction = entry
    return (value_date_gap_days(outflow, transaction), line, account_id)


def _outflow_key(entry: CandidateEntry) -> tuple[date, str, int]:
    """Visit outflows oldest value date first, then by account id and line.

    The spec fixes which *pair* wins, not which outflow is offered first; this
    key makes that choice deterministic and independent of registry order.
    """
    account_id, line, transaction = entry
    return (transaction.value_date, account_id, line)


def _feasible_candidates(
    outflow: Transaction,
    candidates: Sequence[CandidateEntry],
    tolerance_days: int,
) -> list[CandidateEntry]:
    """Candidates whose date window accepts this outflow, nearest first."""
    feasible = [
        entry
        for entry in candidates
        if 0 <= value_date_gap_days(outflow, entry[2]) <= tolerance_days
    ]
    feasible.sort(key=lambda entry: _candidate_key(entry, outflow))
    return feasible


def find_transfers(
    accounts: Sequence[TransferAccount],
    tolerance_days: int = DEFAULT_TOLERANCE_DAYS,
) -> TransferPairing:
    """Pair outflows with inflows across accounts (spec §5.2).

    Outflows are visited in a deterministic order — ascending value date, then
    ascending account id, then ascending line — so the output never depends on
    the order accounts appear in the registry. Each match consumes both rows, so
    a row can appear in at most one pair and as at most one outflow and one
    inflow.

    Args:
        accounts: Every account with its parsed statement.
        tolerance_days: Maximum value-date gap; defaults to spec §5.2's 3.

    Returns:
        The pairing, including which rows were consumed.
    """
    currencies = {account.account_id: account.currency for account in accounts}
    outflows: list[CandidateEntry] = []
    inflows: list[CandidateEntry] = []
    for account in accounts:
        for transaction in account.transactions:
            entry = (account.account_id, int(transaction.line), transaction)
            if transaction.amount < 0:
                outflows.append(entry)
            elif transaction.amount > 0:
                inflows.append(entry)
    outflows.sort(key=_outflow_key)

    used_inflows: set[tuple[str, int]] = set()
    matches: list[TransferMatch] = []

    for entry in outflows:
        account_id, _, outflow = entry
        outflow_currency = currencies[account_id]
        available = [
            candidate
            for candidate in inflows
            if (candidate[0], candidate[1]) not in used_inflows
            and candidate[0] != account_id
            and currencies[candidate[0]] == outflow_currency
            and candidate[2].amount == -outflow.amount
        ]
        feasible = _feasible_candidates(outflow, available, tolerance_days)
        if not feasible:
            continue
        inflow_account_id, inflow_line, inflow = feasible[0]
        used_inflows.add((inflow_account_id, inflow_line))
        matches.append(
            TransferMatch(
                outflow=_endpoint(account_id, outflow),
                inflow=_endpoint(inflow_account_id, inflow),
                gap_days=value_date_gap_days(outflow, inflow),
            )
        )

    paired: set[tuple[str, int]] = set(used_inflows)
    for match in matches:
        paired.add((match.outflow.account_id, int(match.outflow.line)))
    return TransferPairing(
        matches=tuple(matches),
        paired_lines=frozenset(paired),
        unpaired_transfer_candidates=UNPAIRED_TRANSFER_CANDIDATES,
    )


def paired_endpoints(
    accounts: Iterable[TransferAccount],
    pairing: TransferPairing,
) -> tuple[TransferEndpoint, ...]:
    """Every row consumed by a match, in ascending ``(account, line)`` order."""
    endpoints: list[TransferEndpoint] = []
    for account in accounts:
        for transaction in account.transactions:
            if (account.account_id, int(transaction.line)) in pairing.paired_lines:
                endpoints.append(_endpoint(account.account_id, transaction))
    endpoints.sort(key=lambda endpoint: (endpoint.account_id, int(endpoint.line)))
    return tuple(endpoints)
