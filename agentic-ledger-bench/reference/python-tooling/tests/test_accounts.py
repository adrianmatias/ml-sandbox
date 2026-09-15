"""Spec §5 — multi-account reconciliation and transfer detection.

The shared fixture in ``spec/shared-fixtures/`` is the arbiter here: its expected
values come from ``CLARIFICATIONS.md`` §2, not from this implementation.
"""

import json
import unittest
from decimal import Decimal
from typing import ClassVar

from ledger.accounts import AccountsRun, parse_registry, run_accounts
from ledger.amounts import parse_amount
from ledger.dates import parse_date
from ledger.domain import Balance, LineNumber, Transaction
from ledger.errors import RegistryError
from ledger.transfers import (
    TransferAccount,
    find_transfers,
    is_feasible_pair,
    value_date_gap_days,
)

from .helpers import (
    EXPECTED_SHARED_NET_WORTH,
    EXPECTED_SHARED_PAIRS,
    SHARED_FIXTURES,
    as_dict,
    as_list,
    count_of,
    data_of,
    fixture,
    member,
    run_cli,
    scratch_dir,
)


def transaction(line: int, booking: str, value: str, amount: str) -> Transaction:
    """Build a transfer-test transaction with the fields pairing reads."""
    return Transaction(
        line=LineNumber(line),
        booking_date=parse_date(booking),
        value_date=parse_date(value),
        amount=parse_amount(amount),
        balance_after=Balance(Decimal(0)),
        raw_amount=amount,
        raw_balance="0",
        checksum="",
    )


class SharedFixtureTests(unittest.TestCase):
    """Every number ``CLARIFICATIONS.md`` §2 declares, asserted on our output."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the accounts command once for the class."""
        cls.stdin_result = run_cli("accounts", str(SHARED_FIXTURES / "accounts.json"))
        cls.data = data_of(cls.stdin_result.json)

    def test_exit_code_is_zero(self) -> None:
        """The shared fixture is well formed, so the CLI succeeds."""
        self.assertEqual(self.stdin_result.returncode, 0)

    def test_three_accounts_are_reconciled(self) -> None:
        """The registry lists three accounts and all three appear."""
        accounts = as_list(member(self.data, "accounts"), "accounts")
        self.assertEqual(
            [as_dict(e, "account")["id"] for e in accounts], ["A", "B", "C"]
        )

    def test_final_balances_match(self) -> None:
        """A 455.25, B 429.50, C 254.50."""
        accounts = as_list(member(self.data, "accounts"), "accounts")
        balances = {
            as_dict(entry, "account")["id"]: as_dict(entry, "account")["finalBalance"]
            for entry in accounts
        }
        self.assertEqual(balances, {"A": "455.25", "B": "429.50", "C": "254.50"})

    def test_per_account_checks_are_clean(self) -> None:
        """Zero continuity and zero cumulative errors on every account."""
        accounts = as_list(member(self.data, "accounts"), "accounts")
        for entry in accounts:
            account = as_dict(entry, "account")
            with self.subTest(account=account["id"]):
                checks = as_dict(member(account, "checks"), "checks")
                self.assertEqual(checks["continuityErrors"], [])
                self.assertEqual(checks["cumulativeErrors"], [])
                self.assertTrue(checks["reconciled"])

    def test_net_worth_is_1139_25_by_both_computations(self) -> None:
        """Σ(openings + amounts) and Σ(balances) both give 1139.25."""
        self.assertEqual(self.data["netWorthFinal"], EXPECTED_SHARED_NET_WORTH)
        self.assertTrue(self.data["netWorthAgrees"])
        self.assertEqual(self.data["netWorthMismatches"], [])
        series = as_list(member(self.data, "netWorthSeries"), "netWorthSeries")
        last = as_dict(series[-1], "netWorthSeries[-1]")
        self.assertEqual(last["netWorth"], EXPECTED_SHARED_NET_WORTH)
        self.assertEqual(last["netWorthFromBalances"], EXPECTED_SHARED_NET_WORTH)

    def test_exactly_two_transfer_pairs(self) -> None:
        """The shared fixture contains exactly two transfers."""
        transfers = as_dict(member(self.data, "transfers"), "transfers")
        self.assertEqual(transfers["pairCount"], EXPECTED_SHARED_PAIRS)
        self.assertEqual(
            count_of(member(transfers, "pairs"), "pairs"), EXPECTED_SHARED_PAIRS
        )

    def test_the_two_pairs_are_the_expected_ones(self) -> None:
        """A:3 -250 → B:4 +250 and A:4 -300 → C:2 +300, with the right gaps."""
        transfers = as_dict(member(self.data, "transfers"), "transfers")
        pairs = as_list(member(transfers, "pairs"), "pairs")
        summary = [
            (
                as_dict(member(as_dict(pair, "pair"), "outflow"), "outflow")["account"],
                as_dict(member(as_dict(pair, "pair"), "outflow"), "outflow")["line"],
                as_dict(member(as_dict(pair, "pair"), "outflow"), "outflow")["amount"],
                as_dict(member(as_dict(pair, "pair"), "inflow"), "inflow")["account"],
                as_dict(member(as_dict(pair, "pair"), "inflow"), "inflow")["line"],
                as_dict(member(as_dict(pair, "pair"), "inflow"), "inflow")["amount"],
                as_dict(pair, "pair")["gapDays"],
            )
            for pair in pairs
        ]
        self.assertEqual(
            summary,
            [
                ("A", 3, "-250.00", "B", 4, "250.00", 1),
                ("A", 4, "-300.00", "C", 2, "300.00", 0),
            ],
        )

    def test_net_worth_is_conserved_by_the_paired_rows(self) -> None:
        """§5.4: the paired rows cancel exactly."""
        self.assertTrue(self.data["netWorthConserved"])
        detail = as_dict(member(self.data, "netWorthConservedDetail"), "detail")
        self.assertEqual(detail["pairedNetEffect"], "0.00")

    def test_unpaired_counter_is_reserved_at_zero(self) -> None:
        """§5.3: the reserved counter stays 0."""
        transfers = as_dict(member(self.data, "transfers"), "transfers")
        self.assertEqual(transfers["unpairedTransferCandidates"], 0)

    def test_no_row_is_paired_twice(self) -> None:
        """Each row pairs at most once, so the paired-row list has no duplicates."""
        transfers = as_dict(member(self.data, "transfers"), "transfers")
        endpoints = as_list(member(transfers, "pairedRows"), "pairedRows")
        keys = [
            (
                as_dict(entry, "pairedRow")["account"],
                as_dict(entry, "pairedRow")["line"],
            )
            for entry in endpoints
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(keys), 4)

    def test_the_decoy_rows_are_not_paired(self) -> None:
        """The deliberate traps stay unpaired: B:3, B:6, C:4, A:2, A:5."""
        transfers = as_dict(member(self.data, "transfers"), "transfers")
        endpoints = as_list(member(transfers, "pairedRows"), "pairedRows")
        paired = {
            (
                as_dict(entry, "pairedRow")["account"],
                as_dict(entry, "pairedRow")["line"],
            )
            for entry in endpoints
        }
        for decoy in (("B", 3), ("B", 6), ("C", 4), ("A", 2), ("A", 5)):
            with self.subTest(row=decoy):
                self.assertNotIn(decoy, paired)


class TransferRuleTests(unittest.TestCase):
    """§5.2's predicate, tested directly on constructed accounts."""

    def test_gap_days_is_signed_forward(self) -> None:
        """The gap is inflow minus outflow in days."""
        outflow = transaction(2, "05/01/2026", "05/01/2026", "-250.00")
        inflow = transaction(4, "06/01/2026", "06/01/2026", "250.00")
        self.assertEqual(value_date_gap_days(outflow, inflow), 1)

    def test_inflow_before_outflow_is_not_feasible(self) -> None:
        """``inflow.valueDate >= outflow.valueDate`` is required, not optional."""
        outflow = transaction(2, "20/01/2026", "20/01/2026", "-120.50")
        inflow = transaction(4, "15/01/2026", "15/01/2026", "120.50")
        left = TransferAccount("B", "EUR", (outflow,))
        right = TransferAccount("C", "EUR", (inflow,))
        self.assertFalse(is_feasible_pair(left, right))

    def test_tolerance_is_inclusive_of_its_boundary(self) -> None:
        """A gap exactly equal to ``toleranceDays`` is still feasible."""
        outflow = transaction(2, "05/01/2026", "05/01/2026", "-10.00")
        inflow = transaction(4, "08/01/2026", "08/01/2026", "10.00")
        self.assertEqual(value_date_gap_days(outflow, inflow), 3)
        left = TransferAccount("A", "EUR", (outflow,))
        right = TransferAccount("B", "EUR", (inflow,))
        self.assertTrue(is_feasible_pair(left, right, 3))

    def test_beyond_tolerance_is_not_feasible(self) -> None:
        """One day past the tolerance is not a transfer."""
        outflow = transaction(2, "05/01/2026", "05/01/2026", "-10.00")
        inflow = transaction(4, "09/01/2026", "09/01/2026", "10.00")
        pairing = find_transfers(
            (
                TransferAccount("A", "EUR", (outflow,)),
                TransferAccount("B", "EUR", (inflow,)),
            ),
            tolerance_days=3,
        )
        self.assertEqual(pairing.matches, ())

    def test_tolerance_is_configurable(self) -> None:
        """Widening the window turns the same rows into a transfer."""
        outflow = transaction(2, "05/01/2026", "05/01/2026", "-10.00")
        inflow = transaction(4, "09/01/2026", "09/01/2026", "10.00")
        pairing = find_transfers(
            (
                TransferAccount("A", "EUR", (outflow,)),
                TransferAccount("B", "EUR", (inflow,)),
            ),
            tolerance_days=4,
        )
        self.assertEqual(len(pairing.matches), 1)
        self.assertEqual(pairing.matches[0].gap_days, 4)

    def test_different_currencies_never_pair(self) -> None:
        """Equal amounts in different currencies are not a transfer."""
        outflow = transaction(2, "05/01/2026", "05/01/2026", "-10.00")
        inflow = transaction(4, "06/01/2026", "06/01/2026", "10.00")
        pairing = find_transfers(
            (
                TransferAccount("A", "EUR", (outflow,)),
                TransferAccount("B", "USD", (inflow,)),
            )
        )
        self.assertEqual(pairing.matches, ())

    def test_two_outflows_never_pair_with_each_other(self) -> None:
        """Both rows debited: no transfer, however equal the magnitudes."""
        left = transaction(2, "05/01/2026", "05/01/2026", "-120.50")
        right = transaction(4, "05/01/2026", "05/01/2026", "-120.50")
        pairing = find_transfers(
            (
                TransferAccount("B", "EUR", (left,)),
                TransferAccount("C", "EUR", (right,)),
            )
        )
        self.assertEqual(pairing.matches, ())

    def test_same_account_rows_never_pair(self) -> None:
        """An account cannot transfer to itself under §5.2."""
        outflow = transaction(2, "05/01/2026", "05/01/2026", "-10.00")
        inflow = transaction(4, "06/01/2026", "06/01/2026", "10.00")
        pairing = find_transfers((TransferAccount("A", "EUR", (outflow, inflow)),))
        self.assertEqual(pairing.matches, ())

    def test_nearest_date_wins(self) -> None:
        """With two feasible candidates, the closer one is chosen."""
        outflow = transaction(2, "05/01/2026", "05/01/2026", "-10.00")
        near = transaction(4, "06/01/2026", "06/01/2026", "10.00")
        far = transaction(6, "07/01/2026", "07/01/2026", "10.00")
        pairing = find_transfers(
            (
                TransferAccount("A", "EUR", (outflow,)),
                TransferAccount("B", "EUR", (near,)),
                TransferAccount("C", "EUR", (far,)),
            )
        )
        self.assertEqual(len(pairing.matches), 1)
        self.assertEqual(pairing.matches[0].inflow.account_id, "B")

    def test_ties_break_on_the_lowest_line_number(self) -> None:
        """Equal gaps are decided by the lowest line number, deterministically."""
        outflow = transaction(5, "05/01/2026", "05/01/2026", "-10.00")
        first = transaction(20, "06/01/2026", "06/01/2026", "10.00")
        second = transaction(30, "06/01/2026", "06/01/2026", "10.00")
        pairing = find_transfers(
            (
                TransferAccount("A", "EUR", (outflow,)),
                TransferAccount("B", "EUR", (first,)),
                TransferAccount("C", "EUR", (second,)),
            )
        )
        self.assertEqual(pairing.matches[0].inflow.account_id, "B")
        self.assertEqual(int(pairing.matches[0].inflow.line), 20)

    def test_a_row_is_consumed_only_once(self) -> None:
        """Two outflows competing for one inflow: only one pair is produced."""
        first_out = transaction(2, "05/01/2026", "05/01/2026", "-10.00")
        second_out = transaction(3, "05/01/2026", "05/01/2026", "-10.00")
        inflow = transaction(4, "06/01/2026", "06/01/2026", "10.00")
        pairing = find_transfers(
            (
                TransferAccount("A", "EUR", (first_out, second_out)),
                TransferAccount("B", "EUR", (inflow,)),
            )
        )
        self.assertEqual(len(pairing.matches), 1)
        self.assertEqual(len(pairing.paired_lines), 2)

    def test_pairing_is_independent_of_registry_order(self) -> None:
        """Reversing the account order does not change the detected pairs."""
        outflow = transaction(2, "05/01/2026", "05/01/2026", "-10.00")
        inflow = transaction(4, "06/01/2026", "06/01/2026", "10.00")
        forward = find_transfers(
            (
                TransferAccount("A", "EUR", (outflow,)),
                TransferAccount("B", "EUR", (inflow,)),
            )
        )
        backward = find_transfers(
            (
                TransferAccount("B", "EUR", (inflow,)),
                TransferAccount("A", "EUR", (outflow,)),
            )
        )
        self.assertEqual(forward.matches, backward.matches)

    def test_zero_amounts_are_neither_outflows_nor_inflows(self) -> None:
        """A zero row has no direction, so it can never be half a transfer."""
        zero = transaction(2, "05/01/2026", "05/01/2026", "0.00")
        other = transaction(3, "05/01/2026", "05/01/2026", "0.00")
        pairing = find_transfers(
            (
                TransferAccount("A", "EUR", (zero,)),
                TransferAccount("B", "EUR", (other,)),
            )
        )
        self.assertEqual(pairing.matches, ())

    def test_adding_a_third_account_needs_no_parsing_change(self) -> None:
        """§5.6: the same code path reconciles two accounts or three."""
        with scratch_dir("third-account") as scratch:
            (scratch / "accounts.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "toleranceDays": 3,
                        "accounts": [
                            {"id": "A", "currency": "EUR", "openingBalance": "0.00"},
                            {"id": "B", "currency": "EUR", "openingBalance": "0.00"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (scratch / "A.csv").write_text(
                "fecha,fecha valor,importe,saldo\n01/01/2026,01/01/2026,-10.00,-10.00\n",
                encoding="utf-8",
                newline="",
            )
            (scratch / "B.csv").write_text(
                "fecha,fecha valor,importe,saldo\n02/01/2026,02/01/2026,10.00,10.00\n",
                encoding="utf-8",
                newline="",
            )
            two = run_accounts(scratch / "accounts.json")
            (scratch / "accounts.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "toleranceDays": 3,
                        "accounts": [
                            {"id": "A", "currency": "EUR", "openingBalance": "0.00"},
                            {"id": "B", "currency": "EUR", "openingBalance": "0.00"},
                            {"id": "C", "currency": "EUR", "openingBalance": "0.00"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (scratch / "C.csv").write_text(
                "fecha,fecha valor,importe,saldo\n03/01/2026,03/01/2026,10.00,10.00\n",
                encoding="utf-8",
                newline="",
            )
            three = run_accounts(scratch / "accounts.json")
        self.assertEqual(len(two.accounts), 2)
        self.assertEqual(len(three.accounts), 3)
        self.assertEqual(two.pairing.matches, three.pairing.matches)


class RegistryTests(unittest.TestCase):
    """§5 registry parsing: defaults, rejections and the mismatch report."""

    def test_tolerance_defaults_to_three(self) -> None:
        """An absent ``toleranceDays`` means 3 (spec §5.2)."""
        registry = parse_registry(
            '{"version": 1, "accounts": [{"id": "A", "currency": "EUR", "openingBalance": "0"}]}'
        )
        self.assertEqual(registry.tolerance_days, 3)

    def test_tolerance_is_read_from_the_file(self) -> None:
        """A present ``toleranceDays`` is used as given."""
        registry = parse_registry(
            '{"version": 1, "toleranceDays": 7,'
            ' "accounts": [{"id": "A", "currency": "EUR", "openingBalance": "0"}]}'
        )
        self.assertEqual(registry.tolerance_days, 7)

    BAD_REGISTRIES: ClassVar[tuple[str, ...]] = (
        "not json at all",
        "[]",
        '{"version": 9, "accounts": []}',
        '{"version": 1}',
        '{"version": 1, "accounts": [], "extra": 1}',
        '{"version": 1, "accounts": []}',
        '{"version": 1, "accounts": [{"id": "A", "currency": "EUR", "openingBalance": "x"}]}',
        '{"version": 1, "accounts": [{"id": "A"}]}',
        '{"version": 1, "accounts": [{"id": "A", "currency": "EUR",'
        ' "openingBalance": "0", "extra": 1}]}',
        '{"version": 1, "accounts": [{"id": "A", "currency": "EUR", "openingBalance": "0"},'
        ' {"id": "A", "currency": "EUR", "openingBalance": "0"}]}',
        '{"version": 1, "accounts": [{"id": "A", "currency": "EUR", "openingBalance": 0}]}',
        '{"version": 1, "toleranceDays": -1,'
        ' "accounts": [{"id": "A", "currency": "EUR", "openingBalance": "0"}]}',
    )

    def test_bad_registries_are_rejected(self) -> None:
        """Malformed registries raise a ``REGISTRY`` error rather than being guessed at."""
        for document in self.BAD_REGISTRIES:
            with self.subTest(document=document):
                with self.assertRaises(RegistryError) as caught:
                    parse_registry(document)
                self.assertEqual(caught.exception.kind, "REGISTRY")

    def test_declared_opening_mismatch_is_reported_by_the_cli(self) -> None:
        """A registry opening balance that contradicts row 1 is surfaced, not swallowed."""
        run = run_accounts(fixture("registry_mismatch.json"))
        payload = run.payload()
        accounts = as_list(member(payload, "accounts"), "accounts")
        first = as_dict(accounts[0], "account")
        applied = as_dict(member(first, "openingBalanceApplied"), "applied")
        mismatch = as_dict(member(applied, "declaredMismatch"), "declaredMismatch")
        self.assertEqual(mismatch["declared"], "999.00")
        self.assertEqual(mismatch["derived"], "100.00")

    def test_local_registry_fixture_detects_its_transfer(self) -> None:
        """The local A/B/C fixture pairs the single genuine transfer."""
        run = run_accounts(fixture("registry.json"))
        self.assertEqual(len(run.pairing.matches), 1)
        match = run.pairing.matches[0]
        self.assertEqual(match.outflow.account_id, "A")
        self.assertEqual(str(match.outflow.amount), "-250.00")
        self.assertEqual(int(match.outflow.line), 3)
        self.assertEqual(match.inflow.account_id, "C")
        self.assertEqual(str(match.inflow.amount), "250.00")
        self.assertEqual(int(match.inflow.line), 2)
        self.assertEqual(match.gap_days, 1)
        self.assertEqual(match.net_effect, Decimal(0))

    def test_net_worth_agrees_on_the_local_registry(self) -> None:
        """Both net-worth computations agree at every observation point."""
        run: AccountsRun = run_accounts(fixture("registry.json"))
        self.assertTrue(run.parity_holds)
        self.assertEqual(run.net_worth_final, Decimal("800.00"))


if __name__ == "__main__":
    unittest.main()
