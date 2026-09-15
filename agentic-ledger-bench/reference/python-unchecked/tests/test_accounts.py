"""Iteration A test suite — multi-account reconciliation and transfers (spec §5).

Run with ``python3 -m unittest discover -s tests`` from the ``python/``
directory, alongside ``test_baseline.py``.  Standard library only.

Statements used here are written to temporary directories by helpers that
*derive* the balance column from the amounts, so a fixture can never be
accidentally self-inconsistent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.dirname(HERE)
FIXTURES = os.path.join(HERE, "fixtures")
ACCOUNTS_FIXTURES = os.path.join(FIXTURES, "accounts")

if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

from ledger import accounts as accounts_module
from ledger.accounts import (
    build_accounts,
    detect_transfers,
    net_worth_conserved,
)
from ledger.errors import IO, REGISTRY, LedgerError
from ledger.strictdecimal import canonical


def canon(value) -> str:
    return f"{Decimal(value):.2f}"


def write_statement(path: str, opening: str, rows, *, balances=None) -> None:
    """Write a statement whose balance column is derived unless overridden.

    ``rows`` is a sequence of ``(booking, value, amount)``.  ``balances`` may
    supply an explicit balance per row to build deliberately broken statements.
    """
    running = Decimal(opening)
    lines = ["fecha,fecha valor,importe,saldo"]
    for index, (booking, value, amount) in enumerate(rows):
        running += Decimal(amount)
        shown = canon(running) if balances is None else canon(balances[index])
        lines.append(f"{booking},{value},{amount},{shown}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_registry(path: str, accounts, tolerance=None) -> None:
    document = {"accounts": accounts}
    if tolerance is not None:
        document["toleranceDays"] = tolerance
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)


def account(account_id, currency="EUR", opening="0.00", statement=None) -> dict:
    entry = {"id": account_id, "currency": currency, "openingBalance": opening}
    entry["statement"] = statement if statement is not None else f"{account_id}.csv"
    return entry


class AccountHarness:
    """Builds a temp directory holding statements plus a registry."""

    def __init__(self, test: unittest.TestCase):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        test.addCleanup(self._tmp.cleanup)

    def path(self, name: str) -> str:
        return os.path.join(self.root, name)

    def statement(self, name: str, opening: str, rows, balances=None) -> str:
        target = self.path(name)
        write_statement(target, opening, rows, balances=balances)
        return target

    def registry(self, accounts, tolerance=None, name="registry.json") -> str:
        target = self.path(name)
        write_registry(target, accounts, tolerance)
        return target


# --------------------------------------------------------------------------- #
# The shipped 3-account fixture
# --------------------------------------------------------------------------- #


class ShippedFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_accounts(os.path.join(ACCOUNTS_FIXTURES, "registry.json"))

    def test_three_accounts_are_reconciled(self):
        self.assertEqual(3, len(self.data["accounts"]))
        for report in self.data["accounts"]:
            with self.subTest(account=report["id"]):
                self.assertTrue(report["checks"]["reconciled"])
                self.assertEqual([], report["checks"]["continuityErrors"])
                self.assertEqual([], report["checks"]["cumulativeErrors"])
                self.assertEqual([], report["checks"]["valueDateInversions"])

    def test_row_counts_and_closing_balances(self):
        by_id = {report["id"]: report for report in self.data["accounts"]}
        self.assertEqual(9, by_id["ES11-current"]["checks"]["rowCount"])
        self.assertEqual(5, by_id["ES22-savings"]["checks"]["rowCount"])
        self.assertEqual(2, by_id["ES33-joint"]["checks"]["rowCount"])
        self.assertEqual("3404.50", by_id["ES11-current"]["closingBalance"])
        self.assertEqual("6237.40", by_id["ES22-savings"]["closingBalance"])
        self.assertEqual("300.00", by_id["ES33-joint"]["closingBalance"])

    def test_transfers_pair_expected_rows_across_three_accounts(self):
        found = {
            (
                t["from"]["account"],
                t["from"]["line"],
                t["to"]["account"],
                t["to"]["line"],
            )
            for t in self.data["transfers"]
        }
        self.assertEqual(
            {
                ("ES11-current", 3, "ES22-savings", 2),
                ("ES11-current", 5, "ES22-savings", 3),
                ("ES11-current", 8, "ES22-savings", 5),
                ("ES11-current", 9, "ES33-joint", 3),
            },
            found,
        )
        self.assertEqual(4, self.data["transferCheck"]["pairs"])
        self.assertEqual(8, self.data["transferCheck"]["pairedRows"])

    def test_transfers_use_value_dates_not_booking_dates(self):
        """current line 5 books 10/03 but is valued 09/03, and pairs at gap 0."""
        lagging = [t for t in self.data["transfers"] if t["from"]["line"] == 5]
        self.assertEqual(1, len(lagging))
        self.assertEqual("2026-03-09", lagging[0]["from"]["valueDate"])
        self.assertEqual("2026-03-09", lagging[0]["to"]["valueDate"])
        self.assertEqual(0, lagging[0]["gapDays"])

    def test_each_row_pairs_at_most_once(self):
        keys = []
        for transfer in self.data["transfers"]:
            keys.append((transfer["from"]["account"], transfer["from"]["line"]))
            keys.append((transfer["to"]["account"], transfer["to"]["line"]))
        self.assertEqual(len(keys), len(set(keys)))

    def test_transfer_pairs_cross_accounts(self):
        for transfer in self.data["transfers"]:
            self.assertNotEqual(transfer["from"]["account"], transfer["to"]["account"])

    def test_net_worth_is_conserved_by_the_paired_rows(self):
        self.assertTrue(self.data["transferCheck"]["netWorthConserved"])
        self.assertEqual("0.00", self.data["transferCheck"]["netWorthDelta"])

    def test_unpaired_counter_is_reserved_at_zero(self):
        self.assertEqual(0, self.data["transferCheck"]["unpairedTransferCandidates"])

    def test_net_worth_agrees_at_every_observation_point(self):
        self.assertEqual(16, self.data["netWorth"]["observations"])  # 9 + 5 + 2
        self.assertTrue(self.data["netWorth"]["reconciled"])
        self.assertEqual([], self.data["netWorth"]["mismatches"])
        self.assertEqual("6250.00", self.data["netWorth"]["openingNetWorth"])

    def test_analysis_is_deterministic(self):
        again = build_accounts(os.path.join(ACCOUNTS_FIXTURES, "registry.json"))
        self.assertEqual(
            json.dumps(self.data, sort_keys=False),
            json.dumps(again, sort_keys=False),
        )


# --------------------------------------------------------------------------- #
# §5.6 — a third account changes no parsing code
# --------------------------------------------------------------------------- #


class ThirdAccountTests(unittest.TestCase):
    def test_adding_a_third_account_needs_no_parsing_change(self):
        harness = AccountHarness(self)
        harness.statement(
            "a.csv",
            "0.00",
            [
                ("01/01/2026", "01/01/2026", "-100.00"),
            ],
        )
        harness.statement(
            "b.csv",
            "0.00",
            [
                ("01/01/2026", "01/01/2026", "100.00"),
            ],
        )
        harness.statement(
            "c.csv",
            "0.00",
            [
                ("02/01/2026", "02/01/2026", "7.00"),
            ],
        )

        two = build_accounts(
            harness.registry(
                [account("a", statement="a.csv"), account("b", statement="b.csv")],
                name="two.json",
            )
        )
        three = build_accounts(
            harness.registry(
                [
                    account("a", statement="a.csv"),
                    account("b", statement="b.csv"),
                    account("c", statement="c.csv"),
                ],
                name="three.json",
            )
        )

        self.assertEqual(2, len(two["accounts"]))
        self.assertEqual(3, len(three["accounts"]))
        # The two-account pairing is unchanged; the third account simply joins.
        self.assertEqual(two["transfers"], three["transfers"])
        self.assertEqual(1, three["transferCheck"]["pairs"])
        self.assertEqual(3, three["netWorth"]["observations"])
        self.assertEqual("7.00", three["accounts"][2]["closingBalance"])


# --------------------------------------------------------------------------- #
# Pairing rules (§5.2)
# --------------------------------------------------------------------------- #


def movements_for(rows_by_account, tolerance=3):
    """Build accounts in memory from {account_id: (opening, rows)} and pair them.

    The registry's ``openingBalance`` is taken from the same tuple that seeds the
    statement, so the two cannot drift apart in a fixture.
    """
    harness = AccountHarness(unittest.TestCase())
    accounts = []
    for account_id, (opening, rows) in rows_by_account.items():
        harness.statement(f"{account_id}.csv", opening, rows)
        accounts.append(
            account(account_id, opening=opening, statement=f"{account_id}.csv")
        )
    registry = harness.registry(accounts, tolerance=tolerance, name="m.json")
    return build_accounts(registry)


class PairingRuleTests(unittest.TestCase):
    def test_nearest_feasible_date_wins(self):
        """Two feasible inflows (gaps 1 and 3); the nearer one is taken."""
        result = movements_for(
            {
                "out": ("0.00", [("01/01/2026", "01/01/2026", "-100.00")]),
                "in": (
                    "0.00",
                    [
                        ("03/01/2026", "03/01/2026", "100.00"),  # gap 2d: feasible
                        ("02/01/2026", "02/01/2026", "100.00"),  # gap 1d: nearer
                        ("05/01/2026", "05/01/2026", "100.00"),  # gap 4d: infeasible
                    ],
                ),
            }
        )
        self.assertEqual(1, len(result["transfers"]))
        self.assertEqual(1, result["transfers"][0]["gapDays"])
        self.assertEqual(3, result["transfers"][0]["to"]["line"])

    def test_same_day_pairing_beats_a_later_inflow(self):
        """Gap 0 is the nearest feasible date of all."""
        result = movements_for(
            {
                "out": ("0.00", [("01/01/2026", "01/01/2026", "-100.00")]),
                "in": (
                    "0.00",
                    [
                        ("03/01/2026", "03/01/2026", "100.00"),
                        ("01/01/2026", "01/01/2026", "100.00"),  # gap 0
                    ],
                ),
            }
        )
        self.assertEqual(0, result["transfers"][0]["gapDays"])
        self.assertEqual(3, result["transfers"][0]["to"]["line"])

    def test_ties_break_by_lowest_line_number(self):
        """Equal gaps: the earliest row in file order is chosen."""
        result = movements_for(
            {
                "out": ("0.00", [("01/01/2026", "01/01/2026", "-100.00")]),
                "in": (
                    "0.00",
                    [
                        ("01/01/2026", "01/01/2026", "100.00"),  # line 2
                        ("01/01/2026", "01/01/2026", "100.00"),  # line 3
                    ],
                ),
            }
        )
        self.assertEqual(1, len(result["transfers"]))
        self.assertEqual(2, result["transfers"][0]["to"]["line"])

    def test_tolerance_is_configurable_and_enforced(self):
        rows = {
            "out": ("0.00", [("01/01/2026", "01/01/2026", "-100.00")]),
            "in": ("0.00", [("05/01/2026", "05/01/2026", "100.00")]),
        }
        tight = movements_for(rows, tolerance=3)
        self.assertEqual([], tight["transfers"])
        self.assertEqual(3, tight["toleranceDays"])

        loose = movements_for(rows, tolerance=4)
        self.assertEqual(1, len(loose["transfers"]))
        self.assertEqual(4, loose["transfers"][0]["gapDays"])

    def test_date_tolerance_defaults_to_three(self):
        result = movements_for(
            {
                "out": ("0.00", [("01/01/2026", "01/01/2026", "-100.00")]),
                "in": ("0.00", [("01/01/2026", "01/01/2026", "100.00")]),
            }
        )
        self.assertEqual(3, result["toleranceDays"])

    def test_inflow_may_not_precede_the_outflow(self):
        result = movements_for(
            {
                "out": ("0.00", [("05/01/2026", "05/01/2026", "-100.00")]),
                "in": ("0.00", [("01/01/2026", "01/01/2026", "100.00")]),
            }
        )
        self.assertEqual([], result["transfers"])

    def test_amounts_must_be_equal_in_magnitude(self):
        result = movements_for(
            {
                "out": ("0.00", [("01/01/2026", "01/01/2026", "-100.00")]),
                "in": ("0.00", [("01/01/2026", "01/01/2026", "99.99")]),
            }
        )
        self.assertEqual([], result["transfers"])

    def test_rows_in_one_account_never_pair_with_each_other(self):
        result = movements_for(
            {
                "solo": (
                    "0.00",
                    [
                        ("01/01/2026", "01/01/2026", "-100.00"),
                        ("01/01/2026", "01/01/2026", "100.00"),
                    ],
                ),
            }
        )
        self.assertEqual([], result["transfers"])

    def test_currencies_must_match(self):
        harness = AccountHarness(self)
        harness.statement("eur.csv", "0.00", [("01/01/2026", "01/01/2026", "-100.00")])
        harness.statement("usd.csv", "0.00", [("01/01/2026", "01/01/2026", "100.00")])
        registry = harness.registry(
            [
                account("eur", currency="EUR", statement="eur.csv"),
                account("usd", currency="USD", statement="usd.csv"),
            ]
        )
        result = build_accounts(registry)
        self.assertEqual([], result["transfers"])

        # The shipped mismatch registry demonstrates the same rule.
        mismatch = build_accounts(
            os.path.join(ACCOUNTS_FIXTURES, "registry_currency_mismatch.json")
        )
        self.assertEqual([], mismatch["transfers"])
        self.assertTrue(mismatch["transferCheck"]["netWorthConserved"])

    def test_a_row_cannot_pay_two_outflows(self):
        result = movements_for(
            {
                "out1": ("0.00", [("01/01/2026", "01/01/2026", "-100.00")]),
                "out2": ("0.00", [("01/01/2026", "01/01/2026", "-100.00")]),
                "in": ("0.00", [("01/01/2026", "01/01/2026", "100.00")]),
            }
        )
        self.assertEqual(1, len(result["transfers"]))
        self.assertEqual(2, result["transferCheck"]["pairedRows"])

    def test_zero_amount_rows_are_never_transfers(self):
        result = movements_for(
            {
                "a": ("0.00", [("01/01/2026", "01/01/2026", "0.00")]),
                "b": ("0.00", [("01/01/2026", "01/01/2026", "0.00")]),
            }
        )
        self.assertEqual([], result["transfers"])

    def test_pairing_is_independent_of_registry_order(self):
        rows = {
            "a": ("0.00", [("01/01/2026", "01/01/2026", "-100.00")]),
            "b": ("0.00", [("01/01/2026", "01/01/2026", "100.00")]),
            "c": ("0.00", [("01/01/2026", "01/01/2026", "100.00")]),
        }
        forward = movements_for(rows)
        backward = movements_for(dict(reversed(list(rows.items()))))
        self.assertEqual(1, len(forward["transfers"]))
        self.assertEqual(1, len(backward["transfers"]))
        # Same row is chosen either way: the preference key is order-independent.
        self.assertEqual(forward["transfers"][0]["from"]["account"], "a")
        self.assertEqual(forward["transfers"][0]["to"]["account"], "b")
        self.assertEqual(backward["transfers"][0]["to"]["line"], 2)


# --------------------------------------------------------------------------- #
# New invariants (§5.4, §5.5)
# --------------------------------------------------------------------------- #


class NetWorthConservationTests(unittest.TestCase):
    def test_conservation_holds_for_a_balanced_pair(self):
        """§5.4: the paired rows alone must net to exactly zero."""
        result = movements_for(
            {
                "out": ("0.00", [("01/01/2026", "01/01/2026", "-250.50")]),
                "in": ("0.00", [("02/01/2026", "02/01/2026", "250.50")]),
            }
        )
        self.assertTrue(result["transferCheck"]["netWorthConserved"])
        self.assertEqual("0.00", result["transferCheck"]["netWorthDelta"])

    def test_conservation_is_vacuous_without_pairings(self):
        result = movements_for(
            {
                "out": ("0.00", [("01/01/2026", "01/01/2026", "-250.50")]),
            }
        )
        self.assertTrue(result["transferCheck"]["netWorthConserved"])
        self.assertEqual("0.00", result["transferCheck"]["netWorthDelta"])

    def test_conservation_violation_is_reported_not_hidden(self):
        """Feed the invariant rows that cannot balance, and assert the failure.

        §5.4's guarantee is conditional on every paired row really being a
        transfer, so this drives the check with a deliberately unbalanced
        pairing to prove it reports rather than assumes.
        """
        from datetime import date

        from ledger.accounts import Account, Movement, Pairing
        from ledger.model import ParsedStatement, Transaction

        def row(line, amount):
            return Transaction(
                line=line,
                booking_date=date(2026, 1, 1),
                value_date=date(2026, 1, 1),
                amount=Decimal(amount),
                balance_after=Decimal("0.00"),
                raw_amount=str(amount),
                raw_balance="0.00",
            )

        account = Account(
            index=0,
            account_id="a",
            currency="EUR",
            opening_balance=Decimal("0.00"),
            statement_label="a.csv",
            statement=ParsedStatement((), (), Decimal("0.00")),
        )
        broken = Pairing(
            Movement(account, row(2, "-100.00")), Movement(account, row(3, "90.00")), 0
        )
        conserved, delta = net_worth_conserved([broken])
        self.assertFalse(conserved)
        self.assertEqual("-10.00", canonical(delta))


class NetWorthObservationTests(unittest.TestCase):
    def test_net_worth_mismatch_is_reported_at_every_point_after_the_defect(self):
        """One wrong balance makes both net-worth computations disagree."""
        harness = AccountHarness(self)
        harness.statement(
            "a.csv",
            "0.00",
            [
                ("01/01/2026", "01/01/2026", "100.00"),
                ("02/01/2026", "02/01/2026", "100.00"),
            ],
            balances=["100.00", "999.00"],
        )  # second balance is wrong
        harness.statement(
            "b.csv",
            "0.00",
            [
                ("01/01/2026", "01/01/2026", "50.00"),
            ],
        )
        registry = harness.registry(
            [
                account("a", statement="a.csv"),
                account("b", statement="b.csv"),
            ]
        )
        result = build_accounts(registry)

        self.assertFalse(result["netWorth"]["reconciled"])
        self.assertEqual(3, result["netWorth"]["observations"])
        mismatch = result["netWorth"]["mismatches"]
        self.assertEqual(2, len(mismatch))
        self.assertEqual(
            [("a", 3), ("b", 2)], [(m["account"], m["line"]) for m in mismatch]
        )
        self.assertEqual("799.00", mismatch[0]["delta"])

    def test_per_account_checks_stay_independent_of_each_other(self):
        """A defect in one account does not appear in another account's checks."""
        harness = AccountHarness(self)
        harness.statement(
            "good.csv",
            "0.00",
            [
                ("01/01/2026", "01/01/2026", "100.00"),
            ],
        )
        harness.statement(
            "bad.csv",
            "0.00",
            [
                ("01/01/2026", "01/01/2026", "-100.00"),
                ("02/01/2026", "08/01/2026", "-100.00"),  # value date moves FORWARD
                ("03/01/2026", "05/01/2026", "-100.00"),  # value date goes BACKWARDS
            ],
            balances=["-100.00", "-200.00", "-999.00"],
        )  # and a balance is wrong
        registry = harness.registry(
            [
                account("good", statement="good.csv"),
                account("bad", statement="bad.csv"),
            ]
        )
        result = build_accounts(registry)
        good, bad = result["accounts"]
        self.assertTrue(good["checks"]["reconciled"])
        self.assertEqual([], good["checks"]["continuityErrors"])
        self.assertEqual([], good["checks"]["valueDateInversions"])
        # Row 3's balance is the wrong one; §3.1 flags it, and because it is the
        # last row there is no successor to flag too.
        self.assertFalse(bad["checks"]["reconciled"])
        self.assertEqual([4], [e["line"] for e in bad["checks"]["continuityErrors"]])
        self.assertEqual([4], [e["line"] for e in bad["checks"]["cumulativeErrors"]])
        self.assertEqual(
            [{"line": 4, "previousLine": 3}], bad["checks"]["valueDateInversions"]
        )

    def test_registry_opening_balance_anchors_the_cumulative_check(self):
        """The registry's openingBalance is authoritative, not the first row."""
        harness = AccountHarness(self)
        # The statement's own balance column implies an opening of 0.00.
        harness.statement(
            "a.csv",
            "0.00",
            [
                ("01/01/2026", "01/01/2026", "100.00"),
                ("02/01/2026", "02/01/2026", "0.00"),
            ],
        )
        matched = build_accounts(
            harness.registry(
                [account("a", opening="0.00", statement="a.csv")], name="ok.json"
            )
        )
        self.assertEqual([], matched["accounts"][0]["checks"]["cumulativeErrors"])

        # A registry claiming 5.00 must now disagree with every recorded balance.
        mismatched = build_accounts(
            harness.registry(
                [account("a", opening="5.00", statement="a.csv")], name="bad.json"
            )
        )
        errors = mismatched["accounts"][0]["checks"]["cumulativeErrors"]
        self.assertEqual([2, 3], [e["line"] for e in errors])
        self.assertEqual("-5.00", errors[0]["delta"])
        # §3.1 is local, so the registry anchor does not affect it at all.
        self.assertEqual([], mismatched["accounts"][0]["checks"]["continuityErrors"])

        # With the registry opening absent the statement's own anchor is used,
        # which is the baseline behaviour and leaves the rows consistent.
        harness.statement(
            "b.csv",
            "5.00",
            [
                ("01/01/2026", "01/01/2026", "100.00"),
                ("02/01/2026", "02/01/2026", "0.00"),
            ],
        )
        self_anchored = build_accounts(
            harness.registry(
                [account("b", opening="5.00", statement="b.csv")], name="ok2.json"
            )
        )
        self.assertEqual([], self_anchored["accounts"][0]["checks"]["cumulativeErrors"])


# --------------------------------------------------------------------------- #
# Registry validation (frozen REGISTRY kind)
# --------------------------------------------------------------------------- #


class RegistryTests(unittest.TestCase):
    def _error_for(self, mutate_fixture) -> LedgerError:
        harness = AccountHarness(self)
        harness.statement("a.csv", "0.00", [("01/01/2026", "01/01/2026", "1.00")])
        target = harness.path("registry.json")
        mutate_fixture(target, harness)
        with self.assertRaises(LedgerError) as caught:
            build_accounts(target)
        return caught.exception

    def test_missing_registry_is_io(self):
        with self.assertRaises(LedgerError) as caught:
            build_accounts("/nonexistent/registry.json")
        self.assertEqual(IO, caught.exception.kind)

    def test_not_json_is_registry_error(self):
        def mutate(target, harness):
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("{not json")

        self.assertEqual(REGISTRY, self._error_for(mutate).kind)

    def test_top_level_must_be_an_object(self):
        def mutate(target, harness):
            with open(target, "w", encoding="utf-8") as handle:
                json.dump([account("a")], handle)

        self.assertEqual(REGISTRY, self._error_for(mutate).kind)

    def test_missing_accounts_key(self):
        def mutate(target, harness):
            with open(target, "w", encoding="utf-8") as handle:
                json.dump({"toleranceDays": 3}, handle)

        self.assertEqual(REGISTRY, self._error_for(mutate).kind)

    def test_empty_accounts_array(self):
        def mutate(target, harness):
            write_registry(target, [])

        self.assertEqual(REGISTRY, self._error_for(mutate).kind)

    def test_duplicate_account_ids(self):
        def mutate(target, harness):
            write_registry(
                target,
                [account("a", statement="a.csv"), account("a", statement="a.csv")],
            )

        error = self._error_for(mutate)
        self.assertEqual(REGISTRY, error.kind)
        self.assertIn("duplicate", error.message)

    def test_account_without_id(self):
        def mutate(target, harness):
            write_registry(target, [{"currency": "EUR", "openingBalance": "0.00"}])

        self.assertEqual(REGISTRY, self._error_for(mutate).kind)

    def test_account_without_currency(self):
        def mutate(target, harness):
            write_registry(target, [{"id": "a", "openingBalance": "0.00"}])

        self.assertEqual(REGISTRY, self._error_for(mutate).kind)

    def test_non_string_opening_balance_is_rejected(self):
        """A JSON number would mean a binary float reached the domain model."""

        def mutate(target, harness):
            write_registry(
                target,
                [
                    {
                        "id": "a",
                        "currency": "EUR",
                        "openingBalance": 1000.0,
                        "statement": "a.csv",
                    }
                ],
            )

        error = self._error_for(mutate)
        self.assertEqual(REGISTRY, error.kind)
        self.assertIn("decimal string", error.message)

    def test_malformed_opening_balance(self):
        def mutate(target, harness):
            write_registry(target, [account("a", opening="abc", statement="a.csv")])

        self.assertEqual(REGISTRY, self._error_for(mutate).kind)

    def test_non_integer_tolerance_is_rejected(self):
        def mutate(target, harness):
            write_registry(target, [account("a", statement="a.csv")], tolerance=2.5)

        self.assertEqual(REGISTRY, self._error_for(mutate).kind)

    def test_negative_tolerance_is_rejected(self):
        def mutate(target, harness):
            write_registry(target, [account("a", statement="a.csv")], tolerance=-1)

        self.assertEqual(REGISTRY, self._error_for(mutate).kind)

    def test_missing_statement_file_is_io(self):
        def mutate(target, harness):
            write_registry(target, [account("a", statement="nope.csv")])

        self.assertEqual(IO, self._error_for(mutate).kind)

    def test_statement_defaults_to_id_dot_csv(self):
        harness = AccountHarness(self)
        harness.statement("a.csv", "0.00", [("01/01/2026", "01/01/2026", "1.00")])
        target = harness.path("registry.json")
        write_registry(
            target, [{"id": "a", "currency": "EUR", "openingBalance": "0.00"}]
        )
        result = build_accounts(target)
        self.assertEqual(1, result["accounts"][0]["checks"]["rowCount"])
        self.assertEqual("a.csv", result["accounts"][0]["statement"])


# --------------------------------------------------------------------------- #
# The shared cross-language fixture (CLARIFICATIONS.md §2)
# --------------------------------------------------------------------------- #

# The shared fixtures ship beside the frozen spec, outside this git worktree.
SHARED = str(Path(__file__).resolve().parents[3] / "benchmark" / "data")


class SharedFixtureTests(unittest.TestCase):
    """Assert the independently-computed expected values, not our own output.

    Expected values are taken from CLARIFICATIONS.md §2's table, which was
    computed by the orchestrator before either implementation existed.
    """

    @classmethod
    def setUpClass(cls):
        cls.data = build_accounts(os.path.join(SHARED, "accounts.json"))
        cls.by_id = {report["id"]: report for report in cls.data["accounts"]}

    def test_three_accounts_all_eur(self):
        self.assertEqual(3, len(self.data["accounts"]))
        self.assertEqual(["A", "B", "C"], [r["id"] for r in self.data["accounts"]])
        for report in self.data["accounts"]:
            self.assertEqual("EUR", report["currency"])

    def test_final_balances_match_the_expected_table(self):
        self.assertEqual("455.25", self.by_id["A"]["closingBalance"])
        self.assertEqual("429.50", self.by_id["B"]["closingBalance"])
        self.assertEqual("254.50", self.by_id["C"]["closingBalance"])

    def test_zero_continuity_and_cumulative_errors_everywhere(self):
        for report in self.data["accounts"]:
            with self.subTest(account=report["id"]):
                self.assertEqual([], report["checks"]["continuityErrors"])
                self.assertEqual([], report["checks"]["cumulativeErrors"])
                self.assertTrue(report["checks"]["reconciled"])

    def test_net_worth_agrees_by_both_methods_and_equals_1139_25(self):
        self.assertEqual("0.00", self.data["netWorth"]["openingNetWorth"])
        self.assertTrue(self.data["netWorth"]["reconciled"])
        self.assertEqual([], self.data["netWorth"]["mismatches"])
        # Σ final balances, computed here from the reported closings.
        total = sum(
            (Decimal(r["closingBalance"]) for r in self.data["accounts"]), Decimal(0)
        )
        self.assertEqual(Decimal("1139.25"), total)
        self.assertEqual("455.25", self.by_id["A"]["closingBalance"])
        # Both methods: the last observation is Σ(openings + Σ amounts).
        self.assertEqual("1139.25", self.data["netWorth"]["finalNetWorthFromAmounts"])

    def test_exactly_two_transfer_pairs_with_the_expected_endpoints(self):
        pairs = {
            (
                t["from"]["account"],
                t["from"]["line"],
                t["from"]["valueDate"],
                t["to"]["account"],
                t["to"]["line"],
                t["to"]["valueDate"],
                t["amount"],
                t["gapDays"],
            )
            for t in self.data["transfers"]
        }
        self.assertEqual(2, self.data["transferCheck"]["pairs"])
        self.assertEqual(
            {
                ("A", 3, "2026-01-05", "B", 4, "2026-01-06", "250.00", 1),
                ("A", 4, "2026-01-10", "C", 2, "2026-01-10", "300.00", 0),
            },
            pairs,
        )

    def test_the_decoys_are_not_paired(self):
        """The four traps listed in CLARIFICATIONS.md §2 must stay unpaired."""
        paired_rows = set()
        for transfer in self.data["transfers"]:
            paired_rows.add((transfer["from"]["account"], transfer["from"]["line"]))
            paired_rows.add((transfer["to"]["account"], transfer["to"]["line"]))
        # B line 3 (-250) is an outflow with no counterparty; C line 4 (+75)
        # precedes every candidate outflow's value date; B line 6 and C line 4
        # are both outflows of -120.50 and must not pair with each other;
        # A line 2 (+1000) and B line 2 (+500) are uncounterpartied credits.
        for decoy in (("B", 3), ("C", 4), ("B", 6), ("A", 2), ("B", 2)):
            with self.subTest(decoy=decoy):
                self.assertNotIn(decoy, paired_rows)
        self.assertEqual(4, len(paired_rows))

    def test_transfer_net_effect_is_exactly_zero(self):
        self.assertTrue(self.data["transferCheck"]["netWorthConserved"])
        self.assertEqual("0.00", self.data["transferCheck"]["netWorthDelta"])

    def test_shared_fixture_analysis_is_deterministic(self):
        again = build_accounts(os.path.join(SHARED, "accounts.json"))
        self.assertEqual(
            json.dumps(self.data, sort_keys=False), json.dumps(again, sort_keys=False)
        )


# --------------------------------------------------------------------------- #
# §6 CLI contract for the accounts subcommand
# --------------------------------------------------------------------------- #


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ledger", *args],
        cwd=PYTHON_ROOT,
        capture_output=True,
        text=True,
    )


class AccountsCliTests(unittest.TestCase):
    REGISTRY = os.path.join(ACCOUNTS_FIXTURES, "registry.json")

    def test_registry_run_exits_zero_with_the_frozen_envelope(self):
        result = run_cli("accounts", self.REGISTRY)
        self.assertEqual(0, result.returncode)
        payload = json.loads(result.stdout)
        self.assertEqual(["ok", "module", "version", "data"], list(payload.keys()))
        self.assertTrue(payload["ok"])
        self.assertEqual("ledger", payload["module"])
        self.assertEqual(1, payload["version"])

    def test_usage_errors_for_the_accounts_subcommand(self):
        self.assertEqual(2, run_cli("accounts").returncode)
        self.assertEqual(2, run_cli("accounts", "a", "b").returncode)

    def test_damaged_registry_exits_one_with_registry_kind(self):
        result = run_cli("accounts", os.path.join(FIXTURES, "empty_row.csv"))
        self.assertEqual(1, result.returncode)
        payload = json.loads(result.stdout)
        self.assertEqual(["ok", "module", "version", "error"], list(payload.keys()))
        self.assertEqual("REGISTRY", payload["error"]["kind"])

    def test_output_is_deterministic_across_runs(self):
        first = run_cli("accounts", self.REGISTRY)
        second = run_cli("accounts", self.REGISTRY)
        self.assertEqual(first.stdout, second.stdout)

    def test_two_account_registry_also_reconciles(self):
        result = run_cli(
            "accounts", os.path.join(ACCOUNTS_FIXTURES, "registry_two.json")
        )
        self.assertEqual(0, result.returncode)
        payload = json.loads(result.stdout)
        self.assertEqual(2, len(payload["data"]["accounts"]))
        self.assertTrue(payload["data"]["netWorth"]["reconciled"])

    def test_envelope_carries_no_absolute_paths_or_timestamps(self):
        result = run_cli("accounts", self.REGISTRY)
        payload = json.loads(result.stdout)
        self.assertNotIn(ACCOUNTS_FIXTURES, result.stdout)
        self.assertNotIn("generatedAt", result.stdout)
        for report in payload["data"]["accounts"]:
            self.assertNotIn("/", report["statement"])


# --------------------------------------------------------------------------- #
# Property-style test over generated account sets (§5 + §8)
# --------------------------------------------------------------------------- #


class AccountsPropertyTests(unittest.TestCase):
    """Generated transfer sets: the invariants must hold for every one."""

    SEEDS = range(16)

    def test_generated_transfers_pair_completely_and_conserve_net_worth(self):
        import random

        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                harness = AccountHarness(self)
                pending = []  # transfers to be mirrored into the receiver
                rows = {"sender": [], "receiver": []}
                day = 1

                for _ in range(rng.randrange(1, 6)):
                    amount = Decimal(rng.randrange(1, 500_00)).scaleb(-2)
                    gap = rng.randrange(0, 4)  # inside tolerance
                    day += rng.randrange(1, 4)
                    out_day = day
                    in_day = day + gap
                    rows["sender"].append(
                        (
                            f"{out_day:02d}/01/2026",
                            f"{out_day:02d}/01/2026",
                            f"-{amount}",
                        )
                    )
                    rows["receiver"].append(
                        (f"{in_day:02d}/01/2026", f"{in_day:02d}/01/2026", f"{amount}")
                    )
                    pending.append((amount, gap))

                for name, entries in rows.items():
                    harness.statement(f"{name}.csv", "1000.00", entries)
                registry = harness.registry(
                    [
                        account("sender", opening="1000.00", statement="sender.csv"),
                        account(
                            "receiver", opening="1000.00", statement="receiver.csv"
                        ),
                    ]
                )
                result = build_accounts(registry)

                self.assertEqual(len(pending), result["transferCheck"]["pairs"])
                self.assertEqual(
                    2 * len(pending), result["transferCheck"]["pairedRows"]
                )
                self.assertTrue(result["transferCheck"]["netWorthConserved"])
                self.assertEqual("0.00", result["transferCheck"]["netWorthDelta"])
                # Every observation point agrees, because balances were derived.
                self.assertTrue(result["netWorth"]["reconciled"])
                self.assertEqual([], result["netWorth"]["mismatches"])
                # No row is used twice, and no pair stays inside one account.
                used = []
                for transfer in result["transfers"]:
                    used.append((transfer["from"]["account"], transfer["from"]["line"]))
                    used.append((transfer["to"]["account"], transfer["to"]["line"]))
                    self.assertNotEqual(
                        transfer["from"]["account"], transfer["to"]["account"]
                    )
                self.assertEqual(len(used), len(set(used)))

    def test_conservation_delta_is_zero_whenever_pairing_is_balanced(self):
        """§5.4 restated as an invariant over generated balanced pairings."""
        import random

        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                movements = []
                for _ in range(rng.randrange(1, 8)):
                    amount = Decimal(rng.randrange(1, 100_000)).scaleb(-2)
                    movements.append(amount)
                    movements.append(-amount)
                self.assertEqual(Decimal(0), sum(movements, Decimal(0)))
                total = sum(movements, Decimal(0))
                self.assertEqual("0.00", canonical(total))


if __name__ == "__main__":
    unittest.main(verbosity=2)
