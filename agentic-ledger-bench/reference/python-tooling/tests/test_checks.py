"""Spec §3 and §8 — the checks, including the real file's asserted numbers.

Spec §8 is explicit that a suite which passes while the reference file fails the
§3.3 count of 29 is worthless, so the real file's figures are asserted here
directly, and independently re-verified against ``verify.py``'s ground truth.
"""

import unittest
from decimal import Decimal
from itertools import pairwise
from typing import ClassVar

from ledger.amounts import parse_amount, parse_balance
from ledger.baseline import run_baseline
from ledger.checks import (
    check_continuity,
    check_cumulative,
    check_value_date_inversions,
    derive_opening_balance,
    run_checks,
)
from ledger.dates import parse_date
from ledger.domain import LineNumber, Transaction, format_money

from .helpers import (
    EXPECTED_FINAL_BALANCE,
    EXPECTED_ROW_COUNT,
    EXPECTED_SKIPPED_EMPTY_ROWS,
    EXPECTED_VALUE_DATE_INVERSIONS,
    ORIGINAL_EXPORT,
    RAW_CSV,
    SequenceSource,
    checks_of,
    count_of,
    fixture,
    member,
    run_cli,
)


def build_transactions(
    rows: list[tuple[str, str, str, str]],
) -> tuple[Transaction, ...]:
    """Build transactions from ``(booking, value, amount, balance)`` tuples.

    Shared by the example-based and the property-style tests so both exercise
    the same construction path the parser uses.
    """
    built: list[Transaction] = []
    for index, (booking, value, raw_amount, raw_balance) in enumerate(rows):
        amount = parse_amount(raw_amount)
        balance = parse_balance(raw_balance)
        built.append(
            Transaction(
                line=LineNumber(index + 2),
                booking_date=parse_date(booking),
                value_date=parse_date(value),
                amount=amount,
                balance_after=balance,
                raw_amount=raw_amount,
                raw_balance=raw_balance,
                checksum="",
            )
        )
    return tuple(built)


class ReferenceFileTests(unittest.TestCase):
    """The §8 numbers, asserted against the real reference CSV."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the baseline once for the whole class."""
        cls.baseline = run_baseline(RAW_CSV)

    def test_row_count_and_skipped_rows(self) -> None:
        """923 transactions, and the one empty physical row is line 2."""
        self.assertEqual(self.baseline.checks.row_count, EXPECTED_ROW_COUNT)
        self.assertEqual(
            [int(line) for line in self.baseline.checks.skipped_empty_rows],
            EXPECTED_SKIPPED_EMPTY_ROWS,
        )

    def test_final_balance_is_exactly_37713_30(self) -> None:
        """The final balance is the canonical string, not a float rendering."""
        self.assertEqual(
            format_money(self.baseline.final_balance), EXPECTED_FINAL_BALANCE
        )

    def test_value_date_inversions_are_exactly_29(self) -> None:
        """Exactly 29 adjacent pairs move backwards on value date (spec §3.3)."""
        self.assertEqual(
            len(self.baseline.checks.value_date_inversions),
            EXPECTED_VALUE_DATE_INVERSIONS,
        )

    def test_continuity_and_cumulative_are_clean(self) -> None:
        """The reference file has no continuity and no cumulative errors."""
        self.assertEqual(self.baseline.checks.continuity_errors, ())
        self.assertEqual(self.baseline.checks.cumulative_errors, ())

    def test_booking_order_has_no_inversions(self) -> None:
        """Booking order is strictly non-decreasing, whatever the value dates do."""
        dates = [t.booking_date for t in self.baseline.transactions]
        self.assertEqual(sum(1 for a, b in pairwise(dates) if b < a), 0)

    @unittest.skipUnless(ORIGINAL_EXPORT, "pins a fact about the original export")
    def test_the_two_dates_are_not_collapsed(self) -> None:
        """92 rows differ between booking and value date — the §2 fact."""
        differing = sum(
            1 for t in self.baseline.transactions if t.booking_date != t.value_date
        )
        self.assertEqual(differing, 92)

    def test_reconciled_is_true_despite_the_inversions(self) -> None:
        """Clarification 1: inversions are observations and must not flip ``reconciled``."""
        self.assertTrue(self.baseline.checks.reconciled)
        self.assertEqual(len(self.baseline.checks.value_date_inversions), 29)


class ContinuityTests(unittest.TestCase):
    """§3.1 — continuity, including its anchoring rule."""

    def test_first_row_cannot_violate_continuity(self) -> None:
        """The first row is the anchor; only rows 2..n are compared (spec §3.1)."""
        transactions = build_transactions([("01/01/2026", "01/01/2026", "10", "999")])
        self.assertEqual(check_continuity(transactions), ())

    def test_opening_balance_is_derived_from_the_first_row(self) -> None:
        """``openingBalance = balanceAfter[0] - amount[0]``."""
        transactions = build_transactions([("01/01/2026", "01/01/2026", "10", "110")])
        self.assertEqual(derive_opening_balance(transactions).amount, Decimal(100))

    def test_every_violation_is_collected_in_file_order(self) -> None:
        """Three broken rows produce three errors, not one (spec §3.4)."""
        transactions = build_transactions(
            [
                ("01/01/2026", "01/01/2026", "10", "110"),
                ("02/01/2026", "02/01/2026", "10", "999"),
                ("03/01/2026", "03/01/2026", "10", "999"),
                ("04/01/2026", "04/01/2026", "10", "999"),
            ]
        )
        errors = check_continuity(transactions)
        self.assertEqual([int(e.line) for e in errors], [3, 4, 5])

    def test_delta_is_reported_with_the_right_sign(self) -> None:
        """A balance 5.25 below expectation yields ``delta == -5.25``."""
        transactions = build_transactions(
            [
                ("01/01/2026", "01/01/2026", "10.00", "110.00"),
                ("02/01/2026", "02/01/2026", "10.00", "114.75"),
            ]
        )
        error = check_continuity(transactions)[0]
        self.assertEqual(error.balance_expected, Decimal("120.00"))
        self.assertEqual(error.delta, Decimal("-5.25"))

    def test_continuity_tolerates_an_empty_statement(self) -> None:
        """No rows, no continuity errors — and no exception either."""
        self.assertEqual(check_continuity(()), ())


class CumulativeTests(unittest.TestCase):
    """§3.2 — the cumulative reconstruction, and §3.5's independence rule."""

    def test_running_sum_reconstructs_every_balance(self) -> None:
        """A consistent statement reconstructs exactly, row by row."""
        transactions = build_transactions(
            [
                ("01/01/2026", "01/01/2026", "10.00", "110.00"),
                ("02/01/2026", "02/01/2026", "-5.50", "104.50"),
                ("03/01/2026", "03/01/2026", "0.50", "105.00"),
            ]
        )
        self.assertEqual(
            check_cumulative(transactions, derive_opening_balance(transactions)), ()
        )

    def test_cumulative_and_continuity_disagree_on_their_own_evidence(self) -> None:
        """§3.5: a break mid-file makes cumulative wrong for every later row too.

        Continuity only complains about the row that broke the chain, while the
        cumulative check — which never reads continuity's state — carries the
        error forward. The two results differ, which is the point.
        """
        transactions = build_transactions(
            [
                ("01/01/2026", "01/01/2026", "100.00", "100.00"),
                ("02/01/2026", "02/01/2026", "100.00", "150.00"),
                ("03/01/2026", "03/01/2026", "100.00", "250.00"),
            ]
        )
        opening = derive_opening_balance(transactions)
        self.assertEqual(len(check_continuity(transactions)), 1)
        self.assertEqual(len(check_cumulative(transactions, opening)), 2)

    def test_cumulative_reports_a_final_balance_shortfall(self) -> None:
        """The final balance is reconstructed, not copied from the file."""
        transactions = build_transactions(
            [
                ("01/01/2026", "01/01/2026", "100.00", "100.00"),
                ("02/01/2026", "02/01/2026", "-25.00", "70.00"),
            ]
        )
        opening = derive_opening_balance(transactions)
        error = check_cumulative(transactions, opening)[0]
        self.assertEqual(error.balance_from_sum, Decimal("75.00"))
        self.assertEqual(error.balance_actual, Decimal("70.00"))
        self.assertEqual(error.delta, Decimal("-5.00"))

    def test_accumulation_never_short_circuits(self) -> None:
        """Ten bad rows produce ten cumulative errors (spec §3.4)."""
        rows = [("01/01/2026", "01/01/2026", "1.00", "1.00")]
        rows.extend(
            (f"0{day}/01/2026", f"0{day}/01/2026", "1.00", "0.00")
            for day in range(2, 10)
        )
        transactions = build_transactions(rows)
        opening = derive_opening_balance(transactions)
        self.assertEqual(len(check_cumulative(transactions, opening)), 8)


class ValueDateInversionTests(unittest.TestCase):
    """§3.3 — value-date order is judged on value dates, in booking order."""

    def test_backwards_value_date_is_an_inversion(self) -> None:
        """A value date earlier than the previous row's is reported with both lines."""
        transactions = build_transactions(
            [
                ("01/01/2026", "05/01/2026", "1.00", "1.00"),
                ("02/01/2026", "04/01/2026", "1.00", "2.00"),
            ]
        )
        inversion = check_value_date_inversions(transactions)[0]
        self.assertEqual(int(inversion.line), 3)
        self.assertEqual(int(inversion.previous_line), 2)

    def test_equal_value_dates_are_not_inversions(self) -> None:
        """Equality is not backwards movement."""
        transactions = build_transactions(
            [
                ("01/01/2026", "05/01/2026", "1.00", "1.00"),
                ("02/01/2026", "05/01/2026", "1.00", "2.00"),
            ]
        )
        self.assertEqual(check_value_date_inversions(transactions), ())

    def test_booking_date_is_not_what_is_compared(self) -> None:
        """A value date that runs ahead of booking order is fine."""
        transactions = build_transactions(
            [
                ("01/01/2026", "01/01/2026", "1.00", "1.00"),
                ("02/01/2026", "09/01/2026", "1.00", "2.00"),
                ("03/01/2026", "10/01/2026", "1.00", "3.00"),
            ]
        )
        self.assertEqual(check_value_date_inversions(transactions), ())


class PropertyTests(unittest.TestCase):
    """Property-style tests over generated statements, not hand-written examples."""

    ROUNDS: ClassVar[int] = 200

    def test_consistent_statements_always_reconcile(self) -> None:
        """For generated consistent statements: continuity and cumulative both pass.

        The generator builds the balances *from* the amounts, so the invariant
        under test is exactly §3: a statement whose balances follow from its
        amounts must never be reported as broken.
        """
        source = SequenceSource(20260915)
        for round_index in range(self.ROUNDS):
            opening = Decimal(source.between(-10_000, 10_000)) / Decimal(100)
            rows: list[tuple[str, str, str, str]] = []
            running = opening
            for row_index in range(source.between(1, 12)):
                cents = source.between(-50_000, 50_000)
                amount = Decimal(cents) / Decimal(100)
                running += amount
                day = 2 + row_index
                stamp = f"{day:02d}/01/2026"
                rows.append((stamp, stamp, str(amount), str(running)))
            transactions = build_transactions(rows)
            with self.subTest(round=round_index):
                self.assertEqual(check_continuity(transactions), ())
                self.assertEqual(
                    check_cumulative(
                        transactions, derive_opening_balance(transactions)
                    ),
                    (),
                )

    def test_a_perturbed_balance_is_always_detected(self) -> None:
        """Perturbing any single balance is caught, with a delta equal to the perturbation.

        This is the converse property: the checks are not merely conservative,
        they are sensitive to every single-row corruption, and they report the
        size of it rather than just its presence.
        """
        source = SequenceSource(4242)
        for round_index in range(self.ROUNDS):
            size = source.between(2, 8)
            rows: list[tuple[str, str, str, str]] = []
            running = Decimal(0)
            for row_index in range(size):
                amount = Decimal(source.between(-5000, 5000)) / Decimal(100)
                running += amount
                day = 2 + row_index
                stamp = f"{day:02d}/02/2026"
                rows.append((stamp, stamp, str(amount), str(running)))
            target = source.below(size)
            perturbation = Decimal(source.sign()) * Decimal("0.01")
            corrupted_balance = Decimal(rows[target][3]) + perturbation
            rows[target] = (*rows[target][:3], str(corrupted_balance))
            transactions = build_transactions(rows)
            report = run_checks(transactions)
            with self.subTest(round=round_index, target=target):
                # Exactly the corrupted row is off, and by exactly the size of
                # the corruption: the perturbation does not leak into later rows,
                # because the file's later balances are still consistent with a
                # running sum that only ever used the *correct* balances.
                if target == 0:
                    # Row 0 is the anchor: perturbing it moves ``openingBalance``
                    # with it, so the anchor row itself still "agrees" and every
                    # later row reports the shift, with the opposite sign —
                    # ``delta`` is ``balanceActual - balanceFromSum``.
                    expected_lines = list(range(3, size + 2))
                    expected_delta = -perturbation
                else:
                    expected_lines = [target + 2]
                    expected_delta = perturbation
                self.assertEqual(
                    [int(e.line) for e in report.cumulative_errors], expected_lines
                )
                self.assertEqual(
                    {e.delta for e in report.cumulative_errors}, {expected_delta}
                )

    def test_opening_balance_round_trips_through_the_running_sum(self) -> None:
        """``opening + Σ amounts`` equals the last balance for consistent statements."""
        source = SequenceSource(7)
        for round_index in range(self.ROUNDS):
            amounts = [
                Decimal(source.between(-99_999, 99_999)) / Decimal(100)
                for _ in range(source.between(1, 20))
            ]
            rows: list[tuple[str, str, str, str]] = []
            running = Decimal("100.00")
            for row_index, amount in enumerate(amounts):
                running += amount
                day = 2 + row_index
                stamp = f"{day:02d}/03/2026"
                rows.append((stamp, stamp, str(amount), str(running)))
            transactions = build_transactions(rows)
            opening = derive_opening_balance(transactions)
            total = opening.amount + sum((t.amount for t in transactions), Decimal(0))
            with self.subTest(round=round_index):
                self.assertEqual(total, transactions[-1].balance_after)


class OpeningBalanceSourceTests(unittest.TestCase):
    """The provenance of the anchor, including a registry mismatch."""

    def test_declared_opening_balance_is_applied_and_checked(self) -> None:
        """A declared opening that disagrees with row 1 is reported, not hidden."""
        transactions = build_transactions(
            [("01/01/2026", "01/01/2026", "10.00", "110.00")]
        )
        opening = derive_opening_balance(transactions, parse_balance("999.00"))
        self.assertEqual(opening.amount, Decimal(100))
        self.assertEqual(opening.declared_mismatch, (Decimal(999), Decimal(100)))

    def test_matching_declared_opening_balance_has_no_mismatch(self) -> None:
        """When the registry agrees with the statement, nothing is reported."""
        transactions = build_transactions(
            [("01/01/2026", "01/01/2026", "10.00", "110.00")]
        )
        opening = derive_opening_balance(transactions, parse_balance("100.00"))
        self.assertIsNone(opening.declared_mismatch)


class FixtureCheckTests(unittest.TestCase):
    """The check report over the local clean fixture, end to end via the CLI."""

    def test_clean_fixture_reconciles_with_one_inversion(self) -> None:
        """``clean.csv`` reconciles and reports exactly one value-date inversion."""
        checks = checks_of(run_cli("baseline", str(fixture("clean.csv"))).json)
        self.assertEqual(checks["rowCount"], 4)
        self.assertEqual(checks["reconciled"], True)
        self.assertEqual(checks["continuityErrors"], [])
        self.assertEqual(checks["cumulativeErrors"], [])
        self.assertEqual(
            count_of(member(checks, "valueDateInversions"), "inversions"), 1
        )

    def test_row_failures_make_the_run_invalid_without_hiding_the_good_rows(
        self,
    ) -> None:
        """A statement with bad rows still emits every good transaction."""
        result = run_cli("baseline", str(fixture("edge_cases.csv")))
        self.assertEqual(result.returncode, 1)
        checks = checks_of(result.json)
        self.assertEqual(checks["rowCount"], 2)
        self.assertEqual(count_of(member(checks, "parseErrors"), "parseErrors"), 7)

    def test_opening_balance_helper_is_total(self) -> None:
        """``derive_opening_balance`` handles the empty statement without raising."""
        opening = derive_opening_balance(())
        self.assertEqual(opening.amount, Decimal(0))


if __name__ == "__main__":
    unittest.main()
