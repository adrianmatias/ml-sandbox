"""Baseline test suite (spec §3, §4, §6, §8).

Run with ``python3 -m unittest discover -s tests`` from the ``python/``
directory.  Standard library only: no pytest, no hypothesis (spec §9.1).

Randomised input for the property-style tests comes from a seeded
``random.Random`` rather than a property-testing dependency, which keeps the
suite dependency-free and bit-for-bit reproducible while still exercising
*generated* (not enumerated) statements.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.dirname(HERE)
WORKTREE = os.path.dirname(PYTHON_ROOT)
FIXTURES = os.path.join(HERE, "fixtures")
BENCH = os.path.dirname(WORKTREE)
REFERENCE = os.path.join(BENCH, "benchmark", "data", "statement.csv")

# ---------------------------------------------------------------------------
# The assertions below pin facts about the ORIGINAL private bank export the
# experiment first ran on: exact collision line numbers, the exact count of rows
# whose booking and value dates differ, and the literal last-row spelling.
# The published benchmark data is synthetic (see benchmark/make_statement.py)
# and reproduces the *structural* invariants — 923 rows, 37713.30, 29
# inversions, 3 collision dates — but not those incidental values.
#
# Rewriting them would falsify the evidence, so they are gated instead. Set
# ORIGINAL_EXPORT = True when running against the real export.
# ---------------------------------------------------------------------------
ORIGINAL_EXPORT = False


if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

from decimal import Decimal

from ledger import dates, strictdecimal
from ledger.checks import (
    build_baseline,
    build_checks,
    continuity_errors,
    cumulative_errors,
    value_date_inversions,
)
from ledger.cli import main
from ledger.errors import AMOUNT, DATE, HEADER, ROW, LedgerError
from ledger.model import Transaction, parse_statement
from ledger.strictdecimal import AmountError, canonical, parse_amount_token


def fixture(name: str) -> str:
    return os.path.join(FIXTURES, name)


def assert_error(
    test: unittest.TestCase, kind: str, line: int | None, callable_, *args
) -> LedgerError:
    """Assert ``callable_`` raises a LedgerError of exactly ``kind``/``line``."""
    with test.assertRaises(LedgerError) as caught:
        callable_(*args)
    error = caught.exception
    test.assertEqual(kind, error.kind)
    if line is not None:
        test.assertEqual(line, error.line)
    return error


# --------------------------------------------------------------------------- #
# §4.1 amount parsing
# --------------------------------------------------------------------------- #


class AmountParsingTests(unittest.TestCase):
    def test_accepts_dot_and_comma_separators(self):
        self.assertEqual(Decimal("1234.56"), parse_amount_token("1234.56"))
        self.assertEqual(Decimal("1234.56"), parse_amount_token("1234,56"))
        self.assertEqual(Decimal("0.56"), parse_amount_token(",56"))
        self.assertEqual(Decimal("0.56"), parse_amount_token(".56"))
        self.assertEqual(Decimal("-1234.56"), parse_amount_token("-1234,56"))
        self.assertEqual(Decimal("400"), parse_amount_token("400"))
        self.assertEqual(Decimal("400"), parse_amount_token("+400"))
        self.assertEqual(Decimal("0.50"), parse_amount_token("0,50"))

    def test_integers_with_three_digits_are_fine(self):
        # "1234" has no separator at all: unambiguous.
        self.assertEqual(Decimal("1234"), parse_amount_token("1234"))
        self.assertEqual(Decimal("40000"), parse_amount_token("40000"))

    def test_more_than_two_trailing_digits_is_rejected_as_over_precise(self):
        # Only *exactly three* trailing digits are ambiguous (spec §4.1); four
        # are unambiguous but not representable at two decimal places.
        with self.assertRaises(AmountError) as caught:
            parse_amount_token("1.2345")
        self.assertIn("two decimal places", str(caught.exception))

    def test_rejects_ambiguous_three_digit_separator(self):
        for token in ("1.234", "1,234", "-1.234", "0.234"):
            with self.subTest(token=token):
                with self.assertRaises(AmountError) as caught:
                    parse_amount_token(token)
                self.assertIn("ambiguous", str(caught.exception))

    def test_rejects_both_separators(self):
        for token in ("1.234,56", "1,234.56"):
            with self.subTest(token=token):
                with self.assertRaises(AmountError) as caught:
                    parse_amount_token(token)
                self.assertIn("ambiguous", str(caught.exception))

    def test_rejects_malformed(self):
        for token in (
            "abc",
            "1.2.3",
            "--5",
            "+-5",
            "5-",
            "",
            "   ",
            ".",
            ",",
            "1.",
            "1,",
            "1 234",
            "1_000",
        ):
            with self.subTest(token=token):
                with self.assertRaises(AmountError):
                    parse_amount_token(token)

    def test_rejects_more_than_two_decimals(self):
        # Four or more trailing digits are unambiguous (not a thousands
        # separator reading) and simply too precise for money.
        for token in ("1.0055", "0,0012", "12.3456"):
            with self.subTest(token=token):
                with self.assertRaises(AmountError) as caught:
                    parse_amount_token(token)
                self.assertIn("two decimal places", str(caught.exception))

    def test_three_trailing_digits_is_ambiguous_even_when_small(self):
        """`1.005` is the same shape as `1.234`: the rule is about the shape."""
        for token in ("1.005", "12.345", "0,001"):
            with self.subTest(token=token):
                with self.assertRaises(AmountError) as caught:
                    parse_amount_token(token)
                self.assertIn("ambiguous", str(caught.exception))

    def test_rejects_float_from_the_domain_model(self):
        # §4.1: a binary float must not be able to enter the domain model.
        for value in (400.0, -0.1, 1e4, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(AmountError):
                    strictdecimal.as_money(value)
        with self.assertRaises(AmountError):
            strictdecimal.as_money(True)

    def test_accepts_exact_types_only(self):
        self.assertEqual(Decimal("400"), strictdecimal.as_money(400))
        self.assertEqual(Decimal("400"), strictdecimal.as_money("400"))
        self.assertEqual(Decimal("400"), strictdecimal.as_money(Decimal("400")))

    def test_no_float_can_reach_a_transaction(self):
        statement = parse_statement(fixture("empty_row.csv"))
        for transaction in statement.transactions:
            self.assertIsInstance(transaction.amount, Decimal)
            self.assertNotIsInstance(transaction.amount, float)
            self.assertIsInstance(transaction.balance_after, Decimal)


class CanonicalRenderingTests(unittest.TestCase):
    def test_canonical_always_two_places(self):
        self.assertEqual("37713.30", canonical(Decimal("37713.3")))
        self.assertEqual("37713.30", canonical(Decimal("37713.30")))
        self.assertEqual("400.00", canonical(Decimal("400")))
        self.assertEqual("0.00", canonical(Decimal("0")))
        self.assertEqual("-100.00", canonical(Decimal("-100")))
        self.assertEqual("1234567.89", canonical(Decimal("1234567.89")))

    def test_canonical_never_uses_exponent_or_scientific_notation(self):
        for value in ("3.77133E4", "1E+3", "1E-2", "0E-10"):
            rendered = canonical(Decimal(value))
            self.assertNotIn("E", rendered.upper())
            self.assertRegex(rendered, r"^-?\d+\.\d{2}$")

    def test_canonical_normalises_negative_zero(self):
        self.assertEqual("0.00", canonical(Decimal("-0.00")))

    def test_arbitrary_precision_is_preserved(self):
        big = Decimal("123456789012345678901234567890.12")
        self.assertEqual("123456789012345678901234567890.12", canonical(big))


# --------------------------------------------------------------------------- #
# §4.2 date parsing
# --------------------------------------------------------------------------- #


class DateParsingTests(unittest.TestCase):
    def test_accepts_strict_dd_mm_yyyy(self):
        self.assertEqual(dates.parse_date_token("01/02/2026"), dates.date(2026, 2, 1))
        self.assertEqual(dates.parse_date_token("29/02/2024"), dates.date(2024, 2, 29))

    def test_rejects_impossible_calendar_dates(self):
        for token in (
            "31/02/2026",
            "31/04/2026",
            "29/02/2023",
            "00/01/2026",
            "01/13/2026",
        ):
            with self.subTest(token=token):
                with self.assertRaises(dates.DateError):
                    dates.parse_date_token(token)

    def test_rejects_wrong_shapes(self):
        for token in (
            "2026-02-01",
            "1/2/2026",
            "01/02/26",
            "01-02-2026",
            "",
            "  ",
            "01/2/2026",
        ):
            with self.subTest(token=token):
                with self.assertRaises(dates.DateError):
                    dates.parse_date_token(token)

    def test_surrounding_whitespace_is_trimmed_before_parsing(self):
        """A `fecha` field of `  01/02/2026` is a stray-space artifact, not a
        different date format: trimmed, then validated strictly."""
        self.assertEqual(dates.date(2026, 2, 1), dates.parse_date_token(" 01/02/2026 "))
        self.assertEqual(dates.date(2026, 2, 1), dates.parse_date_token("\t01/02/2026"))

    def test_iso_rendering(self):
        self.assertEqual("2021-03-11", dates.iso(dates.date(2021, 3, 11)))


# --------------------------------------------------------------------------- #
# §4 fixtures through the parser
# --------------------------------------------------------------------------- #


class FixtureParsingTests(unittest.TestCase):
    def test_empty_row_is_skipped_not_an_error(self):
        statement = parse_statement(fixture("empty_row.csv"))
        self.assertEqual(2, len(statement.transactions))
        self.assertEqual((2,), statement.skipped_empty_rows)
        # The skipped line is still counted: the first data row is physical line 3.
        self.assertEqual(3, statement.transactions[0].line)
        self.assertEqual(4, statement.transactions[1].line)

    def test_quoted_field(self):
        statement = parse_statement(fixture("quoted_field.csv"))
        self.assertEqual(2, len(statement.transactions))
        self.assertEqual(Decimal("1234.56"), statement.transactions[0].amount)
        # rawAmount is the value as it appeared, minus the CSV quoting itself.
        self.assertEqual("1234,56", statement.transactions[0].raw_amount)
        self.assertEqual("1234.56", canonical(statement.transactions[0].amount))

    def test_comma_decimal_amounts(self):
        statement = parse_statement(fixture("comma_decimal.csv"))
        self.assertEqual(3, len(statement.transactions))
        self.assertEqual("1234.56", canonical(statement.transactions[0].amount))
        self.assertEqual("1134.56", canonical(statement.transactions[1].balance_after))
        self.assertEqual("1234,56", statement.transactions[0].raw_amount)
        # The parser is broader than the delimiter allows: the unquoted form is
        # exercised directly, since in a comma-delimited file it is not
        # expressible as a single field at all.
        self.assertEqual(Decimal("1234.56"), parse_amount_token("1234,56"))
        self.assertEqual(Decimal("2369.12"), parse_amount_token("2369,12"))

    def test_unquoted_comma_decimal_is_a_row_error_not_a_guess(self):
        """In a comma-delimited file `1234,56` really is two fields: report ROW."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "fecha,fecha valor,importe,saldo\n"
                    "11/03/2021,11/03/2021,1234,56,1234,56\n"
                )
            assert_error(self, ROW, 2, parse_statement, path)

    def test_malformed_amounts_are_rejected_with_line_and_kind(self):
        cases = {
            "bad_amount_text.csv": 2,
            "bad_amount_two_dots.csv": 2,
            "bad_amount_multi_sign.csv": 2,
        }
        for name, line in cases.items():
            with self.subTest(fixture=name):
                assert_error(self, AMOUNT, line, parse_statement, fixture(name))

    def test_ambiguous_amount_is_rejected_not_guessed(self):
        error = assert_error(
            self, AMOUNT, 2, parse_statement, fixture("ambiguous_amount.csv")
        )
        self.assertIn("ambiguous", error.message)

    def test_missing_saldo(self):
        assert_error(self, AMOUNT, 3, parse_statement, fixture("missing_saldo.csv"))

    def test_bad_date(self):
        assert_error(self, DATE, 2, parse_statement, fixture("bad_date.csv"))

    def test_impossible_date_does_not_roll_over(self):
        assert_error(self, DATE, 2, parse_statement, fixture("impossible_date.csv"))

    def test_crlf_produces_identical_domain_values(self):
        crlf = parse_statement(fixture("crlf.csv"))
        lf = parse_statement(fixture("empty_row.csv"))
        self.assertEqual(
            [t.as_json() for t in lf.transactions],
            [t.as_json() for t in crlf.transactions],
        )
        self.assertEqual(lf.skipped_empty_rows, crlf.skipped_empty_rows)
        with open(fixture("crlf.csv"), "rb") as handle:
            self.assertIn(b"\r\n", handle.read())

    def test_collision_on_natural_key_does_not_merge_rows(self):
        statement = parse_statement(fixture("collision.csv"))
        self.assertEqual(16, len(statement.transactions))
        keys = [
            (t.booking_date, t.amount, t.balance_after) for t in statement.transactions
        ]
        # Three natural keys occur exactly twice and both rows survive.
        seen = {}
        for key in keys:
            seen[key] = seen.get(key, 0) + 1
        duplicated = [key for key, count in seen.items() if count == 2]
        self.assertEqual(3, len(duplicated))
        for key in duplicated:
            rows = [
                t
                for t in statement.transactions
                if (t.booking_date, t.amount, t.balance_after) == key
            ]
            self.assertEqual(2, len(rows))
            self.assertNotEqual(rows[0].line, rows[1].line)
            # Identical content, therefore identical checksum: the checksum
            # cannot be the identity of a row.
            self.assertEqual(rows[0].checksum, rows[1].checksum)

    def test_collision_rows_are_distinguishable_only_by_line(self):
        """Collision and the §3.1 continuity invariant hold at the same time.

        That is only possible because the colliding pair is separated by a
        compensating pair that returns to the same balance — the shape the
        reference file uses at 243/247, 288/290 and 446/448.
        """
        statement = parse_statement(fixture("collision.csv"))
        by_line = {t.line: t for t in statement.transactions}
        for first, second in ((2, 6), (8, 10), (13, 15)):
            with self.subTest(pair=(first, second)):
                a, b = by_line[first], by_line[second]
                self.assertEqual(
                    (a.booking_date, a.value_date, a.amount, a.balance_after),
                    (b.booking_date, b.value_date, b.amount, b.balance_after),
                )
                # Same natural content, different position: a dedup keyed on
                # natural fields (or on the checksum) would drop the second row.
                self.assertEqual(a.checksum, b.checksum)
        checks = build_checks(statement)
        self.assertEqual([], checks["continuityErrors"])
        self.assertEqual([], checks["cumulativeErrors"])
        self.assertFalse(build_checks(statement)["continuityErrors"])


# --------------------------------------------------------------------------- #
# §3 checks: accumulation and independence
# --------------------------------------------------------------------------- #


class CheckTests(unittest.TestCase):
    def test_continuity_delta_sign_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "fecha,fecha valor,importe,saldo\n"
                    "01/01/2026,01/01/2026,100,100\n"
                    "02/01/2026,02/01/2026,100,250\n"
                )
            statement = parse_statement(path)
        errors = continuity_errors(statement.transactions, statement.opening_balance)
        self.assertEqual(1, len(errors))
        self.assertEqual(
            {
                "line": 3,
                "balanceExpected": "200.00",
                "balanceActual": "250.00",
                "delta": "50.00",
            },
            errors[0],
        )

    def test_cumulative_delta_sign_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "fecha,fecha valor,importe,saldo\n"
                    "01/01/2026,01/01/2026,100,100\n"
                    "02/01/2026,02/01/2026,100,150\n"
                )
            statement = parse_statement(path)
        errors = cumulative_errors(statement.transactions, statement.opening_balance)
        self.assertEqual(
            {
                "line": 3,
                "balanceFromSum": "200.00",
                "balanceActual": "150.00",
                "delta": "-50.00",
            },
            errors[0],
        )

    def test_checks_accumulate_every_violation_and_never_stop_early(self):
        """§3.4: many violations, all reported, in ascending line order."""
        rows = ["fecha,fecha valor,importe,saldo", "01/01/2026,01/01/2026,100,100"]
        # 5 deliberately broken rows, each colliding with the next.
        for day in range(2, 7):
            rows.append(f"0{day}/01/2026,0{day}/01/2026,10,{100 + day * 1000}")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(rows) + "\n")
            statement = parse_statement(path)
            checks = build_checks(statement)
        self.assertEqual(5, len(checks["continuityErrors"]))
        self.assertEqual(5, len(checks["cumulativeErrors"]))
        self.assertEqual(
            [3, 4, 5, 6, 7], [e["line"] for e in checks["continuityErrors"]]
        )
        self.assertEqual(
            [3, 4, 5, 6, 7], [e["line"] for e in checks["cumulativeErrors"]]
        )
        self.assertFalse(checks["reconciled"])

    def test_continuity_and_cumulative_are_computed_independently(self):
        """One wrong saldo; the two checks report different row sets.

        §3.1 anchors row *i* on the recorded balance of row *i-1*, so a single
        bad saldo shows up locally twice (the bad row, and its successor whose
        neighbour moved) and then stops.  §3.2 anchors on the running sum, so
        the same error shows up on the bad row and on *every* row after it,
        because the sum never re-anchors.  Independent implementations must
        disagree like this; one derived from the other's state could not.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "fecha,fecha valor,importe,saldo\n"
                    "01/01/2026,01/01/2026,100,100\n"
                    "02/01/2026,02/01/2026,100,999\n"  # wrong: should be 200
                    "03/01/2026,03/01/2026,100,299\n"  # right relative to 999
                    "04/01/2026,04/01/2026,100,399\n"
                )
            statement = parse_statement(path)
        continuity = continuity_errors(
            statement.transactions, statement.opening_balance
        )
        cumulative = cumulative_errors(
            statement.transactions, statement.opening_balance
        )
        self.assertEqual([3, 4], [e["line"] for e in continuity])
        self.assertEqual([3, 4, 5], [e["line"] for e in cumulative])
        self.assertNotEqual(
            [e["line"] for e in continuity], [e["line"] for e in cumulative]
        )
        # Both agree about the row that is actually wrong, and disagree after it.
        self.assertEqual("799.00", continuity[0]["delta"])
        self.assertEqual("799.00", cumulative[0]["delta"])

    def test_a_single_wrong_saldo_makes_cumulative_never_re_anchor(self):
        """The cumulative deviation persists to the last row (spec §3.2)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "fecha,fecha valor,importe,saldo\n"
                    "01/01/2026,01/01/2026,100,100\n"
                    "02/01/2026,02/01/2026,0,100\n"
                    "03/01/2026,03/01/2026,100,999\n"  # line 4: should be 200
                    "04/01/2026,04/01/2026,0,999\n"
                    "05/01/2026,05/01/2026,19,1018\n"
                )
            statement = parse_statement(path)
        cumulative = cumulative_errors(
            statement.transactions, statement.opening_balance
        )
        self.assertEqual([4, 5, 6], [e["line"] for e in cumulative])
        # The row that is actually wrong is off by 799.00; every row after it
        # inherits this balance error as a pure +800.00-vs-+1.00 drift, and the
        # sum never heals because §3.2 does not re-anchor on recorded balances.
        self.assertEqual("799.00", cumulative[0]["delta"])
        self.assertEqual(
            {"5": "799.00", "6": "799.00"},
            {str(e["line"]): e["delta"] for e in cumulative[1:]},
        )

    def test_value_date_inversions_use_booking_order_and_ignore_equal_dates(self):
        statement = parse_statement(fixture("empty_row.csv"))
        self.assertEqual([], value_date_inversions(statement.transactions))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "fecha,fecha valor,importe,saldo\n"
                    "01/01/2026,10/01/2026,0,0\n"
                    "02/01/2026,10/01/2026,0,0\n"  # equal: not an inversion
                    "03/01/2026,05/01/2026,0,0\n"  # backwards: inversion
                    "04/01/2026,09/01/2026,0,0\n"  # still below 10/01: inversion
                )
            statement = parse_statement(path)
        self.assertEqual(
            [{"line": 4, "previousLine": 3}],
            value_date_inversions(statement.transactions),
        )

    def test_empty_statement_is_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("fecha,fecha valor,importe,saldo\n,,,\n")
            statement = parse_statement(path)
        checks = build_checks(statement)
        self.assertEqual(0, checks["rowCount"])
        self.assertEqual([2], checks["skippedEmptyRows"])
        self.assertEqual([], checks["continuityErrors"])
        self.assertEqual([], checks["cumulativeErrors"])
        self.assertEqual("0.00", checks["openingBalance"])
        self.assertEqual("0.00", checks["finalBalance"])


class HeaderTests(unittest.TestCase):
    def test_wrong_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("fecha,importe,saldo\n")
            assert_error(self, HEADER, 1, parse_statement, path)

    def test_reordered_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("fecha,fecha valor,saldo,importe\n")
            assert_error(self, HEADER, 1, parse_statement, path)

    def test_wrong_field_count_is_a_row_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "fecha,fecha valor,importe,saldo\n01/01/2026,01/01/2026,1\n"
                )
            assert_error(self, ROW, 2, parse_statement, path)


# --------------------------------------------------------------------------- #
# §8 the real reference file
# --------------------------------------------------------------------------- #


class ReferenceFileTests(unittest.TestCase):
    """Spec §8: a suite that passes while the real file fails the 29-inversion
    count is worthless — so assert the real numbers."""

    @classmethod
    def setUpClass(cls):
        cls.statement = parse_statement(REFERENCE)
        cls.checks = build_checks(cls.statement)
        cls.data = build_baseline(cls.statement)

    def test_row_count_is_923(self):
        self.assertEqual(923, self.checks["rowCount"])
        self.assertEqual(923, len(self.data["transactions"]))

    def test_skipped_empty_rows_is_line_2(self):
        self.assertEqual([2], self.checks["skippedEmptyRows"])

    def test_first_transaction_is_at_physical_line_3(self):
        # Header = line 1, skipped empty row = line 2, first data row = line 3.
        self.assertEqual(3, self.data["transactions"][0]["line"])

    def test_value_date_inversions_is_exactly_29(self):
        self.assertEqual(29, len(self.checks["valueDateInversions"]))
        self.assertEqual(29, len(value_date_inversions(self.statement.transactions)))

    def test_booking_order_has_no_inversions(self):
        booking = [t.booking_date for t in self.statement.transactions]
        self.assertEqual(booking, sorted(booking))

    def test_continuity_and_cumulative_are_both_empty(self):
        self.assertEqual([], self.checks["continuityErrors"])
        self.assertEqual([], self.checks["cumulativeErrors"])

    def test_final_balance_reconstructs_exactly(self):
        self.assertEqual("37713.30", self.checks["finalBalance"])
        self.assertEqual("0.00", self.checks["openingBalance"])
        opening = self.statement.opening_balance
        running = opening
        for transaction in self.statement.transactions:
            running += transaction.amount
        self.assertEqual(Decimal("37713.30"), running)
        self.assertEqual(Decimal("37713.30"), self.statement.final_balance)
        self.assertEqual("37713.30", self.data["transactions"][-1]["balanceAfter"])

    @unittest.skipUnless(ORIGINAL_EXPORT, "pins a fact about the original export")
    def test_92_rows_have_distinct_value_dates(self):
        differing = sum(
            1 for t in self.statement.transactions if t.booking_date != t.value_date
        )
        self.assertEqual(92, differing)

    def test_reconciled_ignores_value_date_inversions(self):
        """CLARIFICATIONS.md §1: inversions are observations, not errors.

        The reference file has 29 of them and must still be `reconciled`.
        """
        self.assertTrue(self.checks["reconciled"])
        self.assertEqual(29, len(self.checks["valueDateInversions"]))

    def test_reconciled_is_false_when_a_balance_check_fails(self):
        """The converse of the clarification: a real error does clear it."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "fecha,fecha valor,importe,saldo\n"
                    "01/01/2026,01/01/2026,100,100\n"
                    "02/01/2026,02/01/2026,100,250\n"
                )
            statement = parse_statement(path)
        checks = build_checks(statement)
        self.assertFalse(checks["reconciled"])
        self.assertTrue(checks["continuityErrors"])

    def test_reconciled_is_true_when_only_observations_are_non_empty(self):
        """Inversions + skipped rows, zero errors -> reconciled."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "fecha,fecha valor,importe,saldo\n"
                    ",,,\n"
                    "01/01/2026,10/01/2026,100,100\n"
                    "02/01/2026,05/01/2026,0,100\n"
                )
            statement = parse_statement(path)
        checks = build_checks(statement)
        self.assertEqual([2], checks["skippedEmptyRows"])
        self.assertEqual(
            [{"line": 4, "previousLine": 3}], checks["valueDateInversions"]
        )
        self.assertEqual([], checks["continuityErrors"])
        self.assertEqual([], checks["cumulativeErrors"])
        self.assertTrue(checks["reconciled"])

    @unittest.skipUnless(
        ORIGINAL_EXPORT, "collision line numbers are from the original export"
    )
    def test_known_natural_key_collisions_survive_at_the_stated_lines(self):
        pairs = ((243, 247), (288, 290), (446, 448))
        by_line = {t.line: t for t in self.statement.transactions}
        for first, second in pairs:
            with self.subTest(pair=(first, second)):
                a, b = by_line[first], by_line[second]
                self.assertEqual(
                    (a.booking_date, a.amount, a.balance_after),
                    (b.booking_date, b.amount, b.balance_after),
                )
                # The checksum is a *content* fingerprint, so two genuinely
                # distinct rows with identical content share it.  That is the
                # point: neither the natural key nor the checksum can be used
                # as a dedup key, only the line number distinguishes them.
                self.assertEqual(a.checksum, b.checksum)
                self.assertNotEqual(a.line, b.line)
        self.assertEqual(923, len(by_line))
        self.assertEqual(923, len(set(by_line)))

    def test_every_amount_and_balance_is_a_canonical_decimal_string(self):
        for entry in self.data["transactions"]:
            for key in ("amount", "balanceAfter"):
                self.assertRegex(entry[key], r"^-?\d+\.\d{2}$")

    def test_checksum_is_16_hex_chars_of_the_canonical_preimage(self):
        import hashlib

        entry = self.data["transactions"][0]
        preimage = f"{entry['amount']}|{entry['balanceAfter']}|{entry['bookingDate']}"
        expected = hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]
        self.assertEqual(expected, entry["checksum"])
        self.assertRegex(entry["checksum"], r"^[0-9a-f]{16}$")


# --------------------------------------------------------------------------- #
# Property-style tests (spec §8)
# --------------------------------------------------------------------------- #


def generate_statement(
    rng: random.Random, rows: int = 40
) -> tuple[str, list[Transaction], list[int]]:
    """Generate a synthetic statement from a seeded RNG.

    Amounts are drawn at two decimal places; the balance column is generated by
    exact integer-cent accumulation (never by float arithmetic), so the
    expected continuity/cumulative results are known by construction.  Random
    rows are also blanked out to exercise ``skippedEmptyRows``.
    """
    opening_cents = rng.randrange(-500_00, 500_00)
    lines = ["fecha,fecha valor,importe,saldo"]
    expected: list[Transaction] = []
    skipped: list[int] = []
    balance_cents = opening_cents
    money = Decimal(1).scaleb(-2)
    booking = dates.date(2026, 1, 1)
    value = dates.date(2026, 1, 1)

    for index in range(rows):
        line_number = index + 2
        if rng.random() < 0.1:
            lines.append(",,,")
            skipped.append(line_number)
            continue
        amount_cents = rng.randrange(-250_00, 250_00)
        balance_cents += amount_cents
        booking = booking + dates.timedelta_days(rng.randrange(0, 3))
        # Value dates wander independently, producing inversions on purpose.
        value = value + dates.timedelta_days(rng.choice((-2, -1, 0, 0, 1, 3)))
        amount = Decimal(amount_cents) * money
        balance = Decimal(balance_cents) * money
        lines.append(
            f"{booking.strftime('%d/%m/%Y')},{value.strftime('%d/%m/%Y')},"
            f"{canonical(amount)},{canonical(balance)}"
        )
        expected.append(
            Transaction(
                line=line_number,
                booking_date=booking,
                value_date=value,
                amount=amount,
                balance_after=balance,
                raw_amount=canonical(amount),
                raw_balance=canonical(balance),
            )
        )
    return "\n".join(lines) + "\n", expected, skipped


class PropertyTests(unittest.TestCase):
    """Generated inputs, invariants that must hold for every one of them."""

    SEEDS = range(24)

    def _run(self, seed: int):
        rng = random.Random(seed)
        text, expected, skipped = generate_statement(rng)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "generated.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            statement = parse_statement(path)
            checks = build_checks(statement)
        return statement, checks, expected, skipped

    def test_generated_statements_round_trip_exactly(self):
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                _, checks, expected, skipped = self._run(seed)
                self.assertEqual(len(expected), checks["rowCount"])
                self.assertEqual(skipped, checks["skippedEmptyRows"])
                self.assertEqual([], checks["continuityErrors"])
                self.assertEqual([], checks["cumulativeErrors"])

    def test_parsed_values_equal_independently_generated_values(self):
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                statement, _, expected, _ = self._run(seed)
                self.assertEqual(
                    [t.as_json() for t in expected],
                    [t.as_json() for t in statement.transactions],
                )

    def test_inversion_count_matches_a_naive_recount(self):
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                statement, checks, expected, _ = self._run(seed)
                naive = sum(
                    1
                    for i in range(1, len(expected))
                    if expected[i].value_date < expected[i - 1].value_date
                )
                self.assertEqual(naive, len(checks["valueDateInversions"]))

    def test_injected_corruption_is_always_detected(self):
        """Perturb one balance by +0.01: at least one check must notice, always."""
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                text, _, _ = generate_statement(rng)
                rows = text.splitlines()
                # Only a real data row can be corrupted; generated blank rows
                # (",,,") are skipped records.
                candidates = [
                    index
                    for index in range(1, len(rows))
                    if rows[index].strip(", ") != ""
                ]
                target = rng.choice(candidates)
                fields = rows[target].split(",")
                fields[3] = canonical(Decimal(fields[3]) + Decimal("0.01"))
                rows[target] = ",".join(fields)
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "corrupt.csv")
                    with open(path, "w", encoding="utf-8") as handle:
                        handle.write("\n".join(rows) + "\n")
                    statement = parse_statement(path)
                checks = build_checks(statement)
                self.assertTrue(
                    checks["continuityErrors"] or checks["cumulativeErrors"],
                    f"corruption at line {target + 1} went unnoticed (seed {seed})",
                )

    def test_canonical_strings_are_stable_under_respelling(self):
        """Respellings of the same value parse to one canonical string."""
        rng = random.Random(1234)
        for _ in range(200):
            cents = rng.randrange(-10_000_00, 10_000_00)
            value = Decimal(cents).scaleb(-2)
            spellings = {
                canonical(value),
                f"{value:.2f}",
                str(value),
                value.to_eng_string(),
            }
            if value == value.to_integral_value():
                spellings.add(str(int(value)))
            for spelling in spellings:
                if spelling == canonical(value):
                    continue
                with self.subTest(cents=cents, spelling=spelling):
                    self.assertEqual(
                        canonical(value), canonical(parse_amount_token(spelling))
                    )

    def test_arbitrary_precision_sums_are_exact(self):
        """Summing many decimals must not round at the default context precision."""
        rng = random.Random(99)
        values = [
            Decimal(rng.randrange(-(10**20), 10**20)).scaleb(-2) for _ in range(500)
        ]
        exact = Decimal(sum(int(v.scaleb(2)) for v in values)).scaleb(-2)
        self.assertEqual(exact, strictdecimal.exact_sum(values))
        self.assertEqual(canonical(exact), canonical(strictdecimal.exact_sum(values)))


# --------------------------------------------------------------------------- #
# §6 CLI contract
# --------------------------------------------------------------------------- #


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ledger", *args],
        cwd=PYTHON_ROOT,
        capture_output=True,
        text=True,
    )


class CliTests(unittest.TestCase):
    def test_help_exits_zero(self):
        result = run_cli("--help")
        self.assertEqual(0, result.returncode)
        self.assertIn("baseline", result.stdout)

    def test_no_arguments_is_a_usage_error(self):
        result = run_cli()
        self.assertEqual(2, result.returncode)

    def test_unknown_command_is_a_usage_error(self):
        result = run_cli("frobnicate", "x")
        self.assertEqual(2, result.returncode)

    def test_missing_argument_is_a_usage_error(self):
        result = run_cli("baseline")
        self.assertEqual(2, result.returncode)

    def test_reference_file_exits_zero_with_a_valid_envelope(self):
        result = run_cli("baseline", REFERENCE)
        self.assertEqual(0, result.returncode)
        payload = json.loads(result.stdout)
        self.assertEqual(["ok", "module", "version", "data"], list(payload.keys()))
        self.assertTrue(payload["ok"])
        self.assertEqual("ledger", payload["module"])
        self.assertEqual(1, payload["version"])

    def test_unparseable_input_exits_one_with_the_frozen_error_shape(self):
        result = run_cli("baseline", fixture("ambiguous_amount.csv"))
        self.assertEqual(1, result.returncode)
        payload = json.loads(result.stdout)
        self.assertEqual(["ok", "module", "version", "error"], list(payload.keys()))
        self.assertFalse(payload["ok"])
        self.assertEqual(["kind", "message", "line"], list(payload["error"].keys()))
        self.assertEqual("AMOUNT", payload["error"]["kind"])
        self.assertEqual(2, payload["error"]["line"])

    def test_missing_file_exits_one_with_io_kind(self):
        result = run_cli("baseline", os.path.join(FIXTURES, "does-not-exist.csv"))
        self.assertEqual(1, result.returncode)
        payload = json.loads(result.stdout)
        self.assertEqual("IO", payload["error"]["kind"])
        self.assertEqual(["kind", "message"], list(payload["error"].keys()))

    def test_output_is_deterministic_across_runs(self):
        first = run_cli("baseline", REFERENCE)
        second = run_cli("baseline", REFERENCE)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(first.stdout.encode("utf-8"), second.stdout.encode("utf-8"))

    def test_check_violations_still_exit_zero(self):
        """§6: exit 0 when the file is well-formed, even if checks report violations."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken-balance.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "fecha,fecha valor,importe,saldo\n"
                    "01/01/2026,01/01/2026,100,100\n"
                    "02/01/2026,31/12/2025,100,500\n"  # broken balance AND an inversion
                )
            result = run_cli("baseline", path)
        self.assertEqual(0, result.returncode)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        checks = payload["data"]["checks"]
        self.assertTrue(checks["continuityErrors"])
        self.assertTrue(checks["cumulativeErrors"])
        self.assertTrue(checks["valueDateInversions"])
        self.assertFalse(checks["reconciled"])

    def test_stdout_is_pure_ascii_with_one_trailing_newline(self):
        result = run_cli("baseline", REFERENCE)
        self.assertTrue(result.stdout.endswith("\n"))
        self.assertFalse(result.stdout.endswith("\n\n"))
        result.stdout.encode("ascii")  # raises if non-ASCII slipped in

    def test_envelope_has_no_paths_or_timestamps(self):
        result = run_cli("baseline", REFERENCE)
        payload = json.loads(result.stdout)
        self.assertNotIn("generatedAt", result.stdout)
        for entry in payload["data"]["transactions"]:
            self.assertNotIn("/", entry["rawAmount"])
            self.assertNotIn("/", entry["rawBalance"])

    def test_cli_entry_point_callable_in_process(self):
        """main() is importable and returns the exit code rather than exiting."""
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["baseline", REFERENCE])
        self.assertEqual(0, code)
        self.assertTrue(json.loads(buffer.getvalue())["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
