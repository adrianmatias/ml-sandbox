"""The §4 fixtures, exercised as fixtures.

Covers every case spec §4 names: a fully empty row, a quoted field, a
comma-decimal amount, malformed amounts, an ambiguous ``1.234``, a missing
``saldo``, a non-``DD/MM/YYYY`` date, CRLF line endings and the
``(bookingDate, amount, balanceAfter)`` collision.
"""

import unittest
from typing import ClassVar

from ledger.checks import run_checks
from ledger.statement import parse_statement_file

from .helpers import checks_of, fixture, run_cli
from .make_fixtures import write_fixtures


class FixtureIntegrityTests(unittest.TestCase):
    """The fixture generator must be deterministic and complete."""

    REQUIRED_CASES: ClassVar[tuple[str, ...]] = (
        "clean.csv",
        "edge_cases.csv",
        "collision.csv",
        "registry.json",
        "A.csv",
        "B.csv",
        "C.csv",
    )

    def test_every_required_fixture_is_written(self) -> None:
        """All §4 fixtures exist after generation."""
        written = {path.name for path in write_fixtures()}
        for name in self.REQUIRED_CASES:
            self.assertIn(name, written)

    def test_bom_fixture_really_has_a_bom(self) -> None:
        """``clean.csv`` starts with a UTF-8 BOM, which §2 requires us to accept."""
        self.assertTrue(fixture("clean.csv").read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_crlf_fixture_really_uses_crlf(self) -> None:
        """``collision.csv`` uses CRLF line endings, per §4."""
        self.assertIn(b"\r\n", fixture("collision.csv").read_bytes())


class EdgeCaseFixtureTests(unittest.TestCase):
    """One test per §4 edge case, against the generated fixture."""

    def setUp(self) -> None:
        """Parse ``edge_cases.csv`` once per test."""
        self.parsed = parse_statement_file(fixture("edge_cases.csv"))

    def test_empty_rows_are_skipped_not_errors(self) -> None:
        """Lines 2 and 12 are fully empty and must be skipped, not reported."""
        self.assertEqual(
            [int(line) for line in self.parsed.skipped_empty_rows], [2, 12]
        )

    def test_every_malformed_row_is_reported_in_line_order(self) -> None:
        """All five bad rows are collected — the parser never stops at the first."""
        self.assertEqual(
            [int(f.line) for f in self.parsed.row_errors], [5, 6, 7, 8, 9, 10, 11]
        )

    def test_malformed_amounts_are_amount_errors(self) -> None:
        """``abc``, ``1.2.3``, ``--5`` and the ambiguous ``1.234`` are AMOUNT errors."""
        kinds = {int(f.line): f.kind for f in self.parsed.row_errors}
        self.assertEqual(kinds[5], "AMOUNT")
        self.assertEqual(kinds[6], "AMOUNT")
        self.assertEqual(kinds[7], "AMOUNT")
        self.assertEqual(kinds[8], "AMOUNT")

    def test_ambiguous_thousands_separator_says_so(self) -> None:
        """``1.234`` is rejected as ambiguous, and the message says why."""
        message = next(f.message for f in self.parsed.row_errors if int(f.line) == 8)
        self.assertIn("ambiguous", message)

    def test_missing_saldo_is_a_row_error(self) -> None:
        """A row without ``saldo`` is structurally wrong, not silently zero."""
        kinds = {int(f.line): f.kind for f in self.parsed.row_errors}
        self.assertEqual(kinds[9], "ROW")

    def test_bad_dates_are_date_errors(self) -> None:
        """``2026-01-10`` and ``31/02/2026`` are both rejected as dates."""
        by_line = {int(f.line): f.kind for f in self.parsed.row_errors}
        self.assertEqual(by_line[10], "DATE")
        self.assertEqual(by_line[11], "DATE")

    def test_non_dd_mm_yyyy_shapes_are_rejected(self) -> None:
        """A date that is not exactly ``DD/MM/YYYY`` is refused, not reordered."""
        message = next(f.message for f in self.parsed.row_errors if int(f.line) == 10)
        self.assertIn("DD/MM/YYYY", message)

    def test_quoted_comma_decimal_survives_as_exact_money(self) -> None:
        """A quoted ``1234,56`` is accepted and parsed as exactly 1234.56."""
        amounts = [str(t.amount) for t in self.parsed.transactions]
        self.assertIn("1234.56", amounts)

    def test_exactly_the_good_rows_survive(self) -> None:
        """Only the two well-formed rows become transactions (lines 3 and 4)."""
        self.assertEqual([int(t.line) for t in self.parsed.transactions], [3, 4])


class CollisionFixtureTests(unittest.TestCase):
    """Spec §4: natural-key collisions must not merge real transactions."""

    def test_colliding_rows_are_both_kept(self) -> None:
        """Both members of each colliding pair survive parsing, with distinct lines."""
        parsed = parse_statement_file(fixture("collision.csv"))
        self.assertEqual(len(parsed.transactions), 7)
        self.assertEqual(
            [int(t.line) for t in parsed.transactions], [2, 3, 4, 5, 6, 7, 8]
        )

    def test_colliding_rows_are_identical_on_natural_fields(self) -> None:
        """The collisions are real: (date, amount, balance) repeats exactly, twice."""
        parsed = parse_statement_file(fixture("collision.csv"))
        natural = [
            (t.booking_date, t.amount, t.balance_after) for t in parsed.transactions
        ]
        repeated = [entry for entry in natural if natural.count(entry) > 1]
        self.assertEqual(len(repeated), 4)

    def test_colliding_rows_have_identical_checksums(self) -> None:
        """The checksum covers only natural fields, so it collides too — by design.

        This is why nothing may dedupe on the checksum, and why ``line`` is the
        only safe identity for a row.
        """
        parsed = parse_statement_file(fixture("collision.csv"))
        self.assertEqual(
            parsed.transactions[1].checksum, parsed.transactions[3].checksum
        )
        self.assertEqual(
            parsed.transactions[4].checksum, parsed.transactions[6].checksum
        )
        self.assertEqual(len({t.checksum for t in parsed.transactions}), 5)

    def test_a_colliding_statement_still_reconciles(self) -> None:
        """Collisions are a data fact, not a reconciliation error."""
        parsed = parse_statement_file(fixture("collision.csv"))
        report = run_checks(parsed.transactions)
        self.assertEqual(report.continuity_errors, ())
        self.assertEqual(report.cumulative_errors, ())


class CrlfAndBomTests(unittest.TestCase):
    """§2 BOM tolerance and §4 CRLF handling, through the real CLI."""

    def test_crlf_file_with_collisions_reconciles(self) -> None:
        """The CRLF fixture parses to seven transactions with no errors at all."""
        checks = checks_of(run_cli("baseline", str(fixture("collision.csv"))).json)
        self.assertEqual(checks["rowCount"], 7)
        self.assertEqual(checks["parseErrors"], [])
        self.assertEqual(checks["reconciled"], True)

    def test_bom_file_is_accepted(self) -> None:
        """A BOM on line 1 does not corrupt the header match."""
        checks = checks_of(run_cli("baseline", str(fixture("clean.csv"))).json)
        self.assertEqual(checks["rowCount"], 4)
        self.assertEqual(checks["finalBalance"], "513.50")


if __name__ == "__main__":
    unittest.main()
