"""Spec §4.1 and §4.2 — the parsing rules, including what must be rejected."""

import unittest
from datetime import date
from decimal import Decimal
from typing import ClassVar

from ledger.amounts import parse_amount, parse_balance, parse_decimal
from ledger.dates import parse_date
from ledger.errors import AmountError, DateError


class AcceptedAmountTests(unittest.TestCase):
    """Amounts that must parse to the exact value written in the file."""

    ACCEPTED: ClassVar[tuple[tuple[str, str], ...]] = (
        ("400", "400"),
        ("-9", "-9"),
        ("+12.5", "12.5"),
        ("37713.30", "37713.30"),
        ("0.00", "0.00"),
        ("-0.01", "-0.01"),
        ("1234,56", "1234.56"),
        ("1234.5", "1234.5"),
        ("12345", "12345"),
        ("  42.00  ", "42.00"),
        ("1.234.567,89", "1234567.89"),
        ("1,234,567.89", "1234567.89"),
        ("+0", "0"),
    )

    def test_accepted_amounts(self) -> None:
        """Each accepted spelling yields exactly the expected decimal."""
        for text, expected in self.ACCEPTED:
            with self.subTest(text=text):
                self.assertEqual(parse_decimal(text), Decimal(expected))

    def test_comma_decimal_is_not_a_thousand_separator(self) -> None:
        """``1234,56`` is 1234.56, never 123456 (spec §4.1)."""
        self.assertEqual(parse_amount("1234,56"), Decimal("1234.56"))


class RejectedAmountTests(unittest.TestCase):
    """Amounts that must be rejected, with the reason recorded per case."""

    REJECTED: ClassVar[tuple[tuple[str, str], ...]] = (
        ("", "empty"),
        ("   ", "empty"),
        ("abc", "non-numeric"),
        ("1.2.3", "ambiguous"),
        ("--5", "multi-sign"),
        ("+-5", "multi-sign"),
        ("5-", "non-numeric"),
        ("1,2,3", "ambiguous"),
        ("1.234", "ambiguous"),
        ("1,234", "ambiguous"),
        ("-1.234", "ambiguous"),
        ("1.2345.6", "ambiguous"),
        ("12.34.567", "ambiguous"),
        ("1,234,567", "ambiguous"),
        ("1234,56.78", "ambiguous"),
        ("1e3", "non-numeric"),
        ("NaN", "non-numeric"),
        ("Infinity", "non-numeric"),
        (".", "malformed"),
        ("-", "non-numeric"),
        ("€5", "non-numeric"),
        ("5 000", "non-numeric"),
    )

    def test_rejected_amounts_raise_amount_error(self) -> None:
        """Every malformed or ambiguous spelling raises an ``AMOUNT`` error."""
        for text, reason in self.REJECTED:
            with self.subTest(text=text):
                with self.assertRaises(AmountError) as caught:
                    parse_decimal(text)
                self.assertEqual(caught.exception.kind, "AMOUNT")
                self.assertTrue(caught.exception.message, reason)

    def test_ambiguous_amount_is_not_guessed(self) -> None:
        """``1.234`` must be refused, not read as either 1234 or 1.234."""
        with self.assertRaises(AmountError) as caught:
            parse_decimal("1.234")
        self.assertIn("ambiguous", caught.exception.message)

    def test_balance_uses_the_same_rules(self) -> None:
        """``saldo`` is money too, so ``1.234`` is refused there as well."""
        with self.assertRaises(AmountError):
            parse_balance("1.234")


class ExactDecimalTests(unittest.TestCase):
    """Spec §4.1: a binary float must not be able to enter the domain model."""

    def test_decimal_addition_is_exact(self) -> None:
        """The classic float trap does not occur: 0.1 + 0.2 is exactly 0.3."""
        self.assertEqual(parse_decimal("0.1") + parse_decimal("0.2"), Decimal("0.3"))

    def test_long_fraction_keeps_every_digit(self) -> None:
        """Arbitrary precision: nothing is silently rounded at parse time."""
        text = "0.123456789012345678901234567890"
        self.assertEqual(parse_decimal(text), Decimal(text))

    def test_parsed_value_is_a_decimal_not_a_float(self) -> None:
        """The parsed value's type is ``Decimal``, so no float path exists."""
        self.assertIsInstance(parse_decimal("1.5"), Decimal)
        self.assertNotIsInstance(parse_decimal("1.5"), float)


class DateTests(unittest.TestCase):
    """Spec §4.2: strict ``DD/MM/YYYY`` with real calendar validation."""

    def test_valid_dates_parse(self) -> None:
        """A real date parses to the calendar date it names."""
        self.assertEqual(parse_date("29/02/2024"), date(2024, 2, 29))
        self.assertEqual(parse_date("01/01/2026"), date(2026, 1, 1))
        self.assertEqual(parse_date(" 31/12/2026 "), date(2026, 12, 31))

    def test_invalid_dates_are_rejected(self) -> None:
        """Lenient shapes and impossible days are refused, not rolled over."""
        bad = (
            "31/02/2026",
            "30/02/2026",
            "29/02/2026",
            "1/2/2026",
            "01/2/2026",
            "01/02/26",
            "2026-02-01",
            "01-02-2026",
            "32/01/2026",
            "01/13/2026",
            "00/01/2026",
            "01/00/2026",
            "",
            "abc",
            "01/02/2026x",
        )
        for text in bad:
            with self.subTest(text=text):
                with self.assertRaises(DateError) as caught:
                    parse_date(text)
                self.assertEqual(caught.exception.kind, "DATE")


if __name__ == "__main__":
    unittest.main()
