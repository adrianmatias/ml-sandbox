"""Spec §6 — the CLI contract: interface, envelope, exit codes and determinism."""

import json
import unittest

from .helpers import (
    EXPECTED_FINAL_BALANCE,
    EXPECTED_ROW_COUNT,
    ORIGINAL_EXPORT,
    RAW_CSV,
    SHARED_FIXTURES,
    as_dict,
    as_int,
    as_list,
    as_str,
    checks_of,
    data_of,
    fixture,
    member,
    run_cli,
    scratch_dir,
)


class HelpAndUsageTests(unittest.TestCase):
    """Usage errors exit 2; help exits 0 (spec §6)."""

    def test_help_exits_zero(self) -> None:
        """``--help`` prints the interface and succeeds."""
        result = run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("baseline", result.stdout)
        self.assertIn("accounts", result.stdout)

    def test_short_help_exits_zero(self) -> None:
        """``-h`` behaves like ``--help``."""
        self.assertEqual(run_cli("-h").returncode, 0)

    def test_no_arguments_is_a_usage_error(self) -> None:
        """No command at all is a usage error, not a crash and not success."""
        result = run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage", result.stderr)

    def test_missing_file_argument_is_a_usage_error(self) -> None:
        """``baseline`` without a path is a usage error."""
        self.assertEqual(run_cli("baseline").returncode, 2)

    def test_unknown_command_is_a_usage_error(self) -> None:
        """An unknown command is a usage error."""
        result = run_cli("reconcile", str(RAW_CSV))
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown command", result.stderr)


class EnvelopeTests(unittest.TestCase):
    """The §6 envelope: key order, members and value shapes."""

    def test_success_envelope_key_order_is_frozen(self) -> None:
        """``ok``, ``module``, ``version``, ``data`` — in that exact order."""
        result = run_cli("baseline", str(RAW_CSV))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            list(json.loads(result.stdout).keys()),
            ["ok", "module", "version", "data"],
        )

    def test_error_envelope_key_order_is_frozen(self) -> None:
        """``ok``, ``module``, ``version``, ``error`` — in that exact order."""
        result = run_cli("baseline", str(fixture("does_not_exist.csv")))
        self.assertEqual(
            list(json.loads(result.stdout).keys()),
            ["ok", "module", "version", "error"],
        )

    def test_module_name_and_version_are_stable(self) -> None:
        """Both implementations share ``module``/``version`` verbatim."""
        envelope = run_cli("baseline", str(RAW_CSV)).json
        self.assertEqual(envelope["module"], "ledger")
        self.assertEqual(envelope["version"], 1)

    def test_transactions_carry_every_required_field(self) -> None:
        """Each §3 transaction entry has all eight frozen fields."""
        data = data_of(run_cli("baseline", str(RAW_CSV)).json)
        transactions = as_list(member(data, "transactions"), "transactions")
        first = as_dict(transactions[0], "transactions[0]")
        self.assertEqual(
            list(first.keys()),
            [
                "line",
                "bookingDate",
                "valueDate",
                "amount",
                "balanceAfter",
                "rawAmount",
                "rawBalance",
                "checksum",
            ],
        )

    def test_checks_object_carries_every_required_field(self) -> None:
        """The §3 ``checks`` object has all six specified members plus parse errors."""
        checks = checks_of(run_cli("baseline", str(RAW_CSV)).json)
        for key in (
            "rowCount",
            "skippedEmptyRows",
            "reconciled",
            "continuityErrors",
            "cumulativeErrors",
            "valueDateInversions",
        ):
            self.assertIn(key, checks)

    def test_money_is_always_a_canonical_two_place_string(self) -> None:
        """Every serialised amount keeps exactly two decimal places."""
        data = data_of(run_cli("baseline", str(RAW_CSV)).json)
        transactions = as_list(member(data, "transactions"), "transactions")
        for index, entry in enumerate(transactions):
            row = as_dict(entry, f"transactions[{index}]")
            for key in ("amount", "balanceAfter"):
                text = as_str(member(row, key), key)
                self.assertRegex(text, r"^-?\d+\.\d{2}$")
                self.assertNotIn("E", text)

    @unittest.skipUnless(ORIGINAL_EXPORT, "pins a fact about the original export")
    def test_raw_fields_are_preserved_verbatim(self) -> None:
        """``rawAmount``/``rawBalance`` repeat the file's own spelling."""
        data = data_of(run_cli("baseline", str(RAW_CSV)).json)
        transactions = as_list(member(data, "transactions"), "transactions")
        last = as_dict(transactions[-1], "transactions[-1]")
        self.assertEqual(last["rawAmount"], "-9")
        self.assertEqual(last["rawBalance"], "37713.3")
        self.assertEqual(last["balanceAfter"], EXPECTED_FINAL_BALANCE)


class ErrorEnvelopeTests(unittest.TestCase):
    """§6 error kinds and exit codes for unreadable and unparsable input."""

    def test_missing_file_is_an_io_error_with_exit_1(self) -> None:
        """An unreadable file exits 1 with ``kind: IO``."""
        result = run_cli("baseline", str(fixture("nope.csv")))
        self.assertEqual(result.returncode, 1)
        error = as_dict(member(result.json, "error"), "error")
        self.assertEqual(error["kind"], "IO")

    def test_bad_header_is_a_header_error_with_exit_1(self) -> None:
        """A wrong header exits 1 with ``kind: HEADER`` and names line 1."""
        with scratch_dir("bad-header") as scratch:
            path = scratch / "bad.csv"
            path.write_text(
                "a,b,c,d\n01/01/2026,01/01/2026,1,1\n", encoding="utf-8", newline=""
            )
            result = run_cli("baseline", str(path))
        self.assertEqual(result.returncode, 1)
        error = as_dict(member(result.json, "error"), "error")
        self.assertEqual(error["kind"], "HEADER")
        self.assertEqual(error["line"], 1)

    def test_empty_file_is_a_header_error(self) -> None:
        """A file with no lines at all is a header failure, not an index error."""
        with scratch_dir("empty-file") as scratch:
            path = scratch / "empty.csv"
            path.write_text("", encoding="utf-8", newline="")
            result = run_cli("baseline", str(path))
        self.assertEqual(result.returncode, 1)
        error = as_dict(member(result.json, "error"), "error")
        self.assertEqual(error["kind"], "HEADER")

    def test_unclosed_quote_is_a_row_error(self) -> None:
        """Malformed CSV quoting exits 1 with ``kind: ROW``."""
        with scratch_dir("bad-csv") as scratch:
            path = scratch / "bad.csv"
            path.write_text(
                'fecha,fecha valor,importe,saldo\n01/01/2026,01/01/2026,"1,1\n',
                encoding="utf-8",
                newline="",
            )
            result = run_cli("baseline", str(path))
        self.assertEqual(result.returncode, 1)
        error = as_dict(member(result.json, "error"), "error")
        self.assertEqual(error["kind"], "ROW")


class BehaviourTests(unittest.TestCase):
    """Contract behaviours the spec fixes explicitly."""

    def test_reference_file_exits_zero_despite_the_inversions(self) -> None:
        """A well-formed file exits 0 even when checks report observations."""
        result = run_cli("baseline", str(RAW_CSV))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(checks_of(result.json)["rowCount"], EXPECTED_ROW_COUNT)

    def test_two_runs_are_byte_identical(self) -> None:
        """Determinism: the same input produces the same bytes, twice."""
        first = run_cli("baseline", str(RAW_CSV))
        second = run_cli("baseline", str(RAW_CSV))
        self.assertEqual(first.stdout, second.stdout)

    def test_output_contains_no_absolute_paths(self) -> None:
        """No timestamp, no absolute path, no host-specific text (spec §6)."""
        result = run_cli("baseline", str(RAW_CSV))
        self.assertNotIn(str(RAW_CSV.parent), result.stdout)
        self.assertNotIn("/home/", result.stdout)

    def test_error_arrays_are_ordered_by_line(self) -> None:
        """Reported line numbers ascend, per §6's ordering rule."""
        checks = checks_of(run_cli("baseline", str(fixture("edge_cases.csv"))).json)
        errors = as_list(member(checks, "parseErrors"), "parseErrors")
        lines = [
            as_int(member(as_dict(entry, "parseError"), "line"), "line")
            for entry in errors
        ]
        self.assertEqual(lines, sorted(lines))

    def test_accounts_command_runs_on_the_shared_fixture(self) -> None:
        """``accounts`` accepts the shared registry and reports three accounts."""
        result = run_cli("accounts", str(SHARED_FIXTURES / "accounts.json"))
        self.assertEqual(result.returncode, 0, result.stdout)
        data = data_of(result.json)
        accounts = as_list(member(data, "accounts"), "accounts")
        self.assertEqual(len(accounts), 3)

    def test_accounts_command_reports_a_registry_error(self) -> None:
        """A malformed registry exits 1 with ``kind: REGISTRY``."""
        with scratch_dir("bad-registry") as scratch:
            path = scratch / "accounts.json"
            path.write_text(
                '{"version": 2, "accounts": []}', encoding="utf-8", newline=""
            )
            result = run_cli("accounts", str(path))
        self.assertEqual(result.returncode, 1)
        error = as_dict(member(result.json, "error"), "error")
        self.assertEqual(error["kind"], "REGISTRY")


if __name__ == "__main__":
    unittest.main()
