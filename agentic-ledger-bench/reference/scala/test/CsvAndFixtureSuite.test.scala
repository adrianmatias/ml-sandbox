package ledger

class CsvSuite extends munit.FunSuite:
  import TestSupport.*

  private def fields(line: String): List[String] =
    Csv.split(line, Line(1)).fold(e => fail(e.message), identity)

  test("plain fields"):
    assertEquals(fields("a,b,c,d"), List("a", "b", "c", "d"))

  test("quoted fields keep separators and drop the quotes"):
    assertEquals(fields("\"a,b\",c,d,e"), List("a,b", "c", "d", "e"))

  test("a doubled quote inside a quoted field is one literal quote"):
    assertEquals(fields("\"he said \"\"hi\"\"\",b,c,d"), List("he said \"hi\"", "b", "c", "d"))

  test("empty quoted field is empty, not missing"):
    assertEquals(fields("\"\",b,c,d"), List("", "b", "c", "d"))

  test("unterminated quote is a ROW error"):
    assertEquals(Csv.split("\"a,b,c", Line(4)).left.map(_.kind), Left(ErrorKind.ROW))

  test("quote inside an unquoted field is a ROW error"):
    assertEquals(Csv.split("a\"b,c,d,e", Line(4)).left.map(_.kind), Left(ErrorKind.ROW))

  test("lines: LF and CRLF, trailing newline, BOM"):
    assertEquals(Csv.lines("a\nb\n"), List("a", "b"))
    assertEquals(Csv.lines("a\r\nb\r\n"), List("a", "b"))
    assertEquals(Csv.lines("a\r\nb"), List("a", "b"))
    assertEquals(Csv.lines("\uFEFFa\nb"), List("a", "b"))
    assertEquals(Csv.lines("a"), List("a"))

  test("blank records"):
    assert(Csv.isBlank(List("", "", "", "")))
    assert(Csv.isBlank(List("  ", "")))
    assert(!Csv.isBlank(List("", "0", "", "")))
    assert(Csv.isBlank(List("")))

class FixtureSuite extends munit.FunSuite:
  import TestSupport.*

  test("fully empty row `,,,` is skipped, not an error"):
    val statement = parseOrFail(fixture("edge/empty_row.csv"))
    assertEquals(statement.transactions.map(_.line.value), List(2, 4))
    assertEquals(statement.skippedEmptyRows.map(_.value), List(3))
    val checks = Checks.run(statement)
    assert(checks.reconciled)
    assertEquals(checks.continuityErrors, Nil)

  test("quoted fields parse, including a comma decimal inside quotes"):
    val statement = parseOrFail(fixture("edge/quoted_field.csv"))
    assertEquals(statement.transactions.map(_.amount.canonical), List("100.00", "-50.25"))
    assertEquals(statement.transactions.map(_.balanceAfter.canonical), List("100.00", "49.75"))
    assert(Checks.run(statement).reconciled)

  test("comma decimal separator"):
    val statement = parseOrFail(fixture("edge/comma_decimal.csv"))
    assertEquals(statement.transactions.map(_.amount.canonical), List("1234.56", "-34.56"))
    assertEquals(statement.transactions.map(_.amount), List(money("1234.56"), money("-34.56")))
    assertEquals(statement.transactions.map(_.rawAmount.text), List("1234,56", "-34,56"))
    assert(Checks.run(statement).reconciled)

  test("an unquoted comma decimal is a column-count error, not an amount error"):
    val error = errorOfFixture("edge/comma_decimal_unquoted.csv")
    assertEquals(error.kind, ErrorKind.ROW)
    assertEquals(error.line.map(_.value), Some(2))
    assert(error.message.contains("found 6"), error.message)

  test("malformed amounts are rejected"):
    assertEquals(errorOfFixture("edge/amount_abc.csv").kind, ErrorKind.AMOUNT)
    assertEquals(errorOfFixture("edge/amount_multi_dot.csv").kind, ErrorKind.AMOUNT)
    assertEquals(errorOfFixture("edge/amount_multi_sign.csv").kind, ErrorKind.AMOUNT)

  test("ambiguous 1.234 is rejected, never guessed"):
    val error = errorOfFixture("edge/amount_ambiguous.csv")
    assertEquals(error.kind, ErrorKind.AMOUNT)
    assertEquals(error.line.map(_.value), Some(2))
    assert(error.message.contains("ambiguous"), error.message)
    assertEquals(errorOfFixture("edge/amount_both_separators.csv").kind, ErrorKind.AMOUNT)

  test("missing saldo is a row error at the right line"):
    val error = errorOfFixture("edge/missing_saldo.csv")
    assertEquals(error.kind, ErrorKind.ROW)
    assertEquals(error.line.map(_.value), Some(3))
    assert(error.message.contains("found 3"), error.message)

  test("bad dates are rejected at the right lines, both dates of a row independently"):
    val errors = errorsOf(fixture("edge/bad_date.csv"))
    // line 2 has 31/02/2026 in both date columns -> two DATE errors, not one
    assertEquals(errors.map(_.kind), List(ErrorKind.DATE, ErrorKind.DATE, ErrorKind.DATE))
    assertEquals(errors.map(_.line.map(_.value)), List(Some(2), Some(2), Some(3)))
    assert(errors.head.message.contains("31/02/2026"))
    assert(errors(2).message.contains("2026-01-01"))

  test("CRLF line endings"):
    val statement = parseOrFail(fixture("edge/crlf.csv"))
    assertEquals(statement.transactions.length, 2)
    assertEquals(statement.transactions.map(_.rawAmount.text), List("100", "-50"))

  test("UTF-8 BOM is stripped from the header"):
    val statement = parseOrFail(fixture("edge/bom.csv"))
    assertEquals(statement.transactions.length, 1)
    assertEquals(statement.transactions.head.amount, money("100.00"))

  test("a (bookingDate, amount, balanceAfter) collision keeps both rows"):
    val statement = parseOrFail(fixture("edge/collision.csv"))
    assertEquals(statement.transactions.length, 3)
    val first = statement.transactions(0)
    val third = statement.transactions(2)
    assertEquals(first.bookingDate.iso, third.bookingDate.iso)
    assertEquals(first.amount, third.amount)
    assertEquals(first.balanceAfter, third.balanceAfter)
    assertEquals(first.checksum, third.checksum)
    assert(Checks.run(statement).reconciled)
    // a dedup keyed on natural fields would return 2 rows here
    assertEquals(statement.transactions.map(_.line.value), List(2, 3, 4))

  test("every violation is accumulated, not just the first"):
    val errors = errorsOf(fixture("edge/multi_error.csv"))
    assertEquals(errors.map(_.line.map(_.value)), List(Some(2), Some(3), Some(4), Some(5)))
    assertEquals(errors.map(_.kind), List(ErrorKind.AMOUNT, ErrorKind.DATE, ErrorKind.AMOUNT, ErrorKind.ROW))

  test("header only: no rows, no errors, reconciled"):
    val statement = parseOrFail(fixture("edge/header_only.csv"))
    assertEquals(statement.rowCount, 0)
    assertEquals(statement.openingBalance, None)
    val checks = Checks.run(statement)
    assert(checks.reconciled)
    assertEquals(checks.rowCount, 0)

  test("wrong header is a HEADER error"):
    val error = errorOfFixture("edge/bad_header.csv")
    assertEquals(error.kind, ErrorKind.HEADER)
    assert(error.message.contains("fecha,fecha valor,importe,saldo"))
