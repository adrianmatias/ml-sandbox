package ledger

/** The real reference file is the arbiter for spec 3 (spec 8). */
class ReferenceFileSuite extends munit.FunSuite:
  import TestSupport.*

  private lazy val statement = parseOrFail(referenceCsv, "reference file")
  private lazy val checks = Checks.run(statement)

  test("923 transactions, in file order, every non-skipped line present exactly once"):
    assertEquals(statement.rowCount, 923)
    assertEquals(statement.transactions.map(_.line.value), (3 to 925).toList)

  test("skippedEmptyRows is exactly [2]"):
    assertEquals(statement.skippedEmptyRows.map(_.value), List(2))

  test("the final balance reconstructs exactly 37713.30"):
    assertEquals(checks.rowCount, 923)
    assertEquals(statement.finalBalance.map(_.canonical), Some("37713.30"))
    val opening = statement.openingBalance.getOrElse(fail("no opening balance"))
    val fromSum = opening + Money.sum(statement.transactions.map(_.amount))
    assertEquals(fromSum.canonical, "37713.30")

  test("exactly 29 value-date inversions, and booking order has none"):
    assertEquals(checks.valueDateInversions.length, 29)
    val bookingInversions = statement.transactions
      .zip(statement.transactions.drop(1))
      .count((previous, current) => current.bookingDate.isBefore(previous.bookingDate))
    assertEquals(bookingInversions, 0)

  test("92 of 923 rows have booking date different from value date"):
    val differing = statement.transactions.count(t => t.bookingDate.iso != t.valueDate.iso)
    assertEquals(differing, 92)
    assertEquals(statement.transactions.length - differing, 831)

  test("continuityErrors and cumulativeErrors are both empty"):
    assertEquals(checks.continuityErrors, Nil)
    assertEquals(checks.cumulativeErrors, Nil)

  test("reconciled is true even though there are 29 inversions (CLARIFICATIONS.md)"):
    assert(checks.reconciled)
    assertEquals(checks.valueDateInversions.length, 29)

  test("the three duplicate natural keys survive: lines 243/247, 288/290, 446/448"):
    val byLine = statement.transactions.map(t => t.line.value -> t).toMap
    def naturalKey(line: Int) =
      val t = byLine(line)
      (t.bookingDate.iso, t.valueDate.iso, t.amount.canonical, t.balanceAfter.canonical)
    for (left, right) <- List((243, 247), (288, 290), (446, 448)) do
      assertEquals(naturalKey(left), naturalKey(right), s"lines $left/$right")
    // a dedup on natural fields would have produced 920 rows, not 923
    assertEquals(statement.transactions.map(_.checksum).distinct.length, 923 - 3)

  test("checksums are 16 lowercase hex chars and independently reproducible"):
    val first = statement.transactions.head
    assertEquals(first.line.value, 3)
    assertEquals(first.checksum.hex, "ad6223ae9f8b7c1f") // sha256("400.00|400.00|2021-03-11")[0:8]
    assertEquals(
      Checksum.of(money("400.00"), money("400.00"), Dated.booking(date("11/03/2021"))),
      first.checksum
    )
    assert(statement.transactions.forall(_.checksum.hex.matches("[0-9a-f]{16}")))

  test("raw text is preserved exactly as it appeared"):
    assertEquals(statement.transactions.head.rawAmount.text, "400")
    assertEquals(statement.transactions.head.rawBalance.text, "400")
    assertEquals(statement.transactions.last.rawAmount.text, "-9")
    assertEquals(statement.transactions.last.rawBalance.text, "37713.3")

  test("every amount and balance renders canonically"):
    val texts = statement.transactions.flatMap(t => List(t.amount.canonical, t.balanceAfter.canonical))
    assert(texts.forall(_.matches("-?[0-9]+\\.[0-9]{2}")), "all values are plain notation with 2 decimals")
    assert(texts.forall(!_.contains("E")))

  test("dates render as ISO YYYY-MM-DD"):
    assert(statement.transactions.forall(_.bookingDate.iso.matches("[0-9]{4}-[0-9]{2}-[0-9]{2}")))
    assertEquals(statement.transactions.head.bookingDate.iso, "2021-03-11")
