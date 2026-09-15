package ledger

class ChecksSuite extends munit.FunSuite:
  import TestSupport.*

  // openingBalance is anchored at balanceAfter[0] - amount[0] = 0
  private val broken = statementOf(
    transaction(2, "01/01/2026", "01/01/2026", "100", "100"),
    transaction(3, "02/01/2026", "02/01/2026", "-30", "60"), // balance is 10 short
    transaction(4, "03/01/2026", "01/01/2026", "-10", "50") // value date goes backwards
  )

  test("continuity compares each row against the previous row's balance"):
    val errors = Checks.continuity(broken.transactions)
    assertEquals(errors.map(_.line.value), List(3))
    assertEquals(errors.head.balanceExpected.canonical, "70.00")
    assertEquals(errors.head.balanceActual.canonical, "60.00")
    assertEquals(errors.head.delta.canonical, "-10.00")

  test("cumulative carries its own running sum from the opening balance"):
    val errors = Checks.cumulative(broken.transactions, money("0.00"))
    assertEquals(errors.map(_.line.value), List(3, 4))
    assertEquals(errors.head.balanceFromSum.canonical, "70.00")
    assertEquals(errors(1).balanceFromSum.canonical, "60.00")
    assertEquals(errors(1).balanceActual.canonical, "50.00")

  test("the two checks are independent: cumulative reports a row continuity accepts"):
    val statement = broken
    val continuityLines = Checks.continuity(statement.transactions).map(_.line.value).toSet
    val cumulativeLines = Checks.cumulative(statement.transactions, money("0.00")).map(_.line.value).toSet
    assert(!continuityLines.contains(4), "row 4 is continuous with row 3")
    assert(cumulativeLines.contains(4), "but the running sum still disagrees at row 4")

  test("value-date inversions count only backwards steps in booking order"):
    val inversions = Checks.valueDateInversions(broken.transactions)
    assertEquals(inversions.map(i => (i.line.value, i.previousLine.value)), List((4, 3)))

  test("booking order itself has no inversions here"):
    val forward = statementOf(
      transaction(2, "01/01/2026", "05/01/2026", "10", "10"),
      transaction(3, "02/01/2026", "01/01/2026", "10", "20")
    )
    assertEquals(Checks.continuity(forward.transactions), Nil)
    assertEquals(Checks.valueDateInversions(forward.transactions).map(_.line.value), List(3))

  test("no short-circuiting: every break is reported"):
    val many = statementOf(
      transaction(2, "01/01/2026", "01/01/2026", "100", "100"),
      transaction(3, "02/01/2026", "02/01/2026", "-10", "95"), // break
      transaction(4, "03/01/2026", "03/01/2026", "-10", "85"),
      transaction(5, "04/01/2026", "04/01/2026", "-10", "70"), // break
      transaction(6, "05/01/2026", "05/01/2026", "-10", "60")
    )
    assertEquals(Checks.continuity(many.transactions).map(_.line.value), List(3, 5))

  test("a clean ledger reports nothing"):
    val clean = statementOf(
      transaction(2, "01/01/2026", "01/01/2026", "100", "100"),
      transaction(3, "02/01/2026", "03/01/2026", "-30", "70"),
      transaction(4, "03/01/2026", "03/01/2026", "-20", "50")
    )
    val checks = Checks.run(clean)
    assert(checks.reconciled)
    assertEquals(checks.rowCount, 3)
    assertEquals(clean.openingBalance.map(_.canonical), Some("0.00"))
    assertEquals(clean.finalBalance.map(_.canonical), Some("50.00"))
    assertEquals(checks.continuityErrors, Nil)
    assertEquals(checks.cumulativeErrors, Nil)
    assertEquals(checks.valueDateInversions, Nil)

  test("reconciled ignores value-date inversions (CLARIFICATIONS.md)"):
    val inverted = statementOf(
      transaction(2, "01/01/2026", "02/01/2026", "100", "100"),
      transaction(3, "02/01/2026", "01/01/2026", "0", "100")
    )
    val checks = Checks.run(inverted)
    assertEquals(checks.valueDateInversions.length, 1)
    assert(checks.reconciled, "an inversion is an observation, not a reconciliation error")
    assertEquals(checks.skippedEmptyRows, Nil)

  test("empty statement has no anchor and no errors"):
    val checks = Checks.run(Statement(Nil, Nil))
    assert(checks.reconciled)
    assertEquals(checks.rowCount, 0)
