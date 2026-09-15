package ledger

// ---------------------------------------------------------------------------
// The three baseline checks (spec 3.1 - 3.3)
// ---------------------------------------------------------------------------

final case class ContinuityError(line: Line, balanceExpected: Money, balanceActual: Money, delta: Money)

final case class CumulativeError(line: Line, balanceFromSum: Money, balanceActual: Money, delta: Money)

final case class ValueDateInversion(line: Line, previousLine: Line)

final case class Checks(
    rowCount: Int,
    skippedEmptyRows: List[Line],
    continuityErrors: List[ContinuityError],
    cumulativeErrors: List[CumulativeError],
    valueDateInversions: List[ValueDateInversion]
):
  /**
   * True iff `continuityErrors` and `cumulativeErrors` are empty.
   *
   * The frozen spec said "true iff every array below is empty"; CLARIFICATIONS.md
   * (2026-09-15) makes the binding definition the two error arrays only:
   * `valueDateInversions` and `skippedEmptyRows` are factual observations, not
   * errors, and must not affect this boolean. On the reference file `reconciled`
   * is therefore true even though there are exactly 29 inversions.
   */
  def reconciled: Boolean = continuityErrors.isEmpty && cumulativeErrors.isEmpty

object Checks:

  /** Run all three checks over one statement. Nothing here short-circuits. */
  def run(statement: Statement): Checks =
    Checks(
      rowCount = statement.rowCount,
      skippedEmptyRows = statement.skippedEmptyRows,
      continuityErrors = continuity(statement.transactions),
      cumulativeErrors = statement.openingBalance.map(cumulative(statement.transactions, _)).getOrElse(Nil),
      valueDateInversions = valueDateInversions(statement.transactions)
    )

  /**
   * 3.1 Continuity: `balanceAfter[i] == balanceAfter[i-1] + amount[i]`.
   * Reads only the previous row; it keeps no running total, so it cannot be
   * derived from the cumulative check's state (spec 3.5).
   */
  def continuity(transactions: List[Transaction]): List[ContinuityError] =
    transactions.zip(transactions.drop(1)).flatMap { (previous, current) =>
      val expected = previous.balanceAfter + current.amount
      if expected == current.balanceAfter then Nil
      else List(ContinuityError(current.line, expected, current.balanceAfter, current.balanceAfter - expected))
    }

  /**
   * 3.2 Cumulative: `S(i) = openingBalance + sum(amount[1..i])` compared against
   * `balanceAfter[i]` for every row. Carries its own running sum; it never looks
   * at a neighbouring row's balance to decide what to expect.
   */
  def cumulative(transactions: List[Transaction], openingBalance: Money): List[CumulativeError] =
    val errors = List.newBuilder[CumulativeError]
    var running = openingBalance
    transactions.foreach { transaction =>
      running = running + transaction.amount
      if running != transaction.balanceAfter then
        errors += CumulativeError(
          transaction.line,
          running,
          transaction.balanceAfter,
          transaction.balanceAfter - running
        )
    }
    errors.result()

  /** 3.3 Inversions: adjacent pairs whose value date goes backwards in booking order. */
  def valueDateInversions(transactions: List[Transaction]): List[ValueDateInversion] =
    transactions.zip(transactions.drop(1)).flatMap { (previous, current) =>
      if current.valueDate.isBefore(previous.valueDate) then
        List(ValueDateInversion(current.line, previous.line))
      else Nil
    }

