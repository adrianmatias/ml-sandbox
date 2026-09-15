package ledger

import java.time.LocalDate
import java.time.format.DateTimeFormatter
import scala.util.Random

/**
 * Property-style tests over a small generated domain: random but *valid* ledgers,
 * built from a running balance, so a correct implementation must report nothing.
 */
class PropertySuite extends munit.FunSuite:
  import TestSupport.*

  private val DayMonthYear: DateTimeFormatter = DateTimeFormatter.ofPattern("dd/MM/uuuu")

  /** A ledger whose balances are consistent by construction. */
  private def generatedStatement(seed: Int): Statement =
    val rng = new Random(seed)
    val rows = 1 + rng.nextInt(25)
    val sb = new StringBuilder("fecha,fecha valor,importe,saldo\n")
    var balance = BigDecimal(0)
    var day = LocalDate.of(2026, 1, 1)
    for _ <- 1 to rows do
      day = day.plusDays(1 + rng.nextInt(4))
      val amount = BigDecimal(rng.nextInt(200001) - 100000) / 100
      balance = balance + amount
      sb.append(DayMonthYear.format(day))
        .append(',')
        .append(DayMonthYear.format(day))
        .append(',')
        .append(amount.bigDecimal.toPlainString)
        .append(',')
        .append(balance.bigDecimal.toPlainString)
        .append('\n')
    parseOrFail(sb.toString, s"seed $seed")

  test("property: a ledger built from its own running balance is always reconciled"):
    for seed <- 1 to 300 do
      val statement = generatedStatement(seed)
      val checks = Checks.run(statement)
      assertEquals(checks.continuityErrors, Nil, s"seed $seed")
      assertEquals(checks.cumulativeErrors, Nil, s"seed $seed")
      assert(checks.reconciled, s"seed $seed")
      assert(statement.rowCount >= 1, s"seed $seed")

  test("property: the two independent checks agree on the final balance"):
    for seed <- 1 to 300 do
      val statement = generatedStatement(seed)
      val opening = statement.openingBalance.getOrElse(fail(s"seed $seed: no opening balance"))
      val fromSum = opening + Money.sum(statement.transactions.map(_.amount))
      assertEquals(fromSum, statement.finalBalance.get, s"seed $seed")

  test("property: perturbing one balance is reported by both checks at the expected lines"):
    for seed <- 1 to 200 do
      val statement = generatedStatement(seed)
      val index = new Random(seed * 31 + 7).nextInt(statement.transactions.length)
      val original = statement.transactions(index)
      val perturbed = original.copy(balanceAfter = original.balanceAfter + money("1.00"))
      val transactions = statement.transactions.updated(index, perturbed)

      val cumulative = Checks.cumulative(transactions, statement.openingBalance.get)
      assert(cumulative.nonEmpty, s"seed $seed")
      assertEquals(cumulative.head.line, original.line, s"seed $seed")
      assertEquals(cumulative.head.delta.canonical, "1.00", s"seed $seed")
      // the cumulative check is anchored on amounts, so the wrong balance does not
      // propagate: exactly one row disagrees with the running sum
      assertEquals(cumulative.map(_.line), List(original.line), s"seed $seed")

      // the continuity check walks balances, so the wrong balance breaks its own row
      // (if it has a predecessor) and the row after it, by +1 and -1 respectively
      val continuity = Checks.continuity(transactions)
      val expectedContinuity =
        (if index > 0 then List(original.line -> "1.00") else Nil) ++
          (if index + 1 < transactions.length then List(transactions(index + 1).line -> "-1.00") else Nil)
      assertEquals(continuity.map(e => e.line -> e.delta.canonical), expectedContinuity, s"seed $seed")

  test("property: canonical rendering round-trips through the parser exactly"):
    val rng = new Random(20260915L)
    for _ <- 1 to 500 do
      val cents = rng.nextLong() % 1000000000L
      val value = Money.wrap(BigDecimal(cents) / 100)
      val text = value.canonical
      assert(text.matches("-?[0-9]+\\.[0-9]{2}"), text)
      assertEquals(AmountParser.parseText(text), Right(value))
      assertEquals(value + Money.zero, value)

  test("property: money addition is associative and commutative on the generated domain"):
    val rng = new Random(7L)
    for _ <- 1 to 200 do
      val a = Money.wrap(BigDecimal(rng.nextInt(2000001) - 1000000) / 100)
      val b = Money.wrap(BigDecimal(rng.nextInt(2000001) - 1000000) / 100)
      val c = Money.wrap(BigDecimal(rng.nextInt(2000001) - 1000000) / 100)
      assertEquals((a + b) + c, a + (b + c))
      assertEquals(a + b, b + a)
      assertEquals(a - b, -(b - a))
      assertEquals(Money.sum(List(a, b, c)), a + b + c)

  test("property: checksums are stable and depend only on the three inputs"):
    val rng = new Random(99L)
    for _ <- 1 to 100 do
      val amount = Money.wrap(BigDecimal(rng.nextInt(2000001) - 1000000) / 100)
      val balance = Money.wrap(BigDecimal(rng.nextInt(2000001) - 1000000) / 100)
      val booking = Dated.booking(LocalDate.of(2026, 1, 1).plusDays(rng.nextInt(365)))
      val checksum = Checksum.of(amount, balance, booking)
      assert(checksum.hex.matches("[0-9a-f]{16}"), checksum.hex)
      assertEquals(Checksum.of(amount, balance, booking), checksum)
      if !amount.isZero then
        assert(Checksum.of(amount + money("0.01"), balance, booking) != checksum)
