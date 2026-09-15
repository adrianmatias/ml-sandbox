package ledger

import java.time.LocalDate
import java.time.format.DateTimeFormatter
import scala.collection.mutable.ListBuffer
import scala.util.Random

class AccountsSuite extends munit.FunSuite:
  import TestSupport.*

  private val shared = "../spec/shared-fixtures/accounts.json"
  private val DayMonthYear: DateTimeFormatter = DateTimeFormatter.ofPattern("dd/MM/uuuu")

  private def pairs(report: TransferReport): List[(String, Int, String, Int, Int, String)] =
    report.transfers.map(t =>
      (
        t.outflow.accountId.value,
        t.outflow.line.value,
        t.inflow.accountId.value,
        t.inflow.line.value,
        t.lagDays,
        t.amount.canonical
      )
    )

  // -------------------------------------------------------------------------
  // The shared fixture of CLARIFICATIONS.md
  // -------------------------------------------------------------------------

  test("shared fixture: 3 accounts, all EUR, opening balance 0.00"):
    val accounts = loadAccounts(shared)
    assertEquals(accounts.map(_.id.value), List("A", "B", "C"))
    assertEquals(accounts.map(_.currency.code).distinct, List("EUR"))
    assert(accounts.forall(_.openingBalance.isZero))

  test("shared fixture: per-account final balances A 455.25, B 429.50, C 254.50"):
    val accounts = loadAccounts(shared)
    assertEquals(accounts.map(a => a.id.value -> a.closingBalance.canonical).toMap,
      Map("A" -> "455.25", "B" -> "429.50", "C" -> "254.50"))

  test("shared fixture: no continuity or cumulative errors on any account"):
    val accounts = loadAccounts(shared)
    accounts.foreach { account =>
      val checks = Checks.run(account.statement)
      assertEquals(checks.continuityErrors, Nil, account.id.value)
      assertEquals(checks.cumulativeErrors, Nil, account.id.value)
      assert(checks.reconciled, account.id.value)
    }

  test("shared fixture: exactly the two expected transfer pairs"):
    val report = TransferDetection.detect(loadAccounts(shared), 3)
    assertEquals(pairs(report), List(
      ("A", 3, "B", 4, 1, "250.00"),
      ("A", 4, "C", 2, 0, "300.00")
    ))
    assertEquals(report.transfers.length, 2)

  test("shared fixture: net worth is 1139.25 by both methods"):
    val accounts = loadAccounts(shared)
    val netWorth = NetWorth.observe(accounts)
    assertEquals(netWorth.total.netWorth.canonical, "1139.25")
    assertEquals(netWorth.total.netWorthFromBalances.canonical, "1139.25")
    assert(netWorth.total.agrees)
    assertEquals(netWorth.mismatches, Nil)
    assert(netWorth.points.forall(_.agrees))

  test("shared fixture: transfers conserve net worth exactly"):
    val report = TransferDetection.detect(loadAccounts(shared), 3)
    assert(report.netWorthConserved)
    assertEquals(report.conservationResidual.canonical, "0.00")
    assertEquals(report.unpairedTransferCandidates, 0)

  test("shared fixture: the decoys are not paired"):
    val report = TransferDetection.detect(loadAccounts(shared), 3)
    val paired = report.transfers.flatMap(t => List((t.outflow.accountId.value, t.outflow.line.value), (t.inflow.accountId.value, t.inflow.line.value))).toSet
    // A2/B2 opening credits, B3 -250, A5 +120.50, B6/C4 -120.50, C3 +75
    assert(!paired.contains(("A", 2)) && !paired.contains(("B", 2)))
    assert(!paired.contains(("B", 3)))
    assert(!paired.contains(("A", 5)))
    assert(!paired.contains(("B", 6)) && !paired.contains(("C", 4)))
    assert(!paired.contains(("C", 3)))

  // -------------------------------------------------------------------------
  // Own fixtures
  // -------------------------------------------------------------------------

  test("own fixture: nearest date, tolerance boundary, and currency isolation"):
    val accounts = loadAccounts("fixtures/accounts/registry.json")
    assertEquals(accounts.length, 3)
    val report = TransferDetection.detect(accounts, 3)
    assertEquals(pairs(report), List(
      ("ALPHA", 3, "BETA", 3, 1, "150.00"),
      ("ALPHA", 4, "BETA", 4, 3, "400.00") // exactly at tolerance 3
    ))
    // ALPHA line 5 (-60, value date 20/01) vs BETA line 5 (+60, value date 24/01): gap 4 > 3
    // ALPHA line 3 (-150 EUR) vs GAMMA line 2 (+150 USD): different currency
    assert(report.netWorthConserved)

  test("tolerance is configurable"):
    val accounts = loadAccounts("fixtures/accounts/registry.json")
    assertEquals(TransferDetection.detect(accounts, 2).transfers.length, 1)
    assertEquals(TransferDetection.detect(accounts, 4).transfers.length, 3)

  test("tie-break: equal gaps resolve to the lowest line number"):
    val report = TransferDetection.detect(loadAccounts("fixtures/tiebreak/registry.json"), 3)
    assertEquals(pairs(report), List(("PAYER", 2, "RECV", 2, 1, "500.00")))

  test("nearest feasible date beats a lower line number"):
    val report = TransferDetection.detect(loadAccounts("fixtures/nearest/registry.json"), 3)
    assertEquals(pairs(report), List(("PAYER", 2, "RECV", 3, 0, "700.00")))

  test("a third account needs no change to the parsing code"):
    val two = loadAccounts("fixtures/accounts/registry-2.json")
    val three = loadAccounts("fixtures/accounts/registry.json")
    assertEquals(two.length, 2)
    assertEquals(three.length, 3)
    val twoReport = TransferDetection.detect(two, 3)
    val threeReport = TransferDetection.detect(three, 3)
    assertEquals(pairs(threeReport), pairs(twoReport))
    assert(threeReport.netWorthConserved)

  test("a registry opening balance that disagrees with the statement is reported"):
    val accounts = loadAccounts("fixtures/mismatch/registry.json")
    accounts.foreach { account =>
      val checks = Checks.run(account.statement)
      assertEquals(checks.continuityErrors, Nil, account.id.value)
      assertEquals(checks.cumulativeErrors, Nil, account.id.value)
    }
    val netWorth = NetWorth.observe(accounts)
    assertEquals(netWorth.mismatches.map(_.scope), List("M1", "TOTAL"))
    assertEquals(netWorth.mismatches.head.delta.canonical, "1.00")
    assertEquals(netWorth.points.find(_.account.value == "M1").map(_.netWorth.canonical), Some("1049.00"))
    assertEquals(netWorth.points.find(_.account.value == "M1").map(_.netWorthFromBalances.canonical), Some("1050.00"))

  test("the default statement is <id>.csv, and unknown fields are ignored"):
    val registry = registryOf(shared)
    assertEquals(registry.toleranceDays, 3)
    assertEquals(registry.accounts.map(_.statementPath), List("A.csv", "B.csv", "C.csv"))

  test("registry errors are REGISTRY kind and never partial"):
    val cases = List(
      "[]" -> "must be an object",
      "{}" -> "no \"accounts\"",
      """{"accounts": []}""" -> "no accounts",
      """{"accounts": [{"id": "A", "currency": "eur", "openingBalance": "0.00"}]}""" -> "uppercase",
      """{"accounts": [{"id": "A", "currency": "EUR", "openingBalance": 1000}]}""" -> "must be a string",
      """{"accounts": [{"id": "A", "currency": "EUR", "openingBalance": "1.234"}]}""" -> "ambiguous",
      """{"accounts": [{"id": "A", "currency": "EUR", "openingBalance": "0.00"}, {"id": "A", "currency": "EUR", "openingBalance": "0.00"}]}""" -> "duplicate",
      """{"toleranceDays": -1, "accounts": [{"id": "A", "currency": "EUR", "openingBalance": "0.00"}]}""" -> "non-negative",
      """{"accounts": [{"currency": "EUR", "openingBalance": "0.00"}]}""" -> "missing \"id\""
    )
    cases.foreach { (json, fragment) =>
      Registry.parse(json) match
        case Right(_) => fail(s"should have been rejected: $json")
        case Left(errors) =>
          assert(errors.all.forall(_.kind == ErrorKind.REGISTRY), json)
          assert(errors.all.exists(_.message.contains(fragment)), s"$json -> ${errors.all.map(_.message)}")
    }

  test("a JSON number for openingBalance is refused: no binary float enters the domain"):
    // a fractional JSON number never even reaches the field: the reader has no float case
    Registry.parse("""{"accounts": [{"id": "A", "currency": "EUR", "openingBalance": 1000.00}]}""") match
      case Left(errors) =>
        assert(errors.all.exists(_.message.contains("non-integer number")), errors.all.map(_.message))
      case Right(_) => fail("a fractional JSON number must be rejected")
    // an integer JSON number reaches the field and is refused there, a string being required
    Registry.parse("""{"accounts": [{"id": "A", "currency": "EUR", "openingBalance": 1000}]}""") match
      case Left(errors) =>
        assert(errors.all.exists(_.message.contains("must be a string")), errors.all.map(_.message))
      case Right(_) => fail("openingBalance must be a string")

  // -------------------------------------------------------------------------
  // Property: pairing invariants hold on generated ledgers
  // -------------------------------------------------------------------------

  private def statementFrom(rows: List[(LocalDate, LocalDate, BigDecimal)]): Statement =
    val sb = new StringBuilder("fecha,fecha valor,importe,saldo\n")
    var balance = BigDecimal(0)
    rows.foreach { (booking, value, amount) =>
      balance = balance + amount
      sb.append(DayMonthYear.format(booking))
        .append(',')
        .append(DayMonthYear.format(value))
        .append(',')
        .append(amount.bigDecimal.toPlainString)
        .append(',')
        .append(balance.bigDecimal.toPlainString)
        .append('\n')
    }
    parseOrFail(sb.toString)

  private def account(id: String, currency: String, statement: Statement): Account =
    Account(AccountId(id), Currency.parse(currency).toOption.get, Money.zero, s"$id.csv", statement)

  private def generatedAccounts(seed: Int): (List[Account], Int) =
    val rng = new Random(seed)
    val tolerance = rng.nextInt(4)
    val a = ListBuffer.empty[(LocalDate, LocalDate, BigDecimal)]
    val b = ListBuffer.empty[(LocalDate, LocalDate, BigDecimal)]
    val c = ListBuffer.empty[(LocalDate, LocalDate, BigDecimal)]
    for _ <- 1 to 1 + rng.nextInt(6) do
      val amount = BigDecimal(1 + rng.nextInt(50000)) / 100
      val day = LocalDate.of(2026, 1, 1).plusDays(rng.nextInt(30))
      val lag = rng.nextInt(6)
      a += ((day, day, -amount))
      b += ((day, day.plusDays(lag), amount))
    for _ <- 1 to rng.nextInt(5) do
      val day = LocalDate.of(2026, 1, 1).plusDays(rng.nextInt(30))
      val amount = BigDecimal(rng.nextInt(100000)) / 100
      a += ((day, day, amount))
      b += ((day, day, -amount))
    c += ((LocalDate.of(2026, 1, 5), LocalDate.of(2026, 1, 5), BigDecimal(250)))
    List(
      account("A", "EUR", statementFrom(a.toList)),
      account("B", "EUR", statementFrom(b.toList)),
      account("C", "USD", statementFrom(c.toList))
    ) -> tolerance

  test("property: pairing never reuses a row, always satisfies the predicate, and conserves"):
    for seed <- 1 to 200 do
      val (accounts, tolerance) = generatedAccounts(seed)
      val report = TransferDetection.detect(accounts, tolerance)

      assertEquals(report.conservationResidual, Money.zero, s"seed $seed")
      assert(report.netWorthConserved, s"seed $seed")
      assertEquals(report.unpairedTransferCandidates, 0, s"seed $seed")

      val used = report.transfers.flatMap(t => List((t.outflow.accountIndex, t.outflow.line.value), (t.inflow.accountIndex, t.inflow.line.value)))
      assertEquals(used.distinct.length, used.length, s"seed $seed: a row was used twice")

      report.transfers.foreach { transfer =>
        assert(transfer.outflow.accountIndex != transfer.inflow.accountIndex, s"seed $seed")
        assertEquals(accounts(transfer.outflow.accountIndex).currency, accounts(transfer.inflow.accountIndex).currency, s"seed $seed")
        assert(transfer.outflow.amount.isNegative, s"seed $seed")
        assert(transfer.inflow.amount.isPositive, s"seed $seed")
        assertEquals(transfer.outflow.amount.abs, transfer.inflow.amount, s"seed $seed")
        assertEquals(transfer.amount, transfer.inflow.amount, s"seed $seed")
        assert(transfer.lagDays >= 0 && transfer.lagDays <= tolerance, s"seed $seed")
        assertEquals(transfer.outflow.valueDate.daysUntil(transfer.inflow.valueDate), transfer.lagDays, s"seed $seed")
      }

      assertEquals(TransferDetection.detect(accounts, tolerance), report, s"seed $seed: not deterministic")

  test("property: transfers between accounts with different currencies never pair"):
    val eur = account("E", "EUR", statementFrom(List((LocalDate.of(2026, 1, 5), LocalDate.of(2026, 1, 5), BigDecimal(-100)))))
    val usd = account("U", "USD", statementFrom(List((LocalDate.of(2026, 1, 5), LocalDate.of(2026, 1, 5), BigDecimal(100)))))
    val report = TransferDetection.detect(List(eur, usd), 3)
    assertEquals(report.transfers, Nil)
    assert(report.netWorthConserved)
