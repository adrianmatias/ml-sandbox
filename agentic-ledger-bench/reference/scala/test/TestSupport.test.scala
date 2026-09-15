package ledger

import java.nio.file.{Files, Path}
import java.nio.charset.StandardCharsets

/** Shared helpers for the suites: locating fixtures and failing loudly when absent. */
object TestSupport:

  private def locate(relative: String): Path =
    val candidates = List(
      Path.of(relative),
      Path.of("scala").resolve(relative),
      Path.of("..").resolve(relative),
      Path.of("../..").resolve(relative)
    )
    candidates.find(Files.exists(_)).getOrElse {
      throw new AssertionError(
        s"cannot locate $relative (cwd = ${Path.of("").toAbsolutePath}); tried ${candidates.mkString(", ")}"
      )
    }

  def text(relative: String): String =
    new String(Files.readAllBytes(locate(relative)), StandardCharsets.UTF_8)

  /** Absolute path of a located fixture, for tests that call the CLI by filename. */
  def path(relative: String): String = locate(relative).toString

  /** A file under `fixtures/`. */
  def fixture(name: String): String = text(s"fixtures/$name")

  /** The read-only reference statement shipped with the spec. */
  def referenceCsv: String = text("../raw/kutxa_movimientos_2026-05-25.csv")

  def parseOrFail(source: String, label: String = "<inline>"): Statement =
    StatementParser.parse(source) match
      case Right(statement) => statement
      case Left(errors) =>
        throw new AssertionError(s"$label should have parsed, got: ${errors.all.map(_.message).mkString("; ")}")

  def errorsOf(source: String, label: String = "<inline>"): List[ParseError] =
    StatementParser.parse(source) match
      case Left(errors) => errors.all
      case Right(_) => throw new AssertionError(s"$label should have failed to parse")

  def errorOfFixture(name: String): ParseError = errorsOf(fixture(name), name).head

  def money(text: String): Money =
    AmountParser.parseText(text).fold(message => throw new AssertionError(message), identity)

  def date(text: String): java.time.LocalDate =
    DateParser.parseText(text).fold(message => throw new AssertionError(message), identity)

  /** Build a transaction directly, for check-level tests. */
  def transaction(
      line: Int,
      booking: String,
      value: String,
      amount: String,
      balance: String
  ): Transaction =
    Transaction(
      line = Line(line),
      bookingDate = Dated.booking(date(booking)),
      valueDate = Dated.value(date(value)),
      amount = money(amount),
      balanceAfter = money(balance),
      rawAmount = Raw(amount),
      rawBalance = Raw(balance)
    )

  def statementOf(transactions: Transaction*): Statement = Statement(transactions.toList, Nil)

  /** Parse a registry and load every statement it names. */
  def loadAccounts(registryRelative: String): List[Account] =
    val file = locate(registryRelative)
    val body = new String(Files.readAllBytes(file), StandardCharsets.UTF_8)
    val registry = Registry.parse(body) match
      case Right(value) => value
      case Left(errors) =>
        throw new AssertionError(s"$registryRelative: ${errors.all.map(_.message).mkString("; ")}")
    AccountsLoader.load(registry, file.getParent) match
      case Right(accounts) => accounts
      case Left(errors) =>
        throw new AssertionError(s"$registryRelative: ${errors.all.map(_.message).mkString("; ")}")

  def registryOf(registryRelative: String): Registry =
    val file = locate(registryRelative)
    val body = new String(Files.readAllBytes(file), StandardCharsets.UTF_8)
    Registry.parse(body).fold(
      errors => throw new AssertionError(errors.all.map(_.message).mkString("; ")),
      identity
    )
