package ledger

import java.math.{MathContext, RoundingMode}
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.time.temporal.ChronoUnit
import scala.math.BigDecimal

/** 1-based physical line number in the source CSV; the header is line 1. */
final case class Line(value: Int):
  override def toString: String = value.toString

/** The literal text of a numeric field, preserved for the `raw*` output fields. */
final case class Raw(text: String)

// ---------------------------------------------------------------------------
// Money
// ---------------------------------------------------------------------------

/**
 * Exact decimal money. The constructor is private and the only way in from
 * outside is `Money.parse`, which goes through `BigDecimal(String)`. There is no
 * `Double` parameter anywhere in this type or its factories: a binary float
 * cannot enter the domain model.
 */
final case class Money private (value: BigDecimal):
  def +(other: Money): Money = Money.wrap(value.+(other.value)(Money.MC))
  def -(other: Money): Money = Money.wrap(value.-(other.value)(Money.MC))
  def unary_- : Money = Money.wrap(-value)
  def abs: Money = if value.signum < 0 then -this else this
  def isZero: Boolean = value.signum == 0
  def isNegative: Boolean = value.signum < 0
  def isPositive: Boolean = value.signum > 0

  /** Canonical decimal rendering: plain notation, never `E`, at least 2 decimals. */
  def canonical: String =
    val scale = if value.scale < 2 then 2 else value.scale
    // setScale(Int) never rounds when the scale only grows; it throws otherwise.
    // toPlainString on the underlying java.math.BigDecimal: never exponent notation.
    value.setScale(scale).bigDecimal.toPlainString

  override def toString: String = canonical

object Money:
  /** Explicit context: money arithmetic never happens at binary floating point. */
  val MC: MathContext = MathContext(34, RoundingMode.HALF_UP)

  def wrap(value: BigDecimal): Money = new Money(value)

  val zero: Money = wrap(BigDecimal(0))

  /** Sum of a list, in list order (exact, no rounding at these magnitudes). */
  def sum(values: Iterable[Money]): Money = values.foldLeft(zero)(_ + _)

final case class Checksum(hex: String)

object Checksum:
  /** hex of the first 8 bytes of sha256(amount|balanceAfter|bookingDate). */
  def of(amount: Money, balanceAfter: Money, bookingDate: Dated[Booking]): Checksum =
    val input = s"${amount.canonical}|${balanceAfter.canonical}|${bookingDate.iso}"
    val digest = MessageDigest.getInstance("SHA-256").digest(input.getBytes(StandardCharsets.UTF_8))
    Checksum(digest.take(8).map(b => (b & 0xff).toHexString.reverse.padTo(2, '0').reverse).mkString)

// ---------------------------------------------------------------------------
// Dates: two roles, two incompatible types
// ---------------------------------------------------------------------------

/** Semantic role of a date. Booking and value dates are not interchangeable. */
sealed trait DateRole
sealed trait Booking extends DateRole
sealed trait Value extends DateRole

/**
 * A calendar date tagged with its semantic role. `Dated[Booking]` and
 * `Dated[Value]` are unrelated types, so comparing or substituting one for the
 * other is a compile error rather than a silent bug.
 */
final case class Dated[R <: DateRole] private (date: LocalDate):
  def iso: String = Dated.ISO.format(date)

object Dated:
  val ISO: DateTimeFormatter = DateTimeFormatter.ISO_LOCAL_DATE

  def booking(date: LocalDate): Dated[Booking] = new Dated[Booking](date)
  def value(date: LocalDate): Dated[Value] = new Dated[Value](date)

  extension [R <: DateRole](d: Dated[R])
    def compareTo(other: Dated[R]): Int = d.date.compareTo(other.date)
    def isBefore(other: Dated[R]): Boolean = d.date.isBefore(other.date)
    def isAfter(other: Dated[R]): Boolean = d.date.isAfter(other.date)
    def daysUntil(other: Dated[R]): Int = ChronoUnit.DAYS.between(d.date, other.date).toInt

// ---------------------------------------------------------------------------
// Transactions
// ---------------------------------------------------------------------------

/**
 * A row that parsed cleanly. Every field is a domain value: there is no way to
 * hold a transaction whose amount is a string or whose value date is a booking
 * date. The checksum is derived once, here, and never re-derived elsewhere.
 */
final case class Transaction(
    line: Line,
    bookingDate: Dated[Booking],
    valueDate: Dated[Value],
    amount: Money,
    balanceAfter: Money,
    rawAmount: Raw,
    rawBalance: Raw
):
  val checksum: Checksum = Checksum.of(amount, balanceAfter, bookingDate)

/** Transactions in file order plus the lines skipped because they were empty. */
final case class Statement(transactions: List[Transaction], skippedEmptyRows: List[Line]):
  def rowCount: Int = transactions.length

  /**
   * The anchor implied by the first row: `balanceAfter[0] - amount[0]`.
   * This is the only value the two checks of spec 3.1 and 3.2 share; neither
   * derives its running state from the other.
   */
  def openingBalance: Option[Money] =
    transactions.headOption.map(t => t.balanceAfter - t.amount)

  def finalBalance: Option[Money] = transactions.lastOption.map(_.balanceAfter)

// ---------------------------------------------------------------------------
// Accounts (iteration A)
// ---------------------------------------------------------------------------

final case class Currency(code: String):
  override def toString: String = code

object Currency:
  def parse(text: String): Either[String, Currency] =
    val trimmed = text.trim
    if trimmed.length == 3 && trimmed.forall(c => c >= 'A' && c <= 'Z') then Right(Currency(trimmed))
    else Left(s"currency must be a 3-letter uppercase code, found \"$text\"")

final case class AccountId(value: String):
  override def toString: String = value

/** A declared account: identity, currency and the opening balance from the registry. */
final case class Account(
    id: AccountId,
    currency: Currency,
    openingBalance: Money,
    statementPath: String,
    statement: Statement
):
  def netWorth: Money = openingBalance + Money.sum(statement.transactions.map(_.amount))

  /** Balance after the last row; with no rows nothing moved, so it is the opening balance. */
  def closingBalance: Money = statement.finalBalance.getOrElse(openingBalance)



