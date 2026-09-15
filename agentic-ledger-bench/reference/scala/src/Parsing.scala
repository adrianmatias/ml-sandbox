package ledger

import java.time.LocalDate
import java.time.format.{DateTimeFormatter, ResolverStyle}
import scala.math.BigDecimal

/**
 * Amount parsing (spec 4.1). Accepts `.` or `,` as decimal separator, rejects
 * anything whose separator is ambiguous, and never guesses. The core returns a
 * message; the CSV layer attaches the line number.
 */
object AmountParser:

  private val Digits = "[0-9]+".r

  def parse(raw: String, at: Line): Either[ParseError, Money] =
    parseText(raw).left.map(ParseError.amount(at, _))

  def parseText(raw: String): Either[String, Money] =
    val text = raw.trim
    if text.isEmpty then Left("empty amount")
    else
      val (sign, body) =
        if text.startsWith("-") then (-1, text.drop(1))
        else if text.startsWith("+") then (1, text.drop(1))
        else (1, text)
      if body.isEmpty then Left(s"""amount "$raw" has a sign but no digits""")
      else if body.startsWith("-") || body.startsWith("+") then
        Left(s"""amount "$raw" has more than one sign""")
      else
        val dots = body.count(_ == '.')
        val commas = body.count(_ == ',')
        if dots > 0 && commas > 0 then
          Left(s"""amount "$raw" contains both "." and "," and is ambiguous""")
        else if dots + commas > 1 then
          Left(s"""amount "$raw" contains more than one decimal separator""")
        else
          val separator: Option[Char] =
            if dots == 1 then Some('.') else if commas == 1 then Some(',') else None
          val parts = separator match
            case Some(sep) =>
              val at = body.indexOf(sep.toInt)
              List(body.substring(0, at), body.substring(at + 1))
            case None => List(body)
          parts match
            case List(whole, fraction) if whole.nonEmpty && fraction.nonEmpty =>
              if !Digits.matches(whole) || !Digits.matches(fraction) then
                Left(s"""amount "$raw" is not numeric""")
              else if fraction.length == 3 then
                Left(s"""amount "$raw" is ambiguous: a separator followed by exactly three digits""")
              else Right(Money.wrap(BigDecimal(s"$whole.$fraction") * sign))
            case List(whole) if whole.nonEmpty && Digits.matches(whole) =>
              Right(Money.wrap(BigDecimal(whole) * sign))
            case _ => Left(s"""amount "$raw" is not numeric""")

/**
 * Date parsing (spec 4.2). Strict `DD/MM/YYYY` with real calendar validation:
 * `31/02/2026` is rejected, not rolled over to 2026-03-03.
 */
object DateParser:

  private val Shape = "[0-9]{2}/[0-9]{2}/[0-9]{4}".r
  private val Formatter: DateTimeFormatter =
    DateTimeFormatter.ofPattern("dd/MM/uuuu").withResolverStyle(ResolverStyle.STRICT)

  def parse(raw: String, at: Line): Either[ParseError, LocalDate] =
    parseText(raw).left.map(ParseError.date(at, _))

  def parseText(raw: String): Either[String, LocalDate] =
    val text = raw.trim
    if !Shape.matches(text) then Left(s"""date "$raw" is not DD/MM/YYYY""")
    else
      try Right(LocalDate.parse(text, Formatter))
      catch
        case _: java.time.format.DateTimeParseException =>
          Left(s"""date "$raw" is not a real calendar date""")

/** Row- and file-level parsing, accumulating every violation (spec 3.4). */
object StatementParser:

  val HeaderFields: List[String] = List("fecha", "fecha valor", "importe", "saldo")

  def header: String = HeaderFields.mkString(",")

  def parse(text: String, source: String = ""): Either[Errors, Statement] =
    Csv.lines(text) match
      case Nil => Left(Errors.one(ParseError.header("empty input: expected a header line")))
      case headerLine :: dataLines =>
        Csv.split(headerLine, Line(1)) match
          case Left(error) => Left(Errors.one(error.copy(kind = ErrorKind.HEADER, line = None)))
          case Right(fields) =>
            val found = fields.map(_.trim)
            if found != HeaderFields then
              Left(
                Errors.one(
                  ParseError.header(
                    s"""unexpected header: expected "$header", found "${found.mkString(",")}""""
                  )
                )
              )
            else
              val prefixed = prefixer(source)
              val parsed = dataLines.zipWithIndex.map { (raw, index) =>
                parseRow(raw, Line(index + 2), prefixed)
              }
              Errors.sequence(parsed).map { rows =>
                Statement(
                  transactions = rows.flatten,
                  skippedEmptyRows = rows.zipWithIndex.collect { case (None, index) => Line(index + 2) }
                )
              }

  private def prefixer(source: String): String => String =
    if source.isEmpty then identity else message => s"$source: $message"

  private def parseRow(
      raw: String,
      at: Line,
      prefix: String => String
  ): Either[Errors, Option[Transaction]] =
    Csv.split(raw, at) match
      case Left(error) => Left(Errors.one(error.copy(message = prefix(error.message))))
      case Right(fields) =>
        if Csv.isBlank(fields) then Right(None)
        else if fields.length != 4 then
          Left(Errors.one(ParseError.row(at, prefix(s"expected 4 columns, found ${fields.length}"))))
        else
          val bookingRaw = fields(0)
          val valueRaw = fields(1)
          val amountRaw = fields(2)
          val balanceRaw = fields(3)
          zip4(
            DateParser.parse(bookingRaw, at),
            DateParser.parse(valueRaw, at),
            AmountParser.parse(amountRaw, at),
            AmountParser.parse(balanceRaw, at)
          ) match
            case Left(errors) =>
              val tagged = errors.all.map(e => e.copy(message = prefix(e.message)))
              Left(Errors(tagged.head, tagged.tail))
            case Right((booking, value, amount, balance)) =>
              Right(
                Some(
                  Transaction(
                    line = at,
                    bookingDate = Dated.booking(booking),
                    valueDate = Dated.value(value),
                    amount = amount,
                    balanceAfter = balance,
                    rawAmount = Raw(amountRaw),
                    rawBalance = Raw(balanceRaw)
                  )
                )
              )

  /** Applicative combination of four parses that keeps every error. */
  private def zip4[A, B, C, D](
      ea: Either[ParseError, A],
      eb: Either[ParseError, B],
      ec: Either[ParseError, C],
      ed: Either[ParseError, D]
  ): Either[Errors, (A, B, C, D)] =
    val errors = List(ea, eb, ec, ed).collect { case Left(e) => e }
    errors match
      case Nil =>
        Right(
          (
            ea.toOption.get,
            eb.toOption.get,
            ec.toOption.get,
            ed.toOption.get
          )
        )
      case head :: rest => Left(Errors(head, rest))
