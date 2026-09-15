package ledger

/** Frozen error kinds (spec 6). The `toString` of each case is the wire value. */
enum ErrorKind:
  case IO, HEADER, ROW, AMOUNT, DATE, REGISTRY

final case class ParseError(kind: ErrorKind, message: String, line: Option[Line]):
  def at(line: Line): ParseError = copy(line = Some(line))

object ParseError:
  def io(message: String): ParseError = ParseError(ErrorKind.IO, message, None)
  def header(message: String): ParseError = ParseError(ErrorKind.HEADER, message, None)
  def row(line: Line, message: String): ParseError = ParseError(ErrorKind.ROW, message, Some(line))
  def amount(line: Line, message: String): ParseError = ParseError(ErrorKind.AMOUNT, message, Some(line))
  def date(line: Line, message: String): ParseError = ParseError(ErrorKind.DATE, message, Some(line))
  def registry(message: String): ParseError = ParseError(ErrorKind.REGISTRY, message, None)

/**
 * A non-empty list of errors. Parsing reports every violation it found (spec
 * 3.4) rather than stopping at the first one, so the failure channel cannot be a
 * bare `ParseError`.
 */
final case class Errors(head: ParseError, rest: List[ParseError]):
  def all: List[ParseError] = head :: rest
  def first: ParseError = head
  def size: Int = 1 + rest.length

object Errors:
  def one(error: ParseError): Errors = Errors(error, Nil)

  /** Keep every `Left`, in order, or collect every value. */
  def collect[A](results: List[Either[ParseError, A]]): Either[Errors, List[A]] =
    val values = List.newBuilder[A]
    val errors = List.newBuilder[ParseError]
    results.foreach {
      case Right(a) => values += a
      case Left(e) => errors += e
    }
    val collected = errors.result()
    collected match
      case Nil => Right(values.result())
      case head :: rest => Left(Errors(head, rest))

  /** Collapse the `Left` cases of a list, keeping order, or collect the values. */
  def sequence[A](results: List[Either[Errors, A]]): Either[Errors, List[A]] =
    val values = List.newBuilder[A]
    val errors = List.newBuilder[ParseError]
    var failed = false
    results.foreach {
      case Right(a) => values += a
      case Left(errs) =>
        failed = true
        errs.all.foreach(errors += _)
    }
    val collected = errors.result()
    if failed then Left(Errors(collected.head, collected.tail)) else Right(values.result())
