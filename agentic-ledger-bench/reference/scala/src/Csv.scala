package ledger

/**
 * CSV reading for the Kutxa statement format: comma separated, RFC 4180 quoting
 * for a single record, UTF-8 BOM tolerated on the first line, LF or CRLF.
 */
object Csv:

  val BOM: Char = '\uFEFF'

  /** Physical lines, 1-based by position, with a leading BOM removed. */
  def lines(text: String): List[String] =
    val body = if text.endsWith("\n") then text.dropRight(1) else text
    val raw = body.split("\n", -1).toList.map(stripCr)
    raw match
      case head :: tail if head.startsWith(BOM.toString) => head.drop(1) :: tail
      case other => other

  private def stripCr(line: String): String =
    if line.endsWith("\r") then line.dropRight(1) else line

  /**
   * Split one record into fields. A quote opens a quoted field only at the start
   * of a field; `""` inside a quoted field is one literal quote.
   */
  def split(line: String, at: Line): Either[ParseError, List[String]] =
    val fields = List.newBuilder[String]
    val current = new StringBuilder
    var i = 0
    var inQuotes = false
    var failure: Option[ParseError] = None

    while i < line.length && failure.isEmpty do
      val c = line.charAt(i)
      if inQuotes then
        if c == '"' then
          if i + 1 < line.length && line.charAt(i + 1) == '"' then
            current.append('"')
            i += 1
          else inQuotes = false
        else current.append(c)
      else if c == '"' && current.isEmpty then inQuotes = true
      else if c == ',' then
        fields += current.toString
        current.clear()
      else if c == '"' then failure = Some(ParseError.row(at, "unexpected quote inside an unquoted field"))
      else current.append(c)
      i += 1

    if failure.isEmpty && inQuotes then failure = Some(ParseError.row(at, "unterminated quoted field"))
    failure match
      case Some(error) => Left(error)
      case None =>
        fields += current.toString
        Right(fields.result())

  /** A record with no content at all: `,,,` or a blank line. Skipped, not an error. */
  def isBlank(fields: List[String]): Boolean = fields.forall(_.trim.isEmpty)
