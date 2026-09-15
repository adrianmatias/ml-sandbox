package ledger

/**
 * A minimal JSON tree. Numbers are `Long`: this module emits line numbers and
 * counts, and money is always a decimal *string*, so no floating point value can
 * ever be rendered. Object fields are an ordered list, never a map, so key order
 * is explicit and hash-order iteration is impossible.
 */
enum Json:
  case Str(value: String)
  case Num(value: Long)
  case Bool(value: Boolean)
  case Null
  case Arr(values: List[Json])
  case Obj(fields: List[(String, Json)])

object Json:

  def int(value: Long): Json = Num(value)

  def render(json: Json): String = json match
    case Str(value) => quote(value)
    case Num(value) => value.toString
    case Bool(value) => if value then "true" else "false"
    case Null => "null"
    case Arr(values) => values.map(render).mkString("[", ",", "]")
    case Obj(fields) => fields.map((key, value) => quote(key) + ":" + render(value)).mkString("{", ",", "}")

  private def quote(text: String): String =
    val sb = new StringBuilder("\"")
    text.foreach {
      case '"' => sb.append("\\\"")
      case '\\' => sb.append("\\\\")
      case '\n' => sb.append("\\n")
      case '\r' => sb.append("\\r")
      case '\t' => sb.append("\\t")
      case '\b' => sb.append("\\b")
      case '\f' => sb.append("\\f")
      case c if c < ' ' => sb.append("\\u%04x".format(c.toInt))
      case c => sb.append(c)
    }
    sb.append('"').toString

/** Reader for the accounts registry. Small, strict, no dependency. */
object JsonParser:

  def parse(text: String): Either[String, Json] =
    val state = new State(text)
    state.skipWhitespace()
    for
      value <- state.value()
      _ <- { state.skipWhitespace(); if state.atEnd then Right(()) else Left(state.error("trailing content")) }
    yield value

  private final class State(text: String):
    private var pos = 0

    def atEnd: Boolean = pos >= text.length

    def error(message: String): String =
      var line = 1
      var column = 1
      var i = 0
      while i < pos && i < text.length do
        if text.charAt(i) == '\n' then { line += 1; column = 1 }
        else column += 1
        i += 1
      s"$message at line $line, column $column"

    def skipWhitespace(): Unit =
      while !atEnd && (text.charAt(pos) == ' ' || text.charAt(pos) == '\n' || text.charAt(pos) == '\r' || text.charAt(pos) == '\t') do
        pos += 1

    private def peek: Option[Char] = if atEnd then None else Some(text.charAt(pos))

    def value(): Either[String, Json] =
      peek match
        case None => Left(error("unexpected end of input"))
        case Some('{') => obj()
        case Some('[') => arr()
        case Some('"') => string().map(Json.Str.apply)
        case Some('t') => literal("true", Json.Bool(true))
        case Some('f') => literal("false", Json.Bool(false))
        case Some('n') => literal("null", Json.Null)
        case Some(c) if c == '-' || c.isDigit => number()
        case Some(c) => Left(error(s"unexpected character '$c'"))

    private def literal(word: String, value: Json): Either[String, Json] =
      if text.startsWith(word, pos) then { pos += word.length; Right(value) }
      else Left(error(s"expected $word"))

    private def obj(): Either[String, Json] =
      pos += 1
      skipWhitespace()
      val fields = List.newBuilder[(String, Json)]
      if peek.contains('}') then { pos += 1; Right(Json.Obj(fields.result())) }
      else
        var done = false
        var failure: Option[String] = None
        while !done && failure.isEmpty do
          skipWhitespace()
          string() match
            case Left(message) => failure = Some(message)
            case Right(key) =>
              skipWhitespace()
              if !peek.contains(':') then failure = Some(error("expected ':'"))
              else
                pos += 1
                skipWhitespace()
                value() match
                  case Left(message) => failure = Some(message)
                  case Right(v) =>
                    fields += (key -> v)
                    skipWhitespace()
                    peek match
                      case Some(',') => pos += 1
                      case Some('}') => pos += 1; done = true
                      case _ => failure = Some(error("expected ',' or '}'"))
        failure match
          case Some(message) => Left(message)
          case None => Right(Json.Obj(fields.result()))

    private def arr(): Either[String, Json] =
      pos += 1
      skipWhitespace()
      val values = List.newBuilder[Json]
      if peek.contains(']') then { pos += 1; Right(Json.Arr(values.result())) }
      else
        var done = false
        var failure: Option[String] = None
        while !done && failure.isEmpty do
          skipWhitespace()
          value() match
            case Left(message) => failure = Some(message)
            case Right(v) =>
              values += v
              skipWhitespace()
              peek match
                case Some(',') => pos += 1
                case Some(']') => pos += 1; done = true
                case _ => failure = Some(error("expected ',' or ']'"))
        failure match
          case Some(message) => Left(message)
          case None => Right(Json.Arr(values.result()))

    private def string(): Either[String, String] =
      if !peek.contains('"') then Left(error("expected a string"))
      else
        pos += 1
        val sb = new StringBuilder
        var done = false
        var failure: Option[String] = None
        while !done && failure.isEmpty do
          if atEnd then failure = Some(error("unterminated string"))
          else
            val c = text.charAt(pos)
            pos += 1
            c match
              case '"' => done = true
              case '\\' =>
                if atEnd then failure = Some(error("unterminated escape"))
                else
                  val e = text.charAt(pos)
                  pos += 1
                  e match
                    case '"' => sb.append('"')
                    case '\\' => sb.append('\\')
                    case '/' => sb.append('/')
                    case 'b' => sb.append('\b')
                    case 'f' => sb.append('\f')
                    case 'n' => sb.append('\n')
                    case 'r' => sb.append('\r')
                    case 't' => sb.append('\t')
                    case 'u' =>
                      if pos + 4 > text.length then failure = Some(error("truncated \\u escape"))
                      else
                        val hex = text.substring(pos, pos + 4)
                        try
                          sb.append(Integer.parseInt(hex, 16).toChar)
                          pos += 4
                        catch case _: NumberFormatException => failure = Some(error("bad \\u escape"))
                    case other => failure = Some(error(s"bad escape '\\$other'"))
              case other => sb.append(other)
        failure match
          case Some(message) => Left(message)
          case None => Right(sb.toString)

    private def number(): Either[String, Json] =
      val start = pos
      if peek.contains('-') then pos += 1
      while !atEnd && text.charAt(pos).isDigit do pos += 1
      val raw = text.substring(start, pos)
      if raw.isEmpty || raw == "-" then Left(error("expected a number"))
      else if !atEnd && text.charAt(pos) == '.' then Left(error(s"non-integer number '$raw.' is not supported"))
      else
        try Right(Json.Num(raw.toLong))
        catch case _: NumberFormatException => Left(error(s"number '$raw' is out of range"))
