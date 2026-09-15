package ledger

import java.nio.file.{Files, Path}
import java.nio.charset.StandardCharsets

// ---------------------------------------------------------------------------
// Accounts registry (iteration A)
// ---------------------------------------------------------------------------

/** One entry of the accounts registry. `statementPath` is as written in the file. */
final case class AccountEntry(
    id: AccountId,
    currency: Currency,
    openingBalance: Money,
    statementPath: String
)

final case class Registry(accounts: List[AccountEntry], toleranceDays: Int)

object Registry:
  val DefaultToleranceDays: Int = 3

  /**
   * Registry format (this module's own, recorded in NOTES.md):
   * `{"toleranceDays": 3, "accounts": [{"id": "A", "currency": "EUR",
   * "openingBalance": "1000.00", "statement": "a.csv"}]}`.
   * `toleranceDays` defaults to 3 and `statement` defaults to `<id>.csv`, both
   * resolved relative to the registry file's directory. `openingBalance` must be
   * a string: a JSON number would drag a binary float into the domain.
   * Unknown fields are ignored.
   */
  def parse(text: String): Either[Errors, Registry] =
    JsonParser.parse(text) match
      case Left(message) => Left(Errors.one(ParseError.registry(s"invalid registry JSON: $message")))
      case Right(json) => fromJson(json)

  private def fromJson(json: Json): Either[Errors, Registry] =
    val errors = List.newBuilder[ParseError]

    def field(obj: List[(String, Json)], name: String): Option[Json] =
      obj.find(_._1 == name).map(_._2)

    def asObject(value: Json, what: String): Option[List[(String, Json)]] = value match
      case Json.Obj(fields) => Some(fields)
      case other =>
        errors += ParseError.registry(s"$what must be an object, found ${typeName(other)}")
        None

    def asString(value: Json, what: String): Option[String] = value match
      case Json.Str(text) => Some(text)
      case other =>
        errors += ParseError.registry(s"$what must be a string, found ${typeName(other)}")
        None

    json match
      case Json.Obj(fields) =>
        val tolerance =
          field(fields, "toleranceDays") match
            case None => Registry.DefaultToleranceDays
            case Some(Json.Num(value)) if value >= 0 => value.toInt
            case Some(other) =>
              errors += ParseError.registry(
                s"toleranceDays must be a non-negative integer, found ${typeName(other)}"
              )
              Registry.DefaultToleranceDays

        val entries =
          field(fields, "accounts") match
            case None =>
              errors += ParseError.registry("registry has no \"accounts\" field")
              Nil
            case Some(Json.Arr(values)) if values.isEmpty =>
              errors += ParseError.registry("registry has no accounts")
              Nil
            case Some(Json.Arr(values)) => values.zipWithIndex.flatMap((v, i) => entry(v, s"accounts[$i]", errors))
            case Some(other) =>
              errors += ParseError.registry(s"\"accounts\" must be an array, found ${typeName(other)}")
              Nil

        val duplicates = entries.groupBy(_.id.value).collect { case (id, xs) if xs.length > 1 => id }.toList.sorted
        duplicates.foreach(id => errors += ParseError.registry(s"""duplicate account id "$id""""))

        val collected = errors.result()
        collected match
          case Nil => Right(Registry(entries, tolerance))
          case head :: rest => Left(Errors(head, rest))
      case other =>
        Left(Errors.one(ParseError.registry(s"registry must be an object, found ${typeName(other)}")))

  private def entry(
      value: Json,
      what: String,
      errors: scala.collection.mutable.Builder[ParseError, List[ParseError]]
  ): Option[AccountEntry] =
    value match
      case Json.Obj(fields) =>
        def field(name: String): Option[Json] = fields.find(_._1 == name).map(_._2)
        def requiredString(name: String): Option[String] =
          field(name) match
            case Some(Json.Str(text)) => Some(text)
            case Some(other) =>
              errors += ParseError.registry(s"$what.$name must be a string, found ${typeName(other)}")
              None
            case None =>
              errors += ParseError.registry(s"$what is missing \"$name\"")
              None

        def optionalString(name: String): Option[String] =
          field(name) match
            case Some(Json.Str(text)) => Some(text)
            case Some(other) =>
              errors += ParseError.registry(s"$what.$name must be a string, found ${typeName(other)}")
              None
            case None => None

        val idText = requiredString("id")
        val id = idText.filter(_.trim.nonEmpty)
        if idText.isDefined && id.isEmpty then errors += ParseError.registry(s"$what.id must not be empty")
        val currency = requiredString("currency").map(Currency.parse).flatMap {
          case Right(c) => Some(c)
          case Left(message) =>
            errors += ParseError.registry(s"$what.currency $message")
            None
        }
        val opening = requiredString("openingBalance").flatMap { text =>
          AmountParser.parseText(text) match
            case Right(money) => Some(money)
            case Left(message) =>
              errors += ParseError.registry(s"$what.openingBalance $message")
              None
        }
        val statementText = optionalString("statement")
        val statement = statementText.filter(_.trim.nonEmpty)
        if statementText.isDefined && statement.isEmpty then
          errors += ParseError.registry(s"$what.statement must not be empty")

        for
          i <- id
          c <- currency
          o <- opening
        yield AccountEntry(AccountId(i), c, o, statement.getOrElse(s"$i.csv"))
      case other =>
        errors += ParseError.registry(s"$what must be an object, found ${typeName(other)}")
        None

  private def typeName(json: Json): String = json match
    case Json.Str(_) => "a string"
    case Json.Num(_) => "a number"
    case Json.Bool(_) => "a boolean"
    case Json.Null => "null"
    case Json.Arr(_) => "an array"
    case Json.Obj(_) => "an object"

// ---------------------------------------------------------------------------
// Loading statements for a registry
// ---------------------------------------------------------------------------

object AccountsLoader:

  /** Read and parse one statement per registry entry, keeping every error. */
  def load(registry: Registry, baseDir: Path): Either[Errors, List[Account]] =
    val loaded = registry.accounts.map { entry =>
      val path = baseDir.resolve(entry.statementPath)
      Source.read(path, s"${entry.id.value}: statement ${entry.statementPath}") match
        case Left(error) => Left(Errors.one(error))
        case Right(text) =>
          StatementParser.parse(text, entry.id.value) match
            case Left(errors) => Left(errors)
            case Right(statement) =>
              Right(
                Account(
                  id = entry.id,
                  currency = entry.currency,
                  openingBalance = entry.openingBalance,
                  statementPath = entry.statementPath,
                  statement = statement
                )
              )
    }
    Errors.sequence(loaded)

/** File reading. Error messages never contain an absolute path (spec 6). */
object Source:
  def read(path: Path, describe: String): Either[ParseError, String] =
    try Right(Files.readString(path, StandardCharsets.UTF_8))
    catch
      case error: Exception =>
        Left(ParseError.io(s"cannot read $describe: ${error.getClass.getSimpleName}"))
