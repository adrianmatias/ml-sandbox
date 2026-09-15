package ledger

import java.nio.file.Path

/** Outcome of one CLI invocation: exit code plus the exact bytes for each stream. */
final case class CliResult(exitCode: Int, stdout: String, stderr: String)

/**
 * The frozen CLI of spec 6. `run` is a pure function of its arguments and the
 * files they name, so tests can assert on exit codes and envelopes directly.
 */
object Cli:

  val Module: String = "ledger"
  val Version: Long = 1

  val usage: String =
    """ledger — Kutxa statement ledger
      |
      |Usage:
      |  ledger baseline <statement.csv>   print the baseline JSON envelope
      |  ledger accounts <accounts.json>   print the multi-account JSON envelope
      |  ledger --help                     print this help
      |
      |Exit codes: 0 input well-formed (checks may still report violations),
      |            1 input cannot be parsed, 2 usage error.
      |""".stripMargin

  def ok(data: Json): String =
    Json.render(
      Json.Obj(
        List(
          "ok" -> Json.Bool(true),
          "module" -> Json.Str(Module),
          "version" -> Json.Num(Version),
          "data" -> data
        )
      )
    ) + "\n"

  def fail(error: ParseError): String =
    Json.render(
      Json.Obj(
        List(
          "ok" -> Json.Bool(false),
          "module" -> Json.Str(Module),
          "version" -> Json.Num(Version),
          "error" -> Json.Obj(
            List(
              "kind" -> Json.Str(error.kind.toString),
              "message" -> Json.Str(error.message),
              "line" -> error.line.map(l => Json.Num(l.value.toLong)).getOrElse(Json.Null)
            )
          )
        )
      )
    ) + "\n"

  def run(args: List[String]): CliResult = args match
    case Nil => usageError("no command given")
    case ("--help" | "-h") :: _ => CliResult(0, usage, "")
    case "baseline" :: path :: Nil => baseline(path)
    case "accounts" :: path :: Nil => accounts(path)
    case command :: _ if command == "baseline" || command == "accounts" =>
      usageError(s"$command takes exactly one file argument")
    case other :: _ => usageError(s"unknown command: $other")

  private def usageError(message: String): CliResult =
    CliResult(2, "", s"$message\n\n$usage")

  private def failure(error: ParseError): CliResult = CliResult(1, fail(error), "")

  def baseline(path: String): CliResult =
    Source.read(Path.of(path), "statement file") match
      case Left(error) => failure(error)
      case Right(text) =>
        StatementParser.parse(text) match
          case Left(errors) => failure(errors.first)
          case Right(statement) => CliResult(0, ok(Wire.baselineData(statement)), "")

  def accounts(path: String): CliResult =
    Source.read(Path.of(path), "registry file") match
      case Left(error) => failure(error)
      case Right(text) =>
        Registry.parse(text) match
          case Left(errors) => failure(errors.first)
          case Right(registry) =>
            val baseDir = Option(Path.of(path).toAbsolutePath.normalize.getParent).getOrElse(Path.of("."))
            AccountsLoader.load(registry, baseDir) match
              case Left(errors) => failure(errors.first)
              case Right(accounts) =>
                val transfers = TransferDetection.detect(accounts, registry.toleranceDays)
                val netWorth = NetWorth.observe(accounts)
                CliResult(
                  0,
                  ok(Wire.accountsData(registry.toleranceDays, accounts, transfers, netWorth)),
                  ""
                )
