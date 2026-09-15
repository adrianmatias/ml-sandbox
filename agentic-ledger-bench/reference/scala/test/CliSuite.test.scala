package ledger

class CliSuite extends munit.FunSuite:
  import TestSupport.*

  private def run(args: String*): CliResult = Cli.run(args.toList)

  private val reference = "../raw/kutxa_movimientos_2026-05-25.csv"

  test("exit 0 when the file is well-formed, even when checks report violations"):
    val result = run("baseline", path("fixtures/edge/collision.csv"))
    assertEquals(result.exitCode, 0)
    assert(result.stdout.startsWith("""{"ok":true,"module":"ledger","version":1,"data":{"""), result.stdout.take(80))

  test("exit 0 on the reference file, which has 29 inversions"):
    val result = run("baseline", path(reference))
    assertEquals(result.exitCode, 0)
    assert(result.stdout.contains(""""reconciled":true"""))

  test("exit 1 when the input cannot be parsed, with the frozen kind and line"):
    val result = run("baseline", path("fixtures/edge/amount_abc.csv"))
    assertEquals(result.exitCode, 1)
    assert(result.stdout.contains(""""kind":"AMOUNT""""), result.stdout)
    assert(result.stdout.contains(""""line":2"""), result.stdout)

  test("exit 1 with kind IO when the file cannot be read"):
    val result = run("baseline", "definitely-not-here-9f3a.csv")
    assertEquals(result.exitCode, 1)
    assert(result.stdout.contains(""""kind":"IO""""), result.stdout)
    assert(!result.stdout.contains("/home/"), "IO messages must not leak absolute paths")

  test("exit 2 on usage errors, with usage on stderr and no JSON"):
    for args <- List(List.empty[String], List("nope"), List("baseline"), List("baseline", "a", "b")) do
      val result = Cli.run(args)
      assertEquals(result.exitCode, 2, args.toString)
      assertEquals(result.stdout, "", args.toString)
      assert(result.stderr.contains("Usage:"), args.toString)

  test("--help exits 0 and prints usage on stdout"):
    val result = run("--help")
    assertEquals(result.exitCode, 0)
    assert(result.stdout.contains("baseline <statement.csv>"))
    assertEquals(result.stderr, "")

  test("envelope key order is frozen: ok, module, version, then data or error"):
    val baseline = run("baseline", path("fixtures/edge/empty_row.csv")).stdout
    assert(
      baseline.startsWith("""{"ok":true,"module":"ledger","version":1,"data":{"transactions":["""),
      baseline.take(120)
    )
    assert(
      baseline.contains(""""checks":{"rowCount":2,"skippedEmptyRows":[3],"reconciled":true,"""),
      baseline
    )
    val failure = run("baseline", path("fixtures/edge/bad_header.csv")).stdout
    assert(
      failure.startsWith("""{"ok":false,"module":"ledger","version":1,"error":{"kind":"HEADER","message":""""),
      failure.take(120)
    )
    assert(failure.contains(""""line":null"""), failure)

  test("two runs on the same input are byte-identical"):
    assertEquals(run("baseline", path(reference)).stdout, run("baseline", path(reference)).stdout)
    val registry = "../spec/shared-fixtures/accounts.json"
    assertEquals(run("accounts", path(registry)).stdout, run("accounts", path(registry)).stdout)

  test("no absolute paths, no timestamps, no exponent notation"):
    val out = run("baseline", path(reference)).stdout
    assert(!out.contains("/home/"), "no absolute paths")
    assert(!out.matches("(?s).*\\d{4}-\\d{2}-\\d{2}T.*"), "no timestamps")
    assert(!out.matches("(?s).*\\dE[+-]?\\d.*"), "no exponent notation")
    assert(out.endsWith("}\n"))

  test("canonical amounts: 37713.30, never 37713.3 and never 3.77133E4"):
    val out = run("baseline", path(reference)).stdout
    assert(out.contains(""""balanceAfter":"37713.30""""), "final balance must render with two decimals")
    assert(!out.contains(""""balanceAfter":"37713.3""""), "unpadded decimal must not appear")
    assert(!out.contains("E4"), "exponent notation must not appear")

  test("accounts command: exit 0 on the shared fixture"):
    val result = run("accounts", path("../spec/shared-fixtures/accounts.json"))
    assertEquals(result.exitCode, 0)
    assert(result.stdout.startsWith("""{"ok":true,"module":"ledger","version":1,"data":{"toleranceDays":3,"accounts":["""))
    assert(result.stdout.contains(""""netWorthConserved":true"""))
    assert(result.stdout.contains(""""unpairedTransferCandidates":0"""))

  test("accounts command: kinds for a bad registry"):
    val result = run("accounts", path("fixtures/edge/bad_header.csv"))
    assertEquals(result.exitCode, 1)
    assert(result.stdout.contains(""""kind":"REGISTRY""""), result.stdout)
