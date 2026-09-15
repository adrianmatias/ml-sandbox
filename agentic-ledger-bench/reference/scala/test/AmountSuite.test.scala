package ledger

class AmountSuite extends munit.FunSuite:

  private def ok(raw: String): Money = AmountParser.parseText(raw).fold(m => fail(s"$raw: $m"), identity)
  private def bad(raw: String): String = AmountParser.parseText(raw).fold(identity, m => fail(s"$raw parsed as ${m.canonical}"))

  test("accepts . and , as decimal separator"):
    assertEquals(ok("1234.56").canonical, "1234.56")
    assertEquals(ok("1234,56").canonical, "1234.56")
    assertEquals(ok("-42.59").canonical, "-42.59")
    assertEquals(ok("+42,59").canonical, "42.59")

  test("canonical form: plain notation, at least two decimals, never E"):
    assertEquals(ok("400").canonical, "400.00")
    assertEquals(ok("37713.3").canonical, "37713.30")
    assertEquals(ok("0").canonical, "0.00")
    assertEquals(ok("-0.5").canonical, "-0.50")
    assertEquals(ok("37713.30").canonical, "37713.30")
    assert(!ok("37713.30").canonical.contains("E"))

  test("rejects a separator followed by exactly three digits as ambiguous"):
    assert(bad("1.234").contains("ambiguous"), bad("1.234"))
    assert(bad("1,234").contains("ambiguous"), bad("1,234"))
    assert(bad("1234.567").contains("ambiguous"), bad("1234.567"))
    assert(bad("-1.234").contains("ambiguous"), bad("-1.234"))

  test("rejects a value containing both separators"):
    assert(bad("1.234,56").contains("ambiguous"))
    assert(bad("1,234.56").contains("ambiguous"))

  test("four digits after a separator are not ambiguous"):
    assertEquals(ok("1.2345").canonical, "1.2345")

  test("rejects empty, non-numeric and multi-sign"):
    assertEquals(bad(""), "empty amount")
    assert(bad("abc").nonEmpty)
    assert(bad("1.2.3").nonEmpty)
    assert(bad("--5").contains("more than one sign"))
    assert(bad("-").nonEmpty)
    assert(bad("5.").nonEmpty)
    assert(bad(".5").nonEmpty)
    assert(bad("1,2,3").nonEmpty)

  test("money is exact decimal; a binary float cannot enter the domain"):
    assertEquals((ok("0.1") + ok("0.2")).canonical, "0.30")
    assertEquals(Money.sum(List(ok("0.1"), ok("0.2"))).canonical, "0.30")
    assertEquals(ok("0.1") + ok("0.2"), ok("0.30"))
    // the same sum through Double is famously not 0.3
    assert(0.1 + 0.2 != 0.3)

  test("explicit MathContext is used for arithmetic"):
    assertEquals(Money.MC.getPrecision, 34)
    assertEquals(Money.MC.getRoundingMode, java.math.RoundingMode.HALF_UP)

  test("equality ignores scale, canonical rendering does not"):
    assertEquals(ok("400"), ok("400.00"))
    assertEquals(ok("400").canonical, "400.00")
    assertEquals(ok("400.0").canonical, "400.00")

