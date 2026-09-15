package ledger

class DateSuite extends munit.FunSuite:

  private def bad(raw: String): String = DateParser.parseText(raw).fold(identity, d => fail(s"$raw parsed as $d"))

  test("strict DD/MM/YYYY"):
    assertEquals(DateParser.parseText("01/01/2026"), Right(java.time.LocalDate.of(2026, 1, 1)))
    assertEquals(DateParser.parseText("25/12/2026"), Right(java.time.LocalDate.of(2026, 12, 25)))
    assertEquals(DateParser.parseText("29/02/2024"), Right(java.time.LocalDate.of(2024, 2, 29)))

  test("rejects dates that do not exist rather than rolling them over"):
    assert(bad("31/02/2026").contains("real calendar date"), bad("31/02/2026"))
    assert(bad("29/02/2026").contains("real calendar date"))
    assert(bad("31/04/2026").contains("real calendar date"))
    assert(bad("00/01/2026").contains("real calendar date"))

  test("rejects any shape other than DD/MM/YYYY"):
    assert(bad("2026-01-01").contains("DD/MM/YYYY"))
    assert(bad("1/1/2026").contains("DD/MM/YYYY"))
    assert(bad("01-01-2026").contains("DD/MM/YYYY"))
    assert(bad("01/01/26").contains("DD/MM/YYYY"))
    assert(bad("").contains("DD/MM/YYYY"))
    assert(bad("12/25/2026").contains("real calendar date"))

  test("booking and value date are distinct types"):
    val booking: Dated[Booking] = Dated.booking(java.time.LocalDate.of(2026, 1, 5))
    val value: Dated[Value] = Dated.value(java.time.LocalDate.of(2026, 1, 7))
    assertEquals(booking.iso, "2026-01-05")
    assertEquals(value.iso, "2026-01-07")
    // The following would not compile, which is the point of the two roles:
    //   booking.isBefore(value)   // Required: Dated[Booking], found: Dated[Value]
    assertEquals(value.daysUntil(Dated.value(java.time.LocalDate.of(2026, 1, 10))), 3)
    assert(value.isAfter(Dated.value(java.time.LocalDate.of(2026, 1, 6))))
