package ledger

/** JSON rendering of domain values. Key order here is the frozen output order (spec 3, 6). */
object Wire:

  def money(value: Money): Json = Json.Str(value.canonical)

  def transaction(t: Transaction): Json =
    Json.Obj(
      List(
        "line" -> Json.Num(t.line.value.toLong),
        "bookingDate" -> Json.Str(t.bookingDate.iso),
        "valueDate" -> Json.Str(t.valueDate.iso),
        "amount" -> money(t.amount),
        "balanceAfter" -> money(t.balanceAfter),
        "rawAmount" -> Json.Str(t.rawAmount.text),
        "rawBalance" -> Json.Str(t.rawBalance.text),
        "checksum" -> Json.Str(t.checksum.hex)
      )
    )

  def continuityError(e: ContinuityError): Json =
    Json.Obj(
      List(
        "line" -> Json.Num(e.line.value.toLong),
        "balanceExpected" -> money(e.balanceExpected),
        "balanceActual" -> money(e.balanceActual),
        "delta" -> money(e.delta)
      )
    )

  def cumulativeError(e: CumulativeError): Json =
    Json.Obj(
      List(
        "line" -> Json.Num(e.line.value.toLong),
        "balanceFromSum" -> money(e.balanceFromSum),
        "balanceActual" -> money(e.balanceActual),
        "delta" -> money(e.delta)
      )
    )

  def valueDateInversion(e: ValueDateInversion): Json =
    Json.Obj(
      List(
        "line" -> Json.Num(e.line.value.toLong),
        "previousLine" -> Json.Num(e.previousLine.value.toLong)
      )
    )

  def checks(c: Checks): Json =
    Json.Obj(
      List(
        "rowCount" -> Json.Num(c.rowCount.toLong),
        "skippedEmptyRows" -> Json.Arr(c.skippedEmptyRows.map(l => Json.Num(l.value.toLong))),
        "reconciled" -> Json.Bool(c.reconciled),
        "continuityErrors" -> Json.Arr(c.continuityErrors.map(continuityError)),
        "cumulativeErrors" -> Json.Arr(c.cumulativeErrors.map(cumulativeError)),
        "valueDateInversions" -> Json.Arr(c.valueDateInversions.map(valueDateInversion))
      )
    )

  def baselineData(statement: Statement): Json =
    Json.Obj(
      List(
        "transactions" -> Json.Arr(statement.transactions.map(transaction)),
        "checks" -> checks(Checks.run(statement))
      )
    )

  // -------------------------------------------------------------------------
  // Iteration A
  // -------------------------------------------------------------------------

  def pairedRow(row: PairedRow): Json =
    Json.Obj(
      List(
        "account" -> Json.Str(row.accountId.value),
        "line" -> Json.Num(row.line.value.toLong),
        "valueDate" -> Json.Str(row.valueDate.iso)
      )
    )

  def transfer(t: Transfer): Json =
    Json.Obj(
      List(
        "outflow" -> pairedRow(t.outflow),
        "inflow" -> pairedRow(t.inflow),
        "amount" -> money(t.amount),
        "lagDays" -> Json.Num(t.lagDays.toLong)
      )
    )

  def netWorthPoint(p: NetWorthPoint): Json =
    Json.Obj(
      List(
        "account" -> Json.Str(p.account.value),
        "netWorth" -> money(p.netWorth),
        "netWorthFromBalances" -> money(p.netWorthFromBalances),
        "delta" -> money(p.delta),
        "agrees" -> Json.Bool(p.agrees)
      )
    )

  def netWorthMismatch(m: NetWorthMismatch): Json =
    Json.Obj(
      List(
        "account" -> Json.Str(m.scope),
        "netWorth" -> money(m.netWorth),
        "netWorthFromBalances" -> money(m.netWorthFromBalances),
        "delta" -> money(m.delta)
      )
    )

  def accountsData(
      toleranceDays: Int,
      accounts: List[Account],
      transfers: TransferReport,
      netWorth: NetWorthReport
  ): Json =
    Json.Obj(
      List(
        "toleranceDays" -> Json.Num(toleranceDays.toLong),
        "accounts" -> Json.Arr(
          accounts.map { account =>
            Json.Obj(
              List(
                "id" -> Json.Str(account.id.value),
                "currency" -> Json.Str(account.currency.code),
                "openingBalance" -> money(account.openingBalance),
                "rowCount" -> Json.Num(account.statement.rowCount.toLong),
                "checks" -> checks(Checks.run(account.statement))
              )
            )
          }
        ),
        "transfers" -> Json.Arr(transfers.transfers.map(transfer)),
        "unpairedTransferCandidates" -> Json.Num(transfers.unpairedTransferCandidates.toLong),
        "netWorthConserved" -> Json.Bool(transfers.netWorthConserved),
        "netWorth" -> Json.Arr(netWorth.points.map(netWorthPoint)),
        "netWorthTotal" -> Json.Obj(
          List(
            "netWorth" -> money(netWorth.total.netWorth),
            "netWorthFromBalances" -> money(netWorth.total.netWorthFromBalances),
            "delta" -> money(netWorth.total.delta),
            "agrees" -> Json.Bool(netWorth.total.agrees)
          )
        ),
        "netWorthMismatches" -> Json.Arr(netWorth.mismatches.map(netWorthMismatch))
      )
    )
