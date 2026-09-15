package ledger

/**
 * Iteration A: transfer detection (spec 5.2) and the conservation invariant
 * (spec 5.4).
 *
 * A pairing is a value, not a convention: `Transfer` can only be built by
 * `TransferDetection.pair`, which is the single place that decides what
 * "matched amount and currency within tolerance" means. Nothing downstream
 * re-derives a date difference or an amount.
 */

/** One side of a pairing: which row, in which account, with the values it matched on. */
final case class PairedRow(
    accountIndex: Int,
    accountId: AccountId,
    line: Line,
    amount: Money,
    valueDate: Dated[Value]
)

final case class Transfer(outflow: PairedRow, inflow: PairedRow, amount: Money, lagDays: Int)

final case class TransferReport(transfers: List[Transfer], unpairedTransferCandidates: Int):
  /** Spec 5.4: summed over the rows that are marked as transfers, the change must be exactly zero. */
  def conservationResidual: Money = Money.sum(transfers.map(t => t.outflow.amount + t.inflow.amount))

  def netWorthConserved: Boolean = conservationResidual.isZero

object TransferDetection:

  /**
   * Greedy, deterministic pairing. Candidates are ordered by nearest feasible
   * value date, then by the lowest line number on the outflow side, then on the
   * inflow side; a row is consumed by at most one pairing (spec 5.2).
   */
  def detect(accounts: List[Account], toleranceDays: Int): TransferReport =
    val candidates = for
      outIndex <- accounts.indices.toList
      inIndex <- accounts.indices.toList
      if outIndex != inIndex
      if accounts(outIndex).currency == accounts(inIndex).currency
      outflow <- accounts(outIndex).statement.transactions if outflow.amount.isNegative
      inflow <- accounts(inIndex).statement.transactions if inflow.amount.isPositive
      if outflow.amount.abs == inflow.amount
      lag = outflow.valueDate.daysUntil(inflow.valueDate)
      if lag >= 0 && lag <= toleranceDays
    yield Candidate(
      PairedRow(outIndex, accounts(outIndex).id, outflow.line, outflow.amount, outflow.valueDate),
      PairedRow(inIndex, accounts(inIndex).id, inflow.line, inflow.amount, inflow.valueDate),
      lag
    )

    val ordered = candidates.sortBy(c =>
      (c.lagDays, c.outflow.accountIndex, c.outflow.line.value, c.inflow.accountIndex, c.inflow.line.value)
    )

    val used = scala.collection.mutable.Set.empty[(Int, Int)]
    val chosen = List.newBuilder[Transfer]
    ordered.foreach { candidate =>
      val outflowKey = (candidate.outflow.accountIndex, candidate.outflow.line.value)
      val inflowKey = (candidate.inflow.accountIndex, candidate.inflow.line.value)
      if !used.contains(outflowKey) && !used.contains(inflowKey) then
        used += outflowKey
        used += inflowKey
        chosen += Transfer(candidate.outflow, candidate.inflow, candidate.outflow.amount.abs, candidate.lagDays)
    }

    TransferReport(
      chosen.result().sortBy(t => (t.outflow.accountIndex, t.outflow.line.value)),
      // Reserved for a later iteration (spec 5.3).
      unpairedTransferCandidates = 0
    )

  private final case class Candidate(outflow: PairedRow, inflow: PairedRow, lagDays: Int)

// ---------------------------------------------------------------------------
// Net worth observation points (spec 5.5)
// ---------------------------------------------------------------------------

final case class NetWorthPoint(
    account: AccountId,
    netWorth: Money,
    netWorthFromBalances: Money
):
  def delta: Money = netWorthFromBalances - netWorth
  def agrees: Boolean = delta.isZero

final case class NetWorthTotal(netWorth: Money, netWorthFromBalances: Money):
  def delta: Money = netWorthFromBalances - netWorth
  def agrees: Boolean = delta.isZero

final case class NetWorthMismatch(scope: String, netWorth: Money, netWorthFromBalances: Money, delta: Money)

final case class NetWorthReport(
    points: List[NetWorthPoint],
    total: NetWorthTotal,
    mismatches: List[NetWorthMismatch]
)

object NetWorth:

  /**
   * `netWorth(i) = openingBalance + sum(amount)` and
   * `netWorthFromBalances(i) = balanceAfter` of the last row, observed once per
   * account and once over all accounts. Every disagreement is reported.
   */
  def observe(accounts: List[Account]): NetWorthReport =
    val points = accounts.map { account =>
      NetWorthPoint(account.id, account.netWorth, account.closingBalance)
    }
    val total = NetWorthTotal(
      Money.sum(points.map(_.netWorth)),
      Money.sum(points.map(_.netWorthFromBalances))
    )
    val mismatches =
      points.filterNot(_.agrees).map(p => NetWorthMismatch(p.account.value, p.netWorth, p.netWorthFromBalances, p.delta)) ++
        (if total.agrees then Nil
         else List(NetWorthMismatch("TOTAL", total.netWorth, total.netWorthFromBalances, total.delta)))
    NetWorthReport(points, total, mismatches)
