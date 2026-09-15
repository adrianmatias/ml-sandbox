"""Ledger module — Kutxa statement parsing, checks, and multi-account
reconciliation, implementing the frozen specification ``spec/ledger-spec.md``
(v1, 2026-09-15).

Public surface:

* :mod:`ledger.model`   — domain values and statement parsing (spec §2, §4)
* :mod:`ledger.checks`  — the three baseline checks (spec §3)
* :mod:`ledger.accounts`— iteration A: multi-account reconciliation and
  transfer detection (spec §5)
* :mod:`ledger.envelope`— the frozen JSON envelope (spec §6)

§4.1 ("a binary float must not be able to enter the domain model") is enforced
at the single entry point :func:`ledger.strictdecimal.as_money`: floats are
refused by a runtime type check, so no amount in this module tree is ever a
``float``, and every monetary value is an exact :class:`decimal.Decimal`.
"""

from .envelope import fail, ok

__all__ = ["ok", "fail", "__version_info__"]

#: Contract version carried in every envelope (spec §6).
__version_info__ = 1
