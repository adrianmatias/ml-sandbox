"""The frozen JSON envelope (spec §6).

Key order is fixed by construction — plain ``dict`` insertion order, rendered
with ``sort_keys=False`` — so the bytes are reproducible run to run and
comparable with the other implementation of this contract.

    {"ok": true,  "module": "ledger", "version": 1, "data": {...}}
    {"ok": false, "module": "ledger", "version": 1, "error": {...}}

``json.dumps`` defaults are used apart from ``sort_keys``: ``ensure_ascii``
stays on so output is pure ASCII, separators stay at ``, `` / ``": "``, and no
timestamps, paths or locale-dependent formatting are ever placed in the
payload.
"""

from __future__ import annotations

import json

from .errors import LedgerError

MODULE = "ledger"
VERSION = 1

#: Exit codes (spec §6): well-formed input, unparseable input, usage error.
EXIT_OK = 0
EXIT_UNPARSEABLE = 1
EXIT_USAGE = 2


def ok(data: dict) -> dict:
    """Successful envelope; key order ``ok, module, version, data``."""
    return {"ok": True, "module": MODULE, "version": VERSION, "data": data}


def fail(error: LedgerError) -> dict:
    """Error envelope; key order ``ok, module, version, error``."""
    return {
        "ok": False,
        "module": MODULE,
        "version": VERSION,
        "error": error.envelope_error(),
    }


def render(envelope: dict) -> str:
    """Canonical text form of an envelope: one line, ASCII, no sorted keys.

    The result ends with exactly one newline so a shell redirect produces a
    conventional POSIX text file.
    """
    return json.dumps(envelope, sort_keys=False, ensure_ascii=True) + "\n"
