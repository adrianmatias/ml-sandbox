from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Match:
    boxer_a: str
    boxer_b: str
    is_a_win: bool | None
    date: str
