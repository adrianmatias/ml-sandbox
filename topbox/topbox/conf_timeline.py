from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfTimeline:
    """Configuration for year-end PageRank snapshots.

    Args:
        year_first: First snapshot year.
        year_last: Last snapshot year; None resolves to the current year.
        step: Year step between snapshots.
        min_fights: Minimum fights in a window for the snapshot to be computed.
        top_n: Maximum ranked fighters per snapshot.
    """

    year_first: int = 1950
    year_last: int | None = None
    step: int = 1
    min_fights: int = 30
    top_n: int = 5000
