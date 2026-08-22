from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)

TRAJECTORY_COLUMNS = [
    "boxer",
    "first_ranked_year",
    "last_ranked_year",
    "presence",
    "peak_year",
    "peak_rank",
    "peak_score",
    "peak_offset",
    "last_rank",
    "integral",
]


class TrajectoryDigest:
    """Per-boxer functionals over snapshot rank trajectories."""

    def compute(self, timeline: pd.DataFrame) -> pd.DataFrame:
        """Reduce the long timeline to one row per boxer.

        Args:
            timeline: Long-format DataFrame (snapshot_year, rank, boxer, score).

        Returns:
            DataFrame with peak and integral functionals, best careers first.
        """
        if timeline.empty:
            LOGGER.warning("Empty timeline — returning empty trajectory digest")
            return pd.DataFrame(columns=TRAJECTORY_COLUMNS)

        ordered = timeline.sort_values(["boxer", "snapshot_year"])
        grouped = ordered.groupby("boxer")
        digest = grouped.agg(
            first_ranked_year=("snapshot_year", "min"),
            last_ranked_year=("snapshot_year", "max"),
            presence=("snapshot_year", "count"),
            integral=("score", "sum"),
            peak_rank=("rank", "min"),
            last_rank=("rank", "last"),
        )

        peak_rows = ordered.loc[grouped["rank"].idxmin()].set_index("boxer")
        digest["peak_year"] = peak_rows["snapshot_year"]
        digest["peak_score"] = peak_rows["score"]
        digest = digest.reset_index()
        digest["peak_offset"] = digest["peak_year"] - digest["first_ranked_year"]

        digest = digest.sort_values(
            ["peak_rank", "integral"], ascending=[True, False]
        ).reset_index(drop=True)
        LOGGER.info(f"Trajectory digest built for {len(digest)} boxers")
        return digest[TRAJECTORY_COLUMNS]
