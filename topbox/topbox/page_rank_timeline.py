from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from topbox.conf_timeline import ConfTimeline
from topbox.page_rank_box import PageRankBox

LOGGER = logging.getLogger(__name__)

TIMELINE_COLUMNS = ["snapshot_year", "rank", "boxer", "score"]


class PageRankTimeline:
    """Year-end PageRank snapshots producing per-boxer rank trajectories."""

    def __init__(
        self,
        conf: ConfTimeline | None = None,
        page_rank_box: PageRankBox | None = None,
    ) -> None:
        self.conf = conf or ConfTimeline()
        self.page_rank_box = page_rank_box or PageRankBox(top_n=self.conf.top_n)
        LOGGER.info(f"{self.__dict__}")

    def snapshot_years(self) -> list[int]:
        """Resolve the configured snapshot years, including year_last."""
        year_last = self.conf.year_last or datetime.now().year
        return list(range(self.conf.year_first, year_last + 1, self.conf.step))

    def window_df(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Filter fights fought no later than the snapshot year."""
        dates = pd.to_datetime(df["date"], errors="coerce")
        return df[dates.dt.year <= year]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute one PageRank per snapshot year.

        Args:
            df: Match DataFrame (boxer_a, boxer_b, is_a_win, date).

        Returns:
            Long-format DataFrame with snapshot_year, rank, boxer, score.
        """
        frames: list[pd.DataFrame] = []
        for year in self.snapshot_years():
            window = self.window_df(df, year)
            if len(window) < self.conf.min_fights:
                LOGGER.info(
                    f"Skipping {year}: {len(window)} fights "
                    f"below min_fights={self.conf.min_fights}"
                )
                continue
            ranks = self.page_rank_box.compute(window, round_digits=None)
            ranks.insert(0, "snapshot_year", year)
            frames.append(ranks[["snapshot_year", "rank", "boxer", "score"]])
            LOGGER.info(f"Snapshot {year}: {len(ranks)} fighters ranked")

        if not frames:
            LOGGER.warning("No snapshot produced — returning empty timeline")
            return pd.DataFrame(columns=TIMELINE_COLUMNS)

        timeline = pd.concat(frames, ignore_index=True)
        LOGGER.info(
            f"Timeline built: {len(timeline)} rows across "
            f"{timeline['snapshot_year'].nunique()} snapshots"
        )
        return timeline
