from __future__ import annotations

import pandas as pd
import pytest

from topbox.conf_timeline import ConfTimeline
from topbox.page_rank_timeline import PageRankTimeline


def _fight(a: str, b: str, win: bool | None = True, date: str = "2000-01-01") -> dict:
    return {"boxer_a": a, "boxer_b": b, "is_a_win": win, "date": pd.Timestamp(date)}


def _month_day(year: int, month: int) -> str:
    day = (month * 2) % 27 + 1
    return f"{year}-{month:02d}-{day:02d}"


@pytest.fixture
def match_df() -> pd.DataFrame:
    rows = []
    for year in range(2000, 2005):
        for month in range(1, 13):
            date = _month_day(year, month)
            rows.append(_fight("A", f"Opp{year}{month}", True, date))
            if month % 3 == 0:
                rows.append(_fight("B", f"Opp{year}{month}", False, date))
    return pd.DataFrame(rows)


class TestSnapshotYears:
    def test_explicit_range_inclusive(self) -> None:
        conf = ConfTimeline(year_first=2000, year_last=2003)
        assert PageRankTimeline(conf).snapshot_years() == [2000, 2001, 2002, 2003]

    def test_step_applied(self) -> None:
        conf = ConfTimeline(year_first=2000, year_last=2004, step=2)
        assert PageRankTimeline(conf).snapshot_years() == [2000, 2002, 2004]

    def test_year_last_none_reaches_current_year(self) -> None:
        years = PageRankTimeline(ConfTimeline(year_first=2020)).snapshot_years()
        assert years[-1] >= 2026


class TestWindowDf:
    def test_future_fights_excluded(self, match_df: pd.DataFrame) -> None:
        window = PageRankTimeline().window_df(match_df, 2002)
        years = pd.to_datetime(window["date"]).dt.year
        assert years.max() == 2002

    def test_all_past_kept(self, match_df: pd.DataFrame) -> None:
        window = PageRankTimeline().window_df(match_df, 2030)
        assert len(window) == len(match_df)


class TestPageRankTimelineCompute:
    def test_column_layout(self, match_df: pd.DataFrame) -> None:
        timeline = PageRankTimeline(
            ConfTimeline(year_first=2001, year_last=2004, min_fights=5)
        ).compute(match_df)
        assert list(timeline.columns) == ["snapshot_year", "rank", "boxer", "score"]

    def test_thin_windows_skipped(self, match_df: pd.DataFrame) -> None:
        timeline = PageRankTimeline(
            ConfTimeline(year_first=2001, year_last=2004, min_fights=10_000)
        ).compute(match_df)
        assert timeline.empty

    def test_empty_result_has_columns(self, match_df: pd.DataFrame) -> None:
        timeline = PageRankTimeline(
            ConfTimeline(year_first=2001, year_last=2004, min_fights=10_000)
        ).compute(match_df)
        assert list(timeline.columns) == ["snapshot_year", "rank", "boxer", "score"]

    def test_ranks_start_at_one_per_snapshot(self, match_df: pd.DataFrame) -> None:
        timeline = PageRankTimeline(
            ConfTimeline(year_first=2001, year_last=2004, min_fights=5)
        ).compute(match_df)
        min_ranks = timeline.groupby("snapshot_year")["rank"].min()
        assert (min_ranks == 1).all()

    def test_snapshot_years_strictly_increasing(self, match_df: pd.DataFrame) -> None:
        timeline = PageRankTimeline(
            ConfTimeline(year_first=2001, year_last=2004, step=1, min_fights=5)
        ).compute(match_df)
        years = timeline["snapshot_year"].unique()
        assert list(years) == sorted(years)

    def test_dominant_winner_outranks_loser(self, match_df: pd.DataFrame) -> None:
        timeline = PageRankTimeline(
            ConfTimeline(year_first=2004, year_last=2004, min_fights=5)
        ).compute(match_df)
        rank_a = timeline.loc[timeline["boxer"] == "A", "rank"].iloc[0]
        rank_b = timeline.loc[timeline["boxer"] == "B", "rank"].iloc[0]
        assert rank_a < rank_b

    def test_scores_keep_full_precision(self, match_df: pd.DataFrame) -> None:
        timeline = PageRankTimeline(
            ConfTimeline(year_first=2004, year_last=2004, min_fights=5)
        ).compute(match_df)
        beyond_four = (timeline["score"] * 10**4 % 1 > 1e-9).any()
        assert beyond_four

    def test_late_fighter_absent_from_early_snapshot(
        self, match_df: pd.DataFrame
    ) -> None:
        late = _df_append(match_df, _fight("Latecomer", "A", False, "2004-06-01"))
        timeline = PageRankTimeline(
            ConfTimeline(year_first=2003, year_last=2004, min_fights=5)
        ).compute(late)
        early = timeline[timeline["snapshot_year"] == 2003]["boxer"]
        late_snap = timeline[timeline["snapshot_year"] == 2004]["boxer"]
        assert "Latecomer" not in set(early)
        assert "Latecomer" in set(late_snap)


def _df_append(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)
