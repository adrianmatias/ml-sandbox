from __future__ import annotations

import pandas as pd
import pytest

from topbox.trajectory_digest import TrajectoryDigest


def _row(year: int, rank: int, boxer: str, score: float) -> dict:
    return {"snapshot_year": year, "rank": rank, "boxer": boxer, "score": score}


@pytest.fixture
def timeline_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(2000, 1, "Alpha", 0.50),
            _row(2001, 1, "Alpha", 0.50),
            _row(2002, 2, "Alpha", 0.30),
            _row(2001, 2, "Beta", 0.30),
            _row(2002, 1, "Beta", 0.50),
            _row(2002, 3, "Gamma", 0.20),
        ]
    )


class TestTrajectoryDigest:
    def test_empty_timeline_returns_columns(self) -> None:
        empty = pd.DataFrame(columns=["snapshot_year", "rank", "boxer", "score"])
        digest = TrajectoryDigest().compute(empty)
        assert digest.empty
        assert list(digest.columns) == list(TrajectoryDigest().compute(empty).columns)

    def test_one_row_per_boxer(self, timeline_df: pd.DataFrame) -> None:
        digest = TrajectoryDigest().compute(timeline_df)
        assert len(digest) == 3
        assert set(digest["boxer"]) == {"Alpha", "Beta", "Gamma"}

    def test_peak_detection(self, timeline_df: pd.DataFrame) -> None:
        digest = TrajectoryDigest().compute(timeline_df).set_index("boxer")
        assert digest.loc["Alpha", "peak_rank"] == 1
        assert digest.loc["Alpha", "peak_year"] == 2000
        assert digest.loc["Beta", "peak_rank"] == 1
        assert digest.loc["Beta", "peak_year"] == 2002

    def test_integral_sums_all_scores(self, timeline_df: pd.DataFrame) -> None:
        digest = TrajectoryDigest().compute(timeline_df).set_index("boxer")
        assert digest.loc["Alpha", "integral"] == pytest.approx(1.30)
        assert digest.loc["Beta", "integral"] == pytest.approx(0.80)

    def test_first_last_and_presence(self, timeline_df: pd.DataFrame) -> None:
        digest = TrajectoryDigest().compute(timeline_df).set_index("boxer")
        assert digest.loc["Alpha", "first_ranked_year"] == 2000
        assert digest.loc["Alpha", "last_ranked_year"] == 2002
        assert digest.loc["Alpha", "presence"] == 3
        assert digest.loc["Gamma", "presence"] == 1

    def test_peak_offset_years_from_debut(self, timeline_df: pd.DataFrame) -> None:
        digest = TrajectoryDigest().compute(timeline_df).set_index("boxer")
        assert digest.loc["Alpha", "peak_offset"] == 0
        assert digest.loc["Beta", "peak_offset"] == 1

    def test_last_rank_reflects_final_snapshot(self, timeline_df: pd.DataFrame) -> None:
        digest = TrajectoryDigest().compute(timeline_df).set_index("boxer")
        assert digest.loc["Alpha", "last_rank"] == 2
        assert digest.loc["Beta", "last_rank"] == 1

    def test_sorted_by_peak_then_integral(self, timeline_df: pd.DataFrame) -> None:
        digest = TrajectoryDigest().compute(timeline_df)
        assert digest["boxer"].tolist() == ["Alpha", "Beta", "Gamma"]

    def test_peak_tie_resolves_to_earliest_year(self) -> None:
        timeline = pd.DataFrame(
            [_row(2000, 1, "Alpha", 0.60), _row(2001, 1, "Alpha", 0.55)]
        )
        digest = TrajectoryDigest().compute(timeline)
        assert digest.iloc[0]["peak_year"] == 2000
