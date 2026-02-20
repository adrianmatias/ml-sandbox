from __future__ import annotations

import pandas as pd
import pytest

from topbox.conf import ConfPagerank
from topbox.pagerank import compute_ranks


@pytest.fixture
def sample_df() -> pd.DataFrame:
    data = [
        {"boxer_a": "A", "boxer_b": "B", "is_a_win": True, "date": "2024-01-01"},
        {"boxer_a": "A", "boxer_b": "D", "is_a_win": True, "date": "2024-02-01"},
        {"boxer_a": "B", "boxer_b": "C", "is_a_win": False, "date": "2023-12-01"},
    ]
    return pd.DataFrame(data)


class TestComputeRanks:
    def test_loser_to_winner_mode(self, sample_df: pd.DataFrame) -> None:
        conf = ConfPagerank(top_n=2)
        ranks = compute_ranks(sample_df, conf, mode="loser_to_winner")
        assert len(ranks) == 2
        assert isinstance(ranks[0], tuple)
        assert ranks[0][1] > 0
        boxer_names = [r[0] for r in ranks]
        assert "A" in boxer_names

    def test_winner_to_loser_mode(self, sample_df: pd.DataFrame) -> None:
        conf = ConfPagerank(top_n=3)
        ranks = compute_ranks(sample_df, conf, mode="winner_to_loser")
        assert len(ranks) == 3
        # B and D should rank higher as losers
        boxer_names = [r[0] for r in ranks]
        assert "B" in boxer_names or "D" in boxer_names

    def test_undirected_mode(self, sample_df: pd.DataFrame) -> None:
        conf = ConfPagerank(top_n=2)
        ranks = compute_ranks(sample_df, conf, mode="undirected")
        assert len(ranks) == 2
        assert isinstance(ranks[0], tuple)
