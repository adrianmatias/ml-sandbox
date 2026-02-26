from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import pytest

from topbox.page_rank_box import PageRankBox

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df(*rows: dict) -> pd.DataFrame:
    """Build a minimal match DataFrame from dicts."""
    return pd.DataFrame(rows)


def _fight(a: str, b: str, win: bool | None = True, date: str = "2024-01-01") -> dict:
    return {"boxer_a": a, "boxer_b": b, "is_a_win": win, "date": pd.Timestamp(date)}


# ---------------------------------------------------------------------------
# PageRankBox.edge_weight
# ---------------------------------------------------------------------------


class TestEdgeWeight:
    prb = PageRankBox()
    prb_cons = PageRankBox(is_consolidated=True)

    def get_date(self, years_ago: int) -> pd.Timestamp:
        return pd.Timestamp(datetime.now().year - years_ago, 6, 1)

    def test_current_year_is_near_one(self) -> None:
        w = self.prb.edge_weight(self.get_date(0))
        assert math.isclose(w, 1.0, rel_tol=1e-6)

    def test_one_tau_ago_is_one_over_e(self) -> None:
        # After exactly tau years the decay is exp(-1) ≈ 0.368
        w = self.prb.edge_weight(self.get_date(8))
        assert math.isclose(w, math.exp(-1), rel_tol=1e-6)

    def test_old_fight_hits_floor(self) -> None:
        # 100 years ago → exp(-100/8) ≈ 0.0, clamped to weight_min=0.1
        w = self.prb.edge_weight(self.get_date(100))
        assert w == pytest.approx(0.1)

    def test_weight_never_below_floor(self) -> None:
        for years_ago in range(0, 120, 10):
            assert self.prb.edge_weight(self.get_date(years_ago)) >= 0.1

    def test_consolidated_recent_no_boost(self) -> None:
        w = self.prb_cons.edge_weight(self.get_date(0))
        assert math.isclose(w, 1.0, rel_tol=1e-6)

    def test_consolidated_tau_ago_boosted(self) -> None:
        w = self.prb_cons.edge_weight(self.get_date(8))
        assert math.isclose(w, math.exp(1), rel_tol=1e-6)

    def test_decay_reversal(self) -> None:
        recent_old = self.prb.edge_weight(self.get_date(8))
        cons_old = self.prb_cons.edge_weight(self.get_date(8))
        assert recent_old < 1.0 < cons_old


# ---------------------------------------------------------------------------
# PageRankBox.canonical_pair
# ---------------------------------------------------------------------------


class TestCanonicalPair:
    def test_already_sorted(self) -> None:
        builder = PageRankBox()
        assert builder.canonical_pair("Ali", "Frazier") == ("Ali", "Frazier")

    def test_reversed(self) -> None:
        builder = PageRankBox()
        assert builder.canonical_pair("Frazier", "Ali") == ("Ali", "Frazier")

    def test_equal(self) -> None:
        builder = PageRankBox()
        assert builder.canonical_pair("X", "X") == ("X", "X")


# ---------------------------------------------------------------------------
# PageRankBox.dedup
# ---------------------------------------------------------------------------


class TestDedup:
    def test_removes_mirror_duplicate(self) -> None:
        builder = PageRankBox()
        df = _df(
            _fight("Ali", "Frazier", True),
            _fight("Frazier", "Ali", False),  # same fight from Frazier's profile
        )
        assert len(builder.dedup(df)) == 1

    def test_canonical_ordering_applied(self) -> None:
        builder = PageRankBox()
        # Frazier > Ali, so row gets swapped and win flag flipped
        df = _df(_fight("Frazier", "Ali", False))
        out = builder.dedup(df)
        assert out.iloc[0]["boxer_a"] == "Ali"
        assert out.iloc[0]["boxer_b"] == "Frazier"
        assert out.iloc[0]["is_a_win"] == True  # noqa: E712

    def test_win_flag_unchanged_when_already_canonical(self) -> None:
        builder = PageRankBox()
        df = _df(_fight("Ali", "Frazier", True))
        out = builder.dedup(df)
        assert out.iloc[0]["is_a_win"] == True  # noqa: E712

    def test_different_dates_kept_as_separate_fights(self) -> None:
        builder = PageRankBox()
        df = _df(
            _fight("Ali", "Frazier", True, "1971-03-08"),
            _fight("Ali", "Frazier", False, "1974-01-28"),
        )
        assert len(builder.dedup(df)) == 2

    def test_no_temp_columns_in_output(self) -> None:
        builder = PageRankBox()
        out = builder.dedup(_df(_fight("Ali", "Frazier", True)))
        assert "_key_a" not in out.columns
        assert "_key_b" not in out.columns

    def test_index_reset_after_dedup(self) -> None:
        builder = PageRankBox()
        df = _df(_fight("Ali", "Frazier", True), _fight("Frazier", "Ali", False))
        out = builder.dedup(df)
        assert list(out.index) == [0]

    def test_unrelated_fights_all_kept(self) -> None:
        builder = PageRankBox()
        df = _df(
            _fight("A", "B", True),
            _fight("B", "C", False),
            _fight("A", "D", True),
        )
        assert len(builder.dedup(df)) == 3

    def test_dedup_preserves_draw_flag(self) -> None:
        builder = PageRankBox()
        df = _df(_fight("Ali", "Frazier", None))
        out = builder.dedup(df)
        assert pd.isna(out["is_a_win"].iloc[0])

        df_swap = _df(_fight("Frazier", "Ali", None))
        out_swap = builder.dedup(df_swap)
        assert pd.isna(out_swap["is_a_win"].iloc[0])


# ---------------------------------------------------------------------------
# PageRankBox.build_graph
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def _deduped(self, *rows: dict) -> pd.DataFrame:
        builder = PageRankBox()
        return builder.dedup(_df(*rows))

    def test_loser_to_winner_edge_direction(self) -> None:
        builder = PageRankBox()
        df = self._deduped(_fight("Ali", "Frazier", True))
        G = builder.build_graph(df)
        assert G.has_edge("Frazier", "Ali")
        assert not G.has_edge("Ali", "Frazier")

    def test_winner_to_loser_edge_direction(self) -> None:
        builder = PageRankBox()
        df = self._deduped(_fight("Ali", "Frazier", True))
        G = builder.build_graph(df)
        assert not G.has_edge("Ali", "Frazier")

    def test_na_date_row_skipped(self) -> None:
        builder = PageRankBox()
        df = _df(
            {"boxer_a": "Ali", "boxer_b": "Frazier", "is_a_win": True, "date": pd.NaT}
        )
        G = builder.build_graph(df)
        assert G.number_of_edges() == 0

    def test_draw_adds_mutual_half_weight_edges(self) -> None:
        builder = PageRankBox()
        df = _df(_fight("Ali", "Frazier", None, "2024-01-01"))
        df_dedup = builder.dedup(df)
        G = builder.build_graph(df_dedup)
        full_weight = builder.edge_weight(pd.Timestamp("2024-01-01"))
        half_weight = full_weight * 0.5
        assert G.has_edge("Ali", "Frazier")
        assert G["Ali"]["Frazier"]["weight"] == pytest.approx(half_weight)
        assert G.has_edge("Frazier", "Ali")
        assert G["Frazier"]["Ali"]["weight"] == pytest.approx(half_weight)

    def test_draw_factor_custom_value(self) -> None:
        builder = PageRankBox(draw_share=0.25)
        df = _df(_fight("Ali", "Frazier", None, "2024-01-01"))
        G = builder.build_graph(builder.dedup(df))
        full_weight = builder.edge_weight(pd.Timestamp("2024-01-01"))
        quarter_weight = full_weight * 0.25
        assert G["Ali"]["Frazier"]["weight"] == pytest.approx(quarter_weight)


# ---------------------------------------------------------------------------
# compute_ranks — dedup integration
# ---------------------------------------------------------------------------


class TestComputeRanks:
    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return _df(
            _fight("A", "B", True, "2024-01-01"),
            _fight("A", "D", True, "2024-02-01"),
            _fight("B", "C", False, "2023-12-01"),
        )

    def test_returns_top_n(self, sample_df: pd.DataFrame) -> None:
        ranks = PageRankBox(top_n=2).compute(sample_df)
        assert len(ranks) == 2
        assert ranks.iloc[0]["score"] > 0
        assert "A" in ranks["boxer"].tolist()

    def test_winner_to_loser_returns_top_n(self, sample_df: pd.DataFrame) -> None:
        ranks = PageRankBox(top_n=3).compute(sample_df)
        assert len(ranks) == 3
        names = ranks["boxer"].tolist()
        assert "B" in names or "D" in names

    def test_mirror_rows_count_once(self) -> None:
        # A beat B, recorded from both profiles; graph should have one edge, not two
        df = _df(
            _fight("A", "B", True),
            _fight("B", "A", False),  # mirror
        )
        ranks_mirror = PageRankBox(top_n=2).compute(df)
        df_single = _df(_fight("A", "B", True))
        ranks_single = PageRankBox(top_n=2).compute(df_single)
        # Scores must be identical — dedup collapses to the same graph
        assert ranks_mirror.equals(ranks_single)
