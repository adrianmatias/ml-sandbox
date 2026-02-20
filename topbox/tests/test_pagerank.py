from __future__ import annotations

import pandas as pd
import pytest

from topbox.conf import ConfPagerank
from topbox.pagerank import GraphBuilder, compute_ranks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df(*rows: dict) -> pd.DataFrame:
    """Build a minimal match DataFrame from dicts."""
    return pd.DataFrame(rows)


def _fight(a: str, b: str, win: bool, date: str = "2024-01-01") -> dict:
    return {"boxer_a": a, "boxer_b": b, "is_a_win": win, "date": pd.Timestamp(date)}


# ---------------------------------------------------------------------------
# GraphBuilder.canonical_pair
# ---------------------------------------------------------------------------


class TestCanonicalPair:
    def test_already_sorted(self) -> None:
        builder = GraphBuilder()
        assert builder.canonical_pair("Ali", "Frazier") == ("Ali", "Frazier")

    def test_reversed(self) -> None:
        builder = GraphBuilder()
        assert builder.canonical_pair("Frazier", "Ali") == ("Ali", "Frazier")

    def test_equal(self) -> None:
        builder = GraphBuilder()
        assert builder.canonical_pair("X", "X") == ("X", "X")


# ---------------------------------------------------------------------------
# GraphBuilder.dedup
# ---------------------------------------------------------------------------


class TestDedup:
    def test_removes_mirror_duplicate(self) -> None:
        builder = GraphBuilder()
        df = _df(
            _fight("Ali", "Frazier", True),
            _fight("Frazier", "Ali", False),  # same fight from Frazier's profile
        )
        assert len(builder.dedup(df)) == 1

    def test_canonical_ordering_applied(self) -> None:
        builder = GraphBuilder()
        # Frazier > Ali, so row gets swapped and win flag flipped
        df = _df(_fight("Frazier", "Ali", False))
        out = builder.dedup(df)
        assert out.iloc[0]["boxer_a"] == "Ali"
        assert out.iloc[0]["boxer_b"] == "Frazier"
        assert out.iloc[0]["is_a_win"] == True  # noqa: E712

    def test_win_flag_unchanged_when_already_canonical(self) -> None:
        builder = GraphBuilder()
        df = _df(_fight("Ali", "Frazier", True))
        out = builder.dedup(df)
        assert out.iloc[0]["is_a_win"] == True  # noqa: E712

    def test_different_dates_kept_as_separate_fights(self) -> None:
        builder = GraphBuilder()
        df = _df(
            _fight("Ali", "Frazier", True, "1971-03-08"),
            _fight("Ali", "Frazier", False, "1974-01-28"),
        )
        assert len(builder.dedup(df)) == 2

    def test_no_temp_columns_in_output(self) -> None:
        builder = GraphBuilder()
        out = builder.dedup(_df(_fight("Ali", "Frazier", True)))
        assert "_key_a" not in out.columns
        assert "_key_b" not in out.columns

    def test_index_reset_after_dedup(self) -> None:
        builder = GraphBuilder()
        df = _df(_fight("Ali", "Frazier", True), _fight("Frazier", "Ali", False))
        out = builder.dedup(df)
        assert list(out.index) == [0]

    def test_unrelated_fights_all_kept(self) -> None:
        builder = GraphBuilder()
        df = _df(
            _fight("A", "B", True),
            _fight("B", "C", False),
            _fight("A", "D", True),
        )
        assert len(builder.dedup(df)) == 3


# ---------------------------------------------------------------------------
# GraphBuilder.build_graph
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def _deduped(self, *rows: dict) -> pd.DataFrame:
        builder = GraphBuilder()
        return builder.dedup(_df(*rows))

    def test_loser_to_winner_edge_direction(self) -> None:
        builder = GraphBuilder()
        df = self._deduped(_fight("Ali", "Frazier", True))
        G = builder.build_graph(df, "loser_to_winner")
        assert G.has_edge("Frazier", "Ali")
        assert not G.has_edge("Ali", "Frazier")

    def test_winner_to_loser_edge_direction(self) -> None:
        builder = GraphBuilder()
        df = self._deduped(_fight("Ali", "Frazier", True))
        G = builder.build_graph(df, "winner_to_loser")
        assert G.has_edge("Ali", "Frazier")
        assert not G.has_edge("Frazier", "Ali")

    def test_undirected_graph_type(self) -> None:
        import networkx as nx

        builder = GraphBuilder()
        df = self._deduped(_fight("Ali", "Frazier", True))
        G = builder.build_graph(df, "undirected")
        assert isinstance(G, nx.Graph)
        assert not isinstance(G, nx.DiGraph)

    def test_na_date_row_skipped(self) -> None:
        builder = GraphBuilder()
        df = _df(
            {"boxer_a": "Ali", "boxer_b": "Frazier", "is_a_win": True, "date": pd.NaT}
        )
        G = builder.build_graph(df, "loser_to_winner")
        assert G.number_of_edges() == 0


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

    def test_loser_to_winner_returns_top_n(self, sample_df: pd.DataFrame) -> None:
        ranks = compute_ranks(sample_df, ConfPagerank(top_n=2), "loser_to_winner")
        assert len(ranks) == 2
        assert isinstance(ranks[0], tuple)
        assert ranks[0][1] > 0
        assert "A" in [r[0] for r in ranks]

    def test_winner_to_loser_returns_top_n(self, sample_df: pd.DataFrame) -> None:
        ranks = compute_ranks(sample_df, ConfPagerank(top_n=3), "winner_to_loser")
        assert len(ranks) == 3
        names = [r[0] for r in ranks]
        assert "B" in names or "D" in names

    def test_undirected_mode(self, sample_df: pd.DataFrame) -> None:
        ranks = compute_ranks(sample_df, ConfPagerank(top_n=2), "undirected")
        assert len(ranks) == 2

    def test_mirror_rows_count_once(self) -> None:
        # A beat B, recorded from both profiles; graph should have one edge, not two
        df = _df(
            _fight("A", "B", True),
            _fight("B", "A", False),  # mirror
        )
        ranks_mirror = compute_ranks(df, ConfPagerank(top_n=2), "loser_to_winner")
        df_single = _df(_fight("A", "B", True))
        ranks_single = compute_ranks(
            df_single, ConfPagerank(top_n=2), "loser_to_winner"
        )
        # Scores must be identical — dedup collapses to the same graph
        assert ranks_mirror == ranks_single
