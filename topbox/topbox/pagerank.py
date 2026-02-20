from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import pandas as pd

from topbox.conf import ConfPagerank

LOGGER = logging.getLogger(__name__)


@dataclass
class GraphBuilder:
    def canonical_pair(self, a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        canon = df.apply(
            lambda r: self.canonical_pair(r["boxer_a"], r["boxer_b"]), axis=1
        )
        df = df.copy()
        df["_key_a"] = canon.str[0]
        df["_key_b"] = canon.str[1]
        swapped = df["_key_a"] != df["boxer_a"]
        df["is_a_win"] = df["is_a_win"] ^ swapped
        df["boxer_a"] = df["_key_a"]
        df["boxer_b"] = df["_key_b"]
        df = df.drop_duplicates(subset=["boxer_a", "boxer_b", "date"])
        df = df.drop(columns=["_key_a", "_key_b"]).reset_index(drop=True)
        LOGGER.info(f"Deduped {before} → {len(df)} rows ({before - len(df)} removed)")
        return df

    def build_graph(self, df: pd.DataFrame, mode: str) -> nx.Graph | nx.DiGraph:
        G: nx.Graph | nx.DiGraph = nx.Graph() if mode == "undirected" else nx.DiGraph()
        for _, row in df.iterrows():
            if pd.isna(row["date"]):
                continue
            winner = row["boxer_a"] if row["is_a_win"] else row["boxer_b"]
            loser = row["boxer_b"] if row["is_a_win"] else row["boxer_a"]
            if mode == "loser_to_winner":
                G.add_edge(loser, winner)
            elif mode == "winner_to_loser":
                G.add_edge(winner, loser)
            else:
                G.add_edge(winner, loser)
        return G


def compute_ranks(
    df: pd.DataFrame, conf: ConfPagerank, mode: str
) -> list[tuple[str, float]]:
    """Compute PageRank scores for boxers.

    Deduplicates fights before graph construction so mirror rows from
    different boxer profiles are counted only once.

    Args:
        df: Raw match dataset DataFrame.
        conf: Pagerank config.
        mode: One of loser_to_winner, winner_to_loser, undirected.

    Returns:
        Top N (boxer, score) tuples sorted descending by score.
    """
    builder = GraphBuilder()
    LOGGER.info(f"Computing PageRank on {len(df)} raw rows, mode={mode}")
    df = builder.dedup(df)
    G = builder.build_graph(df, mode)
    pr = nx.pagerank(G, alpha=conf.alpha, max_iter=conf.max_iter, tol=conf.tol)
    top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[: conf.top_n]
    LOGGER.info(f"Top ranks: {top}")
    return top
