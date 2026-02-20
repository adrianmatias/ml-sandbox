from __future__ import annotations

import logging
from typing import List, Tuple

import networkx as nx
import pandas as pd

from topbox.conf import ConfPagerank

LOGGER = logging.getLogger(__name__)


def compute_ranks(
    df: pd.DataFrame, conf: ConfPagerank, mode: str
) -> List[Tuple[str, float]]:
    """Compute PageRank scores for boxers.

    Args:
        df: Match dataset DataFrame.
        conf: Pagerank config.

    Returns:
        Top N boxers with scores.
    """
    LOGGER.info(f"Computing PageRank on {len(df)} matches with mode {mode}")
    if mode == "undirected":
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    for _, row in df.iterrows():
        if pd.isna(row["date"]):
            continue
        winner = row["boxer_a"] if row["is_a_win"] else row["boxer_b"]
        loser = row["boxer_b"] if row["is_a_win"] else row["boxer_a"]
        if mode == "loser_to_winner":
            G.add_edge(loser, winner)
        elif mode == "winner_to_loser":
            G.add_edge(winner, loser)
        elif mode == "undirected":
            G.add_edge(winner, loser)
    pr = nx.pagerank(G, alpha=conf.alpha, max_iter=conf.max_iter, tol=conf.tol)
    top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[: conf.top_n]
    LOGGER.info(f"Top ranks: {top}")
    return top
