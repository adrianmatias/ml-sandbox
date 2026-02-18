from __future__ import annotations

import logging
from typing import List, Tuple

import networkx as nx
import pandas as pd

from topbox.conf import ConfPagerank

LOGGER = logging.getLogger(__name__)


def compute_ranks(df: pd.DataFrame, conf: ConfPagerank) -> List[Tuple[str, float]]:
    """Compute PageRank scores for boxers.

    Args:
        df: Match dataset DataFrame.
        conf: Pagerank config.

    Returns:
        Top N boxers with scores.
    """
    LOGGER.info(f"Computing PageRank on {len(df)} matches")
    G = nx.DiGraph()
    for _, row in df.iterrows():
        if pd.isna(row["date"]):
            continue
        if row["is_a_win"]:
            G.add_edge(row["boxer_a"], row["boxer_b"])
        else:
            G.add_edge(row["boxer_b"], row["boxer_a"])
    pr = nx.pagerank(G, alpha=conf.alpha, max_iter=conf.max_iter, tol=conf.tol)
    top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[: conf.top_n]
    LOGGER.info(f"Top ranks: {top}")
    return top
