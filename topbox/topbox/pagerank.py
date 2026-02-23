from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd
import unicodedata

from topbox.conf import ConfPagerank

LOGGER = logging.getLogger(__name__)


@dataclass
class GraphBuilder:
    def normalize_name(self, name: str) -> str:
        if not isinstance(name, str):
            name = str(name)
        # Strip diacritics (handles every Spanish tilde, French accent, etc.)
        nfkd = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in nfkd if not unicodedata.combining(c))
        # Clean suffixes and punctuation
        name = (
            name.replace(" Jr.", " Jr")
            .replace(" Jr", " Jr")
            .replace(" Sr.", " Sr")
            .replace(".", "")
            .replace("'", "")
            .strip()
        )
        # Load boxing-specific canonical mapping from JSON
        mapping_path = Path(__file__).parent / "name_mapping.json"
        with mapping_path.open() as f:
            mapping = json.load(f)
        return mapping.get(name, name)

    def canonical_pair(self, a: str, b: str) -> tuple[str, str]:
        a = self.normalize_name(a)
        b = self.normalize_name(b)
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
            winner = self.normalize_name(
                row["boxer_a"] if row["is_a_win"] else row["boxer_b"]
            )
            loser = self.normalize_name(
                row["boxer_b"] if row["is_a_win"] else row["boxer_a"]
            )
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
    builder = GraphBuilder()
    LOGGER.info(f"Computing PageRank on {len(df)} raw rows, mode={mode}")
    df = builder.dedup(df)
    G = builder.build_graph(df, mode)
    pr = nx.pagerank(G, alpha=conf.alpha, max_iter=conf.max_iter, tol=conf.tol)
    top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[: conf.top_n]
    LOGGER.info(f"Top ranks after normalization: {top[:10]}")
    return top
