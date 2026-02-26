from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class PageRankBox:
    """PageRank (loser-to-winner DiGraph). Handles draws via mutual 0.5-weight edges."""

    _NICKNAME_RULES = [
        (re.compile(r"\bCanelo\b", re.I), "Canelo Alvarez"),
        (re.compile(r"\bTank\b", re.I), "Gervonta Davis"),
        (re.compile(r"\bRay Leonard\b", re.I), "Sugar Ray Leonard"),
        (re.compile(r"\s+Jr\.?", re.I), " Jr"),
        (re.compile(r"\s+Sr\.?", re.I), " Sr"),
    ]

    def __init__(
        self,
        alpha: float = 0.85,
        max_iter: int = 1000,
        tol: float = 1.0e-6,
        top_n: int = 10,
        draw_share: float = 0.5,
        is_consolidated: bool = False,
    ) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.top_n = top_n
        self.draw_share = draw_share
        self.is_consolidated = is_consolidated
        LOGGER.info(f"{self.__dict__}")

    def normalize_name(self, name: str) -> str:
        if not isinstance(name, str):
            name = str(name)

        nfkd = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in nfkd if not unicodedata.combining(c))

        name = re.sub(r"\s+", " ", name).strip()
        name = name.replace(".", "").replace("'", "")

        for pattern, replacement in self._NICKNAME_RULES:
            name = pattern.sub(replacement, name)

        words = name.split()
        if words:
            cleaned = [words[0]]
            for word in words[1:]:
                if word.lower() != cleaned[-1].lower():
                    cleaned.append(word)
            name = " ".join(cleaned)

        return name.strip()

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
        df["is_a_win"] = df["is_a_win"].where(
            pd.isna(df["is_a_win"]), df["is_a_win"] ^ swapped
        )
        df["boxer_a"] = df["_key_a"]
        df["boxer_b"] = df["_key_b"]
        df = df.drop_duplicates(subset=["boxer_a", "boxer_b", "date"])
        df = df.drop(columns=["_key_a", "_key_b"]).reset_index(drop=True)
        LOGGER.info(f"Deduped {before} → {len(df)} rows")
        return df

    def edge_weight(
        self, date: pd.Timestamp, tau: float = 8.0, weight_min: float = 0.1
    ) -> float:
        """Exponential time-decay weight for a fight edge.

        Args:
            date: Fight date.
            tau: Half-life in years (controls decay rate).
            weight_min: Floor so fights never zeroed.

        Returns:
            Weight ≥ weight_min.
        """
        years_ago = datetime.now().year - date.year
        sign = 1 if self.is_consolidated else -1
        return max(weight_min, np.exp(sign * years_ago / tau))

    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        G = nx.DiGraph()
        for _, row in df.iterrows():
            if pd.isna(row["date"]):
                continue
            weight = self.edge_weight(row["date"])
            a = self.normalize_name(row["boxer_a"])
            b = self.normalize_name(row["boxer_b"])
            if pd.isna(row["is_a_win"]):
                half_weight = weight * self.draw_share
                G.add_edge(a, b, weight=half_weight)
                G.add_edge(b, a, weight=half_weight)
                continue
            is_a_win = row["is_a_win"]
            winner = a if is_a_win else b
            loser = b if is_a_win else a
            G.add_edge(loser, winner, weight=weight)
        return G

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute PageRank (loser-to-winner DiGraph) and return 'rank' DataFrame.
        Draws handled as mutual half-weight edges."""
        LOGGER.info(f"Computing PageRank on {len(df)} raw rows")

        df = self.dedup(df)
        G = self.build_graph(df)

        pr = nx.pagerank(G, alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
        ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)[: self.top_n]

        result_df = pd.DataFrame(ranked, columns=["boxer", "score"])
        result_df.insert(0, "rank", range(1, len(result_df) + 1))
        result_df["score"] = result_df["score"].round(4)

        LOGGER.info(
            f"Top 10 after normalization: {result_df.head(10).to_dict('records')}"
        )
        return result_df
