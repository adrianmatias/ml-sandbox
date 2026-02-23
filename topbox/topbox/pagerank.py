from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

import networkx as nx
import pandas as pd

from topbox.conf import ConfPagerank

LOGGER = logging.getLogger(__name__)


@dataclass
class GraphBuilder:
    _NICKNAME_RULES = [
        (re.compile(r"\bCanelo\b", re.I), "Canelo Alvarez"),
        (re.compile(r"\bTank\b", re.I), "Gervonta Davis"),
        (re.compile(r"\bSugar Ray\b", re.I), "Sugar Ray Leonard"),
        (re.compile(r"\bRay Leonard\b", re.I), "Sugar Ray Leonard"),
        (re.compile(r"\s+Jr\.?", re.I), " Jr"),
        (re.compile(r"\s+Sr\.?", re.I), " Sr"),
    ]

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
        df["is_a_win"] = df["is_a_win"] ^ swapped
        df["boxer_a"] = df["_key_a"]
        df["boxer_b"] = df["_key_b"]
        df = df.drop_duplicates(subset=["boxer_a", "boxer_b", "date"])
        df = df.drop(columns=["_key_a", "_key_b"]).reset_index(drop=True)
        LOGGER.info(f"Deduped {before} → {len(df)} rows")
        return df

    def build_graph(self, df: pd.DataFrame, mode: str) -> nx.Graph | nx.DiGraph:
        G = nx.Graph() if mode == "undirected" else nx.DiGraph()
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
    df: pd.DataFrame, conf: ConfPagerank, mode: str = "loser_to_winner"
) -> pd.DataFrame:
    """Compute PageRank and return ready-to-save DataFrame with 'Rank' column."""
    builder = GraphBuilder()
    LOGGER.info(f"Computing PageRank on {len(df)} raw rows, mode={mode}")

    df = builder.dedup(df)
    G = builder.build_graph(df, mode)

    pr = nx.pagerank(G, alpha=conf.alpha, max_iter=conf.max_iter, tol=conf.tol)
    ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)[: conf.top_n]

    result_df = pd.DataFrame(ranked, columns=["boxer", "score"])
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    result_df["score"] = result_df["score"].round(4)

    LOGGER.info(f"Top 10 after normalization: {result_df.head(10).to_dict('records')}")
    return result_df
