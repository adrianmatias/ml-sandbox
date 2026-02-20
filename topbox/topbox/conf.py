from __future__ import annotations

from dataclasses import dataclass

"""Configuration dataclasses."""


@dataclass(frozen=True)
class ConfCrawler:
    """Crawler configuration.

    Args:
        base_url: Base URL for crawling.
        user_agent: User agent string.
    """

    base_url: str = "https://box.live"
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"
    )


@dataclass(frozen=True)
class ConfDataset:
    """Dataset configuration.

    Args:
        min_date: Minimum date filter (YYYY-MM-DD).
        save_path: Path to save dataset.
    """

    min_date: str | None = None
    save_path: str = "data/matches.parquet"


@dataclass(frozen=True)
class ConfPagerank:
    """PageRank configuration.

    Args:
        alpha: Damping factor.
        max_iter: Maximum iterations.
        tol: Tolerance for convergence.
        top_n: Number of top boxers to return.
    """

    alpha: float = 0.85
    max_iter: int = 100
    tol: float = 1.0e-6
    top_n: int = 10
