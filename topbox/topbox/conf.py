"""Configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    save_path: str = "data/full_boxing_matches_1965_present_again.csv"


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
    max_iter: int = 1000
    tol: float = 1.0e-6
    top_n: int = 10


@dataclass(frozen=True)
class ConfWikipedia:
    """Wikipedia boxer-list configuration.

    Args:
        pages: Wikipedia page titles to scrape for boxer names.
        api_url: MediaWiki action API base URL.
        user_agent: User-Agent sent with every request.
        extra_names: Additional names appended after Wikipedia fetch.
    """

    pages: tuple[str, ...] = field(
        default_factory=lambda: (
            "List_of_WBA_world_champions",
            "List_of_WBC_world_champions",
            "List_of_IBF_world_champions",
            "List_of_WBO_world_champions",
            "List_of_current_world_boxing_champions",
        )
    )
    api_url: str = "https://en.wikipedia.org/w/api.php"
    user_agent: str = "topbox/1.0 (boxing pagerank project; github.com/topbox)"
    extra_names: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ConfWikiCrawler:
    """Wiki crawler configuration.

    Args:
        base_url: Base URL for Wikipedia pages.
        user_agent: User agent string.
        delay: Delay between requests in seconds.
    """

    base_url: str = "https://en.wikipedia.org/wiki/"
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"
    )
    delay: float = 0.5


@dataclass(frozen=True)
class ConfCrawlerMin:
    """Minimal crawler configuration.

    Args:
        user_agent: User agent string.
        timeout: Request timeout in seconds.
        delay: Delay between requests in seconds.
        min_year: Minimum year for fights.
    """

    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    timeout: int = 10
    delay: float = 0.8
    min_year: int = 1965
