from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from topbox.conf import ConfCrawlerWiki
from topbox.domain import Match

LOGGER = logging.getLogger(__name__)


@dataclass
class CrawlerWiki:
    """Minimal crawler for Wikipedia boxer pages using pandas."""

    conf: ConfCrawlerWiki

    def get_fighters(self) -> dict[str, str]:
        """Get dictionary of boxer names to Wikipedia URLs from JSON file."""
        fighters_path = Path(__file__).parent / "fighters.json"
        with fighters_path.open() as f:
            return json.load(f)

    def parse_fight_table(self, tables: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """Find and return the fight table from list of tables."""
        return [t for t in tables if "Date" in t.columns and len(t) > 5]

    def extract_matches(self, name: str, url: str) -> list[Match]:
        """Extract matches from a boxer's Wikipedia page."""
        matches = []
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": self.conf.user_agent},
                timeout=self.conf.timeout,
            )
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            fight_tables = self.parse_fight_table(tables)
            if not fight_tables:
                LOGGER.warning(f"No fight table found for {name}")
                return matches
            for table in fight_tables:
                for _, row in table.iterrows():
                    date_str = str(row.get("Date", "")).strip()
                    if not date_str or "nan" in date_str.lower():
                        continue
                    try:
                        dt = pd.to_datetime(date_str, errors="coerce")
                        if pd.isna(dt) or dt.year < self.conf.min_year:
                            continue
                        date = dt.strftime("%Y-%m-%d")
                    except Exception:
                        continue
                    opp = str(row.get("Opponent", "")).strip()
                    if not opp or "nan" in opp.lower():
                        continue
                    res = str(row.get("Result", "")).lower()
                    is_win = "win" in res or any(
                        x in res for x in ["ko", "ud", "sd", "tdko"]
                    )
                    matches.append(Match(name, opp, is_win, date))
        except requests.RequestException as e:
            LOGGER.error(f"Failed to fetch {url}: {e}")
        except Exception as e:
            LOGGER.error(f"Error processing {name}: {e}")
        return matches

    def crawl_all(self) -> list[Match]:
        """Crawl all fighters and return list of matches."""
        fighters = self.get_fighters()
        matches: list[Match] = []
        for name, url in fighters.items():
            LOGGER.info(f"Fetching {name}...")
            boxer_matches = self.extract_matches(name, url)
            matches.extend(boxer_matches)
            time.sleep(self.conf.delay)
        LOGGER.info(f"Crawled {len(fighters)} boxers, total matches: {len(matches)}")
        return matches


def get_matches(conf: ConfCrawlerWiki) -> list[Match]:
    """Get matches from crawler.

    Args:
        conf: Crawler configuration.

    Returns:
        List of Match objects.
    """
    crawler = CrawlerWiki(conf)
    return crawler.crawl_all()
