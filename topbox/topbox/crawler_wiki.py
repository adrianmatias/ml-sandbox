from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from topbox.conf import ConfWikiCrawler, ConfWikipedia
from topbox.types import Match
from topbox.wikipedia import get_boxer_names

LOGGER = logging.getLogger(__name__)


@dataclass
class CrawlerWiki:
    """Crawler for Wikipedia boxer pages to extract fight records."""

    conf: ConfWikiCrawler
    wiki_conf: ConfWikipedia | None = None

    def boxer_list(self) -> list[str]:
        """Get list of boxer names from Wikipedia."""
        return get_boxer_names(self.wiki_conf)

    def parse_boxer_page(self, boxer_name: str) -> list[Match]:
        """Parse fight records from a boxer's Wikipedia page.

        Args:
            boxer_name: Name of the boxer.

        Returns:
            List of Match objects for the boxer's fights.
        """
        url = f"{self.conf.base_url}{quote(boxer_name.replace(' ', '_'))}"
        try:
            resp = requests.get(
                url, headers={"User-Agent": self.conf.user_agent}, timeout=30
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            LOGGER.error(f"Failed to fetch {url}: {e}")
            return []

        soup = BeautifulSoup(resp.content, "html.parser")
        table = self.find_record_table(soup)
        if not table:
            LOGGER.warning(f"No boxing record table found for {boxer_name}")
            return []

        return self.parse_table(table, boxer_name)

    def find_record_table(self, soup: BeautifulSoup):
        """Find the professional boxing record table in the page.

        Args:
            soup: Parsed HTML soup.

        Returns:
            The table element if found, else None.
        """
        for table in soup.find_all("table", class_="wikitable"):
            caption = table.find("caption")
            if caption and "professional boxing record" in caption.get_text().lower():
                return table
        return None

    def parse_table(self, table: BeautifulSoup, boxer_name: str) -> list[Match]:
        """Parse matches from the boxing record table.

        Args:
            table: The table element.
            boxer_name: Name of the boxer.

        Returns:
            List of Match objects.
        """
        matches = []
        rows = table.find_all("tr")[1:]  # skip header
        for row in rows:
            cols = row.find_all(["td", "th"])
            if len(cols) < 5:
                continue
            result = cols[1].get_text(strip=True).lower()
            opponent = cols[3].get_text(strip=True)
            date_str = cols[6].get_text(strip=True) if len(cols) > 6 else ""

            if not opponent or not result:
                continue

            is_win = result.startswith("win")
            date = self.parse_date(date_str)

            matches.append(Match(boxer_name, opponent, is_win, date))

        LOGGER.info(f"Parsed {len(matches)} matches for {boxer_name}")
        return matches

    def parse_date(self, date_str: str) -> str:
        """Parse date string into YYYY-MM-DD format.

        Args:
            date_str: Raw date string from table.

        Returns:
            Formatted date string.
        """
        match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", date_str)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        months = {
            "january": "01",
            "february": "02",
            "march": "03",
            "april": "04",
            "may": "05",
            "june": "06",
            "july": "07",
            "august": "08",
            "september": "09",
            "october": "10",
            "november": "11",
            "december": "12",
        }
        match = re.search(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", date_str, re.IGNORECASE)
        if match:
            month_str = match.group(1).lower()
            day = match.group(2).zfill(2)
            year = match.group(3)
            month = months.get(month_str)
            if month:
                return f"{year}-{month}-{day}"
        return date_str


def get_matches(conf: ConfWikiCrawler) -> list[Match]:
    """Get matches from Wikipedia for top boxers.

    Args:
        conf: Wiki crawler configuration.

    Returns:
        List of Match objects.
    """
    crawler = CrawlerWiki(conf)
    name_list = crawler.boxer_list()
    matches = []
    for name in name_list[:10]:  # limit for testing
        LOGGER.info(f"Fetching matches for {name}")
        boxer_matches = crawler.parse_boxer_page(name)
        matches.extend(boxer_matches)
        time.sleep(conf.delay)
    return matches
