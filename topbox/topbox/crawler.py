from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import sync_playwright

from topbox.conf import ConfCrawler, ConfWikipedia
from topbox.wikipedia import get_boxer_names

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Match:
    boxer_a: str
    boxer_b: str
    is_a_win: bool
    date: str


@dataclass
class BoxerCrawler:
    conf: ConfCrawler
    wiki_conf: ConfWikipedia | None = None

    def boxer_list(self) -> list[str]:
        return get_boxer_names(self.wiki_conf)

    def top_boxers(self) -> list[tuple[str, str]]:
        name_list = self.boxer_list()
        directory_url = "https://box.live/boxers/"
        LOGGER.info(f"Fetching {directory_url} to match {len(name_list)} names")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=self.conf.user_agent)
            page.goto(directory_url, wait_until="networkidle")
            html = page.content()
            browser.close()

        name_to_url = self.name_to_url(html)
        boxer_list = self.match_names_to_urls(name_list, name_to_url)

        LOGGER.info(f"Mapped {len(boxer_list)}/{len(name_list)} boxers")
        return boxer_list

    def name_to_url(self, html: str) -> dict[str, str]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        name_to_url = {}
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if "/boxers/" in href and href.endswith("/"):
                full_url = (
                    f"https://box.live{href}" if not href.startswith("http") else href
                )
                name = a.get_text(strip=True).split("(")[0].strip()
                if name and len(name) > 2:
                    self.add_keys(name, full_url, name_to_url)
        return name_to_url

    def add_keys(self, name: str, url: str, name_to_url: dict[str, str]) -> None:
        key1 = name.lower()
        key2 = (
            key1.replace(" ", "-").replace("'", "").replace(".", "").replace("jr", "jr")
        )
        name_to_url[key1] = url
        name_to_url[key2] = url

    def match_names_to_urls(
        self, name_list: list[str], name_to_url: dict[str, str]
    ) -> list[tuple[str, str]]:
        boxer_list = []
        for name in name_list:
            key = name.lower()
            url = name_to_url.get(key) or name_to_url.get(key.replace(" ", "-"))
            if url:
                boxer_list.append((name, url))
            else:
                LOGGER.warning(f"No URL for '{name}' on box.live — skipping")
        return boxer_list


def get_top_boxers(
    conf: ConfCrawler, wiki_conf: ConfWikipedia | None = None
) -> list[tuple[str, str]]:
    crawler = BoxerCrawler(conf, wiki_conf)
    return crawler.top_boxers()


def _build_name_to_url(html: str) -> dict[str, str]:
    """Extract name-to-URL mappings from box.live directory HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    name_to_url = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "/boxers/" in href and href.endswith("/"):
            full_url = (
                f"https://box.live{href}" if not href.startswith("http") else href
            )
            name = a.get_text(strip=True).split("(")[0].strip()
            if name and len(name) > 2:
                _add_keys(name, full_url, name_to_url)
    return name_to_url


def _add_keys(name: str, url: str, name_to_url: dict[str, str]) -> None:
    """Add normalized keys for name matching."""
    key1 = name.lower()
    key2 = key1.replace(" ", "-").replace("'", "").replace(".", "").replace("jr", "jr")
    name_to_url[key1] = url
    name_to_url[key2] = url


def _match_names_to_urls(
    names: list[str], name_to_url: dict[str, str]
) -> list[tuple[str, str]]:
    """Match display names to URLs, logging misses."""
    boxers = []
    for name in names:
        key = name.lower()
        url = name_to_url.get(key) or name_to_url.get(key.replace(" ", "-"))
        if url:
            boxers.append((name, url))
        else:
            LOGGER.warning(f"No URL for '{name}' on box.live — skipping")
    return boxers


def parse_boxer_name(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if h1:
        return h1.text.strip()
    return "Unknown Boxer"


def parse_profile_html(html: str, boxer_name: str) -> list[Match]:
    """Parse fight history from box.live profile HTML (2026 structure)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Find the exact fight history table by its caption
    caption = soup.find("caption", string=lambda t: t and "Recent Contests" in t)
    if not caption:
        LOGGER.warning(f"No 'Recent Contests' table found for {boxer_name}")
        return []

    table = caption.find_parent("table")
    LOGGER.info(f"Found fight table: {caption.get_text(strip=True)}")

    matches = []
    rows = table.find_all("tr")[1:]  # skip header

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 3:
            continue

        date = cols[0].get_text(strip=True)
        opp_cell = cols[1]
        result_text = cols[2].get_text(strip=True).lower()

        # Opponent name (sometimes linked)
        opp_link = opp_cell.find("a")
        opponent = (
            opp_link.get_text(strip=True) if opp_link else opp_cell.get_text(strip=True)
        )

        if not date or not opponent:
            continue

        is_win = result_text.startswith("won") or "won" in result_text

        matches.append(Match(boxer_name, opponent, is_win, date))

    LOGGER.info(f"Parsed {len(matches)} fights for {boxer_name}")
    return matches


def get_matches(conf: ConfCrawler) -> list[Match]:
    matches = []
    boxers = get_top_boxers(conf)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for name, url in boxers:
            try:
                page = browser.new_page(user_agent=conf.user_agent)
                page.goto(url, wait_until="networkidle")
                html = page.content()
                Path("data").mkdir(exist_ok=True)
                Path(f"data/{name.replace(' ', '_').lower()}_profile.html").write_text(
                    html, encoding="utf-8"
                )
                page.close()

                parsed_name = parse_boxer_name(html)
                fights = parse_profile_html(html, parsed_name)
                matches.extend(
                    fights[-30:]
                )  # last 30 fights per boxer (adjust as needed)

                LOGGER.info(f"Got {len(fights)} fights for {parsed_name}")
                time.sleep(1.2)  # polite delay
            except Exception as e:
                LOGGER.error(f"Failed {url}: {e}")
        browser.close()

    return matches
