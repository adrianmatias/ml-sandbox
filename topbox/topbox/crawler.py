from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import sync_playwright

from topbox.conf import ConfCrawler

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Match:
    boxer_a: str
    boxer_b: str
    is_a_win: bool
    date: str


# Top ~25 active boxers across divisions (easy to expand)
TOP_BOXERS = [
    ("Oleksandr Usyk", "https://box.live/boxers/oleksandr-usyk/"),
    ("Tyson Fury", "https://box.live/boxers/tyson-fury/"),
    ("Anthony Joshua", "https://box.live/boxers/anthony-joshua/"),
    ("Daniel Dubois", "https://box.live/boxers/daniel-dubois/"),
    ("Deontay Wilder", "https://box.live/boxers/deontay-wilder/"),
    ("Canelo Alvarez", "https://box.live/boxers/canelo-alvarez/"),
    ("Dmitry Bivol", "https://box.live/boxers/dmitry-bivol/"),
    ("Artur Beterbiev", "https://box.live/boxers/artur-beterbiev/"),
    ("Terence Crawford", "https://box.live/boxers/terence-crawford/"),
    ("Errol Spence Jr", "https://box.live/boxers/errol-spence-jr/"),
    ("Jermell Charlo", "https://box.live/boxers/jermell-charlo/"),
    ("Gennady Golovkin", "https://box.live/boxers/gennady-golovkin/"),
    ("Jake Paul", "https://box.live/boxers/jake-paul/"),
    # Add more as you like — format: (name, profile_url)
]


def get_top_boxers(conf: ConfCrawler) -> list[tuple[str, str]]:
    """Return list of (name, profile_url) — currently hardcoded top boxers."""
    return TOP_BOXERS[: conf.max_pages]


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
