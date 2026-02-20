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


# Curated list of high-value boxers (edit this freely — names only!)
# The crawler will automatically find the correct box.live URL for each.
CURATED_BOXERS = [
    "Gennady Golovkin",
    "George Groves",
    "Muhammad Ali",
    "Mike Tyson",
    "Sugar Ray Robinson",
    "Manny Pacquiao",
    "Joe Louis",
    "George Foreman",
    "Marvin Hagler",
    "Sugar Ray Leonard",
    "Roberto Duran",
    "Floyd Mayweather Jr",
    "Lennox Lewis",
    "Evander Holyfield",
    "Oscar De La Hoya",
    "Rocky Marciano",
    "Henry Armstrong",
    "Jack Dempsey",
    "Harry Greb",
    "Carlos Monzon",
    "Roy Jones Jr",
    "Bernard Hopkins",
    "Canelo Alvarez",  # will resolve to saul-canelo-alvarez
    "Oleksandr Usyk",
    "Tyson Fury",
    "Anthony Joshua",
    "Terence Crawford",
    "Errol Spence Jr",
    "Daniel Dubois",
    "Jake Paul",
    "Vasyl Lomachenko",
    "Naoya Inoue",
    "Devin Haney",
    "Gervonta Davis",
    "Shakur Stevenson",
    "Teofimo Lopez",
    "Dmitry Bivol",
    "Artur Beterbiev",
    "Janibek Alimkhanuly",  # will resolve to zhanibek-alimkhanuly
    "Jermell Charlo",
    "Jermall Charlo",
    "David Benavidez",
    "Caleb Plant",
    "Jesse Rodriguez",
    "Leo Santa Cruz",
    "Marco Antonio Barrera",
    "Juan Manuel Marquez",
    "Guillermo Rigondeaux",
    "Nonito Donaire",
    "Wladimir Klitschko",
    "Vitali Klitschko",
    "Joe Frazier",
    "Sonny Liston",
    "Larry Holmes",
    "Thomas Hearns",
    "Julio Cesar Chavez",
    "Pernell Whitaker",
    "Shane Mosley",
    "Erik Morales",
    "Azumah Nelson",
    "Alexis Arguello",
    "Wilfred Benitez",
    "Aaron Pryor",
    "Ricardo Lopez",
    "Eusebio Pedroza",
    "Kosei Tanaka",
    "Roman Gonzalez",
    "Juan Francisco Estrada",
    "Kazuto Ioka",
    "Srisaket Sor Rungvisai",  # will resolve to the long correct slug
    "Andy Ruiz Jr",
    "Deontay Wilder",
    "Luis Ortiz",
    "Joe Joyce",
    "Filip Hrgovic",
    "Conor Benn",
    "Chris Eubank Jr",
    "Liam Smith",
    "Kell Brook",
    "Amir Khan",
    "Carl Frampton",
    "Josh Warrington",
]


def get_top_boxers(conf: ConfCrawler) -> list[tuple[str, str]]:
    """Scrape the boxer directory once and return correct URLs for our curated names."""
    directory_url = "https://box.live/boxers/"
    LOGGER.info(f"Fetching boxer directory from {directory_url} to get correct URLs...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=conf.user_agent)
        page.goto(directory_url, wait_until="networkidle")
        html = page.content()
        browser.close()

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Build lookup table: normalized name → correct full URL
    name_to_url = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "/boxers/" in href and href.endswith("/"):
            full_url = (
                f"https://box.live{href}" if not href.startswith("http") else href
            )
            name = a.get_text(strip=True).split("(")[0].strip()
            if name and len(name) > 2:
                # Multiple lookup keys for robustness
                key1 = name.lower()
                key2 = (
                    name.lower()
                    .replace(" ", "-")
                    .replace("'", "")
                    .replace(".", "")
                    .replace("jr", "jr")
                )
                name_to_url[key1] = full_url
                name_to_url[key2] = full_url

    # Match our curated names to real URLs
    boxers = []
    for display_name in CURATED_BOXERS:
        key = display_name.lower()
        url = name_to_url.get(key) or name_to_url.get(key.replace(" ", "-"))

        if url:
            boxers.append((display_name, url))
        else:
            LOGGER.warning(
                f"Could not find URL for '{display_name}' on box.live — skipping"
            )

    LOGGER.info(
        f"✅ Mapped {len(boxers)}/{len(CURATED_BOXERS)} boxers with correct URLs."
    )
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
