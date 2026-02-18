from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from topbox.conf import ConfCrawler

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Match:
    boxer_a: str
    boxer_b: str
    is_a_win: bool
    date: str


def parse_profile_html(html: str, boxer_name: str) -> list[Match]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="table1")
    if not table:
        LOGGER.warning("No fight table found")
        return []

    matches = []
    rows = table.find_all("tr")[1:]
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 3:
            continue
        date = cols[0].text.strip()
        opp = cols[1].text.strip()
        res = cols[2].text.strip()
        is_win = res.upper().startswith("W")
        matches.append(Match(boxer_name, opp, is_win, date))
    return matches


def get_matches(conf: ConfCrawler) -> list[Match]:
    profiles = [
        ("Oleksandr Usyk", "https://boxrec.com/en/proboxer/430139"),
        ("Tyson Fury", "https://boxrec.com/en/proboxer/159406"),
    ]
    matches = []
    for name, url in profiles[: conf.max_pages]:
        try:
            resp = requests.get(
                url, headers={"User-Agent": conf.user_agent}, timeout=10
            )
            resp.raise_for_status()
            fights = parse_profile_html(resp.text, name)
            matches.extend(fights[-20:])
            LOGGER.info(f"Got {len(fights)} fights for {name}")
        except Exception as e:
            LOGGER.error(f"Failed {url}: {e}")
    return matches
