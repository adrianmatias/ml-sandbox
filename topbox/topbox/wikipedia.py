"""Fetch world boxing champion names from Wikipedia champion list pages."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import requests

from topbox.conf import ConfWikipedia

LOGGER = logging.getLogger(__name__)


@dataclass
class WikiFilter:
    skip_patterns: re.Pattern[str] = field(
        default_factory=lambda: re.compile(
            r"""
            \bvacant\b
            | \bRetired\b
            | \bStripped\b
            | ^WBA$|^WBC$|^IBF$|^WBO$|^IBO$|^WBF$|^NABF$
            | Championship | Organization | Association | Council | Federation
            | \bBoxing\b
            | \bTitle\b | \bBelt\b | \bDivision\b | \bWeight\b | \bClass\b
            | \bList\b
            | \bChampion\b
            | \bRound\b | \bKnockout\b | \bCount\b | \bContest\b
            | vs\.
            | Olympics | Games
            | \bmagazine\b | \bNewspaper\b | \bTimes\b | \bHerald\b | \bShimbun\b
            """,
            re.VERBOSE | re.IGNORECASE,
        )
    )
    paren_re: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"\s*\([^)]+\)\s*$")
    )
    wikilink_re: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"\[\[([^\]|#]+)(?:\|([^\]]+))?\]\]")
    )
    lowercase_start: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"^[a-z]")
    )

    def process_wikilink(self, m: re.Match[str]) -> str | None:
        target = m.group(1).strip()
        display = m.group(2)
        if self.skip_patterns.search(target):
            return None
        raw = (display or target).strip()
        name = self.paren_re.sub("", raw).strip()
        if (
            not name
            or self.skip_patterns.search(name)
            or len(name) < 4
            or name.isupper()
            or self.lowercase_start.match(name)
            or re.fullmatch(r"\d{4}|\d{1,2}\s+\w+\s+\d{4}", name)
        ):
            return None
        return name


@dataclass
class WikipediaFetcher:
    conf: ConfWikipedia
    wiki_filter: WikiFilter = field(default_factory=WikiFilter)

    def fetch_wikitext(self, page: str) -> str:
        params = {
            "action": "parse",
            "page": page,
            "prop": "wikitext",
            "format": "json",
            "formatversion": "2",
        }
        try:
            resp = requests.get(
                self.conf.api_url,
                params=params,
                headers={"User-Agent": self.conf.user_agent},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                LOGGER.warning(f"Wikipedia API error for '{page}': {data['error']}")
                return ""
            return data["parse"]["wikitext"]
        except requests.RequestException as e:
            LOGGER.error(f"Request failed for Wikipedia page '{page}': {e}")
            return ""

    def extract_names_from_wikitext(self, wikitext: str) -> list[str]:
        seen: set[str] = set()
        name_list: list[str] = []
        for m in self.wiki_filter.wikilink_re.finditer(wikitext):
            name = self.wiki_filter.process_wikilink(m)
            if name and name not in seen:
                seen.add(name)
                name_list.append(name)
        return name_list

    def get_boxer_names(self) -> list[str]:
        seen: set[str] = set()
        all_name_list: list[str] = []

        for page in self.conf.pages:
            LOGGER.info(f"Fetching Wikipedia page: {page}")
            wikitext = self.fetch_wikitext(page)
            if not wikitext:
                continue
            name_list = self.extract_names_from_wikitext(wikitext)
            added = 0
            for name in name_list:
                if name not in seen:
                    seen.add(name)
                    all_name_list.append(name)
                    added += 1
            LOGGER.info(
                f"  {page}: {added} new names (total so far: {len(all_name_list)})"
            )

        for name in self.conf.extra_names:
            if name not in seen:
                seen.add(name)
                all_name_list.append(name)

        result = sorted(all_name_list)
        LOGGER.info(f"Wikipedia boxer list complete: {len(result)} unique names")
        return result


def get_boxer_names(conf: ConfWikipedia | None = None) -> list[str]:
    if conf is None:
        conf = ConfWikipedia()
    fetcher = WikipediaFetcher(conf)
    return fetcher.get_boxer_names()
