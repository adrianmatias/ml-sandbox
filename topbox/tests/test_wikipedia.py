"""Tests for the Wikipedia boxer-name fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from topbox.conf import ConfWikipedia
from topbox.wikipedia import WikipediaFetcher, get_boxer_names

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_WIKITEXT = """\
==Heavyweight==
| [[Muhammad Ali]]<br>56–5 (37 KO)
| [[Joe Frazier]]<br>32–4–1 (27 KO)
| [[George Foreman]]<br>76–5 (68 KO)
==Light Heavyweight==
| [[Roy Jones Jr.]]<br>66–9 (47 KO)
| [[Bernard Hopkins]]<br>55–8–2 (32 KO)
| [[Josh Kelly (boxer)|Josh Kelly]]<br>12–0 (7 KO)
==Skip these==
| [[WBA]]
| [[IBF]]
| vacant
| [[Weight class (boxing)|Heavyweight]]
| 2023
| [[Boxing]]
"""

SAMPLE_WIKITEXT_EXTRA = """\
| [[Canelo Álvarez]]<br>60–2–2 (39 KO)
| [[Naoya Inoue]]<br>27–0 (24 KO)
| [[Oleksandr Usyk]]<br>24–0 (15 KO)
"""


# ---------------------------------------------------------------------------
# WikipediaFetcher.extract_names_from_wikitext
# ---------------------------------------------------------------------------


class TestExtractNamesFromWikitext:
    def test_extracts_known_boxers(self) -> None:
        fetcher = WikipediaFetcher(ConfWikipedia())
        names = fetcher.extract_names_from_wikitext(SAMPLE_WIKITEXT)
        assert "Muhammad Ali" in names
        assert "Joe Frazier" in names
        assert "George Foreman" in names
        assert "Roy Jones Jr." in names
        assert "Bernard Hopkins" in names

    def test_strips_disambiguation(self) -> None:
        fetcher = WikipediaFetcher(ConfWikipedia())
        names = fetcher.extract_names_from_wikitext(SAMPLE_WIKITEXT)
        # "Josh Kelly (boxer)" should become "Josh Kelly"
        assert "Josh Kelly" in names
        assert "Josh Kelly (boxer)" not in names

    def test_skips_organisations(self) -> None:
        fetcher = WikipediaFetcher(ConfWikipedia())
        names = fetcher.extract_names_from_wikitext(SAMPLE_WIKITEXT)
        assert "WBA" not in names
        assert "IBF" not in names

    def test_skips_vacant(self) -> None:
        fetcher = WikipediaFetcher(ConfWikipedia())
        names = fetcher.extract_names_from_wikitext(SAMPLE_WIKITEXT)
        assert "vacant" not in names

    def test_skips_weight_class_articles(self) -> None:
        fetcher = WikipediaFetcher(ConfWikipedia())
        names = fetcher.extract_names_from_wikitext(SAMPLE_WIKITEXT)
        assert "Heavyweight" not in names

    def test_skips_year_strings(self) -> None:
        fetcher = WikipediaFetcher(ConfWikipedia())
        names = fetcher.extract_names_from_wikitext(SAMPLE_WIKITEXT)
        assert "2023" not in names

    def test_deduplication(self) -> None:
        fetcher = WikipediaFetcher(ConfWikipedia())
        repeated = SAMPLE_WIKITEXT + SAMPLE_WIKITEXT
        names = fetcher.extract_names_from_wikitext(repeated)
        assert names.count("Muhammad Ali") == 1

    def test_empty_wikitext(self) -> None:
        fetcher = WikipediaFetcher(ConfWikipedia())
        assert fetcher.extract_names_from_wikitext("") == []

    def test_accented_names(self) -> None:
        fetcher = WikipediaFetcher(ConfWikipedia())
        names = fetcher.extract_names_from_wikitext(SAMPLE_WIKITEXT_EXTRA)
        assert "Canelo Álvarez" in names
        assert "Naoya Inoue" in names
        assert "Oleksandr Usyk" in names


# ---------------------------------------------------------------------------
# WikipediaFetcher.fetch_wikitext
# ---------------------------------------------------------------------------


class TestFetchWikitext:
    def test_returns_wikitext_on_success(self) -> None:
        fetcher = WikipediaFetcher(ConfWikipedia())
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "parse": {"wikitext": "==Heavyweight==\n| [[Muhammad Ali]]"}
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("topbox.wikipedia.requests.get", return_value=mock_resp):
            result = fetcher.fetch_wikitext("List_of_WBA_world_champions")

        assert "Muhammad Ali" in result

    def test_returns_empty_on_api_error(self) -> None:
        fetcher = WikipediaFetcher(ConfWikipedia())
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "error": {"code": "missingtitle", "info": "Page not found"}
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("topbox.wikipedia.requests.get", return_value=mock_resp):
            result = fetcher.fetch_wikitext("Nonexistent_Page")

        assert result == ""

    def test_returns_empty_on_network_error(self) -> None:
        import requests as req

        fetcher = WikipediaFetcher(ConfWikipedia())
        with patch(
            "topbox.wikipedia.requests.get", side_effect=req.RequestException("timeout")
        ):
            result = fetcher.fetch_wikitext("List_of_WBA_world_champions")

        assert result == ""


# ---------------------------------------------------------------------------
# get_boxer_names
# ---------------------------------------------------------------------------


class TestGetBoxerNames:
    def _make_mock_get(self, wikitext: str) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"parse": {"wikitext": wikitext}}
        mock_resp.raise_for_status = MagicMock()
        return MagicMock(return_value=mock_resp)

    def test_returns_sorted_deduplicated_names(self) -> None:
        conf = ConfWikipedia(pages=("PageA", "PageB"))
        wikitext_a = "| [[Muhammad Ali]]\n| [[Joe Frazier]]"
        wikitext_b = "| [[Joe Frazier]]\n| [[George Foreman]]"

        responses = [
            MagicMock(
                **{
                    "json.return_value": {"parse": {"wikitext": wikitext_a}},
                    "raise_for_status": MagicMock(),
                }
            ),
            MagicMock(
                **{
                    "json.return_value": {"parse": {"wikitext": wikitext_b}},
                    "raise_for_status": MagicMock(),
                }
            ),
        ]

        with patch("topbox.wikipedia.requests.get", side_effect=responses):
            names = get_boxer_names(conf)

        assert "Muhammad Ali" in names
        assert "Joe Frazier" in names
        assert "George Foreman" in names
        assert names.count("Joe Frazier") == 1
        assert names == sorted(names)

    def test_extra_names_appended(self) -> None:
        conf = ConfWikipedia(pages=("PageA",), extra_names=("Jake Paul",))
        mock_get = self._make_mock_get("| [[Muhammad Ali]]")

        with patch("topbox.wikipedia.requests.get", mock_get):
            names = get_boxer_names(conf)

        assert "Jake Paul" in names

    def test_extra_names_not_duplicated(self) -> None:
        conf = ConfWikipedia(pages=("PageA",), extra_names=("Muhammad Ali",))
        mock_get = self._make_mock_get("| [[Muhammad Ali]]")

        with patch("topbox.wikipedia.requests.get", mock_get):
            names = get_boxer_names(conf)

        assert names.count("Muhammad Ali") == 1

    def test_network_failure_skips_page(self) -> None:
        import requests as req

        conf = ConfWikipedia(pages=("BadPage",))
        with patch(
            "topbox.wikipedia.requests.get", side_effect=req.RequestException("err")
        ):
            names = get_boxer_names(conf)

        assert names == []

    def test_uses_default_conf_when_none(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"parse": {"wikitext": "| [[Muhammad Ali]]"}}
        mock_resp.raise_for_status = MagicMock()

        with patch("topbox.wikipedia.requests.get", return_value=mock_resp) as mock_get:
            names = get_boxer_names(None)

        # Should have made one call per default page (5 pages)
        assert mock_get.call_count == 5
        assert "Muhammad Ali" in names

    def test_minimum_coverage(self) -> None:
        """Live smoke test: real Wikipedia pages should yield many names."""
        pytest.importorskip("requests")
        names = get_boxer_names()
        assert len(names) >= 200, f"Expected 200+ boxers, got {len(names)}"
        # These boxers all held WBA/WBC/IBF/WBO belts and appear on the pages
        assert "Floyd Mayweather Jr." in names
        assert "Manny Pacquiao" in names
        assert "Evander Holyfield" in names
