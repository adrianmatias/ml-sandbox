from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from topbox.conf import ConfWikiCrawler
from topbox.crawler_wiki import CrawlerWiki


@pytest.fixture
def sample_html():
    html = (
        '<html><body><table class="wikitable"><caption>Professional boxing record</caption>'
        "<tr><th>No.</th><th>Result</th><th>Record</th><th>Opponent</th><th>Type</th><th>Round, time</th><th>Date</th><th>Location</th><th>Notes</th></tr>"
        "<tr><td>1</td><td>Win</td><td>1–0</td><td>Opponent A</td><td>KO</td><td>1 (10), 2:30</td><td>January 1, 2020</td><td>Venue</td><td></td></tr>"
        "<tr><td>2</td><td>Loss</td><td>1–1</td><td>Opponent B</td><td>UD</td><td>10</td><td>2020-02-02</td><td>Venue</td><td></td></tr>"
        "</table></body></html>"
    )
    return html


@pytest.fixture
def crawler():
    conf = ConfWikiCrawler()
    return CrawlerWiki(conf)


def test_find_record_table(crawler, sample_html):
    soup = BeautifulSoup(sample_html, "html.parser")
    table = crawler.find_record_table(soup)
    assert table is not None
    assert table.find("caption").get_text().strip() == "Professional boxing record"


def test_parse_table(crawler, sample_html):
    soup = BeautifulSoup(sample_html, "html.parser")
    table = crawler.find_record_table(soup)
    matches = crawler.parse_table(table, "Test Boxer")
    assert len(matches) == 2
    assert matches[0].boxer_a == "Test Boxer"
    assert matches[0].boxer_b == "Opponent A"
    assert matches[0].is_a_win is True
    assert matches[0].date == "2020-01-01"
    assert matches[1].is_a_win is False
    assert matches[1].date == "2020-02-02"


def test_parse_date(crawler):
    assert crawler.parse_date("2020-01-01") == "2020-01-01"
    assert crawler.parse_date("January 1, 2020") == "2020-01-01"
    assert crawler.parse_date("Unknown") == "Unknown"
