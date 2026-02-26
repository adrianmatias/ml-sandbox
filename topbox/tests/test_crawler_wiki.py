from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from topbox.crawler_wiki import CrawlerWiki


@pytest.fixture
def crawler():
    return CrawlerWiki()


def test_get_fighters(crawler):
    fighters = crawler.get_fighter_seed()
    assert isinstance(fighters, dict)
    assert len(fighters) > 50  # has many
    assert "Muhammad Ali" in fighters


@patch("requests.get")
def test_extract_matches_success(mock_get, crawler):
    # Mock response with HTML containing a table
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    html = (  # noqa: E501
        '<html><body><table class="wikitable">'
        "<tr><th>No.</th><th>Result</th><th>Record</th><th>Opponent</th>"
        "<th>Type</th><th>Round, time</th><th>Date</th><th>Location</th><th>Notes</th></tr>"  # noqa: E501
        "<tr><td>1</td><td>Win</td><td>1-0</td><td>Opponent A</td><td>KO</td>"
        "<td>1 (10), 2:30</td><td>2020-01-01</td><td>Venue</td><td></td></tr>"
        "<tr><td>2</td><td>Loss</td><td>1-1</td><td>Opponent B</td><td>UD</td>"
        "<td>10</td><td>2020-02-02</td><td>Venue</td><td></td></tr>"
        "<tr><td>3</td><td>Win</td><td>2-1</td><td>Opponent C</td><td>SD</td>"
        "<td>12</td><td>2020-03-03</td><td>Venue</td><td></td></tr>"
        "<tr><td>4</td><td>Draw</td><td>2-1-1</td><td>Opponent D</td><td>Draw</td>"
        "<td>10</td><td>2020-04-04</td><td>Venue</td><td></td></tr>"
        "<tr><td>5</td><td>Win</td><td>3-1-1</td><td>Opponent E</td><td>TKO</td>"
        "<td>8 (12)</td><td>2020-05-05</td><td>Venue</td><td></td></tr>"
        "<tr><td>6</td><td>Loss</td><td>3-2-1</td><td>Opponent F</td><td>KO</td>"
        "<td>5 (10)</td><td>2020-06-06</td><td>Venue</td><td></td></tr>"
        "</table></body></html>"
    )
    mock_response.text = html
    mock_get.return_value = mock_response

    matches = crawler.extract_matches("Test Boxer", "http://example.com")
    assert len(matches) == 6
    assert matches[0].boxer_a == "Test Boxer"
    assert matches[0].is_a_win is True

    expected_wins = [True, False, True, None, True, False]
    assert [m.is_a_win for m in matches] == expected_wins


@patch("requests.get")
def test_extract_matches_no_table(mock_get, crawler):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.text = "<html><body>No table</body></html>"
    mock_get.return_value = mock_response

    matches = crawler.extract_matches("Test Boxer", "http://example.com")
    assert len(matches) == 0
