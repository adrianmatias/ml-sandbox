from __future__ import annotations

import pytest

from topbox.crawler import parse_profile_html


@pytest.fixture
def sample_html():
    return """
    <table class="table1">
        <tr><th>Date</th><th>Opponent</th><th>Res.</th><th>Method</th></tr>
        <tr><td>2024-02-10</td><td>Foe X</td><td class="bpro-w">W</td><td>TKO</td></tr>
        <tr><td>2023-12-05</td><td>Foe Y</td><td class="bpro-l">L</td><td>UD</td></tr>
    </table>
    """


class TestParseProfileHtml:
    def test_no_table(self) -> None:
        matches = parse_profile_html("<div>no table</div>", "Me")
        assert matches == []

    def test_sample(self, sample_html: str) -> None:
        matches = parse_profile_html(sample_html, "Me")
        assert len(matches) == 2
        assert matches[0].boxer_a == "Me"
        assert matches[0].boxer_b == "Foe X"
        assert matches[0].is_a_win is True
        assert matches[0].date == "2024-02-10"
        assert matches[1].is_a_win is False
