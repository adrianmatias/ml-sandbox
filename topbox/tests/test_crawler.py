from pathlib import Path

import pytest

from topbox.crawler import Match, parse_profile_html


@pytest.fixture(scope="session")
def usyk_html() -> str:
    path = Path("data/oleksandr_usyk_profile.html")
    if not path.exists():
        pytest.skip("Usyk HTML fixture not found – run crawler once first")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def fury_html() -> str:
    path = Path("data/tyson_fury_profile.html")
    if not path.exists():
        pytest.skip("Fury HTML fixture not found")
    return path.read_text(encoding="utf-8")


class TestParseProfileHtmlReal:
    def test_usyk_real_profile(self, usyk_html: str) -> None:
        matches = parse_profile_html(usyk_html, "Oleksandr Usyk")
        assert len(matches) >= 15, f"Expected ~20+ fights, got {len(matches)}"
        assert all(m.boxer_a == "Oleksandr Usyk" for m in matches)
        assert any("Fury" in m.boxer_b for m in matches)  # real fight exists
        assert isinstance(matches[0], Match)

    def test_fury_real_profile(self, fury_html: str) -> None:
        matches = parse_profile_html(fury_html, "Tyson Fury")
        assert len(matches) >= 15
        assert any("Usyk" in m.boxer_b for m in matches)
