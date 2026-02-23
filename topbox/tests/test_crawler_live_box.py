from pathlib import Path

import pytest


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
