from __future__ import annotations

from topbox.conf import ConfCrawler, ConfDataset, ConfPagerank


class TestConfCrawler:
    def test_defaults(self) -> None:
        conf = ConfCrawler()
        assert conf.base_url == "https://boxrec.com"
        assert conf.max_pages == 10
        assert conf.user_agent == "Mozilla/5.0 (compatible; topbox/1.0)"

    def test_override(self) -> None:
        conf = ConfCrawler(base_url="https://test.com", max_pages=5)
        assert conf.base_url == "https://test.com"
        assert conf.max_pages == 5


class TestConfDataset:
    def test_defaults(self) -> None:
        conf = ConfDataset()
        assert conf.min_date is None
        assert conf.save_path == "data/matches.parquet"

    def test_override(self) -> None:
        conf = ConfDataset(min_date="2020-01-01")
        assert conf.min_date == "2020-01-01"


class TestConfPagerank:
    def test_defaults(self) -> None:
        conf = ConfPagerank()
        assert conf.alpha == 0.85
        assert conf.top_n == 10
