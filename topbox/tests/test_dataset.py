from __future__ import annotations

from pathlib import Path

import pytest

from topbox.conf import ConfDataset
from topbox.dataset import Match, create_dataset


@pytest.fixture
def sample_matches() -> list[Match]:
    return [
        Match("A", "B", True, "2024-01-01"),
        Match("B", "C", False, "2023-12-01"),
        Match("A", "D", True, "2024-02-01"),
    ]


class TestCreateDataset:
    def test_basic(self, tmp_path: Path, sample_matches: list[Match]) -> None:
        save_path = tmp_path / "test.parquet"
        conf = ConfDataset(save_path=str(save_path))
        df = create_dataset(sample_matches, conf)
        assert len(df) == 3
        assert "boxer_a" in df.columns
        assert Path(save_path).exists()

    def test_filter_date(self, tmp_path: Path, sample_matches: list[Match]) -> None:
        save_path = tmp_path / "test.parquet"
        conf = ConfDataset(save_path=str(save_path), min_date="2024-01-01")
        df = create_dataset(sample_matches, conf)
        assert len(df) == 2

    def test_raw_rows_preserved(self, tmp_path: Path) -> None:
        # Mirror rows must be kept as-is; dedup is not the dataset's concern
        matches = [
            Match("Ali", "Frazier", True, "1971-03-08"),
            Match("Frazier", "Ali", False, "1971-03-08"),
        ]
        conf = ConfDataset(save_path=str(tmp_path / "t.parquet"))
        df = create_dataset(matches, conf)
        assert len(df) == 2

    def test_empty_matches(self, tmp_path: Path) -> None:
        conf = ConfDataset(save_path=str(tmp_path / "t.parquet"))
        df = create_dataset([], conf)
        assert len(df) == 0
        assert list(df.columns) == ["boxer_a", "boxer_b", "is_a_win", "date"]
