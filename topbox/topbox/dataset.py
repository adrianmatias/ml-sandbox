from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from topbox.domain import Match

LOGGER = logging.getLogger(__name__)


class Dataset:
    def __init__(self, save_path: str, min_date: str):
        self.min_date = min_date
        self.save_path = save_path
        self.df: pd.DataFrame = pd.DataFrame()

        LOGGER.info(f"{self.__dict__}")

    def create_from_matches(self, matches: list[Match]) -> None:
        """Build DataFrame from matches and save."""
        if not matches:
            LOGGER.warning("No matches – empty dataset created")
            self.df = pd.DataFrame(columns=["boxer_a", "boxer_b", "is_a_win", "date"])
            self.save()
            return

        LOGGER.info(f"Creating dataset from {len(matches)} matches")
        self.df = pd.DataFrame([vars(m) for m in matches])
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")

        if self.min_date:
            min_dt = pd.to_datetime(self.min_date)
            self.df = self.df[self.df["date"] >= min_dt]

        self.save()

    def save(self) -> None:
        """Persist to parquet."""
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(self.save_path)
        LOGGER.info(f"Saved dataset: {self.save_path} ({len(self.df):,} rows)")

    def load(self) -> bool:
        """Load from parquet if exists. Returns True on success."""
        path = Path(self.save_path)
        if path.exists():
            self.df = pd.read_parquet(path)
            LOGGER.info(f"Loaded existing dataset: {path} ({len(self.df):,} rows)")
            return True
        return False
