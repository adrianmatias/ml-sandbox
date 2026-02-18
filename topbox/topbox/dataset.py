from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from topbox.conf import ConfDataset
from topbox.crawler import Match

LOGGER = logging.getLogger(__name__)


def create_dataset(matches: list[Match], conf: ConfDataset) -> pd.DataFrame:
    """Create dataset from matches.

    Args:
        matches: List of Match objects.
        conf: Dataset config.

    Returns:
        Filtered DataFrame saved as parquet.
    """
    if len(matches) == 0:
        LOGGER.warning("No matches, returning empty df")
        return pd.DataFrame(columns=["boxer_a", "boxer_b", "is_a_win", "date"])

    LOGGER.info(f"Creating dataset from {len(matches)} matches")
    df = pd.DataFrame([vars(m) for m in matches])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if conf.min_date:
        min_dt = pd.to_datetime(conf.min_date)
        df = df[df["date"] >= min_dt]
    Path(conf.save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(conf.save_path, index=False)
    LOGGER.info(f"Saved dataset: {conf.save_path} ({df.shape[0]} rows)")
    return df
