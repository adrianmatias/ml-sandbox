#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

from topbox.conf import ConfCrawlerWiki, ConfDataset, ConfPagerank
from topbox.crawler_wiki import get_matches
from topbox.dataset import Dataset
from topbox.pagerank import compute_ranks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    Path("data").mkdir(exist_ok=True)

    conf_c = ConfCrawlerWiki()
    conf_d = ConfDataset()
    conf_p = ConfPagerank(top_n=5000)

    ds = Dataset(conf_d)

    if not ds.load():
        matches = get_matches(conf_c)
        ds.create_from_matches(matches)

    ranks_df = compute_ranks(ds.df, conf_p, mode="loser_to_winner")
    ranks_df.to_csv("topbox.csv", index=False)
    logging.info(f"topbox.csv written with {len(ranks_df):,} ranked rows")


if __name__ == "__main__":
    main()
