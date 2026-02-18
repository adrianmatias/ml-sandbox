#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

from topbox.conf import ConfCrawler, ConfDataset, ConfPagerank
from topbox.crawler import get_matches
from topbox.dataset import create_dataset
from topbox.pagerank import compute_ranks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    Path("data").mkdir(exist_ok=True)

    boxer_count = 250
    conf_c = ConfCrawler(max_pages=boxer_count)
    conf_d = ConfDataset()
    conf_p = ConfPagerank(top_n=boxer_count)

    matches = get_matches(conf_c)
    df = create_dataset(matches, conf_d)
    ranks = compute_ranks(df, conf_p)

    print("Top boxers by PageRank:")
    for boxer, score in ranks:
        print(f"{boxer}: {score:.4f}")


if __name__ == "__main__":
    main()
