#!/usr/bin/env python3
from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

from topbox.conf import ConfDataset, ConfPagerank, ConfWikiCrawler
from topbox.crawler_wiki import get_matches
from topbox.dataset import create_dataset
from topbox.pagerank import compute_ranks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    Path("data").mkdir(exist_ok=True)

    conf_c = ConfWikiCrawler()
    conf_d = ConfDataset()
    conf_p = ConfPagerank(top_n=5000)

    if Path(conf_d.save_path).exists():
        df = pd.read_csv(conf_d.save_path)
        logging.info(f"Loaded existing dataset: {conf_d.save_path}")
    else:
        matches = get_matches(conf_c)
        df = create_dataset(matches, conf_d)
    ranks = compute_ranks(df, conf_p, mode="loser_to_winner")

    file_out = "topbox.csv"
    with open(file_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Boxer", "Score"])
        for rank, (boxer, score) in enumerate(ranks, 1):
            writer.writerow([rank, boxer, f"{score:.4f}"])
    logging.info(f"{file_out}")


if __name__ == "__main__":
    main()
