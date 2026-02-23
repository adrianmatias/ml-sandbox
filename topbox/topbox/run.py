#!/usr/bin/env python3
from __future__ import annotations

import logging

from topbox.const import CONST
from topbox.crawler_wiki import get_matches
from topbox.dataset import Dataset
from topbox.page_rank_box import PageRankBox


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s,%(msecs)03d %(levelname)-8s "
            "[%(name)s.%(funcName)s:%(lineno)d] %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    CONST.loc.data.mkdir(exist_ok=True)

    filename_match = CONST.loc.data / "match.parquet"

    ds = Dataset(save_path=filename_match, min_date="1950-01-01")

    if not ds.load():
        matches = get_matches()
        ds.create_from_matches(matches)

    rank_df = PageRankBox(top_n=5000, mode="loser_to_winner").compute(ds.df)
    filename = CONST.loc.data / "topbox.csv"
    logging.info(f"{filename=}")
    rank_df.to_csv(filename, index=False)
    rank_df_len = len(rank_df)
    logging.info(f"{rank_df_len=}")


if __name__ == "__main__":
    main()
