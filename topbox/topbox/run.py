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

    recent_df = PageRankBox(top_n=5000, is_consolidated=False).compute(ds.df)
    recent_filename = CONST.loc.data / "topbox_recent.csv"
    logging.info(f"{recent_filename=}")
    recent_df.to_csv(recent_filename, index=False)
    recent_len = len(recent_df)
    logging.info(f"Recent ranks: {recent_len=}")

    cons_df = PageRankBox(top_n=5000, is_consolidated=True).compute(ds.df)
    cons_filename = CONST.loc.data / "topbox_consolidated.csv"
    logging.info(f"{cons_filename=}")
    cons_df.to_csv(cons_filename, index=False)
    cons_len = len(cons_df)
    logging.info(f"Consolidated ranks: {cons_len=}")


if __name__ == "__main__":
    main()
