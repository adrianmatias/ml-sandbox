#!/usr/bin/env python3
from __future__ import annotations

import logging

from topbox.conf_timeline import ConfTimeline
from topbox.const import CONST
from topbox.crawler_wiki import get_matches
from topbox.dataset import Dataset
from topbox.page_rank_timeline import PageRankTimeline
from topbox.trajectory_digest import TrajectoryDigest


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s,%(msecs)03d %(levelname)-8s "
            "[%(name)s.%(funcName)s:%(lineno)d] %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    filename_match = CONST.loc.data / "match.parquet"
    ds = Dataset(save_path=filename_match, min_date="1950-01-01")

    if not ds.load():
        matches = get_matches()
        ds.create_from_matches(matches)

    timeline = PageRankTimeline(conf=ConfTimeline()).compute(ds.df)
    timeline_filename = CONST.loc.data / "topbox_timeline.csv"
    timeline.to_csv(timeline_filename, index=False)
    logging.info(f"{timeline_filename=} rows={len(timeline)}")

    digest = TrajectoryDigest().compute(timeline)
    digest_filename = CONST.loc.data / "topbox_trajectory.csv"
    digest.to_csv(digest_filename, index=False)
    logging.info(f"{digest_filename=} boxers={len(digest)}")


if __name__ == "__main__":
    main()
