from dataclasses import dataclass


@dataclass
class Path:
    data: str = "data"
    chroma: str = "chroma"


@dataclass
class ConfCrawler:
    url: str = "https://delightfulobservaciones.blogspot.com/"
    post_count_min: int = 2
