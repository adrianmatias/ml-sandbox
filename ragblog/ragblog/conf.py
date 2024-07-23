from dataclasses import dataclass


@dataclass
class Path:
    data: str
    chroma: str


@dataclass
class Conf:
    path: Path


CONF = Conf(path=Path(data="data", chroma="chroma"))
CONF_BLOG_FULL = Conf(path=Path(data="data", chroma="chroma_full"))
