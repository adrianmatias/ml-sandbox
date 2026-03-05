import os.path
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Loc:
    root: Path = Path(os.path.dirname(__file__)).parent
    data: Path = root / "data"
    vect_db: Path = data / "vect_db"


@dataclass(frozen=True)
class Const:
    loc = Loc()


CONST = Const()
