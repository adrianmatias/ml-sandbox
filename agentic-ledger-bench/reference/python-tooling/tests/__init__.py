"""Test suite for the ledger module (see spec §8).

Importing this package puts ``python/`` on ``sys.path`` so the tests can import
``ledger`` no matter which directory the runner was started from.
"""

import sys
from pathlib import Path

_PYTHON_DIR = str(Path(__file__).resolve().parent.parent)
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)
