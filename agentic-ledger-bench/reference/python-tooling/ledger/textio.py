"""The one function that turns a path into text.

Shared by the statement parser and the registry loader so both report an
unreadable file the same way, with the same ``IO`` error kind (§6).
"""

from pathlib import Path

from ledger.errors import InputError


def read_utf8_text(path: Path) -> str:
    """Read a UTF-8 text file, tolerating a BOM on the first line (spec §2).

    Args:
        path: File to read.

    Returns:
        The decoded contents, BOM removed, line endings untouched.

    Raises:
        InputError: If the file cannot be read or is not valid UTF-8.
    """
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise InputError(path.name, exc.strerror or str(exc)) from exc
    except UnicodeDecodeError as exc:
        raise InputError(path.name, f"not valid UTF-8 ({exc.reason})") from exc
