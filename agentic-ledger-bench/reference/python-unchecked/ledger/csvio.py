"""Minimal RFC-4180-style CSV tokeniser (spec §2).

The standard library's ``csv`` module is not used, deliberately:

* it makes runtime behaviour depend on ``csv.field_size_limit`` and on dialect
  sniffing choices that are awkward to freeze, and
* its error behaviour for unbalanced quotes (``_csv.Error`` at an arbitrary
  point, or silent concatenation of continuation lines) is harder to map onto
  the frozen ``ROW`` error than a 40-line explicit state machine.

So quoted fields are handled here, explicitly, per line.  Supported: a field
may be wrapped in ``"``, and ``""`` inside a quoted field is one literal ``"``.
A physical line is exactly one record: embedded newlines inside quotes are not
supported and are reported as ``ROW`` rather than silently merged.
"""

from __future__ import annotations

from .errors import ROW, LedgerError


def tokenize_line(line: str) -> list[str]:
    """Split one physical CSV line into fields, stripping CSV quoting.

    Raises ``LedgerError(ROW, ...)`` for an unterminated quoted field or for
    characters after a closing quote.
    """
    fields: list[str] = []
    buf: list[str] = []
    index = 0
    length = len(line)
    in_quotes = False
    quoted_field = False

    while index < length:
        char = line[index]
        if in_quotes:
            if char == '"':
                if index + 1 < length and line[index + 1] == '"':
                    buf.append('"')
                    index += 2
                    continue
                in_quotes = False
                index += 1
                continue
            buf.append(char)
            index += 1
            continue

        if char == '"':
            if quoted_field:
                raise LedgerError(ROW, "unexpected quote inside a field")
            if buf and "".join(buf).strip() != "":
                raise LedgerError(ROW, "quote inside an unquoted field")
            buf = []
            quoted_field = True
            in_quotes = True
            index += 1
            continue

        if char == ",":
            fields.append("".join(buf))
            buf = []
            quoted_field = False
            index += 1
            continue

        if quoted_field:
            raise LedgerError(ROW, "unexpected text after a closing quote")
        buf.append(char)
        index += 1

    if in_quotes:
        raise LedgerError(ROW, "unterminated quoted field")
    fields.append("".join(buf))
    return fields


def is_empty_record(fields: list[str]) -> bool:
    """True for a record whose every field is blank (``,,,`` in the reference file).

    A wholly blank physical line (``""``) also counts: it carries no data and
    is skipped rather than reported as a field-count error.  The reference file
    has neither, so this is a strict broadening, not a behaviour change.
    """
    return all(field.strip() == "" for field in fields)


def read_csv_lines(path: str, *, encoding: str = "utf-8-sig") -> list[str]:
    """Read a statement file into physical lines, undoing BOM and CRLF.

    ``encoding="utf-8-sig"`` strips a leading BOM when present and is a no-op
    otherwise, so the reader is safe for both the spec's stated UTF-8-with-BOM
    encoding and the BOM-less bytes actually present in the shipped reference
    file (see NOTES.md "spec friction").

    Universal newlines are used, so CRLF and LF files produce identical lines
    (spec §4's CRLF fixture).
    """
    with open(path, "r", encoding=encoding, newline=None) as handle:
        text = handle.read()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        # A trailing newline terminates the last record; it is not a record.
        lines.pop()
    return lines
