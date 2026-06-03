"""CSV cell serialization helpers for Content Ops exports."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any


_SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@")


def csv_cell_value(value: Any) -> Any:
    """Return a CSV-safe cell value while preserving export row contracts."""

    if isinstance(value, (Mapping, list, tuple)):
        text: Any = json.dumps(value, default=str, separators=(",", ":"))
    elif value is None:
        text = ""
    else:
        text = value

    if (
        isinstance(text, str)
        and text
        and text.lstrip().startswith(_SPREADSHEET_FORMULA_PREFIXES)
    ):
        return "'" + text
    return text


__all__ = ["csv_cell_value"]
