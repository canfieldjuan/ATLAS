"""Competitive intelligence MCP package exports."""
from __future__ import annotations

import importlib
from typing import Any

_LOCAL_EXPORTS = {
    "mcp": "server",
}


def __getattr__(name: str) -> Any:
    module_name = _LOCAL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
