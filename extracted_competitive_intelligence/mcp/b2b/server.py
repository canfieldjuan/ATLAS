"""Competitive intelligence MCP server entry point."""
from __future__ import annotations

import importlib
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Callable

from ._shared import TOOL_GROUPS, logger


class _ImportOnlyMCP:
    """Minimal decorator registry for environments without the MCP package."""

    def __init__(self, name: str, **_: Any) -> None:
        self.name = name
        self.settings = SimpleNamespace(host=None, port=None)
        self.tools: dict[str, Callable[..., Any]] = {}

    def tool(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = kwargs.get("name") or getattr(func, "__name__", "")
            if tool_name:
                self.tools[tool_name] = func
            return func

        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return decorator(args[0])
        return decorator

    def remove_tool(self, name: str) -> None:
        self.tools.pop(name, None)

    def run(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("The 'mcp' package is required to run the MCP server")


@asynccontextmanager
async def _lifespan(server: Any):
    """Initialize DB pool on startup, close on shutdown."""
    from ...storage.database import get_db_pool

    pool = get_db_pool()
    if hasattr(pool, "initialize"):
        await pool.initialize()
    logger.info("Competitive intelligence MCP: DB pool initialized")
    yield
    if hasattr(pool, "close"):
        await pool.close()


try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = _ImportOnlyMCP


mcp = FastMCP(
    "extracted-competitive-intelligence",
    instructions=(
        "Competitive intelligence server. Query vendor churn signals, search "
        "enriched reviews, read intelligence reports, identify high-intent "
        "companies, manage scrape targets, and persist competitive reports."
    ),
    lifespan=_lifespan,
)


def _apply_tool_gating(server: Any) -> None:
    """Remove tools from groups not listed in config."""
    try:
        from ...config import settings

        raw = getattr(settings.b2b_churn, "mcp_tool_groups", "all")
    except Exception:
        raw = "all"

    raw = str(raw or "all").strip().lower()
    if raw == "all":
        return

    enabled_groups = {g.strip() for g in raw.split(",") if g.strip()}
    tools_to_remove: list[str] = []
    for group_name, tool_names in TOOL_GROUPS.items():
        if group_name not in enabled_groups:
            tools_to_remove.extend(tool_names)

    removed = 0
    for name in tools_to_remove:
        try:
            server.remove_tool(name)
            removed += 1
        except Exception:
            pass

    if removed:
        logger.info(
            "Competitive intelligence MCP tool gating removed %d tools, enabled groups: %s",
            removed,
            enabled_groups,
        )


def _register_domain_modules() -> None:
    """Import extracted domain modules so decorators register their tools."""
    for module_name in (
        "cross_vendor",
        "displacement",
        "vendor_registry",
        "write_intelligence",
    ):
        importlib.import_module(f"{__package__}.{module_name}")


_register_domain_modules()
_apply_tool_gating(mcp)


def main() -> None:
    transport = "sse" if "--sse" in sys.argv else "stdio"
    if transport == "sse":
        from ...config import settings

        mcp.settings.host = settings.mcp.host
        mcp.settings.port = settings.mcp.b2b_churn_port
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
