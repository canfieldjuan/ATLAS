"""B2B Churn Intelligence MCP Server -- main entry point.

Usage:
    python -m atlas_brain.mcp.b2b                  # stdio
    python -m atlas_brain.mcp.b2b --sse             # SSE HTTP transport
    python -m atlas_brain.mcp.b2b_churn_server      # backward compat (deprecated)
"""

import sys
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from ._shared import logger


@asynccontextmanager
async def _lifespan(server):
    """Initialize DB pool on startup, close on shutdown."""
    from atlas_brain.storage.database import init_database, close_database

    await init_database()
    logger.info("B2B Churn MCP: DB pool initialized")
    yield
    await close_database()


mcp = FastMCP(
    "atlas-b2b-churn",
    instructions=(
        "B2B churn intelligence server for Atlas. "
        "Query vendor churn signals, search enriched reviews, read intelligence "
        "reports, identify high-intent companies, monitor pipeline health, "
        "manage scrape targets, view blog posts, and list affiliate partners. "
        "Data sourced from 16 review sites: G2, Capterra, TrustRadius, Reddit, "
        "Gartner, GetApp, GitHub, HackerNews, PeerSpot, ProductHunt, Quora, "
        "RSS, StackOverflow, TrustPilot, Twitter/X, YouTube."
    ),
    lifespan=_lifespan,
)

# Register all domain modules
from . import signals  # noqa: E402, F401
from . import reviews  # noqa: E402, F401
from . import reports  # noqa: E402, F401
from . import products  # noqa: E402, F401
from . import displacement  # noqa: E402, F401
from . import vendor_history  # noqa: E402, F401
from . import vendor_registry  # noqa: E402, F401
from . import scrape_targets  # noqa: E402, F401
from . import pipeline  # noqa: E402, F401
from . import corrections  # noqa: E402, F401
from . import calibration  # noqa: E402, F401
from . import webhooks  # noqa: E402, F401
from . import crm_events  # noqa: E402, F401
from . import content  # noqa: E402, F401


def main():
    transport = "sse" if "--sse" in sys.argv else "stdio"
    if transport == "sse":
        from atlas_brain.config import settings
        from atlas_brain.mcp.auth import run_sse_with_auth

        mcp.settings.host = settings.mcp.host
        mcp.settings.port = settings.mcp.b2b_churn_port
        run_sse_with_auth(mcp, settings.mcp.host, settings.mcp.b2b_churn_port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
