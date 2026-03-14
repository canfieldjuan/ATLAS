"""B2B Churn Intelligence MCP Server.

DEPRECATED: Use atlas_brain.mcp.b2b instead.

This module is a backward-compatibility bridge. All tools have been
modularized into atlas_brain/mcp/b2b/ (14 domain files, 61 tools).

Usage (both still work):
    python -m atlas_brain.mcp.b2b_churn_server      # legacy
    python -m atlas_brain.mcp.b2b                    # new
"""

import sys

# Re-export mcp so existing configs that import from this module still work
from .b2b.server import mcp  # noqa: F401

if __name__ == "__main__":
    from .b2b.server import main
    main()
