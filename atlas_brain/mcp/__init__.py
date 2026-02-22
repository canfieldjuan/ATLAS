"""
Atlas MCP servers package.

Two standalone MCP servers:
  - CRM server   (atlas_brain.mcp.crm_server)    — customer/contact management
  - Email server (atlas_brain.mcp.email_server)  — send + read email

Each server can run in stdio mode (default, for Claude Desktop / Cursor)
or SSE/HTTP mode (for network-accessible deployment).

    # CRM — stdio
    python -m atlas_brain.mcp.crm_server

    # Email — stdio
    python -m atlas_brain.mcp.email_server

    # CRM — SSE on port 8056
    python -m atlas_brain.mcp.crm_server --sse

    # Email — SSE on port 8057
    python -m atlas_brain.mcp.email_server --sse
"""
