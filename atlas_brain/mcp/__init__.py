"""
Atlas MCP servers package.

Seven standalone MCP servers:
  - CRM server          (atlas_brain.mcp.crm_server)          -- customer/contact management
  - Email server        (atlas_brain.mcp.email_server)        -- send + read email
  - Twilio server       (atlas_brain.mcp.twilio_server)       -- calls, SMS, recordings
  - Calendar server     (atlas_brain.mcp.calendar_server)     -- calendar events, scheduling
  - Invoicing server    (atlas_brain.mcp.invoicing_server)    -- invoices, payments, services
  - Intelligence server (atlas_brain.mcp.intelligence_server) -- reports, risk, interventions
  - B2B Churn server    (atlas_brain.mcp.b2b_churn_server)    -- B2B churn signals, reviews, pipeline

Each server can run in stdio mode (default, for Claude Desktop / Cursor)
or SSE/HTTP mode (for network-accessible deployment).

    # CRM -- stdio
    python -m atlas_brain.mcp.crm_server

    # Email -- stdio
    python -m atlas_brain.mcp.email_server

    # CRM -- SSE on port 8056
    python -m atlas_brain.mcp.crm_server --sse

    # Email -- SSE on port 8057
    python -m atlas_brain.mcp.email_server --sse
"""
