"""
Atlas MCP servers package.

Twelve standalone MCP server entry points:
  - CRM server          (atlas_brain.mcp.crm_server)          -- customer/contact management
  - Email server        (atlas_brain.mcp.email_server)        -- send + read email
  - Twilio server       (atlas_brain.mcp.twilio_server)       -- calls, SMS, recordings
  - Calendar server     (atlas_brain.mcp.calendar_server)     -- calendar events, scheduling
  - Invoicing server    (atlas_brain.mcp.invoicing_server)    -- invoices, payments, services
  - Invoicing readonly (atlas_brain.mcp.invoicing_readonly_server) -- authenticated invoice review
  - Content Ops deflection readonly (atlas_brain.mcp.content_ops_deflection_readonly_server) -- report snapshots
  - Content Ops marketer verify (atlas_brain.mcp.content_ops_marketer_verify_server) -- verify-only draft review
  - Intelligence server (atlas_brain.mcp.intelligence_server) -- reports, risk, interventions
  - B2B Churn server    (atlas_brain.mcp.b2b_churn_server)    -- B2B churn signals, reviews, pipeline
  - Universal Scraper   (atlas_brain.mcp.scraper_server)      -- schema-driven web extraction
  - Memory server       (atlas_brain.mcp.memory_server)       -- knowledge graph + conversation history

Each server can run in stdio mode (default, for Claude Desktop / Cursor)
or SSE/HTTP mode (for network-accessible deployment).

    # CRM -- stdio
    python -m atlas_brain.mcp.crm_server

    # Email -- stdio
    python -m atlas_brain.mcp.email_server

    # Invoicing readonly -- stdio
    python -m atlas_brain.mcp.invoicing_readonly_server

    # Content Ops marketer verify -- stdio
    python -m atlas_brain.mcp.content_ops_marketer_verify_server

    # Memory -- stdio
    python -m atlas_brain.mcp.memory_server

    # CRM -- SSE on port 8056
    python -m atlas_brain.mcp.crm_server --sse

    # Email -- SSE on port 8057
    python -m atlas_brain.mcp.email_server --sse

    # Invoicing readonly -- SSE on port 8065
    ATLAS_MCP_AUTH_TOKEN=<token> python -m atlas_brain.mcp.invoicing_readonly_server --sse

    # Content Ops marketer verify -- SSE on port 8068
    ATLAS_MCP_AUTH_TOKEN=<token> python -m atlas_brain.mcp.content_ops_marketer_verify_server --sse

    # Memory -- SSE on port 8064
    python -m atlas_brain.mcp.memory_server --sse
"""
