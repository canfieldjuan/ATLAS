# PR-Invoicing-Readonly-ChatGPT-MCP

## Why this slice exists

The full Atlas invoicing MCP server exposes read tools and mutating tools on the same surface. That is acceptable for local stdio and bearer-protected HTTP clients, but it is the wrong shape for a ChatGPT-style connector that should not receive write/send/payment tools. The remaining read tools still expose customer financial data, so HTTP mode must stay authenticated even on the read-only surface.

This slice adds a separate authenticated read-only MCP server so connector clients can inspect invoice state without gaining write/send/payment tools.

## Scope (this PR)

1. Add a new `atlas-invoicing-readonly` MCP server that wraps existing invoicing read tools.
2. Add a dedicated HTTP port for the read-only server.
3. Update MCP docs and audit mappings so the new server is tracked by the existing local review bundle.
4. Add tests pinning the exact read-only tool surface.

### Files touched

- `atlas_brain/config.py`
- `atlas_brain/mcp/__init__.py`
- `atlas_brain/config_defaults.py`
- `atlas_brain/mcp/invoicing_readonly_server.py`
- `atlas_brain/services/mcp_client.py`
- `scripts/audit_claude_md_claims.py`
- `scripts/audit_mcp_port_assignments.py`
- `scripts/audit_mcp_tool_names_match_docs.py`
- `tests/test_audit_mcp_port_assignments.py`
- `tests/test_invoicing_readonly_mcp.py`
- `tests/test_pre_push_audit.py`
- `CLAUDE.md`
- `plans/PR-Invoicing-Readonly-ChatGPT-MCP.md`

## Mechanism

The new server creates its own `FastMCP("atlas-invoicing-readonly")` instance and registers only eight read/review tools:

- `get_invoice`
- `list_invoices`
- `search_invoices`
- `list_pending_drafts`
- `customer_balance`
- `payment_history`
- `list_services`
- `get_service`

Each tool delegates to the existing implementation in `atlas_brain.mcp.invoicing_server`, so business logic stays single-source-of-truth. The `--sse` path requires `ATLAS_MCP_AUTH_TOKEN` and wraps the streamable HTTP app with the existing bearer middleware; safety comes from both authentication and the absence of mutating tools on this server. Runtime host/port are read from `ATLAS_MCP_HOST` / `ATLAS_MCP_INVOICING_READONLY_PORT`, using shared MCP default constants so the config field and entrypoint cannot drift while avoiding global `settings` import during MCP bootstrap.

## Intentional

- No write/send tools are re-exported, including `export_invoice_pdf`, because it writes files by default.
- The full invoicing server remains unchanged for write actions; the read-only HTTP server also requires bearer auth because read tools expose financial data.
- The read-only server uses a separate port (`8065`) instead of trying to branch behavior by token or path inside the full server.

## Deferred

- Public Funnel routing is operational, not baked into the Python server. Any public endpoint/path must forward bearer-authenticated requests only.
- OAuth/tool-security metadata for future ChatGPT write tools is deferred. Write tools should not be exposed to connector traffic until that exists.

## Verification

```bash
python -m py_compile atlas_brain/mcp/invoicing_readonly_server.py atlas_brain/config.py
pytest tests/test_invoicing_readonly_mcp.py tests/test_pre_push_audit.py
python scripts/audit_claude_md_claims.py
python scripts/audit_mcp_tool_names_match_docs.py
python scripts/audit_mcp_port_assignments.py
```

## Estimated diff size

| Scope | Estimated LOC |
|---|---:|
| Total | 500 |

One new server, docs, audit mappings, and focused tests. Under the 400 LOC target.
