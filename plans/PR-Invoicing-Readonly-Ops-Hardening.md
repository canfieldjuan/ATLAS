## Why this slice exists

PR #590 added an authenticated read-only invoicing MCP surface for ChatGPT-style connectors. During verification, the endpoint was briefly exposed with a session-local test token, which is exactly the kind of operator mistake this public financial-data surface should make hard.

This slice adds a narrow runtime guard against placeholder or trivially weak read-only MCP tokens and adds a connector smoke command so operators can verify unauthenticated rejection plus authenticated tool-list shape without invoking invoice reads or mutating data.

## Scope (this PR)

1. Reject missing, placeholder, or too-short `ATLAS_MCP_AUTH_TOKEN` values when starting the read-only invoicing HTTP app.
2. Add a read-only connector smoke script that checks:
   - unauthenticated HTTP requests are rejected with `401`;
   - authenticated MCP initialization succeeds;
   - exactly the eight read-only invoicing tools are exposed;
   - no known mutating invoicing tools are exposed.
3. Document the production token guidance and smoke command.
4. Add focused tests for token validation and smoke-script validation behavior.

### Files touched

- `atlas_brain/mcp/invoicing_readonly_server.py`
- `scripts/check_invoicing_readonly_mcp_connector.py`
- `tests/test_invoicing_readonly_mcp.py`
- `tests/test_check_invoicing_readonly_mcp_connector.py`
- `CLAUDE.md`
- `plans/PR-Invoicing-Readonly-Ops-Hardening.md`

## Mechanism

`invoicing_readonly_server._streamable_http_app()` routes through a local `_require_http_auth_token()` helper before wrapping the Streamable HTTP app. The helper is deliberately scoped to this server because the surface exposes customer financial data; the general MCP auth middleware remains unchanged for other servers.

The new script uses a plain unauthenticated HTTP probe for the `401` boundary and the MCP Streamable HTTP client for authenticated tool discovery. It fails closed on missing token, missing tools, extra tools, and any known mutating invoice tool.

## Intentional

- This does not rotate or store tokens. Operators still provide `ATLAS_MCP_AUTH_TOKEN`; the code only rejects obvious unsafe values.
- The smoke command does not call invoice tools. Listing tools is enough to verify connector auth and surface shape without touching customer data.
- The token minimum length is specific to the read-only invoicing HTTP server. It is stricter than the generic middleware because this endpoint is intended for public connector access.

## Deferred

- A managed secret-rotation workflow is deferred until Atlas has a broader MCP secret operations lane.
- A hosted health endpoint is deferred; the smoke script is enough for local and operator verification now.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile atlas_brain/mcp/invoicing_readonly_server.py scripts/check_invoicing_readonly_mcp_connector.py tests/test_check_invoicing_readonly_mcp_connector.py tests/test_invoicing_readonly_mcp.py
.venv/bin/python -m pytest tests/test_invoicing_readonly_mcp.py tests/test_check_invoicing_readonly_mcp_connector.py
bash scripts/local_pr_review.sh --allow-dirty
```

Focused pytest result: 15 passed.

## Estimated diff size

| Area | LOC churn (approx) |
|---|---:|
| Server token guard | ~35 |
| Connector smoke script | ~150 |
| Focused tests | ~105 |
| Docs and plan | ~70 |
| **Total** | **~360** |

This stays under the 400 LOC PR target.
