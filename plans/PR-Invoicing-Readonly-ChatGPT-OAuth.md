## Why this slice exists

The read-only invoicing MCP endpoint works with direct MCP clients that can send a static `Authorization: Bearer ...` header, but ChatGPT online did not connect. The likely mismatch is authentication: ChatGPT custom MCP connectors expect OAuth or no auth, while the current read-only server requires a raw bearer token before discovery.

This slice adds a library-backed OAuth mode for the read-only invoicing MCP server while preserving the existing bearer-token mode.

## Scope (this PR)

1. Add a minimal single-operator OAuth provider for the read-only invoicing MCP server.
2. Wire `ATLAS_MCP_INVOICING_READONLY_AUTH_MODE=oauth` to FastMCP's built-in OAuth/resource-server auth routes.
3. Keep `bearer` mode as the default for existing direct clients.
4. Add focused tests for OAuth provider registration, authorization-code exchange, approval-token enforcement, and server auth-mode wiring.
5. Document the ChatGPT OAuth environment variables and the remaining public-route requirement.

### Files touched

- `atlas_brain/mcp/invoicing_readonly_oauth.py`
- `atlas_brain/mcp/invoicing_readonly_server.py`
- `tests/test_invoicing_readonly_oauth.py`
- `CLAUDE.md`
- `plans/PR-Invoicing-Readonly-ChatGPT-OAuth.md`

## Mechanism

`invoicing_readonly_oauth.py` implements the MCP package's `OAuthAuthorizationServerProvider` protocol with in-memory client, authorization-code, refresh-token, and access-token stores. The provider supports dynamic client registration, PKCE authorization-code exchange through the MCP library's token handler, and a local approval page that requires `ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN`.

When `ATLAS_MCP_INVOICING_READONLY_AUTH_MODE=oauth`, `_streamable_http_app()` configures the existing `FastMCP` instance with `AuthSettings`, `ClientRegistrationOptions`, the provider, and `ProviderTokenVerifier`, then returns the library-authenticated Streamable HTTP app. In bearer mode, the existing static-token middleware remains unchanged.

## Intentional

- OAuth mode uses in-memory state. This is enough for a single hosted connector process; persistence/rotation can be added later if needed.
- The approval step is not auto-approved. Someone who discovers the URL still needs the operator approval token to complete OAuth.
- This does not expose mutating invoicing tools. The underlying MCP tool surface stays the eight read-only tools from PR #590.

## Deferred

- Durable OAuth client/token persistence is deferred until this needs multi-process or restart-resilient connector sessions.
- A dedicated public hostname or extra Tailscale route for OAuth `/.well-known/...` metadata may be required operationally because the current MCP URL is path-prefixed under `/invoicing-readonly`.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile atlas_brain/mcp/invoicing_readonly_oauth.py atlas_brain/mcp/invoicing_readonly_server.py
.venv/bin/python -m pytest tests/test_invoicing_readonly_mcp.py tests/test_invoicing_readonly_oauth.py tests/test_check_invoicing_readonly_mcp_connector.py
bash scripts/local_pr_review.sh --allow-dirty
```

Focused pytest result: 23 passed.

## Estimated diff size

| Area | LOC churn (approx) |
|---|---:|
| OAuth provider | ~275 |
| Server wiring/docs | ~80 |
| Tests | ~265 |
| Plan | ~70 |
| **Total** | **~710** |

This is over the 400 LOC soft target because OAuth support needs a protocol provider plus tests; splitting the provider from server wiring would leave no runnable ChatGPT path.
