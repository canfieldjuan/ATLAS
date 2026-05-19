## Why this slice exists

PR #604 proved OAuth metadata discovery is routable, and the follow-up manual
check proved the full public flow works through token exchange and MCP
`list_tools`. That manual command is not durable: after a restart, Funnel route
change, or ChatGPT connector failure, we need a repeatable operator smoke that
separates Atlas OAuth breakage from ChatGPT-specific behavior.

This slice captures the no-invoice OAuth e2e verification as a reusable script.

## Scope (this PR)

1. Add a read-only e2e smoke for the public OAuth connector flow.
2. Validate dynamic client registration, authorization redirect, operator
   approval, token exchange, and OAuth-authenticated MCP `list_tools`.
3. Require the exact eight read-only invoicing tools and reject mutating/extra
   tools.
4. Add fixture tests for helper behavior and failure modes.
5. Document the command and the required Tailscale route shape.

### Files touched

- `scripts/check_invoicing_readonly_oauth_e2e.py`
- `tests/test_check_invoicing_readonly_oauth_e2e.py`
- `CLAUDE.md`
- `plans/PR-Invoicing-Readonly-OAuth-E2E-Smoke.md`

## Mechanism

The script performs the same public flow ChatGPT needs, but stops at tool
listing:

```text
POST /register
GET /authorize -> /oauth/approve?request_id=...
POST /oauth/approve with operator approval token
POST /token with PKCE verifier
MCP initialize + list_tools with the returned bearer token
```

It uses `https://chat.openai.com/aip/callback` as the default redirect URI, but
never follows that redirect. It extracts the authorization code locally and
uses it at the token endpoint. The final MCP check only lists tools; it does
not call invoice/service/balance/payment tools.

## Intentional

- The script requires an approval token because it exercises the actual
  operator approval boundary. It never prints the approval token, client secret,
  access token, refresh token, or authorization code.
- The script does not persist OAuth clients or tokens; the server remains the
  single-process in-memory provider from PR #600.
- The script is separate from `check_invoicing_readonly_oauth_discovery.py` so
  a cheap metadata-only smoke is still available when the operator does not
  want to approve a client.

## Deferred

- Durable OAuth client/token persistence remains deferred until multi-process
  or restart-resilient connector sessions are needed.
- A real ChatGPT connector click-through remains operational/manual because it
  requires the ChatGPT UI.

## Verification

Planned commands:

```bash
.venv/bin/python -m py_compile scripts/check_invoicing_readonly_oauth_e2e.py tests/test_check_invoicing_readonly_oauth_e2e.py
.venv/bin/python -m pytest tests/test_check_invoicing_readonly_oauth_e2e.py
.venv/bin/python scripts/check_invoicing_readonly_oauth_e2e.py --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-readonly --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-readonly/mcp
bash scripts/local_pr_review.sh --allow-dirty
```

## Estimated diff size

| Area | LOC churn (approx) |
|---|---:|
| E2E smoke script | ~260 |
| Tests | ~200 |
| Docs/plan | ~100 |
| **Total** | **~560** |

This exceeds the soft cap because a networked e2e operator script needs enough
helper seams to test failure behavior without hitting the live connector.
