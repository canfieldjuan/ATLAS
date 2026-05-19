# PR-Invoicing-Draft-Writer-OAuth

## Why this slice exists

`atlas_brain.mcp.invoicing_draft_writer_server` now pins the first safe write
surface for invoicing: draft-only creation/update plus two read helpers. The
next step is making that server connectable from ChatGPT online without
expanding the tool surface.

The read-only invoicing connector proved the OAuth, Tailscale metadata, e2e
smoke, and operator launcher pattern. This slice applies that proven pattern to
the draft-writer server with its own scope and env prefix.

## Scope (this PR)

1. Add OAuth mode to the draft-writer MCP server.
2. Use the draft-write scope `invoices.draft.write` and draft-writer-specific
   env vars.
3. Add discovery and OAuth e2e smoke scripts that validate the exact four-tool
   draft-writer surface.
4. Add an operator launcher for local/public startup guidance.
5. Add tests for server OAuth metadata, smoke helpers, and launcher validation.
6. Update docs with the public rollout commands while keeping send/payment/void
   out of scope.

### Files touched

- `atlas_brain/mcp/invoicing_draft_writer_oauth.py`
- `atlas_brain/mcp/invoicing_draft_writer_server.py`
- `scripts/check_invoicing_draft_writer_oauth_discovery.py`
- `scripts/check_invoicing_draft_writer_oauth_e2e.py`
- `scripts/start_invoicing_draft_writer_oauth_server.py`
- `tests/test_invoicing_draft_writer_oauth.py`
- `tests/test_check_invoicing_draft_writer_oauth_discovery.py`
- `tests/test_check_invoicing_draft_writer_oauth_e2e.py`
- `tests/test_start_invoicing_draft_writer_oauth_server.py`
- `docs/INVOICING_MCP_WRITE_ACCESS_GUARDRAILS.md`
- `docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md`
- `plans/PR-Invoicing-Draft-Writer-OAuth.md`

## Mechanism

The draft-writer OAuth provider subclasses the proven read-only provider but
changes the default scope and approval-page copy. The server reads
`ATLAS_MCP_INVOICING_DRAFT_WRITER_AUTH_MODE`; bearer mode remains the default,
and OAuth mode configures FastMCP auth with `invoices.draft.write`.

The discovery and e2e scripts mirror the read-only smokes with draft-writer env
vars, resource URLs, scope, and exact tool allowlist. The e2e smoke still calls
only `list_tools`, so it verifies OAuth and tool boundaries without creating or
updating invoices.

The operator launcher mirrors the read-only launcher and prints masked env,
Tailscale protected-resource routing, and the two smoke commands.

## Intentional

- OAuth mode is added only to the draft-writer server. The full invoicing server
  remains unchanged and is still not ChatGPT-facing.
- The e2e smoke does not call `create_draft_invoice`; public connector smoke
  should prove auth/tool surface, not mutate invoice data.
- CLAUDE MCP tool-count docs are not updated in this slice because the
  pre-existing auditor only tracks `### ... MCP Server` headings and the public
  port docs for fully listed servers. The draft-writer public rollout commands
  live in the guardrail/runbook docs until we decide to add it to the global MCP
  inventory table.

## Deferred

- Live public Tailscale run and actual ChatGPT connector approval.
- Adding the draft-writer server to global MCP inventory/port tables if we want
  it documented alongside always-on servers.
- Any send/approve/payment/void/PDF/service mutation OAuth scope.

## Verification

- .venv/bin/python -m py_compile atlas_brain/mcp/invoicing_draft_writer_oauth.py atlas_brain/mcp/invoicing_draft_writer_server.py scripts/check_invoicing_draft_writer_oauth_discovery.py scripts/check_invoicing_draft_writer_oauth_e2e.py scripts/start_invoicing_draft_writer_oauth_server.py
- .venv/bin/pytest tests/test_invoicing_draft_writer_oauth.py tests/test_check_invoicing_draft_writer_oauth_discovery.py tests/test_check_invoicing_draft_writer_oauth_e2e.py tests/test_start_invoicing_draft_writer_oauth_server.py
- git diff --check
- bash scripts/local_pr_review.sh

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `atlas_brain/mcp/invoicing_draft_writer_oauth.py` | ~111 |
| `atlas_brain/mcp/invoicing_draft_writer_server.py` | ~76 |
| `scripts/check_invoicing_draft_writer_oauth_discovery.py` | ~215 |
| `scripts/check_invoicing_draft_writer_oauth_e2e.py` | ~401 |
| `scripts/start_invoicing_draft_writer_oauth_server.py` | ~240 |
| `tests/test_invoicing_draft_writer_oauth.py` | ~278 |
| `tests/test_check_invoicing_draft_writer_oauth_discovery.py` | ~161 |
| `tests/test_check_invoicing_draft_writer_oauth_e2e.py` | ~266 |
| `tests/test_start_invoicing_draft_writer_oauth_server.py` | ~223 |
| Docs + plan | ~153 |
| **Total** | **~2,124** |

This intentionally exceeds the 400 LOC soft cap because the safe OAuth rollout
requires server wiring, discovery smoke, e2e smoke, launcher, and regression
tests together. Splitting the smokes from the server would create an exposed
OAuth path without the verification harness that made the read-only rollout
safe.
