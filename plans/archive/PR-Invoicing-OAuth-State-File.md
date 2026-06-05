# Invoicing OAuth State File

## Why this slice exists

The ChatGPT invoicing MCP connectors now work through OAuth, but the OAuth
provider is process-local. After a server restart, ChatGPT can still hold its
registered client credentials and refresh token while the server forgets both.
That creates a predictable operator failure mode: the connector works until the
local MCP server or PC restarts, then token refresh fails and the operator has
to reconnect.

This slice adds optional local state-file persistence for invoicing OAuth client
registrations and refresh tokens. It closes the restart gap without changing the
MCP tool surface or making OAuth approval automatic.

## Scope

1. Add optional state-file persistence to the shared invoicing OAuth provider.
2. Persist registered OAuth clients and refresh tokens only.
3. Wire separate state-file env vars for read-only and draft-writer connectors.
4. Document the env vars and the restart behavior.
5. Cap persisted dynamic client registrations to avoid durable state-file growth
   from unauthenticated `/register` traffic.
6. Add tests proving client/refresh-token state survives provider recreation.

### Files touched

- `atlas_brain/mcp/invoicing_readonly_oauth.py`
- `atlas_brain/mcp/invoicing_readonly_server.py`
- `atlas_brain/mcp/invoicing_draft_writer_oauth.py`
- `atlas_brain/mcp/invoicing_draft_writer_server.py`
- `tests/test_invoicing_readonly_oauth.py`
- `tests/test_invoicing_draft_writer_oauth.py`
- `docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md`
- `docs/INVOICING_MCP_WRITE_ACCESS_GUARDRAILS.md`
- `plans/PR-Invoicing-OAuth-State-File.md`

## Mechanism

`InvoicingReadonlyOAuthProvider` will accept an optional `state_file` path. When
present, the provider loads JSON state at startup and writes it after client
registration, authorization-code exchange, and token revocation. The persisted
state includes OAuth client registrations and refresh tokens. Pending approval
requests, one-time authorization codes, and access tokens remain process-local.

The read-only server reads
`ATLAS_MCP_INVOICING_READONLY_OAUTH_STATE_FILE`. The draft-writer server reads
`ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_STATE_FILE`. Both env vars are optional;
without them, behavior stays in-memory.

Persisted dynamic client registrations are capped because OAuth dynamic client
registration is intentionally public. The cap keeps bogus registrations from
growing durable local state unboundedly.

## Intentional

- Access tokens are not persisted. A restarted connector should refresh using a
  persisted refresh token rather than keep using an old bearer token.
- Pending authorizations and authorization codes are not persisted because they
  are short-lived one-time values.
- The state file is local operator secret material. The provider writes it with
  owner-only permissions and docs place it under `.secrets/`.
- No DB migration. These OAuth connectors are local operator MCP surfaces, not a
  multi-tenant web auth system.
- Dynamic client registration is still public OAuth behavior, but persisted
  client state is capped so abuse cannot grow the state file indefinitely.

## Deferred

- Encrypting the state file at rest can be considered later if these connectors
  move off a local operator machine.
- Applying the same state-file pattern to non-invoicing MCP servers is a future
  per-server rollout.

## Verification

- Focused OAuth tests: 21 passed.
- Python compile check for touched OAuth modules/tests: passed.
- Git whitespace check: passed.
- Local PR review bundle in advisory dirty mode: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Provider persistence | ~90 |
| Server env wiring | ~15 |
| Tests | ~130 |
| Docs | ~25 |
| Plan doc | ~95 |
| **Total** | ~355 |
