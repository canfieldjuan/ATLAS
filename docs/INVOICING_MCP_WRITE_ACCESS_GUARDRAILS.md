# Invoicing MCP write-access guardrails

This is the contract for exposing invoice writes through ChatGPT or any other
remote MCP client.

The read-only invoicing connector is proven. Write access should not reuse the
full `atlas-invoicing` server as-is because that server intentionally includes
local-operator tools with send, payment, void, service, and file-write side
effects.

## Decision

The first write-capable ChatGPT connector must be a separate draft-only server:

```text
atlas_brain.mcp.invoicing_draft_writer_server
```

It should expose the smallest useful surface:

- `create_draft_invoice`
- `update_draft_invoice`
- `get_invoice`
- `list_pending_drafts`

It must not expose the full `atlas_brain.mcp.invoicing_server` tool surface.

## Why separate server

The existing full server is designed for trusted local operators. It contains
tools that can create drafts, update drafts, send invoices, record payments,
void invoices, mutate service agreements, generate PDFs, and write files to
disk.

For a hosted ChatGPT connector, safety should come from the process/tool
boundary, not from model instructions.

A separate server gives us:

- Exact tool allowlist smoke tests.
- Independent OAuth scope and approval copy.
- Separate env prefix and public route.
- Lower chance that a future full-server tool accidentally becomes remote.
- A clean path to add more writes later behind explicit PRs.

## Allowed first write surface

### `create_draft_invoice`

Creates an invoice in `draft` status only.

Required behavior:

- Requires an idempotency key or stable `source_ref`.
- Writes `source="chatgpt_draft_writer"` or an equivalent explicit source.
- Writes metadata identifying the connector and operator-review requirement.
- Does not send email.
- Does not mark the invoice sent.
- Does not record payment.
- Does not void anything.
- Does not export or save a PDF.
- Does not mutate services.

Important implementation detail: do not delegate blindly to the existing
`create_invoice` MCP tool if that would trigger CRM contact resolution or CRM
logging. The draft writer should either call the repository directly with
explicit metadata or use a shared helper that guarantees no external side
effects.

### `update_draft_invoice`

Updates only existing draft invoices.

Required behavior:

- Rejects non-draft invoices.
- Accepts only draft-safe fields such as line items, due date, notes, tax,
  discount, invoice description, and contact name.
- Does not send, approve, void, record payment, export PDF, or mutate service
  state.
- Preserves metadata that marks the invoice as connector-created or
  connector-touched.

### Read helpers

The write connector may include enough read tools to let the operator inspect
the drafts it creates:

- `get_invoice`
- `list_pending_drafts`

Do not copy the full read-only surface unless there is a clear need. The
read-only connector already handles broader invoice review.

## Denied in the first write connector

These full-server tools must not appear in the first write-access connector:

- `approve_and_send`
- `send_invoice`
- `record_payment`
- `mark_void`
- `export_invoice_pdf`
- `create_service`
- `update_service`
- `set_service_status`

These are not small variations of draft write access. They carry external,
financial, or file-system side effects and need separate design.

## OAuth contract

Use a separate OAuth scope:

```text
invoices.draft.write
```

Use a separate env prefix:

```bash
ATLAS_MCP_INVOICING_DRAFT_WRITER_AUTH_MODE=oauth
ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL=https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer
ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL=https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp
ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN=<long-random-token>
ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_STATE_FILE=.secrets/invoicing-draft-writer-oauth-state.json
ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT=<dedicated-port>
```

For local operator runs, prefer storing the approval token in a private file
instead of passing it directly on the command line:

```bash
mkdir -p .secrets
chmod 700 .secrets
python - <<'PY' > .secrets/invoicing-draft-writer-approval-token
import secrets

print(secrets.token_urlsafe(32))
PY
chmod 600 .secrets/invoicing-draft-writer-approval-token
```

The optional OAuth state file stores ChatGPT client registrations and refresh
tokens so the connector survives local MCP server restarts. Keep it under
`.secrets/`; the provider writes it with owner-only permissions. It does not
store pending approval requests, one-time authorization codes, or access tokens.

The approval page copy must say draft-write access, not read-only access.

The connector must not share the read-only approval token. Separate scopes and
separate approval tokens make accidental privilege expansion easier to detect.

## Draft-writer rollout commands

Start the draft-writer OAuth server through the operator launcher:

```bash
.venv/bin/python scripts/start_invoicing_draft_writer_oauth_server.py \
  --approval-token-file .secrets/invoicing-draft-writer-approval-token \
  --dry-run
.venv/bin/python scripts/start_invoicing_draft_writer_oauth_server.py \
  --approval-token-file .secrets/invoicing-draft-writer-approval-token
```

The path-prefixed public URL requires both Funnel routes:

```bash
tailscale funnel --bg --yes \
  --set-path /invoicing-draft-writer \
  http://127.0.0.1:8066
tailscale funnel --bg --yes \
  --set-path /.well-known/oauth-protected-resource/invoicing-draft-writer \
  http://127.0.0.1:8066/.well-known/oauth-protected-resource/invoicing-draft-writer
```

Before public discovery, verify the routes are actually present and pointed at
the draft-writer port:

```bash
.venv/bin/python scripts/check_invoicing_draft_writer_funnel_routes.py
```

The default public route is:

```text
https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp
```

Verify public discovery before connecting ChatGPT:

```bash
.venv/bin/python scripts/check_invoicing_draft_writer_oauth_discovery.py \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp
```

Then verify OAuth token exchange and the exact four-tool surface:

```bash
.venv/bin/python scripts/check_invoicing_draft_writer_oauth_e2e.py \
  --approval-token-file .secrets/invoicing-draft-writer-approval-token \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp
```

The e2e smoke lists tools only. It must not create or update invoices.

After discovery and no-mutation e2e pass, operators can run the explicit live
write smoke:

```bash
.venv/bin/python scripts/check_invoicing_draft_writer_live_write.py \
  --create-blocked-draft \
  --approval-token-file .secrets/invoicing-draft-writer-approval-token \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp
```

That command creates or reuses one idempotent test draft with no customer email
and a zero subtotal, then verifies `get_invoice` by invoice number and
`list_pending_drafts`. The expected result is a blocked draft with `no_email`
and `subtotal_zero`, not a sendable invoice.

If discovery passes but e2e fails with `421 Misdirected Request` or
`Invalid Host header`, the MCP transport host allowlist is stale. OAuth mode
must allow the public host from
`ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL` /
`ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL` while keeping
DNS-rebinding protection enabled.

## Required implementation tests

The server PR must include tests that prove:

- The exposed MCP tool set equals the draft-writer allowlist exactly.
- All denied tools are absent.
- HTTP mode fails closed without auth.
- OAuth mode uses the `invoices.draft.write` scope.
- `create_draft_invoice` creates only `draft` invoices.
- `create_draft_invoice` is idempotent for the same idempotency key/source ref.
- `create_draft_invoice` does not touch CRM or email providers.
- `update_draft_invoice` rejects non-draft invoices.

The public smoke scripts must mirror the read-only connector pattern:

- Discovery smoke checks metadata and unauthenticated `401`.
- OAuth e2e smoke completes the flow and calls only `list_tools`.
- Live write smoke requires explicit acknowledgement and creates only a blocked
  idempotent smoke draft.
- Tool-surface check rejects denied tools by name.

## Required audit metadata

Drafts created or touched through the connector should include metadata similar
to:

```json
{
  "mcp_connector": "invoicing_draft_writer",
  "operator_review_required": true,
  "created_by_remote_connector": true,
  "idempotency_key": "<operator-provided-key>"
}
```

This is not a security boundary. It is an audit and operations affordance so
read-only tools, pending-draft review, and later reports can distinguish
ChatGPT-created drafts from local/manual drafts.

## Later write surfaces

Do not add these until draft-only write access has real usage:

- Send/approve invoice.
- Record payment.
- Void invoice.
- Export and persist PDF.
- Mutate service agreements.

Each later write surface needs its own confirmation model. In particular,
payment and send actions should require stronger operator confirmation than
draft creation because they affect customer-facing or financial state.
