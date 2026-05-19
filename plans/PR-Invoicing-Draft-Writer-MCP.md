# PR-Invoicing-Draft-Writer-MCP

## Why this slice exists

`docs/INVOICING_MCP_WRITE_ACCESS_GUARDRAILS.md` defines the first safe write
surface for ChatGPT invoicing access: draft-only invoice creation/update in a
separate server. The existing full `atlas-invoicing` MCP server is too broad
for remote write access because it also exposes send, payment, void, service,
and PDF/file side effects.

This slice implements the local draft-writer MCP boundary without public OAuth
rollout. It gives us real code and tests for the safe write surface before
adding ChatGPT-facing OAuth wiring.

## Scope (this PR)

1. Add a separate `atlas-invoicing-draft-writer` MCP server.
2. Expose exactly four tools: create draft, update draft, get invoice, and
   list pending drafts.
3. Implement `create_draft_invoice` without CRM, email, send, payment, void,
   PDF, or service side effects.
4. Add a local connector boundary smoke for exact tool-surface validation.
5. Add unit tests for tool allowlist, idempotent draft creation, and boundary
   smoke failures.

### Files touched

- `atlas_brain/mcp/invoicing_draft_writer_server.py`
- `scripts/check_invoicing_draft_writer_mcp_connector.py`
- `tests/test_invoicing_draft_writer_mcp.py`
- `tests/test_check_invoicing_draft_writer_mcp_connector.py`
- `plans/PR-Invoicing-Draft-Writer-MCP.md`

## Mechanism

The new server registers its own `FastMCP("atlas-invoicing-draft-writer")`
instance and exposes only the guardrail-approved tools.

`create_draft_invoice` parses and validates line items, derives a stable
`source_ref` from the required idempotency key, returns the existing invoice on
retry, and otherwise calls the invoice repository directly with explicit
connector metadata. It does not delegate to `create_invoice` because that full
MCP tool can resolve CRM contacts and log CRM interactions.

`update_draft_invoice` delegates to the existing full-server `update_invoice`
tool because that path already rejects non-draft invoices and has no external
send/payment/file side effects.

The boundary smoke mirrors the read-only connector check: unauthenticated HTTP
must be rejected, authenticated `list_tools` must equal the exact allowlist,
and known denied mutating tools must be absent.

## Intentional

- OAuth mode is deferred. This PR proves the write tool boundary locally first;
  public ChatGPT exposure comes after the exact local surface is pinned.
- The server does not include broader read-only tools such as `list_invoices`,
  `customer_balance`, or `payment_history`. The existing read-only connector
  covers that surface.
- `create_draft_invoice` uses direct repository creation instead of the full
  MCP `create_invoice` wrapper to avoid CRM side effects.

## Deferred

- `PR-Invoicing-Draft-Writer-OAuth`: add OAuth mode, discovery smoke, e2e
  smoke, operator launcher, port docs, and ChatGPT connector setup.
- Send/approve, payment, void, PDF export, and service mutation remain
  deferred until draft-only write access is proven.

## Verification

- .venv/bin/python -m py_compile atlas_brain/mcp/invoicing_draft_writer_server.py scripts/check_invoicing_draft_writer_mcp_connector.py
- .venv/bin/pytest tests/test_invoicing_draft_writer_mcp.py tests/test_check_invoicing_draft_writer_mcp_connector.py
- git diff --check
- bash scripts/local_pr_review.sh

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `atlas_brain/mcp/invoicing_draft_writer_server.py` | ~340 |
| `scripts/check_invoicing_draft_writer_mcp_connector.py` | ~139 |
| `tests/test_invoicing_draft_writer_mcp.py` | ~270 |
| `tests/test_check_invoicing_draft_writer_mcp_connector.py` | ~90 |
| `plans/PR-Invoicing-Draft-Writer-MCP.md` | ~91 |
| **Total** | **~930** |

This intentionally exceeds the soft 400 LOC budget because the safe slice needs
the server, exact boundary smoke, and regression tests together. Splitting the
smoke/tests away from the server would temporarily create unpinned write
surface code.
