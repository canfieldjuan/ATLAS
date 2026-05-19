# PR-Invoicing-MCP-Write-Access-Guardrails

## Why this slice exists

The read-only invoicing MCP connector now works in ChatGPT through OAuth. The next step is write access, but the existing full `atlas-invoicing` MCP server mixes draft creation with high-risk operations: sending invoices, recording payments, voiding invoices, mutating services, and exporting PDFs to disk.

This slice creates the guardrail contract before any write-capable ChatGPT connector is exposed. The goal is to make the safe path explicit: write access starts with a separate draft-only connector, not by publishing the full invoicing MCP server.

## Scope (this PR)

1. Document the allowed first write surface for invoicing MCP access.
2. Document the denied tool classes that must stay out of the first write connector.
3. Define the required OAuth, audit, idempotency, and verification gates for the later implementation PR.
4. Cross-link the new guardrail doc from the existing ChatGPT OAuth rollout runbook.

### Files touched

- `docs/INVOICING_MCP_WRITE_ACCESS_GUARDRAILS.md`
- `docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md`
- `plans/PR-Invoicing-MCP-Write-Access-Guardrails.md`

## Mechanism

The new guardrail document classifies the current full invoicing MCP tools into:

- Safe reads already covered by `atlas-invoicing-readonly`.
- Candidate draft-only writes for a future `atlas-invoicing-draft-writer` server.
- Explicitly denied operations for the first write rollout.

It also defines the minimum implementation gates the future code PR must ship: separate server module, separate OAuth scope/env prefix, exact tool allowlist smoke, no send/payment/void/PDF/service tools, idempotency key for draft creation, metadata/audit tagging, and tests that assert the connector cannot expose denied tools.

## Intentional

- No production write tool is added in this slice. The risk is high enough that the contract should land before implementation.
- The first write surface is draft-only. Sending invoices, recording payments, voiding invoices, PDF export, and service mutation remain out of scope.
- The write connector is specified as a separate server instead of adding flags to the full invoicing server. Separate process/tool surface is easier to audit and harder to misconfigure.

## Deferred

- `PR-Invoicing-Draft-Writer-MCP`: add the first draft-only write MCP server and boundary smoke.
- `PR-Invoicing-Draft-Writer-OAuth`: add public OAuth discovery/e2e/launcher for that server if it is not bundled with the server PR.
- Any send/payment/void tools are deferred until draft-only write access has real usage and a stronger confirmation/audit flow.

## Verification

- `git diff --check`
- `bash scripts/local_pr_review.sh --allow-dirty`

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `docs/INVOICING_MCP_WRITE_ACCESS_GUARDRAILS.md` | ~184 |
| `docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md` | ~6 |
| `plans/PR-Invoicing-MCP-Write-Access-Guardrails.md` | ~58 |
| **Total** | **~248** |
