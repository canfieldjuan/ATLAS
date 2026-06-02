# PR: Content Ops Deflection MCP Plan

## Why this slice exists

The invoicing read-only MCP connector proved that Atlas can expose a protected
ChatGPT-facing MCP server online. FAQ deflection reports are the next strongest
candidate because they connect product visibility to a revenue path: ChatGPT can
surface the customer's top support deflection opportunities while the full report
and future generation actions remain paid-gated. Before implementation starts,
the connector needs a written contract because this surface is multi-tenant and
the existing deflection report store is not yet listable.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-mcp
Slice phase: Workflow/process

1. Define the first read-only FAQ deflection MCP connector surface.
2. Name the required store-contract gap for listing reports by tenant.
3. Make tenant identity to account binding the first security invariant.
4. Define the safe content-opportunity projection boundary.
5. Sequence the implementation, OAuth, smoke, and ChatGPT handoff follow-ups.

### Files touched

- `plans/PR-Content-Ops-Deflection-MCP-Plan.md`

## Mechanism

The next implementation PR should follow the proven invoicing read-only MCP
connector pattern for server lifecycle, OAuth approval, launcher, Tailscale
routing, and smoke-test discipline, but it is not a pure mirror. This connector
targets the ChatGPT data-only app / company knowledge / deep-research surface,
so the public ChatGPT-facing MCP tool contract is `search` and `fetch`, not the
custom-tool surface used by invoicing-readonly. FAQ deflection reports are keyed
by account and request, so every tool must resolve exactly one account before it
queries storage. Tool calls must never accept account identifiers from ChatGPT
arguments.

Initial public read-only tools:

- search(query): search the bound account's free deflection report snapshots and
  return `{results:[{id,title,url}]}` as both structured content and JSON-encoded
  text for ChatGPT compatibility.
- fetch(id): fetch one unpaid-safe deflection report document by search result
  ID or request ID and return `{id,title,text,url,metadata}` as both structured
  content and JSON-encoded text.

Internal projections behind `fetch`:

- report list summary for the bound account only.
- deflection report snapshot for one request.
- top deflection questions.
- structured unpaid-safe content opportunities.
- unlock status that exposes lock/payment state only.

The implementation PR must add a report-listing method to the deflection report
store protocol, Postgres adapter, and in-memory test store. The database table is
already indexed for account-scoped listing, but the Python store boundary only
has save, get snapshot, get artifact record, and mark paid today.

The implementation PR must also add an account resolver boundary before OAuth
work starts. In direct or test mode, the account resolver may use typed Atlas
configuration. In OAuth mode, the approval flow must bind the authenticated
connector identity to one account and persist that binding with the OAuth client
or token state. Missing, ambiguous, or malformed account binding must return a
failure envelope and must not call the store. Bearer mode is only for local
boundary smokes and direct clients; user-pasted static bearer tokens and
credentials in connector URLs are not a ChatGPT or Claude connector path.

Content opportunities must be a structured projection, not markdown scraping.
If the current snapshot shape is too thin, the implementation PR should add a
small unpaid-safe accessor or extend the snapshot with non-sensitive fields such
as rank, question, weighted frequency, customer wording, opportunity score,
answer-status category, recommended content action, and unlock hint. The
projection must not expose answers, source IDs, source evidence, markdown,
nested FAQ item payloads, or full artifacts.

OAuth must be a conscious connector decision, not copied blindly from
invoicing-readonly. The public connector should prefer Client ID Metadata
Documents (CIMD) over Dynamic Client Registration (DCR). The authorization
server metadata must advertise client ID metadata support and allow public-client
token exchange with `none` in `token_endpoint_auth_methods_supported`. DCR may
remain only as a bounded direct-client or temporary compatibility fallback.

Protected-resource and authorization metadata must name the OAuth invariants the
connector depends on: S256 PKCE support, a protected-resource `resource` value
that exactly matches the public MCP URL including its Tailscale path prefix,
token-audience validation for that resource, rejection of tokens minted for a
different audience, and no upstream bearer-token passthrough.

Follow-up sequence:

1. Read-only MCP implementation: store list method, account resolver, safe
   opportunity projection, public `search` / `fetch` tools, and exact tool-list
   boundary tests.
2. OAuth connector implementation: approval page, CIMD-first client identity,
   token verification, account binding persistence, and protected-resource
   metadata.
3. Public smoke implementation: discovery smoke, OAuth e2e smoke, Tailscale
   route checker, and operator launcher.
4. ChatGPT handoff: connector setup runbook and live validation against the
   public route.

## Intentional

- This PR is plan-only so the multi-tenant security contract is reviewed before
  code lands.
- The first MCP surface is read-only. Generation, publishing, checkout, and
  paid-unlock mutation tools remain out of scope.
- The first implementation must not copy the operator-scoped invoicing
  assumptions without adding account binding.
- The public ChatGPT data/knowledge surface uses `search` and `fetch`; request
  IDs remain the internal report identifiers behind `fetch`.
- Custom named tools can be reconsidered only for a separate ChatGPT Apps or
  direct-client surface.

## Deferred

- The read-only MCP server implementation is deferred to the next slice.
- OAuth, CIMD support, Tailscale route checks, public smokes, and ChatGPT
  connector handoff are separate follow-up slices after the read-only contract
  lands.
- Paid generation actions such as generate FAQ package, generate landing page,
  or generate blog post from a report remain later revenue-action slices.
- Parked hardening: none.

## Verification

- `bash scripts/local_pr_review.sh --current-pr-body-file <body-file>` -- passed.
- Current OpenAI MCP and Claude connector auth docs were checked before the
  review-response edit.

## Estimated diff size

Estimated: ~140 LOC. This is intentionally a small workflow/process slice whose
only tracked artifact is the implementation plan.

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~140 |
| **Total** | **~140** |
