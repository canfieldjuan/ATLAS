# PR: Content Ops Deflection Readonly MCP

## Why this slice exists

PR #1250 defined the ChatGPT-facing FAQ deflection MCP connector contract:
read-only first, public `search` and `fetch` tools, tenant account binding before
any storage query, account-scoped report listing, and unpaid-safe opportunity
projection. This slice implements that first local server surface so the next
OAuth/CIMD slice has a concrete, tested tool boundary to protect. The diff is
expected to exceed the preferred 400 LOC budget because the MCP tool surface
would be unsafe or untestable if the store list method, account resolver,
projection helpers, docs/audit visibility, runner enrollment, and tenant
negative fixtures were split apart.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-mcp
Slice phase: Vertical slice

1. Add an account-scoped `list_reports` method to the deflection report store
   protocol, in-memory store, and Postgres adapter.
2. Add unpaid-safe deflection opportunity projection helpers from snapshots.
3. Add a read-only `atlas_brain.mcp.content_ops_deflection_readonly_server`
   exposing only `search` and `fetch`.
4. Add typed Atlas MCP config for the direct/test account binding and report
   URL base used before OAuth lands.
5. Add the minimal MCP inventory docs/audit allowlist entries needed for
   mechanical review to see the new server.
6. Add focused tests for exact tool surface, tenant binding failure, tenant
   isolation, safe projections, and store listing.

### Files touched

- `plans/PR-Content-Ops-Deflection-Readonly-MCP.md`
- `atlas_brain/config_defaults.py`
- `atlas_brain/config.py`
- `atlas_brain/mcp/content_ops_deflection_readonly_server.py`
- `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
- `extracted_content_pipeline/deflection_report_access.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `CLAUDE.md`
- `scripts/audit_claude_md_claims.py`
- `scripts/audit_mcp_tool_names_match_docs.py`
- `tests/test_audit_claude_md_claims.py`
- `tests/test_audit_mcp_tool_names_match_docs.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_mcp_content_ops_deflection_readonly.py`

## Mechanism

The server follows the invoicing-readonly lifecycle shape but keeps the public
tool contract to ChatGPT-compatible `search` and `fetch`.

`search(query, limit, paid)` resolves the bound account, calls
`DeflectionReportArtifactStore.list_reports(account_id, limit, paid)`, filters
the bound account's unpaid-safe snapshots by query text, and returns
`{results:[{id,title,url}]}` as structured content plus JSON text. When a query
is present, the store list call is unbounded and `limit` caps only returned
matches, so older matching reports are not hidden behind newer non-matches.

`fetch(id)` resolves the same bound account, loads only that tenant/request row,
and returns `{id,title,text,url,metadata}` as structured content plus JSON text.
The text and metadata include summary counts, top questions, structured content
opportunities, and unlock state. They do not expose answers, source IDs, source
evidence, markdown, nested FAQ item payloads, or full artifacts.

The account resolver is a small boundary used by every tool before storage is
called. For this direct/test slice it resolves from typed `settings.mcp`
configuration. Missing account binding returns a failure envelope and does not
call the store. HTTP auth also reads the typed `settings.mcp.auth_token` field
and wraps the streamable HTTP app with the validated token directly. OAuth
token-to-account binding is deferred, but this resolver is the seam the OAuth
slice will replace or extend.

The new store list method is account-scoped and ordered newest first. The
Postgres adapter uses the existing `(account_id, created_at DESC)` index on
`content_ops_deflection_reports`; the in-memory store mirrors the same tenant
filtering semantics for tests.

## Intentional

- No OAuth, CIMD, protected-resource metadata, Tailscale launcher, or public
  smoke scripts in this slice.
- The CLAUDE.md change is limited to mechanical MCP inventory visibility. Full
  OAuth, route, smoke, and ChatGPT setup runbook details remain deferred.
- Bearer/direct mode is only a local boundary path for this slice, not a
  ChatGPT or Claude connector auth path.
- The public tool names are only `search` and `fetch`. Custom deflection tools
  remain deferred to a separate Apps or direct-client surface.

## Deferred

- OAuth/CIMD account binding, protected-resource metadata, token-audience
  validation, and refresh-token persistence remain in the OAuth connector slice.
- Tailscale route checks, discovery/e2e smokes, and operator launcher remain in
  the public smoke slice.
- ChatGPT connector setup runbook and live validation remain in the handoff
  slice.
- Parked hardening: none.

## Verification

- py_compile for the changed server, store, projection, and config modules -- passed.
- Focused MCP/audit pytest command covering the read-only MCP server and MCP audit fixtures -- 38 passed.
- Focused deflection report pytest command -- 21 passed.
- Review/CI follow-up focused pytest command: `/home/juan-canfield/Desktop/Atlas/.venv/bin/python -m pytest tests/test_mcp_content_ops_deflection_readonly.py tests/test_content_ops_deflection_report.py -q` -- 37 passed.
- Review/CI follow-up extracted-safe pytest command: `pytest tests/test_content_ops_deflection_report.py -q` -- 24 passed.
- Review/CI follow-up MCP pytest command: `/home/juan-canfield/Desktop/Atlas/.venv/bin/python -m pytest tests/test_mcp_content_ops_deflection_readonly.py -q` -- 13 passed.
- Review/CI follow-up dedicated atlas workflow pytest command -- 79 passed.
- MCP claims, port assignment, and tool-name inventory audit commands -- passed.
- Extracted content pipeline validation command -- passed.
- Extracted reasoning import guard command for extracted_content_pipeline -- passed.
- Extracted standalone audit command -- passed.
- ASCII Python policy command -- passed.
- Focused extracted CI enrollment command -- OK: 143 matching tests are enrolled.
- Full extracted pipeline check runner -- passed on review/CI follow-up rerun: extracted reasoning core 295 passed; extracted content pipeline 2928 passed, 10 skipped.
- Local PR review command with the planned PR body file -- passed.

## Estimated diff size

Estimated: ~1170 LOC. This intentionally exceeds the preferred 400 LOC budget
because the read-only MCP server must ship with the account-scoped store list
method, resolver, projection helpers, MCP audit visibility, extracted runner
enrollment, and tenant negative fixtures to be reviewable as a safe vertical
slice.

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~95 |
| Store list method | ~95 |
| Snapshot projection helpers | ~70 |
| MCP server | ~450 |
| Tests | ~375 |
| Config/docs/audits/runner | ~170 |
| **Total** | **~1170** |
