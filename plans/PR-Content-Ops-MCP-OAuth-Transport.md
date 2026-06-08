# PR-Content-Ops-MCP-OAuth-Transport

## Why this slice exists

#1381 landed the verify-only marketer MCP shell, but it still exposes HTTP through bearer mode only. #1353 calls out secure connector transport as the next delivery-layer gap before marketer clients can use the workflow remotely. This slice lands the server-side OAuth mode and refusal gates first, so the later public rollout scripts can stay thin and test the already-existing server contract.

The full dual-client public rollout is too large for one safe PR. This slice deliberately stops before public Tailscale Funnel scripts and live ChatGPT/Claude e2e smokes; it creates the authenticated server mode those follow-up scripts will exercise.

This is slightly over the normal 400 LOC target because the server-side OAuth mode has to carry typed config, operator-facing docs, the OAuth helper, the server auth switch, and focused tests that stub MCP auth surfaces for extracted CI.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. Add content-specific OAuth provider wiring for the marketer verify MCP server with a verify-only scope and operator approval page.
2. Add typed MCP config fields for marketer verify OAuth mode, issuer URL, resource URL, approval token, and optional state file.
3. Switch the server HTTP path between bearer mode and OAuth mode without changing the single `verify_draft` tool surface.
4. Keep OAuth mode fail-closed on missing/short approval config and keep DNS rebinding protection enabled for configured public hosts.

### Files touched

- `plans/PR-Content-Ops-MCP-OAuth-Transport.md`
- `CLAUDE.md`
- `atlas_brain/config.py`
- `atlas_brain/mcp/content_ops_marketer_verify_oauth.py`
- `atlas_brain/mcp/content_ops_marketer_verify_server.py`
- `tests/test_mcp_content_ops_marketer_verify.py`

### Review Contract

- Acceptance criteria:
  - [ ] Bearer mode remains the default and still requires a production-shaped bearer token for HTTP.
  - [ ] OAuth mode installs FastMCP auth settings with the content-ops verify scope and dynamic client registration enabled.
  - [ ] OAuth mode refuses to start without issuer URL, resource URL, and a non-short approval token.
  - [ ] OAuth transport security keeps DNS rebinding protection enabled and allows only localhost plus configured public OAuth/resource hosts.
  - [ ] The MCP tool surface remains exactly `verify_draft`; no generation, publishing, approval, registry mutation, search, or fetch tools are added.
- Affected surfaces: MCP / auth / config / tests.
- Risk areas: security / tenant isolation / backcompat / config / CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R11, R12

## Mechanism

The server gets a content-specific OAuth helper that reuses the existing Atlas authorization-code provider shape and narrows it to a verify-only content scope. The MCP server reads typed settings from `settings.mcp`, validates OAuth settings before serving HTTP, installs FastMCP auth settings and token verification in OAuth mode, and keeps the previous bearer middleware path as the default.

OAuth transport security is generated from the configured issuer/resource URLs. Localhost remains allowed for development, while public hosts must come from the configured URLs. The server keeps the account resolver and review-service delegation from #1381 unchanged, so this slice adds transport authentication without changing verdict behavior.

## Intentional

- This is the first server-side OAuth transport slice, not the full public connector rollout.
- This does not add ChatGPT search/fetch adapters, public Funnel route scripts, live OAuth e2e scripts, or dual-client smoke commands.
- This does not add server-side ContentPR sessions, claims extraction, registry mutation tools, or generation tools.
- Token-bound multi-tenant account derivation remains deferred; this slice preserves the current one-bound-tenant server model and makes remote transport refuse unsafe OAuth configuration.

## Deferred

- `PR-Content-Ops-MCP-OAuth-Smokes`: operator launcher, OAuth discovery smoke, public Funnel route smoke, and OAuth e2e smoke for the marketer verify server.
- `PR-Content-Ops-MCP-Token-Tenant-Binding`: derive the tenant account binding from OAuth token/client state instead of the current one-bound-tenant server config.
- `PR-Content-Ops-MCP-ChatGPT-Adapter`: decide and implement ChatGPT search/fetch compatibility if the rich tool connector path is not sufficient.
- Parked hardening: none.

## Verification

- python -m pytest tests/test_mcp_content_ops_marketer_verify.py -q -- 14 passed.
- python -m pytest tests/test_atlas_content_ops_review_workflow.py tests/test_mcp_content_ops_marketer_verify.py tests/test_content_ops_claim_registry.py -q -- 58 passed.
- python -m py_compile atlas_brain/mcp/content_ops_marketer_verify_server.py atlas_brain/mcp/content_ops_marketer_verify_oauth.py -- passed.
- git diff --check -- passed.
- bash scripts/run_extracted_pipeline_checks.sh -- 3368 passed, 10 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content_ops_mcp_oauth_transport_pr_body.md -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| OAuth helper | ~117 |
| Server auth-mode switch | ~124 |
| Typed config and docs | ~44 |
| Focused tests | ~152 |
| Plan doc | ~75 |
| **Total** | **~512** |
