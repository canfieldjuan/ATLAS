# PR-Content-Ops-MCP-OAuth-Discovery-Smoke

## Why this slice exists

#1387 added server-side OAuth mode for the verify-only marketer MCP server. The next #1353 delivery gap is public connector rollout verification. This slice starts that rollout with the lowest-risk smoke: a marketer-specific OAuth discovery checker that validates the public issuer metadata, protected-resource metadata, and unauthenticated MCP challenge before any approval, token exchange, or tool call happens.

The #1387 review also left a non-blocking DNS allow-list NIT for explicit default ports. This slice touches the same transport surface, so it closes that small restrictive-host edge here instead of carrying it forward.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. Add a marketer verify OAuth discovery smoke script for public metadata and 401 challenge validation.
2. Enroll focused checker tests proving happy path, missing config, and bad challenge failures.
3. Document the discovery smoke command in the marketer MCP section.
4. Fix the explicit default-port DNS allow-list NIT from #1387 and cover it with a focused test.

### Files touched

- `plans/PR-Content-Ops-MCP-OAuth-Discovery-Smoke.md`
- `CLAUDE.md`
- `atlas_brain/mcp/content_ops_marketer_verify_server.py`
- `scripts/check_content_ops_marketer_verify_oauth_discovery.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_check_content_ops_marketer_verify_oauth_discovery.py`
- `tests/test_mcp_content_ops_marketer_verify.py`

### Review Contract

- Acceptance criteria:
  - [ ] The new checker validates authorization metadata, protected-resource metadata, and unauthenticated MCP `WWW-Authenticate` resource metadata.
  - [ ] The checker refuses to run without issuer/resource URLs before touching network transport.
  - [ ] The checker uses the marketer verify scope `content_ops.review.verify`, not invoicing scopes.
  - [ ] Explicit default-port OAuth URLs allow both bare host and canonical host:port variants.
  - [ ] The existing MCP tool surface remains exactly `verify_draft`.
- Affected surfaces: MCP / auth / scripts / docs / CI.
- Risk areas: security / config / CI enrollment / deployment safety.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R11, R12

## Mechanism

The new checker reuses the existing OAuth metadata validation helpers from the draft-writer discovery checker but supplies content-ops-specific environment names, default scope, operator messages, and output. It does not complete OAuth approval or call any MCP tools; it only proves the public discovery documents and unauthenticated challenge are routable and internally consistent.

The DNS allow-list fix keeps the #1387 behavior for implicit-port URLs and adds the bare hostname when an operator explicitly configures the scheme default port, such as `https://example.com:443`.

## Intentional

- This is discovery smoke only; it does not add the operator launcher, Funnel route checker, OAuth e2e token exchange, or dual-client connector smoke.
- The script validates live URLs when run by an operator, but tests mock network helpers and prove the checker branches without external calls.
- The script reuses existing metadata helper behavior instead of refactoring the invoicing checker into a shared library in this slice.

## Deferred

- `PR-Content-Ops-MCP-OAuth-Launcher`: operator launcher that forces OAuth mode, masks secrets, and prints Funnel/discovery/e2e commands.
- `PR-Content-Ops-MCP-OAuth-E2E`: dynamic registration, approval, token exchange, and list-tools smoke.
- `PR-Content-Ops-MCP-Token-Tenant-Binding`: derive tenant account binding from OAuth token/client state.
- Parked hardening: none.

## Verification

- python -m pytest tests/test_check_content_ops_marketer_verify_oauth_discovery.py tests/test_mcp_content_ops_marketer_verify.py -q -- 20 passed.
- python -m pytest tests/test_atlas_content_ops_review_workflow.py tests/test_check_content_ops_marketer_verify_oauth_discovery.py tests/test_mcp_content_ops_marketer_verify.py tests/test_content_ops_claim_registry.py -q -- 64 passed.
- python -m py_compile scripts/check_content_ops_marketer_verify_oauth_discovery.py atlas_brain/mcp/content_ops_marketer_verify_server.py -- passed.
- python scripts/check_content_ops_marketer_verify_oauth_discovery.py --help -- passed.
- git diff --check -- passed.
- bash scripts/run_extracted_pipeline_checks.sh -- 3374 passed, 10 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content_ops_mcp_oauth_discovery_smoke_pr_body.md -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Discovery checker | ~116 |
| Checker tests | ~154 |
| DNS NIT + test | ~14 |
| Docs and CI enrollment | ~6 |
| Plan doc | ~79 |
| **Total** | **~369** |
