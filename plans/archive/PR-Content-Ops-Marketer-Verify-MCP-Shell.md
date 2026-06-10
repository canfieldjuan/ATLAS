# PR-Content-Ops-Marketer-Verify-MCP-Shell

## Why this slice exists

#1380 closed the tenant-binding bridge, so the remaining #1353 delivery gap can move from service plumbing to the first transport surface. This slice gives marketer clients one verify-only MCP entry point that delegates to the existing content-ops review service. It does not add generation, OAuth rollout, multi-step state, or content persistence.

This is intentionally over the normal 400 LOC target because a new MCP server cannot land as only a Python module in this repo. The slice must carry the server, focused tool tests, typed config, CLAUDE inventory, MCP audit maps, CI enrollment, and audit fixture updates together or the mechanical gates drift.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

1. Add a new Content Ops Marketer Verify MCP server with one tool, `verify_draft`.
2. Convert decoded tool input into `ContentOpsReviewRequest` using fail-closed, non-raising coercion for malformed decoded values.
3. Resolve the configured tenant account through an injected resolver and delegate to the existing tenant-bound review service.
4. Enroll the new MCP surface in local docs/audits and CI checks.

### Files touched

- `plans/PR-Content-Ops-Marketer-Verify-MCP-Shell.md`
- `atlas_brain/mcp/content_ops_marketer_verify_server.py`
- `atlas_brain/mcp/__init__.py`
- `tests/test_mcp_content_ops_marketer_verify.py`
- `tests/test_audit_mcp_tool_names_match_docs.py`
- `tests/test_pre_push_audit.py`
- `atlas_brain/config_defaults.py`
- `atlas_brain/config.py`
- `CLAUDE.md`
- `scripts/audit_claude_md_claims.py`
- `scripts/audit_mcp_tool_names_match_docs.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`
- Audit fixture tests if the MCP server inventory maps require fixture updates.

### Review Contract

- Acceptance criteria:
  - [ ] The MCP server exposes exactly `verify_draft` and no generate, publish, registry-mutation, unlock, or approval tools.
  - [ ] Missing tenant binding blocks before registry reads.
  - [ ] Valid decoded input delegates to the tenant-bound review service and returns the service result envelope.
  - [ ] Malformed decoded rows count as missing or unresolved evidence, never as silent approval and never as an exception.
  - [ ] MCP docs, tool-name audits, port audits, and CI enrollment know about the new server.
- Affected surfaces: MCP / config / docs / CI.
- Risk areas: security / tenant isolation / backcompat / config / CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R11, R12

## Mechanism

The new server follows the Content Ops deflection precedent for FastMCP setup, bearer-token HTTP wrapping, test overrides, and configured direct account binding. Its single tool builds a review request from primitive MCP arguments, then calls the tenant-bound review workflow service with the configured account resolver and Postgres claim-registry reader.

Input coercion is deliberately conservative: missing strings stay empty, unknown coverage status becomes unresolved, malformed lists become empty tuples, malformed dates are ignored, and malformed coverage rows become synthetic unresolved required rows. Other unsupported decoded row shapes stay non-raising and empty, but required coverage cannot silently disappear into an approval.

## Intentional

- The slice does not build OAuth, dynamic client registration, public funnel route smokes, or dual-client OAuth e2e checks. #1353 keeps that as the rollout/hardening phase after the first tool surface exists.
- The slice does not create ContentPR session persistence. v1 remains stateless: the marketer client sends the structured draft evidence on each verification call.
- The slice requires extracted claims as structured input. Claims extraction from prose remains deferred LLM or client-owned work.
- The tool verifies the marketer's own draft evidence; it does not generate copy or mutate registry rows.

## Deferred

- `PR-Content-Ops-MCP-OAuth-Transport`: OAuth, dynamic client registration, public HTTPS rollout, dual-client smoke checks, and token-bound per-tenant isolation.
- `PR-Marketer-Verification-MCP-State`: server-side ContentPR/session persistence if stateless client round-trips prove brittle.
- `PR-Content-Ops-Claims-Extraction`: deterministic matcher or LLM-owned extraction for prose-only drafts.
- Parked hardening: none.

## Verification

- python -m pytest tests/test_mcp_content_ops_marketer_verify.py -q -- 10 passed.
- python -m pytest tests/test_audit_mcp_tool_names_match_docs.py tests/test_audit_claude_md_claims.py tests/test_pre_push_audit.py -q -- 20 passed.
- python -m pytest tests/test_atlas_content_ops_review_workflow.py tests/test_mcp_content_ops_marketer_verify.py tests/test_content_ops_claim_registry.py -q -- 54 passed.
- python scripts/audit_claude_md_claims.py -- passed.
- python scripts/audit_mcp_tool_names_match_docs.py -- passed.
- python scripts/audit_mcp_port_assignments.py -- passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main -- OK, 156 matching tests are enrolled.
- python -m py_compile atlas_brain/mcp/content_ops_marketer_verify_server.py -- passed.
- bash scripts/check_ascii_python.sh -- passed.
- bash scripts/run_extracted_pipeline_checks.sh -- 3364 passed, 10 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content_ops_marketer_verify_mcp_shell_pr_body.md -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| MCP server adapter | ~359 |
| Focused MCP tests | ~258 |
| Docs, config, audits, CI enrollment | ~95 |
| Plan doc | ~90 |
| **Total** | **~802** |
