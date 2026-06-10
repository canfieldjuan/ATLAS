# PR - Content-Ops Tenant Binding Bridge

## Why this slice exists

#1377 made the review service tool-shaped, but future marketer MCP callers still
need a safe way to turn connector tenant binding into the `TenantScope` the
service and claim-registry reader already require. #1353 identifies this as the
remaining service seam before MCP transport: FastAPI routes use a request
ContextVar, while MCP precedents expose an account resolver. This slice bridges
that resolver shape into the review service without starting the MCP server.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

Build the tenant-binding bridge for the review service:

1. Add a small account-resolver protocol at the review-service boundary.
2. Add a wrapper that resolves the bound account, builds a `TenantScope`, and
   delegates to the existing review service.
3. Fail closed before registry reads when the binding is missing, malformed, or
   the resolver fails.
4. Preserve the direct `scope=` service path for FastAPI callers and tests.

### Review Contract

- Acceptance criteria:
  - [x] A bound account resolver produces a `TenantScope` passed to the
        registry reader.
  - [x] Whitespace around the resolved account is trimmed before building scope.
  - [x] Missing, blank, or non-string account binding blocks before registry
        reads.
  - [x] Resolver exceptions block before registry reads.
  - [x] Existing direct-scope review service behavior remains unchanged.
  - [x] No MCP transport, OAuth server, DB migration, LLM behavior, or
        generated-asset mutation is added.
- Affected surfaces: host review service, CI-covered service tests.
- Risk areas: tenant isolation, fail-closed behavior, backcompat.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R12.

### Files touched

- `atlas_brain/_content_ops_review_workflow.py`
- `tests/test_atlas_content_ops_review_workflow.py`
- `plans/PR-Content-Ops-Tenant-Binding-Bridge.md`

## Mechanism

The review workflow module gains a protocol for the deflection-style account
resolver boundary and a wrapper that accepts a request, resolver, and registry
reader. The wrapper resolves the account, converts a non-empty string into
`TenantScope`, then delegates to the existing service path. If binding is absent
or resolution raises, the wrapper returns the same blocked result shape used by
the existing service and never calls the registry reader.

The existing direct `scope=` entry point remains unchanged so FastAPI route
callers keep using the already-wired ContextVar bridge.

## Intentional

- This is not the marketer MCP server and does not add OAuth routes, tokens, or
  tool registration.
- This does not change the claim-registry repository; it only ensures future
  connector callers hand that repository a tenant scope.
- The bridge accepts only account binding from an injected resolver, not an
  account id from a draft/tool payload.

## Deferred

- `PR-Marketer-Verification-MCP`: expose verify-only tools after tenant binding
  is available as service plumbing.
- `PR-Content-Ops-MCP-OAuth-Transport`: settle dual-client connector transport
  and OAuth token-to-tenant isolation.
- Parked hardening: none expected.

## Verification

- Command: python -m pytest tests/test_atlas_content_ops_review_workflow.py -q
  - Passed: 24 tests.
- Command: python -m pytest tests/test_atlas_content_ops_review_workflow.py tests/test_atlas_content_ops_scope.py tests/test_content_ops_claim_registry.py -q
  - Passed: 50 tests.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py
  - Passed: 156 matching tests are enrolled.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - Passed: mapped files match Atlas sources; hard Atlas imports clean.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Passed: clean.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Passed: Atlas runtime import findings 0.
- Command: bash scripts/check_ascii_python.sh
  - Passed: extracted content pipeline Python files are ASCII.
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - Passed: 3345 tests, 10 skipped.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content_ops_tenant_binding_bridge_pr_body.md
  - Passed: local PR review passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/_content_ops_review_workflow.py` | 36 |
| `tests/test_atlas_content_ops_review_workflow.py` | 76 |
| `plans/PR-Content-Ops-Tenant-Binding-Bridge.md` | 105 |
| **Total** | **217** |
