# PR-Content-Ops-MCP-ChatGPT-Search-Fetch-Adapter

## Why this slice exists

#1398 pinned the dual-client contract from #1353: Claude rich uses the current `verify_draft` tool surface, while the proven ChatGPT connector profile requires a separate search/fetch surface. That profile now exists in the OAuth e2e checker, but no server surface satisfies it.

This slice builds the deterministic adapter surface first. It keeps the existing Claude-rich verifier unchanged and adds a separate ChatGPT-shaped MCP module with only search and fetch tools. The adapter delegates verification to the same tenant-bound review backend and stores only an in-process verdict handoff for the follow-up fetch.

This PR is expected to exceed the 400 LOC target because the adapter surface and the tenant-isolation tests are not safely separable: without the tests, the first ChatGPT-shaped verifier would have an unproven cross-tenant fetch boundary.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

1. Add a separate Content Ops marketer ChatGPT adapter MCP module exposing exactly search and fetch.
2. Map a JSON review request submitted through search into the existing tenant-bound verify backend.
3. Cache the resulting verdict by tenant-bound result ID so fetch can return the full verdict document without exposing another tenant's result.
4. Return a static adapter contract document when search receives no JSON review request.
5. Enroll adapter tests in the extracted pipeline wrapper and dedicated Content Ops review workflow.

### Files touched

- `plans/PR-Content-Ops-MCP-ChatGPT-Search-Fetch-Adapter.md`
- `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py`
- `tests/test_mcp_content_ops_marketer_verify.py`
- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`

### Review Contract

- Acceptance criteria:
  - [ ] The adapter exposes exactly search and fetch, and does not expose `verify_draft`, generation, publishing, checkout, or registry mutation tools.
  - [ ] A JSON review request submitted through search delegates to the existing review backend for the bound tenant.
  - [ ] A successful search returns a result ID that fetch can use to return the full verdict document.
  - [ ] Fetch fails closed for missing account binding, unknown IDs, and cached results owned by another tenant.
  - [ ] The standalone adapter server reuses the verifier server lifespan so registry reads run with the same database initialization path.
  - [ ] Non-JSON search input returns the adapter contract document instead of running verification.
  - [ ] Existing Claude-rich `verify_draft` tests and tool-surface contract remain unchanged.
  - [ ] The new adapter tests run in the extracted pipeline wrapper and dedicated Content Ops review workflow.
- Affected surfaces: MCP / tenant isolation / review workflow adapter / tests / CI.
- Risk areas: security / public tool-surface compatibility / tenant isolation / maintainability / CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R12

## Mechanism

The new module owns a separate FastMCP instance and imports the existing verifier module for request coercion, account resolution, registry-reader resolution, review-service delegation, and database lifespan management. Search first checks whether the query is a decoded JSON object. If it is not, search returns one static contract result that describes the adapter input shape. If it is JSON, the adapter resolves the tenant, builds the same review request the rich verifier uses, runs the review service, writes the verdict payload to an in-memory cache keyed by a deterministic result ID, and returns that result in ChatGPT search shape.

Fetch resolves the tenant again before reading the cache. Missing account binding, unknown IDs, and account mismatches return non-success metadata rather than a verdict. The cache is intentionally process-local in this slice; durable verdict/session persistence remains a follow-up.

## Intentional

- This is a separate adapter surface, not extra tools on the Claude-rich verifier. The existing verifier remains a one-tool server.
- This PR does not add public OAuth launcher/discovery wiring for the adapter. It proves the tool surface and backend mapping first.
- The process-local cache is enough for the adapter's first two-call vertical path and is explicitly not a durable verdict ledger.
- The adapter accepts JSON in search because ChatGPT's proven connector shape gives us search and fetch only.

## Deferred

- `PR-Content-Ops-MCP-ChatGPT-Adapter-OAuth-Rollout`: add public OAuth/launcher/discovery/e2e wiring for the adapter server.
- `PR-Content-Ops-MCP-Verdict-Session-Persistence`: replace process-local verdict handoff with tenant-scoped durable storage if live connector use needs restart-safe fetch.
- `PR-Content-Ops-MCP-Live-Dual-Client-Rollout`: run public OAuth smokes against real Claude and ChatGPT connector registrations after both public surfaces exist.
- Parked hardening: none.

## Verification

- Passed: python -m pytest tests/test_mcp_content_ops_marketer_verify.py -q
  - 28 passed in 0.24s
- Passed: python -m py_compile atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py
- Passed: git diff --check
- Passed: python -m pytest tests/test_atlas_content_ops_review_workflow.py tests/test_check_content_ops_marketer_verify_oauth_e2e.py tests/test_content_ops_marketer_verify_launcher_contract.py tests/test_mcp_content_ops_marketer_verify.py -q
  - 73 passed in 0.66s
- Passed: bash scripts/run_extracted_pipeline_checks.sh
  - 295 passed in 1.68s for extracted_reasoning_core
  - 3481 passed, 10 skipped in 57.51s for extracted_content_pipeline
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_mcp_chatgpt_adapter_pr_body.md
  - local PR review passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 88 |
| Adapter server | 246 |
| Adapter tests | 129 |
| CI enrollment | 2 |
| **Total** | **465** |

This is over the normal 400 LOC target for the reason named in Why this slice exists.
