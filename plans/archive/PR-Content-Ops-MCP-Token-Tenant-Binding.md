# PR-Content-Ops-MCP-Token-Tenant-Binding

## Why this slice exists

#1390 completed the public OAuth e2e smoke for the verify-only Content Ops marketer MCP connector. The remaining #1353 rollout gap is tenant/account binding: OAuth-mode tool calls still rely on the configured account resolver at review time instead of deriving tenant scope from the authenticated OAuth token.

This slice closes that transport-hardening gap before broader dual-client connector smokes. It keeps bearer/direct mode unchanged, but OAuth mode must stamp the bound tenant onto issued token state and resolve the review-service tenant from the current access token.

Review follow-up: the launcher must validate the same account binding required by server startup, so this PR carries that operator-facing gate instead of leaving a dry-run/server-start mismatch.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. Add tenant binding metadata to the Content Ops marketer OAuth provider for access and refresh tokens.
2. In OAuth mode, resolve the review-service account from the current authenticated OAuth access token instead of the direct configured resolver.
3. Fail closed when OAuth mode is missing the account binding needed to mint tenant-bound tokens.
4. Preserve bearer/direct mode behavior for local and non-OAuth clients.
5. Require and report the account binding in the OAuth launcher dry-run path.
6. Prove token binding, refresh-token preservation, missing-token blocking, bearer-mode fallback, and launcher account validation with focused tests.

### Files touched

- `plans/PR-Content-Ops-MCP-Token-Tenant-Binding.md`
- `atlas_brain/mcp/content_ops_marketer_verify_oauth.py`
- `atlas_brain/mcp/content_ops_marketer_verify_server.py`
- `scripts/start_content_ops_marketer_verify_oauth_server.py`
- `tests/test_mcp_content_ops_marketer_verify.py`
- `tests/test_start_content_ops_marketer_verify_oauth_server.py`

### Review Contract

- Acceptance criteria:
  - [ ] OAuth-issued access tokens carry a tenant account binding.
  - [ ] Refresh-token exchange preserves the same tenant binding for new access tokens.
  - [ ] OAuth-mode review calls without an authenticated tenant-bound token block before registry reads.
  - [ ] OAuth mode fails fast if no account binding exists for token issuance.
  - [ ] The OAuth launcher rejects missing account binding before starting the server.
  - [ ] Bearer/direct mode continues to use the configured direct resolver.
- Affected surfaces: MCP / auth / tenant scope / scripts tests.
- Risk areas: security / tenant isolation / connector rollout / config.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R11, R12

## Mechanism

The Content Ops marketer OAuth provider records the configured account id when it mints an access token, stores the same account id for the refresh token, and restores refresh-token bindings from the OAuth state file. The provider exposes a small account lookup by access-token value.

The MCP server keeps the existing configured resolver for bearer mode. In OAuth mode it returns a token-bound resolver that reads the current MCP authenticated access token from the MCP auth context and asks the provider for the tenant binding. Missing token, missing provider binding, or missing configured account id all fail closed by producing no account id; the existing review service then blocks with `tenant scope required`.

The launcher mirrors the server startup contract by listing the account id in its required env report and rejecting empty values during dry-run validation.

## Intentional

- The source of the tenant account is still the operator-configured single-tenant binding at token issuance time. This PR changes the tool-call path to token-bound resolution; a multi-tenant account picker in the approval page is deferred.
- The tool payload still does not accept tenant ids.
- This does not add ChatGPT search/fetch adapters, dual-client live-client smokes, or `verify_draft` e2e tool calls.
- Cross-layer caller hints are same-method-name references in other OAuth providers/tests; this slice changes only the Content Ops marketer provider/server and its focused MCP tests cover the changed paths.

## Deferred

- `PR-Content-Ops-MCP-Dual-Client-Smoke`: run public OAuth flows against both Claude and the chosen ChatGPT connector surface.
- `PR-Content-Ops-MCP-Approval-Account-Picker`: optional future approval-page account selection if one process needs to approve different tenant accounts per client.
- `PR-Content-Ops-MCP-Connector-Boundary-Smoke`: optional bearer/auth boundary smoke if rollout needs a narrower regression check.
- Parked hardening: none.

## Verification

- Passed: python -m pytest tests/test_mcp_content_ops_marketer_verify.py -q
  - 21 passed in 0.28s
- Passed after review fix: python -m pytest tests/test_start_content_ops_marketer_verify_oauth_server.py tests/test_mcp_content_ops_marketer_verify.py -q
  - 38 passed in 0.33s
- Passed: python -m py_compile atlas_brain/mcp/content_ops_marketer_verify_oauth.py atlas_brain/mcp/content_ops_marketer_verify_server.py
- Passed after review fix: python -m py_compile scripts/start_content_ops_marketer_verify_oauth_server.py
- Passed: git diff --check
- Passed: bash scripts/run_extracted_pipeline_checks.sh
  - 295 passed in 1.69s for extracted_reasoning_core
  - 3433 passed, 10 skipped in 57.51s for extracted_content_pipeline
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content_ops_mcp_token_tenant_binding_pr_body.md
  - local PR review passed
- Passed after review fix: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content_ops_mcp_token_tenant_binding_pr_body.md
  - local PR review passed

## Estimated diff size

| Area | Actual LOC |
|---|---:|
| OAuth provider token binding | +109 |
| MCP server resolver switch | +34 / -2 |
| Focused tests | +203 |
| Launcher account-id gate | +3 |
| Plan doc | +95 |
| **Total** | **446** |

This may exceed the normal 400 LOC target after the launcher review fix. The overage is deliberate because the server validation and operator launcher validation are one contract: splitting them would leave a known dry-run/server-start mismatch in the reviewed PR.
