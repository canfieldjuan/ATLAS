# PR-Content-Ops-MCP-OAuth-E2E

## Why this slice exists

#1387 added the verify-only marketer MCP OAuth transport, #1388 added public discovery validation, and #1389 added the operator launcher. The #1353 delivery tracker now calls for the no-mutation OAuth e2e smoke: prove the public OAuth flow can dynamically register a client, route through the operator approval gate, exchange an authorization code for a token, and use that token to list the MCP tool surface.

This slice is production hardening for connector rollout. It deliberately lists tools only; it must not call `verify_draft` or submit draft content.

This PR exceeds the normal 400 LOC target because the new checker is security/config validation around OAuth and connector exposure. Keeping the checker, command documentation, CI enrollment, and focused failure-branch fixtures together is the smallest defensible slice that proves the e2e smoke fails closed before rollout.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. Add a Content Ops marketer verify OAuth e2e smoke script for public connector rollout.
2. Reuse the existing draft-writer OAuth e2e transport helpers while supplying content-ops env vars, scope, client name, and exact tool surface.
3. Prove config missing values, approval token file handling, token exchange failure, tool-surface failures, and the registration to list-tools sequence with focused tests.
4. Enroll the new e2e tests in the extracted pipeline wrapper.
5. Document the e2e smoke command in the Content Ops Marketer Verify MCP section.
6. Propagate the required approval-token handoff into the launcher-printed e2e command so the operator runbook matches the checker interface.

### Files touched

- `plans/PR-Content-Ops-MCP-OAuth-E2E.md`
- `CLAUDE.md`
- `scripts/check_content_ops_marketer_verify_oauth_e2e.py`
- `scripts/start_content_ops_marketer_verify_oauth_server.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_check_content_ops_marketer_verify_oauth_e2e.py`
- `tests/test_start_content_ops_marketer_verify_oauth_server.py`

### Review Contract

- Acceptance criteria:
  - [ ] The checker uses content-ops env vars and the `content_ops.review.verify` scope.
  - [ ] The checker refuses missing issuer/resource/approval-token values before network transport.
  - [ ] The checker reads an approval token from a local file without printing the secret.
  - [ ] The checker performs the OAuth registration, authorization, approval, token exchange, and authenticated list-tools sequence.
  - [ ] The checker accepts exactly the verify-only tool surface `verify_draft`.
  - [ ] The checker does not call `verify_draft` or any draft/content mutation path.
  - [ ] The launcher-printed e2e command includes the required approval-token handoff.
- Affected surfaces: MCP / auth / scripts / docs / CI.
- Risk areas: security / config / deployment safety / CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R11, R12

## Mechanism

The new checker loads the existing draft-writer OAuth e2e module by file path and reuses its generic transport helpers for client registration, PKCE authorization, approval, token exchange, and authenticated MCP tool listing. The Content Ops wrapper owns only the content-specific command-line parser, environment variable names, default scope, exact expected tool set, no-mutation surface checks, and operator-facing output.

Tests mock the network helper boundaries and sequence functions rather than making public HTTP calls. Live operators can run the script against the public URL after the launcher and discovery smoke pass.

The launcher carries the optional approval-token-file path through `LaunchConfig` for operator guidance only. Its printed e2e command now includes `--approval-token-file`: either the exact quoted path passed to the launcher or a placeholder plus explicit env/export guidance when the token came from dotenv or process env.

## Intentional

- This is a no-mutation e2e smoke. It lists tools only and intentionally does not call `verify_draft`.
- The script reuses the draft-writer e2e helpers instead of extracting a shared library in this slice.
- This does not settle ChatGPT search/fetch adapters or token-derived tenant binding.

## Deferred

- `PR-Content-Ops-MCP-Token-Tenant-Binding`: derive tenant account binding from OAuth token/client state instead of the current configured account binding.
- `PR-Content-Ops-MCP-Dual-Client-Smoke`: run public OAuth flows against both Claude and the chosen ChatGPT connector surface.
- `PR-Content-Ops-MCP-Connector-Boundary-Smoke`: optional bearer/auth boundary smoke if needed after e2e hardening.
- Parked hardening: none.

## Verification

- Passed: python -m pytest tests/test_check_content_ops_marketer_verify_oauth_e2e.py -q
  - 13 passed in 0.26s
- Passed: python -m pytest tests/test_check_content_ops_marketer_verify_oauth_e2e.py tests/test_check_content_ops_marketer_verify_oauth_discovery.py tests/test_mcp_content_ops_marketer_verify.py -q
  - 33 passed in 0.50s
- Passed: python -m py_compile scripts/check_content_ops_marketer_verify_oauth_e2e.py
- Passed: python scripts/check_content_ops_marketer_verify_oauth_e2e.py --help
- Passed: git diff --check
- Passed: bash scripts/run_extracted_pipeline_checks.sh
  - 295 passed in 1.75s for extracted_reasoning_core
  - 3417 passed, 10 skipped in 58.70s for extracted_content_pipeline
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content_ops_mcp_oauth_e2e_pr_body.md
  - local PR review passed
- Passed after review fix: python -m pytest tests/test_start_content_ops_marketer_verify_oauth_server.py tests/test_check_content_ops_marketer_verify_oauth_e2e.py -q
  - 28 passed in 0.31s
- Passed after review fix: bash scripts/run_extracted_pipeline_checks.sh
  - 295 passed in 1.70s for extracted_reasoning_core
  - 3418 passed, 10 skipped in 57.29s for extracted_content_pipeline
- Passed after review fix: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content_ops_mcp_oauth_e2e_pr_body.md
  - local PR review passed

## Estimated diff size

| Area | Actual LOC |
|---|---:|
| E2E checker | +221 |
| Checker tests | +289 |
| Launcher review fix | +33 / -1 |
| Docs and CI enrollment | +11 / -5 |
| Plan doc | +101 |
| **Total** | **661** |

This is 7 files, +655 / -6. It exceeds the normal 400 LOC target for the reason named in Why this slice exists.
