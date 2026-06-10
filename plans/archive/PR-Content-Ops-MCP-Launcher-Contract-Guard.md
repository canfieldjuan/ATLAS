# PR-Content-Ops-MCP-Launcher-Contract-Guard

## Why this slice exists

#1395 merged the token-bound tenant gate for the verify-only Content Ops marketer MCP OAuth path. Its review also identified the third recurring launcher/runbook drift in this OAuth arc: server or checker validation tightened, but the operator launcher still printed a dry-run command that did not satisfy the downstream contract.

This slice turns that review finding into mechanism before the dual-client connector smoke. The launcher output, downstream checker argument parsers, and server-required OAuth account binding should fail in tests when they drift apart.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. Add a focused deterministic test that parses the Content Ops marketer OAuth launcher dry-run commands through the real discovery and e2e checker parsers.
2. Add a focused deterministic test that proves the launcher required-env report includes the account binding required by the OAuth server startup path.
3. Enroll the new test file and the launcher/checker script drift surfaces in the extracted pipeline check wrapper and the dedicated Content Ops review workflow so CI exercises the guard when those scripts change.

### Files touched

- `plans/PR-Content-Ops-MCP-Launcher-Contract-Guard.md`
- `tests/test_content_ops_marketer_verify_launcher_contract.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`

### Review Contract

- Acceptance criteria:
  - [ ] The launcher's printed discovery command is accepted by the real discovery checker parser.
  - [ ] The launcher's printed e2e command is accepted by the real e2e checker parser and carries an approval-token file path.
  - [ ] The launcher required-env report includes the account binding required before OAuth server startup.
  - [ ] The new guard is enrolled in the extracted pipeline check wrapper and the dedicated Content Ops review workflow.
- Affected surfaces: tests / launcher contract / checker contract / CI enrollment.
- Risk areas: deployment safety / backcompat / maintainability.
- Reviewer rules triggered: R1, R2, R5, R10, R12

## Mechanism

The test imports the launcher and checker modules directly, builds a deterministic dry-run launch config, captures the launcher's operator guidance, reconstructs the printed multi-line Python commands, and feeds those argument lists through the real checker parsers. If the launcher drops a required checker flag again, the parser/config assertion fails.

A second assertion pins the server account-binding requirement at the launcher boundary: the required-env report must include `ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_ACCOUNT_ID`, and a whitespace-only value must remain invalid through the launcher validator. The parser tests clear checker env defaults before constructing parsers so ambient shell state cannot satisfy or invalidate the command contract.

## Intentional

- This PR does not change launcher, checker, OAuth provider, or MCP server behavior.
- The guard stays deterministic and offline; it does not start the server, call Tailscale, hit public URLs, or perform OAuth network traffic.
- It only covers the Content Ops marketer verify OAuth launcher. Invoicing launchers can get the same pattern later if needed.
- The test reconstructs commands from printed operator guidance because the drift class is specifically the human-facing dry-run/runbook output.

## Deferred

- `PR-Content-Ops-MCP-Dual-Client-Smoke`: run public OAuth flows against both Claude and the chosen ChatGPT connector surface after this guard is in place.
- `PR-Content-Ops-MCP-Launcher-Contract-Shared-Helper`: optional future extraction if another launcher needs the same guard shape.
- Parked hardening: none.

## Verification

- Passed: python -m pytest tests/test_content_ops_marketer_verify_launcher_contract.py -q
  - 3 passed in 0.37s
- Passed: bash scripts/run_extracted_pipeline_checks.sh
  - 295 passed in 1.66s for extracted_reasoning_core
  - 3449 passed, 10 skipped in 58.57s for extracted_content_pipeline
- Passed after workflow enrollment fix: python -m pytest tests/test_atlas_content_ops_review_workflow.py tests/test_content_ops_marketer_verify_launcher_contract.py tests/test_mcp_content_ops_marketer_verify.py -q
  - 48 passed in 0.35s
- Passed after workflow enrollment fix: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content_ops_mcp_launcher_contract_guard_pr_body.md
  - local PR review passed
- Passed after local reviewer fix: python -m pytest tests/test_content_ops_marketer_verify_launcher_contract.py -q
  - 3 passed in 0.35s
- Passed after local reviewer fix: env ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN=ambient-token python -m pytest tests/test_content_ops_marketer_verify_launcher_contract.py -q
  - 3 passed in 0.35s
- Passed after local reviewer fix: python -m pytest tests/test_atlas_content_ops_review_workflow.py tests/test_content_ops_marketer_verify_launcher_contract.py tests/test_mcp_content_ops_marketer_verify.py -q
  - 48 passed in 0.35s
- Passed after local reviewer fix: python -m py_compile tests/test_content_ops_marketer_verify_launcher_contract.py
- Passed after local reviewer fix: git diff --check
- Passed after local reviewer fix: bash scripts/run_extracted_pipeline_checks.sh
  - 295 passed in 1.69s for extracted_reasoning_core
  - 3449 passed, 10 skipped in 58.12s for extracted_content_pipeline
- Passed after local reviewer fix: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content_ops_mcp_launcher_contract_guard_pr_body.md
  - local PR review passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | +89 |
| Contract guard tests | +168 |
| CI enrollment | +10 |
| **Total** | **267** |

This is under the normal 400 LOC target.
