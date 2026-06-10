# PR-Content-Ops-MCP-Live-Dual-Client-Rollout

## Why this slice exists

#1353 is past the rich verifier, ChatGPT adapter, OAuth rollout, adapter port
env, and adapter help-text slices. The remaining gap is one operator-safe proof
that both public connector surfaces can be smoke tested together without
mutating draft content or printing approval secrets.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Add a dual-client rollout checker that delegates to the existing
   no-mutation OAuth e2e checker for `claude-rich` and
   `chatgpt-search-fetch`.
2. Require separate rich and adapter issuer/resource URL pairs, reject identical
   resource URLs, and prefer approval-token file handoff.
3. Add deterministic tests for command construction, missing inputs, identical
   resource rejection, fail-fast behavior, and secret redaction.
4. Enroll the new checker tests in the extracted wrapper and dedicated workflow.

### Files touched

- `plans/PR-Content-Ops-MCP-Live-Dual-Client-Rollout.md`
- `scripts/check_content_ops_marketer_verify_dual_client_rollout.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`
- `tests/test_check_content_ops_marketer_verify_dual_client_rollout.py`

### Review Contract

- Acceptance criteria:
  - [ ] The wrapper invokes both profiles against their matching URL pairs.
  - [ ] Missing inputs and identical resource URLs fail before any smoke call.
  - [ ] Either profile failure blocks rollout success.
  - [ ] Output omits literal approval-token values, and new tests are enrolled.
- Affected surfaces: operator smoke script / OAuth rollout runbook / CI.
- Risk areas: public connector rollout / approval-token handling / CI
  enrollment / false-green smoke reporting.
- Reviewer rules triggered: R1, R2, R3, R10, R12

## Mechanism

The new script parses two public URL pairs and one approval-token handoff, builds
one argument vector per client profile, and calls the existing OAuth e2e checker
in sequence. It stops on the first non-zero result and prints only profile
labels plus public URLs.

## Intentional

- No live credentials, generated draft content, MCP tool calls, or OAuth logic
  are added; this wrapper is runbook glue.
- Shared issuers remain allowed because distinct resources are the surface guard.

## Deferred

- Manually captured live run artifacts and durable verdict/session persistence
  remain deferred until public registrations or restart-safe fetch need them.
- Parked hardening: none.

## Verification

- Passed: python -m pytest tests/test_check_content_ops_marketer_verify_dual_client_rollout.py tests/test_check_content_ops_marketer_verify_oauth_e2e.py -q
  - 24 passed in 0.49s
- Passed: python -m py_compile scripts/check_content_ops_marketer_verify_dual_client_rollout.py
- Passed: git diff --check
- Passed: python -m pytest tests/test_atlas_content_ops_review_workflow.py tests/test_check_content_ops_marketer_verify_dual_client_rollout.py tests/test_check_content_ops_marketer_verify_oauth_e2e.py tests/test_content_ops_marketer_verify_launcher_contract.py tests/test_start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py tests/test_mcp_content_ops_marketer_verify.py -q
  - 92 passed in 0.91s
- Passed: bash scripts/run_extracted_pipeline_checks.sh
  - 295 passed in 1.70s for extracted_reasoning_core
  - 3517 passed, 10 skipped in 58.93s for extracted_content_pipeline
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_mcp_live_dual_client_rollout_pr_body.md

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 84 |
| Rollout checker | 180 |
| Tests | 129 |
| CI/script enrollment | 6 |
| **Total** | **399** |
