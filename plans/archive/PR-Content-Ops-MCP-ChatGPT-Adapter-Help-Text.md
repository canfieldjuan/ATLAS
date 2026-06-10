# PR-Content-Ops-MCP-ChatGPT-Adapter-Help-Text

## Why this slice exists

#1407 fixed the ChatGPT adapter launcher's port resolution, but review caught one remaining operator-facing drift: `--help` still inherited rich-verifier help text for port, issuer URL, and resource URL defaults. The runtime now uses the adapter port env/default and adapter OAuth paths, so stale help can send operators to env keys that no longer control the adapter launcher.

This slice fixes the documentation at the executable boundary before live dual-client connector smokes.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. Override the ChatGPT adapter launcher's inherited help text for `--port`, `--issuer-url`, and `--resource-url`.
2. Add regression coverage that `--help` names the adapter port env/default and adapter OAuth defaults, not the rich verifier values.

### Files touched

- `plans/PR-Content-Ops-MCP-ChatGPT-Adapter-Help-Text.md`
- `scripts/start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py`
- `tests/test_start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py`

### Review Contract

- Acceptance criteria:
  - [ ] Adapter launcher `--help` names `ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_CHATGPT_PORT`.
  - [ ] Adapter launcher `--help` names the 8069 adapter default.
  - [ ] Adapter launcher `--help` names the ChatGPT adapter issuer/resource defaults.
  - [ ] Adapter launcher `--help` no longer describes the rich verifier port env/default as the adapter override path.
  - [ ] Runtime port resolution behavior from #1407 remains unchanged.
- Affected surfaces: MCP launcher / operator runbook defaults / tests.
- Risk areas: public connector rollout / operator runbook drift.
- Reviewer rules triggered: R1, R2, R5, R12

## Mechanism

The adapter launcher already reuses the rich verifier parser for common arguments. After constructing that parser, this slice rewrites the help strings for the adapter-specific defaults while leaving argument names and parsing behavior unchanged. The test exercises the real parser's formatted help output.

## Intentional

- This PR changes help text only; it does not change launch-time env precedence.
- The adapter launcher continues to reuse the rich verifier parser and validation helpers for shared security behavior.
- No new CI enrollment is needed because the touched test file is already enrolled in the extracted wrapper and dedicated Content Ops workflow.

## Deferred

- `PR-Content-Ops-MCP-Live-Dual-Client-Rollout`: run public OAuth smokes against real Claude and ChatGPT connector registrations after this help-text drift is closed.
- Durable verdict/session persistence remains deferred unless live connector use needs restart-safe fetch.
- Parked hardening: none.

## Verification

- Passed: python -m pytest tests/test_start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py tests/test_content_ops_marketer_verify_launcher_contract.py -q
  - 14 passed in 0.46s
- Passed: python -m pytest tests/test_atlas_content_ops_review_workflow.py tests/test_check_content_ops_marketer_verify_oauth_e2e.py tests/test_content_ops_marketer_verify_launcher_contract.py tests/test_start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py tests/test_mcp_content_ops_marketer_verify.py -q
  - 86 passed in 0.72s
- Passed: python -m py_compile scripts/start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py
- Passed: git diff --check
- Passed: bash scripts/run_extracted_pipeline_checks.sh
  - 295 passed in 1.69s for extracted_reasoning_core
  - 3511 passed, 10 skipped in 57.52s for extracted_content_pipeline
- Planned: bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_mcp_chatgpt_adapter_help_text_pr_body.md

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 71 |
| Launcher | 23 |
| Tests | 13 |
| **Total** | **107** |
