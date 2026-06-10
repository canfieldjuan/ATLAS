# PR-Content-Ops-MCP-Dual-Client-Rollout-Guidance

## Why this slice exists

#1412 added the dual-client rollout checker, but the operator launcher guidance
still prints only one-profile smoke commands. An operator can now start the rich
verifier and ChatGPT adapter successfully without seeing the combined smoke that
proves both public surfaces together.

This slice makes the combined smoke discoverable at the executable boundary.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Product polish

1. Add dual-client rollout smoke guidance to the ChatGPT adapter OAuth launcher.
2. Keep the command token-file based and use the rich verifier defaults plus the
   adapter launcher's resolved public URLs.
3. Add a launcher-contract regression that parses the printed dual-client command
   through the real dual-client checker parser.

### Files touched

- `plans/PR-Content-Ops-MCP-Dual-Client-Rollout-Guidance.md`
- `scripts/start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py`
- `tests/test_content_ops_marketer_verify_launcher_contract.py`

### Review Contract

- Acceptance criteria:
  - [ ] Adapter launcher output includes the dual-client rollout checker command.
  - [ ] The command uses `--approval-token-file`, not a literal token.
  - [ ] The command parses through the real dual-client checker parser.
  - [ ] Rich URLs point at the rich verifier defaults; adapter URLs point at the
        adapter launcher defaults/resolved URLs.
  - [ ] Existing single-profile discovery/e2e guidance remains unchanged.
- Affected surfaces: operator launcher guidance / tests.
- Risk areas: public connector rollout / operator runbook drift / secret
  handling.
- Reviewer rules triggered: R1, R2, R3, R10

## Mechanism

The adapter launcher already imports the rich launcher for shared defaults and
helpers. After the existing ChatGPT search/fetch e2e command, it prints one
dual-client rollout command using the rich default issuer/resource pair and the
adapter config's issuer/resource pair. The existing launcher-contract test file
loads the dual checker and parses that printed command.

## Intentional

- This PR changes operator text only; no server, OAuth, or MCP tool behavior
  changes.
- The rich URLs use defaults because the adapter launcher does not own the rich
  launcher's runtime overrides. Operators using custom rich URLs can edit those
  two public values before running the command.
- Cross-layer caller hints are same-named launcher guidance functions and direct
  launcher tests; only adapter launcher output changes.

## Deferred

- Manual live run artifact capture remains an operator action after public
  Claude and ChatGPT connector registrations are available.
- Durable verdict/session persistence remains deferred unless live connector use
  proves process-local fetch state is insufficient.
- Parked hardening: none.

## Verification

- Passed: python -m pytest tests/test_content_ops_marketer_verify_launcher_contract.py -q
  - 6 passed in 0.39s
- Passed: python -m py_compile scripts/start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py
- Passed: git diff --check
- Passed: python -m pytest tests/test_atlas_content_ops_review_workflow.py tests/test_check_content_ops_marketer_verify_dual_client_rollout.py tests/test_check_content_ops_marketer_verify_oauth_e2e.py tests/test_content_ops_marketer_verify_launcher_contract.py tests/test_start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py tests/test_mcp_content_ops_marketer_verify.py -q
  - 93 passed in 0.88s
- Passed: bash scripts/run_extracted_pipeline_checks.sh
  - 295 passed in 1.68s for extracted_reasoning_core
  - 3518 passed, 10 skipped in 58.32s for extracted_content_pipeline
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_mcp_dual_client_rollout_guidance_pr_body.md

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 89 |
| Launcher guidance | 8 |
| Tests | 27 |
| **Total** | **124** |
