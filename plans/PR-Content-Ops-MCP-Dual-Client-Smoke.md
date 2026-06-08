# PR-Content-Ops-MCP-Dual-Client-Smoke

## Why this slice exists

#1353 names the remaining connector-specific gap after token-bound tenant binding and launcher drift guards: the verify-only Content Ops marketer MCP must be validated for Claude plus the chosen ChatGPT connector surface. The important product decision is already in the tracker: Claude can use the rich `verify_draft` tool directly, while the proven ChatGPT connector pattern in this repo is a `search` and `fetch` surface.

This slice codifies that decision in the OAuth e2e smoke before any adapter work starts. The current server should keep passing the Claude rich profile, and the ChatGPT profile should fail closed unless the listed surface is the proven `search` and `fetch` pair.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Extend the Content Ops marketer verify OAuth e2e smoke with an explicit client-profile contract.
2. Keep the default profile as Claude rich verify-only: exactly `verify_draft`, no generation, publishing, registry mutation, or adapter tools.
3. Add the chosen ChatGPT proven connector profile: exactly `search` and `fetch`, with `verify_draft` treated as incompatible for that profile.
4. Update launcher/operator docs so the current e2e command names the Claude rich profile and the deferred ChatGPT command names the `search` and `fetch` adapter requirement.
5. Enroll the e2e test in the dedicated Content Ops review workflow because this PR changes the e2e checker contract.

### Files touched

- `plans/PR-Content-Ops-MCP-Dual-Client-Smoke.md`
- `CLAUDE.md`
- `scripts/check_content_ops_marketer_verify_oauth_e2e.py`
- `scripts/start_content_ops_marketer_verify_oauth_server.py`
- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`
- `tests/test_check_content_ops_marketer_verify_oauth_e2e.py`
- `tests/test_content_ops_marketer_verify_launcher_contract.py`

### Review Contract

- Acceptance criteria:
  - [ ] The OAuth e2e checker defaults to the Claude rich profile and still accepts exactly `verify_draft`.
  - [ ] The checker can run in ChatGPT proven connector profile and accepts exactly `search` plus `fetch`.
  - [ ] The ChatGPT proven profile rejects the current `verify_draft`-only surface with a clear incompatibility error.
  - [ ] The Claude rich profile rejects `search` or `fetch` as unexpected extras so adapter tools do not leak into the rich verifier.
  - [ ] The e2e smoke still performs registration, approval, token exchange, and list-tools only; it does not call draft verification.
  - [ ] Launcher/docs output names the client profile explicitly.
  - [ ] The focused e2e tests run in the extracted pipeline wrapper and the dedicated Content Ops review workflow.
- Affected surfaces: MCP / auth smoke / operator launcher / docs / CI.
- Risk areas: public tool-surface compatibility / deployment safety / backcompat / CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R12

## Mechanism

The e2e checker gets a small client-profile table keyed by `claude-rich` and `chatgpt-search-fetch`. Each profile owns its expected tool set, denied tool set, and success wording. The existing OAuth flow remains unchanged: register a temporary client, route through operator approval, exchange the code, list MCP tools, and validate the returned tool names against the selected profile.

Tests keep network transport mocked at the same boundaries as the existing e2e tests. The new fixtures exercise both profile happy paths, the ChatGPT rejection of a `verify_draft`-only surface, and the Claude rejection of adapter tools.

## Intentional

- This PR chooses the proven ChatGPT connector surface as `search` and `fetch`; it does not build that adapter.
- The current live Content Ops marketer server is expected to pass `claude-rich`, not `chatgpt-search-fetch`.
- This remains a no-mutation smoke. It lists tools only and does not call `verify_draft`.
- The profile logic stays in the checker instead of the MCP server because this slice validates client compatibility; it does not change runtime behavior.

## Deferred

- `PR-Content-Ops-MCP-ChatGPT-Search-Fetch-Adapter`: add the actual ChatGPT-compatible adapter surface that can satisfy the `chatgpt-search-fetch` profile.
- `PR-Content-Ops-MCP-Live-Dual-Client-Rollout`: run the public OAuth smoke against real Claude and ChatGPT connector registrations once both surfaces exist.
- Parked hardening: none.

## Verification

- Passed: python -m pytest tests/test_check_content_ops_marketer_verify_oauth_e2e.py tests/test_content_ops_marketer_verify_launcher_contract.py -q
  - 21 passed in 0.66s
- Passed: python -m py_compile scripts/check_content_ops_marketer_verify_oauth_e2e.py scripts/start_content_ops_marketer_verify_oauth_server.py
- Passed: git diff --check
- Passed: python -m pytest tests/test_atlas_content_ops_review_workflow.py tests/test_content_ops_marketer_verify_launcher_contract.py tests/test_check_content_ops_marketer_verify_oauth_e2e.py tests/test_mcp_content_ops_marketer_verify.py -q
  - 66 passed in 0.63s
- Passed: bash scripts/run_extracted_pipeline_checks.sh
  - 295 passed in 1.66s for extracted_reasoning_core
  - 3454 passed, 10 skipped in 56.77s for extracted_content_pipeline
- Pending before PR: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content_ops_mcp_dual_client_smoke_pr_body.md

## Estimated diff size

| Area | Actual LOC |
|---|---:|
| Plan doc | 86 |
| E2E checker profile contract | 85 |
| Tests | 100 |
| Launcher/docs/CI enrollment | 26 |
| **Total** | **297** |

This is under the normal 400 LOC target.
