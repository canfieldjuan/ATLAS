# PR-Content-Ops-MCP-ChatGPT-Adapter-OAuth-Rollout

## Why this slice exists

#1401 landed the ChatGPT-compatible search/fetch adapter, but it is still only a local tool module. #1353 now has both halves of the dual-client decision on main: Claude rich uses the existing `verify_draft` OAuth surface, while ChatGPT proven connector mode uses the new search/fetch adapter. The remaining gap before a live dual-client rollout is making the adapter independently runnable through the same public OAuth launcher and checker path.

This slice keeps the rollout deterministic and local-testable. It does not register live external clients and it does not add a durable verdict ledger. It only makes the adapter server routable in OAuth mode, gives operators a ChatGPT-specific launcher path, and pins the launcher output to the existing discovery/e2e checker contract.

This PR exceeds the normal 400 LOC target because the adapter HTTP entrypoint, dedicated launcher, and offline launcher-contract tests are one production-hardening unit: without the tests, the public operator command could drift from the already-built checker profile and silently point ChatGPT at the wrong surface.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. Add Streamable HTTP entrypoint support to the ChatGPT adapter module so it can run as its own OAuth-protected MCP server.
2. Reuse the existing Content Ops marketer verify OAuth provider, tenant account binding, and database lifespan instead of adding a second auth model.
3. Add an operator launcher for the ChatGPT adapter with separate default path/port guidance and a `chatgpt-search-fetch` e2e command.
4. Extend launcher/checker contract tests so the adapter launcher output is parsed by the existing discovery and e2e checker parsers.
5. Enroll the new launcher coverage in the extracted pipeline wrapper and dedicated Content Ops review workflow.

### Files touched

- `plans/PR-Content-Ops-MCP-ChatGPT-Adapter-OAuth-Rollout.md`
- `atlas_brain/mcp/content_ops_marketer_verify_server.py`
- `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py`
- `scripts/start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py`
- `scripts/start_content_ops_marketer_verify_oauth_server.py`
- `tests/test_mcp_content_ops_marketer_verify.py`
- `tests/test_start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py`
- `tests/test_content_ops_marketer_verify_launcher_contract.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`

### Review Contract

- Acceptance criteria:
  - [ ] The adapter can build its own authenticated Streamable HTTP app in OAuth mode.
  - [ ] The adapter OAuth path reuses the verifier OAuth provider, required scope, account binding, and token verifier behavior.
  - [ ] The adapter launcher starts the adapter module, not the rich verifier module.
  - [ ] The adapter launcher prints discovery and e2e smoke commands whose resource URL and client profile match the ChatGPT search/fetch adapter surface.
  - [ ] Existing Claude-rich launcher behavior remains unchanged.
  - [ ] The Claude-rich launcher points ChatGPT rollout guidance to the dedicated adapter launcher instead of printing a stale direct ChatGPT e2e command.
  - [ ] New tests run in the extracted pipeline wrapper and dedicated Content Ops review workflow.
- Affected surfaces: MCP / OAuth launcher / auth configuration / public connector smoke tooling / tests / CI.
- Risk areas: security / tenant isolation / public tool-surface compatibility / operator runbook drift / CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R12

## Mechanism

The adapter module will expose the same HTTP serving shape as the rich verifier, but it will bind that shape to the adapter FastMCP instance. OAuth setup stays centralized in the verifier module: a small verifier helper will configure auth settings and token verification on a supplied MCP instance while continuing to use the existing provider, scope, approval token, account id, and state-file settings. The verifier's own server remains the default caller of that helper, so the rich surface behavior stays unchanged.

The adapter launcher will mirror the existing marketer verify launcher with ChatGPT-specific defaults. It will still set the existing Content Ops marketer verify OAuth env keys because the provider and tenant binding are shared, but it will launch the adapter module and print the e2e command with the `chatgpt-search-fetch` profile. The launcher-contract tests parse the printed commands through the real checker parsers so command drift fails locally.

## Intentional

- This PR does not add a second OAuth provider namespace. The adapter is another transport over the same verify-only tenant binding, not a different product authority.
- The launcher uses separate defaults for adapter path and port so the rich verifier and ChatGPT adapter can be run as separate processes with separate per-process env.
- The existing e2e checker already knows the `chatgpt-search-fetch` profile; this PR wires the adapter to that checker instead of creating a duplicate checker.
- Durable verdict/session persistence stays out of scope because #1401's process-local handoff is enough for the first public adapter smoke.

## Deferred

- `PR-Content-Ops-MCP-Live-Dual-Client-Rollout`: run public OAuth smokes against real Claude and ChatGPT connector registrations after the adapter launcher exists.
- `PR-Content-Ops-MCP-Verdict-Session-Persistence`: replace process-local verdict handoff with tenant-scoped durable storage if live connector use needs restart-safe fetch.
- Optional approval-page account picker remains deferred unless one long-running process must approve multiple tenant accounts.
- Parked hardening: none.

## Verification

- Passed: python -m pytest tests/test_mcp_content_ops_marketer_verify.py tests/test_start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py tests/test_content_ops_marketer_verify_launcher_contract.py tests/test_check_content_ops_marketer_verify_oauth_e2e.py -q
  - 58 passed in 0.67s
- Passed: python -m pytest tests/test_atlas_content_ops_review_workflow.py tests/test_check_content_ops_marketer_verify_oauth_e2e.py tests/test_content_ops_marketer_verify_launcher_contract.py tests/test_start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py tests/test_mcp_content_ops_marketer_verify.py -q
  - 82 passed in 0.75s
- Passed: python -m py_compile atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py scripts/start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py
- Passed: git diff --check
- Passed: bash scripts/run_extracted_pipeline_checks.sh
  - 295 passed in 1.69s for extracted_reasoning_core
  - 3503 passed, 10 skipped in 59.16s for extracted_content_pipeline
- Planned: bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_mcp_chatgpt_adapter_oauth_rollout_pr_body.md

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 94 |
| Adapter HTTP entrypoint and shared auth helper | 88 |
| Adapter launcher | 159 |
| Claude-rich launcher guidance | 11 |
| Tests | 272 |
| CI enrollment | 6 |
| **Total** | **630** |

This is over the normal 400 LOC target for the reason named in Why this slice exists.
