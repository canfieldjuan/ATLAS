# PR-Content-Ops-MCP-ChatGPT-Adapter-Port-Env

## Why this slice exists

#1405 merged the public OAuth launcher path for the ChatGPT search/fetch adapter. Review left one non-blocking but real launcher hardening issue: the adapter launcher loads `.env` and currently lets the rich verifier's shared `ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_PORT=8068` override the adapter's documented `8069` default. That can make the rich verifier and adapter collide when an operator uses one shared dotenv file.

This slice closes that follow-up before live dual-client connector smokes. It gives the adapter launcher a dedicated port env key while still passing the resolved value through the existing runtime port key expected by the MCP module.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. Add a launcher-only ChatGPT adapter port env key for `scripts/start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py`.
2. Make the adapter launcher prefer CLI `--port`, then the dedicated adapter port env, then the adapter default, ignoring the rich verifier port env for adapter defaulting.
3. Keep the launched process compatible with the existing runtime config by writing the resolved adapter port to the shared runtime port env.
4. Add regression tests for shared dotenv collision, dedicated env override, and CLI override precedence.

### Files touched

- `plans/PR-Content-Ops-MCP-ChatGPT-Adapter-Port-Env.md`
- `scripts/start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py`
- `tests/test_start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py`

### Review Contract

- Acceptance criteria:
  - [ ] A shared dotenv containing the rich verifier port does not change the adapter launcher's default port.
  - [ ] The dedicated adapter port env can set the adapter launcher port.
  - [ ] Explicit `--port` still wins over both env keys.
  - [ ] The launched subprocess still receives `ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_PORT` with the resolved adapter port.
  - [ ] Existing adapter launcher validation and smoke guidance remain unchanged.
- Affected surfaces: MCP launcher / operator rollout defaults / tests.
- Risk areas: public connector rollout / operator runbook drift / CI enrollment.
- Reviewer rules triggered: R1, R2, R5, R12

## Mechanism

The adapter launcher will define a dedicated env key for its port, then normalize that into the existing runtime env key after dotenv and process env are loaded. The MCP server module does not need a new settings field because the launcher is the process boundary: the adapter subprocess already reads the shared runtime port key through existing Atlas settings.

## Intentional

- This PR does not add a new Atlas settings field. The dedicated key is launcher-only and exists to resolve operator defaults before process start.
- Directly running the adapter module with `--sse` still uses the existing runtime port setting. The supported public rollout path is the launcher.
- This PR does not change issuer/resource URL env names; those are shared intentionally because both surfaces use the same verify-only OAuth provider semantics.

## Deferred

- `PR-Content-Ops-MCP-Live-Dual-Client-Rollout`: run public OAuth smokes against real Claude and ChatGPT connector registrations after launcher hardening.
- Durable verdict/session persistence remains deferred unless live connector use needs restart-safe fetch.
- Parked hardening: none.

## Verification

- Passed: python -m pytest tests/test_start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py tests/test_content_ops_marketer_verify_launcher_contract.py -q
  - 13 passed in 0.47s
- Passed: python -m pytest tests/test_atlas_content_ops_review_workflow.py tests/test_check_content_ops_marketer_verify_oauth_e2e.py tests/test_content_ops_marketer_verify_launcher_contract.py tests/test_start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py tests/test_mcp_content_ops_marketer_verify.py -q
  - 85 passed in 0.75s
- Passed: python -m py_compile scripts/start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py
- Passed: git diff --check
- Passed: bash scripts/run_extracted_pipeline_checks.sh
  - 295 passed in 1.68s for extracted_reasoning_core
  - 3510 passed, 10 skipped in 59.13s for extracted_content_pipeline
- Planned: bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_mcp_chatgpt_adapter_port_env_pr_body.md

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 73 |
| Launcher | 13 |
| Tests | 48 |
| **Total** | **134** |
