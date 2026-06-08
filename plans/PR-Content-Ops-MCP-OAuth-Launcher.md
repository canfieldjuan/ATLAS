# PR-Content-Ops-MCP-OAuth-Launcher

## Why this slice exists

#1387 added server-side OAuth mode for the verify-only marketer MCP server, and #1388 added public discovery verification. The #1353 delivery tracker still requires an operator-safe launcher before this connector is practical to run against a public Funnel URL. Operators need one command that loads Atlas dotenv files, forces the correct OAuth mode, validates the public issuer/resource/approval settings, masks secrets, and prints the exact public smoke commands.

This is the next smallest production-hardening slice in the OAuth rollout. It deliberately stops before token exchange and live client flows so the launcher can be reviewed independently from the later e2e smoke.

This slice is expected to exceed the normal 400 LOC target because the operator CLI, secret masking, validation branches, Funnel route rendering, and failure-branch tests need to land together. Splitting tests from the launcher would violate the detector coverage rule for this config/security surface.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. Add a Content Ops Marketer Verify OAuth launcher script modeled on the draft-writer launcher.
2. Force OAuth mode for the marketer verify server, validate public URLs, validate the approval token length, and validate the configured port before launching.
3. Print masked operator configuration, path-preserving Tailscale Funnel route commands, the existing discovery smoke command, and the planned e2e smoke command.
4. Enroll focused launcher tests in the extracted pipeline wrapper.
5. Document the launcher command in the Content Ops Marketer Verify MCP section.

### Files touched

- `plans/PR-Content-Ops-MCP-OAuth-Launcher.md`
- `CLAUDE.md`
- `scripts/start_content_ops_marketer_verify_oauth_server.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_start_content_ops_marketer_verify_oauth_server.py`

### Review Contract

- Acceptance criteria:
  - [ ] The launcher defaults to the marketer verify public issuer/resource URLs and port without requiring shell-sourced env.
  - [ ] The launcher forces `ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_AUTH_MODE=oauth`.
  - [ ] The launcher rejects missing or short approval tokens and invalid ports before starting the server process.
  - [ ] Operator output masks bearer and approval token values while still showing which required settings are present.
  - [ ] Operator output prints the content-ops discovery smoke command and a placeholder e2e command for the later e2e slice.
  - [ ] The server command launches `atlas_brain.mcp.content_ops_marketer_verify_server` with `--sse`.
- Affected surfaces: MCP / auth / scripts / docs / CI.
- Risk areas: security / config / deployment safety / CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R11, R12

## Mechanism

The new launcher follows the existing draft-writer operator launcher shape with content-ops-specific constants, env var names, default URLs, server module, and smoke command names. It loads `.env` and `.env.local` by default, lets explicit CLI flags override dotenv/process values where appropriate, and supports an approval-token file so operators can avoid placing secrets in shell history.

The script only validates and starts the existing MCP server. It does not add new auth behavior to the server, does not exchange tokens, and does not call `verify_draft`.

## Intentional

- This slice prints the planned e2e smoke command before the e2e checker exists. That keeps operator runbooks stable while the actual token-exchange smoke remains a separate reviewable PR.
- The launcher uses the same lightweight dotenv parsing pattern as the draft-writer precedent instead of adding a dependency.
- The launcher keeps the current single-bound-tenant server model; token-derived tenant binding stays deferred.

## Deferred

- `PR-Content-Ops-MCP-OAuth-E2E`: dynamic registration, approval, token exchange, and list-tools smoke.
- `PR-Content-Ops-MCP-Token-Tenant-Binding`: derive tenant account binding from OAuth token/client state instead of the current configured account binding.
- `PR-Content-Ops-MCP-Dual-Client-Smoke`: run the public OAuth flow against both Claude and the chosen ChatGPT connector surface after the e2e checker exists.
- Parked hardening: none.

## Verification

- python -m pytest tests/test_start_content_ops_marketer_verify_oauth_server.py -q -- 14 passed.
- python -m pytest tests/test_start_content_ops_marketer_verify_oauth_server.py tests/test_check_content_ops_marketer_verify_oauth_discovery.py tests/test_mcp_content_ops_marketer_verify.py -q -- 34 passed.
- python -m py_compile scripts/start_content_ops_marketer_verify_oauth_server.py -- passed.
- python scripts/start_content_ops_marketer_verify_oauth_server.py --help -- passed.
- git diff --check -- passed.
- bash scripts/run_extracted_pipeline_checks.sh -- 3388 passed, 10 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content_ops_mcp_oauth_launcher_pr_body.md -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Launcher script | ~308 |
| Launcher tests | ~304 |
| Docs and CI enrollment | ~4 |
| Plan doc | ~82 |
| **Total** | **~698** |

See Why for the over-budget justification.
