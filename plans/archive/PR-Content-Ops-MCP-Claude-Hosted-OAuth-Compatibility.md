# PR-Content-Ops-MCP-Claude-Hosted-OAuth-Compatibility

## Why this slice exists
Live Claude.ai setup exposed a #1415 rollout gap: hosted Claude used root OAuth
endpoints while the rich verifier was documented under `/content-ops-marketer`.
This slice makes the working root-alias path repeatable and testable.

## Scope (this PR)
Ownership lane: content-ops/review-contract
Slice phase: Production hardening

Add Claude-hosted root OAuth alias guidance to the rich launcher, a checker for
the observed root authorization redirect, focused tests, and CI enrollment.

### Files touched
- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`
- `plans/PR-Content-Ops-MCP-Claude-Hosted-OAuth-Compatibility.md`
- `scripts/check_content_ops_marketer_verify_claude_hosted_oauth.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/start_content_ops_marketer_verify_oauth_server.py`
- `tests/test_check_content_ops_marketer_verify_claude_hosted_oauth.py`
- `tests/test_content_ops_marketer_verify_launcher_contract.py`
- `tests/test_start_content_ops_marketer_verify_oauth_server.py`

### Review Contract
- Acceptance criteria:
  - [ ] Launcher prints root OAuth Funnel aliases for Claude hosted connectors.
  - [ ] Checker accepts root authorization redirects to the Content Ops approval path.
  - [ ] Checker rejects missing approval redirects, wrong hosts, and non-redirect responses.
  - [ ] Existing rich verifier and ChatGPT adapter commands remain compatible.
- Affected surfaces: MCP OAuth rollout scripts, operator guidance, CI enrollment.
- Risk areas: security, backcompat, deployment safety, maintainability.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R12, R13

## Mechanism
The launcher prints deterministic root OAuth `tailscale funnel` aliases. The
checker sends a Claude-hosted authorization request without following redirects
and requires a redirect to the configured Content Ops approval path. Tests mock
transport and cover success plus failure branches.

## Intentional
No MCP server changes; no generated OAuth client IDs or secrets; checker
defaults to Claude's hosted callback URL and accepts overrides.

## Deferred
Public-client metadata changes and process-manager durability stay deferred. Parked hardening: none.

## Verification
- Passed: focused launcher/checker pytest, 29 passed.
- Passed: dedicated Content Ops workflow pytest, 98 passed.
- Passed: py_compile for launcher and checker.
- Passed: extracted wrapper, 295 reasoning-core passed; 3540 content-pipeline passed, 10 skipped.
- Passed: local PR review with body file.

## Estimated diff size
| Area | Estimated LOC |
| --- | ---: |
| Total | ~398 |

8 files, +398 / -0.
