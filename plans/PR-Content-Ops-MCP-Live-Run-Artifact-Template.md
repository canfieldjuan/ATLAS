# PR-Content-Ops-MCP-Live-Run-Artifact-Template

## Why this slice exists

#1412 and #1413 made the dual-client smoke executable and discoverable, but the
remaining live step is still easy to perform badly: an operator could paste raw
terminal output into an issue and leak tokens, client secrets, authorization
codes, or access tokens.

This slice defines the sanitized artifact we expect from the future live
Claude-plus-ChatGPT connector run before any live credentials exist.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Product polish

1. Add a Content Ops marketer verify live-run artifact runbook.
2. Define the exact commands to run after public Claude and ChatGPT connector
   registrations are available.
3. Define the sanitized evidence fields to capture and the secret fields that
   must never appear.

### Files touched

- `plans/PR-Content-Ops-MCP-Live-Run-Artifact-Template.md`
- `docs/MCP_CONTENT_OPS_MARKETER_VERIFY_LIVE_RUN_ARTIFACT.md`

### Review Contract

- Acceptance criteria:
  - [ ] The doc points operators to the rich launcher, adapter launcher, and
        dual-client rollout checker already on main.
  - [ ] The artifact template records both client profiles, public URL pairs,
        command exit states, and expected tool names.
  - [ ] The doc tells operators to start both MCP servers without `--dry-run`
        before running public discovery and dual-client smokes.
  - [ ] The doc explicitly forbids approval tokens, client secrets,
        authorization codes, access tokens, and refresh tokens in artifacts.
  - [ ] The doc keeps generation, durable persistence, and account-picker work
        out of scope.
- Affected surfaces: operator runbook / live evidence handoff.
- Risk areas: secret hygiene / public connector rollout / scope drift.
- Reviewer rules triggered: R1, R3, R10

## Mechanism

This is a documentation-only slice. The new runbook gives operators a checklist
and copyable artifact envelope for the live connector run. It references the
existing scripts instead of adding new behavior.

## Intentional

- No live run artifact is committed in this PR because public connector
  registrations and secrets are not available in-repo.
- No script, MCP server, OAuth, or persistence behavior changes.
- AI reconciliation: fixed the Codex/reviewer P2 by adding the non-dry-run
  two-server startup step and startup fields in the artifact template.

## Deferred

- Actual live-run artifact capture remains an operator action after public
  Claude and ChatGPT connector registrations are available.
- Durable verdict/session persistence remains deferred unless live connector use
  proves process-local fetch state is insufficient.
- Approval-page account picker remains deferred unless one process must approve
  multiple tenant accounts.
- Parked hardening: none.

## Verification

- Passed: test -s docs/MCP_CONTENT_OPS_MARKETER_VERIFY_LIVE_RUN_ARTIFACT.md
- Passed: git diff --check
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_mcp_live_run_artifact_template_pr_body.md

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 85 |
| Runbook doc | 178 |
| **Total** | **263** |
