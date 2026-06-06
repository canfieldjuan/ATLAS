# Content Ops Reasoning Capability Check

## Why This Slice Exists

AI Content Ops can now consume explicit, single-pass, and multi-pass reasoning
providers. The hosted operations status endpoint already reports whether those
paths are configured, but hosts still have to infer the missing runtime
requirements from several flat booleans.

This slice makes the existing status endpoint a clearer host setup check by
reporting per-mode reasoning capability readiness.

## Scope

- Add a nested reasoning capability report to campaign operations status.
- Report whether explicit, single-pass, and multi-pass modes are configured.
- Report whether each configured mode is runnable.
- Report missing config flags or runtime requirements for unavailable modes.
- Mark which ready mode is active after precedence is applied.
- Keep generation readiness behavior unchanged.

## Mechanism

The router still uses provider presence as the readiness source of truth:

- explicit provider readiness requires an injected reasoning provider
- single-pass readiness requires the config flag plus LLM and skill providers
- multi-pass readiness requires the config flag plus an LLM provider

The new `reasoning.capabilities` object mirrors those rules so admin UIs and
host smoke checks can show the operator exactly what is missing. It also marks
the active mode so consumers do not have to reimplement precedence.

## Intentional

- No provider handles are opened by the status endpoint.
- No LLM, skill, or reasoning provider calls are made.
- Existing `reasoning.mode`, `single_pass_ready`, and `multi_pass_ready` fields
  remain in place for compatibility.
- Explicit providers still take precedence over packaged single-pass or
  multi-pass config when deciding the active mode.

## Deferred

- No standalone CLI is added in this slice; the hosted status endpoint is the
  lowest-conflict capability surface already used by host installs.
- No UI changes are included; the response contract is ready for a later admin
  display pass.
- No deeper reasoning-core health check is added because that would require
  invoking host LLM/provider dependencies.

## Verification

- Campaign operations API tests cover explicit, single-pass, and multi-pass
  capability payloads.
- Focused tests verify generation readiness still follows the existing rules.
- Python compile, diff, and local PR review checks cover the modified files.

### Files Touched

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-Reasoning-Capability-Check.md` | 78 |
| `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | 7 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `extracted_content_pipeline/STATUS.md` | 10 |
| `extracted_content_pipeline/api/campaign_operations.py` | 55 |
| `tests/test_extracted_campaign_api_operations.py` | 54 |

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Status capability helper | ~55 |
| Tests | ~54 |
| Docs and coordination | ~21 |
| Plan doc | ~78 |
| **Total** | ~207 |
