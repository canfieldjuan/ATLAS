# Content Ops Strict Validation Observability

## Why this slice exists

PR #557 exposed strict reasoning validation failures in the Content Ops step
audit. Review left one operator-observability follow-up: emit a grep-able log
record when strict validation blocks generated assets.

## Scope (this PR)

1. Log strict validation blockers from the execution layer when they are
   mirrored into the step reasoning audit.
2. Document the current comma-separated blocker reason contract at the shared
   signal helper.
3. Remove the stale #557 coordination row and claim this follow-up slice.
4. Refresh the Content Ops backlog/status docs for the shipped observability
   follow-up.

### Files touched

- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/reasoning_signals.py`
- `plans/PR-Content-Ops-Strict-Validation-Observability.md`
- `tests/test_extracted_content_ops_execution.py`

## Mechanism

When `_step_reasoning_audit(...)` detects strict validation failures, it still
returns the same `validation_blocked` / `validation_failures` response shape,
and now also logs `content_ops_strict_validation_blocked` at warning level with
the output name, visible failure count, and truncation flag.

## Intentional

No result-shape changes. No blocker wire-format migration. The current
serialized blocker reason remains comma-separated stable IDs; richer structured
blocker payloads are separate contract work.

## Deferred

Host-owned falsification policy wiring for strict presets remains separate
product-policy work.

## Verification

pytest tests/test_extracted_content_ops_execution.py -> 40 passed.
py_compile -> passed. git diff check -> passed. ASCII grep on touched Python
files -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| **Total** | ~140 |
