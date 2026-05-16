# Content Ops Strict Validation Telemetry

## Why this slice exists

PR #556 made `multi_pass_strict` fail closed for report and sales brief
generation. The strict failure reason reached generated-asset result errors,
but the operator-facing Content Ops step reasoning audit did not expose a
stable validation field.

## Scope (this PR)

1. Detect strict validation failures in generated-asset result errors.
2. Mirror those failures into the per-step `reasoning` audit as compact
   validation telemetry.
3. Preserve existing generated-asset result shapes and execution statuses.
4. Refresh the Content Ops backlog/status/coordination docs for the shipped
   telemetry follow-up.

### Files touched

- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/reasoning_signals.py`
- `plans/PR-Content-Ops-Strict-Validation-Telemetry.md`
- `tests/test_extracted_content_ops_execution.py`

## Mechanism

`_step_reasoning_audit(...)` now scans `result["errors"]` for
`reasoning_validation_blocked` entries. When present, it adds
`validation_blocked: true` and a bounded `validation_failures` list containing
the normalized reason, optional `target_id`, and parsed blocker identifiers.
The strict-validation reason signal is shared with the API wrapper so producer
and executor stay on the same string contract.

## Intentional

No report/sales generation contract changes. Services still return their own
`errors`; the executor only mirrors strict validation failures into the compact
reasoning audit so UI/operator automation has one stable place to inspect them.

## Deferred

Host-owned falsification policy wiring for strict presets remains separate
product-policy work.

## Verification

pytest tests/test_extracted_content_ops_execution.py
tests/test_extracted_content_control_surface_api.py -> 86 passed.
py_compile -> passed. git diff check -> passed. ASCII grep on touched Python
files -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| **Total** | ~255 |
