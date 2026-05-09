# PR: Content Ops Reasoning Usage Audit

## Goal

Make the step-level reasoning audit report actual reasoning consumption counts,
not only provider readiness.

## Scope

- Add `contexts_used` to `ContentOpsStepExecution.reasoning` when service
  results expose `reasoning_contexts_used`.
- Preserve the existing readiness fields: `requirement`,
  `service_supports_reasoning`, and `provider_configured`.
- Thread the new audit field through the frontend wire/domain adapter.
- Render the count in the compact reasoning badge.

## Non-Goals

- Do not expose raw reasoning context payloads.
- Do not change generation services or reasoning provider lookup.
- Do not add a drawer or expanded reasoning inspection UI.

## Verification

- `python -m py_compile extracted_content_pipeline/content_ops_execution.py`
- `python -m pytest tests/test_extracted_content_ops_execution.py -q`
- `npm run build` from `atlas-intel-ui`
- `git diff --check`
