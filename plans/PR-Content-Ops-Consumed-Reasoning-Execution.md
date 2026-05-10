# PR-Content-Ops-Consumed-Reasoning-Execution

## Why this slice exists

AI Content Ops execution already reports whether a reasoning provider is wired
and how many contexts a service says it consumed. The frontend contract still
calls the reasoning drawer blocked because `/content-ops/execute` has no stable
field for a service to return the consumed reasoning payload itself.

## Scope (this PR)

1. Add an optional `consumed_contexts` field to the per-step execution reasoning
   audit when a service result returns `consumed_reasoning_contexts`.
2. Keep existing `contexts_used` behavior unchanged for services that only
   report counts.
3. Update the Content Ops frontend contract and package status to describe the
   new optional field.

### Files touched

- `extracted_content_pipeline/content_ops_execution.py`
- `tests/test_extracted_content_ops_execution.py`
- `docs/frontend/content_ops_frontend_contract.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Consumed-Reasoning-Execution.md`

## Mechanism

`_step_reasoning_audit(...)` now looks for `result["consumed_reasoning_contexts"]`
after it mirrors `reasoning_contexts_used`. If the value is a mapping or a list
of mappings, it is copied into `reasoning.consumed_contexts`. Missing or
malformed values are ignored so older services keep their current response
shape.

## Intentional

- This does not force every generator to expose raw consumed context yet. It
  creates the execution contract first so service-level opt-in slices can land
  without changing the API shape again.
- `contexts_used` remains the compact count; `consumed_contexts` is the
  optional drawer-ready payload.
- `signal_extraction` still omits reasoning audit because its catalog
  requirement is `absent`.

## Deferred

- Service-level opt-in for campaign, blog post, report, landing page, and sales
  brief results to populate `consumed_reasoning_contexts` from their prompt
  payloads.
- Frontend drawer UI that renders `reasoning.consumed_contexts`.

## Verification

- `pytest tests/test_extracted_content_ops_execution.py` -> 34 passed
- `python -m py_compile extracted_content_pipeline/content_ops_execution.py tests/test_extracted_content_ops_execution.py` -> passed
- `bash scripts/check_ascii_python.sh` -> passed
- `git diff --check` -> passed
- `bash scripts/run_extracted_pipeline_checks.sh` -> 1432 passed, 1 existing
  torch/pynvml warning

## Estimated diff size

6 files, roughly +120 / -20. Under the 400 LOC review budget.
