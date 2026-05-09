# PR: Content Ops reasoning consumption summary

## Why this slice exists

The frontend contract now correctly treats reasoning as optional,
host-injected context. The remaining backend gap is execution-level
visibility: a completed step shows its runner result, but there is no
compact signal telling the UI whether that step was reasoning-capable
and whether a provider was attached.

## Scope (this PR)

1. Add a compact `reasoning` audit object to `ContentOpsStepExecution`
   for reasoning-capable outputs.
2. Keep the audit intentionally small: catalog requirement, service
   support for `with_reasoning_context`, and provider attachment.
3. Update the frontend contract to distinguish this audit from the
   full consumed reasoning payload.
4. Claim this slice in the coordination table while the PR is open.

### Files touched

- `extracted_content_pipeline/content_ops_execution.py`
- `tests/test_extracted_content_ops_execution.py`
- `docs/frontend/content_ops_frontend_contract.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Reasoning-Consumption-Summary.md`

## Mechanism

The executor computes the per-step audit from the output catalog and
the injected service object. Outputs with
`reasoning_requirement="optional_host_context"` get:

- `requirement`
- `service_supports_reasoning`
- `provider_configured`

Outputs whose catalog requirement is `absent` keep the prior payload
shape unless the service unexpectedly exposes reasoning support.

## Intentional

- No generator behavior changes.
- No UI drawer.
- No full reasoning payload exposure.
- No competitive-intelligence files touched.

## Deferred

- A future drawer-ready field carrying the consumed reasoning context
  itself.
- Frontend rendering for the execution-level reasoning audit.

## Verification

- `python -m py_compile extracted_content_pipeline/content_ops_execution.py`
- `python -m pytest tests/test_extracted_content_ops_execution.py`
- `git diff --check`

## Estimated diff size

- 5 files.
- About 120 inserted lines and 10 deleted lines.
- Well below the 400-line soft PR budget.
