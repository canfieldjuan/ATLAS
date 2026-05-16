# PR: Content Ops Report/Sales Reasoning Presets
## Why this slice exists
PR #554 added the preset catalog; this wires the first runtime consumer.
## Scope (this PR)
Add `reasoning_preset`, report/sales-brief plan metadata, and scoped
`multi_pass_structured` provider construction. Host providers still win.
### Files Touched
`docs/extraction/coordination/inflight.md` `extracted_content_pipeline/api/control_surfaces.py`
`extracted_content_pipeline/content_ops_execution.py` `extracted_content_pipeline/control_surfaces.py`
`extracted_content_pipeline/generation_plan.py` `extracted_content_pipeline/manifest.json`
`plans/PR-Content-Ops-Report-Sales-Reasoning-Presets.md` `tests/test_extracted_content_control_surface_api.py`
`tests/test_extracted_content_generation_plan.py` `tests/test_extracted_content_ops_execution.py`
## Mechanism
The API turns `multi_pass_structured` into a `MultiPassCampaignReasoningProvider`
only for configured `report` and `sales_brief` services.
## Intentional
No strict mode, cache/state store, blog/email/landing provider, or behavior
change when `reasoning_preset` is absent. Runtime structured report/brief
reasoning uses `L3`; the bare provider default stays `L2`.
## Deferred
Runtime `OutputPolicy`/`multi_pass_strict` stay deferred; validation flags are plan metadata here.
## Verification
 pytest focused generation-plan/execution/API suite -> 99 passed.
py_compile -> passed. git diff check -> passed.
ASCII grep on touched Python files -> passed. local PR review -> passed.
## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| **Total** | ~580 |
