# Content Ops Strict Reasoning Policy

## Why this slice exists

PR #555 made packaged `multi_pass_structured` usable for report and sales
brief generation, but `multi_pass_strict` still failed at plan/runtime
boundaries even though the multi-pass provider already supports output
validation and blocking validation.

## Scope (this PR)

1. Allow report and sales brief requests to use `multi_pass_strict`.
2. Map packaged reasoning presets onto the existing `OutputPolicy` runtime
   knobs.
3. Keep `multi_pass_structured` nonblocking, and make `multi_pass_strict`
   fail closed via a strict wrapper that preserves validation blocker details.
4. Keep the catalog honest by leaving falsification disabled until host-owned
   rules are wired.

### Files touched

- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/reasoning_policy.py`
- `plans/PR-Content-Ops-Strict-Reasoning-Policy.md`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_reasoning_policy.py`
- `tests/test_extracted_report_generation.py`
- `tests/test_extracted_sales_brief_generation.py`

## Mechanism

Update generation-plan preset validation, API provider construction, and
focused tests for report/sales strict reasoning. The API still builds only
host-injected LLM ports; no database, provider routing, or storage surface is
added.

## Intentional

No falsification policy is auto-created. The provider supports it, but the
control-surface request has no host-owned falsification rules to pass yet.

## Deferred

Host-configured falsification rules remain separate product-policy work.

## Verification

pytest focused generation-plan/API/policy/report/sales suite -> 132 passed.
py_compile -> passed. git diff check -> passed. ASCII grep on touched Python
files -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| **Total** | ~370 |
