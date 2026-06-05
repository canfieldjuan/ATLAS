# Content Ops Blog Narrative Pack

## Why this slice exists

The reasoning policy audit called out blog posts as the next long-form asset
after report and sales brief. Blog posts can already consume reasoning context,
but packaged structured reasoning still rejects them at the plan and execute
surfaces.

## Scope (this PR)

1. Add `blog_post` to the packaged structured reasoning runtime surface.
2. Keep `multi_pass_strict` unavailable for blog posts until a blog-specific
   blocking policy exists.
3. Add a host-configurable output-to-pack mapping with a blog-safe default.
4. Group packaged reasoning providers by pack name so mixed output requests do
   not reuse the blog pack for report or sales brief.
5. Remove the merged #560 row from the coordination ledger and claim this PR.
6. Refresh status/backlog docs for the shipped blog narrative seam.

### Files touched

- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/reasoning_policy.py`
- `plans/PR-Content-Ops-Blog-Narrative-Pack.md`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_reasoning_policy.py`

## Mechanism

`blog_post` joins packaged runtime only for `multi_pass_structured`. The API
router builds one provider group for blog output using the configured blog
pack, defaulting to `content_ops_blog`, and keeps report/sales brief on the
existing `content_ops_structured` pack.
Execution-service reasoning metadata now accumulates targeted output bindings
so mixed requests can report every provider-bound output correctly.

## Intentional

Strict blog reasoning remains blocked. Blog output validation should stay soft
until a concrete blog-specific blocking policy ships.

## Deferred

Strict blog reasoning remains deferred until a blog-specific blocking policy is
defined. Landing-page narrative planning remains opt-in host wiring, not a
packaged runtime default.

## Verification

pytest tests/test_extracted_content_reasoning_policy.py
tests/test_extracted_content_generation_plan.py
tests/test_extracted_content_control_surface_api.py
tests/test_extracted_content_ops_execution.py -> 148 passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| **Total** | ~620 |
