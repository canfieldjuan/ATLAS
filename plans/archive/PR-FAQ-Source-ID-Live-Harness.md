# PR-FAQ-Source-ID-Live-Harness

## Why this slice exists

#1121 proved `inputs.source_faq_ids` reaches landing/blog execute context through
the route and input-provider repository seam, and #1123 proved the SQL-backed
`get_draft(...)` lookup is tenant-scoped against Postgres. The remaining thin
validation gap is the real execute harness: selected saved FAQ IDs should feed
the actual landing/blog generation services, not only capture-service fakes.

This slice closes that gap without touching FAQ generation or adding a hosted
runbook artifact.

## Scope (this PR)

Ownership lane: content-ops/faq-output-ingestion

Slice phase: Functional validation

1. Add live execute-harness coverage for `inputs.source_faq_ids`.
2. Use the host input-provider repository factory seam to load a saved FAQ draft
   by ID.
3. Execute the real landing-page and blog-post generation services with the
   existing deterministic LLM test harness.
4. Assert selected FAQ wording and resolution evidence reach saved drafts and
   LLM prompts through the normal `/content-ops/execute` route.

### Files touched

- `tests/test_extracted_content_ops_live_execute_harness.py`
- `plans/PR-FAQ-Source-ID-Live-Harness.md`

## Mechanism

The test mounts `create_content_ops_control_surface_router(...)` with:

- `input_provider=build_content_ops_input_provider(...)`
- a fake pool-backed FAQ repository factory implementing `get_draft(...)`
- real `LandingPageGenerationService` and `BlogPostGenerationService`
- the existing deterministic LLM and in-memory draft repositories from the live
  execute harness

The payload names only `inputs.source_faq_ids`; no inline `source_material` is
provided. For each output, the route must load the selected FAQ draft under the
tenant scope, normalize it as FAQ-derived support-ticket source material, and
execute the real service.

## Intentional

- This is test-only functional validation. It does not change runtime behavior.
- The repository is fake because #1123 already proves the real Postgres
  repository isolation boundary; this slice proves the live execute harness
  consumes that boundary correctly.
- This does not add the hosted/live runbook artifact yet. That remains useful
  only if operators need an environment-recorded proof outside the test suite.

## Deferred

- Future PR: hosted/live execute runbook artifact if operators need a recorded
  environment proof that combines route execution and live Postgres in one
  smoke.
- Future PR: richer saved-FAQ picker with search/status filters if operators
  need more than the recent list.
- Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_extracted_content_ops_live_execute_harness.py::test_selected_faq_id_feeds_real_landing_and_blog_generation -q
  - Result: 1 passed.
- Command: python -m pytest tests/test_extracted_content_ops_live_execute_harness.py tests/test_support_ticket_provider_landing_blog_execute.py -q
  - Result: 23 passed.
- Command: python -m py_compile tests/test_extracted_content_ops_live_execute_harness.py
  - Result: passed.
- Command: git diff --check
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Live harness test | ~160 |
| Plan doc | ~80 |
| **Total** | **~240** |
