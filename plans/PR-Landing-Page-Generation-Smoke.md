# PR: Landing Page Generation Smoke

## Why this slice exists

The landing-page work now has SEO/AEO/GEO inputs, prompt threading, save-time
readiness checks, review/edit/repair flows, public rendering, sitemap/index
policy, and repair-lock hardening. Those slices each shipped focused tests, but
we still need one thin end-to-end smoke that proves the normal operator path
works as a chain: Content Ops inputs reach the real landing-page generator,
readiness failures trigger the repair loop, the repaired draft exports as ready,
and the public robots policy only indexes the approved ready page with a real
CTA URL.

Ownership lane: content-ops/landing-page-generation-smoke

## Scope (this PR)

1. Add one focused landing-page generation smoke test using the real
   `execute_content_ops_from_mapping` dispatcher and
   `LandingPageGenerationService`.
2. Use fake LLM/skills/storage ports only at the external boundaries so the
   smoke stays deterministic and offline.
3. Verify SEO/AEO/GEO operator inputs are present in the generated prompt
   payload.
4. Verify a first draft missing readiness is repaired, persisted, exported with
   ready SEO/AEO and GEO summaries, and indexable only after approval with a
   non-placeholder CTA.
5. Enroll the smoke in extracted CI so future landing-page changes run it.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Generation-Smoke.md` | Plan doc for this smoke slice. |
| `tests/test_extracted_landing_page_generation_smoke.py` | Adds the end-to-end landing-page generation smoke. |
| `scripts/run_extracted_pipeline_checks.sh` | Runs the new smoke in extracted pipeline CI. |
| `.github/workflows/extracted_pipeline_checks.yml` | Triggers extracted CI when landing-page test files change. |

## Mechanism

The smoke test builds a real `LandingPageGenerationService` with a fake
repository, LLM, and skill store. It then calls
`execute_content_ops_from_mapping(...)` with `outputs=["landing_page"]` and
real landing-page SEO/AEO/GEO input keys.

The fake LLM returns two parsed JSON payloads: first a structurally valid draft
missing the meta description, then a repaired draft. The service's existing
readiness gate should reject the first payload with
`seo_aeo_readiness:meta_description`, run the repair attempt, and save the
second payload. The test then shapes that saved draft through
`landing_page_draft_export_row(...)` and `public_landing_page_robots(...)`.

## Intentional

- No live LLM, database, or browser dependency. This is a wiring smoke, not a
  provider smoke.
- No new production behavior. The slice only adds coverage over the current
  flow.
- The smoke checks one representative readiness miss rather than every
  validator; individual validators already have dedicated tests.

## Deferred

- Parked hardening: none. Root `HARDENING.md` has no landing-page parked items.
- Blog/deep-dive parked items in `ATLAS-HARDENING.md` are outside this
  landing-page lane.

## Verification

- pytest tests/test_extracted_landing_page_generation_smoke.py -q -> 1 passed.
- pytest tests/test_extracted_landing_page_generation.py tests/test_extracted_landing_page_generation_smoke.py tests/test_extracted_landing_page_export.py tests/test_extracted_content_asset_api.py -q -> 116 passed.
- bash scripts/validate_extracted_content_pipeline.sh -> passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -> passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt -> passed.
- python -m py_compile tests/test_extracted_landing_page_generation_smoke.py -> passed.
- bash scripts/run_extracted_pipeline_checks.sh -> 1780 passed, 1 skipped.

## Estimated diff size

| File | Estimated LOC |
| --- | ---: |
| `plans/PR-Landing-Page-Generation-Smoke.md` | 80 |
| `tests/test_extracted_landing_page_generation_smoke.py` | 255 |
| `scripts/run_extracted_pipeline_checks.sh` | 5 |
| `.github/workflows/extracted_pipeline_checks.yml` | 5 |
| **Total** | **345** |
