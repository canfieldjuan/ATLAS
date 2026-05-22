# PR-Landing-Page-Readiness-Validator-Unification

## Why this slice exists

Landing-page SEO/AEO and GEO readiness is currently computed inside
`landing_page_export.py`. Export rows and public robots use that logic, but
save-time generation only runs the structural landing-page quality pack. That
leaves two separate notions of "ready": a draft can pass generation and persist
even when the export/public readiness checks would immediately mark it
`needs_review`.

This slice makes the landing-page readiness scorer a shared helper and wires it
into the generation quality gate so export, public robots, and save-time repair
use the same source for SEO/AEO/GEO readiness checks.

This PR is intentionally over the normal 400 LOC target because the clean
source-level fix moves the existing readiness helper out of export instead of
copying it into generation. The large diff is mostly code movement plus tests
that pin export parity and generation-time blocking.

## Scope (this PR)

Ownership lane: content-ops/landing-page-readiness-validator

1. Move the SEO/AEO/GEO readiness scoring code out of
   `landing_page_export.py` into `landing_page_readiness.py`.
2. Keep export row shape, public robots behavior, and readiness payload shape
   unchanged.
3. Make `LandingPageGenerationService` run the shared readiness scorer after
   the existing structural quality pack.
4. Feed missing readiness checks into the existing quality repair loop so the
   model gets actionable repair blockers.
5. Add focused tests for the shared helper and generation-time readiness
   blocking/repair behavior.

### Files touched

- `plans/PR-Landing-Page-Readiness-Validator-Unification.md`
- `extracted_content_pipeline/landing_page_readiness.py`
- `extracted_content_pipeline/landing_page_export.py`
- `extracted_content_pipeline/landing_page_generation.py`
- `tests/test_landing_page_readiness.py`
- `tests/test_extracted_landing_page_export.py`
- `tests/test_extracted_landing_page_generation.py`

## Mechanism

`landing_page_readiness.py` owns three public functions:

- `landing_page_seo_aeo_readiness(draft)`
- `landing_page_geo_readiness(draft)`
- `landing_page_readiness_repair_issues(draft)`

Export uses the first two functions directly for rows and public robots.
Generation builds a provisional `LandingPageDraft` from the parsed model output
using the same `_build_draft` path it uses before persistence, then runs
`landing_page_readiness_repair_issues`. Missing SEO/AEO/GEO checks are surfaced
as deterministic repair issues with stable prefixes, e.g.
`seo_aeo_readiness:meta_description` and `geo_readiness:claim_safety`.

The existing structural quality pack remains load-bearing for hard schema and
safety blockers such as missing CTA, invalid slug, unresolved placeholders, and
blocked phrasing. The shared readiness scorer adds the export/public readiness
contract to save-time validation without moving that logic into the
`extracted_quality_gate` package in this slice.

This changes operator-facing behavior: some landing pages that previously
persisted as drafts with `seo_aeo_readiness` or `geo_readiness` set to
`needs_review` will now return `quality_blocked` if the default repair attempt
does not clear every readiness issue. That increase is expected. It means
generation is stopping earlier instead of saving drafts that public robots and
sitemap publishing would later reject. Hosts that need the previous advisory
behavior can still pass `quality_gates_enabled=False`; hosts that want more
repair headroom can raise `landing_page_quality_repair_attempts` up to the
existing cap.

## Intentional

- No readiness payload shape changes. Existing export/UI consumers continue to
  receive `status`, `passed`, `total`, `missing`, and `checks`.
- No public route or UI changes.
- No attempt to delete the structural quality pack. It still catches schema and
  safety issues before readiness scoring runs.
- Readiness issues are repair issues, not a new exception type.
- No default repair-attempt change in this slice. The current default of 1
  preserves the cost model and avoids silently increasing generation spend.
  The existing per-run `landing_page_quality_repair_attempts` input remains the
  operator control for drafts that need more repair passes.

## Deferred

- `PR-Landing-Page-Review-Repair-Action` should expose operator-triggered
  repair for saved drafts.
- `PR-Landing-Page-Editable-Drafts` should add PATCH + structured edit UI.
- `PR-Landing-Page-Public-Prerender` remains later after generation and saved
  draft readiness behavior are consistent.

## Verification

- Passed: `pytest tests/test_landing_page_readiness.py tests/test_extracted_landing_page_export.py tests/test_extracted_landing_page_generation.py -q` - 44 passed.
- Passed: Python compile check for the changed modules and tests.
- Passed: `git diff --check`.
- Passed: local PR review wrapper.

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Plan | ~100 |
| Readiness helper move | ~740 |
| Export/generation wiring | ~60 |
| Tests | ~275 |
| **Total** | **~1175** |
