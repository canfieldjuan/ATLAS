# PR-Blog-GEO-Quality-Gate

## Why this slice exists

PR-Blog-GEO-Readiness-Summary made draft-level GEO readiness visible in
generated-asset review/export rows. The next step is to stop saving newly
generated blog drafts that miss the core draft-level GEO contract.

This slice adds an opt-in GEO quality gate for extracted AI Content Ops blog
generation.

## Scope (this PR)

1. Add opt-in GEO draft validators to `extracted_quality_gate.blog_pack`.
2. Have `BlogPostGenerationService` opt generated blogs into the GEO gate.
3. Add quality-pack tests for passing and failing GEO draft checks.
4. Add generation tests proving GEO-incomplete drafts do not save.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-GEO-Quality-Gate.md` | Plan doc for this slice. |
| `extracted_quality_gate/blog_pack.py` | Add opt-in GEO draft blockers. |
| `extracted_content_pipeline/blog_generation.py` | Pass parsed/generated fields into the GEO gate. |
| `tests/test_extracted_quality_gate_blog_pack.py` | Cover GEO gate pass/fail behavior. |
| `tests/test_extracted_blog_generation.py` | Prove generated drafts block on missing GEO structure. |

## Mechanism

The quality pack remains pure and opt-in. Callers set `require_geo` in the
quality context when they want GEO blockers. The extracted blog generator sets
that flag because generated blog drafts are now expected to satisfy the draft
GEO contract before save.

The gate checks:

- `geo_entity_clarity_missing`
- `geo_answer_first_sections_missing`
- `geo_citable_section_structure_missing`
- `geo_evidence_specificity_missing`
- `geo_freshness_context_missing`
- `geo_faq_coverage_missing`
- `geo_citation_safety_failed`

## Intentional

- No publish-level checks. Crawler-visible HTML, canonical URLs, schema, OG
  images, and indexability belong in a frontend/public-route slice.
- No prompt changes.
- No claim that GEO guarantees AI-engine placement.
- Existing non-GEO direct quality-pack consumers remain unchanged unless they
  set `require_geo`.

## Deferred

- Add publish-level GEO verification.
- Add targeted repair-loop retry text for GEO blockers.
- Share helper logic with the export/readiness module if the contract grows.

## Verification

- Focused quality-pack and blog-generation tests passed.
- Touched-module Python compile check passed.
- Whitespace diff check passed.
- Extracted pipeline check suite passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~70 |
| Quality gate | ~180 |
| Blog generation context | ~10 |
| Tests | ~140 |
| **Total** | **~400** |
