# PR-Landing-Page-Failure-Diagnostics

Ownership lane: `content-ops/landing-page-live-smoke`

## Why this slice exists

The Haiku landing-page live smoke can fail the save path because the landing
quality gate blocks on `geo_readiness:section_semantics`. The blocked result
currently reports only the missing check and repair history, not the candidate
section metadata/body shape that caused the miss. A no-gate inspection run
showed another candidate can pass readiness, so changing prompts without seeing
the failed candidate would be guesswork.

## Scope (this PR)

1. Add a bounded landing-page failed-candidate snapshot for parsed candidates
   that fail quality after all repair attempts.
2. Include the same snapshot when a repair response is unparseable after a
   previous parsed candidate existed.
3. Cover the diagnostics with focused landing-page generation tests.

### Files touched

- `extracted_content_pipeline/landing_page_generation.py`
- `tests/test_extracted_landing_page_generation.py`
- `plans/PR-Landing-Page-Failure-Diagnostics.md`

## Mechanism

The generator already carries the last parsed landing-page JSON through the
quality and repair loop. This PR adds a small helper that extracts safe, bounded
fields from that parsed object: title, slug, hero headline, CTA, meta title,
section count, section kinds/titles/questions, answer summary lengths, body
starts, and repair attempt metadata. Quality-blocked and repair-unparseable
errors include that snapshot.

## Intentional

- The snapshot is diagnostic only; it does not relax any quality gate or save
  blocked drafts.
- Body excerpts are capped so failed live outputs do not dump whole landing-page
  drafts into operational logs.
- No prompt changes in this slice. The live failure needs better observability
  before we change the generation contract.

## Deferred

- Parked hardening: none.
- Fixing the actual `geo_readiness:section_semantics` recurrence is the next
  slice once diagnostics show the offending section shape.

## Verification

- `pytest tests/test_extracted_landing_page_generation.py -q` - 38 passed.
- Live Haiku landing-page smoke with quality gates enabled - passed and returned saved draft id `f83fab79-762d-46d6-885a-37f5df5eb1e4`.
- Extracted content pipeline validation wrapper - passed.
- Extracted reasoning-import guard - passed.
- Extracted standalone audit with debt failure enabled - passed.
- Extracted Python ASCII check - passed.
- Extracted content pipeline sync wrapper - passed.

## Estimated diff size

| Area | Estimate |
|---|---:|
| Landing-page generator diagnostics | ~80 LOC |
| Focused tests | ~55 LOC |
| Plan doc | ~60 LOC |
| **Total** | **~195 LOC** |
