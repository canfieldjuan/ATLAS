# PR-Blog-Readiness-Review-UI

## Why this slice exists

Generated blog draft rows now expose `seo_aeo_readiness` and `geo_readiness`
from the extracted content pipeline, but the Atlas Intel review UI still only
shows the older generic output checks. That forces operators to open the raw row
JSON to see whether a blog draft is SEO/AEO-ready or GEO-ready.

This slice surfaces those existing readiness summaries in the generated asset
review UI.

## Scope (this PR)

1. Add typed readiness fields to `atlas-intel-ui/src/api/contentOps.ts`.
2. Show SEO/AEO and GEO readiness facts for blog post rows.
3. Include the same readiness labels in the blog preview metadata.
4. Keep backend, generation, and quality-gate behavior unchanged.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-Readiness-Review-UI.md` | Plan doc for this slice. |
| `atlas-intel-ui/src/api/contentOps.ts` | Type generated-asset readiness fields. |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | Surface blog SEO/AEO and GEO readiness in review cards and details. |

## Mechanism

The review page already renders compact asset facts on both the list rows and
detail drawer. Blog rows now parse the existing readiness objects and add
human-readable SEO/AEO and GEO facts, including pass counts and the first missing
checks when the status is `needs_review`.

Preview metadata uses the same labels so readiness is visible before opening the
detail drawer.

## Intentional

- No backend changes.
- No generated-asset API contract changes.
- No changes to approval or rejection behavior.
- No attempt to display every individual readiness check in this slice.

## Deferred

- Add a fuller readiness breakdown panel if operators need to inspect every
  passing and missing check without using the raw row.
- Add UI component tests if the Atlas Intel UI gets a test runner.

## Verification

- Atlas UI lint passed.
- Atlas UI production build passed.
- Blog GEO prerender verification passed.
- Whitespace diff check passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~55 |
| API type shape | ~10 |
| Review UI helper/rendering | ~55 |
| Total | ~120 |
