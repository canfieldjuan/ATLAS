# PR-Landing-Page-Index-Policy

## Why this slice exists

Approved generated landing pages can now render publicly, but the first public
renderer intentionally marks every page `noindex,follow`. This slice makes the
indexing decision backend-owned so search discoverability can be enabled only
when the approved page passes the existing landing-page SEO/AEO and GEO gates.

## Scope (this PR)

Ownership lane: content-ops/landing-page-index-policy

1. Add a public landing-page robots policy helper.
2. Keep the default public policy as `noindex,follow`.
3. Return `index,follow` only for approved landing pages whose SEO/AEO and GEO
   readiness summaries are both ready.
4. Include only the robots policy in the public payload, not the underlying
   readiness details.
5. Wire the public renderer to use the backend-provided robots policy.

### Files touched

- `plans/PR-Landing-Page-Index-Policy.md`
- `extracted_content_pipeline/landing_page_export.py`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/pages/PublicLandingPage.tsx`
- `tests/test_extracted_landing_page_export.py`
- `tests/test_extracted_content_asset_api.py`

## Mechanism

The public payload gets a `robots` field from the backend allowlist projection.
The helper evaluates the already-existing `_seo_aeo_readiness` and
`_geo_readiness` summaries and returns `index,follow` only when both are ready
and the draft status is approved. All other public pages remain
`noindex,follow`.

The frontend does not calculate readiness. It reads `robots` from the public
API response and falls back to `noindex,follow` if the field is missing.

## Intentional

- No sitemap or prerender inclusion in this slice.
- No public exposure of readiness summaries, checks, reference ids, metadata,
  tenant scope, generation telemetry, or reasoning fields.
- No change to draft/rejected/expired visibility; those still 404 at the
  public route.

## Deferred

- `PR-Landing-Page-Public-Sitemap` can add sitemap/prerender behavior after the
  indexing policy is merged and reviewed.

## Verification

- Focused backend tests for landing-page export and public asset API - 45 passed.
- Frontend lint in `atlas-intel-ui` - passed.
- Frontend production build in `atlas-intel-ui` - passed.
- Git whitespace check - passed.
- Full extracted pipeline checks through `scripts/run_extracted_pipeline_checks.sh`
  - 1651 passed.
- Local PR review wrapper - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~60 |
| Backend policy | ~25 |
| Frontend wiring | ~5 |
| Tests | ~110 |
| **Total** | **~200** |
