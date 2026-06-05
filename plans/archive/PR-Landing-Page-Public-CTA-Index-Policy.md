# PR: Landing Page Public CTA Index Policy

## Why this slice exists

The landing-page publish contract says placeholder CTA URLs such as `#` and
`/demo` should fail the publish check unless the host has a real route. The
draft quality gate blocks obvious placeholders like `#`, but the public robots
policy can still index an approved, readiness-passing page whose CTA points to
`/demo`.

This slice makes the public index policy reject placeholder CTA destinations.

Ownership lane: content-ops/landing-page-public-index-policy

## Scope (this PR)

1. Add a public CTA URL indexability helper to `landing_page_export.py`.
2. Return `noindex,follow` from `public_landing_page_robots(...)` when the
   public CTA URL is missing, JavaScript, fragment-only, or `/demo`.
3. Keep draft generation and draft quality gates unchanged.
4. Update export tests so ready public pages use a real local route and
   placeholder public CTAs stay noindex.
5. Update public generated-asset API tests so sitemap and renderer fixtures use
   a real local route and `/demo` remains noindex.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Public-CTA-Index-Policy.md` | Plan doc for this public index-policy slice. |
| `extracted_content_pipeline/landing_page_export.py` | Add public CTA URL indexability check to robots policy. |
| `tests/test_extracted_landing_page_export.py` | Cover indexable CTA and placeholder CTA behavior. |
| `tests/test_extracted_content_asset_api.py` | Cover public API/sitemap behavior with the stricter CTA policy. |

## Mechanism

`public_landing_page_robots(...)` remains the publish-surface policy gate. It
still requires approved status and ready SEO/AEO/GEO checks, then adds one
public-only conversion-path check for the page-level CTA URL.

The helper intentionally lives in export/publish shaping rather than the draft
quality gate so operators can still draft or repair pages before a final public
destination is chosen.

## Intentional

- No generator prompt changes.
- No draft quality-gate changes.
- No public route changes.
- No assumptions about a complete frontend route registry beyond the known
  placeholder `/demo`.

## Deferred

- `HARDENING.md` still tracks landing-page repair legacy-lock rollout cleanup
  and repair lock connection hold time. Both are parked under
  `Owner/session: landing-page repair session` and are not required for this
  public CTA index-policy slice.

## Parked hardening

- None added.

## Verification

- Focused landing-page export and generated-asset API tests -> 67 passed.
- Local PR review -> passed.

## Estimated diff size

| File | Estimated LOC |
| --- | ---: |
| `extracted_content_pipeline/landing_page_export.py` | 25 |
| `tests/test_extracted_landing_page_export.py` | 25 |
| `tests/test_extracted_content_asset_api.py` | 15 |
| `plans/PR-Landing-Page-Public-CTA-Index-Policy.md` | 75 |
| **Total** | **140** |
