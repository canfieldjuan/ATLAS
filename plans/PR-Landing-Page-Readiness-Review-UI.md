# PR-Landing-Page-Readiness-Review-UI

## Why this slice exists

PR #710 added landing-page SEO/AEO and GEO readiness summaries to generated
asset rows, and PRs #711, #712, and #717 made the generator respect those
checks more directly. The review UI still only renders the full readiness
breakdown for blog posts, so landing-page reviewers can see compact facts but
must inspect the raw JSON to understand which checks passed or failed.

This slice closes that UI gap without changing generation, persistence, export,
or review behavior.

## Scope (this PR)

1. Reuse the existing readiness breakdown drawer section for landing pages.
2. Keep blog-post readiness rendering unchanged.
3. Add landing-page SEO/AEO and GEO compact facts beside campaign and persona
   facts.
4. Rename the blog-specific helper/component names so the UI reflects the shared
   readiness surface.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Readiness-Review-UI.md` | Plan doc for this UI slice. |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | Render readiness panels for landing pages as well as blog posts. |

## Mechanism

The detail drawer already parses `seo_aeo_readiness` and `geo_readiness` into
`ReadinessPanel` objects for blog posts. This PR promotes that helper from a
blog-only function to an asset-aware function and returns panels for both
`blog_post` and `landing_page` rows.

The rendered card markup stays the same: status, pass count, missing checks,
and per-check pass/fail rows. Landing pages also show SEO/AEO and GEO facts in
the compact Facts block so operators can scan the queue and inspect the detail
drawer without opening Raw Row.

## Intentional

- No backend or API changes; the required fields already ship on generated
  asset rows.
- No new generator, quality-gate, export, approval, or rejection behavior.
- No readiness panels for reports, sales briefs, or FAQ Markdown until those
  asset types have first-class readiness contracts.
- No UI component tests because the Atlas Intel UI does not currently include a
  component test runner.

## Deferred

- PR-Landing-Page-Publish-Verification should add public-rendered URL checks
  once generated landing pages have a concrete hosted publish route.
- PR-Generated-Asset-Readiness-Shared-Component can extract the readiness
  breakdown if more generated asset types adopt the same contract.

## Verification

- `npm ci` in `atlas-intel-ui` -> completed; existing dependency audit reported
  6 vulnerabilities.
- `npm run lint` in `atlas-intel-ui` -> passed.
- `npm run build` in `atlas-intel-ui` -> passed.
- `git diff --check` -> passed with 0 whitespace errors.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~70 |
| Review UI | ~20 |
| Total | ~90 |
