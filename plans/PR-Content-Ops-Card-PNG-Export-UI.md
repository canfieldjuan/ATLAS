# PR: Content Ops Card PNG Export UI

## Why this slice exists

PR #1300 landed the backend `format=png` contract for quote/stat card visual
exports after #1303 proved the live browser path. The remaining deferred
product step is the atlas-intel-ui action that lets marketers download those
PNG images from the generated-assets review page instead of stopping at CSV or
HTML.

This slice keeps the UI change narrow: wire the existing quote/stat card
visual-export controls to the new PNG format, preserve the existing HTML export
action for review/debugging, and prove both card types through the already
enrolled frontend tests.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input-card-png-ui
Slice phase: Product polish

1. Add a typed frontend API helper for
   `GET /content-assets/{asset}/drafts/export?format=png`.
2. Add an `Export PNG` action on the generated-assets review toolbar for
   `quote_card` and `stat_card`, reusing the same filters and busy/error state
   as CSV/HTML exports.
3. Keep `Export HTML` available for quote/stat cards as a review/debug artifact.
4. Extend the existing quote-card and stat-card review-asset tests to cover the
   PNG helper route and UI wiring.

### Files touched

- `plans/PR-Content-Ops-Card-PNG-Export-UI.md`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`
- `atlas-intel-ui/scripts/content-ops-quote-card-review-assets.test.mjs`
- `atlas-intel-ui/scripts/content-ops-stat-card-review-assets.test.mjs`

## Mechanism

`contentOps.ts` already has text helpers for CSV and HTML exports. This slice
adds a binary helper that calls the same asset export route with `format=png`
and returns a `Blob`:

```ts
exportGeneratedAssetDraftsPng(asset, params)
```

`ContentOpsAssetsReview` then calls that helper from a new `Export PNG` button
shown only when `assetSupportsVisualExport(asset)` is true. The download uses
the browser's object URL path and a `.png` filename, while errors continue to
surface through the existing `actionError` banner.

The test changes stay inside the already enrolled quote/stat card review-asset
scripts. They assert that the API helper requests `format=png`, that the UI
source imports/calls the PNG helper, and that the `Export PNG` control is
present for the card assets.

## Intentional

- No backend changes; #1300 is the accepted PNG contract.
- No new test script. The quote/stat review-asset tests are already listed in
  `.github/workflows/atlas_intel_ui_checks.yml`, which avoids the frontend CI
  enrollment gap.
- No id-filter support for quote/stat exports; this continues the #1300
  deferral.
- No live browser rerun in this UI slice; #1303 already captured the live
  Chromium proof and #1300 added the backend contract tests.

## Deferred

- Optional quote/stat id deep links remain deferred until a product path needs
  exact run-result links.
- Product copy/toolbar grouping polish can follow after the action is available
  to operators.

## Parked hardening

None.

## Verification

Ran:

- `cd atlas-intel-ui && npm run test:content-ops-quote-card-review-assets`
  - 6 passed.
- `cd atlas-intel-ui && npm run test:content-ops-stat-card-review-assets`
  - 6 passed.
- `cd atlas-intel-ui && npm run build`
  - passed; Vite built 2,418 modules, generated sitemap, and pre-rendered 16
    public routes.
- `git diff --check`
  - passed.

Still to run before push:

- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-content-ops-card-png-export-ui.md`

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan | 105 |
| API + UI | 83 |
| Existing frontend tests | 72 |
| **Total** | **260** |
