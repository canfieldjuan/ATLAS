# PR: Content Ops Card Visual Export UI

## Why this slice exists

PR #1297 added the backend `format=html` contract for visual quote/stat card
exports, but atlas-intel-ui still exposes only the CSV export action on the
generated-assets review screen. Operators can now call the HTML artifact route
directly, but they cannot download those visual cards from the UI where quote
and stat drafts are reviewed.

This slice completes the UI handoff deferred by #1297: quote-card and
stat-card review tabs get a visible HTML export action that calls the existing
tenant-scoped export endpoint.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Product polish

1. Add a typed `exportGeneratedAssetDraftsHtml(...)` frontend API helper that
   calls `/content-assets/{asset}/drafts/export?format=html`.
2. Add an `Export HTML` action on the generated-assets review screen only for
   `quote_card` and `stat_card`, using the same status/limit filters as CSV.
3. Reuse the existing download helper shape with an HTML MIME type and a
   deterministic filename.
4. Extend the already-enrolled quote-card and stat-card review asset tests to
   prove route shape, UI gating, and visible action text.

### Files touched

- `plans/PR-Content-Ops-Card-Visual-Export-UI.md`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`
- `atlas-intel-ui/scripts/content-ops-quote-card-review-assets.test.mjs`
- `atlas-intel-ui/scripts/content-ops-stat-card-review-assets.test.mjs`

## Mechanism

The API helper mirrors the existing CSV helper:

```ts
export function exportGeneratedAssetDraftsHtml(asset, params) {
  return getAssetText(asset, '/drafts/export', { ...params, format: 'html' })
}
```

The review screen keeps CSV available for every generated asset. A small
`assetSupportsVisualExport(...)` predicate gates the HTML action to
`quote_card` and `stat_card`, matching the backend allowlist from #1297. The
handler downloads the returned static HTML with the same current `params`
object used for CSV export, so the UI does not create a new filter surface or
scope path.

## Intentional

- No PNG/SVG/server-side image export. This UI calls the HTML artifact contract
  accepted in #1297.
- No new frontend workflow step. The changed quote/stat test scripts are
  already listed in `.github/workflows/atlas_intel_ui_checks.yml`.
- No #1268 output-variations work.
- No id-filter support for quote/stat visual exports; the current review tabs
  do not support id filters for those assets.

## Deferred

- PNG/server-side image export remains a later product-polish slice after the
  HTML artifact is used from the UI.
- Optional quote/stat id deep links remain deferred until a product path needs
  exact run-result links.

## Parked hardening

None.

## Verification

- Passed: `cd atlas-intel-ui && npm run test:content-ops-quote-card-review-assets`
  (5 passed)
- Passed: `cd atlas-intel-ui && npm run test:content-ops-stat-card-review-assets`
  (5 passed)
- Passed: `cd atlas-intel-ui && npm run lint`
- Passed: `cd atlas-intel-ui && npm run build`
- Passed: `git diff --check`
- Passed: `rg -n "test:content-ops-(quote-card|stat-card)-review-assets" atlas-intel-ui/package.json .github/workflows/atlas_intel_ui_checks.yml`
  (both modified test scripts are already enrolled in the frontend CI workflow)

## Estimated diff size

Actual git diff: 5 files, +208 / -3.

| Area | LOC |
|---|---:|
| **Total** | **211** |
