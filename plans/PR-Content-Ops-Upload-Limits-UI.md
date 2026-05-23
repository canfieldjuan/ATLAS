# PR-Content-Ops-Upload-Limits-UI

## Why this slice exists

The Content Ops new-run page lets operators upload CSV/JSON/JSONL ingestion
files, but the upload surface only says "Load JSON/JSONL/CSV". Operators have
to discover byte caps, row caps, source-text truncation, and inline deprecation
by trial, error, or backend errors.

PR #872 exposes those limits through the control-surface catalog. This slice
uses that catalog-backed contract in the UI so the upload surface displays the
real backend limits without hardcoded duplicate caps.

## Scope (this PR)

Ownership lane: content-ops/upload-limits-ui

1. Render a compact ingestion-limits summary in `ContentOpsNewRun`.
2. Source the display from `catalog.ingestionLimits`.
3. Align inspect/import request caps with `catalog.ingestionLimits` so the
   banner matches the actual submitted request.
4. Treat `auto` as server detection mode in the UI rather than listing it as a
   concrete file format.
5. Add small formatting helpers and a focused mapper test for the new
   catalog-backed limits.

### Files touched

- `plans/PR-Content-Ops-Upload-Limits-UI.md`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/package.json`
- `atlas-intel-ui/scripts/content-ops-ingestion-limits.test.mjs`

## Mechanism

The page already builds a domain catalog with `fromWireCatalog`. The new
`catalog.ingestionLimits` field contains:

- upload byte cap
- upload row cap
- supported upload formats
- deprecated inline row cap
- source text and sample caps

The ingestion inspector section will render those values near the file upload
controls. Formatting stays local to the page and only transforms numbers into
operator-readable text. The display filters `auto` out of the concrete format
list and renders it separately as automatic server detection.

The inspect/import handlers also submit `maxSourceTextChars` and
`maxSampleLimit` from the same catalog-backed limits. The separate local result
display truncation remains capped at 5 rows so large diagnostics do not flood
the page.

## Intentional

- This does not add client-side file rejection because the backend remains the
  canonical enforcement point.
- This does not remove the inline JSON textarea yet. Inline ingestion is marked
  deprecated in the displayed limits, but compatibility remains.
- This PR is stacked on #872 because it consumes the new catalog field.

## Deferred

- Future PR: remove inline compatibility after the operator compatibility
  window.
- Future PR: add client-side preflight rejection only if operators need faster
  feedback for large local files.
- Parked hardening: none.

## Verification

- Passed: UI build: `npm run build`.
- Passed: UI lint: `npm run lint`.
- Passed: ingestion-limits mapper test:
  `npm run test:content-ops-ingestion-limits` (`2 passed`).
- Passed: `git diff --check`.
- Passed: local PR review via `scripts/local_pr_review.sh`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| UI display, request caps + helpers | ~85 |
| Mapper test + npm script | ~55 |
| **Total** | **~220** |
