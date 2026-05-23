# PR-Content-Ops-Upload-Preflight

## Why this slice exists

The new-run page can now display the backend upload limits from the control
surface catalog, but selecting an oversized local file still marks it as
selected and waits for the server to reject it during inspect/import.

That is unnecessary work for operators and avoidable network load. This slice
uses the same catalog-backed limits to reject obviously oversized files in the
browser before they become the selected ingestion file.

## Scope (this PR)

Ownership lane: content-ops/upload-preflight

1. Add a small domain helper for ingestion file selection preflight.
2. Check selected file size against `catalog.ingestionLimits.fileUpload`.
3. Show the existing file-load error state when a file is too large.
4. Keep backend validation authoritative and unchanged.
5. Extend the ingestion-limits frontend test with preflight coverage.

### Files touched

- `plans/PR-Content-Ops-Upload-Preflight.md`
- `atlas-intel-ui/src/domain/contentOps/ingestionLimits.ts`
- `atlas-intel-ui/src/domain/contentOps/index.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/scripts/content-ops-ingestion-limits.test.mjs`

## Mechanism

`contentOpsIngestionFilePreflightError(file, limits)` returns a human-readable
error string when the selected file exceeds the catalog-backed byte cap. The
React file handler calls the helper before setting `selectedIngestionFile`.

The helper accepts a file-like `{ name, size }` object so the test can exercise
the behavior without a browser or React test harness.

## Intentional

- This only preflights file size. Row count and parse validity still require
  server inspection because the browser should not parse large operator files
  just to duplicate backend logic.
- This does not change inspect/import API calls or backend enforcement.
- This PR builds on #876's catalog-backed limits UI and keeps the same
  ingestion-limits frontend test as the focused coverage surface.

## Deferred

- Future PR: remove inline compatibility after the operator compatibility
  window.
- Future PR: add extension/MIME preflight only if operators repeatedly upload
  unsupported formats.
- Parked hardening: none.

## Verification

- Passed: ingestion-limits mapper/preflight test:
  `npm run test:content-ops-ingestion-limits` (`6 passed`).
- Passed: UI build: `npm run build`.
- Passed: UI lint: `npm run lint`.
- Passed: `git diff --check`.
- Passed: local PR review via `scripts/local_pr_review.sh`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Domain helper + export | ~30 |
| UI handler change | ~20 |
| Test additions | ~60 |
| **Total** | **~190** |
