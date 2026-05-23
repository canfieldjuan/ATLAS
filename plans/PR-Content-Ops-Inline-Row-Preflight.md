# PR-Content-Ops-Inline-Row-Preflight

## Why this slice exists

The Content Ops new-run page still supports deprecated inline JSON rows for
manual/pasted ingestion. The backend enforces the inline row cap from the
control-surface catalog, but the UI currently parses and submits oversized
inline payloads before the server rejects them.

After #880, uploaded files get catalog-backed preflight for obvious local
limits. This slice applies the same pattern to inline rows without touching the
file-upload path or backend behavior.

## Scope (this PR)

Ownership lane: content-ops/inline-row-preflight

1. Add a small domain helper for inline row-count preflight.
2. Check parsed inline rows against `catalog.ingestionLimits.inlineRows.maxRows`.
3. Use the existing invalid-input states for inspect/import when inline rows
   exceed the cap.
4. Keep backend validation authoritative and unchanged.
5. Extend the ingestion-limits frontend test with inline row-count coverage.

### Files touched

- `plans/PR-Content-Ops-Inline-Row-Preflight.md`
- `atlas-intel-ui/src/domain/contentOps/ingestionLimits.ts`
- `atlas-intel-ui/src/domain/contentOps/index.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/scripts/content-ops-ingestion-limits.test.mjs`

## Mechanism

`contentOpsInlineRowsPreflightError(rowCount, limits)` returns a readable error
when a parsed inline JSON payload exceeds the catalog-backed inline row cap.
`ContentOpsNewRun` calls it after JSON parsing and before inspect/import
submission.

The helper fails open if either the row count or cap is non-finite. That keeps
the backend as the final authority if the catalog shape is ever unavailable or
malformed.

## Intentional

- This does not revive or expand inline ingestion; the path remains deprecated.
- This only preflights parsed inline row count. File row count and parse
  validity still belong to server-side inspection.
- This does not change API requests or backend enforcement.

## Deferred

- Future PR: remove inline compatibility after the operator compatibility
  window.
- Future PR: add extension/MIME preflight only if unsupported uploads become
  common.
- Parked hardening: none.

## Verification

- Passed: ingestion-limits mapper/preflight test:
  `npm run test:content-ops-ingestion-limits` (`10 passed`).
- Passed: UI build: `npm run build`.
- Passed: UI lint: `npm run lint`.
- Passed: `git diff --check`.
- Passed: local PR review via `scripts/local_pr_review.sh`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Domain helper + export | ~30 |
| UI handler change | ~25 |
| Test additions | ~45 |
| **Total** | **~180** |
