# PR-Generated-Asset-Repair-Summary

## Why this slice exists

PR #736 made landing-page repair history readable in the generated asset drawer,
but the queue row still hides that signal. Review also noted that the
repair-history fields compile only through `GeneratedAssetDraft`'s index
signature, so a field-name typo would silently render an empty panel.

This slice tightens the generated-asset wire type and adds a row-level repair
summary.

## Scope (this PR)

1. Declare generated asset repair-history fields on the TypeScript wire type.
2. Reuse the drawer repair-history parser for a compact row badge.
3. Keep the detailed per-attempt display in the drawer unchanged.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Generated-Asset-Repair-Summary.md` | Plan doc for this UI contract/readability slice. |
| `atlas-intel-ui/src/api/contentOps.ts` | Adds explicit generated-asset repair telemetry fields. |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | Adds row-level repair summary derived from parsed repair history. |

## Mechanism

`GeneratedAssetDraft` now declares the top-level repair telemetry fields emitted
by the backend:

- `generation_quality_repair_attempts`
- `generation_quality_repair_history`
- `quality_repair_history`
- `metadata`

The asset row calls the existing `assetRepairHistory(row)` helper and renders a
small status badge only when usable repair history is present. The badge reports
whether the final attempt passed or remained blocked, and whether repairs were
needed.

## Intentional

- No backend changes; the merged telemetry persistence already emits the data.
- No new filtering/export behavior; this is queue readability only.
- The drawer remains the source for detailed blockers and repair issues.

## Deferred

- A future generated-asset filtering/export slice can add repair-history CSV
  columns or filters if operators need queue-wide triage.

## Verification

- `npm run lint` from `atlas-intel-ui` -> passed.
- `npm run build` from `atlas-intel-ui` -> passed; Vite built successfully,
  generated the sitemap, and pre-rendered public routes.
- `git diff --check` -> passed with 0 whitespace errors.
- `bash scripts/local_pr_review.sh origin/main` -> passed all wrapper checks:
  pre-push audit wrapper, plan/code consistency, and `git diff --check`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~55 |
| Type/UI | ~70 |
| Total | ~125 |
