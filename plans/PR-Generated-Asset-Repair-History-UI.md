# PR-Generated-Asset-Repair-History-UI

## Why this slice exists

PR #733 persisted landing-page quality repair telemetry on generated asset
drafts, but reviewers still have to inspect the raw row JSON to understand what
happened during each repair attempt. That hides useful review context at the
bottom of the detail drawer.

This slice makes the persisted repair history visible in the generated asset
review UI.

## Scope (this PR)

1. Parse generated asset repair history from the draft row and metadata fallbacks.
2. Render a compact repair-history section in the asset detail drawer when
   repair history is present.
3. Keep the raw row diagnostic dump available for deeper operator debugging.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Generated-Asset-Repair-History-UI.md` | Plan doc for this UI slice. |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | Adds repair-history parsing and drawer rendering. |

## Mechanism

The generated asset review drawer already normalizes optional readiness data
before rendering it. This PR follows that pattern with a small
`assetRepairHistory` parser that accepts array or JSON-stringified values from:

- `generation_quality_repair_history`
- `quality_repair_history`
- the same keys inside `metadata`

When entries exist, the drawer shows each attempt with pass/blocked status,
blockers, and repair issues. Empty arrays are omitted so drafts without repair
telemetry keep the existing layout.

## Intentional

- No backend/API changes; PR #733 already persists the data.
- No changes to generation behavior or repair caps.
- No attempt to hide the raw row because it remains useful for operator
  diagnostics.

## Deferred

- A future generated-asset filtering/export slice can add repair-history
  columns or filters if operators need queue-level triage by repair attempts.
- A future component test slice can cover the drawer once this page has an
  established UI test harness.

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
| UI/parser | ~120 |
| Total | ~175 |
