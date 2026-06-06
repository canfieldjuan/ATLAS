# PR-Landing-Page-Saved-Draft-Repair-UI

## Why this slice exists

The saved-draft landing-page repair API is merged, but operators still have no
in-app action for using it. The review drawer already shows landing-page
readiness and repair history, so the natural next step is a focused repair
button there.

This slice keeps the UI change small: call the new repair endpoint from the
landing-page detail drawer, update the current row in-place, and let the
existing readiness and repair-history panels show the result.

## Scope (this PR)

Ownership lane: content-ops/landing-page-saved-draft-repair-ui

1. Add a frontend API wrapper for
   `POST /content-assets/landing_page/drafts/{id}/repair`.
2. Add a landing-page-only Repair action to the detail drawer.
3. Reuse the existing row busy state, action error handling, readiness panels,
   and repair-history display.
4. Keep approve/reject/edit behavior unchanged.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Saved-Draft-Repair-UI.md` | Plan doc for this UI slice. |
| `atlas-intel-ui/src/api/contentOps.ts` | Add repair result typing and repair API wrapper. |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | Add drawer repair action and row refresh handling. |

## Mechanism

The drawer shows Repair only for non-approved landing pages with an id. Clicking
it calls `repairGeneratedLandingPageDraft(id)`. On success, the returned review
row replaces the matching row in the current list and becomes the open detail
row. On failure, the existing page-level action error and drawer-local repair
error surfaces show the backend message.

## Intentional

- No table-row repair button in this slice. Keeping the mutation inside the
  detail drawer makes the operator inspect readiness before spending an LLM
  repair call.
- No polling. The backend route is synchronous and returns the repaired row.
- No new readiness rendering. Existing panels already consume the repaired row.

## Deferred

- Rate-limit/idempotency indicators can be added after backend rate limits
  exist.
- A future UX pass can add success toasts or inline repair result summaries if
  the drawer feels too quiet after repair.

## Verification

- `npm run build` in `atlas-intel-ui` -> passed.
- `npm run lint` in `atlas-intel-ui` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~65 |
| API wrapper/types | ~25 |
| Review drawer | ~60 |
| Total | ~150 |
