# PR-Landing-Page-Repair-UI-Polish

## Why this slice exists

The saved-draft repair UI is merged. Review flagged two non-blocking but useful
operator polish items: failed repairs currently surface raw JSON, and already
ready landing pages still show the Repair action even though the backend will
no-op.

This slice addresses those two points without changing the repair API contract
or the broader review drawer behavior.

## Scope (this PR)

Ownership lane: content-ops/landing-page-repair-ui-polish

1. Parse repair failure responses into operator-readable messages.
2. Include the first repair blockers in the drawer error when the backend
   returns a failed repair result.
3. Hide the Repair button when all landing-page readiness panels are already
   `ready`.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Repair-UI-Polish.md` | Plan doc for this polish slice. |
| `atlas-intel-ui/src/api/contentOps.ts` | Parse repair endpoint errors instead of showing raw JSON. |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | Hide Repair for already-ready landing pages. |

## Mechanism

`repairGeneratedLandingPageDraft` now handles the fetch directly so it can parse
the repair endpoint's structured error body. When the backend returns
`{detail: {message, repair_result}}`, the UI error becomes a short message plus
the first blocker labels.

The drawer computes repair availability from the same readiness panels it
already renders. If every panel is `ready`, Repair is hidden. Missing readiness
panels still allow Repair because the UI cannot prove the draft is ready.

## Intentional

- No backend changes.
- No new success toast or status surface.
- No attempt to parse every API error globally. This is scoped to the repair
  endpoint because it has a known structured failure shape.

## Deferred

- A broader API error-normalization pass can improve all frontend mutation
  wrappers later.
- A backend rate-limit/idempotency slice can add hard cost controls for repeated
  repair calls.

## Verification

- `npm run build` in `atlas-intel-ui` -> passed.
- `npm run lint` in `atlas-intel-ui` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~65 |
| API error parsing | ~45 |
| Repair button gate | ~10 |
| Total | ~120 |
