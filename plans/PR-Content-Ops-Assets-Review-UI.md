# PR: Content Ops generated assets review UI

## Plan

Expose the generated Content Ops asset review routes in the Atlas UI so an
operator can review real persisted outputs after execution.

## Why this slice exists

The backend mounts `/api/v1/content-assets/*`, but the frontend only shows
execution summaries. Operators need one protected screen to inspect generated
reports, blog posts, landing pages, and sales briefs, approve or reject drafts,
and export the current queue. The diff is above the normal target because the
slice is the smallest useful vertical UI: API wrappers, route wiring, and the
screen need to land together for the feature to be usable.

## Scope (this PR)

1. Add generated-asset API wrappers to the existing Content Ops adapter.
2. Add a protected generated-assets review page with asset/status filters,
   CSV export, and approve/reject actions.
3. Mount the page at `/content-ops/assets`.
4. Add B2B sidebar links for Content Ops execution and asset review.

### Files touched

- `atlas-intel-ui/src/App.tsx`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/components/Sidebar.tsx`
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Assets-Review-UI.md`

## Mechanism

The UI fetches `/api/v1/content-assets/{asset}/drafts`, uses the returned
`id` for `/drafts/review`, and downloads CSV through the existing export
route using authenticated fetch. Status `all` maps to an empty status query
because the backend already treats that as an all-status export.

## Intentional

- Keep the page wire-shaped and lightweight; no new domain mapper layer.
- Disable review actions when a row has no id instead of guessing a natural key.
- Keep CSV export authenticated through `fetch`, not `window.open`.

## Deferred

- Rich asset preview drawers for body/sections/content.
- Bulk approve/reject actions.
- Frontend unit tests for the page; current coverage is TypeScript build plus
  backend route/export tests.

## Verification

- Frontend production build in `atlas-intel-ui`: passed.
- Local PR review wrapper after rebasing onto the row-id prerequisite.

## Estimated diff size

| Area | Estimate |
|---|---:|
| API adapter | ~150 LOC |
| Review page | ~390 LOC |
| Route/sidebar wiring | ~10 LOC |
| Plan + coordination | ~55 LOC |
| Total | ~605 LOC |
