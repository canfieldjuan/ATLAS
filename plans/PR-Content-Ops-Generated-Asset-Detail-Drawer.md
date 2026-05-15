# PR: Content Ops Generated Asset Detail Drawer

## Why This Slice Exists

The generated asset review cards now show richer summaries, but operators still
need a way to inspect full sections, references, and raw persisted fields before
approving or rejecting a draft.

## Scope

Add an in-page detail drawer to
`atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`.

### Files Touched

- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Generated-Asset-Detail-Drawer.md`

## Mechanism

- Add a view-details action on each generated asset row.
- Render a right-side drawer with the existing preview, facts, parsed sections,
  reference ids, and raw row JSON.
- Reuse the existing row helpers; keep review/export API calls unchanged.

## Intentional

- No backend changes.
- No new route.
- No inline editing.

## Deferred

- Dedicated frontend test harness.
- Markdown rendering for full draft bodies.
- Field-specific editors.

## Verification

- `atlas-intel-ui/package.json` build script.
- `scripts/local_pr_review.sh`.
- `git diff --check`.

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Detail drawer UI | ~150 |
| Coordination row | ~2 |
| Plan doc | ~55 |
| **Total** | ~207 |
