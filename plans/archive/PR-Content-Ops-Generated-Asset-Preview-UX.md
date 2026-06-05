# PR: Content Ops Generated Asset Preview UX

## Why This Slice Exists

Generated asset review works, but the current cards only expose a thin preview
for persisted report, blog post, landing page, and sales brief drafts. Operators
need enough context to approve or reject drafts without exporting CSV first.

## Scope

Improve the existing generated asset review card in
`atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`.

### Files Touched

- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`
- `atlas-intel-ui/src/api/contentOps.ts`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Generated-Asset-Preview-UX.md`

## Mechanism

- Add asset-specific preview metadata for counts, target context, topic/type
  fields, reasoning confidence, and generation usage.
- Declare the generated-asset export fields used by the preview cards on the
  wire draft type.
- Improve section and hero extraction for structured report, landing page, and
  sales brief rows.
- Keep the existing review/export API calls unchanged.

## Intentional

- No backend changes.
- No route or navigation changes.
- No new test framework setup.

## Deferred

- Dedicated component tests once the UI test harness exists.
- Full draft detail drawer.
- Inline content editing.

## Verification

- `atlas-intel-ui/package.json` build script.
- `scripts/local_pr_review.sh`.
- `git diff --check`.

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Review page preview helpers | ~125 |
| Wire type fields | ~10 |
| Coordination row | ~2 |
| Plan doc | ~55 |
| **Total** | ~192 |
