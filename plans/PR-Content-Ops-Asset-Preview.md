# PR: Content Ops asset previews

## Why this slice exists

After generated asset review and batch status updates, operators can approve or
reject persisted assets but still see only metadata-level rows. The active AI
Content Ops backlog calls out richer generated-asset previews as the next
operator review UX improvement.

## Scope (this PR)

1. Add compact preview data fields to the generated asset wire type.
2. Render per-row preview snippets for reports, blog posts, landing pages, and
   sales briefs using fields already returned by the existing list endpoint.
3. Replace the merged batch-review coordination claim with this slice.

### Files touched

- `plans/PR-Content-Ops-Asset-Preview.md`
- `docs/extraction/coordination/inflight.md`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`

## Mechanism

The review page now derives a small preview model from each row:

- `blog_post`: description/content excerpt plus tags.
- `report` and `sales_brief`: first section title/body excerpt.
- `landing_page`: hero headline, first section excerpt, and CTA label.

The data is already present in the generated asset list/export rows, so this is
a rendering-only slice.

## Intentional

- No backend changes. The existing API already returns the fields needed for a
  compact preview.
- No full document drawer. This PR improves scanability without introducing a
  larger detail-view interaction.
- No Markdown renderer. Preview bodies are plain excerpts to keep the review
  list dense and safe.

## Deferred

- Full generated-asset detail drawer with rendered Markdown/sections.
- Component-level frontend tests for the review page.

## Verification

```bash
npm --prefix atlas-intel-ui run build
bash scripts/local_pr_review.sh
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `plans/PR-Content-Ops-Asset-Preview.md` | 55 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `atlas-intel-ui/src/api/contentOps.ts` | 10 |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | 120 |
| **Total** | **~189** |
