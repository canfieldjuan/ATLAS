# PR-Content-Ops-FAQ-Asset-Review-UI

## Why this slice exists

faq_markdown is now generated, persisted, exported, reviewable through the
backend API, and render-proofed as Markdown. The Atlas Intel Generated Asset
Review UI still exposes only reports, blog posts, landing pages, and sales
briefs. Operators cannot select FAQ drafts from the UI even though the route
already supports them.

## Scope (this PR)

1. Add faq_markdown to the frontend generated-asset type union.
2. Add a FAQ Markdown asset card to the Generated Asset Review page.
3. Render FAQ-specific previews, facts, and detail sections from the structured
   `items` payload returned by the API.
4. Avoid unsafe raw Markdown HTML rendering.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Asset-Review-UI.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Replace the merged stale row with this in-flight slice. |
| `atlas-intel-ui/src/api/contentOps.ts` | Add FAQ Markdown to the generated asset type and FAQ row fields. |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | Add the FAQ card plus safe preview/detail rendering. |

## Mechanism

The API type union will include faq_markdown, matching the backend asset
choices. The review page will add a fifth asset card and branch on
the FAQ asset id where the existing page already branches by asset:
subtitle, asset-specific facts, preview, and detail drawer content.

FAQ detail rendering reads `items` as an array of records and renders question,
answer, action items, and source labels with React text nodes. It does not use
`dangerouslySetInnerHTML` or a browser Markdown renderer; the raw row remains
available for diagnostics.

## Intentional

- No backend/API route changes. The generated asset API already supports FAQ
  Markdown.
- No Markdown-to-HTML rendering in the browser. The structured `items` payload
  is safer and more reviewable for operators.
- No new frontend test runner is introduced. This UI currently relies on
  TypeScript build coverage.

## Deferred

- A richer public FAQ page renderer can be added if FAQ drafts become a hosted
  customer-facing page, not just an operator review artifact.
- Per-item approve/reject remains out of scope; status review still applies to
  the whole generated FAQ document.

## Verification

- npm ci in atlas-intel-ui - passed, with 6 existing audit findings.
- npm run build in atlas-intel-ui - passed.
- npx eslint src/api/contentOps.ts src/pages/ContentOpsAssetsReview.tsx in atlas-intel-ui - passed.
- npm run lint in atlas-intel-ui - fails on pre-existing no-explicit-any errors
  in atlas-intel-ui/src/pages/b2b/B2BReports.tsx outside this PR diff.
- git diff --check - passed.
- bash scripts/local_pr_review.sh - passed.

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Asset-Review-UI.md` | +74 |
| `docs/extraction/coordination/inflight.md` | +1 / -1 |
| `atlas-intel-ui/src/api/contentOps.ts` | +7 |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | +141 / -12 |
| Total | ~218 |
