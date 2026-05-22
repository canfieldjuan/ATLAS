# PR-Landing-Page-Structured-Data-Preview-UI

## Why this slice exists

PR #751 exports renderer-ready landing-page structured data, but operators
reviewing generated landing pages still only see the raw row dump. The review
drawer should surface the JSON-LD in a compact, scannable panel before approval
or export.

## Scope (this PR)

Ownership lane: content-ops/landing-page-structured-data-preview-ui

1. Add a landing-page-only structured-data panel to the generated asset detail
   drawer.
2. Summarize Schema.org node types, FAQ question count, and canonical-link
   presence.
3. Show the formatted JSON-LD without changing review, approval, export, or
   generation behavior.

### Files touched

- `plans/PR-Landing-Page-Structured-Data-Preview-UI.md`
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`

## Mechanism

The detail drawer reads `row.structured_data` for landing-page assets only. The
helper accepts either object or JSON-string wire shapes, mirrors the existing
row parsing helpers, and derives:

- schema node types from `@graph` / `@type`
- FAQ count from `FAQPage.mainEntity`
- canonical status from `url` or `@id`

The panel is placed near readiness and repair history so reviewers can inspect
whether SEO/AEO/GEO export metadata is usable before approving a generated
landing page.

## Intentional

- No backend changes. PR #751 already added the export field.
- No approval behavior change.
- No public renderer. This is review visibility only.
- No copy-to-clipboard interaction in this slice; raw JSON is already visible
  and selectable.

## Deferred

- `PR-Landing-Page-Public-Renderer` can consume the structured data when a
  public generated landing-page route exists.
- A future UI slice can add copy/download controls if reviewers need them.

## Verification

- `npm run build` in `atlas-intel-ui` - passed.
- `git diff --check` - passed.
- `bash scripts/local_pr_review.sh origin/main` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~55 |
| Review drawer panel | ~80 |
| **Total** | **~135** |
