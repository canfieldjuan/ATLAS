# PR: Blog Review Public Link

## Why this slice exists

PR #1224 made approved generated blog posts publicly readable at `/blog/:slug`,
but the generated-asset review UI still only surfaces a public URL for approved
landing pages. That leaves the operator handoff incomplete: after approving a
generated `blog_post`, the drawer does not show the URL that can be opened,
copied, or checked.

This slice completes the review-to-public handoff for generated blogs by adding
the same public-link affordance already present for landing pages.

## Scope (this PR)

Ownership lane: content-ops/blog-public-productization

Slice phase: Product polish

1. Extend the generated-asset review drawer public URL helper so approved
   `blog_post` rows with a slug produce `/blog/:slug`.
2. Preserve the existing landing-page URL behavior at `/lp/:id/:slug`.
3. Show a pending approval message for blog posts that have a slug but are not
   approved yet.
4. Add a focused frontend source regression test for the blog public-link
   behavior and enroll it in the Atlas Intel UI workflow.

### Files touched

- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`
- `atlas-intel-ui/scripts/content-ops-blog-review-public-link.test.mjs`
- `atlas-intel-ui/package.json`
- `.github/workflows/atlas_intel_ui_checks.yml`
- `plans/PR-Blog-Review-Public-Link.md`

## Mechanism

The drawer already computes:

```ts
const publicUrl = publicLandingPageUrl(row, asset)
```

This PR replaces the landing-page-only helper with a small asset-aware helper:

```ts
publicAssetUrl(row, 'blog_post') -> /blog/:slug, approved only
publicAssetUrl(row, 'landing_page') -> /lp/:id/:slug, approved only
```

The pending-state helper similarly becomes asset-aware so both landing pages and
blog posts can explain why the public link is not available before approval.
The UI panel stays in the detail drawer; no backend or public blog route change
is needed because #1224 already made approved generated posts public.

## Intentional

- No public API changes. The public blog runtime route and approved-status
  predicate shipped in #1224.
- No list-card public-link button. The detail drawer remains the operator
  handoff surface, matching the existing landing-page behavior.
- Blog public URLs are slug-only. Generated blog public routes do not include
  draft IDs.

## Deferred

- Static prerender/sitemap ingestion for generated blogs remains the later
  SEO/GEO hardening slice deferred by #1224.
- A full browser E2E around approving a draft and opening the public URL remains
  deferred; this slice is the UI wiring/contract handoff.
- Parked hardening: none. `HARDENING.md` and `ATLAS-HARDENING.md` were scanned;
  their blog entries are content-generation correctness issues, not this review
  drawer URL handoff.

## Verification

- `cd atlas-intel-ui && npm ci` - passed; npm reports the existing 6 audit
  findings already parked in `HARDENING.md`.
- `cd atlas-intel-ui && npm run test:content-ops-blog-review-public-link` - 2
  passed.
- `cd atlas-intel-ui && npm run test:content-ops-landing-page-e2e-ui` - 4
  passed.
- `cd atlas-intel-ui && npm run test:blog-public-generated-posts` - 4 passed.
- `cd atlas-intel-ui && npm run lint` - passed.
- `cd atlas-intel-ui && npm run build` - passed; TypeScript and Vite build
  completed.
- `cd atlas-intel-ui && npm run verify:blog-geo` - verified 14 blog pages.
- `cd atlas-intel-ui && npm run verify:landing-page-geo` - skipped because no
  generated landing-page sitemap entries were present locally.
- `bash scripts/local_pr_review.sh --current-pr-body-file
  /home/juan-canfield/Desktop/blog-review-public-link-pr-body.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Review drawer helper/UI | ~45 |
| Focused frontend test + script/workflow enrollment | ~55 |
| **Total** | **~175** |
