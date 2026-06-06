# PR-Blog-GEO-Publish-Check

## Why this slice exists

Generated blog drafts now have draft-level SEO/AEO/GEO readiness checks before
save, plus a targeted repair loop. The next gap is publish-level visibility:
the public blog HTML that crawlers and answer engines fetch must keep the
required metadata visible without client-side JavaScript.

This slice adds a build-output verifier for the Vite public blog prerender.

## Scope (this PR)

1. Add an `atlas-intel-ui` verification script for prerendered blog pages.
2. Check every static blog slug in `src/content/blog`.
3. Fail if any built page is missing canonical URL, OG metadata, BlogPosting
   JSON-LD, BreadcrumbList JSON-LD, sitemap inclusion, or indexability.
4. Add an npm script so operators and CI can run the verifier after build.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-GEO-Publish-Check.md` | Plan doc for this slice. |
| `atlas-intel-ui/package.json` | Add the verification npm script. |
| `atlas-intel-ui/scripts/verify-blog-geo-prerender.mjs` | Verify built public blog HTML. |

## Mechanism

The verifier reads static blog slugs from `atlas-intel-ui/src/content/blog`,
then checks `atlas-intel-ui/dist/blog/<slug>/index.html` and
`atlas-intel-ui/dist/sitemap.xml` after `npm run build`.

Each blog page must include crawler-visible:

- `<link rel="canonical">`
- Open Graph URL, type, and image metadata
- `BlogPosting` JSON-LD
- `BreadcrumbList` JSON-LD
- sitemap URL
- no `noindex` robots directive

## Intentional

- No runtime frontend behavior changes.
- No new test framework dependency.
- No FAQPage requirement until public static blog content includes FAQ entries.
- No generated draft contract changes.

## Deferred

- Emit and verify FAQPage JSON-LD once published static blog posts include FAQ
  entries.
- Wire the verifier into remote CI if the team wants frontend build checks to
  gate every PR.

## Verification

- Atlas UI production build passed.
- Blog GEO prerender verification passed across 14 blog pages.
- Script syntax check passed.
- Whitespace diff check passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~60 |
| Package script | ~5 |
| Verification script | ~120 |
| **Total** | **~185** |
