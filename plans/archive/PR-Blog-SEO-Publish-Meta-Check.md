# PR-Blog-SEO-Publish-Meta-Check

## Why this slice exists

The blog publish verifier already protects crawler-visible GEO basics such as
canonical URLs, OG URL/type/image, BlogPosting JSON-LD, breadcrumbs, sitemap
inclusion, and indexability. It does not yet verify the source SEO title and
description actually make it into the prerendered HTML.

That leaves a practical SEO regression gap: a blog post could have good source
metadata while the published route ships the wrong title, missing description,
or incomplete social preview tags.

## Scope (this PR)

1. Extend `atlas-intel-ui/scripts/verify-blog-geo-prerender.mjs` to read each
   blog post's source title and description metadata.
2. Verify the prerendered title tag and meta description match the source
   SEO fields, falling back to the display title and description.
3. Verify OG title/description and Twitter title/description/image are present
   and aligned with the same source metadata.
4. Keep route rendering, blog content, and generation behavior unchanged.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-SEO-Publish-Meta-Check.md` | Plan doc for this slice. |
| `atlas-intel-ui/scripts/verify-blog-geo-prerender.mjs` | Add source-aware SEO and social meta verification. |

## Mechanism

The verifier now parses the existing TypeScript blog content files for `slug`,
`title`, `description`, `seo_title`, and `seo_description`. For each built blog
page, it compares the prerendered HTML against the expected title and
description values.

The check stays local to the existing publish verifier so the GitHub UI workflow
continues to catch public blog metadata regressions in one place.

## Intentional

- No frontend rendering changes.
- No content changes.
- No attempt to parse every possible TypeScript expression in blog files. The
  current blog content uses simple string fields, which is the contract this
  verifier enforces.
- No FAQ schema verification in this slice because current checked-in blog
  posts do not include FAQ data.

## Deferred

- Add FAQPage schema verification when blog content includes FAQ entries.
- Add per-post image verification if generated posts get unique OG images.

## Verification

- Atlas UI lint passed.
- Atlas UI production build passed.
- Blog GEO prerender verification passed across 14 blog pages.
- Whitespace diff check passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~60 |
| Publish verifier | ~75 |
| Total | ~135 |
