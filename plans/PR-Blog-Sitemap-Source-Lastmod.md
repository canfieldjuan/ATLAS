# PR-Blog-Sitemap-Source-Lastmod

## Why this slice exists

The public blog publish verifier now protects canonical URLs, SEO metadata,
BlogPosting JSON-LD, breadcrumbs, sitemap inclusion, and indexability. The
sitemap still gives every blog post the build date as `lastmod`, even when each
post already has a stable source `date`.

That makes the crawler-facing freshness signal noisier than it needs to be.
Rebuilding the site should not make every blog post look newly modified.

## Scope (this PR)

1. Update `atlas-intel-ui/vite.config.ts` so blog sitemap entries use each
   post's source `date` as `lastmod`.
2. Keep non-blog sitemap entries on the build date.
3. Extend `atlas-intel-ui/scripts/verify-blog-geo-prerender.mjs` to assert each
   blog sitemap `lastmod` matches the source post date.
4. Keep route rendering, blog content, and generated metadata unchanged.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-Sitemap-Source-Lastmod.md` | Plan doc for this slice. |
| `atlas-intel-ui/vite.config.ts` | Use source post dates for blog sitemap lastmod. |
| `atlas-intel-ui/scripts/verify-blog-geo-prerender.mjs` | Verify sitemap lastmod against source post dates. |

## Mechanism

The sitemap plugin already scans each blog source file. It now records the
source `date` alongside each slug and writes that date into the blog URL
`lastmod` field.

The publish verifier already parses the same source date for BlogPosting checks.
It now also reads each sitemap URL block and compares `lastmod` to the source
date.

## Intentional

- No content changes.
- No runtime React rendering changes.
- No change to landing, blog index, or root sitemap freshness behavior.
- No generated-post pipeline changes.

## Deferred

- Add `lastmod` from update timestamps if generated blog posts eventually track
  separate published and modified dates.
- Add FAQPage schema verification when blog content includes FAQ entries.

## Verification

- Atlas UI lint passed.
- Atlas UI production build passed.
- Blog GEO prerender verification passed across 14 blog pages.
- Whitespace diff check passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~60 |
| Sitemap plugin | ~25 |
| Publish verifier | ~20 |
| Total | ~105 |
