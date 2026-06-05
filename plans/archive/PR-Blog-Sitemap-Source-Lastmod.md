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
4. Keep blog content unchanged while tightening the source metadata path used
   by sitemap and prerender generation.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-Sitemap-Source-Lastmod.md` | Plan doc for this slice. |
| `atlas-intel-ui/vite.config.ts` | Use required source post metadata for blog sitemap lastmod and prerender output. |
| `atlas-intel-ui/scripts/verify-blog-geo-prerender.mjs` | Verify sitemap lastmod against source post dates and reject duplicate title tags. |

## Mechanism

The Vite build now collects blog metadata from the actual source post files and
requires `slug`, `title`, `description`, and `date` before emitting crawler
metadata. The sitemap and prerender plugins share that collection path, so the
blog URLs, sitemap `lastmod`, title metadata, and BlogPosting JSON-LD all come
from the same source fields.

The publish verifier already parses the same source date for BlogPosting checks.
It now also reads each sitemap URL block and compares `lastmod` to the source
date. It also rejects duplicate title tags in prerendered blog pages.

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
| Plan | ~25 |
| Sitemap and prerender source metadata path | ~95 |
| Publish verifier | ~25 |
| Total | ~145 |
