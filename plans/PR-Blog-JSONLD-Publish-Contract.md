# PR-Blog-JSONLD-Publish-Contract

## Why this slice exists

The blog publish verifier now checks source-aligned title, description, OG, and
Twitter metadata. Its BlogPosting JSON-LD check is still mostly presence-based:
it verifies required fields exist, but not that those fields match the source
blog post contract.

That leaves a crawler-visible structured-data regression gap. A page could ship
the right meta tags while BlogPosting JSON-LD carries stale headline,
description, date, image, or organization data.

## Scope (this PR)

1. Extend `atlas-intel-ui/scripts/verify-blog-geo-prerender.mjs` to parse each
   blog post source date.
2. Verify BlogPosting headline and description match the same source-derived
   title and description used by SEO metadata.
3. Verify BlogPosting published/modified dates match the source post date.
4. Verify BlogPosting image matches the route's OG image.
5. Verify author and publisher organization identity fields stay present.
6. Update `atlas-intel-ui/vite.config.ts` so prerendered BlogPosting JSON-LD
   emits the source-aligned fields the verifier requires.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-JSONLD-Publish-Contract.md` | Plan doc for this slice. |
| `atlas-intel-ui/scripts/verify-blog-geo-prerender.mjs` | Add source-aligned BlogPosting JSON-LD assertions. |
| `atlas-intel-ui/vite.config.ts` | Emit source-aligned BlogPosting dates and organization identity in prerendered HTML. |

## Mechanism

The verifier derives expected BlogPosting values from the existing TypeScript
blog source files and the prerendered HTML. It compares JSON-LD headline,
description, dates, mainEntityOfPage, image, author, and publisher values
against that expected contract.

The prerender plugin now uses the same exact-field source parsing and emits the
date and organization identity fields needed by the publish contract.

This keeps publish-surface SEO/GEO checks in one CI-covered verifier.

## Intentional

- No runtime React rendering changes.
- No content changes.
- No generated-post pipeline changes.
- No new schema types.

## Deferred

- Add FAQPage schema verification when blog content includes FAQ entries.
- Add per-post image verification if generated posts get unique OG images.
- Consider validating HowTo schema once migration posts reliably emit it.

## Verification

- Atlas UI lint passed.
- Atlas UI production build passed.
- Blog GEO prerender verification passed across 14 blog pages.
- Whitespace diff check passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~60 |
| Publish verifier | ~70 |
| Prerender metadata | ~35 |
| Total | ~165 |
