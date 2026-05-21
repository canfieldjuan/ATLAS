# PR-Blog-Shared-Source-Parser

## Why this slice exists

The Atlas Intel blog prerender build and the blog GEO verifier were carrying
separate copies of the same source-post parsing logic. That made chart fallback
behavior harder to keep aligned: a parser fix could land in the build path while
the verifier kept a stale version, or the other way around.

This slice moves the source metadata parser behind one shared helper so the
build and verification paths read `slug`, SEO fields, article body content,
chart specs, and optional FAQ entries through the same contract.

## Scope (this PR)

1. Add a shared `blog-source-metadata.mjs` helper for Atlas Intel blog source
   parsing.
2. Add a `.d.mts` declaration so the TypeScript Vite config gets typed parser
   results when importing the ESM helper.
3. Update `vite.config.ts` to use the shared parser for sitemap generation and
   prerendered blog pages.
4. Update `verify-blog-geo-prerender.mjs` to use the same parser before checking
   crawler-visible blog output.
5. Keep the chart-placeholder safety check in the shared parser so undefined
   chart IDs fail both build and verification.
6. Parse optional source `faq` entries through the same helper.
7. Prerender visible FAQ sections and FAQPage JSON-LD when a source post has FAQ
   entries.
8. Extend the verifier to assert visible FAQ text and FAQPage schema for posts
   that include FAQ entries.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-Shared-Source-Parser.md` | Plan doc for this slice. |
| `atlas-intel-ui/scripts/blog-source-metadata.mjs` | Shared blog source parser with chart and FAQ validation. |
| `atlas-intel-ui/scripts/blog-source-metadata.d.mts` | Type declarations for Vite's TypeScript config import. |
| `atlas-intel-ui/vite.config.ts` | Reuse the shared parser and prerender static FAQ HTML/schema when present. |
| `atlas-intel-ui/scripts/verify-blog-geo-prerender.mjs` | Reuse the shared parser and verify static FAQ HTML/schema when present. |

## Mechanism

The helper reads each TypeScript source post from `src/content/blog`, extracts
the fields already used by the build and verifier, parses the JSON-shaped
`charts` and optional `faq` arrays with the bracket-balanced field reader, and
validates that every `{{chart:id}}` placeholder has matching chart data.

`vite.config.ts` now asks the helper for source metadata when writing sitemap
entries and prerendering article pages. The verifier asks the same helper for the
same source metadata before checking the generated HTML.

When a source post includes FAQ entries, the prerenderer now writes a visible FAQ
section into the crawler-visible article HTML and adds an FAQPage node to the
same JSON-LD graph as BlogPosting and BreadcrumbList. The verifier enforces both
the visible text and schema node only for posts that actually include FAQ data.

## Intentional

- No generated blog content changes.
- No React runtime changes.
- No sitemap contract changes.
- No change to the public blog route shape.
- No broad parser rewrite beyond sharing the source metadata contract.
- No generated FAQ content added to current posts.

## Deferred

- Loading source posts as real modules instead of parsing field literals.
- Generating a build-time manifest artifact consumed by both build and tests.
- Broader FAQ content generation or retrofitting FAQ entries onto existing posts.

## Verification

- Atlas Intel production build.
- Blog GEO prerender verification.
- Whitespace diff check.
- Local PR review.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~85 |
| Shared parser helper and declarations | ~185 |
| Vite config parser removal/wiring plus FAQ fallback | ~205 |
| Verifier parser removal/wiring plus FAQ checks | ~170 |
| Total | ~645 |
