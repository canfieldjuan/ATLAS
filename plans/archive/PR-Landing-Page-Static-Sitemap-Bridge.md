# PR-Landing-Page-Static-Sitemap-Bridge

## Why this slice exists

Generated landing pages now have a backend-owned public sitemap feed, but the
public `atlas-intel-ui` build still emits a static sitemap with only the fixed
marketing and blog routes. This slice bridges those two surfaces so crawler
entry points can include approved generated landing pages without making the
frontend own landing-page readiness logic.

## Scope (this PR)

Ownership lane: content-ops/landing-page-static-sitemap-bridge

1. Add a small sitemap-bridge helper for build-time import of generated landing
   page URLs.
2. Wire the `atlas-intel-ui` Vite sitemap plugin to append generated `/lp/...`
   URLs when a feed URL is configured.
3. Normalize imported landing-page URLs onto the frontend origin and dedupe
   sitemap entries.
4. Keep backend readiness/indexing policy unchanged.
5. Keep static prerender unchanged in this slice.

### Files touched

- `plans/PR-Landing-Page-Static-Sitemap-Bridge.md`
- `atlas-intel-ui/vite.config.ts`
- `atlas-intel-ui/scripts/landing-page-sitemap-bridge.mjs`
- `atlas-intel-ui/scripts/landing-page-sitemap-bridge.d.mts`
- `atlas-intel-ui/scripts/landing-page-sitemap-bridge.test.mjs`
- `atlas-intel-ui/package.json`

## Mechanism

The Vite sitemap plugin builds the existing static URL list, then optionally
fetches a configured generated landing-page sitemap feed from
`VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL`. The helper extracts `<loc>` entries,
keeps only `/lp/...` paths, rewrites those paths to the frontend `BASE_URL`,
and returns sitemap-ready URL objects.

If no feed URL is configured, the build produces the same static sitemap it
does today. If a feed URL is configured but the feed cannot be fetched or
parsed, the helper throws so production builds do not silently ship a partial
generated-page sitemap.

## Intentional

- No frontend copy/layout changes.
- No backend sitemap or robots-policy changes.
- No static prerender for database-backed generated landing pages.
- No import of backend metadata/readiness details into the frontend build.

## Deferred

- `PR-Landing-Page-SEO-GEO-AEO-Input-Contract` should add first-class
  landing-page inputs before more rendering work. The current code audit shows
  review panels, save-time quality gates, and generation repair already exist,
  while user-provided SEO/GEO/AEO inputs still do not have a clean form or a
  broad generation input contract.
- `PR-Landing-Page-Readiness-Validator-Unification` should then centralize the
  export/public-readiness validators with the save-time quality gate so
  generation, review, robots, and sitemap inclusion use one source of truth.
- `PR-Landing-Page-Public-Prerender` can add static HTML prerendering later if
  crawl diagnostics show the SPA route is not enough.

## Verification

- Node unit tests for feed URL resolution, XML loc extraction, origin
  normalization, path filtering, dedupe, and fetch failure behavior.
- `npm --prefix atlas-intel-ui run test:sitemap-bridge`
  - 6 passed.
- `npm --prefix atlas-intel-ui run build`
  - passed without a generated landing-page feed.
  - passed with a `data:` generated landing-page feed and wrote the normalized
    `/lp/111/acme-page` URL into `dist/sitemap.xml`.
- `npm --prefix atlas-intel-ui run verify:blog-geo`
  - verified 14 blog pages.
- `npm --prefix atlas-intel-ui run lint`
  - passed.
- Git whitespace check.
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~85 |
| Sitemap bridge helper | ~95 |
| Vite sitemap wiring | ~25 |
| Node tests/package script | ~95 |
| **Total** | **~300** |
