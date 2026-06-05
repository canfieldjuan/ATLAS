# PR: Landing Page Public Prerender

## Why this slice exists

Generated landing pages now have a public API route, sitemap feed, metadata,
structured data, and review/repair readiness. The frontend route at `/lp/:id/:slug`
still renders the generated page through client-side JavaScript. That means a
crawler can receive the SPA shell before the generated hero, body, CTA, meta,
and JSON-LD are visible in the returned HTML.

This slice makes approved generated landing pages crawler-visible at the
frontend build boundary.

Ownership lane: content-ops/landing-page-public-prerender

## Scope (this PR)

1. Extend the landing-page sitemap bridge so the build can:
   - parse `/lp/{id}/{slug}` entries from the public landing-page sitemap,
   - resolve the public landing-page API base,
   - fetch the approved public landing-page JSON payload for each sitemap entry.
2. Extend the Vite public-route prerender plugin to write static
   `dist/lp/{id}/{slug}/index.html` files for fetched generated landing pages.
3. Include generated landing-page title, description, canonical, body copy, CTA,
   robots policy, and JSON-LD in the static HTML.
4. Escape generated title, description, route body text, CTA URLs, and JSON-LD
   script content in the static HTML path.
5. Add focused bridge tests for sitemap parsing, API-base resolution, and page
   payload fetching.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Public-Prerender.md` | Plan doc for this public prerender slice. |
| `atlas-intel-ui/scripts/landing-page-sitemap-bridge.mjs` | Parse public landing-page sitemap entries and fetch public page payloads for prerendering. |
| `atlas-intel-ui/scripts/landing-page-sitemap-bridge.d.mts` | Expose TypeScript declarations for the new sitemap/prerender helpers. |
| `atlas-intel-ui/scripts/landing-page-sitemap-bridge.test.mjs` | Cover entry parsing, API-base resolution, payload fetching, and noindex filtering. |
| `atlas-intel-ui/vite.config.ts` | Prerender fetched generated landing pages into static public route HTML. |

## Mechanism

The existing sitemap bridge remains the source of generated landing-page URLs.
When `VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL` is configured, the build fetches the
sitemap, extracts approved `/lp/...` entries, fetches each public JSON payload
through `VITE_API_BASE` or the sitemap URL origin, and turns each payload into a
normal `PrerenderedRoute`.

The existing prerender plugin then writes one static HTML file per generated
landing page, the same way it already writes public blog routes. Runtime React
still hydrates and can refetch the same public payload.

## Intentional

- No backend route changes.
- No database changes.
- No public approval policy changes.
- No attempt to prerender when the public landing-page sitemap URL is absent.
- No reliance on generated landing pages being available in local dev builds.

## Deferred

- `HARDENING.md` still tracks landing-page repair legacy-lock rollout cleanup
  and repair lock connection hold time. Both are parked under
  `Owner/session: landing-page repair session` and are not required for public
  prerendering.
- Rich markdown rendering parity can be improved later if operators need exact
  static/React HTML matching. This slice prioritizes crawler-visible text,
  metadata, CTA, and JSON-LD.

## Parked hardening

- None added.

## Verification

- Sitemap bridge test -> 10 passed.
- Atlas Intel UI production build -> passed.
- Blog GEO prerender verification -> verified 14 blog pages.
- Local PR review -> passed.

## Estimated diff size

| File | Estimated LOC |
| --- | ---: |
| `atlas-intel-ui/scripts/landing-page-sitemap-bridge.mjs` | 95 |
| `atlas-intel-ui/scripts/landing-page-sitemap-bridge.d.mts` | 30 |
| `atlas-intel-ui/scripts/landing-page-sitemap-bridge.test.mjs` | 85 |
| `atlas-intel-ui/vite.config.ts` | 190 |
| `plans/PR-Landing-Page-Public-Prerender.md` | 65 |
| **Total** | **465** |
