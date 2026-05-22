# Atlas Intel UI

React/Vite frontend for Atlas Intelligence public pages, generated-asset review,
and B2B intelligence workflows.

## Build Checks

Run the frontend checks from this directory:

```bash
npm run lint
npm run test:landing-page-prerender
npm run build
npm run verify:blog-geo
npm run verify:landing-page-geo
```

The GitHub `Atlas Intel UI Checks` workflow runs the same build and public
prerender verification path.

## Environment

### `VITE_API_BASE`

Set this in production to the Atlas backend origin, without the API path:

```bash
VITE_API_BASE=https://atlas-api.example.com
```

The browser uses this value for `/api/v1/...` calls. The build also uses it
when prerendering generated landing pages, because each sitemap entry is fetched
from:

```text
{VITE_API_BASE}/api/v1/content-assets/landing_page/public/{id}
```

Leave it empty in local development when the Vite dev proxy should handle
`/api` requests.

### `VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL`

Set this only when generated landing pages should be included in the static
public sitemap and prerendered as crawler-visible HTML:

```bash
VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL=https://atlas-api.example.com/api/v1/content-assets/landing_page/public/sitemap.xml
```

When this is configured, the Vite build fetches the sitemap, keeps `/lp/...`
entries, fetches each approved public landing-page payload, and writes static
HTML to:

```text
dist/lp/{id}/{slug}/index.html
```

The landing-page verifier then checks those built files for canonical metadata,
robots policy, JSON-LD, visible body copy, H1, and CTA markup.

If this variable is absent, generated landing-page prerendering is skipped and
`npm run verify:landing-page-geo` exits successfully when no `/lp/...` sitemap
entries exist.
