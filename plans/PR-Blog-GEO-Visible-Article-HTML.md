# PR-Blog-GEO-Visible-Article-HTML

## Why this slice exists

The Atlas Intel UI now verifies blog SEO metadata, BlogPosting JSON-LD,
breadcrumbs, sitemap inclusion, and source-date `lastmod`. It also runs the
publish verifier in CI.

The remaining publish-level GEO gap is crawler-visible article content. A blog
page can expose a correct `<head>` and schema while still serving an empty SPA
root to crawlers that do not execute JavaScript. GEO and AEO need the article
body to be visible in the returned HTML.

## Scope (this PR)

1. Add static blog article HTML to the prerendered `/blog/:slug/index.html`
   output.
2. Keep the React runtime route behavior unchanged.
3. Use the existing blog source file collector as the source of truth.
4. Extend the blog GEO prerender verifier to assert crawler-visible article
   content.
5. Keep landing, blog index, sitemap, and frontend component rendering
   behavior unchanged.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-GEO-Visible-Article-HTML.md` | Plan doc for this slice. |
| `atlas-intel-ui/vite.config.ts` | Render source blog content into prerendered blog route HTML. |
| `atlas-intel-ui/scripts/verify-blog-geo-prerender.mjs` | Verify crawler-visible blog article body output. |

## Mechanism

The Vite prerender plugin already scans each blog source file for required
metadata. This slice extends that source collector to require the post `content`
field. For blog routes only, the plugin renders the markdown content into static
HTML and inserts it into the `#root` element in the prerendered route file.

The generated static body includes a blog article marker, one `h1`, date/author
context, and the rendered source body. Chart placeholders are removed from the
static HTML because the interactive chart components still belong to the React
runtime.

The verifier now reads the built blog HTML and checks that each post has:

- one crawler-visible `h1`
- a prerendered article marker
- a prerendered content marker
- source title visible in the `h1`
- source body text visible in the returned HTML
- enough body text to prove it is not an empty shell

## Intentional

- No React component changes.
- No generated blog content changes.
- No static chart rendering in this slice.
- No FAQ schema changes.
- No claim that GEO guarantees AI-engine placement.

## Deferred

- Static chart summaries for no-JavaScript crawlers.
- Full static FAQ rendering checks when the Atlas Intel source posts include
  FAQ entries.
- Sharing the source parsing helper between the Vite plugin and verifier.

## Verification

- Atlas UI lint.
- Atlas UI production build.
- Blog GEO prerender verification.
- Whitespace diff check.
- Local PR review.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Vite prerender body output | ~65 |
| Publish verifier | ~75 |
| Total | ~215 |
