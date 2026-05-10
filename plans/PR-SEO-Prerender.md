# PR: SEO Pre-render Public Routes

Date: 2026-05-10
Owner: antigravity
Depends on: PR #446 (PR-SEO-Foundation-Fixes) — merged

## Why this slice exists

PR #446 fixed the meta tag *content* in React components, but none of it was
visible to crawlers because the app is a client-side SPA. Googlebot and all AI
crawler agents (GPTBot, PerplexityBot, ClaudeBot) receive a blank
`<div id="root"></div>` when they fetch any URL. This slice adds a zero-
dependency Vite build plugin that bakes correct `<head>` HTML into static
files for every public route at build time, making all SEO/GEO/AEO signals
immediately readable without JavaScript execution.

## Scope (this PR)

1. Add `prerenderPlugin()` to `vite.config.ts` — a build-time plugin that
   generates `dist/<route>/index.html` for each public route with full static
   `<head>` meta baked in.
2. Update `vercel.json` to serve pre-rendered `.html` files for public routes
   before falling back to `index.html` for authenticated SPA routes.
3. No third-party SSR dependency added — pure Node.js `fs` + string
   manipulation in the Vite config.

### Files touched

- `atlas-intel-ui/vite.config.ts`
- `atlas-intel-ui/vercel.json`
- `plans/PR-SEO-Prerender.md` (this file)

## Mechanism

**`prerenderPlugin()`** runs in `closeBundle()` (after Vite finishes emitting
assets). It:

1. Reads `dist/index.html` as a base template.
2. Reads each `src/content/blog/*.ts` file and extracts `slug`, `seo_title`,
   and `seo_description` via regex (no TS transpilation needed at build time —
   these are plain string literals).
3. Builds a static `<head>` block per route using `buildHeadHtml()`:
   - `<title>`, `<meta name="description">`, `<link rel="canonical">`
   - Full OG tag set (`og:title`, `og:description`, `og:url`, `og:type`,
     `og:site_name`, `og:image`, `og:image:width`, `og:image:height`)
   - Twitter card set
   - `<script type="application/ld+json">` with route-appropriate schema
4. Replaces the fallback `<title>` in `index.html` and injects meta before
   `</head>`.
5. Writes the result to `dist/<path>/index.html`.

**`vercel.json` rewrites:** The new rewrite rules check for a pre-rendered
file first (`/landing` → `/landing/index.html`, `/blog/(.*)`→
`/blog/$1/index.html`) before falling through to the SPA catch-all. Vercel
evaluates rewrites in order, so authenticated routes (`/b2b/*`, `/reviews`,
etc.) correctly fall through to `/index.html`.

**Build output:**
```
dist/landing/index.html          ← pre-rendered, full meta
dist/blog/index.html             ← pre-rendered, full meta
dist/blog/<slug>/index.html      ← pre-rendered, per-post title/desc/JSON-LD
dist/index.html                  ← unchanged SPA shell (auth routes)
```

## Intentional

- **No third-party SSR library.** `vite-plugin-ssg` (v0.1.0, installed
  during investigation) requires exporting `ssgOptions` from each page file
  and uses `StaticRouter` — too invasive for what is effectively a one-time
  meta injection. The custom plugin is ~120 lines, has no dependencies beyond
  Node.js builtins, and is trivially auditable.
- **String-regex extraction of blog meta**, not TS import. Importing TS at
  Vite build plugin time requires a transpilation step; regex on known literal
  patterns is faster and has zero failure modes at build time.
- **Only public routes pre-rendered.** Authenticated routes stay SPA —
  crawlers cannot access them and `robots.txt` disallows them anyway.
- **`dist/index.html` is not modified.** The base SPA shell is left intact
  so authenticated routes continue working normally.

## Deferred

- Per-post `og:image` (custom image per blog post) → future content tooling PR
- `BreadcrumbList` schema on blog posts → future NIT PR
- `robots.txt` domain → after custom domain is set
- Incremental static regeneration → not available on Vercel without Next.js;
  acceptable since blog content is updated at build time

## Verification

```bash
cd atlas-intel-ui && npm run build
# Should print: "Pre-rendered 16 public routes"

# Confirm static files exist
ls dist/landing/index.html
ls dist/blog/index.html
ls dist/blog/amazon-review-monitoring-tools-2026-03/index.html

# Confirm meta is baked in (no JS needed)
grep 'og:title' dist/landing/index.html
grep 'og:title' dist/blog/index.html
grep 'og:title' dist/blog/amazon-review-monitoring-tools-2026-03/index.html
grep 'application/ld+json' dist/landing/index.html

# Build exit code
echo "Exit: $?"
```

Expected output:
- `og:title` present in all three files with correct values
- `application/ld+json` count = 1 in landing
- Exit: 0

## Estimated diff size

~200 LOC (vite.config.ts rewrite + vercel.json). Within 400 LOC budget.
