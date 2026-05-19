# PR-Blog-SEO-Infrastructure

## Why this slice exists

`churnsignals.co` is a React SPA. Every URL currently serves the same
`atlas-churn-ui/index.html` shell with a generic
`<title>Churn Signals - B2B Software Churn Intelligence</title>` and
no per-route metadata. Verified live as of this PR:

```
curl -L https://churnsignals.co/blog/hubspot-deep-dive-2026-04
  -> <title>Churn Signals - B2B Software Churn Intelligence</title>
  -> <meta property="og:title" content="Churn Signals" />
```

The team's `SeoHead` component sets per-route title/meta/canonical/
OG/Twitter/JSON-LD in `useEffect`, so the data is correct AFTER
client-side hydration. But:

- **AEO crawlers** (GPTBot, PerplexityBot, ClaudeBot, Bingbot for
  Copilot) typically don't execute JS. They see the empty shell --
  every blog post looks identical.
- **Social unfurls** (LinkedIn, Twitter, Slack, iMessage) never run
  JS. Every share preview shows the generic homepage card.
- **Googlebot** eventually re-renders and indexes per-page meta,
  but the render queue adds weeks of delay.

The result: 80 published blog posts are effectively invisible to AI
engines, social previews, and the first month of Google indexing.
This blocks the long-term goal of publishing blog posts that get
discovered, cited, and clicked.

The sibling `atlas-intel-ui/` already has a `prerenderPlugin` that
writes per-route HTML files at build time
(`atlas-churn-ui` analogue does not). This slice ports that plugin
to `atlas-churn-ui`, adds the missing Vercel rewrite rules so the
prerendered HTML actually takes precedence over the SPA fallback,
ships the `og-default.png` asset that the BlogPost JSON-LD already
references, and adds an `Organization` + `WebSite` graph + a
`BreadcrumbList` per blog post that the existing pipeline doesn't
emit.

A small companion change on `atlas-intel-ui/` adds `Organization` +
`WebSite` (with `SearchAction`) to the `/landing` JSON-LD graph and
`BreadcrumbList` alongside `BlogPosting` on its blog posts, parity
with the churn-ui work.

## Scope

1. Port the prerender plugin from `atlas-intel-ui/vite.config.ts`
   to `atlas-churn-ui/vite.config.ts`. Adapted for churn-ui's
   `BlogPost` content shape (`seo_title`, `seo_description`, `date`,
   `topic_type`, `faq`).
2. Emit per-route `dist/<path>/index.html` at build time for:
     * `/` -- Organization + WebSite (with SearchAction) graph
     * `/blog`, `/methodology`, `/landing` -- basic head + OG/Twitter
     * `/blog/<slug>` -- BlogPosting + BreadcrumbList + FAQPage
       (when `faq` is populated)
3. Add `atlas-churn-ui/vercel.json` with rewrite rules so requests
   like `/blog/<slug>` resolve to `dist/blog/<slug>/index.html`
   instead of falling through to the SPA's `/index.html` shell.
4. Add `atlas-churn-ui/public/og-default.png` (copied from
   `atlas-intel-ui/public/`). The existing `BlogPost.tsx:53`
   JSON-LD references `https://churnsignals.co/og-default.png`;
   without the asset, social unfurls 404 the image.
5. On `atlas-intel-ui/vite.config.ts`: add `Organization` + `WebSite`
   (with `SearchAction`) entries to the existing `LANDING_JSON_LD`
   `@graph`, and add a `BreadcrumbList` entry to the per-blog-post
   `@graph` so intel-ui blog posts emit Home -> Blog -> Post
   breadcrumb schema like churn-ui will.

### Files touched

- `atlas-churn-ui/vite.config.ts`
- `atlas-churn-ui/vercel.json`
- `atlas-churn-ui/public/og-default.png`
- `atlas-intel-ui/vite.config.ts`
- `plans/PR-Blog-SEO-Infrastructure.md`

## Mechanism

The Vite plugin runs in `closeBundle()` after the normal SPA build.
It reads `dist/index.html` as the base shell, then for each route:

1. Replaces the generic `<title>` with a route-specific one.
2. Injects per-route `<meta>` tags (description, canonical,
   `og:title`, `og:description`, `og:image`, `og:image:width`,
   `og:image:height`, `twitter:card`, `twitter:title`,
   `twitter:description`, `twitter:image`).
3. Injects a `<script id="seo-jsonld" type="application/ld+json">`
   block carrying the per-route schema. The matching `id` lets the
   client-side `SeoHead` component replace the same node on
   hydration without producing duplicates.
4. Writes the rendered HTML to `dist/<path>/index.html`.

For blog posts the plugin scrapes `src/content/blog/*.ts` files for
`slug`, `seo_title`, `seo_description`, `date`, and the `faq`
array. The FAQ regex tolerates multi-line array literals and
escaped quotes. Posts with empty `faq` get BlogPosting +
BreadcrumbList only.

Vercel routing: the new `vercel.json` adds explicit rewrites for
`/blog`, `/blog/(.*)`, `/landing`, and `/methodology` before the
SPA catch-all. Without these the SPA fallback rule would match
first and the prerendered files would never be served.

## Intentional

- The prerender plugin runs alongside the existing client-side
  `SeoHead` component rather than replacing it. The two work
  together: prerender renders for crawlers, `SeoHead` updates the
  DOM on client-side route changes via `useEffect`. The same
  `id="seo-jsonld"` lets `SeoHead` replace the prerendered script
  in place rather than creating duplicates.
- BlogPosting + BreadcrumbList + FAQPage are kept as three separate
  `<script>` blocks (with distinct `id` attributes) rather than a
  single `@graph`. That mirrors how the existing pipeline emits
  schema and avoids merging unrelated entities.
- The `BlogPost.tsx` component continues to set its JSON-LD via
  `useEffect`; this PR does not change the client-side behavior
  beyond ensuring the prerendered version is consistent with what
  hydration produces.
- `og-default.png` is force-added (root `.gitignore` has `*.png`).
  Same precedent as `atlas-intel-ui/public/og-default.png` (already
  force-added previously).

## Deferred

- A custom OG image per blog post (currently every post shares
  `og-default.png`). Lower CTR on social shares vs. unique images
  but no rendering breakage.
- `Organization` schema with a verified `sameAs` chain (e.g.,
  Twitter, LinkedIn handles confirmed under the brand). The current
  `sameAs` array points to brand handles that may not yet be
  claimed; verify with the SEO/PR owner before treating as
  authoritative.
- Submitting `https://churnsignals.co/sitemap.xml` to Google Search
  Console. That's an operator action after this deploys, not a code
  change.
- Submitting Bing Webmaster Tools (Bing powers Copilot / ChatGPT
  search). Same operator-action category.
- `atlas-intel-ui` lives on `atlas-intel-ui-two.vercel.app`. The
  auto-generated Vercel preview domain doesn't rank well; a real
  custom domain is needed before its sitemap is worth submitting
  to GSC. Out of scope for this PR.

## Verification

- `npm run build` in `atlas-churn-ui` -> `Sitemap generated with
  83 URLs`, `Pre-rendered 83 public routes`, no TS errors.
- `find atlas-churn-ui/dist -name "index.html" | wc -l` -> 83.
- Spot-check on `dist/blog/hubspot-deep-dive-2026-04/index.html`:
  per-post `<title>HubSpot Reviews 2026: 1680 User Experiences
  Analyzed | Churn Signals</title>`, `og:image` set to
  `https://churnsignals.co/og-default.png`, BlogPosting +
  BreadcrumbList + FAQPage JSON-LD blocks present.
- `npm run build` in `atlas-intel-ui` -> compiles cleanly with the
  Organization + BreadcrumbList additions.
- `git diff --check` -> passed.
- `scripts/local_pr_review.sh` -> expected to pass.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `atlas-churn-ui/vite.config.ts` (prerender plugin add) | ~210 |
| `atlas-churn-ui/vercel.json` (new) | ~12 |
| `atlas-churn-ui/public/og-default.png` (binary; counted minimal) | ~1 |
| `atlas-intel-ui/vite.config.ts` (graph additions) | ~50 |
| Plan doc | ~140 |
| **Total** | **~415** |
