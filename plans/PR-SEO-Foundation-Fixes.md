# PR: SEO Foundation Fixes

Date: 2026-05-09
Owner: antigravity

## Why this slice exists

The SEO/GEO/AEO audit (2026-05-09) identified that `Landing.tsx` has zero
meta management, `SeoHead` never emits `og:image`, `Blog.tsx` duplicates the
meta pattern instead of using `SeoHead`, and the sitemap omits the primary
acquisition page (`/landing`). These gaps mean crawlers and AI engines
(GPTBot, PerplexityBot, ClaudeBot) see incorrect or missing title/description
for the most important public routes. This slice fixes the content layer;
`PR-SEO-Prerender` bakes it into static HTML.

## Scope (this PR)

1. Add `ogImage` + `twitter:image` props to `SeoHead` with a default fallback.
2. Add `SeoHead` to `Landing.tsx` with `SoftwareApplication` + `WebSite` JSON-LD.
3. Replace `Blog.tsx`'s hand-rolled `useEffect` meta block with `SeoHead`.
4. Update `index.html` `<title>` to keyword-targeted string.
5. Update Vite sitemap plugin to include `/landing` at priority `1.0`; demote
   `/` to `0.3` (it redirects to login).
6. Add `public/og-default.png` (1200×630 OG card).

### Files touched

- `atlas-intel-ui/src/components/SeoHead.tsx`
- `atlas-intel-ui/src/pages/Landing.tsx`
- `atlas-intel-ui/src/pages/Blog.tsx`
- `atlas-intel-ui/index.html`
- `atlas-intel-ui/vite.config.ts`
- `atlas-intel-ui/public/og-default.png` (new)
- `plans/PR-SEO-Foundation-Fixes.md` (this file)

## Mechanism

**SeoHead:** adds optional `ogImage?: string` prop. Inside `useEffect`, a
resolved image URL (prop value or the static default) is written to
`og:image`, `og:image:width`, `og:image:height`, and `twitter:image`. All
four are tracked in the `managed` set so cleanup removes them on unmount.

**Landing.tsx:** adds a `landingJsonLd` constant with `@graph` containing
`WebSite` and `SoftwareApplication` nodes. `SoftwareApplication` includes an
`AggregateOffer` with the current pricing range ($49–$399). `SeoHead` is
mounted at the top of the returned JSX, before `<PublicLayout>`.

**Blog.tsx:** the 47-line `useEffect` block is replaced with a single
`<SeoHead>` element. Logic is identical; the shared component now owns it.

**Sitemap plugin:** the `urls` array is reordered so `/landing` is first at
priority `1.0`. The existing `/` entry moves to the bottom at priority `0.3`
with `changefreq: 'monthly'`.

## Intentional

- **`/` stays in sitemap at 0.3** rather than being removed — removing it
  signals a broken site to crawlers. Low priority correctly deprioritizes
  the login redirect without hiding the domain.
- **No SSR/pre-rendering in this PR** — this PR fixes meta *content*;
  `PR-SEO-Prerender` fixes *crawlability*. Coupling them makes rollback
  harder if the pre-renderer breaks the build.
- **No `BreadcrumbList` schema** — scoped to a future NIT PR; not blocking.
- **No custom domain update in `robots.txt`** — blocked on a custom domain
  being configured by the team.

## Deferred

- Pre-rendering public routes → `PR-SEO-Prerender`
- `BreadcrumbList` on BlogPost → future NIT PR
- `updated_date` field on `BlogPost` type → future content PR
- Per-post OG images → future content tooling PR
- `robots.txt` domain → after custom domain is set

## Verification

```bash
cd atlas-intel-ui && npm run build
# Sitemap should list /landing as priority 1.0 and / as 0.3
grep -A4 'landing' dist/sitemap.xml
grep -A4 '<loc>https://atlas-intel-ui-two.vercel.app/</loc>' dist/sitemap.xml

# OG image file present
ls -lh public/og-default.png

# Build clean
echo "Exit: $?"
```

## Estimated diff size

~130 LOC. Well within 400 LOC budget.
