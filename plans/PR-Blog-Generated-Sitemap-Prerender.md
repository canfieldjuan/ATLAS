# PR: Blog Generated Sitemap Prerender

## Why this slice exists

PR #1224 made approved generated blog posts publicly readable at runtime, and
PR #1226 added the review-drawer handoff link. Both plans deliberately deferred
static sitemap/prerender ingestion for generated blogs, so approved DB-backed
posts can be opened at `/blog/:slug` but are still absent from the build-time
SEO/GEO artifacts crawlers consume.

This slice closes that deferred hardening item by teaching the Atlas Intel UI
build to ingest approved generated blog posts from the existing public blog API
when a public feed/base URL is configured, then include those posts in
`sitemap.xml` and generated `/blog/:slug/index.html` files.

## Scope (this PR)

Ownership lane: content-ops/blog-generated-sitemap-prerender

Slice phase: Production hardening

1. Add a small generated-blog build bridge that resolves an optional
   `VITE_PUBLIC_BLOG_POSTS_URL` feed, falling back to
   `VITE_API_BASE/api/v1/blog/published` when configured.
2. Convert the existing public blog list envelope into strict build-time blog
   entries with slug, title, description, date, author, content, charts, FAQ,
   and SEO metadata.
3. Wire generated blog entries into the sitemap plugin with `/blog/:slug`
   URLs and `lastmod` from the public post date.
4. Wire generated blog entries into the prerender plugin, while keeping static
   posts on their trusted static HTML path.
5. Escape generated blog HTML before Markdown rendering so DB-backed approved
   content cannot inject raw HTML into prerendered output.
6. Add mocked transport tests for feed resolution, request shape, malformed
   envelope failure, malformed post failure, dedupe, sitemap rows, and
   prerender-entry shape.
7. Enroll the focused test in `atlas-intel-ui/package.json` and the Atlas
   Intel UI workflow.

### Files touched

- `atlas-intel-ui/scripts/blog-sitemap-bridge.mjs`
- `atlas-intel-ui/scripts/blog-sitemap-bridge.d.mts`
- `atlas-intel-ui/scripts/blog-generated-sitemap-prerender.test.mjs`
- `atlas-intel-ui/vite.config.ts`
- `atlas-intel-ui/package.json`
- `.github/workflows/atlas_intel_ui_checks.yml`
- `plans/PR-Blog-Generated-Sitemap-Prerender.md`

## Mechanism

The build bridge is frontend-build-only. It does not add a new backend route;
it consumes the public route shipped by #1224:

```text
GET ${VITE_PUBLIC_BLOG_POSTS_URL}
or
GET ${VITE_API_BASE}/api/v1/blog/published
```

If no URL is configured, the bridge returns an empty set so local and CI builds
do not need a live backend. If a URL is configured, the response envelope must
be an object with `posts: []`, and each public post must contain the fields
needed to build a canonical page. Missing or malformed envelope/post data fails
closed instead of silently producing a partial sitemap.

`vite.config.ts` will share one blog-route builder for static and generated
posts. Static in-repo posts keep their existing HTML rendering behavior.
Generated posts pass through an escape-first Markdown path before prerender
HTML insertion, matching the safety intent from #1224's runtime generated-post
renderer.

## Intentional

- No backend API changes. The public blog list/detail routes already exist and
  expose approved generated posts.
- No live API dependency in CI. Tests mock `fetch`, and the build remains a
  no-op for generated blog ingestion unless `VITE_PUBLIC_BLOG_POSTS_URL` or
  `VITE_API_BASE` is configured.
- No Content Ops generation/review UI changes. This is only the build-time
  discovery/prerender hardening for already approved posts.
- Static posts stay in the trusted HTML path because existing source posts are
  authored as HTML/Markdown hybrids; generated posts use escape-first rendering.

## Deferred

- Browser E2E against a live approved generated blog post remains out of scope;
  the operator can validate live after deployment by setting the feed/base env.
- A shared public blog wire-model module between runtime TS and build-time Node
  remains deferred because the current Vite build scripts are plain `.mjs`
  modules and the existing runtime adapter is TS/browser-oriented.
- Parked hardening: none. `HARDENING.md` and `ATLAS-HARDENING.md` were scanned;
  existing blog entries are generator content-quality issues, not this
  sitemap/prerender build bridge.

## Verification

- `cd atlas-intel-ui && npm ci` - passed; npm reports the existing 6 audit
  findings already parked in `HARDENING.md`.
- `cd atlas-intel-ui && npm run test:blog-generated-sitemap-prerender` - 9
  passed.
- `cd atlas-intel-ui && npm run test:blog-public-generated-posts` - 4 passed.
- `cd atlas-intel-ui && npm run test:sitemap-bridge` - 10 passed.
- `cd atlas-intel-ui && npm run test:landing-page-prerender` - 4 passed.
- `cd atlas-intel-ui && npm run lint` - passed.
- `cd atlas-intel-ui && npm run build` - passed; baseline build generated 17
  sitemap URLs and 16 public prerendered routes with no generated blog feed
  configured.
- `cd atlas-intel-ui && VITE_PUBLIC_BLOG_POSTS_URL=<data-url fixture> npm run
  build` - passed; fixture build generated 18 sitemap URLs and 17 public
  prerendered routes, including `/blog/generated-prerender-fixture`.
- `cd atlas-intel-ui && rg -n
  "generated-prerender-fixture|<script>alert|javascript:alert|&lt;script&gt;"
  dist/sitemap.xml dist/blog/generated-prerender-fixture/index.html` -
  confirmed the generated URL was emitted, raw `<script>` was escaped, and the
  unsafe markdown link did not retain an `href`.
- `cd atlas-intel-ui && npm run verify:blog-geo` - verified 14 blog pages.
- `cd atlas-intel-ui && npm run verify:landing-page-geo` - passed by skipping
  because no generated landing-page sitemap entries were present locally.
- `bash scripts/local_pr_review.sh --current-pr-body-file
  /home/juan-canfield/Desktop/blog-generated-sitemap-prerender-pr-body.md` - to
  run.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 138 |
| Generated blog bridge + declaration | 215 |
| Focused bridge tests | 170 |
| Vite sitemap/prerender wiring | 179 |
| package/workflow enrollment | 4 |
| **Total** | **709** |

The slice may land slightly over the 400 LOC target because it needs both the
strict build bridge and mocked negative fixtures to prove malformed public API
envelopes fail closed. Splitting the bridge and Vite wiring would leave no
usable generated-blog sitemap/prerender path; splitting the negative fixtures
would violate AGENTS.md §3i for the new build-time parser/bridge.

Final diff summary: 7 files, +655 / -54.
