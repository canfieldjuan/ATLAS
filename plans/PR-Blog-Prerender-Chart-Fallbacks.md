# PR-Blog-Prerender-Chart-Fallbacks

## Why this slice exists

PR-Blog-GEO-Visible-Article-HTML made blog article bodies visible in the
prerendered Atlas Intel HTML. That closed the empty-shell crawler issue, but
the first implementation removed chart placeholders from the static body.

Several public blog posts use charts to expose evidence such as migration
destinations, brands losing customers, and safety consequence counts. For GEO
and AEO, no-JavaScript crawlers should see a readable evidence fallback instead
of losing those chart-backed facts.

## Scope (this PR)

1. Parse source post chart specs during the existing Vite blog source scan.
2. Replace `{{chart:id}}` placeholders with static chart fallback figures in
   prerendered blog route HTML.
3. Render each fallback as a `figcaption` plus table using the chart source data.
4. Extend the blog GEO prerender verifier to reject leftover chart placeholders.
5. Extend the verifier to assert each source chart has a visible fallback.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-Prerender-Chart-Fallbacks.md` | Plan doc for this slice. |
| `atlas-intel-ui/vite.config.ts` | Render static chart fallback figures into prerendered blog bodies. |
| `atlas-intel-ui/scripts/verify-blog-geo-prerender.mjs` | Verify chart placeholders are removed and chart fallback titles are visible. |

## Mechanism

The source post files already contain JSON-shaped `charts` arrays. The Vite
prerender plugin now parses that field alongside `slug`, `title`, `date`,
`author`, and `content`. When rendering the static article body, each
`{{chart:id}}` placeholder is replaced with:

- `figure[data-prerendered-chart]`
- a visible `figcaption` using the chart title
- a simple HTML table using the chart's `x_key` and configured bar series

Interactive chart rendering remains owned by the React runtime. This slice only
adds no-JavaScript HTML fallbacks for crawlers.

## Intentional

- No React component changes.
- No generated blog content changes.
- No changes to chart interactivity.
- No attempt to render SVG/canvas charts at build time.
- No claim that GEO guarantees AI-engine placement.

## Deferred

- Replacing regex source parsing with a shared manifest or build-time source
  module import.
- Richer chart fallback prose summaries.
- FAQ fallback verification when Atlas Intel blog sources include FAQ entries.

## Verification

- Atlas UI lint.
- Atlas UI production build.
- Blog GEO prerender verification.
- Whitespace diff check.
- Local PR review.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~70 |
| Vite chart fallback rendering | ~75 |
| Publish verifier | ~30 |
| Total | ~175 |
