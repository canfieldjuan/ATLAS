# PR-Blog-Affiliate-Disclosure-Prerender

## Why this slice exists

`BlogArticleView.tsx` renders an FTC affiliate-disclosure block for any post
with affiliate content, but it does so in a React `useEffect` -- so the
disclosure only exists after the SPA hydrates. The audience that most needs to
see a disclosure in the *static* HTML is exactly the audience that never runs
the JS:

- **AEO crawlers** (GPTBot, PerplexityBot, ClaudeBot, Bingbot) that quote and
  cite blog content without executing the app.
- **Social unfurl bots** (LinkedIn, Slack, iMessage) that render a preview
  from the raw HTML.

The prerender plugin already emits per-route `dist/blog/<slug>/index.html`
with head metadata + JSON-LD, but the `<body>` is still the empty
`<div id="root"></div>` shell -- so a no-JS agent sees no disclosure on the 28
posts that carry an affiliate link. This slice injects the disclosure into the
prerendered HTML for those posts so the FTC notice is present without JS.

## Scope (this PR)

1. Add an `AFFILIATE_DISCLOSURE_NOSCRIPT` constant: a `<noscript>`-wrapped
   disclosure block whose copy mirrors `BlogArticleView.tsx`.
2. Add an optional `bodyHtml` field to `PrerenderedRoute`.
3. In the blog-route loop, set `bodyHtml` on posts whose `.ts` source has a
   non-empty `affiliate_url` (or inlines Monday.com's affiliate URL) -- parity
   with `BlogArticleView.hasAffiliateContent`.
4. In the render loop, inject `bodyHtml` immediately before `<div id="root">`.

### Files touched

- `atlas-churn-ui/vite.config.ts`
- `plans/PR-Blog-Affiliate-Disclosure-Prerender.md`

## Mechanism

The disclosure is wrapped in `<noscript>` and injected *before*
`<div id="root">`, i.e. outside React's mount container:

- **No-JS agents** (crawlers, unfurl bots, no-JS humans) render the
  `<noscript>` content -- the FTC notice and a `/methodology` link.
- **JS browsers** never render `<noscript>` content, so there is no visual
  flash and no duplicate of the React-rendered disclosure. Because the block
  is outside `#root`, React's `createRoot().render()` never touches it.

The affiliate predicate matches `BlogArticleView.hasAffiliateContent`:
`/"affiliate_url"\s*:\s*"[^"]+"/` (a non-empty `data_context.affiliate_url`)
OR `try.monday.com` inlined in the body. The disclosure copy is held as a
string constant kept in sync with the component's wording.

## Intentional

- **Disclosure only -- no `rel="sponsored"` rewriting.** The original
  crawler-visibility idea paired the disclosure with tagging affiliate anchors
  `rel="sponsored"`. That half is moot here: the prerendered `<body>` has no
  `<a>` tags at all (the article prose is rendered by React at runtime), so
  there is nothing to decorate. The disclosure block is the entire
  crawler-visible surface achievable without full body prerender.
- **`<noscript>`, not a visible body block.** A plain injected block would
  flash before React mounts and duplicate the component's disclosure;
  `<noscript>` is the precise semantic for "no-JS agents only."
- **Copy duplicated, not imported.** The plugin runs at build time over raw
  `.ts` source text and cannot import the React component; the disclosure
  string is duplicated with a comment pointing at the source of truth.
- **Inline-styled, minimal.** The audience is mostly text-extracting bots; a
  small inline style covers the rare no-JS human without pulling in CSS.

## Deferred

- **Full body prerender / SSR.** Rendering the article prose (and its
  affiliate `<a>` tags, which could then carry `rel="sponsored"` statically)
  into the prerendered HTML is a much larger slice -- React SSR or a
  markdown-to-HTML build step. The disclosure is the high-value, low-risk
  subset; full body prerender is the follow-up if AEO citation quality needs
  the prose itself in the static HTML.
- **Per-post custom OG image.** Unchanged from prior SEO-infra work.

## Verification

- `npm run build` (in `atlas-churn-ui`) -> `built in 4.44s`, `Sitemap
  generated with 83 URLs`, `Pre-rendered 83 public routes`, no TS errors
  (route count unchanged -- the change augments bodies, it does not add
  routes).
- `grep -rl "may contain affiliate links" dist/blog/*/index.html | wc -l`
  -> `28` (matches the 28 posts with a non-empty `affiliate_url`).
- Affiliate post `dist/blog/asana-deep-dive-2026-04/index.html` body is
  `<body>\n    <noscript><div ...><strong>Disclosure:</strong> ... See our
  <a href="/methodology">methodology</a>.</div></noscript>\n    <div
  id="root"></div>` -- disclosure + methodology link present, injected before
  `#root`.
- Non-affiliate post `dist/blog/hubspot-deep-dive-2026-04/index.html`:
  `grep -c "may contain affiliate links"` -> `0` (correctly absent).
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `atlas-churn-ui/vite.config.ts` (constant + field + predicate + injection) | ~40 |
| Plan doc | ~115 |
| **Total** | **~155** |
