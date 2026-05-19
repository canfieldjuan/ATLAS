# PR-Blog-Affiliate-Sponsored-Rel

## Why this slice exists

`atlas-churn-ui` already renders a clearly-visible affiliate
disclosure block at the top of every post that has an affiliate
link (`BlogArticleView.tsx` lines 192-204): "This article may
contain affiliate links. If you purchase through these links, we
may earn a commission ...". That covers the FTC visibility
requirement.

What is NOT in place: the `rel="sponsored"` attribute on the actual
affiliate anchor tags. Google's link attribution guidelines say
affiliate links must carry `rel="sponsored"` (or
`rel="nofollow sponsored"`) so the link is correctly classified as
a commercial relationship. The reports/affiliate-system-
investigation.md doc (committed in PR #617) flagged this as a real
legal-risk gap.

Current state:

- **CTA button anchor** (`BlogArticleView.tsx:217-224`): has
  `rel="noopener noreferrer"` regardless of whether the CTA points
  at an affiliate URL or a cal.com booking link.
- **Inline affiliate anchors inside post body content**: the
  `decorateAffiliateLinks` helper iterates anchors whose `href`
  matches `data_context.affiliate_url` and adds visual decoration
  ONLY when the `highlightAffiliateLinks` admin-preview flag is
  true. In production rendering the flag defaults to `false` so the
  function early-returns and the inline affiliate anchors get no
  `rel` tagging at all.

Result: even when readers see the visible disclosure block, the
affiliate anchors themselves don't declare the commercial
relationship to Google or to browser link-info dialogs.

## Scope

1. Refactor `decorateAffiliateLinks` so the `rel="sponsored"`
   tagging path runs whenever there's an affiliate URL, regardless
   of the `highlightAffiliateLinks` preview flag. The visual outline
   + `data-preview-affiliate` markers remain gated on the preview
   flag.
2. Existing `rel` tokens (`noopener`, `noreferrer`) on prose-author
   anchors must be preserved when `sponsored` is added -- the
   security attributes don't go away.
3. Update the CTA button anchor to add `sponsored` to its `rel` when
   `cta.mode === 'affiliate'`. Generic-mode CTAs (cal.com booking,
   internal report) stay with just `noopener noreferrer`.
4. Add tests covering all five branches: inline-affiliate gets
   sponsored in production rendering, existing rel tokens are
   preserved when sponsored is added, non-affiliate anchors are
   untouched, affiliate-mode CTA gets sponsored, generic-mode CTA
   does not get sponsored.

### Files touched

- `atlas-churn-ui/src/components/BlogArticleView.tsx`
- `atlas-churn-ui/src/components/BlogArticleView.test.tsx`
- `plans/PR-Blog-Affiliate-Sponsored-Rel.md`

## Mechanism

`decorateAffiliateLinks(html, { affiliateUrl, highlightAffiliateLinks })`
is split into two concerns inside the same function:

1. **Always**: parse `html` with `DOMParser`, iterate
   `a[href]` elements, find the anchors whose `href` exactly
   matches `affiliateUrl`. For each: read the existing `rel`
   attribute, tokenise into a Set, add `sponsored`, write back.
   Using a Set deduplicates so `<a rel="sponsored noopener">` plus
   another sponsored tag won't produce duplicate tokens.
2. **Preview-only** (gated on `highlightAffiliateLinks`): apply the
   dashed outline style + `data-preview-affiliate="true"` +
   `title="Affiliate link"`. These markers serve the admin draft
   reviewer scanning a post for affiliate placements.

The function early-returns when `affiliateUrl` is empty or
`DOMParser` is unavailable (SSR). Failure during parse falls back
to the un-modified HTML.

The CTA anchor in JSX uses a conditional `rel` value:
```jsx
rel={
  cta.mode === 'affiliate'
    ? 'sponsored noopener noreferrer'
    : 'noopener noreferrer'
}
```

`cta.mode` comes from `resolveBlogArticleCta` (`src/lib/blogCta.ts`)
which returns `'affiliate'` only when both `data_context.affiliate_url`
and a partner name are present, so generic posts can't accidentally
get tagged.

## Intentional

- `sponsored` is the canonical Google attribute for paid /
  affiliate relationships. `nofollow` is the older, more general
  signal. Modern Google guidance treats `sponsored` as implying
  the relationship without `nofollow`; we don't add `nofollow`
  here to keep the rel list short and the intent specific.
- Existing rel tokens (typically `noopener noreferrer`) are
  preserved via a token Set. The merge logic is order-stable
  enough for assertions (`relTokens.contains('sponsored')` etc.)
  even though the rendered string order isn't guaranteed.
- The admin diagnostic anchor in `BlogReview.tsx:806-813` (which
  displays the affiliate URL as a clickable link in a sidebar
  panel) is NOT updated. That anchor is admin-only, not a
  customer-facing recommendation; tagging it as sponsored would
  be misleading.
- `hasAffiliateContent` (`BlogArticleView.tsx:37`) still uses a
  hardcoded substring check for `try.monday.com` as a fallback.
  That's pre-existing technical debt; this PR doesn't address it.

## Deferred

- Per-post unique OG images (currently every post shares
  `og-default.png`).
- Affiliate-program revenue reconciliation (clicks tracked in
  `affiliate_clicks` table, but no link to commission data).
- A more rigorous affiliate-anchor detector that walks all known
  partner URLs from the `affiliate_partners` table instead of just
  matching `data_context.affiliate_url`. Out of scope -- the
  generator only injects ONE partner per post anyway.

## Verification

- `npx vitest run src/components/BlogArticleView.test.tsx` ->
  6 tests passed (1 existing + 5 new).
- `npm run build` in `atlas-churn-ui` -> 83-URL sitemap, no TS
  errors, 83 prerendered routes.
- `git diff --check` -> passed.
- `scripts/local_pr_review.sh` -> expected to pass.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `BlogArticleView.tsx` (decorator refactor + CTA rel) | ~40 |
| `BlogArticleView.test.tsx` (5 new tests + comments) | ~115 |
| Plan doc | ~120 |
| **Total** | **~275** |
