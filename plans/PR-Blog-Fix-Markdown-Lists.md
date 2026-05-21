# PR-Blog-Fix-Markdown-Lists

## Why this slice exists

A per-post correctness audit (logged in `reports/blog-audit-findings.md`,
defect class **D9**) found markdown bullet lists emitted *inside* `<p>` blocks:

```
<p><strong>Top pain points:</strong>
- Overall dissatisfaction
- Pricing
- User experience</p>
```

The markdown->HTML build step treats `<p>...</p>` as a raw HTML block and does
not convert the inner list, so it renders as literal "- item" text instead of
a bulleted list -- visibly broken on the page.

This shape was invisible to the SEO analyzer: the old `detectMarkdownInHtml`
regex (`<p>[^<]*?\n\s*[-*]`) stopped at the first inner tag, so a list after a
`<strong>label:</strong>` never matched (it reported 0). The analyzer was
hardened to scan the whole `<p>` block, which surfaced **61 occurrences across
10 posts**. This PR fixes those 10 posts.

## Scope (this PR)

Convert each markdown-in-`<p>` block into a label paragraph plus a real list:

```
<p><strong>Top pain points:</strong></p>
<ul>
<li>Overall dissatisfaction</li>
<li>Pricing</li>
<li>User experience</li>
</ul>
```

### Files touched

- `plans/PR-Blog-Fix-Markdown-Lists.md`
- `atlas-churn-ui/src/content/blog/crm-landscape-2026-04.ts` (15)
- `atlas-churn-ui/src/content/blog/marketing-automation-landscape-2026-04.ts` (11)
- `atlas-churn-ui/src/content/blog/helpdesk-landscape-2026-04.ts` (10)
- `atlas-churn-ui/src/content/blog/jira-vs-trello-2026-03.ts` (8)
- `atlas-churn-ui/src/content/blog/mailchimp-deep-dive-2026-04.ts` (5)
- `atlas-churn-ui/src/content/blog/tableau-deep-dive-2026-04.ts` (4)
- `atlas-churn-ui/src/content/blog/salesforce-deep-dive-2026-04.ts` (3)
- `atlas-churn-ui/src/content/blog/copper-deep-dive-2026-04.ts` (2)
- `atlas-churn-ui/src/content/blog/switch-to-woocommerce-2026-04.ts` (2)
- `atlas-churn-ui/src/content/blog/microsoft-defender-for-endpoint-deep-dive-2026-04.ts` (1)

## Mechanism

A one-off transform (`/tmp/fix_markdown_lists.py`, not committed) mirrors the
detector: for each `<p>` block containing a `\n- ` / `\n* ` bullet line, it
splits the pre-bullet prefix into its own `<p>` (kept verbatim, e.g. the
`<strong>label:</strong>`) and renders the bullet items as `<ul><li>`. Item
text is preserved verbatim (no rewording, no fabrication). Content-field only
-- no `affiliate_url` / `seo` / `faq` / metadata touched, so prerender,
sitemap, and disclosure wiring are unaffected.

The analyzer-side detector change (extended `markdown_in_html`, plus a new
`form_prompt_quote` detector for the related D1 class) lives in the
`seo-geo-aeo-blog-post` skill, not in this PR.

## Intentional

- **Real `<ul><li>`, not a re-flattened paragraph.** These are genuine lists;
  rendering them as proper HTML lists is the correct fix and improves
  AEO/snippet eligibility.
- **Verbatim items.** The transform only restructures; it does not change item
  text, so no evidence or numbers are altered.
- **Content-field only**, one transform applied uniformly to the 10 flagged
  posts.

## Deferred

- **D1 (form-prompt quotes)** -- 31 posts, separate cleanup.
- **Generator fix** -- emit real `<ul>` rather than markdown-in-`<p>` so the
  shape cannot recur (the root cause).
- D2/D3/D4/D7/D8 -- catalogued, not yet addressed.

## Verification

- `seo-geo-aeo-blog-post` analyzer across the corpus -> `Markdown in <p>
  tags 0 / 0` (was `10 / 61`); `0 CRITICAL`.
- HTML well-formedness validator on all 10 edited posts -> 0 issues
  (balanced `<ul>`/`<li>`).
- `npm run build` (atlas-churn-ui) -> `built in 4.47s`, `Pre-rendered 82
  public routes`, no TS errors.
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| 10 blog `.ts` posts (61 markdown blocks -> `<ul><li>`) | ~245 |
| Plan doc | ~90 |
| **Total** | **~335** |
