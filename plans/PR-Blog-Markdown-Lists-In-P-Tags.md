# PR-Blog-Markdown-Lists-In-P-Tags

## Why this slice exists

The seo-geo-aeo-blog-post skill's baseline analyzer (introduced in PR
#612, refreshed in #634/#636/#638) flagged that 14 of 79 published
blog posts under `atlas-churn-ui/src/content/blog/` contain markdown
bullet syntax (`- item`) nested inside `<p>` tags. The BlogPost
contract says `content` is an HTML string and must use semantic tags
(`<ul><li>`) -- the markdown leaks render as literal dashes in the
browser, breaking list semantics and accessibility.

Total occurrences across the 14 posts: 93 paragraph blocks containing
list-like markdown. Affected slugs include the full 2026-04 batch of
deep dives, switch-to guides, head-to-head comparisons, and best-fit
guides. The 2026-03 Shopify switch guide is also affected.

This slice converts the 14 already-published artifacts to use semantic
`<ul><li>` blocks. It does NOT add a generator-side guard to prevent
recurrence -- that lives in a separate slice (the seo-geo-aeo-blog-post
skill audit catches the pattern post-publish; an upstream check during
generation would require parsing the LLM's HTML output for the same
shape).

## Scope

1. Convert every `<p>` block in the 14 affected post `.ts` files
   that contains `\n- item` (or `\n* item`) lines to:
     * a `<p>` carrying any prose preceding the bullets, and
     * a sibling `<ul>` with each bullet rendered as a `<li>`.
2. Preserve all other content verbatim, including non-bullet `<p>`
   blocks, charts, FAQ items, and the BlogPost metadata.
3. Verify with the analyzer that `Markdown in <p> tags` count drops
   from 14 / 57 to 0 / 0.
4. Verify with `npm run build` in `atlas-churn-ui` that the
   converted posts still compile and render.

### Files touched

- `atlas-churn-ui/src/content/blog/best-hr-hcm-for-51-200-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-project-management-for-201-1000-2026-04.ts`
- `atlas-churn-ui/src/content/blog/brevo-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/close-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/help-scout-vs-zendesk-2026-04.ts`
- `atlas-churn-ui/src/content/blog/insightly-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/linode-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/power-bi-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/sentinelone-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-clickup-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-shopify-2026-03.ts`
- `atlas-churn-ui/src/content/blog/switch-to-zoho-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/why-teams-leave-slack-2026-04.ts`
- `atlas-churn-ui/src/content/blog/woocommerce-deep-dive-2026-04.ts`
- `plans/PR-Blog-Markdown-Lists-In-P-Tags.md`

## Mechanism

A one-off Node script scans each affected `.ts` file and applies a
regex-based replacement against the `content` template literal. For
each `<p>...</p>` block, the script:

1. Splits the inner content on newlines.
2. Walks the lines as a state machine: `prose` -> `bullets` ->
   `trailing`. A bullet line is `^[ \t]*[-*][ \t]+(.+)$`.
3. Emits one `<p>...</p>` carrying the prose lines (if any), followed
   by `<ul>` with one `<li>` per bullet, followed by another `<p>`
   for trailing content (rare).
4. Continuation lines (non-bullet, non-empty) immediately after a
   bullet are appended to the previous bullet's text -- defensive
   against wrapped list items, though the current data does not
   contain any.

The script makes no other content edits. Posts that don't contain
the bullet-in-`<p>` pattern are skipped untouched.

## Intentional

- Surgical content fix: only the 14 affected `.ts` files are
  modified. No generator code change, no test changes, no schema
  changes.
- Bullets that already render as proper `<ul><li>` are unaffected --
  the regex only matches a `<p>` whose inner content has a
  newline followed by a `-` or `*` token, which `<ul>` blocks
  don't satisfy.
- Continuation-line behavior (appending wrap-around text to the
  prior bullet) is implemented but exercises no current data. It's
  documented in code for future generations.
- The seo-geo-aeo-blog-post audit catches the same pattern post-
  publish. After this PR the count goes to zero; if it climbs again
  in a future generation, the analyzer is the canary.

## Deferred

- Generator-side detection / refusal at draft time. A check inside
  `_apply_blog_quality_gate` could grep `content` for bullet-in-`<p>`
  patterns and refuse to publish. That's a separate slice and would
  require coordination with the LLM prompt to also avoid emitting
  markdown lists inside `<p>` blocks in the first place.
- Numbered-list (`1. item`) conversion. Current data does not
  contain numbered lists nested in `<p>` blocks; if a future post
  ships one, the converter will need an `<ol>` branch.
- Cleanup of other markdown leaks (e.g., `**bold**` or backtick
  spans inside `<p>`). The audit hasn't flagged these in production.

## Verification

- `node /home/juan-canfield/.claude/skills/seo-geo-aeo-blog-post/scripts/audit-published-posts.js --repo=atlas-churn-ui` -> `Markdown in <p> tags: 0 / 0` (was `14 / 57`).
- `npm run build` in `atlas-churn-ui` -> 83-URL sitemap, no TS errors.
- `git diff --check` -> passed.
- Spot-check on `close-deep-dive-2026-04.ts`: 4-item bullet block in
  the "Key findings" paragraph converted to `<p>...</p><ul><li>...</li></ul>`;
  surrounding prose preserved.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Affected post `.ts` files (14) -- insertions | ~600 |
| Affected post `.ts` files (14) -- deletions | ~420 |
| Plan doc | ~130 |
| **Total** | **~1150** |

Totals reflect raw insertions + deletions, not net LOC. The high
deletion count comes from each multi-line `<p>...\n- item\n- item...</p>`
block being rewritten as `<p>...</p>\n<ul>\n<li>...</li>\n...</ul>` --
each converted block grows from N+1 lines (prose + N bullets in one
`<p>`) to ~N+4 lines (`<p>`, `<ul>`, N `<li>`, `</ul>`).
