# PR-Blog-Strip-Empty-Blockquotes-Data

## Why this slice exists

A corpus re-audit with the `seo-geo-aeo-blog-post` skill found **170 empty
blockquotes across 43 published posts** -- the dominant CRITICAL finding.
Each is an attribution with no quote text:

```html
<blockquote>
<p>-- Client service manager, verified reviewer on G2</p>
</blockquote>
```

It renders as a citation containing only "-- Client service manager…" and no
quote -- broken for readers, an empty citation to an AI engine.

These are **legacy artifacts**, not a live generator bug. The producer's
quote gate (`_remove_unmatched_quote_lines`) gained attribution-only +
orphan-intro/disclaimer stripping in commit `946a8aad` (2026-05-19), with
144 lines of regression tests (`test_orphan_intro_*`,
`test_full_block_stripped_*`). The 43 affected posts were generated
2026-04-07 / 2026-04-10 -- before that fix. Verified empirically: feeding the
exact shape through the current stripper removes it cleanly. So the source is
already fixed; this PR remediates the data the old producer left behind.

This is the second half of the "both, sequenced" empty-blockquote work.
PR #692 fixed the 44th post (`top-complaint-every-crm-2026-04`, which also had
misattribution disclaimers needing prose judgment); this PR handles the
remaining 43. **Stacked on #692** so it does not re-touch that post.

## Scope (this PR)

Apply the generator's own strip logic retroactively to the already-converted
HTML of the 43 legacy posts:

1. Remove each empty blockquote block.
2. Remove an immediately-following orphan disclaimer `<p>` (the
   acknowledged-misattribution patterns from `_ORPHAN_DISCLAIMER_RES`).
3. Remove the preceding lead-in: if the `<p>` is a pure lead-in (short, ends
   with `:`), drop it; if the lead-in clause is fused with substantive prose
   (`<p>The pricing backlash is specific. One business owner called out the
   $55 monthly cost:</p>`), drop only the trailing lead-in sentence and keep
   the real content (`<p>The pricing backlash is specific.</p>`).
4. Leave generic follow-ons (`This pattern recurs:`) untouched -- the same
   accepted behavior as the fixed generator.

A post-strip well-formedness audit (HTML tag balance + dangling-lead-in scan,
neither of which `tsc` or the SEO analyzer checks) surfaced two follow-ups
folded into this PR:

5. **12 dangling lead-ins** across 8 posts where the fused-clause trim missed
   a `.` immediately followed by `</strong>` (and over-long single-clause
   lead-ins), leaving `...One office manager reported hiring workflow bugs:`
   with no quote. Repaired with prose judgment: `:`->`.` where the clause
   carries real substance (the skill's paraphrase-with-attribution option),
   clause dropped where it was pure scaffolding.
6. **`real-cost-of-copper` deep-repair.** The audit found this post was
   contaminated by keyword-matched off-topic reviews (copper-the-metal, audio
   cables, an ISP complaint, a health "28 day cycle" note), each shipped with
   a disclaimer admitting it -- the same evidence-integrity hard-stop the
   skill flags, in wording the analyzer's regex missed. Removed the Spectrum
   and Optimus blockquotes, the five-item audio-gear "strengths" list, the
   cryptic health-quote paragraph, and FAQ item #4 (which fed FAQPage schema);
   reworded the prose that referenced them. Legitimate Copper-CRM analysis
   (the 90/738 pricing stat, urgency distribution, the "spreadsheet with a
   pretty interface" critique, satisfaction anchors) is preserved.
7. **`close-deep-dive` pulled.** Re-reading the post during the audit showed
   its evidence layer is almost entirely keyword-match noise: the whole
   "What Reviewers Actually Say" section quotes off-topic reviews ("closing
   the accounts", "close my account", a G2 form prompt, "CloseAI's deep
   research"), the competitor list is nonsense (Genshin characters, a Sony
   camera, Android -- the post admits they "do not align with the CRM
   category"), and two dangling disclaimers remain. Unlike copper there is no
   real evidence base to anchor a repair, so the post is removed
   (`git rm`; the `import.meta.glob` loader and the sitemap/prerender derive
   from the blog directory, so deleting the file de-publishes it) and the one
   inbound `related_slugs` reference (in `insightly-vs-zoho-crm-2026-04`) is
   dropped. Root cause is upstream: the common-word vendor name "Close" (like
   "Copper") keyword-matches unrelated reviews in the corpus -- a
   disambiguation gap to fix before these vendors are regenerated.

### Files touched

- `plans/PR-Blog-Strip-Empty-Blockquotes-Data.md`
- `atlas-churn-ui/src/content/blog/asana-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/azure-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-crm-for-51-200-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-hr-hcm-for-51-200-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-project-management-for-201-1000-2026-04.ts`
- `atlas-churn-ui/src/content/blog/brevo-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/clickup-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/close-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/crm-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/gusto-vs-workday-2026-04.ts`
- `atlas-churn-ui/src/content/blog/help-scout-vs-zendesk-2026-04.ts`
- `atlas-churn-ui/src/content/blog/helpdesk-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/hr-hcm-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/hubspot-vs-power-bi-2026-04.ts`
- `atlas-churn-ui/src/content/blog/insightly-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/insightly-vs-zoho-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/jira-vs-mondaycom-2026-04.ts`
- `atlas-churn-ui/src/content/blog/linode-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/looker-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/metabase-vs-tableau-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-defender-for-endpoint-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-teams-vs-notion-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-teams-vs-salesforce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/power-bi-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/project-management-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/real-cost-of-copper-2026-04.ts`
- `atlas-churn-ui/src/content/blog/real-cost-of-woocommerce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/sentinelone-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/shopify-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/slack-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/slack-vs-zoom-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-clickup-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-sentinelone-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-shopify-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-zoho-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-communication-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-e-commerce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-helpdesk-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-marketing-automation-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-project-management-2026-04.ts`
- `atlas-churn-ui/src/content/blog/why-teams-leave-azure-2026-04.ts`
- `atlas-churn-ui/src/content/blog/why-teams-leave-slack-2026-04.ts`
- `atlas-churn-ui/src/content/blog/wrike-deep-dive-2026-04.ts`

## Mechanism

A one-off transform (`/tmp/remediate_empty_blockquotes.py`, not committed)
ported the generator's `_remove_unmatched_quote_lines` rules to HTML: it
matches the analyzer's empty-blockquote shape
(`<blockquote>\s*<p>\s*--[^<]*</p>\s*</blockquote>`), so blockquotes that
carry a real quote `<p>` (two `<p>` elements) are never touched. Lead-in and
disclaimer handling mirror `_looks_like_intro_paragraph` (ends with `:`,
3-180 chars, not a heading/list) and `_ORPHAN_DISCLAIMER_RES`. The
fused-paragraph case splits on sentence boundaries and drops only the trailing
lead-in clause, preserving substantive prose.

No generator code changes -- the producer already does this for new posts.

## Intentional

- **Strip, not regenerate.** Regeneration was investigated and rejected: the
  producer builds date-suffixed slugs (`month_suffix = today.strftime("%Y-%m")`),
  so a re-run would create new `2026-05` files rather than overwriting the
  `2026-04` posts, would rewrite content wholesale (discarding reviewed work),
  and there is no single-slug regeneration entry point.
- **Generic follow-ons left in place.** Sentences like "This pattern recurs:"
  that followed a removed quote are kept, matching the generator's accepted
  behavior. Some now read as standalone analysis; detecting and rewriting them
  risks deleting real insight, so it is left to an optional later prose pass.
- **Fused lead-ins trimmed, not dropped whole.** Where a lead-in clause shared
  a paragraph with substantive prose, only the trailing clause was removed, so
  no real sentence is lost.
- **Stacked on #692.** Based on the misattribution-fix branch so the 44th post
  (`top-complaint-every-crm`) is not re-touched; until #692 merges, this PR's
  diff also shows that post's changes.

## Deferred

- **Optional follow-on prose polish.** Smoothing standalone "This pattern…"
  sentences that once introduced a quote -- cosmetic, not a CRITICAL finding.

## Verification

- `seo-geo-aeo-blog-post` analyzer across the corpus
  (`node scripts/audit-published-posts.js --repo=atlas-churn-ui`) ->
  `Clean posts: 79`, `0 CRITICAL`; `Empty blockquotes 0 / 0` (was 174 across
  44 before this slice + #692).
- `npm run build` (atlas-churn-ui) -> `built in 4.47s`, `Pre-rendered 83
  public routes`, no TS errors (all 43 posts still compile).
- Dry-run diff spot-checked on `gusto-vs-workday`, `jira-vs-mondaycom`
  (10 removals each): real content preserved, lead-in clauses cleanly
  trimmed, no broken HTML.
- Post-strip HTML well-formedness validator (tag balance + parser) across all
  posts -> 0 issues; dangling-lead-in scan -> 0 (was 12 before the
  follow-up fixes).
- Broader misattribution-disclaimer scan across the corpus -> only
  `real-cost-of-copper` (repaired here) and `close-deep-dive` (pulled here)
  carried analyzer-missed disclaimer variants; after both, the corpus audit
  reports `Clean posts: 78`, `0 CRITICAL`.
- `npm run build` -> `Sitemap generated with 82 URLs`, `Pre-rendered 82
  public routes` (was 83; close-deep-dive removed), no TS errors.
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| 42 blog `.ts` posts (170 blockquotes + 12 lead-in trims + copper repair) | ~780 |
| `close-deep-dive-2026-04.ts` removed (contaminated post pulled) | ~370 |
| `insightly-vs-zoho-crm` related_slugs reference dropped | ~1 |
| Plan doc | ~205 |
| **Total** | **~1356** |
