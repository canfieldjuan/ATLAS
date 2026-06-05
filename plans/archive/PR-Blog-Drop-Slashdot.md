# PR-Blog-Drop-Slashdot

Ownership lane: `content-ops/blog-drop-slashdot`

## Why this slice exists

Final slice of the undeclared/excluded quoted-source fix phase. Scoping
(DB-verified) found 9 posts that quote a Slashdot reviewer where EVERY quoted
Slashdot review is `enrichment_status = not_applicable` -- unenriched, uncounted,
and absent from the post's declared sources (e.g. tableau declares
"G2, Gartner, PeerSpot, Reddit / 439 enriched"; its Slashdot reviews are 7, 0
enriched). The quote pool pulled raw Slashdot text outside the analyzed corpus.
Per the maintainer's call, DROP these quotes (they aren't part of the analysis
the post describes) rather than declare a non-counted source.

This clears the `undeclared_quoted_source` class for the Slashdot subset; the
PeerSpot declares (#834) and excluded-source paraphrases (#834/#837/#838) handle
the rest.

## Scope (this PR)

Remove each Slashdot blockquote (9 posts, 13 quotes) plus its dedicated lead-in,
keeping prose coherent:

- **Simple drops** (lead-in + blockquote removed, follow-on already general):
  hubspot-deep-dive-2026-04, zoom-deep-dive-2026-04, tableau-deep-dive-2026-04
  (2 quotes), zoho-crm-deep-dive-2026-04, workday-deep-dive-2026-04,
  shopify-deep-dive-2026-04 (quote 1).
- **Drop + follow-on rework** (the post's analysis paragraph referenced the now-
  removed quote -- reworded to a standalone statement, since the Slashdot review
  is out-of-corpus): intercom-deep-dive-2026-04, switch-to-klaviyo-2026-04
  (2 quotes), shopify-deep-dive-2026-04 (quote 2 -- removed the duplicate quote +
  its "strongest positive quote" analysis; reworded the section-closing line so
  it no longer back-references a stripped quote).
- **Lead-in -> brief theme line** (kept a short non-quoted strength statement so a
  later sentence's referent still resolves): switch-to-zoho-crm-2026-04.

### Files touched

- `plans/PR-Blog-Drop-Slashdot.md`
- `atlas-churn-ui/src/content/blog/hubspot-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/intercom-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/shopify-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-klaviyo-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-zoho-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/tableau-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/workday-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/zoho-crm-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/zoom-deep-dive-2026-04.ts`

## Mechanism

The Slashdot quotes come from `not_applicable` reviews the analysis never
counted, so they're dropped, not paraphrased or declared. Where the surrounding
prose analyzed the dropped quote, the analytical point (the post's own synthesis)
is kept as a standalone statement with the quote framing removed; where it was
purely meta ("This is the strongest positive quote..."), it's removed.

## Intentional

- **Drop, not paraphrase/declare.** These reviews are outside the enriched/counted
  corpus (the zoho L158 quote-pool-vs-count item), so declaring Slashdot would
  misstate the sources and paraphrasing would still surface out-of-corpus text.
- **Follow-on referents fixed.** shopify's section-closing "The quote set is
  small ..." was reworded to "Though the quote set is small ..." so it no longer
  reads as a back-reference to a stripped quote (the drop otherwise introduced an
  `orphaned_quote_reference`; verified fixed).

## Deferred

- **workday `prose_vs_chart_metric`**: PRE-EXISTING (verified on origin/main),
  fixed by the open #831; this branch is off origin/main so it still shows here.
  Not touched (would conflict with #831). workday file-overlaps #831 on different
  lines (L164-167 here vs L180-190 there) -- auto-merges.
- Methodology source-list mentions of excluded sources -- parked (ATLAS-HARDENING,
  via #834).

## Verification

- All 9 posts re-audited (`--slug=`): `undeclared_quoted_source` = 0; 8 fully
  clean; workday shows only the pre-existing `prose_vs_chart_metric=1` (#831),
  confirmed present on origin/main with my changes stashed.
- shopify orphan check: the drop initially introduced `orphaned_quote_reference=1`
  (the section-closing line lost its backing blockquote); reworded -> clean.

## Estimated diff size

| Area | LOC |
|---|---:|
| 9 posts (13 quote drops + follow-on rewords) | ~66 |
| Plan doc | ~80 |
| **Total** | **~146** |
