# PR-Blog-Fix-Misattributed-CRM-Quotes

## Why this slice exists

A corpus re-audit with the `seo-geo-aeo-blog-post` skill flagged
`top-complaint-every-crm-2026-04` as the single worst evidence-integrity
post: 6 misattribution-disclaimer occurrences -- prose that admits its own
quotes are wrong. The skill treats this as a hard-stop, "a category worse
than fabricated evidence because it normalizes data-pipeline failure."

The post shipped with quote blocks that fall into three broken shapes:

- **Empty blockquotes** -- attribution with no quote text
  (`<blockquote><p>-- reviewer on Reddit</p></blockquote>`), each followed by
  a disclaimer like *"That quote is from a compensation discussion, not a CRM
  review."*
- **Wrong-vendor quote** -- a Salesforce review-form prompt inside the Copper
  section, followed by *"That quote is misattributed in the source data (it
  references Salesforce, not Copper)."*
- **Review-form-prompt "quotes"** -- G2's reviewer question
  (`"What do you like best about Zoho CRM"`) presented as a customer quote.

These are upstream quote-pipeline failures (extraction returned attribution
without text, or matched on a surface keyword without checking relevance).
The disclaimers were published as hedges instead of fixing the data.

## Scope (this PR)

1. Remove the 4 empty blockquotes, the wrong-vendor Salesforce block, and the
   Zoho review-form-prompt block -- 6 broken quote blocks total.
2. Remove the 6 accompanying misattribution/hedge disclaimer paragraphs.
3. Preserve every legitimate analytical claim from those paragraphs (timing
   patterns, retention anchors, feature-gating) as plain prose, so no real
   insight is lost.
4. Leave untouched the 3 on-topic, right-vendor Reddit quote fragments
   (Salesforce x2, Zoho x1) that are real and correctly attributed.

### Files touched

- `atlas-churn-ui/src/content/blog/top-complaint-every-crm-2026-04.ts`
- `plans/PR-Blog-Fix-Misattributed-CRM-Quotes.md`

## Mechanism

Per the skill's evidence-integrity fix hierarchy -- (1) restore the real
quote, (2) paraphrase with attribution, (3) omit the block -- option (1) is
unavailable (no access to the original review source for these rows), so each
broken block is omitted. The lead-in line (`One reviewer on Reddit framed it
this way:`) and the disclaimer are removed with it; any real analytical claim
the disclaimer carried is rewritten as a direct statement (e.g. "Copper
reviews show a clear timing pattern: pricing complaints spike near contract
renewal periods").

The post keeps its real evidence: per-vendor review counts and urgency scores
(first-party Churn Signals data), complaint-theme breakdowns, and the three
genuine Reddit quotes. Honest stat-based analysis replaces fabricated quotes.

## Intentional

- **Omit, not paraphrase-as-quote.** The removed blocks had no recoverable
  quote text, so they are dropped entirely rather than converted to invented
  quotes -- the skill forbids publishing a disclaimer or a fabricated stand-in.
- **Three blockquotes kept.** The Salesforce ("moving off a Salesforce overlay
  provider", "using Salesforce for a couple years") and Zoho ("swapped Zoho
  CRM/One for Monday CRM") fragments are real, on-topic, and correctly
  attributed; the analyzer does not flag them.
- **Data-only fix.** The upstream generator gate that emitted attribution
  without quote text (and matched off-topic quotes) is the root cause; that
  fix lands with the corpus-wide empty-blockquote slice (174 occurrences
  across 44 posts), where the generator change can be made and tested once.

## Deferred

- **Generator quote-relevance + non-empty-quote gate.** Folded into the
  follow-up empty-blockquote slice so the producer fix is made and verified
  against the full corpus rather than piecemeal.
- **The other 43 posts with empty blockquotes.** This PR fixes only the
  4 in this post (incidental to removing its misattributed blocks); the
  remaining 170 occurrences are the next slice.

## Verification

- `seo-geo-aeo-blog-post` analyzer, single-post deep-dive:
  `node scripts/audit-published-posts.js
  --repo=<repo> --slug=top-complaint-every-crm-2026-04` -> all detectors `0`,
  `Clean posts: 1`, `0 CRITICAL` (was 6 misattribution + 4 empty blockquotes).
- `npm run build` (atlas-churn-ui) -> `built in 4.38s`, `Pre-rendered 83
  public routes`, no TS errors (the `.ts` still compiles).
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `top-complaint-every-crm-2026-04.ts` (6 blocks removed, prose preserved) | ~30 |
| Plan doc | ~100 |
| **Total** | **~130** |
