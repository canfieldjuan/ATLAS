# PR-Blog-Paraphrase-Excluded-A

Ownership lane: `content-ops/blog-paraphrase-excluded-a`

## Why this slice exists

Second slice of the excluded-source-quote fix (after #834). The widened
`excluded_source_quote` detector found 15 posts quoting allowlist-excluded
sources (Capterra/Trustpilot/TrustRadius/Software Advice). Per the maintainer's
call, paraphrase them: drop the source name + verbatim form, keep the reviewer's
substance. This batch covers 7 posts; the remaining 6 + the Slashdot drops follow
in sibling slices.

These quotes are mostly INLINE (embedded in prose with "writes:"/"notes:"), so
paraphrasing means rewriting the clause in place, not removing a blockquote.
Allowlist quotes in the same paragraphs (G2, Gartner, PeerSpot, Reddit) are left
verbatim.

## Scope (this PR)

- **azure-vs-salesforce-2026-03** (5): paraphrase 2 TrustRadius, 2 Software Advice,
  1 Capterra inline quotes (L142/143/146/148/154); keep the G2, Gartner, PeerSpot,
  Reddit quotes.
- **clickup-deep-dive-2026-04** (1): paraphrase the Trustpilot price-jump quote
  (L173) -- keep the "$9 -> $29" figures.
- **insightly-vs-zoho-crm-2026-04** (1): drop "on Trustpilot" from the L163
  reported-speech line (already paraphrase-style; just the attribution).
- **jira-vs-trello-2026-03** (1): remove the TrustRadius `<p>` from the L290-293
  blockquote (keep the Reddit quote), paraphrase it to a prose sentence after.
- **mailchimp-deep-dive-2026-04** (1): paraphrase the Trustpilot "goodbye to
  Mailchimp" quote (L191).
- **metabase-vs-tableau-2026-04** (1): drop "on Capterra" from the L140
  reported-speech clause (keep the G2 quote).
- **microsoft-teams-vs-notion-2026-04** (1): paraphrase the Trustpilot
  "Good riddance!" displacement quote (L142); keep the Reddit Coda quote.

### Files touched

- `plans/PR-Blog-Paraphrase-Excluded-A.md`
- `atlas-churn-ui/src/content/blog/azure-vs-salesforce-2026-03.ts`
- `atlas-churn-ui/src/content/blog/clickup-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/insightly-vs-zoho-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/jira-vs-trello-2026-03.ts`
- `atlas-churn-ui/src/content/blog/mailchimp-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/metabase-vs-tableau-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-teams-vs-notion-2026-04.ts`

## Mechanism

Excluded source -> the quote can't be shown per the blog source allowlist, so the
clause is rewritten as reported speech (substance kept, source name + quotation
marks dropped). No invention: each paraphrase restates the existing quote. Where a
paraphrase would otherwise leave a dangling "One reviewer notes:" lead-in (jira),
the lead-in is folded into the sentence.

## Intentional

- **Paraphrase, not drop.** Keeps the reviewer insight; only the policy-violating
  attribution/verbatim form is removed.
- **Allowlist quotes untouched.** G2/Gartner/PeerSpot/Reddit quotes in the same
  paragraphs stay verbatim.
- **Methodology declarations untouched.** insightly L148 (and similar) name
  excluded sources in source-distribution prose, not quotes -- the parked
  corpus-composition question (ATLAS-HARDENING, via #834), left as-is.

## Deferred

- Excluded-source paraphrase for the remaining 6 posts (real-cost-of-woocommerce,
  switch-to-clickup-04, switch-to-shopify, teamwork, top-complaint, woocommerce)
  -- next slice.
- Slashdot x9 drop -- separate slice.

## Review follow-up (#837 review)

The reviewer caught a real blind spot: the `excluded_source_quote` detector only
matched the on-form (`reviewer on <Source>`) and missed the **adjective form**
(`<Source> reviewer`), so two posts still carried excluded-source quotes that
audited as 0. Fixed:
- **Detector widened** (untracked auditor): now matches `<role> on <Platform>` AND
  `<Platform> <role>`; a bare dash-form was tried and dropped (it false-positived
  on parenthetical em-dashes in methodology lists, e.g. jira). `--self-test` ALL
  PASS (+3 fixtures, incl. the methodology-FP guard).
- **microsoft-teams-vs-notion**: paraphrased the remaining Trustpilot quotes (the
  verbatim Windows-11 quote, repeated 2x; the embedded "$288 / limited usage and
  product complexity", repeated 4x) -- dropped the `Trustpilot reviewer`
  attribution and the verbatim form.
- **insightly-vs-zoho-crm**: dropped the source name from 4 reported-speech
  attributions (`A Capterra reviewer ...` x2, `... Trustpilot reviewer ...` x2).
- jira-vs-trello was a detector FALSE POSITIVE under the bare dash-form (its
  Capterra/Trustpilot are methodology mentions, parked) -- now clean, no change.

## Verification

- All 7 posts re-audited under the WIDENED detector (`--slug=`):
  `excluded_source_quote` = 0, no findings of any kind. grep confirms no residual
  `<excluded> reviewer` / `reviewer on <excluded>` attributions.
- `excluded_source_quote` detector `--self-test`: ALL PASS.

## Estimated diff size

| Area | LOC |
|---|---:|
| azure-vs-salesforce (5 paraphrases) | ~10 |
| 6 single-quote posts | ~12 |
| review follow-up (microsoft-teams-vs-notion + insightly) | ~20 |
| Plan doc | ~98 |
| **Total** | **~140** |
