# PR-Blog-Fix-Undeclared-Hubspot-Clickup

Ownership lane: `content-ops/blog-fix-undeclared-hubspot-clickup`

## Why this slice exists

First slice of the `undeclared_quoted_source` / excluded-source-quote fix phase.
Scoping (DB-verified) split the quoted-source problem three ways: PeerSpot x2
(allowlist, just undeclared -> declare), Slashdot x9 (allowlist but quoted reviews
are `not_applicable` -> drop, separate slice), and excluded-source quotes
(Capterra/Trustpilot/TrustRadius/Software Advice -> paraphrase per the maintainer's
call). hubspot-deep-dive-2026-03 and switch-to-clickup-2026-03 are the two posts
that need BOTH a PeerSpot declare AND an excluded-source paraphrase, so they're
fixed together (avoids two PRs touching the same files).

A widened detector (`excluded_source_quote`, added to the untracked corpus
auditor) showed the real excluded-source-quote problem is 15 posts / 20
platform-occurrences -- broader than the 6 the `undeclared` detector caught,
because the old detector silently skipped excluded sources the post DECLARES
(e.g. hubspot-03 quotes Trustpilot while listing Trustpilot in its methodology).
This PR is the first batch; the rest follow in sibling slices.

## Scope (this PR)

- **hubspot-deep-dive-2026-03**:
  - L143: declare PeerSpot in the source list (PeerSpot is allowlist AND
    enriched/counted -- 10 enriched reviews -- so the quote at L152 is valid and
    stays; the prose just omitted the source).
  - L156-158: paraphrase the TrustRadius blockquote (excluded source) to prose --
    drop the source name, keep the reviewer's substance (lead gen / data tracking
    / campaigns / positive business impact; senior lead account manager, education).
  - L162-164: paraphrase the Trustpilot blockquote (excluded source) to prose
    ("not worth adopting unless you're prepared to spend a fortune").
- **switch-to-clickup-2026-03**:
  - L124: declare PeerSpot (its two PeerSpot quotes at L160/L187 are allowlist and
    stay once declared).
  - L137-140: paraphrase the TrustRadius blockquote to prose, folding the dangling
    "One reviewer notes:" lead-in into the sentence so no orphaned lead-in remains.
- **Park** (`ATLAS-HARDENING.md`): both posts' methodology DECLARES excluded
  sources (Capterra/Trustpilot) as corpus -- a corpus-composition question
  (is the analysis allowlist-restricted?) distinct from the quote fix; left
  untouched and parked.

### Files touched

- `plans/PR-Blog-Fix-Undeclared-Hubspot-Clickup.md`
- `atlas-churn-ui/src/content/blog/hubspot-deep-dive-2026-03.ts`
- `atlas-churn-ui/src/content/blog/switch-to-clickup-2026-03.ts`
- `ATLAS-HARDENING.md`

## Mechanism

Allowlist source (PeerSpot) -> declare it (quote stays). Excluded source -> the
quote can't be shown per the blog source allowlist, so it's paraphrased to prose
(substance kept, source name dropped, no blockquote since it's no longer
verbatim) -- the same recover->paraphrase precedent set on the orphaned-lead-in
slices. No invention: each paraphrase restates the existing quote's content.

## Intentional

- **Paraphrase, not drop.** Keeps the reviewer insight; only the policy-violating
  attribution/verbatim form is removed.
- **Methodology declarations left as-is, parked.** Removing Capterra/Trustpilot
  from the corpus source-list is a corpus-composition decision (the counts may
  genuinely include those reviews) -- out of scope for a quote fix.
- **No generator edit.** Data-correction for published posts; the quote pool's
  inclusion of excluded/non-counted reviews is the parked zoho L158 item.

## Deferred

- Excluded-source paraphrase for the other 13 posts (azure-vs-salesforce, clickup-04,
  insightly-vs-zoho-crm, jira-vs-trello, mailchimp, metabase-vs-tableau,
  microsoft-teams-vs-notion, real-cost-of-woocommerce, switch-to-clickup-04,
  switch-to-shopify, teamwork, top-complaint, woocommerce) -- sibling slices.
- Slashdot x9 drop -- separate slice.
- Methodology-declares-excluded-sources -- parked (ATLAS-HARDENING).

## Verification

- Widened detector `excluded_source_quote` added to the untracked auditor;
  `--self-test`: ALL PASS (incl. 3 new fixtures: excluded quoted-even-if-declared
  flagged, allowlist not flagged, prose mention not flagged).
- Both posts re-audited (`--slug=`): `excluded_source_quote` = 0,
  `undeclared_quoted_source` = 0, no orphaned-lead-in, NO findings of any kind.

## Estimated diff size

| Area | LOC |
|---|---:|
| hubspot-03 (declare + 2 paraphrases) | ~10 |
| switch-to-clickup-03 (declare + 1 paraphrase) | ~7 |
| ATLAS-HARDENING park entry | ~8 |
| Plan doc | ~94 |
| **Total** | **~119** |
