# PR-Blog-Paraphrase-Excluded-B

Ownership lane: `content-ops/blog-paraphrase-excluded-b`

## Why this slice exists

Third slice of the excluded-source-quote fix (after #834 and #837, batch A).
Paraphrases the remaining 6 posts that quote allowlist-excluded sources
(Capterra/Trustpilot/TrustRadius/Software Advice): drop the source name + verbatim
form, keep the reviewer's substance. After this, the only remaining undeclared/
excluded class is the Slashdot x9 DROP (separate slice).

## Scope (this PR)

- **real-cost-of-woocommerce-2026-04** (L123): paraphrase the Capterra "eCommerce
  for WordPress hands-down" inline quote.
- **switch-to-clickup-2026-04** (L144-148): drop the TrustRadius blockquote + its
  "on TrustRadius" lead-in, fold into one prose sentence (keep the tailor-workflows
  detail).
- **switch-to-shopify-2026-03** (L129-132, L138-141): paraphrase two TrustRadius/
  Trustpilot blockquotes to prose, and update the follow-on referents ("This quote"
  -> "This sentiment"; "This positive quote" -> "This positive note") so they
  still read correctly.
- **teamwork-deep-dive-2026-04** (L179): paraphrase three excluded quotes in one
  line (Capterra + Trustpilot + Capterra); keep the $17/Pro/"no reporting" figures.
- **top-complaint-every-e-commerce-2026-04** (L77, FAQ answer): drop "verified
  reviewer on Software Advice" from the reported-speech line (keep the "3x rate"
  and 54-review-sample facts).
- **woocommerce-deep-dive-2026-04** (L185): paraphrase the Capterra
  "eCommerce for WordPress hands-down" inline quote.

### Files touched

- `plans/PR-Blog-Paraphrase-Excluded-B.md`
- `atlas-churn-ui/src/content/blog/real-cost-of-woocommerce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-clickup-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-shopify-2026-03.ts`
- `atlas-churn-ui/src/content/blog/teamwork-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-e-commerce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/woocommerce-deep-dive-2026-04.ts`

## Mechanism

Excluded source -> rewrite the quote as reported speech (substance kept, source
name + quotation marks dropped, blockquote converted to prose since it's no longer
verbatim). No invention -- each paraphrase restates the existing quote. Where the
post later refers back to a now-paraphrased quote ("This quote ..."), the referent
is updated so the prose stays coherent.

## Intentional

- **Paraphrase, not drop.** Keeps the reviewer insight; only the policy-violating
  attribution/verbatim form is removed.
- **Referents updated.** switch-to-shopify's "This quote"/"This positive quote"
  follow-ons are changed to "sentiment"/"note" so they don't dangle.
- **Figures preserved.** teamwork's $17/Pro/reporting and top-complaint's 3x-rate
  / 54-review facts are kept (real data); only the source attribution drops.

## Deferred

- Slashdot x9 DROP -- separate slice (next).
- Methodology source-list mentions of excluded sources -- parked
  (ATLAS-HARDENING, via #834).

## Review follow-up (#837 widened detector)

After the #837 review widened `excluded_source_quote` to catch the adjective form
(`<Source> reviewer`, not just `reviewer on <Source>`), a residual surfaced in
**teamwork** L301: a second paragraph with two verbatim Trustpilot quotes (Pro
$17 / Premium $33) attributed to "a Trustpilot reviewer" -- the original batch
fixed only L179. Paraphrased L301 to reported speech (kept the $17/$33 figures,
dropped the quotes + the "Trustpilot reviewer ... consumer review platform"
attribution). The other 5 posts were already clean under the widened detector.

## Verification

- All 6 posts re-audited under the WIDENED detector (`--slug=`):
  `excluded_source_quote` = 0. (teamwork still shows `pain_as_strength` -- the DD1
  finding fixed by the open #846, off origin/main; not this slice.)

## Estimated diff size

| Area | LOC |
|---|---:|
| 6 posts (paraphrases + 2 blockquote conversions + referent fixes) | ~26 |
| review follow-up (teamwork L301) | ~2 |
| Plan doc | ~85 |
| **Total** | **~113** |
