# Support-Ticket Generated Content Acceptance Matrix - 2026-05-28

## Scope

This validation re-checks the saved live support-ticket generated-content
artifacts after the support-ticket provider, generated-content evaluator,
descriptive blog, and compact small-upload blog slices landed.

The goal is to separate current accepted outputs from older known-bad regression
artifacts before the next product slice.

## Matrix

| Artifact | Output | Status | Evaluator result | Shape notes |
|---|---|---|---|---|
| `tmp/support_ticket_evidence_contract_live_validation_20260526/landing-page-draft.json` | landing page | Current accepted | Passed | 908 generated-copy words across hero/sections/meta/CTA; SEO/AEO and GEO were recorded ready in the saved export. |
| `tmp/support_ticket_blog_small_upload_live_validation_20260526_policy/blog-post-draft.json` | blog post | Current accepted | Passed | 1,111 generated-content words by this tokenizer, 4 H2 sections, 0 H3 sections; the prior validation recorded 1,095 words and ready SEO/AEO + GEO. |
| `tmp/support_ticket_live_blog_gate_20260525/blog-post-draft-cadence2.json` | blog post | Known-bad regression artifact | Failed as expected | 1,585 generated-content words, 7 H2 sections, 0 H3 sections. Fails on unsupported outcome claims and concrete answer steps without resolution evidence. |

The current accepted artifacts are the latest saved outputs that previous live
validation slices marked as passing after the prompt, quality-policy, and
generated-content evaluator fixes.

The known-bad artifact is intentionally included to prove the detector still
fires on the old failure mode. It is not a candidate output.

## Manual Audit

The current accepted landing page and compact blog pass the deterministic
truthfulness checks that matter for uploaded-ticket inputs:

- support-ticket source context is present
- support-ticket, FAQ, help-center, or answer framing is visible
- source counts are visible
- no invented calendar window or recurring cadence
- no unsupported percentage claims
- no guaranteed ticket-volume, churn, retention, capacity, or time-savings
  outcome claims
- no concrete answer/product steps without verified resolution evidence
- visible source signals include the ticket clusters and/or customer questions

The compact blog artifact is the final `policy` export from the small-upload
live validation. The similarly named non-policy export in
`tmp/support_ticket_blog_small_upload_live_validation_20260526/` is an earlier
intermediate run that passed truthfulness but missed the final compact shape.

## Result

Current state for this lane:

- The latest accepted support-ticket landing page is source-truthful by the
  generated-content evaluator.
- The latest accepted small-upload support-ticket blog is source-truthful by the
  generated-content evaluator and matches the compact section shape.
- The evaluator still catches the older unsupported-benefit and unsupported
  answer-step failure modes.

This is enough to consider the current support-ticket landing/blog generated
content path accepted for the packaged 4-row CSV scenario. It does not prove
arbitrary customer CSVs yet.

## Verification

- Command: python scripts/evaluate_support_ticket_generated_content.py --output landing_page tmp/support_ticket_evidence_contract_live_validation_20260526/landing-page-draft.json --pretty
  - Passed.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post tmp/support_ticket_blog_small_upload_live_validation_20260526_policy/blog-post-draft.json --pretty
  - Passed.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post tmp/support_ticket_live_blog_gate_20260525/blog-post-draft-cadence2.json --pretty
  - Failed as expected on `support_ticket_outcome_claims_grounded` and `support_ticket_answer_steps_grounded`.
- Shape summary command: Python JSON export scanner over the three artifacts.
  - Current landing: 908 generated-copy words.
  - Current compact blog: 1,111 generated-content words, 4 H2, 0 H3.
  - Known-bad blog: 1,585 generated-content words, 7 H2, 0 H3.
