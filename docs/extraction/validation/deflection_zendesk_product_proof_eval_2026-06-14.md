# Deflection Zendesk Product-Proof Eval

Date: 2026-06-14

Issues: #1419, #1440

## What Ran

This validation feeds the committed sanitized Zendesk product-proof corpus
through the real full-thread importer, support-ticket input package, FAQ
deflection report builder, and product-proof evaluator after the question-label
cleanup in #1568.

## Result

| Metric | Value |
|---|---:|
| Tickets evaluated | 50 |
| Expected publishable-answer tickets | 36 |
| Covered publishable-answer tickets | 25 |
| Generated ranked questions | 9 |
| Publishable-answer items | 6 |
| Publishable false-positive sources | 0 |
| Unresolved sources published | 0 |
| Reopened sources published | 0 |
| Private note leaks | 0 |
| Degraded published question labels | 0 |
| Degraded draft question labels recorded | 3 |
| Failed artifact output checks | 0 |
| FAQ warnings | 2 |

Status: `ok`

The run is safety-clean for published answers. Remaining draft-only weak labels
are recorded in the summary JSON, not hidden, so follow-up slices can decide
whether they are launch-blocking polish or expected diagnostics.

FAQ warning codes: `non_repeat_tickets_excluded, duplicate_source_policy_questions`

## Artifact Links

- Summary JSON:
  `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_eval_20260614/summary.json`
- Report excerpt:
  `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_eval_20260614/report_excerpt.md`

## Boundary

This is an offline product-shaped validation run. It does not call live
Zendesk, mutate Stripe state, unlock a paid report, send email, or exercise the
hosted portfolio result page. It pairs with the CFPB full-volume proof: CFPB
proves stress and this corpus proves Zendesk ticket-shape quality.
