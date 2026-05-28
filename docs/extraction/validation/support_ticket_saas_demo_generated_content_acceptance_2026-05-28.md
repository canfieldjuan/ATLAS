# Support-Ticket SaaS Demo Generated Content Acceptance - 2026-05-28

## Scope

This validation runs the support-ticket landing/blog generation path against the
repo's 36-row SaaS demo CSV:

`extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv`

The CSV has 36 synthetic support-ticket rows, 35 direct customer questions, and
7 observed clusters. It is broader than the packaged 4-row support-ticket CSV
used by the current acceptance matrix.

## Result

| Output | Status | Artifact | Notes |
|---|---|---|---|
| Landing page | Accepted | `docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_landing_page.json` | Live generation saved a draft, export context matched the provider package, SEO/AEO and GEO readiness were ready, and the support-ticket generated-content evaluator passed. |
| Blog post | Not accepted | `docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/known_bad_saas_demo_blog_post.json` | The broader SaaS demo blog repeatedly produced unsupported benefit/outcome language. This slice tightened the prompt and evaluator so the final observed bad artifact now fails instead of false-passing. |

The landing path is accepted for this representative CSV shape. The blog path is
not accepted for the 36-row SaaS demo shape yet.

## What Changed During Validation

The first blog run blocked on a cadence false positive: the generated article
quoted a source question that contained "weekly", and the evaluator treated the
word as an invented recurring cadence. The fix was to allow cadence language
only when the sentence includes exact cadence-bearing source wording from the
support-ticket context.

Subsequent runs exposed unsupported outcome claims in several forms:

- a support ticket could have been prevented by an FAQ entry
- a FAQ could address incoming ticket volume
- support ticket volume or repeat-question volume would be reduced
- questions would stop appearing
- the team would handle future instances more efficiently
- customers would find answers faster or resolve without follow-up
- fixed 2-4 week measurement windows or resolution-time improvements were implied

Those forms now have focused negative fixtures in
`tests/test_evaluate_support_ticket_generated_content.py`, and the blog prompt
now names those forbidden claims explicitly.

## Acceptance Notes

The landing page stayed within the current support-ticket evidence contract:

- used uploaded-ticket framing instead of a dated window
- surfaced 36 source rows, 36 included rows, and 35 question-like rows
- mentioned observed ticket clusters and customer wording
- avoided unsupported outcome, cadence, and resolution-step claims

The blog generator still needs a deeper contract change. Pattern blockers now
catch the observed bad artifacts, but the broader run shows Haiku keeps trying
to turn observed support-ticket clusters into promised support outcomes. The
next slice should make the no-outcome descriptive support-ticket blog mode
structural, not only prompt-enforced.

## Verification

- Command: python scripts/smoke_content_ops_live_generation.py --output landing_page --account-id acct_support_ticket_saas_demo_acceptance_20260528_landing --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_generated_content_acceptance_20260528/landing-page-draft.json --output-result tmp/support_ticket_saas_demo_generated_content_acceptance_20260528/landing-page-result.json --evaluate-generated-content --json
  - Passed.
- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post ... --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --evaluate-generated-content --json
  - Multiple runs saved drafts after repair, but manual audit found unsupported outcome language. The final observed bad artifact is committed as a known-bad fixture and now fails the evaluator.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output landing_page docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_landing_page.json --pretty
  - Passed.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/known_bad_saas_demo_blog_post.json --pretty
  - Failed as expected on `support_ticket_outcome_claims_grounded`.
- Command: python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q
  - Passed, 45 tests.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - Passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Passed.
- Command: bash scripts/check_ascii_python.sh
  - Passed.
- Command: bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline
  - Passed.
