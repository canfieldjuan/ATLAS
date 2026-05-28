# PR: Support-Ticket SaaS Demo Blog Fixture Promotion

## Why this slice exists

The live 36-row SaaS demo support-ticket blog path is now accepted after the
observed-shell contract: live Haiku generation saved a draft, generated-content
evaluation passed, SEO/AEO and GEO readiness were ready, and persisted usage
telemetry matched the draft metadata.

The existing SaaS demo acceptance doc still records the blog path as not
accepted and only commits the known-bad blog fixture. This slice updates that
acceptance state and pins it with a minimized accepted fixture so later prompt,
evaluator, or support-ticket-package changes cannot silently regress the
representative SaaS demo blog path.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Functional validation

1. Add a minimized accepted blog-post fixture derived from the latest 36-row
   SaaS demo live export.
2. Update the SaaS demo acceptance doc to mark the blog path accepted for the
   current observed-shell contract.
3. Add focused fixture tests proving the accepted SaaS demo blog passes and the
   existing known-bad SaaS demo blog still fails.
4. Leave new live generation, prompt changes, and broader customer-CSV coverage
   out of scope.

### Files touched

- `plans/PR-Support-Ticket-SaaS-Demo-Blog-Fixture-Promotion.md` - plan doc for this fixture promotion.
- `docs/extraction/validation/support_ticket_saas_demo_generated_content_acceptance_2026-05-28.md` - update the SaaS demo blog acceptance state and verification notes.
- `docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_blog_post.json` - minimized accepted blog fixture.
- `tests/test_support_ticket_saas_demo_generated_content_fixtures.py` - regression tests for the accepted and known-bad SaaS demo blog fixtures.
- `scripts/run_extracted_pipeline_checks.sh` - enroll the new fixture test in the extracted pipeline runner.

## Mechanism

The fixture is derived from the local live-smoke draft export produced by the
observed-shell validation run and keeps only evaluator-relevant fields:
generated copy, source context, readiness summaries, SEO/FAQ metadata, and
visible status/title fields. It drops run-specific ids, account scope, provider
request ids, and token/cost telemetry.

The focused test loads the committed fixture JSON and calls
`evaluate_support_ticket_generated_content(..., output="blog_post")`. The
accepted fixture must pass. The known-bad fixture must fail on the existing
unsupported support-ticket generated-content checks.

## Intentional

- This is fixture promotion, not another live LLM run.
- This does not commit the raw live artifact. The committed fixture is minimized
  and reproducible through the validation doc commands.
- This keeps the existing known-bad fixture because it proves the evaluator still
  catches the prior false-green class.
- This does not touch FAQ Article output or customer-language keyword promotion.

## Deferred

- Future PR: broader accepted fixture set across more customer CSV shapes.
- Future PR: product UI for viewing the accepted generated draft and per-run
  telemetry together.
- Future PR: deterministic renderer if free-form blog generation regresses on
  no-outcome support-ticket claims.
- Parked hardening: none.

## Verification

- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_blog_post.json --pretty
  - Passed.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/known_bad_saas_demo_blog_post.json --pretty
  - Failed as expected on `support_ticket_outcome_claims_grounded`.
- Command: python -m pytest tests/test_support_ticket_saas_demo_generated_content_fixtures.py -q
  - Passed, 2 tests.
- Command: python -m pytest tests/test_support_ticket_saas_demo_generated_content_fixtures.py tests/test_evaluate_support_ticket_generated_content.py -q
  - Passed, 51 tests.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py
  - Passed; 124 matching tests enrolled.
- Command: python -m pytest tests/test_support_ticket_saas_demo_generated_content_fixtures.py tests/test_evaluate_support_ticket_generated_content.py tests/test_audit_extracted_pipeline_ci_enrollment.py -q
  - Passed, 60 tests.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file <PR body file>
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~90 |
| Acceptance doc update | ~35 |
| Accepted fixture | ~370 |
| Fixture tests | ~50 |
| CI runner enrollment | ~1 |
| **Total** | **~550** |

This lands above the 400 LOC soft cap because the fixture needs enough generated
copy, source context, metadata, and readiness summaries to exercise the
evaluator honestly.
