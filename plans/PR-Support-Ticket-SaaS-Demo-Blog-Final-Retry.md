# Support-Ticket SaaS Demo Blog Final Retry

## Why this slice exists

The 36-row SaaS demo blog path has not been accepted yet. The latest live run
saved a draft, but manual review found unsupported benefit claims that the
evaluator missed. `PR-Support-Ticket-Blog-False-Green-Outcome-Claims` tightened
the prompt contract and evaluator so those claims now fail.

This slice reruns the representative SaaS demo blog generation with Haiku after
that stricter contract lands. If the generated draft passes the evaluator and a
manual truthfulness scan, this slice records it as the current accepted fixture.
If it fails, this slice records the failure and leaves the blog path not
accepted.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider

Slice phase: Functional validation

1. Run the 36-row SaaS demo support-ticket blog live smoke with Haiku.
2. Evaluate the saved draft with the support-ticket generated-content evaluator.
3. Manually scan the generated content for unsupported outcome, causality,
   resolution-step, and search-performance claims.
4. Update the SaaS demo validation note with the final retry result.
5. Commit the accepted current blog fixture only if the draft passes both
   automated and manual review.

### Files touched

- `docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_blog_post.json` - accepted blog draft export from the final live retry.
- `docs/extraction/validation/support_ticket_saas_demo_generated_content_acceptance_2026-05-28.md` - final retry status and verification.
- `plans/PR-Support-Ticket-SaaS-Demo-Blog-Final-Retry.md` - this plan.

## Mechanism

The live smoke command uses the repo's support-ticket SaaS demo CSV, the local
Atlas environment files, and the Haiku model override. The generated draft is
exported to tmp, then replayed through
`scripts/evaluate_support_ticket_generated_content.py`.

The acceptance note is updated based on the result. Because the final retry
passed the evaluator, SEO/AEO readiness, GEO readiness, and manual truthfulness
scan, this PR commits the accepted current blog fixture.

## Intentional

- Haiku is used for this validation run because it is the agreed test model for
  live generation cost control.
- This slice does not loosen the evaluator to accept generated text. Acceptance
  requires the generated content to fit the stricter contract.
- The prior known-bad fixture remains in place even if this run succeeds because
  it documents a useful negative case.

## Deferred

- If this retry fails, the next slice should address the newly observed blocker
  at the prompt, evaluator, or readiness-contract source before another live
  retry.
- If this retry succeeds, a follow-up can promote the representative SaaS demo
  blog fixture into a broader scripted acceptance matrix.
- Parked hardening: none.

## Verification

- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528_final_retry --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528_final_retry/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528_final_retry/blog-post-result.json --evaluate-generated-content --json
  - Passed. Saved draft `4ff4f0fe-274e-46ad-9af2-e47475dd749b`; generated-content evaluation passed; SEO/AEO readiness ready; GEO readiness ready.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post tmp/support_ticket_saas_demo_blog_acceptance_20260528_final_retry/blog-post-draft.json --pretty
  - Passed.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_blog_post.json --pretty
  - Passed.
- Command: python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q
  - Passed, 47 tests.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/support-ticket-saas-demo-blog-final-retry-pr-body.md
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Accepted fixture | ~263 |
| Validation note | ~44 |
| Plan doc | ~85 |
| Total | ~392 |
