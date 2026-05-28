# Support-Ticket SaaS Demo Blog Final Retry

## Why this slice exists

The 36-row SaaS demo blog path has not been accepted yet. The latest live run
saved a draft, but manual review found unsupported benefit claims that the
evaluator missed. `PR-Support-Ticket-Blog-False-Green-Outcome-Claims` tightened
the prompt contract and evaluator so those claims now fail.

This slice reruns the representative SaaS demo blog generation with Haiku after
that stricter contract lands. Review found one more source issue: the support
ticket package collapsed three tied 4-ticket categories into `remaining`, which
made the generated draft incorrectly describe them as lower-frequency. Fixing
that rollup exposed another evaluator false-green: the cluster-correct blog
still invented activation, search, discoverability, self-service, and
post-publication outcome claims. This PR fixes the source rollup, tightens the
evaluator/prompt contract for the new false-green shapes, and records that the
blog is still not accepted.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider

Slice phase: Functional validation

1. Fix the support-ticket package cluster rollup so the 36-row SaaS demo CSV
   exposes all nine 4-ticket clusters instead of collapsing three tied clusters
   into a misleading `remaining` bucket.
2. Tighten the support-ticket blog evaluator and prompt contract for the
   activation, discoverability, search, self-service, and post-publication
   outcome claims observed in the cluster-correct retries.
3. Include generated blog metadata in the evaluator text surface so user-facing
   SEO/FAQ metadata cannot carry unsupported claims past the gate.
4. Update the SaaS demo validation note to reject the bad accepted fixture and
   record the latest cluster-correct retry results.
5. Leave no accepted blog fixture committed for this validation date.

### Files touched

- `docs/extraction/validation/support_ticket_saas_demo_generated_content_acceptance_2026-05-28.md` - final retry status and verification.
- `extracted_content_pipeline/blog_generation.py` - support-ticket descriptive prompt forbidden-claim contract.
- `extracted_content_pipeline/support_ticket_generated_content_eval.py` - support-ticket generated-content evaluator coverage.
- `extracted_content_pipeline/support_ticket_input_package.py` - support-ticket cluster rollup limit.
- `plans/PR-Support-Ticket-SaaS-Demo-Blog-Final-Retry.md` - this plan.
- `tests/test_evaluate_support_ticket_generated_content.py` - false-green evaluator coverage.
- `tests/test_smoke_content_ops_support_ticket_package.py` - cluster rollup coverage.

## Mechanism

The package now exposes up to 12 named support-ticket clusters before adding a
`remaining` bucket. That keeps the representative 36-row SaaS demo CSV truthful:
its nine pain categories all have four tickets, so none should be described as
lower-frequency or one-off.

The evaluator now treats blog `metadata` as generated text alongside title,
description, content, tags, and charts. That closes the gap where the article
body could be checked while SEO/FAQ metadata still contained unsupported
prioritization or discoverability claims.

The outcome detector adds focused patterns for the newly observed claims:
activation/workflow blocking, friction/delay reduction, internal-search
appearance, discoverability/ranking, self-service support avoidance, and
post-publication "entry is working" assertions. The descriptive blog prompt
names the same forbidden claim families so generation and validation stay
aligned.

The validation note keeps the accepted landing page and known-bad blog fixture,
does not commit a current accepted blog fixture, and records that the latest
cluster-correct blog retries still failed manual review.

## Intentional

- Haiku is used for this validation run because it is the agreed test model for
  live generation cost control.
- This slice does not loosen the evaluator to accept generated text. Acceptance
  requires the generated content to fit the stricter contract.
- The invalid accepted fixture is deleted instead of replaced. The latest
  cluster-correct blogs still fail review, so there is no accepted blog artifact
  to commit.
- The prior known-bad fixture remains in place because it documents a useful
  negative case.

## Deferred

- The next slice should tighten the descriptive support-ticket blog generation
  contract before another live retry. The current backstop now catches the
  observed false-green claims, but the generator is still being pulled toward
  speculative prioritization and post-publication outcome language.
- A follow-up can promote the representative SaaS demo blog fixture into a
  broader scripted acceptance matrix once a manually accepted blog exists.
- Parked hardening: none.

## Verification

- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528_final_retry --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528_final_retry/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528_final_retry/blog-post-result.json --evaluate-generated-content --json
  - Saved draft `4ff4f0fe-274e-46ad-9af2-e47475dd749b`; generated-content evaluation passed; SEO/AEO readiness ready; GEO readiness ready; manual review rejected it because the package collapsed three tied clusters into `remaining`.
- Command: python scripts/smoke_content_ops_support_ticket_package.py extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --pretty
  - Passed; after the cluster rollup fix, output includes all nine 4-ticket clusters and no `remaining` bucket.
- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528_cluster_fix --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528_cluster_fix/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528_cluster_fix/blog-post-result.json --evaluate-generated-content --json
  - Saved draft `7b587228-7cf6-4bfd-b33e-1689810153e6`; generated-content evaluation passed; SEO/AEO ready; GEO ready; manual review rejected it on unsupported impact, discoverability, and self-service claims.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post tmp/support_ticket_saas_demo_blog_acceptance_20260528_cluster_fix/blog-post-draft.json --pretty
  - Failed as expected after the evaluator was tightened for the first cluster-correct false-green claims.
- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528_cluster_fix_2 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528_cluster_fix_2/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528_cluster_fix_2/blog-post-result.json --evaluate-generated-content --json
  - Saved draft `8adb720a-936b-462e-8659-daef8dbe5f4b`; generated-content evaluation passed; SEO/AEO ready; GEO ready; manual review rejected it on unsupported impact and discoverability claims.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post tmp/support_ticket_saas_demo_blog_acceptance_20260528_cluster_fix_2/blog-post-draft.json --pretty
  - Failed as expected after the evaluator was tightened for the second cluster-correct false-green claims and metadata surface.
- Command: python -m pytest tests/test_evaluate_support_ticket_generated_content.py tests/test_smoke_content_ops_support_ticket_package.py -q
  - Passed, 57 tests.
- Command: python -m pytest tests/test_smoke_content_ops_support_ticket_package.py -q
  - Passed, 8 tests.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/support-ticket-saas-demo-blog-final-retry-pr-body.md
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Validation note | ~83 |
| Cluster rollup fix + tests | ~30 |
| Evaluator and prompt contract | ~235 |
| Plan doc | ~136 |
| Total | ~484 |

Over the 400-LOC soft cap because the same PR needs to record the rejection,
fix the source rollup, and pin the evaluator false-green that made the fixture
unsafe.
