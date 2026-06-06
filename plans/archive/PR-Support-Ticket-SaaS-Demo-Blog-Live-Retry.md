# Support-Ticket SaaS Demo Blog Live Retry

## Why this slice exists

The representative 36-row SaaS demo blog path is still not accepted. The prior
slice tightened the descriptive support-ticket prompt contract so Haiku sees
plain instructions to avoid unsupported prioritization, discoverability,
search, self-service, and post-publication outcome claims before the evaluator
has to catch them.

This slice reruns the live Haiku blog generation against the same SaaS demo CSV
and records whether the blog path is now accepted.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider

Slice phase: Functional validation

1. Run the 36-row support-ticket SaaS demo blog live smoke with Haiku.
2. Export the saved draft and generated result to `tmp/`.
3. Replay the exported draft through the support-ticket generated-content
   evaluator.
4. Manually scan the generated body and metadata for unsupported outcome,
   prioritization, discoverability, search, self-service, resolution-step, and
   timing claims.
5. Update the SaaS demo validation note with the live retry result.
6. Commit an accepted current blog fixture only if both automated and manual
   review pass.

### Files touched

- `docs/extraction/validation/support_ticket_saas_demo_generated_content_acceptance_2026-05-28.md` - live retry status and verification.
- `plans/PR-Support-Ticket-SaaS-Demo-Blog-Live-Retry.md` - this plan.

## Mechanism

The live smoke uses the repo SaaS demo CSV, the Atlas environment files, and the
existing Haiku override env file. It asks the real content-ops generation path
to create a blog post, save the draft, export it to `tmp/`, and run the
generated-content evaluator.

If the automated gates pass, the exported JSON is manually scanned for the same
claim families that caused the previous false greens. Acceptance requires both
passes. If manual review rejects the output, this slice records the failure
without committing a current accepted blog fixture.

## Intentional

- Haiku remains the live validation model for cost control.
- This slice does not change generator code. It validates whether the prompt
  contract change that just landed was enough.
- A fixture is only committed if the generated blog is accepted by both the
  evaluator and manual review.
- The first smoke invocation failed only when writing the `--output-result`
  file because the target `tmp/` directory did not exist. The rerun used the
  same source CSV and Haiku override after creating the export directory.

## Deferred

- This retry still drifted on a softer unsupported benefit claim. The next
  source fix should move from prompt tightening to a deterministic FAQ-shell or
  section-outline scaffold before generation.
- Parked hardening: none.

## Verification

- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528_descriptive_retry --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528_descriptive_retry/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528_descriptive_retry/blog-post-result.json --evaluate-generated-content --json
  - Failed after generation when writing `--output-result` because the target `tmp/` directory did not exist.
- Command: mkdir -p tmp/support_ticket_saas_demo_blog_acceptance_20260528_descriptive_retry
  - Passed.
- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528_descriptive_retry_2 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528_descriptive_retry/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528_descriptive_retry/blog-post-result.json --evaluate-generated-content --json
  - Saved draft `0efa47f7-c77b-4462-841e-990465fda1af`; generated-content evaluation passed; SEO/AEO ready; GEO ready; manual review rejected it on an unsupported future-customer recognition claim.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post tmp/support_ticket_saas_demo_blog_acceptance_20260528_descriptive_retry/blog-post-draft.json --pretty
  - Passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/support-ticket-saas-demo-blog-live-retry-pr-body.md
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Validation note | ~35 |
| Plan doc | ~85 |
| Total | ~120 |
