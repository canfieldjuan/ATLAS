# PR: Support-Ticket SaaS Demo Blog Accepted Fixture

## Why this slice exists

PR-Support-Ticket-SaaS-Demo-Generated-Content-Acceptance accepted the 36-row
SaaS demo landing-page path but left the blog path unaccepted because Haiku kept
turning observed support-ticket clusters into unsupported outcome claims.
PR-Support-Ticket-Blog-Descriptive-Contract then added the structural
`descriptive_no_outcome` contract that should steer no-outcome/no-resolution
support-ticket blogs before the evaluator has to catch them.

This slice tests that fix against the same 36-row SaaS demo CSV with a live
Haiku retry. If the generated blog passes the deterministic support-ticket
generated-content evaluator, this PR commits a minimized accepted blog fixture.
If it still fails, this PR records the failure and parks only the next source
gap.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Functional validation

1. Run a live Haiku `blog_post` generation through
   `scripts/smoke_content_ops_live_generation.py` using
   `extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv`.
2. Export the saved draft and smoke result to `tmp/`.
3. Re-run the deterministic evaluator against the exported draft.
4. Add the small contract forbidden-claim fix exposed by the first retry.
5. Add a short validation report documenting the command, evaluator result, and
   fixture shape.
6. Park the remaining source blocker if the blog path is still not accepted.

### Files touched

- `plans/PR-Support-Ticket-SaaS-Demo-Blog-Accepted-Fixture.md` - Plan doc for this validation slice.
- `docs/extraction/validation/support_ticket_saas_demo_blog_acceptance_2026-05-28.md` - Validation report for the post-contract live blog retry.
- `extracted_content_pipeline/blog_generation.py` - Tighten the descriptive contract's forbidden claims for the unsupported "help customers find answers" phrasing exposed by the first retry.
- `tests/test_extracted_blog_generation.py` - Focused assertion that the structural contract includes the forbidden customer-answer outcome phrasing.
- `HARDENING.md` - Park the remaining GEO entity clarity blocker exposed by the second retry.

## Mechanism

The live command uses the existing smoke harness, real support-ticket provider
packaging, real DB save/export, and the Haiku model routing env:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528 \
  --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528/blog-post-draft.json \
  --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528/blog-post-result.json \
  --evaluate-generated-content \
  --json
```

Attempt 1 failed before save on unsupported "help customers find answers"
outcome language. That phrasing is already blocked by the deterministic
evaluator, so this PR adds it to the structural descriptive contract's
`forbidden_claims` and pins it with a focused test. Attempt 2 then failed before
save on `geo_entity_clarity_missing`, so no accepted fixture is committed.

## Intentional

- This is a bounded Haiku acceptance retry, not another prompt-tuning loop.
- The only generation code change is the small structural contract mismatch
  exposed by attempt 1.
- This stays out of FAQ Markdown/article ownership.
- This does not resolve the cost telemetry schema mismatch; that remains parked.

## Deferred

- Future PR: inspect and fix the support-ticket blog
  `geo_entity_clarity_missing` failure before another live acceptance retry.
- Future PR: commit an accepted SaaS demo blog fixture after the GEO blocker is
  fixed and the live retry saves a passing draft.
- Future PR: add a scripted regression gate after the accepted fixture exists.
- Future PR: validate the same shape against a sanitized real customer export.
- Parked hardening:
  - Support-ticket SaaS demo blog still fails GEO entity clarity after
    descriptive contract.

## Verification

- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528/blog-post-result.json --evaluate-generated-content --json
  - Failed before save on unsupported "help customers find answers" outcome language.
- Command: python -m pytest tests/test_extracted_blog_generation.py tests/test_smoke_content_ops_live_generation.py -q
  - Passed, 102 tests.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - Passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Passed.
- Command: bash scripts/check_ascii_python.sh
  - Passed.
- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528_retry2 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528_retry2/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528_retry2/blog-post-result.json --evaluate-generated-content --json
  - Failed before save on `geo_entity_clarity_missing`.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/support-ticket-saas-demo-blog-accepted-fixture-pr-body.md
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~116 |
| Validation report | ~61 |
| Contract test/fix | ~5 |
| HARDENING note | ~9 |
| **Total** | **~191** |

This stays focused on validation evidence and avoids changing generator code in
the same slice unless the run proves a small source-level correctness issue.
