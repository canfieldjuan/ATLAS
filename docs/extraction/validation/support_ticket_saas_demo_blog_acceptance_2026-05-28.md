# Support-Ticket SaaS Demo Blog Acceptance Retry - 2026-05-28

## Scope

This validation retried the 36-row SaaS demo support-ticket blog path after the
structural `descriptive_no_outcome` contract landed.

Source CSV:

`extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv`

The run used Haiku test routing and the existing live generation smoke harness.

## Result

Status: not accepted.

No blog draft was saved, so there is no accepted fixture in this slice.

| Attempt | Status | Blocking result |
|---|---|---|
| 1 | Failed before save | `support_ticket_generated_content` blocked the unsupported outcome sentence: "The next step is to turn those clusters into published, verified, and measurable FAQ entries that help customers find answers." |
| 2 | Failed before save | `geo_entity_clarity_missing` after quality repair budget was exhausted. |

## What Changed During Validation

Attempt 1 exposed a small mismatch between the new descriptive contract and the
existing deterministic evaluator. The contract already forbade ticket reduction,
deflection, retention, and faster-resolution claims, but it did not explicitly
forbid the broader unsupported outcome phrasing "help customers find answers."
The evaluator blocks that phrasing for no-outcome support-ticket data, so this
slice added that wording to the structural `forbidden_claims` contract and
covered it with a focused test.

Attempt 2 got past the support-ticket outcome detector but failed the blog GEO
entity clarity gate. The generated title contained "Support Ticket FAQ Gaps,"
but the quality gate still returned `geo_entity_clarity_missing` after two
repair attempts. That is now the next source blocker before this SaaS demo blog
path can be accepted.

## Acceptance Notes

The descriptive contract reduced the failure from unsupported outcome claims to
a general blog quality blocker, which is progress, but the 36-row SaaS demo blog
path is still not accepted. The next slice should inspect the support-ticket
blog prompt/quality gate interaction for `geo_entity_clarity_missing` and fix
that source issue before another live acceptance retry.

The repeated `_store_local failed for span=content_ops.llm.complete: column
"account_id" of relation "llm_usage" does not exist` warning also appeared
during both live attempts. That is already parked in `HARDENING.md` as a cost
telemetry schema issue and did not block generation.

## Verification

- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528/blog-post-result.json --evaluate-generated-content --json
  - Failed before save on unsupported "help customers find answers" outcome language.
- Command: python -m pytest tests/test_extracted_blog_generation.py tests/test_smoke_content_ops_live_generation.py -q
  - Passed, 102 tests, after adding the contract forbidden-claim fixture.
- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528_retry2 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528_retry2/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528_retry2/blog-post-result.json --evaluate-generated-content --json
  - Failed before save on `geo_entity_clarity_missing`.
