# PR-Support-Ticket-Blog-Blueprint-Package-Context

## Why this slice exists

The support-ticket input package is now the source of truth for the uploaded
ticket rows that feed Content Ops generation. The live-generation smoke helper
still builds the support-ticket blog blueprint by re-parsing raw CSV rows with
local helper functions for counts, date-window wording, and clusters.

That duplication can drift from the ingestion layer. For functional validation,
the support-ticket CSV should become one package, then both the execute payload
and the seeded blog blueprint should read from that package's normalized inputs.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Functional validation

1. Change the live-generation smoke helper's support-ticket blog blueprint
   payload to derive source context from `build_support_ticket_input_package`.
2. Remove the duplicated raw-row parsing helpers that only existed for the
   support-ticket blog smoke blueprint.
3. Update smoke-helper tests so package-derived source counts, clusters, date
   window wording, and FAQ-entry estimates are covered.

### Files touched

- `plans/PR-Support-Ticket-Blog-Blueprint-Package-Context.md`
- `scripts/smoke_content_ops_live_generation.py`
- `tests/test_smoke_content_ops_live_generation.py`

## Mechanism

`_support_ticket_blog_blueprint_payload` will call
`build_support_ticket_input_package(rows)` and read the normalized package
inputs:

- `source_row_count`
- `included_ticket_row_count`
- `question_like_ticket_count`
- `top_ticket_clusters`
- `source_period`
- `faq_questions`

The seeded blog blueprint's `data_context` and section stats then reflect the
same normalized ticket set that the support-ticket provider sends to execution.

## Intentional

- No FAQ generator behavior changes.
- No DB, LLM, or hosted route changes.
- No scale/backpressure work; the root `HARDENING.md` FAQSCALE-1 item belongs
  to the FAQ scale session and remains parked.
- This only removes duplicate validation-helper parsing; the package remains
  the ingestion-layer source of truth.

## Deferred

- Future PR: run another live DB/LLM validation if we need a fresh saved-draft
  artifact after this helper alignment.
- Parked hardening: none.

## Verification

- `python -m pytest tests/test_smoke_content_ops_live_generation.py -q`
  - passed, 24 tests.
- `python -m pytest tests/test_smoke_content_ops_live_generation.py tests/test_extracted_support_ticket_input_package.py tests/test_support_ticket_provider_landing_blog_execute.py -q`
  - passed, 52 tests.
- `python -m pytest tests/test_smoke_content_ops_support_ticket_package.py tests/test_extracted_support_ticket_input_provider.py -q`
  - passed, 16 tests.
- Python compile check for `scripts/smoke_content_ops_live_generation.py`
  passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~85 |
| Smoke helper | ~120 |
| Tests | ~35 |
| **Total** | **~240** |
