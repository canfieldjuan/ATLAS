# Support-Ticket Blog Descriptive Prompt Contract

## Why this slice exists

The representative SaaS demo blog still fails manual truthfulness review after
the evaluator catches the latest false-green claims. The source issue is that
the generator receives a structured descriptive contract, but the user prompt
still asks for a normal persuasive blog. Haiku keeps filling that genre with
speculative activation, prioritization, discoverability, search, and
post-publication outcome claims.

This slice makes the descriptive support-ticket contract explicit in the prompt
surface before the next live retry.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider

Slice phase: Functional validation

1. Add a support-ticket descriptive prompt addendum for blogs generated from
   questions-only ticket evidence.
2. Tell the generator to use only observed counts, clusters, customer wording,
   and review-needed FAQ shells; neutralize tied-cluster ranking; and keep
   post-publication measurement language non-promissory.
3. Pin that the addendum appears in both first-pass and repair prompts.

### Files touched

- `extracted_content_pipeline/blog_generation.py` - descriptive support-ticket prompt addendum.
- `plans/PR-Support-Ticket-Blog-Descriptive-Prompt-Contract.md` - this plan.
- `tests/test_extracted_blog_generation.py` - prompt contract coverage.

## Mechanism

`_blog_generation_prompts` already receives the enriched blueprint, including
`data_context.support_ticket_blog_mode`. This PR appends a small, explicit user
prompt addendum when that mode is `descriptive_no_outcome`.

The addendum repeats the contract in plain instructions the model sees right
before generation: describe what the uploaded tickets show, avoid business
impact ranking unless the input has outcome evidence, treat tied clusters as
tied, use draft-answer placeholders when resolution evidence is absent, and do
not claim FAQ entries will be discoverable, rank, reduce tickets, or prove they
are working.

Because repair prompts reuse the original base user prompt, the same addendum
also carries into quality repair attempts.

## Intentional

- This PR does not run another live generation. It first fixes the prompt
  contract source that caused the last live retries to drift.
- The evaluator backstop remains unchanged here; the previous slice already
  proved it catches the observed bad drafts.
- The addendum is only applied for support-ticket blogs without measured
  outcomes or resolution evidence.

## Deferred

- Run the 36-row SaaS demo Haiku blog live retry after this prompt contract
  lands.
- If Haiku still drifts after this, the next source fix should narrow the
  section outline or add a deterministic FAQ-shell scaffold before generation.
- Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_extracted_blog_generation.py::test_generate_puts_support_ticket_descriptive_contract_in_prompt tests/test_extracted_blog_generation.py::test_quality_repair_prompt_keeps_support_ticket_descriptive_contract -q
  - Passed, 2 tests.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/support-ticket-blog-descriptive-prompt-contract-pr-body.md
  - Pending.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Prompt addendum | ~45 |
| Tests | ~25 |
| Plan doc | ~70 |
| Total | ~140 |
