# Support-Ticket Evidence Contract Live Validation - 2026-05-26

## Scope

This validation reran live Content Ops support-ticket landing-page and blog-post
generation after the resolution/date/outcome evidence contracts landed in #977
and #979.

The goal was to check the actual customer-facing copy generated from uploaded
support tickets, not just unit-level readiness checks.

Environment:

- Repo: `/home/juan-canfield/Desktop/Atlas-support-ticket-provider`
- Env files:
  - `/home/juan-canfield/Desktop/Atlas/.env`
  - `/home/juan-canfield/Desktop/Atlas/.env.local`
  - `tmp/support_ticket_live_haiku_eval_20260525/haiku.env`
- Artifact directory:
  - `tmp/support_ticket_evidence_contract_live_validation_20260526`
- Model override:
  - Claude Haiku family through OpenRouter

Source input:

- `extracted_content_pipeline/examples/support_ticket_sources.csv`
- 4 uploaded support-ticket rows
- 2 direct customer questions
- 2 ticket clusters:
  - `email and profile updates` - 2 tickets
  - `reporting friction` - 2 tickets
- No dated window
- No measured outcomes
- No verified resolution evidence

## Commands

Landing page:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --account-id acct_support_ticket_evidence_contract_live_validation_20260526_landing \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_evidence_contract_live_validation_20260526/landing-page-draft.json \
  --output-result tmp/support_ticket_evidence_contract_live_validation_20260526/landing-page-result.json \
  --evaluate-generated-content \
  --json
```

Blog post, initial run:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_support_ticket_evidence_contract_live_validation_20260526_blog \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_evidence_contract_live_validation_20260526/blog-post-draft.json \
  --output-result tmp/support_ticket_evidence_contract_live_validation_20260526/blog-post-result.json \
  --evaluate-generated-content \
  --json
```

After each source fix, the blog smoke was rerun with account ids ending in
`blog_fixed`, `blog_fixed2`, `blog_fixed3`, `blog_fixed4`, and `blog_fixed5`.

Regression tests:

```bash
python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q
python -m pytest \
  tests/test_evaluate_support_ticket_generated_content.py \
  tests/test_extracted_support_ticket_input_package.py \
  tests/test_support_ticket_provider_landing_blog_execute.py \
  tests/test_extracted_content_ops_live_execute_harness.py \
  -q
```

## Landing Result

Landing generation passed.

- Saved draft id: `27390fd6-6994-4fa5-8162-7b6c51743de1`
- SEO/AEO readiness: ready
- GEO readiness: ready
- Generated-content evaluation: passed
- Manual audit: no unsupported outcome claims, no unsupported date/cadence
  claims, and no concrete answer steps without resolution evidence.

## Blog Findings

The initial blog draft saved and passed the previous deterministic evaluator,
but manual audit found false greens:

- unsupported outcome and support-volume claims, including fewer tickets,
  faster resolution, reduced repeat work, real time savings, and ticket-volume
  drops
- unsupported extrapolation that more customers would ask the same questions in
  the future
- unsupported product-step examples such as generic account-settings paths
- unsupported resolution claims, including support-team-confirmed capabilities
  when the uploaded tickets had no resolution evidence

The root cause was not just model behavior. The support-ticket package also fed
the model the default secondary keyword `reduce repeat support tickets`, which
pulled the blog toward unsupported outcome promises. That default was changed
to `repeat support questions`.

## Source Fixes

This slice added or tightened the following gates:

- `support_ticket_outcome_claims_grounded` now catches the live false-green
  variants around fewer tickets, faster resolution, reduced repeat work, real
  time savings, support workload, instant answers, future-customer assumptions,
  and ticket-volume drops.
- `support_ticket_answer_steps_grounded` now catches concrete answer steps,
  UI paths, product capability claims, support-team-confirmed resolutions, and
  likely-resolution guesses when `support_ticket_resolution_evidence_present`
  is false.
- The support-ticket blog prompt now explicitly keeps no-resolution answers as
  review-needed placeholders and avoids broad benefits sections when measured
  outcomes are absent.
- The support-ticket package now uses descriptive default secondary keywords
  instead of a promissory ticket-reduction keyword.

## Final Blog Result

The final live blog run did not save a draft. That is the correct safety result
for this slice: the stricter generated-content gate blocked a candidate whose
copy still included unsupported outcome language.

Final run:

- Account id: `acct_support_ticket_evidence_contract_live_validation_20260526_blog_fixed5`
- Saved draft ids: none
- Result: blocked before save
- Blockers included:
  - `support_ticket_generated_content`
  - unsupported outcome language copied into the draft candidate
  - GEO structure blockers on the failed candidate

This means the live path no longer silently saves the misleading support-ticket
blog output found during manual audit.

## Remaining Work

The next product slice should restructure the support-ticket blog source task
so Haiku can produce a passing long-form draft without relying on copied
guardrail language or unsupported benefit claims.

Recommended direction:

- make no-outcome support-ticket blogs explicitly descriptive/diagnostic
- avoid broad benefits sections when measured outcomes are absent
- use section goals around observed clusters, customer wording, review workflow,
  publication checklist, and what to measure next
- keep answer bodies as review-needed placeholders unless resolution evidence is
  present

## Verification

- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q`
  - `40 passed`
- Focused provider/evaluator sweep:
  - `tests/test_evaluate_support_ticket_generated_content.py`
  - `tests/test_extracted_support_ticket_input_package.py`
  - `tests/test_support_ticket_provider_landing_blog_execute.py`
  - `tests/test_extracted_content_ops_live_execute_harness.py`
  - `81 passed`

The final local PR review should rerun the focused sweep and full mechanical
bundle before this PR opens.
