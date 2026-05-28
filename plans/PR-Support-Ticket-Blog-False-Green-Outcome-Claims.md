# Support-Ticket Blog False Green Outcome Claims

## Why this slice exists

The live SaaS demo blog retry after the citable repair contract saved a draft
and passed the current support-ticket generated-content evaluator, but a manual
read still found unsupported benefit claims. The draft said FAQ entries reduce
the number of times the team must answer a question, customers can find answers
themselves, and declining repeat tickets prove the FAQ entry is working.

That is a false green. The support-ticket blog contract allows observed ticket
clusters, customer wording, review-needed FAQ shells, verification work, and
metrics to watch. It does not allow unsupported outcomes or causality claims
from questions-only ticket data.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider

Slice phase: Functional validation

1. Add evaluator coverage for the newly observed false-green SaaS demo claims.
2. Tighten the unsupported-outcome detector so those claims fail before a draft
   can be accepted.
3. Tighten the support-ticket descriptive blog prompt so repair attempts avoid
   the same unsupported benefit phrasing.

### Files touched

- `extracted_content_pipeline/blog_generation.py` - prompt contract wording for support-ticket descriptive blogs.
- `extracted_content_pipeline/support_ticket_generated_content_eval.py` - unsupported outcome claim patterns.
- `tests/test_evaluate_support_ticket_generated_content.py` - regression coverage for the false-green claims.
- `plans/PR-Support-Ticket-Blog-False-Green-Outcome-Claims.md` - this plan.

## Mechanism

The evaluator already sentence-splits generated output and tests each sentence
against unsupported support-ticket outcome patterns. This slice adds patterns for
the specific uncovered claim shapes:

- FAQ entries reducing how many times the team answers a question.
- Customers finding answers themselves from the FAQ.
- Repeat-ticket decline or fewer tickets being treated as proof that the FAQ is
  being found, used, or working.
- Publishing FAQ entries reducing support load over time.
- Customer-wording FAQ entries performing better in search without evidence.

The blog prompt's descriptive support-ticket contract gets matching forbidden
language so the generator is told not to produce those claims in the first place.

## Intentional

- This does not commit the latest live draft as accepted. The draft exposed the
  false green and should not become the current passing fixture.
- This keeps the detector as a regression backstop, not the only control. The
  prompt contract is updated in the same slice so generation and validation move
  together.
- The live retry itself is deferred until this stricter contract lands.

## Deferred

- A follow-up PR should run the SaaS demo blog live smoke again with Haiku after
  this stricter contract lands, then record an accepted fixture only if the
  generated content passes both evaluator and manual truthfulness review.
- Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q
  - Passed, 47 tests.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post tmp/support_ticket_saas_demo_blog_acceptance_20260528_after_repair_contract/blog-post-draft.json --pretty
  - Failed as expected on `support_ticket_outcome_claims_grounded`, including the newly caught unsupported claims from the false-green live draft.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - Passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Passed.
- Command: bash scripts/check_ascii_python.sh
  - Passed.
- Command: bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline
  - Passed; extracted package refreshed from atlas_brain sources.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/support-ticket-blog-false-green-outcome-claims-pr-body.md
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Evaluator patterns | ~25 |
| Prompt contract | ~10 |
| Tests | ~45 |
| Plan doc | ~75 |
| Total | ~175 |
