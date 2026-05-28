# PR: Support-Ticket Discoverability Claim Gate

## Why this slice exists

The accepted SaaS demo blog fixture review caught unsupported help-center and
discoverability language that the generated-content evaluator did not flag:
customers asking repeated questions "without finding answers," FAQ traffic
implying customers found the page, and low traffic implying a page is not
discoverable.

Those claims are not grounded by the support-ticket source package when the
input contains only uploaded questions, counts, clusters, and wording. The
fixture has been cleaned, but the evaluator still needs to catch the claim
shape so future generated blog or landing-page drafts do not false-green.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Robust testing

1. Add focused negative tests for the reviewed help-center/discoverability
   false-green phrases.
2. Tighten the support-ticket generated-content evaluator so those claims fail
   `support_ticket_outcome_claims_grounded`.
3. Keep the change limited to deterministic evaluation; no prompt, live LLM, or
   fixture-promotion changes are included.

### Files touched

- `plans/PR-Support-Ticket-Discoverability-Claim-Gate.md` - plan doc for this evaluator hardening slice.
- `extracted_content_pipeline/support_ticket_generated_content_eval.py` - add narrow unsupported discoverability claim patterns.
- `tests/test_evaluate_support_ticket_generated_content.py` - add regression tests for the reviewed false-green phrases.

## Mechanism

The evaluator already extracts generated text from landing-page/blog exports,
splits it into sentence-like claims, and checks each claim against
`_UNSUPPORTED_SUPPORT_OUTCOME_CLAIM_PATTERNS`. This slice adds narrowly scoped
patterns for:

- repeated questions caused by customers not finding answers in a help center
  or knowledge base
- FAQ/help-center traffic suggesting customers are finding a page or answer
- low traffic implying an entry is not discoverable
- support-team observations implying customers found or did not find FAQ entries

The tests feed those exact reviewed claim shapes through the public
`evaluate_support_ticket_generated_content(..., output="blog_post")` path and
assert the outcome check fails with the expected unsupported claims.

## Intentional

- This does not attempt a broad "discoverability" classifier. The current
  evaluator is regex-based by design, and this slice adds only the reviewed
  false-green shapes.
- This does not block neutral measurement language such as page views, customer
  feedback, or support-team observations when the copy does not infer
  discoverability or outcome causality.
- This is not another accepted-fixture update. PR #1090 already cleaned the
  committed fixture; this slice closes the detector gap that review exposed.

## Deferred

- Future PR: consolidate the growing unsupported-claim pattern list into named
  groups if review says the single tuple is becoming too hard to audit.
- Future PR: mutation-test the support-ticket generated-content evaluator
  branches if checker/evaluator false-greens keep recurring.
- Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q
  - Passed, 50 tests.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_blog_post.json --pretty
  - Passed; accepted fixture still reports `ok=true` and no unsupported claims.
- Command: python -m pytest tests/test_support_ticket_saas_demo_generated_content_fixtures.py -q
  - Passed, 2 tests.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file <PR body file>
  - Passed; advisory overlap warning with #1089 on
    `extracted_content_pipeline/support_ticket_generated_content_eval.py` and
    `tests/test_evaluate_support_ticket_generated_content.py`, no blocking
    drift.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~70 |
| Evaluator patterns | ~35 |
| Tests | ~55 |
| **Total** | **~160** |
