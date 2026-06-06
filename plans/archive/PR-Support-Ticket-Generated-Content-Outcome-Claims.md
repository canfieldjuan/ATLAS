# Support Ticket Generated Content Outcome Claims

## Why this slice exists

The support-ticket landing and blog paths now pass deterministic source-context
and readiness gates, but a product-content audit of the saved Haiku artifacts
found another truthfulness gap: generated copy can turn uploaded support-ticket
patterns into guaranteed outcomes such as support tickets dropping immediately
or future support interactions being prevented. Uploaded tickets show where
customers are stuck; they do not prove future volume reduction, churn reduction,
or immediate support impact.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Functional validation

1. Document the product-content audit against the existing live Haiku landing
   page and blog saved-draft exports.
2. Add a support-ticket generated-content evaluator check that fails guaranteed
   future support/customer outcome claims when the source context only contains
   uploaded ticket evidence.
3. Tighten the landing-page and blog prompts so support-ticket benefits use
   cautious language unless the source context includes real outcome metrics.
4. Add focused negative and false-positive tests for the new detector.

### Files touched

- `plans/PR-Support-Ticket-Generated-Content-Outcome-Claims.md`
- `docs/extraction/validation/support_ticket_generated_content_product_audit_2026-05-25.md`
- `extracted_content_pipeline/support_ticket_generated_content_eval.py`
- `tests/test_evaluate_support_ticket_generated_content.py`
- `extracted_content_pipeline/skills/digest/landing_page_generation.md`
- `atlas_brain/skills/digest/blog_post_generation.md`
- `extracted_content_pipeline/skills/digest/blog_post_generation.md`

## Mechanism

The evaluator already checks source context, unsupported uploaded-ticket
timeframes, unsupported cadence, source-backed percentages, and source-signal
visibility. This slice adds one more truthfulness check for guaranteed outcome
language, such as:

```text
Support tickets for those questions drop.
The 2 FAQ entries will prevent future support interactions.
Answering these two questions will reduce incoming support tickets immediately.
```

The check does not block cautious, defensible copy such as "can help reduce
repeat tickets" or "track whether ticket volume drops after publication."

## Intentional

- No new live LLM spend in this slice. It audits already-saved Haiku artifacts
  and adds deterministic coverage for the discovered issue.
- This does not change FAQ generation internals or standalone FAQ article
  shape; that remains owned by the FAQ lane.
- The detector is intentionally narrow. It targets guaranteed future support,
  queue, churn, or customer-retention outcomes, not every ordinary mention of
  reducing repeat questions.

## Deferred

- Future PR: rerun live Haiku landing and blog smokes after this prompt/evaluator
  change and record whether the new generated drafts pass the stricter
  outcome-claim gate.
- Parked hardening: none.

## Verification

Ran locally:

- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q`
  - 25 passed
- `python -m py_compile extracted_content_pipeline/support_ticket_generated_content_eval.py`
  - passed
- `python scripts/evaluate_support_ticket_generated_content.py --output landing_page tmp/support_ticket_live_haiku_eval_20260525/landing-page-draft.json --pretty`
  - failed as expected on `support_ticket_outcome_claims_grounded`
- `python scripts/evaluate_support_ticket_generated_content.py --output blog_post tmp/support_ticket_live_blog_gate_20260525/blog-post-draft-cadence2.json --pretty`
  - failed as expected on `support_ticket_outcome_claims_grounded`
- `bash scripts/local_pr_review.sh`
  - passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan and validation doc | ~170 |
| Evaluator | ~60 |
| Tests | ~70 |
| Prompt updates | ~8 |
| **Total** | **~308** |
