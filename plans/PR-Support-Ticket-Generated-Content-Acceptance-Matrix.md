# PR: Support-Ticket Generated Content Acceptance Matrix

## Why this slice exists

The support-ticket provider path now has route wiring, stress coverage, live
Haiku validation, and generated-content evaluators. After the cost/cache detour,
the next useful product step is to re-establish where generated blog and landing
content stands using the actual saved live artifacts already produced in this
lane.

This slice does not change generation logic unless the current evaluator finds
a data-truthfulness failure in the latest accepted artifacts. Its job is to make
"done for now" explicit: which generated support-ticket outputs pass today,
which older artifacts are known-bad regression examples, and which future
acceptance work remains.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Functional validation

1. Run the deterministic support-ticket generated-content evaluator against the
   latest accepted live landing-page and blog-post draft exports.
2. Run the evaluator against at least one older known-bad saved draft to confirm
   the detector still catches unsupported outcome claims.
3. Record a compact acceptance matrix with artifact paths, output type,
   evaluator result, and any manual audit notes.
4. Leave live LLM generation, FAQ generation, and new generated-copy tuning out
   of scope unless the accepted artifacts fail.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Support-Ticket-Generated-Content-Acceptance-Matrix.md` | Plan doc for this validation slice. |
| `docs/extraction/validation/support_ticket_generated_content_acceptance_matrix_2026-05-28.md` | Acceptance matrix for current support-ticket generated artifacts. |

## Mechanism

The validation uses the existing deterministic evaluator:

```bash
python scripts/evaluate_support_ticket_generated_content.py --output <type> <draft-export> --pretty
```

The matrix separates current accepted artifacts from older regression artifacts.
Accepted artifacts must pass. Known-bad artifacts are allowed to fail, and their
failure must be the expected unsupported generated-content class.

## Intentional

- This is a validation slice, not a new generator prompt slice.
- This does not spend on a new live model call. The saved exports already prove
  the route, DB save, export, and evaluator path from previous live slices.
- This does not touch FAQ generation. FAQ ownership remains in the parallel FAQ
  lane.

## Deferred

- Future PR: broader acceptance testing across more customer CSV shapes once we
  choose the representative datasets.
- Future PR: live generation rerun if the saved-artifact matrix shows stale or
  insufficient evidence for a model/prompt combination.
- Future PR: decide whether to promote the acceptance matrix into a scripted
  regression check once the representative artifact set is stable.
- Parked hardening: none planned.

## Verification

- Command: python scripts/evaluate_support_ticket_generated_content.py --output landing_page tmp/support_ticket_evidence_contract_live_validation_20260526/landing-page-draft.json --pretty
  - Passed.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post tmp/support_ticket_blog_small_upload_live_validation_20260526_policy/blog-post-draft.json --pretty
  - Passed.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post tmp/support_ticket_live_blog_gate_20260525/blog-post-draft-cadence2.json --pretty
  - Failed as expected on `support_ticket_outcome_claims_grounded` and
    `support_ticket_answer_steps_grounded`.
- Command: Python JSON export scanner over the three artifacts.
  - Current landing: 908 generated-copy words.
  - Current compact blog: 1,111 generated-content words, 4 H2, 0 H3.
  - Known-bad blog: 1,585 generated-content words, 7 H2, 0 H3.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/support-ticket-generated-content-acceptance-matrix-pr-body.md
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Validation doc | ~120 |
| **Total** | **~200** |
