# PR-Deflection-Cost-Resolvability-Scoring

## Why this slice exists

Parent issue atlas-portfolio#324 asks the report to discriminate cost from
resolvability: expensive repeats with no proven answer should stay visibly
expensive work, not lose priority merely because no draftable resolution exists.

Root cause: `priority_score` mixes benchmark support cost, upstream opportunity,
and status/resolvability weights in the same unbounded scale. That lets the
resolvability/status component and upstream opportunity score compete directly
with repeat cost, so the queue can treat "has an answer" / "already covered" as
part of the same ranking axis instead of using status and section placement to
explain resolvability separately.

This change fixes the root in the producer scoring helper. It does not add
another paid decision surface or continue review-control work.

## Scope (this PR)

Ownership lane: deflection/324-cost-resolvability-scoring
Slice phase: Functional validation

1. Make benchmark support cost the dominant `priority_score` component for paid
   action rows.
2. Bound the opportunity/status add-ons so resolvability labels can explain the
   row without drowning the cost ranking.
3. Keep the existing section discrimination: unresolved repeats remain
   `Needs answer` / `top_unresolved_repeats`, drafted answers remain
   `Draft ready`, and already-covered pain remains
   `already_covered_still_recurring`.
4. Add a regression where an expensive unresolved repeat outranks cheaper
   resolved/proven rows while still exposing `missing_answer` and no publishable
   answer copy.

### Review Contract

- Acceptance criteria:
  - A high-cost unresolved repeat outranks cheaper drafted/proven rows in
    `priority_fix_queue`.
  - The high-cost unresolved row keeps `status: Needs answer`, `fix_type:
    create_missing_answer`, and a `missing_answer` driver; it does not gain a
    publishable answer.
  - Lower-cost drafted and already-covered rows keep their resolvability/status
    labels and sections.
  - Low-confidence/suppressed rows remain excluded from `top_unresolved_repeats`.
- Affected surfaces: paid `deflection.v1` report-model action scoring and
  focused report producer tests.
- Risk areas: paid report ordering, score drift in docs/examples, and preserving
  the grounded-resolution promise.
- Reviewer rules triggered: R1, R2, R10, R13, R14.
- boundary-probe: Required. This PR changes ranking/scoring; review should
  inspect the high-cost unresolved vs cheaper resolved-row probe.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Cost-Resolvability-Scoring.md`
- `tests/test_content_ops_deflection_report.py`

## Mechanism

`_action_priority_score` scales the benchmark support-cost component above the
status/opportunity add-ons. Status weights remain present but small and bounded,
so they break close ties and keep unresolved/proven rows explainable without
becoming the primary ranking axis. The upstream `opportunity_score` is also
capped when folded into `priority_score`; it stays visible as its own field and
sort tie-breaker, but it no longer overwhelms repeat-cost pressure.

The queue sort remains deterministic: `priority_score`, then support cost,
opportunity score, then original rank. Section membership remains driven by
status, so this slice changes scoring priority rather than customer-facing
section shape.

## Intentional

- No paid-surface/review-control changes. Those require explicit operator
  approval and are outside #324's literal scoring ask.
- No new report sections or contract fields. This is a scoring calibration, not
  another shape expansion.
- Support cost remains the existing benchmark estimate (`ticket_count *
  assisted_contact_cost`), not a savings guarantee or tenant-specific cost.

## Deferred

- Tenant-specific assisted-contact cost calibration if a future input provides a
  defensible per-tenant cost basis.
- Any paid decision-surface/review-control expansion unless the operator
  explicitly approves that product shape.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_content_ops_deflection_report.py -k "priority_score or priority_queue_scores_status_and_csat_signals or cost_ahead_of_resolvability or priority_fix_queue_keeps_pdf_limit_items" - passed, 3 selected.
- Command: python -m pytest tests/test_content_ops_deflection_report.py -q - passed, 168 tests.
- Command: python -m pytest tests/test_content_ops_faq_report_contract_docs.py - passed, 5 tests.
- Command: python scripts/generate_deflection_snapshot_example.py --check - passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh - passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- Command: bash scripts/check_ascii_python.sh - passed.
- Command: bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline - passed; diff unchanged after sync.
- Command: python scripts/sync_pr_plan.py plans/PR-Deflection-Cost-Resolvability-Scoring.md - updated Files touched and diff size from the actual diff.
- Pending after this review fix: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/deflection-cost-resolvability-scoring-pr-body.md.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 12 |
| `docs/frontend/content_ops_faq_report_contract.md` | 7 |
| `extracted_content_pipeline/faq_deflection_report.py` | 53 |
| `plans/PR-Deflection-Cost-Resolvability-Scoring.md` | 118 |
| `tests/test_content_ops_deflection_report.py` | 194 |
| **Total** | **384** |
