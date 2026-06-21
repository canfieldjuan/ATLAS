# PR-Deflection-Priority-Status-Scoring

## Why this slice exists

#1612's locked report-shape plan says the paid report should become a work
queue, not a ticket archive. #1745 landed S1: the `deflection.v1` action
sections and fail-closed Snapshot projection. That contract intentionally used
a conservative deterministic baseline; it exposes action rows, but the priority
queue still sorts mostly by estimated support cost and the status classifier
does not yet prove the S2 cases from the plan.

This slice implements S2's narrow behavior: deterministic priority scoring and
status classification for the action sections. The goal is not a renderer
refresh; it is to make the persisted model tell a support lead which repeat to
fix first and why, using only repeat volume, benchmark support cost, answer
evidence status, outcome/CSAT signal, and evidence confidence already present in
the model.

This slice exceeds the 400 LOC soft budget because the persisted model contract
changes and the checked-in frontend example JSON is producer-synced. Splitting
the example update from the classifier would make the code and documented
contract disagree.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-actionability
Slice phase: Functional validation

1. Add an explicit deterministic `priority_score` and bounded
   `priority_drivers` to paid action rows.
2. Tune status classification so low-confidence/evidence-sparse rows are not
   promoted as publishable work, unresolved repeats stay actionable, drafted
   answers remain publish-ready, and already-covered-but-still-recurring rows
   are called out when CSAT/reopen signals show unresolved customer pain.
3. Sort the priority queue by the S2 score, then support cost/opportunity as
   stable tie-breakers.
4. Keep result-page/email/PDF rendering unchanged; S3/S4 consume this model
   contract later.

### Review Contract

- Acceptance criteria:
  - High-repeat rows with negative CSAT/reopen signal outrank otherwise similar
    high-repeat rows with good/absent CSAT.
  - Unresolved repeats with no proven answer receive `Needs answer` and a
    missing-answer driver.
  - Scoped resolution evidence with publishable answer copy receives
    `Draft ready`.
  - Scoped resolution evidence with customer pain after publication receives
    `Already covered but still recurring`.
  - Evidence-sparse or one-ticket rows receive `Low confidence` and do not enter
    `top_unresolved_repeats`.
  - Owner lane and fix type remain deterministic-only with `Unknown` fallback;
    no LLM/cloud calls are introduced in report rendering.
- Affected surfaces: paid `deflection.v1` report model action-section data,
  frontend contract example/docs, focused report model tests, and this plan.
- Risk areas: changing paid model shape, priority ordering regressions, and
  accidentally implying CSAT/cost precision beyond the benchmark/source data.
- Reviewer rules triggered: R1, R2, R10, R13, R14.
- boundary-probe: Required. This PR changes a classifier/ranking gate; review
  should inspect the same-count CSAT ordering probe and low-confidence
  rejection probe.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Priority-Status-Scoring.md`
- `tests/test_content_ops_deflection_report.py`

## Mechanism

Action rows gain an internal scoring helper. The score starts from the existing
opportunity/support-cost shape and adds deterministic weights for status,
negative CSAT, reopened tickets, and confidence. The score is persisted as
`priority_score` so later renderers can explain the queue without recomputing
private heuristics.

`priority_drivers` is a bounded list of safe enum strings such as
`high_repeat_volume`, `missing_answer`, `negative_csat`, `reopened_after_answer`,
`draft_ready`, and `low_confidence`. The labels describe why an item is ordered
where it is without echoing raw customer text, ticket IDs, or evidence quotes.

The classifier remains deterministic-only:

- empty/one-ticket/evidence-sparse rows become `Low confidence`;
- rows without scoped resolution evidence become `Needs answer`;
- scoped answers with reopen/negative-CSAT signal become
  `Already covered but still recurring`;
- scoped answers with answer copy become `Draft ready`;
- scoped evidence without answer copy remains `Needs review`.

The queue sort key changes to `priority_score`, then estimated support cost,
opportunity score, and original rank for stable ties.

## Intentional

- No renderer refresh. Result page, email, and PDF layout changes remain S3/S4.
- No LLM, cloud classifier, or inferred product taxonomy. Owner lane stays
  deterministic from existing topic data with `Unknown` fallback.
- No source-cost precision. Support cost remains benchmark-only unless future
  source data supplies a defensible cost basis.
- No Snapshot expansion. These action sections remain paid-only and absent from
  the free projection.

## Deferred

- S3 result-page actionable dashboard using the priority score/drivers.
- S4 curated email/PDF refresh using the same model sections.
- S5 cross-surface QA checks for action-section agreement.
- S6 monthly delta and macro/writeback upsell fields.

Parked hardening: none.

## Verification

- Focused report and frontend contract-doc tests -- 95 passed.
- Snapshot/report drift tests -- 10 passed.
- Combined focused report, frontend contract-doc, and snapshot/report drift
  tests -- 105 passed.
- Sparse-evidence review regression covered in the focused report tests:
  `ticket_count >= 2` with fewer than two sources now reports `Low confidence`,
  `confidence: low`, and receives the low-confidence score penalty.
- Python compile check for the report producer and focused report test module
  -- passed.
- Plan sync check for this plan against `origin/main` -- passed.
- Extracted package gauntlet:
  manifest validation, reasoning import audit, standalone audit, and ASCII
  policy -- passed.
- Full extracted pipeline check bundle -- passed; reasoning core 295 passed,
  extracted content 4764 passed / 15 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 132 |
| `docs/frontend/content_ops_faq_report_contract.md` | 6 |
| `extracted_content_pipeline/faq_deflection_report.py` | 92 |
| `plans/PR-Deflection-Priority-Status-Scoring.md` | 143 |
| `tests/test_content_ops_deflection_report.py` | 168 |
| **Total** | **541** |
