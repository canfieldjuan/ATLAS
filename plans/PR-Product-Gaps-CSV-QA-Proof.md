# PR-Product-Gaps-CSV-QA-Proof

## Why this slice exists

#1847 is the S4 proof slice for the CSV-first Product Gap arc. Prior slices
landed ingestion/routing (#1849), action fields and delivery surfaces (#1854),
and the paid Jira-copy affordance (#1855), but the remaining proof is split
across generic tests. Root cause: the product-gap vertical has buyer-visible
surfaces, yet the explicit QA observation still reports only generic paid-row
counts, and the synthetic CSV login proof does not assert the uncapped evidence
export for that same product-gap scenario.

This PR fixes the proof gap without adding new product shape: it extends the
existing CSV login vertical test to assert the evidence export audit details,
and extends the paid result-page QA observation to count Product Gap cards and
copy-ready Jira handoffs.

## Scope (this PR)

Ownership lane: deflection/product-gaps-report-shape
Slice phase: Functional validation

1. Add Product Gap card/Jira handoff counts to the existing paid result-page QA
   observation and enroll those counts in the hosted QA scorecard.
2. Extend the synthetic CSV login product-gap test so the same vertical proves
   evidence-export question and row details.
3. Keep report-model fields, snapshot/free projection, and email behavior
   unchanged.

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Product-Gaps-CSV-QA-Proof.md`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py`

### Review Contract

- Acceptance criteria:
  - [ ] The CSV login/discoverability fixture proves repeated product-gap
        owner lane, cost, routing metadata, action fields, and evidence export
        question/row audit details together.
  - [ ] The paid result-page QA observation exposes Product Gap card and Jira
        handoff counts, and the hosted QA scorecard validates them.
  - [ ] Locked/free result pages still do not render paid product-gap fields.
  - [ ] No new backend action/report fields or snapshot-safe fields are added.
  - [ ] No `source_ids`, `top_evidence`, or raw quotes are added to hosted card
        rendering.
- Affected surfaces: report/evidence-export tests, paid result-page QA
  observation, result-page smoke tests.
- Risk areas: privacy boundary, QA count truthfulness, scope creep into report
  contract.
- Reviewer rules triggered: R1, R2, R3, R9, R10, R12, R14.

## Mechanism

Thread the already-computed `gapItems` from `renderPaidArtifact` into
`resultPageQaObservation`, then report bounded counts for rendered Product Gap
cards and copy-ready Jira handoffs in the observation JSON. The Jira count uses
the same `jiraHandoffText` helper as the renderer, so the observation counts the
controls the page actually renders. The hosted QA scorecard now enrolls those
count/displayed-row keys and derives their model-side totals from the existing
`no_proven_answer_count`. The renderer already gates paid artifact rendering on
unlocked state, so the new counts remain absent on locked/free pages.

For backend proof, reuse the existing synthetic CSV login fixture and assert
the evidence export generated from the same artifact includes the product-gap
question, source IDs, evidence rows, and `needs_review` answer linkage.

## Intentional

- No new report-model fields; this is QA/proof around the existing #1849/#1854
  shape.
- No email changes; #1854 already covers compact email rendering and no raw
  evidence leak.
- No plan archive cleanup; another active PR has touched `plans/INDEX.md`, so
  this slice stays conflict-free.

## Deferred

- Closing/updating the parent issue checklist is deferred until this PR lands
  and the review verdict confirms S4 is complete.
- #1853 demo-derive remains paused/downstream until the product-gap arc is
  closed.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_content_ops_deflection_report.py::test_csv_product_gap_owner_lane_vertical_routes_login_gap -q` - passed.
- `python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_full_report_qa_harness_defers_result_page_action_row_observer tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py::test_hosted_smoke_scorecard_passes_sanitized_surface_observations -q` - passed.
- `node portfolio-ui/scripts/faq-deflection-result-page.test.mjs` - passed.
- `npm --prefix portfolio-ui run test:deflection-full-report-qa-hosted-smoke` - passed.
- Portfolio UI workflow-equivalent commands (`test:deflection-upload-shell`, `test:deflection-atlas-proxy`, `test:deflection-result`, `test:deflection-full-report-qa-hosted-smoke`) - passed.
- `scripts/run_extracted_pipeline_checks.sh` - passed (`4999 passed, 15 skipped`).
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/product-gaps-csv-qa-proof-pr-body.md` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_deflection_report.py` | 19 |
| `plans/PR-Product-Gaps-CSV-QA-Proof.md` | 105 |
| `portfolio-ui/api/content-ops/deflection/result-page.js` | 30 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 19 |
| `tests/test_content_ops_deflection_report.py` | 38 |
| `tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py` | 6 |
| **Total** | **217** |
