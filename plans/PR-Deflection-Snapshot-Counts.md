# PR-Deflection-Snapshot-Counts

## Why this slice exists

ATLAS issue #1272 is the backend prerequisite for atlas-portfolio#196 after the results-page teaser landed in atlas-portfolio#199. The portfolio needs real measured inputs for the inline Support Tax projection and locked 6-N rows, but the current unpaid snapshot exposes only question text, weighted ranking score, and teaser metadata. `weighted_frequency` is not a customer-specific ticket count, so using it for spend copy would violate the derived-number doctrine.

This slice exposes raw ticket counts from the FAQ item/source evidence already used by the report generator, and keeps the paid trust boundary intact by publishing only rank and count for rows beyond the free top-N.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection
Slice phase: Functional validation

1. Add raw `ticket_count` to each free `top_questions` row using real item counts or source-row cardinality, never `weighted_frequency`.
2. Add `summary.repeat_ticket_count` as the real total repeat-ticket volume available to the frontend projection.
3. Add `locked_questions` for ranks beyond the free top-N with only `{ rank, ticket_count }`.
4. Keep snapshot privacy fail-closed: no locked question text, answers, steps, evidence, source IDs, Markdown, FAQ result payloads, or term mappings leak through the new fields.
5. Update frontend contract docs and the canonical snapshot example.

### Files touched

- `plans/PR-Deflection-Snapshot-Counts.md` - Plan doc for this slice.
- `extracted_content_pipeline/faq_deflection_report.py` - Build and serialize raw-count and locked-row snapshot fields.
- `scripts/smoke_content_ops_deflection_paid_postgres.py` - Keep the paid-gate smoke fixture on the current snapshot shape.
- `tests/test_content_ops_deflection_report.py` - Pin raw count selection, locked-row withholding, and weighted-score non-fallback behavior.
- `tests/test_extracted_content_control_surface_api.py` - Pin execute-route snapshot counts on the unpaid gated response.
- `tests/test_extracted_content_ops_live_execute_harness.py` - Pin uncapped paid artifact snapshots with locked rank/count rows.
- `tests/test_faq_deflection_paid_postgres_smoke.py` - Keep the paid-gate smoke assertions and Postgres fake aligned with the current snapshot/store contract.
- `tests/test_smoke_content_ops_deflection_submit_handoff.py` - Keep the submit-handoff smoke fixture aligned with the current snapshot shape.
- `tests/test_content_ops_faq_report_contract_docs.py` - Keep producer docs/examples in sync and reject paid-field leaks.
- `docs/frontend/content_ops_faq_report_contract.md` - Document raw counts and locked rows.
- `docs/frontend/content_ops_faq_deflection_checkout_contract.md` - Document count-dependent projection rules.
- `docs/frontend/content_ops_faq_deflection_snapshot_example.json` - Canonical example payload.

## Mechanism

`build_deflection_snapshot` computes a raw count per FAQ item from `ticket_count` when present, otherwise from the real `source_ids` cardinality. If neither exists, the count is zero and the summary projection does not invent a replacement from `weighted_frequency`.

Visible top questions keep the existing fields and add `ticket_count`. The snapshot summary adds `repeat_ticket_count` as the sum of raw item counts across the artifact. Rows after `top_n` are projected into a new `locked_questions` tuple containing only rank and raw ticket count.

## Intentional

- This is an ATLAS producer slice only. atlas-portfolio will consume the fields in a follow-up PR under atlas-portfolio#196.
- `weighted_frequency` remains in the payload as ranking metadata, but it is deliberately not used as a fallback count.
- `source_ids` are counted server-side only. The IDs themselves remain stripped from the free snapshot.
- Locked rows publish every post-top-N rank with count metadata only. This supports 6-N FOMO rows without exposing customer wording before payment.
- Cross-layer caller hints were inspected. The execute-route caller is covered by `test_execute_generation_route_returns_snapshot_for_unpaid_deflection_report` and the live execute harness; the readonly opportunity helper is covered by `test_deflection_snapshot_content_opportunities_are_unpaid_safe`; `portfolio-ui` is a legacy/non-canonical consumer and the active atlas-portfolio handoff is covered by docs/examples; broad `as_dict` and `_Pool` hints are name-only false positives outside this slice's call path.

## Deferred

- atlas-portfolio rendering for the Support Tax projection, locked 6-N rows, and cost-per-ticket benchmark citation.
- Live report validation after ATLAS deploys this producer change and the portfolio consumes it.

Parked hardening: none.

## Verification

- Command: pytest tests/test_content_ops_deflection_report.py::test_deflection_snapshot_strips_answers_evidence_and_sources tests/test_content_ops_deflection_report.py::test_deflection_snapshot_counts_are_raw_and_locked_rows_hide_questions tests/test_content_ops_deflection_report.py::test_deflection_snapshot_content_opportunities_are_unpaid_safe tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_returns_snapshot_for_unpaid_deflection_report tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_deflection_snapshot_example_matches_producer_shape -q - 5 passed.
- Command: pytest tests/test_content_ops_deflection_report.py tests/test_content_ops_faq_report_contract_docs.py tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_returns_snapshot_for_unpaid_deflection_report tests/test_extracted_content_ops_live_execute_harness.py::test_deflection_report_execute_uncaps_paid_artifact_and_keeps_snapshot_top_n tests/test_faq_deflection_paid_postgres_smoke.py::test_deflection_paid_postgres_smoke_seeds_locks_marks_paid_and_rereads tests/test_smoke_content_ops_deflection_submit_handoff.py -q - 60 passed, 1 warning.
- Command: bash scripts/validate_extracted_content_pipeline.sh - passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt - passed, 0 findings.
- Command: bash scripts/check_ascii_python.sh - passed.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main - passed, 144 matching tests enrolled.
- Command: bash scripts/run_extracted_pipeline_checks.sh - 2973 passed, 10 skipped, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 72 |
| Snapshot projection | 23 |
| Smoke fixture | 3 |
| Tests | 115 |
| Docs/examples | 29 |
| **Total** | **242** |
