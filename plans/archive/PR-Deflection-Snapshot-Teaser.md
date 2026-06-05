# PR-Deflection-Snapshot-Teaser

## Why this slice exists

Issue #1262 identifies a conversion gap in the unpaid FAQ deflection results page: the snapshot proves volume and ranking, but it shows no answer quality. The paid gate should not expose the report, evidence, source IDs, or the full drafted-answer set, but the free snapshot can safely expose one bounded, scoped, resolution-backed drafted answer plus a few body-withheld previews. This slice adds that backend projection so atlas-portfolio can render a real teaser instead of mocks.

The diff is slightly over the 400 LOC target because the contract is only safe when the producer, execute-route config, frontend docs, canonical example, and regression tests land together. Splitting those pieces would leave either an undocumented payload or docs/examples that are not producer-pinned.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection
Slice phase: Product polish

1. Add a `teaser` block to `DeflectionSnapshot` with one full scoped drafted answer and at most a configured number of body-withheld previews.
2. Select teaser candidates deterministically from items whose `answer_evidence_status` is `resolution_evidence` and `resolution_evidence_scope` is `scoped`.
3. Keep snapshot privacy fail-closed: no evidence quotes, source IDs, Markdown, term mappings, or non-teaser answer bodies leak.
4. Thread a default teaser preview-count config through the execute route snapshot gate.
5. Update frontend contract docs and the canonical snapshot example.

### Files touched

- `plans/PR-Deflection-Snapshot-Teaser.md` - Plan doc for this slice.
- `extracted_content_pipeline/faq_deflection_report.py` - Build and serialize the bounded teaser projection.
- `extracted_content_pipeline/api/control_surfaces.py` - Add route config for teaser preview count.
- `tests/test_content_ops_deflection_report.py` - Pin teaser selection, privacy, and fail-closed behavior.
- `tests/test_extracted_content_control_surface_api.py` - Pin the teaser preview-count config validation branch.
- `tests/test_extracted_content_ops_live_execute_harness.py` - Prove execute-route snapshots include the bounded teaser.
- `tests/test_content_ops_faq_report_contract_docs.py` - Keep producer docs/examples in sync.
- `docs/frontend/content_ops_faq_report_contract.md` - Document the snapshot teaser type.
- `docs/frontend/content_ops_faq_deflection_checkout_contract.md` - Document the unpaid teaser contract.
- `docs/frontend/content_ops_faq_deflection_snapshot_example.json` - Canonical example payload.

## Mechanism

The snapshot builder keeps summary/top questions unchanged and adds `teaser`. Eligible teaser items must have `answer_evidence_status == "resolution_evidence"`, `resolution_evidence_scope == "scoped"`, and a real drafted answer body. The full answer chooses the first eligible mid-rank item when possible, preferring rank 4+ so the highest-frequency answers stay locked; smaller reports fall back to the last eligible item so a one-answer report can still show a teaser. Previews are deterministic eligible items excluding the full answer, capped by config.

The full teaser answer includes only unpaid-safe render fields: rank, question, answer, steps, status/scope, weighted frequency, and source count. Preview entries include rank, question, status/scope, weighted frequency, step count, source count, and `body_withheld: true`. No evidence, source IDs, term mappings, nested FAQ item payloads, or Markdown are projected.

## Intentional

- This is a backend payload slice only. Portfolio rendering, blur treatment, and landing-page demo placement remain in atlas-portfolio.
- The teaser is omitted as an empty block when no item passes the scoped resolution-evidence plus drafted-answer gate.
- The source count is structural metadata, not source identity; it lets the frontend show proof shape without leaking source IDs.
- Cross-layer caller hints were inspected. Existing router/config callers remain compatible because the new preview count has a default, and the execute-route snapshot path is covered here; portfolio-ui is a deferred consumer and its contract docs are updated.

## Deferred

- atlas-portfolio rendering for the snapshot teaser and landing-page before/after demo.
- Live customer validation after this backend payload lands and the portfolio consumes it.

Parked hardening: none.

## Verification

- Command: pytest tests/test_content_ops_deflection_report.py::test_deflection_snapshot_includes_bounded_fail_closed_teaser tests/test_content_ops_deflection_report.py::test_deflection_snapshot_teaser_empty_when_no_scoped_resolution_evidence tests/test_content_ops_deflection_report.py::test_deflection_snapshot_rejects_negative_teaser_preview_count tests/test_extracted_content_ops_live_execute_harness.py::test_deflection_report_execute_uncaps_paid_artifact_and_keeps_snapshot_top_n tests/test_extracted_content_control_surface_api.py::test_content_ops_config_rejects_negative_deflection_teaser_preview_count tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_deflection_snapshot_example_matches_producer_shape -q - 6 passed.
- Command: pytest tests/test_content_ops_faq_report_contract_docs.py -q - 5 passed.
- Command: pytest tests/test_content_ops_deflection_report.py tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_returns_snapshot_for_unpaid_deflection_report tests/test_extracted_content_control_surface_api.py::test_content_ops_config_rejects_negative_deflection_teaser_preview_count tests/test_extracted_content_ops_live_execute_harness.py::test_live_execute_route_returns_faq_deflection_report_artifact tests/test_extracted_content_ops_live_execute_harness.py::test_deflection_report_execute_uncaps_paid_artifact_and_keeps_snapshot_top_n -q - 32 passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh - passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt - passed, 0 findings.
- Command: bash scripts/check_ascii_python.sh - passed.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main - passed, 144 matching tests enrolled.
- Command: bash scripts/run_extracted_pipeline_checks.sh - 2965 passed, 10 skipped, 1 warning.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/deflection-snapshot-teaser-pr-body.md - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 75 |
| Snapshot projection | 92 |
| API config | 17 |
| Tests | 202 |
| Docs/examples | 58 |
| **Total** | **444** |
